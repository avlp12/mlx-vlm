"""Parity for the opt-in segment-aligned routed-expert GEMM (MLX_VLM_GLM5_MOE_GEMM).

``Glm5NextTiledSwitchGLU`` pads every expert's run of sorted rows out to a multiple
of ``R`` so no ``gather_qmm_rhs`` 16-row block ever spans two experts.  The change is
purely a row *layout* change: the same ``BlockMMA`` runs over the same ``BK=32`` K
steps into the same fp32 accumulator, so the result is expected to be **bit-identical**
to the stock ``SwitchGLU`` -- these tests pin that rather than a tolerance.

They also pin that the path declines what it is not written for: decode-shaped and
otherwise low-occupancy routes (where padding a whole ``R``-row tile per active expert
would cost more than the boundary passes it removes) fall back to stock.
"""

import os

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models.glm5_next.moe_gemm import (
    Glm5NextTiledSwitchGLU,
    choose_tile_rows,
    min_rows,
    moe_gemm_enabled,
    segment_tile_plan,
)
from mlx_vlm.models.switch_layers import SwitchGLU

# GLM-5.3-Flash routed-expert geometry, scaled down on the hidden dims so the test
# runs in a second; E / top_k / group_size / bits are the live values.
E, TOP_K, GROUP, BITS = 288, 8, 64, 4
HIDDEN, INTER = 256, 128


def _pair(seed=0, hidden=HIDDEN, inter=INTER, experts=E):
    mx.random.seed(seed)
    ref = SwitchGLU(hidden, inter, experts, bias=False)
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(ref, name, getattr(ref, name).to_quantized(GROUP, BITS, mode="affine"))
    mx.eval(ref.parameters())
    new = Glm5NextTiledSwitchGLU(hidden, inter, experts, bias=False)
    # Share the *same* quantized parameter arrays: any difference is the kernel path.
    new.gate_proj, new.up_proj, new.down_proj = (
        ref.gate_proj,
        ref.up_proj,
        ref.down_proj,
    )
    new.activation = ref.activation
    return ref, new


def _route(tokens, seed=0, experts=E, hidden=HIDDEN):
    mx.random.seed(seed + 1000)
    x = mx.random.normal((1, tokens, hidden)).astype(mx.bfloat16)
    logits = mx.random.normal((1, tokens, experts))
    idx = mx.argpartition(-logits, TOP_K - 1, axis=-1)[..., :TOP_K].astype(mx.uint32)
    mx.eval(x, idx)
    return x, idx


def test_env_toggle_defaults_off(monkeypatch):
    monkeypatch.delenv("MLX_VLM_GLM5_MOE_GEMM", raising=False)
    assert moe_gemm_enabled() is False
    monkeypatch.setenv("MLX_VLM_GLM5_MOE_GEMM", "1")
    assert moe_gemm_enabled() is True
    monkeypatch.setenv("MLX_VLM_GLM5_MOE_GEMM", "0")
    assert moe_gemm_enabled() is False

    class Cfg:
        moe_prefill_gemm = True

    monkeypatch.delenv("MLX_VLM_GLM5_MOE_GEMM", raising=False)
    assert moe_gemm_enabled(Cfg()) is True


@pytest.mark.parametrize("rows_per_tile", [16, 32, 64])
def test_segment_tile_plan_is_a_valid_permutation(rows_per_tile):
    _, idx = _route(1024, seed=3)
    flat = idx.flatten()
    sorted_idx = mx.sort(flat).astype(mx.uint32)
    slot, tile_expert, n_tiles, used = segment_tile_plan(sorted_idx, E, rows_per_tile)
    assert used == rows_per_tile

    slots = slot.tolist()
    assert len(set(slots)) == len(slots), "slots must be distinct"
    assert max(slots) < n_tiles * rows_per_tile
    # Every row lands in a tile owned by its own expert, and no tile mixes experts.
    experts_of = tile_expert.tolist()
    for row, s in enumerate(slots):
        assert experts_of[s // rows_per_tile] == int(sorted_idx[row].item())
    # Padding is exactly ceil(c_e / R) * R.
    counts = {}
    for e in sorted_idx.tolist():
        counts[e] = counts.get(e, 0) + 1
    expected = sum((c + rows_per_tile - 1) // rows_per_tile for c in counts.values())
    assert n_tiles == expected


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("rows_per_tile", ["auto", 16, 32])
def test_bit_identical_to_stock_switch_glu(monkeypatch, seed, rows_per_tile):
    monkeypatch.setenv("MLX_VLM_GLM5_MOE_GEMM_ROWS", str(rows_per_tile))
    monkeypatch.setenv("MLX_VLM_GLM5_MOE_GEMM_MIN", "64")
    ref, new = _pair(seed=seed)
    x, idx = _route(2048, seed=seed)
    want, got = ref(x, idx), new(x, idx)
    mx.eval(want, got)
    assert want.shape == got.shape
    assert mx.array_equal(
        want, got
    ), f"max abs {float(mx.max(mx.abs(want.astype(mx.float32) - got.astype(mx.float32))))}"


def test_ragged_route_including_empty_experts(monkeypatch):
    """Hot/cold route: some experts get hundreds of rows, some get none."""
    monkeypatch.setenv("MLX_VLM_GLM5_MOE_GEMM_ROWS", "16")
    monkeypatch.setenv("MLX_VLM_GLM5_MOE_GEMM_MIN", "64")
    ref, new = _pair(seed=7)
    mx.random.seed(11)
    tokens = 1024
    x = mx.random.normal((1, tokens, HIDDEN)).astype(mx.bfloat16)
    # Bias the router hard so only ~a third of the experts are ever selected.
    bias = mx.concatenate([mx.full((E // 3,), 8.0), mx.zeros((E - E // 3,)) - 8.0])
    logits = mx.random.normal((1, tokens, E)) + bias
    idx = mx.argpartition(-logits, TOP_K - 1, axis=-1)[..., :TOP_K].astype(mx.uint32)
    mx.eval(x, idx)
    assert len(set(idx.flatten().tolist())) < E, "route must leave experts empty"
    want, got = ref(x, idx), new(x, idx)
    mx.eval(want, got)
    assert mx.array_equal(want, got)


def test_auto_rows_matches_the_measured_winner():
    """R=32's extra padding must only be taken when the bm=32 tile pays for it.

    Reference points are the measured whole-layer A/B (E=288, top-8, M3 Ultra,
    speedup over stock at R=16 / R=32):

        512 tokens ( 14.2 rows/expert)  1.337x / 0.972x  -> 16
       1024        ( 28.4)              1.220x / 1.203x  -> 16
       1536        ( 42.7)              1.150x / 1.006x  -> 16
       2048        ( 56.9)              1.107x / 1.110x  -> 16 (tie)
       3072        ( 85.3)              1.074x / 1.076x  -> 16 (tie)
       4096        (113.8)              1.051x / 1.061x  -> 32
       8192        (227.6)              1.020x / 1.062x  -> 32
    """
    for tokens, want in ((512, 16), (1024, 16), (1536, 16), (4096, 32), (8192, 32)):
        _, idx = _route(tokens, seed=5)
        sorted_idx = mx.sort(idx.flatten()).astype(mx.uint32)
        counts = (
            mx.zeros((E,), mx.int32)
            .at[sorted_idx.astype(mx.int32)]
            .add(mx.ones((sorted_idx.size,), mx.int32))
        )
        got = choose_tile_rows(counts)
        assert got == want, (
            f"{tokens} tokens ({tokens*TOP_K/E:.1f} rows/expert): "
            f"chose R={got}, measured winner is R={want}"
        )


def test_env_rows_override_beats_auto(monkeypatch):
    monkeypatch.setenv("MLX_VLM_GLM5_MOE_GEMM_ROWS", "32")
    monkeypatch.setenv("MLX_VLM_GLM5_MOE_GEMM_MIN", "64")
    ref, new = _pair(seed=9)
    x, idx = _route(1536, seed=9)  # auto would pick R=16 here
    want, got = ref(x, idx), new(x, idx)
    mx.eval(want, got)
    assert mx.array_equal(want, got)


def test_decode_shapes_fall_back_to_stock(monkeypatch):
    """The tiled layout must not fire on decode / speculative-block shapes."""
    monkeypatch.delenv("MLX_VLM_GLM5_MOE_GEMM_MIN", raising=False)
    monkeypatch.delenv("MLX_VLM_GLM5_MOE_GEMM_ROWS", raising=False)
    ref, new = _pair(seed=2)
    gate = min_rows(E, 16)
    assert gate == 16 * E * 3 // 4  # 3456 routed rows = 432 tokens at top-8
    for tokens in (1, 8, 64, 256):
        x, idx = _route(tokens, seed=2)
        assert idx.size < gate
        want, got = ref(x, idx), new(x, idx)
        mx.eval(want, got)
        assert mx.array_equal(want, got)


def test_multi_layer_greedy_token_identity(monkeypatch):
    """64-token greedy walk through a stack of MoE blocks, 5 seeds.

    Not a tolerance check -- with bit-identical blocks the argmax stream can only
    match, so this pins that the *wiring* (sort, pad, unpad, unsort, weighting)
    is right end to end and that nothing drifts across 8 stacked layers.
    """
    monkeypatch.setenv("MLX_VLM_GLM5_MOE_GEMM_ROWS", "16")
    monkeypatch.setenv("MLX_VLM_GLM5_MOE_GEMM_MIN", "64")
    layers, tokens, vocab = 8, 64, 512

    for seed in range(5):
        ref, new = _pair(seed=seed)
        blocks = [_pair(seed=seed * 31 + i) for i in range(layers)]
        mx.random.seed(seed + 500)
        emb = mx.random.normal((vocab, HIDDEN)).astype(mx.bfloat16) * 0.05
        head = mx.random.normal((vocab, HIDDEN)).astype(mx.bfloat16) * 0.05
        router = [
            mx.random.normal((E, HIDDEN)).astype(mx.float32) * 0.05
            for _ in range(layers)
        ]
        toks = mx.random.randint(0, vocab, (1, tokens))
        mx.eval(emb, head, toks, *router)

        def run(pick_new):
            h = emb[toks]
            for i in range(layers):
                logits = h.astype(mx.float32) @ router[i].T
                idx = mx.argpartition(-logits, TOP_K - 1, axis=-1)[..., :TOP_K]
                scores = mx.softmax(
                    mx.take_along_axis(logits, idx, axis=-1), axis=-1
                ).astype(h.dtype)
                mlp = blocks[i][1] if pick_new else blocks[i][0]
                y = mlp(h, idx.astype(mx.uint32))
                h = h + (y * scores[..., None]).sum(axis=-2).astype(h.dtype)
            return mx.argmax(h.astype(mx.float32) @ head.T, axis=-1)

        want, got = run(False), run(True)
        mx.eval(want, got)
        assert mx.array_equal(want, got), f"seed {seed}: token stream diverged"


def test_model_wiring_selects_the_tiled_switch_glu(monkeypatch):
    import mlx_vlm.models.glm5_next.language as glm5
    from mlx_vlm.models.glm5_next.config import TextConfig

    cfg = dict(
        model_type="glm5_next_text",
        vocab_size=512,
        hidden_size=HIDDEN,
        intermediate_size=512,
        moe_intermediate_size=INTER,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        n_shared_experts=1,
        n_routed_experts=E,
        routed_scaling_factor=2.5,
        kv_lora_rank=64,
        q_lora_rank=64,
        qk_rope_head_dim=0,
        v_head_dim=32,
        qk_nope_head_dim=32,
        num_experts_per_tok=TOP_K,
        first_k_dense_replace=0,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        index_topk=64,
        index_head_dim=32,
        index_n_heads=4,
        layer_types=["linear_attention"],
        mlp_layer_types=["sparse"],
        linear_attn_config={
            "num_heads": 8,
            "gate_lower_bound": -5.0,
            "head_dim": 32,
            "short_conv_kernel_size": 4,
        },
    )
    monkeypatch.delenv("MLX_VLM_GLM5_MOE_GEMM", raising=False)
    assert (
        type(glm5.Glm5NextMoE(TextConfig.from_dict(dict(cfg))).switch_mlp) is SwitchGLU
    )

    monkeypatch.setenv("MLX_VLM_GLM5_MOE_GEMM", "1")
    moe = glm5.Glm5NextMoE(TextConfig.from_dict(dict(cfg)))
    assert isinstance(moe.switch_mlp, Glm5NextTiledSwitchGLU)
    # Parameter tree must be identical to the stock module's, or the checkpoint
    # would not load into it.
    monkeypatch.setenv("MLX_VLM_GLM5_MOE_GEMM", "0")
    stock = glm5.Glm5NextMoE(TextConfig.from_dict(dict(cfg)))

    def keys(tree, prefix=""):
        out = set()
        for k, v in tree.items():
            p = f"{prefix}{k}"
            if isinstance(v, dict):
                out |= keys(v, p + ".")
            else:
                out.add((p, tuple(v.shape)))
        return out

    assert keys(moe.parameters()) == keys(stock.parameters())


# --------------------------------------------------------------------------- #
# Micro-bench: python -m mlx_vlm.tests.test_glm5_next_moe_gemm [chunk ...]
# --------------------------------------------------------------------------- #


def _bench():
    import math
    import sys
    import time

    hidden, inter, experts, top_k, layers = 4096, 2048, 288, 8, 42
    chunks = [int(a) for a in sys.argv[1:]] or [2048]
    baseline_tok_s = 450.0

    def timeit(fn, warm=3, iters=8):
        for _ in range(warm):
            mx.eval(fn())
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        mx.synchronize()
        return (time.perf_counter() - t0) / iters

    def quant(shape):
        s = math.sqrt(1.0 / shape[-1])
        w = mx.random.uniform(low=-s, high=s, shape=shape).astype(mx.bfloat16)
        q, sc, b = mx.quantize(w, group_size=GROUP, bits=BITS, mode="affine")
        mx.eval(q, sc, b)
        return q, sc, b

    # --- where the gap comes from: one gate/up projection, three routes -------
    print(
        f"--- one gate/up projection, x[{chunks[0]*top_k},{hidden}] "
        f"@ w[{experts},{inter},{hidden}] q{BITS} g{GROUP} ---"
    )
    n = chunks[0] * top_k
    flops = 2.0 * n * hidden * inter
    q, sc, b = quant((experts, inter, hidden))
    x1 = mx.random.normal((n, 1, hidden)).astype(mx.bfloat16)
    mx.eval(x1)

    def counts_to_idx(counts):
        i = mx.concatenate(
            [mx.full((c,), e, dtype=mx.uint32) for e, c in enumerate(counts) if c]
        )
        mx.eval(i)
        return i

    base, rem = divmod(n, experts)
    routes = {
        "sorted, mean 56.9 rows/expert": counts_to_idx(
            [base + (1 if e < rem else 0) for e in range(experts)]
        ),
        "sorted, counts forced to x16": counts_to_idx(
            [
                ((n // 16) // experts + (1 if e < (n // 16) % experts else 0)) * 16
                for e in range(experts)
            ]
        ),
    }
    for name, idx in routes.items():
        dt = timeit(
            lambda: mx.gather_qmm(
                x1,
                q,
                sc,
                b,
                rhs_indices=idx,
                transpose=True,
                group_size=GROUP,
                bits=BITS,
                mode="affine",
                sorted_indices=True,
            )
        )
        print(f"  gather_qmm {name:<32s} {dt*1e3:8.3f} ms  {flops/dt/1e12:6.2f} TFLOPS")
    qd, scd, bd = quant((inter, hidden))
    x2 = x1.reshape(n, hidden)
    dt = timeit(
        lambda: mx.quantized_matmul(
            x2, qd, scd, bd, transpose=True, group_size=GROUP, bits=BITS, mode="affine"
        )
    )
    print(
        f"  {'dense quantized_matmul, same M/N/K':<43s} {dt*1e3:8.3f} ms  "
        f"{flops/dt/1e12:6.2f} TFLOPS"
    )
    del q, sc, b, qd, scd, bd, x1, x2

    # --- whole-layer A/B -----------------------------------------------------
    mx.random.seed(0)
    ref, new = _pair(seed=0, hidden=hidden, inter=inter, experts=experts)
    for chunk in chunks:
        mx.random.seed(1)
        x = mx.random.normal((1, chunk, hidden)).astype(mx.bfloat16)
        lg = mx.random.normal((1, chunk, experts))
        idx = mx.argpartition(-lg, top_k - 1, axis=-1)[..., :top_k].astype(mx.uint32)
        mx.eval(x, idx)
        wall = chunk / baseline_tok_s * 1e3
        t_ref = timeit(lambda: ref(x, idx))
        print(
            f"\n--- chunk {chunk} ({chunk*top_k} rows, "
            f"{chunk*top_k/experts:.1f} rows/expert) ---"
        )
        print(
            f"  stock SwitchGLU        {t_ref*1e3:8.3f} ms/layer  "
            f"{t_ref*1e3*layers:8.1f} ms/chunk  "
            f"({t_ref*1e3*layers/wall*100:.1f}% of a {wall:.0f} ms chunk)"
        )
        for rows in (16, 32):
            os.environ["MLX_VLM_GLM5_MOE_GEMM_ROWS"] = str(rows)
            os.environ["MLX_VLM_GLM5_MOE_GEMM_MIN"] = "64"
            same = bool(mx.array_equal(ref(x, idx), new(x, idx)))
            t_new = timeit(lambda: new(x, idx))
            print(
                f"  tiled R={rows:<3d}             {t_new*1e3:8.3f} ms/layer  "
                f"{t_new*1e3*layers:8.1f} ms/chunk   {t_ref/t_new:.4f}x   "
                f"e2e {(t_ref-t_new)*1e3*layers/wall*100:+5.2f}%   "
                f"bit-identical={same}"
            )
        del x, lg, idx


if __name__ == "__main__":
    _bench()
