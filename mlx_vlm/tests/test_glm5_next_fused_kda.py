"""Eager vs fused equivalence for the GLM-5-Next KDA single-token decode step.

``MLX_VLM_GLM5_FUSED_KDA=1`` replaces the ~30 tiny dispatches that make up the
post-projection half of ``Glm5NextLinearAttention`` (conv1d window update, silu,
two L2 norms, the safe forget gate, beta, the gated delta rule and the gated
RMSNorm) with a single ``mx.fast.metal_kernel`` launch per layer.

The kernel is a rounding-faithful transcription, not an approximation: it is
expected to be *bit-identical* to the eager path, including the fp32 recurrent
state carried across steps.  These tests pin that, and pin that the fast path
correctly declines anything it is not written for (prefill, S>1, batched or
left-padded decode).  A speculative capture is *not* declined: the kernel has a
capture variant that emits the ``gdn_sink`` tensors, checked here against the
eager sink and against the rollback replay those tensors feed.

``MLX_VLM_GLM5_FUSED_KDA_QPROJ=1`` additionally folds the two small quantized
projections (``f_b_proj`` / ``g_b_proj``) into the same launch.

Run the 34-layer micro-bench with ``python -m mlx_vlm.tests.test_glm5_next_fused_kda``.
"""

import os
import re

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_vlm.models.glm5_next.language as glm5
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next.fused_kda import fused_kda_probe

# GLM-5.3-Flash text_config, restricted to what the KDA layer reads.  The kernel
# is parameterised by linear_num_heads / linear_head_dim / short_conv_kernel_size
# / gate_lower_bound / rms_norm_eps, so those are the live values verbatim.
_CFG = dict(
    model_type="glm5_next_text",
    vocab_size=1024,
    hidden_size=4096,
    intermediate_size=12288,
    moe_intermediate_size=2048,
    num_hidden_layers=1,
    num_attention_heads=64,
    num_key_value_heads=64,
    n_shared_experts=1,
    n_routed_experts=288,
    routed_scaling_factor=2.5,
    kv_lora_rank=512,
    q_lora_rank=1536,
    qk_rope_head_dim=0,
    v_head_dim=256,
    qk_nope_head_dim=256,
    num_experts_per_tok=8,
    first_k_dense_replace=3,
    max_position_embeddings=1048576,
    rms_norm_eps=1e-05,
    index_topk=2048,
    index_head_dim=128,
    index_n_heads=32,
    layer_types=["linear_attention"],
    mlp_layer_types=["dense"],
    linear_attn_config={
        "num_heads": 64,
        "gate_lower_bound": -5.0,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
    },
)


def _config():
    return TextConfig.from_dict(dict(_CFG))


def _layer(config, seed=0):
    mx.random.seed(seed)
    layer = glm5.Glm5NextLinearAttention(config)

    def rand(tree):
        if isinstance(tree, dict):
            return {k: rand(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rand(v) for v in tree]
        return (mx.random.normal(tree.shape) * 0.05).astype(mx.bfloat16)

    layer.update(rand(layer.parameters()))
    H, D = config.linear_num_heads, config.linear_head_dim
    layer.conv1d.weight = (mx.random.normal(layer.conv1d.weight.shape) * 0.5).astype(
        mx.bfloat16
    )
    # A_log / dt_bias are kept in fp32 by the converter's cast_predicate.
    layer.forget_gate.A_log = (mx.random.normal((H,)) * 0.5).astype(mx.float32)
    layer.forget_gate.dt_bias = (mx.random.normal((H * D,)) * 0.5).astype(mx.float32)
    layer.o_norm.weight = (mx.ones((D,)) + 0.02 * mx.random.normal((D,))).astype(
        mx.bfloat16
    )
    # The live build quantises the KDA projections to 8-bit, group 64.
    nn.quantize(layer, group_size=64, bits=8)
    mx.eval(layer.parameters())
    return layer


def _cache(config, batch=1, seed=1):
    mx.random.seed(seed)
    H, D, K = (
        config.linear_num_heads,
        config.linear_head_dim,
        config.linear_conv_kernel_dim,
    )
    cache = ArraysCache(size=2)
    # "Warmed" states: a decode step never sees the zero-initialised cache.
    cache[0] = (mx.random.normal((batch, K - 1, 3 * H * D)) * 0.3).astype(mx.bfloat16)
    cache[1] = (mx.random.normal((batch, H, D, D)) * 0.05).astype(mx.float32)
    mx.eval(cache[0], cache[1])
    return cache


def _clone(cache):
    out = ArraysCache(size=2)
    out[0] = mx.array(cache[0])
    out[1] = mx.array(cache[1])
    mx.eval(out[0], out[1])
    return out


def _max_abs_rel(a, b):
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    d = mx.abs(a32 - b32)
    return float(d.max()), float((d / mx.maximum(mx.abs(a32), 1e-30)).max())


@pytest.fixture(autouse=True)
def _reset_toggle():
    saved = (glm5._FUSED_KDA_ENV, glm5._FUSED_KDA_QPROJ_ENV, glm5._FUSED_KDA_MAX_BATCH)
    yield
    glm5._FUSED_KDA_ENV, glm5._FUSED_KDA_QPROJ_ENV, glm5._FUSED_KDA_MAX_BATCH = saved


def _allow_batch(batch):
    """Let a parity cell run at a width the shipped cap does not yet permit.

    The cap is a policy number about evidence; this suite is where the evidence
    comes from, so it has to be able to run ahead of it.  Restored by the autouse
    fixture, so a parity cell can never leave the cap raised for anything else.
    """
    glm5._FUSED_KDA_MAX_BATCH = max(batch, glm5._FUSED_KDA_MAX_BATCH)


def _set_toggle(layer, on, qproj=False):
    glm5._FUSED_KDA_ENV = on
    glm5._FUSED_KDA_QPROJ_ENV = qproj
    layer._fused_kda = None
    layer._fused_kda_qproj = None
    layer._fused_kda_ty = None
    layer._fused_kda_qproj_ty = None


def _require_device(config, kind, layer=None):
    """Skip where the GPU cannot run this pipeline at any supported size."""
    kwargs = dict(
        kind=kind,
        num_heads=config.linear_num_heads,
        head_dim=config.linear_head_dim,
        conv_kernel_size=config.linear_conv_kernel_dim,
        dtype=mx.bfloat16,
        state_dtype=mx.float32,
    )
    if kind == "qproj":
        kwargs["bits"] = int(layer.forget_gate.f_b_proj.bits)
        kwargs["group_size"] = int(layer.forget_gate.f_b_proj.group_size)
    if fused_kda_probe(**kwargs) is None:
        pytest.skip(f"device threadgroup cap below the {kind} kernel's requirement")


# Every batch width the eager-vs-fused parity suite is run at.  _FUSED_KDA_MAX_BATCH
# is required to be one of these (test_fused_kda_batch_cap_is_covered_by_parity), so
# the cap cannot be raised to a width nothing has ever been compared at.
_PARITY_BATCHES = (2, 4, 8, 16, 32)


def _masks(kind, batch, steps, seed=99):
    """The three regimes batched decode actually produces."""
    if kind == "none":
        return [None] * steps
    if kind == "all-true":
        return [mx.ones((batch, 1), dtype=mx.bool_)] * steps
    mx.random.seed(seed)
    m = [mx.random.uniform(shape=(batch, 1)) > 0.35 for _ in range(steps)]
    mx.eval(m)
    return m


def test_fused_kda_matches_eager_over_32_decode_steps():
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    layer = _layer(config)
    eager_cache = _cache(config)
    fused_cache = _clone(eager_cache)

    mx.random.seed(4321)
    steps = [
        mx.random.normal((1, 1, config.hidden_size)).astype(mx.bfloat16)
        for _ in range(32)
    ]
    mx.eval(steps)

    worst = (0.0, 0.0)
    for x in steps:
        _set_toggle(layer, False)
        eager_out = layer(x, None, eager_cache)
        _set_toggle(layer, True)
        fused_out = layer(x, None, fused_cache)
        mx.eval(eager_out, fused_out, eager_cache.cache, fused_cache.cache)
        for ref, got in (
            (eager_out, fused_out),
            (eager_cache[0], fused_cache[0]),
            (eager_cache[1], fused_cache[1]),
        ):
            worst = max(worst, _max_abs_rel(ref, got))

    # The transcription reproduces every rounding point of the eager chain, so
    # this is exact rather than merely close.  Keep a bf16-scale tolerance in the
    # assertion so a future MLX rounding change degrades to "still fine" rather
    # than a hard failure, but report the real number.
    assert worst[0] <= 2.0**-6, f"max abs diff {worst[0]:.3e} (rel {worst[1]:.3e})"
    assert worst == (0.0, 0.0), f"expected bit-identical, got {worst}"


def test_fused_kda_capture_matches_eager_gdn_sink():
    """With a drafter attached the capture variant must reproduce gdn_sink exactly.

    Those tensors are not observable in the forward output -- they are replayed by
    ``rollback_speculative_cache`` on a partial accept -- so they need their own
    check, including running the rollback consumer on both sinks.
    """
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    from mlx_vlm.models.gated_delta import gated_delta_update

    config = _config()
    layer = _layer(config)
    eager_cache = _cache(config)
    fused_cache = _clone(eager_cache)

    mx.random.seed(2468)
    steps = [
        mx.random.normal((1, 1, config.hidden_size)).astype(mx.bfloat16)
        for _ in range(32)
    ]
    mx.eval(steps)

    names = ["q", "k", "v", "a", "b", "A_log", "dt_bias", "state", "conv_input"]
    worst = (0.0, 0.0)
    for x in steps:
        eager_sink, fused_sink = [], []
        _set_toggle(layer, False)
        eager_out = layer(x, None, eager_cache, gdn_sink=eager_sink)
        _set_toggle(layer, True)
        fused_out = layer(x, None, fused_cache, gdn_sink=fused_sink)
        assert len(eager_sink) == len(fused_sink) == 1

        # Replay the rollback consumer (accept the single token) on both sinks.
        def replay(entry):
            return gated_delta_update(
                entry[0][:, :1],
                entry[1][:, :1],
                entry[2][:, :1],
                entry[3][:, :1],
                entry[4][:, :1],
                entry[5],
                entry[6],
                state=entry[7],
                lower_bound=entry[10],
            )[1]

        eager_roll, fused_roll = replay(eager_sink[0]), replay(fused_sink[0])
        mx.eval(
            eager_out,
            fused_out,
            eager_cache.cache,
            fused_cache.cache,
            eager_sink[0][:9],
            fused_sink[0][:9],
            eager_roll,
            fused_roll,
        )
        assert eager_sink[0][9] == fused_sink[0][9] == config.linear_conv_kernel_dim
        assert eager_sink[0][10] == fused_sink[0][10] == config.linear_lower_bound
        pairs = [
            (eager_out, fused_out),
            (eager_cache[0], fused_cache[0]),
            (eager_cache[1], fused_cache[1]),
            (eager_roll, fused_roll),
        ]
        for i, name in enumerate(names):
            ref, got = eager_sink[0][i], fused_sink[0][i]
            assert ref.shape == got.shape, f"{name}: {ref.shape} vs {got.shape}"
            assert ref.dtype == got.dtype, f"{name}: {ref.dtype} vs {got.dtype}"
            pairs.append((ref, got))
        for ref, got in pairs:
            worst = max(worst, _max_abs_rel(ref, got))

    assert worst == (0.0, 0.0), f"expected bit-identical sink, got {worst}"


def test_fused_kda_qproj_matches_eager_over_32_decode_steps():
    """MLX_VLM_GLM5_FUSED_KDA_QPROJ folds f_b_proj / g_b_proj into the kernel.

    The in-kernel GEMV transcribes MLX's affine ``qmv_quad`` partition, so it is
    bit-identical too -- a plain per-element dequant dot instead disagrees on
    ~0.01% of elements, which was enough to flip greedy tokens on 2 of 5 seeds.
    """
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    layer = _layer(config)
    _set_toggle(layer, True, qproj=True)
    if not layer._fused_kda_qproj_ready(mx.bfloat16, mx.float32):
        pytest.skip("projection fold unsupported for this quantization")

    eager_cache = _cache(config)
    qproj_cache = _clone(eager_cache)
    mx.random.seed(1357)
    steps = [
        mx.random.normal((1, 1, config.hidden_size)).astype(mx.bfloat16)
        for _ in range(32)
    ]
    mx.eval(steps)

    worst = (0.0, 0.0)
    for x in steps:
        _set_toggle(layer, False)
        eager_out = layer(x, None, eager_cache)
        _set_toggle(layer, True, qproj=True)
        qproj_out = layer(x, None, qproj_cache)
        mx.eval(eager_out, qproj_out, eager_cache.cache, qproj_cache.cache)
        for ref, got in (
            (eager_out, qproj_out),
            (eager_cache[0], qproj_cache[0]),
            (eager_cache[1], qproj_cache[1]),
        ):
            worst = max(worst, _max_abs_rel(ref, got))
    assert worst == (0.0, 0.0), f"expected bit-identical, got {worst}"


def test_fused_kda_qproj_declines_unsupported_quantization():
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    from mlx_vlm.models.glm5_next.fused_kda import fused_kda_qproj_supported

    config = _config()
    layer = _layer(config)
    fb, gb = layer.forget_gate.f_b_proj, layer.g_b_proj
    D = config.linear_head_dim
    assert fused_kda_qproj_supported(fb, gb, head_dim=D)
    # qmv_quad is only dispatched for head_dim in {64, 128} and bits == 8.
    assert not fused_kda_qproj_supported(fb, gb, head_dim=256)
    assert not fused_kda_qproj_supported(nn.Linear(D, D, bias=False), gb, head_dim=D)
    # An unfolded (dequantized) projection has no scales.
    assert not fused_kda_qproj_supported(fb, nn.Linear(D, D, bias=False), head_dim=D)


def test_fused_kda_declines_ineligible_shapes():
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    layer = _layer(config)
    _set_toggle(layer, True)
    assert layer._fused_kda_ready()

    cache = _cache(config)
    ref = mx.zeros((1, 1, config.hidden_size), mx.bfloat16)
    ok = dict(B=1, S=1, mask=None, cache=cache, gdn_sink=None, ref=ref)
    assert layer._fused_kda_eligible(**ok)
    assert not layer._fused_kda_eligible(**{**ok, "B": 2})
    assert not layer._fused_kda_eligible(**{**ok, "S": 8})
    # A bool [B, S] mask is supported now (batched decode always sends one);
    # anything else still declines.
    assert layer._fused_kda_eligible(**{**ok, "mask": mx.array([[True]])})
    assert not layer._fused_kda_eligible(
        **{**ok, "mask": mx.ones((1, 1), dtype=mx.float32)}
    )
    assert layer._fused_kda_eligible(**{**ok, "gdn_sink": []})  # capture variant
    assert not layer._fused_kda_eligible(**{**ok, "cache": None})
    assert not layer._fused_kda_eligible(**{**ok, "cache": ArraysCache(size=2)})
    assert not layer._fused_kda_eligible(**{**ok, "cache": _cache(config, batch=2)})
    assert not layer._fused_kda_eligible(
        **{**ok, "ref": mx.zeros((1, 1, config.hidden_size), mx.float32)}
    )


def test_fused_kda_prefill_then_decode_agrees():
    """Prefill (eager either way) followed by a fused decode step."""
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    layer = _layer(config)
    mx.random.seed(99)
    prompt = mx.random.normal((1, 16, config.hidden_size)).astype(mx.bfloat16)
    token = mx.random.normal((1, 1, config.hidden_size)).astype(mx.bfloat16)
    mx.eval(prompt, token)

    outs = []
    caches = []
    for on in (False, True):
        cache = ArraysCache(size=2)
        _set_toggle(layer, on)
        layer(prompt, None, cache)
        outs.append(layer(token, None, cache))
        caches.append(cache)
    mx.eval(outs, [c.cache for c in caches])
    assert _max_abs_rel(outs[0], outs[1]) == (0.0, 0.0)
    assert _max_abs_rel(caches[0][1], caches[1][1]) == (0.0, 0.0)


def test_toggle_off_uses_eager_path():
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    layer = _layer(config)
    _set_toggle(layer, False)
    assert not layer._fused_kda_ready()
    cache = _cache(config)
    x = mx.random.normal((1, 1, config.hidden_size)).astype(mx.bfloat16)
    before = mx.array(cache[1])
    out = layer(x, None, cache)
    mx.eval(out, cache.cache)
    assert not bool(mx.all(before == cache[1]))  # eager path still advanced state


def _bench(n_layers=34, iters=20, warmup=5):  # pragma: no cover - manual bench
    import time

    config = _config()
    layers = [_layer(config, seed=i) for i in range(n_layers)]
    h = mx.random.normal((1, 1, config.hidden_size)).astype(mx.bfloat16)
    hq = mx.random.normal(
        (1, 1, config.linear_num_heads * config.linear_head_dim)
    ).astype(mx.bfloat16)
    hd = mx.random.normal((1, 1, config.linear_head_dim)).astype(mx.bfloat16)
    mx.eval(h, hq, hd)
    for layer in layers:
        mx.eval(layer._fused_in_proj(h))

    def timeit(fn):
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1000.0

    def sweep(fn):
        def run():
            mx.eval([fn(i) for i in range(n_layers)])

        return run

    for tag, on, qp in (
        ("eager", False, False),
        ("fused", True, False),
        ("qproj", True, True),
    ):
        for layer in layers:
            _set_toggle(layer, on, qproj=qp)
        caches = [_cache(config, seed=100 + i) for i in range(n_layers)]
        full = timeit(sweep(lambda i: layers[i](h, None, caches[i])))
        gemv = (
            timeit(sweep(lambda i: layers[i]._fused_in_proj(h)[0]))
            + timeit(sweep(lambda i: layers[i].o_proj(hq)))
            + timeit(sweep(lambda i: layers[i].g_b_proj(hd)))
        )
        print(
            f"{tag:5s} full={full:7.3f} ms  gemv_floor={gemv:7.3f} ms  "
            f"chain={full - gemv:7.3f} ms"
        )


if __name__ == "__main__":  # pragma: no cover
    _bench()


@pytest.mark.parametrize("batch", _PARITY_BATCHES)
@pytest.mark.parametrize("mask_kind", ["none", "all-true", "ragged"])
def test_fused_kda_batched_matches_eager(batch, mask_kind):
    """Batched decode, across the three mask regimes it actually sees.

    BatchGenerator sets left_padding on the ArraysCache even for a uniform-length
    batch, so batched decode always arrives with a bool mask -- all-true once the
    rows have caught up, ragged while some are still left-padded.  The eager path
    uses it only to zero the pre-conv input of a padded row (history and the
    recurrence still run), and the kernel does the same, per (batch row, head)
    threadgroup.  Every reduction is per (batch, head) and unchanged, so this is
    exact for the same reason the B=1 path is.
    """
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    _require_device(config, "base")
    layer = _layer(config)
    n_steps = 32
    masks = _masks(mask_kind, batch, n_steps)
    mx.random.seed(20260831)
    steps = [
        mx.random.normal((batch, 1, config.hidden_size)).astype(mx.bfloat16)
        for _ in range(n_steps)
    ]
    mx.eval(steps)

    eager_cache = _cache(config, batch=batch)
    fused_cache = _clone(eager_cache)
    _allow_batch(batch)
    worst = (0.0, 0.0)
    for x, m in zip(steps, masks):
        _set_toggle(layer, False)
        eager_out = layer(x, m, eager_cache)
        _set_toggle(layer, True)
        assert layer._fused_kda_ready()
        assert layer._fused_kda_eligible(batch, 1, m, fused_cache, None, x)
        fused_out = layer(x, m, fused_cache)
        mx.eval(eager_out, fused_out, eager_cache.cache, fused_cache.cache)
        for ref, got in (
            (eager_out, fused_out),
            (eager_cache[0], fused_cache[0]),
            (eager_cache[1], fused_cache[1]),
        ):
            worst = max(worst, _max_abs_rel(ref, got))
    assert worst == (0.0, 0.0), f"expected bit-identical, got {worst}"


def test_fused_kda_batch_gate():
    """The batch cap, the cache-shape contract and the mask contract."""
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    _require_device(config, "base")
    layer = _layer(config)
    _set_toggle(layer, True)
    cache = _cache(config, batch=4)
    ref = mx.zeros((4, 1, config.hidden_size), mx.bfloat16)
    ok = dict(B=4, S=1, mask=None, cache=cache, gdn_sink=None, ref=ref)
    assert layer._fused_kda_eligible(**ok)
    assert layer._fused_kda_eligible(**{**ok, "mask": mx.ones((4, 1), dtype=mx.bool_)})
    cap = glm5._FUSED_KDA_MAX_BATCH
    assert layer._fused_kda_eligible(
        B=cap,
        S=1,
        mask=mx.ones((cap, 1), dtype=mx.bool_),
        cache=_cache(config, batch=cap),
        gdn_sink=None,
        ref=mx.zeros((cap, 1, config.hidden_size), mx.bfloat16),
    )
    over = glm5._FUSED_KDA_MAX_BATCH + 1
    assert not layer._fused_kda_eligible(
        B=over,
        S=1,
        mask=None,
        cache=_cache(config, batch=over),
        gdn_sink=None,
        ref=mx.zeros((over, 1, config.hidden_size), mx.bfloat16),
    )
    # cache batch must match the request
    assert not layer._fused_kda_eligible(**{**ok, "cache": _cache(config, batch=2)})
    assert not layer._fused_kda_eligible(
        **{**ok, "mask": mx.ones((4, 1), dtype=mx.float32)}
    )


@pytest.mark.parametrize("batch", [2, 8, 16, 32])
def test_fused_kda_batched_capture_matches_eager_gdn_sink(batch):
    """The speculative sink must be exact at B > 1 too (rollback replays it)."""
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    from mlx_vlm.models.gated_delta import gated_delta_update

    config = _config()
    _require_device(config, "base")
    layer = _layer(config)
    eager_cache = _cache(config, batch=batch)
    fused_cache = _clone(eager_cache)
    _allow_batch(batch)
    names = ["q", "k", "v", "a", "b", "A_log", "dt_bias", "state", "conv_input"]
    mx.random.seed(4242)
    worst = (0.0, 0.0)
    for _ in range(8):
        x = mx.random.normal((batch, 1, config.hidden_size)).astype(mx.bfloat16)
        mx.eval(x)
        eager_sink, fused_sink = [], []
        _set_toggle(layer, False)
        eager_out = layer(x, None, eager_cache, gdn_sink=eager_sink)
        _set_toggle(layer, True)
        fused_out = layer(x, None, fused_cache, gdn_sink=fused_sink)

        def replay(e):
            return gated_delta_update(
                e[0][:, :1],
                e[1][:, :1],
                e[2][:, :1],
                e[3][:, :1],
                e[4][:, :1],
                e[5],
                e[6],
                state=e[7],
                lower_bound=e[10],
            )[1]

        eager_roll, fused_roll = replay(eager_sink[0]), replay(fused_sink[0])
        mx.eval(
            eager_out,
            fused_out,
            eager_cache.cache,
            fused_cache.cache,
            eager_sink[0][:9],
            fused_sink[0][:9],
            eager_roll,
            fused_roll,
        )
        pairs = [
            (eager_out, fused_out),
            (eager_cache[1], fused_cache[1]),
            (eager_roll, fused_roll),
        ]
        for i, name in enumerate(names):
            r, g = eager_sink[0][i], fused_sink[0][i]
            assert r.shape == g.shape, f"{name}: {r.shape} vs {g.shape}"
            pairs.append((r, g))
        for r, g in pairs:
            worst = max(worst, _max_abs_rel(r, g))
    assert worst == (0.0, 0.0), f"expected bit-identical sink, got {worst}"


# --------------------------------------------------------------------------
# Why the batch cap is a policy number and not a kernel limit.
#
# _FUSED_KDA_MAX_BATCH says how wide a batch parity has been run to.  That is
# only an honest thing to say if widening the batch really does leave the
# compiled pipeline alone -- otherwise the number would be hiding a threadgroup
# budget or a register cliff, and raising it would be shipping a degraded
# kernel.  The two tests below check that structurally, on the CPU, so the claim
# survives a future edit to the kernel rather than resting on a code comment.
_TG_LIMIT_BYTES = 32768  # Apple GPU threadgroup memory budget (Apple7 onward)


def _kernel_sources():
    from mlx_vlm.models.glm5_next import fused_kda as fk

    return {
        "base": fk._SOURCE,
        "capture": fk._SOURCE + fk._SINK_SOURCE,
        "qproj": fk._qproj_source(fk._SOURCE),
    }


def _strip_comments(src):
    # The kernel sources contain no string or character literals, so removing
    # /* */ and // spans is exact here.
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return "\n".join(re.sub(r"//.*$", "", line) for line in src.splitlines())


def test_fused_kda_kernel_source_is_batch_agnostic():
    """No variant's *code* may mention B; only grid.z and the buffer extents may.

    This is the whole basis for one threadgroup probe covering every batch size
    and for the cap being raisable by measurement alone.  ``B`` appears in the
    sources five times, every one of them inside a comment describing a buffer
    shape -- if a future edit makes a real expression depend on B, the compiled
    pipeline stops being batch-invariant and this fails.
    """
    ident_b = re.compile(r"(?<![A-Za-z0-9_])B(?![A-Za-z0-9_])")
    for name, src in _kernel_sources().items():
        offending = [
            line.strip()
            for line in _strip_comments(src).splitlines()
            if ident_b.search(line)
        ]
        assert not offending, f"{name} kernel reads B in code: {offending}"
        # ... and it is genuinely mentioned in the prose, so the check above is
        # not passing because the regex is broken.
        assert ident_b.search(src), f"{name}: regex found no B at all"


def test_fused_kda_threadgroup_footprint_is_b_independent_and_fits():
    """Every threadgroup allocation is sized by the head dim, never by the batch.

    One threadgroup serves one (batch row, head) pair, so a wider batch buys more
    threadgroups and not bigger ones.  At the live GLM-5.3-Flash dims that is
    3084 B for the base and capture pipelines and 3596 B with the projection
    fold, against a 32 KiB budget -- the same at B=1 and at B=16.
    """
    D = _CFG["linear_attn_config"]["head_dim"]
    decl = re.compile(r"threadgroup\s+float\s+(\w+)\s*\[([^\]]+)\]")
    expected = {"base": 3084, "capture": 3084, "qproj": 3596}
    for name, src in _kernel_sources().items():
        total = 0
        for var, extent in decl.findall(_strip_comments(src)):
            extent = extent.strip()
            assert extent == "D" or extent.isdigit(), (
                f"{name}: threadgroup array {var}[{extent}] is not sized by the "
                "head dim or a constant -- a batch-sized one would break the "
                "one-threadgroup-per-(row, head) mapping"
            )
            total += (D if extent == "D" else int(extent)) * 4
        assert total == expected[name], f"{name}: {total} B, expected {expected[name]}"
        assert total <= _TG_LIMIT_BYTES


def test_fused_kda_batch_cap_is_env_tunable():
    """The cap can be moved without editing the model (and is the A/B lever).

    Read at import, like the gather-gate constants next to it, so this needs a
    fresh interpreter rather than a reload -- reloading would swap the layer
    class out from under the rest of the session.
    """
    import subprocess
    import sys

    prog = (
        "import mlx_vlm.models.glm5_next.language as g\nprint(g._FUSED_KDA_MAX_BATCH)"
    )
    for env_value, expect in ((None, "16"), ("8", "8"), ("32", "32"), ("2", "2")):
        env = dict(os.environ)
        env.pop("MLX_VLM_GLM5_FUSED_KDA_MAX_BATCH", None)
        if env_value is not None:
            env["MLX_VLM_GLM5_FUSED_KDA_MAX_BATCH"] = env_value
        out = subprocess.run(
            [sys.executable, "-c", prog],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        assert out.stdout.strip() == expect, out.stderr


def test_fused_kda_qproj_fold_still_declines_a_wide_batch():
    """Raising the base cap must not drag the projection fold along with it.

    The fold inverts as the batch grows (it re-reads the weights per (row, head)
    threadgroup while mx.quantized_matmul reads them once for all rows), so it
    keeps its own, much lower cap.  Nothing about the B=16 extension changes it.
    """
    assert glm5._FUSED_KDA_QPROJ_MAX_BATCH == 2
    assert glm5._FUSED_KDA_QPROJ_MAX_BATCH < glm5._FUSED_KDA_MAX_BATCH


def test_fused_kda_batch_cap_is_covered_by_parity():
    """The shipped cap may only be a width this file actually compares.

    The cap is a claim about evidence -- "eager and fused agree this far" -- so
    it is worth making it impossible to raise the number without adding the
    evidence.  Widening _PARITY_BATCHES costs one line and 32 carried decode
    steps per mask regime; that is the price of moving the default.
    """
    assert glm5._FUSED_KDA_MAX_BATCH in (1,) + _PARITY_BATCHES, (
        f"cap is {glm5._FUSED_KDA_MAX_BATCH} but parity only runs at "
        f"{_PARITY_BATCHES}"
    )
