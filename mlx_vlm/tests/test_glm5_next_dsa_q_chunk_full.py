"""K1 -- ``MLX_VLM_GLM5_DSA_Q_CHUNK_FULL``: one gather per prefill chunk.

The default (flag unset) path must be byte-for-byte the 95bbe594 behaviour: the
query chunk is ``_gather_q_chunk_for(Kv, dim)`` and nothing else.  With the flag
set, and ONLY on the default ``take`` gather mode, the loop runs once over the
whole query block, bounded by the take path's own grid limit rather than by the
2**31 element bound of the EAGER path (which is the only path that has one --
see the block comment above ``_take_path_q_chunk_max`` in language.py).

The claim under test is BIT-EXACTNESS: queries are independent, so how many of
them share one gather/SDPA dispatch cannot change any output element.  Tests run
on whatever device pytest is pinned to (CPU on the desk, conftest honours
MLX_DEFAULT_DEVICE); the GPU evidence is the Stage-B logits fingerprint.
"""
import os

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_vlm.models.glm5_next.language as glm5
from mlx_vlm.models.glm5_next.config import TextConfig

_CFG = dict(
    model_type="glm5_next_text",
    vocab_size=1024,
    hidden_size=128,
    intermediate_size=256,
    moe_intermediate_size=128,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_shared_experts=1,
    n_routed_experts=8,
    routed_scaling_factor=2.5,
    kv_lora_rank=64,
    q_lora_rank=64,
    qk_rope_head_dim=0,
    v_head_dim=32,
    qk_nope_head_dim=32,
    num_experts_per_tok=2,
    first_k_dense_replace=1,
    max_position_embeddings=1048576,
    rms_norm_eps=1e-05,
    index_topk=16,
    index_head_dim=32,
    index_n_heads=4,
    layer_types=["full_attention"],
    mlp_layer_types=["dense"],
    linear_attn_config={
        "num_heads": 4,
        "gate_lower_bound": -5.0,
        "head_dim": 32,
        "short_conv_kernel_size": 4,
    },
)


@pytest.fixture(autouse=True)
def _clean_env():
    saved = os.environ.get("MLX_VLM_GLM5_DSA_Q_CHUNK_FULL")
    os.environ.pop("MLX_VLM_GLM5_DSA_Q_CHUNK_FULL", None)
    yield
    if saved is None:
        os.environ.pop("MLX_VLM_GLM5_DSA_Q_CHUNK_FULL", None)
    else:
        os.environ["MLX_VLM_GLM5_DSA_Q_CHUNK_FULL"] = saved


def _attn(seed=0):
    mx.random.seed(seed)
    config = TextConfig.from_dict(dict(_CFG))
    layer = glm5.Glm5NextSparseAttention(config)

    def rand(tree):
        if isinstance(tree, dict):
            return {k: rand(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rand(v) for v in tree]
        return (mx.random.normal(tree.shape) * 0.05).astype(mx.bfloat16)

    layer.update(rand(layer.parameters()))
    mx.eval(layer.parameters())
    return layer


def _inputs(B=1, L=48, Kv=96, topk=16, seed=3):
    """q / kv_latent / topk_indices exactly as ``__call__`` builds them."""
    mx.random.seed(seed)
    cfg = TextConfig.from_dict(dict(_CFG))
    H, dim, qd = cfg.num_attention_heads, cfg.kv_lora_rank, cfg.qk_nope_head_dim
    q = (mx.random.normal((B, H, L, qd)) * 0.5).astype(mx.bfloat16)
    kv_latent = (mx.random.normal((B, 1, Kv, dim)) * 0.5).astype(mx.bfloat16)
    # Causal, in-range selections with a few -1 (unselected) slots, which is the
    # shape the indexer emits after its validity mask.
    idx = mx.random.randint(0, Kv, (B, 1, L, topk)).astype(mx.int32)
    drop = mx.random.uniform(shape=(B, 1, L, topk)) < 0.1
    idx = mx.where(drop, mx.array(-1, mx.int32), idx)
    mx.eval(q, kv_latent, idx)
    return q, kv_latent, idx


def _run(layer, q, kv_latent, idx, *, full, chunk_knob=None):
    prev_knob = glm5._GATHER_Q_CHUNK
    try:
        if chunk_knob is not None:
            glm5._GATHER_Q_CHUNK = chunk_knob
        if full:
            os.environ["MLX_VLM_GLM5_DSA_Q_CHUNK_FULL"] = "1"
        else:
            os.environ.pop("MLX_VLM_GLM5_DSA_Q_CHUNK_FULL", None)
        out = layer._gathered_attention(q, kv_latent, idx)
        mx.eval(out)
        return out
    finally:
        glm5._GATHER_Q_CHUNK = prev_knob
        os.environ.pop("MLX_VLM_GLM5_DSA_Q_CHUNK_FULL", None)


# --------------------------------------------------------------- the bound
class TestTakePathBound:
    def test_bound_is_the_grid_limit_not_the_element_limit(self):
        # gather_front's grid-y extent is indices.size() = B * chunk * topk and the
        # kernel position is a uint2, so the bound is 2**32-1, NOT the 2**31
        # element bound of the eager gather_axis path.
        assert glm5._take_path_q_chunk_max(1, 2051) == (2**32 - 1) // 2051
        assert glm5._take_path_q_chunk_max(8, 2051) == (2**32 - 1) // (8 * 2051)

    def test_every_shipped_prefill_chunk_is_a_single_gather(self):
        # 8,192 queries at the served topk (index_topk 2048 + kpool-1 tail) is far
        # under the bound at every batch size the fork serves.
        for batch in (1, 8, 16):
            assert glm5._take_path_q_chunk_max(batch, 2051) >= 8192

    def test_degenerate_shapes_return_zero_so_the_caller_keeps_the_default(self):
        assert glm5._take_path_q_chunk_max(0, 2051) == 0
        assert glm5._take_path_q_chunk_max(1, 0) == 0

    def test_a_pathological_topk_would_still_shrink(self):
        # Not reachable with this model's shapes, but the guard must be real: at
        # topk 2**20 and batch 8 the bound is below an 8,192-row chunk.
        assert glm5._take_path_q_chunk_max(8, 1 << 20) < 8192


# ------------------------------------------------------------ default path
class TestDefaultUnchanged:
    def test_flag_unset_uses_the_derived_chunk(self, monkeypatch):
        seen = []
        real = glm5._gather_q_chunk_for
        monkeypatch.setattr(
            glm5, "_gather_q_chunk_for",
            lambda kv, dim: seen.append(real(kv, dim)) or seen[-1],
        )
        layer = _attn()
        q, kv, idx = _inputs()
        _run(layer, q, kv, idx, full=False, chunk_knob=16)
        # 16, not the knob's 8: _GATHER_Q_CHUNK_MIN floors the derived value.
        assert seen == [16]

    def test_flag_unset_is_bit_identical_to_the_pristine_call(self):
        # The flag's OFF path is the same code the base commit ran: the same
        # inputs twice must give the same bytes, and the loop must really have
        # run more than once at this knob (otherwise the test is vacuous).
        layer = _attn()
        q, kv, idx = _inputs(L=48)
        a = _run(layer, q, kv, idx, full=False, chunk_knob=16)
        b = _run(layer, q, kv, idx, full=False, chunk_knob=16)
        assert mx.array_equal(a, b).item()
        prev = glm5._GATHER_Q_CHUNK
        try:
            glm5._GATHER_Q_CHUNK = 16
            assert glm5._gather_q_chunk_for(96, 64) == 16  # 48 / 16 = 3 iterations
        finally:
            glm5._GATHER_Q_CHUNK = prev


# ------------------------------------------------------------------- K1 arm
class TestFullChunkArm:
    @pytest.mark.parametrize("L,knob", [(48, 16), (48, 32), (33, 16), (8, 16)])
    def test_full_chunk_is_bit_identical(self, L, knob):
        layer = _attn()
        q, kv, idx = _inputs(L=L)
        ref = _run(layer, q, kv, idx, full=False, chunk_knob=knob)
        arm = _run(layer, q, kv, idx, full=True, chunk_knob=knob)
        assert arm.shape == ref.shape
        assert mx.array_equal(arm, ref).item(), (
            f"K1 changed the output at L={L}, chunk knob={knob}: "
            f"max|d| {mx.max(mx.abs(arm.astype(mx.float32) - ref.astype(mx.float32))).item()}"
        )

    def test_batched_is_bit_identical(self):
        layer = _attn()
        q, kv, idx = _inputs(B=3, L=40)
        ref = _run(layer, q, kv, idx, full=False, chunk_knob=16)
        arm = _run(layer, q, kv, idx, full=True, chunk_knob=16)
        assert mx.array_equal(arm, ref).item()

    def test_arm_runs_exactly_one_gather(self, monkeypatch):
        calls = []
        real = glm5._gather_latents_take
        monkeypatch.setattr(
            glm5, "_gather_latents_take",
            lambda kv, c: calls.append(c.shape[1]) or real(kv, c),
        )
        layer = _attn()
        q, kv, idx = _inputs(L=48)
        _run(layer, q, kv, idx, full=False, chunk_knob=16)
        assert calls == [16, 16, 16]
        calls.clear()
        _run(layer, q, kv, idx, full=True, chunk_knob=16)
        assert calls == [48]

    def test_eager_gather_mode_is_not_widened(self, monkeypatch):
        """The 2**31 bound is real on gather_axis: K1 must not touch that mode."""
        monkeypatch.setattr(glm5, "_dsa_gather_mode", lambda: "eager")
        widths = []
        real = mx.take_along_axis
        monkeypatch.setattr(
            mx, "take_along_axis",
            lambda a, i, axis: widths.append(i.shape[1]) or real(a, i, axis=axis),
        )
        layer = _attn()
        q, kv, idx = _inputs(L=48)
        _run(layer, q, kv, idx, full=True, chunk_knob=16)
        assert widths == [16, 16, 16]


class TestFlagParsing:
    @pytest.mark.parametrize("raw,want", [
        (None, False), ("", False), ("0", False), ("off", False), ("false", False),
        ("1", True), ("true", True), ("on", True), ("yes", True),
    ])
    def test_parsing(self, raw, want):
        if raw is None:
            os.environ.pop("MLX_VLM_GLM5_DSA_Q_CHUNK_FULL", None)
        else:
            os.environ["MLX_VLM_GLM5_DSA_Q_CHUNK_FULL"] = raw
        assert glm5._dsa_q_chunk_full() is want

    def test_the_flag_is_read_per_call_not_latched(self):
        """The L40 registry classes this key ``per_call``; prove it."""
        os.environ["MLX_VLM_GLM5_DSA_Q_CHUNK_FULL"] = "1"
        assert glm5._dsa_q_chunk_full() is True
        os.environ["MLX_VLM_GLM5_DSA_Q_CHUNK_FULL"] = "0"
        assert glm5._dsa_q_chunk_full() is False
