"""V7/L36: the stock-op chunk-parallel KDA prefill scan, against the scan it replaces.

The claim under test is NOT bit-exactness -- it cannot be.  ``chunk_kda`` sums
the same recurrence in a different order (chunked cumulative gates, the delta
rule re-expressed as one triangular solve per chunk, a WY/UT transform), so the
contract is: the outputs and the CARRIED STATE agree with
``gated_delta_update``'s sequential reference to fp32 round-off, and the
disagreement does NOT grow with the number of chunks.  That last one is the
falsifier that matters: a chunk form whose state drifts per chunk would pass at
S=256 and destroy a 32k prefill.

Everything here runs on CPU.  ``gated_delta_update`` routes to
``gated_delta_ops`` (the ops reference the Metal scan is transcribed from) off
the GPU, which is exactly the oracle we want.

Tolerances are stated as ratios to the reference tensor's own max magnitude, not
elementwise: an output element that is 1e-9 in a tensor whose scale is 1e-1
carries no information, and an elementwise rtol would be a test of denormals.
"""
import math

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_vlm.models.glm5_next.chunk_kda as C
import mlx_vlm.models.glm5_next.language as glm5
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.gated_delta import gated_delta_update
from mlx_vlm.models.glm5_next.config import TextConfig

LB = -5.0

# The shipped scan is fp32-accumulate over bf16 operands; the chunk form is fp32
# throughout.  8e-6 of the tensor's own max is ~2^-17 -- three bits inside bf16's
# own resolution, i.e. below what either path can claim to represent.
ATOL_REL = 8e-6


def _rel(got, ref):
    """max|got-ref| / max|ref| -- both cast to fp32 first."""
    got = got.astype(mx.float32)
    ref = ref.astype(mx.float32)
    denom = float(mx.abs(ref).max().item())
    return float(mx.abs(got - ref).max().item()) / max(denom, 1e-30)


def _l2norm(x, eps=1e-6):
    return x * mx.rsqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def _inputs(B, S, H, D, *, gate="spread", seed=0, dtype=mx.float32):
    """q/k/v in the post-glue convention (L2-normalised, q pre-scaled) + gates.

    ``gate`` selects the log-gate regime:
      spread -- per-(head, channel) offsets so the log-gate covers [-5, ~0] with
                p10 near the -4.37 nats/token the live model rests at;
      resting-- every channel near -4.37 (the p0.1 the design note calls out);
      floor  -- every channel pinned at the -5.0 lower bound (worst case for the
                sub-block exponent range);
      open   -- a ~= 0, i.e. no decay at all (worst case for the triangular
                solve: (I+T) is furthest from the identity).
    """
    mx.random.seed(seed)
    q = (_l2norm(mx.random.normal((B, S, H, D))) * (D**-0.5)).astype(dtype)
    k = _l2norm(mx.random.normal((B, S, H, D))).astype(dtype)
    v = (mx.random.normal((B, S, H, D)) * 0.5).astype(dtype)
    if gate == "spread":
        a = mx.random.normal((B, S, H, D)) * 0.8 + (mx.random.normal((1, 1, H, D)) * 2.5 + 1.0)
    else:
        target = {"resting": -4.37, "floor": -4.999, "open": -1e-6}[gate] / LB
        z = math.log(target / (1.0 - target))
        a = mx.random.normal((B, S, H, D)) * (0.7 if gate != "open" else 0.0) + z
    b = mx.random.normal((B, S, H))
    state = mx.random.normal((B, H, D, D)) * 0.1
    return dict(
        q=q, k=k, v=v, a=a, b=b,
        A_log=mx.zeros((H, 1)), dt_bias=mx.zeros((H, D)), state=state,
    )


def _both(d, **kw):
    ref = gated_delta_update(
        d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"],
        state=d["state"], lower_bound=LB,
    )
    got = C.chunk_kda_update(
        d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"],
        state=d["state"], lower_bound=LB, **kw,
    )
    return got, ref


# --------------------------------------------------------------------------- #
# equivalence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("c", [16, 32, 64, 128])
def test_matches_sequential_reference_across_chunk_sizes(c):
    d = _inputs(1, 384, 4, 64)
    got, ref = _both(d, chunk=c, sub=16)
    assert got is not None
    assert _rel(got[0], ref[0]) < ATOL_REL
    assert _rel(got[1], ref[1]) < ATOL_REL


@pytest.mark.parametrize("gate", ["spread", "resting", "floor", "open"])
def test_matches_sequential_reference_across_gate_regimes(gate):
    d = _inputs(1, 256, 4, 64, gate=gate)
    got, ref = _both(d, chunk=64, sub=16)
    assert got is not None
    assert not bool(mx.isnan(got[0]).any().item())
    assert not bool(mx.isinf(got[0]).any().item())
    assert _rel(got[0], ref[0]) < ATOL_REL
    assert _rel(got[1], ref[1]) < ATOL_REL


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_matches_sequential_reference_in_low_precision_inputs(dtype):
    # The glue hands q/k/v down in the model dtype (language.py:1180-1182); both
    # paths then accumulate in fp32, so the gap must stay at the OUTPUT cast's
    # resolution, not grow.
    d = _inputs(1, 256, 4, 64, dtype=dtype)
    got, ref = _both(d, chunk=64, sub=16)
    assert got is not None
    assert got[0].dtype == dtype and got[1].dtype == mx.float32
    assert _rel(got[0], ref[0]) < 2e-4  # one bf16/fp16 ulp of the tensor max
    assert _rel(got[1], ref[1]) < ATOL_REL


@pytest.mark.parametrize("B,H,D", [(1, 8, 128), (2, 4, 64)])
def test_matches_sequential_reference_across_shapes(B, H, D):
    d = _inputs(B, 256, H, D)
    got, ref = _both(d, chunk=64, sub=16)
    assert got is not None
    assert got[0].shape == (B, 256, H, D)
    assert got[1].shape == (B, H, D, D)
    assert _rel(got[0], ref[0]) < ATOL_REL
    assert _rel(got[1], ref[1]) < ATOL_REL


@pytest.mark.parametrize("S", [130, 200, 383])
def test_partial_last_chunk_is_padded_not_wrong(S):
    # Padded tokens get log_g = 0 and beta = 0, so they neither decay nor write
    # the state; if that were wrong the CARRIED STATE would move, not just y.
    d = _inputs(1, S, 4, 64)
    got, ref = _both(d, chunk=64, sub=16)
    assert got is not None
    assert got[0].shape[1] == S
    assert _rel(got[0], ref[0]) < ATOL_REL
    assert _rel(got[1], ref[1]) < ATOL_REL


# --------------------------------------------------------------------------- #
# the carry -- the contract this lever lives or dies on
# --------------------------------------------------------------------------- #
def test_state_carries_across_chunks():
    """Two prefill steps that hand the state over == one step of both halves."""
    S = 256
    d = _inputs(1, 2 * S, 4, 64, seed=3)
    ref = gated_delta_update(
        d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"],
        state=d["state"], lower_bound=LB,
    )
    first = C.chunk_kda_update(
        d["q"][:, :S], d["k"][:, :S], d["v"][:, :S], d["a"][:, :S], d["b"][:, :S],
        d["A_log"], d["dt_bias"], state=d["state"], lower_bound=LB, chunk=64, sub=16,
    )
    second = C.chunk_kda_update(
        d["q"][:, S:], d["k"][:, S:], d["v"][:, S:], d["a"][:, S:], d["b"][:, S:],
        d["A_log"], d["dt_bias"], state=first[1], lower_bound=LB, chunk=64, sub=16,
    )
    y = mx.concatenate([first[0], second[0]], axis=1)
    assert _rel(y, ref[0]) < ATOL_REL
    assert _rel(second[1], ref[1]) < ATOL_REL


def test_state_drift_does_not_grow_with_chunk_count():
    """KILL condition for the lever, as a test: error flat in the chunk count.

    A per-chunk state error would compound over the 128 chunks of one 8192-token
    prefill step (and 512 of a 32k prompt).  Measured here from 2 to 128 chunks;
    the bound is the same constant at both ends, no slope allowed.
    """
    errs = {}
    for n in (2, 8, 32, 128):
        d = _inputs(1, 64 * n, 1, 32, seed=1)
        got, ref = _both(d, chunk=64, sub=16)
        errs[n] = _rel(got[1], ref[1])
    assert max(errs.values()) < ATOL_REL, errs
    # no growth: 64x more chunks may not cost more than 4x the error
    assert errs[128] <= 4.0 * max(errs[2], 1e-9), errs


# --------------------------------------------------------------------------- #
# numerics guard: the sub-block exponent bound
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "lower_bound,expect",
    [(-5.0, (64, 16)), (-8.0, (64, 8)), (-10.0, (64, 8)), (-24.0, (64, 2)),
     (-100.0, None), (None, None)],
)
def test_chunk_geometry_respects_the_fp32_exponent_bound(lower_bound, expect):
    # e^{c_sub * |lower_bound|} must stay inside fp32; the sub-block shrinks (or
    # the lever declines) rather than emitting inf * 0 = NaN on the diagonal.
    assert C.chunk_geometry(64, 16, lower_bound) == expect
    if expect is not None:
        assert expect[1] * abs(lower_bound) <= C._MAX_SUB_EXPONENT


def test_chunk_geometry_rejects_indivisible_or_zero():
    assert C.chunk_geometry(64, 24, -5.0) is None
    assert C.chunk_geometry(0, 16, -5.0) is None
    assert C.chunk_geometry(64, 0, -5.0) is None


def test_strict_lower_inverse_is_exact():
    mx.random.seed(7)
    n = 16
    idx = mx.arange(n)
    # entries at the scale the real matrix has: T_ij = beta_i * W_ij with
    # |W_ij| <= 1 and beta in (0, 1).
    t = mx.where(idx[:, None] > idx[None, :], mx.random.normal((3, n, n)) * 0.3, 0.0)
    inv = C._strict_lower_inverse(t, n)
    prod = inv @ (mx.eye(n) + t)
    assert float(mx.abs(prod - mx.eye(n)).max().item()) < 1e-4


def test_no_overflow_when_every_channel_sits_on_the_gate_floor():
    d = _inputs(1, 256, 4, 64, gate="floor")
    got, ref = _both(d, chunk=128, sub=16)  # widest chunk = widest exponent span
    assert got is not None
    assert not bool(mx.isnan(got[1]).any().item())
    assert _rel(got[1], ref[1]) < ATOL_REL


# --------------------------------------------------------------------------- #
# declines -- every case the chunk form does not cover keeps the shipped path
# --------------------------------------------------------------------------- #
def test_declines_masked_calls():
    d = _inputs(1, 256, 4, 64)
    out = C.chunk_kda_update(
        d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"],
        state=d["state"], mask=mx.ones((1, 256), dtype=mx.bool_), lower_bound=LB,
        chunk=64, sub=16,
    )
    assert out is None


def test_declines_unbounded_gate_and_short_sequences():
    d = _inputs(1, 256, 4, 64)
    args = (d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"])
    assert C.chunk_kda_update(*args, state=d["state"], lower_bound=None) is None
    short = _inputs(1, 64, 4, 64)
    assert C.chunk_kda_update(
        short["q"], short["k"], short["v"], short["a"], short["b"],
        short["A_log"], short["dt_bias"], state=short["state"], lower_bound=LB,
        chunk=64, sub=16,
    ) is None


def test_env_default_is_fused(monkeypatch):
    monkeypatch.setattr(C, "_MODE_ENV", None)
    monkeypatch.delenv("MLX_VLM_GLM5_KDA_PREFILL_MODE", raising=False)
    assert C.kda_prefill_mode() == "fused"
    monkeypatch.setattr(C, "_MODE_ENV", None)
    monkeypatch.setenv("MLX_VLM_GLM5_KDA_PREFILL_MODE", "CHUNK")
    assert C.kda_prefill_mode() == "chunk"


def test_env_chunk_size_defaults(monkeypatch):
    monkeypatch.setattr(C, "_CHUNK_ENV", None)
    monkeypatch.setattr(C, "_SUB_ENV", None)
    monkeypatch.delenv("MLX_VLM_GLM5_KDA_CHUNK", raising=False)
    monkeypatch.delenv("MLX_VLM_GLM5_KDA_CHUNK_SUB", raising=False)
    assert C.kda_chunk_size() == 64
    assert C.kda_chunk_sub() == 16


# --------------------------------------------------------------------------- #
# the wiring in language.py
# --------------------------------------------------------------------------- #
_H, _D, _K = 4, 128, 4
_CFG = dict(
    model_type="glm5_next_text", vocab_size=1024, hidden_size=512,
    intermediate_size=1024, moe_intermediate_size=512, num_hidden_layers=1,
    num_attention_heads=8, num_key_value_heads=8, n_shared_experts=1,
    n_routed_experts=8, routed_scaling_factor=2.5, kv_lora_rank=128,
    q_lora_rank=256, qk_rope_head_dim=0, v_head_dim=64, qk_nope_head_dim=64,
    num_experts_per_tok=2, first_k_dense_replace=1,
    max_position_embeddings=1048576, rms_norm_eps=1e-05, index_topk=64,
    index_head_dim=128, index_n_heads=8, layer_types=["linear_attention"],
    mlp_layer_types=["dense"],
    linear_attn_config={"num_heads": _H, "gate_lower_bound": -5.0,
                        "head_dim": _D, "short_conv_kernel_size": _K},
)


def _layer(seed=0):
    mx.random.seed(seed)
    layer = glm5.Glm5NextLinearAttention(TextConfig.from_dict(dict(_CFG)))

    def rand(tree):
        if isinstance(tree, dict):
            return {k: rand(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rand(v) for v in tree]
        return (mx.random.normal(tree.shape) * 0.05).astype(mx.bfloat16)

    layer.update(rand(layer.parameters()))
    layer.conv1d.weight = (mx.random.normal(layer.conv1d.weight.shape) * 0.5).astype(mx.bfloat16)
    layer.forget_gate.A_log = (mx.random.normal((_H,)) * 0.5).astype(mx.float32)
    layer.forget_gate.dt_bias = (mx.random.normal((_H * _D,)) * 0.5).astype(mx.float32)
    layer.o_norm.weight = (mx.ones((_D,)) + 0.02 * mx.random.normal((_D,))).astype(mx.bfloat16)
    mx.eval(layer.parameters())
    return layer


def _cache(seed=1):
    mx.random.seed(seed)
    cache = ArraysCache(size=2)
    cache[0] = (mx.random.normal((1, _K - 1, 3 * _H * _D)) * 0.3).astype(mx.bfloat16)
    cache[1] = (mx.random.normal((1, _H, _D, _D)) * 0.05).astype(mx.float32)
    mx.eval(cache[0], cache[1])
    return cache


def _forward(layer, x, mode, monkeypatch):
    monkeypatch.setattr(C, "_MODE_ENV", mode)
    cache = _cache()
    y = layer(x, cache=cache)
    mx.eval(y, cache[0], cache[1])
    return y, cache


def test_layer_chunk_mode_matches_the_shipped_prefill_path(monkeypatch):
    layer = _layer()
    x = (mx.random.normal((1, 256, 512)) * 0.5).astype(mx.bfloat16)
    y_ref, c_ref = _forward(layer, x, "fused", monkeypatch)
    y_chunk, c_chunk = _forward(layer, x, "chunk", monkeypatch)
    assert _rel(y_chunk, y_ref) < 3e-3      # bf16 o_proj output, one ulp is 8e-3
    assert _rel(c_chunk[1], c_ref[1]) < ATOL_REL
    assert bool(mx.array_equal(c_chunk[0], c_ref[0]).item())  # conv state untouched


def test_layer_default_mode_leaves_the_shipped_path_bit_identical(monkeypatch):
    layer = _layer()
    x = (mx.random.normal((1, 256, 512)) * 0.5).astype(mx.bfloat16)
    monkeypatch.setattr(C, "_MODE_ENV", None)
    monkeypatch.delenv("MLX_VLM_GLM5_KDA_PREFILL_MODE", raising=False)
    cache = _cache()
    y = layer(x, cache=cache)
    monkeypatch.setattr(C, "_MODE_ENV", "fused")
    cache2 = _cache()
    y2 = layer(x, cache=cache2)
    assert bool(mx.array_equal(y, y2).item())
    assert bool(mx.array_equal(cache[1], cache2[1]).item())


def test_prefill_kernel_declines_when_chunk_mode_is_on(monkeypatch):
    layer = _layer()
    monkeypatch.setattr(C, "_MODE_ENV", "chunk")
    cache = _cache()
    ref = mx.zeros((1, 256, _H * _D), dtype=mx.bfloat16)
    assert not layer._fused_kda_prefill_eligible(1, 256, None, cache, None, ref)


def test_preamble_compile_knob_changes_nothing(monkeypatch):
    d = _inputs(1, 384, 4, 64)
    got, ref = _both(d, chunk=64, sub=16)
    monkeypatch.setattr(C, "_COMPILE_ENV", True)
    monkeypatch.setattr(C, "_PREPARE_C", {})
    hot = C.chunk_kda_update(
        d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"],
        state=d["state"], lower_bound=LB, chunk=64, sub=16,
    )
    assert C.kda_chunk_compile() is True
    assert _rel(hot[0], ref[0]) < ATOL_REL
    assert _rel(hot[1], ref[1]) < ATOL_REL
    # fusing the preamble is arithmetic-preserving, not merely close
    assert bool(mx.array_equal(hot[0], got[0]).item())
    assert bool(mx.array_equal(hot[1], got[1]).item())


def test_compile_knob_defaults_off(monkeypatch):
    monkeypatch.setattr(C, "_COMPILE_ENV", None)
    monkeypatch.delenv("MLX_VLM_GLM5_KDA_CHUNK_COMPILE", raising=False)
    assert C.kda_chunk_compile() is False
