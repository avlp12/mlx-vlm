"""The DFlash context truncation, hoisted in front of ``fc``.

Every sliding layer of a DFlash/DFlash2 drafter discards all but the last
``sliding_window - 1`` context positions before it does anything with them
(``DFlashAttention.__call__``).  ``fc`` (concat_dim -> hidden) and
``hidden_norm`` run BEFORE that discard and are position-wise, so on a long
prompt almost all of their work lands on rows that are thrown away one line
later.  ``DFlashDraftModel._pretruncate_ctx`` moves the discard in front of
them; ``MLX_VLM_DFLASH_FC_PRETRUNC=0`` restores the pre-fix sequence, which is
what every reference below is built from.

SCOPE OF THE IDENTITY CLAIM.  ``fc`` being position-wise makes the surviving
rows mathematically identical, but bit-identity is a property of the KERNEL, not
of the algebra: MLX picks a matmul implementation from the shapes, so changing
the row count can change the reduction order.  That is a per-device, per-shape
empirical fact, and these tests say so rather than asserting a bit-equality that
only one backend happens to deliver.

  * CPU (mlx 0.32.1.dev20260902).  Row-stable at every geometry these tests
    exercise: bfloat16, 8-bit-quantized ``fc``, and float32 with a realistic
    contraction dim all compare EQUAL.  float32 with a tiny contraction dim
    (K=32) is not, and drifts by ~2e-7 -- pinned as a bound below.  (One
    caveat kept honest: sweeping 32 context seeds at the toy 32->16 bfloat16
    geometry found seed 19 at S=40 producing a 1-bit change on CPU as well, so
    even CPU row-stability is a measured property of these shapes and this
    seed, not a guarantee.  The seed the tests use is pinned in ``_run``.)

  * Metal / M3 Ultra (mlx 0.32.1.dev20260902, measured 2026-09-03).  The
    quantized matmul has several row-count regimes -- the result of
    ``fc(ctx[:M])`` row 0 changes at M = 1, 2, 15, 33 and ~100, and is then
    stable for every M >= 100 up to 4096.  The toy geometry below keeps only
    ``sliding_window - 1 == 7`` rows, which lands in the smallest regime, so the
    q8 arm is NOT bit-identical there; the surviving difference is <= 4.34
    bfloat16 ulp of the row maximum in the drafter hidden state (over 32 context
    seeds) and moves the drafter's argmax over its vocabulary head in 0 of 384
    draft positions.

  * THE SHIPPED GEOMETRY IS EXACT ON METAL.  Pre-truncation only fires when
    ``S > sliding_window - 1``, so with the shipped GLM-5.3-Flash DFlash2 window
    of 2048 the two row counts compared are always M_new = 2047 and M_ref = S >=
    2048 -- both inside the stable M >= 100 regime.  Measured at the shipped
    ``fc`` (20480 -> 4096, 8-bit affine, group 64, bfloat16 activations) for
    S in {2048, 2049, 3000, 4096, 5000, 8192, 12000, 16384}: 0 of 8,384,512
    output elements differ, max|diff| = 0.0, for both the q8 and the bf16
    ``fc``.  ``test_shipped_fc_geometry_is_row_count_stable`` pins that.

  * LIVE RECEIPT.  R29 (``/Users/gesicht/glm53flash/logs/sweep10/R29_VERDICT.md``,
    2026-09-03, epsilon, 16,394-token prompt x 256 tokens, ABAB x 3) ran the real
    model with pre-truncation ON and measured accept/round 2.9231 -> 2.9231
    (delta 0.000) and text sha1 ``18d1fe50b8beac71513a`` identical against the
    pre-truncation-free tree.  The optimisation therefore stays DEFAULT ON.

The tests below assert bit equality on CPU, and on Metal a documented tolerance
plus argmax equality wherever the toy window drops into a small-M kernel regime.
"""

import os

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.speculative.drafters.dflash2 import DFlash2DraftModel, ModelConfig

FLAG = "MLX_VLM_DFLASH_FC_PRETRUNC"

# ``mx.default_device()`` is pinned by conftest from MLX_DEFAULT_DEVICE.
ON_GPU = mx.default_device() == mx.gpu

# bfloat16 keeps 7 explicit mantissa bits, so one ulp is 2**-8 of the binade.
BF16_ULP = 2.0**-8
# Metal tolerance for the toy small-M geometries, as a multiple of one bfloat16
# ulp of the row maximum.  Worst measured 2026-09-03 over 32 context seeds at
# 32->16 / q8 / window 8: 4.34 ulp on the drafter hidden state.  8 is that with
# a factor of two of headroom; it is still ~1/30 of a bfloat16 mantissa.
GPU_ULP_TOL = 8.0


def _config(layer_types, sliding_window, hidden=16, n_target=2):
    return ModelConfig.from_dict(
        {
            "architectures": ["DFlash2DraftModel"],
            "model_type": "qwen3",
            "is_causal": False,
            "hidden_size": hidden,
            "intermediate_size": 2 * hidden,
            "num_hidden_layers": len(layer_types),
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "max_position_embeddings": 8192,
            "num_target_layers": n_target,
            "layer_types": list(layer_types),
            "sliding_window": sliding_window,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000},
            "dflash_config": {
                "block_size": 3,
                "runtime_block_size": 3,
                "conv_group_size": 4,
                "conv_kernel_size": 2,
                "mask_token_id": 31,
                "selector_rank": 4,
                "selector_top_k": 4,
                "target_layer_ids": list(range(n_target)),
            },
        }
    )


def _drafter(
    layer_types, sliding_window, hidden=16, n_target=2, dtype=mx.bfloat16, quantize=False
):
    mx.random.seed(0)
    config = _config(layer_types, sliding_window, hidden, n_target)
    model = DFlash2DraftModel(config)
    # ``bind`` wants a whole target model; ``_hidden`` only needs the embedding,
    # and the argmax check below needs a head, so both are stubbed here.
    model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    model.embed_scale = 1.0
    model.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    if quantize:
        # Only ``fc`` -- it is the projection this change moves work out of, and
        # the shipped checkpoint carries it 8-bit.
        model.fc = nn.QuantizedLinear.from_linear(model.fc, group_size=32, bits=8)
    if dtype is not None:
        model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model


def _run(model, S, L=3, flag="1"):
    """One ``_hidden`` call on fresh caches, under an explicit flag value."""
    mx.random.seed(1)
    width = len(model.config.target_layer_ids) * model.config.hidden_size
    ctx = mx.random.normal((1, S, width)).astype(model.fc.weight.dtype
                                                 if model.fc.weight.dtype != mx.uint32
                                                 else mx.bfloat16)
    inputs = mx.arange(L, dtype=mx.int32)[None, :] % model.config.vocab_size
    cache = model.make_cache()
    old = os.environ.get(FLAG)
    os.environ[FLAG] = flag
    try:
        out = model._hidden(inputs, ctx, cache)
        mx.eval(out, [c.state for c in cache])
    finally:
        if old is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = old
    return out, cache


def _assert_same(new, ref, what):
    """Bit-equal on CPU; within a documented bfloat16 ulp bound on Metal.

    The CPU arm is the strict one and is never relaxed: the kernels there are
    row-stable at these shapes, so any change is a real regression.
    """
    if not ON_GPU:
        assert mx.array_equal(new, ref), f"{what}: pre-truncated output differs"
        return
    a = ref.astype(mx.float32)
    b = new.astype(mx.float32)
    scale = float(mx.max(mx.abs(a)))
    if scale == 0.0:
        assert mx.array_equal(new, ref), f"{what}: differs on an all-zero reference"
        return
    worst = float(mx.max(mx.abs(a - b)))
    ulps = worst / (BF16_ULP * scale)
    assert ulps <= GPU_ULP_TOL, (
        f"{what}: Metal row-count regime moved the result by {ulps:.2f} bfloat16 "
        f"ulp of the row maximum (max|diff|={worst:.4e}, max|ref|={scale:.4e}), "
        f"over the documented {GPU_ULP_TOL} ulp bound"
    )


def _assert_identical(model, S, L=3):
    ref, ref_cache = _run(model, S, L=L, flag="0")
    new, new_cache = _run(model, S, L=L, flag="1")
    assert new.shape == ref.shape
    _assert_same(new, ref, f"S={S}: hidden")
    # What the drafter actually emits must not move at all, on either device.
    a_ref = mx.argmax(model._logits(ref), axis=-1)
    a_new = mx.argmax(model._logits(new), axis=-1)
    mx.eval(a_ref, a_new)
    assert mx.array_equal(a_new, a_ref), f"S={S}: drafter argmax moved"
    for i, (a, b) in enumerate(zip(ref_cache, new_cache)):
        # Offsets are integer bookkeeping: exact on every device, always.
        assert a.offset == b.offset, f"S={S}: layer {i} offset {b.offset} != {a.offset}"
        _assert_same(b.keys, a.keys, f"S={S}: layer {i} cache keys")
        _assert_same(b.values, a.values, f"S={S}: layer {i} cache values")
    return new_cache


@pytest.mark.parametrize("S", [5, 40, 512])
def test_pretruncated_hidden_is_bit_identical_bf16(S):
    """Serving dtype: same drafter, same inputs, both paths -- equal, not close.

    Measured equal on CPU and on Metal at this geometry.
    """
    _assert_identical(_drafter(["sliding_attention"] * 2, sliding_window=8), S)


@pytest.mark.parametrize("S", [5, 40, 512])
def test_pretruncated_hidden_is_bit_identical_q8_fc(S):
    """Shipped checkpoint carries ``fc`` 8-bit; the hoist must survive that too.

    Exact on CPU.  On Metal the toy window keeps 7 rows, which is inside MLX's
    small-M quantized-matmul regime, so this is the bounded arm -- see the module
    docstring, and ``test_shipped_fc_geometry_is_row_count_stable`` for the row
    counts the shipped window actually produces.
    """
    _assert_identical(
        _drafter(["sliding_attention"] * 2, sliding_window=8, quantize=True), S
    )


def test_pretruncated_hidden_is_bit_identical_at_a_real_window():
    """The shipped geometry's window (2048) on a 4096-row context."""
    _assert_identical(_drafter(["sliding_attention"] * 2, sliding_window=2048), 4096)


@pytest.mark.skipif(
    not ON_GPU,
    reason="cost, not correctness: the shipped fc is ~700 GFLOP per arm and takes "
    "minutes on CPU.  It exists to pin a Metal row-count-regime property; CPU "
    "row-stability at the small geometries is covered by the tests above.",
)
@pytest.mark.parametrize("quantize", [True, False])
def test_shipped_fc_geometry_is_row_count_stable(quantize):
    """The claim the optimisation actually rests on, at the shipped ``fc`` shape.

    GLM-5.3-Flash-DFlash2 ships ``fc`` at 5 * 4096 -> 4096 with a 2048 window, so
    the hoist only ever compares M_new = 2047 against M_ref = S >= 2048.  Both are
    in MLX's stable row-count regime on Metal, and this asserts bit equality --
    no tolerance.  If a future MLX changes the tiling here, that is exactly what
    this test exists to catch.
    """
    K, N, KEEP, S = 5 * 4096, 4096, 2047, 4096
    mx.random.seed(0)
    fc = nn.Linear(K, N, bias=False)
    if quantize:
        fc = nn.QuantizedLinear.from_linear(fc, group_size=64, bits=8)
    fc.set_dtype(mx.bfloat16)
    mx.eval(fc.parameters())

    mx.random.seed(1)
    ctx = mx.random.normal((1, S, K)).astype(mx.bfloat16)
    skip = S - KEEP
    ref = fc(ctx)[:, skip:]
    new = fc(ctx[:, skip:])
    mx.eval(ref, new)
    differing = int(mx.sum(ref != new))
    assert differing == 0, (
        f"shipped fc {K}->{N} q8={quantize}: {differing}/{ref.size} elements moved "
        f"when the row count went {S} -> {KEEP} "
        f"(max|diff|={float(mx.max(mx.abs(ref.astype(mx.float32) - new.astype(mx.float32)))):.4e})"
    )


@pytest.mark.parametrize("S", [5, 40, 512])
def test_offset_advance_matches_the_per_layer_discard(S):
    """Every layer must end where the per-layer discard used to leave it.

    The proposal block's keys are concatenated outside the cache, so only the
    context advances the offset: ``skip`` (hoisted or per-layer) plus the
    resident window.  That sums to ``S`` either way.
    """
    model = _drafter(["sliding_attention"] * 2, sliding_window=8)
    keep = model.config.sliding_window - 1
    skip = max(0, S - keep)
    resident = min(S, keep)
    _, ref_cache = _run(model, S, flag="0")
    _, new_cache = _run(model, S, flag="1")
    for i, (a, b) in enumerate(zip(ref_cache, new_cache)):
        assert b.offset == skip + resident == S, f"layer {i}: {b.offset}"
        assert b.offset == a.offset


def test_pretruncation_declines_when_a_layer_is_not_sliding():
    """A full-attention layer sees the whole context, so nothing may be hoisted."""
    model = _drafter(["sliding_attention", "full_attention"], sliding_window=8)
    assert model._uniform_ctx_keep() is None
    ctx = mx.random.normal((1, 40, 32))
    cache = model.make_cache()
    before = [c.offset for c in cache]
    assert model._pretruncate_ctx(ctx, cache).shape == ctx.shape
    assert [c.offset for c in cache] == before


def test_flag_off_restores_the_prefix_free_sequence():
    model = _drafter(["sliding_attention"] * 2, sliding_window=8)
    ctx = mx.random.normal((1, 40, 32))
    cache = model.make_cache()
    os.environ[FLAG] = "0"
    try:
        assert model._pretruncate_ctx(ctx, cache).shape[1] == 40
        assert [c.offset for c in cache] == [0, 0]
    finally:
        os.environ.pop(FLAG, None)
    assert model._pretruncate_ctx(ctx, cache).shape[1] == 7
    assert [c.offset for c in cache] == [33, 33]


def test_default_is_on():
    """R29 measured the real model unchanged with this on; it ships on."""
    os.environ.pop(FLAG, None)
    from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import _fc_pretrunc_enabled

    assert _fc_pretrunc_enabled() is True


def test_identity_is_kernel_scoped_at_float32_small_k():
    """Documented exception, kept honest rather than kept quiet.

    float32 with a 32-wide contraction takes a different GEMM path whose
    reduction order depends on the row count, so the hoist is mathematically --
    but not bitwise -- identical there.  The bound, not the inequality, is what
    is asserted: if MLX ever makes that path row-stable this test still passes.
    """
    model = _drafter(["sliding_attention"] * 2, sliding_window=8, dtype=mx.float32)
    ref, _ = _run(model, 40, flag="0")
    new, _ = _run(model, 40, flag="1")
    assert float(mx.max(mx.abs(ref - new))) < 1e-5
