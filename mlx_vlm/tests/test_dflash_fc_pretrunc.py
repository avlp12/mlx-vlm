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
the row count can change the reduction order.  Measured on mlx 0.32.1 / CPU:
bfloat16 (the drafter's serving dtype), 8-bit quantized (the shipped
GLM-5.3-Flash-DFlash2-q8 checkpoint) and float32 with a realistic contraction
dim are all row-stable and compare EQUAL; float32 with a tiny contraction dim
(K=32) is not, and drifts by ~2e-7.  The tests below therefore assert bit
equality at the dtypes that actually serve, and pin the float32/small-K case as
a bounded difference rather than pretending it is exact.
"""

import os

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.speculative.drafters.dflash2 import DFlash2DraftModel, ModelConfig

FLAG = "MLX_VLM_DFLASH_FC_PRETRUNC"


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
    # ``bind`` wants a whole target model; ``_hidden`` only needs the embedding.
    model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    model.embed_scale = 1.0
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


def _assert_identical(model, S, L=3):
    ref, ref_cache = _run(model, S, L=L, flag="0")
    new, new_cache = _run(model, S, L=L, flag="1")
    assert new.shape == ref.shape
    assert mx.array_equal(new, ref), f"S={S}: pre-truncated output differs"
    for i, (a, b) in enumerate(zip(ref_cache, new_cache)):
        assert a.offset == b.offset, f"S={S}: layer {i} offset {b.offset} != {a.offset}"
        assert mx.array_equal(b.keys, a.keys), f"S={S}: layer {i} cache keys differ"
        assert mx.array_equal(b.values, a.values), f"S={S}: layer {i} cache values differ"
    return new_cache


@pytest.mark.parametrize("S", [5, 40, 512])
def test_pretruncated_hidden_is_bit_identical_bf16(S):
    """Serving dtype: same drafter, same inputs, both paths -- equal, not close."""
    _assert_identical(_drafter(["sliding_attention"] * 2, sliding_window=8), S)


@pytest.mark.parametrize("S", [5, 40, 512])
def test_pretruncated_hidden_is_bit_identical_q8_fc(S):
    """Shipped checkpoint carries ``fc`` 8-bit; the hoist must survive that too."""
    _assert_identical(
        _drafter(["sliding_attention"] * 2, sliding_window=8, quantize=True), S
    )


def test_pretruncated_hidden_is_bit_identical_at_a_real_window():
    """The shipped geometry's window (2048) on a 4096-row context."""
    _assert_identical(_drafter(["sliding_attention"] * 2, sliding_window=2048), 4096)


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


def test_identity_is_kernel_scoped_at_float32_small_k():
    """Documented exception, kept honest rather than kept quiet.

    float32 with a 32-wide contraction takes a different CPU GEMM path whose
    reduction order depends on the row count, so the hoist is mathematically --
    but not bitwise -- identical there.  The bound, not the inequality, is what
    is asserted: if MLX ever makes that path row-stable this test still passes.
    """
    model = _drafter(["sliding_attention"] * 2, sliding_window=8, dtype=mx.float32)
    ref, _ = _run(model, 40, flag="0")
    new, _ = _run(model, 40, flag="1")
    assert float(mx.max(mx.abs(ref - new))) < 1e-5
