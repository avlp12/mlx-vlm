"""Bit-identity for the DFlash2 drafter's compiled small-op chains.

``MLX_VLM_DFLASH_COMPILE=1`` fuses the launch-bound halves of a draft block:
the four norm + kernel-projection + two-offset dynamic-convolution regions that
sandwich self_attn, and the selector's per-position greedy Viterbi step.

What is deliberately *not* inside the boundary, and why:
  * ``self_attn`` -- it reads and writes the rotating KV cache, and mx.compile is
    known on this model family to bake ``cache.offset`` into the trace and then
    return stale results with no exception raised.
  * ``self.mlp`` -- swiglu -> nn.silu -> sigmoid -> exp, and JIT'd Metal
    substitutes the fast exp approximation for the precise one, which moves the
    last bit.
  * any walk with a caller-supplied ``sample_proposal`` -- an opaque callable
    that may draw randomness, which a trace would freeze.

With those excluded the fused path is expected to be *bit-identical*, and these
tests pin that over a cache-carrying 32-block run, which is the shape of the
failure the exclusions are there to prevent (a frozen offset only diverges once
the cache has advanced).
"""

import os

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_vlm.speculative.drafters.dflash2.dflash2 as dflash2_mod
from mlx_vlm.speculative.drafters.dflash2 import DFlash2DraftModel, ModelConfig

BLOCKS = 32


def _config(hidden=128, layers=2, vocab=512, block_size=8):
    cfg = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "is_causal": False,
        "hidden_size": hidden,
        "intermediate_size": hidden * 2,
        "num_hidden_layers": layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "vocab_size": vocab,
        "max_position_embeddings": 4096,
        "num_target_layers": 4,
        "layer_types": ["sliding_attention"] * layers,
        "sliding_window": 64,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000},
        "dflash_config": {
            "block_size": block_size,
            "runtime_block_size": block_size,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": vocab - 1,
            "selector_rank": 32,
            "selector_top_k": 8,
            "target_layer_ids": [0, 1, 2, 3],
        },
    }
    return ModelConfig.from_dict(cfg)


def _drafter(config, seed=0):
    mx.random.seed(seed)
    model = DFlash2DraftModel(config)

    def rand(tree):
        if isinstance(tree, dict):
            return {k: rand(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rand(v) for v in tree]
        return (mx.random.normal(tree.shape) * 0.08).astype(mx.bfloat16)

    model.update(rand(model.parameters()))
    # bind() normally borrows these from the 320B target; the drafter owns
    # neither, so stand them in directly.
    model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    model.embed_tokens.weight = (
        mx.random.normal(model.embed_tokens.weight.shape) * 0.08
    ).astype(mx.bfloat16)
    model.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    model.lm_head.weight = (
        mx.random.normal(model.lm_head.weight.shape) * 0.08
    ).astype(mx.bfloat16)
    mx.eval(model.parameters(), model.embed_tokens.parameters(),
            model.lm_head.parameters())
    return model


def _greedy_sampler(logits):
    return mx.argmax(logits, axis=-1)


def _run_blocks(model, config, blocks=BLOCKS, seed=3):
    """A cache-carrying run: each round advances the drafter's rotating KV cache
    by the 'accepted' context, exactly as the live speculative loop does.

    Records the pre-logits *float* hidden as well as the drafted tokens.  The
    tokens go through an argmax, which would swallow a last-bit difference; the
    hidden state is what the compiled conv/norm regions actually produce, so it
    is the sharp equality.  This is draft_block's body, unrolled so the
    intermediate is observable.
    """
    mx.random.seed(seed)
    cache = model.make_cache()
    concat = len(config.target_layer_ids) * config.hidden_size
    mask_id = int(config.mask_token_id)
    last = 0
    toks, hiddens = [], []
    for i in range(blocks):
        s_ctx = 1 + (i % 3)                       # accepted tokens vary, as live
        target_hidden = (
            mx.random.normal((1, s_ctx, concat)) * 0.3
        ).astype(mx.bfloat16)
        anchor = mx.array([last], dtype=mx.int32)
        masks = mx.full((1, config.block_size - 1), mask_id, dtype=mx.int32)
        draft_inputs = mx.concatenate([anchor[:, None], masks], axis=1)
        draft_hidden = model._hidden(draft_inputs, target_hidden, cache)[:, 1:]
        logits = model._logits(draft_hidden)
        tokens = model.candidate_selector.select(
            draft_hidden, logits, anchor, _greedy_sampler
        ).astype(mx.int32)
        mx.eval(draft_hidden, tokens)
        hiddens.append(draft_hidden)
        toks.append(tokens)
        last = int(tokens[0, -1].item())
    return toks, hiddens, cache


def _set_compile(flag):
    os.environ["MLX_VLM_DFLASH_COMPILE"] = "1" if flag else "0"
    dflash2_mod._COMPILE_ENV = None


@pytest.fixture(autouse=True)
def _restore_env():
    yield
    os.environ.pop("MLX_VLM_DFLASH_COMPILE", None)
    dflash2_mod._COMPILE_ENV = None


def test_compile_is_bit_identical_over_32_blocks():
    config = _config()
    _set_compile(False)
    eager_model = _drafter(config)
    eager, eager_h, eager_cache = _run_blocks(eager_model, config)

    _set_compile(True)
    fused_model = _drafter(config)            # same seed -> same weights
    fused, fused_h, fused_cache = _run_blocks(fused_model, config)

    assert len(eager) == BLOCKS
    for i, (a, b) in enumerate(zip(eager_h, fused_h)):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert bool(mx.all(a == b)), f"block {i}: draft hidden differs (float)"
    for i, (a, b) in enumerate(zip(eager, fused)):
        assert bool(mx.all(a == b)), f"block {i}: drafted tokens differ"

    # the KV the rounds carried must match too -- a frozen cache.offset shows up
    # in the state before it shows up in the tokens
    for li, (ca, cb) in enumerate(zip(eager_cache, fused_cache)):
        assert ca.offset == cb.offset, f"layer {li}: cache offset drifted"
        for sa, sb in zip(ca.state, cb.state):
            if sa is None or sb is None:
                continue
            assert bool(mx.all(sa == sb)), f"layer {li}: cache state differs"


def test_blocks_actually_vary():
    """Negative control.  An untrained drafter argmaxes to one dominant token, so
    token equality alone would be nearly vacuous -- which is exactly why the
    equality above is asserted on the float hidden.  Pin that the hidden really
    does move from block to block (the cache is advancing and being read)."""
    config = _config()
    _set_compile(False)
    model = _drafter(config)
    _, hiddens, cache = _run_blocks(model, config)
    assert not bool(mx.all(hiddens[0] == hiddens[1]))
    assert not bool(mx.all(hiddens[0] == hiddens[-1]))
    assert cache[0].offset > config.block_size, "KV cache must have advanced"


def test_selector_greedy_step_matches_eager():
    config = _config()
    _set_compile(False)
    model = _drafter(config)
    sel = model.candidate_selector
    mx.random.seed(11)
    B, L = 1, config.block_size - 1
    hidden = (mx.random.normal((B, L, config.hidden_size)) * 0.3).astype(mx.bfloat16)
    logits = (mx.random.normal((B, L, config.vocab_size)) * 0.5).astype(mx.bfloat16)
    anchor = mx.array([7], dtype=mx.int32)
    mx.eval(hidden, logits, anchor)
    ref = sel.select(hidden, logits, anchor, _greedy_sampler)
    mx.eval(ref)

    _set_compile(True)
    got = sel.select(hidden, logits, anchor, _greedy_sampler)
    mx.eval(got)
    assert bool(mx.all(ref == got))


def test_sampled_walk_is_never_compiled():
    """A sample_proposal walk must stay eager: a trace would freeze the RNG."""
    config = _config()
    _set_compile(True)
    model = _drafter(config)
    sel = model.candidate_selector

    class _Sampler:
        def __call__(self, logits):
            return mx.argmax(logits, axis=-1)

        def sample_proposal(self, scores):
            return mx.argmax(scores, axis=-1)

    mx.random.seed(13)
    B, L = 1, config.block_size - 1
    hidden = (mx.random.normal((B, L, config.hidden_size)) * 0.3).astype(mx.bfloat16)
    logits = (mx.random.normal((B, L, config.vocab_size)) * 0.5).astype(mx.bfloat16)
    anchor = mx.array([5], dtype=mx.int32)
    mx.eval(hidden, logits, anchor)
    out = sel.select(hidden, logits, anchor, _Sampler())
    mx.eval(out)
    assert sel._fused_step is None, "sampled walk must not build a compiled step"
    assert out.shape == (B, L)


def test_toggle_defaults_off():
    os.environ.pop("MLX_VLM_DFLASH_COMPILE", None)
    dflash2_mod._COMPILE_ENV = None
    assert dflash2_mod._compile_enabled() is False


@pytest.mark.parametrize("block_size", [3, 8])
def test_bit_identical_across_block_sizes(block_size):
    """mx.compile caches per shape; the live loop uses more than one block size
    (the adaptive floor is 3), so both must trace and both must match."""
    config = _config(block_size=block_size)
    _set_compile(False)
    _, eager_h, _ = _run_blocks(_drafter(config), config, blocks=8)
    _set_compile(True)
    _, fused_h, _ = _run_blocks(_drafter(config), config, blocks=8)
    for a, b in zip(eager_h, fused_h):
        assert bool(mx.all(a == b))
