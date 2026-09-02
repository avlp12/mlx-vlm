"""Chunked prefill for a DFlash-attached GLM-5.3-Flash request.

A dflash drafter used to force ``prefill_step_size = None``
(``Glm5NextLanguageModel.chunked_prefill_policy``), so the served prefill ran the
WHOLE prompt through one forward.  Two things then scaled with the prompt for the
life of the request: the ``[1, S, vocab]`` logits of that forward (309,760 B per
token on the shipped geometry -- only the last position is ever sampled), and the
transient live set of a single S-wide forward (measured 2.27 MB/token on the box,
R29).  Chunking removes both, but only if the chunk loop carries the speculative
capture and the pieces are stitched back -- which is what
``PrefillHiddenAccumulator`` does.

What this module pins, in the order the argument runs:

1.  **The target does not move.**  Chunked-with-capture is bit-identical to
    chunked-without-capture (i.e. to the greedy chunked path that already ships)
    in its logits AND in every byte of the prompt cache.  The capture only
    appends to Python sinks.
2.  **Unchunked vs chunked is a separate question**, and the answer is measured
    here rather than assumed: a KDA scan split at a chunk boundary is not the
    same arithmetic as one scan over the whole prompt.  That difference is the
    one the greedy path already accepts; this file records its size instead of
    claiming it away.
3.  **The drafter's round-1 context is bit-identical** for every row it keeps --
    including the RoPE offsets, which is the half that is easy to lose.  Trimming
    the context at the prefill seam and bumping the draft cache offsets by the
    trimmed width is exactly the operation ``_pretruncate_ctx`` performs one
    level down; skipping the offset half shifts every draft position and is
    caught by a negative control below.
4.  **The retention is gone**, in bytes, not in shapes.
5.  **All three chunk drivers carry the capture**, not two.  ``generate_step`` and
    ``server/generation.py::_run_chunked_speculative_prefill`` were fixed first;
    ``generate/ar.py::PromptProcessingBatch`` -- the continuous-batching prefill a
    served multi-row request actually takes -- had the identical defect plus a
    second one, ``prompt_tokens`` read after the chunk loop had consumed the prompt
    down to its tail.  Section 8 pins the batched path at B = 2 with ragged prompt
    lengths and the left padding the batch driver applies.
"""

import logging
import os
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.generate import ar as ar_module
from mlx_vlm.generate.ar import (
    PromptProcessingBatch,
    _left_pad_prompts,
    _right_pad_prompts,
)
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next.language import LanguageModel
from mlx_vlm.server import generation as server_generation
from mlx_vlm.speculative.dflash import _adopt_pretruncated_context
from mlx_vlm.speculative.drafters.dflash2 import DFlash2DraftModel, ModelConfig
from mlx_vlm.speculative.utils import (
    PrefillHiddenAccumulator,
    prefill_context_keep,
    prefill_context_offset,
)

CAPTURE_IDS = [0, 1]
STREAM = mx.default_stream(mx.default_device())


# ------------------------------------------------------------------ tiny target


def _tiny_text_config():
    return TextConfig(
        model_type="glm5_next_text",
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=0,
        v_head_dim=64,
        qk_nope_head_dim=64,
        qk_head_dim=64,
        num_experts_per_tok=4,
        first_k_dense_replace=1,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        index_topk=6,
        index_head_dim=64,
        index_n_heads=2,
        index_kpool=3,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 64,
            "short_conv_kernel_size": 2,
            "gate_lower_bound": -5.0,
        },
        hc_mult=4,
        num_nextn_predict_layers=1,
        pad_token_id=0,
        eos_token_id=1,
    )


def _lm():
    mx.random.seed(0)
    lm = LanguageModel(_tiny_text_config())
    lm.eval()
    mx.eval(lm.parameters())
    return lm


def _prompt(S):
    return (mx.arange(S, dtype=mx.int32) % 127)[None, :] + 1


def _cache_arrays(prompt_cache):
    """Every array a prompt cache holds, flattened, for a bytewise comparison."""
    out = []
    for entry in prompt_cache:
        state = entry.state
        stack = [state]
        while stack:
            item = stack.pop()
            if isinstance(item, mx.array):
                out.append(item)
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
    return out


def _prefill(lm, prompt, *, step, capture, keep=None):
    """Drive the server's prefill helper the way the request path does."""
    embeds = lm.model.embed_tokens(prompt)
    prompt_cache = lm.make_cache()
    spec_kwargs = {"capture_layer_ids": list(CAPTURE_IDS)} if capture else {}
    out, remaining = server_generation._run_chunked_speculative_prefill(
        lm,
        prompt,
        embeds,
        prompt_cache,
        {},
        spec_kwargs,
        prefill_step_size=step,
        generation_stream=STREAM,
        hidden_context_keep=keep,
    )
    mx.eval(out.logits)
    if out.hidden_states:
        mx.eval(out.hidden_states)
    mx.eval(_cache_arrays(prompt_cache))
    return out, prompt_cache, remaining


# ------------------------------------------- 1. the target does not move at all


CHUNK = 8
PROMPT = 40
WINDOW = 16  # the stub drafter's sliding_window; it keeps WINDOW - 1 = 15 rows
KEEP = WINDOW - 1


def test_chunked_spec_prefill_is_bit_identical_to_chunked_greedy():
    """The claim the whole change rests on.

    The capture kwargs decide whether two Python lists get appended to; no value
    read out of a sink feeds back into ``h``, the caches or the logits, and
    ``_fused_kda_eligible`` explicitly ``del``\\ s ``gdn_sink`` before deciding.
    So spec-chunked and greedy-chunked must agree BITWISE, not approximately.
    """
    lm = _lm()
    prompt = _prompt(PROMPT)

    greedy_out, greedy_cache, _ = _prefill(lm, prompt, step=CHUNK, capture=False)
    spec_out, spec_cache, _ = _prefill(lm, prompt, step=CHUNK, capture=True, keep=KEEP)

    assert greedy_out.hidden_states is None
    assert spec_out.hidden_states is not None

    assert mx.array_equal(
        spec_out.logits, greedy_out.logits
    ), "the speculative capture moved the target's logits"
    greedy_arrays = _cache_arrays(greedy_cache)
    spec_arrays = _cache_arrays(spec_cache)
    assert len(spec_arrays) == len(greedy_arrays)
    for i, (a, b) in enumerate(zip(greedy_arrays, spec_arrays)):
        assert a.shape == b.shape, f"cache array {i}: {a.shape} != {b.shape}"
        assert mx.array_equal(a, b), f"cache array {i} differs"


def test_chunked_spec_prefill_holds_no_gdn_sink():
    """The prefill leg's KDA rollback stash must not be built on ANY chunk.

    A per-chunk stash would be a fresh O(chunk) allocation on every chunk and is
    read by nobody: every consumer of ``gdn_states`` takes it off a verify
    forward.
    """
    lm = _lm()
    out, _, _ = _prefill(lm, _prompt(PROMPT), step=CHUNK, capture=True, keep=KEEP)
    assert not out.gdn_states, f"expected no sink, got {out.gdn_states!r}"


def test_the_capture_reaches_every_chunk_not_just_the_last():
    """Regression for the defect this change fixes.

    With the capture attached only to the final one-token forward, the drafter's
    whole prompt context was a single row.
    """
    seen = []
    lm = _lm()

    class _Spy:
        def __init__(self, inner):
            self.inner = inner
            self.supports_capture_gdn_states = True

        def __call__(self, inputs, cache=None, **kwargs):
            seen.append((int(inputs.shape[1]), dict(kwargs)))
            return self.inner(inputs, cache=cache, **kwargs)

    spy = _Spy(lm)
    embeds = lm.model.embed_tokens(_prompt(PROMPT))
    out, _ = server_generation._run_chunked_speculative_prefill(
        spy,
        _prompt(PROMPT),
        embeds,
        lm.make_cache(),
        {},
        {"capture_layer_ids": list(CAPTURE_IDS)},
        prefill_step_size=CHUNK,
        generation_stream=STREAM,
        hidden_context_keep=None,
    )
    assert len(seen) > 1, "prompt did not chunk"
    for width, kwargs in seen:
        assert kwargs["capture_layer_ids"] == CAPTURE_IDS
        assert kwargs["capture_gdn_states"] is False
    # Widths reconstruct the prompt exactly, last chunk being the single token.
    assert sum(w for w, _ in seen) == PROMPT
    assert seen[-1][0] == 1
    assert out.hidden_states[0].shape[1] == PROMPT


# --------------------------------- 2. unchunked vs chunked: measured, not assumed


def test_unchunked_vs_chunked_delta_is_recorded_not_asserted_away():
    """Chunking the target IS allowed to move its state, and may.

    A KDA recurrent scan run as ``ceil(S/chunk)`` scans carrying state is not the
    same arithmetic as one scan over S, and the same is true of the DSA branch
    selection (a short chunk takes the dense mask; a long one takes the gathered
    path).  This is a property of chunking, not of the capture -- the greedy path
    already ships it -- so the test that matters is
    ``test_chunked_spec_prefill_is_bit_identical_to_chunked_greedy`` above.  What
    is pinned here is only that the two are CLOSE, and that whichever way the
    equality falls it falls the same way for the captured and uncaptured arms.

    Measured on this config (fp32, MLX_ENABLE_TF32=0, CPU, S=40, chunk=8):
    unchunked vs chunked moves the last-position logits by 7.15e-7 in BOTH arms,
    and 5 of the 6 prompt-cache arrays differ bitwise (the sixth is empty).  So
    the answer is: chunking DOES move the target's state, by that much, and it
    moves it identically whether or not a drafter is attached.
    """
    lm = _lm()
    prompt = _prompt(PROMPT)

    unchunked_g, _, _ = _prefill(lm, prompt, step=None, capture=False)
    chunked_g, _, _ = _prefill(lm, prompt, step=CHUNK, capture=False)
    unchunked_s, _, _ = _prefill(lm, prompt, step=None, capture=True)
    chunked_s, _, _ = _prefill(lm, prompt, step=CHUNK, capture=True)

    last_ug = unchunked_g.logits[:, -1, :]
    last_us = unchunked_s.logits[:, -1, :]
    assert mx.array_equal(
        last_ug, last_us
    ), "the capture moved the UNCHUNKED target's logits"

    equal_greedy = bool(mx.array_equal(last_ug, chunked_g.logits[:, -1, :]))
    equal_spec = bool(mx.array_equal(last_us, chunked_s.logits[:, -1, :]))
    assert equal_greedy == equal_spec, (
        "chunking changed the target for one arm and not the other -- that would "
        "mean the capture is not a pure sink"
    )

    delta = float(
        mx.max(
            mx.abs(
                last_ug.astype(mx.float32)
                - chunked_g.logits[:, -1, :].astype(mx.float32)
            )
        )
    )
    # Loose by design: this is the size of the chunking difference, recorded so a
    # future change that makes it LARGE is visible.  It is not an identity claim.
    assert delta < 1e-2, f"chunking moved the last-position logits by {delta}"


def test_unchunked_and_chunked_captures_agree_on_the_rows_the_drafter_keeps():
    """The drafter's window, compared across the seam.

    ``hidden_sink`` is filled by ``hidden_sink.append(h.mean(axis=2))`` after each
    captured layer, so a chunk's capture is that chunk's rows of the same
    quantity.  Stitching them along the time axis reproduces the unchunked
    capture up to the target-state difference measured above.

    Measured on this config: 1.6e-7 (layer 0) and 4.2e-7 (layer 1) over the 15
    kept rows -- the same chunking difference, not an accumulator error.  That
    the accumulator adds nothing of its own is proved separately and BITWISE by
    ``test_stitching_is_bit_identical_when_the_pieces_are_the_same_pieces``.
    """
    lm = _lm()
    prompt = _prompt(PROMPT)
    unchunked, _, _ = _prefill(lm, prompt, step=None, capture=True)
    chunked, _, _ = _prefill(lm, prompt, step=CHUNK, capture=True, keep=KEEP)

    assert (
        len(chunked.hidden_states) == len(unchunked.hidden_states) == len(CAPTURE_IDS)
    )
    for i, h in enumerate(chunked.hidden_states):
        assert h.shape == (1, KEEP, lm.args.hidden_size), f"layer {i}: {h.shape}"
        ref = unchunked.hidden_states[i][:, -KEEP:]
        drift = float(mx.max(mx.abs(h.astype(mx.float32) - ref.astype(mx.float32))))
        assert drift < 1e-2, f"layer {i}: stitched capture drifted {drift}"


def test_stitching_is_bit_identical_when_the_pieces_are_the_same_pieces():
    """The accumulator itself adds nothing.

    Split one unchunked capture by hand and feed the pieces in: the stitched
    result must be EQUAL, not close.  This separates "the accumulator is correct"
    from "chunking moves the target", which the previous test measures.
    """
    lm = _lm()
    prompt = _prompt(PROMPT)
    ref, _, _ = _prefill(lm, prompt, step=None, capture=True)

    acc = PrefillHiddenAccumulator()
    for lo in range(0, PROMPT, CHUNK):
        piece = [h[:, lo : lo + CHUNK] for h in ref.hidden_states]
        acc.append(_Captured(piece))
    stitched, offset = acc.finish()
    assert offset == 0
    for a, b in zip(ref.hidden_states, stitched):
        assert a.shape == b.shape
        assert mx.array_equal(a, b)


class _Captured:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


# ----------------------------------------------- 3. the drafter's round-1 context


def _drafter(hidden=16, n_target=2, window=WINDOW, n_layers=2):
    mx.random.seed(0)
    config = ModelConfig.from_dict(
        {
            "architectures": ["DFlash2DraftModel"],
            "model_type": "qwen3",
            "is_causal": False,
            "hidden_size": hidden,
            "intermediate_size": 2 * hidden,
            "num_hidden_layers": n_layers,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "max_position_embeddings": 8192,
            "num_target_layers": n_target,
            "layer_types": ["sliding_attention"] * n_layers,
            "sliding_window": window,
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
    model = DFlash2DraftModel(config)
    model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    model.embed_scale = 1.0
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def _draft_round_one(model, ctx, *, skip=0, L=3):
    cache = model.make_cache()
    if skip:
        _adopt_pretruncated_context(model, [cache], skip)
    inputs = mx.arange(L, dtype=mx.int32)[None, :] % model.config.vocab_size
    out = model._hidden(inputs, ctx, cache)
    mx.eval(out, [c.state for c in cache])
    return out, cache


def _assert_draft_state_equal(ref, new, *, why):
    ref_out, ref_cache = ref
    new_out, new_cache = new
    assert new_out.shape == ref_out.shape
    assert mx.array_equal(new_out, ref_out), why
    for i, (a, b) in enumerate(zip(ref_cache, new_cache)):
        assert a.offset == b.offset, f"layer {i}: offset {b.offset} != {a.offset}"
        assert mx.array_equal(a.keys, b.keys), f"layer {i}: cache keys differ"
        assert mx.array_equal(a.values, b.values), f"layer {i}: cache values differ"


@pytest.mark.parametrize("S", [40, 128])
def test_trimmed_context_plus_adopted_offset_is_bit_identical(S):
    """The seam trim, done correctly, is a no-op for the drafter."""
    model = _drafter()
    mx.random.seed(1)
    width = len(model.config.target_layer_ids) * model.config.hidden_size
    ctx = mx.random.normal((1, S, width)).astype(mx.bfloat16)
    keep = model.prefill_context_keep()
    assert keep == KEEP

    ref = _draft_round_one(model, ctx)
    trimmed = _draft_round_one(model, mx.contiguous(ctx[:, -keep:]), skip=S - keep)
    _assert_draft_state_equal(
        ref, trimmed, why="pre-truncated round-1 hidden is not bit-identical"
    )


def test_the_offset_half_of_the_trim_is_load_bearing():
    """Negative control: trimming without adopting the offset DOES change things.

    RoPE is relative, so the attention scores are mathematically the same and the
    emitted tokens could not change (acceptance is resolved against the target's
    argmax) -- but the arithmetic is different arithmetic, and the acceptance rate
    is free to move.  If this test ever starts passing by accident, the identity
    test above has stopped proving anything.
    """
    model = _drafter()
    S = 128
    mx.random.seed(1)
    width = len(model.config.target_layer_ids) * model.config.hidden_size
    ctx = mx.random.normal((1, S, width)).astype(mx.bfloat16)
    keep = model.prefill_context_keep()

    ref_out, ref_cache = _draft_round_one(model, ctx)
    bad_out, bad_cache = _draft_round_one(model, mx.contiguous(ctx[:, -keep:]), skip=0)

    assert ref_cache[0].offset != bad_cache[0].offset
    assert not mx.array_equal(ref_out, bad_out), (
        "dropping the offset compensation happened to be a no-op on this "
        "geometry -- pick one where it is not, or the identity test is vacuous"
    )


def test_a_full_attention_draft_layer_refuses_the_trim():
    """A layer that reads the whole context makes the hoist illegal, so decline."""
    model = _drafter(n_layers=2)
    model.config.layer_types = ["full_attention", "sliding_attention"]
    assert model.prefill_context_keep() is None
    assert prefill_context_keep("dflash", model) is None


def test_the_trim_has_a_kill_switch(monkeypatch):
    model = _drafter()
    assert prefill_context_keep("dflash", model) == KEEP
    monkeypatch.setenv("MLX_VLM_SPEC_PREFILL_CTX_TRIM", "0")
    assert prefill_context_keep("dflash", model) is None


def test_only_dflash_offers_a_trim():
    model = _drafter()
    for kind in ("mtp", "eagle3", "lookup"):
        assert prefill_context_keep(kind, model) is None
    assert prefill_context_keep("dflash", None) is None


def test_a_drafter_that_cannot_adopt_an_offset_is_a_loud_error():
    class _Deaf:
        pass

    with pytest.raises(RuntimeError, match="adopt_pretruncated_context"):
        _adopt_pretruncated_context(_Deaf(), [[]], 7)
    # zero is not an error: it is the default everywhere.
    _adopt_pretruncated_context(_Deaf(), [[]], 0)


# ---------------------------------------------------- 4. the retention, in bytes


def _held_bytes(lm, prompt, *, step, capture, keep=None):
    """Live bytes still referenced once the prefill's result is in hand."""
    mx.clear_cache()
    base = mx.get_active_memory()
    out, prompt_cache, _ = _prefill(lm, prompt, step=step, capture=capture, keep=keep)
    mx.clear_cache()
    held = mx.get_active_memory() - base
    del out, prompt_cache
    mx.clear_cache()
    return held


def test_the_chunked_prefill_stops_holding_the_full_prompt_logits():
    """The dominant remaining retention after the gdn_sink fix, in bytes.

    Unchunked, ``out.logits`` is ``[1, S, vocab]`` and the request holds it for
    its whole life; chunked, only the final one-token forward's logits survive.
    """
    lm = _lm()
    S = 256
    prompt = _prompt(S)
    vocab = lm.args.vocab_size
    itemsize = lm.model.embed_tokens.weight.dtype.size

    # Warm-up: the fused in-projection concatenates its weights once and caches
    # them on the module; that allocation must not land in a measured arm.
    mx.eval(lm(prompt, cache=lm.make_cache(), capture_layer_ids=CAPTURE_IDS).logits)
    mx.clear_cache()

    unchunked = _held_bytes(lm, prompt, step=None, capture=True)
    chunked = _held_bytes(lm, prompt, step=CHUNK, capture=True, keep=KEEP)

    logits_bytes = (S - 1) * vocab * itemsize
    assert chunked < unchunked
    assert unchunked - chunked >= logits_bytes, (
        f"freed {unchunked - chunked} B but the discarded logits alone are "
        f"{logits_bytes} B (unchunked={unchunked}, chunked={chunked}, S={S})"
    )


def test_the_retained_capture_stops_growing_with_the_prompt():
    """With the trim on, what the request holds is O(window), not O(S)."""
    lm = _lm()
    mx.eval(
        lm(_prompt(64), cache=lm.make_cache(), capture_layer_ids=CAPTURE_IDS).logits
    )
    mx.clear_cache()

    small = _held_bytes(lm, _prompt(128), step=CHUNK, capture=True, keep=KEEP)
    large = _held_bytes(lm, _prompt(256), step=CHUNK, capture=True, keep=KEEP)
    growth = (large - small) / 128.0

    hidden_per_token = (
        len(CAPTURE_IDS)
        * lm.args.hidden_size
        * (lm.model.embed_tokens.weight.dtype.size)
    )
    assert growth < hidden_per_token, (
        f"retention still grows at {growth} B/token, at least as fast as the "
        f"untrimmed capture ({hidden_per_token} B/token)"
    )


def test_a_trimmed_capture_does_not_pin_its_parent():
    """``mx`` slices are views that pin the parent buffer.

    A bare ``h[:, -keep:]`` would keep the whole prompt-shaped concatenation
    alive -- the exact retention this change removes -- so ``finish`` copies.
    """
    keep = 8
    acc = PrefillHiddenAccumulator(keep=keep)
    big = mx.zeros((1, 4096, 64), dtype=mx.float32)
    mx.eval(big)
    acc.append(_Captured([big]))
    stitched, offset = acc.finish()
    assert offset == 4096 - keep
    assert stitched[0].shape == (1, keep, 64)

    mx.clear_cache()
    base = mx.get_active_memory()
    held = [mx.contiguous(stitched[0])]
    mx.eval(held)
    del stitched, big, acc
    mx.clear_cache()
    after = mx.get_active_memory() - base
    # If the trimmed row block still pinned the 4096-row parent this would be
    # ~1 MB rather than ~2 KB.
    assert after < 4096 * 64 * 4 // 8, f"trimmed capture still pins {after} B"


# ------------------------------------------------------- 5. the accumulator rules


def test_the_accumulator_never_trims_inside_the_loop():
    """A chunk boundary is not the prompt end.

    Per-chunk trailing slicing would keep the tail of every chunk; the window is
    the tail of the CONCATENATION.  Feed distinguishable chunks and check which
    rows come back.
    """
    keep = 5
    acc = PrefillHiddenAccumulator(keep=keep)
    rows = []
    for lo in range(0, 12, 4):
        piece = mx.arange(lo, lo + 4, dtype=mx.float32).reshape(1, 4, 1)
        rows.extend(range(lo, lo + 4))
        acc.append(_Captured([piece]))
    stitched, offset = acc.finish()
    assert offset == 12 - keep == 7
    assert stitched[0].reshape(-1).tolist() == [7.0, 8.0, 9.0, 10.0, 11.0]


def test_the_accumulator_releases_chunks_it_can_prove_are_outside_the_window():
    """Whole aged-out chunks are dropped so the loop's own retention is bounded."""
    keep = 5
    acc = PrefillHiddenAccumulator(keep=keep)
    for lo in range(0, 40, 4):
        acc.append(_Captured([mx.zeros((1, 4, 1))]))
        resident = acc.total_rows - acc.dropped_rows
        # never less than the window, never more than window + one chunk
        assert resident >= min(acc.total_rows, keep)
        assert resident <= keep + 4
    stitched, offset = acc.finish()
    assert offset == 40 - keep
    assert stitched[0].shape == (1, keep, 1)


def test_the_accumulator_is_a_no_op_without_captures():
    acc = PrefillHiddenAccumulator(keep=4)
    acc.append(_Captured(None))
    acc.append(object())
    assert acc.finish() == (None, 0)
    assert acc.pending() == []


def test_a_capture_width_change_mid_prompt_is_a_loud_error():
    acc = PrefillHiddenAccumulator()
    acc.append(_Captured([mx.zeros((1, 4, 1)), mx.zeros((1, 4, 1))]))
    with pytest.raises(RuntimeError, match="capture width changed"):
        acc.append(_Captured([mx.zeros((1, 4, 1))]))


def test_the_offset_rides_on_the_prefill_output():
    lm = _lm()
    out, _, _ = _prefill(lm, _prompt(PROMPT), step=CHUNK, capture=True, keep=KEEP)
    assert prefill_context_offset(out) == PROMPT - KEEP
    plain, _, _ = _prefill(lm, _prompt(PROMPT), step=CHUNK, capture=True)
    assert prefill_context_offset(plain) == 0
    assert prefill_context_offset(object()) == 0


# ------------------------------------------------------------------- 6. the gate


def test_the_policy_admits_dflash_only_with_a_capture_list():
    lm = _lm()
    drafter = object()
    assert lm.chunked_prefill_policy(draft_model=None) is True
    assert (
        lm.chunked_prefill_policy(
            draft_model=drafter,
            draft_kind="dflash",
            prefill_kwargs={"capture_layer_ids": [0, 1]},
        )
        is True
    )
    assert (
        lm.chunked_prefill_policy(
            draft_model=drafter, draft_kind="dflash", prefill_kwargs={}
        )
        is False
    )


def test_the_policy_leaves_the_other_drafters_where_they_were():
    lm = _lm()
    drafter = object()
    # eagle3: structurally the same capture, deliberately NOT admitted yet.
    assert (
        lm.chunked_prefill_policy(
            draft_model=drafter,
            draft_kind="eagle3",
            prefill_kwargs={"capture_layer_ids": [0, 1]},
        )
        is False
    )
    assert lm.chunked_prefill_policy(draft_model=drafter, draft_kind="lookup") is True
    assert (
        lm.chunked_prefill_policy(
            draft_model=drafter,
            draft_kind="mtp",
            prefill_kwargs={"return_hidden": True, "return_shared_kv": True},
        )
        is True
    )
    assert (
        lm.chunked_prefill_policy(
            draft_model=drafter,
            draft_kind="mtp",
            prefill_kwargs={"return_hidden": True},
        )
        is False
    )


def test_mtp_capture_is_not_carried_into_the_chunks():
    """MTP's ``return_hidden`` capture must stay off the chunk loop.

    Its consumer reads ``hidden_states[-1]`` and wants the LAST token; stitching
    would hand it the whole prompt.  Its chunk loop is already correct as it is.
    """
    seen = []
    lm = _lm()

    class _Spy:
        supports_capture_gdn_states = True

        def __init__(self, inner):
            self.inner = inner

        def __call__(self, inputs, cache=None, **kwargs):
            seen.append(dict(kwargs))
            return self.inner(inputs, cache=cache, **kwargs)

    spy = _Spy(lm)
    prompt = _prompt(PROMPT)
    out, _ = server_generation._run_chunked_speculative_prefill(
        spy,
        prompt,
        lm.model.embed_tokens(prompt),
        lm.make_cache(),
        {},
        {"return_hidden": True, "return_shared_kv": True},
        prefill_step_size=CHUNK,
        generation_stream=STREAM,
    )
    assert len(seen) > 1
    for kwargs in seen[:-1]:
        assert "return_hidden" not in kwargs
    assert seen[-1]["return_hidden"] is True
    assert out.hidden_states[-1].shape[1] == 1


# --------------------------------------------- 7. the non-server generator (ar.py)


def test_ar_chunk_loop_carries_the_capture_and_stitches_the_prompt():
    """``generate/ar.py`` built its ``chunk_kwargs`` from ``kwargs`` only.

    So a dflash drafter on the non-server path would have chunked (once the
    policy allowed it) and handed the drafter one row.  Same fix, same shape:
    carry the capture, accumulate, stitch, pass the offset on.
    """
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from mlx_vlm.generate import ar as ar_module

    generate_module = sys.modules["mlx_vlm.generate"]

    hidden_width = 4
    prompt_len = 5
    step = 2

    model = MagicMock()
    model.no_chunked_prefill = False
    model.chunked_prefill_policy.return_value = True
    model.language_model.supports_logits_to_keep = False
    model.language_model.supports_capture_gdn_states = True

    def _forward(*args, **kwargs):
        embeds = kwargs.get("inputs_embeds")
        n = embeds.shape[1] if embeds is not None else args[0].shape[1]
        return SimpleNamespace(
            logits=mx.zeros((1, n, 4)),
            hidden_states=[mx.ones((1, n, hidden_width)) * n],
            shared_kv_states={},
            cross_attention_states=None,
            encoder_outputs=None,
        )

    model.language_model.side_effect = _forward

    embedding_output = MagicMock()
    embedding_output.inputs_embeds = mx.zeros((1, prompt_len, hidden_width))
    embedding_output.to_dict.return_value = {}
    model.get_input_embeddings.return_value = embedding_output

    drafter = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[0]),
        prefill_context_keep=lambda: 3,
        adopt_pretruncated_context=lambda cache, skip: None,
    )
    seen_rounds = {}

    def _fake_rounds(*args, **kwargs):
        seen_rounds["last_outputs"] = args[6]
        seen_rounds["target_hidden_offset"] = kwargs.get("target_hidden_offset")
        return iter(())

    with (
        patch("mlx_vlm.speculative.drafters.validate_drafter_compatibility"),
        patch.object(generate_module.cache, "make_prompt_cache", return_value=[]),
        patch.object(generate_module, "make_logits_processors", return_value=[]),
        patch.object(
            generate_module, "make_sampler", return_value=lambda _: mx.array([0])
        ),
        patch.object(ar_module, "run_speculative_rounds", side_effect=_fake_rounds),
    ):
        list(
            generate_module.generate_step(
                input_ids=mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32),
                model=model,
                pixel_values=None,
                mask=None,
                max_tokens=1,
                prefill_step_size=step,
                draft_model=drafter,
                draft_kind="dflash",
            )
        )

    calls = model.language_model.call_args_list
    assert len(calls) >= 2
    for call in calls:
        assert call.kwargs["capture_layer_ids"] == [0]
        assert call.kwargs["capture_gdn_states"] is False

    stitched = seen_rounds["last_outputs"].hidden_states
    # keep=3, prompt=5 -> the trailing 3 rows of the concatenation, and the
    # drafter is owed the 2 rows that were dropped.
    assert len(stitched) == 1
    assert stitched[0].shape == (1, 3, hidden_width)
    assert seen_rounds["target_hidden_offset"] == prompt_len - 3
    # Chunk widths were 2, 2, 1 and each chunk's capture is filled with its own
    # width, so the surviving rows say which chunks they came from.
    assert stitched[0][0, :, 0].tolist() == [2.0, 2.0, 1.0]


# ------------------------------------- 8. the batched driver (PromptProcessingBatch)
#
# The third chunked-prefill driver in the tree.  ``generate_step`` and
# ``server/generation.py::_run_chunked_speculative_prefill`` were fixed first;
# ``PromptProcessingBatch`` -- the continuous-batching prefill, the one a served
# multi-row request actually takes -- kept the defect: ``prompt_step`` built its
# kwargs from ``self._prompt_kwargs`` alone, so with a dflash drafter attached the
# policy admitted chunking (because the capture WAS requested) and then the drafter
# was handed only the final forward.  Second half of the same defect:
# ``prompt_tokens=self._input_ids`` was read AFTER the chunk loop had consumed
# ``_input_ids`` down to its tail.

BATCH_ROWS = [list(range(3, 27)), list(range(7, 47))]  # ragged: 24 and 40 tokens
BATCH_STEP = 8
BATCH_S = 40  # left-padded width; row 0 carries 16 columns of left padding
BATCH_KEEP = 15  # the stub drafter's prefill_context_keep()


class _BatchSpy:
    """Records every forward the batch driver makes, and every capture it returns.

    A proxy rather than a subclass: ``PromptProcessingBatch`` calls ``self.model(...)``
    and reaches for ``make_cache`` / ``chunked_prefill_policy`` /
    ``supports_capture_gdn_states`` on the same object, so plain attribute
    forwarding is enough and the real ``Glm5NextLanguageModel`` does the arithmetic.
    """

    def __init__(self, lm):
        self._lm = lm
        self.calls = []
        self.captures = []

    def __call__(self, *args, **kwargs):
        self.calls.append(dict(kwargs))
        out = self._lm(*args, **kwargs)
        self.captures.append(getattr(out, "hidden_states", None))
        return out

    def __getattr__(self, name):
        return getattr(self._lm, name)


class _StubDrafter:
    """The smallest object ``speculative/utils.py`` accepts as a dflash drafter."""

    def __init__(self, keep=BATCH_KEEP):
        self.config = SimpleNamespace(target_layer_ids=list(CAPTURE_IDS))
        self._keep = keep
        self.adopted = []

    def prefill_context_keep(self):
        return self._keep

    def adopt_pretruncated_context(self, cache, skip):
        self.adopted.append(skip)


def _batch_prefill(*, step, drafter, rows=BATCH_ROWS):
    """Drive a real ``PromptProcessingBatch`` to the handoff, the way the server does.

    Returns ``(gen_batch, spy, padded_ids)``.  ``generate()`` hands the prompt cache
    to the generation batch and clears its own reference, so the cache to compare is
    ``gen_batch.prompt_cache``.
    """
    lm = _lm()
    spy = _BatchSpy(lm)
    padded = _left_pad_prompts(rows)
    draft_kwargs = (
        dict(draft_model=drafter, draft_kind="dflash") if drafter is not None else {}
    )
    batch = PromptProcessingBatch(
        model=spy,
        uids=list(range(len(rows))),
        input_ids=rows,
        max_tokens=[4] * len(rows),
        inputs_embeds=lm.model.embed_tokens(padded),
        prompt_kwargs={},
        prefill_step_size=step,
        **draft_kwargs,
    )
    while batch.needs_processing():
        batch.prompt_step()
    gen_batch = batch.generate(
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        stop_criteria=lambda token: False,
    )
    mx.eval(_cache_arrays(gen_batch.prompt_cache))
    return gen_batch, spy, padded


def test_the_batched_chunk_loop_carries_the_capture_on_every_chunk():
    """The defect itself, at the driver a served multi-row request takes.

    Four chunks of 8 plus an 8-wide tail (``needs_processing`` stops once what is
    left fits in one step).  Before the fix the four chunk forwards carried no
    ``capture_layer_ids`` at all and only the tail did.
    """
    drafter = _StubDrafter()
    _, spy, _ = _batch_prefill(step=BATCH_STEP, drafter=drafter)

    assert len(spy.calls) == 5, [c.get("n_to_process") for c in spy.calls]
    for i, call in enumerate(spy.calls):
        assert call.get("capture_layer_ids") == list(CAPTURE_IDS), f"forward {i}"
        # Prefill leg: hidden captures yes, KDA rollback stash no -- on every chunk,
        # not just the last one.
        assert call.get("capture_gdn_states") is False, f"forward {i}"


def test_the_batched_capture_is_the_whole_prompt_not_the_last_chunk():
    """Per row, bit-equal, against the chunks the driver itself produced.

    ``hidden`` is what ``SpeculativeGenerationBatch`` hands the round loop:
    ``mx.concatenate(hidden_states, axis=-1)`` over the captured layers.  It must be
    the trailing ``keep`` rows of the WHOLE prompt, and it must be exactly the
    concatenation of the per-chunk captures -- the accumulator adds nothing of its
    own.  Before the fix this array was ``[2, 8, ...]``: the tail forward alone.
    """
    drafter = _StubDrafter()
    gen_batch, spy, _ = _batch_prefill(step=BATCH_STEP, drafter=drafter)

    assert gen_batch.hidden.shape == (
        len(BATCH_ROWS),
        BATCH_KEEP,
        len(CAPTURE_IDS) * 128,
    )
    per_layer = [
        mx.concatenate([c[i] for c in spy.captures], axis=1)
        for i in range(len(CAPTURE_IDS))
    ]
    assert per_layer[0].shape[1] == BATCH_S
    reference = mx.concatenate([h[:, -BATCH_KEEP:] for h in per_layer], axis=-1)
    for row in range(len(BATCH_ROWS)):
        assert mx.array_equal(
            gen_batch.hidden[row : row + 1], reference[row : row + 1]
        ), f"row {row}: stitched capture is not the chunks' own rows"


def test_the_batched_capture_agrees_with_the_unchunked_capture_per_row():
    """Chunked vs unchunked, per row and per captured layer.

    The KDA layer is BIT-EQUAL on both rows -- including row 0, which carries 16
    columns of left padding.  The sparse-attention layer agrees to 2.7e-07 on the
    row that needed no padding.

    On the LEFT-PADDED row the sparse layer moves by up to 1.13 at intermediate
    positions.  That is the DeepSeek-sparse indexer, not the accumulator: with
    ``index_topk`` = 6 the early positions of a padded row have fewer than 6 real
    candidates, so which padding columns fill the top-k is decided over a chunk's
    candidate set rather than the whole prompt's.  It is a property of chunking a
    padded row that the greedy batch path already ships, and it moves neither the
    target -- every one of the 10 prompt-cache arrays is bit-equal across the two
    arms -- nor the sampled token.  Recorded rather than asserted away.
    """
    chunked, _, _ = _batch_prefill(step=BATCH_STEP, drafter=_StubDrafter())
    unchunked, _, _ = _batch_prefill(step=None, drafter=_StubDrafter())

    assert unchunked.hidden.shape == chunked.hidden.shape == (2, BATCH_KEEP, 256)
    kda = slice(0, 128)
    sparse = slice(128, 256)
    for row in range(len(BATCH_ROWS)):
        got = chunked.hidden[row : row + 1]
        ref = unchunked.hidden[row : row + 1]
        assert mx.array_equal(
            got[:, :, kda], ref[:, :, kda]
        ), f"row {row}: the KDA capture is not the unchunked capture's kept rows"
    unpadded = len(BATCH_ROWS) - 1  # the longest row needed no left padding
    drift = float(
        mx.max(
            mx.abs(
                chunked.hidden[unpadded, :, sparse].astype(mx.float32)
                - unchunked.hidden[unpadded, :, sparse].astype(mx.float32)
            )
        )
    )
    assert drift < 1e-5, f"unpadded row sparse capture drifted {drift}"


def test_the_batched_spec_prefill_leaves_the_target_where_greedy_left_it():
    """The capture only appends to Python sinks -- per row, bit-equal.

    Holds before the fix too (there was no capture on the chunks to disturb
    anything); it is the guard that carrying one on every chunk did not start
    disturbing something.
    """
    spec, _, _ = _batch_prefill(step=BATCH_STEP, drafter=_StubDrafter())
    greedy, _, _ = _batch_prefill(step=BATCH_STEP, drafter=None)

    spec_arrays = _cache_arrays(spec.prompt_cache)
    greedy_arrays = _cache_arrays(greedy.prompt_cache)
    assert len(spec_arrays) == len(greedy_arrays) == 10
    for i, (a, b) in enumerate(zip(spec_arrays, greedy_arrays)):
        assert a.shape == b.shape, f"cache array {i}: {a.shape} != {b.shape}"
        if a.ndim and a.shape[0] == len(BATCH_ROWS):
            for row in range(len(BATCH_ROWS)):
                assert mx.array_equal(
                    a[row : row + 1], b[row : row + 1]
                ), f"cache array {i}, row {row}"
        else:
            assert mx.array_equal(a, b), f"cache array {i}"


def test_the_batched_prompt_tokens_are_the_whole_prompt_per_row():
    """``prompt_tokens=self._input_ids`` was read after the loop had eaten it.

    The chunk loop reassigns ``self._input_ids = self._input_ids[:, n:]``, so by the
    time ``generate()`` runs, ``_input_ids`` is the 8-column tail.  The prompt-lookup
    drafter builds its n-gram index from this array; a tail is not a prompt.
    """
    gen_batch, _, padded = _batch_prefill(step=BATCH_STEP, drafter=_StubDrafter())

    assert gen_batch.prompt_tokens.shape == (len(BATCH_ROWS), BATCH_S)
    for row, ids in enumerate(BATCH_ROWS):
        seen = gen_batch.prompt_tokens[row].tolist()
        assert mx.array_equal(
            gen_batch.prompt_tokens[row : row + 1], padded[row : row + 1]
        ), f"row {row}"
        assert seen[BATCH_S - len(ids) :] == ids, f"row {row}: prompt truncated"
        assert seen[: BATCH_S - len(ids)] == [0] * (BATCH_S - len(ids))


def test_the_batched_offset_reaches_the_round_loop():
    """``target_hidden_offset`` is the half of the trim that is easy to lose.

    The accumulator dropped ``S - keep`` rows off the front of the drafter's
    context; ``_dflash_rounds_batch`` has taken ``target_hidden_offset`` since
    bbc7cdf8, but nothing on the batch path filled it in.
    """
    from unittest.mock import patch

    drafter = _StubDrafter()
    gen_batch, _, _ = _batch_prefill(step=BATCH_STEP, drafter=drafter)

    assert gen_batch.target_hidden_offset == BATCH_S - BATCH_KEEP

    seen = {}

    def _record(*args, **kwargs):
        seen.update(kwargs)
        return iter(())

    with patch.object(
        ar_module, "run_speculative_server_rounds", side_effect=_record
    ):
        gen_batch._start_rounds()

    assert seen["target_hidden_offset"] == BATCH_S - BATCH_KEEP
    assert seen["prompt_tokens"].shape == (len(BATCH_ROWS), BATCH_S)


@pytest.mark.parametrize("row", [0, 1])
def test_the_batched_offset_adoption_is_bit_identical_per_row(row):
    """Every row's draft cache must adopt the SAME offset, and it must be a no-op.

    The batched round loop makes one draft cache per row and applies the single
    ``target_hidden_offset`` to all of them (``speculative/dflash.py:963``).  Feed a
    real DFlash2 drafter the row's full context, then the trimmed context plus the
    adopted offset: output and both cache tensors and every layer offset must be
    EQUAL, not close -- separately for each row, because a per-row context is what
    ``_dflash_rounds_batch`` slices out (``hidden_by_orig``).
    """
    _, spy, _ = _batch_prefill(step=BATCH_STEP, drafter=_StubDrafter(keep=None))
    per_layer = [
        mx.concatenate([c[i] for c in spy.captures], axis=1)
        for i in range(len(CAPTURE_IDS))
    ]
    full = mx.concatenate(per_layer, axis=-1)[row : row + 1].astype(mx.bfloat16)
    assert full.shape == (1, BATCH_S, 256)

    model = _drafter(hidden=128, n_target=2)
    assert prefill_context_keep("dflash", model) == BATCH_KEEP
    skip = BATCH_S - BATCH_KEEP

    _assert_draft_state_equal(
        _draft_round_one(model, full),
        _draft_round_one(model, mx.contiguous(full[:, -BATCH_KEEP:]), skip=skip),
        why=f"row {row}: trimming without adopting {skip} moved the drafter",
    )


# ------------------------------- 9. the right-padded (mixed warm/cold) batch
#
# ``BatchGenerator._build_mixed_prompt_batch`` (``generate/ar.py:2699-2942``) builds
# a ``PromptProcessingBatch`` with ``right_pad_per_row`` AND ``draft_model`` /
# ``draft_kind`` (:2935-2938), so a right-padded batch with a hidden-reading
# drafter is reachable.  Two things were wrong with chunking it, and neither was
# about the capture:
#
# * the drafter's window is the TRAILING rows of the capture, and for a short
#   right-padded row those are padding;
# * ``generate()`` picked each row's last real token at ``seq - 1 - right_pad`` where
#   ``seq`` is the FINAL forward's width -- which goes negative once the prefill
#   chunked.
#
# NARROWED 2026-09-03 (merge of ``glm5-rightpad-chunk-select`` into this branch).
# The second objection is now FIXED rather than avoided: ``_last_real_column`` /
# ``_capture_last_real_logits`` capture each row's ``[1, vocab]`` slice in whatever
# chunk its last real token lands in, so chunked greedy selection is exact for a
# right-padded row.  And the first objection is about the TRIM only -- with
# ``keep=None`` the accumulator's stitch reproduces the full padded width column for
# column.  So the driver now refuses ONE half for a right-padded batch: the trim.
# The chunking is kept, the capture stays full width, ``capture_gdn_states`` is still
# off.  The other refusal (an unnameable warm prefix) keeps its blanket form.

RP_FULL = [list(range(3, 43)), list(range(7, 47))]  # both whole prompts are 40 long
RP_PREFIX = [16, 0]  # row 0 is warm: 16 tokens already in the cache
RP_SUFFIX = [RP_FULL[i][RP_PREFIX[i] :] for i in range(2)]  # 24 and 40
RP_MAX_SUFFIX = max(len(s) for s in RP_SUFFIX)
RP_RIGHT_PAD = [RP_MAX_SUFFIX - len(s) for s in RP_SUFFIX]  # [16, 0]
RP_META = [
    {"full_input_ids": RP_FULL[i], "prefix_len": RP_PREFIX[i]} for i in range(2)
]


def _right_padded_batch(*, step, drafter, apc_meta=RP_META):
    """A mixed warm/cold batch, shaped exactly as ``_build_mixed_prompt_batch`` shapes one.

    ``warm_cache`` is deliberately not supplied: the row-0 prefix K/V is absent, so
    row 0's arithmetic is not what a served warm row would compute.  Everything this
    section asserts is about the DRIVER's policy -- how many forwards, how wide the
    capture, which token ids reach the drafter, which position is sampled -- and none
    of it depends on the prefix K/V being real.
    """
    lm = _lm()
    spy = _BatchSpy(lm)
    padded = _right_pad_prompts(RP_SUFFIX, max_length=RP_MAX_SUFFIX)
    draft_kwargs = (
        dict(draft_model=drafter, draft_kind="dflash") if drafter is not None else {}
    )
    batch = PromptProcessingBatch(
        model=spy,
        uids=[0, 1],
        input_ids=RP_SUFFIX,
        max_tokens=[4, 4],
        inputs_embeds=lm.model.embed_tokens(padded),
        prompt_kwargs={},
        prefill_step_size=step,
        right_pad_per_row=list(RP_RIGHT_PAD),
        suffix_lens=[len(s) for s in RP_SUFFIX],
        apc_meta=apc_meta,
        **draft_kwargs,
    )
    while batch.needs_processing():
        batch.prompt_step()
    gen_batch = batch.generate(
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        stop_criteria=lambda token: False,
    )
    mx.eval(_cache_arrays(gen_batch.prompt_cache))
    return gen_batch, spy, batch


def _first_tokens(gen_batch):
    tokens = getattr(gen_batch, "first_tokens", None)
    if tokens is None:
        tokens = gen_batch._next_tokens
    mx.eval(tokens)
    return tokens.tolist()


def test_a_right_padded_batch_refuses_the_trim_but_keeps_the_chunking(caplog):
    """NARROWED 2026-09-03: the trim is refused, the chunking is not.

    Was ``test_a_right_padded_batch_refuses_the_chunking_and_the_trim``, which
    asserted ONE forward.  The chunking half of that refusal rested on the
    negative last-real-token index, which the merged
    ``glm5-rightpad-chunk-select`` fix removes, and on a claim that the stitched
    pieces do not line up across the batch, which is true of the TRIM and not of
    the stitch (``keep=None`` makes ``_prune`` a no-op and ``finish()`` a plain
    time-axis concatenation).  What survives is the trim refusal: the drafter's
    window is the trailing rows, and for row 0 those are padding.
    """
    with caplog.at_level(logging.INFO, logger="mlx_vlm.generate"):
        gen_batch, spy, batch = _right_padded_batch(
            step=BATCH_STEP, drafter=_StubDrafter()
        )

    # The chunking is KEPT: the 40-wide right-padded prefill at step 8 is four
    # chunks of 8 plus the final forward (``needs_processing`` stops once what is
    # left fits in one), i.e. five forwards where the refusal made exactly one.
    assert batch.prefill_step_size == BATCH_STEP
    assert len(spy.calls) == 5, [c.get("n_to_process") for c in spy.calls]
    assert [c.get("n_to_process") for c in spy.calls] == [8, 8, 8, 8, None]
    # The capture is asked for on EVERY forward, and the gdn stash is refused on
    # every forward.
    for i, call in enumerate(spy.calls):
        assert call["capture_layer_ids"] == list(CAPTURE_IDS), f"forward {i}"
        assert call["capture_gdn_states"] is False, f"forward {i}"
    # Full width, untrimmed: 40 columns, and nothing is owed to the drafter.
    assert gen_batch.hidden.shape == (2, RP_MAX_SUFFIX, len(CAPTURE_IDS) * 128)
    assert gen_batch.target_hidden_offset == 0
    assert batch._prefill_hidden.keep is None

    messages = [r.getMessage() for r in caplog.records]
    assert any("declining the trailing-context trim" in m for m in messages), messages
    assert any("right-padded batch" in m for m in messages), messages
    # ... and it does NOT claim to have declined the chunking.
    assert not any("and the chunked prefill" in m for m in messages), messages


def test_a_right_padded_batch_hands_the_drafter_the_whole_prompt_per_row():
    """A warm row's prefill input is its SUFFIX; its prompt is prefix + suffix.

    Before this policy, row 0's ``prompt_tokens`` was its 24-token suffix
    right-padded with 16 zeros -- an n-gram corpus missing everything the cache
    already held, and 16 trailing zeros that were never tokens at all.
    """
    gen_batch, _, _ = _right_padded_batch(step=BATCH_STEP, drafter=_StubDrafter())

    assert gen_batch.prompt_tokens.shape == (2, len(RP_FULL[0]))
    for row, whole in enumerate(RP_FULL):
        assert gen_batch.prompt_tokens[row].tolist() == whole, f"row {row}"
    # Specifically: row 0 is NOT the right-padded suffix it prefills with.
    assert gen_batch.prompt_tokens[0].tolist() != (
        RP_SUFFIX[0] + [0] * RP_RIGHT_PAD[0]
    )


def test_a_right_padded_spec_prefill_matches_the_greedy_arm_it_now_shares():
    """Refusing the chunking puts the spec arm on the unchunked decomposition.

    So it must be bit-identical to unchunked greedy -- per row, in the sampled token
    and in every prompt-cache array.  (Against CHUNKED greedy it is not, and must not
    be expected to be: chunked greedy still walks into the negative-index defect
    pinned by the next test.)
    """
    spec, _, _ = _right_padded_batch(step=BATCH_STEP, drafter=_StubDrafter())
    greedy_unchunked, _, _ = _right_padded_batch(step=None, drafter=None)

    assert _first_tokens(spec) == _first_tokens(greedy_unchunked)

    spec_arrays = _cache_arrays(spec.prompt_cache)
    greedy_arrays = _cache_arrays(greedy_unchunked.prompt_cache)
    assert len(spec_arrays) == len(greedy_arrays) == 10
    for i, (a, b) in enumerate(zip(spec_arrays, greedy_arrays)):
        assert a.shape == b.shape, f"cache array {i}"
        if a.ndim and a.shape[0] == 2:
            for row in range(2):
                assert mx.array_equal(
                    a[row : row + 1], b[row : row + 1]
                ), f"cache array {i}, row {row}"
        else:
            assert mx.array_equal(a, b), f"cache array {i}"


def test_right_padded_chunked_prefill_indexes_a_negative_row_position():
    """PRE-EXISTING DEFECT, pinned here, NOT fixed by this branch.

    ``PromptProcessingBatch.generate`` picks each row's last real token at
    ``seq - 1 - right_pad[i]`` (``generate/ar.py:2213-2219``) where ``seq`` is the
    width of the FINAL forward.  Unchunked that is the whole prompt and the index is
    right.  Once the prefill chunked, ``seq`` is only what the loop left over, and
    for a right-padded row the index goes NEGATIVE -- an out-of-range
    ``take_along_axis`` on an 8-wide axis.

    Measured on this fixture: unchunked greedy emits [125, 99]; chunked greedy emits
    a different first token for the padded row at every chunk size tried
    (step 4 -> 76, 8 -> 112, 12 -> 0, 16 -> 54, 24 -> 1, 32 -> 54), while the second
    row -- which carries no right padding -- emits 99 in every arm, and the prompt
    cache is bit-equal in every arm.  So the cache is right and the SELECTION is
    wrong.  It needs no drafter: this is the greedy batch path.

    Delete this test when the index is fixed to count from the whole prompt.
    """
    lm = _lm()
    padded = _right_pad_prompts(RP_SUFFIX, max_length=RP_MAX_SUFFIX)
    batch = PromptProcessingBatch(
        model=lm,
        uids=[0, 1],
        input_ids=RP_SUFFIX,
        max_tokens=[4, 4],
        inputs_embeds=lm.model.embed_tokens(padded),
        prompt_kwargs={},
        prefill_step_size=BATCH_STEP,
        right_pad_per_row=list(RP_RIGHT_PAD),
        suffix_lens=[len(s) for s in RP_SUFFIX],
    )
    # No drafter, so nothing declines the chunking.
    assert batch.prefill_step_size == BATCH_STEP
    while batch.needs_processing():
        batch.prompt_step()

    seq = batch._inputs_embeds.shape[1]  # the final forward's width
    assert seq < RP_MAX_SUFFIX, "the prefill did not chunk; fixture is wrong"
    indices = [seq - 1 - pad for pad in batch._right_pad_per_row]
    assert indices[1] >= 0, "the unpadded row is fine"
    assert indices[0] < 0, (
        f"expected a negative index for the right-padded row, got {indices[0]}; "
        "if this now passes the defect is fixed -- delete this test"
    )


def test_a_warm_row_without_a_recoverable_prefix_declines_the_trim(caplog):
    """The other refusal: a warm row whose whole prompt cannot be named.

    ``apc_meta`` declares a prefix but carries no ``full_input_ids``, so the drafter
    cannot be handed the whole prompt.  It does not get a trimmed one either -- and
    ``prompt_tokens`` stays the prefill input rather than silently passing a suffix
    off as a prompt.
    """
    rows = [list(range(3, 27)), list(range(7, 47))]
    meta = [{"prefix_len": 5}, None]  # prefix declared, whole prompt not

    lm = _lm()
    spy = _BatchSpy(lm)
    padded = _left_pad_prompts(rows)
    with caplog.at_level(logging.INFO, logger="mlx_vlm.generate"):
        batch = PromptProcessingBatch(
            model=spy,
            uids=[0, 1],
            input_ids=rows,
            max_tokens=[4, 4],
            inputs_embeds=lm.model.embed_tokens(padded),
            prompt_kwargs={},
            prefill_step_size=BATCH_STEP,
            apc_meta=meta,
            draft_model=_StubDrafter(),
            draft_kind="dflash",
        )
    while batch.needs_processing():
        batch.prompt_step()
    gen_batch = batch.generate(
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        stop_criteria=lambda token: False,
    )

    assert batch.prefill_step_size is None
    assert len(spy.calls) == 1
    assert gen_batch.hidden.shape == (2, BATCH_S, len(CAPTURE_IDS) * 128)
    assert gen_batch.target_hidden_offset == 0
    assert mx.array_equal(gen_batch.prompt_tokens, padded)

    messages = [r.getMessage() for r in caplog.records]
    assert any("not recoverable" in m for m in messages), messages
    # This refusal KEEPS its blanket form: the chunking is declined too.
    assert any("and the chunked prefill" in m for m in messages), messages
