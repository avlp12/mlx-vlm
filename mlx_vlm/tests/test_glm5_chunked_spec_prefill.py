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
"""

import os

import mlx.core as mx
import mlx.nn as nn
import pytest

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
