"""A model with recurrent layers must not be handed a RIGHT-PADDED prefill.

``BatchGenerator._build_mixed_prompt_batch`` (``generate/ar.py``) admits a warm
APC-prefix row and a cold row into ONE ``PromptProcessingBatch``, right-padding
every row's suffix to the longest one so the suffix RoPE positions line up.  APC
is on by default (``server/runtime_config.py``), so this is what a served pair of
concurrent requests takes whenever either of them hits the prefix cache.

Measured on the tiny CPU fixture below (2 layers: one KDA, one DSA; greedy, no
drafter, chunk-size independent), a right-padded row's PROMPT logits agree with
the same row run ALONE to 9.5e-07 -- and its DECODE trajectory parts company at
the first step.  Two independent defects were behind that:

D1 -- the indexer's derived pool state survived the roll.
    ``BatchKVCache.finalize()`` (``models/cache.py``) rolls the K/V buffer from
    right- to left-padding, but the GLM-5 DSA indexer keeps three derived
    attributes on that same cache object (``glm5_next/language.py``:
    ``_pool``/``_fpool``, absolute column indices in the PRE-roll frame, and
    ``_no_pad``, which a right-padded prefill sets True because it marks the
    padding VALID).  The S == 1 incremental-pool branch is admitted by exactly
    ``_no_pad`` and reuses the complete pool blocks verbatim, so after the roll
    it selected blocks ``right_pad[i]`` columns to the LEFT of the row's data.
    The fix is the invalidation ``give_back_sparse_cache_rows`` already performs
    after its own per-row shift; it is asserted directly below.

D2 -- the KDA (linear-attention) state absorbed the padding, and cannot be rolled.
    ``ArraysCache.make_mask`` has a ``lengths`` branch that a right-padded batch
    never reaches (``PromptProcessingBatch`` sets ``left_padding = [0] * B`` for
    such a batch and ``left_padding`` wins that ``if``); and even with the mask
    restored, the conv state is taken at the padded column and the forget gate
    decays the recurrent state for ``right_pad[i]`` extra steps.  A recurrent
    state does not record the column it came from, so unlike a K/V buffer there
    is nothing to roll it back with.

D1 was fixed.  D2 cannot be, at this layer: it is a property of right padding
itself.  Measured here, per fixture (row 0 padded, row 1 not; step-1 decode
logits vs the same row run alone, on a 6.33 logit scale):

    suffix 22 / pad 18:  1.127  ->  0.093     tokens agree after (by margin)
    suffix 23 / pad 17:  1.232  ->  0.157     tokens still differ
    suffix 24 / pad 16:  1.476  ->  0.131     tokens still differ
    suffix 25 / pad 15:  1.503  ->  1.589     tokens still differ
    suffix 26 / pad 14:  2.046  ->  0.186     tokens agree after (by margin)
    suffix 27 / pad 13:  2.290  ->  0.936     tokens agree after (by margin)

Three of six fixtures come out with the right tokens after D1, and none of them
comes out CORRECT: the residual is never zero, and which side of a token margin
it lands on is luck.  So the ruling is not "D1 fixed it" but decline: the
capability ``model_supports_right_padded_prefill`` answers False for any model
whose ``make_cache()`` prototype holds an ``ArraysCache``, and
``_apply_right_pad_policy`` then batches only rows whose suffix lengths are
EQUAL -- equal suffixes mean no right padding at all, which is the only shape
this model's KDA layers can share a prefill in.  Left-padded cold-only batching
is untouched; the defect is right padding.

The throughput cost is real and is counted: ``prefill_batch_refusal_counts()``
(surfaced in the server's runtime snapshot as ``prefill_batch_refusals``) reports
``right_pad_kda`` events and ``right_pad_kda_rows_deferred`` rows.

DEVICE.  Everything here is CPU (``MLX_DEFAULT_DEVICE=cpu``, pinned by conftest).
The policy claims are integer bookkeeping and device-free.  The decode-equality
claims are float and are asserted as token equality on CPU only; on Metal the
KDA scan drifts by ~1e-06 between batch widths and this 2-layer randomly
initialised model can flip a token on that, exactly as
``test_glm5_rightpad_chunked_select`` documents for its own fixtures.
"""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx_vlm.generate import ar as ar_module
from mlx_vlm.generate.ar import (
    BatchGenerator,
    PromptProcessingBatch,
    _right_pad_prompts,
    model_supports_right_padded_prefill,
    prefill_batch_refusal_counts,
    reset_prefill_batch_refusal_counts,
)
from mlx_vlm.models import cache as cache_mod
from mlx_vlm.models.glm5_next import language as glm5
from mlx_vlm.tests.test_glm5_rightpad_chunked_select import _lm, _pads

ON_GPU = mx.default_device() == mx.gpu

# Two suffixes of EQUAL length (25) and one longer one (40).  A batch of the
# first two carries no right padding; adding the third forces 15 columns of it.
SUF_A = list(range(3, 28))
SUF_B = list(range(40, 65))
SUF_C = list(range(7, 47))
# A 22-token suffix: right-padded to 40 it is the one fixture in the sweep whose
# decode selection AND tokens come out equal to the singleton once D1 is fixed.
SUF_D = list(range(3, 25))

# Worst prompt-side disagreement measured between a right-padded row and the
# same row run alone: 9.54e-07 on both devices.  The prompt is not the defect.
PROMPT_TOL = 2e-06
# The D2 residual on the (25, 40) fixture at decode step 1, measured 1.589 after
# the D1 fix (1.503 before it).  Asserted as a floor, so this test starts failing
# the day someone makes right padding actually work -- at which point the policy
# below can be relaxed, and should be.
KDA_RESIDUAL_FLOOR = 1e-2


def _run(suffixes, emit=4):
    """Prefill ``suffixes`` (right-padded iff their lengths differ) and decode.

    Returns ``(tokens_per_row, sampler_logits_per_step, gen_batch)``.
    ``sampler_logits[0]`` is the PROMPT selection; ``[t]`` for t >= 1 is decode
    step ``t``.
    """
    pads, width = _pads(suffixes)
    lm = _lm()
    embeds = lm.model.embed_tokens(_right_pad_prompts(suffixes, max_length=width))
    pad_kwargs = (
        dict(right_pad_per_row=list(pads), suffix_lens=[len(s) for s in suffixes])
        if any(pads)
        else {}
    )
    batch = PromptProcessingBatch(
        model=lm,
        uids=list(range(len(suffixes))),
        input_ids=suffixes,
        max_tokens=[emit] * len(suffixes),
        inputs_embeds=embeds,
        prompt_kwargs={},
        prefill_step_size=None,
        **pad_kwargs,
    )
    while batch.needs_processing():
        batch.prompt_step()
    logits = []

    def sampler(logprobs):
        logits.append(mx.array(logprobs))
        return mx.argmax(logprobs, axis=-1)

    gen = batch.generate(sampler=sampler, stop_criteria=lambda token: False)
    rows = [[] for _ in suffixes]
    for _ in range(emit):
        tokens, _, _, _ = gen._step()
        for i, token in enumerate(tokens):
            rows[i].append(int(token))
    mx.eval(logits)
    return rows, logits, gen


def _sequence(uid, ids, prefix_len_unused=0):
    """One entry of the tuple list ``_build_mixed_prompt_batch`` consumes."""
    return (
        uid,
        list(ids),
        4,
        {"inputs_embeds": mx.ones((1, len(ids), 4))},
        [],
        None,
    )


def _fake_generator(model):
    """A ``BatchGenerator`` with only the attributes the mixed builder reads.

    ``object.__new__`` for the same reason ``test_generate`` uses it for this
    builder: ``__init__`` wires a wired-memory limit and a tokenizer that have
    nothing to do with admission.
    """
    bg = object.__new__(BatchGenerator)
    bg.model = model
    bg.apc_manager = object()
    bg.vault = None
    bg.prefill_step_size = None
    bg.kv_bits = None
    bg.kv_group_size = 64
    bg.kv_quant_scheme = "affine"
    bg.apc_mode = "block"
    bg._right_pad_capability = None
    bg._unprocessed_sequences = []
    # ``close()``/``__del__`` reach for this; object.__new__ never ran __init__.
    bg._wire_stack = None
    return bg


def _capture_built_batches(bg, sequences, picks):
    """Run the real builder with a fake ``PromptProcessingBatch``.

    Returns the kwargs the builder would have constructed the batch with, or
    ``None`` if it declined.
    """
    captured = {}

    def fake_prompt_batch(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    with (
        patch.object(BatchGenerator, "_apc_pick_for", side_effect=list(picks)),
        patch.object(
            ar_module._apc, "make_warm_batch_kv_cache_multi", return_value=([], 4)
        ),
        patch.object(
            ar_module,
            "_generate_module_override",
            lambda name, default: fake_prompt_batch,
        ),
    ):
        batch = bg._build_mixed_prompt_batch(sequences)
    return captured if batch is not None else None


def _pick(prefix_len, full_ids):
    return {
        "matched_blocks": [],
        "prefix_len": prefix_len,
        "extra_hash": 0,
        "full_input_ids": list(full_ids),
    }


# --------------------------------------------------------- 1. the capability


def test_glm5_next_declines_right_padded_prefill():
    """Declared on the class AND derived from the cache prototype -- both False."""
    lm = _lm()
    assert lm.supports_right_padded_prefill is False
    assert model_supports_right_padded_prefill(lm) is False

    # The derivation on its own, with no declaration in reach: a bare object
    # that only forwards this model's ``make_cache``.
    proxy = SimpleNamespace(make_cache=lm.make_cache)
    assert model_supports_right_padded_prefill(proxy) is False
    assert any(
        isinstance(c, cache_mod.ArraysCache) for c in lm.make_cache()
    ), "fixture no longer has a linear-attention layer"


def test_a_kv_only_model_still_right_pads():
    """The capability must not become a blanket refusal: KV-only models are fine.

    Their whole state is the per-column K/V buffer that ``finalize()`` rolls.
    """
    kv_only = SimpleNamespace(
        make_cache=lambda: [cache_mod.KVCache() for _ in range(2)]
    )
    assert model_supports_right_padded_prefill(kv_only) is True

    nested = SimpleNamespace(
        make_cache=lambda: [
            cache_mod.CacheList(cache_mod.KVCache(), cache_mod.KVCache())
        ]
    )
    assert model_supports_right_padded_prefill(nested) is True

    # A model with no prototype at all gets plain BatchKVCache from _make_cache.
    assert model_supports_right_padded_prefill(SimpleNamespace()) is True

    # ... and the recurrent state is found even when it is nested in a CacheList,
    # which is how falcon_h1 and inkling build theirs.
    hybrid = SimpleNamespace(
        make_cache=lambda: [
            cache_mod.CacheList(cache_mod.ArraysCache(size=2), cache_mod.KVCache())
        ]
    )
    assert model_supports_right_padded_prefill(hybrid) is False


def test_an_unbuildable_prototype_declines_rather_than_assumes():
    def boom():
        raise RuntimeError("no prototype")

    model = SimpleNamespace(make_cache=boom)
    assert model_supports_right_padded_prefill(model) is False


# ------------------------------------------------------------ 2. the policy


def test_the_builder_splits_a_mixed_batch_with_unequal_suffixes():
    """THE ruling.  B = 2, one warm row, unequal suffixes -> no right padding.

    At 47b503a3 this built ONE batch with ``right_pad_per_row = [0, 15]``; the
    padded row's decode then diverged from the first step (see the module
    docstring).  Now the head-length group is admitted alone and the other row
    goes back to the queue, to be served in its own batch.
    """
    reset_prefill_batch_refusal_counts()
    lm = _lm()
    bg = _fake_generator(lm)
    warm = list(range(100))  # 100 tokens, 75 of them warm -> suffix 25
    cold = list(range(40))  # cold -> suffix 40
    sequences = [_sequence(1, warm), _sequence(2, cold)]
    picks = [_pick(75, warm), None]

    built = _capture_built_batches(bg, sequences, picks)

    assert built is not None
    assert built["uids"] == [1]
    assert built["right_pad_per_row"] == [0]
    assert built["suffix_lens"] == [25]
    counts = prefill_batch_refusal_counts()
    assert counts["right_pad_kda"] == 1
    assert counts["right_pad_kda_rows_deferred"] == 1

    # The deferred row is served in its own batch on the next pass.
    pending = bg._pending_after_admission(
        sequences, len(sequences), SimpleNamespace(uids=built["uids"])
    )
    assert [s[0] for s in pending] == [2]


def test_equal_suffix_rows_still_share_one_batch():
    """The fast path is kept, not removed: equal suffixes need no padding."""
    reset_prefill_batch_refusal_counts()
    bg = _fake_generator(_lm())
    warm = list(range(100))
    cold = list(range(25))
    sequences = [_sequence(1, warm), _sequence(2, cold)]
    picks = [_pick(75, warm), None]  # suffixes 25 and 25

    built = _capture_built_batches(bg, sequences, picks)

    assert built is not None
    assert built["uids"] == [1, 2]
    assert built["right_pad_per_row"] == [0, 0]
    assert prefill_batch_refusal_counts() == {}


def test_the_policy_admits_the_whole_equal_length_group_not_just_one_row():
    """B = 3, lengths (25, 25, 40): the two that match batch, the third defers."""
    reset_prefill_batch_refusal_counts()
    bg = _fake_generator(_lm())
    warm = list(range(100))
    sequences = [
        _sequence(1, warm),
        _sequence(2, list(range(25))),
        _sequence(3, list(range(40))),
    ]
    picks = [_pick(75, warm), None, None]

    built = _capture_built_batches(bg, sequences, picks)

    assert built["uids"] == [1, 2]
    assert built["right_pad_per_row"] == [0, 0]
    assert prefill_batch_refusal_counts()["right_pad_kda_rows_deferred"] == 1


def test_the_policy_anchors_on_the_queue_head_so_no_row_starves():
    """The admitted group is the HEAD row's, even when a bigger group exists.

    Anchoring on the largest group, or on the first warm row, would let a stream
    of same-length arrivals defer the oldest row forever.  Here rows 1 and 2 are
    the same length and row 0 (the head) is alone; row 0 is what gets served.
    """
    reset_prefill_batch_refusal_counts()
    bg = _fake_generator(_lm())
    head = list(range(100))
    sequences = [
        _sequence(1, head),
        _sequence(2, list(range(40))),
        _sequence(3, list(range(40))),
    ]
    picks = [_pick(75, head), None, None]

    built = _capture_built_batches(bg, sequences, picks)

    assert built["uids"] == [1]
    assert prefill_batch_refusal_counts()["right_pad_kda_rows_deferred"] == 2


def test_a_head_group_with_no_warm_row_falls_back_to_the_cold_path():
    """Declining outright is correct here, and the caller left-pads instead.

    There is nothing to build a warm batch out of once the head group is all
    cold, so the builder returns ``None`` and the caller's cold-only path admits
    the whole window LEFT-padded -- which is sound.  The cost, and it is a real
    one, is that the warm row in that window re-prefills its prefix this round.
    """
    reset_prefill_batch_refusal_counts()
    bg = _fake_generator(_lm())
    warm = list(range(100))
    sequences = [_sequence(1, list(range(40))), _sequence(2, warm)]
    picks = [None, _pick(75, warm)]  # head is cold, suffix 40; warm row suffix 25

    assert _capture_built_batches(bg, sequences, picks) is None
    assert prefill_batch_refusal_counts()["right_pad_kda"] == 1


def test_a_kv_only_model_keeps_the_right_padded_batch():
    """The policy fires on the capability, not on the shape of the batch."""
    reset_prefill_batch_refusal_counts()
    bg = _fake_generator(SimpleNamespace(layers=[object()]))
    warm = list(range(100))
    sequences = [_sequence(1, warm), _sequence(2, list(range(40)))]
    picks = [_pick(75, warm), None]

    built = _capture_built_batches(bg, sequences, picks)

    assert built["uids"] == [1, 2]
    assert built["right_pad_per_row"] == [15, 0]
    assert prefill_batch_refusal_counts() == {}


def test_pending_after_admission_keeps_the_deferred_rows_at_the_front():
    bg = _fake_generator(_lm())
    window = [_sequence(i, list(range(4))) for i in (1, 2, 3)]
    bg._unprocessed_sequences = window + [_sequence(9, list(range(4)))]

    pending = bg._pending_after_admission(window, 3, SimpleNamespace(uids=[2]))

    assert [s[0] for s in pending] == [1, 3, 9]


# --------------------------------------- 3. what the admitted batches decode


@pytest.mark.skipif(
    ON_GPU, reason="token equality across batch widths is CPU-only here"
)
def test_the_admitted_group_decodes_like_singletons():
    """The claim the policy exists to buy, on real forwards.

    The equal-suffix pair the policy admits decodes token for token like the two
    rows run alone, for four steps; so does the deferred row in its own batch.
    """
    grouped, _, _ = _run([SUF_A, SUF_B])
    alone_a, _, _ = _run([SUF_A])
    alone_b, _, _ = _run([SUF_B])
    deferred, _, _ = _run([SUF_C])
    alone_c, _, _ = _run([SUF_C])

    assert grouped == [alone_a[0], alone_b[0]]
    assert deferred == alone_c


@pytest.mark.skipif(
    ON_GPU, reason="token equality across batch widths is CPU-only here"
)
def test_the_batch_the_old_builder_would_have_made_still_decodes_wrong():
    """The KDA residual, recorded rather than fixed -- this is D2.

    ``_run([SUF_A, SUF_C])`` is exactly the batch 47b503a3 built for this pair:
    row 0 right-padded by 15.  Its PROMPT logits agree with the singleton to
    9.5e-07 and its DECODE parts company at step 1 by 1.589 on a 6.33 scale.
    The unpadded row is unharmed, which is what makes this padding and not
    batching.

    If this test ever fails because the residual vanished, right padding has been
    made to work and ``_apply_right_pad_policy`` should be relaxed accordingly.
    """
    padded, padded_logits, _ = _run([SUF_A, SUF_C])
    alone_a, logits_a, _ = _run([SUF_A])
    alone_c, logits_c, _ = _run([SUF_C])

    prompt_drift = float(mx.max(mx.abs(padded_logits[0][0] - logits_a[0][0])))
    assert prompt_drift < PROMPT_TOL, prompt_drift

    step1 = float(mx.max(mx.abs(padded_logits[1][0] - logits_a[1][0])))
    assert step1 > KDA_RESIDUAL_FLOOR, step1
    assert padded[0] != alone_a[0]

    # The row that carries no padding is bit-clean at the prompt and stays
    # within the batch-width drift for every decode step.
    assert padded[1] == alone_c[0]
    for t in range(len(padded_logits)):
        drift = float(mx.max(mx.abs(padded_logits[t][1] - logits_c[t][0])))
        assert drift < PROMPT_TOL, (t, drift)


# ------------------------------------------------- 4. D1, asserted directly


def _indexer_cache(gen_batch):
    """The DSA layer's indexer side cache (layer 1; layer 0 is the KDA one)."""
    return gen_batch.prompt_cache[1].caches[1]


def test_finalize_invalidates_the_indexer_pool_after_rolling_right_padding():
    """D1.  The roll leaves no pool behind, and no ``_no_pad`` licence to use one.

    Built directly, bypassing the policy: the point is that a consumer that
    still right-pads is at least INDEXER-correct.  At 47b503a3 this cache came
    out of ``finalize()`` with ``_pool`` set (absolute columns in the pre-roll
    frame) and ``_no_pad`` True, which is precisely the pair that admits the
    S == 1 incremental branch.
    """
    _, _, gen = _run([SUF_A, SUF_C], emit=0)
    indexer = _indexer_cache(gen)

    assert getattr(indexer, "_pool", None) is None
    assert getattr(indexer, "_fpool", None) is None
    assert getattr(indexer, "_no_pad", None) is False
    # The roll did happen -- otherwise the assertions above are vacuous.
    assert indexer.left_padding.tolist() == [15, 0]


def test_the_first_decode_step_repools_the_whole_row_instead_of_reusing_blocks():
    """D1, at the mechanism: with the pool dropped, step 1 takes the eager path.

    ``_pooled_states`` is called on the FULL width (41 = 40 prefill columns + 1
    decode token), not on a short suffix, which is what the incremental branch
    would have passed.  This is the observable difference the invalidation buys.
    """
    _, _, gen = _run([SUF_A, SUF_C], emit=0)
    seen = []
    original = glm5.Glm5NextIndexer._pooled_states

    def spy(self, keys, gate, valid, layout=None):
        seen.append(keys.shape[1])
        return original(self, keys, gate, valid, layout=layout)

    glm5.Glm5NextIndexer._pooled_states = spy
    try:
        gen._step()
    finally:
        glm5.Glm5NextIndexer._pooled_states = original

    assert seen, "the indexer did not pool at all"
    assert max(seen) == 41, seen


def test_a_left_padded_batch_is_untouched_by_the_invalidation():
    """Nothing above may cost the ordinary cold batch anything.

    A batch with no right padding never enters the ``finalize()`` branch, so its
    pool survives its own decode steps exactly as before -- the incremental
    branch is still reachable, and it is still what a same-length batch takes.
    """
    _, _, gen = _run([SUF_A, SUF_B], emit=1)
    indexer = _indexer_cache(gen)

    assert indexer.left_padding.tolist() == [0, 0]
    assert getattr(indexer, "_pool", None) is not None
    assert getattr(indexer, "_no_pad", None) is True


def _decode_step_topk(gen_batch):
    """The indexer's selected KV columns on ONE decode step, per DSA layer."""
    seen = []
    original = glm5.Glm5NextIndexer.__call__

    def spy(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        seen.append(result)
        return result

    glm5.Glm5NextIndexer.__call__ = spy
    try:
        gen_batch._step()
    finally:
        glm5.Glm5NextIndexer.__call__ = original
    return seen


@pytest.mark.skipif(ON_GPU, reason="selection ties resolve per device; CPU pin")
def test_the_decode_selection_is_the_singletons_once_the_pool_is_invalidated():
    """D1, as the selection it corrupted -- the sharpest form of the claim.

    The padded row's cache is left-padded by ``pad`` after the roll, so its
    absolute columns are the singleton's plus ``pad``.  Shift them back and the
    two selections must be the same set.  Measured on this fixture (suffix 22,
    pad 18), before and after the invalidation:

        before:  batch [27 28 29 30 31 32 39 40] -> shifted [ 9 10 11 12 13 14 21 22]
        after:   batch [24 25 26 30 31 32 39 40] -> shifted [ 6  7  8 12 13 14 21 22]
        singleton                                           [ 6  7  8 12 13 14 21 22]

    One pool block -- ``index_kpool`` = 3 contiguous columns -- was being taken
    from the pre-roll frame.  That is D1 exactly.

    D2 is untouched by this and still visible one fixture over: at suffix 25 /
    pad 15 the shifted selection is [9 10 11 21 22 23 24 25] against the
    singleton's [3 4 5 21 22 23 24 25] AFTER the fix, because the KDA state
    feeding the indexer's query is itself wrong.  No pool bookkeeping can reach
    that, which is why the batch is declined rather than repaired.
    """
    pad = len(SUF_C) - len(SUF_D)
    _, _, batched = _run([SUF_D, SUF_C], emit=0)
    _, _, alone = _run([SUF_D], emit=0)

    got = _decode_step_topk(batched)
    ref = _decode_step_topk(alone)
    assert len(got) == len(ref) == 1, (len(got), len(ref))

    batch_row0 = got[0][0, 0, 0]
    shifted = mx.where(batch_row0 >= 0, batch_row0 - pad, batch_row0)
    mx.eval(shifted)
    assert sorted(int(x) for x in shifted.tolist()) == sorted(
        int(x) for x in ref[0][0, 0, 0].tolist()
    )
