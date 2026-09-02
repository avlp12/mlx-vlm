"""Per-row batched DFlash2 rollback: no clamp, and every row lands where its
own committed tokens put it.

The uniform-acceptance clamp (``test_dflash_ragged_rollback.py``) bought
correctness with tokens: every row above the batch minimum gave back
``accepted_i - min(accepted)`` every round and re-drafted them next round.
These tests drive the real ``_dflash_rounds_batch`` over a real (tiny)
same-architecture Glm5Next target with RAGGED accepts and NO clamp, and assert
the same invariant the clamp was protecting:

    after the round, each row's cache is what a fresh forward over exactly that
    row's committed tokens produces.

The caches are the ones a served batch actually gets -- ``generate/ar.py``'s
``_make_cache`` conversion, i.e. ``CacheList(BatchKVCache, BatchKVCache)`` for
the sparse layers and ``ArraysCache`` for the KDA layers -- because per-row
lengths are representable only there. On a scalar-offset ``KVCache`` the loud
guard still fires; that is pinned below too.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.speculative import dflash as dflash_utils
from mlx_vlm.speculative.common import (
    _batch_acceptance_must_be_uniform,
    _format_speculative_stats,
    _record_per_row_rollback,
    _reset_per_row_rollback,
    _supports_per_row_rollback,
)
from mlx_vlm.tests.test_dflash_ragged_rollback import (
    BLOCK_TOTAL,
    PROMPT,
    _tiny_glm5_next_target,
)

# Distinct per row, so a row that ends up with another row's tokens is visible
# in the cache and not merely in a counter.
DRAFTS = {0: [11, 12, 13, 14], 1: [21, 22, 23, 24], 2: [31, 32, 33, 34]}
BONUS = [5, 7, 9]
# The ragged round under test: row 0 accepts 3 of its 4 drafts, row 1 accepts 1.
RAGGED = [3, 1]
# The token the stub "target" emits at the rejection point, plus the row index.
# It MUST be below vocab_size (32): that token is fed back as the next round's
# input, and an out-of-range id is an unchecked gather past the embedding table
# -- reading whatever memory happens to be there. With 90 the two-round and B=3
# cases failed about a third of the time, with values that depended on what the
# process had allocated earlier; nothing about the rollback was involved.
SENTINEL = 28

# A batched B>1 reduction and a B=1 reference do not agree bit for bit in
# float32; the same tolerance (and the same reason) as the clamp tests.
_ROW_TOL = 1e-4


@pytest.fixture(scope="module")
def model():
    """One tiny target for the whole module: the model is stateless between
    tests (every test brings its own caches) and building it is the slow part."""
    mx.random.seed(3)
    m = _tiny_glm5_next_target()
    mx.eval(m.parameters())
    return m


class _StubDrafter:
    """Fixed per-row draft block -- just the surface the batch loop touches."""

    requires_uniform_batch_acceptance = False
    dflash_deferred_walk = False

    def __init__(self):
        self.config = SimpleNamespace(
            block_size=BLOCK_TOTAL,
            runtime_block_size=BLOCK_TOTAL,
            target_layer_ids=[0],
        )
        self.accept_lens = []
        self.draft_lens = []
        self.rows = []

    def reset(self, model=None):
        self.accept_lens = []
        self.draft_lens = []

    def make_cache(self):
        return []

    def draft_block(self, bonus, hidden, cache, bs, sampler, token_dtype):
        # The batch loop drafts row by row, in ACTIVE-slot order, and the active
        # set shrinks when a row finishes -- so the row a call belongs to is
        # tracked by the caller through ``rows``, not by a modular counter.
        row = self.rows.pop(0)
        return mx.array([DRAFTS[row][: bs - 1]], dtype=token_dtype)


class _RoundsDone(Exception):
    """Stops the loop the instant the Nth rollback has been applied -- the only
    window in which a round's cache state is observable."""


def _drive(model, accepts_per_round, *, rounds_to_run=1, stop_check=None):
    """Run ``rounds_to_run`` real ``_dflash_rounds_batch`` rounds on batched
    caches, dictating only the per-row acceptance.

    Returns (drafter, emitted_per_row, cache, accepts_seen_by_the_rollback).
    """
    B = len(accepts_per_round[0])
    target = SimpleNamespace(language_model=model)
    cache = _make_cache(model, [0] * B)
    model(mx.array([PROMPT] * B, dtype=mx.int32), cache=cache)
    hidden = mx.zeros((B, 1, model.args.hidden_size), dtype=mx.float32)

    drafter = _StubDrafter()
    rounds = list(accepts_per_round)
    # Active-slot -> original-row for the CURRENT round; the walk stub and the
    # drafter stub both need it, and it changes when a row is filtered out.
    active = [list(range(B))]

    def ragged_walk(draft_tokens, target_tokens, budgets):
        accepted = rounds.pop(0)
        rows = draft_tokens.tolist()
        # SENTINEL is in vocabulary on purpose -- see its definition.
        return list(accepted), [
            (rows[i][:a] + [SENTINEL + active[0][i]])[: budgets[i]]
            for i, a in enumerate(accepted)
        ]

    original_walk = dflash_utils._speculative_walk_batch
    dflash_utils._speculative_walk_batch = ragged_walk

    seen = []
    original_rollback = model.rollback_speculative_cache

    def spy(caches, gdn_states, accepted_arg, block_size):
        seen.append(
            ([int(v) for v in accepted_arg.reshape(-1).tolist()], int(block_size))
        )
        original_rollback(caches, gdn_states, accepted_arg, block_size)
        if len(seen) >= rounds_to_run:
            raise _RoundsDone
        # The next round drafts for whoever is left; the loop filters finished
        # rows right after this call, so recompute the active map from the
        # emitted state the caller can see.
        active[0] = [i for i in active[0] if not finished[i]]
        drafter.rows.extend(active[0])

    model.rollback_speculative_cache = spy
    finished = [False] * B

    def _stop(row, token):
        stop = bool(stop_check(row, token)) if stop_check is not None else False
        finished[row] = finished[row] or stop
        return stop

    drafter.rows.extend(active[0])
    emitted = [[] for _ in range(B)]
    try:
        gen = dflash_utils._dflash_rounds_batch(
            target,
            drafter,
            cache,
            hidden,
            first_bonus=mx.array(BONUS[:B], dtype=mx.int32),
            max_tokens=64,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            greedy_sampling=True,
            stop_check=_stop,
        )
        try:
            for tokens_out, _ in gen:
                for i, token in enumerate(tokens_out):
                    if token is not None:
                        emitted[i].append(int(token))
        except _RoundsDone:
            pass
        else:  # pragma: no cover - the round always reaches the rollback
            pytest.fail("the round never reached rollback_speculative_cache")
    finally:
        model.rollback_speculative_cache = original_rollback
        dflash_utils._speculative_walk_batch = original_walk
    assert len(seen) == rounds_to_run, f"expected {rounds_to_run} rollbacks, got {seen}"
    return drafter, emitted, cache, seen


def _reference_cache(model, row, emitted_row):
    """A fresh single-row forward over exactly what this row has committed.

    Each round emits [accepted drafts..., the target's own token], and that last
    token is the bonus the NEXT round feeds back in -- so across any number of
    rounds the committed tokens are ``[first bonus] + emitted[:-1]``. Unchanged
    from the clamp tests: it holds per row whether or not the rows agree.
    """
    cache = model.make_cache()
    model(mx.array([PROMPT], dtype=mx.int32), cache=cache)
    committed = [BONUS[row]] + emitted_row[:-1]
    model(mx.array([committed], dtype=mx.int32), cache=cache)
    return cache, committed


def _row_signature(cache_entry):
    from mlx_vlm.models.cache import ArraysCache

    out = []
    if isinstance(cache_entry, ArraysCache):
        for i, arr in enumerate(cache_entry.cache):
            out.append((f"kda[{i}]", None, arr))
        return out
    for i, sub in enumerate(cache_entry.caches):
        keys = None if sub.keys is None else sub.keys[:, :, : sub.offset]
        values = None if sub.values is None else sub.values[:, :, : sub.offset]
        out.append((f"kv[{i}].keys", int(sub.offset), keys))
        out.append((f"kv[{i}].values", int(sub.offset), values))
    return out


def _worst_diff(got, ref):
    if got is None or ref is None or (got.size == 0 and ref.size == 0):
        return 0.0
    assert got.shape == ref.shape, f"shape {got.shape} vs {ref.shape}"
    return float(mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))))


def _assert_row_matches_its_own_forward(model, cache, row, emitted_row, *, slot=None):
    """The contract: this row's cache == a fresh forward over its own tokens."""
    slot = row if slot is None else slot
    ref_cache, committed = _reference_cache(model, row, emitted_row)
    for layer, (got_entry, ref_entry) in enumerate(zip(cache, ref_cache)):
        got_sig = _row_signature(got_entry.extract(slot))
        ref_sig = _row_signature(ref_entry)
        for (name, got_off, got_arr), (_, ref_off, ref_arr) in zip(got_sig, ref_sig):
            assert got_off == ref_off, (
                f"row {row} layer {layer} {name}: cache offset {got_off} but a "
                f"forward over {committed} lands at {ref_off} -- the row is "
                "holding the KV of tokens it rejected"
            )
            worst = _worst_diff(got_arr, ref_arr)
            assert worst <= _ROW_TOL, (
                f"row {row} layer {layer} {name}: max abs diff {worst:.3e} "
                f"against a fresh forward over {committed} -- the row was "
                "rolled back to another row's accept count"
            )
    return committed


def test_ragged_round_is_not_clamped_and_every_row_keeps_its_own_length(model):

    drafter, emitted, cache, seen = _drive(model, [RAGGED])

    # The rollback saw the RAGGED accepts -- no clamp on the way in ...
    assert [a for a, _ in seen] == [RAGGED], seen
    # ... and every row emitted its own (accepted + 1) tokens, not the batch
    # minimum's. This is the +68%-at-B=8 lever, in miniature.
    assert [len(e) for e in emitted] == [a + 1 for a in RAGGED], emitted
    assert drafter.accept_lens == RAGGED

    for row in range(2):
        _assert_row_matches_its_own_forward(model, cache, row, emitted[row])


def test_per_row_rollback_composes_over_two_ragged_rounds(model):
    """One ragged round can be right by luck; two cannot.

    Round 2 drafts from round 1's bonus and verifies into round 1's rolled-back
    cache, so an off-by-one between what a row emitted and the state it was left
    in only surfaces on the second pass -- and the per-row left padding round 1
    introduced has to still be honoured (by the mask, by the indexer's pooling
    anchor, and by the next append) when round 2 adds more of it.
    """

    drafter, emitted, cache, seen = _drive(model, [RAGGED, RAGGED], rounds_to_run=2)

    assert [a for a, _ in seen] == [RAGGED, RAGGED], seen
    assert [len(e) for e in emitted] == [2 * (a + 1) for a in RAGGED], emitted
    for row in range(2):
        _assert_row_matches_its_own_forward(model, cache, row, emitted[row])

    # Row 1 is now four positions short of row 0 and says so, per row.
    latent = cache[1][0]
    assert [int(v) for v in latent.left_padding.tolist()] == [0, 4]
    assert [int(v) for v in latent.offset.tolist()] == [12, 8]


def test_a_row_finishing_midway_leaves_the_survivors_intact(model):
    """B=3, ragged, one row stopped in round 1 -> the continuous-batching filter.

    The filter runs immediately after the rollback (dflash.py, right below the
    ``rollback_speculative_cache`` call) and re-indexes the batch axis of every
    cache. The rows that survive it must still satisfy the invariant, with their
    own per-row padding carried through the filter's own left-shift.
    """

    drafter, emitted, cache, seen = _drive(
        model,
        [[3, 1, 2], [2, 1]],
        rounds_to_run=2,
        stop_check=lambda row, token: row == 2,
    )

    assert [a for a, _ in seen] == [[3, 1, 2], [2, 1]], seen
    # Round 2 ran on the two survivors only.
    assert seen[1][0] == [2, 1]
    # The finished row is gone from the batch axis of every cache.
    for entry in cache:
        for sub in getattr(entry, "caches", [entry]):
            assert sub.batch_size == 2, type(sub).__name__

    # Slot 0 is original row 0, slot 1 is original row 1.
    for slot, row in enumerate((0, 1)):
        _assert_row_matches_its_own_forward(
            model, cache, row, emitted[row], slot=slot
        )


def test_the_receipt_reads_clamped_zero_and_counts_what_was_kept(model):
    """``clamped N tok`` is the number a per-row build is read for: it must be 0,
    and the tokens the clamp would have taken must be attributable."""

    drafter, _, _, _ = _drive(model, [RAGGED, RAGGED], rounds_to_run=2)

    kept = 2 * sum(a - min(RAGGED) for a in RAGGED)
    assert getattr(drafter, "clamped_tokens", 0) == 0
    assert drafter.per_row_kept_tokens == kept
    assert drafter.speculative_total_per_row_kept == kept
    assert ", clamped 0 tok, per-row kept 4 tok" in _format_speculative_stats(drafter)


def test_the_per_row_counter_is_per_request_and_lifetime():
    drafter = SimpleNamespace()
    _record_per_row_rollback(drafter, 0)
    assert not hasattr(drafter, "speculative_total_per_row_kept")
    _record_per_row_rollback(drafter, 3)
    _record_per_row_rollback(drafter, 2)
    assert drafter.per_row_kept_tokens == 5
    assert drafter.speculative_total_per_row_kept == 5
    _reset_per_row_rollback(drafter)
    assert drafter.per_row_kept_tokens == 0
    assert drafter.speculative_total_per_row_kept == 5


def test_uniform_accepts_take_the_rectangular_path_untouched(model):
    """The common case must not pay for the ragged one: with equal accepts no
    row is shifted, so no left padding appears and no pool is dropped."""

    drafter, emitted, cache, seen = _drive(model, [[2, 2]])

    assert [a for a, _ in seen] == [[2, 2]], seen
    latent = cache[1][0]
    assert [int(v) for v in latent.left_padding.tolist()] == [0, 0]
    assert getattr(drafter, "per_row_kept_tokens", 0) == 0
    for row in range(2):
        _assert_row_matches_its_own_forward(model, cache, row, emitted[row])


def test_the_requirement_is_read_off_the_caches_not_the_class(model):
    """glm5_next still DECLARES the static requirement -- flipping it to False
    would turn an unsupported cache into a mid-generation crash -- and answers
    the cache-aware question per cache."""
    from mlx_vlm.models.glm5_next.language import LanguageModel

    assert LanguageModel.requires_uniform_batch_acceptance is True
    drafter = SimpleNamespace()

    batched = _make_cache(model, [0, 0])
    assert model.supports_per_row_speculative_rollback(batched)
    assert _supports_per_row_rollback(model, batched)
    assert not _batch_acceptance_must_be_uniform(drafter, model, batched)

    plain = model.make_cache()
    assert not model.supports_per_row_speculative_rollback(plain)
    assert _batch_acceptance_must_be_uniform(drafter, model, plain)

    # A quantized batch cache holds packed tuples that cannot be shifted; the
    # clamp has to stay for it.
    quantized = _make_cache(model, [0, 0], kv_bits=4)
    assert not model.supports_per_row_speculative_rollback(quantized)
    assert _batch_acceptance_must_be_uniform(drafter, model, quantized)

    # And a model that says nothing at all keeps its ragged accepts, as before.
    assert not _supports_per_row_rollback(SimpleNamespace(), batched)


def test_rollback_refuses_ragged_accepts_it_cannot_represent(model):
    """The loud guard survives: a cache with no per-row length must crash, not
    silently trim by the batch maximum."""
    cache = model.make_cache()
    model(mx.array([PROMPT, PROMPT], dtype=mx.int32), cache=cache)

    with pytest.raises(RuntimeError, match="per-row rollback needs a batched"):
        model.rollback_speculative_cache(
            cache, [], mx.array(RAGGED, dtype=mx.int32), BLOCK_TOTAL
        )


def test_the_clamp_still_runs_when_the_target_cannot_do_per_row(model):
    """End to end on unsupported caches: the loop clamps, so the guard above is
    unreachable rather than merely loud."""

    target = SimpleNamespace(language_model=model)
    cache = model.make_cache()  # scalar-offset: no per-row length
    model(mx.array([PROMPT, PROMPT], dtype=mx.int32), cache=cache)
    hidden = mx.zeros((2, 1, model.args.hidden_size), dtype=mx.float32)

    def ragged_walk(draft_tokens, target_tokens, budgets):
        rows = draft_tokens.tolist()
        return list(RAGGED), [
            (rows[i][:a] + [90 + i])[: budgets[i]] for i, a in enumerate(RAGGED)
        ]

    original_walk = dflash_utils._speculative_walk_batch
    dflash_utils._speculative_walk_batch = ragged_walk
    seen = []
    original_rollback = model.rollback_speculative_cache

    def spy(caches, gdn_states, accepted_arg, block_size):
        seen.append([int(v) for v in accepted_arg.reshape(-1).tolist()])
        original_rollback(caches, gdn_states, accepted_arg, block_size)
        raise _RoundsDone

    model.rollback_speculative_cache = spy
    drafter = _StubDrafter()
    drafter.rows.extend([0, 1])
    try:
        gen = dflash_utils._dflash_rounds_batch(
            target,
            drafter,
            cache,
            hidden,
            first_bonus=mx.array(BONUS[:2], dtype=mx.int32),
            max_tokens=64,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            greedy_sampling=True,
        )
        try:
            for _ in gen:
                pass
        except _RoundsDone:
            pass
    finally:
        model.rollback_speculative_cache = original_rollback
        dflash_utils._speculative_walk_batch = original_walk

    assert seen == [[min(RAGGED)] * 2], seen
    assert drafter.clamped_tokens == sum(a - min(RAGGED) for a in RAGGED)
    assert getattr(drafter, "per_row_kept_tokens", 0) == 0


def test_the_next_step_reads_the_padded_rows_the_same_as_unpadded_ones(model):
    """The cache check cannot see a wrong SELECTION -- this does.

    In a 2-layer fixture the sparse layer is last, so what its indexer selects
    never reaches another layer's cache: a row whose left padding is not honoured
    by the indexer still passes the cache invariant. So take one more decode step
    on the rolled-back batch and compare each row's logits against a single-row
    forward over that row's own tokens. This is what pins the property the whole
    design rests on -- the indexer anchors its pools at the row's first VALID key
    (``_pool_layout``), so selection is equivariant under left padding, provided
    the vacated window is actually marked invalid.
    """

    drafter, emitted, cache, seen = _drive(model, [RAGGED, RAGGED], rounds_to_run=2)
    assert [a for a, _ in seen] == [RAGGED, RAGGED], seen

    # One more step, batched, each row on its own next token.
    nxt = [emitted[row][-1] for row in range(2)]
    batched = model(mx.array([[t] for t in nxt], dtype=mx.int32), cache=cache).logits

    for row in range(2):
        ref_cache, committed = _reference_cache(model, row, emitted[row])
        single = model(
            mx.array([[nxt[row]]], dtype=mx.int32), cache=ref_cache
        ).logits
        worst = _worst_diff(batched[row : row + 1], single)
        assert worst <= 1e-3, (
            f"row {row}: next-step logits differ by {worst:.3e} from a forward "
            f"over {committed} + [{nxt[row]}] -- the row's left padding is being "
            "attended or selected"
        )
