"""V1b: a speculative batch that GROWS.

Rows leaving a batched DFlash2 round loop were always supported; rows arriving
were not.  ``SpeculativeGenerationBatch.extend`` raised on a non-empty batch and
``BatchGenerator._next`` returned before the prefill path while one was alive, so
a served batch was whatever the first queue drain happened to see and never grew
again -- V1 measured rows/round 3.29 (3.91 with the 40 ms coalescing window) out
of 8-16 waiting, against greedy's 14.

These tests are CPU-only and drive the REAL ``_dflash_rounds_batch`` over the
real (tiny) Glm5Next target the ragged/per-row rollback tests use, plus the real
``SpeculativeGenerationBatch`` over a fake round loop.  The invariant they pin is
the one the rollback tests already pin, applied to a row that was not there when
the loop started:

    after the round, each row's cache is what a fresh forward over exactly that
    row's committed tokens produces.

ON IDENTITY, AND WHAT IS NOT CLAIMED HERE.  A row admitted mid-stream does NOT
produce the same token sequence it would have produced in a batch of its own, and
neither do the incumbents.  That is not a defect of admission: the tree already
measured that a B > 1 reduction and a B == 1 reference disagree in float32
(``test_dflash_perrow_rollback.py`` compares them at 1e-4, "not bit for bit, and
the same reason as the clamp tests"), and that verifying S tokens in one forward
is not bit-identical to S sequential forwards, so a near-tie argmax flips and
everything after it diverges (``speculative/dflash.py::_adaptive_k_enabled``, the
ON IDENTITY paragraph, receipts logs/sweep3/CARD_spec_row_r18.json).  Batch
composition is an input to those reductions.  What IS pinned:

  * the STATE identity above -- exact offsets, per-row caches within 1e-4 of the
    row's own forward -- for the admitted row and for every incumbent;
  * exact token identity for the incumbents when the walk is stubbed, i.e. that
    admission changes no BOOKKEEPING (uid mapping, active-slot map, bonus
    tokens, emit counts), only arithmetic;
  * that the flag off is byte-identical to the tree before this change.
"""

from types import SimpleNamespace
from typing import List, Optional

import mlx.core as mx
import pytest

from mlx_vlm.generate import ar as ar_mod
from mlx_vlm.generate.ar import SpeculativeGenerationBatch, _extend_cache, _make_cache
from mlx_vlm.speculative import dflash as dflash_utils
from mlx_vlm.speculative.drafters.qwen3_dflash.batched_cache import BatchDFlashKVCache
from mlx_vlm.tests.test_dflash_ragged_rollback import (
    BLOCK_TOTAL,
    PROMPT,
    _tiny_glm5_next_target,
)

# The admitted row's prompt: a DIFFERENT sequence, so a row that ends up reading
# an incumbent's cache is visible in the numbers and not merely in a counter.
LATE_PROMPT = [3, 5, 7, 9]
LATE_PROMPT_SHORT = [3, 5, 7]

DRAFTS = {0: [11, 12, 13, 14], 1: [21, 22, 23, 24], 2: [31, 32, 33, 34]}
BONUS = [5, 7]
LATE_BONUS = 9
SENTINEL = 28
_ROW_TOL = 1e-4
_ROW_OF_BONUS = {BONUS[0]: 0, BONUS[1]: 1, LATE_BONUS: 2}


@pytest.fixture(scope="module")
def model():
    mx.random.seed(3)
    m = _tiny_glm5_next_target()
    mx.eval(m.parameters())
    return m


class _StubDrafter:
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
        self.rows: List[int] = []
        self.caches_made = 0
        self.boundary = None
        self.pending_boundary = False

    def reset(self, model=None):
        self.accept_lens = []
        self.draft_lens = []

    def make_cache(self):
        self.caches_made += 1
        return []

    def draft_block(self, bonus, hidden, cache, bs, sampler, token_dtype):
        # V1d: the round loop now rolls the cache back BEFORE it emits, so the
        # window in which a round's cache state is observable opens once that
        # round's tokens have been yielded and closes when the next round
        # drafts.  The rollback spy arms ``pending_boundary``; the first draft
        # call of the next round runs the hook there.
        if self.pending_boundary and self.boundary is not None:
            self.pending_boundary = False
            self.boundary()
        # The row is read off the BONUS rather than off a queue: the bonus is
        # distinct per row in every round (the first bonuses, then SENTINEL+row),
        # and an admission changes the drafting ORDER, which a queue would have
        # to be told about separately.
        row = _ROW_OF_BONUS.get(int(bonus))
        if row is None:
            row = int(bonus) - SENTINEL
        return mx.array([DRAFTS[row][: bs - 1]], dtype=token_dtype)


class _RoundsDone(Exception):
    """Stops the loop the instant the Nth rollback has been applied."""


def _prefill(model, prompt, rows=1):
    cache = _make_cache(model, [0] * rows)
    model(mx.array([prompt] * rows, dtype=mx.int32), cache=cache)
    return cache


def _drive(
    model,
    accepts_per_round,
    *,
    admit_before_round: Optional[int] = None,
    late_prompt=LATE_PROMPT,
):
    """Run real ``_dflash_rounds_batch`` rounds over B=2, optionally admitting a
    third row at the boundary before round ``admit_before_round`` (1-based).

    Returns (emitted_per_row, cache, accepts_seen, admitted_rows).
    """
    B = len(accepts_per_round[0])
    target = SimpleNamespace(language_model=model)
    cache = _prefill(model, PROMPT, rows=B)
    hidden = mx.zeros((B, 1, model.args.hidden_size), dtype=mx.float32)

    drafter = _StubDrafter()
    rounds = list(accepts_per_round)
    active = [list(range(B))]
    total_rows = [B]
    emitted: List[List[int]] = [[] for _ in range(B + 1)]
    finished = [False] * (B + 1)
    admitted: List[dict] = []
    polls = [0]

    def ragged_walk(draft_tokens, target_tokens, budgets):
        accepted = rounds.pop(0)
        rows = draft_tokens.tolist()
        return list(accepted), [
            (rows[i][:a] + [SENTINEL + active[0][i]])[: budgets[i]]
            for i, a in enumerate(accepted)
        ]

    def admission():
        polls[0] += 1
        if admit_before_round is None or polls[0] != admit_before_round:
            return None
        late_cache = _prefill(model, late_prompt, rows=1)
        # Exactly what SpeculativeGenerationBatch._admission_poll does: the
        # caller owns the batch-dimension merge of the TARGET caches, and the
        # list object the generator closed over must stay the same object.
        cache[:] = _extend_cache(cache, late_cache)
        row = total_rows[0]
        total_rows[0] += 1
        active[0] = active[0] + [row]
        record = {
            "rows_hidden": [mx.zeros((1, 1, model.args.hidden_size), dtype=mx.float32)],
            "bonus": [LATE_BONUS],
            "row_ids": [0],
            "target_hidden_offset": [0],
            "max_tokens": 64,
        }
        admitted.append(record)
        return record

    seen = []
    original_rollback = model.rollback_speculative_cache
    original_walk = dflash_utils._speculative_walk_batch
    dflash_utils._speculative_walk_batch = ragged_walk

    def spy(caches, gdn_states, accepted_arg, block_size):
        seen.append([int(v) for v in accepted_arg.reshape(-1).tolist()])
        original_rollback(caches, gdn_states, accepted_arg, block_size)
        drafter.pending_boundary = True

    def _round_boundary():
        if len(seen) >= len(accepts_per_round):
            raise _RoundsDone
        active[0] = [i for i in active[0] if not finished[i]]

    model.rollback_speculative_cache = spy
    drafter.boundary = _round_boundary

    def _stop(row, token):
        return finished[row]

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
            admission=admission,
        )
        try:
            for tokens_out, _ in gen:
                for i, token in enumerate(tokens_out):
                    if token is not None:
                        emitted[i].append(int(token))
        except _RoundsDone:
            pass
        else:  # pragma: no cover
            pytest.fail("the round never reached rollback_speculative_cache")
    finally:
        model.rollback_speculative_cache = original_rollback
        dflash_utils._speculative_walk_batch = original_walk

    return emitted, cache, seen, admitted


def _reference_cache(model, prompt, first_bonus, emitted_row):
    cache = model.make_cache()
    model(mx.array([prompt], dtype=mx.int32), cache=cache)
    committed = [first_bonus] + emitted_row[:-1]
    if committed:
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


def _assert_row_matches_its_own_forward(
    model, cache, slot, prompt, first_bonus, emitted_row
):
    ref_cache, committed = _reference_cache(model, prompt, first_bonus, emitted_row)
    for layer, (got_entry, ref_entry) in enumerate(zip(cache, ref_cache)):
        got_sig = _row_signature(got_entry.extract(slot))
        ref_sig = _row_signature(ref_entry)
        for (name, got_off, got_arr), (_, ref_off, ref_arr) in zip(got_sig, ref_sig):
            assert got_off == ref_off, (
                f"slot {slot} layer {layer} {name}: cache offset {got_off} but a "
                f"forward over {committed} lands at {ref_off}"
            )
            worst = _worst_diff(got_arr, ref_arr)
            assert worst <= _ROW_TOL, (
                f"slot {slot} layer {layer} {name}: max abs diff {worst:.3e} "
                f"against a fresh forward over {committed}"
            )
    return committed


# ==========================================================================
# The round loop
# ==========================================================================
def test_a_row_admitted_at_a_round_boundary_decodes_from_the_next_round(model):
    emitted, cache, seen, admitted = _drive(
        model, [[3, 1], [2, 2, 2]], admit_before_round=2
    )

    assert len(admitted) == 1, "the loop polls the admission channel once a round"
    # Round 1 saw two rows; round 2 saw three -- the batch GREW mid-stream,
    # which is the entire point.
    assert [len(a) for a in seen] == [2, 3], seen
    # The admitted row emitted only from the round it joined.
    assert len(emitted[0]) == 3 + 1 + 2 + 1
    assert len(emitted[1]) == 1 + 1 + 2 + 1
    assert len(emitted[2]) == 2 + 1
    assert emitted[2] == DRAFTS[2][:2] + [SENTINEL + 2], emitted[2]


def test_the_admitted_row_lands_where_its_own_forward_puts_it(model):
    emitted, cache, seen, _ = _drive(model, [[3, 1], [2, 2, 2]], admit_before_round=2)
    # Slot 2 is the admitted row: appended at the end of the active order, which
    # is the end of the target caches' batch dimension.
    _assert_row_matches_its_own_forward(
        model, cache, 2, LATE_PROMPT, LATE_BONUS, emitted[2]
    )


def test_the_incumbents_are_undisturbed_by_the_admission(model):
    emitted, cache, seen, _ = _drive(model, [[3, 1], [2, 2, 2]], admit_before_round=2)
    for slot, row in enumerate((0, 1)):
        _assert_row_matches_its_own_forward(
            model, cache, slot, PROMPT, BONUS[row], emitted[row]
        )


def test_a_shorter_late_prompt_is_left_padded_into_the_batch(model):
    """The admitted row's prompt need not be the incumbents' length.

    ``_extend_cache`` left-pads the shorter side and carries per-row offsets;
    this is the same path ``GenerationBatch.extend`` has always used, exercised
    here with a LIVE speculative batch on the other side of it.
    """
    emitted, cache, seen, _ = _drive(
        model,
        [[3, 1], [2, 2, 2]],
        admit_before_round=2,
        late_prompt=LATE_PROMPT_SHORT,
    )
    assert [len(a) for a in seen] == [2, 3], seen
    _assert_row_matches_its_own_forward(
        model, cache, 2, LATE_PROMPT_SHORT, LATE_BONUS, emitted[2]
    )


def test_the_incumbents_emit_exactly_what_they_would_have_without_the_row(model):
    """Token identity for the incumbents, with the walk stubbed.

    The stub decides acceptance, so the target forward's numerics are out of the
    comparison and what is left is the BOOKKEEPING: the active-slot map, the
    bonus feedback and the emit loop.  Those must not move when a row joins.
    (With the real walk they do move -- batch composition is an input to the
    verify reduction; see this module's docstring.)
    """
    with_row, _, _, _ = _drive(model, [[3, 1], [2, 2, 2]], admit_before_round=2)
    without, _, _, _ = _drive(model, [[3, 1], [2, 2]])
    assert with_row[0] == without[0]
    assert with_row[1] == without[1]


def test_no_admission_channel_is_the_loop_that_shipped(model):
    """``admission=None`` -- not one poll, not one branch taken."""
    emitted, cache, seen, admitted = _drive(model, [[3, 1], [2, 2]])
    assert admitted == []
    assert [len(a) for a in seen] == [2, 2], seen


# ==========================================================================
# BatchDFlashKVCache.grow
# ==========================================================================
def test_grow_keeps_the_layout_invariant():
    c = BatchDFlashKVCache(2)
    c.set_pending_lengths([3, 2])
    k = mx.ones((2, 1, 3, 4))
    c.update_and_fetch(k, k)
    assert c.left_padding == [0, 1] and c.context_lengths == [3, 2]

    c.grow(1, 7)
    assert c.batch_size == 3
    # column c of row j is real  <=>  c >= left_padding[j]; a new row has none.
    assert c.left_padding == [0, 1, c.size()]
    assert c.context_lengths == [3, 2, 0]
    assert min(c.left_padding) == 0
    assert c.keys.shape[0] == 3
    # The offset is the row's own pre-truncation credit, not the batch's.
    assert c.offset.tolist() == [3, 2, 7]


def test_a_grown_row_takes_its_own_context_on_the_next_round():
    c = BatchDFlashKVCache(2)
    c.set_pending_lengths([3, 2])
    c.update_and_fetch(mx.ones((2, 1, 3, 4)), mx.ones((2, 1, 3, 4)))
    c.grow(1, 0)

    block = mx.arange(3 * 1 * 2 * 4, dtype=mx.float32).reshape(3, 1, 2, 4)
    c.set_pending_lengths([2, 2, 2])
    keys, _ = c.update_and_fetch(block, block)
    assert c.context_lengths == [5, 4, 2]
    # The grown row's real columns are exactly the block rows it was handed.
    real = keys[2, :, c.left_padding[2] :, :]
    assert float(mx.max(mx.abs(real - block[2]))) == 0.0


def test_grow_is_a_no_op_for_a_non_positive_count():
    c = BatchDFlashKVCache(2)
    c.grow(0)
    c.grow(-1)
    assert c.batch_size == 2


# ==========================================================================
# SpeculativeGenerationBatch: the bookkeeping around the round loop
# ==========================================================================
HIDDEN = 8


def _spec_batch(uids=(1, 2), *, first=5, draft_kind="dflash", processors=None,
                max_tokens=64, cache=None):
    return SpeculativeGenerationBatch(
        model=SimpleNamespace(),
        draft_model=SimpleNamespace(),
        draft_kind=draft_kind,
        uids=list(uids),
        first_tokens=mx.array([first + i for i in range(len(uids))], dtype=mx.int32),
        prompt_cache=[] if cache is None else cache,
        sampler=lambda logits: logits,
        stop_criteria=lambda token: False,
        max_tokens=[max_tokens] * len(uids),
        hidden=mx.zeros((len(uids), 1, HIDDEN), dtype=mx.float32),
        shared_kv_states=None,
        prompt_tokens=None,
        logits_processors=processors,
    )


def _batch_cache(rows, length):
    from mlx_vlm.models.cache import BatchKVCache

    c = BatchKVCache([0] * rows)
    c.update_and_fetch(
        mx.zeros((rows, 1, length, 4)), mx.zeros((rows, 1, length, 4))
    )
    return [c]


def _drain(batch, steps):
    out = []
    for _ in range(steps):
        out.append(batch.next())
    return out


def test_extend_still_refuses_on_an_active_batch_when_the_flag_is_off(monkeypatch):
    monkeypatch.delenv("MLX_VLM_SPEC_EXTEND_ACTIVE", raising=False)
    batch = _spec_batch()
    assert batch.accepts_extension() is False
    with pytest.raises(RuntimeError, match="Cannot extend an active speculative"):
        batch.extend(_spec_batch(uids=(3,)))


def test_an_empty_batch_is_still_adopted_wholesale(monkeypatch):
    monkeypatch.delenv("MLX_VLM_SPEC_EXTEND_ACTIVE", raising=False)
    batch = _spec_batch(uids=())
    other = _spec_batch(uids=(3, 4))
    batch.extend(other)
    assert batch.uids == [3, 4] and batch._loop_rows == [0, 1]


def test_a_non_dflash_or_structured_batch_refuses_to_grow(monkeypatch):
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_ACTIVE", "1")
    assert _spec_batch(draft_kind="mtp").accepts_extension() is False
    # A grammar ledger is built ONCE per batch, sized to it.
    processor = SimpleNamespace(kind="grammar")
    monkeypatch.setattr(
        ar_mod, "resolve_structured_processor", lambda *a, **k: None
    )
    assert _spec_batch(processors=[[processor], None]).accepts_extension() is False


def test_the_admission_channel_is_wired_only_when_the_batch_may_grow(monkeypatch):
    seen = {}

    def fake_rounds(*args, **kwargs):
        seen.update(kwargs)
        yield [None, None], None

    monkeypatch.setattr(ar_mod, "run_speculative_server_rounds", fake_rounds)

    monkeypatch.delenv("MLX_VLM_SPEC_EXTEND_ACTIVE", raising=False)
    batch = _spec_batch()
    batch._start_rounds()
    next(batch._rounds_iter)
    assert seen["admission"] is None

    seen.clear()
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_ACTIVE", "1")
    batch = _spec_batch()
    batch._start_rounds()
    next(batch._rounds_iter)
    assert callable(seen["admission"])


def test_the_target_caches_are_merged_at_the_poll_not_at_the_extend(monkeypatch):
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_ACTIVE", "1")
    cache = _batch_cache(2, 4)
    batch = _spec_batch(cache=cache)
    batch.extend(_spec_batch(uids=(3,), first=9, cache=_batch_cache(1, 4)))

    # The bookkeeping is immediate -- the server's capacity check must see the
    # row the instant it is admitted ...
    assert batch.uids == [1, 2, 3] and len(batch) == 3
    # ... and the caches are NOT, because the round loop may be suspended
    # mid-round with a verify done and its rollback still pending.
    assert cache[0].keys.shape[0] == 2

    record = batch._admission_poll()
    assert record is not None and record["bonus"] == [9]
    assert cache[0].keys.shape[0] == 3
    assert batch.prompt_cache is cache, "the generator closed over THIS list"
    assert batch._loop_rows == [0, 1, 2]
    # A second poll has nothing left to hand over.
    assert batch._admission_poll() is None


def test_an_admitted_rows_bonus_token_is_emitted_exactly_once(monkeypatch):
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_ACTIVE", "1")

    def fake_rounds(*args, **kwargs):
        admission = kwargs["admission"]
        while True:
            admission()
            yield [None, None, None], None

    monkeypatch.setattr(ar_mod, "run_speculative_server_rounds", fake_rounds)
    batch = _spec_batch(cache=_batch_cache(2, 4))
    first = batch.next()
    assert [r.token for r in first] == [5, 6]

    batch.extend(_spec_batch(uids=(3,), first=9, cache=_batch_cache(1, 4)))
    late = batch.next()
    assert [(r.uid, r.token) for r in late] == [(3, 9)], (
        "the round loop starts an admitted row at emitted=1, so THIS class owes "
        "the caller that bonus token"
    )
    assert batch._num_tokens == [1, 1, 1]
    # And never again.
    assert batch.next() == []


def test_the_loop_row_index_is_mapped_back_to_this_batchs_uid(monkeypatch):
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_ACTIVE", "1")

    def fake_rounds(*args, **kwargs):
        admission = kwargs["admission"]
        yield [70, 71], None
        admission()
        yield [80, 81, 82], None

    monkeypatch.setattr(ar_mod, "run_speculative_server_rounds", fake_rounds)
    batch = _spec_batch(cache=_batch_cache(2, 4))
    batch.next()  # first tokens
    assert [(r.uid, r.token) for r in batch.next()] == [(1, 70), (2, 71)]
    batch.extend(_spec_batch(uids=(3,), first=9, cache=_batch_cache(1, 4)))
    assert [(r.uid, r.token) for r in batch.next()] == [(3, 9)]
    assert [(r.uid, r.token) for r in batch.next()] == [(1, 80), (2, 81), (3, 82)]


def test_a_row_that_stops_on_its_first_token_never_enters_the_loop(monkeypatch):
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_ACTIVE", "1")
    polls = []

    def fake_rounds(*args, **kwargs):
        while True:
            polls.append(kwargs["admission"]())
            yield [None, None], None

    monkeypatch.setattr(ar_mod, "run_speculative_server_rounds", fake_rounds)
    cache = _batch_cache(2, 4)
    batch = _spec_batch(cache=cache)
    batch.stop_criteria = lambda token: token == 9
    batch.next()

    batch.extend(_spec_batch(uids=(3,), first=9, cache=_batch_cache(1, 4)))
    responses = batch.next()
    assert [(r.uid, r.finish_reason) for r in responses] == [(3, "stop")]
    batch.next()
    assert polls == [None], (
        "a row that stopped on its bonus token has nothing to decode, and its "
        "cache columns leave with it rather than riding along"
    )
    assert cache[0].keys.shape[0] == 2


def test_rows_queued_when_the_loop_ended_get_a_loop_of_their_own(monkeypatch):
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_ACTIVE", "1")
    starts = []

    def fake_rounds(*args, **kwargs):
        starts.append(list(kwargs["first_bonus"].reshape(-1).tolist()))
        return iter(())  # the loop ends immediately: every row finished

    monkeypatch.setattr(ar_mod, "run_speculative_server_rounds", fake_rounds)
    batch = _spec_batch(cache=_batch_cache(2, 4))
    batch.next()
    batch.extend(_spec_batch(uids=(3,), first=9, cache=_batch_cache(1, 4)))
    batch.next()  # emits the admitted row's bonus
    assert batch.next() == [], "the restart consumes a step and emits nothing"
    final = batch.next()  # the new loop is built (and, for this fake, ends) here
    assert starts == [[5, 6], [9]], starts
    assert batch._loop_rows == [2], (
        "the new generator's row 0 is this batch's row 2; every callback the "
        "loop is handed goes through _global_row"
    )
    # Every row answers.  Without the restart the queued row would have been
    # marked finished by the FIRST loop's StopIteration without ever decoding,
    # holding a cache nothing would read -- a request that never answers.
    assert sorted(r.uid for r in final) == [1, 2, 3]


# ==========================================================================
# BatchGenerator._next: the prefill/decode interleave policy
# ==========================================================================
class _FakeSpecBatch:
    is_speculative = True

    def __init__(self, *, accepts=True, rows=2):
        self._accepts = accepts
        self._rows = rows
        self.logits_processors = []
        self.prompt_cache = []
        self.calls = 0

    def __len__(self):
        return self._rows

    def accepts_extension(self):
        return self._accepts

    def next(self):
        self.calls += 1
        return []


def _fake_generator(gen_batch, *, capacity=16):
    gen = ar_mod.BatchGenerator.__new__(ar_mod.BatchGenerator)
    gen._generation_batch = gen_batch
    gen._prompt_batch = None
    gen._unprocessed_sequences = []
    gen.completion_batch_size = capacity
    gen.prefill_batch_size = 1
    gen._gen_tokens_counter = 0
    gen._steps_counter = 0
    gen._cache_eval_interval = 0
    gen._spec_prefill_phase = 0
    gen._prompt_tokens_counter = 0
    gen._prompt_time_counter = 0
    gen._wire_stack = None
    return gen


def _reached_prefill(gen):
    """``_next`` returns after the decode step unless it fell through.

    The tell is ``_unprocessed_sequences``: nothing else touches it, and the
    admission path is the only way a live speculative batch can reach it.
    """
    seen = []
    gen._build_mixed_prompt_batch = lambda seqs: seen.append(seqs) or None
    gen._unprocessed_sequences = [("uid", [1], 8, {}, None, None)]
    gen._pending_after_admission = lambda *a, **k: []
    try:
        gen._next()
    except Exception:  # pragma: no cover - the cold path needs a real model
        pass
    return bool(seen)


def test_a_live_speculative_batch_blocks_prefill_by_default(monkeypatch):
    monkeypatch.delenv("MLX_VLM_SPEC_EXTEND_ACTIVE", raising=False)
    gen = _fake_generator(_FakeSpecBatch(accepts=False))
    assert _reached_prefill(gen) is False
    assert gen._generation_batch.calls == 1, "the decode step still happens"


def test_with_admission_on_the_prefill_path_is_reachable(monkeypatch):
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_ACTIVE", "1")
    gen = _fake_generator(_FakeSpecBatch(accepts=True))
    assert _reached_prefill(gen) is True


def test_the_batch_may_not_grow_past_the_capacity_bound(monkeypatch):
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_ACTIVE", "1")
    # MLX_VLM_MAX_NUM_SEQS is ``completion_batch_size``; a full batch does not
    # prefill a row it could not admit.
    gen = _fake_generator(_FakeSpecBatch(accepts=True, rows=4), capacity=4)
    assert _reached_prefill(gen) is False


def test_the_interleave_policy_pays_one_chunk_every_n_decode_steps(monkeypatch):
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_ACTIVE", "1")
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_PREFILL_EVERY", "3")
    assert ar_mod.spec_extend_prefill_every() == 3
    gen = _fake_generator(_FakeSpecBatch(accepts=True))
    reached = [_reached_prefill(gen) for _ in range(6)]
    assert reached == [False, False, True, False, False, True], reached


def test_a_junk_interleave_setting_is_the_default_and_never_zero(monkeypatch):
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_PREFILL_EVERY", "not-a-number")
    assert ar_mod.spec_extend_prefill_every() == 1
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_PREFILL_EVERY", "0")
    assert ar_mod.spec_extend_prefill_every() == 1


def test_a_single_row_keeps_the_scalar_loop_unless_it_may_grow(monkeypatch):
    """``admission=None`` must not move a B == 1 request off ``_dflash_rounds``.

    The scalar loop has no active-slot map to append to, so a batch that may
    grow is routed to ``_dflash_rounds_batch`` even at B == 1.  That is a
    different (equivalent, not bit-identical) code path, which is exactly why it
    is reached only with the flag on.
    """
    from mlx_vlm.speculative import utils as spec_utils

    taken = []
    monkeypatch.setattr(
        spec_utils,
        "_dflash_rounds",
        lambda *a, **k: taken.append("scalar") or iter(()),
    )
    monkeypatch.setattr(
        spec_utils,
        "_dflash_rounds_batch",
        lambda *a, **k: taken.append("batch") or iter(()),
    )
    monkeypatch.setattr(spec_utils, "_validate_speculative_sampling", lambda *a: None)

    common = dict(
        draft_kind="dflash",
        first_bonus=mx.array([5], dtype=mx.int32),
        max_tokens=8,
        sampler=lambda x: x,
        greedy_sampling=True,
    )
    list(
        spec_utils.run_speculative_server_rounds(
            SimpleNamespace(), SimpleNamespace(), [], None, **common
        )
    )
    list(
        spec_utils.run_speculative_server_rounds(
            SimpleNamespace(),
            SimpleNamespace(),
            [],
            None,
            admission=lambda: None,
            **common,
        )
    )
    assert taken == ["scalar", "batch"], taken


def test_admission_is_refused_loudly_for_a_drafter_that_cannot_do_it():
    from mlx_vlm.speculative import utils as spec_utils

    gen = spec_utils.run_speculative_server_rounds(
        SimpleNamespace(),
        SimpleNamespace(),
        [],
        None,
        draft_kind="mtp",
        first_bonus=mx.array([5], dtype=mx.int32),
        max_tokens=8,
        sampler=lambda x: x,
        admission=lambda: None,
    )
    with pytest.raises(NotImplementedError, match="dflash only"):
        next(gen)
