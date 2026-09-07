"""L31 session capture on a B > 1 SPECULATIVE batch: which cache row is that uid?

``BatchGenerator.capture_session`` located a uid's KV row with
``_all_uids.index(uid)`` and handed that number straight to
``snapshot_prompt_cache_row``.  For the plain ``GenerationBatch`` that is right:
``filter()`` rewrites ``uids`` and ``prompt_cache`` together, so the two stay
aligned.  For ``SpeculativeGenerationBatch`` it is WRONG, and silently so:

  * ``_all_uids`` never shrinks -- that is the whole point of it, the finished
    row must stay locatable for the capture window;
  * ``prompt_cache`` is the ROUND LOOP's, and the loop filters the batch
    dimension down to its ACTIVE slots every time a row finishes
    (``speculative/dflash.py``, the "Continuous batching: filter out finished
    sequences" block; ``mtp.py`` and ``eagle3.py`` do the same).

So the moment any row of a speculative batch finishes, every later row's
``_all_uids`` index is too large by the number of rows that left, and the
capture stores either a ZERO-ROW cache (index past the end: mx slicing yields a
width-0 row and no exception) or -- worse -- A DIFFERENT, STILL-LIVE REQUEST'S
KV under this conversation's session key.

These tests drive the real ``_dflash_rounds_batch`` through the real
``SpeculativeGenerationBatch`` over the tiny Glm5Next target the rollback tests
use, with a stub drafter and a stubbed walk, and identify the captured row BY
ITS PROMPT: the three rows are prefilled with three DIFFERENT prompts, and the
first ``len(prompt)`` KV columns of a row are that prompt's own and are never
rolled back.  CPU only.
"""

from types import SimpleNamespace
from typing import List, Optional

import mlx.core as mx
import pytest

from mlx_vlm import apc as _apc
from mlx_vlm import context_vault as _cv
from mlx_vlm.generate import ar as ar_mod
from mlx_vlm.generate.ar import BatchGenerator, SpeculativeGenerationBatch, _make_cache
from mlx_vlm.speculative import dflash as dflash_utils
from mlx_vlm.tests.test_dflash_ragged_rollback import _tiny_glm5_next_target

BLOCK_TOTAL = 5
# One prompt per row, all the same length (so no left padding) and pairwise
# different, so a snapshot taken from the wrong row is visible in the KV and not
# merely in a counter.
PROMPTS = {0: [2, 4, 6, 8], 1: [3, 5, 7, 9], 2: [10, 12, 14, 16]}
DRAFTS = {0: [11, 12, 13, 14], 1: [21, 22, 23, 24], 2: [17, 18, 19, 20]}
BONUS = [5, 7, 9]
SENTINEL = 28
_ROW_OF_BONUS = {5: 0, 7: 1, 9: 2}
_ROW_OF_FIRST_DRAFT = {11: 0, 21: 1, 17: 2}
UIDS = [101, 102, 103]
_TOL = 1e-4


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
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def reset(self, model=None):
        self.accept_lens = []
        self.draft_lens = []

    def make_cache(self):
        return []

    def draft_block(self, bonus, hidden, cache, bs, sampler, token_dtype):
        # The row is read off the BONUS: distinct per row in every round (the
        # first bonuses, then SENTINEL + row).
        row = _ROW_OF_BONUS.get(int(bonus))
        if row is None:
            row = int(bonus) - SENTINEL
        return mx.array([DRAFTS[row][: bs - 1]], dtype=token_dtype)


def _prompt_kv_reference(model, row):
    """A single-row forward over just that row's PROMPT."""
    cache = model.make_cache()
    model(mx.array([PROMPTS[row]], dtype=mx.int32), cache=cache)
    return cache


def _kv_prefix(entry, n):
    """First ``n`` KV columns of a single-row cache entry, per layer."""
    out = []
    subs = getattr(entry, "caches", None)
    if subs is None:
        return out
    for i, sub in enumerate(subs):
        keys = getattr(sub, "keys", None)
        if keys is None:
            continue
        out.append((f"kv[{i}]", keys[:, :, :n, :]))
    return out


def _prompt_mismatch(snapshot, reference, n):
    """Worst |diff| between a snapshot's prompt columns and a row's own forward.

    ``inf`` when the snapshot has no row at all (the out-of-range case: a
    width-0 extraction, which is what an index past the end silently produces).
    """
    worst = 0.0
    compared = 0
    for got_entry, ref_entry in zip(snapshot, reference):
        for (name, got), (_, ref) in zip(_kv_prefix(got_entry, n), _kv_prefix(ref_entry, n)):
            if got.shape != ref.shape:
                return float("inf")
            worst = max(
                worst,
                float(mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)))),
            )
            compared += 1
    if compared == 0:
        return float("inf")
    return worst


def _drive(model, *, stops, rounds, capture_uid):
    """Run the real round loop until ``capture_uid`` finishes; capture there.

    ``stops`` is a set of ``(round_number, token)`` pairs.  Round-aware on
    purpose: every round ends every row on that row's SENTINEL, so a plain
    token set cannot say "stop on the last token of round 2" without also
    stopping on the last token of round 1.  And the last token of a round is
    where a row has to stop for its capture to be legal at all -- a row that
    stops MID-block leaves the cache holding the committed tokens after it,
    which is the V1d refusal (``cache_ahead_of_emitted``), not this file's
    subject.

    Returns (captured_row_cache, cache, rows_in_cache_at_capture, batch).
    """
    active_now = {"rows": [0, 1, 2]}
    remaining = list(rounds)
    round_no = {"n": 0}

    def walk(draft_tokens, target_tokens, budgets):
        accepted = remaining.pop(0)
        round_no["n"] += 1
        rows = draft_tokens.tolist()
        out = []
        for i, a in enumerate(accepted):
            orig = _ROW_OF_FIRST_DRAFT[int(rows[i][0])]
            out.append((rows[i][:a] + [SENTINEL + orig])[: budgets[i]])
        return list(accepted), out

    cache = _make_cache(model, [0, 0, 0])
    model(mx.array([PROMPTS[r] for r in range(3)], dtype=mx.int32), cache=cache)

    batch = SpeculativeGenerationBatch(
        model=SimpleNamespace(language_model=model),
        draft_model=_StubDrafter(),
        draft_kind="dflash",
        uids=list(UIDS),
        first_tokens=mx.array(BONUS, dtype=mx.int32),
        prompt_cache=cache,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        stop_criteria=lambda token: (round_no["n"], int(token)) in stops,
        max_tokens=[64] * 3,
        hidden=mx.zeros((3, 1, model.args.hidden_size), dtype=mx.float32),
        shared_kv_states=None,
        prompt_tokens=None,
        greedy_sampling=True,
    )

    # The key a real conversation carries: the row's own prompt plus every
    # token the row has emitted.  It has to be the real one -- the end-of-turn
    # rung's length is checked against it (V1d), so a placeholder key would be
    # refused before the row lookup this file is about ever ran.
    session_tokens = {uid: list(PROMPTS[i]) for i, uid in enumerate(UIDS)}
    generator = SimpleNamespace(
        vault=object(),
        _session_tokens=session_tokens,
        _generation_batch=batch,
    )
    captured = {}

    def spy(vault, key, row_cache, **kwargs):
        captured["row_cache"] = row_cache
        return True

    original_walk = dflash_utils._speculative_walk_batch
    original_record = ar_mod._context_vault.record_session_turn
    dflash_utils._speculative_walk_batch = walk
    ar_mod._context_vault.record_session_turn = spy
    try:
        for _ in range(12):
            if len(batch) == 0:
                break
            for response in batch.next():
                if response.token is not None:
                    session_tokens[response.uid].append(int(response.token))
                if response.finish_reason is None:
                    continue
                # Exactly where the server captures: inside the window between
                # finish_reason and remove() (server/generation.py::_step).
                BatchGenerator.capture_session(
                    generator, response.uid, session_id="s"
                )
                if response.uid == capture_uid:
                    return (
                        captured.get("row_cache"),
                        cache,
                        int(cache[1].caches[0].keys.shape[0]),
                        batch,
                    )
            active_now["rows"] = [
                i for i, done in enumerate(batch._finished) if not done
            ]
    finally:
        dflash_utils._speculative_walk_batch = original_walk
        ar_mod._context_vault.record_session_turn = original_record
    pytest.fail(f"uid {capture_uid} never finished")


# ==========================================================================
# The defect
# ==========================================================================
def test_a_later_rows_capture_is_not_taken_from_past_the_end(model):
    """Row 0 leaves in round 1; row 2 finishes in round 2.

    ``_all_uids.index(103)`` is still 2, but the cache is two rows wide by then
    and row 2 IS the second of them.  An index of 2 extracts a width-0 row:
    ``snapshot_prompt_cache_row`` returns a cache with an offset and no data,
    which is then stored under the conversation's session key.
    """
    row_cache, cache, rows_at_capture, batch = _drive(
        model,
        stops={(1, SENTINEL + 0), (2, SENTINEL + 2)},
        rounds=[[3, 3, 2], [3, 3]],
        capture_uid=103,
    )
    assert rows_at_capture == 2, (
        "the round loop filters the finished row out of the target caches at "
        "the round boundary; this test needs that to have happened"
    )
    assert row_cache, "the capture stored nothing at all"
    worst = _prompt_mismatch(row_cache, _prompt_kv_reference(model, 2), len(PROMPTS[2]))
    assert worst <= _TOL, (
        f"the captured cache's prompt columns differ from row 2's own forward "
        f"by {worst:.3e}: this session rung is not this conversation's KV"
    )


def test_a_later_rows_capture_is_not_a_live_neighbours_kv(model):
    """The same defect one row over, where the wrong index is IN range.

    Row 0 leaves in round 1; row 1 finishes in round 2 while row 2 is still
    decoding.  ``_all_uids.index(102)`` is 1, and cache row 1 is now row 2's --
    a DIFFERENT, still-live request's KV, stored under row 1's session key.
    """
    row_cache, cache, rows_at_capture, batch = _drive(
        model,
        stops={(1, SENTINEL + 0), (2, SENTINEL + 1)},
        rounds=[[3, 2, 2], [3, 3]],
        capture_uid=102,
    )
    assert rows_at_capture == 2
    assert row_cache, "the capture stored nothing at all"
    mine = _prompt_mismatch(row_cache, _prompt_kv_reference(model, 1), len(PROMPTS[1]))
    neighbour = _prompt_mismatch(
        row_cache, _prompt_kv_reference(model, 2), len(PROMPTS[2])
    )
    assert neighbour > _TOL, (
        "the captured cache is row 2's -- a live neighbour's KV under row 1's "
        "session key"
    )
    assert mine <= _TOL, (
        f"the captured cache's prompt columns differ from row 1's own forward "
        f"by {mine:.3e}"
    )


def test_the_uid_to_cache_row_map_follows_the_loops_filter(model):
    """The mapping itself, named: uid -> the row the CACHE holds it in."""
    _, cache, rows_at_capture, batch = _drive(
        model,
        stops={(1, SENTINEL + 0), (2, SENTINEL + 2)},
        rounds=[[3, 3, 2], [3, 3]],
        capture_uid=103,
    )
    assert batch._all_uids == UIDS, "the stable list never shrinks"
    assert batch.cache_row_for_uid(102) == 0
    assert batch.cache_row_for_uid(103) == 1
    assert batch.cache_row_for_uid(101) is None, (
        "row 0's columns left with it at the round boundary; there is no cache "
        "row to snapshot, and refusing is the only right answer"
    )


# ==========================================================================
# What must not move
# ==========================================================================
def test_the_greedy_batch_still_indexes_by_uids(monkeypatch):
    """``GenerationBatch`` compacts uids and prompt_cache together."""
    from mlx_vlm.models.cache import BatchKVCache

    cache = BatchKVCache([0, 0])
    keys = mx.arange(2 * 1 * 3 * 4, dtype=mx.float32).reshape(2, 1, 3, 4)
    cache.update_and_fetch(keys, keys)
    seen = {}

    def spy(vault, key, row_cache, **kwargs):
        seen["row_cache"] = row_cache
        return True

    monkeypatch.setattr(ar_mod._context_vault, "record_session_turn", spy)
    gb = SimpleNamespace(uids=["a", "b"], prompt_cache=[cache])
    generator = SimpleNamespace(
        vault=object(), _session_tokens={"b": [1]}, _generation_batch=gb
    )
    assert BatchGenerator.capture_session(generator, "b", session_id="s") is True
    got = seen["row_cache"][0]
    assert float(mx.max(mx.abs(got.keys[:, :, :3, :] - keys[1:2]))) == 0.0


def test_a_batch_that_never_reported_a_row_map_is_unchanged(monkeypatch):
    """Back-compat: no map (a stubbed loop, a kind that does not report) is the
    ``_all_uids`` index, exactly as before."""
    from mlx_vlm.models.cache import BatchKVCache

    cache = BatchKVCache([0, 0])
    keys = mx.arange(2 * 1 * 3 * 4, dtype=mx.float32).reshape(2, 1, 3, 4)
    cache.update_and_fetch(keys, keys)
    seen = {}

    def spy(vault, key, row_cache, **kwargs):
        seen["row_cache"] = row_cache
        return True

    monkeypatch.setattr(ar_mod._context_vault, "record_session_turn", spy)
    gb = SimpleNamespace(
        uids=["b"], _all_uids=["a", "b"], prompt_cache=[cache]
    )
    generator = SimpleNamespace(
        vault=object(), _session_tokens={"b": [1]}, _generation_batch=gb
    )
    assert BatchGenerator.capture_session(generator, "b", session_id="s") is True
    got = seen["row_cache"][0]
    assert float(mx.max(mx.abs(got.keys[:, :, :3, :] - keys[1:2]))) == 0.0


def test_an_out_of_range_row_is_refused_not_snapshotted_empty():
    """``snapshot_prompt_cache_row`` past the end must not return a width-0 row.

    Defence in depth for the same defect: a wrong index that lands past the end
    used to produce a cache with an OFFSET and no data, which stores and then
    restores as garbage.
    """
    from mlx_vlm.models.cache import BatchKVCache

    cache = BatchKVCache([0, 0])
    keys = mx.zeros((2, 1, 3, 4), dtype=mx.float32)
    cache.update_and_fetch(keys, keys)
    assert _apc.snapshot_prompt_cache_row([cache], 1) is not None
    assert _apc.snapshot_prompt_cache_row([cache], 2) is None
    assert _apc.snapshot_prompt_cache_row([cache], -1) is None


# ==========================================================================
# V1b: the same map, on a batch that GROWS
# ==========================================================================
def _batch_cache(rows, length):
    from mlx_vlm.models.cache import BatchKVCache

    cache = BatchKVCache([0] * rows)
    cache.update_and_fetch(
        mx.zeros((rows, 1, length, 4)), mx.zeros((rows, 1, length, 4))
    )
    return [cache]


def _spec_batch(uids, *, first=5, cache=None):
    return SpeculativeGenerationBatch(
        model=SimpleNamespace(),
        draft_model=SimpleNamespace(),
        draft_kind="dflash",
        uids=list(uids),
        first_tokens=mx.array(
            [first + i for i in range(len(uids))], dtype=mx.int32
        ),
        prompt_cache=[] if cache is None else cache,
        sampler=lambda logits: logits,
        stop_criteria=lambda token: False,
        max_tokens=[64] * len(uids),
        hidden=mx.zeros((len(uids), 1, 8), dtype=mx.float32),
        shared_kv_states=None,
        prompt_tokens=None,
    )


def test_the_map_is_the_loops_rows_translated_not_the_loops_rows(monkeypatch):
    """A row admitted mid-stream: the loop's slot is not this batch's row.

    ``_loop_rows`` already carries that translation for every other callback
    (tokens, budgets, stop checks); the cache map goes through the same one, or
    a capture after a restart snapshots a different request.
    """
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_ACTIVE", "1")

    def fake_rounds(*args, **kwargs):
        admission = kwargs["admission"]
        active_rows = kwargs["active_rows"]
        # Round 1: rows 0 and 1, both alive.
        active_rows([0, 1])
        yield [70, 71], None
        # Row 0 finished and left the cache; then a row is admitted.
        active_rows([1])
        pending = admission()
        if pending:
            active_rows([1, 2])
        yield [None, 80, 81], None

    monkeypatch.setattr(ar_mod, "run_speculative_server_rounds", fake_rounds)
    cache = _batch_cache(2, 4)
    batch = _spec_batch((1, 2), cache=cache)
    batch.next()  # first tokens
    batch.next()  # round 1
    assert batch.cache_row_for_uid(1) == 0 and batch.cache_row_for_uid(2) == 1

    batch.extend(_spec_batch((3,), first=9, cache=_batch_cache(1, 4)))
    batch.next()  # the admitted row's bonus token
    batch.next()  # the round that polls and grows
    assert batch._loop_rows == [0, 1, 2]
    assert batch.cache_row_for_uid(1) is None, "row 0's columns left the cache"
    assert batch.cache_row_for_uid(2) == 0
    assert batch.cache_row_for_uid(3) == 1


def test_a_restarted_loop_does_not_leave_the_map_pointing_at_the_old_cache(
    monkeypatch,
):
    """``_restart_rounds_for_pending`` swaps the WHOLE cache for the queued
    record's; the map has to move with it, not merely be overwritten later."""
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_ACTIVE", "1")

    def fake_rounds(*args, **kwargs):
        return iter(())  # every row finished: the loop ends at once

    monkeypatch.setattr(ar_mod, "run_speculative_server_rounds", fake_rounds)
    batch = _spec_batch((1, 2), cache=_batch_cache(2, 4))
    batch.next()
    batch.extend(_spec_batch((3,), first=9, cache=_batch_cache(1, 4)))
    batch.next()  # the admitted row's bonus
    batch.next()  # the first loop ends; the batch restarts on the queued row
    assert batch._loop_rows == [2]
    assert batch._cache_rows == [2], (
        "the restart's cache is the queued record's, one row wide, and that row "
        "is this batch's row 2"
    )
    assert batch.cache_row_for_uid(3) == 0
    assert batch.cache_row_for_uid(1) is None


def test_a_queued_row_has_no_cache_row_until_the_merge(monkeypatch):
    """``_admit`` extends the bookkeeping now and the caches at the next round
    boundary; between the two there is no row to snapshot."""
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_ACTIVE", "1")
    cache = _batch_cache(2, 4)
    batch = _spec_batch((1, 2), cache=cache)
    batch.extend(_spec_batch((3,), first=9, cache=_batch_cache(1, 4)))
    assert batch._all_uids == [1, 2, 3]
    assert batch.cache_row_for_uid(3) is None, (
        "the donor's cache has not been merged yet; refusing beats snapshotting "
        "a row that is not there"
    )
    batch._admission_poll()
    assert batch.cache_row_for_uid(3) == 2
