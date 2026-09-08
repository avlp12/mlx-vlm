"""``MLX_VLM_DFLASH_ROUND_TIMERS`` on the SCALAR DFlash round loop.

The gate was instrumented only in ``_dflash_rounds_batch``.  A served B == 1
request takes ``_dflash_rounds`` (``speculative/utils.py``: ``batch_size == 1
and admission is None``), so on the single-stream rail the knob was a silent
no-op: it neither perturbed the run nor produced a draft/verify split, and the
ms/round a decode cost model is fitted from came from the one loop the timers
never covered.

These tests drive the real ``_dflash_rounds`` over the real (tiny) glm5_next
target the rollback tests use -- real verify forward, real
``rollback_speculative_cache`` -- with a stub drafter, and pin the three things
that make a timing receipt worth reading:

  (a) the timers change nothing.  Same emitted tokens, ON and OFF.
  (b) the split is closed: draft + verify + emit + rollback accounts for the
      round's own wall clock (the remainder is the untimed prologue).
  (c) the row's ``accepted``/``depth`` are the numbers the loop REPORTS for
      that round (``accept_lens``/``draft_lens``), not a second opinion.

CPU only; no model is loaded from disk.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.speculative import dflash as dflash_utils
from mlx_vlm.tests.test_dflash_ragged_rollback import _tiny_glm5_next_target

BLOCK_TOTAL = 5
PROMPT = [2, 4, 6, 8]
FIRST_BONUS = 5
DRAFTS = [11, 12, 13, 14]
SENTINEL = 28
# Dictated acceptance, cycled: a full-accept round (4) skips the rollback
# entirely, a zero-accept round rolls the whole block back.  Both have to leave
# a closed record.
SCRIPT = [3, 0, 4, 1, 2]


class _StubDrafter:
    """The surface the scalar loop actually touches -- not a model."""

    dflash_deferred_walk = False

    def __init__(self):
        self.config = SimpleNamespace(
            block_size=BLOCK_TOTAL,
            runtime_block_size=BLOCK_TOTAL,
            target_layer_ids=[0],
        )
        self.accept_lens = []
        self.draft_lens = []

    def reset(self, model=None):
        self.accept_lens = []
        self.draft_lens = []
        return []  # the scalar loop's draft cache

    def make_cache(self):
        return []

    def draft_block(self, bonus, hidden, cache, bs, sampler, token_dtype, **kw):
        return mx.array([DRAFTS[: bs - 1]], dtype=token_dtype)


@pytest.fixture(scope="module")
def model():
    mx.random.seed(3)
    m = _tiny_glm5_next_target()
    mx.eval(m.parameters())
    return m


def _scripted_walk(script):
    """A walk that dictates acceptance, keeping the real shape contract."""
    state = {"round": 0}

    def walk(draft_tokens, target_tokens, budget, **kwargs):
        accepted = script[state["round"] % len(script)]
        state["round"] += 1
        row = draft_tokens.reshape(-1).tolist()
        accepted = min(accepted, len(row))
        tokens = row[:accepted] + ([SENTINEL] if accepted < len(row) else [])
        if not tokens:  # a full block with nothing to append still emits
            tokens = [SENTINEL]
        return accepted, tokens[: max(1, budget)]

    return walk


def _run(model, monkeypatch, *, timers, max_tokens=18, script=None):
    """One real scalar round loop.  Returns (emitted tokens, drafter)."""
    monkeypatch.setenv("MLX_VLM_DFLASH_ROUND_TIMERS", "1" if timers else "0")
    dflash_utils._ROUND_TIMERS_ENV = None
    if script is not None:
        monkeypatch.setattr(dflash_utils, "_speculative_walk", _scripted_walk(script))

    target = SimpleNamespace(language_model=model)
    cache = model.make_cache()
    model(mx.array([PROMPT], dtype=mx.int32), cache=cache)
    hidden = mx.zeros((1, 1, model.args.hidden_size), dtype=mx.float32)
    drafter = _StubDrafter()
    tokens = []
    rounds = dflash_utils._dflash_rounds(
        target,
        drafter,
        cache,
        hidden,
        first_bonus=FIRST_BONUS,
        max_tokens=max_tokens,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        draft_block_size=BLOCK_TOTAL,
        use_model_initial_block_size=False,
        greedy_sampling=True,
    )
    try:
        for token, _ in rounds:
            tokens.append(int(token))
    finally:
        rounds.close()
        dflash_utils._ROUND_TIMERS_ENV = None
    return tokens, drafter


def test_the_scalar_round_timers_are_off_by_default(model, monkeypatch):
    """OFF is the absence of the attributes, not a zero: a reader must be able
    to tell an untimed run from a timed run that measured nothing."""
    monkeypatch.delenv("MLX_VLM_DFLASH_ROUND_TIMERS", raising=False)
    dflash_utils._ROUND_TIMERS_ENV = None
    try:
        assert dflash_utils._round_timers_enabled() is False
    finally:
        dflash_utils._ROUND_TIMERS_ENV = None

    _, drafter = _run(model, monkeypatch, timers=False)
    assert not hasattr(drafter, "speculative_draft_seconds")
    assert not hasattr(drafter, "speculative_round_timings")
    assert not hasattr(drafter, "speculative_timed_rounds")


def test_the_timers_do_not_change_the_emitted_tokens(model, monkeypatch):
    """(a) The extra mx.evals are waits, not arithmetic."""
    off, drafter_off = _run(model, monkeypatch, timers=False)
    on, drafter_on = _run(model, monkeypatch, timers=True)
    assert off == on
    assert off  # the loop actually ran
    assert drafter_off.accept_lens == drafter_on.accept_lens
    assert drafter_off.draft_lens == drafter_on.draft_lens


def test_the_timers_do_not_change_the_emitted_tokens_under_a_scripted_walk(
    model, monkeypatch
):
    """Same, with acceptance dictated so partial-accept rounds (and therefore
    the rollback, and therefore the third sync) fire on every round."""
    off, _ = _run(model, monkeypatch, timers=False, script=SCRIPT)
    on, _ = _run(model, monkeypatch, timers=True, script=SCRIPT)
    assert off == on
    assert off


def test_each_round_row_is_a_closed_split(model, monkeypatch):
    """(b) draft + verify + emit + rollback ~= the round's own wall clock."""
    _, drafter = _run(model, monkeypatch, timers=True, script=SCRIPT)
    rows = drafter.speculative_round_timings
    assert rows, "the timed loop published no rounds"
    assert drafter.speculative_timed_rounds == len(rows)
    for row in rows:
        assert row["loop"] == "dflash_scalar"
        assert row["sync_added"] is True
        assert row["complete"] is True
        parts = row["draft"] + row["verify"] + row["emit"] + row["rollback"]
        assert row["total"] > 0.0
        assert parts <= row["total"] * (1.0 + 1e-9)
        assert parts >= row["total"] * 0.95, (
            f"un-timed remainder {row['other']:.6f}s is more than 5% of the "
            f"round's {row['total']:.6f}s: {row}"
        )
        assert row["other"] >= -1e-9
        for key in ("draft", "verify", "emit", "rollback"):
            assert row[key] >= 0.0

    # The cumulative attributes -- the ones the batch loop already publishes and
    # /metrics reads -- are the same seconds, summed.
    assert drafter.speculative_draft_seconds == pytest.approx(
        sum(r["draft"] for r in rows), rel=1e-9
    )
    assert drafter.speculative_verify_seconds == pytest.approx(
        sum(r["verify"] for r in rows), rel=1e-9
    )
    assert drafter.speculative_emit_seconds == pytest.approx(
        sum(r["emit"] for r in rows), rel=1e-9
    )
    assert drafter.speculative_rollback_seconds == pytest.approx(
        sum(r["rollback"] for r in rows), rel=1e-9
    )
    assert drafter.speculative_round_seconds == pytest.approx(
        sum(r["total"] for r in rows), rel=1e-9
    )


def test_the_row_reports_the_rounds_own_acceptance_and_depth(model, monkeypatch):
    """(c) One round, one acceptance: the timing row and ``accept_lens`` are the
    same number, and ``depth`` is the drafted width the round was charged for."""
    _, drafter = _run(model, monkeypatch, timers=True, script=SCRIPT)
    rows = drafter.speculative_round_timings
    assert len(rows) == len(drafter.accept_lens) == len(drafter.draft_lens)
    assert [r["accepted"] for r in rows] == list(drafter.accept_lens)
    assert [r["depth"] for r in rows] == list(drafter.draft_lens)
    assert [r["i"] for r in rows] == list(range(len(rows)))
    # A scripted round accepts what the script says (clipped by the emit budget
    # of the final round), so the acceptance really did vary across rows.
    assert len(set(r["accepted"] for r in rows)) > 1
    # A full-accept round rolls nothing back and is charged nothing for it.  The
    # predicate is ``accepted_emitted``, not ``accepted``: an emit budget that
    # stops the row mid-block leaves the walk's number at the full width while
    # the CACHE is rolled back to what was emitted, which is exactly the
    # distinction the two fields exist to keep (common.py::_record_budget_clamp).
    rolled = [r for r in rows if r["accepted_emitted"] < r["depth"]]
    assert rolled, "no round rolled back; the rollback bucket was never exercised"
    for row in rows:
        if row["accepted_emitted"] == row["depth"]:
            assert row["rollback"] == 0.0
        else:
            assert row["rollback"] > 0.0


def test_the_split_reaches_the_metrics_snapshot(model, monkeypatch):
    """The output channel is the batch loop's own: drafter attributes ->
    ``_speculative_stats_snapshot`` -> ``GET /metrics``, which is what
    bench/ops/gdn_e2e_arms.py polls before/after a run.  ``draft_seconds`` and
    ``verify_seconds`` keep their exact meaning (an existing reader is
    unaffected); the emit/rollback/round fields and the rows are additive."""
    app_module = pytest.importorskip("mlx_vlm.server.app")
    _, drafter = _run(model, monkeypatch, timers=True, script=SCRIPT)
    monkeypatch.setattr(
        app_module.runtime,
        "response_generator",
        SimpleNamespace(draft_model=drafter, draft_kind="dflash"),
        raising=False,
    )
    snapshot = app_module._speculative_stats_snapshot()
    for key in (
        "draft_seconds",
        "verify_seconds",
        "emit_seconds",
        "rollback_seconds",
        "round_seconds",
        "timed_rounds",
        "round_timings",
        "round_timer_sync_points",
    ):
        assert key in snapshot, key
    assert snapshot["timed_rounds"] == len(snapshot["round_timings"])
    assert snapshot["round_timer_sync_points"] == ["draft", "verify", "rollback"]
    assert snapshot["round_seconds"] == pytest.approx(
        sum(r["total"] for r in snapshot["round_timings"]), rel=1e-9
    )

    # An UNTIMED drafter must publish none of it: absence is not zero.
    monkeypatch.setattr(
        app_module.runtime,
        "response_generator",
        SimpleNamespace(draft_model=_StubDrafter(), draft_kind="dflash"),
        raising=False,
    )
    untimed = app_module._speculative_stats_snapshot()
    assert "emit_seconds" not in untimed
    assert "round_timings" not in untimed
