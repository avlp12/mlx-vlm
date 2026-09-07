"""``MLX_VLM_MTP_ROUND_TIMERS`` on the SCALAR MTP round loop.

Same gap as the DFlash one (``test_dflash_scalar_round_timers.py``): the gate
was wired only into ``_mtp_rounds_batch``, so a B == 1 MTP request produced no
split at all.  ``_mtp_rounds`` YIELDS in the middle of its round -- the drafter
cache sync, the rollback and the shared-KV update all run after the tokens have
left -- so its row is published before the yields and completed after them, and
the consumer's time inside the yield belongs to nobody.  These tests pin that
the row is complete, closed, and free of the yield.

CPU only, stubs throughout; no model is loaded.
"""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx_vlm.speculative import mtp as mtp_utils
from mlx_vlm.speculative.mtp import _mtp_rounds


class _Draft:
    def __init__(self):
        self.config = SimpleNamespace(block_size=3)
        self.accept_lens = []
        self.draft_lens = []

    def set_shared_kv(self, *args, **kwargs):
        pass

    def reset(self, model):
        pass

    def draft_block(self, *args, **kwargs):
        return mx.array([[7, 8]], dtype=mx.int32)


class _LM:
    def __init__(self):
        self.rollbacks = []

    def rollback_speculative_cache(self, *args):
        self.rollbacks.append(args)


def _run(timers, max_tokens=5):
    mtp_utils._MTP_ROUND_TIMERS_ENV = None
    lm = _LM()
    draft = _Draft()
    verify = SimpleNamespace(
        hidden=mx.zeros((1, 3, 2), dtype=mx.float32),
        shared_kv_states={},
        gdn_states=None,
        target_tokens=None,
    )
    try:
        with (
            patch.object(mtp_utils, "_mtp_round_timers_enabled", lambda: timers),
            patch.object(mtp_utils, "_mtp_verify_target", return_value=verify),
            patch.object(mtp_utils, "_mtp_acceptance_walk", return_value=(1, [7, 9])),
        ):
            tokens = [
                int(tok)
                for tok, _ in _mtp_rounds(
                    SimpleNamespace(language_model=lm),
                    draft,
                    [SimpleNamespace(offset=0)],
                    mx.zeros((1, 1, 2), dtype=mx.float32),
                    {},
                    first_bonus=1,
                    max_tokens=max_tokens,
                    sampler=lambda logits: mx.argmax(logits, axis=-1),
                    draft_block_size=3,
                    token_dtype=mx.int32,
                    greedy_sampling=True,
                )
            ]
    finally:
        mtp_utils._MTP_ROUND_TIMERS_ENV = None
    return tokens, draft, lm


def test_the_mtp_scalar_timers_change_nothing_and_are_absent_when_off():
    off, draft_off, lm_off = _run(False)
    on, draft_on, lm_on = _run(True)
    assert off == on
    assert off
    assert draft_off.accept_lens == draft_on.accept_lens
    assert len(lm_off.rollbacks) == len(lm_on.rollbacks)
    assert not hasattr(draft_off, "speculative_round_timings")
    assert not hasattr(draft_off, "speculative_draft_seconds")


def test_the_mtp_scalar_row_is_closed_and_excludes_the_yield():
    _, draft, _ = _run(True)
    rows = draft.speculative_round_timings
    assert rows
    # Every round but possibly the LAST is complete: the emit loop can ``return``
    # from inside a yield on the max_tokens edge, and the row is published before
    # the yields precisely so that round is not silently dropped.  It says so
    # instead of pretending its post-yield buckets were measured and were zero.
    assert all(row["complete"] for row in rows[:-1])
    for row in rows:
        assert row["loop"] == "mtp_scalar"
        assert row["sync_added"] is True
        parts = row["draft"] + row["verify"] + row["emit"] + row["rollback"]
        assert row["total"] > 0.0
        assert parts == pytest.approx(row["total"], rel=0.05)
    assert [r["accepted"] for r in rows] == list(draft.accept_lens)
    assert [r["depth"] for r in rows] == list(draft.draft_lens)
    assert draft.speculative_timed_rounds == len(rows)
    assert draft.speculative_round_seconds == pytest.approx(
        sum(r["total"] for r in rows), rel=1e-9
    )
