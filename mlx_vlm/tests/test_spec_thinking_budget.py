"""A thinking budget on the SPECULATIVE serving path, and the latent bug under it.

The server used to refuse ``thinking_budget`` whenever a drafter was loaded.
That refusal was unimplemented-not-impossible: a budget never has to change what
the target would have produced, only to STOP the accepted walk at an exact
position.  For every ``j < accepted`` the drafted token IS the target's own
token, so emitting ``draft[:k]`` is byte-identical to the autoregressive
reference truncated at ``k``.

Truncating mid-block was not safe before this branch, though, and that is the
first half of this file.  Every walk already truncated ``new_tokens`` by a
budget while returning ``accepted`` at its PRE-truncation value, and every cache
consumer downstream -- the DFlash hidden window and rollback guard, the MTP
positions ledger the sampled coupling is keyed to, the rollback length itself --
reads ``accepted``.  Nothing broke while the only budget was ``max_tokens``
(truncation meant the row ended), and everything would have broken the moment a
budget cut a row mid-stream: live KV for tokens the stream never emitted.

The second half is the budget itself: an emit cap that stops the walk, and a
forced round that places ``\\n</think>`` as the front of the draft so the
target's bonus is conditioned on the CLOSED thinking block rather than on
whatever the drafter guessed.

Everything here is CPU stubs.  The "target" is a first-order deterministic map
``f(t) = 7t + 3 mod V``: its greedy token at block position j is ``f(input[j])``,
which makes the autoregressive reference a closed form and makes "the drafter
was right" and "the walk accepted" the same statement.
"""

import contextlib
import os
from threading import Event
from types import SimpleNamespace
from typing import List, Optional

import mlx.core as mx
import pytest

from mlx_vlm.generate import ar as ar_mod
from mlx_vlm.prompt_utils import template_references_kw
from mlx_vlm.server import generation as server_generation
from mlx_vlm.server.app import (
    _speculative_stats_snapshot,
    _thinking_config_snapshot,
)
from mlx_vlm.server.runtime import runtime as server_runtime
from mlx_vlm.speculative import dflash as dflash_utils
from mlx_vlm.speculative import mtp as mtp_utils
from mlx_vlm.speculative import utils as speculative_utils
from mlx_vlm.speculative.common import (
    _speculative_walk,
    _speculative_walk_batch,
    _speculative_walk_batch_uniform_acceptance,
    accepted_from_emitted,
    accepted_list_from_emitted,
    place_forced_draft_prefix,
    round_emit_plan,
)
from mlx_vlm.utils import ThinkingBudgetCriteria

VOCAB = 64
HIDDEN = 4
BLOCK = 5  # verify block total: 1 bonus + 4 drafted
# Stand-ins for the two ids ThinkingBudgetCriteria forces, kept in vocabulary
# (a bonus token is fed straight back as the next round's input).
NEWLINE_ID = 41
END_THINK_ID = 42


def nxt(token: int) -> int:
    """The scripted target's greedy continuation."""
    return (int(token) * 7 + 3) % VOCAB


def ar_reference(bonus: int, count: int) -> List[int]:
    """``count`` tokens the target alone would emit after ``bonus``."""
    out = []
    token = int(bonus)
    for _ in range(count):
        token = nxt(token)
        out.append(token)
    return out


# ==========================================================================
# R1 -- the walks: acceptance must describe what was EMITTED
# ==========================================================================
class TestAcceptedFromEmitted:
    def test_a_full_round_is_unchanged(self):
        assert accepted_from_emitted([9, 8, 7]) == 2
        assert accepted_list_from_emitted([[9, 8, 7], [1]]) == [2, 0]

    def test_an_empty_row_accepted_nothing(self):
        # ``len - 1`` would be -1 and would trim a cache backwards.
        assert accepted_from_emitted([]) == 0

    @pytest.mark.parametrize("cap", [1, 2, 3, 4])
    def test_b1_walk_accepted_matches_the_emitted_prefix(self, cap):
        draft = mx.array([[nxt(5), nxt(nxt(5)), nxt(nxt(nxt(5)))]], dtype=mx.int32)
        target = mx.array(
            [[nxt(5), nxt(nxt(5)), nxt(nxt(nxt(5))), nxt(nxt(nxt(nxt(5))))]],
            dtype=mx.int32,
        )
        accepted, new_tokens = _speculative_walk(draft, target, cap)
        assert accepted == 3, "the raw walk still reports what it verified"
        assert len(new_tokens) == min(4, cap)
        assert accepted_from_emitted(new_tokens) == min(4, cap) - 1

    def test_batch_walk_accepted_matches_the_emitted_prefix_per_row(self):
        draft = mx.array([[10, 11, 12], [20, 21, 22]], dtype=mx.int32)
        target = mx.array([[10, 11, 12, 13], [20, 99, 22, 23]], dtype=mx.int32)
        accepted, new_tokens = _speculative_walk_batch(draft, target, [2, 9])
        assert accepted == [3, 1], "raw acceptance is per row and pre-truncation"
        assert new_tokens == [[10, 11], [20, 99]]
        assert accepted_list_from_emitted(new_tokens) == [1, 1]


class TestForcedPrefixWalks:
    def test_forced_positions_are_accepted_without_comparison(self):
        # The target disagrees at BOTH forced positions and agrees nowhere else.
        draft = mx.array([[NEWLINE_ID, END_THINK_ID, 12]], dtype=mx.int32)
        target = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        accepted, new_tokens = _speculative_walk(
            draft, target, 9, forced_prefix_len=2
        )
        assert accepted == 2
        assert new_tokens == [NEWLINE_ID, END_THINK_ID, 3], (
            "the bonus must be the target's token AT the position after the "
            "forced run, i.e. conditioned on the closed thinking block"
        )

    def test_a_forced_run_longer_than_the_draft_is_clamped(self):
        draft = mx.array([[NEWLINE_ID, END_THINK_ID]], dtype=mx.int32)
        target = mx.array([[1, 2, 3]], dtype=mx.int32)
        accepted, new_tokens = _speculative_walk(
            draft, target, 9, forced_prefix_len=17
        )
        assert accepted == 2
        assert new_tokens == [NEWLINE_ID, END_THINK_ID, 3]

    def test_uniform_acceptance_forces_only_what_every_row_asked_for(self):
        draft = mx.array([[NEWLINE_ID, 11], [20, 21]], dtype=mx.int32)
        target = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int32)
        accepted, new_tokens = _speculative_walk_batch_uniform_acceptance(
            draft, target, [0, 0], [9, 9], forced_prefix_lens=[1, 0]
        )
        # Row 1 asked for nothing; forcing position 0 there would commit an
        # unverified drafted token, so the mixed batch forces nothing.
        assert accepted == [0, 0]
        assert new_tokens == [[1], [4]]

    def test_uniform_acceptance_forces_when_every_row_asked(self):
        draft = mx.array([[NEWLINE_ID, 11], [NEWLINE_ID, 21]], dtype=mx.int32)
        target = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int32)
        accepted, new_tokens = _speculative_walk_batch_uniform_acceptance(
            draft, target, [0, 0], [9, 9], forced_prefix_lens=[1, 1]
        )
        assert accepted == [1, 1]
        assert new_tokens == [[NEWLINE_ID, 2], [NEWLINE_ID, 5]]


class TestRoundEmitPlan:
    def test_no_criteria_leaves_the_budget_alone(self):
        assert round_emit_plan(0, None, None, 4, 7) == (7, [])

    def test_a_positive_cap_narrows_the_budget(self):
        assert round_emit_plan(0, lambda row: 3, lambda row: [], 4, 7) == (3, [])

    def test_a_zero_cap_with_forced_ids_buys_the_run_plus_a_bonus(self):
        budget, forced = round_emit_plan(
            0, lambda row: 0, lambda row: [NEWLINE_ID, END_THINK_ID], 4, 7
        )
        assert (budget, forced) == (3, [NEWLINE_ID, END_THINK_ID])

    def test_a_zero_cap_with_nothing_to_force_still_makes_progress(self):
        # A 0 budget and no forced ids would spin the round loop forever.
        assert round_emit_plan(0, lambda row: 0, lambda row: [], 4, 7) == (1, [])

    def test_the_forced_run_cannot_outgrow_the_drafted_width(self):
        budget, forced = round_emit_plan(
            0, lambda row: 0, lambda row: [NEWLINE_ID, END_THINK_ID], 1, 7
        )
        assert forced == [NEWLINE_ID]
        assert budget == 2


def test_place_forced_draft_prefix_only_touches_the_front_of_the_named_rows():
    draft = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int32)
    out = place_forced_draft_prefix(draft, [[NEWLINE_ID, END_THINK_ID], []])
    assert out.tolist() == [[NEWLINE_ID, END_THINK_ID, 3], [4, 5, 6]]
    assert out.dtype == draft.dtype


def test_place_forced_draft_prefix_is_a_noop_without_forced_rows():
    draft = mx.array([[1, 2, 3]], dtype=mx.int32)
    assert place_forced_draft_prefix(draft, [[], []]) is draft


# ==========================================================================
# Shared scripted target / drafter for the round loops
# ==========================================================================
class _ScriptedTarget:
    """Greedy next token at block position j is ``f(verify_input[j])``.

    Deterministic, first order, and in vocabulary -- so the autoregressive
    reference is a closed form and an accepted draft and a correct draft are
    the same event.
    """

    def __init__(self):
        self.rollbacks: List[tuple] = []
        self.verify_widths: List[int] = []

    def __call__(self, verify_input, cache=None, **kwargs):
        rows = verify_input.tolist()
        batch, width = len(rows), len(rows[0])
        self.verify_widths.append(width)
        onehot = [[[0.0] * VOCAB for _ in range(width)] for _ in range(batch)]
        for i in range(batch):
            for j in range(width):
                onehot[i][j][nxt(rows[i][j])] = 1.0
        return SimpleNamespace(
            hidden_states=[mx.zeros((batch, width, HIDDEN), dtype=mx.float32)],
            logits=mx.array(onehot, dtype=mx.float32),
            gdn_states=None,
        )

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        value = (
            [int(v) for v in accepted.reshape(-1).tolist()]
            if isinstance(accepted, mx.array)
            else accepted
        )
        self.rollbacks.append((value, int(block_size)))


class _ScriptedDrafter:
    """Proposes the target's own continuation from the bonus it was handed."""

    requires_uniform_batch_acceptance = False

    def __init__(self, deferred: bool = False):
        self.config = SimpleNamespace(
            block_size=BLOCK, runtime_block_size=BLOCK, target_layer_ids=[0]
        )
        self.dflash_deferred_walk = deferred
        self.accept_lens: List[float] = []
        self.draft_lens: List[int] = []

    def reset(self, model=None, **kwargs):
        self.accept_lens = []
        self.draft_lens = []
        return []

    def make_cache(self):
        return []

    def draft_block(self, bonus, hidden, cache, bs, sampler, token_dtype, **kwargs):
        seeds = (
            [int(bonus)]
            if not isinstance(bonus, mx.array) or bonus.ndim == 0
            else [int(v) for v in bonus.reshape(-1).tolist()]
        )
        return mx.array(
            [ar_reference(seed, bs - 1) for seed in seeds], dtype=token_dtype
        )


@pytest.fixture
def pinned_block(monkeypatch):
    """Pin the verify block total so the emit cap is the only variable.

    The width ladder would otherwise narrow the block in response to the cap,
    which is a real optimisation (and is tested separately) but hides the thing
    these tests are about: what happens when the cap lands INSIDE the block.
    """
    monkeypatch.setattr(
        dflash_utils,
        "_dflash_next_block_size",
        lambda drafter, requested, remaining, initial=None: BLOCK,
    )


def _run_dflash_b1(
    *,
    bonus: int,
    max_tokens: int,
    deferred: bool,
    emit_limit=None,
    forced_draft_ids=None,
):
    target = _ScriptedTarget()
    drafter = _ScriptedDrafter(deferred=deferred)
    emitted = [
        token
        for token, _ in dflash_utils._dflash_rounds(
            SimpleNamespace(language_model=target),
            drafter,
            [],
            mx.zeros((1, 1, HIDDEN), dtype=mx.float32),
            first_bonus=bonus,
            max_tokens=max_tokens,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            greedy_sampling=True,
            emit_limit=emit_limit,
            forced_draft_ids=forced_draft_ids,
        )
    ]
    return target, drafter, emitted


# ==========================================================================
# R1 -- the DFlash B=1 round loop
# ==========================================================================
@pytest.mark.parametrize("deferred", [False, True], ids=["eager", "deferred"])
class TestDflashSingletonEmitCap:
    def test_an_uncapped_run_reproduces_the_autoregressive_reference(
        self, pinned_block, deferred
    ):
        _, _, emitted = _run_dflash_b1(bonus=5, max_tokens=13, deferred=deferred)
        assert emitted == ar_reference(5, 12), (
            "12 not 13: the caller already yielded the first bonus"
        )

    def test_a_capped_stream_is_the_reference_truncated_at_the_same_point(
        self, pinned_block, deferred
    ):
        _, _, emitted = _run_dflash_b1(
            bonus=5, max_tokens=13, deferred=deferred, emit_limit=lambda row: 2
        )
        assert emitted == ar_reference(5, 12), (
            "an emit cap changes WHERE the round stops, never WHICH token it "
            "emits: draft[j] == target[j] for every j < accepted"
        )

    def test_the_rollback_is_called_with_the_emitted_count_not_the_verified_one(
        self, pinned_block, deferred
    ):
        target, _, _ = _run_dflash_b1(
            bonus=5, max_tokens=9, deferred=deferred, emit_limit=lambda row: 2
        )

        # The drafter is perfect, so every round verified all 4 drafts. A cap of
        # 2 emits 2 of the 5 block tokens, so the cache must be trimmed to 1
        # accepted draft. Pre-fix, ``accepted == bs - 1`` skipped the rollback
        # entirely and left three verified-but-unemitted tokens live.
        assert target.rollbacks, (
            "no rollback at all is the pre-fix behaviour: accepted was bs-1"
        )
        assert all(
            accepted == 1 and block == BLOCK for accepted, block in target.rollbacks
        ), target.rollbacks

    def test_an_uncapped_perfect_round_still_skips_the_rollback(
        self, pinned_block, deferred
    ):
        target, _, _ = _run_dflash_b1(bonus=5, max_tokens=11, deferred=deferred)
        assert target.rollbacks == [], (
            "the fix must not cost a rollback on the steady-state full-accept "
            "round -- that is the whole speculative win"
        )

    def test_the_receipt_separates_verified_acceptance_from_the_budget_giveback(
        self, pinned_block, deferred
    ):
        _, drafter, _ = _run_dflash_b1(
            bonus=5, max_tokens=9, deferred=deferred, emit_limit=lambda row: 2
        )
        # 4 rounds of 2 emitted tokens each, from emitted=1 to emitted=9.
        assert drafter.accept_lens and all(a == 4 for a in drafter.accept_lens), (
            "acceptance stats stay pre-truncation: the drafter WAS right about "
            "four tokens, and a budget give-back is not a drafter miss"
        )
        assert drafter.budget_clamped_tokens == 3 * len(drafter.accept_lens)
        assert (
            drafter.speculative_total_budget_clamped
            == drafter.budget_clamped_tokens
        )


# ==========================================================================
# R3 -- the forced-draft round
# ==========================================================================
def _run_dflash_b1_forcing(*, deferred, force_after, forced_ids, max_tokens):
    """Drive the B=1 loop and flip the budget ON after ``force_after`` tokens.

    The flag is flipped from the CONSUMER side, between the round that emitted
    the last free token and the round that reads ``emit_limit`` -- which is
    exactly where the criteria flips it in the server, since the criteria is
    driven by the emitted tokens themselves.
    """
    state = {"forcing": False}
    target = _ScriptedTarget()
    drafter = _ScriptedDrafter(deferred=deferred)
    emitted: List[int] = []
    rounds = dflash_utils._dflash_rounds(
        SimpleNamespace(language_model=target),
        drafter,
        [],
        mx.zeros((1, 1, HIDDEN), dtype=mx.float32),
        first_bonus=5,
        max_tokens=max_tokens,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        greedy_sampling=True,
        emit_limit=lambda row: 0 if state["forcing"] else None,
        forced_draft_ids=lambda row: list(forced_ids) if state["forcing"] else [],
    )
    for token, _ in rounds:
        emitted.append(int(token))
        if len(emitted) == force_after:
            state["forcing"] = True
        elif state["forcing"] and len(emitted) >= force_after + len(forced_ids) + 1:
            state["forcing"] = False
    return target, drafter, emitted


@pytest.mark.parametrize("deferred", [False, True], ids=["eager", "deferred"])
def test_a_forced_round_emits_exactly_the_closing_run_then_a_target_bonus(
    pinned_block, deferred
):
    """The boundary round: ``\\n</think>`` placed, then the target's own token.

    Conditioning is the whole point -- the bonus is drawn at the position AFTER
    the closing run, so it is the model's continuation of a CLOSED thinking
    block rather than of whatever the drafter had proposed.
    """
    _, _, emitted = _run_dflash_b1_forcing(
        deferred=deferred,
        force_after=BLOCK,
        forced_ids=[NEWLINE_ID, END_THINK_ID],
        max_tokens=16,
    )
    assert emitted[:BLOCK] == ar_reference(5, BLOCK), (
        "the free rounds before the boundary are the plain reference"
    )
    assert emitted[BLOCK : BLOCK + 3] == [
        NEWLINE_ID,
        END_THINK_ID,
        nxt(END_THINK_ID),
    ], "forced run, then the target's bonus conditioned on it"
    assert emitted[BLOCK + 3] == nxt(nxt(END_THINK_ID)), (
        "and the stream resumes from the closed block, not from where the "
        "drafter had been heading"
    )


@pytest.mark.parametrize("deferred", [False, True], ids=["eager", "deferred"])
def test_a_one_token_forced_run_also_lands(pinned_block, deferred):
    """Half a closing run: the state the round after ``\\n`` is emitted sees."""
    _, _, emitted = _run_dflash_b1_forcing(
        deferred=deferred,
        force_after=BLOCK,
        forced_ids=[END_THINK_ID],
        max_tokens=14,
    )
    assert emitted[BLOCK : BLOCK + 2] == [END_THINK_ID, nxt(END_THINK_ID)]


@pytest.mark.parametrize("deferred", [False, True], ids=["eager", "deferred"])
def test_the_forced_round_rolls_the_cache_back_to_what_it_emitted(
    pinned_block, deferred
):
    target, _, _ = _run_dflash_b1_forcing(
        deferred=deferred,
        force_after=BLOCK,
        forced_ids=[NEWLINE_ID, END_THINK_ID],
        max_tokens=16,
    )
    assert (2, BLOCK) in target.rollbacks, (
        "the forced round emitted 3 of 5 block tokens, so exactly 2 drafted "
        f"positions may stay live; saw {target.rollbacks}"
    )


# ==========================================================================
# R1 -- the DFlash continuous-batching loop
# ==========================================================================
def _run_dflash_batch(*, bonuses, max_tokens, emit_limit=None, forced_draft_ids=None):
    target = _ScriptedTarget()
    drafter = _ScriptedDrafter()
    rows = [[] for _ in bonuses]
    for tokens_out, _ in dflash_utils._dflash_rounds_batch(
        SimpleNamespace(language_model=target),
        drafter,
        [],
        mx.zeros((len(bonuses), 1, HIDDEN), dtype=mx.float32),
        first_bonus=mx.array(bonuses, dtype=mx.int32),
        max_tokens=max_tokens,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        greedy_sampling=True,
        emit_limit=emit_limit,
        forced_draft_ids=forced_draft_ids,
    ):
        for i, token in enumerate(tokens_out):
            if token is not None:
                rows[i].append(int(token))
    return target, drafter, rows


class TestDflashBatchEmitCap:
    def test_an_uncapped_batch_reproduces_the_reference_per_row(self, pinned_block):
        _, _, rows = _run_dflash_batch(bonuses=[5, 9], max_tokens=13)
        assert rows == [ar_reference(5, 12), ar_reference(9, 12)]

    def test_a_row_only_capped_row_still_matches_its_own_reference(self, pinned_block):
        _, _, rows = _run_dflash_batch(
            bonuses=[5, 9],
            max_tokens=13,
            emit_limit=lambda row: 2 if row == 0 else None,
        )
        assert rows == [ar_reference(5, 12), ar_reference(9, 12)]

    def test_the_rollback_sees_the_emitted_count_for_every_row(self, pinned_block):
        target, _, _ = _run_dflash_batch(
            bonuses=[5, 9],
            max_tokens=BLOCK * 4,
            emit_limit=lambda row: 2 if row == 0 else None,
        )
        assert target.rollbacks, "a ragged emit must roll the cache back"
        accepted, block = target.rollbacks[0]
        assert block == BLOCK
        # Row 0 emitted 2 of its 5 block tokens, so exactly 1 drafted position
        # may stay live; row 1 emitted all 5 and keeps all 4.
        assert accepted == [1, BLOCK - 1], accepted

    def test_an_uncapped_batch_never_touches_the_budget_counter(self, pinned_block):
        _, drafter, _ = _run_dflash_batch(bonuses=[5, 9], max_tokens=BLOCK * 2 + 1)
        assert getattr(drafter, "budget_clamped_tokens", 0) == 0, (
            "the counter must stay a signal, not a constant: a round that "
            "emitted everything it verified gave nothing back"
        )

    def test_the_budget_giveback_is_counted_when_a_row_is_capped(self, pinned_block):
        _, drafter, _ = _run_dflash_batch(
            bonuses=[5, 9],
            max_tokens=BLOCK * 4,
            emit_limit=lambda row: 2 if row == 0 else None,
        )
        assert drafter.budget_clamped_tokens >= 3, (
            "row 0 verified 4 drafts and emitted 1 in its first round alone"
        )
        assert (
            drafter.speculative_total_budget_clamped
            == drafter.budget_clamped_tokens
        ), "per-request and lifetime counters agree inside one request"


def _run_dflash_batch_forcing(*, bonuses, force_after, forced_ids, max_tokens):
    """Force row 0 only, once row 0 has emitted ``force_after`` tokens."""
    state = {"forcing": False}
    target = _ScriptedTarget()
    drafter = _ScriptedDrafter()
    rows = [[] for _ in bonuses]
    for tokens_out, _ in dflash_utils._dflash_rounds_batch(
        SimpleNamespace(language_model=target),
        drafter,
        [],
        mx.zeros((len(bonuses), 1, HIDDEN), dtype=mx.float32),
        first_bonus=mx.array(bonuses, dtype=mx.int32),
        max_tokens=max_tokens,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        greedy_sampling=True,
        emit_limit=lambda row: 0 if (row == 0 and state["forcing"]) else None,
        forced_draft_ids=lambda row: (
            list(forced_ids) if (row == 0 and state["forcing"]) else []
        ),
    ):
        for i, token in enumerate(tokens_out):
            if token is not None:
                rows[i].append(int(token))
        if len(rows[0]) == force_after:
            state["forcing"] = True
        elif state["forcing"] and len(rows[0]) >= force_after + len(forced_ids) + 1:
            state["forcing"] = False
    return target, drafter, rows


def test_a_forced_row_does_not_force_its_neighbours(pinned_block):
    """Only the row whose budget tripped gets the closing run placed.

    The other row's draft is untouched and its walk still compares against the
    target -- forcing a position for a row that did not ask would commit an
    unverified drafted token into its cache.
    """
    _, _, rows = _run_dflash_batch_forcing(
        bonuses=[5, 9],
        force_after=BLOCK,
        forced_ids=[NEWLINE_ID, END_THINK_ID],
        max_tokens=BLOCK * 3,
    )
    assert rows[0][BLOCK : BLOCK + 3] == [
        NEWLINE_ID,
        END_THINK_ID,
        nxt(END_THINK_ID),
    ]
    assert rows[1] == ar_reference(9, len(rows[1])), (
        "row 1 never asked for a budget and must walk normally"
    )


# ==========================================================================
# R1 -- the MTP continuous-batching loop and its positions ledger
# ==========================================================================
class _MTPTarget:
    """Same scripted map, through the MTP verify surface."""

    def __init__(self):
        self.rollbacks: List[tuple] = []

    def speculative_verify_logits(self, verify_input, prompt_cache, sampler):
        rows = verify_input.tolist()
        batch, width = len(rows), len(rows[0])
        target = mx.array(
            [[nxt(token) for token in row] for row in rows], dtype=mx.int32
        )
        hidden = mx.zeros((batch, width, HIDDEN), dtype=mx.float32)
        return hidden, {}, None, target

    def rollback_speculative_cache(self, prompt_cache, gdn_states, accepted, bs):
        self.rollbacks.append((list(accepted), int(bs)))


class _MTPDrafter:
    prefer_requested_block_size = True

    def __init__(self):
        self.config = SimpleNamespace(block_size=BLOCK)
        self.accept_lens: List[float] = []
        self.draft_lens: List[int] = []
        self.kv_valid_lens: List[List[int]] = []

    def reset(self, model=None, left_padding=None):
        self.accept_lens = []
        self.draft_lens = []

    def set_shared_kv(self, shared_kv, kv_offset=None, position=None,
                      kv_valid_len=None, left_padding=None):
        self.kv_valid_lens.append([int(v) for v in kv_valid_len.tolist()])

    def draft_block(self, bonus, hidden, cache, bs, sampler, token_dtype, **kwargs):
        seeds = [int(v) for v in mx.array(bonus).reshape(-1).tolist()]
        return mx.array(
            [ar_reference(seed, bs - 1) for seed in seeds], dtype=token_dtype
        )


def _run_mtp(*, bonuses, max_tokens, emit_limit=None, forced_draft_ids=None,
             prefill_len=7):
    target = _MTPTarget()
    drafter = _MTPDrafter()
    prompt_cache = [SimpleNamespace(offset=prefill_len)]
    rows = [[] for _ in bonuses]
    for tokens_out, _ in mtp_utils._mtp_rounds_batch(
        SimpleNamespace(language_model=target),
        drafter,
        prompt_cache,
        mx.zeros((len(bonuses), 1, HIDDEN), dtype=mx.float32),
        {},
        first_bonus=mx.array(bonuses, dtype=mx.int32),
        max_tokens=max_tokens,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        greedy_sampling=True,
        emit_limit=emit_limit,
        forced_draft_ids=forced_draft_ids,
    ):
        for i, token in enumerate(tokens_out):
            if token is not None:
                rows[i].append(int(token))
    return target, drafter, rows


class TestMTPPositionsLedger:
    def test_an_uncapped_run_reproduces_the_reference(self):
        _, _, rows = _run_mtp(bonuses=[5], max_tokens=13)
        assert rows == [ar_reference(5, 12)]

    def test_the_ledger_advances_by_the_emitted_count_under_a_cap(self):
        """The sampled coupling is keyed by (row, absolute position).

        A round that emits fewer tokens than it verified must NOT skip the
        positions it threw away, or the next round draws the target's sample
        for a position the stream never reached and the proposal and the target
        stop sharing a key.  ``accepted + 1`` was that skip: with a cap of 2 the
        walk verified 2 drafts (raw accepted 2, ledger +3) while the round
        emitted 2 tokens.
        """
        prefill = 7
        _, drafter, _ = _run_mtp(
            bonuses=[5], max_tokens=9, emit_limit=lambda row: 2, prefill_len=prefill
        )
        ledger = [entry[0] for entry in drafter.kv_valid_lens]
        assert ledger[0] == prefill
        deltas = [ledger[i + 1] - ledger[i] for i in range(len(ledger) - 1)]
        assert deltas and all(delta == 2 for delta in deltas), (
            f"a 2-token round must advance the ledger by 2, got {deltas}"
        )

    def test_the_ledger_advances_by_the_full_block_with_no_cap(self):
        _, drafter, _ = _run_mtp(bonuses=[5], max_tokens=11, prefill_len=3)
        ledger = [entry[0] for entry in drafter.kv_valid_lens]
        deltas = [ledger[i + 1] - ledger[i] for i in range(len(ledger) - 1)]
        assert deltas and all(delta == BLOCK for delta in deltas), deltas

    def test_the_rollback_sees_the_emitted_count(self):
        target, _, _ = _run_mtp(bonuses=[5], max_tokens=9, emit_limit=lambda row: 2)
        assert target.rollbacks
        assert all(accepted == [1] for accepted, _ in target.rollbacks), (
            target.rollbacks
        )

    def test_a_forced_mtp_round_emits_the_closing_run_then_a_bonus(self):
        state = {"forcing": False}
        target = _MTPTarget()
        drafter = _MTPDrafter()
        rows = [[]]
        for tokens_out, _ in mtp_utils._mtp_rounds_batch(
            SimpleNamespace(language_model=target),
            drafter,
            [SimpleNamespace(offset=7)],
            mx.zeros((1, 1, HIDDEN), dtype=mx.float32),
            {},
            first_bonus=mx.array([5], dtype=mx.int32),
            max_tokens=BLOCK * 3,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            greedy_sampling=True,
            emit_limit=lambda row: 0 if state["forcing"] else None,
            forced_draft_ids=lambda row: (
                [NEWLINE_ID, END_THINK_ID] if state["forcing"] else []
            ),
        ):
            for token in tokens_out:
                if token is not None:
                    rows[0].append(int(token))
            if len(rows[0]) == BLOCK:
                state["forcing"] = True
            elif state["forcing"] and len(rows[0]) >= BLOCK + 3:
                state["forcing"] = False
        assert rows[0][:BLOCK] == ar_reference(5, BLOCK)
        assert rows[0][BLOCK : BLOCK + 3] == [
            NEWLINE_ID,
            END_THINK_ID,
            nxt(END_THINK_ID),
        ]


# ==========================================================================
# R2 -- SpeculativeGenerationBatch threading
# ==========================================================================
class _FakeCriteria:
    """The surface ``SpeculativeGenerationBatch`` actually calls."""

    def __init__(self, limit=None, forced=()):
        self.seen: List[int] = []
        self._limit = limit
        self._forced = list(forced)

    def __call__(self, token_id):
        self.seen.append(int(token_id))
        return None

    def tokens_before_budget_stop(self):
        return self._limit

    def pending_forced_sequence(self):
        return list(self._forced)


def _make_spec_batch(criteria_list, uids=(1,)):
    batch = ar_mod.SpeculativeGenerationBatch(
        model=SimpleNamespace(),
        draft_model=SimpleNamespace(),
        draft_kind="dflash",
        uids=list(uids),
        first_tokens=mx.array([5] * len(uids), dtype=mx.int32),
        prompt_cache=[],
        sampler=lambda logits: logits,
        stop_criteria=lambda token: False,
        max_tokens=[64] * len(uids),
        hidden=mx.zeros((len(uids), 1, HIDDEN), dtype=mx.float32),
        shared_kv_states=None,
        prompt_tokens=None,
        thinking_budget_criteria=list(criteria_list),
    )
    return batch


class TestSpeculativeBatchWiring:
    def test_missing_rows_are_padded_with_none(self):
        batch = _make_spec_batch([], uids=(1, 2))
        assert batch.thinking_budget_criteria == [None, None]
        assert batch._emit_limit(0) is None
        assert batch._forced_draft_ids(1) == []

    def test_the_emit_limit_and_forced_ids_come_from_the_row_criteria(self):
        criteria = _FakeCriteria(limit=0, forced=[NEWLINE_ID, END_THINK_ID])
        batch = _make_spec_batch([criteria])
        assert batch._emit_limit(0) == 0
        assert batch._forced_draft_ids(0) == [NEWLINE_ID, END_THINK_ID]

    def test_the_hooks_are_passed_only_when_a_row_has_a_budget(self, monkeypatch):
        seen = {}

        def fake_rounds(*args, **kwargs):
            seen.update(kwargs)
            yield [None], None

        monkeypatch.setattr(ar_mod, "run_speculative_server_rounds", fake_rounds)

        batch = _make_spec_batch([None])
        batch._start_rounds()
        next(batch._rounds_iter)  # generators are lazy; nothing runs until now
        assert seen["emit_limit"] is None
        assert seen["forced_draft_ids"] is None

        seen.clear()
        batch = _make_spec_batch([_FakeCriteria()])
        batch._start_rounds()
        next(batch._rounds_iter)
        assert callable(seen["emit_limit"])
        assert callable(seen["forced_draft_ids"])

    def test_every_emitted_token_reaches_the_criteria_in_order(self, monkeypatch):
        criteria = _FakeCriteria()
        holder = {}

        def fake_rounds(*args, **kwargs):
            holder["stop_check"] = kwargs["stop_check"]
            yield [None], None

        monkeypatch.setattr(ar_mod, "run_speculative_server_rounds", fake_rounds)
        batch = _make_spec_batch([criteria])
        batch._start_rounds()
        next(batch._rounds_iter)
        for token in (11, 12, 13):
            holder["stop_check"](0, token)
        assert criteria.seen == [11, 12, 13], (
            "the criteria's forced-sequence ledger advances once per emitted "
            "token, exactly as it does on the autoregressive path"
        )


# ==========================================================================
# R2/R3 -- the real criteria's non-consuming peek
# ==========================================================================
class _TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"\n": [NEWLINE_ID], "</think>": [END_THINK_ID], "<think>": [40]}[text]


def _criteria(budget):
    return ThinkingBudgetCriteria(
        tokenizer=_TinyTokenizer(),
        thinking_budget=budget,
        thinking_end_token="</think>",
        thinking_start_token="<think>",
        enable_thinking=True,
        prompt_preopens_thinking=True,
    )


class TestThinkingBudgetCriteriaPeek:
    def test_nothing_pends_before_the_budget_trips(self):
        criteria = _criteria(3)
        assert criteria.pending_forced_sequence() == []
        assert criteria.tokens_before_budget_stop() == 4

    def test_the_whole_closing_run_is_visible_the_moment_it_trips(self):
        criteria = _criteria(2)
        for token in (10, 11, 12):
            criteria(token)
        assert criteria.pending_forced_sequence() == [NEWLINE_ID, END_THINK_ID]
        assert criteria.tokens_before_budget_stop() == 0

    def test_peeking_does_not_consume(self):
        criteria = _criteria(2)
        for token in (10, 11, 12):
            criteria(token)
        first = criteria.pending_forced_sequence()
        assert criteria.pending_forced_sequence() == first
        assert criteria.pending_forced_sequence() == first

    def test_emitting_the_run_consumes_it_exactly_once(self):
        criteria = _criteria(2)
        for token in (10, 11, 12):
            criteria(token)
        criteria(NEWLINE_ID)
        assert criteria.pending_forced_sequence() == [END_THINK_ID]
        criteria(END_THINK_ID)
        assert criteria.pending_forced_sequence() == []
        assert criteria.tokens_before_budget_stop() is None, (
            "the block is closed, so the budget no longer applies"
        )

    def test_the_budget_does_not_apply_outside_a_thinking_block(self):
        criteria = ThinkingBudgetCriteria(
            tokenizer=_TinyTokenizer(),
            thinking_budget=2,
            thinking_end_token="</think>",
            thinking_start_token="<think>",
            enable_thinking=True,
            prompt_preopens_thinking=False,
        )
        assert criteria.tokens_before_budget_stop() is None


def test_the_criteria_and_the_round_plan_compose_into_the_closing_run():
    """End to end at the plan level: trip the budget, then get `\\n</think>`."""
    criteria = _criteria(2)
    for token in (10, 11, 12):
        criteria(token)
    budget, forced = round_emit_plan(
        0,
        lambda row: criteria.tokens_before_budget_stop(),
        lambda row: criteria.pending_forced_sequence(),
        BLOCK - 1,
        64,
    )
    assert forced == [NEWLINE_ID, END_THINK_ID]
    assert budget == 3


# ==========================================================================
# R4 -- the refusal, narrowed
# ==========================================================================
def _generator_with_draft_kind(kind):
    gen = server_generation.ResponseGenerator.__new__(
        server_generation.ResponseGenerator
    )
    gen.draft_model = object()
    gen.draft_kind = kind
    gen.processor = None
    gen._ready = Event()
    gen._ready.set()
    gen.wait_until_ready = lambda: None
    return gen


class TestThinkingBudgetRefusalNarrowing:
    @pytest.mark.parametrize("kind", ["eagle3", "lookup"])
    def test_the_legacy_loop_kinds_still_refuse(self, kind, monkeypatch):
        monkeypatch.delenv("MLX_VLM_DFLASH_CONTINUOUS_BATCHING", raising=False)
        gen = _generator_with_draft_kind(kind)
        args = server_generation.GenerationArguments(max_tokens=8, thinking_budget=4)
        with pytest.raises(ValueError, match="legacy speculative"):
            gen.generate("hi", args=args)

    def test_dflash_with_continuous_batching_off_still_refuses(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_DFLASH_CONTINUOUS_BATCHING", "0")
        gen = _generator_with_draft_kind("dflash")
        args = server_generation.GenerationArguments(max_tokens=8, thinking_budget=4)
        with pytest.raises(ValueError, match="CONTINUOUS_BATCHING=0"):
            gen.generate("hi", args=args)

    @pytest.mark.parametrize("kind", ["dflash", "mtp"])
    def test_the_continuous_batching_kinds_no_longer_refuse(self, kind, monkeypatch):
        monkeypatch.delenv("MLX_VLM_DFLASH_CONTINUOUS_BATCHING", raising=False)
        gen = _generator_with_draft_kind(kind)
        args = server_generation.GenerationArguments(max_tokens=8, thinking_budget=4)
        # The refusal is gone; the call now fails further in, on the fake
        # generator's missing preprocessing. Anything but the old ValueError.
        with pytest.raises(Exception) as excinfo:
            gen.generate("hi", args=args)
        assert "thinking_budget" not in str(excinfo.value)

    def test_the_structured_refusal_is_untouched(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_DFLASH_CONTINUOUS_BATCHING", raising=False)
        gen = _generator_with_draft_kind("dflash")
        args = server_generation.GenerationArguments(
            max_tokens=8, logits_processors=[lambda toks, logits: logits]
        )
        with pytest.raises(ValueError, match="Structured response_format"):
            gen.generate("hi", args=args)


# ==========================================================================
# R5 -- thinking is a property of the template, not a default
# ==========================================================================
NO_THINKING_KW_TEMPLATE = (
    "{%- set clear_thinking = clear_thinking if clear_thinking is defined "
    "else false %}{%- for m in messages %}{{ m.content }}"
    "{%- if not clear_thinking and m.reasoning_content %}"
    "{{ m.reasoning_content }}{%- endif %}{%- endfor %}"
)
THINKING_KW_TEMPLATE = (
    "{%- if enable_thinking %}<think>{%- endif %}"
    "{%- for m in messages %}{{ m.content }}{%- endfor %}"
)


class TestTemplateReferencesKw:
    def test_a_template_without_the_variable_is_detected(self):
        processor = SimpleNamespace(chat_template=NO_THINKING_KW_TEMPLATE)
        assert template_references_kw(processor, "enable_thinking") is False
        assert template_references_kw(processor, "clear_thinking") is True
        assert template_references_kw(processor, "reasoning_content") is True

    def test_a_template_with_the_variable_is_detected(self):
        processor = SimpleNamespace(chat_template=THINKING_KW_TEMPLATE)
        assert template_references_kw(processor, "enable_thinking") is True
        assert template_references_kw(processor, "clear_thinking") is False

    def test_the_tokenizer_template_counts_too(self):
        processor = SimpleNamespace(
            chat_template=None,
            tokenizer=SimpleNamespace(chat_template=THINKING_KW_TEMPLATE),
        )
        assert template_references_kw(processor, "enable_thinking") is True

    def test_a_dict_template_counts_too(self):
        processor = SimpleNamespace(
            chat_template={"default": THINKING_KW_TEMPLATE}
        )
        assert template_references_kw(processor, "enable_thinking") is True

    def test_an_override_wins_the_search(self):
        processor = SimpleNamespace(chat_template=NO_THINKING_KW_TEMPLATE)
        assert (
            template_references_kw(
                processor,
                "enable_thinking",
                chat_template_override=THINKING_KW_TEMPLATE,
            )
            is True
        )

    def test_no_template_at_all_references_nothing(self):
        assert template_references_kw(None, "enable_thinking") is False


class TestAlwaysOnThinking:
    def _gen(self, template):
        gen = server_generation.ResponseGenerator.__new__(
            server_generation.ResponseGenerator
        )
        gen.processor = SimpleNamespace(chat_template=template)
        return gen

    def test_a_template_without_the_variable_turns_thinking_on(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_ENABLE_THINKING", raising=False)
        gen = self._gen(NO_THINKING_KW_TEMPLATE)
        args = server_generation.GenerationArguments(max_tokens=8)
        assert args.enable_thinking is False
        gen._apply_always_on_thinking(args)
        assert args.enable_thinking is True, (
            "the template renders the same prompt either way, so a False here "
            "only disarms the budget criteria and the thinking-aware processor"
        )

    def test_a_template_with_the_variable_is_left_alone(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_ENABLE_THINKING", raising=False)
        gen = self._gen(THINKING_KW_TEMPLATE)
        args = server_generation.GenerationArguments(max_tokens=8)
        gen._apply_always_on_thinking(args)
        assert args.enable_thinking is False

    def test_an_explicit_request_choice_wins(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_ENABLE_THINKING", raising=False)
        gen = self._gen(NO_THINKING_KW_TEMPLATE)
        args = server_generation.GenerationArguments(
            max_tokens=8, enable_thinking=False, enable_thinking_explicit=True
        )
        gen._apply_always_on_thinking(args)
        assert args.enable_thinking is False

    def test_an_explicit_env_setting_wins(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_ENABLE_THINKING", "0")
        gen = self._gen(NO_THINKING_KW_TEMPLATE)
        args = server_generation.GenerationArguments(max_tokens=8)
        gen._apply_always_on_thinking(args)
        assert args.enable_thinking is False
        assert server_generation.server_enable_thinking_is_explicit() is True

    def test_no_processor_means_no_opinion(self):
        gen = server_generation.ResponseGenerator.__new__(
            server_generation.ResponseGenerator
        )
        gen.processor = None
        assert gen._thinking_always_on() is False

    def test_a_processor_with_no_template_means_no_opinion(self):
        gen = self._gen(None)
        assert gen._thinking_always_on() is False


class TestClearThinkingTemplateKwarg:
    def test_it_is_passed_when_the_template_reads_it(self):
        args = server_generation.GenerationArguments(max_tokens=8, clear_thinking=False)
        processor = SimpleNamespace(chat_template=NO_THINKING_KW_TEMPLATE)
        assert args.to_template_kwargs(processor)["clear_thinking"] is False

    def test_it_is_withheld_when_the_template_ignores_it(self):
        args = server_generation.GenerationArguments(max_tokens=8, clear_thinking=False)
        processor = SimpleNamespace(chat_template=THINKING_KW_TEMPLATE)
        assert "clear_thinking" not in args.to_template_kwargs(processor)

    def test_it_is_absent_when_nothing_set_it(self):
        args = server_generation.GenerationArguments(max_tokens=8)
        processor = SimpleNamespace(chat_template=NO_THINKING_KW_TEMPLATE)
        assert "clear_thinking" not in args.to_template_kwargs(processor)

    def test_a_true_request_value_survives(self):
        args = server_generation.GenerationArguments(max_tokens=8, clear_thinking=True)
        processor = SimpleNamespace(chat_template=NO_THINKING_KW_TEMPLATE)
        assert args.to_template_kwargs(processor)["clear_thinking"] is True

    @pytest.mark.parametrize(
        "raw,expected",
        [(None, False), ("0", False), ("1", True), ("true", True), ("no", False)],
    )
    def test_the_server_default_is_false(self, monkeypatch, raw, expected):
        if raw is None:
            monkeypatch.delenv("MLX_VLM_CLEAR_THINKING", raising=False)
        else:
            monkeypatch.setenv("MLX_VLM_CLEAR_THINKING", raw)
        assert server_generation.get_server_clear_thinking() is expected


# ==========================================================================
# R6 -- the receipt
# ==========================================================================
def test_budget_clamped_is_in_the_metrics_snapshot(monkeypatch):
    drafter = SimpleNamespace(
        speculative_total_rounds=10,
        speculative_total_batch_rounds=10,
        speculative_total_row_rounds=10,
        speculative_total_drafted=40,
        speculative_total_accepted=31.0,
        speculative_total_clamped=2,
        speculative_total_budget_clamped=7,
        speculative_total_per_row_kept=0,
    )
    monkeypatch.setattr(
        server_runtime,
        "response_generator",
        SimpleNamespace(draft_model=drafter, draft_kind="dflash"),
        raising=False,
    )
    snapshot = _speculative_stats_snapshot()
    assert snapshot["budget_clamped"] == 7
    assert snapshot["clamped"] == 2, (
        "the two give-backs are different: the batch's uniform clamp and the "
        "row's own emit budget"
    )


def test_budget_clamped_reads_zero_on_a_drafter_that_never_clamped(monkeypatch):
    monkeypatch.setattr(
        server_runtime,
        "response_generator",
        SimpleNamespace(draft_model=SimpleNamespace(), draft_kind="mtp"),
        raising=False,
    )
    assert _speculative_stats_snapshot()["budget_clamped"] == 0


# ==========================================================================
# Observability -- the GPU panel LV could not tell "budget never armed" from
# "budget armed and never tripped", because nothing on the server said either.
# ==========================================================================
class _PreopenedTokenizer:
    """A GLM-5.3-Flash-shaped tokenizer: <think> and </think> are single ids."""

    def encode(self, text, add_special_tokens=False):
        return {"\n": [198], "</think>": [154842], "<think>": [154841]}[text]


def _generator_with_template(template, processor=None):
    gen = server_generation.ResponseGenerator.__new__(
        server_generation.ResponseGenerator
    )
    gen.processor = processor or SimpleNamespace(chat_template=template)
    gen.tokenizer = _PreopenedTokenizer()
    return gen


class TestPromptPreopenedThinking:
    """The template's generation prompt ENDS in ``<think>``.

    The model therefore never emits a think-start token, so the criteria has to
    start INSIDE the block or it counts nothing for the whole response and the
    budget never forces a close -- which is exactly the shape of the LV panel's
    null result.
    """

    def _criteria(self, *, enable_thinking, budget=256):
        gen = _generator_with_template(NO_THINKING_KW_TEMPLATE)
        args = server_generation.GenerationArguments(
            max_tokens=1024,
            thinking_budget=budget,
            enable_thinking=enable_thinking,
        )
        # ...<|assistant|><think> -- the last id of the rendered prompt.
        input_ids = mx.array([[1, 2, 3, 154828, 154841]], dtype=mx.int32)
        return gen._make_thinking_budget_criteria(args, input_ids)

    def test_a_prompt_that_ends_in_think_starts_the_criteria_inside(self):
        criteria = self._criteria(enable_thinking=True)
        assert criteria.prompt_preopens_thinking is True
        assert criteria.in_thinking is True, (
            "if this is False the counter never increments: the generated "
            "stream contains no <think> to switch it on"
        )

    def test_it_forces_the_closing_run_at_budget_plus_one(self):
        criteria = self._criteria(enable_thinking=True, budget=256)
        forced = []
        for _ in range(300):
            criteria(9999)
            token = criteria.pop_forced_token_id()
            if token is not None:
                forced.append(token)
        assert forced[:2] == [198, 154842], (
            "\\n then </think>, starting at the 257th generated token"
        )

    def test_thinking_off_disarms_it_entirely(self):
        criteria = self._criteria(enable_thinking=False)
        assert criteria.in_thinking is False
        for _ in range(300):
            criteria(9999)
            assert criteria.pop_forced_token_id() is None, (
                "MLX_VLM_ENABLE_THINKING=0 is an operator saying off, and it "
                "must keep disarming the budget"
            )


class TestThinkingConfigSnapshot:
    def _snapshot(self, generator):
        server_runtime.response_generator = generator
        try:
            return _thinking_config_snapshot()
        finally:
            server_runtime.response_generator = None

    def test_it_reports_the_budget_the_env_actually_set(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_THINKING_BUDGET", "256")
        monkeypatch.delenv("MLX_VLM_ENABLE_THINKING", raising=False)
        snap = self._snapshot(_generator_with_template(NO_THINKING_KW_TEMPLATE))
        assert snap["server_budget"] == 256
        assert snap["enable_thinking_env_set"] is False
        assert snap["template_references_enable_thinking"] is False
        assert snap["always_on"] is True

    def test_an_unset_budget_reads_none_not_zero(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_THINKING_BUDGET", raising=False)
        snap = self._snapshot(_generator_with_template(NO_THINKING_KW_TEMPLATE))
        assert snap["server_budget"] is None, (
            "None and 0 are different arms and the receipt must say which"
        )

    def test_env_off_is_distinguishable_from_env_unset(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_ENABLE_THINKING", "0")
        snap = self._snapshot(_generator_with_template(NO_THINKING_KW_TEMPLATE))
        assert snap["server_enable_thinking"] is False
        assert snap["enable_thinking_env_set"] is True, (
            "the whole point: 0 and unset both read False, and only this field "
            "separates 'operator said off' from 'the template decides'"
        )

    def test_a_template_that_owns_the_variable_is_not_always_on(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_ENABLE_THINKING", raising=False)
        snap = self._snapshot(_generator_with_template(THINKING_KW_TEMPLATE))
        assert snap["template_references_enable_thinking"] is True
        assert snap["always_on"] is False

    def test_it_survives_a_server_with_no_generator(self):
        server_runtime.response_generator = None
        snap = _thinking_config_snapshot()
        assert snap["always_on"] is None
        assert snap["template_references_enable_thinking"] is None

    def test_clear_thinking_is_reported_both_ways(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_CLEAR_THINKING", "1")
        snap = self._snapshot(_generator_with_template(NO_THINKING_KW_TEMPLATE))
        assert snap["server_clear_thinking"] is True
        assert snap["template_references_clear_thinking"] is True


# ==========================================================================
# LV2 root cause: the CLI made MLX_VLM_ENABLE_THINKING permanently "set"
# ==========================================================================
class TestCliDoesNotFakeAnExplicitThinkingMode:
    """``--enable-thinking`` absent must leave the env var absent.

    ``action="store_true"`` with a ``False`` default and an UNCONDITIONAL
    ``os.environ[...] = "0"`` write meant every server ever started reported
    ``server_enable_thinking_is_explicit() == True`` -- an operator decision
    nobody made.  Always-on thinking then short-circuited on its "an operator
    said off" early return, and the LV2 panel logged
    ``enabled=False always_on_template=True`` for every arm.
    """

    def _run_cli(self, monkeypatch, extra_argv):
        """Run the real ``main()`` over a THROWAWAY environment.

        ``main()`` writes a dozen preload/env keys; letting them escape into the
        process environment made a later test try to fetch a model named "demo"
        from the Hub. Swapping ``os.environ`` for a plain dict keeps every write
        inside this test while ``os.environ.get`` -- which is how the server
        reads all of them -- still sees it.
        """
        import sys as _sys

        import mlx_vlm.server.cli as server_cli

        sandbox = dict(os.environ)
        for name in ("MLX_VLM_ENABLE_THINKING", "MLX_VLM_THINKING_BUDGET"):
            sandbox.pop(name, None)
        monkeypatch.setattr(os, "environ", sandbox)
        monkeypatch.setattr(
            _sys, "argv", ["mlx_vlm.server", "--model", "demo", *extra_argv]
        )
        monkeypatch.setattr(server_cli.uvicorn, "run", lambda *a, **k: None)
        server_cli.main()
        return sandbox

    def test_without_the_flag_the_env_var_stays_unset(self, monkeypatch):
        env = self._run_cli(monkeypatch, [])
        assert "MLX_VLM_ENABLE_THINKING" not in env, (
            "an unwritten env var is what lets the template decide; writing "
            "'0' here is the server telling itself the operator said off"
        )
        assert server_generation.server_enable_thinking_is_explicit() is False
        assert server_generation.get_server_enable_thinking() is False, (
            "the effective default is unchanged -- only its explicitness is"
        )

    def test_with_the_flag_it_is_set_on(self, monkeypatch):
        env = self._run_cli(monkeypatch, ["--enable-thinking"])
        assert env["MLX_VLM_ENABLE_THINKING"] == "1"
        assert server_generation.server_enable_thinking_is_explicit() is True

    def test_an_inherited_value_survives_a_launch_without_the_flag(
        self, monkeypatch
    ):
        import sys as _sys

        import mlx_vlm.server.cli as server_cli

        sandbox = dict(os.environ)
        sandbox["MLX_VLM_ENABLE_THINKING"] = "1"
        monkeypatch.setattr(os, "environ", sandbox)
        monkeypatch.setattr(_sys, "argv", ["mlx_vlm.server", "--model", "demo"])
        monkeypatch.setattr(server_cli.uvicorn, "run", lambda *a, **k: None)
        server_cli.main()
        assert sandbox["MLX_VLM_ENABLE_THINKING"] == "1", (
            "the launcher's own export must not be clobbered by an argparse "
            "default the operator never typed"
        )


# ==========================================================================
# End to end: a chat.completions body with NO enable_thinking field
# ==========================================================================
GLM_SHAPED_TEMPLATE = (
    "{%- set clear_thinking = clear_thinking if clear_thinking is defined "
    "else false %}{%- for m in messages %}<|user|>{{ m.content }}"
    "{%- if not clear_thinking and m.reasoning_content is defined %}"
    "{{ m.reasoning_content }}{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}<|assistant|><think>{%- endif %}"
)


def _glm_shaped_generator():
    """A ResponseGenerator carrying only what the thinking path reads.

    Mirrors the served build: the chat template has no ``enable_thinking``
    variable and its generation prompt ends in ``<think>``, and the tokenizer
    resolves both think markers to single ids.
    """
    gen = server_generation.ResponseGenerator.__new__(
        server_generation.ResponseGenerator
    )
    gen.processor = SimpleNamespace(chat_template=GLM_SHAPED_TEMPLATE, config=None)
    gen.tokenizer = _PreopenedTokenizer()
    return gen


def _resolve_like_the_server(request, generator):
    """``_build_gen_args`` -> ``_apply_always_on_thinking`` -> criteria.

    The exact order ``ResponseGenerator.generate`` runs them in, minus the
    tokenizer/model work: normalization on the request thread, the always-on
    decision at the top of ``generate``, then the criteria built from the
    prompt's own ids.
    """
    from mlx_vlm.server import request_normalization as rn

    args = rn._build_gen_args(request, generator.processor)
    generator._apply_always_on_thinking(args)
    # ...<|assistant|><think>: the last id of a prompt this template rendered.
    input_ids = mx.array([[7, 8, 9, 154841]], dtype=mx.int32)
    criteria = generator._make_thinking_budget_criteria(args, input_ids)
    return args, criteria


class TestChatCompletionsBodyWithoutEnableThinking:
    def _request(self, **extra):
        from mlx_vlm.server.schemas import ChatRequest

        return ChatRequest(
            model="demo",
            messages=[{"role": "user", "content": "spec"}],
            stream=True,
            max_tokens=1024,
            temperature=0.0,
            **extra,
        )

    def test_a_body_without_the_field_does_not_count_as_explicit(
        self, monkeypatch
    ):
        monkeypatch.delenv("MLX_VLM_ENABLE_THINKING", raising=False)
        monkeypatch.delenv("MLX_VLM_THINKING_BUDGET", raising=False)
        request = self._request()
        assert "enable_thinking" not in request.model_fields_set, (
            "a pydantic Optional default must not land in model_fields_set"
        )
        args, _ = _resolve_like_the_server(request, _glm_shaped_generator())
        assert args.enable_thinking_explicit is False
        assert args.enable_thinking is True

    def test_the_budget_arm_resolves_armed_and_preopened(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_ENABLE_THINKING", raising=False)
        monkeypatch.delenv("MLX_VLM_THINKING_BUDGET", raising=False)
        generator = _glm_shaped_generator()
        args, criteria = _resolve_like_the_server(
            self._request(thinking_budget=256), generator
        )
        # The three fields the server's "Thinking resolved:" line reports.
        assert args.enable_thinking is True, "enabled"
        assert generator._thinking_always_on() is True, "always_on_template"
        assert args.thinking_budget == 256, "budget"
        assert criteria is not None and criteria.in_thinking is True, "preopened"

    def test_it_then_forces_the_close_at_budget_plus_one(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_ENABLE_THINKING", raising=False)
        monkeypatch.delenv("MLX_VLM_THINKING_BUDGET", raising=False)
        _, criteria = _resolve_like_the_server(
            self._request(thinking_budget=256), _glm_shaped_generator()
        )
        forced = []
        for _ in range(300):
            criteria(9999)
            token = criteria.pop_forced_token_id()
            if token is not None:
                forced.append(token)
        assert forced[:2] == [198, 154842]

    def test_an_operator_who_says_off_is_still_obeyed(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_ENABLE_THINKING", "0")
        args, criteria = _resolve_like_the_server(
            self._request(thinking_budget=256), _glm_shaped_generator()
        )
        assert args.enable_thinking is False
        assert criteria.in_thinking is False

    def test_a_body_that_says_off_is_still_obeyed(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_ENABLE_THINKING", raising=False)
        args, criteria = _resolve_like_the_server(
            self._request(thinking_budget=256, enable_thinking=False),
            _glm_shaped_generator(),
        )
        assert args.enable_thinking_explicit is True
        assert args.enable_thinking is False
        assert criteria.in_thinking is False
