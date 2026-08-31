"""Unit tests for the prompt-lookup matcher and drafter policy."""

import mlx.core as mx
import pytest

from mlx_vlm.speculative.drafters.prompt_lookup import (
    PromptLookupConfig,
    PromptLookupDraftModel,
)
from mlx_vlm.speculative.drafters.prompt_lookup import ngram as ngram_mod
from mlx_vlm.speculative.drafters.prompt_lookup.ngram import NgramIndex


# ------------------------------------------------------------------ matcher
def test_proposes_the_continuation_of_the_matched_span():
    ix = NgramIndex(n_min=3, n_max=5)
    ix.extend([1, 2, 3, 4, 5, 6, 1, 2, 3])
    tokens, n, pos = ix.propose(3)
    assert tokens == [4, 5, 6]
    assert n == 3 and pos == 2


def test_abstains_without_a_match():
    ix = NgramIndex(n_min=3, n_max=5)
    ix.extend([1, 2, 3, 4, 5, 6, 7, 8])
    assert ix.propose(4) == ([], 0, -1)


def test_abstains_below_n_min():
    ix = NgramIndex(n_min=3, n_max=5)
    ix.extend([7, 7])
    assert ix.propose(4)[0] == []


def test_prefers_the_longest_n():
    # suffix [8,9,1,2,3]: the 5-gram occurs once earlier (continuation 40),
    # the 3-gram [1,2,3] also occurs at an older spot (continuation 99).
    ix = NgramIndex(n_min=3, n_max=5)
    ix.extend([1, 2, 3, 99, 0, 8, 9, 1, 2, 3, 40, 41, 0, 8, 9, 1, 2, 3])
    tokens, n, _ = ix.propose(2)
    assert n == 5 and tokens == [40, 41]


def test_skips_the_suffixs_own_occurrence():
    """The n-gram ending at the current last token is in the index too; using it
    would propose the token that follows the cursor, which does not exist."""
    ix = NgramIndex(n_min=3, n_max=3)
    ix.extend([5, 6, 7])
    assert ix.propose(2)[0] == []       # only occurrence is the suffix itself


def test_takes_the_most_recent_earlier_occurrence():
    ix = NgramIndex(n_min=3, n_max=3, keep=2)
    ix.extend([1, 2, 3, 10, 1, 2, 3, 20, 1, 2, 3])
    tokens, _, pos = ix.propose(1)
    assert tokens == [20] and pos == 6   # the second [1,2,3], not the first


def test_continuation_runs_past_the_matched_span():
    """The proposal is 'what followed last time', which deliberately keeps
    running past the end of the matched n-gram -- that is what lets a drafter
    mid-quotation copy a long span, not just n tokens of it."""
    ix = NgramIndex(n_min=3, n_max=3)
    ix.extend([1, 2, 3, 4, 1, 2, 3])
    assert ix.propose(8)[0] == [4, 1, 2, 3]


def test_continuation_is_truncated_at_the_context_end():
    ix = NgramIndex(n_min=3, n_max=3)
    ix.extend([1, 2, 3, 4, 5, 1, 2, 3])
    assert ix.propose(2)[0] == [4, 5]
    assert len(ix.propose(99)[0]) == 5      # everything after position 2


def test_incremental_append_equals_bulk_build():
    seq = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 3, 1, 4, 1, 5]
    a = NgramIndex(n_min=3, n_max=5)
    a.extend(seq)
    b = NgramIndex(n_min=3, n_max=5)
    for t in seq:
        b.append(t)
    assert a.propose(4) == b.propose(4)
    assert a.tokens == b.tokens


def test_hash_collisions_cannot_produce_a_wrong_draft(monkeypatch):
    """Keys are rolling hashes, so collisions are possible; every hit is
    re-verified against the real tokens.  Force every key to collide and the
    matcher must still only ever propose after a genuine match."""
    monkeypatch.setattr(ngram_mod, "_hash", lambda tokens, start, n: 0)
    # keep must span the whole context here: with every key colliding, one deque
    # holds every position, and the genuine match would otherwise be evicted.
    ix = NgramIndex(n_min=3, n_max=3, keep=64)
    ix.extend([1, 2, 3, 40, 7, 8, 9, 50, 1, 2, 3])
    tokens, n, pos = ix.propose(1)
    assert tokens == [40] and pos == 2     # not 50, which only collided
    ix2 = NgramIndex(n_min=3, n_max=3, keep=64)
    ix2.extend([1, 2, 3, 40, 7, 8, 9])
    assert ix2.propose(1)[0] == []         # all-colliding but nothing matches


def test_reset_clears_everything():
    ix = NgramIndex(n_min=3, n_max=3)
    ix.extend([1, 2, 3, 4, 1, 2, 3])
    ix.reset()
    assert len(ix) == 0 and ix.propose(2)[0] == []


@pytest.mark.parametrize("n_min,n_max,keep", [(0, 3, 2), (4, 3, 2), (3, 5, 1)])
def test_rejects_bad_parameters(n_min, n_max, keep):
    with pytest.raises(ValueError):
        NgramIndex(n_min=n_min, n_max=n_max, keep=keep)


# ------------------------------------------------------------------ drafter
def _drafter(**kw):
    cfg = PromptLookupConfig(**{"n_min": 3, "n_max": 5, "block_size": 8, **kw})
    return PromptLookupDraftModel(cfg)


def test_draft_block_abstains_as_an_empty_proposal():
    d = _drafter()
    d.set_context([1, 2, 3, 4, 5, 6, 7])
    out = d.draft_block(7, None, None, 8, None, mx.int32)
    assert out.shape == (1, 0)
    assert d.abstentions == 1


def test_draft_block_proposes_the_match():
    d = _drafter(adaptive=False)
    d.set_context([1, 2, 3, 4, 5, 6, 1, 2, 3])
    out = d.draft_block(3, None, None, 8, None, mx.int32)
    # block_size 8 -> up to 7 proposed; the continuation of position 2 runs on
    # past the matched span, which is the point.
    assert out.tolist() == [[4, 5, 6, 1, 2, 3]]


def test_observe_extends_the_index():
    d = _drafter(adaptive=False)
    d.set_context([1, 2, 3, 4, 5, 6])
    assert d.draft_block(6, None, None, 8, None, mx.int32).shape == (1, 0)
    d.observe([1, 2, 3])
    assert d.draft_block(3, None, None, 8, None, mx.int32).tolist() == [
        [4, 5, 6, 1, 2, 3]
    ]


def test_reset_clears_context_and_stats():
    d = _drafter()
    d.set_context([1, 2, 3, 4, 1, 2, 3])
    d.draft_block(3, None, None, 8, None, mx.int32)
    d.reset(None)
    assert d.abstentions == 0 and d.match_lens == [] and len(d.index) == 0


def test_adaptive_gate_shrinks_then_recovers():
    """A workload that stops quoting must collapse the verify width, and must
    still be able to widen again -- that is what explore_tokens buys."""
    d = _drafter(adaptive=True, adaptive_window=2, explore_tokens=1)
    assert d._budget(8) == 7                     # no history yet: full width
    for _ in range(12):
        d.note_round(0)
    assert d._budget(8) == 1                     # collapsed to the explore floor
    for _ in range(12):
        d.note_round(7)
    assert d._budget(8) == 7                     # recovered


def test_adaptive_off_always_uses_full_width():
    d = _drafter(adaptive=False)
    for _ in range(12):
        d.note_round(0)
    assert d._budget(8) == 7


def test_budget_never_exceeds_the_block():
    d = _drafter(adaptive=True, adaptive_window=2)
    for _ in range(12):
        d.note_round(100)
    assert d._budget(4) == 3
    assert d._budget(2) == 1


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("MLX_VLM_LOOKUP_NMIN", "2")
    monkeypatch.setenv("MLX_VLM_LOOKUP_NMAX", "6")
    monkeypatch.setenv("MLX_VLM_LOOKUP_ADAPTIVE", "0")
    cfg = PromptLookupConfig.from_env()
    assert (cfg.n_min, cfg.n_max, cfg.adaptive) == (2, 6, False)


def test_drafter_is_greedy_only():
    assert PromptLookupDraftModel.requires_greedy_sampling is True
