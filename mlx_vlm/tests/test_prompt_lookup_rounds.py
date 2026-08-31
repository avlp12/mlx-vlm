"""Greedy identity for the prompt-lookup speculative round loop.

The speculative contract is that the emitted stream is *exactly* what plain
greedy decoding would have produced -- the drafter may only change how fast the
tokens arrive, never which tokens arrive.  For a lookup drafter that contract is
the whole safety story, because the drafter itself is unprincipled: it copies
text without any model of whether the copy is right.

These tests drive ``_lookup_rounds`` against a scripted first-order target whose
greedy continuation is known in closed form, so the reference is exact and the
accept / reject / abstain paths can all be forced deliberately:

  * the prompt contains a span that *starts* like the generated text and then
    diverges, which makes the drafter propose confidently and be wrong -- the
    partial-accept + rollback path;
  * the target's own cycle eventually re-enters the index, which makes the
    drafter right -- the full-accept path;
  * early rounds have no match at all -- the abstain path, where the round must
    degenerate into a plain single-token decode instead of ending generation.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.speculative.drafters.prompt_lookup import (
    PromptLookupConfig,
    PromptLookupDraftModel,
)
from mlx_vlm.speculative.lookup import _lookup_rounds

VOCAB = 128
# A 6-cycle: greedy from 10 is 11,12,13,14,15,10,11,...
NEXT = {10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 10}
DEFAULT = 7
NEXT.setdefault(DEFAULT, DEFAULT)
CYCLE = 6           # 10 -> 11 -> ... -> 15 -> 10; the self-loop on 7 is not part of it


def _next(tok: int) -> int:
    return NEXT.get(int(tok), DEFAULT)


def _greedy_reference(first_bonus: int, n: int):
    out, t = [], int(first_bonus)
    for _ in range(n):
        t = _next(t)
        out.append(t)
    return out


class _StubCache:
    def __init__(self):
        self.offset = 0


class _StubLM:
    """First-order greedy target with a KV cache offset and a real rollback."""

    def __init__(self):
        self.cache = [_StubCache()]
        self.verify_widths = []
        self.rollbacks = []
        self.capture_args = []

    def __call__(self, inputs, cache=None, capture_layer_ids=None,
                 speculative_verify=False, **kwargs):
        self.capture_args.append(capture_layer_ids)
        row = inputs.reshape(-1).tolist()
        self.verify_widths.append(len(row))
        cache[0].offset += len(row)
        logits = mx.zeros((1, len(row), VOCAB))
        onehot = mx.zeros((len(row), VOCAB))
        idx = mx.array([[_next(t)] for t in row], dtype=mx.int32)
        onehot = mx.put_along_axis(onehot, idx, mx.array(1.0), axis=-1)
        return SimpleNamespace(
            logits=onehot[None],
            gdn_states=[],
            hidden_states=None,
        )

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        trim = block_size - (int(accepted) + 1)
        self.rollbacks.append((int(accepted), int(block_size), trim))
        caches[0].offset -= trim
        return trim


def _sampler(logits):
    return mx.argmax(logits, axis=-1)


def _drive(prompt, first_bonus, max_tokens, **cfg_kw):
    lm = _StubLM()
    model = SimpleNamespace(language_model=lm)
    cfg = PromptLookupConfig(**{"n_min": 3, "n_max": 5, "block_size": 8, **cfg_kw})
    drafter = PromptLookupDraftModel(cfg)
    prompt_cache = lm.cache
    prompt_cache[0].offset = len(prompt)
    out = [
        tok
        for tok, _ in _lookup_rounds(
            model,
            drafter,
            prompt_cache,
            prompt_tokens=mx.array([prompt], dtype=mx.int32),
            first_bonus=first_bonus,
            max_tokens=max_tokens,
            sampler=_sampler,
            greedy_sampling=True,
        )
    ]
    return out, drafter, lm


# --------------------------------------------------------------- the contract
@pytest.mark.parametrize("max_tokens", [8, 32, 64])
def test_output_is_exactly_greedy(max_tokens):
    # The prompt starts like the generated text and then diverges, so the
    # drafter is confidently wrong for a while.
    prompt = [10, 11, 12, 99, 98, 97, 96, 7, 7, 7]
    out, drafter, _ = _drive(prompt, 10, max_tokens)
    # _lookup_rounds emits max_tokens-1: the caller already yielded the bonus.
    assert out == _greedy_reference(10, max_tokens - 1)
    assert len(drafter.accept_lens) > 0


def test_output_is_exactly_greedy_without_any_match():
    """A prompt sharing nothing with the output: every round abstains and the
    loop must degenerate into plain decoding rather than stopping."""
    prompt = [50, 51, 52, 53, 54, 55]
    out, drafter, lm = _drive(prompt, 10, 12, n_min=5, n_max=5)
    assert out == _greedy_reference(10, 11)
    assert drafter.abstentions > 0
    assert 1 in lm.verify_widths          # at least one plain decode-width round


def test_all_three_paths_are_exercised():
    prompt = [10, 11, 12, 99, 98, 97, 96, 7, 7, 7]
    out, drafter, lm = _drive(prompt, 10, 48)
    accepts = drafter.accept_lens
    assert any(a == 0 for a in accepts), "no full-rejection round"
    assert any(a > 0 for a in accepts), "no accepting round"
    assert drafter.abstentions > 0, "no abstention"
    assert lm.rollbacks, "rollback never exercised"
    assert out == _greedy_reference(10, 47)


def test_rollback_trims_the_cache_consistently():
    prompt = [10, 11, 12, 99, 98, 97, 96, 7, 7, 7]
    out, _, lm = _drive(prompt, 10, 40)
    # every emitted token, plus the bonus that seeded the round loop, must be
    # exactly what the cache holds beyond the prompt
    assert lm.cache[0].offset == len(prompt) + 1 + len(out)


def test_verify_always_passes_an_empty_capture_list():
    """None would leave gdn_states unallocated and silently break rollback."""
    prompt = [10, 11, 12, 99, 98, 97, 96, 7, 7, 7]
    _drive(prompt, 10, 16)
    _, _, lm = _drive(prompt, 10, 16)
    assert lm.capture_args and all(c == [] for c in lm.capture_args)


def test_abstention_costs_exactly_one_verify_position():
    prompt = [50, 51, 52, 53, 54, 55]
    _, drafter, lm = _drive(prompt, 10, 6, n_min=5, n_max=5)
    # every round abstained, so every verify was a single token
    assert set(lm.verify_widths) == {1}
    assert not lm.rollbacks


def test_quoting_workload_accepts_long_runs():
    """The win case: the generated text re-enters a span already in context.

    The prompt has to be a *coherent* continuation into the first bonus -- an
    earlier draft of this test ended the prompt on the same token the bonus
    then repeated, which planted a two-token artifact that appears nowhere else
    in the context and cost two rounds of abstention before matching resumed.
    That is real behaviour (the drafter copies context artifacts and the
    verifier catches them, which ``test_all_three_paths_are_exercised`` covers),
    but it is not the win case.
    """
    prompt = [10] + _greedy_reference(10, 23)      # ends on 15
    bonus = _next(prompt[-1])                      # == 10, a natural next token
    out, drafter, lm = _drive(prompt, bonus, 24, adaptive=False)
    assert out == _greedy_reference(bonus, 23)
    # The proposal is bounded by the *distance back to the match*: the copy runs
    # from pos+1 and cannot run past the end of the context, so a period-P
    # repetition caps a draft at P tokens however large the block is.  Here the
    # target cycles with period 6, so 6 -- not block_size-1 = 7 -- is saturation.
    # Real document quoting matches far back, so this bound does not bind there.
    assert max(drafter.accept_lens) == CYCLE
    mean_accept = sum(drafter.accept_lens) / len(drafter.accept_lens)
    assert mean_accept >= 4, f"mean accept {mean_accept} too low on pure quoting"
    assert not lm.rollbacks, "a perfectly quoted span needs no rollback"


def test_draft_length_is_bounded_by_distance_to_the_match():
    """Pin the bound directly: a far-away match can fill the whole block."""
    tail = _greedy_reference(10, 23)
    filler = [7] * 40                     # 7 -> 7, so the run is self-consistent
    prompt = [10] + tail + filler
    bonus = _next(prompt[-1])
    _, drafter, _ = _drive(prompt, bonus, 16, adaptive=False)
    # the filler run matches itself one token back, so the copy is capped at 1
    assert max(drafter.accept_lens) <= 1


def test_non_greedy_is_refused():
    lm = _StubLM()
    model = SimpleNamespace(language_model=lm)
    drafter = PromptLookupDraftModel(PromptLookupConfig())
    with pytest.raises(ValueError, match="greedy-only"):
        list(
            _lookup_rounds(
                model, drafter, lm.cache, prompt_tokens=None, first_bonus=10,
                max_tokens=4, sampler=_sampler, greedy_sampling=False,
            )
        )


def test_target_without_rollback_is_refused():
    lm = SimpleNamespace(__call__=None)
    model = SimpleNamespace(language_model=SimpleNamespace())
    drafter = PromptLookupDraftModel(PromptLookupConfig())
    with pytest.raises(RuntimeError, match="rollback_speculative_cache"):
        list(
            _lookup_rounds(
                model, drafter, [], prompt_tokens=None, first_bonus=10,
                max_tokens=4, sampler=_sampler, greedy_sampling=True,
            )
        )


def test_never_drafts_past_the_token_budget():
    """Tokens drafted beyond max_tokens are verify width spent on output that is
    thrown away."""
    cycle = _greedy_reference(10, 24)
    _, _, lm = _drive([10] + cycle, 10, 4, adaptive=False)
    assert max(lm.verify_widths) <= 4
