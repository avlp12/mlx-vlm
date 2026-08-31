"""Prompt-lookup speculative drafter: an n-gram copy of the live context.

Motivation.  On the warm-context-vault workload -- long cached documents,
doc-grounded answering -- the model spends much of its output quoting or
lightly paraphrasing spans that are already in its own context.  Wherever it
does, the next few tokens are literally sitting in the prompt, and a drafter
that simply finds them costs no GPU work at all.  This is the decode-side twin
of the vault: the vault removes the prefill of a repeated document, this removes
the decode of a repeated span.

What it is not.  It has no weights, no forward pass and no probability
distribution, so it cannot participate in the non-greedy rejection-sampling
correction -- there is nothing to correct against.  It is greedy-only, and under
greedy the speculative contract still holds exactly: the verifier accepts a
drafted token only where it equals the target's own argmax, so the emitted
stream is bit-identical to plain greedy decoding.

Economics.  A DFlash2 round pays ~6 ms of drafter encode plus ~4 ms of drafter
GPU before the target ever runs.  A lookup round pays neither.  Its only cost is
that the verify forward runs at L = K+1 instead of L = 1, which on a 288-expert
MoE is not free.  So the failure mode is not "slow drafter", it is "wasted
verify width on a workload that does not quote", and the defence is to make the
proposal length track the measured accept rate -- see ``_budget``.
"""

import math
from typing import Any, Callable, List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn

from .config import PromptLookupConfig
from .ngram import NgramIndex


class PromptLookupDraftModel(nn.Module):
    """Weight-free drafter matching the mlx-vlm speculative drafter interface."""

    # No draft distribution exists, so only the greedy walk is sound.
    requires_greedy_sampling = True
    # Nothing to capture from the target: the round loop passes hidden=None.
    needs_target_hidden = False

    def __init__(self, config: Optional[PromptLookupConfig] = None):
        super().__init__()
        self.config = config or PromptLookupConfig()
        self.config.validate()
        self.index = NgramIndex(
            n_min=self.config.n_min,
            n_max=self.config.n_max,
            keep=self.config.keep,
        )
        self.accept_lens: List[float] = []
        self.draft_lens: List[int] = []
        # Match bookkeeping, reported alongside accept_lens so a receipt can
        # separate "no match found" from "matched but rejected".
        self.match_lens: List[int] = []
        self.abstentions = 0
        self._ema: Optional[float] = None
        self._last_proposed = 0

    # ------------------------------------------------------------- lifecycle
    def reset(self, target_model: Any = None) -> None:
        """Mirrors the drafter interface.  There is no KV cache to build, so the
        returned cache is None and the round loop must not index it."""
        del target_model
        self.index.reset()
        self.accept_lens = []
        self.draft_lens = []
        self.match_lens = []
        self.abstentions = 0
        self._ema = None
        self._last_proposed = 0
        return None

    def validate_target_compatibility(self, target_model: Any) -> None:
        # Any target works: the drafter reads token ids, not hidden states.
        del target_model

    def bind(self, target_model: Any) -> "PromptLookupDraftModel":
        del target_model
        return self

    def set_context(self, tokens: Sequence[int]) -> None:
        """Seed the index with the prompt.  O(len(tokens))."""
        self.index.reset()
        self.index.extend(tokens)

    def observe(self, tokens: Sequence[int]) -> None:
        """Commit accepted tokens.  Only committed tokens are indexed, so a
        rejected speculative tail never needs to be rolled back out."""
        for t in tokens:
            self.index.append(int(t))

    # ---------------------------------------------------------------- policy
    def _budget(self, block_size: int) -> int:
        """How many tokens this round may propose *if the matcher finds a match*.

        Abstention is not this function's job.  Whether a match exists is known
        for free, before any GPU work, and a round with no match verifies a
        single token and is therefore exactly a plain decode step and exactly
        free.  So the drafter is already self-limiting on workloads that do not
        quote, and this only has to answer the second question: when I do match,
        how far into the match is it still worth proposing?

        That question has a marginal answer.  One extra proposed token widens the
        verify by ``verify_cost_per_token`` decode-steps and buys one token if it
        survives, so it is worth proposing while P(survive) exceeds that cost.
        Modelling the accepted run as geometric with mean ``e`` (the EMA over
        proposing rounds) gives per-token survival p = e/(1+e), so the last
        worthwhile position is ``log(cost)/log(p)``.

        Two earlier revisions were wrong in opposite directions and both were
        caught by the quote sweep rather than by reasoning:
          * gating *abstention* on the same EMA locked the gate at zero -- a
            collapsed gate proposes nothing, so nothing is accepted, so it never
            recovers -- and dropped pure quoting from 2.30x to 1.08x;
          * a flat ``int(e+1)`` ceiling was far too timid, forfeiting about half
            the available win on verbatim quoting (2.30x -> 1.81x).
        """
        hard = max(1, int(block_size) - 1)
        if not self.config.adaptive or self._ema is None:
            return hard
        e = max(0.0, float(self._ema))
        p = e / (1.0 + e)                       # per-token survival
        if p <= 0.0:
            return 1                            # a found match is evidence
        cost = self.config.verify_cost_per_token
        if p >= 1.0:
            return hard
        return max(1, min(hard, int(math.log(cost) / math.log(p))))

    def _record(self, accepted: float) -> None:
        w = max(1, int(self.config.adaptive_window))
        alpha = 2.0 / (w + 1.0)
        self._ema = (
            float(accepted)
            if self._ema is None
            else (1 - alpha) * self._ema + alpha * float(accepted)
        )

    def note_round(self, accepted: float) -> None:
        """Called by the round loop after the walk.  Only rounds that actually
        proposed inform the width gate; an abstention says nothing about how
        good this drafter's matches are, and folding it in is what made the
        earlier revision lock at zero."""
        if self._last_proposed > 0:
            self._record(accepted)

    # ----------------------------------------------------------------- draft
    def draft_block(
        self,
        last_bonus: Any,
        hidden: Any = None,
        cache: Any = None,
        block_size: int = 8,
        sampler: Optional[Callable] = None,
        token_dtype: mx.Dtype = mx.int32,
        **kwargs,
    ) -> mx.array:
        """Propose the continuation of the longest context match.

        ``hidden``, ``cache`` and ``sampler`` are accepted and ignored so the
        signature matches the other drafters; an empty [1, 0] result is a
        deliberate abstention and the round loop degenerates to a plain decode
        step.
        """
        del hidden, cache, sampler, kwargs
        if not isinstance(last_bonus, int):
            last_bonus = int(mx.array(last_bonus).reshape(-1)[0].item())
        k = self._budget(block_size)
        tokens, n_matched, _pos = self.index.propose(k)
        self._last_proposed = len(tokens)
        self.match_lens.append(len(tokens))
        if not tokens:
            self.abstentions += 1
            return mx.zeros((1, 0), dtype=token_dtype)
        return mx.array([tokens], dtype=token_dtype)

    def stats(self) -> dict:
        rounds = len(self.accept_lens)
        return {
            "rounds": rounds,
            "abstentions": self.abstentions,
            "match_rate": (rounds - self.abstentions) / rounds if rounds else 0.0,
            "accept_per_round": (
                sum(self.accept_lens) / rounds if rounds else 0.0
            ),
            "draft_per_round": (
                sum(self.draft_lens) / len(self.draft_lens) if self.draft_lens else 0.0
            ),
            "accept_lens": list(self.accept_lens),
            "draft_lens": list(self.draft_lens),
            "match_lens": list(self.match_lens),
            "ema_accept": self._ema,
        }


Model = PromptLookupDraftModel

__all__ = ["PromptLookupDraftModel", "Model"]
