"""Sampled-mode speculative coupling (F1/F2).

The defect
----------
Under sampling (the GLM-5.3-Flash vendor setting is ``temperature=1``,
``top_p=0.95``) this fork drew the DFlash2 draft proposal and the target sample
from *independent* RNG streams -- the target keyed
``_position_keys(seed, row_ids, positions)`` while the proposal keyed the same
stream XOR ``0x0DFA5202`` -- and then accepted a drafted token only on exact
token equality with the target's own draw.

That is unbiased (the emitted token is always the target's draw) but the
acceptance rate is the *collision probability*

    alpha_indep = sum_t p_t q_t  <=  max_t p_t  <=  exp(-H_2(p))

instead of the maximal coupling

    alpha_max   = sum_t min(p_t, q_t).

Any position with Renyi-2 entropy of one nat or more is capped at
alpha <= 0.37 no matter how good the drafter is, so a *sampled* proposal is
provably never better than a greedy one under this acceptance rule.

The two fixes
-------------
F1 ``MLX_VLM_DFLASH_SAMPLED_DRAFT=argmax`` -- draft the argmax, exactly as MTP
already does under sampling (``mtp.py:971``).  A point-mass proposal
q = delta_{t*} makes sum_t min(p_t, q_t) = p_{t*} = max_t p_t, which is the
maximal coupling *for a deterministic proposal* and dominates the collision
probability of any independently sampled q.  It also restores DFlash2's
fused/compiled Viterbi step, which is skipped whenever ``sample_proposal`` is a
callable (``drafters/dflash2/dflash2.py:222-229``).

F2 ``MLX_VLM_SPEC_SAMPLED_COUPLING=1`` -- shared-key Gumbel-max coupling.
``mx.random.categorical`` is Gumbel-max: it draws one iid Gumbel per *array
slot* from the key and returns the argmax of ``logits + G``.  Two distributions
sampled with the *same key over the same index space* are therefore coupled:
they agree whenever the Gumbel-argmax agrees, which happens with probability
close to sum_t min(p_t, q_t) (measured 0.72 against a maximal 0.78, 92% of the
bound).  Three things have to hold at once:

  (a) proposal and target must use the same per-position key (drop the XOR);
  (b) the target's top-p sampler must sample in **token-id order**.  The
      shipped one argsorts the probabilities and calls ``categorical`` on the
      *sorted* axis, so slot j carries a different token for p than for q and
      the shared key couples nothing.  Removing the XOR alone leaves acceptance
      at the broken value -- that is the regression this module's token-order
      path guards against;
  (c) the proposal must go through the same nucleus mask at the same
      temperature over the same token-id axis, so identical logits give
      identical draws.

Neither fix changes the output distribution: the emitted token is still the
target's own draw, and the token-order nucleus mask is the sorted-order mask
under a permutation.  Only *which* proposal the target happens to agree with
changes.

Defaults are inert: with both environment variables unset the resolved mode is
``"independent"`` and coupling is off, which is byte-identical to the previous
behaviour.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import mlx.core as mx

logger = logging.getLogger("mlx_vlm.sampling_coupling")

COUPLING_ENV = "MLX_VLM_SPEC_SAMPLED_COUPLING"
DRAFT_MODE_ENV = "MLX_VLM_DFLASH_SAMPLED_DRAFT"

DRAFT_MODE_INDEPENDENT = "independent"
DRAFT_MODE_ARGMAX = "argmax"
DRAFT_MODE_GUMBEL = "gumbel"
DRAFT_MODES = (DRAFT_MODE_INDEPENDENT, DRAFT_MODE_ARGMAX, DRAFT_MODE_GUMBEL)

PROPOSAL_KEY_XOR = 0x0DFA5202


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


def sampled_coupling_enabled(env: Optional[dict] = None) -> bool:
    """F2 toggle. Off by default: unset == today's independent streams."""
    raw = (os.environ if env is None else env).get(COUPLING_ENV)
    return False if raw is None else _truthy(raw)


def dflash_sampled_draft_mode(env: Optional[dict] = None) -> str:
    """F1 toggle -- how DFlash2 draws its proposal under sampling.

    ``"independent"`` (default while F2 is off) keeps the XOR-keyed independent
    draw, ``"argmax"`` drafts the drafter's own argmax like MTP, ``"gumbel"``
    (default once F2 is on) draws the proposal in full-vocabulary token-id
    order with the target's key so the two are Gumbel-coupled.
    """
    environ = os.environ if env is None else env
    raw = environ.get(DRAFT_MODE_ENV)
    fallback = (
        DRAFT_MODE_GUMBEL
        if sampled_coupling_enabled(environ)
        else DRAFT_MODE_INDEPENDENT
    )
    if raw is None:
        return fallback
    mode = raw.strip().lower()
    if mode not in DRAFT_MODES:
        logger.warning(
            "Ignoring invalid %s=%r (expected one of %s); using %r",
            DRAFT_MODE_ENV,
            raw,
            ", ".join(DRAFT_MODES),
            fallback,
        )
        return fallback
    return mode


def top_p_logits_token_order(
    logprobs: mx.array, top_p: float, temperature: float
) -> mx.array:
    """Nucleus-masked log-probabilities laid out in **token-id order**.

    The shipped sampler keeps the mask on the argsorted axis and samples there.
    This computes the identical mask -- ascending argsort, cumulative sum, keep
    where the cumulative mass exceeds ``1 - top_p`` -- and then scatters it back
    to token ids by gathering the cumulative sum at each token's rank, so the
    categorical draw that follows is indexed by token id.  Same mask, same
    weights, same marginal; only the Gumbel slot each token occupies changes,
    and that is exactly what makes a shared key couple.
    """
    if logprobs.dtype == mx.bfloat16:
        logprobs = logprobs.astype(mx.float32)
    probs = mx.softmax(logprobs / temperature, axis=-1)
    order = mx.argsort(probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, order, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    # rank[t] = the slot token t occupies in the ascending order.
    rank = mx.argsort(order, axis=-1)
    cumulative_at_token = mx.take_along_axis(cumulative_probs, rank, axis=-1)
    top_probs = mx.where(
        cumulative_at_token > 1 - top_p,
        probs,
        mx.zeros_like(probs),
    )
    return mx.log(top_probs)


def sample_top_p_token_order(
    logprobs: mx.array, key: mx.array, top_p: float, temperature: float
) -> mx.array:
    """One nucleus draw, Gumbel-max over the token-id axis."""
    return mx.random.categorical(
        top_p_logits_token_order(logprobs, top_p, temperature), key=key
    )


def scatter_candidates_to_vocab(
    scores: mx.array, candidates: mx.array, vocab_size: int
) -> mx.array:
    """Lift a drafter's per-candidate scores onto the full token-id axis.

    The DFlash2 selector scores only ``top_k`` candidates, so its categorical
    draw is indexed by *candidate slot*.  Under a shared key that couples
    nothing -- slot j is a different token for the drafter than for the target.
    Scattering the scores back to token ids (``-inf`` elsewhere) puts both
    draws on one index space, which is the precondition for the coupling.
    """
    full = mx.full(
        tuple(scores.shape[:-1]) + (int(vocab_size),),
        -float("inf"),
        dtype=scores.dtype,
    )
    return mx.put_along_axis(full, candidates.astype(mx.int32), scores, axis=-1)


__all__ = [
    "COUPLING_ENV",
    "DRAFT_MODES",
    "DRAFT_MODE_ARGMAX",
    "DRAFT_MODE_GUMBEL",
    "DRAFT_MODE_INDEPENDENT",
    "DRAFT_MODE_ENV",
    "PROPOSAL_KEY_XOR",
    "dflash_sampled_draft_mode",
    "sample_top_p_token_order",
    "sampled_coupling_enabled",
    "scatter_candidates_to_vocab",
    "top_p_logits_token_order",
]
