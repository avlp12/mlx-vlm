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

F2 ``MLX_VLM_DFLASH_SAMPLED_DRAFT=gumbel`` (the default) -- shared-key
Gumbel-max coupling.  ``mx.random.categorical`` is Gumbel-max: it draws one iid
Gumbel per *array slot* from the key and returns the argmax of ``logits + G``.
Two distributions sampled with the *same key over the same index space* are
therefore coupled: they agree whenever the Gumbel-argmax agrees, which happens
with probability close to sum_t min(p_t, q_t) (measured 0.72 against a maximal
0.78, 93% of the bound).  Three things go into it, and they are NOT equally
important:

  (a) LOAD-BEARING.  Proposal and target must use the same per-position key
      (drop the XOR).  Without it there is no coupling at all: 0.03.
  (b) LOAD-BEARING.  The target's top-p sampler must sample in **token-id
      order**.  The shipped one argsorts the probabilities and calls
      ``categorical`` on the *sorted* axis, so slot j carries a different token
      for p than for q and the shared key couples nothing.  Removing the XOR
      alone only reaches 0.07; adding the token-order axis reaches 0.72.
  (c) WORTH ABOUT 2%.  Applying the same nucleus mask, at the same temperature,
      to the proposal before its draw.  This one is a refinement, not the
      mechanism: the drafter's scores are DFlash2 **Viterbi path scores**
      evaluated at the target's temperature, not a calibrated conditional law,
      so masking them makes the proposal a slightly better q -- it does not
      make it the target's distribution, and nothing downstream assumes it is.

Neither fix changes the output distribution: the emitted token is still the
target's own draw, and the token-order nucleus mask is the sorted-order mask
under a permutation.  Only *which* proposal the target happens to agree with
changes.

Environment contract (precedence)
---------------------------------
``MLX_VLM_DFLASH_SAMPLED_DRAFT`` is the one authoritative setting and resolves
to exactly one effective mode:

    independent  ->  coupled=False, XOR-keyed slot-axis proposal (pre-fix)
    argmax       ->  coupled=False, argmax draft (F1)
    gumbel       ->  coupled=True,  shared-key token-order coupling (F2)

``MLX_VLM_SPEC_SAMPLED_COUPLING`` is a pure alias kept for the served rail's A/B
switch: it is consulted ONLY when the draft mode is unset (or unparseable),
where truthy selects ``gumbel`` and falsy selects ``independent``.  There is no
combination that yields a coupled target with an uncoupled proposal -- that
third, broken configuration was reachable before this resolver existed.

With both unset the effective mode is ``gumbel``: the coupling is the serving
default as of the 43.0 -> 54.4 tok/s panel.  ``MLX_VLM_DFLASH_SAMPLED_DRAFT=
independent`` restores the pre-fix behaviour byte-identically.

Empty or whitespace-only values count as unset.  An unparseable value warns
once per distinct value (never per round) and falls through to the default.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Set, Tuple

import mlx.core as mx

logger = logging.getLogger("mlx_vlm.sampling_coupling")

COUPLING_ENV = "MLX_VLM_SPEC_SAMPLED_COUPLING"
DRAFT_MODE_ENV = "MLX_VLM_DFLASH_SAMPLED_DRAFT"

DRAFT_MODE_INDEPENDENT = "independent"
DRAFT_MODE_ARGMAX = "argmax"
DRAFT_MODE_GUMBEL = "gumbel"
DRAFT_MODES = (DRAFT_MODE_INDEPENDENT, DRAFT_MODE_ARGMAX, DRAFT_MODE_GUMBEL)

#: Both variables unset == the coupling.  Flipped from ``independent`` after the
#: served panel measured 43.0 -> 54.4 tok/s and the exactness review.
DEFAULT_DRAFT_MODE = DRAFT_MODE_GUMBEL

PROPOSAL_KEY_XOR = 0x0DFA5202

_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off")

# (variable, offending value) pairs already reported.  R6: the round loop
# resolves the mode every round, so an unguarded warning would print per round.
_WARNED: Set[Tuple[str, str]] = set()


def _reset_env_warnings() -> None:
    """Test hook: forget which invalid values have already been reported."""
    _WARNED.clear()


def _warn_once(name: str, raw: str, fallback: str) -> None:
    key = (name, raw)
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(
        "Ignoring invalid %s=%r; using %r. Valid values: %s.",
        name,
        raw,
        fallback,
        ", ".join(DRAFT_MODES) if name == DRAFT_MODE_ENV else "0/1",
    )


def _env_value(environ, name: str) -> Optional[str]:
    """R6: an empty or whitespace-only value is the same as not setting it."""
    raw = environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def dflash_sampled_draft_mode(env: Optional[dict] = None) -> str:
    """The ONE effective sampled-draft mode.  See the module docstring."""
    environ = os.environ if env is None else env

    raw = _env_value(environ, DRAFT_MODE_ENV)
    if raw is not None:
        mode = raw.lower()
        if mode in DRAFT_MODES:
            return mode
        _warn_once(DRAFT_MODE_ENV, raw, DEFAULT_DRAFT_MODE)

    alias = _env_value(environ, COUPLING_ENV)
    if alias is None:
        return DEFAULT_DRAFT_MODE
    lowered = alias.lower()
    if lowered in _TRUE_VALUES:
        return DRAFT_MODE_GUMBEL
    if lowered in _FALSE_VALUES:
        return DRAFT_MODE_INDEPENDENT
    _warn_once(COUPLING_ENV, alias, DEFAULT_DRAFT_MODE)
    return DEFAULT_DRAFT_MODE


def sampled_coupling_enabled(env: Optional[dict] = None) -> bool:
    """Is the target/proposal pair coupled?  Exactly ``mode == "gumbel"``.

    Derived from the mode rather than read from ``MLX_VLM_SPEC_SAMPLED_COUPLING``
    directly, so a coupled target can never be paired with a slot-axis proposal.
    """
    return dflash_sampled_draft_mode(env) == DRAFT_MODE_GUMBEL


def top_p_logits_token_order(
    logprobs: mx.array, top_p: float, temperature: float
) -> mx.array:
    """Nucleus-masked log-probabilities laid out in **token-id order**.

    The shipped sampler keeps the mask on the argsorted axis and samples there.
    This computes the identical mask -- ascending argsort, cumulative sum, keep
    where the cumulative mass exceeds ``1 - top_p`` -- and then scatters the
    cumulative sum straight back to token ids, so the categorical draw that
    follows is indexed by token id.  Same mask, same weights, same marginal;
    only the Gumbel slot each token occupies changes, and that is exactly what
    makes a shared key couple.

    One argsort, one scatter.  The scatter replaced a second ``argsort`` of the
    permutation (``rank = argsort(order)``); the two are bit-identical because
    ``order`` is a permutation, so the scatter writes every slot exactly once.
    """
    if logprobs.dtype == mx.bfloat16:
        logprobs = logprobs.astype(mx.float32)
    probs = mx.softmax(logprobs / temperature, axis=-1)
    order = mx.argsort(probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, order, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    cumulative_at_token = mx.put_along_axis(
        mx.zeros_like(probs), order, cumulative_probs, axis=-1
    )
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
    "DEFAULT_DRAFT_MODE",
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
