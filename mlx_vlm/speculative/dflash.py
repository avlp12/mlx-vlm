import os
from typing import Any, Callable, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .common import (
    _batch_acceptance_must_be_uniform,
    _dflash_block_total,
    _record_per_row_rollback,
    _record_speculative_round,
    _record_uniform_clamp,
    _reset_per_row_rollback,
    _reset_uniform_clamp,
    _speculative_walk,
    _speculative_walk_batch,
    _SpeculativeSamplerRNG,
    generation_stream,
)


_ADAPTIVE_K_ENV = None
# Default width policy: a FIXED verify block total, at the drafter's trained
# width.  0 disables it and falls through to the legacy threshold ladder.
_FIXED_WIDTH_ENV = None
_ROUND_FIXED = None
_ROUND_COST = None
_ADAPTIVE_K_WINDOW = None
_ADAPTIVE_K_MINROUNDS = None


def _fixed_width() -> int:
    """The shipped default verify block total.  0 means "not this policy".

    R24 measured fixed block-total 8 against the shipped adaptive-K policy on
    four workloads, n=3 paired ABAB, and it won every one on worst-pair:
    code 1.0663, prose 1.0820, chat 1.0798, prose@16,394 1.0832 (receipts
    logs/sweep3/CARD_spec_row_r24_p1024.json and _r24_p16k.json).  R20 had put
    the same comparison behind a rule that let a disputed, non-shipping
    alternative veto it; R24 compared against what actually ships.

    8 is not a tuned number: it is the drafter's own trained block size
    (dflash_config.block_size), so this policy proposes exactly the width the
    checkpoint was trained to propose and never more.
    """
    global _FIXED_WIDTH_ENV
    if _FIXED_WIDTH_ENV is None:
        try:
            _FIXED_WIDTH_ENV = int(os.environ.get("MLX_VLM_DFLASH_FIXED_WIDTH", "8"))
        except (TypeError, ValueError):
            _FIXED_WIDTH_ENV = 8
    return _FIXED_WIDTH_ENV


def _adaptive_k_enabled() -> bool:
    """Pick the block width that maximises measured throughput instead of
    ratcheting on thresholds.  OPT-IN since R24: set
    ``MLX_VLM_DFLASH_ADAPTIVE_K=1``.

    THERE ARE THREE WIDTH POLICIES, and "adaptive off" no longer means "ladder":

        fixed (DEFAULT)  block total ``MLX_VLM_DFLASH_FIXED_WIDTH``, default 8 --
                         the drafter's own trained ``dflash_config.block_size``.
                         See ``_fixed_width``.
        adaptive-K       this function, ``MLX_VLM_DFLASH_ADAPTIVE_K=1``.
        ladder (legacy)  reachable only with ``MLX_VLM_DFLASH_FIXED_WIDTH=0``
                         and adaptive off.

    R24 measured fixed 8 against adaptive-K on four workloads, n=3 paired ABAB,
    and fixed won every one on worst-pair: code 1.0663, prose 1.0820, chat
    1.0798, prose@16,394 1.0832 (logs/sweep3/CARD_spec_row_r24_p1024.json,
    _r24_p16k.json).  Adaptive-K is kept as the documented alternative that lost
    R20 and R24, not as a fallback anyone should reach for by default.

    It deliberately takes precedence over ``prefer_requested_block_size`` (the
    harness's --fixed-block): pinning the width is exactly what this replaces,
    so honouring the pin would make the comparison a no-op.

    IT SHIPPED OFF, AND THE REASON IT SHIPPED OFF WAS THE COST MODEL, NOT THE
    POLICY.  With the stale constants (see _round_cost_params) this lost 11% to
    the very ladder it was meant to replace, because a fixed cost overstated by
    1.43x and a marginal cost understated by 2x pinned the argmax at the cap for
    every hazard at or above 0.81 -- it was not adapting, it was just choosing
    block 8 with extra steps.  On the refitted constants it wins both workloads.

    Measured against the shipped ladder, one load, arms interleaved in-process,
    three cycles, natural prompts at a 1024-token prime (receipt
    logs/sweep3/R9_spec_width_r9.json, analysis R9_RESULT.json):

        workload   adaptive-K vs ladder, per cycle
        code       +2.34%  +7.56%  +7.06%
        prose      +4.41%  +4.49%  +4.66%

    Six paired comparisons, six positive, on two workloads that disagree about
    almost everything else.

    WHY THE LADDER LOSES, mechanically.  The ladder has no notion of what a
    drafted token costs, so it can only ratchet between thresholds calibrated to
    nothing measurable; on prose it settles at a mean width of 3.03 and takes 118
    rounds for 256 tokens, where the cost model takes 106.

    THE COST MODEL IS AHEAD OF THE LADDER, NOT AT THE OPTIMUM -- and R24 closed
    that question against it.  Fixed W=5 already beat it on prose, 42.33 against
    38.83 tok/s, and fixed 8 went on to beat it on all four R24 workloads.  That gap was investigated
    under PA780 and it does NOT close: see _empirical_enabled for the three
    independent reasons and logs/sweep3/R12_RESULT.json for the receipts.  It is
    the drafter's cost of workload-blindness and it is structural, not a tuning
    miss.

    ON IDENTITY, AND THIS PARAGRAPH USED TO BE WRONG.  It previously claimed
    that all width policies produced identical decoded text, "which is what
    speculative decoding's exactness guarantee predicts".  R18 measured the
    opposite: every policy decoded DIFFERENT text, including greedy against
    every speculative policy (logs/sweep3/CARD_spec_row_r18.json).  Speculative
    decoding is exact in exact arithmetic, but verifying S tokens in one forward
    is not bit-identical to S sequential forwards in floating point -- a
    near-tie argmax flips once and everything after it diverges.

    What IS true, and what the receipts show: each policy is internally
    reproducible (one sha1 per policy per workload across three cycles), every
    arm returns the same token count, and the round accounting closes.  That is
    the identity standard; bit-identical text across widths is not an
    fp-achievable one, and a check asserting it was retired after firing nine
    times in R20 as pure noise.

    So width selection moves wall clock AND the exact token stream, though not
    the distribution it is drawn from.  A user toggling the policy sees
    different text.
    """
    global _ADAPTIVE_K_ENV
    if _ADAPTIVE_K_ENV is None:
        _ADAPTIVE_K_ENV = os.environ.get(
            "MLX_VLM_DFLASH_ADAPTIVE_K", "0"
        ).lower() in ("1", "true", "yes", "on")
    return _ADAPTIVE_K_ENV


def _round_cost_params():
    """Round cost in units of one plain decode step: ``fixed + cost * K``.

    The marginal cost is high because GLM-5.3-Flash is a 288-expert MoE: eight
    tokens route to ~58 distinct experts instead of 8, so widening the verify
    multiplies FFN weight traffic.  On a dense model verify width would be
    nearly free and the optimum would be much wider; here it is not.

    Both constants belong to the target and the box, not to the drafter, so
    they are env-tunable and must be refitted when the serving stack changes.
    THEY WERE NOT, AND THE STACK CHANGED TWICE.

    The original fit (probe_verify_width_v2.py, committed at 9a9b7d82) read
    verify = 44.7 + 4.75*L ms against a 35.5 ms decode step, giving fixed = 1.87
    and cost = 0.134.  That commit precedes BOTH 4bb3b97b -- which generalised
    MLA absorption to L > 1, so a verify block stopped materialising the latent
    KV cache -- and 5c59cfa7, which found the server's speculative loop running
    unwired and paging on every forward.  Each bug hid the other's magnitude, so
    the width model was fitted on a cost curve that no longer exists.

    Refitted from the campaign's own post-fix receipt (EAGLE3_REBASED_GATES.json:
    verify 36.54 / 63.95 / 91.75 ms at W = 1 / 4 / 8, draft 1.717 ms per drafted
    token, greedy decode step 32.84 ms).  Fitting verify on the W >= 2 anchors
    gives 36.15 + 6.95*W ms; adding (W-1) drafted tokens and dividing by the
    decode step gives fixed = 1.312 and cost = 0.264.  The shipped pair overstated
    the fixed cost by 1.43x and understated the marginal cost by 2x, which is the
    entire argument of the width argmax.

    Measured live, one load, in-process interleaved, 3 cycles, natural prompts at
    a 1024-token prime (receipt logs/sweep3/R9_spec_width_r9.json, analysis
    R9_RESULT.json).  Adaptive-K driven by each pair, median of cycles 1-2:

        constants     code tok/s    prose tok/s
        shipped          41.93          42.01
        measured         51.00          38.60

    Code +21.4%, prose -5.6%.  The prose regression is NOT the cost constants
    failing: feeding the realised acceptance back through this same model says
    the wider choice was right there (2.415 tokens/round at width 3.56 scores
    0.0370 tok/ms against 3.084 at width 4.80 scoring 0.0406).  What fails is the
    NUMERATOR -- E[accepted] predicted as sum(p^k) from a pooled hazard.

    CORRECTION.  This used to say the 16-round window starves itself of evidence.
    That is wrong, and it was wrong when I wrote it: the estimator is
    asymptotically consistent at any fixed width, because for a geometric process
    p_hat -> E[a]/(E[a] + 1 - p^d) and substituting E[a] = p(1-p^d)/(1-p) the
    (1 - p^d) cancels exactly.  Fewer observations per round is not a biased
    estimate.  The real defect is misspecification of the geometric SHAPE; see
    _empirical_enabled for the measured misfit and for why it does not close.

    Against the shipped default path (the threshold ladder, adaptive-K off)
    adaptive-K with these constants wins BOTH workloads -- code +2.34/+7.56/+7.06%
    and prose +4.41/+4.49/+4.66% across three paired cycles.  Turning it on by
    default is a policy change and is deliberately NOT made here.
    """
    global _ROUND_FIXED, _ROUND_COST
    if _ROUND_FIXED is None:
        _ROUND_FIXED = float(os.environ.get("MLX_VLM_DFLASH_ROUND_FIXED", 1.3124))
        _ROUND_COST = float(os.environ.get("MLX_VLM_DFLASH_ROUND_COST", 0.2639))
    return _ROUND_FIXED, _ROUND_COST


def _dflash_hazard(draft_model: nn.Module) -> Optional[float]:
    """Truncated-geometric MLE of the per-token acceptance hazard.

    Measured on our receipts the hazard is essentially flat across positions in
    the block (code: 0.78 0.85 0.82 0.84 0.78 0.81 0.82), so acceptance is a
    constant-hazard process and one scalar describes it.  A round that accepted
    ``a`` of ``d`` drafted tokens contributes ``a`` successes and -- only when it
    was cut short -- exactly one observed failure; a round that accepted all of
    them is right-censored and contributes no failure.  Laplace-smoothed so an
    all-accept window explores wider instead of dividing by zero.
    """
    global _ADAPTIVE_K_WINDOW, _ADAPTIVE_K_MINROUNDS
    if _ADAPTIVE_K_WINDOW is None:
        _ADAPTIVE_K_WINDOW = int(os.environ.get("MLX_VLM_DFLASH_ADAPTIVE_K_WINDOW", 16))
        _ADAPTIVE_K_MINROUNDS = int(
            os.environ.get("MLX_VLM_DFLASH_ADAPTIVE_K_MINROUNDS", 4)
        )
    accept_lens = getattr(draft_model, "accept_lens", None) or []
    draft_lens = getattr(draft_model, "draft_lens", None) or []
    recent = [
        (float(a), int(d))
        for a, d in zip(accept_lens[-_ADAPTIVE_K_WINDOW:], draft_lens[-_ADAPTIVE_K_WINDOW:])
        if int(d) > 0
    ]
    if len(recent) < _ADAPTIVE_K_MINROUNDS:
        return None
    successes = sum(a for a, _ in recent)
    failures = sum(1 for a, d in recent if a < d)
    p = (successes + 0.5) / (successes + failures + 1.0)
    return min(0.98, max(0.05, p))


def _dflash_block_size_for_hazard(
    p: float,
    cap: int,
    floor: int = 2,
    fixed: Optional[float] = None,
    cost: Optional[float] = None,
) -> int:
    """argmax over block widths of (1 + E[accepted]) / round cost.

    E[accepted] for a geometric hazard truncated at K = cap-1 drafted tokens is
    sum_j p^j, and the round costs fixed + cost*K, so the optimum is a genuine
    interior maximum: too narrow wastes the fixed cost, too wide pays for a token
    that probably will not survive.

    ``fixed``/``cost`` default to the DFlash block-drafter constants but are
    parameters because the same optimisation answers the MTP rollout-depth
    question with different economics: a DFlash block costs one drafter forward
    however wide it is, whereas each MTP rollout step costs its own forward, so
    MTP's marginal cost per unit depth is several times higher and its optimum
    correspondingly shallower.  ``floor`` may be 1, which means "propose
    nothing" -- for MTP that is a plain decode step, and it is how the never-lose
    guarantee falls out of the same formula instead of needing a separate gate.
    """
    if fixed is None or cost is None:
        f, c = _round_cost_params()
        fixed = f if fixed is None else fixed
        cost = c if cost is None else cost
    best, best_gain = floor, -1.0
    e = 0.0
    pk = 1.0
    if floor <= 1:
        best_gain = 1.0 / fixed          # width 1 == propose nothing
        best = 1
    for width in range(2, cap + 1):
        pk *= p
        e += pk                       # E[accepted] at width-1 drafted tokens
        gain = (1.0 + e) / (fixed + cost * (width - 1))
        if gain > best_gain:
            best_gain, best = gain, width
    return max(floor, min(cap, best))


_HAZ_EMPIRICAL = None
_HAZ_PARAMS = None


def _empirical_enabled() -> bool:
    """MLX_VLM_DFLASH_HAZARD_EMPIRICAL=1 -- measure E[accepted] per width instead
    of predicting it from a hazard and a geometric shape.

    WHY THIS EXISTS.  The cost model's denominator (fixed + cost*(W-1)) was
    refitted and validated.  Its NUMERATOR is the problem: it predicts
    E[accepted] as sum(p^k) from a single pooled hazard, and acceptance on this
    stack is not actually a constant-hazard geometric process.  Fitting one p per
    workload and predicting per width misfits by width, and misfits in the
    direction that flips the ranking (measured on the fixed-width arms of
    logs/sweep3/R9_spec_width_r9.json):

        prose, pooled p = 0.711     code, pooled p = 0.806
          W4  1.577 vs 1.383  +14.0%   W4  1.980 vs 2.122   -6.7%
          W5  1.832 vs 2.072  -11.6%   W5  2.402 vs 2.200   +9.2%
          W8  2.236 vs 2.312   -3.3%   W8  3.238 vs 3.250   -0.4%

    On prose the shape overstates W4 and understates W5, so the model ranks W4
    above W5 while measured throughput ranks W5 far above it (42.31 against 36.47
    tok/s).  And because a geometric MLE fitted at narrow widths returns a lower
    p than one fitted at wide widths (0.660 at W4 against 0.754 at W5), narrowing
    lowers the estimate, which justifies narrowing again.

    NOTE THAT THIS IS NOT THE MECHANISM I FIRST RECORDED.  The docstring on
    _round_cost_params and on _adaptive_k_enabled said the 16-round window
    starves itself of evidence.  That is wrong: the estimator is asymptotically
    consistent at any fixed width, because for a geometric process
    p_hat -> E[a]/(E[a] + 1 - p^d) and the (1 - p^d) cancels exactly.  Fewer
    observations per round is not a biased estimate.  The loop is real but it
    runs through the SHAPE assumption, not through the sample size.

    AND THIS POLICY DOES NOT FIX IT.  Measured against the shipped configuration,
    one load, 4 policies interleaved, 3 cycles (logs/sweep3/R9_spec_width_r12.json,
    analysis R12_RESULT.json): code 49.44 against 51.43 (-3.9%) and prose 37.53
    against 38.83 (-3.3%).  It lost on both, and prose never came near fixed W5's
    42.33.  Three reasons that is not a tuning miss:

      1  dosage.  A probe every 11 rounds over a 106-round generation is 9 probe
         rounds against 7 candidate widths needing 2 observations each.  Mean
         width came out 3.545 against adaptive-K's 3.557 -- it never moved, it
         just paid for 9 wasted rounds.
      2  the exploration cannot be afforded at ANY dosage.  Separating W4 from W5
         on prose (E=1.383 sd=1.221 against E=2.074 sd=1.601) needs 33
         observations per width at 2 sigma -- 65 rounds, against a generation
         that finishes in 83 rounds at W5.  Even a 1-sigma coin flip costs 17.
         And the ceiling of any width policy on prose IS W5, so a learner's best
         case is W5 minus an exploration cost comparable to the whole run.
      3  no better fixed shape model rescues it either.  The correction the
         geometric prediction needs (measured / predicted) is 0.878 on prose
         against 1.072 on code at W4, and 1.131 against 0.916 at W5 -- OPPOSITE
         directions at the two widths whose ranking decides the question.  The
         correction is workload-dependent by nature.

    Kept in the tree, off, with the numbers, so it is not rediscovered.
    """
    global _HAZ_EMPIRICAL
    if _HAZ_EMPIRICAL is None:
        _HAZ_EMPIRICAL = os.environ.get(
            "MLX_VLM_DFLASH_HAZARD_EMPIRICAL", "0"
        ).lower() in ("1", "true", "yes", "on")
    return _HAZ_EMPIRICAL


def _empirical_params():
    """window per width, minimum observations before trusting a width, probe period."""
    global _HAZ_PARAMS
    if _HAZ_PARAMS is None:
        _HAZ_PARAMS = (
            int(os.environ.get("MLX_VLM_DFLASH_HAZARD_WINDOW", 8)),
            int(os.environ.get("MLX_VLM_DFLASH_HAZARD_MIN_OBS", 2)),
            int(os.environ.get("MLX_VLM_DFLASH_HAZARD_PROBE_EVERY", 11)),
        )
    return _HAZ_PARAMS


def _dflash_width_table(draft_model, window: int):
    """Accepted-draft counts bucketed by the width they were actually drafted at.

    Keyed on drafted length, so bucket ``d`` describes block width ``d + 1``.
    """
    accept_lens = getattr(draft_model, "accept_lens", None) or []
    draft_lens = getattr(draft_model, "draft_lens", None) or []
    table = {}
    for a, d in zip(accept_lens, draft_lens):
        d = int(d)
        if d > 0:
            table.setdefault(d, []).append(float(a))
    return {d: v[-window:] for d, v in table.items()}


def _dflash_block_size_empirical(draft_model, p, cap, floor, fixed, cost):
    """argmax of (1 + measured E[accepted at w]) / (fixed + cost*(w-1)).

    A width with too few observations falls back to the geometric prediction, so
    this STARTS exactly where the hazard model starts and only departs where it
    has measured something the shape got wrong.  The periodic probe is what makes
    the narrow trap inescapable by construction rather than by luck: no candidate
    width can be starved of observations by the policy's own choices.
    """
    window, min_obs, probe_every = _empirical_params()
    table = _dflash_width_table(draft_model, window)
    n_rounds = len(getattr(draft_model, "draft_lens", None) or [])
    cands = list(range(floor, cap + 1))
    if not cands:
        return cap
    if probe_every > 0 and n_rounds and n_rounds % probe_every == 0:
        # least-observed candidate; ties go to the WIDER one, because the trap
        # this exists to escape is always a narrow one.
        return max(cands, key=lambda w: (-len(table.get(w - 1, ())), w))
    best, best_gain = floor, -1.0
    for w in cands:
        obs = table.get(w - 1, ())
        if len(obs) >= min_obs:
            e = sum(obs) / len(obs)
        else:
            e = sum(p ** j for j in range(1, w))
        gain = (1.0 + e) / (fixed + cost * (w - 1))
        if gain > best_gain:
            best_gain, best = gain, w
    return best


def _dflash_next_block_size(
    draft_model: nn.Module,
    requested_block_total: int,
    remaining_budget: int,
    initial_block_size: Optional[int] = None,
) -> int:
    """Choose the next DFlash verify block size from recent acceptance.

    DFlash checkpoints advertise a trained block size, usually 16. Treat that
    as the ceiling and back off quickly when deeper positions are mostly
    rejected. When acceptance is strong at the current depth, grow back toward
    the configured ceiling.
    """
    block_total = min(requested_block_total, remaining_budget)
    if block_total <= 1:
        return block_total
    if _adaptive_k_enabled():
        # Cost-model policy: supersedes both the threshold ladder below and the
        # fixed-width pin.  The two cannot compose -- they drive the same
        # actuator from different objectives, and the ladder has no notion of
        # what a drafted token costs, so it can only ratchet between thresholds
        # calibrated to nothing measurable.  Every block-8 receipt in our corpus
        # ran with --fixed-block, so the ladder has no measured behaviour here
        # to preserve either.
        p = _dflash_hazard(draft_model)
        if p is not None:
            floor = max(2, int(getattr(draft_model, "dflash_min_block_size", 2)))
            if _empirical_enabled():
                f, c = _round_cost_params()
                return _dflash_block_size_empirical(
                    draft_model, p, block_total, floor, f, c
                )
            return _dflash_block_size_for_hazard(p, block_total, floor=floor)
        if initial_block_size is not None:
            return min(block_total, max(2, int(initial_block_size)))
        return block_total
    if getattr(draft_model, "prefer_requested_block_size", False):
        return block_total

    fixed = _fixed_width()
    if fixed > 0:
        # SHIPPED DEFAULT since R24.  An explicit --fixed-block pin
        # (prefer_requested_block_size, checked above) still wins.
        #
        # initial_block_size is DELIBERATELY IGNORED here, and the first version
        # of this branch got that wrong.  It honoured initial_block_size on the
        # reasoning that "a caller that names a width means it" -- but
        # dflash_initial_block_size is a DRAFTER ATTRIBUTE, not a per-request
        # argument, and dflash.py:756 passes it on EVERY round.  DFlash2 sets it
        # to 3 (drafters/dflash2/dflash2.py:259), so the shipped default silently
        # resolved to width 3 on the single-sequence server path and never
        # reached 8.  Lane 3's X3 T1 server logged rounds=105 drafted=209, i.e.
        # block total 2.99, which is that bug and not a measurement artefact.
        #
        # The ladder below consults initial_block_size only when there is no
        # acceptance history yet -- it is a WARM-UP HINT for an adapting policy.
        # A fixed policy has nothing to warm up into, so honouring it converted a
        # first-round hint into a permanent pin.  Fixed means fixed.
        return min(block_total, max(2, fixed))

    accept_lens = getattr(draft_model, "accept_lens", None) or []
    draft_lens = getattr(draft_model, "draft_lens", None) or []
    recent = [
        (float(a), int(d))
        for a, d in zip(accept_lens[-8:], draft_lens[-8:])
        if int(d) > 0
    ]
    if not recent:
        if initial_block_size is not None:
            return min(block_total, max(2, int(initial_block_size)))
        return block_total

    current = min(block_total, max(2, recent[-1][1] + 1))
    min_total = min(
        block_total,
        max(2, int(getattr(draft_model, "dflash_min_block_size", 4))),
    )
    drafted = sum(d for _, d in recent)
    accepted = sum(a for a, _ in recent)
    accept_rate = accepted / drafted
    mean_accept = accepted / len(recent)

    # mean_accept is bounded by the drafted length (current-1), so at
    # current=3 the bound IS 2.0 and a constant 2.0 floor is satisfied by
    # anything short of a perfect recent window.  That is a downward ratchet
    # BIAS, not a permanent pin: escape upward still happens once accept_rate
    # and the full-hit rate clear their thresholds (simulated 2026-09-02,
    # logs/upstream/UPSTREAM_2074_MERGE_RECONCILE_2026-09-02.md section 4c).
    # Scale the floor below 2.0 for small blocks (2026-08-30).
    if accept_rate < 0.30 or mean_accept < min(2.0, 0.5 * (current - 1)):
        if current >= 8:
            return max(min_total, min(block_total, current // 2))
        return max(min_total, min(block_total, current - 2))

    if accept_rate < 0.50:
        return max(min_total, min(block_total, current - 2))

    full_hits = sum(1 for a, d in recent if a >= d)
    full_hit_rate = full_hits / len(recent)
    if accept_rate >= 0.85 and full_hit_rate >= 0.75:
        return min(block_total, current + 2)

    return min(block_total, current)


def _dflash_committed_hidden_segments(
    hidden_full: mx.array, new_tokens_list: List[List[int]]
) -> List[mx.array]:
    return [
        hidden_full[i : i + 1, : len(new_tokens), :]
        for i, new_tokens in enumerate(new_tokens_list)
    ]


def _supports_positioned_target_sampling(sampler: Callable) -> bool:
    return callable(getattr(sampler, "sample_target", None))


class _PositionedDraftSampler:
    def __init__(
        self,
        sampler: Callable,
        *,
        row_ids: List[int],
        positions: List[int],
    ):
        self.sampler = sampler
        self.row_ids = [int(row_id) for row_id in row_ids]
        self.positions = [int(position) for position in positions]

    def __call__(self, logits: mx.array) -> mx.array:
        if logits.ndim == 1:
            batch, length = 1, 1
        elif logits.ndim == 2:
            batch, length = logits.shape[0], 1
        else:
            batch, length = logits.shape[0], logits.shape[1]
        if batch != len(self.row_ids):
            raise ValueError(
                "Draft sampler row count does not match logits batch size."
            )

        rows = [row_id for row_id in self.row_ids for _ in range(length)]
        positions = [
            position + offset for position in self.positions for offset in range(length)
        ]
        sampled = self.sampler.sample_target(
            logits.reshape(batch * length, logits.shape[-1]),
            row_ids=rows,
            positions=positions,
        )
        self.positions = [position + length for position in self.positions]
        return sampled.reshape(logits.shape[:-1])

    def sample_proposal(self, logits: mx.array) -> mx.array:
        sample_proposal = getattr(self.sampler, "sample_proposal", None)
        if not callable(sample_proposal):
            return mx.argmax(logits, axis=-1)
        if logits.ndim == 1:
            batch, length = 1, 1
        elif logits.ndim == 2:
            batch, length = logits.shape[0], 1
        else:
            batch, length = logits.shape[0], logits.shape[1]
        if batch != len(self.row_ids):
            raise ValueError(
                "Draft sampler row count does not match logits batch size."
            )
        rows = [row_id for row_id in self.row_ids for _ in range(length)]
        positions = [
            position + offset for position in self.positions for offset in range(length)
        ]
        sampled = sample_proposal(
            logits.reshape(batch * length, logits.shape[-1]),
            row_ids=rows,
            positions=positions,
        )
        self.positions = [position + length for position in self.positions]
        return sampled.reshape(logits.shape[:-1])


def _sample_dflash_target_block(
    logits: mx.array,
    sampler: Callable[[mx.array], mx.array],
    *,
    row_ids: List[int],
    base_positions: List[int],
) -> mx.array:
    batch, length, vocab_size = logits.shape
    logprobs = _dflash_target_logprobs(logits)
    flat_logprobs = logprobs.reshape(batch * length, vocab_size)
    positions = [
        int(base_position) + position
        for base_position in base_positions
        for position in range(length)
    ]
    rows = [int(row_id) for row_id in row_ids for _ in range(length)]
    return sampler.sample_target(
        flat_logprobs,
        row_ids=rows,
        positions=positions,
    ).reshape(batch, length)


def _dflash_target_logprobs(logits: mx.array) -> mx.array:
    return mx.stack(
        [
            row - mx.logsumexp(row, axis=-1, keepdims=True)
            for row in (logits[:, position, :] for position in range(logits.shape[1]))
        ],
        axis=1,
    )


def _sample_dflash_target_walk(
    logits: mx.array,
    draft_tokens: mx.array,
    sampler: Callable[[mx.array], mx.array],
    budgets: List[int],
    *,
    row_ids: List[int],
    base_positions: List[int],
) -> Tuple[List[int], List[List[int]]]:
    if _supports_positioned_target_sampling(sampler):
        target_tokens = _sample_dflash_target_block(
            logits,
            sampler,
            row_ids=row_ids,
            base_positions=base_positions,
        )
        mx.async_eval(target_tokens)
        return _speculative_walk_batch(draft_tokens, target_tokens, budgets)

    batch, length, _ = logits.shape
    draft_count = int(draft_tokens.shape[1])
    draft_rows = draft_tokens.tolist()
    logprobs = _dflash_target_logprobs(logits)

    for position in range(length):
        target_tokens = sampler(logprobs[:, position, :])
        mx.eval(target_tokens)
        target_rows = [int(token) for token in target_tokens.reshape(-1).tolist()]
        if position < draft_count and all(
            target_rows[row] == draft_rows[row][position] for row in range(batch)
        ):
            continue

        new_tokens = []
        for row, budget in enumerate(budgets):
            tokens = draft_rows[row][:position]
            if len(tokens) < budget:
                tokens.append(target_rows[row])
            new_tokens.append(tokens[:budget])
        return [position] * batch, new_tokens

    return [draft_count] * batch, [
        draft_rows[row][: budgets[row]] for row in range(batch)
    ]


def _dflash_deferred_walk_enabled(
    draft_model: nn.Module, *, greedy_sampling: bool
) -> bool:
    """Whether a round should use the deferred greedy walk.

    ``MLX_VLM_DFLASH_DEFERRED`` forces the choice; otherwise the drafter
    attribute ``dflash_deferred_walk`` decides and defaults to on. Sampled
    decoding always stays eager: its walk draws target samples in position
    order and cannot be reduced to a token comparison.
    """
    if not greedy_sampling:
        return False
    raw = os.environ.get("MLX_VLM_DFLASH_DEFERRED")
    if raw is not None:
        return raw.lower() in ("1", "true", "yes", "on")
    return bool(getattr(draft_model, "dflash_deferred_walk", True))


def _dflash_pack_greedy_walk(
    draft_tokens: mx.array, target_tokens: mx.array, batch: int
) -> mx.array:
    """Resolve the greedy walk on device into one flat host transfer.

    Layout is ``[accepted per row..., target row per row...]``. Acceptance is
    the length of the leading run where the drafter matched the target's
    greedy choice, so every committed token is already a prefix of the target
    row and the drafted tokens never have to come back to the host.
    """
    target_rows = target_tokens.reshape(batch, -1)
    n_draft = int(draft_tokens.size) // batch
    if n_draft == 0:
        accepted = mx.zeros((batch,), dtype=mx.int32)
    else:
        matched = draft_tokens.reshape(batch, -1) == target_rows[:, :n_draft]
        accepted = mx.sum(mx.cumprod(matched.astype(mx.int32), axis=1), axis=1)
    return mx.concatenate(
        [accepted.astype(mx.int32), target_rows.reshape(-1).astype(mx.int32)],
        axis=0,
    )


def _speculative_walk_deferred_greedy(
    packed: mx.array,
    budget: int,
) -> Tuple[int, List[int]]:
    """Materialize a packed greedy walk. Mirrors ``_speculative_walk``."""
    values = packed.tolist()
    accepted = int(values[0])
    return accepted, values[1 : accepted + 2][:budget]


def _speculative_walk_batch_deferred_greedy(
    packed: mx.array,
    batch: int,
    budgets: List[int],
) -> Tuple[List[int], List[List[int]]]:
    """Materialize a packed greedy walk for B > 1."""
    values = packed.tolist()
    accepted_list = [int(value) for value in values[:batch]]
    n_target = (len(values) - batch) // batch
    new_tokens_list: List[List[int]] = []
    for row, accepted in enumerate(accepted_list):
        start = batch + row * n_target
        new_tokens_list.append(values[start : start + accepted + 1][: budgets[row]])
    return accepted_list, new_tokens_list


def _adopt_pretruncated_context(draft_model, draft_caches, target_hidden_offset: int):
    """Give a drafter back the offset a caller's context trim took from it.

    A chunked prefill may hand round 1 only the trailing window of the prompt
    (``speculative/utils.py::PrefillHiddenAccumulator``).  The drafter would
    normally discard that prefix itself and add its width to every layer cache's
    offset; when the caller discarded it first, the offset has to be supplied or
    the drafter's absolute RoPE positions move.  Loud rather than silent: a
    non-zero offset a drafter cannot accept is a wiring bug.
    """
    if not target_hidden_offset:
        return
    adopt = getattr(draft_model, "adopt_pretruncated_context", None)
    if not callable(adopt):
        raise RuntimeError(
            f"{type(draft_model).__name__} was handed a pre-truncated target "
            f"hidden (offset {target_hidden_offset}) but does not implement "
            "adopt_pretruncated_context; its context RoPE positions would be "
            "wrong. Set MLX_VLM_SPEC_PREFILL_CTX_TRIM=0 or teach the drafter."
        )
    for cache in draft_caches:
        adopt(cache, int(target_hidden_offset))


def _dflash_rounds(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    *,
    first_bonus: int,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    use_model_initial_block_size: bool = True,
    greedy_sampling: bool = True,
    target_hidden_offset: int = 0,
) -> Generator[Tuple[int, None], None, None]:
    """DFlash speculative-decoding **round loop**.

    draft → verify → walk → rollback. ``generate_step`` is responsible
    for prefill, sampling the first bonus token, and packaging the
    captured hidden states into ``hidden``.
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache. "
            "This target does not currently support DFlash speculative decoding."
        )

    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = _dflash_block_total(
        draft_model, draft_block_size, ignore_runtime=_fixed_width() > 0
    )
    draft_cache = draft_model.reset(model)
    _adopt_pretruncated_context(draft_model, [draft_cache], target_hidden_offset)
    _reset_uniform_clamp(draft_model)
    _reset_per_row_rollback(draft_model)
    positioned_sampling = _supports_positioned_target_sampling(sampler)
    sampler_rng = _SpeculativeSamplerRNG(
        draft_model,
        enabled=not greedy_sampling and not positioned_sampling,
    )
    # Greedy rounds resolve acceptance on device and pull the round back in a
    # single transfer instead of two eager token rows (2026-08-30).
    deferred_walk = _dflash_deferred_walk_enabled(
        draft_model, greedy_sampling=greedy_sampling
    )
    prepare_target_hidden = getattr(draft_model, "prepare_target_hidden", None)
    hidden_is_prepared = callable(prepare_target_hidden)
    if hidden_is_prepared:
        hidden = prepare_target_hidden(hidden)
        mx.async_eval(hidden)

    b = first_bonus
    emitted = 1  # the first bonus has already been yielded by the caller

    while emitted < max_tokens:
        bs = _dflash_next_block_size(
            draft_model,
            block_total,
            max_tokens - emitted + 1,
            (
                getattr(draft_model, "dflash_initial_block_size", None)
                if use_model_initial_block_size
                else None
            ),
        )
        if bs <= 1:
            break

        draft_kwargs = {"target_hidden_prepared": True} if hidden_is_prepared else {}
        draft_sampler = (
            _PositionedDraftSampler(
                sampler,
                row_ids=[0],
                positions=[emitted],
            )
            if not greedy_sampling and positioned_sampling
            else sampler
        )
        draft_tokens = sampler_rng.draft_tokens(
            draft_model.draft_block,
            b,
            hidden,
            draft_cache,
            bs,
            draft_sampler,
            token_dtype,
            **draft_kwargs,
        )
        mx.async_eval(draft_tokens)

        with mx.stream(generation_stream):
            verify_input = mx.concatenate(
                [mx.array([[b]], dtype=token_dtype), draft_tokens],
                axis=1,
            )
            verify_out = lm(
                verify_input,
                cache=prompt_cache,
                capture_layer_ids=target_layer_ids,
                speculative_verify=True,
            )
            hidden = mx.concatenate(verify_out.hidden_states, axis=-1)
            if greedy_sampling:
                target_tokens = sampler(verify_out.logits)
                if deferred_walk:
                    walk_packed = _dflash_pack_greedy_walk(
                        draft_tokens, target_tokens, 1
                    )
        if greedy_sampling:
            mx.async_eval(walk_packed if deferred_walk else target_tokens, hidden)
        else:
            mx.async_eval(hidden)

        if greedy_sampling and deferred_walk:
            accepted, new_tokens = _speculative_walk_deferred_greedy(
                walk_packed, max_tokens - emitted
            )
        elif greedy_sampling:
            accepted, new_tokens = _speculative_walk(
                draft_tokens, target_tokens, max_tokens - emitted
            )
        else:
            accepted_list, new_tokens_list = _sample_dflash_target_walk(
                verify_out.logits,
                draft_tokens,
                sampler,
                [max_tokens - emitted],
                row_ids=[0],
                base_positions=[emitted],
            )
            accepted = accepted_list[0]
            new_tokens = new_tokens_list[0]
            sampler_rng.target_sampled(sync_draft=not positioned_sampling)
        _record_speculative_round(draft_model, accepted, bs - 1)

        if accepted < bs - 1:
            hidden = hidden[:, : accepted + 1, :]
        b = new_tokens[-1] if new_tokens else b

        if accepted < bs - 1:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache, verify_out.gdn_states, accepted, bs
                )

        if hidden_is_prepared and emitted + len(new_tokens) < max_tokens:
            hidden = prepare_target_hidden(hidden)
            mx.async_eval(hidden)

        # Emit after scheduling the next context projection so its execution
        # can overlap server-side detokenization and response handling.
        for tok in new_tokens:
            yield tok, None
            emitted += 1
            if emitted >= max_tokens:
                return

        verify_out = None


def _dflash_rounds_batch(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    *,
    first_bonus: mx.array,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    stop_check: Optional[Callable[[int, int], bool]] = None,
    greedy_sampling: bool = True,
    row_ids: Optional[List[int]] = None,
    target_hidden_offset: int = 0,
) -> Generator[Tuple[List[Optional[int]], None], None, None]:
    """Batch DFlash speculative-decoding round loop (B > 1).

    Supports continuous batching: when a sequence finishes (EOS or
    max_tokens), it is filtered out of the target caches and the
    drafter cache is reinitialized for the new batch size.

    ``stop_check(seq_idx, token_id) -> bool`` is an optional callback
    that returns True to stop a sequence (e.g. EOS detection).

    Yields ``(tokens_list, None)`` where ``tokens_list[i]`` is the
    token for sequence ``i`` (or ``None`` if that sequence has nothing
    to emit this step).
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement " "rollback_speculative_cache."
        )

    B = first_bonus.shape[0]
    row_ids = list(range(B)) if row_ids is None else list(row_ids)
    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = _dflash_block_total(
        draft_model, draft_block_size, ignore_runtime=_fixed_width() > 0
    )
    draft_model.reset(model)
    _reset_uniform_clamp(draft_model)
    _reset_per_row_rollback(draft_model)
    positioned_sampling = _supports_positioned_target_sampling(sampler)
    sampler_rng = _SpeculativeSamplerRNG(
        draft_model,
        enabled=not greedy_sampling and not positioned_sampling,
    )
    # Greedy rounds resolve acceptance on device and pull the round back in a
    # single transfer instead of two eager token rows (2026-08-30).
    deferred_walk = _dflash_deferred_walk_enabled(
        draft_model, greedy_sampling=greedy_sampling
    )
    draft_caches = [draft_model.make_cache() for _ in range(B)]
    _adopt_pretruncated_context(draft_model, draft_caches, target_hidden_offset)

    # Per-sequence state tracked by ORIGINAL index so the caller sees
    # stable indices in the yielded token lists.
    b = first_bonus.tolist()  # active bonus tokens
    emitted = [1] * B
    finished = [False] * B
    active_idx = list(range(B))  # maps active-slot → original-index
    hidden_by_orig = [hidden[i : i + 1] for i in range(B)]

    total_emitted = sum(emitted)

    while len(active_idx) > 0:
        remaining = [
            max(1, max_tokens - emitted[active_idx[j]] + 1)
            for j in range(len(active_idx))
        ]
        bs = _dflash_next_block_size(draft_model, block_total, min(remaining))
        if bs <= 1:
            break

        n_active = len(active_idx)
        b_active = [b[active_idx[j]] for j in range(n_active)]
        b_arr = mx.array(b_active, dtype=token_dtype)

        # Draft rowwise: the DFlash drafter cache is scalar-offset and has
        # proven unsafe as a single batched cache on MLX/Metal. Target verify
        # remains batched below.
        def draft_active_rows():
            return mx.concatenate(
                [
                    draft_model.draft_block(
                        int(b_active[j]),
                        hidden_by_orig[active_idx[j]],
                        draft_caches[active_idx[j]],
                        bs,
                        (
                            _PositionedDraftSampler(
                                sampler,
                                row_ids=[row_ids[active_idx[j]]],
                                positions=[emitted[active_idx[j]]],
                            )
                            if not greedy_sampling and positioned_sampling
                            else sampler
                        ),
                        token_dtype,
                    )
                    for j in range(n_active)
                ],
                axis=0,
            )

        draft_tokens = sampler_rng.draft_tokens(
            draft_active_rows,
        )

        with mx.stream(generation_stream):
            verify_input = mx.concatenate([b_arr[:, None], draft_tokens], axis=1)
            verify_out = lm(
                verify_input,
                cache=prompt_cache,
                capture_layer_ids=target_layer_ids,
                speculative_verify=True,
            )
            hidden_full = mx.concatenate(verify_out.hidden_states, axis=-1)
            if greedy_sampling:
                target_tokens = sampler(verify_out.logits)
                if deferred_walk:
                    walk_packed = _dflash_pack_greedy_walk(
                        draft_tokens, target_tokens, n_active
                    )
        if greedy_sampling:
            mx.async_eval(walk_packed if deferred_walk else target_tokens, hidden_full)
        else:
            mx.async_eval(hidden_full)

        budgets = [max_tokens - emitted[active_idx[j]] for j in range(n_active)]
        if greedy_sampling and deferred_walk:
            accepted_list, new_tokens_list = _speculative_walk_batch_deferred_greedy(
                walk_packed, n_active, budgets
            )
        elif greedy_sampling:
            accepted_list, new_tokens_list = _speculative_walk_batch(
                draft_tokens, target_tokens, budgets
            )
        else:
            accepted_list, new_tokens_list = _sample_dflash_target_walk(
                verify_out.logits,
                draft_tokens,
                sampler,
                budgets,
                row_ids=[row_ids[active_idx[j]] for j in range(n_active)],
                base_positions=[emitted[active_idx[j]] for j in range(n_active)],
            )
            sampler_rng.target_sampled(sync_draft=not positioned_sampling)

        # Ragged accepts and a RECTANGULAR rollback cannot both be right.
        # ``rollback_speculative_cache`` on a target that cannot represent
        # per-row accept counts in the caches it was handed reduces the batch to
        # ONE trim length and ONE replay prefix, so a row that accepted fewer
        # tokens than the batch maximum would keep the live KV of tokens it
        # rejected -- real, attended, wrong tokens -- while its emitted stream
        # says otherwise. Clamp to the batch minimum BEFORE the rollback and
        # BEFORE the emit loop below so tokens and cache agree for every row.
        # The rows above the minimum give back the difference and re-draft it
        # next round: correctness over throughput, counted so the price shows
        # up in the receipt.
        #
        # A target that CAN roll these caches back per row (glm5_next on batched
        # caches: per-row KDA replay length via the gated-delta mask, per-row KV
        # length via a right-shift into left padding) keeps the ragged accepts
        # and pays nothing. The predicate is evaluated against the live caches,
        # not the class, because the same model answers differently for a
        # scalar-offset or quantized cache.
        if n_active > 1 and len(set(accepted_list)) > 1:
            if _batch_acceptance_must_be_uniform(draft_model, lm, prompt_cache):
                uniform = min(len(nt) - 1 for nt in new_tokens_list)
                _record_uniform_clamp(
                    draft_model, sum(a - uniform for a in accepted_list if a > uniform)
                )
                new_tokens_list = [nt[: uniform + 1] for nt in new_tokens_list]
                accepted_list = [uniform] * n_active
            else:
                # What the clamp would have cost, kept instead. This is the B8
                # lever's own receipt: it is the numerator of the tokens/round
                # the per-row path buys back.
                _record_per_row_rollback(
                    draft_model,
                    sum(a - min(accepted_list) for a in accepted_list),
                )

        min_accepted = min(accepted_list)
        accepted_arr = mx.array(accepted_list)

        hidden_segments = _dflash_committed_hidden_segments(
            hidden_full, new_tokens_list
        )
        for j in range(n_active):
            orig = active_idx[j]
            if hidden_segments[j].shape[1] > 0:
                hidden_by_orig[orig] = hidden_segments[j]

        for a in accepted_list:
            _record_speculative_round(draft_model, a, bs - 1)

        # Emit (map active slots back to original indices)
        max_new = max(len(nt) for nt in new_tokens_list) if new_tokens_list else 0
        for pos in range(max_new):
            tokens_out: List[Optional[int]] = [None] * B
            for j in range(n_active):
                orig = active_idx[j]
                if pos < len(new_tokens_list[j]) and not finished[orig]:
                    tok = new_tokens_list[j][pos]
                    tokens_out[orig] = tok
                    emitted[orig] += 1
                    if emitted[orig] >= max_tokens:
                        finished[orig] = True
                    if stop_check is not None and stop_check(orig, tok):
                        finished[orig] = True
            yield tokens_out, None

        # Update bonus tokens
        for j in range(n_active):
            orig = active_idx[j]
            if new_tokens_list[j]:
                b[orig] = new_tokens_list[j][-1]

        if min_accepted < bs - 1:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache, verify_out.gdn_states, accepted_arr, bs
                )

        # --- Continuous batching: filter out finished sequences ---
        keep_slots = [j for j in range(n_active) if not finished[active_idx[j]]]
        if len(keep_slots) < n_active:
            if len(keep_slots) == 0:
                break
            # Filter target caches (BatchKVCache supports this)
            keep_mx = mx.array(keep_slots, dtype=mx.int32)
            for c in prompt_cache:
                if hasattr(c, "filter"):
                    c.filter(keep_mx)
            # Update active index mapping
            active_idx = [active_idx[j] for j in keep_slots]

        verify_out = None
        total_emitted = sum(emitted)
