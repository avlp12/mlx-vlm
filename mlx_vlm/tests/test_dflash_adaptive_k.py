"""Cost-model adaptive block width (MLX_VLM_DFLASH_ADAPTIVE_K=1).

Acceptance on this stack is a constant-hazard process -- measured per-position
hazard on code is 0.78 0.85 0.82 0.84 0.78 0.81 0.82, i.e. flat -- so one scalar
p describes a workload and the best block width is whatever maximises
(1 + E[accepted]) / (fixed + cost*K).  With fixed = 1.63 decode-steps and
cost = 0.186 per drafted token (measured directly by probe_verify_width_v2.py), that
optimum is interior: too narrow wastes the fixed cost, too wide buys tokens that
will not survive.

The policy this replaces is a threshold ladder with no cost model.  It cannot
find an optimum, only ratchet between hand-set thresholds -- and every block-8
receipt in our corpus ran with --fixed-block, so it has never actually run here.
"""

import os
from types import SimpleNamespace

import pytest

import mlx_vlm.speculative.dflash as dflash


class _Drafter:
    dflash_min_block_size = 2

    def __init__(self, accept, draft):
        self.accept_lens = list(accept)
        self.draft_lens = list(draft)


def _reset():
    dflash._ADAPTIVE_K_ENV = None
    dflash._FIXED_WIDTH_ENV = None
    dflash._ROUND_FIXED = None
    dflash._ROUND_COST = None
    dflash._ADAPTIVE_K_WINDOW = None
    dflash._ADAPTIVE_K_MINROUNDS = None


@pytest.fixture(autouse=True)
def _env():
    keep = {k: os.environ.get(k) for k in (
        "MLX_VLM_DFLASH_ADAPTIVE_K", "MLX_VLM_DFLASH_ROUND_FIXED",
        "MLX_VLM_DFLASH_ROUND_COST", "MLX_VLM_DFLASH_ADAPTIVE_K_WINDOW",
        "MLX_VLM_DFLASH_ADAPTIVE_K_MINROUNDS")}
    _reset()
    yield
    for k, v in keep.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    _reset()


def test_default_on_and_the_env_is_the_kill_switch():
    """The default flipped once the cost model stopped describing the wrong machine.

    It shipped OFF because on the stale constants it lost 11% to the very ladder
    it was meant to replace.  On the refitted ones it beats the ladder on both
    workloads, six paired cycles out of six (logs/sweep3/R9_spec_width_r9.json).
    The kill switch stays, and this pins both halves: that the knob turns the
    policy on, and that setting the variable to 0 really does hand control back
    to the ladder -- including handing back the pin, which adaptive-K
    deliberately overrides.

    R24 CHANGED THE DEFAULT: adaptive-K is no longer what runs with no
    environment set (fixed block total 8 is), so this test now SELECTS the
    policy it is about instead of reaching it by omission.  The assertions
    below are unchanged; only the selection is.
    """
    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "1"
    os.environ.pop("MLX_VLM_DFLASH_FIXED_WIDTH", None)
    _reset()
    assert dflash._adaptive_k_enabled() is True

    # Prose-like: hazard 0.66, and the cost model narrows to 3.  The pin is
    # ignored on purpose -- overriding it is the whole point of the policy.
    d = _Drafter([2] * 8, [7] * 8)
    d.prefer_requested_block_size = True
    assert dflash._dflash_next_block_size(d, 8, 64) == 3

    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "0"
    _reset()
    assert dflash._adaptive_k_enabled() is False
    d = _Drafter([2] * 8, [7] * 8)
    d.prefer_requested_block_size = True
    assert dflash._dflash_next_block_size(d, 8, 64) == 8   # pin honoured again

    # Below the minimum round count there is no hazard to act on.  With
    # adaptive-K off the FIXED policy is what answers now, and it returns 8
    # whether or not a caller offers an opener -- initial_block_size is a
    # warm-up hint for an ADAPTING policy, and honouring it in the fixed policy
    # was the defect that pinned the shipped default to DFlash2's hint of 3.
    # The ladder's (unchanged) use of the hint is covered by
    # test_ladder_still_treats_initial_block_size_as_a_warmup_hint.
    os.environ.pop("MLX_VLM_DFLASH_ADAPTIVE_K", None)
    _reset()
    d = _Drafter([1, 2], [7, 7])
    assert dflash._dflash_hazard(d) is None
    assert dflash._dflash_next_block_size(d, 8, 64) == 8
    assert dflash._dflash_next_block_size(d, 8, 64, initial_block_size=3) == 8


def test_the_default_widens_for_code_and_narrows_for_prose():
    """The behaviour the default is shipped FOR, not merely that it is on.

    One drafter shape, three acceptance histories: the cost model has to move the
    width in different directions.  A threshold ladder cannot express this, which
    is why it settles narrow on prose and spends 118 rounds on 256 tokens where
    the cost model spends 106.

    Selected explicitly since R24 made fixed width 8 the default.
    """
    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "1"
    os.environ.pop("MLX_VLM_DFLASH_FIXED_WIDTH", None)
    _reset()
    assert dflash._dflash_next_block_size(_Drafter([2] * 8, [7] * 8), 8, 64) == 3   # p 0.66
    assert dflash._dflash_next_block_size(_Drafter([6] * 8, [7] * 8), 8, 64) == 6   # p 0.85
    assert dflash._dflash_next_block_size(_Drafter([7] * 8, [7] * 8), 8, 64) == 8   # p 0.98


def test_hazard_is_the_truncated_geometric_mle():
    d = _Drafter([2, 3, 2, 1], [7, 7, 7, 7])
    # 8 successes, 4 observed failures (every round was cut short), +Laplace
    assert dflash._dflash_hazard(d) == pytest.approx((8 + 0.5) / (8 + 4 + 1.0))


def test_full_accept_rounds_are_right_censored():
    """A round that accepted everything drafted saw no rejection, so it must not
    contribute a failure -- otherwise a perfect drafter looks imperfect and the
    width never grows."""
    full = _Drafter([7] * 8, [7] * 8)
    cut = _Drafter([7, 7, 7, 7, 7, 7, 7, 6], [7] * 8)
    assert dflash._dflash_hazard(full) > dflash._dflash_hazard(cut)
    # raw MLE would be 56.5/57 = 0.991; the estimate is clamped to 0.98 so an
    # all-accept window cannot drive the width to a degenerate certainty
    assert dflash._dflash_hazard(full) == pytest.approx(0.98)
    # Laplace smoothing bites harder on a thin all-accept window, so a narrow
    # block cannot bootstrap itself to certainty off a handful of rounds.
    small = _Drafter([2] * 8, [2] * 8)
    assert dflash._dflash_hazard(small) == pytest.approx((16 + 0.5) / (16 + 1.0))
    assert dflash._dflash_hazard(small) < dflash._dflash_hazard(full)


def test_hazard_is_clamped_at_both_ends():
    dead = _Drafter([0] * 40, [7] * 40)
    assert dflash._dflash_hazard(dead) == pytest.approx(0.05)


def test_cold_start_falls_back():
    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "1"
    _reset()
    d = _Drafter([1, 2], [7, 7])                     # below min rounds
    assert dflash._dflash_hazard(d) is None
    assert dflash._dflash_next_block_size(d, 8, 64) == 8
    assert dflash._dflash_next_block_size(d, 8, 64, initial_block_size=3) == 3


@pytest.mark.parametrize(
    "p,expected", [(0.55, 3), (0.68, 4), (0.75, 4), (0.81, 5), (0.85, 6), (0.93, 8)]
)
def test_optimum_width_by_hazard(p, expected):
    """These moved when the cost constants were refitted, and that is the point.

    Under the old fixed=1.87 / cost=0.134 the table read 4, 5, 7, 8, 8, 8 -- the
    argmax pinned at the cap for every hazard at or above 0.81, because the fit
    understated the marginal cost of a drafted token by 2x.  The refit
    (see _round_cost_params) pulls the optimum back to a genuine interior
    maximum across the whole useful range.
    """
    assert dflash._dflash_block_size_for_hazard(p, 8) == expected


def test_the_shipped_constants_are_the_refitted_ones():
    """A guard on the number itself, so a silent revert is a loud failure.

    Provenance is in _round_cost_params; the live A/B that justifies it is
    logs/sweep3/R9_spec_width_r9.json.
    """
    dflash._ROUND_FIXED = dflash._ROUND_COST = None
    os.environ.pop("MLX_VLM_DFLASH_ROUND_FIXED", None)
    os.environ.pop("MLX_VLM_DFLASH_ROUND_COST", None)
    fixed, cost = dflash._round_cost_params()
    assert (round(fixed, 4), round(cost, 4)) == (1.3124, 0.2639)
    dflash._ROUND_FIXED = dflash._ROUND_COST = None


def test_optimum_is_an_interior_maximum():
    """Not monotone in the cap: at a middling hazard the best width is strictly
    inside the range, which is exactly what a ratchet cannot express."""
    fixed, cost = dflash._round_cost_params()
    p = 0.70
    gains = []
    for w in range(2, 17):
        e = sum(p ** j for j in range(1, w))
        gains.append((1 + e) / (fixed + cost * (w - 1)))
    assert gains.index(max(gains)) not in (0, len(gains) - 1)
    assert dflash._dflash_block_size_for_hazard(p, 16) == 2 + gains.index(max(gains))


def test_prose_like_narrows_and_code_like_does_not():
    """The measured motivation: prose sits at hazard ~0.68 and runs at block 8
    for 1.02x, when its own optimum is block 5 for ~1.13x.  Code is already at
    its optimum and must not move."""
    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "1"
    _reset()
    prose = _Drafter([2, 2, 3, 2, 2, 1, 3, 2], [7] * 8)
    code = _Drafter([7, 6, 7, 7, 5, 7, 7, 6], [7] * 8)
    assert dflash._dflash_next_block_size(prose, 8, 64) < 7
    assert dflash._dflash_next_block_size(code, 8, 64) == 8


def test_collapses_on_a_useless_drafter():
    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "1"
    _reset()
    dead = _Drafter([0] * 8, [7] * 8)
    assert dflash._dflash_next_block_size(dead, 8, 64) == 2


def test_min_block_size_is_respected():
    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "1"
    _reset()
    dead = _Drafter([0] * 8, [7] * 8)
    dead.dflash_min_block_size = 3
    assert dflash._dflash_next_block_size(dead, 8, 64) == 3


def test_budget_clamps_the_width():
    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "1"
    _reset()
    good = _Drafter([7] * 8, [7] * 8)
    assert dflash._dflash_next_block_size(good, 8, 3) == 3


def test_adaptive_k_overrides_the_fixed_pin():
    """Deliberate: --fixed-block is what this is replacing, so honouring the pin
    would make the A/B a no-op."""
    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "1"
    _reset()
    prose = _Drafter([2] * 8, [7] * 8)
    prose.prefer_requested_block_size = True
    assert dflash._dflash_next_block_size(prose, 8, 64) < 8


def test_cost_params_are_env_tunable():
    """fixed/cost are properties of the target and the box, not the drafter."""
    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "1"
    os.environ["MLX_VLM_DFLASH_ROUND_COST"] = "0.030"   # the (wrong) sweep value
    _reset()
    # a near-free draft token pushes the optimum to the cap
    assert dflash._dflash_block_size_for_hazard(0.68, 8) == 8
    os.environ["MLX_VLM_DFLASH_ROUND_COST"] = "0.134"
    _reset()
    assert dflash._dflash_block_size_for_hazard(0.68, 8) == 5


# --- R24: fixed block total 8 is the shipped default -------------------------
#
# The knob is MEMOISED (dflash._ADAPTIVE_K_ENV, _FIXED_WIDTH_ENV are module
# globals filled on first read), so a test that only sets the environment
# variable would pass while measuring nothing once any other test has read it.
# Every case below resets the globals explicitly; that is what _reset() is for.


def test_default_width_policy_is_fixed_eight():
    """No environment at all -> block total 8, the drafter's trained width."""
    for k in ("MLX_VLM_DFLASH_ADAPTIVE_K", "MLX_VLM_DFLASH_FIXED_WIDTH"):
        os.environ.pop(k, None)
    _reset()
    assert dflash._adaptive_k_enabled() is False
    assert dflash._fixed_width() == 8
    d = _Drafter([7] * 8, [7] * 8)          # acceptance the ladder would grow on
    assert dflash._dflash_next_block_size(d, 8, 64) == 8
    d2 = _Drafter([0] * 8, [7] * 8)         # acceptance the ladder would shrink on
    assert dflash._dflash_next_block_size(d2, 8, 64) == 8, "fixed policy must not adapt"


def test_default_width_policy_respects_remaining_budget():
    for k in ("MLX_VLM_DFLASH_ADAPTIVE_K", "MLX_VLM_DFLASH_FIXED_WIDTH"):
        os.environ.pop(k, None)
    _reset()
    assert dflash._dflash_next_block_size(_Drafter([7] * 8, [7] * 8), 8, 3) == 3


def test_adaptive_k_knob_restores_adaptive_policy():
    """The documented escape hatch: the alternative that lost R20/R24."""
    os.environ.pop("MLX_VLM_DFLASH_FIXED_WIDTH", None)
    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "1"
    _reset()
    try:
        assert dflash._adaptive_k_enabled() is True
        low = dflash._dflash_next_block_size(_Drafter([2] * 8, [7] * 8), 8, 64)
        high = dflash._dflash_next_block_size(_Drafter([7] * 8, [7] * 8), 8, 64)
        assert low != high, "adaptive-K must vary the width with acceptance"
        assert low < 8
    finally:
        os.environ.pop("MLX_VLM_DFLASH_ADAPTIVE_K", None)
        _reset()


def test_ladder_still_reachable_when_fixed_disabled():
    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "0"
    os.environ["MLX_VLM_DFLASH_FIXED_WIDTH"] = "0"
    _reset()
    try:
        assert dflash._fixed_width() == 0
        assert dflash._dflash_next_block_size(SimpleNamespace(accept_lens=[], draft_lens=[]),
                                              16, 20) == 16
    finally:
        for k in ("MLX_VLM_DFLASH_ADAPTIVE_K", "MLX_VLM_DFLASH_FIXED_WIDTH"):
            os.environ.pop(k, None)
        _reset()


def test_explicit_pin_still_wins_over_the_default():
    for k in ("MLX_VLM_DFLASH_ADAPTIVE_K", "MLX_VLM_DFLASH_FIXED_WIDTH"):
        os.environ.pop(k, None)
    _reset()
    d = _Drafter([7] * 8, [7] * 8)
    d.prefer_requested_block_size = True
    assert dflash._dflash_next_block_size(d, 16, 64) == 16


# --- the shipped default must survive the drafter's own attributes -----------
#
# The first version of the fixed policy honoured initial_block_size, which is
# passed from draft_model.dflash_initial_block_size on EVERY round
# (dflash.py:756). DFlash2 sets it to 3, so the default silently resolved to
# width 3 on the single-sequence server path and never reached 8. Lane 3's X3
# T1 server logged rounds=105 drafted=209 -> block total 2.99, which is that
# defect. These pin the resolution end-to-end.


def test_default_ignores_drafter_initial_block_size():
    """A DFlash2-shaped drafter must still resolve to 8, not to its hint of 3."""
    for k in ("MLX_VLM_DFLASH_ADAPTIVE_K", "MLX_VLM_DFLASH_FIXED_WIDTH"):
        os.environ.pop(k, None)
    _reset()
    d = _Drafter([1] * 8, [2] * 8)
    d.dflash_initial_block_size = 3
    assert dflash._dflash_next_block_size(d, 8, 64, 3) == 8, (
        "initial_block_size is a warm-up hint for an adapting policy; a fixed "
        "policy must not be pinned by it"
    )
    assert dflash._dflash_next_block_size(d, 8, 64) == 8


def test_ladder_still_treats_initial_block_size_as_a_warmup_hint():
    """The ladder's use of the hint is unchanged -- only the fixed policy ignores it."""
    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "0"
    os.environ["MLX_VLM_DFLASH_FIXED_WIDTH"] = "0"
    _reset()
    try:
        cold = SimpleNamespace(accept_lens=[], draft_lens=[])
        assert dflash._dflash_next_block_size(cold, 16, 20, 4) == 4
    finally:
        for k in ("MLX_VLM_DFLASH_ADAPTIVE_K", "MLX_VLM_DFLASH_FIXED_WIDTH"):
            os.environ.pop(k, None)
        _reset()


def test_explicit_pin_still_beats_the_drafter_hint():
    for k in ("MLX_VLM_DFLASH_ADAPTIVE_K", "MLX_VLM_DFLASH_FIXED_WIDTH"):
        os.environ.pop(k, None)
    _reset()
    d = _Drafter([1] * 8, [2] * 8)
    d.dflash_initial_block_size = 3
    d.prefer_requested_block_size = True
    assert dflash._dflash_next_block_size(d, 16, 64, 3) == 16
