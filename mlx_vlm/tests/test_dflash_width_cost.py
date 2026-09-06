"""The three things the width controller gets wrong, each behind its own knob.

L21 ran adaptive-K on the served path and it drafted 15% more tokens than fixed
w4 while accepting 3% fewer.  The analysis
(docs/drafts/sweep11/DFLASH_WIDTH_CONTROLLER_ANALYSIS_20260906.md) named three
mechanisms and one instrument that was missing:

  * the DENOMINATOR is linear and the truth is concave -- measured per-width
    round wall clock 33.51 43.40 53.13 57.74 68.36 75.38 72.50 80.72 ms has a
    FALLING marginal cost and an outright dip, which no straight line can hold.
    ``MLX_VLM_DFLASH_ROUND_MS``.
  * the NUMERATOR assumes a flat hazard, and the implied p depends on the width
    it was measured at (code 0.8315@W4, 0.7852@W6, 0.7646@W8), which a flat
    hazard cannot do.  ``MLX_VLM_DFLASH_SURVIVAL_NUMERATOR``.
  * following the argmax every round makes a width MIXTURE, and a mixture is not
    its mean width.  ``MLX_VLM_DFLASH_WIDTH_DWELL``.
  * ``MLX_VLM_DFLASH_ROUND_TIMERS`` was instrumented only in the BATCH loop, so
    on the served single-sequence path -- the path every ms/round in the receipts
    came from -- it was a silent no-op.

Every one of them is off by default, and the first test in each section is the
one that says so: with no environment set the shipped policy must be the same
arithmetic it was before, not merely a similar answer.
"""

import os
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mlx_vlm.speculative.dflash as dflash


# the R20 charter_retake per-width p50 round wall clock, W=1..8, W=1 being the
# no-draft decode step that normalises the rest
R20_MS = "33.51,43.4,53.13,57.74,68.36,75.38,72.50,80.72"
R20 = [33.51, 43.4, 53.13, 57.74, 68.36, 75.38, 72.50, 80.72]

ENV_KEYS = (
    "MLX_VLM_DFLASH_ADAPTIVE_K",
    "MLX_VLM_DFLASH_FIXED_WIDTH",
    "MLX_VLM_DFLASH_ROUND_FIXED",
    "MLX_VLM_DFLASH_ROUND_COST",
    "MLX_VLM_DFLASH_ROUND_MS",
    "MLX_VLM_DFLASH_DECODE_MS",
    "MLX_VLM_DFLASH_SURVIVAL_NUMERATOR",
    "MLX_VLM_DFLASH_WIDTH_DWELL",
    "MLX_VLM_DFLASH_ROUND_TIMERS",
    "MLX_VLM_DFLASH_ADAPTIVE_K_WINDOW",
    "MLX_VLM_DFLASH_ADAPTIVE_K_MINROUNDS",
    "MLX_VLM_DFLASH_HAZARD_EMPIRICAL",
    "MLX_VLM_DFLASH_DEFERRED",
)


def _reset():
    """Every knob here is MEMOISED in a module global, so a test that only sets
    the environment variable would measure nothing once any other test has read
    it."""
    dflash._ADAPTIVE_K_ENV = None
    dflash._FIXED_WIDTH_ENV = None
    dflash._ROUND_FIXED = None
    dflash._ROUND_COST = None
    dflash._ADAPTIVE_K_WINDOW = None
    dflash._ADAPTIVE_K_MINROUNDS = None
    dflash._ROUND_MS_TABLE = None
    dflash._ROUND_MS_DECODE = None
    dflash._ROUND_MS_WARNED = False
    dflash._SURVIVAL_NUMERATOR = None
    dflash._WIDTH_DWELL = None
    dflash._WIDTH_EXTRAS = None
    dflash._ROUND_TIMERS_ENV = None
    dflash._HAZ_EMPIRICAL = None
    dflash._HAZ_PARAMS = None


@pytest.fixture(autouse=True)
def _env():
    keep = {k: os.environ.get(k) for k in ENV_KEYS}
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    _reset()
    yield
    for k, v in keep.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    _reset()


class _Drafter:
    dflash_min_block_size = 2

    def __init__(self, accept, draft):
        self.accept_lens = list(accept)
        self.draft_lens = list(draft)


def _linear_den(width, fixed=1.3124, cost=0.2639):
    return fixed + cost * (width - 1)


def _argmax(den, numerator, cap=8, floor=2):
    """The controller's objective, written out independently of the module."""
    best, best_gain, e = floor, -1.0, 0.0
    if floor <= 1:
        best, best_gain = 1, 1.0 / den(1)
    for w in range(2, cap + 1):
        e += numerator(w - 1)
        gain = (1.0 + e) / den(w)
        if gain > best_gain:
            best_gain, best = gain, w
    return best


# --------------------------------------------------------------------------
# (a) table parsing
# --------------------------------------------------------------------------


def test_no_table_by_default():
    """The load-bearing assertion for every default in this file: with nothing
    set there is no table, so ``_dflash_block_size_for_hazard`` never reaches the
    new branch at all."""
    assert dflash._round_cost_table() is None


def test_table_parses_and_normalises_on_the_w1_decode_step():
    os.environ["MLX_VLM_DFLASH_ROUND_MS"] = R20_MS
    _reset()
    table, decode_ms = dflash._round_cost_table()
    assert list(table) == R20
    assert decode_ms == pytest.approx(33.51), "W=1 is the no-draft decode step"
    # the denominator is in decode-step units, so W=1 is exactly 1.0 and the
    # rest are ratios against it -- directly comparable to fixed + cost*(W-1)
    assert dflash._round_table_den(1, table, decode_ms, 0.2639) == pytest.approx(1.0)
    assert dflash._round_table_den(6, table, decode_ms, 0.2639) == pytest.approx(
        75.38 / 33.51
    )
    # AND THE DIP SURVIVES.  W=7 costs less than W=6; a validator that demanded
    # a non-decreasing table would throw away the one measurement this exists to
    # carry.
    assert dflash._round_table_den(7, table, decode_ms, 0.2639) < dflash._round_table_den(
        6, table, decode_ms, 0.2639
    )


def test_decode_ms_rebases_the_table():
    """The only knob needed to move a table measured on another box."""
    os.environ["MLX_VLM_DFLASH_ROUND_MS"] = R20_MS
    os.environ["MLX_VLM_DFLASH_DECODE_MS"] = "40.0"
    _reset()
    table, decode_ms = dflash._round_cost_table()
    assert decode_ms == pytest.approx(40.0)
    assert dflash._round_table_den(4, table, decode_ms, 0.2639) == pytest.approx(
        57.74 / 40.0
    )


@pytest.mark.parametrize(
    "raw",
    [
        "33.51",              # a single entry is not a curve
        "",                   # empty
        "   ",                # blank
        "33.51,nope,53.13",   # not numbers
        "33.51,0,53.13",      # a free round is not a measurement
        "33.51,-4.0",         # negative wall clock
        "33.51;43.4",         # wrong separator -> one unparseable token
    ],
)
def test_malformed_tables_are_ignored_not_fatal(raw, caplog):
    """A typo in a cost table must not take the width policy down with it."""
    os.environ["MLX_VLM_DFLASH_ROUND_MS"] = raw
    _reset()
    with caplog.at_level("WARNING", logger="mlx_vlm.speculative.dflash"):
        assert dflash._round_cost_table() is None
        # ... and it is still the shipped linear policy underneath
        assert dflash._dflash_block_size_for_hazard(0.80, 8) == 5
    if raw.strip():
        assert sum("MLX_VLM_DFLASH_ROUND_MS" in r.message for r in caplog.records) == 1


def test_the_malformed_warning_is_logged_once(caplog):
    os.environ["MLX_VLM_DFLASH_ROUND_MS"] = "33.51,nope"
    _reset()
    with caplog.at_level("WARNING", logger="mlx_vlm.speculative.dflash"):
        for _ in range(5):
            assert dflash._round_cost_table() is None
    assert len(caplog.records) == 1


def test_a_bad_decode_ms_falls_back_to_the_w1_entry(caplog):
    os.environ["MLX_VLM_DFLASH_ROUND_MS"] = R20_MS
    os.environ["MLX_VLM_DFLASH_DECODE_MS"] = "0"
    _reset()
    with caplog.at_level("WARNING", logger="mlx_vlm.speculative.dflash"):
        table, decode_ms = dflash._round_cost_table()
    assert decode_ms == pytest.approx(33.51)
    assert len(caplog.records) == 1


def test_beyond_the_table_the_linear_form_takes_over():
    """A table is a measurement, not a model: it has nothing to say about widths
    nobody measured, so the last two entries extrapolate a line."""
    table = (10.0, 12.0, 14.0)
    for w in range(1, 7):
        assert dflash._round_table_den(w, table, 10.0, 0.2639) == pytest.approx(
            1.0 + 0.2 * (w - 1)
        )


def test_a_falling_last_pair_does_not_extrapolate_free_tokens():
    """The dip's own hazard.  Extrapolating 75.38 -> 72.50 forever would make
    every further token cheaper than free and drive the argmax to the cap."""
    # last pair falls; the whole-table mean marginal (+2.5/width) takes over
    assert dflash._round_table_den(4, (10.0, 20.0, 15.0), 10.0, 0.2639) == pytest.approx(
        (15.0 + 2.5) / 10.0
    )
    # a table that falls end to end has no positive slope at all: fall back to
    # the fitted marginal cost
    assert dflash._round_table_den(4, (20.0, 15.0, 10.0), 20.0, 0.2639) == pytest.approx(
        (10.0 + 0.2639 * 20.0) / 20.0
    )
    # and the denominator stays strictly positive however far out you go
    assert dflash._round_table_den(64, (20.0, 15.0, 10.0), 20.0, 0.2639) > 0.0


# --------------------------------------------------------------------------
# (b) the argmax the table moves, against the linear model
# --------------------------------------------------------------------------
#
# Hand-computed at p = 0.80 and p = 0.92, cap 8, floor 2, on the shipped
# constants fixed = 1.3124 / cost = 0.2639.  E(W) = sum_{j<W} p^j.
#
#   p = 0.80        E(W)     linear den   gain      table den    gain
#     W2           0.80000     1.5763    1.14191     1.29513    1.38982
#     W3           1.44000     1.8402    1.32594     1.58549    1.53895
#     W4           1.95200     2.1041    1.40298     1.72307    1.71322
#     W5           2.36160     2.3680    1.41959 *   2.03999    1.64785
#     W6           2.68928     2.6319    1.40176     2.24948    1.64006
#     W7           2.95142     2.8958    1.36454     2.16353    1.82638 *
#     W8           3.16114     3.1597    1.31694     2.40883    1.72745
#   linear argmax 5, table argmax 7 -- the table WIDENS a middling workload,
#   because the dip makes W=7 cheaper than W=6.
#
#   p = 0.92        E(W)     linear den   gain      table den    gain
#     W2           0.92000     1.5763    1.21804     1.29513    1.48247
#     W3           1.76640     1.8402    1.50331     1.58549    1.74482
#     W4           2.54509     2.1041    1.68485     1.72307    2.05743
#     W5           3.26148     2.3680    1.79961     2.03999    2.08897
#     W6           3.92056     2.6319    1.86959     2.24948    2.18742
#     W7           4.52692     2.8958    1.90860     2.16353    2.55458 *
#     W8           5.08476     3.1597    1.92574 *   2.40883    2.52602
#   linear argmax 8, table argmax 7 -- the table NARROWS a high-acceptance
#   workload off the cap, because the linear model overcharges W=6/undercharges
#   W=8 and the measurement does not.
#
# Two workloads, opposite directions, one table: that is the property a straight
# line cannot have, and it is the reason the analysis calls C2 the only route
# that lands code at 4 and p512g64 wide.


@pytest.mark.parametrize(
    "p,linear_width,table_width", [(0.80, 5, 7), (0.92, 8, 7)]
)
def test_the_table_moves_the_argmax_off_the_linear_choice(p, linear_width, table_width):
    _reset()
    assert dflash._dflash_block_size_for_hazard(p, 8) == linear_width
    assert dflash._dflash_block_size_for_hazard(p, 8) == _argmax(
        _linear_den, lambda j: p ** j
    )

    os.environ["MLX_VLM_DFLASH_ROUND_MS"] = R20_MS
    _reset()
    assert dflash._dflash_block_size_for_hazard(p, 8) == table_width
    assert dflash._dflash_block_size_for_hazard(p, 8) == _argmax(
        lambda w: R20[w - 1] / R20[0], lambda j: p ** j
    )


def test_the_table_reaches_the_served_policy():
    """Not just the kernel: the width the round loop would actually ask for."""
    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "1"
    os.environ["MLX_VLM_DFLASH_ROUND_MS"] = R20_MS
    _reset()
    prose = _Drafter([2] * 8, [7] * 8)                 # p = 0.66, linear says 3
    assert dflash._dflash_next_block_size(prose, 8, 64) == 4
    mid = _Drafter([6] * 8, [7] * 8)                   # p = 0.8509, linear says 6
    assert dflash._dflash_next_block_size(mid, 8, 64) == 7
    full = _Drafter([7] * 8, [7] * 8)                  # p clamped to 0.98
    assert dflash._dflash_next_block_size(full, 8, 64) == 8


def test_the_table_is_not_applied_to_mtp_economics():
    """``_dflash_block_size_for_hazard`` also answers MTP's rollout depth, and a
    caller that names BOTH fixed and cost is describing a different machine: an
    MTP step costs its own forward, so a DFlash round-cost table says nothing
    about it."""
    os.environ["MLX_VLM_DFLASH_ROUND_MS"] = R20_MS
    _reset()
    assert dflash._round_cost_table() is not None
    assert (
        dflash._dflash_block_size_for_hazard(0.05, 8, floor=1, fixed=1.0, cost=0.5) == 1
    )
    for p in (0.55, 0.80, 0.92, 0.98):
        assert dflash._dflash_block_size_for_hazard(
            p, 8, floor=1, fixed=1.0, cost=0.5
        ) == _argmax(lambda w: 1.0 + 0.5 * (w - 1), lambda j: p ** j, floor=1)


# --------------------------------------------------------------------------
# (c) the default path is unchanged
# --------------------------------------------------------------------------

# frozen from the shipped policy before any of this landed
DEFAULT_SWEEP = [
    (0.05, 2), (0.20, 2), (0.35, 2), (0.50, 2), (0.55, 3), (0.60, 3), (0.65, 3),
    (0.674, 4), (0.70, 4), (0.75, 4), (0.772, 5), (0.80, 5), (0.824, 5),
    (0.85, 6), (0.876, 7), (0.899, 8), (0.92, 8), (0.95, 8), (0.98, 8),
]


@pytest.mark.parametrize("p,expected", DEFAULT_SWEEP)
def test_default_widths_are_byte_for_byte_the_old_policy(p, expected):
    """With none of the new variables set the controller must not merely agree,
    it must run the same arithmetic -- so this asserts the frozen table AND the
    formula, independently written."""
    _reset()
    assert dflash._round_cost_table() is None
    assert dflash._survival_numerator_enabled() is False
    assert dflash._width_dwell() == 0
    assert dflash._dflash_block_size_for_hazard(p, 8) == expected
    assert dflash._dflash_block_size_for_hazard(p, 8) == _argmax(
        _linear_den, lambda j: p ** j
    )


def test_the_argmax_boundaries_are_where_they_were():
    """Every crossover of the shipped linear policy, pinned to 1e-5 in p.

    The analysis quotes 3->4 at 0.674, 4->5 at 0.772, 5->6 at 0.824, 6->7 at
    0.876, 7->8 at 0.899.  Four of those are the rounded truth; **5->6 is at
    0.83214, not 0.824** -- at p = 0.824 the shipped policy still answers 5.  The
    numbers below are the ones the code actually has, which is what a regression
    guard has to hold.
    """
    _reset()
    for below, at, lo, hi in (
        (0.50135, 0.50136, 2, 3),
        (0.67305, 0.67306, 3, 4),
        (0.77161, 0.77162, 4, 5),
        (0.83213, 0.83214, 5, 6),
        (0.87165, 0.87166, 6, 7),
        (0.89878, 0.89879, 7, 8),
    ):
        assert dflash._dflash_block_size_for_hazard(below, 8) == lo
        assert dflash._dflash_block_size_for_hazard(at, 8) == hi
    assert dflash._dflash_block_size_for_hazard(0.824, 8) == 5


def test_the_shipped_default_width_policy_is_untouched():
    """None of this is reachable at all without MLX_VLM_DFLASH_ADAPTIVE_K=1: the
    shipped policy is fixed block total 8 and it must still answer 8."""
    _reset()
    assert dflash._fixed_width() == 8
    assert dflash._dflash_next_block_size(_Drafter([7] * 8, [7] * 8), 8, 64) == 8
    assert dflash._dflash_next_block_size(_Drafter([0] * 8, [7] * 8), 8, 64) == 8


def test_the_new_knobs_do_nothing_to_the_fixed_policy():
    """Set all three; with adaptive-K off none of them has a path to the width."""
    os.environ["MLX_VLM_DFLASH_ROUND_MS"] = R20_MS
    os.environ["MLX_VLM_DFLASH_SURVIVAL_NUMERATOR"] = "1"
    os.environ["MLX_VLM_DFLASH_WIDTH_DWELL"] = "4"
    _reset()
    assert dflash._dflash_next_block_size(_Drafter([2] * 8, [7] * 8), 8, 64) == 8


# --------------------------------------------------------------------------
# (e1) C3: the per-position survival numerator
# --------------------------------------------------------------------------


def test_survival_is_off_by_default():
    assert dflash._survival_numerator_enabled() is False


def test_survival_curve_is_laplace_smoothed_and_right_censored():
    """A round is scored only on the positions it actually PROPOSED -- the same
    censoring _dflash_hazard applies to the cut-short round."""
    d = _Drafter([0] * 8 + [3] * 8, [7] * 16)
    curve = dflash._dflash_survival_curve(d, 16)
    # positions 1..3 survived in the 8 rounds that accepted 3; 4..7 in none
    assert curve[:3] == pytest.approx([(8 + 0.5) / (16 + 1.0)] * 3)
    assert curve[3:] == pytest.approx([0.5 / 17.0] * 4)
    # a narrower round is not counted against positions it never drafted
    mixed = _Drafter([2, 2, 2, 2], [2, 2, 7, 7])
    mixed_curve = dflash._dflash_survival_curve(mixed, 16)
    assert mixed_curve[0] == pytest.approx((4 + 0.5) / (4 + 1.0))   # 4 at risk
    assert mixed_curve[2] == pytest.approx(0.5 / 3.0)               # 2 at risk


def test_survival_disagrees_with_the_geometric_shape_and_wins_the_argmax():
    """Hand-computed, cap 8, floor 2, shipped constants.

    History: 16 rounds all drafting 7, eight accepting 0 and eight accepting 3.
    This is emphatically NOT a flat-hazard process, and the two numerators say
    different things about it.

      geometric  p = (24 + 0.5) / (24 + 16 + 1) = 24.5/41 = 0.597561
        E(W)  W2 0.59756  W3 0.95464  W4 1.16802  W5 1.29553  W6 1.37172
        gain  W2 1.01349  W3 1.06218 *  W4 1.03038  W5 0.96940  W6 0.90113
        -> width 3

      survival  S_1 = S_2 = S_3 = 8.5/17 = 0.5, S_4..S_7 = 0.5/17 = 0.029412
        E(W)  W2 0.50000  W3 1.00000  W4 1.50000  W5 1.52941  W6 1.55882
        gain  W2 0.95159  W3 1.08683  W4 1.18816 *  W5 1.06816  W6 0.97223
        -> width 4

    The geometric shape stops at 3 because a pooled hazard fitted on a history
    where more than half the rounds accept nothing cannot see that the rounds
    which DO accept, accept all the way to position 3.
    """
    hist = _Drafter([0] * 8 + [3] * 8, [7] * 16)
    p = dflash._dflash_hazard(hist)
    assert p == pytest.approx(24.5 / 41.0)

    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "1"
    _reset()
    assert dflash._dflash_next_block_size(hist, 8, 64) == 3

    os.environ["MLX_VLM_DFLASH_SURVIVAL_NUMERATOR"] = "1"
    _reset()
    assert dflash._dflash_next_block_size(hist, 8, 64) == 4
    # and against the objective written out here, not read from the module
    curve = [0.5] * 3 + [0.5 / 17.0] * 4
    assert dflash._dflash_block_size_for_hazard(p, 8, survival=curve) == _argmax(
        _linear_den, lambda j: curve[j - 1]
    ) == 4


def test_survival_falls_back_to_the_geometric_shape_past_the_curve():
    """Positions nobody drafted have no measurement, so p^j answers there."""
    _reset()
    curve = [0.9, 0.9]
    p = 0.5
    assert dflash._dflash_block_size_for_hazard(p, 8, survival=curve) == _argmax(
        _linear_den, lambda j: curve[j - 1] if j <= 2 else p ** j
    )


def test_survival_with_no_history_is_none():
    assert dflash._dflash_survival_curve(_Drafter([], []), 16) is None
    assert dflash._dflash_survival_curve(_Drafter([1], [0]), 16) is None


# --------------------------------------------------------------------------
# (e2) C4: width hysteresis
# --------------------------------------------------------------------------


def test_dwell_is_off_by_default():
    assert dflash._width_dwell() == 0
    d = _Drafter([], [])
    for w in (3, 8, 2, 8):
        assert dflash._dflash_dwell_width(d, w) == w


def test_dwell_holds_the_width_until_the_argmax_has_held():
    os.environ["MLX_VLM_DFLASH_WIDTH_DWELL"] = "3"
    _reset()
    d = _Drafter([], [])
    assert dflash._dflash_dwell_width(d, 3) == 3     # nothing held yet: adopt
    assert dflash._dflash_dwell_width(d, 8) == 3     # 1st
    assert dflash._dflash_dwell_width(d, 8) == 3     # 2nd
    assert dflash._dflash_dwell_width(d, 8) == 8     # 3rd consecutive: change
    assert dflash._dflash_dwell_width(d, 8) == 8


def test_a_single_stray_round_cannot_spend_the_dwell():
    os.environ["MLX_VLM_DFLASH_WIDTH_DWELL"] = "3"
    _reset()
    d = _Drafter([], [])
    assert dflash._dflash_dwell_width(d, 4) == 4
    assert dflash._dflash_dwell_width(d, 8) == 4
    assert dflash._dflash_dwell_width(d, 8) == 4
    assert dflash._dflash_dwell_width(d, 4) == 4     # argmax came back: reset
    assert dflash._dflash_dwell_width(d, 8) == 4     # counting from 1 again
    assert dflash._dflash_dwell_width(d, 8) == 4
    assert dflash._dflash_dwell_width(d, 8) == 8


def test_a_flapping_argmax_never_changes_the_width():
    os.environ["MLX_VLM_DFLASH_WIDTH_DWELL"] = "2"
    _reset()
    d = _Drafter([], [])
    assert dflash._dflash_dwell_width(d, 5) == 5
    for w in (3, 8, 2, 7, 4, 6):
        assert dflash._dflash_dwell_width(d, w) == 5


def test_dwell_reaches_the_served_policy_and_still_respects_the_budget():
    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "1"
    os.environ["MLX_VLM_DFLASH_WIDTH_DWELL"] = "3"
    _reset()
    d = _Drafter([2] * 8, [7] * 8)                    # argmax 3
    assert dflash._dflash_next_block_size(d, 8, 64) == 3
    d.accept_lens = [7] * 8                           # argmax 8 from here on
    assert dflash._dflash_next_block_size(d, 8, 64) == 3
    assert dflash._dflash_next_block_size(d, 8, 64) == 3
    assert dflash._dflash_next_block_size(d, 8, 64) == 8
    # a held width predates the budget that is left now, so the clamp still bites
    assert dflash._dflash_next_block_size(d, 8, 2) == 2


# --------------------------------------------------------------------------
# (d) C1: round timers on the served B=1 loop
# --------------------------------------------------------------------------

HIDDEN = 4
VOCAB = 64
BLOCK = 4


class _StubTarget:
    """The only surface ``_dflash_rounds`` touches on the target."""

    def __init__(self, accepts):
        self.accepts = list(accepts)
        self.round = 0
        self.last_draft = []
        self.forwards = 0

    @staticmethod
    def _onehot(row):
        return mx.array(
            [[[1.0 if v == int(t) else 0.0 for v in range(VOCAB)] for t in row]]
        )

    def __call__(self, ids, cache=None, capture_layer_ids=None, **kw):
        self.forwards += 1
        s = int(ids.shape[1])
        a = min(self.accepts[self.round % len(self.accepts)], len(self.last_draft))
        row = list(self.last_draft[:a]) + [40 + (self.round % 8)]
        row += [50] * (s - len(row))
        self.round += 1
        return SimpleNamespace(
            logits=self._onehot(row[:s]),
            hidden_states=[mx.zeros((1, s, HIDDEN))],
            gdn_states=["gdn"],
        )

    def rollback_speculative_cache(self, caches, gdn, accepted, bs):
        return 0


class _StubDrafter:
    dflash_min_block_size = 2
    dflash_deferred_walk = True

    def __init__(self, target):
        self.config = SimpleNamespace(
            target_layer_ids=[0], block_size=BLOCK, runtime_block_size=BLOCK
        )
        self.accept_lens = []
        self.draft_lens = []
        self._target = target

    def reset(self, model):
        return ["draft-cache"]

    def draft_block(self, b, hidden, draft_cache, bs, sampler, token_dtype, **kw):
        self._target.last_draft = [10 + i for i in range(bs - 1)]
        return mx.array([self._target.last_draft], dtype=token_dtype)


def _run_b1_rounds(max_tokens=12, accepts=(1, 2, 0)):
    target = _StubTarget(accepts)
    drafter = _StubDrafter(target)
    tokens = []
    rounds = dflash._dflash_rounds(
        SimpleNamespace(language_model=target),
        drafter,
        [SimpleNamespace(offset=0)],
        mx.zeros((1, 1, HIDDEN)),
        first_bonus=7,
        max_tokens=max_tokens,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        draft_block_size=BLOCK,
        use_model_initial_block_size=False,
    )
    for tok, _ in rounds:
        tokens.append(int(tok))
    rounds.close()
    return drafter, target, tokens


def test_b1_round_timers_are_absent_by_default():
    """The default must not pay for an instrument nobody asked for -- and an
    absent counter, not a zero one, is what says the timing arm never ran."""
    _reset()
    assert dflash._round_timers_enabled() is False
    drafter, target, tokens = _run_b1_rounds()
    assert target.forwards > 1 and tokens
    assert getattr(drafter, "speculative_draft_seconds", None) is None
    assert getattr(drafter, "speculative_verify_seconds", None) is None


def test_b1_round_timers_record_both_halves_when_enabled():
    """The gap the analysis found: ROUND_TIMERS was instrumented only in
    ``_dflash_rounds_batch``, so on the SERVED single-sequence path -- the path
    every ms/round in the L21 receipts came from -- it produced nothing and
    perturbed nothing.  Any cost model refitted from those numbers was fitted to
    a split that was never measured."""
    os.environ["MLX_VLM_DFLASH_ROUND_TIMERS"] = "1"
    _reset()
    assert dflash._round_timers_enabled() is True
    drafter, target, tokens = _run_b1_rounds()
    assert target.forwards > 1 and tokens
    assert drafter.speculative_draft_seconds > 0.0
    assert drafter.speculative_verify_seconds > 0.0


def test_b1_timers_do_not_change_what_the_loop_emits():
    """Timing is allowed to perturb the CLOCK -- it forces an mx.eval between the
    halves, and the batch loop's docstring says to read the arms against each
    other -- but it must not perturb the tokens."""
    _reset()
    _, _, untimed = _run_b1_rounds()
    os.environ["MLX_VLM_DFLASH_ROUND_TIMERS"] = "1"
    _reset()
    _, _, timed = _run_b1_rounds()
    assert timed == untimed


def test_b1_and_batch_timers_use_the_same_counters():
    """Same attributes, so the server's existing readout covers both paths."""
    import inspect

    src = inspect.getsource(dflash._dflash_rounds)
    assert "_record_draft_seconds(draft_model" in src
    assert "_record_verify_seconds(draft_model" in src
    assert "timed = _round_timers_enabled()" in src


# --------------------------------------------------------------------------
# the default path must be the SAME STATEMENTS, not merely the same answer
# --------------------------------------------------------------------------
#
# A GPU panel on this branch reported the served B=1 rail 11% slower than
# d17a97fe at IDENTICAL widths and acceptance, which is the signature of added
# per-round host work or an added device sync.  These are the guards that make
# that claim testable without a GPU: an mx.eval / mx.async_eval census of the
# round loop, and a gate that keeps every opt-in behind ONE memoised boolean.
#
# The census below was taken on the base commit d17a97fe with this exact stub
# (5 rounds, max_tokens 12, accepts 1/2/0, block total 4):
#
#     mx.eval = 0        mx.async_eval = 15
#
# Zero evals is the load-bearing half: the round loop never waits, it only ever
# submits, and every wait belongs to the walk that reads the tokens back.  A
# single mx.eval added between the draft and the verify would serialise the
# pipeline the async_evals exist to build, and would not change a single width
# or acceptance count while doing it -- exactly the symptom that was reported.

BASE_EVAL_COUNTS = (0, 15)   # (mx.eval, mx.async_eval) on d17a97fe


def _count_b1_evals(max_tokens=12, accepts=(1, 2, 0)):
    real_eval, real_async = mx.eval, mx.async_eval
    counts = [0, 0]

    def counted_eval(*a, **kw):
        counts[0] += 1
        return real_eval(*a, **kw)

    def counted_async(*a, **kw):
        counts[1] += 1
        return real_async(*a, **kw)

    mx.eval, mx.async_eval = counted_eval, counted_async
    try:
        drafter, target, tokens = _run_b1_rounds(max_tokens, accepts)
    finally:
        mx.eval, mx.async_eval = real_eval, real_async
    return tuple(counts), drafter, target, tokens


def test_the_default_round_loop_issues_exactly_the_base_evals():
    """No added sync, no added submit, with the new variables unset."""
    _reset()
    counts, drafter, target, tokens = _count_b1_evals()
    assert counts == BASE_EVAL_COUNTS, (
        "the default B=1 round loop must issue the same mx.eval/mx.async_eval "
        "calls as d17a97fe; anything extra is per-round overhead the shipped "
        "policy never asked for"
    )
    assert counts[0] == 0, "the default loop must never WAIT inside a round"
    assert target.forwards == 5 and len(tokens) == 11


def test_all_three_new_knobs_set_still_add_no_evals_to_the_round_loop():
    """The width policy is not allowed to reach the round loop at all: none of
    these knobs may add a sync, whatever they do to the chosen width."""
    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "1"
    os.environ["MLX_VLM_DFLASH_ROUND_MS"] = R20_MS
    os.environ["MLX_VLM_DFLASH_SURVIVAL_NUMERATOR"] = "1"
    os.environ["MLX_VLM_DFLASH_WIDTH_DWELL"] = "2"
    _reset()
    counts, _, _, _ = _count_b1_evals()
    assert counts[0] == 0


def test_the_timing_arm_is_the_only_thing_that_adds_evals():
    """Two per round, and only with MLX_VLM_DFLASH_ROUND_TIMERS=1."""
    os.environ["MLX_VLM_DFLASH_ROUND_TIMERS"] = "1"
    _reset()
    counts, drafter, target, tokens = _count_b1_evals()
    assert counts[0] == 2 * target.forwards
    assert counts[1] == BASE_EVAL_COUNTS[1], "and it must not add submits"
    assert drafter.speculative_draft_seconds > 0.0


def test_the_extras_gate_is_one_memoised_boolean():
    """Every opt-in is behind ``_width_policy_extras_enabled``, which is read
    once per process -- so the default round pays one global compare, not one
    environment lookup per knob per round."""
    _reset()
    assert dflash._WIDTH_EXTRAS is None
    assert dflash._width_policy_extras_enabled() is False
    assert dflash._WIDTH_EXTRAS is False
    for var, value in (
        ("MLX_VLM_DFLASH_ROUND_MS", R20_MS),
        ("MLX_VLM_DFLASH_SURVIVAL_NUMERATOR", "1"),
        ("MLX_VLM_DFLASH_WIDTH_DWELL", "1"),
    ):
        for k in ("MLX_VLM_DFLASH_ROUND_MS", "MLX_VLM_DFLASH_SURVIVAL_NUMERATOR",
                  "MLX_VLM_DFLASH_WIDTH_DWELL"):
            os.environ.pop(k, None)
        os.environ[var] = value
        _reset()
        assert dflash._width_policy_extras_enabled() is True, var


def test_the_default_width_call_touches_none_of_the_opt_in_machinery(monkeypatch):
    """The strongest form of "unchanged": with the knobs unset, the survival
    curve, the cost table and the dwell state are never even consulted."""
    os.environ["MLX_VLM_DFLASH_ADAPTIVE_K"] = "1"
    _reset()
    dflash._width_policy_extras_enabled()          # resolve the gate first

    def _boom(*a, **kw):                            # pragma: no cover
        raise AssertionError("opt-in machinery reached on the default path")

    monkeypatch.setattr(dflash, "_dflash_survival_curve", _boom)
    monkeypatch.setattr(dflash, "_dflash_dwell_width", _boom)
    monkeypatch.setattr(dflash, "_round_cost_table", _boom)
    assert dflash._dflash_next_block_size(_Drafter([2] * 8, [7] * 8), 8, 64) == 3
    assert dflash._dflash_block_size_for_hazard(0.80, 8) == 5
