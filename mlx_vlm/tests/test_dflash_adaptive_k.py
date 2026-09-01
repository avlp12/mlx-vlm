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

import pytest

import mlx_vlm.speculative.dflash as dflash


class _Drafter:
    dflash_min_block_size = 2

    def __init__(self, accept, draft):
        self.accept_lens = list(accept)
        self.draft_lens = list(draft)


def _reset():
    dflash._ADAPTIVE_K_ENV = None
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
    The kill switch stays, and this pins both halves: that the default is on, and
    that setting the variable to 0 really does hand control back to the ladder --
    including handing back the pin, which adaptive-K deliberately overrides.
    """
    os.environ.pop("MLX_VLM_DFLASH_ADAPTIVE_K", None)
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

    # Below the minimum round count there is no hazard to act on, so the
    # fallbacks are unchanged by the flip: full block, or the caller's opener.
    os.environ.pop("MLX_VLM_DFLASH_ADAPTIVE_K", None)
    _reset()
    d = _Drafter([1, 2], [7, 7])
    assert dflash._dflash_hazard(d) is None
    assert dflash._dflash_next_block_size(d, 8, 64) == 8
    assert dflash._dflash_next_block_size(d, 8, 64, initial_block_size=3) == 3


def test_the_default_widens_for_code_and_narrows_for_prose():
    """The behaviour the default is shipped FOR, not merely that it is on.

    One drafter shape, three acceptance histories: the cost model has to move the
    width in different directions.  A threshold ladder cannot express this, which
    is why it settles narrow on prose and spends 118 rounds on 256 tokens where
    the cost model spends 106.
    """
    os.environ.pop("MLX_VLM_DFLASH_ADAPTIVE_K", None)
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
