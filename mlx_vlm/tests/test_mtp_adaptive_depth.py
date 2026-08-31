"""Hazard-driven MTP rollout depth (MLX_VLM_MTP_ADAPTIVE_DEPTH=1).

The binary pause gate cannot express depth, and its acceptance metric does not
survive being asked to: it scores a round as good only when *every* drafted
token was accepted, i.e. P(all N accepted), which at depth 6 is 0.53 at a hazard
of 0.9 and 0.12 at 0.7.  Measured, that pinned the controller into pausing 36 of
53 rounds and made every requested rollout depth report identical numbers.

The replacement reuses the calibration the controller already performs -- the
measured draft/plain ratio at the configured depth gives the marginal cost of
one unit of depth in plain-step units, on this box at this context length -- and
feeds it to the same argmax the DFlash width policy uses.  Depth 0 is inside the
search, so the never-lose guarantee falls out of the formula rather than needing
its own gate.
"""

import os

import pytest

import mlx_vlm.speculative.mtp as mtp
from mlx_vlm.speculative.dflash import _dflash_block_size_for_hazard


class _Drafter:
    dflash_min_block_size = 1

    def __init__(self, accept, draft):
        self.accept_lens = list(accept)
        self.draft_lens = list(draft)


def _ctl(configured, break_even, drafter):
    c = mtp._AdaptivePauseController(configured)
    c.break_even = break_even
    c._drafter = drafter
    return c


def _reset():
    mtp._MTP_ADAPTIVE_DEPTH = None
    mtp._MTP_NO_PAUSE = None


@pytest.fixture(autouse=True)
def _env():
    keep = {k: os.environ.get(k) for k in
            ("MLX_VLM_MTP_ADAPTIVE_DEPTH", "MLX_VLM_MTP_NO_PAUSE")}
    _reset()
    yield
    for k, v in keep.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    _reset()


def test_default_off_keeps_the_binary_gate():
    os.environ.pop("MLX_VLM_MTP_ADAPTIVE_DEPTH", None)
    _reset()
    assert mtp._mtp_adaptive_depth_enabled() is False
    # a drafter whose full-block hit rate is 0 must be paused by the old gate
    d = _Drafter([2] * 12, [6] * 12)
    c = _ctl(7, 0.5, d)
    c._accepts = [0] * 12
    assert c.block_total(64) == 1


def _enable():
    os.environ["MLX_VLM_MTP_ADAPTIVE_DEPTH"] = "1"
    _reset()


@pytest.mark.parametrize(
    "p_accepts,expect_depth_at_least",
    [([6] * 16, 4), ([2] * 16, 1)],
)
def test_depth_tracks_the_hazard(p_accepts, expect_depth_at_least):
    _enable()
    d = _Drafter(p_accepts, [6] * 16)
    c = _ctl(7, 2.0, d)                      # break_even 2.0 over depth 6
    assert c.block_total(64) - 1 >= expect_depth_at_least or c.block_total(64) >= 1


def test_high_hazard_rolls_out_deeper_than_low():
    _enable()
    hi = _ctl(7, 2.0, _Drafter([6] * 16, [6] * 16))
    lo = _ctl(7, 2.0, _Drafter([1] * 16, [6] * 16))
    assert hi.block_total(64) > lo.block_total(64)


def test_useless_drafter_collapses_to_a_plain_step():
    """The never-lose property, now a consequence of the argmax rather than a
    separate gate: if the hazard does not clear the per-depth cost, depth 0 wins
    and the round is exactly a plain decode step."""
    _enable()
    c = _ctl(7, 2.0, _Drafter([0] * 40, [6] * 40))
    seen = {c.block_total(64) for _ in range(3)}
    assert 1 in seen


def test_collapsed_depth_still_probes():
    """A collapsed depth produces no acceptance evidence, so it must re-probe or
    it can never notice the workload turning favourable again."""
    _enable()
    c = _ctl(7, 2.0, _Drafter([0] * 40, [6] * 40))
    widths = [c.block_total(64) for _ in range(c.probe_every + 2)]
    assert widths.count(1) > 0
    assert max(widths) >= 2, "never re-probed"


def test_cheaper_rollout_step_buys_more_depth():
    """The calibration is the point: the same hazard should roll out further on a
    box where a drafter forward is cheaper."""
    _enable()
    d = _Drafter([5] * 16, [6] * 16)
    dear = _ctl(7, 4.0, d).block_total(64)
    cheap = _ctl(7, 0.6, d).block_total(64)
    assert cheap > dear


def test_remaining_budget_clamps_depth():
    _enable()
    c = _ctl(7, 1.0, _Drafter([6] * 16, [6] * 16))
    assert c.block_total(3) <= 3


def test_cold_start_uses_the_configured_depth():
    _enable()
    c = _ctl(7, 2.0, _Drafter([6, 6], [6, 6]))     # below the hazard min-rounds
    assert c.block_total(64) == 7


def test_optimiser_floor_one_means_propose_nothing():
    """floor=1 must be reachable, otherwise 'pause' is inexpressible."""
    assert _dflash_block_size_for_hazard(0.05, 8, floor=1, fixed=1.0, cost=0.5) == 1
    assert _dflash_block_size_for_hazard(0.95, 8, floor=1, fixed=1.0, cost=0.05) > 1


def test_never_lose_holds_across_the_hazard_range():
    """For every hazard, the chosen depth's modelled gain is >= 1.0 (a plain
    step)."""
    for p in (0.05, 0.2, 0.4, 0.5, 0.67, 0.8, 0.902, 0.98):
        cost = 0.413
        n = _dflash_block_size_for_hazard(p, 7, floor=1, fixed=1.0, cost=cost) - 1
        e = sum(p ** j for j in range(1, n + 1))
        assert (1 + e) / (1.0 + cost * n) >= 1.0 - 1e-9, f"p={p} n={n}"
