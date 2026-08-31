"""The sequencing guard: refuse to load while another heavy run is resident.

Written after the 2026-08-31 freeze, in which a shutting-down server held
183 GiB, the next load started anyway, and the box needed a power cycle.  The
rule is not subtle -- two residents of that size do not fit -- so these tests
are about the guard being *unskippable and honest*: that it refuses rather than
assumes, that "the guard ran and found the fleet quiet" leaves a receipt
distinguishable from "nobody called the guard", and that it covers the peer box
rather than only the one running it.
"""

import pytest

from mlx_vlm.tp import fleet


_QUIET = """\
  4096   101 /usr/sbin/cfprefsd
 512000   202 python -m pytest
"""

_BUSY = """\
  4096   101 /usr/sbin/cfprefsd
191938560   999 python -m mlx_vlm.server --model quasar
"""


def test_parses_rss_in_kib():
    """``ps`` reports kibibytes on macOS; reading them as bytes would put the
    threshold a thousand times too high and never fire."""
    heavy = fleet._parse_ps(_BUSY, threshold_gb=20)
    assert len(heavy) == 1
    assert heavy[0]["pid"] == 999
    assert 180 < heavy[0]["rss_gb"] < 190


def test_a_pytest_process_is_not_heavy():
    assert fleet._parse_ps(_QUIET, threshold_gb=20) == []


def test_quiet_fleet_returns_a_receipt():
    r = fleet.require_quiet_fleet(threshold_gb=20, label="unit",
                                  ps_runner=lambda host: _QUIET)
    assert r["quiet"] is True and r["label"] == "unit"
    assert r["checked"] == ["localhost"]


def test_busy_fleet_refuses_and_names_the_process():
    with pytest.raises(fleet.HeavyRunActive, match="mlx_vlm.server"):
        fleet.require_quiet_fleet(threshold_gb=20, label="tp load",
                                  ps_runner=lambda host: _BUSY)


def test_the_peer_box_is_checked_too():
    """A shard on the peer is exactly as fatal as one here, and the peer is the
    box nobody is looking at."""
    seen = []

    def runner(host):
        seen.append(host)
        return _BUSY if host else _QUIET

    with pytest.raises(fleet.HeavyRunActive, match="10.0.0.2"):
        fleet.require_quiet_fleet(["10.0.0.1", "10.0.0.2"], threshold_gb=20,
                                  ps_runner=runner)
    assert None in seen and "10.0.0.2" in seen


def test_an_uninspectable_box_refuses_rather_than_assumes_quiet():
    """The failure that matters is "I could not tell": treating an ssh timeout
    as an all-clear is how the guard would get out of the way at the exact
    moment it is needed."""
    def runner(host):
        raise fleet.HeavyRunActive("ssh timed out")

    with pytest.raises(fleet.HeavyRunActive):
        fleet.require_quiet_fleet(["10.0.0.1", "10.0.0.2"], ps_runner=runner)


def test_ignore_pids_lets_a_driver_exempt_itself():
    r = fleet.require_quiet_fleet(threshold_gb=20, ignore_pids=[999],
                                  ps_runner=lambda host: _BUSY)
    assert r["quiet"] is True


def test_wait_for_quiet_queues_then_proceeds():
    """The bulk-capture wrapper's behaviour: queue behind the running job."""
    states = [_BUSY, _BUSY, _QUIET]
    slept = []

    def runner(host):
        return states[min(len(slept), len(states) - 1)]

    r = fleet.wait_for_quiet(threshold_gb=20, poll_s=0,
                             ps_runner=runner, sleep=slept.append,
                             now=lambda: 0.0)
    assert r["quiet"] is True and r["polls"] == 2


def test_wait_for_quiet_gives_up_loudly():
    """Waiting forever and starting anyway are both worse than saying which
    process is in the way."""
    clock = iter([0.0, 0.0, 10_000.0, 20_000.0])

    with pytest.raises(fleet.HeavyRunActive, match="waited"):
        fleet.wait_for_quiet(threshold_gb=20, timeout_s=1.0, poll_s=0,
                             ps_runner=lambda host: _BUSY,
                             sleep=lambda s: None, now=lambda: next(clock))


def test_self_rss_is_a_number_for_this_process():
    """Used by the server teardown to prove the model was released rather than
    assert it."""
    assert fleet.self_rss_gb() > 0


def test_self_rss_of_a_dead_pid_is_zero():
    assert fleet.self_rss_gb(2 ** 30) == 0.0


def test_cli_exits_75_when_busy(monkeypatch, capsys):
    """EX_TEMPFAIL, so a shell driver can `|| exit` or retry without parsing."""
    monkeypatch.setattr(fleet, "_run_ps", lambda host=None, timeout=20.0: _BUSY)
    assert fleet.main(["--label", "x"]) == 75
    assert '"quiet": false' in capsys.readouterr().out


def test_cli_exits_0_when_quiet(monkeypatch, capsys):
    monkeypatch.setattr(fleet, "_run_ps", lambda host=None, timeout=20.0: _QUIET)
    assert fleet.main(["--label", "x"]) == 0
    assert '"quiet": true' in capsys.readouterr().out
