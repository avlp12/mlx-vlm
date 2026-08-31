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
Processes: 500 total
PID   MEM  COMMAND
1113  838M UGREEN NAS Helpe
1777  362M python3.13
===PS===
 1113 /Applications/UGREEN NAS Helper.app/Contents/MacOS/helper
 1777 python -m pytest
"""

_BUSY = """\
Processes: 500 total
PID   MEM  COMMAND
999   183G Python
1113  838M UGREEN NAS Helpe
===PS===
  999 python -m mlx_vlm.server --model quasar
 1113 /Applications/UGREEN NAS Helper.app/Contents/MacOS/helper
"""


def test_rss_is_the_wrong_metric_and_footprint_is_the_right_one():
    """The measurement this module exists because of.

    8 GiB of live ``mx.array`` moves this process's RSS by ~0.01 GB: MLX
    allocates Metal buffers and macOS does not count them in ``resident_size``.
    An RSS threshold of any value is therefore a guard that never fires -- which
    is consistent with the 2026-08-31 freeze having happened at all, since the
    EAGLE-3 wrapper's guard was exactly that.  ``ri_phys_footprint`` tracked the
    same allocation to within 0.5%.
    """
    import mlx.core as mx
    import subprocess

    before_fp = fleet.phys_footprint_gb()
    before_rss = float(subprocess.run(
        ["ps", "-o", "rss=", "-p", str(__import__("os").getpid())],
        capture_output=True, text=True).stdout.strip() or 0) * 1024 / 1024**3
    hold = mx.zeros((1024, 1024, 512), dtype=mx.float32)   # 2 GiB
    mx.eval(hold)
    after_fp = fleet.phys_footprint_gb()
    after_rss = float(subprocess.run(
        ["ps", "-o", "rss=", "-p", str(__import__("os").getpid())],
        capture_output=True, text=True).stdout.strip() or 0) * 1024 / 1024**3
    del hold
    assert after_fp - before_fp > 1.5, "footprint must see a 2 GiB allocation"
    assert after_rss - before_rss < 0.5, "RSS must not (this is the whole point)"


def test_parses_top_mem_units():
    for cell, gb in (("838M", 0.818), ("12G", 12.0), ("1234K", 0.0011),
                     ("8211M+", 8.018), ("183G", 183.0)):
        got = fleet._parse_mem(cell)
        assert abs(got - gb) < 0.01, (cell, got, gb)
    assert fleet._parse_mem("") is None
    assert fleet._parse_mem("junk") is None


def test_command_lines_come_from_ps_not_from_top():
    """``top``'s COMMAND column is truncated to 16 characters, which is not
    enough to tell one python from another -- and "which python" is the whole
    diagnostic."""
    heavy = fleet._parse_ps(_BUSY, threshold_gb=20)
    assert len(heavy) == 1
    assert heavy[0]["pid"] == 999
    assert heavy[0]["footprint_gb"] == 183.0
    assert "mlx_vlm.server" in heavy[0]["cmd"]


def test_back_compat_rss_key_still_carries_the_footprint():
    """Existing receipts read ``rss_gb``; the key stays, the number is right."""
    heavy = fleet._parse_ps(_BUSY, threshold_gb=20)
    assert len(heavy) == 1 and 180 < heavy[0]["rss_gb"] < 190


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


def test_self_footprint_is_a_number_for_this_process():
    """Used by the server teardown to prove the model was released rather than
    assert it."""
    assert fleet.self_rss_gb() > 0
    assert fleet.phys_footprint_gb() > 0


def test_self_footprint_of_a_dead_pid_is_zero():
    assert fleet.phys_footprint_gb(2 ** 30) == 0.0


def test_cli_exits_75_when_busy(monkeypatch, capsys):
    """EX_TEMPFAIL, so a shell driver can `|| exit` or retry without parsing."""
    monkeypatch.setattr(fleet, "_run_ps", lambda host=None, timeout=20.0: _BUSY)
    assert fleet.main(["--label", "x"]) == 75
    assert '"quiet": false' in capsys.readouterr().out


def test_cli_exits_0_when_quiet(monkeypatch, capsys):
    monkeypatch.setattr(fleet, "_run_ps", lambda host=None, timeout=20.0: _QUIET)
    assert fleet.main(["--label", "x"]) == 0
    assert '"quiet": true' in capsys.readouterr().out
