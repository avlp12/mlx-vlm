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


_MEM_OK = """
===MEM===
total = 4096.00M  used = 2448.50M  free = 1647.50M  (encrypted)
System-wide memory free percentage: 41%
VMFREEGB 180.0 120.0
"""

_MEM_STRAINED = """
===MEM===
total = 42912.00M  used = 39901.00M  free = 3011.00M  (encrypted)
System-wide memory free percentage: 6%
VMFREEGB 30.0 449.4
"""

# The measured 2026-09-01 post-incident state: swap drained and the percentage
# is only just under threshold, but 54.7 GB cannot hold an 85.5 GiB shard.
_MEM_NO_HEADROOM = """
===MEM===
total = 4096.00M  used = 2080.00M  free = 2016.00M  (encrypted)
System-wide memory free percentage: 11%
VMFREEGB 54.7 449.4
"""

_QUIET_HEAD = """\
Processes: 500 total
PID   MEM  COMMAND
1113  838M UGREEN NAS Helpe
1777  362M python3.13
===PS===
 1113 /Applications/UGREEN NAS Helper.app/Contents/MacOS/helper
 1777 python -m pytest
"""
_QUIET = _QUIET_HEAD + _MEM_OK
_QUIET_STRAINED = _QUIET_HEAD + _MEM_STRAINED

_BUSY = """\
Processes: 500 total
PID   MEM  COMMAND
999   183G Python
1113  838M UGREEN NAS Helpe
===PS===
  999 python -m mlx_vlm.server --model quasar
 1113 /Applications/UGREEN NAS Helper.app/Contents/MacOS/helper
""" + _MEM_OK


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


# ------------------------------------------------------- memory pressure gate
def test_parses_swap_and_free_percentage():
    m = fleet._parse_memory(_QUIET_STRAINED)
    # 39901 MiB reported by sysctl is 38.97 GiB -- the units are MiB, and
    # reading them as GB would put the threshold 1000x off.
    assert 38.5 < m["swap_used_gb"] < 39.5
    assert m["free_pct"] == 6.0


def test_pressure_gate_refuses_with_nothing_co_resident():
    """The 2026-09-01 incident shape.

    One 86 GiB shard on a 512 GB box drove swap to 39.9 of 42 GB with 6% free
    and the run died with a Metal "Insufficient Memory" while the machine went
    unresponsive.  Nothing was co-resident: the pressure had accumulated across
    the night's repeated load/unload cycles.  A guard that only counts big
    processes is blind to that by construction.
    """
    with pytest.raises(fleet.HeavyRunActive, match="memory pressure"):
        fleet.require_quiet_fleet(threshold_gb=20, label="x",
                                  ps_runner=lambda host: _QUIET_STRAINED)


def test_pressure_receipt_records_the_numbers_even_when_quiet():
    r = fleet.require_quiet_fleet(threshold_gb=20, label="x",
                                  ps_runner=lambda host: _QUIET)
    assert r["quiet"] is True
    assert r["pressure"]["localhost"]["free_pct"] == 41.0
    assert r["pressure"]["localhost"]["swap_used_gb"] == 2.39
    assert r["strained"] == {}


def test_memory_snapshot_is_available_to_harnesses():
    m = fleet.memory_snapshot(ps_runner=lambda host: _QUIET_STRAINED)
    assert m["swap_used_gb"] is not None and m["free_pct"] == 6.0


def test_a_missing_memory_section_does_not_block(monkeypatch):
    """An older peer, or a box where memory_pressure is unavailable, must not
    be refused for a check it cannot answer -- the process check still applies."""
    text = _QUIET_HEAD
    r = fleet.require_quiet_fleet(threshold_gb=20, ps_runner=lambda host: text)
    assert r["quiet"] is True
    # Assert the INTENT -- every field unanswered -- rather than an exact key
    # set. The key set legitimately grew when the reclaimable accounting added
    # inactive/file-backed counts, and an exact-equality assertion turned that
    # into a spurious failure about something the test does not care about.
    pressure = r["pressure"]["localhost"]
    assert set(pressure) >= {"swap_used_gb", "free_pct", "free_gb", "wired_gb"}
    assert all(v is None for v in pressure.values()), pressure


def test_peer_without_page_counts_downgrades_to_free_only(caplog):
    """An older peer sends no inactive/file-backed counts. Under the reclaimable
    accounting it must be judged by the OLDER, STRICTER rule -- not refused for
    the question, and never waved through silently."""
    head = _QUIET_HEAD + (
        "===MEM===\n"
        "total = 0.00M  used = 0.00M  free = 0.00M\n"
        "System-wide memory free percentage: 90%\n"
        "VMFREEGB 400.0 6.0\n"          # two fields only: the older probe
    )
    with caplog.at_level("WARNING"):
        r = fleet.require_headroom_box(None, 183.0, label="old peer",
                                       ps_runner=lambda host: head,
                                       accounting="reclaimable")
    assert r["accounting"] == "free_only"
    assert r["accounting_downgraded_from"] == "reclaimable"
    assert "free_only" in r["downgrade_reason"]
    assert any("free_only" in m for m in caplog.messages), caplog.messages
    # 400 GB free clears 183+60, so the stricter rule still passes it.
    assert r["headroom_ok"] is True
    # And the stricter rule must actually BE stricter: the same peer with only
    # 100 GB free is refused, where reclaimable would have had nothing to add.
    thin = head.replace("VMFREEGB 400.0 6.0", "VMFREEGB 100.0 6.0")
    with pytest.raises(fleet.HeavyRunActive, match="free_only"):
        fleet.require_headroom_box(None, 183.0, label="old peer thin",
                                   ps_runner=lambda host: thin,
                                   accounting="reclaimable")


def test_absolute_headroom_is_what_a_load_actually_needs():
    """A percentage is a weak predictor of "will this load fit".

    Measured after the incident: swap had drained to 2 GB and free was 11%,
    which passes both of those checks -- while 449 GB of a 512 GB box was wired
    and held by no visible process, leaving 54.7 GB for an 85.5 GiB shard.
    """
    text = _QUIET_HEAD + _MEM_NO_HEADROOM
    m = fleet._parse_memory(text)
    assert m["free_gb"] == 54.7 and m["wired_gb"] == 449.4
    assert m["swap_used_gb"] < fleet.MAX_SWAP_GB      # swap check passes
    assert m["free_pct"] >= fleet.MIN_FREE_PCT        # percentage passes
    with pytest.raises(fleet.HeavyRunActive, match="free < "):
        fleet.require_quiet_fleet(threshold_gb=20, label="x",
                                  ps_runner=lambda host: text)


def test_headroom_gate_names_the_wired_holder():
    text = _QUIET_HEAD + _MEM_NO_HEADROOM
    try:
        fleet.require_quiet_fleet(threshold_gb=20, ps_runner=lambda host: text)
        raise AssertionError("should have refused")
    except fleet.HeavyRunActive as e:
        assert "wired 449.4GB" in str(e)


def test_enough_headroom_passes():
    r = fleet.require_quiet_fleet(threshold_gb=20, ps_runner=lambda host: _QUIET)
    assert r["quiet"] is True
    assert r["pressure"]["localhost"]["free_gb"] == 180.0


# ------------------------------------------------------------- GPU liveness
def test_gpu_probe_is_a_subprocess_with_a_timeout(monkeypatch):
    """It has to be a subprocess: a wedged Metal device ignores signals, so an
    in-process probe would hang the very guard meant to detect the hang."""
    seen = {}

    class _R:
        returncode = 0

    def fake_run(cmd, **kw):
        seen["cmd"], seen["timeout"] = cmd, kw.get("timeout")
        return _R()

    monkeypatch.setattr(fleet.subprocess, "run", fake_run)
    assert fleet.gpu_responsive() is True
    assert seen["timeout"] is not None, "the probe must be bounded"
    assert "mlx.core" in " ".join(seen["cmd"])


def test_a_wedged_gpu_reports_not_responsive(monkeypatch):
    def boom(cmd, **kw):
        raise fleet.subprocess.TimeoutExpired(cmd, kw.get("timeout", 1))

    monkeypatch.setattr(fleet.subprocess, "run", boom)
    assert fleet.gpu_responsive() is False


def test_gpu_probe_targets_the_peer_over_ssh(monkeypatch):
    seen = {}

    class _R:
        returncode = 0

    monkeypatch.setattr(fleet.subprocess, "run",
                        lambda cmd, **kw: (seen.update(cmd=cmd), _R())[1])
    fleet.gpu_responsive("m3ms@10.0.0.2")
    assert seen["cmd"][0] == "ssh" and "m3ms@10.0.0.2" in seen["cmd"]


# ------------------------------------------------------- headroom for a load
def _mem(free_gb, wired_gb, pct=50):
    return (_QUIET_HEAD + f"""
===MEM===
total = 4096.00M  used = 0.00M  free = 4096.00M  (encrypted)
System-wide memory free percentage: {pct}%
VMFREEGB {free_gb} {wired_gb}
""")


def test_headroom_refuses_the_state_that_froze_the_box():
    """The 2026-09-01 14:31 freeze, encoded.

    A box at 118 GB free carrying ~103 GB of leaked wired debt PASSES a
    100 GB floor, then loads an 86 GiB shard and runs at ~32 GB.  Minutes later
    the peer's worker was terminated under memory pressure, the surviving rank
    wedged for its full 900 s timeout, and the box froze.  The floor was not
    wrong, it was incomplete: what matters is free >= load + margin.
    """
    with pytest.raises(fleet.HeavyRunActive, match="headroom"):
        fleet.require_headroom(fleet.SHARD_GB["tp"], label="lc arm",
                               ps_runner=lambda h: _mem(118.0, 102.9))


def test_headroom_allows_a_clean_box():
    r = fleet.require_headroom(fleet.SHARD_GB["tp"],
                               ps_runner=lambda h: _mem(400.0, 8.0))
    assert r["headroom_ok"] is True and r["load_gb"] == 86.0


def test_headroom_scales_with_the_load_not_a_fixed_floor():
    """200 GB free is plenty for a TP shard and not enough for a single-box
    model.  One floor cannot express both."""
    fleet.require_headroom(fleet.SHARD_GB["tp"],
                           ps_runner=lambda h: _mem(200.0, 8.0))
    with pytest.raises(fleet.HeavyRunActive):
        fleet.require_headroom(fleet.SHARD_GB["single"],
                               ps_runner=lambda h: _mem(200.0, 8.0))


def test_headroom_names_the_wired_debt_in_the_refusal():
    try:
        fleet.require_headroom(fleet.SHARD_GB["tp"],
                               ps_runner=lambda h: _mem(118.0, 102.9))
        raise AssertionError("should have refused")
    except fleet.HeavyRunActive as e:
        assert "already wired" in str(e)


# ------------------------------------------------------ sweep degradation
def test_debtwatch_stops_a_sweep_that_is_leaking():
    """A per-arm threshold cannot see a trend.

    Four lc arms were queued on 2026-09-01; arm 1 aborted and leaked ~94 GB,
    arm 2 passed the (level-based) gate into the reduced margin, and the box
    froze.  This watches the trend.
    """
    seq = iter([_mem(400.0, 8.0), _mem(300.0, 102.0)])
    w = fleet.DebtWatch(ps_runner=lambda h: next(seq))
    with pytest.raises(fleet.HeavyRunActive, match="growing|grown"):
        w.check("arm 2")


def test_debtwatch_allows_a_clean_sweep():
    seq = iter([_mem(400.0, 8.0), _mem(398.0, 9.0)])
    w = fleet.DebtWatch(ps_runner=lambda h: next(seq))
    r = w.check("arm 2")
    assert r["growth_gb"]["localhost"] <= 1.5


def test_single_box_size_tracks_the_batch():
    """SHARD_GB["single"] is the B=8 measurement, not a constant.

    Reproduces the six measured peaks to within a gibibyte, and pins the two
    that matter for a gate: at B=8 the table entry is right, and above it the
    table under-requests -- by 29 GiB at B=32, which is half of require_headroom's
    default margin handed back silently.
    """
    measured = {1: 173.9, 2: 175.3, 4: 177.7, 8: 183.2, 16: 193.5, 32: 211.9}
    for batch, peak in measured.items():
        assert abs(fleet.single_box_gb(batch) - peak) <= 1.0, batch
    assert abs(fleet.single_box_gb(8) - fleet.SHARD_GB["single"]) <= 1.0
    assert fleet.single_box_gb(32) - fleet.SHARD_GB["single"] > 25.0
    with pytest.raises(ValueError):
        fleet.single_box_gb(0)


def test_headroom_refuses_a_wide_batch_a_fixed_size_would_have_admitted():
    """The under-request, as a gate outcome rather than an arithmetic claim.

    265 GB free clears a 183 GiB "single" load plus the 60 GB margin, so the
    fixed size admits a B=32 arm.  The arm actually peaks at 212 GiB, which does
    not clear it -- and being wrong in that direction is how boxes freeze.
    """
    fleet.require_headroom(fleet.SHARD_GB["single"],
                           ps_runner=lambda h: _mem(265.0, 8.0))
    with pytest.raises(fleet.HeavyRunActive, match="headroom"):
        fleet.require_headroom(fleet.single_box_gb(32),
                               ps_runner=lambda h: _mem(265.0, 8.0))
