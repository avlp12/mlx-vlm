"""Sequencing guard: never start a heavy run while another one is still resident.

WHY THIS FILE EXISTS.  On 2026-08-31 an HTTP end-to-end test was shutting down
and hung with its 183 GiB model still mapped; the next long-context TP load
started anyway, and the box froze hard enough to need a power cycle.  Nothing
about that was subtle -- two 190 GiB residents do not fit in 512 GiB -- and
nothing about it was detectable from inside either process.  It is a *fleet*
property, so the check has to look at the fleet.

WHY IT DOES NOT USE RSS.  The obvious implementation, and the one the EAGLE-3
bulk-capture wrapper uses::

    while [ -n "$(ps -A -o rss,command | awk '$1>20000000 && /[Pp]ython/')" ];
    do sleep 30; done

cannot work, and measuring says so: 8 GiB of live ``mx.array`` moves this
process's RSS by 0.01 GB.  MLX allocates Metal buffers, and macOS does not
count them in ``resident_size``.  A 85.5 GiB shard reports about 3 GB of RSS.
So an RSS threshold of any value is a guard that never fires -- which is
consistent with the freeze having happened at all.

What does track it is the task's **physical footprint**: ``footprint(1)``,
``vmmap --summary``, ``top``'s MEM column and ``proc_pid_rusage``'s
``ri_phys_footprint`` all reported 8.0-8.5 GB for that same allocation.  This
module reads ``top`` for the fleet scan (one call per box, sorted by memory,
so the big processes are the cheap ones to find) and ``proc_pid_rusage`` for
"how big am I right now", which needs no subprocess.

Usable from a shell driver too::

    python -m mlx_vlm.tp.fleet --wait --hosts 10.0.0.1,10.0.0.2 --label myrun
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import socket
import subprocess
import time
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "HeavyRunActive",
    "HEAVY_FOOTPRINT_GB",
    "heavy_processes",
    "phys_footprint_gb",
    "require_quiet_fleet",
    "self_rss_gb",
    "wait_for_quiet",
]

# 20 GB: well above any tokenizer/driver/pytest process, well below any shard
# (85 GiB TP) or whole model (183 GiB single-box).  The gap is two orders of
# magnitude, so the threshold does not need tuning.
HEAVY_FOOTPRINT_GB = float(os.environ.get("MLX_VLM_TP_HEAVY_RSS_GB", "20"))
HEAVY_RSS_GB = HEAVY_FOOTPRINT_GB          # back-compat alias

_GB = 1024.0 ** 3
_PS_SEP = "===PS==="
# One call per box: ``top`` for the numbers (its COMMAND column is truncated to
# 16 characters, which is not enough to tell one python from another) and
# ``ps`` for the full command lines, joined on pid.
# The separator is single-quoted: the peer's login shell is zsh, where a bare
# ``===PS===`` triggers equals-expansion and the probe fails with
# "==PS=== not found" -- which the guard then correctly reports as "cannot
# inspect", but for the wrong reason.
_PROBE = ("/usr/bin/top -l 1 -o mem -n 40 -stats pid,mem; "
          f"echo '{_PS_SEP}'; /bin/ps -A -o pid=,command=")


class HeavyRunActive(RuntimeError):
    """A heavy run is still resident somewhere on the fleet."""


def _run_ps(host: Optional[str] = None, timeout: float = 30.0) -> str:
    cmd = ["/bin/sh", "-c", _PROBE]
    if host:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
               _PROBE]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0 or _PS_SEP not in out.stdout:
        raise HeavyRunActive(
            f"cannot inspect {host or 'localhost'}: rc={out.returncode} "
            f"{out.stderr.strip()[:200]}. Refusing rather than assuming quiet.")
    return out.stdout


_UNITS = {"B": 1.0, "K": 1024.0, "M": 1024.0 ** 2, "G": 1024.0 ** 3,
          "T": 1024.0 ** 4}


def _parse_mem(token: str) -> Optional[float]:
    """``top``'s MEM cell -> GB.  ``838M``, ``12G``, ``1234K``, ``8211M+``."""
    tok = token.strip().rstrip("+-")
    if not tok:
        return None
    unit = tok[-1].upper()
    if unit in _UNITS:
        tok, mult = tok[:-1], _UNITS[unit]
    else:
        mult = 1.0
    try:
        return float(tok) * mult / _GB
    except ValueError:
        return None


def _parse_ps(text: str, threshold_gb: float, ignore_pids: Sequence[int] = ()) \
        -> List[Dict[str, object]]:
    """Heavy processes from a combined ``top`` + ``ps`` probe."""
    top_text, _, ps_text = text.partition(_PS_SEP)
    names: Dict[int, str] = {}
    for line in ps_text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            names[int(parts[0])] = parts[1]

    heavy, in_table = [], False
    for line in top_text.splitlines():
        cells = line.split()
        if not cells:
            continue
        if cells[0] == "PID":
            in_table = True
            continue
        if not in_table or not cells[0].isdigit() or len(cells) < 2:
            continue
        pid = int(cells[0])
        gb = _parse_mem(cells[1])
        if gb is None or gb < threshold_gb or pid in ignore_pids:
            continue
        heavy.append({"pid": pid, "footprint_gb": round(gb, 1),
                      "rss_gb": round(gb, 1),   # back-compat key
                      "cmd": names.get(pid, "?")[:160]})
    return heavy


class _RUsageInfoV2(ctypes.Structure):
    """``rusage_info_v2`` from <libproc.h>: the first fields are stable."""

    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("_pad", ctypes.c_uint64 * 32),
    ]


def phys_footprint_gb(pid: Optional[int] = None) -> float:
    """This (or another) process's physical footprint in GB, 0.0 if gone.

    The number a shutdown has to move.  ``resident_size`` does not: MLX's Metal
    buffers are simply not in it, so a teardown that logs "unloaded" while RSS
    is unchanged looks identical to one that worked -- and also to one that did
    not.  ``ri_phys_footprint`` is what ``footprint(1)`` and Activity Monitor
    report, and it tracked an 8 GiB ``mx.array`` to within 0.5%.
    """
    pid = os.getpid() if pid is None else int(pid)
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
        buf = _RUsageInfoV2()
        if libc.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(2),
                                ctypes.byref(buf)) != 0:
            return 0.0
        return buf.ri_phys_footprint / _GB
    except (OSError, AttributeError, ValueError):
        return 0.0


def self_rss_gb(pid: Optional[int] = None) -> float:
    """Kept for callers; returns the physical footprint, not ``resident_size``.

    The name is a historical accident and the number is the correct one: see
    :func:`phys_footprint_gb` for why RSS is the wrong question on this stack.
    """
    return phys_footprint_gb(pid)


def heavy_processes(
    hosts: Sequence[str] = (),
    threshold_gb: float = HEAVY_FOOTPRINT_GB,
    ignore_pids: Sequence[int] = (),
    ps_runner: Optional[Callable[[Optional[str]], str]] = None,
) -> Dict[str, List[Dict[str, object]]]:
    """{host: [heavy process records]} for this box plus every named peer.

    ``hosts`` are addresses on the fast link; the local one is recognised and
    inspected without ssh.  ``ps_runner`` exists so the policy above can be
    tested without a fleet.
    """
    runner = ps_runner or _run_ps
    local_names = {"", "localhost", "127.0.0.1", socket.gethostname()}
    try:
        local_names |= set(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    targets: List[Optional[str]] = [None]
    for h in hosts:
        addr = h.split("@")[-1]
        if addr not in local_names:
            targets.append(h)
    found: Dict[str, List[Dict[str, object]]] = {}
    for t in targets:
        found[t or "localhost"] = _parse_ps(runner(t), threshold_gb, ignore_pids)
    return found


def require_quiet_fleet(
    hosts: Sequence[str] = (),
    threshold_gb: float = HEAVY_FOOTPRINT_GB,
    label: str = "",
    ignore_pids: Sequence[int] = (),
    ps_runner: Optional[Callable[[Optional[str]], str]] = None,
) -> dict:
    """Raise unless every named box is free of heavy python residents.

    Returns a receipt.  The receipt is the point: "the guard ran and the fleet
    was quiet" and "nobody called the guard" must not look the same afterwards.
    """
    found = heavy_processes(hosts, threshold_gb, ignore_pids, ps_runner)
    busy = {h: procs for h, procs in found.items() if procs}
    receipt = {
        "checked": sorted(found),
        "threshold_gb": threshold_gb,
        "label": label,
        "when": time.strftime("%FT%T%z"),
        "quiet": not busy,
        "busy": busy,
    }
    if busy:
        detail = "; ".join(
            f"{h}: " + ", ".join(f"pid {p['pid']} {p['footprint_gb']}GB {p['cmd']}"
                                 for p in procs)
            for h, procs in busy.items())
        raise HeavyRunActive(
            f"refusing to start {label or 'this run'}: a previous heavy run is "
            f"still resident ({detail}). Verify its teardown before loading "
            f"again -- two residents of this size do not fit.")
    return receipt


def wait_for_quiet(
    hosts: Sequence[str] = (),
    threshold_gb: float = HEAVY_FOOTPRINT_GB,
    label: str = "",
    timeout_s: float = 1800.0,
    poll_s: float = 30.0,
    ignore_pids: Sequence[int] = (),
    ps_runner: Optional[Callable[[Optional[str]], str]] = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> dict:
    """Block until the fleet is quiet, then return the receipt.

    The queueing form of :func:`require_quiet_fleet`, for a driver that is happy
    to wait its turn (the bulk-capture wrapper's behaviour).  It still fails
    loudly on timeout: waiting forever and starting anyway are both worse than
    saying which process is in the way.
    """
    started = now()
    waited = 0
    while True:
        try:
            receipt = require_quiet_fleet(
                hosts, threshold_gb, label, ignore_pids, ps_runner)
            receipt["waited_s"] = round(now() - started, 1)
            receipt["polls"] = waited
            return receipt
        except HeavyRunActive as e:
            if now() - started >= timeout_s:
                raise HeavyRunActive(
                    f"{e} (waited {timeout_s:.0f}s)") from None
            if waited == 0:
                logger.info("fleet busy; waiting for it to go quiet: %s", e)
            waited += 1
            sleep(poll_s)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hosts", default=os.environ.get("MLX_VLM_GLM5_TP_HOSTS", ""))
    ap.add_argument("--threshold-gb", type=float, default=HEAVY_FOOTPRINT_GB)
    ap.add_argument("--label", default="")
    ap.add_argument("--wait", action="store_true",
                    help="queue behind the running job instead of refusing")
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    a = ap.parse_args(argv)
    hosts = [h.strip() for h in a.hosts.split(",") if h.strip()]
    fn = wait_for_quiet if a.wait else require_quiet_fleet
    kw = {"timeout_s": a.timeout_s} if a.wait else {}
    try:
        receipt = fn(hosts, a.threshold_gb, a.label, **kw)
    except HeavyRunActive as e:
        print(json.dumps({"quiet": False, "error": str(e)}, indent=1))
        return 75  # EX_TEMPFAIL: try again later
    print(json.dumps(receipt, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
