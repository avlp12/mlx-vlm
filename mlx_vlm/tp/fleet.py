"""Sequencing guard: never start a heavy run while another one is still resident.

WHY THIS FILE EXISTS.  On 2026-08-31 an HTTP end-to-end test was shutting down
and hung with its 183 GiB model still mapped; the next long-context TP load
started anyway, and the box froze hard enough to need a power cycle.  Nothing
about that was subtle -- two 190 GiB residents do not fit in 512 GiB -- and
nothing about it was detectable from inside either process.  It is a *fleet*
property, so the check has to look at the fleet.

The pattern is lifted from the EAGLE-3 bulk-capture wrapper, which had it
right::

    while [ -n "$(ps -A -o rss,command | awk '$1>20000000 && /[Pp]ython/')" ];
    do sleep 30; done

That is: refuse (or wait) while any python process holds more than 20 GB RSS.
This module is the same rule, made reusable, made to cover the peer box, and
made to return a receipt instead of a silence -- so a driver that skipped the
wait can be told apart from one that waited and found the fleet quiet.

Usable from a shell driver too::

    python -m mlx_vlm.tp.fleet --wait --hosts 10.0.0.1,10.0.0.2 --label myrun
"""
from __future__ import annotations

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
    "HEAVY_RSS_GB",
    "heavy_processes",
    "require_quiet_fleet",
    "self_rss_gb",
    "wait_for_quiet",
]

# 20 GB: well above any tokenizer/driver/pytest process, well below any shard
# (85 GiB TP) or whole model (183 GiB single-box).  The gap is two orders of
# magnitude, so the threshold does not need tuning.
HEAVY_RSS_GB = float(os.environ.get("MLX_VLM_TP_HEAVY_RSS_GB", "20"))

# ``ps`` reports RSS in kibibytes on macOS.
_KIB = 1024.0
_GB = 1024.0 ** 3


class HeavyRunActive(RuntimeError):
    """A heavy run is still resident somewhere on the fleet."""


def _run_ps(host: Optional[str] = None, timeout: float = 20.0) -> str:
    cmd = ["ps", "-A", "-o", "rss=,pid=,command="]
    if host:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
               "ps -A -o rss=,pid=,command="]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise HeavyRunActive(
            f"cannot inspect {host or 'localhost'}: rc={out.returncode} "
            f"{out.stderr.strip()[:200]}. Refusing rather than assuming quiet.")
    return out.stdout


def _parse_ps(text: str, threshold_gb: float, ignore_pids: Sequence[int] = ()) \
        -> List[Dict[str, object]]:
    heavy = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            rss_kib, pid = float(parts[0]), int(parts[1])
        except ValueError:
            continue
        gb = rss_kib * _KIB / _GB
        if gb < threshold_gb or pid in ignore_pids:
            continue
        heavy.append({"pid": pid, "rss_gb": round(gb, 1), "cmd": parts[2][:160]})
    return heavy


def self_rss_gb(pid: Optional[int] = None) -> float:
    """RSS of one process in GB, or 0.0 if it is gone.

    Used by the server teardown to *prove* the model was released rather than
    assert it: a shutdown that logs "unloaded" while RSS is unchanged is the
    exact failure this campaign already paid for once.
    """
    pid = os.getpid() if pid is None else int(pid)
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return 0.0
    raw = out.stdout.strip()
    if not raw:
        return 0.0
    try:
        return float(raw.split()[0]) * _KIB / _GB
    except (ValueError, IndexError):
        return 0.0


def heavy_processes(
    hosts: Sequence[str] = (),
    threshold_gb: float = HEAVY_RSS_GB,
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
    threshold_gb: float = HEAVY_RSS_GB,
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
            f"{h}: " + ", ".join(f"pid {p['pid']} {p['rss_gb']}GB {p['cmd']}"
                                 for p in procs)
            for h, procs in busy.items())
        raise HeavyRunActive(
            f"refusing to start {label or 'this run'}: a previous heavy run is "
            f"still resident ({detail}). Verify its teardown before loading "
            f"again -- two residents of this size do not fit.")
    return receipt


def wait_for_quiet(
    hosts: Sequence[str] = (),
    threshold_gb: float = HEAVY_RSS_GB,
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
    ap.add_argument("--threshold-gb", type=float, default=HEAVY_RSS_GB)
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
