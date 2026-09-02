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

import _thread
import ctypes
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "HeavyRunActive",
    "HEAVY_FOOTPRINT_GB",
    "MAX_SWAP_GB",
    "SWAP_CEIL_GB",
    "MIN_FREE_GB",
    "MIN_FREE_PCT",
    "GpuWedged",
    "gpu_responsive",
    "heavy_processes",
    "memory_snapshot",
    "phys_footprint_gb",
    "SHARD_GB",
    "SINGLE_BOX_BASE_GB",
    "SINGLE_BOX_PER_ROW_GB",
    "single_box_gb",
    "require_headroom",
    "require_headroom_box",
    "require_quiet_box",
    "require_quiet_fleet",
    "self_rss_gb",
    "wait_for_quiet",
]

# 20 GB: well above any tokenizer/driver/pytest process, well below any shard
# (85 GiB TP) or whole model (183 GiB single-box).  The gap is two orders of
# magnitude, so the threshold does not need tuning.
HEAVY_FOOTPRINT_GB = float(os.environ.get("MLX_VLM_TP_HEAVY_RSS_GB", "20"))
HEAVY_RSS_GB = HEAVY_FOOTPRINT_GB          # back-compat alias

# A second resident is not the only way to wedge the box, and on 2026-09-01 it
# was not the way that happened: one 86 GiB shard on a 512 GB machine drove
# swap to 39.9 of 42 GB with 6% free, and the run died with a Metal
# "Insufficient Memory" while the box went unresponsive.  Nothing was
# co-resident -- the pressure had accumulated across the night's repeated
# load/unload cycles.  A guard that only counts big processes cannot see that,
# so it also has to look at the state those processes leave behind.
MAX_SWAP_GB = float(os.environ.get("MLX_VLM_TP_MAX_SWAP_GB", "8"))
MIN_FREE_PCT = float(os.environ.get("MLX_VLM_TP_MIN_FREE_PCT", "10"))
# The check that would actually have stopped the incident.  A percentage is a
# weak predictor of "will this load fit": what matters is whether the free
# pages exceed what the shard is about to ask for.  Measured 2026-09-01 on
# gesicht: 449 GB of a 512 GB box was wired and held by no visible process,
# leaving 55 GB free -- and an 86 GiB shard was loaded into it anyway, which is
# how the box reached 95% swap.  Default 100 GB covers an 85.5 GiB TP shard
# plus working space; raise it for a single-box 183 GiB load.
MIN_FREE_GB = float(os.environ.get("MLX_VLM_TP_MIN_FREE_GB", "100"))
# Typical resident sizes on this fleet, so a caller can say what it is about to
# load instead of guessing a floor.
SHARD_GB = {"tp": 86.0, "single": 183.0}

# How require_headroom_box() counts headroom.
#
#   free_only    -- vm_stat "Pages free" alone.  The original rule.  Safe, and
#                   wrong often enough to matter: after a box has read a 171 GB
#                   model once, the page cache holds nearly all of RAM as
#                   INACTIVE file-backed pages, "free" reads ~40 GB, and the
#                   gate refuses a load the box can trivially serve by evicting
#                   clean cache.
#   reclaimable  -- free + min(inactive, file-backed).  Clean file-backed pages
#                   are evictable without swapping, so they are headroom.
#
# The default is reclaimable, ratified 2026-09-02 after a validation load on
# gesicht: free_only said 0.3 GB and refused; reclaimable said 487.0 GB and
# passed; the load then completed in 26.2 SECONDS to a 169.03 GB footprint
# without swap, because the model was already sitting in the page cache that
# free_only refuses to count.  The WATCHED-LOAD clause below is mandatory under
# it and is always on: "reclaimable" is a claim about the future, so it is
# checked while the load happens rather than asserted before it.
GATE_ACCOUNTING = os.environ.get("MLX_VLM_FLEET_GATE_ACCOUNTING", "reclaimable")
# Abort floor for a watched load.
#
# DEVIATION, FLAGGED FOR RATIFICATION: the floor was specified as "free < 20 GB".
# On a box that has read a 171 GB model, "Pages free" is STRUCTURALLY near zero
# -- macOS holds the page cache at whatever size fits and reclaims on demand.
# Measured on gesicht immediately before the validation run: free 2.7 GB,
# inactive 479.7 GB, swap 0.  A literal free-based floor fires instantly on a
# perfectly healthy box, so it cannot be the trigger.
#
# The floor therefore reads the SAME quantity the gate now counts -- free plus
# reclaimable clean file-backed pages.  That is the faithful translation of the
# intent ("the load is losing the race"), because when a load really is losing,
# the reclaimable pool is what collapses.  Swap GROWTH during the load is the
# other primary trigger; a static swap residue is not (see SWAP_CEIL_GB).
# RATIFIED 2026-09-02; the DEVIATION note above is kept because the reasoning is
# the justification, not just history.
WATCH_FLOOR_GB = float(os.environ.get("MLX_VLM_FLEET_WATCH_FLOOR_GB", "20"))
WATCH_FREE_FLOOR_GB = WATCH_FLOOR_GB  # back-compat alias
WATCH_POLL_S = float(os.environ.get("MLX_VLM_FLEET_WATCH_POLL_S", "5"))
# Static swap the gate tolerates.
#
# The ratified rule said swap == 0.  That was stricter than the campaign's own
# standard -- playbook §3.2 says "swap <= threshold" and the harnesses have used
# SWAP_CEIL_GB=4 throughout -- and it bit: after ten sequential 183 GB server
# loads gesicht held 0.20 GB of STALE swap while sitting on ~476 GB of headroom
# (free 82.5 + inactive 393.5), stable over 60 s with no pressure, and the gate
# refused every lane behind it.
#
# A static residue is not a pressure signal.  macOS leaves pages in the swapfile
# after the pressure that wrote them is long gone and never proactively faults
# them back.  The signal that actually predicted the I876/I877 freezes was swap
# GROWING during a load, which WatchedLoad now watches for directly -- so the
# static ceiling can be generous without giving anything up.
SWAP_CEIL_GB = float(os.environ.get("MLX_VLM_FLEET_SWAP_CEIL_GB", "1.0"))
# Growth during a watched load that means the box is losing.  Small, because
# this is a derivative: any sustained growth at all is the failure mode.
SWAP_GROWTH_ABORT_GB = float(
    os.environ.get("MLX_VLM_FLEET_SWAP_GROWTH_GB", "0.05"))


def gate_accountings(mem: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Both headroom numbers for one box, so a receipt can show what each rule
    would have said.  Returns None for an accounting the probe cannot support
    (an older peer sends no inactive/file-backed fields)."""
    free = mem.get("free_gb")
    inact, fileb = mem.get("inactive_gb"), mem.get("file_backed_gb")
    recl = None
    if free is not None and inact is not None and fileb is not None:
        recl = round(free + min(inact, fileb), 1)
    return {"free_only": free, "reclaimable": recl}

# ``single`` is one operating point, not a constant, and treating it as a
# constant is a way to under-request.  A single-box decode arm's peak grows with
# the batch it serves, because the KDA recurrent state (B*H*D*D*4 per layer), the
# KV cache and the prompt logits are all per-row.  Measured on the 320B tree
# (receipts logs/tp2/kda_bench_*_202609011436.json, 2026-09-01):
#
#     B      1      2      4      8     16     32
#   GiB  173.9  175.3  177.7  183.2  193.5  211.9
#
# which is linear in B to within 0.82 GiB:  peak_GiB = 173.0 + 1.23 * B.
# 183.0 -- the number in the table above -- is that line at B = 8.1, so the
# entry is right for the batch it was taken at and wrong everywhere else: it
# over-requests below B=8 and under-requests above it, by 9.7 GiB at B=16, 29.4
# at B=32 and 68.7 at B=64.  require_headroom is only as good as the size it is
# handed, so callers serving a batch should hand it this instead.
SINGLE_BOX_BASE_GB = 173.0
SINGLE_BOX_PER_ROW_GB = 1.23


def single_box_gb(batch: int) -> float:
    """Peak resident size, in GiB, of a single-box decode arm serving ``batch``.

    ``batch`` is required rather than defaulted: the whole point is that the
    answer depends on it, and a default would reintroduce the constant this
    replaces.  Prefill sets the peak, not decode, so this assumes the campaign's
    512-token prompt; a much longer one moves the intercept, not the slope.

    Two things this line does NOT cover, both measured 2026-09-01: enlarged MLX
    command buffers (MLX_MAX_MB_PER_BUFFER=2048) add ~39 GiB at B=16 -- 232.7
    GiB observed against 193.5 without -- and the fit is drawn from fused-KDA
    runs on one model at one quantization.  Size a load with either of those
    changed by measuring it, not by extrapolating this.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    return round(SINGLE_BOX_BASE_GB + SINGLE_BOX_PER_ROW_GB * batch, 1)

_GB = 1024.0 ** 3
_PS_SEP = "===PS==="
# One call per box: ``top`` for the numbers (its COMMAND column is truncated to
# 16 characters, which is not enough to tell one python from another) and
# ``ps`` for the full command lines, joined on pid.
# The separator is single-quoted: the peer's login shell is zsh, where a bare
# ``===PS===`` triggers equals-expansion and the probe fails with
# "==PS=== not found" -- which the guard then correctly reports as "cannot
# inspect", but for the wrong reason.
_MEM_SEP = "===MEM==="
_PROBE = ("/usr/bin/top -l 1 -o mem -n 40 -stats pid,mem; "
          f"echo '{_PS_SEP}'; /bin/ps -A -o pid=,command=; "
          f"echo '{_MEM_SEP}'; /usr/sbin/sysctl -n vm.swapusage; "
          "/usr/bin/memory_pressure -Q 2>/dev/null | "
          "/usr/bin/grep -i 'free percentage'; "
          # Free and wired in GB.  vm_stat is the only place the *absolute*
          # headroom is visible, and absolute headroom is what a load needs.
          "/usr/bin/vm_stat | /usr/bin/awk '/Pages free/{f=$3} "
          "/Pages wired down/{w=$4} /Pages inactive/{i=$3} "
          "/File-backed pages/{fb=$3} END{gsub(/[.]/,\"\",f);gsub(/[.]/,\"\",w);"
          "gsub(/[.]/,\"\",i);gsub(/[.]/,\"\",fb);"
          "printf \"VMFREEGB %.1f %.1f %.1f %.1f\\n\", f*16384/2^30, "
          "w*16384/2^30, i*16384/2^30, fb*16384/2^30}'")


class LoadAborted(RuntimeError):
    """A watched load was stopped because the box started losing headroom."""


class WatchedLoad:
    """Poll local memory during a load and abort if the box starts to lose.

    Mandatory under GATE_ACCOUNTING == "reclaimable".  The reclaimable rule
    promises that clean file-backed pages will be evicted rather than swapped;
    this is what checks that the promise is being kept, while it is being kept
    or broken, instead of asserting it beforehand.

    Abort is by ``interrupt_main()``, which raises KeyboardInterrupt in the main
    thread and unwinds through every context manager on the way out -- so the
    wired_limit exits and the streams are cleared.  A bare SIGTERM to a process
    holding a shard leaks wired memory that only a reboot returns, so it is
    never used here.
    """

    def __init__(self, label: str = "", floor_gb: float = None,
                 poll_s: float = None, expect: Optional[dict] = None):
        self.label = label
        self.floor_gb = WATCH_FLOOR_GB if floor_gb is None else floor_gb
        self.poll_s = WATCH_POLL_S if poll_s is None else poll_s
        self.expect = expect
        self.trajectory: List[dict] = []
        self.swap_base: Optional[float] = None   # set at __enter__
        self.fired: Optional[str] = None
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    @staticmethod
    def sample() -> dict:
        out = {"t": round(time.time(), 2)}
        try:
            vm = subprocess.run(["/usr/bin/vm_stat"], capture_output=True,
                                text=True, timeout=10).stdout
            g = 16384 / _GB
            for line, key in (("Pages free", "free_gb"),
                              ("Pages wired down", "wired_gb"),
                              ("Pages inactive", "inactive_gb"),
                              ("File-backed pages", "file_backed_gb")):
                for ln in vm.splitlines():
                    if ln.startswith(line):
                        n = float(ln.rsplit(":", 1)[1].strip().rstrip("."))
                        out[key] = round(n * g, 1)
                        break
            sw = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                                capture_output=True, text=True, timeout=10).stdout
            used = sw.split("used =")[1].split()[0]
            mult = _UNITS.get(used[-1].upper(), 1.0)
            out["swap_used_gb"] = round(float(used[:-1]) * mult / _GB, 2)
        except Exception as exc:                       # a probe failure is not
            out["probe_error"] = str(exc)[:120]        # a reason to kill a load
        return out

    def _run(self):
        while not self._stop.wait(self.poll_s):
            s = self.sample()
            self.trajectory.append(s)
            swap = s.get("swap_used_gb")
            free, inact = s.get("free_gb"), s.get("inactive_gb")
            fileb = s.get("file_backed_gb")
            avail = None
            if free is not None and inact is not None and fileb is not None:
                avail = free + min(inact, fileb)
                s["available_gb"] = round(avail, 1)
            why = None
            growth = None
            if swap is not None and self.swap_base is not None:
                growth = swap - self.swap_base
                s["swap_growth_gb"] = round(growth, 3)
            if growth is not None and growth > SWAP_GROWTH_ABORT_GB:
                # THE signal. A static residue is tolerated by the gate; swap
                # that GROWS while we load is the box losing the race, and it is
                # what the I876/I877 freezes actually looked like.
                why = (f"swap grew {growth:.2f}GB during the load "
                       f"({self.swap_base:.2f} -> {swap:.2f}GB), over the "
                       f"{SWAP_GROWTH_ABORT_GB:.2f}GB abort threshold")
            elif avail is not None and avail < self.floor_gb:
                why = (f"available (free {free:.1f} + reclaimable "
                       f"{min(inact, fileb):.1f}) fell to {avail:.1f}GB, "
                       f"below the {self.floor_gb:.0f}GB floor")
            if why:
                self.fired = why
                logging.error("[watched-load] ABORTING %s: %s", self.label, why)
                _thread.interrupt_main()
                return

    def __enter__(self):
        before = self.sample()
        # Baseline for the growth clause. Whatever is already in the swapfile is
        # not this load's doing; only what we add counts against us.
        self.swap_base = before.get("swap_used_gb")
        self.trajectory.append(dict(before, phase="before"))
        self._t = threading.Thread(target=self._run, daemon=True,
                                   name="watched-load")
        self._t.start()
        return self

    def __exit__(self, et, ev, tb):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=self.poll_s + 5)
        self.trajectory.append(dict(self.sample(), phase="after"))
        if self.fired and et is KeyboardInterrupt:
            raise LoadAborted(f"{self.label}: {self.fired}") from ev
        return False

    def receipt(self) -> dict:
        return {"label": self.label, "floor_gb": self.floor_gb,
                "poll_s": self.poll_s, "expected": self.expect,
                "swap_base_gb": self.swap_base,
                "swap_growth_abort_gb": SWAP_GROWTH_ABORT_GB,
                "fired": self.fired, "trajectory": self.trajectory}


class HeavyRunActive(RuntimeError):
    """A heavy run is still resident somewhere on the fleet."""


class GpuWedged(RuntimeError):
    """The Metal device is not executing work, so a load would hang."""


# A 4x4 add.  If THIS does not come back, nothing will.
_GPU_PROBE = "import mlx.core as mx; mx.eval(mx.ones((4,4))+1)"


def gpu_responsive(host: Optional[str] = None, timeout_s: float = 25.0,
                   python: Optional[str] = None) -> bool:
    """Does the Metal device still execute a trivial kernel?

    Memory is not the only way a box goes unusable.  On 2026-09-01 gesicht
    reached a state where ``mx.eval(mx.ones((4,4)) + 1)`` never returned, with
    302 GB free and nothing resident -- the device was wedged, most likely by
    the session's repeated aborted runs, in the same way the jaccl fabric had
    wedged earlier.  Every memory check passed and a load into that box would
    simply have hung.

    Run in a SUBPROCESS with a timeout, because the whole point is that the
    operation cannot be interrupted in-process: a wedged device ignores signals
    (see tp.transport.Deadman).  Killing a fresh child costs nothing.
    """
    py = python or sys.executable
    cmd = [py, "-c", _GPU_PROBE]
    if host:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
               f"{py} -c {_GPU_PROBE!r}"]
    try:
        return subprocess.run(cmd, capture_output=True,
                              timeout=timeout_s).returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except (OSError, subprocess.SubprocessError):
        return False


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


def _parse_memory(text: str) -> Dict[str, Optional[float]]:
    """swap used (GB) and system-wide free percentage from the probe tail."""
    out: Dict[str, Optional[float]] = {"swap_used_gb": None, "free_pct": None,
                                       "free_gb": None, "wired_gb": None,
                                       "inactive_gb": None,
                                       "file_backed_gb": None}
    out["free_gb"] = None
    _, _, mem = text.partition(_MEM_SEP)
    for line in mem.splitlines():
        low = line.lower()
        if "used =" in low and "total =" in low:
            try:
                after = line.split("used")[1]
                val = after.split("=")[1].split()[0]
                unit = val[-1].upper()
                num = float(val[:-1]) if unit in _UNITS else float(val)
                mult = _UNITS.get(unit, 1.0)
                out["swap_used_gb"] = round(num * mult / _GB, 2)
            except (IndexError, ValueError):
                pass
        elif line.strip().startswith("VMFREEGB"):
            f = line.split()
            try:
                out["free_gb"] = round(float(f[1]), 1)
                out["wired_gb"] = round(float(f[2]), 1)
            except (IndexError, ValueError):
                pass
            # Fields 3 and 4 are newer than fields 1 and 2; a peer running an
            # older build sends only two, so these stay None and the reclaimable
            # accounting refuses to apply rather than guessing.
            try:
                out["inactive_gb"] = round(float(f[3]), 1)
                out["file_backed_gb"] = round(float(f[4]), 1)
            except (IndexError, ValueError):
                pass
        elif "free percentage" in low:
            try:
                out["free_pct"] = float(low.split(":")[1].strip().rstrip("%"))
            except (IndexError, ValueError):
                pass
    return out


def memory_snapshot(host: Optional[str] = None,
                    ps_runner: Optional[Callable[[Optional[str]], str]] = None
                    ) -> Dict[str, Optional[float]]:
    """Swap + free-percentage for one box, for a harness receipt.

    Heavy harnesses should record this before and after: "the run died" and
    "the run died with the box at 95% swap" are different findings, and only
    the second one tells you what to change.
    """
    runner = ps_runner or _run_ps
    return _parse_memory(runner(host))


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


class DebtWatch:
    """Abort a multi-arm sweep when the boxes are losing memory to it.

    Every watchdog abort leaks roughly a shard, permanently, so a sweep of N
    arms runs with a monotonically shrinking margin -- and a per-arm threshold
    cannot see that, because each individual check only asks "is there enough
    right now".  On 2026-09-01 four lc arms were queued, arm 1 aborted and leaked
    ~94 GB, arm 2 was approved into the reduced margin, and the box froze.

    This watches the trend instead of the level: record wired at sweep start,
    and refuse to start another arm once it has grown by more than one shard.
    """

    def __init__(self, hosts: Sequence[str] = (), tolerance_gb: float = 90.0,
                 ps_runner: Optional[Callable[[Optional[str]], str]] = None):
        self.hosts = tuple(hosts)
        self.tolerance_gb = tolerance_gb
        self._runner = ps_runner
        self.baseline = self._wired()

    def _wired(self) -> Dict[str, Optional[float]]:
        runner = self._runner or _run_ps
        out: Dict[str, Optional[float]] = {}
        targets: List[Optional[str]] = [None] + [h for h in self.hosts
                                                 if h not in ("10.0.0.1",)]
        for t in targets:
            out[t or "localhost"] = _parse_memory(runner(t)).get("wired_gb")
        return out

    def growth(self) -> Dict[str, float]:
        now = self._wired()
        return {h: round((now.get(h) or 0.0) - (v or 0.0), 1)
                for h, v in self.baseline.items()}

    def check(self, label: str = "") -> dict:
        # Poll once: two polls per check would double the ssh cost and could
        # disagree with each other.
        growth = self.growth()
        grown = {h: g for h, g in growth.items() if g > self.tolerance_gb}
        if grown:
            raise HeavyRunActive(
                f"stopping the sweep before {label or 'the next arm'}: wired "
                f"memory has grown since it started ("
                + "; ".join(f"{h} +{g:.0f}GB" for h, g in grown.items())
                + f", tolerance {self.tolerance_gb:.0f}GB). Something is leaking "
                f"a shard per arm; each further arm starts with less margin than "
                f"the gate assumes.")
        return {"baseline_wired_gb": self.baseline, "growth_gb": growth}


def require_headroom(load_gb: float, hosts: Sequence[str] = (),
                     margin_gb: float = 60.0, label: str = "",
                     ps_runner: Optional[Callable[[Optional[str]], str]] = None
                     ) -> dict:
    """Refuse unless free memory exceeds what this load will ASK FOR, plus room.

    THE LESSON THIS ENCODES.  ``require_quiet_fleet`` tests free memory against
    a fixed floor *before* a load, which says nothing about what is left
    *after* it.  With MIN_FREE_GB=100 and an 86 GiB shard, a box at 118 GB free
    passes the gate and then runs at ~32 GB -- and on 2026-09-01 that is exactly
    what happened: the gate approved a load onto a box already carrying ~100 GB
    of leaked debt, the peer's worker was terminated under memory pressure
    minutes later, the surviving rank wedged for its full 900 s timeout, and
    the box froze.  I had written down that this check was "too tight in
    hindsight" the previous night and had not changed it.

    The floor is not wrong, it is incomplete: what matters is
    ``free >= load + margin``.  Callers pass the size of the thing they are
    about to load (SHARD_GB has the usual ones), so the guard scales with the
    request instead of assuming one.
    """
    receipt = require_quiet_fleet(hosts, label=label or f"{load_gb:.0f}GB load",
                                  ps_runner=ps_runner)
    need = load_gb + margin_gb
    short = {}
    for host, m in (receipt.get("pressure") or {}).items():
        free = m.get("free_gb")
        if free is not None and free < need:
            short[host] = (f"{free:.1f}GB free, needs {need:.0f}GB "
                           f"({load_gb:.0f} load + {margin_gb:.0f} margin"
                           + (f", {m['wired_gb']:.0f} already wired)"
                              if m.get("wired_gb") else ")"))
    receipt.update({"load_gb": load_gb, "margin_gb": margin_gb,
                    "headroom_ok": not short, "short": short})
    if short:
        receipt["quiet"] = False
        raise HeavyRunActive(
            f"refusing {label or 'this load'}: not enough headroom for what it "
            f"will allocate (" + "; ".join(f"{h}: {w}" for h, w in short.items())
            + "). A floor check passes here and the box still runs out.")
    return receipt


def require_headroom_box(host: Optional[str], load_gb: float,
                         margin_gb: float = 60.0, label: str = "",
                         ps_runner: Optional[Callable[[Optional[str]], str]] = None,
                         threshold_gb: float = HEAVY_FOOTPRINT_GB,
                         accounting: Optional[str] = None) -> dict:
    """Headroom + quiet check scoped to ONE box, ignoring every other machine.

    require_headroom() folds in localhost because a TP load needs both boxes.
    That makes it refuse a peer-only load whenever THIS box is busy with another
    lane's work -- the false negative that stops two lanes sharing a fleet.  This
    probes exactly one machine and answers about that machine only.
    """
    runner = ps_runner or _run_ps
    who = host if host not in (None, "", "localhost", "127.0.0.1") else "localhost"
    text = runner(None if who == "localhost" else host)
    procs = _parse_ps(text, threshold_gb)
    mem = _parse_memory(text)
    need = load_gb + margin_gb
    acct = accounting or GATE_ACCOUNTING
    avail = gate_accountings(mem)
    # Record what BOTH rules would have said, always, whichever one decides.
    verdicts = {k: (None if v is None else bool(v >= need))
                for k, v in avail.items()}
    receipt = {"box": who, "label": label, "when": time.strftime("%FT%T%z"),
               "load_gb": load_gb, "margin_gb": margin_gb, "pressure": mem,
               "busy": procs, "quiet": not procs, "accounting": acct,
               "available_gb": avail, "would_pass": verdicts,
               "need_gb": need}
    # One model load per box.  Unchanged, and still a hard requirement.
    if procs:
        detail = ", ".join(f"pid {p['pid']} {p['footprint_gb']}GB {p['cmd']}"
                           for p in procs)
        raise HeavyRunActive(
            f"refusing {label or 'this load'} on {who}: something heavy is "
            f"already resident there ({detail}).")
    # Swap must be under the ceiling.  reclaimable promises clean pages, so a box
    # under real memory pressure has none to promise -- but a STATIC residue left
    # by an earlier load is not pressure, and refusing on it strands a box with
    # hundreds of GB free.  Growth during the load is the pressure signal, and
    # WatchedLoad owns that.
    swap = mem.get("swap_used_gb")
    receipt["swap_ceil_gb"] = SWAP_CEIL_GB
    if swap is not None and swap > SWAP_CEIL_GB:
        receipt["quiet"] = False
        raise HeavyRunActive(
            f"refusing {label or 'this load'} on {who}: swap in use is "
            f"{swap:.2f}GB, over the {SWAP_CEIL_GB:.2f}GB ceiling. That is "
            "enough to mean real pressure rather than a stale residue.")
    if acct not in avail:
        raise ValueError(f"unknown gate accounting {acct!r}; "
                         f"expected one of {sorted(avail)}")
    have = avail[acct]
    if have is None:
        # A peer that cannot answer must not be REFUSED for the question -- it
        # gets the older, stricter rule instead. Refusing here would take a box
        # offline for running an older build, which is a worse failure than
        # being conservative about its headroom. Never a silent pass: the
        # downgrade is logged and lands in the receipt.
        fallback = "free_only"
        receipt["accounting_downgraded_from"] = acct
        receipt["accounting"] = acct = fallback
        receipt["downgrade_reason"] = (
            "probe reported no inactive/file-backed page counts "
            "(older build on this box?); falling back to free_only")
        logger.warning(
            "[gate] %s: %s -- headroom for %s judged under free_only instead",
            who, receipt["downgrade_reason"], label or "this load")
        have = avail[fallback]
    if have is None:
        raise HeavyRunActive(
            f"refusing {label or 'this load'} on {who}: the probe reported no "
            "usable memory numbers at all, so headroom cannot be judged.")
    if have < need:
        receipt["quiet"] = False
        extra = ""
        if acct == "reclaimable":
            extra = (f" [free {mem.get('free_gb')}GB + reclaimable "
                     f"{have - (mem.get('free_gb') or 0):.1f}GB]")
        raise HeavyRunActive(
            f"refusing {label or 'this load'} on {who}: {have:.1f}GB available "
            f"under {acct}, needs {need:.0f}GB ({load_gb:.0f} load + "
            f"{margin_gb:.0f} margin"
            + (f", {mem['wired_gb']:.0f} already wired)" if mem.get("wired_gb")
               else ")") + extra)
    receipt["headroom_ok"] = True
    return receipt


def require_quiet_box(host: Optional[str] = None, **kw) -> dict:
    """Gate ONE box, for work that only loads on that box.

    With two lanes running on two machines, gating the whole fleet makes the
    guard a global mutex and the lanes serialise for no reason -- a single-box
    measurement on the peer does not care what this box is holding.  Whole-fleet
    gating stays the default for anything that loads on both (TP).
    """
    hosts = () if host in (None, "", "localhost", "127.0.0.1") else (host,)
    kw.setdefault("label", f"box {host or 'localhost'}")
    return require_quiet_fleet(hosts, **kw)


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
    runner = ps_runner or _run_ps
    local_names = {"", "localhost", "127.0.0.1", socket.gethostname()}
    try:
        local_names |= set(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    targets: List[Optional[str]] = [None]
    for h in hosts:
        if h.split("@")[-1] not in local_names:
            targets.append(h)
    found: Dict[str, List[Dict[str, object]]] = {}
    pressure: Dict[str, Dict[str, Optional[float]]] = {}
    for t in targets:
        text = runner(t)
        name = t or "localhost"
        found[name] = _parse_ps(text, threshold_gb, ignore_pids)
        pressure[name] = _parse_memory(text)
    busy = {h: procs for h, procs in found.items() if procs}
    strained = {}
    for h, m in pressure.items():
        why = []
        if m.get("swap_used_gb") is not None and m["swap_used_gb"] > MAX_SWAP_GB:
            why.append(f"swap {m['swap_used_gb']}GB > {MAX_SWAP_GB}GB")
        if m.get("free_pct") is not None and m["free_pct"] < MIN_FREE_PCT:
            why.append(f"free {m['free_pct']}% < {MIN_FREE_PCT}%")
        if m.get("free_gb") is not None and m["free_gb"] < MIN_FREE_GB:
            why.append(f"only {m['free_gb']}GB free < {MIN_FREE_GB}GB needed"
                       + (f" (wired {m['wired_gb']}GB)"
                          if m.get("wired_gb") else ""))
        if why:
            strained[h] = "; ".join(why)
    receipt = {
        "checked": sorted(found),
        "threshold_gb": threshold_gb,
        "max_swap_gb": MAX_SWAP_GB,
        "min_free_pct": MIN_FREE_PCT,
        "min_free_gb": MIN_FREE_GB,
        "label": label,
        "when": time.strftime("%FT%T%z"),
        "quiet": not busy and not strained,
        "busy": busy,
        "pressure": pressure,
        "strained": strained,
    }
    if strained and not busy:
        raise HeavyRunActive(
            f"refusing to start {label or 'this run'}: the box is under memory "
            f"pressure with nothing co-resident ("
            + "; ".join(f"{h}: {w}" for h, w in strained.items())
            + "). This is the 2026-09-01 shape -- pressure accumulated across "
            "load/unload cycles, not a second model. Let it drain (or recycle "
            "the process) before loading again.")
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
