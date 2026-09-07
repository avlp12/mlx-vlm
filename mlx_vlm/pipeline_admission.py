"""Service-start admission for the resident pipeline tail (L38 A7).

The bench tail was launched by a supervisor that had already proved the box was
quiet: an exclusive box flock, a 65-90 s idle GPU window, a wired ceiling, and a
process/shard-holder scan (``bench/ops/pp_cooperation_supervisor.py``,
``bench/ops/pp_cooperation_common.py``).  A *served* tail has no supervisor.  It
is started once by launchd or by hand and then lives for days, so the campaign's
per-run gates have to be re-cut into two pieces:

* the ones that can only be answered **before the weights load** -- exclusivity,
  wired budget, "is another heavy model already resident" -- become this module,
  and a refusal is a non-zero exit with a sentence saying why;
* the one that has to be answered **continuously** -- is the link still healthy
  -- becomes :class:`RailSampler`, which does not latch a STOP but raises a
  ``degraded`` flag that the head's ``CircuitBreaker`` can act on.

Nothing here imports MLX, touches the GPU, or loads a model: every reading is a
``vm_stat``/``ps``/file-header fact, and every source of fact is injectable so
the whole module is testable on CPU with no fleet.

Health / ping contract (for the head side, A3)
----------------------------------------------
``TailDaemon.status()`` gains an ``admission`` object (this module's
:meth:`AdmissionReport.to_dict`) and a ``rail`` object (:meth:`RailSampler.snapshot`),
and a top-level ``degraded`` boolean that mirrors ``rail["degraded"]``.

The tail's ``ping`` reply carries the same two fields, but **only when the head
asks for them**, because ``pipeline_runtime.PipelineHead.ping`` compares the ack
for equality with ``{"cmd": "ping", "ok": True}`` and an unconditional extra key
would break every head that has not been updated.  The head side of A3 has one
change to make::

    # head sends
    {"cmd": "ping", "rail": true}
    # tail answers
    {"cmd": "ping", "ok": true, "degraded": false,
     "rail": {"degraded": false, "samples": 12, "window": 32, "min_samples": 8,
              "p95_bound_s": 1.5, "recover_factor": 0.8, "p95_wire_s": 0.41,
              "last_wire_s": 0.39, "p95_prefill_s": 4.2, "last_prefill_s": 4.1,
              "transitions": 0, "observed": 12}}

A head that sees ``degraded is True`` on a ping should treat the peer as failing
(``CircuitBreaker.record_failure``) instead of committing a prefill to it; the
tail keeps serving, so the decision stays with the head.  A head that sends a
bare ``{"cmd": "ping"}`` gets the old two-key ack and is unaffected.  ``degraded`` is
hysteretic: it latches at ``p95 > bound`` and clears at
``p95 <= bound * recover_factor``, so a rail sitting on the bound cannot flap the
breaker once per request.

Registered wired policy
-----------------------
The campaign's registered policy lives in
``bench/ops/pp_cooperation_child.py`` (``WIRED05_LIMIT_POLICY``) and is mirrored
in ``docs/drafts/sweep11/PREREG_PP_COOPERATION_WIRED_05.json`` and in every
``bench/ops/manifests/pp05_*.json`` (``repeat_registration.wired_limit_policy``):

    {"requested_limit_bytes": 483183820800,           # 450 GiB exactly
     "require_at_most_device_recommendation": true,
     "restore_previous_limit": true,
     "scope": "run05_only"}

Two things that policy is NOT: it is not a measurement of resident memory, and
its ``scope`` is a single registered run.  A served tail is outside that scope,
so this module treats 450 GiB as a *conventional cap* it inherits rather than a
registration it is entitled to, records ``policy_scope`` in the receipt, and
lets an operator override the number.  ``require_at_most_device_recommendation``
is honoured by clamping the cap to the device's recommended working set when a
caller supplies a ``device_info`` reader.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import threading
import time

__all__ = [
    "AdmissionRefused",
    "AdmissionReport",
    "BoxLock",
    "HEAVY_MODEL_PATTERNS",
    "BOX_BUSY_PATTERNS",
    "REGISTERED_WIRED_POLICY",
    "RailSampler",
    "admit",
    "default_lock_path",
    "heavy_processes",
    "shard_footprint_bytes",
    "wired_bytes",
    "wired_decision",
]

GIB = 1024 ** 3

# Verbatim from bench/ops/pp_cooperation_child.py:21-26 (run05).  Copied, not
# imported: bench/ops is not on the serving import path and this file must not
# grow a dependency on the campaign harness.
REGISTERED_WIRED_POLICY = {
    "requested_limit_bytes": 483183820800,  # 450 GiB
    "require_at_most_device_recommendation": True,
    "restore_previous_limit": True,
    "scope": "run05_only",
}

# The fleet's "one heavy model load per box" rule, as the measurement queues
# spell it (bench queue scripts wait on exactly this alternation before they
# claim a box).  ``rsync`` and the micro-probes are box-busy, not model-resident,
# so they are kept in a second tuple an operator can opt into.
HEAVY_MODEL_PATTERNS = (
    "mlx_vlm.server",
    "gdn_e2e_arms",
    "l7b_prefill",
    "l20_driver",
    "l20_request",
    "l21_dflash",
    "l22_decode",
    "l26_batch",
    "l27_prefill",
    "l28_prefill",
    "pp_cooperation",
)
BOX_BUSY_PATTERNS = HEAVY_MODEL_PATTERNS + (
    "probe_moe",
    "gather_qmm_rhs",
    "indexer_score",
    "rsync",
)

DEFAULT_LOCK_PATH = "~/glm53flash/run/pp_tail.lock"
DEFAULT_RAIL_WINDOW = 32
DEFAULT_RAIL_MIN_SAMPLES = 8
DEFAULT_RAIL_P95_BOUND_S = 2.0
DEFAULT_RAIL_RECOVER_FACTOR = 0.8


class AdmissionRefused(RuntimeError):
    """Start-time refusal.  Raised BEFORE any weight is touched."""

    def __init__(self, gate: str, message: str, detail=None):
        super().__init__(f"[{gate}] {message}")
        self.gate = gate
        self.message = message
        self.detail = detail or {}


# --------------------------------------------------------------------- flock


def default_lock_path() -> str:
    return os.environ.get("MLX_VLM_PIPELINE_LOCK") or DEFAULT_LOCK_PATH


class BoxLock:
    """One tail per box, enforced by an advisory exclusive flock.

    ``flock`` and not a pidfile: the kernel drops the lock when the holder dies
    however it dies, so a tail that was SIGKILLed does not lock the box out of
    its own restart.  The descriptor is kept open for the life of the service --
    closing it releases the lock -- and the file body is a human-readable
    receipt of who holds it, used only for the refusal message.

    A2b adds one liveness step to the refusal path.  The kernel's guarantee is
    about the lock, not about the RECORD: a body naming a pid that is gone
    means either a race (the holder died between our flock attempt and our
    read) or a descriptor some other process inherited.  Neither may cost the
    box its own restart, so a stale record buys exactly one more attempt, with
    a log line saying so -- and if the lock is still genuinely held, the
    refusal says the record was stale rather than pretending a dead pid holds
    it.  What this never does is steal a lock somebody is holding.
    """

    def __init__(self, path=None):
        self.path = Path(os.path.expanduser(str(path or default_lock_path())))
        self.fd = None
        self.holder = None
        self.reclaimed = False

    def acquire(self, *, pid=None, note=None, is_alive=None, log=None):
        pid = os.getpid() if pid is None else int(pid)
        log = log or (lambda line: print(line, flush=True))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            self.holder = _read_holder(fd)
            holder_pid = (
                self.holder.get("pid") if isinstance(self.holder, dict) else None
            )
            stale = holder_pid is not None and not _pid_is_alive(
                holder_pid, is_alive
            )
            stale_and_held = False
            if stale:
                log(
                    f"[tail-admission] lock {self.path} names pid {holder_pid}, "
                    "which is gone -- a stale record must not lock the box out "
                    "of its own restart; retrying the flock once"
                )
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    stale_and_held = True
                else:
                    stale_and_held = False
            if not stale or stale_and_held:
                os.close(fd)
                who = ""
                if isinstance(self.holder, dict) and self.holder.get("pid"):
                    who = f" (held by pid {self.holder['pid']}"
                    if self.holder.get("since"):
                        who += f", since {self.holder['since']}"
                    who += ")"
                    if stale:
                        who = (
                            f" (the recorded holder pid {holder_pid} is gone, "
                            "but the lock is still held -- another process has "
                            "the descriptor)"
                        )
                raise AdmissionRefused(
                    "flock",
                    f"another pipeline tail already owns this box{who}; "
                    f"lock {self.path} -- refusing to start a second resident tail "
                    f"(no weights were loaded). Set MLX_VLM_PIPELINE_LOCK to run a "
                    f"second, deliberately separate service.",
                    {"lock_path": str(self.path), "holder": self.holder,
                     "stale_holder": bool(stale),
                     "errno": getattr(exc, "errno", None)},
                ) from None
            self.reclaimed = True
            log(
                f"[tail-admission] reclaimed the stale lock {self.path} "
                f"(previous holder pid {holder_pid} is gone)"
            )
        self.fd = fd
        body = json.dumps(
            {
                "pid": pid,
                "since": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "note": note or "mlx_vlm.pipeline_prefill --role tail",
            }
        )
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, (body + "\n").encode())
        try:
            os.fsync(fd)
        except OSError:
            pass
        return self

    def release(self):
        """Idempotent, and safe to call from any exit path -- including one
        that is already unwinding an exception."""
        if self.fd is None:
            return
        fd, self.fd = self.fd, None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    # context-manager sugar for tests and short-lived callers
    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


def _pid_is_alive(pid, is_alive=None) -> bool:
    """Liveness by signal 0.  Only ``ProcessLookupError`` proves a pid is gone:
    a pid we are not allowed to signal belongs to someone else and is alive,
    and an unreadable record is treated as alive (refusing is the safe error)."""
    if is_alive is not None:
        return bool(is_alive(pid))
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (TypeError, ValueError, OverflowError):
        return True
    except OSError:
        return True
    return True


def _read_holder(fd):
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 4096).decode("utf-8", "replace").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return {"raw": raw}


# --------------------------------------------------------------- wired policy


_PAGE_RE = re.compile(r"page size of (\d+) bytes")
_WIRED_RE = re.compile(r"Pages wired down:\s*(\d+)")


def _read_vm_stat():
    return subprocess.run(
        ["/usr/bin/vm_stat"], capture_output=True, text=True, check=True
    ).stdout


def wired_bytes(vm_stat_reader=None) -> int:
    """Currently wired bytes, the same way the campaign preflight reads them
    (``pp_cooperation_common.snapshot``): page size x "Pages wired down"."""
    text = (vm_stat_reader or _read_vm_stat)()
    page = _PAGE_RE.search(text)
    wired = _WIRED_RE.search(text)
    if not (page and wired):
        raise AdmissionRefused(
            "wired", "vm_stat parse failed; refusing to guess wired memory",
            {"vm_stat": text[:400]},
        )
    return int(page.group(1)) * int(wired.group(1))


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def shard_footprint_bytes(model_path, lo, hi, n_layers=None, prune=True):
    """Bytes this stage will actually hold, from the safetensors headers only.

    Reads each shard's header (an 8-byte length plus a JSON dict of
    ``{name: {..., data_offsets: [a, b]}}``) and sums the tensors this stage
    keeps.  No tensor data is read and MLX is never imported, so the estimate
    costs milliseconds and cannot itself wire memory.  The keep/drop rule
    mirrors ``pipeline_prefill.load_stage``: layers outside ``[lo, hi)`` go, and
    when pruning, so do the vision tower, ``embed_tokens`` for a non-head stage
    and ``lm_head`` for a non-tail stage.

    Returns ``(bytes, detail)``.  ``bytes`` is ``None`` when the directory holds
    no readable safetensors -- an unknown footprint is reported as unknown, not
    as zero.
    """
    root = Path(os.path.expanduser(str(model_path)))
    files = sorted(root.glob("*.safetensors")) if root.is_dir() else []
    detail = {
        "model_path": str(root),
        "shard_files": len(files),
        "source": "safetensors_header",
        "lo": int(lo),
        "hi": int(hi),
        "prune": bool(prune),
    }
    if not files:
        detail["source"] = "unknown"
        detail["why"] = "no *.safetensors under model_path"
        return None, detail
    total = 0
    kept = dropped = 0
    max_layer = -1
    for path in files:
        try:
            with open(path, "rb") as fh:
                raw = fh.read(8)
                if len(raw) != 8:
                    continue
                (hdr_len,) = struct.unpack("<Q", raw)
                if hdr_len <= 0 or hdr_len > 512 * 1024 * 1024:
                    continue
                header = json.loads(fh.read(hdr_len).decode("utf-8"))
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            detail.setdefault("unreadable", []).append(f"{path.name}: {exc!r}")
            continue
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            offsets = meta.get("data_offsets")
            if not (isinstance(offsets, (list, tuple)) and len(offsets) == 2):
                continue
            size = int(offsets[1]) - int(offsets[0])
            m = _LAYER_RE.search(name)
            if m is not None:
                idx = int(m.group(1))
                max_layer = max(max_layer, idx)
                if lo <= idx < hi:
                    total += size
                    kept += 1
                else:
                    dropped += 1
                continue
            if prune and _dropped_non_layer(name, lo, hi, n_layers, max_layer):
                dropped += 1
                continue
            total += size
            kept += 1
    if kept == 0:
        detail["source"] = "unknown"
        detail["why"] = "safetensors headers held no tensor this stage keeps"
        return None, detail
    detail.update(tensors_kept=kept, tensors_dropped=dropped)
    return total, detail


def _dropped_non_layer(name, lo, hi, n_layers, max_layer):
    lowered = name.lower()
    if "vision" in lowered or "visual" in lowered:
        return True
    if "embed_tokens" in lowered and lo > 0:
        return True
    end = n_layers if n_layers is not None else (max_layer + 1 if max_layer >= 0 else None)
    if "lm_head" in lowered and end is not None and hi < end:
        return True
    return False


def wired_decision(
    *,
    shard_bytes,
    cap_bytes=None,
    vm_stat_reader=None,
    device_info=None,
    current_wired=None,
    allow_overcommit=False,
):
    """Would loading this stage push the box past the registered cap?

    ``current_wired + shard_bytes <= cap`` admits.  The cap defaults to the
    registered 450 GiB and is clamped to the device recommendation when a
    ``device_info`` reader is supplied (the policy's
    ``require_at_most_device_recommendation``).  An unknown ``shard_bytes`` is
    reported and admitted -- this gate refuses on evidence, never on ignorance --
    but the receipt says so, so the health line does not read like a pass.
    """
    env_cap = os.environ.get("MLX_VLM_PIPELINE_WIRED_CAP_BYTES")
    if cap_bytes is None and env_cap:
        cap_bytes = int(env_cap)
        cap_source = "env:MLX_VLM_PIPELINE_WIRED_CAP_BYTES"
    elif cap_bytes is None:
        cap_bytes = int(REGISTERED_WIRED_POLICY["requested_limit_bytes"])
        cap_source = "registered:pp_cooperation_child.WIRED05_LIMIT_POLICY"
    else:
        cap_bytes = int(cap_bytes)
        cap_source = "caller"
    recommended = None
    if device_info is not None and REGISTERED_WIRED_POLICY[
        "require_at_most_device_recommendation"
    ]:
        try:
            info = device_info() or {}
            value = info.get("max_recommended_working_set_size")
            if isinstance(value, int) and value > 0:
                recommended = value
                if recommended < cap_bytes:
                    cap_bytes = recommended
                    cap_source += "+device_recommendation"
        except Exception as exc:  # noqa: BLE001 -- a query, never a gate
            recommended = f"unavailable: {exc!r}"
    if current_wired is None:
        current_wired = wired_bytes(vm_stat_reader)
    current_wired = int(current_wired)
    known = isinstance(shard_bytes, int) and shard_bytes >= 0
    projected = current_wired + int(shard_bytes) if known else None
    ok = True if projected is None else projected <= cap_bytes
    decision = {
        "policy": "run05_wired_limit_policy",
        "policy_scope": REGISTERED_WIRED_POLICY["scope"],
        "cap_bytes": cap_bytes,
        "cap_source": cap_source,
        "cap_gib": round(cap_bytes / GIB, 2),
        "device_recommended_bytes": recommended,
        "current_wired_bytes": current_wired,
        "current_wired_gib": round(current_wired / GIB, 2),
        "shard_bytes": int(shard_bytes) if known else None,
        "shard_gib": round(int(shard_bytes) / GIB, 2) if known else None,
        "projected_bytes": projected,
        "projected_gib": round(projected / GIB, 2) if projected is not None else None,
        "headroom_bytes": (cap_bytes - projected) if projected is not None else None,
        "ok": bool(ok or allow_overcommit),
        "refused": bool(not ok and not allow_overcommit),
        "overcommit_allowed": bool(allow_overcommit),
        "shard_known": known,
    }
    if not known:
        decision["reason"] = "shard footprint unknown; admitted without a budget check"
    elif not ok:
        decision["reason"] = (
            f"projected {decision['projected_gib']} GiB "
            f"(wired {decision['current_wired_gib']} + shard {decision['shard_gib']}) "
            f"exceeds the registered cap {decision['cap_gib']} GiB"
        )
        if allow_overcommit:
            decision["reason"] += " -- admitted anyway by --allow-wired-overcommit"
    return decision


# ------------------------------------------------------------ box preflight


def _read_ps():
    return subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,command="],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _parse_ps(raw):
    rows = []
    for line in raw.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3 or not (fields[0].isdigit() and fields[1].isdigit()):
            continue
        rows.append((int(fields[0]), int(fields[1]), fields[2]))
    return rows


def _ancestors(rows, pid):
    parent = {p: pp for p, pp, _ in rows}
    seen = {pid}
    cur = pid
    for _ in range(64):
        cur = parent.get(cur)
        if not cur or cur in seen:
            break
        seen.add(cur)
    return seen


def heavy_processes(patterns=None, ps_reader=None, own_pid=None, ignore_pids=()):
    """Other processes on this box that already hold a heavy model.

    Own PID **and its ancestor chain** are excluded: the campaign child runs the
    tail inside its own ``pp_cooperation_child.py`` process and under a
    ``pp_cooperation_supervisor.py`` parent, both of which match the pattern
    list -- a gate that refuses because of its own launcher is a gate nobody
    keeps switched on.
    """
    patterns = tuple(patterns or _env_patterns() or HEAVY_MODEL_PATTERNS)
    rows = _parse_ps((ps_reader or _read_ps)())
    own_pid = os.getpid() if own_pid is None else int(own_pid)
    skip = _ancestors(rows, own_pid) | {int(p) for p in ignore_pids}
    rx = re.compile("|".join(re.escape(p) for p in patterns)) if patterns else None
    found = []
    for pid, ppid, cmd in rows:
        if pid in skip or rx is None:
            continue
        m = rx.search(cmd)
        if m is None:
            continue
        if not _looks_like_the_process_itself(cmd, m.group(0)):
            # The name appearing in an argv is not the same as the model being
            # resident: the fleet's queue scripts are `zsh -c ... grep -qE
            # "mlx_vlm.server|..."` wrappers whose own command line contains
            # every pattern in the list, which is why those scripts filter
            # `grep -v "zsh -c source"` and `grep -v grep` before believing a
            # match.  Same filter, stated as a rule: the argv0 has to be a
            # python (pp_cooperation_common.process_blockers uses the same
            # test) or be the matched tool itself.
            continue
        found.append({"pid": pid, "ppid": ppid, "pattern": m.group(0),
                      "command": cmd[:300]})
    return found


def _looks_like_the_process_itself(cmd, pattern):
    head = cmd.split()[0] if cmd.split() else ""
    exe = Path(head).name.lower()
    if exe in ("grep", "egrep", "fgrep", "pgrep", "rg", "ps", "awk", "sed"):
        return False
    return "python" in exe or pattern.lower() in exe


def _env_patterns():
    raw = os.environ.get("MLX_VLM_PIPELINE_PREFLIGHT_PATTERNS")
    if not raw:
        return None
    return tuple(x.strip() for x in raw.split(",") if x.strip())


# ------------------------------------------------------------- rail sampler


def _p95(values):
    if not values:
        return None
    ordered = sorted(values)
    # nearest-rank; with n < 20 this is the max, which is what a small window
    # of a slow rail should report rather than an interpolated fiction
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


class RailSampler:
    """Per-request wire/prefill timings -> a sticky ``degraded`` flag.

    The campaign latched STOP when a rail went bad.  A service must not: it has
    a request in flight and a head that can fall back on its own.  So this
    sampler only *reports*, and the head's ``CircuitBreaker`` decides.  Sliding
    window, nearest-rank p95, and hysteresis so a rail parked on the bound does
    not toggle the breaker every request.
    """

    def __init__(
        self,
        window=DEFAULT_RAIL_WINDOW,
        p95_bound_s=DEFAULT_RAIL_P95_BOUND_S,
        min_samples=DEFAULT_RAIL_MIN_SAMPLES,
        recover_factor=DEFAULT_RAIL_RECOVER_FACTOR,
    ):
        self.window = max(1, int(window))
        self.p95_bound_s = float(p95_bound_s)
        self.min_samples = max(1, int(min_samples))
        self.recover_factor = float(recover_factor)
        self._lock = threading.Lock()
        self._wire = []
        self._prefill = []
        self._degraded = False
        self._transitions = 0
        self._total = 0

    @property
    def degraded(self) -> bool:
        with self._lock:
            return self._degraded

    def observe(self, wire_s, prefill_s=None):
        """Record one request; returns the (possibly new) degraded flag."""
        if wire_s is None or not math.isfinite(float(wire_s)):
            return self.degraded
        with self._lock:
            self._wire.append(float(wire_s))
            del self._wire[: max(0, len(self._wire) - self.window)]
            if prefill_s is not None and math.isfinite(float(prefill_s)):
                self._prefill.append(float(prefill_s))
                del self._prefill[: max(0, len(self._prefill) - self.window)]
            self._total += 1
            self._reassess_locked()
            return self._degraded

    def _reassess_locked(self):
        if len(self._wire) < self.min_samples:
            return
        p95 = _p95(self._wire)
        if not self._degraded and p95 > self.p95_bound_s:
            self._degraded = True
            self._transitions += 1
        elif self._degraded and p95 <= self.p95_bound_s * self.recover_factor:
            self._degraded = False
            self._transitions += 1

    def reset(self):
        with self._lock:
            self._wire = []
            self._prefill = []
            self._degraded = False

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "degraded": self._degraded,
                "samples": len(self._wire),
                "window": self.window,
                "min_samples": self.min_samples,
                "p95_bound_s": self.p95_bound_s,
                "recover_factor": self.recover_factor,
                "p95_wire_s": _round(_p95(self._wire)),
                "last_wire_s": _round(self._wire[-1] if self._wire else None),
                "p95_prefill_s": _round(_p95(self._prefill)),
                "last_prefill_s": _round(self._prefill[-1] if self._prefill else None),
                "transitions": self._transitions,
                "observed": self._total,
            }


def _round(value, places=4):
    return None if value is None else round(float(value), places)


# --------------------------------------------------------------- the gate


class AdmissionReport:
    """What the three start-time gates decided, verbatim, for the health line."""

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self.lock = None
        self.gates = {}
        self.admitted = None
        self.refused_gate = None
        self.at = time.time()

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "admitted": self.admitted,
            "refused_gate": self.refused_gate,
            "at": round(self.at, 3),
            "lock_path": str(self.lock.path) if self.lock is not None else None,
            "gates": self.gates,
        }

    def release(self):
        if self.lock is not None:
            self.lock.release()


def admit(
    args,
    *,
    ps_reader=None,
    vm_stat_reader=None,
    device_info=None,
    shard_bytes=None,
    lock_factory=BoxLock,
):
    """Run the three start-time gates in cost order and return a report.

    Order matters: the flock is free and settles exclusivity, the process scan
    is one ``ps``, and the wired budget is the only one that reads the model
    directory.  Every one of them completes before a single weight is touched;
    a refusal raises :class:`AdmissionRefused` and the caller exits non-zero.
    """
    report = AdmissionReport(enabled=True)
    try:
        # 1. one tail per box
        lock_path = getattr(args, "lock_file", None) or default_lock_path()
        if getattr(args, "no_lock", False):
            report.gates["flock"] = {"ok": True, "skipped": "--no-lock"}
        else:
            lock = lock_factory(lock_path)
            try:
                lock.acquire(note=f"--role tail --port {getattr(args, 'port', '?')}")
            except AdmissionRefused as exc:
                report.gates["flock"] = {"ok": False, "reason": exc.message,
                                         **exc.detail}
                raise
            report.lock = lock
            report.gates["flock"] = {"ok": True, "lock_path": str(lock.path),
                                     "pid": os.getpid(),
                                     "reclaimed": bool(
                                         getattr(lock, "reclaimed", False)
                                     )}

        # 2. one heavy model per box
        allow_shared = bool(
            getattr(args, "allow_shared_box", False)
            or os.environ.get("MLX_VLM_PIPELINE_ALLOW_SHARED_BOX") == "1"
        )
        found = heavy_processes(ps_reader=ps_reader)
        gate = {
            "ok": allow_shared or not found,
            "allow_shared_box": allow_shared,
            "patterns": list(_env_patterns() or HEAVY_MODEL_PATTERNS),
            "found": found,
        }
        report.gates["preflight"] = gate
        if found and not allow_shared:
            names = ", ".join(f"{p['pattern']}(pid {p['pid']})" for p in found[:4])
            gate["reason"] = (
                f"{len(found)} heavy MLX process(es) already resident: {names}"
            )
            raise AdmissionRefused(
                "preflight",
                gate["reason"]
                + " -- one heavy model load per box; refusing to start "
                  "(no weights were loaded). Pass --allow-shared-box to override.",
                {"found": found},
            )

        # 3. wired budget
        if shard_bytes is None:
            shard_bytes = _configured_shard_bytes(args)
        detail = None
        if shard_bytes is None:
            shard_bytes, detail = shard_footprint_bytes(
                getattr(args, "model", ""),
                getattr(args, "split", 0),
                getattr(args, "layers", 0),
                n_layers=getattr(args, "layers", None),
                prune=bool(getattr(args, "prune", True)),
            )
        decision = wired_decision(
            shard_bytes=shard_bytes,
            cap_bytes=getattr(args, "wired_cap_bytes", None),
            vm_stat_reader=vm_stat_reader,
            device_info=device_info,
            allow_overcommit=bool(getattr(args, "allow_wired_overcommit", False)),
        )
        if detail is not None:
            decision["shard_detail"] = detail
        report.gates["wired"] = decision
        if decision["refused"]:
            raise AdmissionRefused(
                "wired",
                decision["reason"]
                + " -- refusing to start (no weights were loaded). Raise the cap "
                  "with --wired-cap-bytes or accept it with "
                  "--allow-wired-overcommit.",
                {"decision": decision},
            )
        report.admitted = True
        return report
    except AdmissionRefused as exc:
        report.admitted = False
        report.refused_gate = exc.gate
        report.release()
        exc.detail.setdefault("report", report.to_dict())
        raise


def _configured_shard_bytes(args):
    value = getattr(args, "shard_bytes", None)
    if value is None:
        value = os.environ.get("MLX_VLM_PIPELINE_SHARD_BYTES")
    if value in (None, ""):
        return None
    return int(value)


def rail_sampler_from_args(args) -> RailSampler:
    return RailSampler(
        window=int(
            getattr(args, "rail_window", None)
            or os.environ.get("MLX_VLM_PIPELINE_RAIL_WINDOW")
            or DEFAULT_RAIL_WINDOW
        ),
        p95_bound_s=float(
            getattr(args, "rail_p95_s", None)
            or os.environ.get("MLX_VLM_PIPELINE_RAIL_P95_S")
            or DEFAULT_RAIL_P95_BOUND_S
        ),
        min_samples=int(
            getattr(args, "rail_min_samples", None) or DEFAULT_RAIL_MIN_SAMPLES
        ),
    )


def describe(argv=None):
    """``python -m mlx_vlm.pipeline_admission`` -- print what the gates see now
    without starting anything.  A dry run for an operator about to launch."""
    import argparse

    p = argparse.ArgumentParser(description="pipeline tail admission dry run")
    p.add_argument("--model", default="")
    p.add_argument("--split", type=int, default=23)
    p.add_argument("--layers", type=int, default=45)
    p.add_argument("--lock-file", default=None)
    p.add_argument("--port", type=int, default=39200)
    args = p.parse_args(argv)
    args.no_lock = True
    args.allow_shared_box = True
    args.allow_wired_overcommit = True
    args.prune = True
    report = admit(args)
    report.release()
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(describe())
