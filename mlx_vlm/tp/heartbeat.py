"""Progress-carrying liveness side-channel for tensor-parallel serving.

WHY THIS EXISTS
===============
Every control verb in mode-level TP rides the *same* ``all_sum`` the sharded
layers already need (see ``worker.py``).  That is a good protocol -- it needs no
second transport and it cannot desynchronise from the data path -- but it has one
structural blind spot: **the only channel between the ranks is the collective
that hangs.**  When rank 0 stops driving, rank 1 blocks inside ``all_sum``
forever, holding its entire shard, and nothing can tell it why.

``worker.py`` says so in its own words, at the point where the fix belongs:

    # A correct liveness signal would need a heartbeat from rank 0, which is a
    # protocol change.  Until then the bound below is time-only.

and explains why the time-only bound had to ship DISABLED:

    A serving deployment can legitimately sit idle between requests for hours,
    and a time-only bound cannot tell "idle" from "orphaned" -- killing rank 1
    on a quiet server would be a worse bug than the one this guards.

This module is that protocol change.  It is deliberately NOT a data path: it
carries no tensors, it never participates in a collective, and if it fails
completely the system degrades to exactly today's behaviour.

LIVENESS IS NOT ENOUGH -- THE BEAT CARRIES PROGRESS
===================================================
A naive heartbeat makes the failure it is meant to catch WORSE.  If rank 0's
driving thread wedges inside a collective and a *separate* thread keeps sending
"I am alive", rank 1 concludes health and waits forever with more confidence
than before.  That is precisely the jaccl vault-class hang.

So the beat carries the driver's PROGRESS COUNTERS, not its pulse:

    (state, epoch, fwd_idx, verb_seq, monotonic_ns)

``fwd_idx`` is ``transport._FWD_IDX``, the index of the current all_sum within
the current forward -- a counter that ALREADY exists and is ALREADY maintained
for exactly this purpose (transport.py: "so a stalled pair can be compared
position by position rather than in aggregate: 'rank 0 reached #57 and rank 1
reached #58' names the reduce").  Until now there was no channel on which to
make that comparison.  Now there is, and it runs on both sides.

WHY A SENDER THREAD IS CORRECT HERE, AND STRICTLY BETTER THAN EMITTING INLINE
============================================================================
The driving thread updates an in-process snapshot (``note()``); a daemon thread
transmits whatever the snapshot currently says, at ~4 Hz.  This is not the
"separate-thread liveness beat" failure above, because the receiver's test is
staleness of CONTENT, not arrival of packets.

It is also strictly more informative than emitting from the driving thread,
because it separates two failures that inline emission would merge into one:

    driver wedged  -> packets KEEP ARRIVING, fwd_idx FROZEN   -> PEER_STALLED
    process dead   -> packets STOP                            -> PEER_DEAD

If the beat were emitted inline, a thread blocked inside a collective could not
emit at all, and both failures would look identical (silence).  The pre-registered
outcome of lane 1's vault-hang reproduction depends on being able to tell them
apart: a PEER_STALLED verdict localises the wedge to rank 0's driving thread,
while a stalled *beat* would instead indicate the process itself is gone.

TRANSPORT
=========
UDP.  Deliberately: "latest wins" is exactly heartbeat semantics, there is no
connection state to wedge, and no retransmit timer that could itself become the
thing that hangs.  Loss is a non-event -- the next beat carries the same state.

HARD CONSTRAINT: never point this at the jaccl coordinator port.  ``transport.py``
records that a plain TCP connect to it is accepted as a rank joining and broke
formation 1/1 times.  ``Beacon`` refuses to bind or send to that port.

ADDRESSING
==========
Defaults to the dedicated 10GBASE-T pair (gesicht en0 10.0.1.1 <-> epsilon en0
10.0.1.2): its own port, its own copper, its own PHY, so it is independent of
the Thunderbolt cable jaccl runs on -- a TB5 cable fault cannot silence it.
Measured 215 us TCP RTT p50 at 64 B, ~1000x faster than the 250 ms beat period.
See DEFAULT_HOSTS for the full four-path comparison and why latency does not
decide this choice.

Both endpoints are env-overridable so that moving to a genuine 10 GbE link (or
any other pair) is a configuration change, not a code change:

    MLX_VLM_TP_HB_LOCAL   host:port to bind      (default per-rank, below)
    MLX_VLM_TP_HB_PEER    host:port to send to   (default per-rank, below)
    MLX_VLM_TP_HB_PORT    port for both          (default 39600)
    MLX_VLM_TP_HB_HOSTS   comma-separated per-rank hosts, overrides the defaults
    MLX_VLM_TP_HB_HZ      beat rate              (default 4.0)
    MLX_VLM_TP_HB_STALL_S fwd_idx frozen bound   (default 30.0)
    MLX_VLM_TP_HB_DEAD_S  no-packet bound        (default 10.0)
    MLX_VLM_TP_HB=0       disable entirely (degrades to today's behaviour)
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# The dedicated 10GBASE-T link, which is the only path between the boxes that is
# PHYSICALLY independent of the Thunderbolt cable jaccl runs on: its own RJ45
# port, its own copper, its own PHY.  A TB5 cable or port fault cannot take it
# down, so a heartbeat here distinguishes "the fast link died" from "rank 0
# wedged" -- which the previous default could not.
#
# Lane 7 measured all four candidate paths, 64 B application TCP round trip,
# TCP_NODELAY, 200 untimed warm-ups, two interleaved passes under comparable load:
#
#     path                        p50        min      1-stream throughput
#     tbnet (Thunderbolt 5)     65 us      47 us      47,470 Mbit/s
#     10GbE en0  <-- default   215 us     128 us       9,403 Mbit/s
#     169.254 (tunnelled)      180 us     143 us         341 Mbit/s
#     Wi-Fi                   5987 us    3702 us         330 Mbit/s
#
# Latency does NOT decide this: at 4 Hz the beat period is 250 ms, so every
# candidate except Wi-Fi is ~1000x faster than it needs to be.  INDEPENDENCE
# decides it, and only en0 has the physical kind.  The 169.254 pair is inferred
# to tunnel over the same TB5 cable (six "Ethernet Adapter" pseudo-NICs, one per
# TB receptacle; the active one is the 4th and the only connected receptacle is
# #4), so it protects against a jaccl/RDMA SOFTWARE wedge but not a cable fault.
#
# The 10GbE's fatter tail (p99 340-405 us vs 265-283 us) is consistent with
# energy-efficient-ethernet parking the PHY between packets -- visible in
# `ifconfig en0` as "energy-efficient-ethernet".  Irrelevant at 4 Hz; if a future
# use ever needs the tail, EEE is the first thing to look at.
DEFAULT_HOSTS = ("10.0.1.1", "10.0.1.2")

# The previous default, kept because it is a genuine fallback: if en0 is
# unplugged this still works and still catches the dominant (software) hang class.
FALLBACK_HOSTS = ("169.254.30.147", "169.254.240.246")
DEFAULT_PORT = 39600
JACCL_COORDINATOR_DEFAULT_PORT = 39500      # never bind or send here

MAGIC = b"GLM5HB01"
# <8s magic | B state | x pad | H rank | q epoch | q fwd_idx | q verb_seq | q ns
_FMT = "<8sBxHqqqq"
BEAT_BYTES = struct.calcsize(_FMT)          # 44

STATE_IDLE = 0        # parked between requests; rank 1 must wait indefinitely
STATE_DRIVING = 1     # a forward/verb sequence is in flight; progress expected
STATE_EXITING = 2     # orderly shutdown; peer may release immediately

_STATE_NAMES = {STATE_IDLE: "IDLE", STATE_DRIVING: "DRIVING", STATE_EXITING: "EXITING"}

# Verdicts from Beacon.poll()
HEALTHY = "HEALTHY"
PEER_STALLED = "PEER_STALLED"
PEER_DEAD = "PEER_DEAD"
PEER_EXITING = "PEER_EXITING"
UNKNOWN = "UNKNOWN"          # never heard from the peer yet; NOT a kill signal


def enabled() -> bool:
    return os.environ.get("MLX_VLM_TP_HB", "1").lower() not in ("0", "false", "no", "off")


@dataclass(frozen=True)
class Beat:
    state: int
    rank: int
    epoch: int
    fwd_idx: int
    verb_seq: int
    ns: int

    def pack(self) -> bytes:
        return struct.pack(_FMT, MAGIC, self.state, self.rank,
                           self.epoch, self.fwd_idx, self.verb_seq, self.ns)

    @staticmethod
    def unpack(buf: bytes) -> Optional["Beat"]:
        """Return None for anything that is not one of our beats.

        A UDP socket will happily receive stray traffic; a malformed datagram
        must never be able to influence a liveness decision, so this validates
        length and magic and returns None rather than raising."""
        if len(buf) != BEAT_BYTES:
            return None
        magic, state, rank, epoch, fwd_idx, verb_seq, ns = struct.unpack(_FMT, buf)
        if magic != MAGIC:
            return None
        return Beat(state, rank, epoch, fwd_idx, verb_seq, ns)

    def describe(self) -> str:
        return (f"state={_STATE_NAMES.get(self.state, self.state)} rank={self.rank} "
                f"epoch={self.epoch} fwd_idx={self.fwd_idx} verb_seq={self.verb_seq}")


def _addr_from_env(name: str) -> Optional[Tuple[str, int]]:
    v = os.environ.get(name, "").strip()
    if not v:
        return None
    host, _, port = v.rpartition(":")
    if not host:
        raise ValueError(f"{name} must be host:port, got {v!r}")
    return (host, int(port))


def resolve_addrs(rank: int, size: int = 2) -> Tuple[Tuple[str, int], Tuple[str, int]]:
    """(local_bind, peer) for this rank.  Env overrides win."""
    port = int(os.environ.get("MLX_VLM_TP_HB_PORT", DEFAULT_PORT))
    hosts = os.environ.get("MLX_VLM_TP_HB_HOSTS", "").strip()
    hosts = tuple(h.strip() for h in hosts.split(",") if h.strip()) or DEFAULT_HOSTS
    if len(hosts) < size:
        raise ValueError(f"MLX_VLM_TP_HB_HOSTS needs {size} hosts, got {hosts}")
    local = _addr_from_env("MLX_VLM_TP_HB_LOCAL") or (hosts[rank], port)
    peer = _addr_from_env("MLX_VLM_TP_HB_PEER") or (hosts[1 - rank], port)
    for who, (_h, p) in (("local", local), ("peer", peer)):
        if p == int(os.environ.get("MLX_JACCL_COORDINATOR_PORT",
                                   JACCL_COORDINATOR_DEFAULT_PORT)):
            raise ValueError(
                f"heartbeat {who} port {p} is the jaccl coordinator port. A packet "
                f"to that port is accepted as a rank joining and breaks formation "
                f"(transport.py records this failing 1/1). Choose another port.")
    return local, peer


class Beacon:
    """Bidirectional progress beacon.  Both ranks run one.

    Rank 0 announces what it is driving; rank 1 announces how far it has followed.
    Each side therefore holds the other's counters and either can NAME a divergent
    reduce without the collective's cooperation.
    """

    def __init__(self, rank: int, size: int = 2, *, sock=None, clock=time.monotonic):
        self.rank = int(rank)
        self.size = int(size)
        self._clock = clock
        self.hz = float(os.environ.get("MLX_VLM_TP_HB_HZ", "4.0"))
        self.stall_s = float(os.environ.get("MLX_VLM_TP_HB_STALL_S", "30.0"))
        self.dead_s = float(os.environ.get("MLX_VLM_TP_HB_DEAD_S", "10.0"))

        self._lock = threading.Lock()
        self._snap = Beat(STATE_IDLE, self.rank, -1, 0, 0, time.monotonic_ns())
        # last received beat, when it arrived, and when its fwd_idx last CHANGED
        self._peer: Optional[Beat] = None
        self._peer_at: float = 0.0
        self._peer_progress_at: float = 0.0
        self._peer_progress_key: Optional[tuple] = None
        self._started_at = self._clock()

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

        if sock is not None:                     # unit tests inject a fake
            self.sock = sock
            self.local = getattr(sock, "local", ("test", 0))
            self.peer = getattr(sock, "peer", ("test", 1))
        else:
            self.local, self.peer = resolve_addrs(self.rank, self.size)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(self.local)
            self.sock.settimeout(0.25)

    # -- driver side -------------------------------------------------------
    # STATE and COUNTERS are updated by DIFFERENT call sites, and the split is
    # load-bearing.
    #
    # MLX IS LAZY.  ``mx.distributed.all_sum`` only BUILDS a node; the collective
    # does not execute, and cannot block, until ``mx.eval``.  So bracketing
    # all_sum would bracket the wrong thing entirely -- the wedge happens inside
    # eval, somewhere else in the call stack.  Therefore:
    #
    #   note_progress()  -- from transport.all_sum, every constructed reduce.
    #                       Moves the counters.  Says nothing about state.
    #   driving()        -- around the mx.eval that actually blocks.
    #                       Says "a collective is OUTSTANDING right now".
    #
    # The receiver's PEER_STALLED test is (state == DRIVING) AND (counters
    # frozen), so it fires only when a collective is genuinely outstanding and
    # progress has stopped.  Idle-between-requests reports IDLE and waits
    # forever, which is the property that let the time-only bound ship disabled.
    def note(self, state: Optional[int] = None, *, epoch: Optional[int] = None,
             fwd_idx: Optional[int] = None, verb_seq: Optional[int] = None):
        """Update the snapshot the sender thread transmits.

        Every field is optional and defaults to "leave as is", so a progress
        update cannot accidentally clear the state and vice versa.  Cheap: one
        lock and a tuple build.
        """
        with self._lock:
            cur = self._snap
            self._snap = Beat(
                cur.state if state is None else int(state),
                self.rank,
                cur.epoch if epoch is None else int(epoch),
                cur.fwd_idx if fwd_idx is None else int(fwd_idx),
                cur.verb_seq if verb_seq is None else int(verb_seq),
                time.monotonic_ns(),
            )

    def note_progress(self, *, epoch: Optional[int] = None,
                      fwd_idx: Optional[int] = None, verb_seq: Optional[int] = None):
        """Counters only.  Called once per constructed reduce."""
        self.note(None, epoch=epoch, fwd_idx=fwd_idx, verb_seq=verb_seq)

    @contextlib.contextmanager
    def driving(self):
        """Mark a collective as outstanding for the duration of the block.

        Wrap the ``mx.eval`` that forces the reduce, not its construction.
        Restores the previous state on the way out, including on exception, so a
        raised TPDesync cannot leave the peer believing we are still driving.
        """
        with self._lock:
            prev = self._snap.state
        self.note(STATE_DRIVING)
        try:
            yield self
        finally:
            self.note(prev if prev != STATE_EXITING else STATE_EXITING)

    def snapshot(self) -> Beat:
        with self._lock:
            return self._snap

    # -- follower side -----------------------------------------------------
    def peer_beat(self) -> Optional[Beat]:
        with self._lock:
            return self._peer

    def poll(self) -> Tuple[str, Optional[str]]:
        """(verdict, human reason).  Pure function of observed state; no I/O."""
        now = self._clock()
        with self._lock:
            peer, at, prog_at = self._peer, self._peer_at, self._peer_progress_at
        if peer is None:
            # Never heard from the peer.  This is NOT a kill signal: formation may
            # still be in progress, and killing rank 1 on a silent-but-healthy
            # start is the exact bug the removed ppid probe caused.
            return UNKNOWN, None
        if peer.state == STATE_EXITING:
            return PEER_EXITING, f"peer announced EXITING ({peer.describe()})"
        age = now - at
        if age > self.dead_s:
            return PEER_DEAD, (f"no beat for {age:.1f}s (bound {self.dead_s}s); "
                               f"last was {peer.describe()}")
        if peer.state == STATE_IDLE:
            # Legitimately parked.  Wait indefinitely -- this is the case that
            # forced the time-only bound to ship disabled.
            return HEALTHY, None
        frozen = now - prog_at
        if peer.state == STATE_DRIVING and frozen > self.stall_s:
            return PEER_STALLED, (
                f"peer says DRIVING but its progress counters have not moved for "
                f"{frozen:.1f}s (bound {self.stall_s}s): {peer.describe()}. The "
                f"peer's process is alive -- beats are still arriving -- so the "
                f"wedge is in its driving thread, not the box.")
        return HEALTHY, None

    def divergence(self, my_fwd_idx: int) -> Optional[str]:
        """Name the divergent reduce, which the collective itself cannot."""
        peer = self.peer_beat()
        if peer is None or peer.fwd_idx == my_fwd_idx:
            return None
        return (f"rank {self.rank} reached all_sum #{my_fwd_idx}; peer rank "
                f"{peer.rank} reached #{peer.fwd_idx} ({peer.describe()})")

    # -- threads -----------------------------------------------------------
    def _send_loop(self):
        period = 1.0 / max(self.hz, 0.1)
        while not self._stop.is_set():
            try:
                self.sock.sendto(self.snapshot().pack(), self.peer)
            except OSError as e:
                logger.debug("[hb] send failed (non-fatal): %s", e)
            self._stop.wait(period)

    def _recv_loop(self):
        while not self._stop.is_set():
            try:
                buf, _ = self.sock.recvfrom(256)
            except (socket.timeout, TimeoutError):
                continue
            except OSError as e:
                if self._stop.is_set():
                    return
                logger.debug("[hb] recv failed (non-fatal): %s", e)
                time.sleep(0.05)
                continue
            self.ingest(buf)

    def ingest(self, buf: bytes) -> Optional[Beat]:
        """Feed one datagram.  Separated from the socket so tests can drive it."""
        b = Beat.unpack(buf)
        if b is None or b.rank == self.rank:
            return None                    # junk, or our own packet looped back
        now = self._clock()
        with self._lock:
            key = (b.epoch, b.fwd_idx, b.verb_seq, b.state)
            if key != self._peer_progress_key:
                self._peer_progress_key = key
                self._peer_progress_at = now
            self._peer = b
            self._peer_at = now
        return b

    def start(self) -> "Beacon":
        if self._threads:
            return self
        now = self._clock()
        with self._lock:
            self._peer_at = self._peer_progress_at = now
        for fn, name in ((self._send_loop, "tp-hb-send"), (self._recv_loop, "tp-hb-recv")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        logger.info("[hb] rank %d beacon up: bound %s, peer %s, %.1f Hz "
                    "(stall %.0fs, dead %.0fs)", self.rank, self.local, self.peer,
                    self.hz, self.stall_s, self.dead_s)
        return self

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        try:
            self.sock.close()
        except Exception:
            pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


# -- module-level singleton, so worker.py and tp_mode.py share one beacon ----
_BEACON: Optional[Beacon] = None


def beacon() -> Optional[Beacon]:
    return _BEACON


def init_beacon(rank: int, size: int = 2, **kw) -> Optional[Beacon]:
    """Idempotent.  Returns None (and logs) when disabled or unusable -- a
    heartbeat that cannot start must never prevent serving."""
    global _BEACON
    if _BEACON is not None or not enabled():
        return _BEACON
    try:
        _BEACON = Beacon(rank, size, **kw).start()
    except Exception:
        logger.warning("[hb] beacon could not start; continuing without a "
                       "side-channel (degrades to the time-only bound)", exc_info=True)
        _BEACON = None
    return _BEACON


def shutdown_beacon(announce_exit: bool = True):
    global _BEACON
    if _BEACON is None:
        return
    try:
        if announce_exit:
            _BEACON.note(STATE_EXITING)
            # one last datagram, best effort, so the peer releases promptly
            try:
                _BEACON.sock.sendto(_BEACON.snapshot().pack(), _BEACON.peer)
            except Exception:
                pass
        _BEACON.stop()
    finally:
        _BEACON = None


def stall_probe(_window_s: float = 0.0) -> Optional[str]:
    """Deadman-compatible harm probe.

    ``transport.Deadman`` polls ``harm_probe(window_s)`` and aborts when it
    returns a reason.  ``worker._ctrl_recv`` currently passes a probe that always
    returns None, because -- as the source says -- "there is no valid host-side
    liveness signal here".  This is that signal, and unlike the default probe
    (``_display_stalled``, which shells out to /usr/bin/log for ~0.75 s and forces
    a ~9% duty cycle while hung) it is a non-blocking read of already-received
    state, and it is a DIRECT signal rather than the display compositor used as a
    proxy for harm.
    """
    b = _BEACON
    if b is None:
        return None
    verdict, reason = b.poll()
    if verdict in (PEER_STALLED, PEER_DEAD, PEER_EXITING):
        return f"{verdict}: {reason}"
    return None
