"""Unit + loopback tests for the TP progress side-channel (mlx_vlm/tp/heartbeat.py).

The three rank-1 rules are tested against a FAKE socket and a FAKE clock, so the
tests are deterministic and take no wall time:

    IDLE                            -> wait indefinitely (never kill)
    DRIVING with fwd_idx frozen     -> PEER_STALLED
    no beat at all                  -> PEER_DEAD

plus the two properties that make the design safe:

    never-heard-from-peer           -> UNKNOWN, not a kill
    malformed / foreign datagram    -> ignored, cannot influence a verdict

There is deliberately no MLX import here: heartbeat.py is stdlib-only, so this
whole file runs on a box with no model and no GPU.
"""
import importlib.util
import os
import socket
import sys
import threading
import time

import pytest

_HB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "mlx_vlm", "tp", "heartbeat.py")
_spec = importlib.util.spec_from_file_location("mlx_vlm_tp_heartbeat_under_test", _HB)
hb = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = hb
_spec.loader.exec_module(hb)


class FakeClock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, dt): self.t += dt


class FakeSocket:
    """Captures sends; never touches the network."""
    local = ("fake-local", 39600)
    peer = ("fake-peer", 39600)

    def __init__(self): self.sent = []
    def sendto(self, buf, addr): self.sent.append((buf, addr))
    def recvfrom(self, n): raise TimeoutError
    def settimeout(self, t): pass
    def setsockopt(self, *a): pass
    def bind(self, a): pass
    def close(self): pass


def make(rank=1, clock=None):
    """A beacon with no threads started -- poll() is a pure function of state."""
    return hb.Beacon(rank, 2, sock=FakeSocket(), clock=clock or FakeClock())


def beat_from_peer(b, *, state, epoch=1, fwd_idx=0, verb_seq=0, rank=0):
    return hb.Beat(state, rank, epoch, fwd_idx, verb_seq, 0).pack()


# --------------------------------------------------------------- wire format
def test_beat_roundtrip():
    b = hb.Beat(hb.STATE_DRIVING, 0, 3, 57, 12, 999)
    assert hb.Beat.unpack(b.pack()) == b


@pytest.mark.parametrize("junk", [b"", b"short", b"X" * hb.BEAT_BYTES,
                                  b"GLM5HB01" + b"\x00" * 99])
def test_malformed_datagrams_are_ignored(junk):
    """A UDP socket receives whatever the network sends it. Junk must never be
    able to move a liveness verdict."""
    c = FakeClock(); b = make(clock=c)
    b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=5))
    before = b.poll()
    assert b.ingest(junk) is None
    assert b.poll() == before


def test_own_packets_are_ignored():
    """Both ranks bind the same port; a looped-back self-beat must not count as
    the peer being alive."""
    b = make(rank=1)
    assert b.ingest(hb.Beat(hb.STATE_DRIVING, 1, 1, 9, 9, 0).pack()) is None
    assert b.poll()[0] == hb.UNKNOWN


# ------------------------------------------------------------- the three rules
def test_unknown_before_first_beat_is_not_a_kill():
    """The removed ppid probe killed healthy rank 1s twelve seconds in. Silence
    at startup must never be a kill signal."""
    c = FakeClock(); b = make(clock=c)
    c.advance(3600.0)
    assert b.poll() == (hb.UNKNOWN, None)


def test_idle_peer_waits_indefinitely():
    """RULE 1. A serving deployment can sit idle for hours. IDLE must never
    escalate, no matter how long."""
    c = FakeClock(); b = make(clock=c)
    for _ in range(200):
        b.ingest(beat_from_peer(b, state=hb.STATE_IDLE))
        c.advance(0.25)
    assert b.poll()[0] == hb.HEALTHY
    # ... even though its counters never moved for far longer than stall_s
    assert c.t - b._peer_progress_at > b.stall_s


def test_driving_with_progress_is_healthy():
    """verb_seq is the counter that must advance: fwd_idx is reset to 0 at every
    forward boundary by transport.reset_forward_counter, so it cannot carry the
    ordering (see Beacon._progress_key)."""
    c = FakeClock(); b = make(clock=c)
    for i in range(400):
        b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING,
                                fwd_idx=i % 101, verb_seq=i))
        c.advance(0.25)
        assert b.poll()[0] == hb.HEALTHY


def test_driving_with_frozen_fwd_idx_is_stalled():
    """RULE 2, the jaccl vault-class hang: packets keep arriving because the
    sender thread is fine, but the driving thread stopped advancing."""
    c = FakeClock(); b = make(clock=c)
    b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=57))
    assert b.poll()[0] == hb.HEALTHY
    # keep beating the SAME counters, well past stall_s but always within dead_s
    for _ in range(int(b.stall_s / 0.25) + 8):
        c.advance(0.25)
        b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=57))
    verdict, reason = b.poll()
    assert verdict == hb.PEER_STALLED
    assert "fwd_idx=57" in reason
    assert "driving thread" in reason   # the verdict localises the wedge


def test_no_beat_is_dead():
    """RULE 3: the process or the box is gone."""
    c = FakeClock(); b = make(clock=c)
    b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=1))
    c.advance(b.dead_s + 1.0)
    verdict, reason = b.poll()
    assert verdict == hb.PEER_DEAD
    assert "no beat" in reason


def test_dead_takes_precedence_over_idle():
    """An IDLE peer that then dies must still be reported dead -- otherwise
    'IDLE waits forever' would mask a real death."""
    c = FakeClock(); b = make(clock=c)
    b.ingest(beat_from_peer(b, state=hb.STATE_IDLE))
    c.advance(b.dead_s + 1.0)
    assert b.poll()[0] == hb.PEER_DEAD


def test_exiting_releases_immediately():
    c = FakeClock(); b = make(clock=c)
    b.ingest(beat_from_peer(b, state=hb.STATE_EXITING))
    assert b.poll()[0] == hb.PEER_EXITING


def _beat_for(b, c, seconds, **kw):
    """Feed beats at the real 4 Hz rate for `seconds` of fake time.

    Advancing the clock in one jump instead would trip dead_s (10 s) long before
    stall_s (30 s) and test nothing about staleness -- which is exactly what the
    first version of this test did."""
    for _ in range(int(seconds / 0.25)):
        c.advance(0.25)
        b.ingest(beat_from_peer(b, **kw))


def test_stall_clock_resets_on_any_counter_change():
    """epoch or verb_seq moving is progress too, not just fwd_idx."""
    c = FakeClock(); b = make(clock=c)
    b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=1, verb_seq=1))
    _beat_for(b, c, b.stall_s - 1.0, state=hb.STATE_DRIVING, fwd_idx=1, verb_seq=1)
    assert b.poll()[0] == hb.HEALTHY          # not yet stale
    # one beat with a CHANGED counter restarts the stall clock ...
    c.advance(0.25)
    b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=1, verb_seq=2))
    # ... so another nearly-full stall window still reads healthy
    _beat_for(b, c, b.stall_s - 1.0, state=hb.STATE_DRIVING, fwd_idx=1, verb_seq=2)
    assert b.poll()[0] == hb.HEALTHY          # the stall clock restarted


def test_silence_is_diagnosed_before_freeze():
    """dead_s (10 s) < stall_s (30 s) by default, and the ordering matters: a
    peer that stops sending is DEAD, not STALLED, because a frozen driving thread
    still has a live sender thread. Getting this backwards would report every
    dead peer as a wedge and send lane 1 hunting the wrong failure."""
    c = FakeClock(); b = make(clock=c)
    assert b.dead_s < b.stall_s
    b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=7))
    c.advance(b.dead_s + 0.5)                  # silence, shorter than stall_s
    assert b.poll()[0] == hb.PEER_DEAD


# ------------------------------------------------------------------ divergence
def test_divergence_names_the_reduce():
    """The whole point of carrying fwd_idx: transport.py keeps the counter so a
    stalled pair can be compared 'position by position', but until now there was
    no channel to compare it on."""
    b = make(rank=1)
    b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=57))
    msg = b.divergence(58)
    assert "#58" in msg and "#57" in msg
    assert b.divergence(57) is None


# ----------------------------------------------------------------- harm probe
def test_stall_probe_is_none_when_no_beacon():
    hb._BEACON = None
    assert hb.stall_probe(30.0) is None


def test_stall_probe_reports_stall():
    c = FakeClock(); b = make(clock=c)
    hb._BEACON = b
    try:
        b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=7))
        assert hb.stall_probe(30.0) is None
        for _ in range(int(b.stall_s / 0.25) + 8):
            c.advance(0.25)
            b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=7))
        out = hb.stall_probe(30.0)
        assert out and out.startswith(hb.PEER_STALLED)
    finally:
        hb._BEACON = None


# ------------------------------------------------------------------ addressing
def test_coordinator_port_is_refused(monkeypatch):
    """transport.py: a connect to the jaccl coordinator port is accepted as a
    rank joining and broke formation 1/1. Never point the beacon at it."""
    monkeypatch.setenv("MLX_VLM_TP_HB_PORT", "39500")
    with pytest.raises(ValueError, match="coordinator port"):
        hb.resolve_addrs(0)


def test_hosts_are_overridable(monkeypatch):
    """Moving to a real 10 GbE pair must be configuration, not code."""
    monkeypatch.setenv("MLX_VLM_TP_HB_HOSTS", "10.0.1.1,10.0.1.2")
    monkeypatch.setenv("MLX_VLM_TP_HB_PORT", "39600")
    local, peer = hb.resolve_addrs(0)
    assert local == ("10.0.1.1", 39600) and peer == ("10.0.1.2", 39600)
    local, peer = hb.resolve_addrs(1)
    assert local == ("10.0.1.2", 39600) and peer == ("10.0.1.1", 39600)


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("MLX_VLM_TP_HB", "0")
    assert hb.enabled() is False
    assert hb.init_beacon(0) is None


# ---------------------------------------------------------------- integration
@pytest.mark.timeout(30)
def test_loopback_two_beacons_real_sockets():
    """Two real UDP beacons on 127.0.0.1, real threads, real clock.

    Drives the full lifecycle: progress -> HEALTHY, freeze -> PEER_STALLED, and
    checks the reverse direction carries rank 1's own counters back so rank 0 can
    name a divergence too.
    """
    p0, p1 = _free_udp_port(), _free_udp_port()
    env = {"MLX_VLM_TP_HB_HZ": "40", "MLX_VLM_TP_HB_STALL_S": "1.0",
           "MLX_VLM_TP_HB_DEAD_S": "2.0"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    a = b = None
    try:
        a = _local_beacon(0, p0, p1)
        b = _local_beacon(1, p1, p0)
        a.start(); b.start()

        for i in range(1, 40):
            a.note(hb.STATE_DRIVING, epoch=1, fwd_idx=i, verb_seq=i)
            b.note(hb.STATE_DRIVING, epoch=1, fwd_idx=i, verb_seq=i)
            time.sleep(0.02)
        assert b.poll()[0] == hb.HEALTHY, b.poll()
        assert b.peer_beat() is not None and b.peer_beat().rank == 0

        # rank 0's driving thread wedges: it stops calling note(), but its sender
        # thread keeps transmitting the frozen snapshot.
        a.note(hb.STATE_DRIVING, epoch=1, fwd_idx=57, verb_seq=57)
        time.sleep(1.6)
        verdict, reason = b.poll()
        assert verdict == hb.PEER_STALLED, (verdict, reason)
        assert "fwd_idx=57" in reason
        # and rank 0 can name the divergence from its side
        b.note(hb.STATE_DRIVING, epoch=1, fwd_idx=58, verb_seq=58)
        time.sleep(0.15)
        assert "#57" in (a.divergence(57) or "") or a.divergence(57) is None
        assert "#58" in (a.divergence(57) or "#58")
    finally:
        for x in (a, b):
            if x is not None:
                x.stop()
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def _free_udp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _local_beacon(rank, my_port, peer_port):
    os.environ["MLX_VLM_TP_HB_LOCAL"] = f"127.0.0.1:{my_port}"
    os.environ["MLX_VLM_TP_HB_PEER"] = f"127.0.0.1:{peer_port}"
    try:
        return hb.Beacon(rank, 2)
    finally:
        os.environ.pop("MLX_VLM_TP_HB_LOCAL", None)
        os.environ.pop("MLX_VLM_TP_HB_PEER", None)


# ------------------------------------------- state / counter split semantics
# MLX is lazy: all_sum only BUILDS the reduce and mx.eval is where it blocks.
# So counters move at construction (note_progress) and the "a collective is
# outstanding" claim is made only around the eval (driving()).  These tests pin
# that split, because getting it wrong breaks the design in one of two ways:
# bracketing all_sum would never catch a wedge, and never bracketing would make
# every idle gap look like one.

def test_note_progress_does_not_change_state():
    b = make(rank=0)
    b.note(hb.STATE_IDLE)
    b.note_progress(fwd_idx=5, verb_seq=5)
    assert b.snapshot().state == hb.STATE_IDLE
    assert b.snapshot().fwd_idx == 5


def test_driving_sets_and_restores_state():
    b = make(rank=0)
    b.note(hb.STATE_IDLE)
    with b.driving():
        assert b.snapshot().state == hb.STATE_DRIVING
    assert b.snapshot().state == hb.STATE_IDLE


def test_driving_restores_on_exception():
    """A TPDesync raised out of mx.eval must not leave the peer believing we are
    still inside a collective."""
    b = make(rank=0)
    b.note(hb.STATE_IDLE)
    with pytest.raises(RuntimeError):
        with b.driving():
            raise RuntimeError("desync")
    assert b.snapshot().state == hb.STATE_IDLE


def test_exiting_survives_a_driving_block():
    b = make(rank=0)
    b.note(hb.STATE_EXITING)
    with b.driving():
        pass
    assert b.snapshot().state == hb.STATE_EXITING


def test_idle_between_requests_is_never_a_wedge():
    """THE central safety property.

    Rank 0 finishes a request and parks. Its counters are frozen -- there are no
    more reduces to construct -- for far longer than stall_s. Rank 1 must wait,
    because the peer is not claiming to be inside a collective. Killing here is
    the "worse bug" the original time-only bound was disabled to avoid.
    """
    c = FakeClock(); rx = make(rank=1, clock=c)
    tx = make(rank=0)
    with tx.driving():                      # a reduce runs ...
        tx.note_progress(epoch=1, fwd_idx=101, verb_seq=101)
        rx.ingest(tx.snapshot().pack())
    rx.ingest(tx.snapshot().pack())         # ... and returns: state back to IDLE
    for _ in range(int((b_stall := rx.stall_s) * 4) + 40):
        c.advance(0.25)
        rx.ingest(tx.snapshot().pack())     # same frozen counters, IDLE
    assert c.t - rx._peer_progress_at > b_stall
    assert rx.poll()[0] == hb.HEALTHY


def test_wedge_inside_a_collective_is_caught():
    """The complement: identical frozen counters, but the peer never left the
    driving block. This is the jaccl vault-class hang and it must be named."""
    c = FakeClock(); rx = make(rank=1, clock=c)
    tx = make(rank=0)
    cm = tx.driving()
    cm.__enter__()                          # enters the eval and never returns
    tx.note_progress(epoch=1, fwd_idx=57, verb_seq=57)
    for _ in range(int(rx.stall_s * 4) + 40):
        c.advance(0.25)
        rx.ingest(tx.snapshot().pack())     # beats keep arriving -- sender is fine
    verdict, reason = rx.poll()
    assert verdict == hb.PEER_STALLED
    assert "fwd_idx=57" in reason and "driving thread" in reason


# ------------------------------------------------- default path (10GbE, I97x)
def test_default_hosts_are_the_dedicated_10gbe_pair(monkeypatch):
    """The default must be the path with PHYSICAL independence from the
    Thunderbolt cable jaccl runs on, not merely the lowest-latency one.

    Measured 64 B TCP RTT p50: tbnet 65 us, 10GbE 215 us, tunnelled 180 us.
    tbnet is 3x faster and is still the WRONG choice -- it is the link whose
    failure the beat has to be able to report.  At 4 Hz all three are ~1000x
    faster than needed, so independence decides and latency does not.
    """
    monkeypatch.delenv("MLX_VLM_TP_HB_HOSTS", raising=False)
    monkeypatch.delenv("MLX_VLM_TP_HB_LOCAL", raising=False)
    monkeypatch.delenv("MLX_VLM_TP_HB_PEER", raising=False)
    monkeypatch.setenv("MLX_VLM_TP_HB_PORT", "39600")
    assert hb.DEFAULT_HOSTS == ("10.0.1.1", "10.0.1.2")
    local, peer = hb.resolve_addrs(0)
    assert local == ("10.0.1.1", 39600) and peer == ("10.0.1.2", 39600)
    local, peer = hb.resolve_addrs(1)
    assert local == ("10.0.1.2", 39600) and peer == ("10.0.1.1", 39600)


def test_fallback_hosts_still_declared():
    """If en0 is unplugged the tunnelled pair still catches the dominant
    (software) hang class, so it stays a documented fallback rather than being
    deleted."""
    assert hb.FALLBACK_HOSTS == ("169.254.30.147", "169.254.240.246")
    assert hb.FALLBACK_HOSTS != hb.DEFAULT_HOSTS


# ------------------------------------- progress = strict advance (lane 1, I98x)
# The stall timer used to reset whenever the progress key CHANGED. UDP is
# entitled to reorder, so an older beat arriving after a newer one restarted the
# window. That failed safe -- detection late by one window, never a false kill --
# but it is not the contract.

def test_reordered_datagram_does_not_reset_the_stall_timer():
    """An older beat arriving after a newer one must not look like progress."""
    c = FakeClock(); b = make(clock=c)
    b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=50, verb_seq=50))
    started = b._peer_progress_at
    c.advance(1.0)
    # the network hands us a beat from BEFORE the one we already have
    b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=49, verb_seq=49))
    assert b._peer_progress_at == started, "a reordered beat reset the stall timer"
    # ... and the peer still wedges on schedule
    for _ in range(int(b.stall_s / 0.25) + 8):
        c.advance(0.25)
        b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=50, verb_seq=50))
    assert b.poll()[0] == hb.PEER_STALLED


def test_reordered_datagram_still_counts_as_LIVENESS():
    """It does not feed stall_s, but it does prove the sender thread is alive, so
    it must still feed dead_s. Conflating the two would report a live-but-slow
    peer as dead."""
    c = FakeClock(); b = make(clock=c)
    b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=50, verb_seq=50))
    for _ in range(int(b.dead_s / 0.25) + 8):
        c.advance(0.25)
        b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=49, verb_seq=49))
    assert b.poll()[0] != hb.PEER_DEAD, "a live peer sending reorders was called dead"


def test_forward_boundary_is_progress_not_regression():
    """THE TRAP IN THE OBVIOUS FIX.

    transport.reset_forward_counter() sets _FWD_IDX back to 0 at the start of
    every forward (server/tp_mode.py:385 and tp/worker.py:566 both call it), so
    fwd_idx runs 1..101 and drops to 1 again. Ordering on a tuple that CONTAINS
    fwd_idx -- (epoch, fwd_idx, verb_seq) -- reads a forward boundary as a
    REGRESSION, so the stall timer stops resetting and PEER_STALLED fires on a
    perfectly healthy peer. That is a false kill, the failure the ppid probe was
    deleted for. Only (epoch, verb_seq) is monotonic.

    Modelled faithfully: a 4 Hz beat SAMPLES the counters, and at ~33-92 ms per
    forward several forwards complete between beats. So consecutive beats show
    fwd_idx at arbitrary phases of its 1..101 cycle while verb_seq accumulates
    monotonically. Sampling fwd_idx in lockstep with the beat would hide the bug,
    because the naive rule would still ratchet once per cycle.
    """
    import random
    rng = random.Random(20260902)               # seeded: deterministic, unsynchronised
    c = FakeClock(); b = make(clock=c)
    seq = 0
    # 2000 beats = 500 s at 4 Hz. Long enough that the naive rule's gap
    # distribution is exercised rather than sampled once: under it, progress
    # registers only when the sampled fwd_idx ties the running max, i.e. with
    # probability ~1/101 per beat, so gaps are geometric with mean ~101 beats
    # (25 s) and roughly 30% of them exceed the 30 s stall bound. Over ~20 gaps
    # the chance of never exceeding it is under 0.1%. A fixed stride would hit
    # exactly every 101 beats with zero variance and hide the bug -- it did.
    for n in range(1, 2000):
        seq += 7                                # ~7 forwards' worth of reduces
        fwd = rng.randint(1, 101)               # arbitrary phase of the cycle
        b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING,
                                fwd_idx=fwd, verb_seq=seq))
        c.advance(0.25)
        assert b.poll()[0] == hb.HEALTHY, (
            f"healthy peer reported {b.poll()[0]} at beat {n} "
            f"(fwd_idx={fwd}, verb_seq={seq}) -- fwd_idx is in the ordering")


def test_frozen_peer_is_never_rebaselined():
    """Regression guard. A wedged driver re-sends the SAME counters forever. An
    earlier version counted equal keys toward the restart run, re-baselined after
    8 beats, and made PEER_STALLED unreachable -- it silently disabled the whole
    module. Equal keys must be inert."""
    c = FakeClock(); b = make(clock=c)
    b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=57, verb_seq=57))
    frozen_at = b._peer_progress_at
    # long enough to pass BOTH the restart run and the stall bound
    n = max(b.REBASELINE_AFTER * 6, int(b.stall_s / 0.25) + 8)
    for _ in range(n):
        c.advance(0.25)
        b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=57, verb_seq=57))
    assert b._peer_progress_at == frozen_at, "a frozen peer re-baselined the timer"
    assert b.poll()[0] == hb.PEER_STALLED


def test_peer_restart_rebaselines_instead_of_stalling_forever():
    """If rank 0 restarts, its counters go back to zero and stay below the high
    water mark permanently. Without a re-baseline the timer would never reset and
    a healthy fresh peer would be reported stalled forever."""
    c = FakeClock(); b = make(clock=c)
    for i in range(1, 40):
        b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=i, verb_seq=i))
        c.advance(0.25)
    # peer restarts: epoch and verb_seq reset
    for i in range(1, b.REBASELINE_AFTER + 4):
        c.advance(0.25)
        b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, epoch=0,
                                fwd_idx=i, verb_seq=i))
    assert b.poll()[0] == hb.HEALTHY, "a restarted peer was reported stalled"
    # and it tracks the restarted peer from there on
    for i in range(b.REBASELINE_AFTER + 4, b.REBASELINE_AFTER + 40):
        c.advance(0.25)
        b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, epoch=0,
                                fwd_idx=i, verb_seq=i))
    assert b.poll()[0] == hb.HEALTHY


def test_isolated_reorders_never_accumulate_to_a_rebaseline():
    """A reorder every other beat must not eventually look like a restart: any
    in-order beat clears the run."""
    c = FakeClock(); b = make(clock=c)
    for i in range(2, 200, 2):
        b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING, fwd_idx=i, verb_seq=i))
        c.advance(0.125)
        b.ingest(beat_from_peer(b, state=hb.STATE_DRIVING,
                                fwd_idx=i - 1, verb_seq=i - 1))   # stale
        c.advance(0.125)
        assert b._peer_regressions < b.REBASELINE_AFTER
    assert b.poll()[0] == hb.HEALTHY
