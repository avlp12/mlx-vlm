"""A peer that has left must be noticed BEFORE the next collective (I1528).

On 2026-09-08 rank 1 died with TPDesync, ran its ``finally``, released its 85
GiB shard and announced EXITING on the heartbeat side-channel.  Rank 0 spun at
~200% CPU for forty minutes, ignored three SIGTERMs and a POST /unload, and had
to be left wedged.  Three things were wrong, and all three are tested here.

1. NOBODY WAS LISTENING.  ``init_beacon`` had exactly one caller,
   ``tp/worker.py`` (rank 1).  Rank 0 never started a beacon, so rank 1's
   EXITING announcement -- the whole point of the side-channel -- had no
   receiver.  ``maybe_load_tp`` now starts one.

2. NOTHING CHECKED BEFORE ENTERING.  ``_ctrl_send`` walked into ``all_sum``
   unconditionally.  Once inside, there is no way back: the wait is in
   jaccl/Metal and no signal, thread or timer on the host can preempt it
   (transport.Deadman says so in its own docstring).  It now raises
   ``TPPeerGone`` first, which unwinds the generation thread normally and lets
   the server's ordinary unload release the model -- no signals, no os._exit.

3. THE WATCHDOG HAD NOTHING ARMED.  ``_Watchdog`` only times an ``_inflight``
   step, and the MTP verify never armed one (see test_tp_mtp_mirror.py).  Its
   poll now also reads a peer verdict, which does not depend on arming.
"""

import contextlib
import socket
import threading
import time
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mlx_vlm.tp.heartbeat as HB
import mlx_vlm.tp.worker as W
from mlx_vlm.server import tp_mode as T

mx.set_default_device(mx.cpu)


@pytest.fixture(autouse=True)
def _clean():
    W.clear_peer_gone()
    yield
    W.clear_peer_gone()


class _CountingSum:
    """An all_sum that must never be reached once the peer is known gone."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return x


def _patch_transport(monkeypatch, all_sum):
    import mlx_vlm.tp.transport as X

    monkeypatch.setattr(X, "all_sum", all_sum)
    monkeypatch.setattr(X, "driving", lambda: contextlib.nullcontext())
    monkeypatch.setattr(X, "set_epoch", lambda e: None)


def test_ctrl_send_refuses_before_the_collective(monkeypatch):
    monkeypatch.setenv(W.ENV_MAX_TOK, "64")
    summed = _CountingSum()
    _patch_transport(monkeypatch, summed)
    W.mark_peer_gone("rank 1 exited (test)")
    with pytest.raises(W.TPPeerGone, match="rank 1 exited"):
        W._ctrl_send(W.OP_FORWARD, 1, mx.zeros((1, 4), dtype=mx.int32))
    assert summed.calls == 0, (
        "the collective was entered anyway -- that is the unpreemptible spin")


def test_even_exit_does_not_enter_the_collective(monkeypatch):
    """A farewell to a peer that is gone is the same infinite wait.

    Both callers of the EXIT verb already fall back to reaping the peer over
    ssh, which is the only thing left that can act on it.
    """
    monkeypatch.setenv(W.ENV_MAX_TOK, "64")
    summed = _CountingSum()
    _patch_transport(monkeypatch, summed)
    W.mark_peer_gone("rank 1 exited (test)")
    with pytest.raises(W.TPPeerGone):
        W._ctrl_send(W.OP_EXIT, 1, None)
    assert summed.calls == 0


def test_the_refusal_is_bounded(monkeypatch):
    monkeypatch.setenv(W.ENV_MAX_TOK, "64")
    _patch_transport(monkeypatch, _CountingSum())
    W.mark_peer_gone("gone")
    t0 = time.monotonic()
    with pytest.raises(W.TPPeerGone):
        W._ctrl_send(W.OP_FORWARD, 1, mx.zeros((1, 1), dtype=mx.int32))
    assert time.monotonic() - t0 < 1.0


# ---------------------------------------------------------------- the sockets
def _free_udp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _beacon_pair(monkeypatch):
    """Two real UDP beacons on loopback: rank 0's and rank 1's."""
    p0, p1 = _free_udp_port(), _free_udp_port()
    monkeypatch.setenv("MLX_VLM_TP_HB_LOCAL", f"127.0.0.1:{p0}")
    monkeypatch.setenv("MLX_VLM_TP_HB_PEER", f"127.0.0.1:{p1}")
    b0 = HB.Beacon(0, 2)
    monkeypatch.setenv("MLX_VLM_TP_HB_LOCAL", f"127.0.0.1:{p1}")
    monkeypatch.setenv("MLX_VLM_TP_HB_PEER", f"127.0.0.1:{p0}")
    b1 = HB.Beacon(1, 2)
    return b0, b1


def test_rank0_hears_the_peer_leave_and_the_next_verb_refuses(monkeypatch):
    """End to end over two sockets: rank 1 announces EXITING, rank 0 raises.

    This is the incident's exact sequence.  Rank 1's ``worker_loop`` finally
    calls ``shutdown_beacon(announce_exit=True)`` on its way out of TPDesync;
    all that was missing on 2026-09-08 was a beacon on rank 0 to receive it.
    """
    monkeypatch.setenv(W.ENV_MAX_TOK, "64")
    summed = _CountingSum()
    _patch_transport(monkeypatch, summed)
    b0, b1 = _beacon_pair(monkeypatch)
    monkeypatch.setattr(HB, "_BEACON", b0)
    b0.start()
    try:
        assert W.peer_gone() is None, "a peer we have not heard from is not dead"
        b1.note(HB.STATE_EXITING)
        b1.sock.sendto(b1.snapshot().pack(), b1.peer)
        deadline = time.monotonic() + 5.0
        while W.peer_gone() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        gone = W.peer_gone()
        assert gone is not None and "EXITING" in gone, f"never noticed: {gone!r}"
        with pytest.raises(W.TPPeerGone):
            W._ctrl_send(W.OP_FORWARD, 1, mx.zeros((1, 4), dtype=mx.int32))
        assert summed.calls == 0
    finally:
        b0.stop()
        b1.sock.close()


def test_a_silent_peer_is_not_a_dead_one(monkeypatch):
    """UNKNOWN must never gate a verb: formation silence is not death."""
    b0, b1 = _beacon_pair(monkeypatch)
    monkeypatch.setattr(HB, "_BEACON", b0)
    try:
        assert W.peer_gone() is None
    finally:
        b0.sock.close()
        b1.sock.close()


# --------------------------------------------------------------- the watchdog
def test_watchdog_fires_on_a_gone_peer_without_waiting_out_the_timeout():
    fired = []
    w = T._Watchdog(300.0, poll_s=0.01,
                    on_timeout=lambda label, waited: fired.append(label),
                    peer_probe=lambda: "peer announced EXITING",
                    on_peer_gone=lambda why: None)
    w.arm("forward b=1 s=4")
    w.poll_once()
    assert fired and "peer gone" in fired[0], (
        "a step in flight against a dead peer must not wait out 300 s")


def test_watchdog_only_records_when_no_step_is_in_flight():
    """An orderly shutdown takes the peer away too; that must not abort us."""
    fired, marked = [], []
    w = T._Watchdog(300.0, poll_s=0.01,
                    on_timeout=lambda label, waited: fired.append(label),
                    peer_probe=lambda: "peer announced EXITING",
                    on_peer_gone=marked.append)
    w.poll_once()
    assert not fired, "aborted the process over an idle peer"
    assert marked == ["peer announced EXITING"], \
        "the verdict must still be recorded, so the next verb refuses"


def test_watchdog_still_times_out_a_slow_step():
    fired = []
    w = T._Watchdog(0.0, poll_s=0.01,
                    on_timeout=lambda label, waited: fired.append(label),
                    peer_probe=lambda: None)
    w.arm("forward b=1 s=4")
    time.sleep(0.01)
    w.poll_once()
    assert fired == ["forward b=1 s=4"]


def test_watchdog_marks_the_peer_by_default(monkeypatch):
    """The default ``on_peer_gone`` is the module gate, so the two connect."""
    w = T._Watchdog(300.0, poll_s=0.01, on_timeout=lambda *a: None,
                    peer_probe=lambda: "PEER_DEAD: no beat for 12s")
    w.poll_once()
    assert W.peer_gone() == "PEER_DEAD: no beat for 12s"


# ------------------------------------------------------------- the unwind
class _FakeLM:
    config = "cfg"

    def __call__(self, ids, cache=None, **kw):
        return SimpleNamespace(logits=mx.zeros((1, 1, 4)))

    def make_cache(self):
        return [SimpleNamespace(offset=0)]


def test_the_mirror_unwinds_and_releases_without_a_signal(monkeypatch):
    """What the operator could not get on the day: the model, back, by unwinding."""
    monkeypatch.setattr(T, "_reap_peer_workers", lambda hosts: None)
    monkeypatch.setattr(T, "tp_hosts", lambda: [])
    m = T.MirroredLanguageModel(_FakeLM())
    W.mark_peer_gone("rank 1 exited with TPDesync")

    def _send(op, epoch, ids, **kw):
        raise W.TPPeerGone("peer gone")

    monkeypatch.setattr(T, "_ctrl_send", _send)
    with pytest.raises(W.TPPeerGone):
        m(mx.zeros((1, 1), dtype=mx.int32), cache=[SimpleNamespace(offset=0)])
    # ... and teardown completes anyway, dropping the model reference, which is
    # what "the model it holds is NOT released yet" was complaining about.
    assert m.shutdown() is False
    assert m._lm is None and m._wire is None


def test_rank0_starts_its_own_beacon(monkeypatch):
    """The one-line omission at the root of (1): rank 0 never called this."""
    calls = []

    def _init(rank, size=2, **kw):
        calls.append((rank, size))
        return SimpleNamespace(note=lambda *a, **k: None)

    monkeypatch.setattr(HB, "init_beacon", _init)
    assert T._start_rank0_beacon(["10.0.0.1", "10.0.0.2"]) is True
    assert calls == [(0, 2)]


def test_a_beacon_that_will_not_start_does_not_stop_the_serve(monkeypatch):
    def _boom(rank, size=2, **kw):
        raise OSError("no such interface")

    monkeypatch.setattr(HB, "init_beacon", _boom)
    assert T._start_rank0_beacon(["10.0.0.1", "10.0.0.2"]) is False
