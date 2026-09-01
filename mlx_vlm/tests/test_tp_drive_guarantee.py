"""Rank 0 must never leave rank 1 blocked in the control wait.

Rank 1 spends its life inside a blocking all_sum waiting for the next verb, so
any path where rank 0 stops issuing verbs orphans it -- holding its entire shard
-- until something kills the box.  That is the whole TP hang family, and it was
reproduced directly: with rank 0 alive-but-idle after N-1 collectives, rank 1
blocked on its Nth and had to be killed by a timeout.

These tests force each way rank 0 can stop and assert an EXIT went out.
"""
import pytest

from mlx_vlm.server import tp_mode


class _Mirror:
    """Just enough of the mirror to exercise _release_peer / _announce."""
    _release_peer = tp_mode.MirroredLanguageModel._release_peer
    _announce = tp_mode.MirroredLanguageModel._announce

    def __init__(self):
        self._closed = False
        self._epoch = 7
        self._watchdog = None


@pytest.fixture
def sent(monkeypatch):
    calls = []
    monkeypatch.setattr(tp_mode, "_ctrl_send",
                        lambda op, epoch, ids, **kw: calls.append((op, epoch)))
    monkeypatch.setattr(tp_mode, "_reap_peer_workers", lambda hosts: None)
    monkeypatch.setattr(tp_mode, "tp_hosts", lambda: [])
    return calls


def test_announce_failure_releases_the_peer(monkeypatch, sent):
    """A raise inside the verb send must still free rank 1."""
    m = _Mirror()
    boom = RuntimeError("link died mid-verb")

    def fail(op, epoch, ids, **kw):
        if op != tp_mode.OP_EXIT:
            raise boom
        sent.append((op, epoch))

    monkeypatch.setattr(tp_mode, "_ctrl_send", fail)
    with pytest.raises(RuntimeError):
        m._announce(tp_mode.OP_FORWARD, None)
    assert (tp_mode.OP_EXIT, 7) in sent, "peer was orphaned by a failed announce"


def test_release_is_idempotent(sent):
    """Two releases must not send two EXITs -- rank 1 is already gone."""
    m = _Mirror()
    m._release_peer("first")
    m._release_peer("second")
    assert sent.count((tp_mode.OP_EXIT, 7)) == 1


def test_release_falls_back_to_reaping(monkeypatch):
    """If EXIT cannot be sent, the peer must still be reaped, not abandoned."""
    reaped = []
    monkeypatch.setattr(tp_mode, "_ctrl_send",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no link")))
    monkeypatch.setattr(tp_mode, "_reap_peer_workers", lambda hosts: reaped.append(hosts))
    monkeypatch.setattr(tp_mode, "tp_hosts", lambda: ["10.0.0.2"])
    m = _Mirror()
    m._release_peer("link gone")
    assert reaped == [["10.0.0.2"]], "peer neither released nor reaped"


def test_a_closed_mirror_does_not_resend(sent):
    m = _Mirror()
    m._closed = True
    m._release_peer("already shut")
    assert sent == []
