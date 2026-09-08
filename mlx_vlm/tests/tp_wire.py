"""A two-rank TP transport, in one process, for CPU tests.

The TP control plane has no side channel: rank 0 fills an int32 vector, rank 1
contributes zeros, and the sum is the message.  Pairing is therefore positional
-- the i-th collective of rank 0 meets the i-th of rank 1 -- and "rank 0 issued
a collective rank 1 did not" is invisible to the transport.  jaccl does not
report a size mismatch either (tp/worker.py records 8 elements against 256
completing silently and returning 3.0 to both ranks), which is why the reserved
echo words exist.

This module models exactly that, so a test can drive both ranks and watch a
desync appear.  It is a helper, not a test file.
"""

from __future__ import annotations

import contextlib
import threading

import mlx.core as mx

import mlx_vlm.tp.worker as W


class PeerNeverCame(Exception):
    """A collective whose other half never arrived: the hang, bounded."""


class Wire:
    """Pair the i-th collective of rank 0 with the i-th of rank 1.

    On a size mismatch each rank gets its own buffer plus whatever prefix of the
    peer's overlaps it.  That is the conservative model, and it preserves the
    property the echo agreement was built on: the reserved words at the TAIL of
    the control vector keep the contributor's own value when the peer's buffer
    is a different length, so they stop cancelling.
    """

    def __init__(self, timeout=10.0):
        self.slots = {}
        self.cv = threading.Condition()
        self.timeout = timeout

    def exchange(self, rank, row):
        with self.cv:
            i = self.slots.get(f"n{rank}", 0)
            self.slots[f"n{rank}"] = i + 1
            self.slots[(rank, i)] = list(row)
            self.cv.notify_all()
            peer = 1 - rank
            if not self.cv.wait_for(lambda: (peer, i) in self.slots, self.timeout):
                raise PeerNeverCame(
                    f"rank {rank} waited {self.timeout}s for the peer's "
                    f"collective #{i}: this is the live hang, in a test")
            other = self.slots[(peer, i)]
        out = list(row)
        for k in range(min(len(out), len(other))):
            out[k] += other[k]
        return out

    def data_reduce(self, rank, width=32):
        """One collective from inside a sharded forward: not a control message."""
        self.exchange(rank, [0] * width)


def patch_transport(monkeypatch, wire):
    """One ``all_sum`` for both ranks; the caller is identified by thread name."""
    import mlx_vlm.tp.transport as X

    def all_sum(x):
        rank = 0 if threading.current_thread().name == "rank0" else 1
        row = x[0].tolist() if x.ndim == 2 else x.reshape(-1).tolist()
        return mx.array([wire.exchange(rank, [int(v) for v in row])],
                        dtype=mx.int32)

    monkeypatch.setattr(X, "all_sum", all_sum)
    monkeypatch.setattr(X, "driving", lambda: contextlib.nullcontext())
    monkeypatch.setattr(X, "set_epoch", lambda e: None)


def rank1_worker_module():
    """A SECOND copy of tp.worker, so rank 1 has its own ``_LAST_SHAPE``.

    The echo agreement compares two per-rank module globals.  Running both ranks
    against one import would compare a value with itself and could never fail --
    which is to say it would never reproduce anything.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_tp_worker_rank1", W.__file__)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "mlx_vlm.tp"          # so ``from ..tp.transport`` resolves
    spec.loader.exec_module(mod)
    return mod


def run_pair(monkeypatch, rank0_script, rank1_body, *, max_tok=64, join_s=25.0):
    """Drive both ranks to completion; return (rank 0 error, rank 1 error).

    ``rank0_script(wire)`` and ``rank1_body(wire, R1)`` run on threads named
    ``rank0``/``rank1``; ``R1`` is rank 1's private copy of tp.worker.
    """
    monkeypatch.setenv(W.ENV_MAX_TOK, str(max_tok))
    wire = Wire()
    patch_transport(monkeypatch, wire)
    R1 = rank1_worker_module()
    W._LAST_SHAPE[:] = [0, 0, 0]
    R1._LAST_SHAPE[:] = [0, 0, 0]
    W.clear_peer_gone()
    R1.clear_peer_gone()
    out = {}

    def r0():
        try:
            rank0_script(wire)
        except BaseException as e:            # noqa: BLE001 - reported, not raised
            out["r0"] = e

    def r1():
        try:
            rank1_body(wire, R1)
        except BaseException as e:            # noqa: BLE001
            out["r1"] = e

    t0 = threading.Thread(target=r0, name="rank0")
    t1 = threading.Thread(target=r1, name="rank1")
    t1.start(); t0.start()
    t0.join(join_s); t1.join(join_s)
    assert not t0.is_alive() and not t1.is_alive(), "a rank never finished"
    return out.get("r0"), out.get("r1")
