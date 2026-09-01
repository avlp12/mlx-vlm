"""The handshake is the only thing standing between us and silent corruption.

Measured on this fleet: a collective whose ranks disagree about SIZE does not
fail on jaccl.  With rank 0 contributing 8 elements and rank 1 contributing 256,
jaccl summed the overlapping prefix, handed 3.0 to both ranks, and said nothing.
Ring raises "connection to a peer was lost" for the same input.

So a width disagreement would corrupt quietly on the fast backend, and only the
formation-time handshake catches it.  These tests pin that guard.
"""
import pytest

from mlx_vlm.tp import worker


def test_handshake_refuses_a_protocol_mismatch(monkeypatch):
    import mlx.core as mx

    mine = [worker.PROTO_VERSION, worker.HEADER, worker._max_tok(), 0, 0, 0, 0, 0]
    # peer reports a different protocol version
    peer = list(mine); peer[0] += 1
    total = [a + b for a, b in zip(mine, peer)]

    monkeypatch.setattr(worker, "_max_tok", lambda: mine[2])
    import mlx_vlm.tp.transport as tr
    monkeypatch.setattr(tr, "all_sum", lambda x: mx.array([total], dtype=mx.int32))
    monkeypatch.setattr(tr, "tp_size", lambda: 2)

    with pytest.raises(worker.TPUnavailable, match="proto_version mismatch"):
        worker._proto_handshake(timeout_s=5)


def test_handshake_refuses_a_payload_width_mismatch(monkeypatch):
    import mlx.core as mx

    mine = [worker.PROTO_VERSION, worker.HEADER, worker._max_tok(), 0, 0, 0, 0, 0]
    peer = list(mine); peer[2] += 64          # different MLX_VLM_GLM5_TP_MAX_TOK
    total = [a + b for a, b in zip(mine, peer)]

    import mlx_vlm.tp.transport as tr
    monkeypatch.setattr(tr, "all_sum", lambda x: mx.array([total], dtype=mx.int32))
    monkeypatch.setattr(tr, "tp_size", lambda: 2)

    with pytest.raises(worker.TPUnavailable, match="max_tokens mismatch"):
        worker._proto_handshake(timeout_s=5)


def test_handshake_accepts_agreement(monkeypatch):
    import mlx.core as mx

    mine = [worker.PROTO_VERSION, worker.HEADER, worker._max_tok(), 0, 0, 0, 0, 0]
    total = [v * 2 for v in mine]
    import mlx_vlm.tp.transport as tr
    monkeypatch.setattr(tr, "all_sum", lambda x: mx.array([total], dtype=mx.int32))
    monkeypatch.setattr(tr, "tp_size", lambda: 2)
    got = worker._proto_handshake(timeout_s=5)
    assert got["proto_version"] == worker.PROTO_VERSION
