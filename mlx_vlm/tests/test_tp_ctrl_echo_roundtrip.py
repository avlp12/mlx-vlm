"""End-to-end rank0/rank1 round trip through the real _ctrl_send/_ctrl_recv,
covering the negation bug: rank 1 must contribute -_LAST_SHAPE, not zeros.

``worker._LAST_SHAPE`` is a single module global; in production each rank is
a separate process with its own copy, so the two roles never share state.
Here we fake that separation by pointing ``worker._LAST_SHAPE`` at a
different Python list before invoking each role's function -- the module
functions only ever look the name up at call time and mutate in place
(``_LAST_SHAPE[:] = ...``), so retargeting the module attribute between calls
keeps rank 0's and rank 1's shadows independent, exactly like two processes
would be.

The fake ``all_sum`` bridges the two sides' sequential calls into the single
combined result a real collective would hand to both ranks: rank 1's
contribution for the verb about to run is precomputed from its own shadow
(mirroring the production formula, so the test also serves as an assertion
that _ctrl_recv builds exactly that row) and staged before rank 0's send;
rank 0's call consumes it and produces the total; rank 1's subsequent
_ctrl_recv call is handed that same total regardless of what it itself
passes in (a real collective would have handed both ranks the identical
sum), with its own contribution checked against the staged value first.
"""
import pytest
import mlx.core as mx

from mlx_vlm.tp import worker
import mlx_vlm.tp.transport as tr


def _rank1_contrib_row(shape, n):
    row = [0] * (worker.HEADER + n)
    row[worker.HEADER + n - worker.ECHO_WORDS:] = [-int(v) for v in shape]
    return row


class _Bridge:
    """Stages rank 1's contribution, then feeds combined totals to both
    calls of all_sum that make up one verb round."""

    def __init__(self, n):
        self.n = n
        self._staged_contrib = None
        self._expected_contrib = None
        self._last_total = None

    def stage(self, rank1_shape):
        row = _rank1_contrib_row(rank1_shape, self.n)
        self._staged_contrib = row
        self._expected_contrib = row

    def fake_all_sum(self, x):
        row = [int(v) for v in x[0].tolist()]
        if self._staged_contrib is not None:
            # rank 0's send: combine with the pre-staged rank-1 contribution.
            total = [a + b for a, b in zip(row, self._staged_contrib)]
            self._last_total = total
            self._staged_contrib = None
            return mx.array([total], dtype=mx.int32)
        # rank 1's recv: this row IS rank 1's real contribution, built inside
        # _ctrl_recv from its own _LAST_SHAPE shadow. It must equal what we
        # staged before rank 0's send (same shadow, read before either side
        # mutated it) -- this is the actual assertion that _ctrl_recv
        # contributes -_LAST_SHAPE, not zeros (the bug this fix closes).
        assert self._last_total is not None, "rank 1 recv'd before rank 0 sent"
        assert row == self._expected_contrib, (
            "rank 1's contribution did not match -_LAST_SHAPE: got "
            f"{row!r}, expected {self._expected_contrib!r}")
        total = self._last_total
        self._last_total = None
        self._expected_contrib = None
        return mx.array([total], dtype=mx.int32)


@pytest.fixture
def bridge(monkeypatch):
    n = worker._max_tok()
    b = _Bridge(n)
    monkeypatch.setattr(tr, "all_sum", b.fake_all_sum)
    return b


def _round(bridge, rank0_shape, rank1_shape, op, epoch, ids):
    """One verb: rank 0 sends, rank 1 receives. Returns rank1's decoded msg."""
    orig_last_shape = worker._LAST_SHAPE
    try:
        bridge.stage(rank1_shape)
        worker._LAST_SHAPE = rank0_shape
        worker._ctrl_send(op, epoch, ids)

        worker._LAST_SHAPE = rank1_shape
        msg = worker._ctrl_recv()
        return msg
    finally:
        worker._LAST_SHAPE = orig_last_shape


def test_make_cache_then_two_forwards_no_desync(bridge):
    """MAKE_CACHE(epoch=1) -> FORWARD -> FORWARD must not raise TPDesync.

    This is exactly the sequence that died on op2 before the fix: rank 1
    contributing zeros made the post-MAKE_CACHE echo read as "epoch: ranks
    differ by 1" on the first FORWARD.
    """
    r0_shape = [0, 0, 0]
    r1_shape = [0, 0, 0]

    msg = _round(bridge, r0_shape, r1_shape, worker.OP_MAKE_CACHE, 1, None)
    assert msg.op == worker.OP_MAKE_CACHE
    assert r0_shape == r1_shape == [0, 0, 1]

    ids = mx.array([[7] * 15], dtype=mx.int32)
    msg = _round(bridge, r0_shape, r1_shape, worker.OP_FORWARD, 1, ids)
    assert msg.op == worker.OP_FORWARD
    assert msg.batch == 1 and msg.seqlen == 15
    assert r0_shape == r1_shape == [1, 15, 1]

    msg = _round(bridge, r0_shape, r1_shape, worker.OP_FORWARD, 1, ids)
    assert msg.op == worker.OP_FORWARD
    assert r0_shape == r1_shape == [1, 15, 1]


def test_genuine_mismatch_raises_tpdesync(bridge):
    """If rank 1's tracked previous shape has actually drifted from rank 0's,
    the echo must not cancel and TPDesync must be raised -- not silently
    accepted."""
    r0_shape = [1, 15, 1]
    r1_shape = [1, 14, 1]  # rank 1 thinks the last seqlen was 14, not 15

    ids = mx.array([[7] * 15], dtype=mx.int32)
    with pytest.raises(worker.TPDesync, match="seqlen"):
        _round(bridge, r0_shape, r1_shape, worker.OP_FORWARD, 1, ids)


def test_rank1_pins_ctrl_batch_seqlen_attribute_names():
    """_decode_checked reads msg.batch/msg.seqlen -- pin those field names so
    a rename of Ctrl silently reintroduces the AttributeError this fix also
    closed (the dataclass/NamedTuple never had .b/.s)."""
    fields = worker.Ctrl._fields
    assert "batch" in fields and "seqlen" in fields
    assert "b" not in fields and "s" not in fields
