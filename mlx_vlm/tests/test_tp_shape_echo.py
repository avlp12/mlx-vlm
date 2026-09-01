"""Per-verb shape agreement, carried inside the control message.

The formation handshake proves the ranks agree on protocol version and vector
widths ONCE. Nothing checked it again, and the fast backend will not: a
collective whose ranks disagree about SIZE completes silently on jaccl --
measured on this fleet at 8 elements against 256, returning 3.0 to both ranks
with no diagnostic, where ring raises "connection to a peer was lost".

So a runtime disagreement about batch or sequence length is not a crash, it is
wrong output. These tests pin the check that makes it loud.
"""
import pytest

from mlx_vlm.tp import worker


def _row(op=worker.OP_FORWARD, epoch=1, shape=None, flat=None, echo=None):
    return worker.encode(op, epoch, shape, flat, n=worker._max_tok(), echo=echo)


def test_agreement_reads_as_zero():
    """rank0's +view plus rank1's -view cancels when they agree."""
    mine = [2, 8, 5]
    r0 = _row(echo=mine)
    r1 = _row(echo=[-v for v in mine])
    summed = [a + b for a, b in zip(r0, r1)]
    assert worker.read_echo(summed) == [0, 0, 0]
    assert worker._shape_disagreement(summed) is None


@pytest.mark.parametrize("field,idx,delta", [("batch", 0, 1),
                                             ("seqlen", 1, -3),
                                             ("epoch", 2, 7)])
def test_disagreement_is_named_not_silent(field, idx, delta):
    """A drift in any field must be reported, and must name the field."""
    mine = [2, 8, 5]
    theirs = list(mine); theirs[idx] += delta
    r0 = _row(echo=mine)
    r1 = _row(echo=[-v for v in theirs])
    summed = [a + b for a, b in zip(r0, r1)]
    drift = worker._shape_disagreement(summed)
    assert drift is not None, f"a {field} disagreement went undetected"
    assert field in drift
    assert str(-delta) in drift or str(delta) in drift


def test_echo_survives_a_full_payload():
    """The reserved words must not be clobbered by a maximum-size forward."""
    n = worker._max_tok()
    room = n - worker.ECHO_WORDS
    flat = list(range(room))
    row = _row(shape=(1, room), flat=flat, echo=[4, 9, 2])
    assert worker.read_echo(row) == [4, 9, 2]
    assert worker.decode(row).ids == flat


def test_a_forward_that_would_clobber_the_echo_is_refused():
    n = worker._max_tok()
    with pytest.raises(worker.TPUnavailable, match="reserved agreement words"):
        _row(shape=(1, n), flat=list(range(n)))


def test_proto_version_bumped_so_mixed_ranks_fail_at_formation():
    """An old peer cannot silently skip the check -- the handshake catches it."""
    assert worker.PROTO_VERSION >= 3
