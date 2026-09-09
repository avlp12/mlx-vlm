import math

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx_vlm.tp.gdn_rows import prepare_gdn_rows


def capture(steps=2, kernel=4, state=True, lower=-5.0):
    # H == B deliberately catches accidental shape-based constant filtering.
    batch, heads, dim = 3, 3, 2

    def data(shape):
        return mx.arange(math.prod(shape)).reshape(shape)

    q = data((batch, steps, heads, dim))
    return (
        q, q + 1, q + 2, q + 3, data((batch, steps, heads)),
        data((heads, 1)), data((heads, dim)),
        data((batch, heads, dim, dim)) if state else None,
        data((batch, steps + kernel - 1, 3 * heads * dim)), kernel, lower,
    )


@pytest.mark.parametrize("steps,kernel,state,lower", [
    (1, 4, True, -5.0), (4, 4, True, -5.0), (2, 1, False, None),
])
def test_producer_shapes_reorder_only_rows(steps, kernel, state, lower):
    original = capture(steps, kernel, state, lower)
    captures = [original]
    staged = prepare_gdn_rows(captures, [2, 0], batch_size=3)
    assert captures[0] is original and staged is not captures
    for field in (0, 1, 2, 3, 4, 7, 8):
        if original[field] is None:
            assert staged[0][field] is None
        else:
            assert staged[0][field].tolist() == [
                original[field].tolist()[i] for i in (2, 0)
            ]
            assert original[field].shape[0] == 3
    for field in (5, 6, 9, 10):
        assert staged[0][field] is original[field]


@pytest.mark.parametrize("indices", [[], [True], [1, 1], [-1], [3], [0.5], None])
def test_invalid_indices(indices):
    with pytest.raises(ValueError):
        prepare_gdn_rows([capture()], indices, batch_size=3)


@pytest.mark.parametrize("batch", [True, 0, -1, 3.0, None])
def test_invalid_batch(batch):
    with pytest.raises(ValueError):
        prepare_gdn_rows([capture()], [0], batch_size=batch)


@pytest.mark.parametrize("field,bad", [
    (0, None), (1, mx.zeros((3, 2, 3, 1))), (4, mx.zeros((3, 2))),
    (5, mx.zeros((3, 2))), (6, None), (7, mx.zeros((2, 3, 2, 2))),
    (8, mx.zeros((3, 2, 18))), (9, True), (10, float("nan")),
])
def test_invalid_later_capture_does_not_stage_or_mutate(monkeypatch, field, bad):
    first = capture()
    second = list(capture())
    second[field] = bad
    second = tuple(second)
    captures = [first, second]

    def unexpected_take(*args, **kwargs):
        pytest.fail("staging occurred before complete validation")

    monkeypatch.setattr(mx, "take", unexpected_take)
    with pytest.raises(ValueError):
        prepare_gdn_rows(captures, [2, 0], batch_size=3)
    assert captures[0] is first and captures[1] is second


@pytest.mark.parametrize("captures", [None, (), [()], [[None] * 11]])
def test_invalid_capture_container(captures):
    with pytest.raises(ValueError):
        prepare_gdn_rows(captures, [0], batch_size=3)


def test_empty_capture_list():
    assert prepare_gdn_rows([], [0], batch_size=3) == []
