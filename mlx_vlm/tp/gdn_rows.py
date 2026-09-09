"""Stage row selection for GLM's eleven-member speculative GDN captures."""

import math

import mlx.core as mx


def prepare_gdn_rows(captures, keep_indices, *, batch_size):
    """Return a new capture list, leaving the original list and tuples untouched.

    Validate every capture before staging arrays. The caller owns installation
    and serialization with cache use; this helper performs no epoch/wire work.
    Empty capture lists are valid (no captured GDN layers). Empty row selections
    must instead use epoch release. Fields 5, 6, 9 and 10 are layer constants,
    even when their leading dimension happens to equal the batch size.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(keep_indices, (list, tuple)) or not keep_indices:
        raise ValueError("keep_indices must be a nonempty list or tuple")
    keep_indices = tuple(keep_indices)
    if any(type(i) is not int or i < 0 or i >= batch_size for i in keep_indices):
        raise ValueError("keep_indices must contain in-range integers (not bool)")
    if len(set(keep_indices)) != len(keep_indices):
        raise ValueError("keep_indices must be unique")
    if type(captures) is not list:
        raise ValueError("captures must be a list")

    def array_shape(value, shape):
        if not isinstance(value, mx.array) or value.shape != shape:
            raise ValueError("GDN array has incompatible schema shape")

    for capture in captures:
        if type(capture) is not tuple or len(capture) != 11:
            raise ValueError("GDN capture must be an eleven-member tuple")
        q, k, v, a, b, a_log, dt_bias, state, conv, kernel, lower = capture
        if not isinstance(q, mx.array) or q.ndim != 4:
            raise ValueError("GDN q must have shape [B, S, H, D]")
        batch, steps, heads, dim = q.shape
        if batch != batch_size or min(steps, heads, dim) < 1:
            raise ValueError("GDN q has incompatible dimensions")
        for value in (k, v, a):
            array_shape(value, q.shape)
        array_shape(b, (batch, steps, heads))
        array_shape(a_log, (heads, 1))
        array_shape(dt_bias, (heads, dim))
        if state is not None:
            array_shape(state, (batch, heads, dim, dim))
        if type(kernel) is not int or kernel < 1:
            raise ValueError("GDN convolution kernel must be a positive integer")
        array_shape(conv, (batch, steps + kernel - 1, 3 * heads * dim))
        if lower is not None and (
            type(lower) not in (int, float) or not math.isfinite(lower)
        ):
            raise ValueError("GDN lower bound must be finite numeric or None")

    indices = mx.array(keep_indices, dtype=mx.int32)
    staged = []
    for capture in captures:
        members = list(capture)
        arrays = []
        for field in (0, 1, 2, 3, 4, 7, 8):
            if members[field] is not None:
                members[field] = mx.take(members[field], indices, axis=0)
                arrays.append(members[field])
        mx.eval(*arrays)
        staged.append(tuple(members))
    return staged
