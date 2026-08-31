"""Weight sharding primitives for tensor parallelism, quantization-aware.

Two directions, in Megatron's terms:

* **column parallel** -- split the *output* features.  Each rank produces a
  slice of the output and no communication is needed; the next op must know it
  is working on a slice.  Slicing is along axis 0 of every array a linear owns,
  so it is alignment-free.

* **row parallel** -- split the *input* features.  Each rank consumes a slice of
  the input and produces a *partial* output; the full result is the sum over
  ranks, so a row-parallel linear is always followed by an all-reduce.

Quantized layouts make row-parallel the delicate one.  For affine
``QuantizedLinear`` the arrays are::

    weight  (out, in // pack)        pack = 32 // bits
    scales  (out, in // group_size)
    biases  (out, in // group_size)   <- quantization zero points, NOT the affine bias

so splitting the input dimension requires ``in // tp`` to be divisible by both
``pack`` and ``group_size``.  At GLM-5.3-Flash dims (in=4096, tp=2, bits=4,
group_size=64) that is 2048, divisible by 8 and by 64, so it is exact -- but the
check is enforced rather than assumed, because a silently misaligned slice
produces plausible-looking garbage.

The affine ``bias`` of a row-parallel linear must be added exactly once, so it
is kept on rank 0 and dropped elsewhere.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

COLUMN = "column"
ROW = "row"


def _split(x: mx.array, axis: int, rank: int, size: int) -> mx.array:
    n = x.shape[axis]
    if n % size != 0:
        raise ValueError(f"cannot split axis {axis} of size {n} across {size} ranks")
    k = n // size
    idx = [slice(None)] * x.ndim
    idx[axis] = slice(rank * k, (rank + 1) * k)
    return mx.contiguous(x[tuple(idx)])


def _is_quant(m) -> bool:
    return "scales" in m


def _in_features(m) -> int:
    """Logical input width of a quantized linear, from the scales.

    Not derivable from the packed weight: at 6 bits mlx stores 3 uint32 per 16
    values, so weight.shape[-1] is 192 for in=1024 -- a ratio of 5.333, not an
    integer "pack factor". The scales carry exactly one entry per group, so
    ``scales.shape[-1] * group_size`` is the only reliable source.
    """
    return m["scales"].shape[-1] * m.group_size


def _check_row_split(m, size: int, axis: int):
    if getattr(m, "mode", "affine") != "affine":
        raise NotImplementedError(
            f"tp sharding supports affine quant only, got {m.mode}"
        )
    in_features = _in_features(m)
    packed = m["weight"].shape[axis]
    if in_features % (size * m.group_size) != 0:
        raise ValueError(
            f"row-parallel split of in_features={in_features} across {size} "
            f"ranks is not aligned to group_size={m.group_size}"
        )
    if packed % size != 0:
        raise ValueError(
            f"row-parallel split of the packed axis ({packed} uint32) across "
            f"{size} ranks is not integral at bits={m.bits}"
        )


def shard_out(m, rank: int, size: int):
    """Column parallel: keep this rank's slice of the output features."""
    if size == 1:
        return m
    m.weight = _split(m["weight"], 0, rank, size)
    if _is_quant(m):
        m.scales = _split(m["scales"], 0, rank, size)
        if "biases" in m:
            m.biases = _split(m["biases"], 0, rank, size)
    if "bias" in m:
        m.bias = _split(m["bias"], 0, rank, size)
    return m


def shard_in(m, rank: int, size: int):
    """Row parallel: keep this rank's slice of the input features.

    The caller owes an all-reduce on the output.
    """
    if size == 1:
        return m
    if _is_quant(m):
        _check_row_split(m, size, 1)
        m.weight = _split(m["weight"], 1, rank, size)
        m.scales = _split(m["scales"], 1, rank, size)
        if "biases" in m:
            m.biases = _split(m["biases"], 1, rank, size)
    else:
        m.weight = _split(m["weight"], 1, rank, size)
    if "bias" in m and rank != 0:
        # added once, on rank 0, after the all-reduce
        m.bias = mx.zeros_like(m["bias"])
    return m


def shard_axis0(m, rank: int, size: int):
    """Split a MultiLinear-style (num_heads, out, in) weight by head."""
    if size == 1:
        return m
    m.weight = _split(m["weight"], 0, rank, size)
    if _is_quant(m):
        m.scales = _split(m["scales"], 0, rank, size)
        if "biases" in m:
            m.biases = _split(m["biases"], 0, rank, size)
    return m


def shard_experts_out(m, rank: int, size: int):
    """Column parallel inside a SwitchGLU-style (num_experts, out, in) weight."""
    if size == 1:
        return m
    m.weight = _split(m["weight"], 1, rank, size)
    if _is_quant(m):
        m.scales = _split(m["scales"], 1, rank, size)
        if "biases" in m:
            m.biases = _split(m["biases"], 1, rank, size)
    return m


def shard_experts_in(m, rank: int, size: int):
    """Row parallel inside a SwitchGLU-style (num_experts, out, in) weight."""
    if size == 1:
        return m
    if _is_quant(m):
        _check_row_split(m, size, 2)
    m.weight = _split(m["weight"], 2, rank, size)
    if _is_quant(m):
        m.scales = _split(m["scales"], 2, rank, size)
        if "biases" in m:
            m.biases = _split(m["biases"], 2, rank, size)
    return m


# ------------------------------------------------------------------ selftest


def _clone(m):
    import copy

    return copy.deepcopy(m)


def selftest(size: int = 2, verbose: bool = True) -> dict:
    """Prove each primitive against the unsharded module, at rounding scale."""
    mx.random.seed(0)
    out = {}

    def rel(a, b):
        return float(mx.max(mx.abs(a - b)) / mx.maximum(mx.max(mx.abs(b)), 1e-6))

    for bits in (4, 6, 8):
        gs = 64
        lin = nn.Linear(1024, 512, bias=True)
        q = nn.QuantizedLinear.from_linear(lin, group_size=gs, bits=bits)
        x = mx.random.normal((3, 1024))
        ref = q(x)

        # column parallel: concatenate the per-rank output slices
        cols = [shard_out(_clone(q), r, size)(x) for r in range(size)]
        out[f"column_q{bits}"] = rel(mx.concatenate(cols, axis=-1), ref)

        # row parallel: each rank sees its input slice, outputs are summed
        parts = []
        k = 1024 // size
        for r in range(size):
            parts.append(shard_in(_clone(q), r, size)(x[..., r * k : (r + 1) * k]))
        out[f"row_q{bits}"] = rel(sum(parts), ref)

    # unquantized
    lin = nn.Linear(1024, 512, bias=True)
    x = mx.random.normal((3, 1024))
    ref = lin(x)
    cols = [shard_out(_clone(lin), r, size)(x) for r in range(size)]
    out["column_fp"] = rel(mx.concatenate(cols, axis=-1), ref)
    k = 1024 // size
    parts = [
        shard_in(_clone(lin), r, size)(x[..., r * k : (r + 1) * k]) for r in range(size)
    ]
    out["row_fp"] = rel(sum(parts), ref)

    # alignment guard must fire
    try:
        q = nn.QuantizedLinear.from_linear(nn.Linear(96, 32, bias=False), group_size=64, bits=4)
        shard_in(_clone(q), 0, 2)
        out["alignment_guard"] = "DID NOT FIRE"
    except ValueError:
        out["alignment_guard"] = "ok"

    if verbose:
        for k_, v in out.items():
            print(f"  {k_:16} {v}")
    return out


if __name__ == "__main__":
    selftest()
