import math
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .activations import swiglu

_SEG_ALIGN_ENV = None


def _moe_segment_align() -> int:
    """Row alignment for MoE expert segments before ``gather_qmm``.  0 disables.

    WHY.  With sorted indices, ``GatherQMM::eval_gpu`` takes the ``gather_qmm_rhs`` branch
    (mlx quantized.cpp:1904, needs ``M == 1 && B >= 16 && right_sorted && B/E >= 4``), whose
    kernel ``affine_gather_qmm_rhs`` (kernels/quantized.metal:151) tiles rows at **BM = 16** and
    runs a FULL K-loop per distinct expert inside each tile.  A 16-row tile that straddles an
    expert boundary therefore runs the whole K-loop twice.  Padding each expert's segment up to a
    multiple of 16 removes the straddle, at the cost of the padding rows.

    Measured on an M3 Ultra, 288 experts, top-8, K=4096 N=2048 4-bit g64, no model load
    (receipt logs/sweep6/lane5_VERDICT_L2_L4.md):

        T      rows    passes nat -> pad    natural    padded    kernel speedup
        512    4,096      532 -> 358        7.084 ms   4.769 ms      1.485x
        2048  16,384    1,296 -> 1,157     16.009 ms  14.203 ms      1.127x
        4096  32,768    2,315 -> 2,173     28.001 ms  26.157 ms      1.071x
        8192  65,536    4,360 -> 4,233     52.052 ms  50.447 ms      1.032x

    Time per K-loop pass is constant to 0.6% between the two arms, so the cost model is the
    boundary-pass count and nothing else.  A Zipf(0.5) routing draw gives 1.402x at T=512, so
    the win is not an artifact of a uniform draw.

    Bit-exact: the padding rows REPEAT each segment's last row, which keeps ``indices[order]``
    correct by construction and can never introduce a NaN, and the real rows of the padded
    output are bit-identical to the unpadded output.

    E2E, epsilon, real text, ABAB, n=3 + discarded warm-up, 16,384-token prefill
    (receipt logs/sweep6/SWEEP6_L2_e2e_E1.json):

        chunk  512:  1.1023 / 1.0981 / 1.1005   median 1.1005, worst pair 1.0981
        chunk 2048:  1.0394 / 1.0462 / 1.0494   median 1.0462, worst pair 1.0394

    On the REAL router: 42 of 42 sparse layers pad, rows +13.55% (the uniform proxy used in the
    microbench gave +13.2%, so the proxy was sound).  Output is **bitwise identical** -- max
    |dlogit| 0.0 and 32 greedy tokens identical -- because the transform is exactly
    output-preserving.

    THE COST TO WATCH IS PEAK MEMORY, not time: 190.5 -> 196.1 GB at the 512 chunk and
    198.1 -> 210.0 GB at 2048 (+11.9 GB).  The padding rows are real transients.  Peak at B=8/16
    with this on is NOT measured, and the fleet gate's SHARD_GB["single"] = 183 is already stale
    against the 198 GB the OFF arm reaches.

    DEFAULT OFF.  Recommended ON for B=1 prefill once the peak-memory gate is re-fitted; keep it
    OFF under batched serving until B=8/16 peaks exist.  ``MLX_VLM_MOE_SEGMENT_ALIGN=16`` to
    enable (``1``, ``true`` and ``on`` are accepted and mean 16).
    """
    global _SEG_ALIGN_ENV
    if _SEG_ALIGN_ENV is None:
        v = os.environ.get("MLX_VLM_MOE_SEGMENT_ALIGN", "0").strip().lower()
        if v in ("1", "true", "yes", "on"):
            n = 16
        else:
            try:
                n = int(v)
            except ValueError:
                n = 0
        _SEG_ALIGN_ENV = max(0, n)
    return _SEG_ALIGN_ENV


def _segment_align_order(sorted_indices, num_experts, align):
    """Padded gather order and the positions of the real rows within it.

    Returns ``(order_pad, real_pos)``: ``order_pad`` maps a padded row to the sorted row it
    takes (padding entries repeat their segment's last row), and ``real_pos`` maps a sorted row
    to its position in the padded layout.

    COSTS ONE HOST SYNC.  ``R_pad`` depends on the VALUES of ``sorted_indices``, so the padded
    gather's output shape cannot be derived on the GPU.  The static worst-case bound
    ``R + (align-1)*E`` needs no sync but pads far harder: measured 1.015x at T=2048 against
    1.130x for this path, i.e. it throws the win away.  The sync is 0.42-0.49 ms and there is
    one per MoE layer -- about 19 ms per 2048-token chunk against a ~228 ms saving.
    """
    idx = np.array(sorted_indices, copy=False)          # <- the sync
    R = idx.shape[0]
    counts = np.bincount(idx, minlength=num_experts)[:num_experts]
    padded = ((counts + align - 1) // align) * align
    seg_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pad_start = np.concatenate([[0], np.cumsum(padded)[:-1]])
    R_pad = int(padded.sum())
    e_of_pos = np.repeat(np.arange(num_experts), padded)
    off = np.arange(R_pad) - pad_start[e_of_pos]
    order_pad = seg_start[e_of_pos] + np.minimum(off, np.maximum(counts[e_of_pos] - 1, 0))
    real_pos = np.repeat(pad_start, counts) + (np.arange(R) - np.repeat(seg_start, counts))
    return mx.array(order_pad.astype(np.uint32)), mx.array(real_pos.astype(np.uint32))


def _gather_sort(x, indices, num_experts=None):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    sorted_indices = indices[order]

    align = _moe_segment_align() if num_experts else 0
    # Only worth it where the model actually reaches affine_gather_qmm_rhs: that branch needs
    # B / E >= 4 (mlx quantized.cpp:1904). Below it the kernel is a different one and padding
    # would add rows for nothing.
    if align > 1 and indices.size >= 4 * num_experts:
        order_pad, real_pos = _segment_align_order(sorted_indices, num_experts, align)
        return (
            x.flatten(0, -3)[order[order_pad] // M],
            sorted_indices[order_pad],
            real_pos[inv_order],
        )
    return x.flatten(0, -3)[order // M], sorted_indices, inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(
                x, indices, num_experts=self.gate_proj.num_experts
            )
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(
                x, indices, num_experts=self.fc1.num_experts
            )
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
