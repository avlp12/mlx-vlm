import math
import os
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .activations import swiglu

_SEG_ALIGN_ENV = None

# --------------------------------------------------------------------------- #
# MLX_VLM_GLM5_PREFILL_CPU_FRACTION: hand a leading slice of the routed-expert
# rows of each prefill MoE chunk to the CPU, concurrently with the GPU.
#
# WHY.  Prefill is COMPUTE-bound, not bandwidth-bound: at chunk 8192 the box runs
# at 410-443 input tok/s against a 700 tok/s GEMM ceiling (33.5 GFLOP/token over a
# measured 23.5 TFLOP/s 4-bit qmm peak), i.e. ~58 % of the GPU's compute.  The CPU
# (M3 Ultra, 32 cores, AMX) is idle and sustains a measured 4.6-5.3 TFLOP/s on
# bf16 GEMM at the exact shapes this path issues.  Weight traffic in prefill is
# ~2 % of the chunk (weights are read once per 8192-token chunk, not once per
# token as in decode), so the 730 GB/s unified-memory read ceiling that killed CPU
# co-STREAMING for decode (dossier P1) does not bind here.  The routed experts are
# the largest single block: 38.0 % of prefill wall at 32k / chunk 8192 (L7-c).
#
# MECHANISM (option A' -- see docs/CPU_COCOMPUTE_L27_2026-09-06.md for the
# alternatives that were rejected and why).  ``_gather_sort`` has already sorted
# the chunk's rows by expert id, so a CONTIGUOUS PREFIX of rows is exactly a set
# of whole experts.  The split point is chosen on the row cumsum, so the CPU gets
# the requested FRACTION OF ROWS whatever the router skew, while touching only the
# leading ``E_cpu`` experts' weights.  That distinction is the whole design:
#
#   * a within-expert row split (take the first f rows of EVERY expert) would need
#     all 288 experts' weights in bf16 -- 609 GB per chunk of dequantisation.
#     Dead.
#   * a whole-expert split by index (option B) touches the same weights as this
#     one but load-balances on expert COUNT, and the real router is skewed enough
#     (L7A: p10 rows/expert 9.0 against a 57 mean at chunk 2048) that a fixed
#     expert count is not a fixed row count.  This path is option B's memory
#     behaviour with option A's load balance.
#   * a layer split (option C) has nothing of the right size: the three dense
#     layers plus the shared expert are 4.3 % of prefill, against the ~15-25 % the
#     CPU needs to be worth its own latency.
#
# THE DEQUANTISATION RUNS ON THE GPU, NOT THE CPU.  Measured on this box, MLX
# 0.32.1, 32 cores (all numbers CPU-stream, 4096x2048 4-bit g64 affine):
#
#     mx.dequantize      ->  0.53 Gparam/s bf16 out, 1.24 Gparam/s f32 out
#     mx.gather_qmm      ->  0.39 GFLOP/s   (21.8 s for one 512x4096x2048 call)
#     mx.gather_mm       ->  0.06 TFLOP/s, and f32 only
#     mx.matmul bf16     ->  4.62 (M=227) / 5.10 (M=512) / 5.34 (M=2048) TFLOP/s
#     mx.matmul f32      ->  2.00 (M=512) / 4.64 (M=2048) TFLOP/s
#
# So the CPU's ONLY fast primitive here is a plain dense ``matmul`` on an already
# dequantised, natively-laid-out ``(N, K)`` weight -- which is why this path
# dequantises the leading experts on the ACCELERATOR stream (a pure bandwidth op:
# 2.56 B/param read+write, ~1.07 s per 8192-token chunk if the CPU took ALL 288
# experts, so ~0.2 s at the useful fractions) and then issues one CPU ``matmul``
# per expert segment.  Dequantising on the CPU instead would cost ~98,000 s per
# chunk; ``gather_qmm`` on the CPU reproduces the dossier's C14 kill (20.05 s)
# exactly.  Both are permanently closed by the numbers above.
#
# NUMERICS.  The CPU arm is NOT bit-identical to the GPU arm: the weights are
# rounded to bf16 (or f32, see MLX_VLM_GLM5_PREFILL_CPU_DTYPE) by ``mx.dequantize``
# before the GEMM, where ``gather_qmm`` dequantises inside the kernel, and the
# accumulation order differs.  This path therefore CANNOT be defended by a logits
# sha and must pass the campaign's KL gate (KL <= 0.042075 against the unchunked
# reference, docs/PERF_BASELINE_B0_2026-09-05.md rule 2) before any default-on
# proposal.  At fraction 0 the path is not entered at all and the graph is
# byte-identical to the arm without this commit.
_PREFILL_CPU_FRACTION_ENV = None
_PREFILL_CPU_MIN_ROWS_ENV = None
_PREFILL_CPU_DTYPE_ENV = None
_PREFILL_CPU_STREAM = None

PREFILL_CPU_FRACTION_ENV = "MLX_VLM_GLM5_PREFILL_CPU_FRACTION"
PREFILL_CPU_MIN_ROWS_ENV = "MLX_VLM_GLM5_PREFILL_CPU_MIN_ROWS"
PREFILL_CPU_DTYPE_ENV = "MLX_VLM_GLM5_PREFILL_CPU_DTYPE"

# Default row floor.  Decode at B=1 routes 8 rows and never sorts; a 8192-token
# prefill chunk routes 65,536.  4096 keeps the whole lever on the prefill side of
# that gap with three orders of magnitude of margin, so no decode step can pay the
# one host sync ``_cpu_cocompute_plan`` costs.
DEFAULT_PREFILL_CPU_MIN_ROWS = 4096

# Hard cap.  Past 0.5 the CPU is the critical path by construction (it is ~4x
# slower per FLOP than the GPU on the same GEMM), so the knob refuses to express
# a configuration that can only lose.
MAX_PREFILL_CPU_FRACTION = 0.5


def _prefill_cpu_fraction() -> float:
    """Fraction of each sorted prefill MoE chunk's ROWS to run on the CPU.  0 = off.

    Parsed once per process (module reload resets it, as the tests do).  Anything
    unparseable, negative, or NaN reads as 0.0 -- OFF -- rather than raising, so a
    typo in a serving env can never fail a request; values above
    ``MAX_PREFILL_CPU_FRACTION`` clamp down to it.
    """
    global _PREFILL_CPU_FRACTION_ENV
    if _PREFILL_CPU_FRACTION_ENV is None:
        raw = os.environ.get(PREFILL_CPU_FRACTION_ENV, "0").strip()
        try:
            f = float(raw)
        except ValueError:
            f = 0.0
        if not (f == f) or f <= 0.0:  # NaN or off
            f = 0.0
        _PREFILL_CPU_FRACTION_ENV = min(f, MAX_PREFILL_CPU_FRACTION)
    return _PREFILL_CPU_FRACTION_ENV


def set_prefill_cpu_fraction(value) -> float:
    """Set the CPU co-compute fraction IN-PROCESS; returns the effective value.

    The A/B for this lever has to be paired arms inside ONE model load -- a
    process-per-arm comparison puts a ~6 % effect back into the cross-process noise
    [I892], and reloading this module mid-run would orphan the already-constructed
    ``SwitchGLU`` instances from their class.  So the knob is settable, exactly as
    ``MLX_VLM_GLM5_FUSED_KDA_BLOCK`` is.  Same clamping and same defensive parse as
    the env path; ``None`` restores the env value on the next read.
    """
    global _PREFILL_CPU_FRACTION_ENV
    if value is None:
        _PREFILL_CPU_FRACTION_ENV = None
        return _prefill_cpu_fraction()
    try:
        f = float(value)
    except (TypeError, ValueError):
        f = 0.0
    if not (f == f) or f <= 0.0:
        f = 0.0
    _PREFILL_CPU_FRACTION_ENV = min(f, MAX_PREFILL_CPU_FRACTION)
    return _PREFILL_CPU_FRACTION_ENV


# Per-call accounting for the co-compute arm.  Pure Python ints, touched only when
# the path is entered, so the OFF arm pays nothing.  The harness turns these into
# the per-fraction CPU/GPU busy estimates without a second instrumented pass --
# the row counts are exact, so only the TFLOP/s constants are assumptions.
_COCOMPUTE_STATS = {
    "calls": 0,
    "cpu_rows": 0,
    "gpu_rows": 0,
    "cpu_experts": 0,
    "cpu_weight_params": 0,
}


def reset_cocompute_stats() -> None:
    for k in _COCOMPUTE_STATS:
        _COCOMPUTE_STATS[k] = 0


def cocompute_stats() -> Dict[str, Any]:
    """Snapshot of the co-compute accounting since the last reset.

    ``cpu_weight_params`` is the number of weight parameters dequantised on the
    accelerator stream for the CPU arm -- the term that has to be charged BACK to
    the GPU when the split is priced, and the reason a large fraction is not free.
    """
    s = dict(_COCOMPUTE_STATS)
    total = s["cpu_rows"] + s["gpu_rows"]
    s["total_rows"] = total
    s["cpu_row_fraction"] = (s["cpu_rows"] / total) if total else 0.0
    return s


def _prefill_cpu_min_rows() -> int:
    """Row floor below which the CPU co-compute path is skipped (decode guard)."""
    global _PREFILL_CPU_MIN_ROWS_ENV
    if _PREFILL_CPU_MIN_ROWS_ENV is None:
        raw = os.environ.get(PREFILL_CPU_MIN_ROWS_ENV, "").strip()
        try:
            n = int(raw)
        except ValueError:
            n = DEFAULT_PREFILL_CPU_MIN_ROWS
        _PREFILL_CPU_MIN_ROWS_ENV = max(1, n)
    return _PREFILL_CPU_MIN_ROWS_ENV


def _prefill_cpu_dtype():
    """Compute dtype for the CPU arm, or ``None`` to follow the activation dtype.

    ``bfloat16`` (the activation dtype in this build) keeps the dequantised weight
    at 2 B/param, so the accelerator-side dequantisation reads+writes 2.56 B/param.
    ``float32`` doubles that traffic but was measured slightly faster on the CPU at
    large M (4.64 vs 2.76 TFLOP/s at M=2048) and rounds the weight once instead of
    twice; it is the arm to try if the KL gate is tight.
    """
    global _PREFILL_CPU_DTYPE_ENV
    if _PREFILL_CPU_DTYPE_ENV is None:
        raw = os.environ.get(PREFILL_CPU_DTYPE_ENV, "").strip().lower()
        _PREFILL_CPU_DTYPE_ENV = {
            "": None,
            "auto": None,
            "bf16": mx.bfloat16,
            "bfloat16": mx.bfloat16,
            "f32": mx.float32,
            "fp32": mx.float32,
            "float32": mx.float32,
            "f16": mx.float16,
            "fp16": mx.float16,
            "float16": mx.float16,
        }.get(raw, None)
    return _PREFILL_CPU_DTYPE_ENV


def _prefill_cpu_stream():
    """The dedicated CPU stream the co-compute arm runs on.

    A NEW stream rather than ``mx.default_stream(mx.cpu)`` for two reasons: it can
    never contend with whatever else in the process uses the default CPU stream,
    and it stays a genuinely distinct stream when the default device is itself the
    CPU -- which is how the CPU-only tests exercise this plumbing without a GPU.
    """
    global _PREFILL_CPU_STREAM
    if _PREFILL_CPU_STREAM is None:
        _PREFILL_CPU_STREAM = mx.new_stream(mx.cpu)
    return _PREFILL_CPU_STREAM


def _cpu_cocompute_plan(
    sorted_indices,
    num_experts: int,
    fraction: Optional[float] = None,
    min_rows: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Where to cut the expert-sorted rows so the CPU gets ~``fraction`` of them.

    Returns ``None`` when the split is not worth taking (too few rows, fraction
    off, no expert boundary at or below the target, or the cut would hand the CPU
    every expert).  Otherwise a dict with:

      ``rows``    -- number of leading rows for the CPU (an exact expert boundary)
      ``experts`` -- number of leading experts those rows cover
      ``counts``  -- ``numpy`` row count per leading expert, summing to ``rows``

    COSTS ONE HOST SYNC, like ``_segment_align_order`` and for the same reason: the
    cut point depends on the VALUES of ``sorted_indices``.  Measured at ~0.45 ms
    per MoE layer, ~19 ms per 8192-token chunk against a ~18.5 s chunk (0.1 %).
    The cut is taken on the row CUMSUM rather than on the expert index, so a skewed
    router (L7A measured p10 rows/expert 9.0 against a mean of 57) still yields the
    requested share of ROWS -- which is what the load balance is actually about.
    """
    if fraction is None:
        fraction = _prefill_cpu_fraction()
    if fraction <= 0.0:
        return None
    if min_rows is None:
        min_rows = _prefill_cpu_min_rows()

    idx = np.array(sorted_indices, copy=False)  # <- the sync
    total = int(idx.shape[0])
    if total < min_rows:
        return None
    target = int(fraction * total)
    if target < 1:
        return None

    counts = np.bincount(idx, minlength=num_experts)[:num_experts]
    cum = np.cumsum(counts)
    # Largest prefix of experts whose rows still fit under the target.
    n_experts = int(np.searchsorted(cum, target, side="right"))
    if n_experts <= 0 or n_experts >= num_experts:
        return None
    rows = int(cum[n_experts - 1])
    if rows <= 0:
        return None
    return {"rows": rows, "experts": n_experts, "counts": counts[:n_experts]}


def _cpu_segment_mm(x, w, counts, bias=None):
    """``x @ w[e].T`` per expert segment, on whatever stream the caller is in.

    ``x`` is ``(R, K)`` with rows already grouped by expert in ``counts`` order;
    ``w`` is ``(E, N, K)`` -- the NATIVE ``SwitchLinear`` layout, deliberately not
    pre-transposed.  ``w[e].swapaxes(-1, -2)`` measured 4.62-5.34 TFLOP/s on the
    CPU against 3.64 for a materialised ``(K, N)`` copy, so the transpose is a view
    and stays one.

    Zero-row experts are skipped: the router routinely leaves experts empty in a
    chunk, and a 0-row ``matmul`` is a dispatch for nothing.
    """
    outs = []
    start = 0
    for e in range(len(counts)):
        c = int(counts[e])
        if c == 0:
            continue
        seg = mx.matmul(x[start : start + c], w[e].swapaxes(-1, -2))
        if bias is not None:
            seg = seg + bias[e]
        outs.append(seg)
        start += c
    if not outs:
        return None
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)


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

    def dequantized_prefix(self, n_experts: int, dtype=None):
        """bf16/f32 ``(n_experts, output_dims, input_dims)`` weights for experts ``[0, n)``.

        Used by the CPU co-compute arm, and issued on the CALLER's stream -- which
        is the accelerator, deliberately: ``mx.dequantize`` is a pure bandwidth op
        there (2.56 B/param at bf16) but runs at 0.53 Gparam/s on the CPU, which is
        ~4 orders of magnitude too slow to be on the co-compute critical path.
        """
        biases = self.get("biases")
        return mx.dequantize(
            self["weight"][:n_experts],
            self["scales"][:n_experts],
            None if biases is None else biases[:n_experts],
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            dtype=dtype,
        )

    def bias_prefix(self, n_experts: int):
        """Per-expert output bias for experts ``[0, n)``, or ``None`` if unbiased."""
        return self["bias"][:n_experts] if "bias" in self else None

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

    def dequantized_prefix(self, n_experts: int, dtype=None):
        """Weights for experts ``[0, n)``; already dense, so this is a slice + cast."""
        w = self["weight"][:n_experts]
        return w if dtype is None or w.dtype == dtype else w.astype(dtype)

    def bias_prefix(self, n_experts: int):
        return self["bias"][:n_experts] if "bias" in self else None

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

    def _cocompute(self, x, idx, plan):
        """Run the leading ``plan['rows']`` expert-sorted rows on the CPU stream,
        the rest on the accelerator stream, and concatenate.

        ORDER IS THE MECHANISM, so it is spelled out.  (1) The accelerator
        dequantises the CPU arm's three weight prefixes and is told to start
        immediately (``mx.async_eval``), because everything the CPU does depends on
        them.  (2) The CPU graph is built and handed to its own stream, also with
        ``async_eval``, so it is enqueued BEFORE the accelerator's own MoE half is
        built rather than at the join.  (3) Only then is the accelerator's tail
        built.  The two arms are on different streams with no edge between them, so
        MLX runs them concurrently and the only synchronisation is the final
        ``concatenate``.

        The accelerator therefore waits exactly as long as the CPU arm overruns the
        accelerator arm -- which is why the fraction must be chosen at (or just
        under) the balance point rather than as high as possible.  See the
        prediction table in docs/CPU_COCOMPUTE_L27_2026-09-06.md.
        """
        n, n_experts, counts = plan["rows"], plan["experts"], plan["counts"]
        _COCOMPUTE_STATS["calls"] += 1
        _COCOMPUTE_STATS["cpu_rows"] += n
        _COCOMPUTE_STATS["gpu_rows"] += int(x.shape[0]) - n
        _COCOMPUTE_STATS["cpu_experts"] += n_experts
        _COCOMPUTE_STATS["cpu_weight_params"] += n_experts * (
            self.up_proj.input_dims * self.up_proj.output_dims
            + self.gate_proj.input_dims * self.gate_proj.output_dims
            + self.down_proj.input_dims * self.down_proj.output_dims
        )
        main = mx.default_stream(mx.default_device())
        cpu = _prefill_cpu_stream()
        dtype = _prefill_cpu_dtype() or x.dtype

        with mx.stream(main):
            w_up = self.up_proj.dequantized_prefix(n_experts, dtype=dtype)
            w_gate = self.gate_proj.dequantized_prefix(n_experts, dtype=dtype)
            w_down = self.down_proj.dequantized_prefix(n_experts, dtype=dtype)
            b_up = self.up_proj.bias_prefix(n_experts)
            b_gate = self.gate_proj.bias_prefix(n_experts)
            b_down = self.down_proj.bias_prefix(n_experts)
            x_cpu = x[:n]
        mx.async_eval(w_up, w_gate, w_down, x_cpu)

        out_dtype = x.dtype
        with mx.stream(cpu):
            xs = x_cpu.reshape(n, -1)
            if xs.dtype != dtype:
                xs = xs.astype(dtype)
            h_up = _cpu_segment_mm(xs, w_up, counts, b_up)
            h_gate = _cpu_segment_mm(xs, w_gate, counts, b_gate)
            h = self.activation(h_up, h_gate)
            y_cpu = _cpu_segment_mm(h, w_down, counts, b_down)
            y_cpu = mx.expand_dims(y_cpu.astype(out_dtype), -2)
        mx.async_eval(y_cpu)

        with mx.stream(main):
            x_gpu, idx_gpu = x[n:], idx[n:]
            g_up = self.up_proj(x_gpu, idx_gpu, sorted_indices=True)
            g_gate = self.gate_proj(x_gpu, idx_gpu, sorted_indices=True)
            y_gpu = self.down_proj(
                self.activation(g_up, g_gate), idx_gpu, sorted_indices=True
            )
            return mx.concatenate([y_cpu, y_gpu], axis=0)

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
        # DEFAULT OFF.  One float compare when the knob is unset; the co-compute
        # plan (and its host sync) is never built, and the graph below is the one
        # that shipped before this lever existed.
        plan = (
            _cpu_cocompute_plan(idx, self.gate_proj.num_experts)
            if do_sort and not self.training and _prefill_cpu_fraction() > 0.0
            else None
        )
        if plan is not None:
            x = self._cocompute(x, idx, plan)
        else:
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
