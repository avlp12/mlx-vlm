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


_SMALL_M_ENV = None


def _small_m_policy():
    """(enabled, max_rows_per_expert) for the small-M expert-GEMM dispatch policy.

    THE BRANCH WE ARE STEERING.  ``mlx/backend/metal/quantized.cpp:1583`` (v0.32.0; the same
    condition at :1904 in the 0.32.1 dev build) dispatches ``GatherQMM`` as::

        if (M == 1 && B >= 16 && right_sorted_ && B / E >= 4) -> gather_qmm_rhs
        else if (M >= vector_limit)                           -> gather_qmm
        else if (transpose_)                                  -> gather_qmv

    ``B`` is the ROW count (``indices.size``) and ``E`` the expert count, so the switch is a
    pure function of **rows per expert**: ``rpe = B / E``, threshold ``rpe >= 4``.  For
    GLM-5.3-Flash (E=288, top_k=8) that is exactly ``tokens >= 144``.

    THE LOSS BAND.  ``affine_gather_qmm_rhs`` tiles rows at BM=16 and runs a full K-loop per
    distinct expert inside a tile, so its cost is (K-loop passes) = rows/16 + (boundary passes
    ~ one per expert that owns any row).  With E=288 the boundary term is a CONSTANT ~288
    passes; the useful term is rows/16 = tokens/2.  At the moment the branch turns on
    (tokens=144) the useful term is 72 passes against 288 boundary passes -- 80% overhead --
    so ``gather_qmv``, whose cost is linear in rows with no per-expert constant, is far cheaper.
    The probe (docs/logs/glm53_kernels/{gesicht,epsilon}/moe_dispatch_probe.json, mlx 0.32.0,
    E=288 top_k=8 K=4096 N=2048 q4 g64, whole SwitchGLU = 3 gather_qmm) measures the cliff::

        tokens   rpe   sorted us/token   branch
          142    3.94       75.16        gather_qmv
          143    3.97       75.06        gather_qmv
          144    4.00      113.37        gather_qmm_rhs      <- +51% per token, in one token
          160    4.44      108.50        gather_qmm_rhs
          256    7.11       74.97        gather_qmm_rhs      <- back to the 143 level
          512   14.22       48.99        gather_qmm_rhs
         2048   56.89       28.89        gather_qmm_rhs

    Fitting those two arms (gesicht; epsilon is within 3%) gives

        gather_qmv, sorted :  ms = 0.803 + 0.01096 * E_touched + 0.005897 * rows
        gather_qmm_rhs     :  ms = 13.39 + 0.02233 * tokens   ( = 0.0447 ms x (rows/16 + 288) )

    and the rhs fit's 13.39 ms intercept is 288 x 0.0465 ms, i.e. exactly the per-expert
    boundary-pass constant -- the same pass model that reproduces the L2 receipt's measured
    pass counts (rows/16 + 288 = 544 / 1312 / 4384 against the measured 532 / 1296 / 4360 at
    T = 512 / 2048 / 8192).  Two independent receipts, one cost model.

    WHAT THE FORK CAN DO WITHOUT TOUCHING MLX.  Three candidates, only one survives:

      (i)   don't sort -> MLX takes gather_qmv.  REJECTED by the probe: unsorted gather_qmv is
            153-163 us/token at every size (23.4 ms at tokens=144 against 16.3 ms for rhs).
            Sorting is what makes gather_qmv cheap; the sort is not the problem, the branch is.
      (ii)  keep the sort, SPLIT the sorted rows into k contiguous slabs each with fewer than
            4*E rows, so every slab takes gather_qmv on already-sorted rows.  Each slab spans
            ~1/k of the expert range, so the k slabs together touch the same experts once --
            the weight traffic does NOT multiply by k, only the k kernel launches do.  This is
            the shipped mechanism.
      (iii) pad up to a size where rhs wins.  REJECTED analytically: rhs ms is monotonically
            increasing in rows, so padding rows can only add cost.  (The other padding lever,
            MLX_VLM_MOE_SEGMENT_ALIGN, is actively harmful here: at rpe=4 nearly every expert
            has fewer than 16 rows, so aligning to 16 inflates 1152 rows to ~4608.  This policy
            therefore suppresses segment alignment on the slabs it creates.)

    CROSSOVER.  With slab count k = ceil(rows / (4E - 1)) and E_touched ~ E:

        split(T, k) = k*0.803 + 0.01096*(E + k-1) + 0.005897*8T
        rhs(T)      = 0.02233*T + 13.39

    k=2 crosses rhs at T ~ 325 tokens (rpe ~ 9.0) and k=3 at T ~ 292 (rpe ~ 8.1); since k steps
    2 -> 3 exactly at rpe = 8, the win region is exactly the k=2 regime, ``4 <= rpe < 8``.
    A deliberately pessimistic model that re-pays the FULL per-slab expert traffic (i.e. treats
    each slab as if it touched all 288 experts, which is what k copies of the probe's own
    tokens=T/k point would cost) crosses earlier, at rpe ~ 6.7.  The default threshold is the
    pessimistic one -- 6.0 -- so that the shipped band is a win under BOTH models; the
    6.0 - 8.0 stretch is model-dependent and is what the GPU sweep is for.

        rpe   tokens   k   split(loc)   split(pess)   rhs      gain vs rhs
        4.00     144   2     11.51 ms     13.86 ms   16.61 ms  +44% / +20%
        4.44     160   2     12.28        14.78      16.97     +38% / +15%
        5.33     192   2     13.81        16.54      17.68     +28% / +7%
        6.00     216   2     14.94        17.86      18.21     +22% / +2%     <- default cut
        7.11     256   2     16.85        19.81      19.11     +13% / -4%
        8.00     288   3     19.17        24.81      19.83      +3% / -20%    <- hard cap

    NUMERICS.  This changes WHICH GPU KERNEL runs, so it is NOT bit-identical to the default on
    a Metal device: gather_qmv and the steel-tiled gather_qmm_rhs reduce K in different orders.
    It IS exactly output-preserving as an ALGEBRAIC transform -- slabbing partitions rows, and
    every row's expert GEMM is independent of every other row -- which is what the CPU tests
    assert (on CPU there is no branch at all, so split and unsplit run the identical kernel and
    the outputs are bitwise equal).  Note the default already has this discontinuity: MLX itself
    changes kernel between 143 and 144 tokens today.  Promotion past default-off therefore owes
    the speculative-acceptance rail, not just a greedy tok/s number.

    GPU ONLY.  The branch being steered lives in the METAL backend; the CPU backend ignores
    ``sorted_indices`` entirely (asserted by
    tests/test_moe_small_m_policy.py::test_cpu_backend_has_no_sorted_branch), so on CPU the
    slabs buy nothing and cost the extra launches: measured 2-6% slower on the CPU smoke arm of
    bench/l37_moe_small_m.py (E=32 top_k=4, tokens 32/40/56, speedup 0.978 / 0.964 / 0.944
    against a 1.3% control floor).  Deliberately NOT device-gated in code -- the decision stays
    a pure function of shapes, which is what keeps it sync-free and compile-safe -- so do not
    turn it on for a CPU-served box.

    TP.  ``shard_experts_out``/``_in`` (mlx_vlm/tp/shard.py:133,145) split axis 1/2 of the
    (num_experts, out, in) weight, never axis 0, so ``num_experts`` and therefore this decision
    are identical at TP1 and TP2.

    ``MLX_VLM_MOE_SMALL_M_POLICY=auto`` to enable (``1``/``true``/``on`` also accepted),
    ``off``/``0`` to disable.  DEFAULT OFF.  ``MLX_VLM_MOE_SMALL_M_MAX_RPE`` overrides the
    upper edge (default 6.0, hard-capped at 8.0 because k=3 is a loss under both models).
    """
    global _SMALL_M_ENV
    if _SMALL_M_ENV is None:
        v = os.environ.get("MLX_VLM_MOE_SMALL_M_POLICY", "off").strip().lower()
        enabled = v in ("auto", "1", "true", "yes", "on")
        try:
            rpe = float(os.environ.get("MLX_VLM_MOE_SMALL_M_MAX_RPE", "6.0"))
        except ValueError:
            rpe = 6.0
        # Below 4.0 the policy can never fire (MLX is already on gather_qmv); above 8.0 the
        # slab count is 3+, which both cost models call a loss.
        rpe = min(max(rpe, 4.0), 8.0)
        _SMALL_M_ENV = (enabled, rpe)
    return _SMALL_M_ENV


def _small_m_slabs(n_rows, num_experts):
    """Number of contiguous slabs to cut the sorted rows into; 1 means "leave MLX alone".

    Decided from SHAPES ONLY (``indices.size`` and the expert count), never from index VALUES,
    so it costs no host sync and is stable under ``mx.compile`` (which retraces per shape).
    """
    if not num_experts:
        return 1
    enabled, max_rpe = _small_m_policy()
    if not enabled:
        return 1
    rpe = n_rows / num_experts
    # rpe < 4: MLX already takes gather_qmv, nothing to steer.
    # rpe >= max_rpe: the rhs branch has amortised its per-expert boundary constant; leave it.
    if rpe < 4.0 or rpe >= max_rpe:
        return 1
    # Every slab must fall strictly below the branch condition B / E >= 4, i.e. <= 4E-1 rows.
    limit = 4 * num_experts - 1
    k = -(-n_rows // limit)  # ceil
    return k if k > 1 else 1


def _slab_bounds(n_rows, k):
    """``k`` near-equal contiguous [start, stop) row ranges covering ``n_rows``."""
    step = -(-n_rows // k)  # ceil, so every slab is <= step rows and the last one is short
    return [(a, min(a + step, n_rows)) for a in range(0, n_rows, step)]


def _gather_sort(x, indices, num_experts=None, allow_align=True):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    sorted_indices = indices[order]

    # ``allow_align`` is False when the small-M policy is going to slab these rows onto
    # gather_qmv: BM=16 segment padding exists only to help gather_qmm_rhs, and in the small-M
    # band it would inflate the row count several-fold for a kernel that never runs.
    align = _moe_segment_align() if (num_experts and allow_align) else 0
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

    def _experts(self, x, idx, sorted_indices):
        x_up = self.up_proj(x, idx, sorted_indices=sorted_indices)
        x_gate = self.gate_proj(x, idx, sorted_indices=sorted_indices)
        return self.down_proj(
            self.activation(x_up, x_gate), idx, sorted_indices=sorted_indices
        )

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        # THE DISPATCH DECISION (see _small_m_policy for the cost model and the receipts).
        # Shapes only -- no host sync, safe under mx.compile.
        n_slabs = (
            _small_m_slabs(indices.size, self.gate_proj.num_experts) if do_sort else 1
        )
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(
                x,
                indices,
                num_experts=self.gate_proj.num_experts,
                allow_align=(n_slabs == 1),
            )
        if self.training:
            idx = mx.stop_gradient(idx)
        if n_slabs > 1:
            # Rows are sorted, so a contiguous slab is sorted too and spans ~1/k of the expert
            # range; running gate/up/down per slab also halves the peak intermediate transient.
            x = mx.concatenate(
                [
                    self._experts(x[a:b], idx[a:b], True)
                    for a, b in _slab_bounds(idx.size, n_slabs)
                ],
                axis=0,
            )
        else:
            x = self._experts(x, idx, do_sort)

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

    def _experts(self, x, idx, sorted_indices):
        x = self.fc1(x, idx, sorted_indices=sorted_indices)
        x = self.activation(x)
        return self.fc2(x, idx, sorted_indices=sorted_indices)

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        n_slabs = _small_m_slabs(indices.size, self.fc1.num_experts) if do_sort else 1
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(
                x,
                indices,
                num_experts=self.fc1.num_experts,
                allow_align=(n_slabs == 1),
            )
        if self.training:
            idx = mx.stop_gradient(idx)
        if n_slabs > 1:
            x = mx.concatenate(
                [
                    self._experts(x[a:b], idx[a:b], True)
                    for a, b in _slab_bounds(idx.size, n_slabs)
                ],
                axis=0,
            )
        else:
            x = self._experts(x, idx, do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
