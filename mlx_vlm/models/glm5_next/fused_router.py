# Copyright © 2026 Apple Inc.
"""v3 row #2 -- the fused MoE router/top-k for the B=1 decode path.

WHY
---
``docs/drafts/sweep11/V3_P1_ATTRIBUTION_20260907.md`` (ledger I1487) measures the
GLM-5.3-Flash decode router at **1.85 ms/step** across 42 sparse layers against a
**0.136 ms byte floor** -- 13.6x its own bytes.  The P0 census
(``bench/hwdossier/v3_dispatch_census.py``, receipt
``logs/sweep11/V3_P0_CENSUS_20260907/d512_e288.json``) shows why: the router emits
**7 compute primitives per layer**, 294 per step, for 2.36 MB of weight.  With
``n_group == 1`` (the shipped GLM-5.3-Flash config) those 7 are

    1 Matmul                                 the fp32 router GEMV  [1,4096]x[4096,288]
    2 Sigmoid                                mx.sigmoid(gates)
    3 CompiledBroadcastAddNegative           -(scores + e_score_correction_bias)
    4 ArgPartition                           mx.argpartition(-scores, kth=top_k-1)
    5 GatherAxis                             mx.take_along_axis(orig_scores, inds)
    6 Sum                                    scores.sum(-1, keepdims=True)
    7 CompiledBroadcastDivideBroadcastMultiply   scores / denom * routed_scaling_factor

(plus 4 pure-view nodes: Flatten, Transpose, Unflatten, Slice).

This module keeps primitive 1 -- it is a real 2.36 MB GEMV, not launch overhead --
and folds primitives 2..7 into a single ``mx.fast.metal_kernel`` launch, so the
class goes **7 -> 2 dispatches per layer, 294 -> 84 per step**.  At the P1-measured
5.51 us/dispatch that is 210 x 5.51 us = **1.157 ms/step** with *zero* byte change,
which is the calibration experiment for the dispatch model.

BIT-IDENTITY ON METAL -- the argument, in full
----------------------------------------------
The one hazard is the top-k *order*: ``inds`` feeds ``gather_qmm`` and the
denominator is an fp32 sum over the top_k gathered scores **in ``inds`` order**, so
any permutation changes the result in the last ulp.  ``mx.argpartition`` makes no
ordering promise -- but on Metal it does not partition at all:

    mlx/backend/metal/sort.cpp:342-353  ``ArgPartition::eval_gpu``
        "// We direct arg partition to sort for now"  -> gpu_merge_sort(..., argsort=true)

so on the GPU ``mx.argpartition(-s, kth)[..., :k]`` is exactly
``mx.argsort(-s)[..., :k]``.  That sort is *stable*: the comparator is a strict
``a < b`` with NaN forced last (``mlx/backend/metal/kernels/sort.h:40-58``), the
per-thread pass only swaps on strict less (:66-84), and the merge takes from the
B side only when ``op(b, a)`` is strictly true (:142) -- with idxs initialised to
iota (:467), equal keys therefore keep ascending index order.

Hence the Metal eager order is **descending biased score, ties broken by the lower
expert index**, which this kernel reproduces exactly by construction.  Negation is
exact and order-reversing in IEEE-754 (and -0.0 / +0.0 compare equal either way),
so selecting the smallest ``-(s+b)`` is the same total order as the largest
``s+b``.

The remaining arithmetic is transliterated op for op:

  * ``sigmoid``  -- MLX's Metal Sigmoid is NOT ``1/(1+exp(-x))``.  It is
    ``y = 1/(1+exp(|x|)); return x < 0 ? y : 1-y``
    (``mlx/backend/metal/kernels/unary_ops.h:308-314``).  The kernel below calls the
    identical expression with the identical ``metal::exp`` / ``metal::abs``.  Both
    MLX's own libraries and ``mx.fast.metal_kernel`` compile with the *default*
    ``CompileOptions{math_mode = MathMode::Safe}``
    (``mlx/backend/common/metal_kernel.h:14``, ``mlx/backend/metal/device.cpp:40-61``),
    so ``metal::exp`` lowers the same way in both.  **RISK, stated:** this is the
    one op that is argued rather than executed here -- the desk has no GPU minutes,
    so the equality of ``metal::exp`` between MLX's unary/compiled libraries and a
    custom-kernel library is a source-level argument.  The queue arm
    (``bench/ops/queues/gesicht_v2a_router_20260907.sh``) carries a natural-panel
    text-sha arm that falsifies it if wrong.

  * ``denominator`` -- the eager ``scores.sum(-1)`` over a row of ``top_k`` (=8)
    floats takes ``row_reduce_small`` (row_size 8 <= 64,
    ``mlx/backend/metal/reduce.cpp:571``), whose first branch gives **one thread per
    output** looping ``total = op(v, total)`` from ``Op::init`` = 0.0f in ascending
    element order (``kernels/reduction/reduce_row.h:166-181, 225-236``).  The kernel
    below accumulates in exactly that order.

  * ``scores / denom * routed_scaling_factor`` -- divide then multiply, matching
    ``CompiledBroadcastDivideBroadcastMultiply``.

  * NaN: MLX's comparator sorts NaN keys last; the kernel maps a NaN key to
    ``+inf``.  These agree unless a *genuine* ``+inf`` key is present, which needs
    ``e_score_correction_bias`` to contain ``-inf``.  Documented divergence, not
    reachable from a trained checkpoint.

ON CPU the kernel is refused (``mx.fast.metal_kernel`` is GPU-only,
``mlx/backend/common/metal_kernel.cpp:40-50``) and the eager path runs unchanged --
exactly like ``fused_kda`` (fused_kda.py:665) and the mHC kernel
(hyper_connection.py:745).  ``reference_select`` below is a device-independent
transliteration of the kernel used by the identity rail
(``tests/test_glm5_router_fused.py``).

DEFAULT OFF.  ``MLX_VLM_GLM5_ROUTER_FUSED=1`` opts in; unset is byte-identical.
"""

import functools
import os
from typing import Any, Optional, Tuple

import mlx.core as mx

ROUTER_FUSED_ENV = "MLX_VLM_GLM5_ROUTER_FUSED"

# The decode shape this kernel is for.  One threadgroup per row and a serial
# top-k inside it is the right trade at B*S <= 8; at prefill widths the eager
# argsort is a real parallel sort and fusing would lose.  Same reasoning (and the
# same guard style) as _HC_FUSED_MIN_ROWS in deepseek_v4/hyper_connection.py:86,
# with the inequality reversed.
MAX_FUSED_ROWS = 8

# One simdgroup per row: cross-lane reductions inside a single simdgroup make the
# threadgroup barriers essentially free.  Must match ``best_*[32]`` in the source.
LANES = 32

# Threadgroup memory is 2*4*E + E bytes; 4096 experts = 36 KB > the 32 KB limit.
MAX_FUSED_EXPERTS = 3000


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def router_fused_enabled(config: Any = None) -> bool:
    flag = getattr(config, "router_fused", None) if config is not None else None
    if flag is not None:
        return bool(flag)
    return _env_flag(ROUTER_FUSED_ENV)


def fused_router_supported(
    rows: int,
    n_experts: int,
    top_k: int,
    n_group: int,
    norm_topk_prob: bool,
    dtype=mx.float32,
) -> bool:
    """Every condition that must hold before the kernel may replace the eager path."""
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    if dtype != mx.float32:
        return False
    if rows < 1 or rows > MAX_FUSED_ROWS:
        return False
    if n_group != 1:
        # The group-limited branch (put_along_axis over n_group buckets) is a
        # different kernel and is not reachable from the shipped GLM-5.3-Flash
        # config (n_group == topk_group == 1); left on the eager path.
        return False
    if top_k < 1 or top_k > 32:
        return False
    if n_experts < top_k or n_experts > MAX_FUSED_EXPERTS:
        return False
    if not norm_topk_prob:
        # top_k > 1 with norm off is representable (template NORM), but no
        # shipped config uses it, so it stays off the fused rail unproven.
        return False
    return True


# --------------------------------------------------------------------------- #
# the kernel
# --------------------------------------------------------------------------- #

_SOURCE = r"""
    // One threadgroup of LANES threads per row.  E, K, NORM are template ints.
    const uint row  = threadgroup_position_in_grid.y;
    const uint lane = thread_position_in_threadgroup.x;

    threadgroup float tg_score[E];   // sigmoid(logit): the UN-biased score
    threadgroup float tg_key[E];     // -(score + bias): the argsort key
    threadgroup uchar tg_used[E];
    threadgroup float best_val[32];
    threadgroup uint  best_idx[32];
    threadgroup uint  tg_sel[K];

    const float inf = metal::numeric_limits<float>::infinity();

    for (uint e = lane; e < E; e += 32u) {
        // mlx/backend/metal/kernels/unary_ops.h:308-314, verbatim.
        const float x = logits[(ulong)row * (ulong)E + (ulong)e];
        const float y = 1 / (1 + metal::exp(metal::abs(x)));
        const float s = (x < 0) ? y : 1 - y;
        const float k = -(s + bias[e]);
        tg_score[e] = s;
        // LessThan (sort.h:40-58) orders NaN last; +inf is the stand-in.
        tg_key[e]   = metal::isnan(k) ? inf : k;
        tg_used[e]  = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // K rounds of "smallest key, ties to the lowest expert index" == the first K
    // of a stable ascending argsort of tg_key.
    for (uint r = 0; r < (uint)K; ++r) {
        float bv = inf;
        uint  bi = (uint)E;
        for (uint e = lane; e < E; e += 32u) {
            if (tg_used[e]) { continue; }
            const float v = tg_key[e];
            // strict < only, so the first (lowest) index wins a tie
            if (bi == (uint)E || v < bv) { bv = v; bi = e; }
        }
        best_val[lane] = bv;
        best_idx[lane] = bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0) {
            float wv = inf;
            uint  wi = (uint)E;
            for (uint t = 0; t < 32u; ++t) {
                const uint i = best_idx[t];
                if (i >= (uint)E) { continue; }
                const float v = best_val[t];
                if (wi == (uint)E || v < wv || (v == wv && i < wi)) { wv = v; wi = i; }
            }
            tg_sel[r] = wi;
            if (wi < (uint)E) { tg_used[wi] = 1; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        float w[K];
        for (uint r = 0; r < (uint)K; ++r) {
            const uint i = tg_sel[r];
            w[r] = (i < (uint)E) ? tg_score[i] : 0.0f;
        }
        // reduce_row.h:166-181 -- sequential from Op::init = 0.0f, ascending.
        float denom = 0.0f;
        if (NORM) {
            for (uint r = 0; r < (uint)K; ++r) { denom = w[r] + denom; }
        }
        for (uint r = 0; r < (uint)K; ++r) {
            const ulong o = (ulong)row * (ulong)K + (ulong)r;
            inds[o]  = tg_sel[r];
            gates[o] = NORM ? ((w[r] / denom) * rsf) : (w[r] * rsf);
        }
    }
"""


@functools.lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="glm5_router_topk",
        input_names=["logits", "bias", "rsf"],
        output_names=["inds", "gates"],
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def fused_group_expert_select(
    logits: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool = True,
) -> Tuple[mx.array, mx.array]:
    """(inds[..., top_k] uint32, gates[..., top_k] float32) in one launch."""
    lead = logits.shape[:-1]
    n_experts = logits.shape[-1]
    rows = 1
    for d in lead:
        rows *= d
    flat = logits.reshape(rows, n_experts)
    inds, gates = _kernel()(
        inputs=[flat, e_score_correction_bias, float(routed_scaling_factor)],
        template=[
            ("E", int(n_experts)),
            ("K", int(top_k)),
            # matches the eager guard ``if top_k > 1 and norm_topk_prob``
            ("NORM", bool(norm_topk_prob and top_k > 1)),
        ],
        grid=(LANES, rows, 1),
        threadgroup=(LANES, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    shape = tuple(lead) + (top_k,)
    return inds.reshape(shape), gates.reshape(shape)


# --------------------------------------------------------------------------- #
# device-independent transliteration of the kernel (identity rail + docs)
# --------------------------------------------------------------------------- #

def reference_select(
    scores: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool = True,
) -> Tuple[mx.array, mx.array]:
    """The kernel's algorithm, line for line, on any device.

    Takes the **sigmoid output** rather than the logits on purpose: the sigmoid
    itself is the one op whose Metal/CPU equality cannot be executed on this desk
    (see the module docstring), so the rail factors it out and tests everything
    that *is* testable -- key formation, the stable top-k, the un-biased gather,
    the sequential denominator and the divide-then-scale.
    """
    import numpy as np

    s = np.asarray(scores, dtype=np.float32)
    b = np.asarray(e_score_correction_bias, dtype=np.float32)
    lead = s.shape[:-1]
    E = s.shape[-1]
    flat = np.ascontiguousarray(s.reshape(-1, E))
    rows = flat.shape[0]
    inf = np.float32(np.inf)

    # tg_key: -(score + bias), NaN -> +inf   (kernel lines "const float k = ...")
    key = (-(flat + b[None, :])).astype(np.float32)
    key = np.where(np.isnan(key), inf, key).astype(np.float32)

    # K rounds of masked argmin.  np.argmin returns the FIRST minimal position,
    # which is exactly the kernel's "strict <, so the lowest index wins a tie".
    used = np.zeros((rows, E), dtype=bool)
    sel = np.zeros((rows, top_k), dtype=np.int64)
    for r in range(top_k):
        masked = np.where(used, inf, key)
        pick = masked.argmin(axis=-1)
        sel[:, r] = pick
        used[np.arange(rows), pick] = True

    w = np.take_along_axis(flat, sel, axis=-1).astype(np.float32)
    if top_k > 1 and norm_topk_prob:
        # reduce_row.h:166-181 -- sequential from 0.0f, ascending element order.
        denom = np.zeros(rows, dtype=np.float32)
        for j in range(top_k):
            denom = (w[:, j] + denom).astype(np.float32)
        w = (w / denom[:, None]).astype(np.float32)
    gates = (w * np.float32(routed_scaling_factor)).astype(np.float32)
    inds = sel.astype(np.uint32)

    shape = tuple(lead) + (top_k,)
    return mx.array(inds.reshape(shape)), mx.array(gates.reshape(shape))


def metal_semantics_select(
    scores: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool = True,
    sum_order: str = "sequential",
) -> Tuple[mx.array, mx.array]:
    """The EAGER path as it executes **on Metal**, expressed in MLX ops.

    Two substitutions, each from MLX source:

    1. ``mx.argpartition`` -> ``mx.argsort``.  ``ArgPartition::eval_gpu``
       (``mlx/backend/metal/sort.cpp:342-353``) literally comments "We direct arg
       partition to sort for now" and calls ``gpu_merge_sort(..., argsort=true)``.
       On CPU ``mx.argsort`` is a ``std::stable_sort`` with the same NaN-last
       comparator (``mlx/backend/cpu/sort.cpp:196-205``), so this is an exact CPU
       model of the Metal ordering, including the ascending-index tie-break.

    2. ``scores.sum(-1)`` -> a sequential accumulation.  On Metal a row of 8
       floats takes ``row_reduce_small``'s one-thread branch
       (``mlx/backend/metal/reduce.cpp:571`` + ``reduction/reduce_row.h:166-181``),
       i.e. ``total = v + total`` from ``0.0f`` in ascending order.  MLX's **CPU**
       ``Sum`` does not use that order: measured on 2,000 random rows the two
       differ in 959 of them by up to 8.94e-08 absolute in the final gate.  Pass
       ``sum_order="mx"`` to get the literal CPU eager arithmetic instead.
    """
    orig = scores
    biased = scores + e_score_correction_bias
    inds = mx.argsort(-biased, axis=-1)[..., :top_k]
    w = mx.take_along_axis(orig, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        if sum_order == "sequential":
            denom = mx.zeros(w.shape[:-1], dtype=w.dtype)
            for j in range(top_k):
                denom = w[..., j] + denom
            denom = denom[..., None]
        elif sum_order == "mx":
            denom = w.sum(axis=-1, keepdims=True)
        else:
            raise ValueError(f"unknown sum_order {sum_order!r}")
        w = w / denom
    return inds, w * routed_scaling_factor
