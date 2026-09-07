# Copyright (c) 2026 Apple Inc.

import os
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

# L13 (2026-09-05): whole-block mx.compile fusion for the HyperConnection
# norm/mix/inject path.  Default OFF -- unverified on GPU, this module has no
# GPU here to measure on (see mlx_vlm/tests/test_glm5_next_hc_compile.py for
# the CPU parity/shape coverage this toggle gets).  Same `_env_flag`-style
# read-once-cache-in-a-global pattern as glm5_next/language.py; duplicated
# rather than imported because language.py imports THIS module (importing back
# would cycle).
_HC_COMPILE_ENV = None


def _hc_compile_enabled() -> bool:
    global _HC_COMPILE_ENV
    if _HC_COMPILE_ENV is None:
        _HC_COMPILE_ENV = os.environ.get(
            "MLX_VLM_GLM5_HC_COMPILE", "0"
        ).lower() in ("1", "true", "yes", "on")
    return _HC_COMPILE_ENV


# ---------------------------------------------------------------- L34 (fused)
# MLX_VLM_GLM5_HC_FUSED: hand-written Metal kernels for the two ends of the
# hyper-connection cycle that are still many small element-wise dispatches --
# the pre-norm (fp32 cast + rms_norm) in FRONT of the mixing matmul, and the
# residual inject (`hc_expand`) BEHIND the block.  The middle (the fn
# projection, and the sinkhorn+collapse) is deliberately untouched: the
# projection is real parallel work (I906 fused it and measured 22x SLOWER at
# B=1) and the sinkhorn+collapse already IS one kernel above.
#
# This is a PREFILL lever, not a decode lever.  At B*L=1 row the three
# dispatches this replaces cost ~6 us each and the bytes are noise; at
# B*L=8192 rows each dispatch is a full round trip of a [B,S,hc_mult,D] fp32
# tensor and the bytes ARE the cost (same argument, and the same regime, as
# Glm5NextLinearAttention._fused_kda_prefill_step).  Hence the row floor:
# below it we stay on the compiled/eager path that decode was tuned on.
#
# DEFAULT ON since I1450 (2026-09-07).  Both flags were opt-in when the kernels
# landed at 5cd9d8b5; the registered KL gate has since passed, so serving takes
# them by default and the flags became opt-OUT:
# ``MLX_VLM_GLM5_HC_FUSED=0`` (or ``_HC_PRENORM_FUSED=0`` for just the pre-norm)
# restores the 5cd9d8b5 behaviour, i.e. the compiled/eager path.
#
# The expand kernel reassociates nothing but does let the Metal compiler
# contract mul+add into fma, so it is NOT bit-identical to `mx.matmul` +
# `mx.compile` -- see ``test_expand_kernel_arithmetic_matches_eager_fp32``,
# which pins the drift at a few fp32 ulp.  Promotion therefore rests on the
# rule-13 rail (natural-panel text sha + speculative acceptance parity) and the
# registered KL gate, not on a bit-identical logit fingerprint.
_HC_FUSED_ENV = None
_HC_PRENORM_ENV = None

# An explicit "0"/"false"/"no"/"off" (any case) turns a default-ON lever off.
# Everything else -- INCLUDING the empty string -- is ON, so a launcher that
# emits ``NAME=`` for a variable it did not set (the shape of the TP passthrough
# bug fixed in server/tp_mode.py::launch_worker) lands on the DEFAULT rather
# than silently disabling the lever on one rank only.
#
# Semantics copied verbatim from ``mlx_vlm/generate/common.py::_env_default_on``
# (glm5-serve-unified @ 6e92b5bf) rather than imported: that module does
# ``from ..models import cache`` at import time, so a models -> generate import
# here would cycle.  Same reason ``_hc_compile_enabled`` above duplicates
# language.py's ``_env_flag``.
_ENV_OFF = ("0", "false", "no", "off")


def _env_default_on(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return True
    return raw.strip().lower() not in _ENV_OFF

# Rows (B*L) below which the fused path is not taken.  512 is chosen as
# "clearly prefill": one threadgroup per row * D/4 threads keeps every core
# busy, which is exactly the property B=1 decode lacks.  UNCHANGED by the I1450
# default-ON promotion: decode (B*L = 1, and every speculative width up to 511)
# still takes the eager/compiled path it was tuned on, so the promotion moves
# prefill only.  Pinned by ``test_expand_row_floor_rejects_decode_shapes`` and
# ``test_row_floor_holds_with_flags_defaulted_on``.
_HC_FUSED_MIN_ROWS = int(os.environ.get("MLX_VLM_GLM5_HC_FUSED_MIN_ROWS", "512"))


def _hc_fused_enabled() -> bool:
    """Master gate, DEFAULT ON.  ``MLX_VLM_GLM5_HC_FUSED=0`` restores 5cd9d8b5."""
    global _HC_FUSED_ENV
    if _HC_FUSED_ENV is None:
        _HC_FUSED_ENV = _env_default_on("MLX_VLM_GLM5_HC_FUSED")
    return _HC_FUSED_ENV


def _hc_prenorm_fused_enabled() -> bool:
    """Second gate for the pre-norm kernel, DEFAULT ON.

    The expand kernel only has to match a 4x4-by-4xD matmul; this one has to
    match ``mx.fast.rms_norm``'s own reduction tree over a 16384-long row, and
    that tree is an implementation detail of the installed MLX.  It keeps its
    own flag so a fingerprint failure can be attributed to one kernel rather
    than two: ``MLX_VLM_GLM5_HC_PRENORM_FUSED=0`` drops just the pre-norm and
    leaves the expand kernel on.  Still ANDed with MLX_VLM_GLM5_HC_FUSED, so
    turning the master flag off turns this off too.
    """
    global _HC_PRENORM_ENV
    if _HC_PRENORM_ENV is None:
        _HC_PRENORM_ENV = _env_default_on("MLX_VLM_GLM5_HC_PRENORM_FUSED")
    return _HC_PRENORM_ENV and _hc_fused_enabled()


def _reset_flag_cache() -> None:
    """Test hook: drop the read-once env caches so a monkeypatched environment
    is re-read.  Not used by serving -- the caches exist precisely so serving
    reads ``os.environ`` once per process."""
    global _HC_FUSED_ENV, _HC_PRENORM_ENV
    _HC_FUSED_ENV = None
    _HC_PRENORM_ENV = None


def _metal_ok() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


# ------------------------------------------------------------------ expand
_HC_EXPAND_SOURCE = """
    uint d4  = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;

    constexpr uint D4 = (uint)D / 4;
    if (d4 >= D4) {
        return;
    }

    using T4 = vec<T, 4>;

    const device float* p = (const device float*)post + row * HC;
    const device float* c = (const device float*)comb + row * HC * HC;

    const device T* xrow = (const device T*)x_in     + row * (uint)D;
    const device T* rrow = (const device T*)residual + row * (uint)(HC * D);
    device T*       orow = (device T*)out            + row * (uint)(HC * D);

    // x is [B, L, D] and broadcasts over the hyper axis: one load, HC uses.
    float4 xv = float4(*(const device T4*)(xrow + d4 * 4));

    // residual is [B, L, HC, D]: HC loads, HC uses each.  Held in registers so
    // the HC*HC MACs below touch device memory exactly once per element --
    // that is the whole point of the kernel.
    float4 rv[HC];
    for (int j = 0; j < HC; ++j) {
        rv[j] = float4(*(const device T4*)(rrow + j * (uint)D + d4 * 4));
    }

    // out[h] = post[h] * x + sum_j comb[j][h] * residual[j]
    //
    // Association is the eager one: the j-sum accumulates in ASCENDING j (the
    // K axis of `mx.matmul(comb.swapaxes(-1,-2), residual)`), and the post*x
    // term is added LAST, because the eager code is `y = post*x` then
    // `y = y + matmul(...)`.  Written with separate `*` and `+` rather than
    // fma() for that reason -- though Metal's default fp-contract may still
    // fuse them, which is why this is fingerprint-gated and not asserted
    // bit-identical.
    for (int h = 0; h < HC; ++h) {
        float4 acc = c[h] * rv[0];
        for (int j = 1; j < HC; ++j) {
            acc = acc + c[j * HC + h] * rv[j];
        }
        float4 y = p[h] * xv + acc;
        *(device T4*)(orow + h * (uint)D + d4 * 4) = T4(y);
    }
"""


def _make_hc_expand_kernel():
    if not _metal_ok():
        return None
    return mx.fast.metal_kernel(
        name="hc_expand_fused",
        input_names=["x_in", "residual", "post", "comb"],
        output_names=["out"],
        source=_HC_EXPAND_SOURCE,
        ensure_row_contiguous=True,
    )


_hc_expand_kernel = _make_hc_expand_kernel()


def hc_expand_fused_eligible(x, residual, post, comb) -> bool:
    """Per-call preconditions.  Deliberately total: any surprise falls back."""
    if _hc_expand_kernel is None:
        return False
    if x.ndim != 3 or residual.ndim != 4:
        return False
    B, L, D = x.shape
    if residual.shape != (B, L, comb.shape[-1], D):
        return False
    HC = comb.shape[-1]
    if D % 4 != 0 or HC < 1 or HC > 8:
        return False
    if B * L < _HC_FUSED_MIN_ROWS:
        return False
    if residual.dtype != x.dtype:
        return False
    if post.dtype != mx.float32 or comb.dtype != mx.float32:
        return False
    if post.shape != (B, L, HC) or comb.shape != (B, L, HC, HC):
        return False
    return True


def _hc_expand_fused(x, residual, post, comb):
    B, L, D = x.shape
    HC = comb.shape[-1]
    d4 = D // 4
    tg = min(256, d4)
    grid_x = ((d4 + tg - 1) // tg) * tg
    (out,) = _hc_expand_kernel(
        inputs=[x, residual, post, comb],
        template=[("T", x.dtype), ("HC", HC), ("D", D)],
        grid=(grid_x, B * L, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, L, HC, D)],
        output_dtypes=[x.dtype],
    )
    return out


# ------------------------------------------------------------------ pre-norm
_HC_PRENORM_SOURCE = """
    uint lid  = thread_position_in_threadgroup.x;
    uint row  = threadgroup_position_in_grid.x;
    uint lane = lid % 32;
    uint sg   = lid / 32;

    constexpr uint N = (uint)(HC * D);          // flattened hyper row
    constexpr uint NT = (uint)NTHREADS;
    constexpr uint SG = NT / 32;

    const device T* xrow = (const device T*)x_in + row * N;
    device float*   zrow = (device float*)z_out  + row * N;

    threadgroup float part[SG];

    // Sum of squares.  Per-thread order: N_READS=4 CONTIGUOUS elements per
    // thread per stride step, stride NT*4 -- the same walk mx.fast.rms_norm's
    // single-row kernel uses -- then simd_sum, then a scalar fold over the
    // simdgroup partials.  fp32 accumulator, exactly like the eager path,
    // whose input is already an exact bf16->fp32 widening.
    float acc = 0.0f;
    for (uint r = 0; r < N; r += NT * 4) {
        uint o = r + lid * 4;
        float x0 = (float)xrow[o + 0];
        float x1 = (float)xrow[o + 1];
        float x2 = (float)xrow[o + 2];
        float x3 = (float)xrow[o + 3];
        acc += x0 * x0;
        acc += x1 * x1;
        acc += x2 * x2;
        acc += x3 * x3;
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        part[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float inv_shared;
    if (lid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < SG; ++i) {
            total += part[i];
        }
        inv_shared = metal::precise::rsqrt(total / (float)N + EPS_INT * 1e-9f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = inv_shared;

    for (uint r = 0; r < N; r += NT * 4) {
        uint o = r + lid * 4;
        zrow[o + 0] = (float)xrow[o + 0] * inv;
        zrow[o + 1] = (float)xrow[o + 1] * inv;
        zrow[o + 2] = (float)xrow[o + 2] * inv;
        zrow[o + 3] = (float)xrow[o + 3] * inv;
    }
"""


def _make_hc_prenorm_kernel():
    if not _metal_ok():
        return None
    return mx.fast.metal_kernel(
        name="hc_prenorm_fused",
        input_names=["x_in"],
        output_names=["z_out"],
        source=_HC_PRENORM_SOURCE,
        ensure_row_contiguous=True,
    )


_hc_prenorm_kernel = _make_hc_prenorm_kernel()

_HC_PRENORM_NTHREADS = 256


def hc_prenorm_fused_eligible(x) -> bool:
    if _hc_prenorm_kernel is None:
        return False
    if x.ndim != 4:
        return False
    B, L, H, D = x.shape
    n = H * D
    if n % (_HC_PRENORM_NTHREADS * 4) != 0:
        return False
    if B * L < _HC_FUSED_MIN_ROWS:
        return False
    return True


def _hc_prenorm_fused(x, norm_eps):
    B, L, H, D = x.shape
    (z,) = _hc_prenorm_kernel(
        inputs=[x],
        template=[
            ("T", x.dtype),
            ("HC", H),
            ("D", D),
            ("NTHREADS", _HC_PRENORM_NTHREADS),
            ("EPS_INT", round(norm_eps / 1e-9)),
        ],
        grid=(B * L * _HC_PRENORM_NTHREADS, 1, 1),
        threadgroup=(_HC_PRENORM_NTHREADS, 1, 1),
        output_shapes=[(B, L, H * D)],
        output_dtypes=[mx.float32],
    )
    return z


def _make_hc_sinkhorn_collapse_kernel():
    """Fused sinkhorn + collapse: eliminates one dispatch per HC cycle.

    1. BRANCHLESS SINKHORN: all 32 lanes in simd group 0 execute identical
       instructions. Lanes >= HC use multiplicative mask (active=0) instead
       of divergent branches - eliminates SIMD serialization.
    2. PARALLEL SINKHORN: lanes 0-3 each own one comb row. Column norm
       via simd_sum() - free SIMD shuffle.
    3. NATIVE bfloat4 LOADS: single 64-bit load yields 4 bfloat16 values;
       cast to float4 is a free hardware conversion.
    4. FMA CHAINS: collapse uses fused multiply-add for 3 of 4 terms.
    """
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid  = thread_position_in_threadgroup.x;
        uint row  = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg   = tid / 32;

        constexpr int MIX      = (2 + HC) * HC;
        constexpr int BASE_OFF = 2 * HC;
        constexpr float EPS = EPS_INT * 1e-9;

        const device float* mix      = (const device float*)mixes + row * MIX;
        device float*       post_out = (device float*)post + row * HC;
        device float*       comb_out = (device float*)comb + row * HC * HC;

        threadgroup float pre_shared[HC];

        // ================================================================
        // PHASE 1: Branchless sinkhorn on simd group 0
        //   All 32 lanes execute identical instructions. Lanes >= HC
        //   compute on clamped indices but multiply by active=0, so they
        //   contribute zero to simd_sum. No divergent branches in the loop.
        // ================================================================
        if (sg == 0) {
            const float pre_scale  = scale[0];
            const float post_scale = scale[1];
            const float comb_scale = scale[2];

            const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
            const uint  llane  = metal::min(lane, (uint)(HC - 1));

            // Pre/post sigmoids: all lanes compute, only active lanes write
            float pre_z  = mix[llane]      * pre_scale  + base[llane];
            float post_z = mix[HC + llane] * post_scale + base[HC + llane];
            float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
            float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

            if (lane < (uint)HC) {
                pre_shared[lane] = pre_v;
                post_out[lane]   = post_v;
            }

            // Comb softmax: load + mask. Inactive lanes load row 0 (safe)
            // but multiply by active=0 so they hold zeros.
            float4 v = (*(const device float4*)(mix  + BASE_OFF + llane * HC)
                            * comb_scale
                      + *(const device float4*)(base + BASE_OFF + llane * HC))
                     * active;

            float row_max = metal::max(metal::max(v.x, v.y),
                                       metal::max(v.z, v.w));
            float4 e = metal::fast::exp(v - row_max) * active;
            float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS))
                     + EPS * active;

            // Initial column normalization
            float4 col_inv = 1.0f / (float4(
                simd_sum(r.x), simd_sum(r.y),
                simd_sum(r.z), simd_sum(r.w)
            ) + EPS);
            r *= col_inv;

            // Sinkhorn iterations: zero branches in the loop body
            for (int iter = 1; iter < ITERS; ++iter) {
                // Row norm + re-clamp inactive lanes
                r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;

                // Col norm via simd_sum
                col_inv = 1.0f / (float4(
                    simd_sum(r.x), simd_sum(r.y),
                    simd_sum(r.z), simd_sum(r.w)
                ) + EPS);
                r *= col_inv;
            }

            if (lane < (uint)HC) {
                *(device float4*)(comb_out + lane * HC) = r;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // PHASE 2: Collapse - all 256 threads, vectorized
        // ================================================================
        const float p0 = pre_shared[0];
        const float p1 = pre_shared[1];
        const float p2 = pre_shared[2];
        const float p3 = pre_shared[3];

        const device T* x_row  = (const device T*)x_in
                                         + row * (HC * D);
        device U*       out_row = (device U*)collapsed
                                         + row * D;

        using T4 = vec<T, 4>;
        using U4 = vec<U, 4>;
        const device T4* x_row0 = (const device T4*)(x_row + 0*D);
        const device T4* x_row1 = (const device T4*)(x_row + 1*D);
        const device T4* x_row2 = (const device T4*)(x_row + 2*D);
        const device T4* x_row3 = (const device T4*)(x_row + 3*D);
        device U4*       out4   = (device U4*)out_row;

        constexpr uint D4 = (uint)D / 4;

        // ================================================================
        // PHASE 3 (FOLD_NORM only): the RMSNorm that follows the collapse.
        //
        // Every caller does norm(collapse(x)) -- input_layernorm on the
        // attention half, post_attention_layernorm on the FFN half -- so the
        // norm is a second dispatch reading back what this kernel just wrote.
        // The collapse already holds those values in registers, and 90 of these
        // cycles run per decode step in a dependent chain where a launch costs
        // far more than this arithmetic. Fold it.
        //
        // The collapsed row is kept in registers (ACC float4s per thread), the
        // sum of squares is reduced across the threadgroup, then the same
        // registers are rescaled and written. D=4096 with 256 threads gives
        // ACC=4, i.e. 16 floats per thread.
        // ================================================================
        // NOTE: these are C++ TEMPLATE PARAMETERS, not preprocessor macros.
        // `#if FOLD_NORM` compiles to `#if 0` -- the preprocessor cannot see a
        // template argument -- and silently takes the false branch. That cost a
        // debugging round here: the folded output came back bit-identical to the
        // UNNORMED collapse. The same defect is latent in the `#if (D % 4)`
        // tail this replaces, which has always been dead code and was only
        // correct because every D we use is a multiple of 4. Use ordinary `if`
        // on a compile-time-constant condition; the compiler folds it.
        threadgroup float ssq_red[8];        // 256 threads / 32 = 8 simdgroups
        threadgroup float rms_scale_shared;

        if (FOLD_NORM) {
            constexpr uint ACC = (D4 + 255) / 256;
            float4 acc[ACC];
            float ssq = 0.0f;
            uint n = 0;
            for (uint d4 = tid; d4 < D4; d4 += 256) {
                float4 x0 = float4(x_row0[d4]);
                float4 x1 = float4(x_row1[d4]);
                float4 x2 = float4(x_row2[d4]);
                float4 x3 = float4(x_row3[d4]);

                float4 result = fma(float4(p0), x0,
                                fma(float4(p1), x1,
                                fma(float4(p2), x2, float4(p3) * x3)));
                acc[n++] = result;
                ssq += dot(result, result);
            }

            float sg_sum = simd_sum(ssq);
            if (lane == 0) {
                ssq_red[sg] = sg_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                float total = 0.0f;
                for (uint i = 0; i < 8; ++i) {
                    total += ssq_red[i];
                }
                rms_scale_shared =
                    metal::rsqrt(total / (float)D + NORM_EPS_INT * 1e-9f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float rms_scale = rms_scale_shared;

            // Index norm_w as scalars rather than casting to a vector
            // pointer: MLX places it in the `constant` address space, and a
            // C-style cast to `const device T4*` is a cross-address-space cast
            // the Metal compiler rejects. Which space MLX picks is its own
            // heuristic, so not depending on it is the robust choice; norm_w is
            // 8 KB and stays in cache either way.
            n = 0;
            for (uint d4 = tid; d4 < D4; d4 += 256) {
                const uint b = d4 * 4;
                float4 w = float4((float)norm_w[b + 0], (float)norm_w[b + 1],
                                  (float)norm_w[b + 2], (float)norm_w[b + 3]);
                out4[d4] = U4(acc[n++] * rms_scale * w);
            }
        } else {
            for (uint d4 = tid; d4 < D4; d4 += 256) {
                float4 x0 = float4(x_row0[d4]);
                float4 x1 = float4(x_row1[d4]);
                float4 x2 = float4(x_row2[d4]);
                float4 x3 = float4(x_row3[d4]);

                float4 result = fma(float4(p0), x0,
                                fma(float4(p1), x1,
                                fma(float4(p2), x2, float4(p3) * x3)));

                out4[d4] = U4(result);
            }

            // Scalar tail for D not divisible by 4. A runtime `if`, for the
            // reason in the note above.
            if (D % 4 != 0) {
                for (uint d = D4 * 4 + tid; d < (uint)D; d += 256) {
                    float val = p0*(float)x_row[0*D+d] + p1*(float)x_row[1*D+d]
                              + p2*(float)x_row[2*D+d] + p3*(float)x_row[3*D+d];
                    out_row[d] = (U)val;
                }
            }
        }
    """

    return mx.fast.metal_kernel(
        name="hc_sinkhorn_collapse",
        input_names=["x_in", "mixes", "scale", "base", "norm_w"],
        output_names=["collapsed", "post", "comb"],
        source=source,
        ensure_row_contiguous=True,
    )


_hc_sinkhorn_collapse_kernel = _make_hc_sinkhorn_collapse_kernel()


def hc_fold_norm_supported(x, norm_w, D) -> bool:
    """Can this call fold the following RMSNorm into the collapse?

    The fold keeps the collapsed row in registers (ACC = ceil(D/4/256) float4s
    per thread), so a large D would spill and lose more than the launch is
    worth. It also takes the vectorised path only, so D must be a multiple of 4.
    """
    if norm_w is None:
        return False
    if D % 4 != 0 or D > 8192:
        return False
    return norm_w.dtype == x.dtype and norm_w.shape == (D,)


def _hc_kernel(x, y, mixes, scale, base, hc_mult, sinkhorn_iters, eps,
               norm_w=None, norm_eps=1e-5):
    """Fused sinkhorn + collapse, optionally with the following RMSNorm folded in.

    With ``norm_w`` the third output is norm(collapse(x)) rather than
    collapse(x), which removes one dispatch from a chain that runs 90 times per
    decode step. Without it the behaviour is exactly as before.
    """
    B, L, H, D = x.shape
    fold = hc_fold_norm_supported(x, norm_w, D)
    # A dummy that is never read when FOLD_NORM is 0: metal_kernel wants every
    # declared input present, and a 4-element array costs nothing.
    w = norm_w if fold else mx.zeros((4,), dtype=x.dtype)

    return _hc_sinkhorn_collapse_kernel(
        inputs=[x, mixes, scale, base, w],
        template=[
            ("T", x.dtype),
            ("U", x.dtype),
            ("HC", hc_mult),
            ("ITERS", sinkhorn_iters),
            ("D", D),
            ("EPS_INT", round(eps / 1e-9)),
            ("FOLD_NORM", 1 if fold else 0),
            ("NORM_EPS_INT", round(norm_eps / 1e-9)),
        ],
        grid=(B * L * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, D), (B, L, hc_mult), (B, L, hc_mult, hc_mult)],
        output_dtypes=[x.dtype, mx.float32, mx.float32],
    )


@mx.compile
def _hc_split_sinkhorn_ops(
    mixes: mx.array,
    scale: mx.array,
    base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    mixes = mixes.astype(mx.float32)
    scale = scale.astype(mx.float32)
    base = base.astype(mx.float32)
    pre_scale, post_scale, comb_scale = scale[0], scale[1], scale[2]

    pre = mx.sigmoid(mixes[..., :hc_mult] * pre_scale + base[:hc_mult]) + eps
    post = 2 * mx.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * post_scale + base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].reshape(
        *mixes.shape[:-1], hc_mult, hc_mult
    ) * comb_scale + base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _hc_ops(x, y, mixes, scale, base, hc_mult, sinkhorn_iters, eps):
    pre, post, comb = _hc_split_sinkhorn_ops(
        mixes, scale, base, hc_mult, sinkhorn_iters, eps
    )
    return (pre[..., None] * y).sum(axis=2).astype(x.dtype), post, comb


def _hc_preamble_impl(x: mx.array, fn_t: mx.array, norm_eps: float):
    """Pre-norm + mixing matmul: the part of ``HyperConnection.__call__`` that
    runs unconditionally, before the GPU-kernel/ops branch.  Pure elementwise
    cast, a flatten, an RMSNorm and one matmul against the (frozen) mixing
    weight -- no reshape whose target size depends on a *value* read out of a
    traced shape, so this is a `shapeless=True` candidate: recompiling on
    B/L changes is not required, only on ndim/dtype changes.
    """
    y = x.astype(mx.float32)
    z = mx.fast.rms_norm(y.flatten(-2), None, norm_eps)
    mixes = z @ fn_t
    return y, mixes


# Module-level cache: one compiled callable, shared by every HyperConnection
# instance (attn_hc and ffn_hc across all 45 layers) -- the weights (fn_t) are
# passed as an explicit input on every call, never captured, so nothing here
# is instance-specific and a single compiled object is correct for all of
# them. mx.compile's own internal shape cache handles the handful of distinct
# (B, L) shapes seen (prefill chunk width + tail, plus decode S=1..8).
_hc_preamble_compiled = mx.compile(_hc_preamble_impl, shapeless=True)


def _hc_full_ops_impl(x, y, mixes, scale, base, hc_mult, sinkhorn_iters, eps):
    """Whole non-GPU-kernel HyperConnection body, one compiled unit: sigmoid
    pre/post gates + comb softmax + sinkhorn iterations (same math as
    ``_hc_split_sinkhorn_ops``, inlined here so it fuses with the final
    reduction into ONE compiled graph) followed by the ``(pre * y).sum``
    collapse. This is the "whole block" fusion the I964 note flags as never
    tried -- I964 only compiled the sinkhorn sub-expression.

    Not `shapeless`: ``comb`` is reshaped to ``(*mixes.shape[:-1], hc_mult,
    hc_mult)``, a reshape whose target size is read from a traced shape value,
    which is exactly the case shapeless compilation does not cover (matches
    the existing `_hc_split_sinkhorn_ops` precedent, also plain `mx.compile`
    for the same reason). Recompiles per distinct (B, L); acceptable per the
    task brief since prefill only visits the 8192-token chunk shape plus one
    tail shape.
    """
    mixes = mixes.astype(mx.float32)
    scale = scale.astype(mx.float32)
    base = base.astype(mx.float32)
    pre_scale, post_scale, comb_scale = scale[0], scale[1], scale[2]

    pre = mx.sigmoid(mixes[..., :hc_mult] * pre_scale + base[:hc_mult]) + eps
    post = 2 * mx.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * post_scale + base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].reshape(
        *mixes.shape[:-1], hc_mult, hc_mult
    ) * comb_scale + base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)

    xc = (pre[..., None] * y).sum(axis=2).astype(x.dtype)
    return xc, post, comb


# Module-level cache, same rationale as _hc_preamble_compiled: hc_mult and
# sinkhorn_iters are plain python ints/floats (config-derived, identical
# across all HyperConnection instances in one model), scale/base are the only
# per-instance arrays and they are explicit inputs, not captured.
_hc_full_ops_compiled = mx.compile(_hc_full_ops_impl)


class HyperConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps

        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = mx.zeros((mix, self.hc_mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)

    def __call__(self, x: mx.array, norm_w=None, norm_eps: float = 1e-5):
        """Hyper-connection.

        ``norm_w`` folds the RMSNorm that every caller applies to the collapsed
        output into the fused kernel, so the collapse and the norm are one
        dispatch instead of two. The folded norm reads the collapse from
        registers, before it is rounded to the output dtype, so it is very
        slightly MORE precise than norming the stored result -- the two agree to
        about one ulp of the output dtype, not bit-exactly.
        """
        B, L, H, D = x.shape
        hc_compile = _hc_compile_enabled()
        use_ops = (
            self.training
            or mx.default_device() != mx.gpu
            or not mx.metal.is_available()
        )
        # L34: on the fused-kernel path ``y`` is DEAD -- the sinkhorn+collapse
        # kernel reads ``x`` and widens it itself -- so the fp32 cast exists
        # only to feed the rms_norm, and the two collapse into one kernel that
        # reads bf16 and writes the fp32 normed row.  The ops path still needs
        # ``y`` for its collapse, so it keeps the cast.
        y = None
        if (
            not use_ops
            and _hc_prenorm_fused_enabled()
            and hc_prenorm_fused_eligible(x)
        ):
            mixes = _hc_prenorm_fused(x, self.norm_eps) @ self.fn.T
        elif hc_compile:
            y, mixes = _hc_preamble_compiled(x, self.fn.T, self.norm_eps)
        else:
            y = x.astype(mx.float32)
            z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
            mixes = z @ self.fn.T
        if use_ops:
            if hc_compile:
                xc, post, comb = _hc_full_ops_compiled(
                    x, y, mixes, self.scale, self.base,
                    self.hc_mult, self.sinkhorn_iters, self.hc_eps,
                )
            else:
                xc, post, comb = _hc_ops(
                    x, y, mixes, self.scale, self.base,
                    self.hc_mult, self.sinkhorn_iters, self.hc_eps,
                )
            # Same contract on the ops path, so a caller that hands us norm_w
            # gets a normed collapse whichever path ran.
            if norm_w is not None:
                xc = mx.fast.rms_norm(xc, norm_w, norm_eps)
            return xc, post, comb

        return _hc_kernel(
            x,
            y,
            mixes,
            self.scale,
            self.base,
            self.hc_mult,
            self.sinkhorn_iters,
            self.hc_eps,
            norm_w=norm_w,
            norm_eps=norm_eps,
        )


@mx.compile
def _hc_expand_op(x, residual, post, comb):
    y = post[..., None] * x[:, :, None, :].astype(mx.float32)
    y = y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))
    return y.astype(x.dtype)


def hc_expand(x, residual, post, comb):
    """Residual inject.

    Eager/compiled form is ``(post * x) + comb^T @ residual``, which
    ``mx.compile`` lowers to an fp32 cast of ``residual``, the batched matmul,
    and one fused element-wise kernel -- three passes over a
    ``[B, L, hc_mult, D]`` fp32 tensor.  At prefill width those passes ARE the
    cost, so L34 replaces them with a single kernel that reads the bf16 inputs
    once and writes the bf16 output once (see ``_HC_EXPAND_SOURCE``).

    The fused kernel is the DEFAULT since I1450; ``MLX_VLM_GLM5_HC_FUSED=0``,
    a sub-512-row call, and any box without Metal all land on
    ``_hc_expand_op`` below, which is unchanged.
    """
    if _hc_fused_enabled() and hc_expand_fused_eligible(x, residual, post, comb):
        return _hc_expand_fused(x, residual, post, comb)
    return _hc_expand_op(x, residual, post, comb)


class HyperHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.fn = mx.zeros(
            (self.hc_mult, self.hc_mult * config.hidden_size), dtype=mx.float32
        )
        self.base = mx.zeros((self.hc_mult,), dtype=mx.float32)
        self.scale = mx.ones((1,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        y = x.astype(mx.float32)
        z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
        mixes = z @ self.fn.T
        pre = mx.sigmoid(mixes * self.scale + self.base) + self.hc_eps
        return (pre[..., None] * y).sum(axis=2).astype(x.dtype)
