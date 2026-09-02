"""Fused flash-style attention for MLA head dims that MLX cannot fuse.

MLX v0.32.0 refuses to fuse ``scaled_dot_product_attention`` for this model twice over
(mlx/backend/metal/scaled_dot_product_attention.cpp, ``ScaledDotProductAttention::use_fallback``):

  * ``sdpa_full_supported_head_dim``   is {64, 80, 128}          -- absorbed MLA is 512.
  * ``sdpa_vector_supported_head_dim`` is {64, 96, 128, 256}     -- absorbed MLA is 512, AND
    the vector path additionally requires ``qL * gqa_factor <= 32`` while absorbed MLA runs
    ``gqa_factor == 64`` (one latent KV "head" shared by all 64 query heads).

So every attention in the 11 DSA layers takes the COMPOSITE fallback (mlx/fast.cpp), which
materialises the ``[B, 1, H, L, Kv]`` score tensor three times over (matmul -> mask -> softmax
-> matmul).  This module replaces that with a single ``mx.fast.metal_kernel`` launch.

WHY HEAD DIM 512 DOES NOT FIT MLX'S OWN STEEL ATTENTION
--------------------------------------------------------
``mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h`` gives EVERY simdgroup the
whole head dim:  ``MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile`` with ``TD = BD / 8``.
At BD=128 that is 16 fragments = 32 fp32 registers/thread; at BD=512 it is 64 fragments =
**128 registers/thread** for the accumulator alone.  And it stages Q in threadgroup memory:
``BQ * (BD + padQ) * sizeof(T)`` = 32 * 520 * 2 = **33.3 KB**, over the 32 KB threadgroup limit.
Two independent walls, both at BD=512.

WHAT THIS KERNEL DOES INSTEAD
-----------------------------
Partition the HEAD DIM across simdgroups instead of replicating it into each one.  With
NSG = 32 simdgroups laid out WM=4 x WN=8:

  * O tile [BQ=32, D=512] fp32 is owned as (8 rows) x (64 cols) per simdgroup = 8 fragments
    = **16 fp32 registers/thread** (vs steel's 128).
  * S tile [BQ=32, BK=64] is exactly 4x8 = 32 fragments = one per simdgroup, so the QK matmul
    needs no cross-simdgroup split and every simdgroup does the full 512-deep contraction.
  * Q, K and V are read with ``simdgroup_load`` straight out of the ``device`` address space,
    so threadgroup memory holds only the scores (8 KB), the bf16 probabilities (4.6 KB) and
    three 32-float row vectors.  ~13 KB of the 32 KB budget.

That is only affordable because the model is MQA-shaped: ONE latent KV row serves all 64 query
heads, so the 8-fold widening of the simdgroup grid over the head dim does not multiply K/V
traffic.  And in absorbed MLA ``k is v`` -- the same latent -- so the KV tile is fetched ONCE
and used for both matmuls, halving KV bytes against a generic fused kernel.

CANONICAL SHAPES
----------------
The kernel works on a 3-D view so one kernel serves both call sites:

    q   [G,  R, D]      G row-groups of R rows
    kv  [Gk, N, D]      Gk in {1, G}; group g reads kv group ``g / (G // Gk)``
    out [G,  R, D]

  * dense MLA prefill      q [B,H,L,512] -> [B*H, L, 512],  kv [B,1,N,512] -> [B, N, 512]
  * gathered MLA (top-k)   q [Bx,H,1,512] -> [Bx, H, 512],  kv [Bx,1,N,512] -> [Bx, N, 512]
    (at L==1 with a single KV head the head axis and the query axis are interchangeable --
     that fold is what turns 64 broadcast GEMVs into one GEMM; see _gathered_attention.)

NUMERICS
--------
fp32 score and output accumulation, bf16 inputs/outputs, probabilities rounded to bf16 before
the PV matmul (which is what the composite path does too: mx.softmax writes bf16).  The softmax
uses ``metal::fast::exp`` to match MLX's own ``softmax.h`` (``softmax_exp`` = ``fast::exp``) and
``sdpa_vector.h``.  Fully-masked rows follow MLX's fused convention (finite_min sentinel), not
the composite's; the model never sends one (``_gathered_attention`` forces a key and zeroes the
row itself; causal and top-k-with-tail always keep at least one key).
"""

import math
from typing import Optional

import mlx.core as mx

_HEADER = r"""
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

// Lane -> (col, row) inside an 8x8 simdgroup fragment.  Transcribed from
// mlx/backend/metal/kernels/steel/attn/mma.h  BaseMMAFrag<T,8,8>::get_coord.
// kElemRows == 1 is what makes the per-row softmax rescale below a single scalar
// multiply per thread: a thread's two elements always live in the SAME row.
inline short2 mla_frag_coord(ushort lane) {
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  return short2(fn, fm);
}
"""

# BQ x BK score tile; NSG = (BQ/8) * (BK/8) simdgroups, and the same grid is reused for the
# O tile as WM=(BQ/8) x WN=(BK/8) with each simdgroup owning D/WN output columns.
_BQ = 32
_BK = 64
_NSG = (_BQ // 8) * (_BK // 8)          # 32
_NTHREADS = _NSG * 32                   # 1024

_SOURCE = r"""
  constexpr int BQ  = """ + str(_BQ) + r""";
  constexpr int BK  = """ + str(_BK) + r""";
  constexpr int WM  = BQ / 8;                 // 4  simdgroup rows
  constexpr int WN  = BK / 8;                 // 8  simdgroup cols
  constexpr int NSG = WM * WN;                // 32 simdgroups
  constexpr int DCOLS = D / WN;               // 64 output cols per simdgroup
  constexpr int NFRAG = DCOLS / 8;            // 8  O fragments per simdgroup
  constexpr int LDS = BK + 8;                 // score tile row stride (pad: bank conflicts)
  constexpr int LDP = BK + 8;                 // prob tile row stride

  const uint sg   = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint qt   = threadgroup_position_in_grid.x;   // query tile
  const uint g    = threadgroup_position_in_grid.z;   // row group

  const int R = p[0];
  const int N = p[1];          // keys, already a multiple of BK (caller pads)
  const int GK_DIV = p[2];     // kv group = g / GK_DIV
  const int MSK_GS = p[3];     // mask strides; 0 disables that axis
  const int MSK_RS = p[4];
  const int NV     = p[5];     // valid keys; keys >= NV are sentinel-masked (padding)

  const device bfloat16_t* Q  = q  + (ulong)g * R * D;
  const device bfloat16_t* KV = kv + (ulong)(g / GK_DIV) * N * D;

  threadgroup float  Sm[BQ * LDS];
  threadgroup bfloat16_t Pm[BQ * LDP];
  threadgroup float  mrow[BQ];
  threadgroup float  lrow[BQ];
  threadgroup float  crow[BQ];
  threadgroup int    tile_live;

  const short2 fc = mla_frag_coord(lane);
  const short sn = fc.x;      // col within fragment (this thread owns sn, sn+1)
  const short sm = fc.y;      // row within fragment
  const uint  wm = sg / WN;
  const uint  wn = sg % WN;

  const int q_row0 = qt * BQ;

  // ---- init row state
  for (uint i = thread_position_in_threadgroup.x; i < BQ; i += NSG * 32) {
    mrow[i] = -3.0e38f;
    lrow[i] = 0.0f;
  }

  simdgroup_float8x8 O[NFRAG];
  for (int f = 0; f < NFRAG; ++f) O[f] = simdgroup_float8x8(0);

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int nk_tiles = N / BK;
  for (int kb = 0; kb < nk_tiles; ++kb) {
    const int k0 = kb * BK;

    // ---- tile liveness: skip a tile no query in this block can see.
    // Generic over causal and over the indexer's scattered top-k mask; this is where the
    // composite's wasted work lives (it multiplies the full rectangle and masks afterwards).
#if HAS_MASK
    {
      if (thread_position_in_threadgroup.x == 0) tile_live = 0;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      int live = 0;
      for (uint i = thread_position_in_threadgroup.x; i < BQ * BK; i += NSG * 32) {
        int r = q_row0 + int(i / BK);
        int c = k0 + int(i % BK);
        if (r < R && c < NV) {
          ulong mo = (ulong)(MSK_GS ? g * MSK_GS : 0) + (ulong)(MSK_RS ? r * MSK_RS : 0) + (ulong)c;
          if (mask[mo]) { live = 1; break; }
        }
      }
      if (live) tile_live = 1;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tile_live == 0) continue;
    }
#endif

    // ---- S = Q . K^T   (one 8x8 fragment per simdgroup, full D contraction)
    simdgroup_float8x8 S = simdgroup_float8x8(0);
    {
      // Batch UNROLL fragment loads before the mmas.  With one 8x8 output fragment per
      // simdgroup the natural loop is 2 device loads per mma and stalls on load latency;
      // issuing UNROLL pairs up front lets the compiler overlap them.  This is the single
      // largest throughput lever measured on this kernel.
      constexpr int UNROLL = 4;
      const device bfloat16_t* qp = Q + (ulong)(q_row0 + 8 * wm) * D;
      const device bfloat16_t* kp = KV + (ulong)(k0 + 8 * wn) * D;
      for (int d = 0; d < D; d += 8 * UNROLL) {
        simdgroup_bfloat8x8 A[UNROLL], B[UNROLL];
#pragma clang loop unroll(full)
        for (int u = 0; u < UNROLL; ++u) {
          simdgroup_load(A[u], qp + d + 8 * u, D);
          simdgroup_load(B[u], kp + d + 8 * u, D, ulong2(0, 0), true);   // K^T
        }
#pragma clang loop unroll(full)
        for (int u = 0; u < UNROLL; ++u)
          simdgroup_multiply_accumulate(S, A[u], B[u], S);
      }
    }
    simdgroup_store(S, Sm + (8 * wm) * LDS + 8 * wn, LDS);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- online softmax: simdgroup `sg` owns row `sg` (NSG == BQ), 2 cols per lane
    {
      const int r = sg;
      const int gr = q_row0 + r;
      float v0 = -3.0e38f, v1 = -3.0e38f;
      const int c0 = int(lane) * 2;
      if (gr < R) {
        v0 = Sm[r * LDS + c0]     * SCALE;
        v1 = Sm[r * LDS + c0 + 1] * SCALE;
        const int gc0 = k0 + c0;
        if (gc0     >= NV) v0 = -3.0e38f;
        if (gc0 + 1 >= NV) v1 = -3.0e38f;
#if HAS_MASK
        {
          ulong mo = (ulong)(MSK_GS ? g * MSK_GS : 0) + (ulong)(MSK_RS ? gr * MSK_RS : 0);
          if (gc0     <  NV && !mask[mo + gc0])     v0 = -3.0e38f;
          if (gc0 + 1 <  NV && !mask[mo + gc0 + 1]) v1 = -3.0e38f;
        }
#endif
      }
      const float mt = simd_max(fmax(v0, v1));
      const float mprev = mrow[r];
      const float mnew = fmax(mprev, mt);
      const float corr = fast::exp(mprev - mnew);
      const float p0 = fast::exp(v0 - mnew);
      const float p1 = fast::exp(v1 - mnew);
      const float st = simd_sum(p0 + p1);
      Pm[r * LDP + c0]     = bfloat16_t(p0);
      Pm[r * LDP + c0 + 1] = bfloat16_t(p1);
      if (lane == 0) {
        mrow[r] = mnew;
        lrow[r] = lrow[r] * corr + st;
        crow[r] = corr;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- rescale O by the row correction (kElemRows == 1 => one scalar per thread)
    {
      const float c = crow[8 * wm + sm];
      for (int f = 0; f < NFRAG; ++f) {
        thread auto& e = O[f].thread_elements();
        e[0] *= c; e[1] *= c;
      }
    }

    // ---- O += P . V   (V is the same latent tile as K in absorbed MLA)
    {
      const threadgroup bfloat16_t* pp = Pm + (8 * wm) * LDP;
      const device bfloat16_t* vp = KV + (ulong)k0 * D + wn * DCOLS;
      for (int kk = 0; kk < BK; kk += 8) {
        simdgroup_bfloat8x8 A;
        simdgroup_load(A, pp + kk, LDP);
        simdgroup_bfloat8x8 B[NFRAG];
#pragma clang loop unroll(full)
        for (int f = 0; f < NFRAG; ++f)
          simdgroup_load(B[f], vp + (ulong)kk * D + f * 8, D);
#pragma clang loop unroll(full)
        for (int f = 0; f < NFRAG; ++f)
          simdgroup_multiply_accumulate(O[f], A, B[f], O[f]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // ---- epilogue: divide by the softmax denominator and store bf16
  {
    const int gr = q_row0 + 8 * int(wm) + sm;
    if (gr < R) {
      const float den = lrow[8 * wm + sm];
      const float inv = den > 0.0f ? 1.0f / den : 0.0f;
      device bfloat16_t* op = out + (ulong)g * R * D + (ulong)gr * D + wn * DCOLS + sn;
      for (int f = 0; f < NFRAG; ++f) {
        thread auto& e = O[f].thread_elements();
        op[f * 8]     = bfloat16_t(e[0] * inv);
        op[f * 8 + 1] = bfloat16_t(e[1] * inv);
      }
    }
  }
"""

_KERNEL_CACHE = {}


def _kernel(D: int, has_mask: bool, scale: float):
    key = (D, has_mask, scale)
    k = _KERNEL_CACHE.get(key)
    if k is None:
        src = _SOURCE.replace("SCALE", f"{scale!r}f")
        src = (f"#define HAS_MASK {1 if has_mask else 0}\n"
               f"  constexpr int D = {D};\n") + src
        k = mx.fast.metal_kernel(
            name=f"mla_flash_d{D}_{'m' if has_mask else 'n'}",
            input_names=["q", "kv", "p"] + (["mask"] if has_mask else []),
            output_names=["out"],
            header=_HEADER,
            source=src,
            ensure_row_contiguous=True,
        )
        _KERNEL_CACHE[key] = k
    return k


def mla_flash_attention(
    q: mx.array,
    kv: mx.array,
    scale: float,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """Fused MQA attention on the canonical 3-D view.

    q    [G, R, D]  bfloat16
    kv   [Gk, N, D] bfloat16, Gk in {1, G}; k and v are the same tensor (absorbed MLA)
    mask [Gm, Rm, N] bool, Gm in {1, G}, Rm in {1, R}, or None
    ->   [G, R, D]  bfloat16
    """
    G, R, D = q.shape
    Gk, N, Dk = kv.shape
    if Dk != D:
        raise ValueError(f"kv head dim {Dk} != q head dim {D}")
    if D % 64 != 0:
        raise ValueError(f"head dim {D} must be a multiple of 64")
    if G % Gk != 0:
        raise ValueError(f"kv groups {Gk} must divide q groups {G}")
    if q.dtype != mx.bfloat16 or kv.dtype != mx.bfloat16:
        raise ValueError("fused MLA attention is bfloat16-only")

    n_valid = N
    pad = (-N) % _BK
    if pad:
        kv = mx.pad(kv, [(0, 0), (0, pad), (0, 0)])
        N = N + pad

    has_mask = mask is not None
    if has_mask:
        if mask.dtype != mx.bool_:
            raise ValueError("only boolean masks are supported")
        m = mask
        while m.ndim < 3:
            m = m[None]
        Gm, Rm, Nm = m.shape
        if Nm != n_valid:
            raise ValueError(f"mask key axis {Nm} != {n_valid}")
        if Gm not in (1, G) or Rm not in (1, R):
            raise ValueError(f"mask shape {m.shape} not broadcastable to ({G},{R},{n_valid})")
        m = mx.contiguous(m)
        msk_gs = Rm * Nm if Gm > 1 else 0
        msk_rs = Nm if Rm > 1 else 0
    else:
        msk_gs = msk_rs = 0

    p = mx.array([R, N, G // Gk, msk_gs, msk_rs, n_valid], dtype=mx.int32)
    inputs = [q, kv, p] + ([m] if has_mask else [])
    n_qtiles = (R + _BQ - 1) // _BQ
    return _kernel(D, has_mask, scale)(
        inputs=inputs,
        output_shapes=[(G, R, D)],
        output_dtypes=[mx.bfloat16],
        grid=(_NTHREADS * n_qtiles, 1, G),
        threadgroup=(_NTHREADS, 1, 1),
    )[0]
