"""Index-consuming (sparse) MLA attention for the DSA prefill gather.

WHY THIS FILE EXISTS
--------------------
``Glm5NextSparseAttention._gathered_attention`` (language.py:2375) materialises the
selected latents with

    kv_g = mx.take_along_axis(
        mx.broadcast_to(kv_latent, (B, lc, Kv, dim)),                 # language.py:2405
        mx.broadcast_to(clamped[:, a0:a1, :, None], (B, lc, topk, dim)),
        axis=2)

and then hands ``kv_g`` to ``_mqa_sdpa``.  Measured on the served build (receipt
logs/sweep11/V5_PREFILL_SYNC_20260907/v5.json, ledger I1493) that gather is the single
largest inefficiency in prefill: 167.3 ms per DSA layer at depth 0 against a 24.05 ms
bandwidth floor (6.96x) and 268.2 ms at depth 24576 against 29.59 ms (9.06x); the
gathered-attention CORE (gather + sdpa, the only quantity the harness measures
DIRECTLY -- the split between them is extrapolated from q-chunk 0 and its
``core_extrapolation_ratio`` is 1.45 at depth 0 and 8.05 at depth 24576, i.e. the split
is unreliable at depth) is 260.8 / 273.7 ms per DSA layer against a 90.1 ms compute floor.

WHY THE GATHER RUNS AT ~1/7 OF BANDWIDTH -- the mechanism, from the MLX sources
-------------------------------------------------------------------------------
``take_along_axis`` lowers to the ``GatherAxis`` primitive (mlx/ops.cpp:3750-3775) and on
Metal to ``gather_axis`` (mlx/backend/metal/indexing.cpp:440-514, kernel
mlx/backend/metal/kernels/indexing/gather_axis.h).  The dispatch is

    grid = (size_post, idx.shape[axis], size_pre) = (dim=512, topk=2051, lc)

i.e. **one thread per output ELEMENT** -- 2 bytes of payload per thread.  Worse, the
kernel is specialised on ``src.flags().row_contiguous`` and ``idx.flags().row_contiguous``
(indexing.cpp:462-464) and BOTH operands here are ``broadcast_to`` views, so neither is
row-contiguous and every thread runs TWO ``elem_to_loc`` address computations
(gather_axis.h:28-40), each an ndim-1 = 3 deep chain of integer div/mod, to move two
bytes.  At depth 0 that is 8192*2051*512 = 8.6e9 threads per layer and ~5e10 integer
divisions; the 103 GB/s effective rate the receipt implies is an ALU wall, not a memory
wall.  (The same ``src.size() > INT32_MAX`` test at indexing.cpp:454 is the 2**31 cliff
that ``_gather_q_chunk_for`` (language.py:494) already works around by SHRINKING the query
chunk with depth -- 256 at depth 0, 64 at depth 24576 -- which is also why the number of
gather launches grows 32 -> 128 with depth, see v5.json kernel_census.)

MLX ALREADY HAS THE RIGHT KERNEL, ON A DIFFERENT PRIMITIVE
----------------------------------------------------------
``Gather::eval_gpu`` has a fast path (mlx/backend/metal/indexing.cpp:431-476) taken when

    src.row_contiguous && nidx == 1 && axes[0] == 0 && idx.row_contiguous
    && slice_size == src.strides()[0]

which dispatches ``gather_front`` (kernels/indexing/gather_front.h): grid
``(slice_size / work_per_thread, n_indices)``, ONE index load per output row, no
``elem_to_loc`` at all, and ``work_per_thread = 2`` for a 2-byte dtype with slice_size > 8.
``mx.take(kv2d, idx, axis=0)`` with ``kv2d`` [Kv, dim] row-contiguous and ``idx`` [G, topk]
contiguous hits exactly that path (mx.take -> gather(a, indices, axis=0,
slice_sizes=[1,dim]), mlx/ops.cpp:3690-3720), and produces the SAME [G, topk, dim] tensor,
element for element, as the take_along_axis form.  That is mode ``take`` below: a bit-exact
drop-in that deletes the ALU wall and leaves a pure 1 KB-row copy.

MODES  (env ``MLX_VLM_GLM5_DSA_GATHER_KERNEL``, default off/eager)
------------------------------------------------------------------
  eager  (default, or "0"/"off")  language.py's take_along_axis, untouched.
  take                            gather_front via mx.take.  BIT-EXACT with eager.
  fused                           this file's ``mla_sparse_flash_attention``: one metal
                                  kernel that consumes the INDEX LIST and never
                                  materialises [G, topk, dim] at all.  NOT bit-exact
                                  (fp32 online-softmax reassociation, bf16 P) -- gate it
                                  with the campaign's KL gate, not with identity.

BYTES, PER DSA LAYER, S = 8192, topk(W) = 2051, dim = 512, bf16
---------------------------------------------------------------
  eager : write kv_g 17.2 GB + scattered read 17.2 GB + sdpa re-reads kv_g twice
          (QK and PV) 34.4 GB                                  ~= 69 GB
  take  : same 17.2 + 17.2 write/read, same 34.4 re-read       ~= 69 GB, but at
          gather_front's ~1 index-load-per-row instead of 2 elem_to_loc per element
  fused : scattered read 17.2 GB, once (BQ=64 covers all 64 head rows of a query in
          one tile, so each selected latent row is read exactly once)  = 17.2 GB
          -> 23.7 ms at 726.5 GB/s; compute floor 90.1 ms at 24.44 TFLOP/s, so the
          fused path is COMPUTE-bound and its floor is the sdpa floor.

STATUS: the ``fused`` kernel has never executed.  It was written and compile-checked
(``xcrun metal -std=metal3.1``) inside a measurement window with the GPU forbidden; its
CPU reference (``mla_sparse_reference_online``) is what has been checked against the eager
path.  Treat every fused number as a prediction.
"""

import math
import os
from typing import Optional, Tuple

import mlx.core as mx

# --------------------------------------------------------------------------- #
# mode selection
# --------------------------------------------------------------------------- #

_MODE = None
_VALID_MODES = ("eager", "take", "fused")


def dsa_gather_mode() -> str:
    """``MLX_VLM_GLM5_DSA_GATHER_KERNEL`` in {eager|take|fused}.  Default eager."""
    global _MODE
    if _MODE is None:
        v = os.environ.get("MLX_VLM_GLM5_DSA_GATHER_KERNEL", "").strip().lower()
        if v in ("", "0", "off", "false", "no", "eager"):
            _MODE = "eager"
        elif v in _VALID_MODES:
            _MODE = v
        else:
            raise ValueError(
                f"MLX_VLM_GLM5_DSA_GATHER_KERNEL={v!r} not in {_VALID_MODES}"
            )
    return _MODE


def _reset_mode_cache() -> None:
    """Tests only: re-read the env on the next call."""
    global _MODE
    _MODE = None


# --------------------------------------------------------------------------- #
# mode "take": bit-exact gather on MLX's gather_front fast path
# --------------------------------------------------------------------------- #


def gather_latents_take(kv_latent: mx.array, clamped: mx.array) -> mx.array:
    """Gather [B, lc, topk, dim] selected latents through ``Gather``'s front fast path.

    kv_latent  [B, 1, Kv, dim]   clamped  [B, lc, topk] (already clipped to [0, Kv))
    ->         [B, lc, topk, dim], element-identical to the take_along_axis form.

    ``mx.take(a, idx, axis=0)`` on a row-contiguous 2-D ``a`` and a contiguous ``idx``
    takes ``Gather::eval_gpu``'s gather_front branch (indexing.cpp:431).  The batch axis is
    folded into the row index (b * Kv) so the whole thing stays ONE gather even when B > 1;
    that also keeps a batched, non-contiguous cache slice on the fast path after the single
    reshape-copy below.
    """
    B, one, Kv, dim = kv_latent.shape
    assert one == 1, f"kv_latent must be [B, 1, Kv, dim], got {kv_latent.shape}"
    lc, topk = clamped.shape[1], clamped.shape[2]
    # reshape() copies iff the cache slice is not row-contiguous (true only for B > 1);
    # the copy is one pass over the latent (33 MB at 32k) and happens once per q-chunk.
    kv2d = kv_latent.reshape(B * Kv, dim)
    idx = clamped
    if B > 1:
        idx = idx + (mx.arange(B, dtype=idx.dtype) * Kv)[:, None, None]
    out = mx.take(kv2d, idx.reshape(B * lc, topk), axis=0)   # [B*lc, topk, dim]
    return out.reshape(B, lc, topk, dim)


# --------------------------------------------------------------------------- #
# mode "fused": one kernel, index list in, attention out, no [G, topk, dim]
# --------------------------------------------------------------------------- #
#
# GEOMETRY, and why it is not the dense kernel's.
#
# fused_mla_attn.py can read K straight out of `device` with simdgroup_load because its
# key rows are CONTIGUOUS.  Here they are not: row j of the tile is kv[idx[g, k0+j]].
# simdgroup_load has no per-row indirection, so the tile must be STAGED in threadgroup
# memory -- and that staging is what fixes the tiling:
#
#   BK * D * 2 bytes must fit the 32 KB threadgroup budget alongside the score tile.
#   BK = 16 -> 16 * 520 * 2 = 16.6 KB (row stride padded to D+8 = 520, which keeps every
#   row 16-byte aligned so the staging copy can use uint4).  BK = 32 would be 33.3 KB:
#   over budget on its own.  So BK = 16, and with SFRAG = 1 that fixes WN = BK/8 = 2,
#   hence DCOLS = D/WN = 256 and NFRAG = 32 O-fragments = 64 fp32 accumulator registers
#   per thread (the dense kernel has 8 fragments / 16 registers; steel's 512-dim form
#   would need 128).  That is the main untested risk in this file: 64 accumulator
#   registers may cost occupancy or spill.  MLX_VLM_GLM5_DSA_GATHER_TILE lets the GPU
#   sweep try the alternatives without editing the source.
#
#   BQ = 64 is not a free parameter either: R = num_heads = 64 (the MQA fold, see
#   _mqa_fold_enabled), and BQ >= R is what makes each selected latent row be read
#   EXACTLY ONCE per query.  BQ = 32 would read the whole 2 MB key set twice per query
#   and hand back half the win; it is offered only as a register-pressure escape hatch.
#
# Staging a tile costs exactly BK*D*2 bytes of device traffic and each 1 KB row is one
# aligned burst of 8 cache lines with no waste.  The indexer selects `index_kpool` = 4
# CONSECUTIVE keys per pool (language.py:2183-2189: pool_indices expanded by kpool), so
# a BK = 16 tile is 4 runs of 4 KB, not 16 random rows -- better locality than the model
# nominally promises.

_TILE = None


def _tile() -> Tuple[int, int]:
    """(BQ, BK) from ``MLX_VLM_GLM5_DSA_GATHER_TILE`` ("64x16" default)."""
    global _TILE
    if _TILE is None:
        v = os.environ.get("MLX_VLM_GLM5_DSA_GATHER_TILE", "64x16").lower()
        bq, _, bk = v.partition("x")
        _TILE = (int(bq), int(bk))
        if _TILE[0] % 8 or _TILE[1] % 8:
            raise ValueError(f"tile {_TILE} must be a multiple of 8 in both axes")
    return _TILE


_HEADER = r"""
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

// Lane -> (col, row) inside an 8x8 simdgroup fragment.  Transcribed from
// mlx/backend/metal/kernels/steel/attn/mma.h  BaseMMAFrag<T,8,8>::get_coord, exactly as
// in fused_mla_attn.py: kElemRows == 1, so a thread's two elements share a ROW and the
// softmax rescale is one scalar multiply per thread.
inline short2 mla_sparse_frag_coord(ushort lane) {
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  return short2(fn, fm);
}
"""

_SOURCE = r"""
  constexpr int WM  = BQ / 8;              // simdgroup rows
  constexpr int WN  = BK / 8;              // simdgroup cols (SFRAG == 1)
  constexpr int NSG = WM * WN;
  constexpr int NTHREADS = NSG * 32;
  constexpr int DCOLS = D / WN;            // output cols owned by one simdgroup
  constexpr int NFRAG = DCOLS / 8;         // O fragments per simdgroup
  constexpr int LDK = D + 8;               // staged key row stride (multiple of 8)
  constexpr int LDS = BK + 8;              // score tile row stride
  constexpr int VPR = D / 8;               // uint4 chunks per staged row
  // Softmax lane geometry.  NTHREADS / BQ threads cooperate on one query row, so
  // LPR = (BQ/8)*(BK/8)*32/BQ = BK/2 and every lane owns CPL = 2 score columns; the
  // xor-shuffle reduction below spans LPR lanes and stays inside one simdgroup because
  // LPR divides 32 (BK <= 64).  NSG * (32/LPR) == BQ identically, so the whole score
  // tile is covered by exactly one pass, whatever the tile.
  constexpr int LPR = BK / 2;              // lanes per query row
  constexpr int CPL = 2;                   // score cols per lane
  constexpr int RPSG = 32 / LPR;           // query rows per simdgroup

  const uint tid  = thread_position_in_threadgroup.x;
  const uint sg   = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint qt   = threadgroup_position_in_grid.x;   // query tile
  const uint g    = threadgroup_position_in_grid.z;   // query group (one query)

  const int R  = p[0];      // rows per group (== num_heads under the MQA fold)
  const int K  = p[1];      // selected keys per group
  const int KV = p[2];      // rows in the latent cache (bounds check only)

  const device bfloat16_t* Q = q + (ulong)g * R * D;
  const device int*     idxp = idx + (ulong)g * K;

  threadgroup uint4  Ktv[BK * (LDK / 8)];
  threadgroup float  Sm[BQ * LDS];
  threadgroup bfloat16_t Pm[BQ * LDS];
  threadgroup float  mrow[BQ];
  threadgroup float  lrow[BQ];
  threadgroup float  crow[BQ];
  threadgroup bool   kok[BK];
  threadgroup bfloat16_t* Kt = (threadgroup bfloat16_t*)Ktv;

  const short2 fc = mla_sparse_frag_coord(lane);
  const short sn = fc.x;
  const short sm = fc.y;
  const uint  wm = sg / WN;
  const uint  wn = sg % WN;
  const int q_row0 = int(qt) * BQ;

  for (uint i = tid; i < BQ; i += NTHREADS) { mrow[i] = -3.0e38f; lrow[i] = 0.0f; }

  simdgroup_float8x8 O[NFRAG];
  for (int f = 0; f < NFRAG; ++f) O[f] = simdgroup_float8x8(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int nk_tiles = (K + BK - 1) / BK;
  for (int kb = 0; kb < nk_tiles; ++kb) {
    const int k0 = kb * BK;

    // ---- stage BK gathered latent rows.  THIS IS THE GATHER: one aligned 1 KB burst
    //      per selected key, straight into threadgroup memory, never written to device.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = tid; t < BK * VPR; t += NTHREADS) {
      const int r = int(t) / VPR;
      const int c = int(t) % VPR;
      const int key = k0 + r;
      int row = -1;
      if (key < K) { row = idxp[key]; if (row >= KV) row = -1; }
      threadgroup uint4* dst = Ktv + r * (LDK / 8);
      dst[c] = (row >= 0) ? ((const device uint4*)(kv + (ulong)row * D))[c] : uint4(0);
      if (c == 0) kok[r] = (row >= 0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- S = Q . K^T, full D contraction, one 8x8 fragment per simdgroup
    {
      simdgroup_float8x8 S0 = simdgroup_float8x8(0);
      const device bfloat16_t* qp = Q + (ulong)(q_row0 + 8 * wm) * D;
      const threadgroup bfloat16_t* kp = Kt + (ulong)(8 * wn) * LDK;
      for (int d = 0; d < D; d += 8) {
        simdgroup_bfloat8x8 A, B0;
        simdgroup_load(A, qp + d, D);
        simdgroup_load(B0, kp + d, LDK, ulong2(0, 0), true);   // K^T from threadgroup
        simdgroup_multiply_accumulate(S0, A, B0, S0);
      }
      simdgroup_store(S0, Sm + (8 * wm) * LDS + 8 * wn, LDS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- online softmax.  LPR lanes cooperate on one query row; the xor-shuffle
    //      reduction stays inside one simdgroup because LPR divides 32.
    {
      const int r  = int(sg) * RPSG + int(lane) / LPR;
      const int c0 = (int(lane) % LPR) * CPL;
      const int gr = q_row0 + r;
      float v[CPL];
      float mt = -3.0e38f;
      for (int t = 0; t < CPL; ++t) {
        float x = -3.0e38f;
        const int c = c0 + t;
        if (gr < R && (k0 + c) < K && kok[c]) x = Sm[r * LDS + c] * SCALE;
        v[t] = x;
        mt = fmax(mt, x);
      }
      for (int off = 1; off < LPR; off <<= 1) mt = fmax(mt, simd_shuffle_xor(mt, ushort(off)));
      const float mprev = mrow[r];
      const float mnew = fmax(mprev, mt);
      const float corr = fast::exp(mprev - mnew);
      float sl = 0.0f;
      for (int t = 0; t < CPL; ++t) {
        const float pt = (v[t] <= -1.0e37f) ? 0.0f : fast::exp(v[t] - mnew);
        sl += pt;
        Pm[r * LDS + c0 + t] = bfloat16_t(pt);
      }
      for (int off = 1; off < LPR; off <<= 1) sl += simd_shuffle_xor(sl, ushort(off));
      if ((int(lane) % LPR) == 0) {
        mrow[r] = mnew;
        lrow[r] = lrow[r] * corr + sl;
        crow[r] = corr;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- rescale O (kElemRows == 1 -> one scalar per thread)
    {
      const float c = crow[8 * wm + sm];
      for (int f = 0; f < NFRAG; ++f) {
        thread auto& e = O[f].thread_elements();
        e[0] *= c; e[1] *= c;
      }
    }

    // ---- O += P . V.  V is the SAME staged tile as K (absorbed MLA: k is v), so the
    //      gathered rows are read once for both matmuls.
    {
      const threadgroup bfloat16_t* pp = Pm + (8 * wm) * LDS;
      const threadgroup bfloat16_t* vp = Kt + wn * DCOLS;
      for (int kk = 0; kk < BK; kk += 8) {
        simdgroup_bfloat8x8 A;
        simdgroup_load(A, pp + kk, LDS);
        for (int f = 0; f < NFRAG; ++f) {
          simdgroup_bfloat8x8 B;
          simdgroup_load(B, vp + (ulong)kk * LDK + f * 8, LDK);
          simdgroup_multiply_accumulate(O[f], A, B, O[f]);
        }
      }
    }
  }

  // ---- epilogue.  A row with no valid key has lrow == 0 -> zeros, which is exactly what
  //      _gathered_attention produces today via its `attn * row_has_keys` post-multiply.
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


def _kernel_source(D: int, BQ: int, BK: int, scale: float) -> str:
    src = _SOURCE.replace("SCALE", f"{scale!r}f")
    return (f"  constexpr int D  = {D};\n"
            f"  constexpr int BQ = {BQ};\n"
            f"  constexpr int BK = {BK};\n") + src


def _kernel(D: int, BQ: int, BK: int, scale: float):
    key = (D, BQ, BK, scale)
    k = _KERNEL_CACHE.get(key)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"mla_sparse_d{D}_q{BQ}_k{BK}",
            input_names=["q", "kv", "idx", "p"],
            output_names=["out"],
            header=_HEADER,
            source=_kernel_source(D, BQ, BK, scale),
            ensure_row_contiguous=True,
        )
        _KERNEL_CACHE[key] = k
    return k


def emit_msl(D: int = 512, BQ: int = 64, BK: int = 16, scale: float = 0.1) -> str:
    """The standalone .metal text MLX would compile, for ``xcrun metal -std=metal3.1``.

    Mirrors ``write_signature`` in mlx/backend/common/metal_kernel.cpp:52-176 (inputs in
    declaration order, then outputs, then the default attribute list) so a clean compile
    here means a clean compile inside MLX.
    """
    return (
        # MLX prepends metal::utils() to every custom kernel
        # (mlx/backend/metal/custom_kernel.cpp:53), which is where bfloat16_t comes from
        # (mlx/backend/metal/kernels/bf16.h:9).  Reproduce the minimum of it here.
        "#include <metal_stdlib>\nusing namespace metal;\ntypedef bfloat bfloat16_t;\n"
        + _HEADER
        + "\n[[kernel]] void mla_sparse_probe(\n"
        "  const device bfloat16_t* q [[buffer(0)]],\n"
        "  const device bfloat16_t* kv [[buffer(1)]],\n"
        "  const device int* idx [[buffer(2)]],\n"
        "  const device int* p [[buffer(3)]],\n"
        "  device bfloat16_t* out [[buffer(4)]],\n"
        "  uint3 thread_position_in_grid [[thread_position_in_grid]],\n"
        "  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],\n"
        "  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],\n"
        "  uint3 threads_per_threadgroup [[threads_per_threadgroup]],\n"
        "  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],\n"
        "  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {\n"
        + _kernel_source(D, BQ, BK, scale)
        + "\n}\n"
    )


def mla_sparse_flash_attention(
    q: mx.array,
    kv: mx.array,
    idx: mx.array,
    scale: float,
) -> mx.array:
    """Top-k sparse MQA attention that consumes the INDEX LIST, not a gathered tensor.

    q    [G, R, D]  bfloat16   one group per query, R rows (the folded head axis)
    kv   [Kv, D]    bfloat16   the whole latent cache, row-contiguous
    idx  [G, K]     int32      selected rows of ``kv``; < 0 or >= Kv means "no key"
    ->   [G, R, D]  bfloat16

    Not bit-exact against the eager path: fp32 online softmax with a different summation
    order and bf16 probabilities (which is also what the composite does -- mx.softmax
    writes bf16).  ``mla_sparse_reference_online`` is the same algorithm on CPU.
    """
    G, R, D = q.shape
    Kv, Dk = kv.shape
    if Dk != D:
        raise ValueError(f"kv head dim {Dk} != q head dim {D}")
    if idx.shape[0] != G:
        raise ValueError(f"idx groups {idx.shape[0]} != q groups {G}")
    if q.dtype != mx.bfloat16 or kv.dtype != mx.bfloat16:
        raise ValueError("sparse MLA attention is bfloat16-only")
    if idx.dtype != mx.int32:
        idx = idx.astype(mx.int32)
    BQ, BK = _tile()
    if D % (8 * (BK // 8)) or D % 8:
        raise ValueError(f"head dim {D} incompatible with tile {(BQ, BK)}")
    K = idx.shape[1]
    p = mx.array([R, K, Kv], dtype=mx.int32)
    n_qtiles = (R + BQ - 1) // BQ
    nthreads = (BQ // 8) * (BK // 8) * 32
    return _kernel(D, BQ, BK, scale)(
        inputs=[q, kv, idx, p],
        output_shapes=[(G, R, D)],
        output_dtypes=[mx.bfloat16],
        grid=(nthreads * n_qtiles, 1, G),
        threadgroup=(nthreads, 1, 1),
    )[0]


# --------------------------------------------------------------------------- #
# CPU references
# --------------------------------------------------------------------------- #


def mla_sparse_reference(
    q: mx.array, kv: mx.array, idx: mx.array, scale: float, p_bf16: bool = True
) -> mx.array:
    """Textbook one-pass reference: gather, score, softmax, weight.  Runs on CPU.

    ``p_bf16`` rounds the (unnormalised) probabilities to bfloat16 before the PV matmul.
    That is not an approximation choice, it is what BOTH paths already do: the fused
    kernel must, because ``simdgroup_bfloat8x8`` takes bf16 operands, and MLX's composite
    fallback does too because ``mx.softmax`` on a bf16 score tensor writes bf16.  With it
    ON this reference and ``mla_sparse_reference_online`` differ only by the online
    tiling, which is exact -- so a mismatch is a transcription bug.  With it OFF the two
    differ by ~1 bfloat16 ulp (3.9e-3 relative), which is the honest size of the
    numerical change the fused path introduces and the reason identity is not the gate.
    """
    G, R, D = q.shape
    Kv = kv.shape[0]
    valid = (idx >= 0) & (idx < Kv)
    kg = mx.take(kv, mx.clip(idx, 0, Kv - 1), axis=0).astype(mx.float32)  # [G,K,D]
    s = (q.astype(mx.float32) @ kg.transpose(0, 2, 1)) * scale            # [G,R,K]
    neg = mx.array(-3.0e38, dtype=mx.float32)
    s = mx.where(valid[:, None, :], s, neg)
    m = mx.max(s, axis=-1, keepdims=True)
    pu = mx.where(s <= -1.0e37, mx.zeros_like(s), mx.exp(s - m))
    if p_bf16:
        pu = pu.astype(mx.bfloat16).astype(mx.float32)
    den = mx.sum(pu, axis=-1, keepdims=True)
    o = (pu @ kg) / mx.where(den > 0, den, mx.ones_like(den))
    return mx.where(den > 0, o, mx.zeros_like(o)).astype(mx.bfloat16)


def mla_sparse_reference_online(
    q: mx.array,
    kv: mx.array,
    idx: mx.array,
    scale: float,
    bk: Optional[int] = None,
    p_bf16: bool = True,
) -> mx.array:
    """The KERNEL's algorithm, on CPU: BK-tiled online softmax, bf16 probabilities.

    Same tiling, same rescale recurrence, same "round P to bf16 before the PV matmul", so
    a mismatch against this is a transcription bug rather than a reassociation artifact.
    Only the fp32 accumulation ORDER inside each 8x8 mma differs from the kernel.

    NOTE ON THE 1e-3 GATE.  With ``p_bf16`` on, this and ``mla_sparse_reference`` are NOT
    equal to 1e-3 and cannot be: the online form rounds P against the RUNNING row max and
    the one-pass form against the FINAL row max, so the two round to different bf16 values
    and land ~1 bfloat16 ulp (2**-8 = 3.9e-3) apart -- below the ulp of the bf16 output
    they are both written into.  ``p_bf16=False`` makes the two forms algebraically
    identical and they then agree to ~1e-7, which is the check that catches a
    transcription bug.  Both checks are in tests/test_glm5_sparse_mla_attn.py; the fused
    path is gated on KL, never on identity.
    """
    G, R, D = q.shape
    Kv = kv.shape[0]
    K = idx.shape[1]
    BK = bk if bk is not None else _tile()[1]
    qf = q.astype(mx.float32)
    kvf = kv.astype(mx.float32)
    m = mx.full((G, R, 1), -3.0e38, dtype=mx.float32)
    l = mx.zeros((G, R, 1), dtype=mx.float32)
    o = mx.zeros((G, R, D), dtype=mx.float32)
    for k0 in range(0, K, BK):
        k1 = min(k0 + BK, K)
        sub = idx[:, k0:k1]
        ok = (sub >= 0) & (sub < Kv)
        rows = mx.take(kvf, mx.clip(sub, 0, Kv - 1), axis=0)          # [G,bk,D]
        rows = mx.where(ok[:, :, None], rows, mx.zeros_like(rows))    # staged zeros
        s = (qf @ rows.transpose(0, 2, 1)) * scale                    # [G,R,bk]
        s = mx.where(ok[:, None, :], s, mx.full(s.shape, -3.0e38, dtype=mx.float32))
        mt = mx.max(s, axis=-1, keepdims=True)
        mnew = mx.maximum(m, mt)
        corr = mx.exp(m - mnew)
        pu = mx.where(s <= -1.0e37, mx.zeros_like(s), mx.exp(s - mnew))
        if p_bf16:
            pu = pu.astype(mx.bfloat16).astype(mx.float32)             # kernel rounds here
        l = l * corr + mx.sum(pu, axis=-1, keepdims=True)
        o = o * corr + (pu @ rows)
        m = mnew
    inv = mx.where(l > 0, 1.0 / mx.where(l > 0, l, mx.ones_like(l)), mx.zeros_like(l))
    return (o * inv).astype(mx.bfloat16)


def threadgroup_bytes(D: int = 512, BQ: int = 64, BK: int = 16) -> int:
    """Threadgroup memory the kernel asks for, for the 32 KB budget check."""
    ldk, lds = D + 8, BK + 8
    return (BK * ldk * 2) + (BQ * lds * 4) + (BQ * lds * 2) + (3 * BQ * 4) + BK
