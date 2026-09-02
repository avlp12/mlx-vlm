"""Custom 4-bit affine quantized GEMV (M=1) as an ``mx.fast.metal_kernel``.

Route B of the GLM-5.3-Flash decode-bandwidth campaign: lands on STOCK mlx,
no MLX rebuild required.  Targets the routed-expert stream, which is the
largest single byte stream of a B=1 decode step for this checkpoint
(42 sparse layers x 8 experts x 25.17 M weights x 0.5625 B = 4.756 GB/token;
the same arithmetic over all 288 experts reproduces the 171.2 GB on-disk size).

Numerics are a deliberate re-implementation of MLX's ``qmv_fast_impl`` /
``qdot`` (mlx/backend/metal/kernels/quantized.h, v0.32.1):

  * fp32 accumulators throughout (MLX: ``typedef float U``)
  * affine dequant folded as ``scale * sum(q_i * x_i) + bias * sum(x_i)``
  * the nibble "pre-divide x" trick instead of shifts -- x lanes 0..3 of each
    packed uint16 are divided by 1, 16, 256, 4096 and the RAW masked bits are
    multiplied in.  Division by powers of two is exact, so this is bit-identical
    to shifting.
  * one ``simd_sum`` across the 32 lanes at the end, lane 0 stores.

The ONLY intended difference is the access pattern: ``packs_per_thread``
(bytes read per thread per k-step) is a tunable rather than MLX's hard-wired 2
(= 8 bytes).  With ``packs_per_thread == 2, rows_per_simd == 4,
simdgroups == 2`` this kernel is the numerical twin of MLX's and is used as the
correctness control.

Identity class: any configuration with ``packs_per_thread != 2`` regroups the
fp32 partial sums, so it is NOT bit-identical to MLX by construction -- it is
the same mathematical expression with a different, equally valid fp32
summation order.
"""

from __future__ import annotations

import functools
from typing import Optional

import mlx.core as mx

__all__ = ["qmv4", "gather_qmv4", "build_kernel", "supported"]

# --------------------------------------------------------------------------
# Metal source generation.  Constants are baked in per configuration so that
# every loop is a compile-time trip count and fully unrollable.
# --------------------------------------------------------------------------

_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
"""


def _vector_load(pt: int) -> str:
    """Emit code loading ``pt`` uint32 words from ``wrow`` into ``wv[0..pt-1]``."""
    if pt == 1:
        return "        wv[0] = wrow[0];\n"
    if pt == 2:
        return (
            "        { uint2 t = *((const device uint2*)wrow);\n"
            "          wv[0] = t.x; wv[1] = t.y; }\n"
        )
    if pt == 4:
        return (
            "        { uint4 t = *((const device uint4*)wrow);\n"
            "          wv[0] = t.x; wv[1] = t.y; wv[2] = t.z; wv[3] = t.w; }\n"
        )
    if pt == 8:
        return (
            "        { const device uint4* tv = (const device uint4*)wrow;\n"
            "          uint4 t0 = tv[0]; uint4 t1 = tv[1];\n"
            "          wv[0] = t0.x; wv[1] = t0.y; wv[2] = t0.z; wv[3] = t0.w;\n"
            "          wv[4] = t1.x; wv[5] = t1.y; wv[6] = t1.z; wv[7] = t1.w; }\n"
        )
    raise ValueError(f"unsupported packs_per_thread {pt}")


def _source(K: int, N: int, gs: int, pt: int, rs: int, nsg: int, loadv: bool) -> str:
    PACK = 8  # 32 / 4 bits
    VPT = PACK * pt  # values per thread per k-step
    BLOCK = VPT * 32  # values per simdgroup per k-step
    NBLK = K // BLOCK
    WROW = K // PACK  # uint32 words per weight row
    GROW = K // gs  # groups per weight row
    SPT = gs // VPT  # threads sharing one group  (>= 1)
    GPB = BLOCK // gs  # groups advanced per k-step

    if loadv:
        wload = _vector_load(pt)
        wref = "wv"
        wdecl = "        uint wv[%d];\n" % pt
        # reinterpret the register-resident packs as uint16 lanes
        pick = lambda i: (
            "(ushort)(wv[%d] & 0xffffu)" % (i // 2)
            if i % 2 == 0
            else "(ushort)(wv[%d] >> 16)" % (i // 2)
        )
    else:
        wload = ""
        wdecl = ""
        wref = None
        pick = lambda i: "wl[%d]" % i

    # inner dequant/dot over the thread's VPT values (VPT/4 uint16 lanes)
    dot_lines = []
    for i in range(VPT // 4):
        w_i = pick(i)
        dot_lines.append(
            "        {{ uint ww = (uint)({w});\n"
            "          a += xt[{a0}] * (float)(ww & 0x000fu)\n"
            "             + xt[{a1}] * (float)(ww & 0x00f0u)\n"
            "             + xt[{a2}] * (float)(ww & 0x0f00u)\n"
            "             + xt[{a3}] * (float)(ww & 0xf000u); }}\n".format(
                w=w_i, a0=4 * i, a1=4 * i + 1, a2=4 * i + 2, a3=4 * i + 3
            )
        )
    dot_body = "".join(dot_lines)

    # x load + pre-divide, matching MLX's load_vector<4bit> exactly
    xl = []
    for i in range(0, VPT, 4):
        # NOTE: the running `xsum` is deliberately computed with T-typed
        # (bf16) intermediates, because MLX's load_vector writes
        #   sum += x[i] + x[i+1] + x[i+2] + x[i+3];
        # with all four operands of type T.  Promoting that to fp32 would be
        # *more* accurate but would break bit-identity with MLX in the
        # control cell, which is the whole point of the control cell.
        xl.append(
            "      xsum += (float)(xp[{i0}] + xp[{i1}] + xp[{i2}] + xp[{i3}]);\n"
            "      xt[{i0}] = (float)xp[{i0}];\n"
            "      xt[{i1}] = (float)xp[{i1}] / 16.0f;\n"
            "      xt[{i2}] = (float)xp[{i2}] / 256.0f;\n"
            "      xt[{i3}] = (float)xp[{i3}] / 4096.0f;\n".format(
                i0=i, i1=i + 1, i2=i + 2, i3=i + 3
            )
        )
    x_body = "".join(xl)

    row_loop = (
        "    #pragma clang loop unroll(full)\n"
        "    for (int r = 0; r < {RS}; ++r) {{\n"
        "      const device uint* wrow = wp + r * {WROW};\n"
        "{WDECL}"
        "{WLOAD}"
        "{WLALIAS}"
        "      float s = (float)sp[r * {GROW}];\n"
        "      float bb = (float)bp[r * {GROW}];\n"
        "      float a = 0.0f;\n"
        "{DOT}"
        "      acc[r] += s * a + xsum * bb;\n"
        "    }}\n"
    ).format(
        RS=rs, WROW=WROW, GROW=GROW, WDECL=wdecl, WLOAD=wload, DOT=dot_body,
        WLALIAS=(
            ""
            if loadv
            else "      const device ushort* wl = (const device ushort*)wrow;\n"
        ),
    )

    src = """
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint tgy = threadgroup_position_in_grid.y;
  const uint bz  = threadgroup_position_in_grid.z;

  const uint e = eidx[bz];
  const int out_row = (int)(tgy * {ROWS_PER_TG} + simd_gid * {RS});

  const device uint* wp = w + (ulong)e * {N} * {WROW}
                            + (ulong)out_row * {WROW} + simd_lid * {PT};
  const device T* sp = scales + (ulong)e * {N} * {GROW}
                              + (ulong)out_row * {GROW} + simd_lid / {SPT};
  const device T* bp = biases + (ulong)e * {N} * {GROW}
                              + (ulong)out_row * {GROW} + simd_lid / {SPT};
  const device T* xp = x + simd_lid * {VPT};

  float xt[{VPT}];
  float acc[{RS}];
  #pragma clang loop unroll(full)
  for (int r = 0; r < {RS}; ++r) {{ acc[r] = 0.0f; }}

  for (int kb = 0; kb < {NBLK}; ++kb) {{
    float xsum = 0.0f;
{XBODY}
{ROWLOOP}
    wp += {WSTEP};
    sp += {GPB};
    bp += {GPB};
    xp += {BLOCK};
  }}

  #pragma clang loop unroll(full)
  for (int r = 0; r < {RS}; ++r) {{
    float v = simd_sum(acc[r]);
    if (simd_lid == 0) {{
      y[(ulong)bz * {N} + out_row + r] = (T)v;
    }}
  }}
""".format(
        ROWS_PER_TG=nsg * rs,
        RS=rs,
        N=N,
        WROW=WROW,
        PT=pt,
        GROW=GROW,
        SPT=SPT,
        VPT=VPT,
        NBLK=NBLK,
        XBODY=x_body,
        ROWLOOP=row_loop,
        WSTEP=BLOCK // PACK,
        GPB=GPB,
        BLOCK=BLOCK,
    )
    return src



# --------------------------------------------------------------------------
# 6-bit affine variant.  MLX packs 4 six-bit values into 3 BYTES
# (get_pack_factor<6,32>() == 4, get_bytes_per_pack<6,32>() == 3), so there is
# no natural vector load and LOADV does not apply.  Reproduces
# load_vector<...,6> (x pre-divided by 1, 64, 16, 4) and qdot<...,6>
# (quantized.h lines 87-95 and 266-281) exactly.
#
# NOTE for upstream: qdot's 6-bit branch advances `x_thread += 4 * i` and
# `w += 3 * i` INSIDE the loop, so the advances accumulate (0, +4, +8 -> 12
# instead of 8).  It is latent because qmv_fast only ever calls it with
# values_per_thread == 8 (two iterations).  Any attempt to raise
# packs_per_thread for 6-bit would hit it.  Flagged, not exploited.
# --------------------------------------------------------------------------


def _source6(K, N, gs, pt, rs, nsg):
    PACK = 4          # values per pack
    BPP = 3           # bytes per pack
    VPT = PACK * pt
    BLOCK = VPT * 32
    NBLK = K // BLOCK
    WROW_B = K * BPP // PACK      # bytes per weight row
    GROW = K // gs
    SPT = gs // VPT
    GPB = BLOCK // gs

    xl = []
    for i in range(0, VPT, 4):
        xl.append(
            "      xsum += (float)(xp[{i0}] + xp[{i1}] + xp[{i2}] + xp[{i3}]);\n"
            "      xt[{i0}] = (float)xp[{i0}];\n"
            "      xt[{i1}] = (float)xp[{i1}] / 64.0f;\n"
            "      xt[{i2}] = (float)xp[{i2}] / 16.0f;\n"
            "      xt[{i3}] = (float)xp[{i3}] / 4.0f;\n".format(
                i0=i, i1=i + 1, i2=i + 2, i3=i + 3
            )
        )
    x_body = "".join(xl)

    dot = []
    for i in range(VPT // 4):
        dot.append(
            "        {{ uint w0 = (uint)wl[{b0}]; uint w1 = (uint)wl[{b1}];\n"
            "          uint w2 = (uint)wl[{b2}];\n"
            "          a += (float)(w0 & 0x3fu) * xt[{x0}];\n"
            "          a += (float)(w0 & 0xc0u) * xt[{x1}];\n"
            "          a += (float)(w1 & 0x0fu) * (xt[{x1}] * 256.0f);\n"
            "          a += (float)(w1 & 0xf0u) * xt[{x2}];\n"
            "          a += (float)(w2 & 0x03u) * (xt[{x2}] * 256.0f);\n"
            "          a += (float)(w2 & 0xfcu) * xt[{x3}]; }}\n".format(
                b0=3 * i, b1=3 * i + 1, b2=3 * i + 2,
                x0=4 * i, x1=4 * i + 1, x2=4 * i + 2, x3=4 * i + 3,
            )
        )
    dot_body = "".join(dot)

    row_loop = (
        "    #pragma clang loop unroll(full)\n"
        "    for (int r = 0; r < {RS}; ++r) {{\n"
        "      const device uchar* wl = wp + r * {WROW_B};\n"
        "      float s = (float)sp[r * {GROW}];\n"
        "      float bb = (float)bp[r * {GROW}];\n"
        "      float a = 0.0f;\n"
        "{DOT}"
        "      acc[r] += s * a + xsum * bb;\n"
        "    }}\n"
    ).format(RS=rs, WROW_B=WROW_B, GROW=GROW, DOT=dot_body)

    return """
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint tgy = threadgroup_position_in_grid.y;
  const uint bz  = threadgroup_position_in_grid.z;

  const uint e = eidx[bz];
  const int out_row = (int)(tgy * {ROWS_PER_TG} + simd_gid * {RS});

  const device uchar* wp = ((const device uchar*)w)
        + (ulong)e * {N} * {WROW_B} + (ulong)out_row * {WROW_B}
        + simd_lid * {BPT};
  const device T* sp = scales + (ulong)e * {N} * {GROW}
                              + (ulong)out_row * {GROW} + simd_lid / {SPT};
  const device T* bp = biases + (ulong)e * {N} * {GROW}
                              + (ulong)out_row * {GROW} + simd_lid / {SPT};
  const device T* xp = x + simd_lid * {VPT};

  float xt[{VPT}];
  float acc[{RS}];
  #pragma clang loop unroll(full)
  for (int r = 0; r < {RS}; ++r) {{ acc[r] = 0.0f; }}

  for (int kb = 0; kb < {NBLK}; ++kb) {{
    float xsum = 0.0f;
{XBODY}
{ROWLOOP}
    wp += {WSTEP};
    sp += {GPB};
    bp += {GPB};
    xp += {BLOCK};
  }}

  #pragma clang loop unroll(full)
  for (int r = 0; r < {RS}; ++r) {{
    float v = simd_sum(acc[r]);
    if (simd_lid == 0) {{
      y[(ulong)bz * {N} + out_row + r] = (T)v;
    }}
  }}
""".format(
        ROWS_PER_TG=nsg * rs, RS=rs, N=N, WROW_B=WROW_B, BPT=pt * BPP,
        GROW=GROW, SPT=SPT, VPT=VPT, NBLK=NBLK, XBODY=x_body,
        ROWLOOP=row_loop, WSTEP=BLOCK * BPP // PACK, GPB=GPB, BLOCK=BLOCK,
    )

def supported(K: int, N: int, gs: int, pt: int, rs: int, nsg: int,
              bits: int = 4) -> Optional[str]:
    """Return None if the configuration is dispatchable, else the reason."""
    if bits not in (4, 6):
        return f"bits {bits} not implemented"
    PACK = 8 if bits == 4 else 4
    VPT = PACK * pt
    BLOCK = VPT * 32
    if gs % VPT != 0:
        return f"group_size {gs} not a multiple of values_per_thread {VPT}"
    if K % BLOCK != 0:
        return f"K {K} not a multiple of block {BLOCK}"
    if N % (nsg * rs) != 0:
        return f"N {N} not a multiple of rows_per_threadgroup {nsg * rs}"
    if nsg * 32 > 1024:
        return "threadgroup too large"
    return None


@functools.lru_cache(maxsize=512)
def build_kernel(K, N, gs=64, pt=4, rs=4, nsg=2, loadv=True, bits=4):
    why = supported(K, N, gs, pt, rs, nsg, bits)
    if why is not None:
        raise ValueError(f"unsupported qmv config: {why}")
    if bits == 6:
        loadv = False
        src = _source6(K, N, gs, pt, rs, nsg)
    else:
        src = _source(K, N, gs, pt, rs, nsg, bool(loadv))
    name = f"qmvB_b{bits}_k{K}_n{N}_g{gs}_p{pt}_r{rs}_s{nsg}_v{int(loadv)}"
    return mx.fast.metal_kernel(
        name=name,
        input_names=["w", "scales", "biases", "x", "eidx"],
        output_names=["y"],
        header=_HEADER,
        source=src,
        ensure_row_contiguous=True,
    )


def _run(w, scales, biases, x, eidx, K, N, B, gs, pt, rs, nsg, loadv, dtype,
         bits=4):
    kern = build_kernel(K, N, gs, pt, rs, nsg, loadv, bits)
    ntg_y = N // (nsg * rs)
    return kern(
        inputs=[w, scales, biases, x, eidx],
        template=[("T", dtype)],
        grid=(32, nsg * ntg_y, B),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, N)],
        output_dtypes=[dtype],
    )[0]


def qmv4(x, w, scales, biases, *, group_size=64, pt=4, rs=4, nsg=2,
         loadv=True, bits=4):
    """y[N] = W[N,K] @ x[K] for a single 2-D affine quantized matrix (M=1)."""
    K = x.shape[-1]
    N = w.shape[-2]
    zero = mx.zeros((1,), dtype=mx.uint32)
    return _run(
        w, scales, biases, x.reshape(-1), zero, K, N, 1,
        group_size, pt, rs, nsg, loadv, x.dtype, bits,
    )


def gather_qmv4(
    x, w, scales, biases, rhs_indices, *,
    group_size=64, pt=4, rs=4, nsg=2, loadv=True, bits=4,
):
    """y[B,N] = W[idx_b][N,K] @ x[K], one dispatch over all B gathered rows.

    ``x`` is a single token vector shared by every gathered expert -- exactly
    the B=1 decode MoE case (lhs_indices are all zero there).
    """
    K = x.shape[-1]
    N = w.shape[-2]
    idx = rhs_indices.reshape(-1).astype(mx.uint32)
    B = idx.shape[0]
    return _run(
        w, scales, biases, x.reshape(-1), idx, K, N, B,
        group_size, pt, rs, nsg, loadv, x.dtype, bits,
    )
