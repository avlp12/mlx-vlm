"""Fused KDA for the PREFILL chunk (S up to the prefill chunk width, B==1 first).

WHAT THE EAGER PREFILL PATH ACTUALLY IS, because the name "chunked gated delta
rule" is misleading here.  ``models/gated_delta.py`` contains no chunk-parallel
(matmul) formulation: at S>1 ``gated_delta_update`` dispatches
``gated_delta_kernel``, a *sequential per-token scan* in Metal that keeps the
state in registers, one thread per (key lane, value row).  "Chunked at 8192"
refers to the prompt being fed to the model in 8192-token blocks, not to an
intra-chunk matmul.  So the recurrence is already one dispatch per layer per
chunk; what runs as many small MLX ops around it is the GLUE:

    concat -> conv1d -> silu -> split/reshape -> f_b_proj -> 2x fp32 l2norm
    ... recurrence ...
    g_b_proj -> hand-rolled gated RMSNorm (Glm5NextRMSNormGated, ~12 dispatches)

At S=8192 that glue is NOT launch-bound (33 launches x 15 us = 0.5 ms against a
measured ~59 ms per layer): it is memory traffic.  Every arrow above
materialises a ``[1, S, H*D]`` bf16 tensor -- 128 MB per tensor at S=8192,
H*D=8192 -- and there are roughly a dozen of them.  Fusing therefore buys
*bandwidth*, not dispatch count, which is the opposite of what the S<=8 verify
block (``fused_kda_verify_block``) buys.  The two kernels look alike and are
paid for by different physics; do not transfer one's measurement to the other.

THE GEOMETRY PROBLEM, and why this is not just "the block kernel with a bigger
S".  ``fused_kda_verify_block`` runs ONE threadgroup per (batch row, head): at
B=1 that is ``H`` threadgroups of 32*TY threads.  ``gated_delta_kernel`` instead
runs one *thread* per (key lane, value row), i.e. 32*D threads per head =
``D/4`` threadgroups per head.  At H=64, D=128 that is 64 threadgroups against
2048 -- a 4x drop in resident threads and, at B=1, only 64 threadgroups for an
80-core GPU.  Widening S does not fix that; it is per-token parallelism.

So this kernel keeps the block kernel's arithmetic verbatim but restores the
recurrence partition by splitting the VALUE axis across threadgroups: ``NV``
threadgroups per (row, head), threadgroup ``nv`` owning value rows
``[nv*D/NV, (nv+1)*D/NV)``.  At NV = D/TY (4 at D=128, TY=32) every thread owns
exactly one value row and 32 key elements -- the same partition, the same lanes
and the same operand order as ``gated_delta_kernel``, hence bit-identical.

Splitting the value axis costs two things:

  * the pre-recurrence glue (conv, silu, both L2 norms, the safe gate) is over
    the KEY axis, which every threadgroup needs in full, so it is recomputed NV
    times and q/k/a are re-read NV times.  That is ~2.4x the read traffic on
    three of the ~12 tensors the eager path materialises, against removing the
    other nine entirely;

  * the gated RMSNorm reduces over the VALUE axis, which is now split, so it
    cannot stay in the same kernel.  It moves to a second, fully parallel launch
    (``fused_kda_prefill_norm``) reading a ``[B, S, H, D]`` bf16 intermediate.
    That intermediate is exactly what ``gated_delta_kernel`` already wrote (in
    the input dtype, ``y[dv] = static_cast<InT>(out)``), so the split costs no
    traffic the eager path did not already pay, and the norm kernel replaces
    ~12 eager dispatches with one.

``NV == 1`` keeps the norm in-kernel (one launch per layer, the maximally fused
arm) and is retained so the two geometries can be measured against each other:
NV=1 is minimum traffic and minimum parallelism, NV=D/TY is maximum parallelism
and 2.4x the read traffic on q/k/a.  Which wins is a measurement, not an
argument, so both ship behind the same flag.

Numerics: every arithmetic line is copied from ``fused_kda._BLOCK_SOURCE``,
which is itself a transcription of the eager chain including where it rounds,
and which ``tests/test_glm5_next_fused_kda_block.py`` pins bit-identical.  The
reduction partitions (simd_sum over 32 key lanes; MLX's row_reduce 4-reads
partition in the L2 and RMS norms) are unchanged at every NV and TY.  The claim
under test in ``test_glm5_next_fused_kda_prefill.py`` is therefore bit-exactness,
not a tolerance.

Not covered here: speculative capture.  The prefill kernel deliberately does not
emit the ``gdn_sink`` q/k/v tensors -- at S=8192 those are 3 x 128 MB per layer
of pure waste -- so a call that carries a sink falls back to the eager path.
"""

import logging
from typing import Optional, Tuple

import mlx.core as mx

from .fused_kda import _HEADER

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# The scan kernel.
#
# `fuse_norm` picks between the two geometries described above.  They are two
# pipelines rather than one branch because mx.fast.metal_kernel derives the
# function signature from input_names/output_names, and the split variant does
# not take `gate` / `o_w` at all (they belong to the second launch).
# --------------------------------------------------------------------------- #
def _make_scan_source(fuse_norm: bool) -> str:
    # Phase 1 writes the recurrence output either to threadgroup memory (so the
    # in-kernel RMSNorm can reduce over it) or straight to the [B, S, H, D]
    # intermediate.  Either way the value stored is `static_cast<T>(o)`, which is
    # what gated_delta_kernel itself stores.
    if fuse_norm:
        y_store = "sy[dv] = float(static_cast<T>(o));"
        gate_load = "      sgate[d] = float(gate[tok_off + h * (uint)D + d]);\n"
        sy_decl = "  threadgroup float sy[D];\n  threadgroup float sgate[D];\n"
    else:
        y_store = "y[tok_off + h * (uint)D + dv] = static_cast<T>(o);"
        gate_load = ""
        sy_decl = ""

    norm_phase = (
        """
    // ------------------------------------------------------------- phase 2
    // Gated RMSNorm over the value axis, fp32, then back to T.  Only reachable
    // at NV == 1: at NV > 1 this threadgroup holds a slice of the value axis and
    // cannot reduce over it, so the norm is a second launch.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index_in_threadgroup == 0u) {
      float po = 0.0f;
      for (int blk = 0; blk < RBLK; ++blk) {
        uint base = (uint)(blk * 128) + 4u * lane;
        for (int i = 0; i < 4; ++i) po = sq_acc(po, sy[base + i]);
      }
      uint base = (uint)(RBLK * 128) + 4u * lane;
      if (4u * lane + 4u <= (uint)REXTRA) {
        for (int i = 0; i < 4; ++i) po = sq_acc(po, sy[base + i]);
      } else {
        for (int i = 0; 4u * lane + (uint)i < (uint)REXTRA; ++i) {
          po = sq_acc(po, sy[base + i]);
        }
      }
      po = simd_sum(po);
      if (lane == 0u) {
        shr[0] = metal::precise::rsqrt(po / (float)D + norm_eps);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      float rn = shr[0];
      for (uint d = tid; d < (uint)D; d += NT) {
        float x = sy[d] * rn;
        x = float(o_w[d]) * x;
        x = x * mlx_sigmoid_precise<float>(sgate[d]);
        y[tok_off + h * (uint)D + d] = static_cast<T>(x);
      }
    }
"""
        if fuse_norm
        else ""
    )

    return f"""
  // grid.z is B * H * NV threadgroups.  z is laid out (b, h) major, nv minor so
  // the NV threadgroups that re-read the same q/k/a rows are adjacent and hit
  // the same cache lines.
  const uint znv  = threadgroup_position_in_grid.z;
  const uint bh   = znv / (uint)NV;
  const uint nv   = znv - bh * (uint)NV;
  const uint b    = bh / (uint)H;
  const uint h    = bh - b * (uint)H;
  const uint lane = thread_position_in_threadgroup.x;
  const uint ty   = thread_position_in_threadgroup.y;
  const uint tid  = thread_index_in_threadgroup;
  const uint S    = (uint)nsteps;

  constexpr int NT     = 32 * TY;
  constexpr int RBLK   = D / 128;
  constexpr int REXTRA = D - RBLK * 128;
  constexpr int NDK    = D / 32;      // key elements per lane
  constexpr int DVPT   = D / NV;      // value rows per threadgroup
  constexpr int NDV    = DVPT / TY;   // value rows per thread
  constexpr uint QKVD  = (uint)(H * D);
  constexpr uint CDIM  = 3u * QKVD;
  constexpr uint KM1   = (uint)(K - 1);
  const uint dv_off    = nv * (uint)DVPT;
  const size_t cs_off  = (size_t)b * KM1 * CDIM;

  threadgroup float sq[D];
  threadgroup float sk[D];
  threadgroup float sv[D];
  threadgroup float sg[D];
{sy_decl}  threadgroup float shr[3];
  // Rolling pre-conv window: the last K-1 inputs oldest-to-newest, circular.
  threadgroup T twin[(K - 1) * 3 * D];

  // State in registers for the WHOLE chunk: loaded once here, stored once after
  // the loop.  This threadgroup owns value rows [dv_off, dv_off + DVPT) of the
  // [B, H, D, D] state, a contiguous slab, so the load and store are coalesced.
  device const ST* si = state_in  + (size_t)bh * D * D;
  device ST*       so = state_out + (size_t)bh * D * D;
  float st[NDV][NDK];
  for (int j = 0; j < NDV; ++j) {{
    uint dv = dv_off + ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i) {{
      st[j][i] = float(si[(size_t)dv * D + NDK * lane + i]);
    }}
  }}

  // Seed the window from the cache, oldest at slot 0 -- the order the eager path
  // lays out concatenate([conv_state, mixed], axis=1).
  for (uint idx = tid; idx < KM1 * 3u * (uint)D; idx += NT) {{
    uint slot = idx / (3u * (uint)D);
    uint r    = idx - slot * 3u * (uint)D;
    uint part = r / (uint)D;
    uint d    = r - part * (uint)D;
    uint c    = part * QKVD + h * (uint)D + d;
    twin[slot * 3u * (uint)D + r] = conv_state[cs_off + (size_t)slot * CDIM + c];
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float a_exp = metal::precise::exp(A_log[h]);

  for (uint t = 0u; t < S; ++t) {{
    const size_t tok_off = ((size_t)b * S + t) * QKVD;   // [B, S, H*D]
    // ------------------------------------------------------------ phase 0a
    // Depthwise causal conv over [window ; x_t] then silu.  For token t the K
    // taps are conv_input[t .. t+K-1]: slots ((t + j) mod K-1) oldest-first,
    // then x_t itself.
    for (uint idx = tid; idx < 3u * (uint)D; idx += NT) {{
      uint part = idx / (uint)D;
      uint d    = idx - part * (uint)D;
      uint c    = part * QKVD + h * (uint)D + d;
      device const T* wc = conv_w + (size_t)c * K;
      float acc = 0.0f;
      for (uint j = 0; j + 1 < (uint)K; ++j) {{
        uint slot = (t + j) % KM1;
        acc += float(twin[slot * 3u * (uint)D + idx]) * float(wc[j]);
      }}
      // The eager path zeroes the PRE-conv input of a masked (row, token), so
      // the zero lands here -- before both the conv and the window write.
      T xnew = valid[(size_t)b * S + t]
                 ? ((part == 0u) ? mq[tok_off + h * (uint)D + d]
                  : ((part == 1u) ? mk[tok_off + h * (uint)D + d]
                                  : mv[tok_off + h * (uint)D + d]))
                 : static_cast<T>(0);
      acc += float(xnew) * float(wc[K - 1]);

      T xb  = static_cast<T>(acc);      // mx.conv1d writes its output in T
      T sig = mlx_sigmoid_fast(xb);     // nn.silu = x * mx.sigmoid(x), compiled
      T sl  = xb * sig;
      if (part == 0u)      sq[d] = float(sl);
      else if (part == 1u) sk[d] = float(sl);
      else                 sv[d] = float(sl);
    }}
    // Deferred past this barrier: until every thread has read its taps, the slot
    // about to be overwritten is still live.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < 3u * (uint)D; idx += NT) {{
      uint part = idx / (uint)D;
      uint d    = idx - part * (uint)D;
      T xnew = valid[(size_t)b * S + t]
                 ? ((part == 0u) ? mq[tok_off + h * (uint)D + d]
                  : ((part == 1u) ? mk[tok_off + h * (uint)D + d]
                                  : mv[tok_off + h * (uint)D + d]))
                 : static_cast<T>(0);
      twin[(t % KM1) * 3u * (uint)D + idx] = xnew;
    }}

    // ------------------------------------------------------------ phase 0b
    for (uint d = tid; d < (uint)D; d += NT) {{
      float av = float(a[tok_off + h * (uint)D + d]) + dt_bias[h * (uint)D + d];
      sg[d]    = metal::precise::exp(lower_bound * mlx_sigmoid_fast<float>(a_exp * av));
{gate_load}    }}
    if (tid == 0u) {{
      shr[2] = float(mlx_sigmoid_precise(bvec[((size_t)b * S + t) * (uint)H + h]));
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ------------------------------------------------------------ phase 0c
    // q = l2norm(q) * D^-0.5 ; k = l2norm(k), both rounded back to T.  Same
    // partition and accumulation order as MLX's row_reduce_simple.
    if (simdgroup_index_in_threadgroup == 0u) {{
      float pq = 0.0f, pk = 0.0f;
      for (int blk = 0; blk < RBLK; ++blk) {{
        uint base = (uint)(blk * 128) + 4u * lane;
        for (int i = 0; i < 4; ++i) {{
          pq = sq_acc(pq, sq[base + i]);
          pk = sq_acc(pk, sk[base + i]);
        }}
      }}
      uint base = (uint)(RBLK * 128) + 4u * lane;
      if (4u * lane + 4u <= (uint)REXTRA) {{
        for (int i = 0; i < 4; ++i) {{
          pq = sq_acc(pq, sq[base + i]);
          pk = sq_acc(pk, sk[base + i]);
        }}
      }} else {{
        for (int i = 0; 4u * lane + (uint)i < (uint)REXTRA; ++i) {{
          pq = sq_acc(pq, sq[base + i]);
          pk = sq_acc(pk, sk[base + i]);
        }}
      }}
      pq = simd_sum(pq);
      pk = simd_sum(pk);
      if (lane == 0u) {{
        shr[0] = metal::precise::rsqrt(pq + 1.0e-6f);
        shr[1] = metal::precise::rsqrt(pk + 1.0e-6f);
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{
      float rq = shr[0], rk = shr[1];
      for (uint d = tid; d < (uint)D; d += NT) {{
        sq[d] = float(static_cast<T>((sq[d] * rq) * qscale));
        sk[d] = float(static_cast<T>(sk[d] * rk));
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ------------------------------------------------------------- phase 1
    // Gated delta rule.  Lane `lane` owns key elements [NDK*lane, NDK*lane+NDK)
    // and this threadgroup owns value rows [dv_off, dv_off+DVPT) -- at
    // NV = D/TY that is exactly gated_delta_kernel's one-thread-per-(lane, row)
    // partition, with the same operand order, hence the same bits.
    {{
      float beta = shr[2];
      for (int j = 0; j < NDV; ++j) {{
        uint dv = dv_off + ty + (uint)TY * (uint)j;
        float kv = 0.0f;
        for (int i = 0; i < NDK; ++i) {{
          uint s = NDK * lane + i;
          st[j][i] = st[j][i] * sg[s];
          kv += st[j][i] * sk[s];
        }}
        kv = simd_sum(kv);
        float delta = (sv[dv] - kv) * beta;
        float o = 0.0f;
        for (int i = 0; i < NDK; ++i) {{
          uint s = NDK * lane + i;
          st[j][i] = st[j][i] + sk[s] * delta;
          o += st[j][i] * sq[s];
        }}
        o = simd_sum(o);
        if (thread_index_in_simdgroup == 0u) {{
          {y_store}
        }}
      }}
    }}
{norm_phase}
    // sq/sk/sv are about to be overwritten by the next token's conv.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}

  // ---------------------------------------------------------------- epilogue
  // One state store for the whole chunk.  The cache window is the last K-1
  // pre-conv rows, oldest-first, which the circular buffer already holds at
  // slots ((S + j) mod K-1) -- correct for S >= K-1 and S < K-1 alike, because
  // the FIFO never distinguished cached rows from new ones.  Only nv == 0
  // writes it: every threadgroup of a head computed the same window.
  for (int j = 0; j < NDV; ++j) {{
    uint dv = dv_off + ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i) {{
      so[(size_t)dv * D + NDK * lane + i] = static_cast<ST>(st[j][i]);
    }}
  }}
  if (nv == 0u) {{
    for (uint idx = tid; idx < KM1 * 3u * (uint)D; idx += NT) {{
      uint slot = idx / (3u * (uint)D);
      uint r    = idx - slot * 3u * (uint)D;
      uint part = r / (uint)D;
      uint d    = r - part * (uint)D;
      uint c    = part * QKVD + h * (uint)D + d;
      conv_state_out[cs_off + (size_t)slot * CDIM + c] =
          twin[((S + slot) % KM1) * 3u * (uint)D + r];
    }}
  }}
"""


# The gated RMSNorm as one launch, for the NV > 1 geometry.  One simdgroup per
# (b, s, h) row, TY rows per threadgroup; the reduction is MLX's row_reduce
# partition (4 contiguous reads per lane, then simd_sum) so it matches
# Glm5NextRMSNormGated's `(x * x).mean(-1)` bit for bit.
_NORM_SOURCE = """
  const uint row = threadgroup_position_in_grid.x * (uint)TY
                 + thread_position_in_threadgroup.y;   // (b*S + s)*H + h
  if (row >= (uint)nrows) return;
  const uint lane = thread_position_in_threadgroup.x;
  constexpr int RBLK   = D / 128;
  constexpr int REXTRA = D - RBLK * 128;
  const size_t off = (size_t)row * D;

  float po = 0.0f;
  for (int blk = 0; blk < RBLK; ++blk) {
    uint base = (uint)(blk * 128) + 4u * lane;
    for (int i = 0; i < 4; ++i) po = sq_acc(po, float(yin[off + base + i]));
  }
  uint base = (uint)(RBLK * 128) + 4u * lane;
  if (4u * lane + 4u <= (uint)REXTRA) {
    for (int i = 0; i < 4; ++i) po = sq_acc(po, float(yin[off + base + i]));
  } else {
    for (int i = 0; 4u * lane + (uint)i < (uint)REXTRA; ++i) {
      po = sq_acc(po, float(yin[off + base + i]));
    }
  }
  po = simd_sum(po);
  float rn = metal::precise::rsqrt(po / (float)D + norm_eps);
  for (uint d = lane; d < (uint)D; d += 32u) {
    float x = float(yin[off + d]) * rn;
    x = float(o_w[d]) * x;
    x = x * mlx_sigmoid_precise<float>(float(gate[off + d]));
    y[off + d] = static_cast<T>(x);
  }
"""

_SCAN_INPUTS_FUSED = [
    "mq", "mk", "mv", "conv_state", "conv_w", "a", "bvec", "A_log", "dt_bias",
    "state_in", "gate", "o_w", "lower_bound", "qscale", "norm_eps", "valid",
    "nsteps",
]
_SCAN_INPUTS_SPLIT = [
    "mq", "mk", "mv", "conv_state", "conv_w", "a", "bvec", "A_log", "dt_bias",
    "state_in", "lower_bound", "qscale", "valid", "nsteps",
]
_SCAN_OUTPUTS = ["y", "state_out", "conv_state_out"]

_KERNELS = {}
_KERNEL_TRIED = False


def _kernel(kind: str):
    """``kind`` in {"fused", "split", "norm"}; ``None`` if Metal is unavailable."""
    global _KERNEL_TRIED
    if not _KERNEL_TRIED:
        _KERNEL_TRIED = True
        if mx.metal.is_available():
            _KERNELS["fused"] = mx.fast.metal_kernel(
                name="glm5_kda_prefill_scan_fused",
                input_names=_SCAN_INPUTS_FUSED,
                output_names=_SCAN_OUTPUTS,
                header=_HEADER,
                source=_make_scan_source(True),
            )
            _KERNELS["split"] = mx.fast.metal_kernel(
                name="glm5_kda_prefill_scan_split",
                input_names=_SCAN_INPUTS_SPLIT,
                output_names=_SCAN_OUTPUTS,
                header=_HEADER,
                source=_make_scan_source(False),
            )
            _KERNELS["norm"] = mx.fast.metal_kernel(
                name="glm5_kda_prefill_norm",
                input_names=["yin", "gate", "o_w", "norm_eps", "nrows"],
                output_names=["y"],
                header=_HEADER,
                source=_NORM_SOURCE,
            )
    return _KERNELS.get(kind)


_ONES_MASK = {}


def _all_valid(n: int) -> mx.array:
    m = _ONES_MASK.get(n)
    if m is None:
        m = mx.ones((n,), dtype=mx.bool_)
        mx.eval(m)
        _ONES_MASK[n] = m
    return m


def prefill_geometry(head_dim: int, nv: Optional[int] = None, ty: int = 32):
    """(nv, ty) for a head dim, defaulting to gated_delta_kernel's partition.

    The default puts one value row on each thread (NDV == 1), which is exactly
    what ``gated_delta_kernel`` does, so the fused scan is resident with the same
    thread count as the recurrence it replaces instead of 1/NV of it.  Falls back
    to fewer value threads when the head dim does not divide.
    """
    d = head_dim
    if ty > 32:
        ty = 32
    while ty > 1 and d % ty:
        ty //= 2
    if nv is None:
        nv = max(1, d // ty)
    while nv > 1 and (d % nv or (d // nv) % ty):
        nv -= 1
    return nv, ty


def fused_kda_prefill_supported(*, num_heads: int, head_dim: int,
                                conv_kernel_size: int,
                                lower_bound: Optional[float]) -> bool:
    """Config-level preconditions.  Mirrors ``fused_kda_supported``."""
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    if lower_bound is None:
        return False          # only the "safe gate" branch is transcribed
    if conv_kernel_size < 2:
        return False
    if head_dim % 32 or head_dim <= 0 or num_heads <= 0:
        return False
    return _kernel("fused") is not None


def fused_kda_prefill_scan(
    q_in: mx.array,
    k_in: mx.array,
    v_in: mx.array,
    conv_state: mx.array,
    conv_w: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    gate: Optional[mx.array],
    o_weight: Optional[mx.array],
    *,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    lower_bound: float,
    norm_eps: float,
    mask: Optional[mx.array] = None,
    nv: int = 1,
    ty: int = 32,
) -> Tuple[mx.array, ...]:
    """The whole S-token KDA chunk in one launch (plus one for the norm at nv>1).

    Shapes match ``fused_kda_verify_block``: ``q_in/k_in/v_in/a`` and (when fused)
    ``gate`` are ``[B, S, H*D]``, ``b`` is ``[B, S, H]``, ``state`` is
    ``[B, H, D, D]``, ``conv_state`` is ``[B, K-1, 3*H*D]``, ``mask`` is an
    optional ``[B, S]`` bool applied to the pre-conv input.

    Returns ``(y, state_out, conv_state_out)``.  At ``nv == 1`` ``y`` is
    ``[B, S, H*D]`` and final -- exactly what o_proj consumes.  At ``nv > 1``
    ``y`` is the ``[B, S, H, D]`` pre-norm recurrence output and the caller must
    run ``fused_kda_prefill_norm``; ``fused_kda_prefill`` does both.

    ``S`` is a runtime scalar, never a template parameter: prefill chunks are
    ragged at the tail of a prompt and a templated S would compile a fresh
    pipeline per chunk width.
    """
    H, D, K = num_heads, head_dim, conv_kernel_size
    B, S = q_in.shape[0], q_in.shape[1]
    dt = q_in.dtype
    valid = _all_valid(B * S) if mask is None else mask.reshape(B * S)
    fused = nv == 1
    kernel = _kernel("fused" if fused else "split")
    template = [
        ("T", dt), ("ST", state.dtype), ("H", H), ("D", D), ("K", K),
        ("TY", ty), ("NV", nv),
    ]
    head = [q_in, k_in, v_in, conv_state, conv_w, a, b, A_log, dt_bias, state]
    if fused:
        inputs = head + [
            gate, o_weight, float(lower_bound), float(D**-0.5), float(norm_eps),
            valid, int(S),
        ]
        y_shape = (B, S, H * D)
    else:
        inputs = head + [
            float(lower_bound), float(D**-0.5), valid, int(S),
        ]
        y_shape = (B, S, H, D)
    return kernel(
        inputs=inputs,
        template=template,
        grid=(32, ty, B * H * nv),
        threadgroup=(32, ty, 1),
        output_shapes=[y_shape, state.shape, conv_state.shape],
        output_dtypes=[dt, state.dtype, dt],
    )


def fused_kda_prefill_norm(
    y_pre: mx.array, gate: mx.array, o_weight: mx.array, *, norm_eps: float,
    head_dim: int, ty: int = 8,
) -> mx.array:
    """Gated RMSNorm over the value axis of ``[B, S, H, D]``, one launch."""
    B, S, H, D = y_pre.shape
    rows = B * S * H
    kernel = _kernel("norm")
    (y,) = kernel(
        inputs=[y_pre, gate, o_weight, float(norm_eps), int(rows)],
        template=[("T", y_pre.dtype), ("D", head_dim), ("TY", ty)],
        grid=(32 * ((rows + ty - 1) // ty), ty, 1),
        threadgroup=(32, ty, 1),
        output_shapes=[(B, S, H * D)],
        output_dtypes=[y_pre.dtype],
    )
    return y


def fused_kda_prefill(
    q_in: mx.array,
    k_in: mx.array,
    v_in: mx.array,
    conv_state: mx.array,
    conv_w: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    gate: mx.array,
    o_weight: mx.array,
    *,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    lower_bound: float,
    norm_eps: float,
    mask: Optional[mx.array] = None,
    nv: int = 1,
    ty: int = 32,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Scan (+ norm at nv>1).  Returns ``(y [B,S,H*D], state, conv_state)``."""
    y, state_out, conv_out = fused_kda_prefill_scan(
        q_in, k_in, v_in, conv_state, conv_w, a, b, A_log, dt_bias, state,
        gate if nv == 1 else None,
        o_weight if nv == 1 else None,
        num_heads=num_heads, head_dim=head_dim,
        conv_kernel_size=conv_kernel_size, lower_bound=lower_bound,
        norm_eps=norm_eps, mask=mask, nv=nv, ty=ty,
    )
    if nv != 1:
        B, S = q_in.shape[0], q_in.shape[1]
        y = fused_kda_prefill_norm(
            y, gate.reshape(B, S, num_heads, head_dim), o_weight,
            norm_eps=norm_eps, head_dim=head_dim,
        )
    return y, state_out, conv_out


# (dtype, state dtype, H, D, K, nv, ty) -> usable (nv, ty), or None if the device
# will not run this pipeline at any admissible threadgroup size.  Same reason as
# fused_kda.fused_kda_probe: maxTotalThreadsPerThreadgroup is a per-pipeline
# limit set by register pressure, reported as a ValueError at eval time far from
# the call site.  Halving TY is partition-preserving (every reduction keeps the
# same lanes and operand order), so a degraded launch is still bit-identical --
# but NV must be lowered with it to keep NDV integral, which is why the probe
# returns a pair.
_PROBE_CACHE = {}


def fused_kda_prefill_probe(
    *, num_heads: int, head_dim: int, conv_kernel_size: int, dtype, state_dtype,
    nv: Optional[int] = None, ty: int = 32,
) -> Optional[Tuple[int, int]]:
    key = (dtype, state_dtype, num_heads, head_dim, conv_kernel_size, nv, ty)
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    H, D, K = num_heads, head_dim, conv_kernel_size
    result = None
    t = ty
    while t >= 1:
        cand_nv, cand_ty = prefill_geometry(D, nv, t)
        try:
            outs = fused_kda_prefill(
                mx.zeros((1, 2, H * D), dtype),
                mx.zeros((1, 2, H * D), dtype),
                mx.zeros((1, 2, H * D), dtype),
                mx.zeros((1, K - 1, 3 * H * D), dtype),
                mx.zeros((3 * H * D, K, 1), dtype),
                mx.zeros((1, 2, H * D), dtype),
                mx.zeros((1, 2, H), dtype),
                mx.zeros((H,), mx.float32),
                mx.zeros((H * D,), mx.float32),
                mx.zeros((1, H, D, D), state_dtype),
                mx.zeros((1, 2, H * D), dtype),
                mx.zeros((D,), dtype),
                num_heads=H, head_dim=D, conv_kernel_size=K,
                lower_bound=-5.0, norm_eps=1e-5, nv=cand_nv, ty=cand_ty,
            )
            mx.eval(outs)
        except ValueError as exc:
            if "threads per threadgroup" not in str(exc):
                raise
            t //= 2
            continue
        except RuntimeError as exc:
            logger.info("glm5_next fused KDA prefill unavailable: %s", exc)
            break
        result = (cand_nv, cand_ty)
        break
    _PROBE_CACHE[key] = result
    if result is None:
        logger.info(
            "glm5_next fused KDA prefill declined: this device's threadgroup "
            "limit is below the kernel's requirement at every supported size"
        )
    return result
