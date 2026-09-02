"""Fused single-token decode step for the GLM-5-Next KDA (Kimi Delta Attention) core.

At S=1 the post-projection half of ``Glm5NextLinearAttention.__call__`` is a
long tail of tiny elementwise / small-reduction kernels: the causal conv1d window
update, silu, two L2 norms, the forget-gate softplus-free "safe gate", the sigmoid
beta, the gated delta-rule state update and finally the gated RMSNorm.  Each of
those is a separate GPU dispatch, and at 34 KDA layers per decode step the launch
overhead dominates the (tiny) arithmetic.

This module folds *all* of them into ONE ``mx.fast.metal_kernel`` launch per layer.
One threadgroup handles one head, so the two cross-``head_dim`` reductions that the
chain needs (the L2 norms over the key axis and the RMSNorm over the value axis)
both stay inside threadgroup memory.  The recurrent state ``[head_dim, head_dim]``
lives in device memory and is streamed through registers exactly once.

The arithmetic is a line-by-line transcription of the eager path, including where
it rounds to the input dtype: ``mx.conv1d`` writes bfloat16, ``nn.silu`` rounds
twice (sigmoid then product), ``gated_delta_kernel`` writes its output ``y`` in the
input dtype before the gated RMSNorm reads it back as float32, and ``beta`` is a
bfloat16 sigmoid.  State accumulation stays float32, matching the eager kernel.

A ``capture=True`` variant additionally emits the tensors ``gdn_sink`` carries for
speculative rollback (post-conv q/k/v and the pre-conv window), straight out of
threadgroup memory, so a drafter-attached single-token step keeps the fusion.

Not a drop-in for prefill: this is decode-only (S=1, no SSM mask).  Batched
decode *is* covered -- ``grid.z`` becomes ``B * H`` threadgroups, one per (batch
row, head) -- and the per-row validity mask that batched decode always carries is
applied in the one place the eager path applies it, at the pre-conv input.
``Glm5NextLinearAttention`` falls back to the eager path whenever any of those
preconditions does not hold, and caps the batch it will serve at a width parity
has actually been run to (``_FUSED_KDA_MAX_BATCH``).
"""

import logging
from typing import Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

# MLX's own elementwise ops, transcribed so the fused kernel rounds identically:
#   mlx/backend/metal/kernels/unary_ops.h -> Sigmoid / Exp / Rsqrt
_HEADER = """
// MLX's Sigmoid (mlx/backend/metal/kernels/unary_ops.h), with two subtleties
// that each cost exactness if ignored:
//
//  * Instantiate on the SAME type the eager op used.  Metal's native `bfloat`
//    rounds after every arithmetic step, so evaluating in float32 and rounding
//    once at the end disagrees with mx.sigmoid(bfloat16) on ~15% of elements.
//
//  * Pick the right exp.  MLX's Sigmoid is written with `metal::exp`, which
//    resolves to the precise implementation inside MLX's *precompiled* Metal
//    library but to the fast approximation inside a JIT'd kernel (mx.compile
//    and mx.fast.metal_kernel share that JIT).  In the KDA chain both appear:
//    `nn.silu` and `compute_g_safe` are mx.compile'd (-> _fast), while the beta
//    sigmoid and the output-gate sigmoid are plain ops (-> _precise).  MLX's
//    Exp op spells `metal::precise::exp` in source, so mx.exp is precise in
//    both.  Mixing them up is a 1-ulp disagreement on ~0.1-50% of elements.
template <typename U>
inline U mlx_sigmoid_precise(U x) {
  U e = static_cast<U>(metal::precise::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}

template <typename U>
inline U mlx_sigmoid_fast(U x) {
  U e = static_cast<U>(metal::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}

// `(x * x).sum(-1)` in MLX materialises the squares before reducing, so the
// multiply rounds before the add.  Writing `acc += v * v` here would let the
// compiler contract it into an fma and silently change the last bit of every
// L2 / RMS norm, so contraction is disabled for this one helper.
#pragma clang fp contract(off)
inline float sq_acc(float acc, float v) {
  return v * v + acc;
}
#pragma clang fp contract(on)
"""

_SOURCE = """
  // grid.z is B * H threadgroups: one per (batch row, head).  B never appears
  // in this source -- only in grid.z and the buffer extents -- so the compiled
  // pipeline (and therefore its threadgroup limit) is identical at every B.
  const uint bh   = threadgroup_position_in_grid.z;
  const uint b    = bh / (uint)H;
  const uint h    = bh - b * (uint)H;
  const uint lane = thread_position_in_threadgroup.x;
  const uint ty   = thread_position_in_threadgroup.y;
  const uint tid  = thread_index_in_threadgroup;

  constexpr int NT   = 32 * TY;      // threads per threadgroup
  constexpr int RBLK = D / 128;      // MLX row-reduce blocks (32 lanes x 4 reads)
  constexpr int REXTRA = D - RBLK * 128;
  constexpr int NDK  = D / 32;       // key-dim elements held per thread
  constexpr int NDV  = D / TY;       // value-dim rows walked per thread
  constexpr uint QKVD = (uint)(H * D);
  constexpr uint CDIM = 3u * QKVD;   // conv1d channel count
  const uint qkv_off  = b * QKVD;                 // [B, 1, H*D] rows
  const size_t cs_off = (size_t)b * (K - 1) * CDIM;  // [B, K-1, 3*H*D]

  threadgroup float sq[D];
  threadgroup float sk[D];
  threadgroup float sv[D];
  threadgroup float sg[D];
  threadgroup float sgate[D];
  threadgroup float sy[D];
  threadgroup float shr[3];          // 0: rsqrt(q), 1: rsqrt(k), 2: beta

  // Issue the recurrent-state read first: it is 4 MB per layer and by far the
  // longest-latency operation here, so it overlaps the conv / gate / norm work
  // below instead of starting after three threadgroup barriers.
  device const ST* si = state_in  + (size_t)bh * D * D;   // [B, H, D, D]
  device ST*       so = state_out + (size_t)bh * D * D;
  float st[NDV][NDK];
  for (int j = 0; j < NDV; ++j) {
    uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i) {
      st[j][i] = float(si[(size_t)dv * D + NDK * lane + i]);
    }
  }

  // ---------------------------------------------------------------- phase 0a
  // Depthwise causal conv1d over the K-tap window [conv_state ; x_t], then silu,
  // and shift the cached window.  This head owns 3*D of the 3*H*D channels
  // (one slice each of the q / k / v thirds of the fused in-projection).
  for (uint idx = tid; idx < 3u * (uint)D; idx += NT) {
    uint part = idx / (uint)D;
    uint d    = idx - part * (uint)D;
    uint c    = part * QKVD + h * (uint)D + d;
    device const T* wc = conv_w + (size_t)c * K;
    float acc = 0.0f;
    for (uint j = 0; j + 1 < (uint)K; ++j) {
      acc += float(conv_state[cs_off + (size_t)j * CDIM + c]) * float(wc[j]);
    }
    // `mixed = mx.where(mask[..., None], mixed, 0)`: batched decode masks the
    // *pre-conv* input of a padded row (history and the recurrence still run),
    // so the zero has to land here, before both the conv and the window write.
    T xnew = valid[b] ? ((part == 0u) ? mq[qkv_off + h * (uint)D + d]
                      : ((part == 1u) ? mk[qkv_off + h * (uint)D + d]
                                      : mv[qkv_off + h * (uint)D + d]))
                      : static_cast<T>(0);
    acc += float(xnew) * float(wc[K - 1]);

    T xb  = static_cast<T>(acc);           // mx.conv1d writes its output in T
    T sig = mlx_sigmoid_fast(xb);          // nn.silu = x * mx.sigmoid(x), compiled
    T sl  = xb * sig;
    if (part == 0u)      sq[d] = float(sl);
    else if (part == 1u) sk[d] = float(sl);
    else                 sv[d] = float(sl);

    // new window = [old[1 .. K-2], x_t]
    for (uint j = 0; j + 2 < (uint)K; ++j) {
      conv_state_out[cs_off + (size_t)j * CDIM + c] =
          conv_state[cs_off + (size_t)(j + 1) * CDIM + c];
    }
    conv_state_out[cs_off + (size_t)(K - 2) * CDIM + c] = xnew;
  }

  // ---------------------------------------------------------------- phase 0b
  // Safe forget gate  g = exp(lb * sigmoid(exp(A_log) * (a + dt_bias)))  in fp32,
  // beta = sigmoid(b) rounded to T, and the output gate pulled into shared mem.
  {
    float a_exp = metal::precise::exp(A_log[h]);   // mx.exp -> precise
    for (uint d = tid; d < (uint)D; d += NT) {
      float av = float(a[qkv_off + h * (uint)D + d]) + dt_bias[h * (uint)D + d];
      sg[d]    = metal::precise::exp(lower_bound * mlx_sigmoid_fast<float>(a_exp * av));
      sgate[d] = float(gate[qkv_off + h * (uint)D + d]);
    }
    if (tid == 0u) {
      shr[2] = float(mlx_sigmoid_precise(bvec[b * (uint)H + h]));  // beta = mx.sigmoid(b), in T
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---------------------------------------------------------------- phase 0c
  // q = l2norm(q) * D^-0.5 ; k = l2norm(k), both rounded back to T.
  if (simdgroup_index_in_threadgroup == 0u) {
    // Same partition and same accumulation order as MLX's row_reduce_simple
    // (N_READS = 4 contiguous elements per lane, then simd_sum), so the fp32
    // sum is bit-identical to `(x * x).sum(-1)`.
    float pq = 0.0f, pk = 0.0f;
    for (int blk = 0; blk < RBLK; ++blk) {
      uint base = (uint)(blk * 128) + 4u * lane;
      for (int i = 0; i < 4; ++i) {
        pq = sq_acc(pq, sq[base + i]);
        pk = sq_acc(pk, sk[base + i]);
      }
    }
    uint base = (uint)(RBLK * 128) + 4u * lane;
    if (4u * lane + 4u <= (uint)REXTRA) {
      for (int i = 0; i < 4; ++i) {
        pq = sq_acc(pq, sq[base + i]);
        pk = sq_acc(pk, sk[base + i]);
      }
    } else {
      for (int i = 0; 4u * lane + (uint)i < (uint)REXTRA; ++i) {
        pq = sq_acc(pq, sq[base + i]);
        pk = sq_acc(pk, sk[base + i]);
      }
    }
    pq = simd_sum(pq);
    pk = simd_sum(pk);
    if (lane == 0u) {
      shr[0] = metal::precise::rsqrt(pq + 1.0e-6f);
      shr[1] = metal::precise::rsqrt(pk + 1.0e-6f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    float rq = shr[0], rk = shr[1];
    for (uint d = tid; d < (uint)D; d += NT) {
      sq[d] = float(static_cast<T>((sq[d] * rq) * qscale));
      sk[d] = float(static_cast<T>(sk[d] * rk));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ----------------------------------------------------------------- phase 1
  // Gated delta rule, one time step.  Identical arithmetic and identical simd
  // reduction partition to models/gated_delta.py's kernel: lane `lane` owns key
  // elements [NDK*lane, NDK*lane + NDK).
  {
    float beta = shr[2];
    for (int j = 0; j < NDV; ++j) {
      uint dv = ty + (uint)TY * (uint)j;
      float kv = 0.0f;
      for (int i = 0; i < NDK; ++i) {
        uint s = NDK * lane + i;
        st[j][i] = st[j][i] * sg[s];
        kv += st[j][i] * sk[s];
      }
      kv = simd_sum(kv);
      float delta = (sv[dv] - kv) * beta;
      float o = 0.0f;
      for (int i = 0; i < NDK; ++i) {
        uint s = NDK * lane + i;
        st[j][i] = st[j][i] + sk[s] * delta;
        o += st[j][i] * sq[s];
      }
      o = simd_sum(o);
      if (thread_index_in_simdgroup == 0u) {
        sy[dv] = float(static_cast<T>(o));   // gated_delta writes y in T
      }
      for (int i = 0; i < NDK; ++i) {
        so[(size_t)dv * D + NDK * lane + i] = static_cast<ST>(st[j][i]);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ----------------------------------------------------------------- phase 2
  // Gated RMSNorm over the value axis, in fp32, then back to T.
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
      y[qkv_off + h * (uint)D + d] = static_cast<T>(x);
    }
  }
"""

# Appended to _SOURCE for the capture variant.  `sq` / `sk` / `sv` still hold the
# post-conv, post-L2-norm q / k / v at this point (phases 1-2 only touch `sy`), so
# the sink tensors come straight out of threadgroup memory -- the same values the
# recurrence consumed, hence bit-identical to what the eager path stashes.
_SINK_SOURCE = """
  // --------------------------------------------------------------- sink emit
  // Speculative capture: hand back exactly the tensors gdn_sink carries, so
  // rollback_speculative_cache replays the accepted prefix on identical inputs.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = tid; d < (uint)D; d += NT) {
    q_out[qkv_off + h * (uint)D + d] = static_cast<T>(sq[d]);
    k_out[qkv_off + h * (uint)D + d] = static_cast<T>(sk[d]);
    v_out[qkv_off + h * (uint)D + d] = static_cast<T>(sv[d]);
  }
  // conv_input = concatenate([conv_state, mixed], axis=1), i.e. [1, K, 3*H*D].
  for (uint idx = tid; idx < 3u * (uint)D; idx += NT) {
    uint part = idx / (uint)D;
    uint d    = idx - part * (uint)D;
    uint c    = part * QKVD + h * (uint)D + d;
    for (uint j = 0; j + 1 < (uint)K; ++j) {
      conv_input_out[(size_t)b * K * CDIM + (size_t)j * CDIM + c] =
          conv_state[cs_off + (size_t)j * CDIM + c];
    }
    conv_input_out[(size_t)b * K * CDIM + (size_t)(K - 1) * CDIM + c] =
        valid[b] ? ((part == 0u) ? mq[qkv_off + h * (uint)D + d]
                 : ((part == 1u) ? mk[qkv_off + h * (uint)D + d]
                                 : mv[qkv_off + h * (uint)D + d]))
                 : static_cast<T>(0);
  }
"""

# ---------------------------------------------------------------------------
# Optional extra fold: f_b_proj / g_b_proj done in-kernel.
#
# Both are Linear(head_dim, num_heads*head_dim) affine-quantized GEMVs whose only
# consumer is this kernel, so folding them in removes two dispatches per layer at
# the cost of streaming ~2 MB more weight.
#
# It is written as a transcription of MLX's affine ``qmv_quad`` path
# (mlx/backend/metal/kernels/quantized.h: load_vector + qdot + quad_sum, bits==8)
# rather than as a generic dot product: one quad per output row, quad lane `l`
# owning x[VPT*l : VPT*l+VPT] with VPT = head_dim/4, one scale/bias pair per
# thread, and the row total closed by quad_sum.  Same partition and same
# accumulation order as MLX, so the folded projection is bit-identical to
# mx.quantized_matmul -- verified for head_dim in {64, 128}.  (A plain
# per-element ``x * (scale*q + bias)`` dot instead disagrees on ~0.01% of
# elements, which is enough to flip greedy tokens.)
#
# Element e of row r is byte e of the row for bits==8:
#   ((w[r, e/4] >> (8 * (e % 4))) & 0xff) * scales[r, e/GS] + biases[r, e/GS]
_QPROJ_COMPUTE = """
  // ------------------------------------------------------------- phase 0a-pre
  {
    constexpr int VPT   = D / 4;        // values_per_thread
    constexpr int SSPT  = GS / VPT;     // scale_step_per_thread
    constexpr int NQUAD = NT / 4;
    constexpr int NG    = D / GS;
    const uint qg   = tid / 4u;         // quad index within the threadgroup
    const uint qlid = tid % 4u;         // lane within the quad

    // Rows [0, D) are f_b_proj, [D, 2D) are g_b_proj; a quad owns one row.
    for (uint t = qg; t < 2u * (uint)D; t += (uint)NQUAD) {
      uint proj = t / (uint)D;
      uint d    = t - proj * (uint)D;
      uint row  = h * (uint)D + d;
      device const T* xp =
          (proj == 0u ? fa : ga) + b * (uint)D + qlid * (uint)VPT;
      float xs = 0.0f;                                  // load_vector's `sum`
      for (int i = 0; i < VPT; ++i) xs += float(xp[i]);
      size_t wb = (size_t)row * D + (size_t)qlid * VPT; // byte offset (bits == 8)
      uint gi = row * (uint)NG + qlid / (uint)SSPT;
      float sc = float(proj == 0u ? fbs[gi] : gbs[gi]);
      float bi = float(proj == 0u ? fbb[gi] : gbb[gi]);
      float accum = 0.0f;                               // qdot
      for (int i = 0; i < VPT; ++i) {
        size_t bo = wb + (size_t)i;
        uint word = (proj == 0u) ? fbw[bo / 4u] : gbw[bo / 4u];
        accum += float(xp[i]) * float((word >> (8u * (uint)(bo % 4u))) & 0xffu);
      }
      float r = quad_sum(sc * accum + xs * bi);
      if (qlid == 0u) {
        if (proj == 0u) sa[d] = float(static_cast<T>(r));      // f_b_proj -> T
        else            sgate[d] = float(static_cast<T>(r));   // g_b_proj -> T
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
"""


def _qproj_source(source: str) -> str:
    """Derive the fold-the-GEMVs source from the validated base source."""
    out = source.replace(
        "  threadgroup float shr[3];",
        "  threadgroup float sa[D];          // in-kernel f_b_proj output\n"
        "  threadgroup float shr[3];",
        1,
    )
    out = out.replace(
        "  // ---------------------------------------------------------------- phase 0a\n",
        _QPROJ_COMPUTE
        + "\n  // ---------------------------------------------------------------- phase 0a\n",
        1,
    )
    out = out.replace(
        "      float av = float(a[qkv_off + h * (uint)D + d]) + "
        "dt_bias[h * (uint)D + d];",
        "      float av = sa[d] + dt_bias[h * (uint)D + d];",
        1,
    )
    out = out.replace(
        "      sgate[d] = float(gate[qkv_off + h * (uint)D + d]);\n", "", 1
    )
    return out


_INPUT_NAMES = [
    "mq",
    "mk",
    "mv",
    "conv_state",
    "conv_w",
    "a",
    "bvec",
    "A_log",
    "dt_bias",
    "state_in",
    "gate",
    "o_w",
    "lower_bound",
    "qscale",
    "norm_eps",
    "valid",
]
_OUTPUT_NAMES = ["y", "state_out", "conv_state_out"]
_SINK_OUTPUT_NAMES = _OUTPUT_NAMES + ["q_out", "k_out", "v_out", "conv_input_out"]

# "a" / "gate" (the f_b_proj / g_b_proj outputs) are replaced by their inputs and
# quantized weights when the projections are folded in.
_QPROJ_INPUT_NAMES = [n for n in _INPUT_NAMES if n not in ("a", "gate")] + [
    "fa",
    "fbw",
    "fbs",
    "fbb",
    "ga",
    "gbw",
    "gbs",
    "gbb",
]

_KERNELS = {}
_KERNEL_TRIED = False

# Batched decode always supplies a per-row bool mask -- BatchGenerator sets
# left_padding on the ArraysCache even for a uniform-length batch -- so the
# kernel takes one unconditionally.  A mask-free call gets a cached all-true
# vector rather than allocating one per step.
_ONES_MASK = {}


def _all_valid(batch: int) -> mx.array:
    m = _ONES_MASK.get(batch)
    if m is None:
        m = mx.ones((batch,), dtype=mx.bool_)
        mx.eval(m)
        _ONES_MASK[batch] = m
    return m


# Threadgroup y-extent: 32 * TY threads per threadgroup, one threadgroup per head.
#
# TY only controls how many value-dim rows each thread walks (NDV = head_dim / TY)
# and how many output rows each quad walks in the projection fold.  Every
# reduction -- the simd_sum over the 32 key lanes, the 32-lane row reduction in
# the two norms, and quad_sum in the folded GEMV -- is over the same lanes with
# the same operand order at every TY, so lowering it is partition-preserving and
# stays bit-identical.  That matters because `maxTotalThreadsPerThreadgroup` is a
# *per-pipeline* limit driven by register pressure: some GPUs (notably the
# virtualized ones on CI runners) admit 1024 threads for the base kernel but cap
# the higher-pressure qproj pipeline lower.  We probe downwards.
_TY_CANDIDATES = (32, 16, 8, 4)


def _kernel(kind: str = "base"):
    """``kind`` in {"base", "capture", "qproj"}; ``None`` if Metal is unavailable.

    Three objects rather than one: mx.fast.metal_kernel derives the function
    signature from input_names/output_names, so each variant needs its own.
    """
    global _KERNEL_TRIED
    if not _KERNEL_TRIED:
        _KERNEL_TRIED = True
        if mx.metal.is_available():
            _KERNELS["base"] = mx.fast.metal_kernel(
                name="glm5_kda_decode_step",
                input_names=_INPUT_NAMES,
                output_names=_OUTPUT_NAMES,
                header=_HEADER,
                source=_SOURCE,
            )
            _KERNELS["capture"] = mx.fast.metal_kernel(
                name="glm5_kda_decode_step_capture",
                input_names=_INPUT_NAMES,
                output_names=_SINK_OUTPUT_NAMES,
                header=_HEADER,
                source=_SOURCE + _SINK_SOURCE,
            )
            _KERNELS["block"] = mx.fast.metal_kernel(
                name="glm5_kda_verify_block",
                input_names=_BLOCK_INPUT_NAMES,
                output_names=_BLOCK_OUTPUT_NAMES,
                header=_HEADER,
                source=_BLOCK_SOURCE,
            )
            _KERNELS["qproj"] = mx.fast.metal_kernel(
                name="glm5_kda_decode_step_qproj",
                input_names=_QPROJ_INPUT_NAMES,
                output_names=_OUTPUT_NAMES,
                header=_HEADER,
                source=_qproj_source(_SOURCE),
            )
    return _KERNELS.get(kind)


# (kind, dtype, state dtype, H, D, K, bits, group_size) -> usable TY, or None if
# the device cannot run this variant at any admissible threadgroup size.
_TY_PROBE_CACHE = {}


def _probe_launch(kind, ty, dt, st, num_heads, head_dim, conv_kernel_size, bits, gs):
    """Run one throwaway launch at ``ty`` and force it, so the driver's
    per-pipeline threadgroup limit is exercised here rather than mid-forward."""
    h, d, k = num_heads, head_dim, conv_kernel_size
    zeros = lambda shape, dtype: mx.zeros(shape, dtype=dtype)  # noqa: E731
    args = dict(
        q_in=zeros((1, 1, h * d), dt),
        k_in=zeros((1, 1, h * d), dt),
        v_in=zeros((1, 1, h * d), dt),
        conv_state=zeros((1, k - 1, 3 * h * d), dt),
        conv_w=zeros((3 * h * d, k, 1), dt),
        b=zeros((1, 1, h), dt),
        A_log=zeros((h,), mx.float32),
        dt_bias=zeros((h * d,), mx.float32),
        state=zeros((1, h, d, d), st),
        o_weight=zeros((d,), dt),
    )
    if kind == "qproj":
        pack = 32 // bits
        w = zeros((h * d, d // pack), mx.uint32)
        sc = zeros((h * d, d // gs), dt)
        proj = _ProbeLinear(w, sc, bits, gs)
        args["a"] = None
        args["gate"] = None
        qproj = (zeros((1, 1, d), dt), proj, zeros((1, 1, d), dt), proj)
    else:
        args["a"] = zeros((1, 1, h * d), dt)
        args["gate"] = zeros((1, 1, h * d), dt)
        qproj = None
    outs = fused_kda_decode_step(
        args["q_in"],
        args["k_in"],
        args["v_in"],
        args["conv_state"],
        args["conv_w"],
        args["a"],
        args["b"],
        args["A_log"],
        args["dt_bias"],
        args["state"],
        args["gate"],
        args["o_weight"],
        num_heads=h,
        head_dim=d,
        conv_kernel_size=k,
        lower_bound=-5.0,
        norm_eps=1e-5,
        ty=ty,
        qproj=qproj,
    )
    mx.eval(outs)


class _ProbeLinear:
    """Minimal stand-in carrying just what the folded GEMV reads."""

    def __init__(self, weight, scales, bits, group_size):
        self.weight = weight
        self.scales = scales
        self.biases = scales
        self.bits = bits
        self.group_size = group_size


def fused_kda_probe(
    *,
    kind: str,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    dtype,
    state_dtype,
    bits: Optional[int] = None,
    group_size: Optional[int] = None,
) -> Optional[int]:
    """Largest threadgroup extent this device will run ``kind`` at, else ``None``.

    ``maxTotalThreadsPerThreadgroup`` is a *per-pipeline* limit set by register
    pressure, not a device constant: a GPU can admit 1024 threads for the base
    kernel and cap the heavier qproj pipeline lower (CI's virtualized runners cap
    it at 640).  MLX reports that as a ValueError at eval time, far from the call
    site, so probe it up front and remember the answer.  Lowering TY is
    partition-preserving -- every reduction keeps the same lanes and the same
    operand order -- so a degraded launch is still bit-identical.
    """
    key = (
        kind,
        dtype,
        state_dtype,
        num_heads,
        head_dim,
        conv_kernel_size,
        bits,
        group_size,
    )
    if key in _TY_PROBE_CACHE:
        return _TY_PROBE_CACHE[key]
    result = None
    for ty in _TY_CANDIDATES:
        if head_dim % ty:
            continue
        try:
            _probe_launch(
                kind,
                ty,
                dtype,
                state_dtype,
                num_heads,
                head_dim,
                conv_kernel_size,
                bits,
                group_size,
            )
        except ValueError as exc:
            if "threads per threadgroup" not in str(exc):
                raise
            continue
        except RuntimeError as exc:  # kernel would not build on this device
            logger.info("glm5_next fused KDA (%s) unavailable: %s", kind, exc)
            break
        result = ty
        break
    _TY_PROBE_CACHE[key] = result
    if result is None:
        logger.info(
            "glm5_next fused KDA (%s) declined: this device's threadgroup limit "
            "is below the kernel's requirement at every supported size",
            kind,
        )
    elif result != _TY_CANDIDATES[0]:
        logger.info(
            "glm5_next fused KDA (%s) running at a reduced threadgroup "
            "(%d threads): the device caps this pipeline below %d.  Results are "
            "unchanged; this only lowers occupancy.",
            kind,
            32 * result,
            32 * _TY_CANDIDATES[0],
        )
    return result


def fused_kda_supported(
    *,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    lower_bound: Optional[float],
) -> bool:
    """Shape/feature preconditions for the fused kernel (config-level, not per-step).

    Bit-identity with the eager path is verified for ``head_dim <= 128`` (where
    MLX's row reduction uses a 32-wide thread row, which the in-kernel L2 / RMS
    reductions mirror).  Larger head dims stay numerically correct but may differ
    from the eager path in the last bit of those two reductions.
    """
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    if lower_bound is None:
        return False  # only the "safe gate" branch is transcribed
    if conv_kernel_size < 2:
        return False
    if head_dim % 32 != 0:
        return False
    if not any(head_dim % ty == 0 for ty in _TY_CANDIDATES):
        return False
    if num_heads <= 0:
        return False
    return _kernel("base") is not None


def fused_kda_qproj_supported(f_b_proj, g_b_proj, *, head_dim: int) -> bool:
    """Can f_b_proj / g_b_proj be folded into the kernel?

    Requires both to be affine-quantized with the same bits/group_size, a packed
    layout the kernel can address, and a lane-aligned input dim.
    """
    mods = (f_b_proj, g_b_proj)
    if not all(hasattr(m, "scales") and hasattr(m, "biases") for m in mods):
        return False
    if len({getattr(m, "mode", "affine") for m in mods}) != 1:
        return False
    if getattr(mods[0], "mode", "affine") != "affine":
        return False
    if len({m.bits for m in mods}) != 1 or len({m.group_size for m in mods}) != 1:
        return False
    bits, group_size = mods[0].bits, mods[0].group_size
    # The in-kernel GEMV transcribes MLX's affine qmv_quad, which MLX only
    # dispatches for these shapes -- outside them the fold would still be
    # correct but no longer bit-identical, so decline instead.
    if bits != 8:
        return False
    if head_dim not in (64, 128):
        return False
    values_per_thread = head_dim // 4
    if group_size % values_per_thread or head_dim % group_size:
        return False
    pack = 32 // bits
    for m in mods:
        if m.weight.dtype != mx.uint32:
            return False
        if m.weight.shape[-1] != head_dim // pack:
            return False
        if m.scales.shape[-1] != head_dim // group_size:
            return False
        if m.scales.shape != m.biases.shape:
            return False
    return _kernel("qproj") is not None


def fused_kda_decode_step(
    q_in: mx.array,
    k_in: mx.array,
    v_in: mx.array,
    conv_state: mx.array,
    conv_w: mx.array,
    a: Optional[mx.array],
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    gate: Optional[mx.array],
    o_weight: mx.array,
    *,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    lower_bound: float,
    norm_eps: float,
    mask: Optional[mx.array] = None,
    ty: int = 32,
    capture: bool = False,
    qproj: Optional[Tuple] = None,
) -> Tuple[mx.array, ...]:
    """One fused KDA decode step, S=1, any batch size.

    Args (S=1):
      q_in, k_in, v_in: ``[B, 1, H*D]`` pre-conv projections, dtype ``T``.
      conv_state:       ``[B, K-1, 3*H*D]`` cached conv window, dtype ``T``.
      conv_w:           ``[3*H*D, K, 1]`` depthwise conv weight, dtype ``T``.
      a:                ``[B, 1, H*D]`` forget-gate ``f_b_proj`` output, dtype ``T``.
      b:                ``[B, 1, H]`` beta logits, dtype ``T``.
      A_log:            ``[H]`` float32.   dt_bias: ``[H*D]`` float32.
      state:            ``[B, H, D, D]`` recurrent state (float32 in practice).
      gate:             ``[B, 1, H*D]`` ``g_b_proj`` output, dtype ``T``.
      o_weight:         ``[D]`` gated-RMSNorm weight, dtype ``T``.
      mask:             optional ``[B, 1]`` bool; a false row has its pre-conv
                        input zeroed, matching the eager
                        ``mx.where(mask[..., None], mixed, 0)``.

    Returns ``(y, state_out, conv_state_out)`` where ``y`` is ``[B, 1, H*D]`` and is
    exactly what the eager path feeds to ``o_proj``.

    With ``capture=True`` it additionally returns
    ``(q_out, k_out, v_out, conv_input_out)``: the post-conv / post-L2-norm q, k, v
    as ``[B, 1, H, D]`` and ``concatenate([conv_state, mixed], axis=1)`` as
    ``[B, K, 3*H*D]`` -- the tensors ``gdn_sink`` carries for speculative rollback.

    One threadgroup per (batch row, head).  B does not appear in the kernel
    source, only in grid.z and the buffer extents, so the compiled pipeline is
    identical at every B and the threadgroup probe result carries over.
    """
    H, D, K = num_heads, head_dim, conv_kernel_size
    B = q_in.shape[0]
    dt = q_in.dtype
    valid = _all_valid(B) if mask is None else mask.reshape(B)
    kind = "capture" if capture else ("qproj" if qproj is not None else "base")
    kernel = _kernel(kind)
    out_shapes = [(B, 1, H * D), state.shape, conv_state.shape]
    out_dtypes = [dt, state.dtype, dt]
    if capture:
        out_shapes += [(B, 1, H, D)] * 3 + [(B, K, 3 * H * D)]
        out_dtypes += [dt] * 4
    head = [q_in, k_in, v_in, conv_state, conv_w]
    tail = [b, A_log, dt_bias, state, o_weight]
    scalars = [float(lower_bound), float(head_dim**-0.5), float(norm_eps), valid]
    template = [
        ("T", dt),
        ("ST", state.dtype),
        ("H", num_heads),
        ("D", head_dim),
        ("K", conv_kernel_size),
        ("TY", ty),
    ]
    if qproj is not None:
        fa, f_b_proj, ga, g_b_proj = qproj
        inputs = (
            head
            + tail
            + scalars
            + [
                fa,
                f_b_proj.weight,
                f_b_proj.scales,
                f_b_proj.biases,
                ga,
                g_b_proj.weight,
                g_b_proj.scales,
                g_b_proj.biases,
            ]
        )
        template += [("BITS", f_b_proj.bits), ("GS", f_b_proj.group_size)]
    else:
        inputs = head + [a] + tail[:4] + [gate, o_weight] + scalars
    return kernel(
        inputs=inputs,
        template=template,
        grid=(32, ty, B * num_heads),
        threadgroup=(32, ty, 1),
        output_shapes=out_shapes,
        output_dtypes=out_dtypes,
    )


# ===========================================================================
# S>1: the speculative verify block, one kernel per layer for the whole block.
#
# WHAT THIS REPLACES, precisely.  The recurrence was ALREADY fused at S>1 --
# gated_delta_update falls through to gated_delta_kernel whenever head_dim is a
# multiple of 32, and that kernel carries its own `for t` scan with the state in
# registers.  So this is NOT "fusing an unfused scan".  What runs eager at S>1 is
# the glue around it: the conv window concat/slice/copy, silu, the two fp32 L2
# norms, the beta sigmoid, and the hand-rolled gated RMSNorm at language.py:423
# (~12 dispatches on its own, and it does not call mx.fast.rms_norm).  Roughly 33
# fusable launches per layer x 34 KDA layers.
#
# At lane 1's measured 14.72 us for a DEPENDENT launch -- the right instrument
# here, since this is a serial chain of tiny kernels with nothing in flight to
# hide a launch behind -- that bounds the prize at ~16.5 ms per verify forward,
# ~17.5% of a 94 ms W=8 verify.  UPPER BOUND, per the fusion ledger's R1: kernel
# count x dispatch cost bounds the prize, it does not predict it.
#
# WHAT THIS BUYS, and one thing it does NOT:
#   * the ~33 dependent launches per layer collapse to one -- measured at a flat
#     ~3.2 ms per verify forward against a 3.03 ms dispatch-elimination ceiling,
#     i.e. essentially all of it, because these are serial launches with nothing
#     in flight to hide them (GAP2_RESULT_ACD.json);
#   * q/k/v/g never reach device memory between the conv and the recurrence;
#   * it does NOT save state round trips. An earlier version of this note claimed
#     W round trips collapse to one. That was wrong: gated_delta_kernel already
#     loads the state before its `for t` loop and stores it after, so the eager
#     path was already paying one per block. The saving is dispatch count only,
#     which is exactly why the measurement lands on a pure dispatch ceiling.
#
# The saving is FLAT in W -- 34 layers x ~33 launches does not depend on the block
# width -- so the PERCENTAGE shrinks as W grows: +7.3% at W=2, +4.1% at W=8, about
# +4.7% at adaptive-K's measured mean width of 4.6.
#
# S is a RUNTIME scalar, not a template parameter.  Adaptive-K varies the width
# every round (measured mean 4.6 on code), and a templated S would compile a new
# pipeline per width and pay a cold compile on the hot path -- the failure mode
# compile_width_churn.json was written to look for.  One pipeline serves W=1..16.
_BLOCK_SOURCE = """
  const uint bh   = threadgroup_position_in_grid.z;
  const uint b    = bh / (uint)H;
  const uint h    = bh - b * (uint)H;
  const uint lane = thread_position_in_threadgroup.x;
  const uint ty   = thread_position_in_threadgroup.y;
  const uint tid  = thread_index_in_threadgroup;
  const uint S    = (uint)nsteps;

  constexpr int NT   = 32 * TY;
  constexpr int RBLK = D / 128;
  constexpr int REXTRA = D - RBLK * 128;
  constexpr int NDK  = D / 32;
  constexpr int NDV  = D / TY;
  constexpr uint QKVD = (uint)(H * D);
  constexpr uint CDIM = 3u * QKVD;
  constexpr uint KM1  = (uint)(K - 1);
  const size_t cs_off = (size_t)b * KM1 * CDIM;

  threadgroup float sq[D];
  threadgroup float sk[D];
  threadgroup float sv[D];
  threadgroup float sg[D];
  threadgroup float sgate[D];
  threadgroup float sy[D];
  threadgroup float shr[3];
  // Rolling pre-conv window: the last K-1 inputs, oldest-to-newest, as a circular
  // buffer.  3*D per slot (this head's slice of the q/k/v thirds).  At D=128,
  // K=4 that is 1152 elements -- a few KB next to the ~3 KB of scratch above.
  threadgroup T twin[(K - 1) * 3 * D];

  // State in registers for the WHOLE block: loaded here, stored after the loop.
  device const ST* si = state_in  + (size_t)bh * D * D;
  device ST*       so = state_out + (size_t)bh * D * D;
  float st[NDV][NDK];
  for (int j = 0; j < NDV; ++j) {
    uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i) {
      st[j][i] = float(si[(size_t)dv * D + NDK * lane + i]);
    }
  }

  // Seed the window from the cache, oldest at slot 0 -- the same order the eager
  // path lays out concatenate([conv_state, mixed], axis=1).
  for (uint idx = tid; idx < KM1 * 3u * (uint)D; idx += NT) {
    uint slot = idx / (3u * (uint)D);
    uint r    = idx - slot * 3u * (uint)D;
    uint part = r / (uint)D;
    uint d    = r - part * (uint)D;
    uint c    = part * QKVD + h * (uint)D + d;
    twin[slot * 3u * (uint)D + r] = conv_state[cs_off + (size_t)slot * CDIM + c];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float a_exp = metal::precise::exp(A_log[h]);

  for (uint t = 0u; t < S; ++t) {
    const uint tok_off = (b * S + t) * QKVD;      // [B, S, H*D]
    // ------------------------------------------------------------ phase 0a
    // Depthwise causal conv over [window ; x_t].  For token t the K taps are
    // conv_input[t .. t+K-1], which in the circular window are slots
    // ((t + j) mod K-1) oldest-first, then x_t itself.
    for (uint idx = tid; idx < 3u * (uint)D; idx += NT) {
      uint part = idx / (uint)D;
      uint d    = idx - part * (uint)D;
      uint c    = part * QKVD + h * (uint)D + d;
      device const T* wc = conv_w + (size_t)c * K;
      float acc = 0.0f;
      for (uint j = 0; j + 1 < (uint)K; ++j) {
        uint slot = (t + j) % KM1;
        acc += float(twin[slot * 3u * (uint)D + idx]) * float(wc[j]);
      }
      // The eager path zeroes the PRE-conv input of a masked row/token, so the
      // zero lands here -- before both the conv and the window write.
      T xnew = valid[b * S + t]
                 ? ((part == 0u) ? mq[tok_off + h * (uint)D + d]
                  : ((part == 1u) ? mk[tok_off + h * (uint)D + d]
                                  : mv[tok_off + h * (uint)D + d]))
                 : static_cast<T>(0);
      acc += float(xnew) * float(wc[K - 1]);

      T xb  = static_cast<T>(acc);
      T sig = mlx_sigmoid_fast(xb);
      T sl  = xb * sig;
      if (part == 0u)      sq[d] = float(sl);
      else if (part == 1u) sk[d] = float(sl);
      else                 sv[d] = float(sl);
    }
    // The window write is deferred past this barrier: until every thread has
    // finished reading its taps, the slot about to be overwritten is still live.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Retire the oldest slot and install x_t in its place.
    for (uint idx = tid; idx < 3u * (uint)D; idx += NT) {
      uint part = idx / (uint)D;
      uint d    = idx - part * (uint)D;
      T xnew = valid[b * S + t]
                 ? ((part == 0u) ? mq[tok_off + h * (uint)D + d]
                  : ((part == 1u) ? mk[tok_off + h * (uint)D + d]
                                  : mv[tok_off + h * (uint)D + d]))
                 : static_cast<T>(0);
      twin[(t % KM1) * 3u * (uint)D + idx] = xnew;
    }

    // ------------------------------------------------------------ phase 0b
    for (uint d = tid; d < (uint)D; d += NT) {
      float av = float(a[tok_off + h * (uint)D + d]) + dt_bias[h * (uint)D + d];
      sg[d]    = metal::precise::exp(lower_bound * mlx_sigmoid_fast<float>(a_exp * av));
      sgate[d] = float(gate[tok_off + h * (uint)D + d]);
    }
    if (tid == 0u) {
      shr[2] = float(mlx_sigmoid_precise(bvec[(b * S + t) * (uint)H + h]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ------------------------------------------------------------ phase 0c
    if (simdgroup_index_in_threadgroup == 0u) {
      float pq = 0.0f, pk = 0.0f;
      for (int blk = 0; blk < RBLK; ++blk) {
        uint base = (uint)(blk * 128) + 4u * lane;
        for (int i = 0; i < 4; ++i) {
          pq = sq_acc(pq, sq[base + i]);
          pk = sq_acc(pk, sk[base + i]);
        }
      }
      uint base = (uint)(RBLK * 128) + 4u * lane;
      if (4u * lane + 4u <= (uint)REXTRA) {
        for (int i = 0; i < 4; ++i) {
          pq = sq_acc(pq, sq[base + i]);
          pk = sq_acc(pk, sk[base + i]);
        }
      } else {
        for (int i = 0; 4u * lane + (uint)i < (uint)REXTRA; ++i) {
          pq = sq_acc(pq, sq[base + i]);
          pk = sq_acc(pk, sk[base + i]);
        }
      }
      pq = simd_sum(pq);
      pk = simd_sum(pk);
      if (lane == 0u) {
        shr[0] = metal::precise::rsqrt(pq + 1.0e-6f);
        shr[1] = metal::precise::rsqrt(pk + 1.0e-6f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      float rq = shr[0], rk = shr[1];
      for (uint d = tid; d < (uint)D; d += NT) {
        sq[d] = float(static_cast<T>((sq[d] * rq) * qscale));
        sk[d] = float(static_cast<T>(sk[d] * rk));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ------------------------------------------------------- sink emit (t)
    // The verify path ALWAYS carries a gdn_sink -- that is how a rejected round
    // is replayed to the accepted token -- so the tensors it needs are emitted
    // unconditionally rather than behind a second kernel variant.  sq/sk/sv hold
    // the post-conv, post-L2-norm values the recurrence is about to consume, so
    // these are bit-identical to what the eager path stashes.  The only sink
    // member not produced here is conv_input, which is a concat of tensors the
    // caller already holds and costs it one dispatch.
    {
      const uint hd_off = ((b * S + t) * (uint)H + h) * (uint)D;
      for (uint d = tid; d < (uint)D; d += NT) {
        q_out[hd_off + d] = static_cast<T>(sq[d]);
        k_out[hd_off + d] = static_cast<T>(sk[d]);
        v_out[hd_off + d] = static_cast<T>(sv[d]);
      }
    }

    // ------------------------------------------------------------- phase 1
    // Gated delta rule.  Same arithmetic and the same simd partition as both the
    // S=1 kernel above and gated_delta_kernel's own t-loop -- lane `lane` owns
    // key elements [NDK*lane, NDK*lane+NDK).  The ONLY difference from the S=1
    // kernel is that `st` is not written out here; it carries to the next token.
    {
      float beta = shr[2];
      for (int j = 0; j < NDV; ++j) {
        uint dv = ty + (uint)TY * (uint)j;
        float kv = 0.0f;
        for (int i = 0; i < NDK; ++i) {
          uint s = NDK * lane + i;
          st[j][i] = st[j][i] * sg[s];
          kv += st[j][i] * sk[s];
        }
        kv = simd_sum(kv);
        float delta = (sv[dv] - kv) * beta;
        float o = 0.0f;
        for (int i = 0; i < NDK; ++i) {
          uint s = NDK * lane + i;
          st[j][i] = st[j][i] + sk[s] * delta;
          o += st[j][i] * sq[s];
        }
        o = simd_sum(o);
        if (thread_index_in_simdgroup == 0u) {
          sy[dv] = float(static_cast<T>(o));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ------------------------------------------------------------- phase 2
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // ---------------------------------------------------------------- epilogue
  // One state store for the whole block, and the cache window = the last K-1
  // pre-conv rows, oldest-first.  The circular buffer already holds exactly
  // those, at slots ((S + j) mod K-1) -- which is correct for W >= K-1 (all new
  // tokens) and for W < K-1 (a mix of cached rows and new tokens) alike, with no
  // special case, because the FIFO never distinguished them.
  for (int j = 0; j < NDV; ++j) {
    uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i) {
      so[(size_t)dv * D + NDK * lane + i] = static_cast<ST>(st[j][i]);
    }
  }
  for (uint idx = tid; idx < KM1 * 3u * (uint)D; idx += NT) {
    uint slot = idx / (3u * (uint)D);
    uint r    = idx - slot * 3u * (uint)D;
    uint part = r / (uint)D;
    uint d    = r - part * (uint)D;
    uint c    = part * QKVD + h * (uint)D + d;
    conv_state_out[cs_off + (size_t)slot * CDIM + c] =
        twin[((S + slot) % KM1) * 3u * (uint)D + r];
  }
"""

_BLOCK_OUTPUT_NAMES = _OUTPUT_NAMES + ["q_out", "k_out", "v_out"]
_BLOCK_INPUT_NAMES = [
    "mq", "mk", "mv", "conv_state", "conv_w", "a", "bvec", "A_log", "dt_bias",
    "state_in", "gate", "o_w", "lower_bound", "qscale", "norm_eps", "valid",
    "nsteps",
]


def fused_kda_verify_block(
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
    ty: int = 32,
) -> Tuple[mx.array, ...]:
    """The whole S-token KDA block in one launch.  S is runtime, not templated.

    Shapes mirror the S=1 entry point with the singleton time axis widened:
    ``q_in/k_in/v_in/a/gate`` are ``[B, S, H*D]``, ``b`` is ``[B, S, H]``, and the
    optional ``mask`` is ``[B, S]`` bool -- the eager path zeroes the pre-conv
    input per (row, token), and so does the kernel.

    Returns ``(y, state_out, conv_state_out, q, k, v)``.  ``y`` is ``[B, S, H*D]``,
    exactly what the eager path hands to ``o_proj``; ``conv_state_out`` is the last
    K-1 pre-conv rows, i.e. the eager ``conv_input[:, -(K-1):, :]``; and q/k/v are
    the post-conv, post-L2-norm ``[B, S, H, D]`` tensors ``gdn_sink`` carries for
    speculative rollback.  They are emitted unconditionally because the verify
    path always carries a sink.

    One threadgroup per (batch row, head), same as S=1 -- so B>1 x S>1 needs no
    new machinery, only a ``[B, S]`` mask instead of ``[B]``.
    """
    H, D, K = num_heads, head_dim, conv_kernel_size
    B, S = q_in.shape[0], q_in.shape[1]
    dt = q_in.dtype
    if mask is None:
        valid = mx.ones((B * S,), dtype=mx.bool_)
    else:
        valid = mask.reshape(B * S)
    kernel = _kernel("block")
    return kernel(
        inputs=[
            q_in, k_in, v_in, conv_state, conv_w, a, b, A_log, dt_bias, state,
            gate, o_weight,
            float(lower_bound), float(head_dim**-0.5), float(norm_eps), valid,
            int(S),
        ],
        template=[
            ("T", dt), ("ST", state.dtype), ("H", num_heads), ("D", head_dim),
            ("K", conv_kernel_size), ("TY", ty),
        ],
        grid=(32, ty, B * num_heads),
        threadgroup=(32, ty, 1),
        output_shapes=[(B, S, H * D), state.shape, conv_state.shape]
        + [(B, S, H, D)] * 3,
        output_dtypes=[dt, state.dtype, dt, dt, dt, dt],
    )
