"""Fused single-token decode step for the GLM-5-Next KDA (Kimi Delta Attention) core.

At B=1, S=1 the post-projection half of ``Glm5NextLinearAttention.__call__`` is a
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

Not a drop-in for prefill: this is decode-only (B=1, S=1, no SSM mask).
``Glm5NextLinearAttention`` falls back to the eager path whenever any of those
preconditions does not hold.
"""

from typing import Optional, Tuple

import mlx.core as mx

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
  const uint h    = threadgroup_position_in_grid.z;
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
  device const ST* si = state_in  + (size_t)h * D * D;
  device ST*       so = state_out + (size_t)h * D * D;
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
      acc += float(conv_state[(size_t)j * CDIM + c]) * float(wc[j]);
    }
    T xnew = (part == 0u) ? mq[h * (uint)D + d]
           : ((part == 1u) ? mk[h * (uint)D + d] : mv[h * (uint)D + d]);
    acc += float(xnew) * float(wc[K - 1]);

    T xb  = static_cast<T>(acc);           // mx.conv1d writes its output in T
    T sig = mlx_sigmoid_fast(xb);          // nn.silu = x * mx.sigmoid(x), compiled
    T sl  = xb * sig;
    if (part == 0u)      sq[d] = float(sl);
    else if (part == 1u) sk[d] = float(sl);
    else                 sv[d] = float(sl);

    // new window = [old[1 .. K-2], x_t]
    for (uint j = 0; j + 2 < (uint)K; ++j) {
      conv_state_out[(size_t)j * CDIM + c] = conv_state[(size_t)(j + 1) * CDIM + c];
    }
    conv_state_out[(size_t)(K - 2) * CDIM + c] = xnew;
  }

  // ---------------------------------------------------------------- phase 0b
  // Safe forget gate  g = exp(lb * sigmoid(exp(A_log) * (a + dt_bias)))  in fp32,
  // beta = sigmoid(b) rounded to T, and the output gate pulled into shared mem.
  {
    float a_exp = metal::precise::exp(A_log[h]);   // mx.exp -> precise
    for (uint d = tid; d < (uint)D; d += NT) {
      float av = float(a[h * (uint)D + d]) + dt_bias[h * (uint)D + d];
      sg[d]    = metal::precise::exp(lower_bound * mlx_sigmoid_fast<float>(a_exp * av));
      sgate[d] = float(gate[h * (uint)D + d]);
    }
    if (tid == 0u) {
      shr[2] = float(mlx_sigmoid_precise(bvec[h]));  // beta = mx.sigmoid(b), in T
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
      y[h * (uint)D + d] = static_cast<T>(x);
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
    q_out[h * (uint)D + d] = static_cast<T>(sq[d]);
    k_out[h * (uint)D + d] = static_cast<T>(sk[d]);
    v_out[h * (uint)D + d] = static_cast<T>(sv[d]);
  }
  // conv_input = concatenate([conv_state, mixed], axis=1), i.e. [1, K, 3*H*D].
  for (uint idx = tid; idx < 3u * (uint)D; idx += NT) {
    uint part = idx / (uint)D;
    uint d    = idx - part * (uint)D;
    uint c    = part * QKVD + h * (uint)D + d;
    for (uint j = 0; j + 1 < (uint)K; ++j) {
      conv_input_out[(size_t)j * CDIM + c] = conv_state[(size_t)j * CDIM + c];
    }
    conv_input_out[(size_t)(K - 1) * CDIM + c] =
        (part == 0u) ? mq[h * (uint)D + d]
      : ((part == 1u) ? mk[h * (uint)D + d] : mv[h * (uint)D + d]);
  }
"""

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
]
_OUTPUT_NAMES = ["y", "state_out", "conv_state_out"]
_SINK_OUTPUT_NAMES = _OUTPUT_NAMES + ["q_out", "k_out", "v_out", "conv_input_out"]

_KERNEL = None
_KERNEL_SINK = None
_KERNEL_TRIED = False

# Threadgroup y-extent.  32 * TY threads per threadgroup, one threadgroup per head;
# TY must divide head_dim and 32 * TY must stay <= 1024.
_TY = 32


def _kernels():
    # Two objects rather than one: mx.fast.metal_kernel derives the function
    # signature from output_names, so the capture variant needs its own kernel.
    global _KERNEL, _KERNEL_SINK, _KERNEL_TRIED
    if _KERNEL_TRIED:
        return _KERNEL, _KERNEL_SINK
    _KERNEL_TRIED = True
    if not mx.metal.is_available():
        return None, None
    _KERNEL = mx.fast.metal_kernel(
        name="glm5_kda_decode_step",
        input_names=_INPUT_NAMES,
        output_names=_OUTPUT_NAMES,
        header=_HEADER,
        source=_SOURCE,
    )
    _KERNEL_SINK = mx.fast.metal_kernel(
        name="glm5_kda_decode_step_capture",
        input_names=_INPUT_NAMES,
        output_names=_SINK_OUTPUT_NAMES,
        header=_HEADER,
        source=_SOURCE + _SINK_SOURCE,
    )
    return _KERNEL, _KERNEL_SINK


def _kernel():
    return _kernels()[0]


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
    if head_dim % 32 != 0 or head_dim % _TY != 0:
        return False
    if 32 * _TY > 1024:
        return False
    if num_heads <= 0:
        return False
    return _kernel() is not None


def fused_kda_decode_step(
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
    capture: bool = False,
) -> Tuple[mx.array, ...]:
    """One fused KDA decode step.

    Args (all B=1, S=1):
      q_in, k_in, v_in: ``[1, 1, H*D]`` pre-conv projections, dtype ``T``.
      conv_state:       ``[1, K-1, 3*H*D]`` cached conv window, dtype ``T``.
      conv_w:           ``[3*H*D, K, 1]`` depthwise conv weight, dtype ``T``.
      a:                ``[1, 1, H*D]`` forget-gate ``f_b_proj`` output, dtype ``T``.
      b:                ``[1, 1, H]`` beta logits, dtype ``T``.
      A_log:            ``[H]`` float32.   dt_bias: ``[H*D]`` float32.
      state:            ``[1, H, D, D]`` recurrent state (float32 in practice).
      gate:             ``[1, 1, H*D]`` ``g_b_proj`` output, dtype ``T``.
      o_weight:         ``[D]`` gated-RMSNorm weight, dtype ``T``.

    Returns ``(y, state_out, conv_state_out)`` where ``y`` is ``[1, 1, H*D]`` and is
    exactly what the eager path feeds to ``o_proj``.

    With ``capture=True`` it additionally returns
    ``(q_out, k_out, v_out, conv_input_out)``: the post-conv / post-L2-norm q, k, v
    as ``[1, 1, H, D]`` and ``concatenate([conv_state, mixed], axis=1)`` as
    ``[1, K, 3*H*D]`` -- the tensors ``gdn_sink`` carries for speculative rollback.
    """
    base, sink = _kernels()
    kernel = sink if capture else base
    H, D, K = num_heads, head_dim, conv_kernel_size
    dt = q_in.dtype
    out_shapes = [(1, 1, H * D), state.shape, conv_state.shape]
    out_dtypes = [dt, state.dtype, dt]
    if capture:
        out_shapes += [(1, 1, H, D)] * 3 + [(1, K, 3 * H * D)]
        out_dtypes += [dt] * 4
    return kernel(
        inputs=[
            q_in,
            k_in,
            v_in,
            conv_state,
            conv_w,
            a,
            b,
            A_log,
            dt_bias,
            state,
            gate,
            o_weight,
            float(lower_bound),
            float(head_dim**-0.5),
            float(norm_eps),
        ],
        template=[
            ("T", dt),
            ("ST", state.dtype),
            ("H", num_heads),
            ("D", head_dim),
            ("K", conv_kernel_size),
            ("TY", _TY),
        ],
        grid=(32, _TY, num_heads),
        threadgroup=(32, _TY, 1),
        output_shapes=out_shapes,
        output_dtypes=out_dtypes,
    )
