"""Chunk-parallel KDA prefill scan built from stock MLX ops (lever V7 / L36).

WHAT THIS REPLACES.  ``gated_delta_update`` (../gated_delta.py:269) dispatches
``gated_delta_kernel``, one threadgroup per (b, h) with a sequential ``for t``
token loop holding the [Dv, Dk] state in registers.  At prefill width that is
64 threadgroups doing 8192 dependent steps: measured 1.48 TFLOP/s = 6 % of this
box's peak, latency/serialisation bound, NOT bandwidth bound (2 %).  This module
computes the SAME recurrence as a sequence of batched GEMMs -- the FLOP count
goes UP ~1.2x, the arithmetic rate is what the lever buys.

THE RECURRENCE (must match, exactly, mlx_vlm/models/gated_delta.py:133-175 ==
the ops reference for the kernel at :37-106, as called from
glm5_next/language.py:1755).  Per (b, h), with state ``S`` of shape [Dv, Dk],
q/k L2-normalised (q additionally scaled by Dk**-0.5) by
``Glm5NextLinearAttention._kda_glue_pre`` (language.py:1180-1181):

    S_t' = S_{t-1} * diag(g_t)                      # per-KEY-channel decay
    d_t  = beta_t * (v_t - S_t' k_t)
    S_t  = S_t' + d_t k_t^T
    y_t  = S_t q_t                                  # POST-update state

with ``g_t = exp(a_t)`` and the log-gate (``compute_g_safe``, gated_delta.py:13)

    a_t = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))   in [lower_bound, 0]

i.e. a per-(head, key-channel) vector gate, ``a_t <= 0``.  Note ``y_t`` uses the
POST-update state, so the intra-chunk output matrix is lower triangular
INCLUSIVE of the diagonal while the delta-rule matrix is STRICTLY lower.

Internally this module transposes to ``M = S^T`` of shape [Dk, Dv] so every
matmul is a plain right-multiply; the cache layout [B, H, Dv, Dk] is restored on
exit.

THE CHUNK FORM.  With a chunk of ``c`` tokens and A_i = sum_{l<=i} a_l (local,
inclusive):

    M_i' = diag(e^{A_i}) M_0 + sum_{j<i} diag(e^{A_i - A_j}) k_j u_j^T
    u_i  = beta_i (v_i - M_i'^T k_i)
         = beta_i v_i - beta_i (k_i e^{A_i})^T M_0 - sum_{j<i} beta_i W_ij u_j
    W_ij = k_i^T diag(e^{A_i - A_j}) k_j
    y_i  = (q_i e^{A_i})^T M_0 + sum_{j<=i} P_ij u_j ,  P_ij = q_i^T diag(e^{A_i-A_j}) k_j
    M_c  = diag(e^{A_c}) M_0 + Kbar^T U ,  Kbar_j = k_j e^{A_c - A_j}

so with T = tril(beta_i W_ij, -1) and ONE triangular solve done outside the
state loop,

    [Wmat | Uv] = (I + T)^{-1} [diag(beta) (k e^{A}) | beta v]
    u  = Uv - Wmat M_0            <- the only state-dependent work left
    y  = (q e^{A}) M_0 + P u
    M  = diag(e^{A_c}) M_0 + Kbar^T u

The Python loop over the S/c chunks therefore carries THREE matmuls and three
elementwise ops; everything else is batched over all chunks at once.

NUMERICS (why there is a second, smaller block size).  e^{A_i - A_j} cannot be
factored as e^{A_i} * e^{-A_j} over a whole chunk: the resting log-gate is
~-4.37 nats/token (p0.1) and the floor is ``lower_bound`` = -5.0, so over 64
tokens the exponent spans up to 320 nats -- e^{+320} overflows fp32 to inf and
e^{-320} flushes to 0, and the two meet as inf*0 = NaN on the DIAGONAL of the
chunk, where the true value is 1.  The exponent range that fp32 can carry is
+-88, so the factorisation is only legal over ``c_sub`` tokens with
``c_sub * |lower_bound| <= 80``: c_sub = 16 at lower_bound = -5 (the same 16 an
Apple threadgroup would force, arrived at independently).  So W and P are built
BLOCKWISE over c/c_sub sub-blocks, each block pair (I, J) referenced to the
cumulative gate at the start of ITS OWN sub-block:

    e^{A_i - A_j} = e^{A_i - r_I} * e^{r_I - r_J} * e^{r_J - A_j}
                     <=1  (i>=blk I start)   <=1 (I>=J)    <= e^{c_sub|LB|}

Every factor is either bounded by 1 or by e^{c_sub|LB|}; nothing overflows, and
underflow to 0 is the correct answer.  This is the "sub-chunk-end-referenced
exponent form" I1416 called mandatory.

ACCUMULATION.  q/k/v arrive bf16 from the glue (language.py:1180-1182 casts back
to ``in_dtype``); this module upcasts to fp32 and keeps the state, the gates, the
cumulative sums, the triangular solve and all three in-loop matmuls in fp32 --
strictly more precision than the kernel, which reads bf16 operands and
accumulates in fp32 registers.  The only optional downcast is ``mm_dtype`` for
the two intra-chunk block GEMMs (the largest transient), default fp32.

NOT bit-identical to the sequential kernel: the sums are re-associated and the
delta rule is re-expressed as a triangular solve (WY transform).  Judged by the
teacher-forced KL gate, not by a fingerprint.
"""

import math
import os
from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# --------------------------------------------------------------------------- #
# MLX_VLM_GLM5_KDA_PREFILL_MODE = fused | chunk     (default: fused = unchanged)
# MLX_VLM_GLM5_KDA_CHUNK        = state-recurrence chunk size c   (default 64)
# MLX_VLM_GLM5_KDA_CHUNK_SUB    = factorisation sub-block c_sub   (default 16)
# MLX_VLM_GLM5_KDA_CHUNK_MIN_S  = smallest S the chunk path takes (default 512)
# MLX_VLM_GLM5_KDA_CHUNK_COMPILE= mx.compile the loop-invariant preamble (off)
#
# Read once, like every other lever in this package (language.py:168).
# --------------------------------------------------------------------------- #
_MODE_ENV: Optional[str] = None
_CHUNK_ENV: Optional[int] = None
_SUB_ENV: Optional[int] = None
_MIN_S_ENV: Optional[int] = None
_COMPILE_ENV: Optional[bool] = None

# fp32 exp() overflows at 88.7; 80 leaves the product of the two clamped factors
# (each <= |k| <= 1) a decade of head-room and is a compile-time property of
# lower_bound, not of the data.
_MAX_SUB_EXPONENT = 80.0


def kda_prefill_mode() -> str:
    global _MODE_ENV
    if _MODE_ENV is None:
        _MODE_ENV = os.environ.get("MLX_VLM_GLM5_KDA_PREFILL_MODE", "fused").strip().lower()
    return _MODE_ENV


def kda_chunk_size() -> int:
    global _CHUNK_ENV
    if _CHUNK_ENV is None:
        _CHUNK_ENV = int(os.environ.get("MLX_VLM_GLM5_KDA_CHUNK", "64"))
    return _CHUNK_ENV


def kda_chunk_sub() -> int:
    global _SUB_ENV
    if _SUB_ENV is None:
        _SUB_ENV = int(os.environ.get("MLX_VLM_GLM5_KDA_CHUNK_SUB", "16"))
    return _SUB_ENV


def kda_chunk_min_s() -> int:
    global _MIN_S_ENV
    if _MIN_S_ENV is None:
        _MIN_S_ENV = int(os.environ.get("MLX_VLM_GLM5_KDA_CHUNK_MIN_S", "512"))
    return _MIN_S_ENV


def kda_chunk_compile() -> bool:
    """Fuse the loop-invariant elementwise preamble (default OFF).

    OFF by default on the I1310 precedent: MLX_VLM_GLM5_KDA_GLUE_COMPILE was a
    pure-arithmetic-preserving fusion too and still cost 1.5 accepted tokens per
    speculative round.  This one only ever runs at prefill width, but it defaults
    off until it has been measured on the box, not because it is unsafe.
    """
    global _COMPILE_ENV
    if _COMPILE_ENV is None:
        _COMPILE_ENV = os.environ.get(
            "MLX_VLM_GLM5_KDA_CHUNK_COMPILE", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
    return _COMPILE_ENV


def chunk_geometry(
    chunk: Optional[int] = None,
    sub: Optional[int] = None,
    lower_bound: Optional[float] = None,
) -> Optional[Tuple[int, int]]:
    """``(c, c_sub)`` this configuration may legally run, or None to decline.

    Declines rather than clamps: a chunk size the exponent bound does not cover
    is how a NaN reaches a 32k prefill.
    """
    c = kda_chunk_size() if chunk is None else int(chunk)
    cs = kda_chunk_sub() if sub is None else int(sub)
    if c <= 0 or cs <= 0 or c % cs != 0:
        return None
    if lower_bound is None:
        # Unbounded softplus gate: no compile-time exponent bound exists.
        return None
    if cs * abs(float(lower_bound)) > _MAX_SUB_EXPONENT:
        # Fall back to the largest legal power-of-two sub-block that divides c.
        while cs > 1 and (cs * abs(float(lower_bound)) > _MAX_SUB_EXPONENT or c % cs != 0):
            cs //= 2
        if cs < 1 or c % cs != 0 or cs * abs(float(lower_bound)) > _MAX_SUB_EXPONENT:
            return None
    return c, cs


def _strict_lower_inverse(t: mx.array, n: int) -> mx.array:
    """``(I + T)^{-1}`` for T strictly lower triangular, exactly, by doubling.

    T is nilpotent (T^n = 0), so (I - N)^{-1} = prod_s (I + N^{2^s}) with
    N = -T and s up to ceil(log2(n)) - 1.  ceil(log2(16)) - 1 = 3 -> 6 matmuls.
    No mx.linalg.inv: that has no GPU implementation in MLX 0.32 and would host-
    sync 8192 tiny systems per layer.
    """
    eye = mx.eye(n, dtype=t.dtype)
    npow = -t
    acc = eye + npow
    steps = max(0, int(math.ceil(math.log2(max(n, 2)))) - 1)
    for _ in range(steps):
        npow = npow @ npow
        acc = acc @ (eye + npow)
    return acc


def _prepare(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    log_g: mx.array,
    beta: mx.array,
    c: int,
    c_sub: int,
    pad: int,
    mm_dtype: Optional[mx.Dtype] = None,
):
    """Everything that does NOT depend on the carried state, for ALL chunks.

    Pure (no state, no cache, no Python control flow that depends on values), so
    it is one ``mx.compile`` away from having its elementwise chain fused -- see
    ``MLX_VLM_GLM5_KDA_CHUNK_COMPILE``.  Returns the five tensors the state loop
    consumes:

      wq    [B,H,N,2c,Dk]   rows [Wmat ; q e^{A}] -- one GEMM against the state
      uv    [B,H,N,c,Dv]    (I+T)^{-1} (beta v)
      p_mat [B,H,N,c,c]     lower-INCLUSIVE intra-chunk output matrix
      kbar_t[B,H,N,Dk,c]    k_j e^{A_c - A_j}, transposed for the state write
      gc    [B,H,N,Dk,1]    e^{A_c}, the whole-chunk decay of the carried state
    """
    B, S, H, Dk = k.shape
    Dv = v.shape[-1]
    ns = c // c_sub
    n_chunks = (S + pad) // c

    def _headfirst(x):
        x = x.astype(mx.float32).transpose(0, 2, 1, 3)  # [B, H, S, D]
        if pad:
            x = mx.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)])
        return x.reshape(B, H, n_chunks, c, x.shape[-1])

    q5 = _headfirst(q)
    k5 = _headfirst(k)
    v5 = _headfirst(v)
    lg = log_g.astype(mx.float32).transpose(0, 2, 1, 3)
    if pad:
        # A padded token must be a no-op: log_g = 0 (no decay) and beta = 0
        # (no state write), so its row of every matrix is zero and the carried
        # state is untouched.  Its y row is sliced off at the end.
        lg = mx.pad(lg, [(0, 0), (0, 0), (0, pad), (0, 0)])
    lg = lg.reshape(B, H, n_chunks, c, Dk)
    bt = beta.astype(mx.float32).transpose(0, 2, 1)  # [B, H, S]
    if pad:
        bt = mx.pad(bt, [(0, 0), (0, 0), (0, pad)])
    bt = bt.reshape(B, H, n_chunks, c, 1)

    # ---- cumulative log-gates, and the two references --------------------- #
    A = mx.cumsum(lg, axis=-2)  # inclusive, [B, H, N, c, Dk], <= 0
    A_end = A[..., c - 1 : c, :]  # [B, H, N, 1, Dk]
    A_blk = A.reshape(B, H, n_chunks, ns, c_sub, Dk)
    # r_I = cumulative BEFORE the first token of sub-block I  (exclusive)
    r = (A - lg).reshape(B, H, n_chunks, ns, c_sub, Dk)[..., 0, :]  # [B,H,N,ns,Dk]

    pin = mx.exp(A_blk - r[..., None, :])  # <= 1
    qout = mx.exp(r[..., None, :] - A_blk)  # in [1, e^{c_sub |LB|}]
    # I >= J => r_I - r_J <= 0.  The I < J half would overflow, so clamp it to
    # 0 before exp(); those blocks are entirely below the causal mask.
    rr = mx.minimum(r[..., :, None, :] - r[..., None, :, :], 0.0)
    R = mx.exp(rr)  # [B, H, N, ns, ns, Dk]

    kp = k5.reshape(B, H, n_chunks, ns, c_sub, Dk) * pin
    qp = q5.reshape(B, H, n_chunks, ns, c_sub, Dk) * pin
    kq = k5.reshape(B, H, n_chunks, ns, c_sub, Dk) * qout

    # One GEMM for both intra-chunk matrices: rows [k-block ; q-block].
    left = mx.concatenate([kp, qp], axis=-2)  # [B,H,N,ns,2c_sub,Dk]
    left = left[..., :, None, :, :] * R[..., :, :, None, :]  # [B,H,N,nsI,nsJ,2c_sub,Dk]
    right = mx.swapaxes(kq, -1, -2)[..., None, :, :, :]  # [B,H,N,1,nsJ,Dk,c_sub]
    if mm_dtype is not None:
        left = left.astype(mm_dtype)
        right = right.astype(mm_dtype)
    blocks = (left @ right).astype(mx.float32)  # [B,H,N,nsI,nsJ,2c_sub,c_sub]

    def _assemble(x):  # [B,H,N,nsI,nsJ,c_sub,c_sub] -> [B,H,N,c,c]
        return mx.swapaxes(x, 4, 5).reshape(B, H, n_chunks, c, c)

    w_full = _assemble(blocks[..., :c_sub, :])
    p_full = _assemble(blocks[..., c_sub:, :])

    idx = mx.arange(c)
    strict = idx[:, None] > idx[None, :]
    incl = idx[:, None] >= idx[None, :]
    t_mat = mx.where(strict, w_full, 0.0) * bt  # beta_i W_ij, strictly lower
    p_mat = mx.where(incl, p_full, 0.0)

    # ---- one triangular solve per chunk, OUTSIDE the state loop ----------- #
    ka = k5 * mx.exp(A)  # k_j e^{A_j}, <= |k|
    qa = q5 * mx.exp(A)  # q_i e^{A_i}
    kbar = k5 * mx.exp(A_end - A)  # k_j e^{A_c - A_j}, <= |k|
    rhs = mx.concatenate([ka * bt, v5 * bt], axis=-1)  # [B,H,N,c,Dk+Dv]

    t6 = mx.swapaxes(
        t_mat.reshape(B, H, n_chunks, ns, c_sub, ns, c_sub), 4, 5
    )  # [B,H,N,nsI,nsJ,c_sub,c_sub]
    diag = t6[:, :, :, mx.arange(ns), mx.arange(ns)]  # [B,H,N,ns,c_sub,c_sub]
    dinv = _strict_lower_inverse(diag, c_sub)
    rhs_b = rhs.reshape(B, H, n_chunks, ns, c_sub, Dk + Dv)
    xs = []
    for i in range(ns):  # block forward substitution, ns = c / c_sub steps
        acc = rhs_b[..., i, :, :]
        for j in range(i):
            acc = acc - t6[..., i, j, :, :] @ xs[j]
        xs.append(dinv[..., i, :, :] @ acc)
    x = mx.stack(xs, axis=-3).reshape(B, H, n_chunks, c, Dk + Dv)
    wmat = x[..., :Dk]  # u = Uv - Wmat @ M0
    uv = x[..., Dk:]

    # [Wmat ; q e^{A}] is one GEMM against the state.
    wq = mx.concatenate([wmat, qa], axis=-2)  # [B,H,N,2c,Dk]
    kbar_t = mx.swapaxes(kbar, -1, -2)  # [B,H,N,Dk,c]
    gc = mx.swapaxes(A_end, -1, -2)  # [B,H,N,Dk,1] holder for exp below
    gc = mx.exp(gc)

    return wq, uv, p_mat, kbar_t, gc


# mx.compile keys its own cache on the ARRAY arguments' shapes/dtypes only, so a
# change in (c, c_sub, pad, mm_dtype) at the same S would silently reuse a stale
# graph.  Key on them here instead of hoping.
_PREPARE_C: dict = {}


def _chunk_scan(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    log_g: mx.array,
    beta: mx.array,
    state: mx.array,
    c: int,
    c_sub: int,
    mm_dtype: Optional[mx.Dtype] = None,
) -> Tuple[mx.array, mx.array]:
    """Chunked scan.

    q, k: [B, S, H, Dk] (already L2-normalised, q pre-scaled)
    v:    [B, S, H, Dv]
    log_g:[B, S, H, Dk]  natural-log per-key-channel gate, <= 0
    beta: [B, S, H]
    state:[B, H, Dv, Dk]  (cache layout)
    returns y [B, S, H, Dv] and state [B, H, Dv, Dk], both fp32
    """
    B, S, H, Dk = k.shape
    Dv = v.shape[-1]
    n_chunks = (S + c - 1) // c
    pad = n_chunks * c - S

    if kda_chunk_compile():
        key = (c, c_sub, pad, mm_dtype)
        prep = _PREPARE_C.get(key)
        if prep is None:
            prep = mx.compile(
                partial(_prepare, c=c, c_sub=c_sub, pad=pad, mm_dtype=mm_dtype)
            )
            _PREPARE_C[key] = prep
        wq, uv, p_mat, kbar_t, gc = prep(q, k, v, log_g, beta)
    else:
        wq, uv, p_mat, kbar_t, gc = _prepare(
            q, k, v, log_g, beta, c, c_sub, pad, mm_dtype
        )

    # The only state-carrying work: 3 matmuls + 4 elementwise per chunk.
    m = mx.swapaxes(state.astype(mx.float32), -1, -2)  # [B,H,Dk,Dv]
    ys = []
    for n in range(n_chunks):
        z = wq[:, :, n] @ m  # [B,H,2c,Dv]
        u = uv[:, :, n] - z[..., :c, :]
        ys.append(z[..., c:, :] + p_mat[:, :, n] @ u)
        m = m * gc[:, :, n] + kbar_t[:, :, n] @ u
    y = mx.stack(ys, axis=2).reshape(B, H, n_chunks * c, Dv)
    if pad:
        y = y[:, :, :S, :]
    return y.transpose(0, 2, 1, 3), mx.swapaxes(m, -1, -2)


def chunk_kda_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    lower_bound: Optional[float] = None,
    chunk: Optional[int] = None,
    sub: Optional[int] = None,
    mm_dtype: Optional[mx.Dtype] = None,
) -> Optional[Tuple[mx.array, mx.array]]:
    """Drop-in for ``gated_delta_update`` at prefill width, or None to decline.

    Same argument order and the same (y, state) contract, so the call site is a
    branch, not a rewrite.  Returns ``None`` -- caller keeps the shipped path --
    for every case this form does not cover: a mask (the kernel zeroes y on a
    masked row while the ops reference does not, gated_delta.py:92 vs :175, and
    a chunk form must not have to guess which), an unbounded gate, a chunk
    geometry the fp32 exponent bound does not cover, or S below the width where
    the GEMM form can pay for its transients.
    """
    if mask is not None:
        return None
    geom = chunk_geometry(chunk, sub, lower_bound)
    if geom is None:
        return None
    c, c_sub = geom
    B, S, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    # Two chunks is the smallest width that exercises the carry; the env floor
    # only applies to the serving path (chunk left unset), so a test can ask for
    # a small geometry explicitly.
    min_s = 2 * c if chunk is not None else max(2 * c, kda_chunk_min_s())
    if S < min_s:
        return None
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)
    beta = mx.sigmoid(b.astype(mx.float32))
    # log of compute_g_safe (gated_delta.py:13) -- never exponentiated whole.
    log_g = float(lower_bound) * mx.sigmoid(
        mx.exp(A_log.astype(mx.float32)) * (a.astype(mx.float32) + dt_bias.astype(mx.float32))
    )
    if log_g.ndim == 3:  # scalar gate broadcast to every key channel
        log_g = mx.broadcast_to(log_g[..., None], (B, S, Hv, Dk))
    y, new_state = _chunk_scan(q, k, v, log_g, beta, state, c, c_sub, mm_dtype)
    return y.astype(q.dtype), new_state.astype(state.dtype)
