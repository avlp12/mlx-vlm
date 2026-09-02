import os
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, CacheList, KVCache
from ..deepseek_v4.hyper_connection import HyperConnection, hc_expand
from ..deepseek_v32.language import DeepseekV32MoE
from ..deepseek_v32.language import Model as DSV32Model
from ..deepseek_v32.language import MoEGate, group_expert_select
from ..gated_delta import gated_delta_update
from ..mla import MultiLinear
from ..mlp import DeepseekMLP
from .config import ModelConfig, TextConfig
from .fused_kda import (
    fused_kda_decode_step,
    fused_kda_probe,
    fused_kda_qproj_supported,
    fused_kda_supported,
    fused_kda_verify_block,
)
from .qmv_custom import DEFAULT_GEOMETRY, maybe_qmv, qmv_applicable
from .speculative_verifier import Glm5NextExactSpeculativeVerifier, verify_logits

_SPECULATIVE_VERIFIER = Glm5NextExactSpeculativeVerifier()

_FUSED_KDA_ENV = None
_FUSED_KDA_QPROJ_ENV = None
_IDX_FAST_ENV = None

# Pools per growth step for the incremental indexer pool buffers (one pool is
# ``index_kpool`` tokens, so 512 pools = 2048 tokens of headroom per growth).
_IDX_POOL_STEP = 512

# How wide a batch the fused kernel is allowed to serve.  This is a *policy* cap
# recording how far parity has actually been run -- not a kernel limit, and the
# distinction is checkable rather than asserted: the literal ``B`` does not occur
# in any of the three kernel sources outside a comment (pinned by
# test_fused_kda_kernel_source_is_batch_agnostic).  One threadgroup serves one
# (batch row, head) pair, so widening the batch adds threadgroups; it does not
# make any threadgroup ask for more.  At the live dims (H=64, D=128) each one
# takes 3084 B of threadgroup memory (3596 B with the projection fold) out of
# Apple's 32 KiB budget and holds st[D/TY][D/32] = 16 floats per thread, at every
# B.  What does grow with B is device traffic: the recurrent state is
# B*H*D*D*4 bytes per layer, so a decode step streams 2.12 GiB through the 34
# KDA layers at B=8 and 4.25 GiB at B=16.  That is a throughput question, which
# is why raising this number needs measurements and not a kernel change.
#
# Env-tunable, so a width can be tried (or walked back) on a device that measures
# differently without editing the model.  It is also the A/B lever for moving it:
# the same process, the same code path, one number apart.  The default only ever
# moves on a live measurement -- test_fused_kda_batch_cap_is_covered_by_parity
# refuses any default the parity matrix below does not actually exercise.
#
# 16 as of 2026-09-01, on a paired single-box measurement (M3 Ultra, the 320B
# tree, two arms per cap alternating which led, 64 timed steps each): B=16
# total throughput 141.9 tok/s fused vs 133.2 eager, +6.50%, and the two pairs
# agreed (+6.91% / +6.09%).  B=32 is equally bit-identical and measured +3.76%
# (178.6 vs 172.2) on two pairs that scattered too much to adopt on.  It was then
# measured again -- five pairs total -- and the answer did not improve:
#
#   per-pair gain  +1.29  +6.30  +2.36  +3.95  +8.57 %   median +5.00%
#   fused arms     176.5  180.8  183.0  183.3  183.7     spread 3.94%
#   eager arms     174.2  170.1  178.7  176.4  169.2     spread 5.48%
#
# Every pair is positive and the median is well over the bar, so the effect is
# almost certainly real -- but the pre-registered rule asks for the WORST pair to
# clear +2% with spread inside the campaign's 0.5%, and the worst pair is -1.26%
# against spreads of 3.9% and 5.5%.  The noise is not in the kernel: the five
# fused arms rise monotonically (a warm-up trend, and the last three agree to
# 0.40%), while the eager arms scatter 5.5% with no trend at all.  A B=32 eager
# step drives ~30 dispatches per layer over 34 layers at a 211.9 GiB peak that
# is evicting page cache, and that is what is being measured.  Until the eager
# reference can be measured stably, 32 stays opt-in.  B=8 held at 1.00% across
# all ten arms, so the pairing itself is sound.
#
# Receipts: logs/tp2/kda_b16_parity_202609011436.json,
# kda_bench_x{1,2}_cap{8,32}_202609011436.json,
# kda_followup_s{1,2,3}_cap{8,32}_202609011456.json.
#
# Not to be combined with the PA733 command-buffer settings at this width:
# MLX_MAX_MB_PER_BUFFER=2048 costs 7.5% at B=16 (131.2 vs 141.9 tok/s) and adds
# 39 GiB of peak.  It is a B=1 lever; see kda_followup_buf_*.
_FUSED_KDA_MAX_BATCH = int(os.environ.get("MLX_VLM_GLM5_FUSED_KDA_MAX_BATCH", "16"))

# The projection fold stops paying once the batch is wide enough to amortise the
# weight read.  mx.quantized_matmul reads f_b/g_b once for all B rows (it becomes
# a GEMM); folding it in makes every (batch row, head) threadgroup re-read its
# head's rows, so weight traffic scales with B while the saving stays at two
# dispatches.  Measured on the 34-layer sweep (M3 Ultra, KDA stack ms,
# base-fused vs +qproj): B=1 8.82 -> 8.55, B=2 10.06 -> 9.49, B=4 10.85 -> 11.65,
# B=8 15.83 -> 20.95.  Crossover sits between 2 and 4.
_FUSED_KDA_QPROJ_MAX_BATCH = 2

# Verify-block width the S>1 kernel will serve.  Matched to the batch cap for the
# same reason: it is the width the parity matrix has actually been run to.
_FUSED_KDA_MAX_WIDTH = int(os.environ.get("MLX_VLM_GLM5_FUSED_KDA_MAX_WIDTH", "16"))

# Kill switch for the S>1 block kernel.  Default ON because it is bit-exact, but
# it has to be flippable IN-PROCESS: the only honest A/B for a kernel this small
# is paired arms inside one load, and a process-per-arm comparison would put the
# measurement back into the cross-process noise [I892] named.
_FUSED_KDA_BLOCK = os.environ.get(
    "MLX_VLM_GLM5_FUSED_KDA_BLOCK", "1"
).lower() in ("1", "true", "yes", "on")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")


def _fused_kda_qproj_enabled() -> bool:
    # Second, stricter opt-in.  Folding f_b_proj / g_b_proj into the kernel drops
    # two dispatches per layer but reorders the quantized dot product relative to
    # mx.quantized_matmul, so the default fused path stays provably bit-identical
    # and this one is measured at rounding scale instead.
    global _FUSED_KDA_QPROJ_ENV
    if _FUSED_KDA_QPROJ_ENV is None:
        _FUSED_KDA_QPROJ_ENV = _env_flag("MLX_VLM_GLM5_FUSED_KDA_QPROJ")
    return _FUSED_KDA_QPROJ_ENV


_SYNC_TRACE_ENV = None


def _sync_trace() -> bool:
    global _SYNC_TRACE_ENV
    if _SYNC_TRACE_ENV is None:
        _SYNC_TRACE_ENV = _env_flag("MLX_VLM_GLM5_SYNC_TRACE")
    return _SYNC_TRACE_ENV


def _idx_fast_enabled() -> bool:
    # Opt-in fast decode path for the DSA indexer's *active* regime (T >
    # index_topk).  The eager path redoes O(T) work every step -- a full-context
    # ``visible`` mask, an O(T) reduction inside ``_visible_tail`` and an O(P)
    # re-concatenation of the pool cache -- none of which can change between
    # steps when the sequence is unpadded and single-stream.  Off by default
    # until it has live mileage; the bypass regime and prefill are untouched.
    global _IDX_FAST_ENV
    if _IDX_FAST_ENV is None:
        _IDX_FAST_ENV = _env_flag("MLX_VLM_GLM5_IDX_FAST")
    return _IDX_FAST_ENV


_MLA_ABSORB_MULTI_ENV = None


def _mla_absorb_multi_enabled() -> bool:
    """Use the absorbed MLA form for L > 1 as well as L == 1.

    The absorbed form keeps k = v = kv_latent ([B, 1, T, kv_lora_rank]) and pushes
    the per-head projections onto the query and the output. The materialised form
    expands the latent cache into [B, num_heads, T, head_dim] for BOTH k and v --
    64x more data at this config -- on every step. That expansion is why a width-8
    verify block costs 24.5x a width-1 step, and why its cost climbs as the cache
    grows within a single generation.

    It is algebraically exact and independent of L: q_h (c W_k[h]^T)^T =
    (q_h W_k[h]) c^T, and attn (c W_v[h]^T) = (attn c) W_v[h]^T. It is clean at
    this config in particular because qk_rope_head_dim == 0, so there is no
    positional component that would have to be split out and carried separately.

    DEFAULT ON. Set MLX_VLM_GLM5_MLA_ABSORB_MULTI=0 to restore the materialised
    form. Measured before flipping, at width 8:

        T=512   1748.30 -> 91.33 ms   19.14x
        T=1024   194.81 -> 91.75 ms    2.12x
        T=2048   109.15 -> 105.97 ms   1.03x
        T=8192   109.27 -> 109.31 ms   1.00x

    The win is entirely BELOW index_topk (2048). Above it the indexer returns
    topk and lines 1284-1309 route L>1 into _gathered_attention, which returns
    before this branch is ever reached -- so there is nothing left to fix there.
    Below it the indexer bypasses, topk_indices is None, and L>1 fell into the
    materialised form. In speculation terms that band went from 0.09x (spec was
    ELEVEN TIMES SLOWER than plain decode) to 1.67x at hazard 0.86.

    On identity: not bit-exact, because the contraction order changes. It is
    inside this model's noise floor -- flipping a SINGLE bfloat16 ulp in one
    embedding element gives 96.88% token match over 256 carried tokens, while
    this change gives 95.70%, against a 100% determinism control. Per layer the
    two branches differ by 3.09e-05 on scores versus a bf16 ulp of ~3.9e-03, and
    under exact dequantization the two orientations agree to 6.5e-09.
    """
    global _MLA_ABSORB_MULTI_ENV
    if _MLA_ABSORB_MULTI_ENV is None:
        v = os.environ.get("MLX_VLM_GLM5_MLA_ABSORB_MULTI")
        _MLA_ABSORB_MULTI_ENV = True if v is None else v not in ("0", "", "false", "False")
    return _MLA_ABSORB_MULTI_ENV


def _fused_kda_enabled() -> bool:
    # Opt-in: the fused decode kernel replaces ~30 dispatches per KDA layer with
    # one, but it is decode-only and only the "safe gate" variant is transcribed,
    # so it stays behind a flag until it has live mileage.
    global _FUSED_KDA_ENV
    if _FUSED_KDA_ENV is None:
        _FUSED_KDA_ENV = _env_flag("MLX_VLM_GLM5_FUSED_KDA")
    return _FUSED_KDA_ENV


# Query chunk for gathered multi-query sparse attention: bounds the gathered
# K/V transient to O(chunk * index_topk) latents at prefill while staying wide
# enough to keep the GPU busy. The short speculative-verify block (L <= 8) is
# always a single chunk.
_GATHER_Q_CHUNK = int(os.environ.get("MLX_VLM_GLM5_GATHER_Q_CHUNK", "1024"))

# Floor for the depth-derived query chunk below.  Only reached past ~260k of
# context; it exists so the Python loop cannot be driven to a pathological
# iteration count by an arbitrarily deep cache.
_GATHER_Q_CHUNK_MIN = int(os.environ.get("MLX_VLM_GLM5_GATHER_Q_CHUNK_MIN", "16"))

# MLX indexes a gather with 32-bit arithmetic while the operand has fewer than
# 2**31 elements and takes a slower path at or above it.  The operand here is
# the broadcast ``(B, chunk, Kv, dim)`` in _gathered_attention, so the boundary
# lands on ``chunk * Kv * dim`` and moves with the cache depth -- which means no
# constant chunk can stay on the fast side of it at every context length.
#
# Measured on the 320B tree, one load, gate 24576 so chunks 11-19 take the
# gathered path, real source+prose, two interleaved cycles reproducing to 0.3%
# (receipt logs/sweep3/R2b_arms_r5.json, analysis R5_RESULT.json):
#
#     chunk   gather region c11-19      vs 512
#       512   84074.8 / 83973.4 ms        --
#       256   82036.6 / 82038.1 ms      +2.4%
#       128   70813.6 / 70650.3 ms     +15.8%
#        64   59339.9 / 59609.7 ms     +29.2%
#
# and the step is not gradual.  The 128 arm crosses the boundary exactly at
# chunk 15, where 128 * 32768 * 512 == 2**31, and its per-chunk cost jumps
# 6564 -> 8894 ms at that chunk and nowhere else.  512 and 256 are both above
# the boundary for every depth past 8192, which is why they differ by only 2.4%
# and why sweeping upward from the 1024 default (AIF I827) never found this.
#
# Identity: bit-exact.  A 30720-token prefill plus 48 greedy tokens at chunk
# 512, 128 and 64 gives byte-identical first-step logits (max |diff| 0.0 on a
# logit scale of 20.75), identical tokens and identical text, with a repeated
# control arm demonstrating determinism.  Receipt logs/sweep3/
# R5_identity_cliff_c1.json.  That check is mandatory rather than a formality:
# mlx#4437 concerns this same 32-bit boundary as a CORRECTNESS issue, and the
# test_gather_q_chunk_* cases below exist to fail loudly if the arithmetic ever
# stops holding.
def _gather_q_chunk_for(kv_len: int, dim: int) -> int:
    """Largest query chunk that keeps ``chunk * kv_len * dim`` under 2**31.

    Rounded down to a power of two so the chunk still divides the prefill block
    evenly and the gathered shapes stay regular, then clamped into
    ``[_GATHER_Q_CHUNK_MIN, _GATHER_Q_CHUNK]`` -- the env knob keeps its meaning
    as the upper bound, and this only ever lowers it.
    """
    if kv_len <= 0 or dim <= 0:
        return _GATHER_Q_CHUNK
    bound = (2**31 - 1) // (kv_len * dim)
    chunk = 1
    while chunk * 2 <= bound:
        chunk *= 2
    return max(_GATHER_Q_CHUNK_MIN, min(_GATHER_Q_CHUNK, chunk))

# Context length above which gathered prefill beats the dense masked path.
#
# HISTORY, because the number moved twice and for different reasons.  The PR's
# microbench put the per-chunk crossover near 16k; end-to-end on M3 Ultra the
# gathered path lost until much deeper, and 32768 measured best against a
# two-point grid of {32768, 65536} (AIF I819, receipt
# unified_gate32768_32k65k131k.json).  That value went upstream in PR #2087.
#
# It was measured against a gather path that cost ~40% more than it needs to.
# A gate is a crossover between two costs and _gather_q_chunk_for above moved
# one of them: with the query chunk off the 2**31 boundary a gather chunk costs
# ~6.6 s instead of ~9.3 s at these dims, so the depth at which gather starts
# beating dense falls a long way.  Re-swept on one load, real source+prose,
# 4 interleaved cycles with a declared and discarded warm-up arm (receipt
# logs/sweep3/R8_gate_band_r8.json, analysis R8_RESULT.json):
#
#     gate     wall, 16384-token prefill, 4 cycles (ms)
#    32768     48819  48834  49766  50671
#    16384     47725  47616  49008  49865
#    12288     46952  47158  48341  49043
#
# 12288 wins every cycle.  The decisive region is chunks 5-6, the only place
# 12288 and 16384 differ, and there 12288 wins by 785/731/819/885 ms -- against
# a within-cycle control spread of 0.17-1.24% on chunks 0-4, which are identical
# work in all three arms.  The per-chunk curves put the crossover between chunk
# 4 and chunk 5 exactly: dense wins by 403 ms at depth 10240 and gather wins by
# 47 ms at 12288.  Arms run in the order 32768, 16384, 12288 and the box heats
# through a cycle, so 12288 is measured under a headwind and wins anyway.
#
# Paired with the depth-derived query chunk this is 257.4 -> 325.7 tok/s at a
# 40960-token prefill, 3/3 cycles in one load (logs/sweep3/R7_HEADLINE_RESULT.json).
#
# Identity is CHAOS-LIMITED, not bit-exact, and that is not new: any gather-gate
# value produces a different token stream from any other, which is what I819's
# 65536 -> 32768 move already shipped.  At prime 16384 with 48 greedy tokens
# 12288 gives identical tokens and identical text to 32768 with differing logits
# (max 0.77 on a scale of 27.5); at prime 30720 with 64 tokens the same class of
# change does flip a token at index 15.  Both are what chaos-limited means.
#
# NOTE ON THE PUBLISHED DEFAULT.  mlx-vlm PR #2087 carries 32768, taken from our
# I819 data, and that remains correct for the configuration it was measured in --
# a gather path pinned to a fixed query chunk.  The shipped default here and the
# published recommendation are allowed to differ; reconciling them upstream is a
# separate decision and not made by this change.
#
# The TP lane keeps its own default (_TP_GATHER_MIN_CONTEXT, 65536) and is
# unaffected by this line -- but that value was fitted against the same
# pre-formula gather cost and is stale for the same reason.  It needs its own
# re-sweep on two boxes.
_GATHER_MIN_CONTEXT = int(os.environ.get("MLX_VLM_GLM5_GATHER_MIN_CONTEXT", "12288"))


# --------------------------------------------------------------- indexer memo
# The DSA lightning indexer splits cleanly in two halves:
#
#   (a) the *scoring* half -- pool_keys, scores, index_scores -- which depends on
#       the cached keys / gate scores, i.e. on hidden states, and is genuinely
#       per-layer work; and
#   (b) the *visibility* half -- pool_indices, pool_valid, pool_end, visible,
#       pool_visible, valid_candidates and the always-select tail -- which is a
#       pure function of the padding mask history (`valid`) and the sequence
#       lengths.  It never reads a key, a gate score or a hidden state.
#
# Every DSA layer inside one Glm5NextModel forward is handed the same fa_mask
# object and advances its indexer cache in lockstep, so all of them rebuild
# byte-identical (b) tensors.  At 32k context that redundancy measures 0.32 ms
# per 512-query chunk on the GPU; over 11 DSA layers x 4 chunks it is 14.2 ms of
# a 2048-token prefill step (29.6 ms at 131k).  Computing it once and handing
# back the *same* mx.array objects removes 10/11 of that.
#
# Correctness is by construction, not by numerical agreement: the memo returns
# the identical array object the first layer built.  It is scoped to a single
# Glm5NextModel.__call__ and only serves indexers that model registered, so the
# MTP head (mtp.py, its own indexer and its own cache) and direct sub-module
# callers (the attribution probes, which invoke layer.self_attn by hand) always
# take the original path.
_VIS_MEMO_CTX = None
_VIS_MEMO_ENV = None
_VIS_MEMO_MB = None
_VIS_MEMO_VERIFY = None


def _vis_memo_enabled() -> bool:
    # Opt-OUT (default on): bit-exact by object identity, self-scoped, and it
    # only ever removes work.  MLX_VLM_GLM5_VIS_MEMO=0 restores per-layer
    # recomputation for A/B measurement.
    global _VIS_MEMO_ENV
    if _VIS_MEMO_ENV is None:
        _VIS_MEMO_ENV = os.environ.get("MLX_VLM_GLM5_VIS_MEMO", "1").lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
    return _VIS_MEMO_ENV


def _vis_memo_budget_bytes() -> int:
    # The chunk loop is inside the layer loop, so the memo has to hold every
    # chunk of the current forward at once: n_chunks * chunk * P bools.  A
    # 2048-token step at 131k context is ~67 MB; an 8192-token step is ~268 MB.
    # Past the budget we stop memoizing (and fall back to recomputation) rather
    # than trade prefill latency for peak memory.
    global _VIS_MEMO_MB
    if _VIS_MEMO_MB is None:
        _VIS_MEMO_MB = int(os.environ.get("MLX_VLM_GLM5_VIS_MEMO_MB", "1024"))
    return _VIS_MEMO_MB * 1024 * 1024


def _vis_memo_verify() -> bool:
    # Debug gate used by the parity test: recompute every memoized tensor and
    # assert equality with the cached one, on every layer.
    global _VIS_MEMO_VERIFY
    if _VIS_MEMO_VERIFY is None:
        _VIS_MEMO_VERIFY = _env_flag("MLX_VLM_GLM5_VIS_MEMO_VERIFY")
    return _VIS_MEMO_VERIFY


class _VisibilityMemo:
    """Per-forward store for the layer-invariant half of the DSA indexer."""

    __slots__ = ("owners", "layout", "chunks", "nbytes", "budget")

    def __init__(self, owners):
        self.owners = owners          # ids of the indexers this model owns
        self.layout = {}              # (B, T) -> pool layout tuple
        self.chunks = {}              # (B, T, S, c0, c1) -> chunk bundle
        self.nbytes = 0
        self.budget = _vis_memo_budget_bytes()

    def serves(self, indexer) -> bool:
        return id(indexer) in self.owners

    def charge(self, nbytes: int) -> bool:
        if self.nbytes + nbytes > self.budget:
            return False
        self.nbytes += nbytes
        return True


def _active_vis_memo(indexer):
    ctx = _VIS_MEMO_CTX
    if ctx is None or not _vis_memo_enabled() or not ctx.serves(indexer):
        return None
    return ctx


def _assert_same(what, cached, fresh):
    """MLX_VLM_GLM5_VIS_MEMO_VERIFY=1: prove the memo hands back exactly the
    tensor the layer would have built itself.  Used by the parity test; never
    runs on the hot path."""
    for i, (a, b) in enumerate(zip(cached, fresh)):
        if a is None or b is None:
            if a is not b:
                raise AssertionError(f"{what}[{i}]: None mismatch")
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(
                f"{what}[{i}]: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
            )
        if not bool(mx.all(a == b)):
            raise AssertionError(f"{what}[{i}]: value mismatch")



# ---------------------------------------------------------------------------
# Shared-expert decode levers (both opt-in, both DEFAULT OFF, both bit-exact).
#
# The shared expert is the worst-occupied significant op in this checkpoint at
# B=1: MLX's affine qmv puts 8 output rows in one 64-thread threadgroup, so
# gate_proj/up_proj (N=2048) launch only 256 threadgroups and down_proj (N=4096)
# only 512, against the 2048-4096 the routed experts get for free from the
# gather's batch dimension.
#
#   R1  MLX_VLM_GLM5_PACK_SHARED   concatenate the shared expert's gate_proj and
#       up_proj on the OUTPUT axis, so one quantized_matmul replaces two.  Affine
#       groups run along the INPUT axis within each output row, so stacking rows
#       is bit-identical by construction -- no tensor is dequantized.  (Measured:
#       max_ulp 0 over 4096 outputs at 6-bit.)  This is the shared-expert analogue
#       of the routed-expert packing on branch glm5-pack-gate-up, which
#       deliberately excludes shared_experts.
#
#   R2  MLX_VLM_GLM5_SHARED_QMV    route the shared expert's M=1 projections
#       through qmv_custom, whose rows-per-threadgroup is tunable.  At rows=1,
#       simdgroups=1 the same work launches 2048-4096 threadgroups instead of
#       256-512.  Bit-identical by construction: changing which thread owns an
#       output row does not change that row's fp32 accumulation order or its
#       simd_sum.  (Measured: max_ulp 0 on all 36 cells.)
#
# NOTE: these are unproven on wall-clock until the epsilon slope run lands.  They
# ship OFF.  The R2 dispatch condition lives in qmv_custom.qmv_applicable() so
# tests can assert on it directly rather than re-deriving it.
# ---------------------------------------------------------------------------
PACK_SHARED_ENV = "MLX_VLM_GLM5_PACK_SHARED"
SHARED_QMV_ENV = "MLX_VLM_GLM5_SHARED_QMV"
# R3: hold the MoE router weight in fp32 instead of re-casting it every call.
ROUTER_FP32_ENV = "MLX_VLM_GLM5_ROUTER_FP32"

_SHARED_EXPERTS = "shared_experts."
_SHARED_GATE_MARKER = _SHARED_EXPERTS + "gate_proj."
_LINEAR_PARAMS = ("weight", "scales", "biases", "bias")


def pack_shared_enabled(config: Any = None) -> bool:
    flag = getattr(config, "pack_shared_gate_up", None) if config is not None else None
    return bool(flag) if flag is not None else _env_flag(PACK_SHARED_ENV)


def shared_qmv_enabled(config: Any = None) -> bool:
    flag = getattr(config, "shared_qmv", None) if config is not None else None
    return bool(flag) if flag is not None else _env_flag(SHARED_QMV_ENV)


def router_fp32_enabled(config: Any = None) -> bool:
    """R3: keep ``mlp.gate.weight`` in fp32 at load so the router GEMV stops
    materialising a [288, 4096] fp32 copy on every layer of every token.

    The checkpoint stores the router weight as BF16 [288, 4096] and
    ``Glm5NextMoEGate.__call__`` does ``self.weight.astype(mx.float32)``, so
    today each of the 42 routers reads 2.36 MB bf16, writes 4.72 MB fp32 and
    then reads that 4.72 MB back for the matmul -- 11.80 MB per layer, 495.6 MB
    per token.  Holding fp32 makes it a single 4.72 MB read, 198.2 MB per token.

    Memory: the fp32 copies total 198.2 MB, but 99.1 MB of that was already
    resident as bf16, so the INCREMENTAL cost is +99.1 MB.

    Bit-exact: bf16 -> fp32 is lossless, so the pre-cast tensor is byte-identical
    to what astype produces today and the matmul receives identical inputs.
    """
    flag = getattr(config, "router_weight_fp32", None) if config is not None else None
    return bool(flag) if flag is not None else _env_flag(ROUTER_FP32_ENV)


def fuse_shared_gate_up(weights: dict) -> dict:
    """Concatenate ``shared_experts.gate_proj`` and ``up_proj`` on the output axis.

    Only ``shared_experts.`` as a whole path component matches, so the routed
    ``switch_mlp.gate_proj`` and the three dense ``mlp.gate_proj`` layers are
    untouched.
    """
    prefixes = set()
    for key in weights:
        offset = key.rfind(_SHARED_GATE_MARKER)
        if offset < 0 or (offset > 0 and key[offset - 1] != "."):
            continue
        if key[offset + len(_SHARED_GATE_MARKER) :] in _LINEAR_PARAMS:
            prefixes.add(key[: offset + len(_SHARED_EXPERTS)])

    for prefix in prefixes:
        for name in _LINEAR_PARAMS:
            gate_key = f"{prefix}gate_proj.{name}"
            up_key = f"{prefix}up_proj.{name}"
            fused_key = f"{prefix}gate_up_proj.{name}"
            if fused_key in weights:
                weights.pop(gate_key, None)
                weights.pop(up_key, None)
            elif gate_key in weights and up_key in weights:
                weights[fused_key] = mx.concatenate(
                    [weights.pop(gate_key), weights.pop(up_key)], axis=0
                )
    return weights


def shared_mlp_quantization_aliases(path: str) -> tuple:
    """Point a packed ``shared_experts.gate_up_proj`` at the checkpoint's
    per-module quantization entries (gate and up are always quantized alike)."""
    marker = ".shared_experts.gate_up_proj"
    if not path.endswith(marker):
        return ()
    base = path[: -len(marker)] + ".shared_experts"
    aliases = [f"{base}.gate_proj", f"{base}.up_proj"]
    stripped = base.removeprefix("language_model.")
    if stripped != base:
        aliases.extend([f"{stripped}.gate_proj", f"{stripped}.up_proj"])
    return tuple(aliases)


class Glm5NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array) -> mx.array:
        dt = hidden_states.dtype
        x = hidden_states.astype(mx.float32)
        var = (x * x).mean(-1, keepdims=True)
        x = x * mx.rsqrt(var + self.eps)
        x = self.weight.astype(mx.float32) * x
        x = x * mx.sigmoid(gate.astype(mx.float32))
        return x.astype(dt)


class Glm5NextForgetGate(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.qkv_dim = self.head_dim * self.num_heads
        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.dt_bias = mx.zeros(self.qkv_dim)
        self.A_log = mx.zeros(self.num_heads)
        self.safe_gate_lower_bound = config.linear_lower_bound

    def __call__(self, hidden_states: mx.array) -> mx.array:
        B, S, _ = hidden_states.shape
        fg = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (fg.astype(mx.float32) + self.dt_bias.astype(mx.float32)).reshape(
            B, S, self.num_heads, self.head_dim
        )
        decay = mx.exp(self.A_log.astype(mx.float32)).reshape(1, 1, self.num_heads, 1)
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * mx.sigmoid(decay * g)
        g_softplus = mx.where(g > 20.0, g, mx.log(1.0 + mx.exp(g)))
        return -decay * g_softplus


def _l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def recurrent_kimi_delta(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
):
    # Reference O(S) recurrence for Kimi Delta Attention, kept as the readable
    # spec and the equivalence oracle for tests. The forward path runs this on
    # the shared fused gated_delta kernel (see Glm5NextLinearAttention).
    dt = query.dtype
    query = _l2norm(query.astype(mx.float32))
    key = _l2norm(key.astype(mx.float32))
    value = value.astype(mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    B, S, H, Dk = key.shape
    Dv = value.shape[-1]
    query = query * (Dk**-0.5)
    if state is None:
        state = mx.zeros((B, H, Dk, Dv), dtype=mx.float32)
    else:
        state = state.astype(mx.float32)
    outs = []
    for i in range(S):
        q_i = query[:, i]
        k_i = key[:, i]
        v_i = value[:, i]
        g_i = mx.exp(g[:, i])[..., None]
        b_i = beta[:, i][..., None]
        state = state * g_i
        kv_mem = (state * k_i[..., None]).sum(axis=-2)
        delta = (v_i - kv_mem) * b_i
        state = state + k_i[..., None] * delta[..., None, :]
        out_i = (state * q_i[..., None]).sum(axis=-2)
        outs.append(out_i)
    out = mx.stack(outs, axis=1).astype(dt)
    return out, state


class Glm5NextLinearAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.qkv_dim = self.num_heads * self.head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.q_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)

        self.conv_dim = self.qkv_dim * 3
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.forget_gate = Glm5NextForgetGate(config)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = Glm5NextRMSNormGated(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(self.qkv_dim, self.hidden_size, bias=False)
        self.fuse_in = True
        self._fused_ready = False
        self._fused_kda = None
        self._fused_kda_qproj = None
        self._fused_kda_ty = None
        self._fused_kda_qproj_ty = None

    def _fused_in_proj(self, inputs):
        # q,k,v,f_a,g_a,b all take `inputs`; fuse into one matmul via a lossless
        # output-axis concat of the (quantized) weights, built once and cached.
        # Returns ``None`` (and disables fusion) if the six projections do not share a
        # single quantization -- a mixed-precision conversion would otherwise dequantize
        # five of them with mods[0]'s group_size/bits.
        if not self._fused_ready:
            mods = [
                self.q_proj,
                self.k_proj,
                self.v_proj,
                self.forget_gate.f_a_proj,
                self.g_a_proj,
                self.b_proj,
            ]
            quantized = [hasattr(m, "scales") for m in mods]
            homogeneous = all(quantized) or not any(quantized)
            if homogeneous and all(quantized):
                homogeneous = (
                    len({m.group_size for m in mods}) == 1
                    and len({m.bits for m in mods}) == 1
                )
            if not homogeneous:
                self.fuse_in = False
                self._fused_ready = True
                return None
            pts, acc = [], 0
            for m in mods[:-1]:
                acc += m.weight.shape[0]
                pts.append(acc)
            self._split_pts = pts
            self._fq = hasattr(mods[0], "scales")
            self._fw = mx.concatenate([m.weight for m in mods], axis=0)
            if self._fq:
                self._fs = mx.concatenate([m.scales for m in mods], axis=0)
                self._fb = mx.concatenate([m.biases for m in mods], axis=0)
                self._gs, self._bits = mods[0].group_size, mods[0].bits
            self._fused_ready = True
        if self._fq:
            out = mx.quantized_matmul(
                inputs,
                self._fw,
                self._fs,
                self._fb,
                transpose=True,
                group_size=self._gs,
                bits=self._bits,
            )
        else:
            out = inputs @ self._fw.T
        return mx.split(out, self._split_pts, axis=-1)

    def _fused_kda_ready(self) -> bool:
        # Config-level capability, resolved once per module.
        if self._fused_kda is None:
            self._fused_kda = _fused_kda_enabled() and fused_kda_supported(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                conv_kernel_size=self.conv_kernel_size,
                lower_bound=self.forget_gate.safe_gate_lower_bound,
            )
        return self._fused_kda

    def _fused_kda_qproj_ready(self, dtype=None, state_dtype=None) -> bool:
        # Needs the dtypes to probe the device: the fold is a separate pipeline
        # with its own threadgroup limit, so it can be declined (or run at a
        # smaller threadgroup) independently of the base kernel.
        if self._fused_kda_qproj is None:
            if dtype is None:
                return False
            supported = (
                _fused_kda_qproj_enabled()
                and self._fused_kda_ready()
                and fused_kda_qproj_supported(
                    self.forget_gate.f_b_proj,
                    self.g_b_proj,
                    head_dim=self.head_dim,
                )
            )
            ty = None
            if supported:
                ty = fused_kda_probe(
                    kind="qproj",
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    conv_kernel_size=self.conv_kernel_size,
                    dtype=dtype,
                    state_dtype=state_dtype,
                    bits=int(self.forget_gate.f_b_proj.bits),
                    group_size=int(self.forget_gate.f_b_proj.group_size),
                )
            self._fused_kda_qproj_ty = ty
            self._fused_kda_qproj = ty is not None
        return self._fused_kda_qproj

    def _fused_kda_eligible(self, B, S, mask, cache, gdn_sink, ref) -> bool:
        # Per-step preconditions.  Anything unusual (prefill, S>1 verify block,
        # a checkpoint whose gate params were not kept in fp32) falls back to the
        # eager path.  A speculative capture is fine: the kernel's capture variant
        # emits the gdn_sink tensors itself.
        del gdn_sink
        if S != 1 or B < 1 or B > _FUSED_KDA_MAX_BATCH:
            return False
        # Batched decode hands down a per-row bool mask (BatchGenerator sets
        # left_padding even for a uniform batch); the kernel applies it exactly
        # where the eager path does.  Anything else -- a non-bool mask, or a
        # shape this layer would broadcast differently -- falls back.
        if mask is not None and (mask.dtype != mx.bool_ or mask.shape != (B, S)):
            return False
        if cache is None or cache[0] is None or cache[1] is None:
            return False
        H, D, K = self.num_heads, self.head_dim, self.conv_kernel_size
        if cache[0].shape != (B, K - 1, 3 * H * D):
            return False
        if cache[1].shape != (B, H, D, D):
            return False
        fg = self.forget_gate
        if fg.A_log.dtype != mx.float32 or fg.dt_bias.dtype != mx.float32:
            return False
        if fg.A_log.size != H or fg.dt_bias.size != H * D:
            return False
        dt = ref.dtype
        if not (
            self.conv1d.weight.dtype == dt
            and self.o_norm.weight.dtype == dt
            and cache[0].dtype == dt
            and self.conv1d.weight.shape == (3 * H * D, K, 1)
            and self.o_norm.weight.size == D
        ):
            return False
        # Device capability, probed once (see fused_kda_probe).  B does not enter
        # the kernel source, so one probe covers every batch size.
        if self._fused_kda_ty is None:
            ty = fused_kda_probe(
                kind="base",
                num_heads=H,
                head_dim=D,
                conv_kernel_size=K,
                dtype=dt,
                state_dtype=cache[1].dtype,
            )
            if ty is None:
                self._fused_kda = False  # stop retrying; stay on the eager path
                return False
            self._fused_kda_ty = ty
        return True

    def _fused_kda_block_eligible(self, B, S, mask, cache, ref) -> bool:
        """Same preconditions as the S=1 kernel, with the time axis opened up.

        Deliberately NOT reusing _fused_kda_eligible: that one asserts S == 1,
        and a shared helper with an S-shaped hole in it is how a width nobody
        parity-tested reaches a kernel.
        """
        if not _FUSED_KDA_BLOCK:
            return False
        if S < 2 or S > _FUSED_KDA_MAX_WIDTH:
            return False
        if B < 1 or B > _FUSED_KDA_MAX_BATCH:
            return False
        # The eager path zeroes the pre-conv input per (row, token); the kernel
        # takes the same [B, S] bool and does the same thing.  Any other mask
        # shape or dtype would broadcast differently, so it falls back.
        if mask is not None and (mask.dtype != mx.bool_ or mask.shape != (B, S)):
            return False
        if cache is None or cache[0] is None or cache[1] is None:
            return False
        H, D, K = self.num_heads, self.head_dim, self.conv_kernel_size
        if cache[0].shape != (B, K - 1, 3 * H * D):
            return False
        if cache[1].shape != (B, H, D, D):
            return False
        fg = self.forget_gate
        if fg.A_log.dtype != mx.float32 or fg.dt_bias.dtype != mx.float32:
            return False
        if fg.A_log.size != H or fg.dt_bias.size != H * D:
            return False
        dt = ref.dtype
        if not (
            self.conv1d.weight.dtype == dt
            and self.o_norm.weight.dtype == dt
            and cache[0].dtype == dt
            and self.conv1d.weight.shape == (3 * H * D, K, 1)
            and self.o_norm.weight.size == D
        ):
            return False
        if self._fused_kda_ty is None:
            ty = fused_kda_probe(
                kind="base",
                num_heads=H,
                head_dim=D,
                conv_kernel_size=K,
                dtype=dt,
                state_dtype=cache[1].dtype,
            )
            if ty is None:
                self._fused_kda = False
                return False
            self._fused_kda_ty = ty
        return True

    def _fused_kda_block(
        self, q_o, k_o, v_o, fa_o, ga_o, b_o, cache, gdn_sink, mask, mixed
    ) -> mx.array:
        """The whole speculative verify block in one kernel launch per layer.

        What this replaces is the GLUE, not the recurrence: gated_delta_update
        already dispatched a fused scan at S>1.  The eager tail around it -- conv
        window concat/slice/copy, silu, two fp32 L2 norms, the beta sigmoid and
        the hand-rolled gated RMSNorm -- is roughly 33 small dependent launches
        per layer, and at 34 KDA layers that is what this collapses to 34.

        It does NOT save state round trips -- gated_delta_kernel already loaded
        the state before its own `for t` loop and stored it after, so the eager
        path paid one per block too.  The win is dispatch count, measured flat at
        ~3.2 ms per verify forward against a 3.03 ms ceiling.
        """
        fg = self.forget_gate
        H, D = self.num_heads, self.head_dim
        B, S, _ = q_o.shape
        a = fg.f_b_proj(fa_o)
        gate = self.g_b_proj(ga_o)
        entry_state = cache[1]
        y, state_out, conv_state_out, q_s, k_s, v_s = fused_kda_verify_block(
            q_o,
            k_o,
            v_o,
            cache[0],
            self.conv1d.weight,
            a,
            b_o,
            fg.A_log,
            fg.dt_bias,
            entry_state,
            gate,
            self.o_norm.weight,
            num_heads=H,
            head_dim=D,
            conv_kernel_size=self.conv_kernel_size,
            lower_bound=fg.safe_gate_lower_bound,
            norm_eps=self.o_norm.eps,
            mask=mask if (mask is not None and mask.dtype == mx.bool_) else None,
            ty=self._fused_kda_ty,
        )
        if gdn_sink is not None:
            # Same eleven members the eager path stashes, in the same order, so
            # rollback_speculative_cache cannot tell which path produced them.
            # conv_input is the one member the kernel does not emit -- it is a
            # concat of tensors already in hand, and costs one dispatch.
            masked = (
                mx.where(mask[..., None], mixed, 0)
                if (mask is not None and mask.dtype == mx.bool_)
                else mixed
            )
            gdn_sink.append(
                (
                    q_s,
                    k_s,
                    v_s,
                    a.reshape(B, S, H, D),
                    b_o,
                    fg.A_log.reshape(H, 1),
                    fg.dt_bias.reshape(H, D),
                    entry_state,
                    mx.concatenate([cache[0], masked], axis=1),
                    self.conv_kernel_size,
                    fg.safe_gate_lower_bound,
                )
            )
        cache[0] = conv_state_out
        cache[1] = state_out
        cache.advance(S)
        return self.o_proj(y.reshape(B, S, -1))

    def _fused_kda_step(
        self, q_o, k_o, v_o, fa_o, ga_o, b_o, cache, gdn_sink=None, mask=None
    ) -> mx.array:
        # One custom Metal kernel for the whole post-projection chain: conv1d
        # window update + silu, both L2 norms, the safe forget gate, beta, the
        # gated delta-rule state update and the gated RMSNorm.  With a drafter
        # attached the capture variant also emits the gdn_sink tensors, so the
        # single-token draft/plain steps of a speculative round keep the fusion
        # (the S>1 verify block still runs eager).
        fg = self.forget_gate
        H, D = self.num_heads, self.head_dim
        capture = gdn_sink is not None
        # The capture variant hands back `a` in gdn_sink, so it keeps the
        # projections outside; folding them in is for the plain decode step.
        use_qproj = (
            not capture
            and q_o.shape[0] <= _FUSED_KDA_QPROJ_MAX_BATCH
            and self._fused_kda_qproj_ready(q_o.dtype, cache[1].dtype)
        )
        qproj = (fa_o, fg.f_b_proj, ga_o, self.g_b_proj) if use_qproj else None
        ty = self._fused_kda_qproj_ty if use_qproj else self._fused_kda_ty
        a = None if use_qproj else fg.f_b_proj(fa_o)
        gate = None if use_qproj else self.g_b_proj(ga_o)
        entry_state = cache[1]
        outs = fused_kda_decode_step(
            q_o,
            k_o,
            v_o,
            cache[0],
            self.conv1d.weight,
            a,
            b_o,
            fg.A_log,
            fg.dt_bias,
            entry_state,
            gate,
            self.o_norm.weight,
            num_heads=H,
            head_dim=D,
            conv_kernel_size=self.conv_kernel_size,
            lower_bound=fg.safe_gate_lower_bound,
            norm_eps=self.o_norm.eps,
            mask=mask,
            ty=ty,
            capture=capture,
            qproj=qproj,
        )
        y, state, conv_state = outs[:3]
        if gdn_sink is not None:
            q_n, k_n, v_n, conv_input = outs[3:]
            gdn_sink.append(
                (
                    q_n,
                    k_n,
                    v_n,
                    a.reshape(-1, 1, H, D),
                    b_o,
                    fg.A_log.reshape(H, 1),
                    fg.dt_bias.reshape(H, D),
                    entry_state,
                    conv_input,
                    self.conv_kernel_size,
                    fg.safe_gate_lower_bound,
                )
            )
        cache[0] = conv_state
        cache[1] = state
        cache.advance(1)
        return self.o_proj(y)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        gdn_sink: Optional[list] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape
        fused = self._fused_in_proj(inputs) if self.fuse_in else None
        if fused is not None:
            q_o, k_o, v_o, fa_o, ga_o, b_o = fused
            if self._fused_kda_ready() and self._fused_kda_eligible(
                B, S, mask, cache, gdn_sink, q_o
            ):
                return self._fused_kda_step(
                    q_o, k_o, v_o, fa_o, ga_o, b_o, cache, gdn_sink, mask
                )
            mixed = mx.concatenate([q_o, k_o, v_o], axis=-1)
        else:
            q_o, k_o, v_o = (
                self.q_proj(inputs),
                self.k_proj(inputs),
                self.v_proj(inputs),
            )
            mixed = mx.concatenate([q_o, k_o, v_o], axis=-1)
            fa_o = self.forget_gate.f_a_proj(inputs)
            ga_o = self.g_a_proj(inputs)
            b_o = self.b_proj(inputs)
            if self._fused_kda_ready() and self._fused_kda_eligible(
                B, S, mask, cache, gdn_sink, q_o
            ):
                return self._fused_kda_step(
                    q_o, k_o, v_o, fa_o, ga_o, b_o, cache, gdn_sink, mask
                )
        if self._fused_kda_ready() and self._fused_kda_block_eligible(
            B, S, mask, cache, q_o
        ):
            return self._fused_kda_block(
                q_o, k_o, v_o, fa_o, ga_o, b_o, cache, gdn_sink, mask, mixed
            )
        if mask is not None and mask.dtype == mx.bool_:
            mixed = mx.where(mask[..., None], mixed, 0)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )
        conv_input = mx.concatenate([conv_state, mixed], axis=1)
        if cache is not None:
            cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1) :, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = mx.split(conv_out, [self.qkv_dim, 2 * self.qkv_dim], axis=-1)
        q = q.reshape(B, S, self.num_heads, self.head_dim)
        k = k.reshape(B, S, self.num_heads, self.head_dim)
        v = v.reshape(B, S, self.num_heads, self.head_dim)

        fg = self.forget_gate
        a = fg.f_b_proj(fa_o).reshape(B, S, self.num_heads, self.head_dim)
        in_dtype = q.dtype
        q = (_l2norm(q.astype(mx.float32)) * (self.head_dim**-0.5)).astype(in_dtype)
        k = _l2norm(k.astype(mx.float32)).astype(in_dtype)

        state = cache[1] if cache is not None else None
        A_log = fg.A_log.reshape(self.num_heads, 1)
        dt_bias = fg.dt_bias.reshape(self.num_heads, self.head_dim)
        lower_bound = fg.safe_gate_lower_bound
        if gdn_sink is not None:
            # Speculative verify: run the fast kernel (no per-step capture) and stash the
            # block inputs + entry state, so a rejected round can be replayed to the
            # accepted token with the kernel in rollback_speculative_cache -- far cheaper
            # than a per-step ops loop on the hot verify path.
            gdn_sink.append(
                (
                    q,
                    k,
                    v,
                    a,
                    b_o,
                    A_log,
                    dt_bias,
                    state,
                    conv_input,
                    self.conv_kernel_size,
                    lower_bound,
                )
            )
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b_o,
            A_log,
            dt_bias,
            state=state,
            lower_bound=lower_bound,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(S)

        gate = self.g_b_proj(ga_o).reshape(B, S, self.num_heads, self.head_dim)
        out = self.o_norm(out, gate).reshape(B, S, -1)
        return self.o_proj(out)


class Glm5NextIndexer(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.index_kpool = args.index_kpool
        self.index_kpool_always_select_tail = args.index_kpool_always_select_tail
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        # Tensor parallelism splits the indexer's head axis, and the scorer
        # CONTRACTS that axis -- so each rank ends up holding a partial sum and
        # the full score only exists after an all-reduce.  ``_tp_reduce`` is
        # that reduce; ``mlx_vlm.tp.glm5_next.shard_dsa`` installs it and it is
        # None (identity) everywhere else.
        self._tp_reduce = None
        # The weight scale must use the GLOBAL head count, or the summed
        # partials come out scaled by sqrt(size) relative to the unsharded
        # reference.  Uniform and positive, so it cannot change the top-k --
        # but it does change the scores, and the scores are compared against a
        # reference in the validator.
        self._scale_heads = self.n_heads
        self.index_kpool_compress_ape = mx.zeros((self.index_kpool, self.head_dim))
        self.index_kpool_compress_gate = mx.zeros((self.head_dim, self.dim))

    def _pool_layout(self, valid, S):
        """The half of the pooling that depends only on `valid` and the length.

        Split out of _pooled_states verbatim (same ops, same order) so a whole
        Glm5NextModel forward can build it once and share it across the DSA
        layers -- see _VisibilityMemo.  Returns everything the keys/gate half
        needs plus the two outputs the selector needs.
        """
        B = valid.shape[0]
        kp = self.index_kpool
        P = (S + kp - 1) // kp
        any_valid = mx.any(valid, axis=-1)
        first_key = mx.where(
            any_valid, mx.argmax(valid.astype(mx.int32), axis=-1), mx.array(S)
        )
        pool_offsets = mx.arange(P * kp).reshape(1, P, kp)
        pool_indices = first_key[:, None, None] + pool_offsets
        safe = mx.clip(pool_indices, 0, S - 1)
        flat = safe.reshape(B, P * kp)
        grouped_valid = (
            mx.take_along_axis(valid.astype(mx.int32), flat, axis=1).reshape(B, P, kp)
            > 0
        )
        grouped_valid = grouped_valid & (pool_indices < S)
        pool_valid = mx.all(grouped_valid, axis=-1)
        pool_indices = mx.where(grouped_valid, pool_indices, -1)
        return flat, grouped_valid, pool_indices, pool_valid

    def _pooled_states(self, keys, gate_scores, valid, layout=None):
        B, S, hd = keys.shape
        kp = self.index_kpool
        P = (S + kp - 1) // kp
        if layout is None:
            layout = self._pool_layout(valid, S)
        flat, grouped_valid, pool_indices, pool_valid = layout
        idxC = mx.broadcast_to(flat[..., None], (B, P * kp, hd))
        grouped_keys = mx.take_along_axis(keys, idxC, axis=1).reshape(B, P, kp, hd)
        grouped_gate = mx.take_along_axis(gate_scores, idxC, axis=1).reshape(
            B, P, kp, hd
        )
        logits = grouped_gate + self.index_kpool_compress_ape[None, None]
        logits = mx.where(grouped_valid[..., None], logits, -1e30)
        probs = mx.softmax(logits, axis=2)
        probs = mx.where(mx.isnan(probs), 0.0, probs)
        pool_keys = mx.sum(probs * grouped_keys, axis=2)
        return pool_keys, pool_indices, pool_valid

    def _visible_tail(self, visible, valid):
        B, S, Kv = visible.shape
        kp = self.index_kpool
        mtw = kp - 1
        any_valid = mx.any(valid, axis=-1)
        first_key = mx.where(
            any_valid, mx.argmax(valid.astype(mx.int32), axis=-1), mx.array(Kv)
        )
        visible_count = mx.sum(visible.astype(mx.int32), axis=-1)
        tail_count = visible_count - (visible_count // kp) * kp
        tail_offsets = mx.arange(mtw)
        tail_start = first_key[:, None] + visible_count - tail_count
        tail_indices = tail_start[..., None] + tail_offsets
        tail_valid = (tail_offsets[None, None, :] < tail_count[..., None]) & (
            tail_indices < Kv
        )
        kv_idx = mx.clip(tail_indices, 0, Kv - 1)
        tail_vis = mx.take_along_axis(visible, kv_idx, axis=-1)
        tail_indices = mx.where(tail_valid & tail_vis, tail_indices, -1)
        return tail_indices

    # ------------------------------------------------------------------ fast
    # Incremental decode path for the ACTIVE indexer regime (T > index_topk).
    #
    # The eager path recomputes, every step and for every one of the 11 DSA
    # layers, three things that provably cannot change:
    #   (a) ``valid`` / ``kv_pos`` / ``visible`` / ``pool_visible`` -- with a
    #       single unpadded stream every cached position is valid and every
    #       position is <= the current query position, so ``visible`` is all-True
    #       and ``valid_candidates`` collapses to ``pool_valid``;
    #   (b) ``_visible_tail`` -- an O(T) reduction whose result, once (a) holds,
    #       is a closed form in T alone;
    #   (c) ``mx.concatenate`` of the stable pool prefix -- pools are append-only
    #       during decode, so this is an O(P) copy of data that is already there.
    # This path keeps (per step) only the work that genuinely depends on the new
    # token: repooling the one incomplete pool, the query-dependent pool scoring,
    # and the top-k.  It is written to be bit-identical to the eager path.

    def _pool_buffers(self, cache):
        """Preallocated append-only pool store; seeded from ``cache._pool``."""
        kp = self.index_kpool
        buf = getattr(cache, "_fpool", None)
        if buf is not None:
            return buf
        ck, ci, cv, t_prev = cache._pool
        B, n = ck.shape[0], ck.shape[1]
        cap = ((n + _IDX_POOL_STEP - 1) // _IDX_POOL_STEP + 1) * _IDX_POOL_STEP
        pk = mx.zeros((B, cap, self.head_dim), dtype=ck.dtype)
        pi = mx.zeros((B, cap, kp), dtype=ci.dtype)
        pv = mx.zeros((B, cap), dtype=cv.dtype)
        pk[:, :n] = ck
        pi[:, :n] = ci
        pv[:, :n] = cv
        buf = [pk, pi, pv, t_prev, cap]
        cache._fpool = buf
        # The eager incremental path keys off ``_pool``; drop it so that if this
        # path ever becomes ineligible mid-stream the fallback is a correct full
        # rebuild rather than a stale prefix.
        cache._pool = None
        return buf

    def _decode_fast(self, x, q, packed_full, cache, T):
        kp, hd = self.index_kpool, self.head_dim
        B = packed_full.shape[0]
        buf = self._pool_buffers(cache)
        pk, pi, pv, t_prev, cap = buf

        # --- (c) repool only the trailing incomplete pool, in place ----------
        n_stable = t_prev // kp
        s0 = n_stable * kp
        tail = packed_full[:, s0:]
        pk_s, pi_s, pv_s = self._pool_tail(tail, s0, T - s0)
        P = n_stable + 1
        if P > cap:
            cap += _IDX_POOL_STEP
            grow = [mx.zeros((B, cap) + a.shape[2:], dtype=a.dtype)
                    for a in (pk, pi, pv)]
            for g, a in zip(grow, (pk, pi, pv)):
                g[:, : a.shape[1]] = a
            pk, pi, pv = grow
        pk[:, n_stable:P] = pk_s
        pi[:, n_stable:P] = pi_s
        pv[:, n_stable:P] = pv_s
        buf[:] = [pk, pi, pv, T, cap]

        # `pool_end` is never needed here: it exists only to gather `visible`,
        # which is all-True on this path, so no buffer is kept for it.
        pool_keys, pool_indices, pool_valid = pk[:, :P], pi[:, :P], pv[:, :P]

        # --- scoring: genuinely query-dependent, recomputed in full ----------
        select_k = min(self.index_topk // kp, P)
        scores = q @ pool_keys[:, None].swapaxes(-1, -2)
        scores = mx.maximum(scores * self.softmax_scale, 0.0)
        weights = self.weights_proj(x) * (self._scale_heads**-0.5)
        index_scores = (weights[:, :, None, :] @ scores).squeeze(2)
        # Reduce before the top-k, exactly as the eager path does.  Two reasons,
        # and the second is the one that bites:
        #
        # (1) correctness -- this is the same head-axis contraction, so under
        #     sharding it is the same partial sum, and ranking it selects the
        #     wrong blocks;
        # (2) COUNT PARITY -- the eager path issues one reduce per query chunk,
        #     so at S=1 it issues exactly one.  If this path issued none, the
        #     number of collectives in a forward would depend on which path a
        #     rank took, and the two ranks do not always take the same one (a
        #     vault-restored cache has no _pool/_no_pad, so it is forced eager
        #     while a live cache is not).  A mirror whose collective count
        #     depends on local cache state deadlocks the moment they differ.
        if self._tp_reduce is not None:
            index_scores = self._tp_reduce(index_scores)
        # (a) valid_candidates == pool_valid: `visible` is all-True here.
        valid_candidates = mx.broadcast_to(pool_valid[:, None], (B, 1, P))
        index_scores = mx.where(valid_candidates, index_scores, -1e30)

        order = mx.argsort(-index_scores, axis=-1)
        selected = order[..., :select_k]
        selected_valid = mx.take_along_axis(valid_candidates, selected, axis=-1)
        sel_exp = mx.broadcast_to(selected[..., None], (B, 1, select_k, kp))
        topk = mx.take_along_axis(
            mx.broadcast_to(pool_indices[:, None], (B, 1, P, kp)), sel_exp, axis=2
        ).reshape(B, 1, select_k * kp)
        sv = mx.broadcast_to(
            selected_valid[..., None], (B, 1, select_k, kp)
        ).reshape(B, 1, select_k * kp)
        topk = mx.where(sv, topk, -1)

        # --- (b) closed-form _visible_tail ----------------------------------
        if self.index_kpool_always_select_tail and kp > 1:
            topk = mx.concatenate([topk, self._tail_fast(B, T, topk.dtype)], axis=-1)
        width = self.index_topk + (
            kp - 1 if (self.index_kpool_always_select_tail and kp > 1) else 0
        )
        if topk.shape[-1] < width:
            topk = mx.concatenate(
                [topk, mx.full((B, 1, width - topk.shape[-1]), -1, dtype=topk.dtype)],
                axis=-1,
            )
        # `valid_cur` is all-True on this path, so the final mask is the identity.
        return topk[:, None, ..., :width].astype(mx.int32)

    def _pool_tail(self, tail, s0, n):
        """`_pooled_states` specialised to the one trailing pool at decode.

        With `n = T - s0` in [1, kpool] and every cached position valid, the
        whole index apparatus of `_pooled_states` -- any/argmax/where for
        `first_key`, the arange, the clip, three `take_along_axis` calls and the
        validity reductions -- collapses to constants known on the host:
            first_key    = 0
            safe[j]      = min(j, n-1)
            grouped_valid[j] = (j < n)
            pool_valid   = (n == kpool)
            pool_indices[j]  = s0 + j  if j < n else -1
        Only the gather, the gate softmax and the weighted sum remain, and each
        is left byte-for-byte as the eager path performs it.
        """
        kp, hd = self.index_kpool, self.head_dim
        B = tail.shape[0]
        # Load-bearing: the caller's `t_prev == T - 1` guard is what bounds the
        # tail to a single pool (n = (t_prev % kp) + 1).  Fail loudly rather than
        # silently pool only the first kp tokens if that ever stops holding.
        if not 1 <= n <= kp:
            raise ValueError(f"indexer fast tail expects 1..{kp} tokens, got {n}")
        key = (kp, n, s0)
        cached = getattr(self, "_ptc", None)
        if cached is None or cached[0] != key:
            idx = mx.array([min(j, n - 1) for j in range(kp)], dtype=mx.int32)
            gv = mx.array([[j < n] for j in range(kp)], dtype=mx.bool_)
            pi = mx.array(
                [[[s0 + j if j < n else -1 for j in range(kp)]]], dtype=mx.int32
            )
            pv = mx.array([[n == kp]], dtype=mx.bool_)
            self._ptc = cached = (key, idx, gv, pi, pv)
        _, idx, gv, pi, pv = cached

        gk = mx.take(tail[..., :hd], idx, axis=1)[:, None]
        gg = mx.take(tail[..., hd : 2 * hd], idx, axis=1)[:, None]
        logits = gg + self.index_kpool_compress_ape[None, None]
        logits = mx.where(gv, logits, -1e30)
        probs = mx.softmax(logits, axis=2)
        probs = mx.where(mx.isnan(probs), 0.0, probs)
        pool_keys = mx.sum(probs * gk, axis=2)
        return (
            pool_keys,
            mx.broadcast_to(pi, (B, 1, kp)),
            mx.broadcast_to(pv, (B, 1)),
        )

    def _tail_fast(self, B, T, dtype):
        # With every position visible and valid, _visible_tail collapses to a
        # closed form: first_key = 0, visible_count = T, tail_count = T % kp,
        # tail_start = T - tail_count, tail_vis all-True, so
        #   tail_valid[o] = (o < tail_count) & (tail_start + o < T) = o < tail_count.
        # Built host-side: 3 int32s, no GPU reduction over T.
        kp = self.index_kpool
        c = T % kp
        vals = [(T - c + o) if o < c else -1 for o in range(kp - 1)]
        return mx.broadcast_to(mx.array(vals, dtype=dtype), (B, 1, kp - 1))

    def __call__(self, x, qr, mask, cache=None):
        B, S, _ = x.shape
        q = self.wq_b(qr).reshape(B, S, self.n_heads, self.head_dim)
        k = self.k_norm(self.wk(x)).reshape(B, S, self.head_dim)
        gate_scores = x @ self.index_kpool_compress_gate.swapaxes(-1, -2)

        if mask is not None and mask.dtype == mx.bool_ and mask.shape == (B, S):
            valid_cur = mask
        else:
            valid_cur = mx.ones((B, S), dtype=mx.bool_)

        # Pack per-token state and append to the indexer cache so pooling/selection
        # run over the full cached sequence -- unifies prefill and incremental decode.
        packed = mx.concatenate(
            [k, gate_scores, valid_cur.astype(k.dtype)[..., None]], axis=-1
        )
        if cache is not None:
            keys, _ = cache.update_and_fetch(packed[:, None], mx.zeros((B, 1, S, 0)))
            packed_full = keys[:, 0]
        else:
            packed_full = packed
        T = packed_full.shape[1]
        # Short-context bypass: when the whole cache fits within index_topk the indexer
        # would select every token, so skip the O(T) pooling/scoring/topk and let the
        # DSA fall through to dense MLA. The cache is already updated above so state
        # stays consistent; the full pool is rebuilt once when T first exceeds index_topk.
        if getattr(self, "bypass_short", True) and T <= self.index_topk:
            return None
        # Fast incremental decode: single unpadded stream, exactly one new token
        # since the pool cache was last built, and a pool state to build on.
        # Anything else (prefill, S>1 verify block, batched or left-padded
        # decode, a rollback that shortened the cache) takes the eager path.
        if (
            _idx_fast_enabled()
            and S == 1
            and B == 1
            and cache is not None
            and mask is None
            and getattr(cache, "_no_pad", False)
            and (
                getattr(cache, "_fpool", None) is not None
                or getattr(cache, "_pool", None) is not None
            )
            and (cache._fpool[3] if getattr(cache, "_fpool", None) is not None
                 else cache._pool[3]) == T - 1
            and (cache._fpool[0] if getattr(cache, "_fpool", None) is not None
                 else cache._pool[0]).shape[0] == B
        ):
            return self._decode_fast(x, q, packed_full, cache, T)
        k_full, gate_full, valid_ch = mx.split(
            packed_full, [self.head_dim, 2 * self.head_dim], axis=-1
        )
        valid = valid_ch[..., 0] > 0

        offset = T - S
        kv_len = T
        kv_pos = mx.arange(T)

        # Incremental pooling at decode: complete pools are stable across steps, so
        # recompute only the suffix (last partial pool + any new pool) and reuse the
        # cached complete pools -- turns the per-step pool cost from O(T) to O(kpool).
        # Exact; falls back to full pooling on prefill, when padding is present, or when
        # the cached pool's batch axis no longer matches the current batch. That last
        # guard matters under continuous batching: BatchGenerator grows/shrinks the
        # batch (extend/filter) on the batch axis but does not carry this per-cache
        # _pool along, so a stale _pool must be discarded and rebuilt for one step.
        if (
            S == 1
            and cache is not None
            and getattr(cache, "_pool", None) is not None
            and getattr(cache, "_no_pad", False)
            and cache._pool[0].shape[0] == B
            and cache._pool[3] == T - S
        ):
            ck, ci, cv, t_prev = cache._pool
            n_stable = t_prev // self.index_kpool
            s0 = n_stable * self.index_kpool
            pk_s, pi_s, pv_s = self._pooled_states(
                k_full[:, s0:], gate_full[:, s0:], valid[:, s0:]
            )
            pi_s = mx.where(pi_s >= 0, pi_s + s0, -1)
            pool_keys = mx.concatenate([ck[:, :n_stable], pk_s], axis=1)
            pool_indices = mx.concatenate([ci[:, :n_stable], pi_s], axis=1)
            pool_valid = mx.concatenate([cv[:, :n_stable], pv_s], axis=1)
        else:
            memo = _active_vis_memo(self)
            layout = None
            if memo is not None:
                key = (B, T)
                layout = memo.layout.get(key)
                if layout is None:
                    layout = self._pool_layout(valid, T)
                    memo.layout[key] = layout
                elif _vis_memo_verify():
                    _assert_same("pool_layout", layout, self._pool_layout(valid, T))
            pool_keys, pool_indices, pool_valid = self._pooled_states(
                k_full, gate_full, valid, layout=layout
            )
            if cache is not None:
                # Instrumented: this is the only forced host-side sync between
                # the sixth hidden all-reduce and the indexer's own reduce, and
                # a TP pair was observed stalled in exactly that window after a
                # vault restore.  Gated so it costs nothing normally.
                if _sync_trace():
                    import logging as _lg
                    _lg.getLogger("glm5.sync").info(
                        "indexer no_pad sync ENTER T=%s S=%s", T, S)
                    cache._no_pad = bool(mx.all(valid))
                    _lg.getLogger("glm5.sync").info(
                        "indexer no_pad sync EXIT  -> %s", cache._no_pad)
                else:
                    cache._no_pad = bool(mx.all(valid))
        if cache is not None:
            cache._pool = (pool_keys, pool_indices, pool_valid, T)
            cache._fpool = None
        P = pool_keys.shape[1]
        select_k = min(self.index_topk // self.index_kpool, P)
        pool_end = mx.clip(pool_indices[..., -1], 0, kv_len - 1)
        pool_keys_t = pool_keys[:, None].swapaxes(-1, -2)
        tail_on = self.index_kpool_always_select_tail and self.index_kpool > 1
        output_width = self.index_topk + (self.index_kpool - 1 if tail_on else 0)

        # Chunk over the query dimension. A one-shot prefill otherwise materializes
        # [B, S, n_heads, P] scores (O(S*P)) and OOMs at long context; chunking bounds
        # peak to O(chunk*P). Decode (S=1) is a single chunk -> identical to before.
        chunk = 512 if S > 512 else S
        # The visibility half of the chunk body (visible -> pool_visible ->
        # valid_candidates, and the always-select tail) reads only `valid`,
        # `pool_end` and the positions, so one forward computes it once and every
        # DSA layer reuses the same arrays.  memo is None for the MTP head, for
        # direct sub-module callers, and when MLX_VLM_GLM5_VIS_MEMO=0.
        memo = _active_vis_memo(self)
        out = []
        for c0 in range(0, S, chunk):
            c1 = min(c0 + chunk, S)
            cs = c1 - c0
            bundle = memo.chunks.get((B, T, S, c0, c1)) if memo is not None else None
            if bundle is None:
                q_pos = offset + mx.arange(c0, c1)
                visible = (kv_pos[None, None, :] <= q_pos[None, :, None]) & valid[
                    :, None, :
                ]
            scores = q[:, c0:c1] @ pool_keys_t
            scores = mx.maximum(scores * self.softmax_scale, 0.0)
            weights = self.weights_proj(x[:, c0:c1]) * (self._scale_heads**-0.5)
            # Contract the head axis as a batched matmul rather than an
            # elementwise product + sum: it never materializes the
            # [B, cs, n_heads, P] product (halving the scorer transient),
            # accumulates in fp32 for low-precision inputs (selection ties
            # resolve as the fp32 reference does), and is structurally immune
            # to the large-strided-shape reduction issue of mx.sum over a
            # non-last axis (ml-explore/mlx#3784) should the chunk ever grow.
            index_scores = (weights[:, :, None, :] @ scores).squeeze(2)
            # Reduce BEFORE the top-k.  Each rank has contracted only its own
            # half of the head axis, so ranking a partial sum selects a
            # different set of KV blocks on each rank -- and the ranks then
            # feed different gathered states into the o_proj all-reduce, whose
            # sum is two halves of different computations.  It does not hang
            # and it does not look wrong; it just is.
            if self._tp_reduce is not None:
                index_scores = self._tp_reduce(index_scores)
            if bundle is None:
                pool_visible = mx.take_along_axis(
                    visible, mx.broadcast_to(pool_end[:, None, :], (B, cs, P)), axis=-1
                )
                valid_candidates = pool_visible & pool_valid[:, None]
                tail = self._visible_tail(visible, valid) if tail_on else None
                bundle = (valid_candidates, tail)
                if memo is not None:
                    nbytes = valid_candidates.size * valid_candidates.itemsize
                    if memo.charge(nbytes):
                        memo.chunks[(B, T, S, c0, c1)] = bundle
                del visible
            elif _vis_memo_verify():
                q_pos = offset + mx.arange(c0, c1)
                vis = (kv_pos[None, None, :] <= q_pos[None, :, None]) & valid[:, None, :]
                pv = mx.take_along_axis(
                    vis, mx.broadcast_to(pool_end[:, None, :], (B, cs, P)), axis=-1
                )
                _assert_same(
                    "chunk_visibility",
                    bundle,
                    (
                        pv & pool_valid[:, None],
                        self._visible_tail(vis, valid) if tail_on else None,
                    ),
                )
            valid_candidates, tail = bundle
            index_scores = mx.where(valid_candidates, index_scores, -1e30)
            order = mx.argsort(-index_scores, axis=-1)
            selected = order[..., :select_k]
            selected_valid = mx.take_along_axis(valid_candidates, selected, axis=-1)
            pi = mx.broadcast_to(pool_indices[:, None], (B, cs, P, self.index_kpool))
            sel_exp = mx.broadcast_to(
                selected[..., None], (B, cs, select_k, self.index_kpool)
            )
            selected_indices = mx.take_along_axis(pi, sel_exp, axis=2)
            topk = selected_indices.reshape(B, cs, select_k * self.index_kpool)
            sv = mx.broadcast_to(
                selected_valid[..., None], (B, cs, select_k, self.index_kpool)
            ).reshape(B, cs, select_k * self.index_kpool)
            topk = mx.where(sv, topk, -1)
            if tail_on:
                topk = mx.concatenate([topk, tail], axis=-1)
            if topk.shape[-1] < output_width:
                pad = mx.full(
                    (B, cs, output_width - topk.shape[-1]), -1, dtype=topk.dtype
                )
                topk = mx.concatenate([topk, pad], axis=-1)
            topk = topk[..., :output_width]
            topk = mx.where(valid_cur[:, c0:c1][..., None], topk, -1)
            out.append(topk)
        topk = out[0] if len(out) == 1 else mx.concatenate(out, axis=1)
        return topk[:, None].astype(mx.int32)


class Glm5NextSparseAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.use_nope = config.mla_use_nope or config.qk_rope_head_dim == 0
        # GLM-5-Next is NoPE by design (qk_rope_head_dim=0, mla_use_nope=True); the
        # config carries no rope parameters. Fail loudly rather than run wrong math
        # if a future config ever requests a RoPE MLA.
        if not self.use_nope:
            raise NotImplementedError(
                "glm5_next implements NoPE MLA only; qk_rope_head_dim>0 with "
                "mla_use_nope=False is not supported."
            )
        self.q_head_dim = config.qk_nope_head_dim
        self.scale = self.q_head_dim**-0.5

        self.q_a_proj = nn.Linear(
            self.hidden_size, self.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank, bias=config.attention_bias
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.indexer = Glm5NextIndexer(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        kv_latent = self.kv_a_layernorm(compressed_kv)
        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, _ = cache[0].update_and_fetch(kv_latent, kv_latent)
        else:
            cache = [None] * 2

        topk_indices = self.indexer(x, qr, mask, cache=cache[1])
        if (
            topk_indices is not None
            and L > 1
            # A quantized latent cache fetches K/V as quantized tuples, which the
            # gather cannot index; those shapes keep the dense masked path below.
            and not (cache[0] is not None and hasattr(cache[0], "group_size"))
            and (L <= 8 or kv_latent.shape[2] >= _GATHER_MIN_CONTEXT)
            and getattr(self, "use_gathered_attention", True)
        ):
            # Multi-query sparse attention: gather the top-k selected latents per
            # query and attend O(L*topk), instead of masking over all Kv (O(L*Kv)).
            # The indexer already selects causally per query, so no extra mask is
            # needed. This is the decode-path selection generalized to any block:
            # it makes the short speculative-verify block affordable at long
            # context, and turns prefill from O(S*T) dense-masked attention into
            # O(S*topk) -- near depth-flat prefill instead of linear decay.
            if (
                cache is not None
                and cache[0] is not None
                and cache[1] is not None
                and cache[1].keys is not None
            ):
                cache[0].keys = mx.depends(
                    cache[0].keys, (cache[1].keys, cache[1].values)
                )
            return self._gathered_attention(q, kv_latent, topk_indices)
        attn_mask = mask
        if topk_indices is not None:
            Kv = kv_latent.shape[2]
            valid_sel = topk_indices >= 0
            if L == 1:
                clamped = mx.clip(topk_indices[:, :, 0, :], 0, Kv - 1)
                idx = clamped[..., None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                sel_mask = valid_sel[:, :, 0, :][:, :, None, :]
                if mask is not None and mask.dtype == mx.bool_:
                    # Single-stream decode passes mask=None here; under continuous
                    # batching the batched cache supplies a left-pad mask that can be
                    # 4-D ([B, 1, 1, Kv]) while `clamped` is 3-D. At S=1 the mask is
                    # purely per-key (no causal), so reduce it to [B, Kv] and gather the
                    # selected key positions -- rank-agnostic and batch-safe.
                    mkeys = mask.reshape(B, -1, Kv)[:, 0, :]
                    gathered = mx.take_along_axis(
                        mx.broadcast_to(mkeys[:, None, :], (B, clamped.shape[1], Kv)),
                        clamped,
                        axis=-1,
                    )
                    sel_mask = sel_mask & gathered[:, :, None, :]
                attn_mask = sel_mask
            else:
                shape = list(topk_indices.shape)
                shape[-1] = Kv + 1
                safe_idx = mx.where(valid_sel, topk_indices, Kv)
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, safe_idx, mx.array(True), axis=-1
                )
                sparse_mask = sparse_mask[..., :Kv]
                if mask is not None and mask.dtype == mx.bool_:
                    sparse_mask = sparse_mask & mask
                attn_mask = sparse_mask

        if (
            cache is not None
            and cache[0] is not None
            and cache[1] is not None
            and cache[1].keys is not None
        ):
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        # attn_mask and the resulting score shape [B, num_heads, L, T] are the
        # same either way -- only the k/v representation differs -- so masking and
        # broadcasting are unaffected by this choice.
        absorb = L == 1 or _mla_absorb_multi_enabled()
        if absorb:
            q = self.embed_q(q)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=attn_mask
        )
        if absorb:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    def _gathered_attention(self, q, kv_latent, topk_indices):
        # Per-query top-k gather: each query attends only to its selected latents
        # (O(L*topk)) rather than a mask over all Kv (O(L*Kv)). Queries are chunked
        # so the gathered-K/V transient stays O(chunk * topk) however long the
        # prefill; a short verify block is a single chunk and unchanged.
        B, H, L, _ = q.shape
        Kv = kv_latent.shape[2]
        dim = kv_latent.shape[-1]
        # topk_indices is [B, 1, L, topk] (axis 1 is the broadcast head); take per-query.
        sel = topk_indices[:, 0, :, :]  # [B, L, topk]
        topk = sel.shape[-1]
        sel_valid = sel >= 0
        # A fully-masked SDPA row is implementation-defined: force key 0 for a
        # query with no selected keys to keep the softmax finite, then zero its
        # output below -- matching the dense masked path, which yields zero rows
        # for such queries (reachable only for padded queries under batching, or
        # with index_kpool_always_select_tail off).
        row_has_keys = mx.any(sel_valid, axis=-1, keepdims=True)  # [B, L, 1]
        sel_valid = sel_valid | ~row_has_keys
        clamped = mx.clip(sel, 0, Kv - 1)
        q_e = self.embed_q(q)  # [B, H, L, dim]
        outs = []
        # Depth-derived, not constant: see _gather_q_chunk_for.  At L <= 8 (a
        # speculative verify block) every candidate chunk exceeds L, so this is
        # a single iteration exactly as before.
        q_chunk = _gather_q_chunk_for(Kv, dim)
        for a0 in range(0, L, q_chunk):
            a1 = min(a0 + q_chunk, L)
            lc = a1 - a0
            kv_g = mx.take_along_axis(
                mx.broadcast_to(kv_latent, (B, lc, Kv, dim)),
                mx.broadcast_to(clamped[:, a0:a1, :, None], (B, lc, topk, dim)),
                axis=2,
            )  # [B, lc, topk, dim]
            q_bl = q_e[:, :, a0:a1].transpose(0, 2, 1, 3).reshape(B * lc, H, 1, dim)
            kv_bl = kv_g.reshape(B * lc, 1, topk, dim)
            valid = sel_valid[:, a0:a1].reshape(B * lc, 1, 1, topk)
            o = scaled_dot_product_attention(
                q_bl, kv_bl, kv_bl, cache=None, scale=self.scale, mask=valid
            )  # [B*lc, H, 1, dim]
            outs.append(o.reshape(B, lc, H, dim).transpose(0, 2, 1, 3))
        attn = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
        attn = attn * row_has_keys.astype(attn.dtype)[:, None, :, 0, None]
        out = self.unembed_out(attn).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class Glm5NextClampedSwiGLU(nn.Module):
    # GLM-5-Next clamps the SwiGLU activation in the text stack (config.swiglu_limit):
    # the gate is clamped above and the up projection on both sides before silu(gate)*up.
    # SwitchGLU invokes activation(x_up, x_gate).
    def __init__(self, limit: Optional[float]):
        super().__init__()
        self.limit = limit

    def __call__(self, x_up: mx.array, x_gate: mx.array) -> mx.array:
        if self.limit is not None:
            x_gate = mx.clip(x_gate, a_min=None, a_max=self.limit)
            x_up = mx.clip(x_up, a_min=-self.limit, a_max=self.limit)
        return nn.silu(x_gate) * x_up


class Glm5NextPackedSharedMLP(nn.Module):
    """Shared-expert MLP whose gate and up projections live in one Linear.

    ``gate_up_proj`` holds ``[2 * intermediate, hidden]`` with gate stacked above
    up.  The clamp/silu/mul and the down projection are byte-for-byte the ones in
    ``Glm5NextMLP``; only the number of dispatches changes.
    """

    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_up_proj = nn.Linear(
            self.hidden_size, 2 * self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.limit = config.swiglu_limit
        self.use_qmv = shared_qmv_enabled(config)

    def _proj(self, layer, x):
        if self.use_qmv:
            y = maybe_qmv(layer, x, **DEFAULT_GEOMETRY)
            if y is not None:
                return y
        return layer(x)

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self._proj(self.gate_up_proj, x)
        gate = gate_up[..., : self.intermediate_size]
        up = gate_up[..., self.intermediate_size :]
        if self.limit is not None:
            gate = mx.clip(gate, a_min=None, a_max=self.limit)
            up = mx.clip(up, a_min=-self.limit, a_max=self.limit)
        return self._proj(self.down_proj, nn.silu(gate) * up)


class Glm5NextMLP(DeepseekMLP):
    # Dense / shared-expert MLP with the clamped SwiGLU (matches the reference text MLP).
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__(
            config, hidden_size=hidden_size, intermediate_size=intermediate_size
        )
        self.limit = config.swiglu_limit
        # Default OFF. Only Glm5NextMoE turns this on, and only for its
        # shared_experts instance -- the three dense MLP layers share this class
        # and are NOT part of the shared-expert item.
        self.use_qmv = False

    def _proj(self, layer, x):
        if getattr(self, "use_qmv", False):
            y = maybe_qmv(layer, x, **DEFAULT_GEOMETRY)
            if y is not None:
                return y
        return layer(x)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self._proj(self.gate_proj, x)
        up = self._proj(self.up_proj, x)
        if self.limit is not None:
            gate = mx.clip(gate, a_min=None, a_max=self.limit)
            up = mx.clip(up, a_min=-self.limit, a_max=self.limit)
        return self._proj(self.down_proj, nn.silu(gate) * up)


class Glm5NextMoEGate(MoEGate):
    # Router logits in fp32 (reference uses moe_router_dtype=float32) so near-tie top-k
    # membership matches the reference rather than flipping under bf16 rounding.
    def __call__(self, x: mx.array):
        # NOTE: mx.array.astype(dtype) does not short-circuit when the dtype
        # already matches (checked on mlx 0.32.1 -- it returns a distinct array),
        # so this guard is load-bearing, not cosmetic.  With R3 enabled the
        # weight arrives fp32 and no cast kernel is emitted at all.
        w = self.weight
        if w.dtype != mx.float32:
            w = w.astype(mx.float32)
        logits = x.astype(mx.float32) @ w.T
        return group_expert_select(
            logits,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class Glm5NextMoE(DeepseekV32MoE):
    # Sparse MoE with the clamped SwiGLU on both the routed experts and the shared expert,
    # and an fp32 router.
    def __init__(self, config):
        super().__init__(config)
        self.switch_mlp.activation = Glm5NextClampedSwiGLU(config.swiglu_limit)
        self.gate = Glm5NextMoEGate(config)
        if config.n_shared_experts is not None:
            inter = config.moe_intermediate_size * config.n_shared_experts
            cls = (
                Glm5NextPackedSharedMLP
                if pack_shared_enabled(config)
                else Glm5NextMLP
            )
            self.shared_experts = cls(config, intermediate_size=inter)
            # R2 applies to the SHARED expert only. The dense MLP layers use the
            # same class and stay on MLX.
            self.shared_experts.use_qmv = shared_qmv_enabled(config)


class Glm5NextDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        layer_type = config.layer_types[layer_idx]
        self.is_linear = layer_type == "linear_attention"
        if self.is_linear:
            self.self_attn = Glm5NextLinearAttention(config)
        else:
            self.self_attn = Glm5NextSparseAttention(config)

        is_sparse = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and config.mlp_layer_types[layer_idx] == "sparse"
        )
        self.mlp = Glm5NextMoE(config) if is_sparse else Glm5NextMLP(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)
        self.compile_ffn = True
        self._ffn_c = None
        # DEFAULT OFF: measured, not demonstrated. In the configuration that
        # ships (compile_ffn already on) three ABBA-paired cycles of 120 timed
        # steps gave +0.324, +0.235, -0.147 ms -- mean +0.137 ms on a 33.9 ms
        # step, 0.4%, with the sign flipping. That is indistinguishable from
        # zero, and a default-on compiled graph per layer costs compile time,
        # memory and a real staleness hazard (_attn_pre_c captures the attn_hc
        # it traced) for no measured gain. The mechanism is kept, tested and
        # TP-guarded so it can be switched on if a future change makes the
        # attention-half glue heavy enough to matter.
        self.compile_attn = False
        self._attn_pre_c = None

    def _attn_prologue(self, x: mx.array):
        """Stateless attention-half prologue: hyper-connection + input norm.

        The FFN half is compiled whole because it touches no cache.  The
        attention half cannot be -- ``self_attn`` advances the KV / recurrent
        state, and ``mx.compile`` would trace that mutation once and then replay
        a stale graph.  So the compiled unit is the glue in FRONT of the
        attention, which is pure in ``x`` and this module's own weights.  The
        glue BEHIND it, ``hc_expand``, already carries ``@mx.compile``.

        ``ffn_hc`` is the same HyperConnection class and already runs inside a
        compiled block today, so the fused sinkhorn+collapse ``metal_kernel`` is
        known to survive tracing.
        """
        xc, post, comb = self.attn_hc(x)
        return self.input_layernorm(xc), post, comb

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        gdn_sink: Optional[list] = None,
    ) -> mx.array:
        residual = x
        # Same gate as the FFN half below, for the same reason and with the same
        # per-shape compile cache: B=1 and S<=8 covers plain decode and the
        # speculative verify block, and adaptive-K varies S only inside that
        # window, so the variant count stays bounded at 8 per layer.
        if self.compile_attn and x.shape[0] == 1 and x.shape[1] <= 8:
            if self._attn_pre_c is None:
                self._attn_pre_c = mx.compile(self._attn_prologue)
            h, post, comb = self._attn_pre_c(x)
        else:
            h, post, comb = self._attn_prologue(x)
        if self.is_linear:
            r = self.self_attn(h, mask, cache, gdn_sink=gdn_sink)
        else:
            r = self.self_attn(h, mask, cache)
        x = hc_expand(r, residual, post, comb)
        # Compile the FFN block for single-stream decode (B=1) at small sequence lengths:
        # S=1 decode and the short S=block_size speculative-verify block. mx.compile keeps
        # a per-shape cache, so each small S compiles once. Batched/prefill shapes stay on
        # the eager path (compiling the 288-expert MoE there spikes memory / can OOM).
        if self.compile_ffn and x.shape[0] == 1 and x.shape[1] <= 8:
            if self._ffn_c is None:
                self._ffn_c = mx.compile(self._ffn_block)
            return self._ffn_c(x)
        return self._ffn_block(x)

    def _ffn_block(self, x: mx.array) -> mx.array:
        # Stateless FFN half (no cache) -> compiles cleanly at a fixed decode shape.
        residual = x
        xc, post, comb = self.ffn_hc(x)
        m = self.mlp(self.post_attention_layernorm(xc))
        return hc_expand(m, residual, post, comb)


class Glm5NextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hc_mult = config.hc_mult
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Glm5NextDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ssm_idx = next((i for i, l in enumerate(self.layers) if l.is_linear), 0)
        self.fa_idx = next((i for i, l in enumerate(self.layers) if not l.is_linear), 0)
        # Indexers this model owns, so the per-forward visibility memo can never
        # serve a foreign one (the MTP head keeps its own indexer + cache).
        self._dsa_indexer_ids = frozenset(
            id(l.self_attn.indexer)
            for l in self.layers
            if not l.is_linear and hasattr(l.self_attn, "indexer")
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        gdn_sink: Optional[list] = None,
        hidden_sink: Optional[list] = None,
        capture_layer_ids: Optional[list] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        fa_cache = cache[self.fa_idx]
        fa_mask = create_attention_mask(
            h, fa_cache[0] if fa_cache else None, return_array=True
        )
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])

        h = mx.broadcast_to(
            h[:, :, None, :], (h.shape[0], h.shape[1], self.hc_mult, h.shape[2])
        )
        h = mx.contiguous(h)

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        # Open a visibility memo for the span of this forward only.  Every DSA
        # layer below is handed the same fa_mask and advances its indexer cache
        # in lockstep, so the mask-only half of the indexer is identical across
        # them; outside this block (MTP head, attribution probes calling
        # layer.self_attn by hand) the memo is closed and nothing changes.
        global _VIS_MEMO_CTX
        outer_memo = _VIS_MEMO_CTX
        _VIS_MEMO_CTX = (
            _VisibilityMemo(self._dsa_indexer_ids) if _vis_memo_enabled() else None
        )
        try:
            for i, (layer, c) in enumerate(zip(self.layers, cache)):
                mask = ssm_mask if layer.is_linear else fa_mask
                h = layer(h, mask=mask, cache=c, gdn_sink=gdn_sink)
                if i in capture_set and hidden_sink is not None:
                    hidden_sink.append(h.mean(axis=2))
        finally:
            _VIS_MEMO_CTX = outer_memo

        h = h.mean(axis=2)
        if hidden_sink is not None and not capture_set:
            hidden_sink.append(h)  # pre-final-norm hidden for the nextn drafter
        return self.norm(h)

    # ---------------------------------------------------------------- pipeline
    # Two-box layer-pipelined prefill splits this stack at a layer boundary and
    # runs the halves on different machines. The tensor that crosses is the
    # mHC-expanded hidden -- (B, S, hc_mult, D), i.e. hc_mult times a plain
    # residual stream -- because the streams are only collapsed after the last
    # layer, above. KDA recurrent state and DSA latent/indexer KV are per-layer,
    # so each half keeps its own caches and nothing else crosses.

    def pipeline_forward(
        self,
        h: Optional[mx.array],
        cache,
        lo: int,
        hi: int,
        inputs: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        """Run decoder layers [lo, hi) over one prefill chunk.

        ``lo == 0``: embed ``inputs``/``inputs_embeds`` and broadcast the mHC
        streams.  Otherwise ``h`` is the boundary tensor from the previous
        stage and is consumed as-is.  Returns the boundary tensor after layer
        ``hi - 1``; call :meth:`pipeline_finish` on the last stage.

        Masks depend only on (chunk length, cache offset) and offsets are equal
        for every layer of a kind, so any local layer of that kind supplies the
        mask -- a stage does not need the other half's caches to build one.
        """
        if lo == 0:
            h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        local = range(lo, hi)
        ssm_i = next((i for i in local if self.layers[i].is_linear), None)
        fa_i = next((i for i in local if not self.layers[i].is_linear), None)
        fa_cache = cache[fa_i] if fa_i is not None else None
        fa_mask = (
            create_attention_mask(
                h, fa_cache[0] if fa_cache else None, return_array=True
            )
            if fa_i is not None
            else None
        )
        ssm_mask = create_ssm_mask(h, cache[ssm_i]) if ssm_i is not None else None

        if lo == 0:
            h = mx.broadcast_to(
                h[:, :, None, :], (h.shape[0], h.shape[1], self.hc_mult, h.shape[2])
            )
            h = mx.contiguous(h)

        for i in local:
            layer = self.layers[i]
            h = layer(
                h, mask=ssm_mask if layer.is_linear else fa_mask, cache=cache[i]
            )
        return h

    def pipeline_finish(self, h: mx.array) -> mx.array:
        """Collapse the mHC streams and apply the final norm (last stage only)."""
        return self.norm(h.mean(axis=2))


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = Glm5NextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        return_hidden = kwargs.pop("return_hidden", False)
        return_shared_kv = kwargs.pop("return_shared_kv", False)
        skip_logits = kwargs.pop("skip_logits", False)
        capture_layer_ids = kwargs.pop("capture_layer_ids", None)
        # glm5_next verifies with a plain capturing forward (no exact kernel), so
        # speculative_verify only needs to be consumed here.
        kwargs.pop("speculative_verify", False)
        # A capture list is supplied whenever a drafter is attached: collect each KDA
        # layer's per-step recurrent state so a round can be rolled back on rejection.
        gdn_sink: Optional[list] = [] if capture_layer_ids is not None else None
        # With a non-empty capture list the DFlash drafter reads the per-layer hidden;
        # with an empty list (MTP) or return_hidden the nextn drafter reads the
        # pre-final-norm hidden, applying its own norm (the DeepSeek-derived convention).
        hidden_sink: Optional[list] = (
            [] if (capture_layer_ids is not None or return_hidden) else None
        )

        out = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            gdn_sink=gdn_sink,
            hidden_sink=hidden_sink,
            capture_layer_ids=capture_layer_ids,
        )
        # Only the last few positions' logits are ever needed for generation; slicing
        # before the (vocab-wide) projection skips it on discarded prefill positions.
        nlk = kwargs.get("num_logits_to_keep", 0)
        logits = None
        if not skip_logits:
            logits = self._logits(out[:, -nlk:, :] if nlk else out)

        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            gdn_states=gdn_sink,
            shared_kv_states={} if return_shared_kv else None,
        )

    def pipeline_prefill_head(
        self, inputs=None, inputs_embeds=None, cache=None, split: int = 0
    ) -> mx.array:
        """Stage-A half of a pipelined prefill chunk -> the boundary tensor."""
        return self.model.pipeline_forward(
            None, cache, 0, split, inputs=inputs, inputs_embeds=inputs_embeds
        )

    @property
    def pipeline_num_layers(self) -> int:
        return len(self.model.layers)

    def _logits(self, normed_hidden: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed_hidden)
        return self.lm_head(normed_hidden)

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        # `hidden` is the pre-final-norm hidden captured for the drafter; apply the
        # final norm, then the (shared) exact-verify LM head.
        return verify_logits(self, self.model.norm(hidden))

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, inputs: mx.array, cache):
        out = _SPECULATIVE_VERIFIER(
            self, inputs, cache=cache, capture_layer_ids=[], skip_logits=True
        )
        return out.hidden_states[-1], out.shared_kv_states, out.gdn_states

    def speculative_verify_logits(self, inputs: mx.array, cache, sampler):
        out = _SPECULATIVE_VERIFIER(self, inputs, cache=cache, capture_layer_ids=[])
        return (
            out.hidden_states[-1],
            out.shared_kv_states,
            out.gdn_states,
            sampler(out.logits),
        )

    def chunked_prefill_policy(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        prompt_cache=None,
        draft_model=None,
        draft_kind=None,
        prefill_kwargs=None,
    ) -> bool:
        del input_ids, inputs_embeds, prompt_cache
        prefill_kwargs = prefill_kwargs or {}
        if draft_model is None:
            return True
        if draft_kind == "lookup":
            # The lookup drafter reads token ids, not captured hidden states, so
            # nothing has to survive a chunk boundary and prefill can chunk
            # exactly as it does with no drafter attached.
            return True
        if draft_kind == "mtp":
            return bool(prefill_kwargs.get("return_hidden", False)) and bool(
                prefill_kwargs.get("return_shared_kv", False)
            )
        return draft_kind is None

    def rollback_speculative_cache(
        self,
        caches: List[Any],
        gdn_states: list,
        accepted,
        block_size: int,
    ) -> int:
        if isinstance(accepted, int):
            accepted_list = [int(accepted)]
        elif isinstance(accepted, mx.array):
            accepted_list = [int(x) for x in accepted.reshape(-1).tolist()]
        else:
            accepted_list = [int(x) for x in accepted]
        max_a = max(accepted_list)
        trim = block_size - (max_a + 1)
        is_batch = len(accepted_list) > 1

        gdn_idx = 0
        for c in caches:
            if c is None:
                continue
            if isinstance(c, ArraysCache):
                # KDA (linear-attention) layer: replay the fast gated-delta kernel from
                # the entry state over just the accepted prefix (n tokens) to recover the
                # rolled-back recurrent state, and slice the conv window to match. Uniform
                # acceptance, so n is shared across the batch.
                q_, k_, v_, a_, b_, A_log_, dt_bias_, init_state, conv_input, K, lb = (
                    gdn_states[gdn_idx]
                )
                gdn_idx += 1
                n = max_a + 1
                _, state_n = gated_delta_update(
                    q_[:, :n],
                    k_[:, :n],
                    v_[:, :n],
                    a_[:, :n],
                    b_[:, :n],
                    A_log_,
                    dt_bias_,
                    state=init_state,
                    lower_bound=lb,
                )
                c[1] = state_n
                c[0] = conv_input[:, n : n + K - 1]
            else:
                # Sparse (MLA + lightning-indexer) layer: trim both KV caches, then roll
                # the indexer pool back to the trimmed length by keeping only the pool
                # blocks that are still fully in-range. The incremental decode path then
                # rebuilds just the last partial block (O(index_kpool)) instead of the
                # whole pool (O(T)) -- critical for long context.
                if trim > 0 and c.is_trimmable():
                    c.trim(trim)
                indexer_cache = c[1]
                pool = getattr(indexer_cache, "_pool", None)
                if pool is not None:
                    pk, pi, pv, t = pool
                    t2 = t - trim
                    if t2 <= 0:
                        indexer_cache._pool = None
                    else:
                        n_stable = t2 // self.args.index_kpool
                        indexer_cache._pool = (
                            pk[:, :n_stable],
                            pi[:, :n_stable],
                            pv[:, :n_stable],
                            t2,
                        )
        return max_a

    def sanitize(self, weights):
        weights = {k: v for k, v in weights.items() if "mtp." not in k}
        weights = DSV32Model.sanitize(self, weights)

        remapped = {}
        conv_parts = {}
        fg_parts = ("A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight")
        for k, v in weights.items():
            nk = k.replace(".hc_attn_", ".attn_hc.").replace(".hc_ffn_", ".ffn_hc.")

            fused = False
            for part in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"):
                suffix = ".self_attn." + part
                if nk.endswith(suffix):
                    prefix = nk[: -len(part)]
                    conv_parts.setdefault(prefix, {})[part[0]] = v
                    fused = True
                    break
            if fused:
                continue

            for p in fg_parts:
                suffix = ".self_attn." + p
                if nk.endswith(suffix):
                    nk = nk[: -len(p)] + "forget_gate." + p
                    break

            remapped[nk] = v

        for prefix, parts in conv_parts.items():
            if all(c in parts for c in ("q", "k", "v")):
                remapped[prefix + "conv1d.weight"] = mx.concatenate(
                    [parts["q"], parts["k"], parts["v"]], axis=0
                )
            else:
                for c, w in parts.items():
                    remapped[prefix + c + "_conv1d.weight"] = w

        weights = remapped
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        # Heal already-converted checkpoints: mHC base/scale and KDA gate params must be
        # fp32 (see cast_predicate); restore them on load if a prior convert cast to bf16.
        for k, v in list(weights.items()):
            keep = (
                ".attn_hc." in k
                or ".ffn_hc." in k
                or k.endswith("A_log")
                or k.endswith("dt_bias")
            )
            if keep and mx.issubdtype(v.dtype, mx.floating) and v.dtype != mx.float32:
                weights[k] = v.astype(mx.float32)
        if pack_shared_enabled(getattr(self, "config", None)):
            weights = fuse_shared_gate_up(weights)
        if router_fp32_enabled(getattr(self, "config", None)):
            for k, v in list(weights.items()):
                if k.endswith("mlp.gate.weight") and mx.issubdtype(
                    v.dtype, mx.floating
                ):
                    if v.dtype != mx.float32:
                        weights[k] = v.astype(mx.float32)
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        # Keep these in fp32 through convert: the fused mHC kernel reads attn_hc/ffn_hc
        # `base` via a float4 pointer (a bf16 base yields a wrong comb matrix), and the
        # KDA gate params (A_log, dt_bias) are fp32-sensitive.
        def predicate(k):
            if "e_score_correction_bias" in k:
                return False
            if ".attn_hc." in k or ".ffn_hc." in k:
                return False
            if k.endswith("A_log") or k.endswith("dt_bias"):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if (
                path.endswith("mlp.gate")
                or "e_score_correction_bias" in path
                or ".indexer" in path
            ):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=2))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches
