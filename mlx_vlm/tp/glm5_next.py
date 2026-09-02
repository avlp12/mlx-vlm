"""Megatron-style TP=2 sharding for glm5_next, decode path.

Sync map (per decode step, GLM-5.3-Flash: 45 layers = 34 KDA + 11 DSA):

    34  KDA  o_proj row-parallel                      -> 1 all-reduce each
    11  DSA  o_proj row-parallel                       -> 1 all-reduce each
    11  DSA  indexer head-axis contraction             -> 1 per query CHUNK
    45  MLP/MoE down_proj row-parallel                -> 1 all-reduce each
    ---
    101 all-reduces / decode step (S=1, so one indexer chunk per DSA layer)

The indexer term is per query chunk (``chunk = 512 if S > 512``), so a prefill
of S tokens costs ``11 * ceil(S/512)`` there rather than 11.  Both ranks derive
the chunk count from S alone, which rank 0 announces, so the counts agree.

Until 2026-09-01 the indexer reduce was installed but never called -- the model
had no call site -- so the real count was 90/step and each rank ranked its own
half of the head scores.  ``tp/validate.py`` checked that summed partials match
the reference, which was true and did not catch it, because it supplied the
reduce itself as a capture hook.

What is deliberately NOT sharded:

* the MLA latent (``kv_a_proj_with_mqa`` -> kv_lora_rank) is shared by every
  head, so it is replicated and both ranks hold the whole DSA KV cache. There is
  no KV memory saving on the DSA layers, only on the weights.
* the MoE router, so both ranks route identically and their partial expert
  outputs sum to the right thing.
* mHC (``attn_hc``/``ffn_hc``), every norm, ``embed_tokens`` and ``lm_head``:
  these read the full hidden, which is replicated after each all-reduce.

The KDA recurrence needs no communication at all: its state is
``(B, H, D, D)`` -- head-local -- and the conv is depthwise, so a head split
keeps every channel's history on the rank that owns it.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .shard import (
    _split,
    shard_axis0,
    shard_experts_in,
    shard_experts_out,
    shard_in,
    shard_out,
)


class AllReduce(nn.Module):
    """Wrap a module so its output is summed across ranks."""

    def __init__(self, inner, reduce_fn):
        super().__init__()
        self.inner = inner
        self._reduce = reduce_fn

    def __call__(self, *a, **k):
        return self._reduce(self.inner(*a, **k))


def _shard_depthwise_conv(conv, rank: int, size: int, qkv_dim: int):
    """Split a depthwise conv whose channels are [q | k | v], each qkv_dim wide.

    A contiguous half is wrong: the layer splits the conv output back into q, k
    and v, so the rank must own the same head slice inside each of the three
    blocks. Weight layout is (channels, kernel, 1).
    """
    local = qkv_dim // size
    idx = mx.concatenate(
        [
            mx.arange(t * qkv_dim + rank * local, t * qkv_dim + (rank + 1) * local)
            for t in range(3)
        ]
    )
    conv.weight = mx.contiguous(mx.take(conv["weight"], idx, axis=0))
    if "bias" in conv:
        conv.bias = mx.contiguous(mx.take(conv["bias"], idx, axis=0))
    # depthwise: groups == channels, so the group count shrinks with the shard
    if hasattr(conv, "groups"):
        conv.groups = conv["weight"].shape[0]
    return conv


def shard_kda(attn, rank: int, size: int, reduce_fn):
    H, D = attn.num_heads, attn.head_dim
    if H % size:
        raise ValueError(f"KDA heads {H} not divisible by tp size {size}")
    qkv_dim = attn.qkv_dim

    for name in ("q_proj", "k_proj", "v_proj", "g_b_proj", "b_proj"):
        shard_out(getattr(attn, name), rank, size)
    shard_out(attn.forget_gate.f_b_proj, rank, size)

    fg = attn.forget_gate
    fg.A_log = _split(fg["A_log"], 0, rank, size)                 # (H,)
    fg.dt_bias = _split(fg["dt_bias"], 0, rank, size)             # (H*D,), head-major
    fg.num_heads = H // size
    fg.qkv_dim = qkv_dim // size

    _shard_depthwise_conv(attn.conv1d, rank, size, qkv_dim)

    shard_in(attn.o_proj, rank, size)

    attn.num_heads = H // size
    attn.qkv_dim = qkv_dim // size
    attn.conv_dim = attn.qkv_dim * 3
    # the fused kernel takes H as a template constant and grids B*H, so a half
    # shard just compiles a second pipeline; threadgroup memory is sized by D
    attn._fused_ready = False
    attn._fused_kda = None
    attn._fused_kda_qproj = None
    return attn


def shard_dsa(attn, rank: int, size: int, reduce_fn):
    H = attn.num_heads
    if H % size:
        raise ValueError(f"DSA heads {H} not divisible by tp size {size}")
    shard_out(attn.q_b_proj, rank, size)
    shard_axis0(attn.embed_q, rank, size)
    shard_axis0(attn.unembed_out, rank, size)
    shard_in(attn.o_proj, rank, size)
    attn.num_heads = H // size

    ix = attn.indexer
    if ix.n_heads % size:
        raise ValueError(f"indexer heads {ix.n_heads} not divisible by tp size {size}")
    shard_out(ix.wq_b, rank, size)
    shard_out(ix.weights_proj, rank, size)
    ix.n_heads = ix.n_heads // size
    # the scorer's head axis is contracted, so each rank holds a partial sum;
    # reduce before the top-k. The weight scale keeps the GLOBAL head count.
    ix._tp_reduce = reduce_fn
    return attn


def shard_mlp(mlp, rank: int, size: int):
    shard_out(mlp.gate_proj, rank, size)
    shard_out(mlp.up_proj, rank, size)
    shard_in(mlp.down_proj, rank, size)
    return mlp


def shard_moe(moe, rank: int, size: int):
    sm = moe.switch_mlp
    shard_experts_out(sm.gate_proj, rank, size)
    shard_experts_out(sm.up_proj, rank, size)
    shard_experts_in(sm.down_proj, rank, size)
    if hasattr(moe, "shared_experts") and moe.shared_experts is not None:
        shard_mlp(moe.shared_experts, rank, size)
    # router stays replicated so both ranks pick the same experts
    return moe


def shard_layer(layer, rank: int, size: int, reduce_fn):
    # The FFN block is mx.compile'd for B=1, S<=8 (models/glm5_next/language.py:
    # 1417).  Under TP that block contains the MoE's all-reduce, i.e. a
    # *distributed* op inside a traced-and-cached graph.  The full-stack
    # lockstep validator already refused to compile it (tp/validate_full.py);
    # the serving path never did, and it bit on 2026-09-01: the first
    # generation in a process completes, the second stalls a couple of decode
    # steps in, with rank 1 waiting for a collective rank 0 did not issue.
    # Compiling a collective is not something to get subtly right -- do not
    # compile it.
    layer.compile_ffn = False
    layer._ffn_c = None
    # The attention-half prologue (attn_hc + input_layernorm) is replicated, not
    # sharded, so it should contain no collective -- the all-reduce wraps
    # self_attn, which sits outside it.  Compiling it under TP is nevertheless
    # the same CLASS of risk that bit on 2026-09-01, and a few hundred
    # microseconds is not worth finding out in the distributed path.  Off here,
    # on everywhere else.
    layer.compile_attn = False
    layer._attn_pre_c = None
    if layer.is_linear:
        shard_kda(layer.self_attn, rank, size, reduce_fn)
    else:
        shard_dsa(layer.self_attn, rank, size, reduce_fn)
    layer.self_attn = AllReduce(layer.self_attn, reduce_fn)

    mlp = layer.mlp
    if hasattr(mlp, "switch_mlp"):
        shard_moe(mlp, rank, size)
    else:
        shard_mlp(mlp, rank, size)
    layer.mlp = AllReduce(mlp, reduce_fn)
    return layer


# The gather gate is a per-LANE tuning constant, not a per-model one.
#
# Measured 2026-09-01 as a per-chunk prefill cost curve (prep/tp2/lc_curve.py),
# 2048-token chunks, GLM-5.3-Flash q4:
#
#   single-box   dense 3924 + 252.8*chunk ms | gather plateau 8670 ms
#   TP=2         dense 2306 + 122.9*chunk ms | gather plateau 6394 ms
#
# TP splits the attention heads, so the dense O(S*T) term halves (252.8 ->
# 122.9, a 2.06x ratio) while the gather path's fixed cost falls only 1.36x --
# it is dominated by indexer and gather work that parallelises less.  The
# crossover therefore moves right: ~38k single-box (which is what makes 32768
# the right default there, as originally measured) and ~68k under TP.  At the
# 32768 default a TP prefill to 65k spends 153.8 s instead of a projected
# 132.4 s all-dense: 13.9% given away.
#
# Why 65536 and not higher, stated as arithmetic because the curves stop at
# 65k: extrapolating dense to 131k (chunk 64) gives 2306 + 122.9*64 = 10.2
# s/chunk against a flat 6.4 s gather plateau, so gather is well ahead beyond
# the ~68k crossover.  65536 engages it just about exactly where it starts
# paying.  The 131k end of that is extrapolation, not measurement.
_TP_GATHER_MIN_CONTEXT = 65536


def _apply_tp_gather_default() -> None:
    """Raise the gather gate for the TP lane unless the operator chose one."""
    import os

    from ..models.glm5_next import language as _lang

    if os.environ.get("MLX_VLM_GLM5_GATHER_MIN_CONTEXT"):
        return                      # explicit wins, on either lane
    _lang._GATHER_MIN_CONTEXT = _TP_GATHER_MIN_CONTEXT


def shard_model(model, rank: int, size: int, reduce_fn=None) -> dict:
    """In-place TP surgery on a loaded glm5_next. Returns a small report."""
    if reduce_fn is None:
        from .transport import all_sum as reduce_fn  # noqa: N813

    # Both ranks call shard_model, so setting the lane default here reaches
    # them both without an env passthrough -- and passthroughs are how rank 1
    # got NAME= and died at import earlier today.
    _apply_tp_gather_default()
    lm = model.language_model.model if hasattr(model, "language_model") else model
    n_kda = n_dsa = n_moe = n_dense = 0
    for layer in lm.layers:
        if layer is None:
            continue
        shard_layer(layer, rank, size, reduce_fn)
        n_kda += bool(layer.is_linear)
        n_dsa += not layer.is_linear
        if hasattr(layer.mlp.inner, "switch_mlp"):
            n_moe += 1
        else:
            n_dense += 1
    return {
        "rank": rank,
        "size": size,
        "kda_layers": n_kda,
        "dsa_layers": n_dsa,
        "moe_layers": n_moe,
        "dense_layers": n_dense,
        "all_reduces_per_step": n_kda + 2 * n_dsa + n_moe + n_dense,
    }
