"""Validate the TP=2 sharding of glm5_next per module, on a tiny config.

Each sharded piece is "column-parallel work, then row-parallel work", so the
unsharded output must equal the SUM over ranks of the sharded outputs. That
makes every module checkable in one process, with no distributed group, no
169 GB checkpoint and no GPU contention -- which is the only way to debug
sharding at a sane pace.

Bit-identity is not expected: summing two partial products reorders float
additions. The bar is rounding scale.
"""

from __future__ import annotations

import copy
import json

import mlx.core as mx

from ..models.cache import ArraysCache, CacheList, KVCache
from ..models.glm5_next.config import TextConfig
from ..models.glm5_next.language import Glm5NextDecoderLayer
from .glm5_next import shard_dsa, shard_kda, shard_mlp, shard_moe

TINY = dict(
    model_type="glm5_next",
    vocab_size=512,
    hidden_size=128,
    intermediate_size=64,
    moe_intermediate_size=32,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_shared_experts=1,
    n_routed_experts=8,
    routed_scaling_factor=1.0,
    kv_lora_rank=32,
    q_lora_rank=32,
    qk_rope_head_dim=0,
    v_head_dim=16,
    qk_nope_head_dim=16,
    num_experts_per_tok=2,
    first_k_dense_replace=3,
    max_position_embeddings=1024,
    rms_norm_eps=1e-6,
    index_topk=8,
    index_head_dim=16,
    index_n_heads=2,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full"],
    mlp_layer_types=["dense", "dense", "dense", "sparse"],
    linear_attn_config={"num_heads": 4, "head_dim": 16, "short_conv_kernel_size": 4,
                        "gate_lower_bound": -5.0},
    index_kpool=4,
    mla_use_nope=True,
    hc_mult=4,
)


# Quantized variant. Dimensions are chosen so every ROW-parallel split stays
# aligned to group_size=64 after halving, and so no quantized linear has an
# input narrower than one group:
#   KDA/DSA o_proj in = 256 -> 128 per rank (128 % 64 == 0)
#   dense MLP down in = 256 -> 128 ; MoE down in = 128 -> 64
#   g_b_proj / f_b_proj in = head_dim = 64 -> exactly one group
TINY_Q = dict(
    TINY,
    hidden_size=256,
    intermediate_size=256,
    moe_intermediate_size=128,
    num_attention_heads=4,
    v_head_dim=64,
    qk_nope_head_dim=64,
    kv_lora_rank=64,
    q_lora_rank=64,
    index_head_dim=64,
    index_n_heads=4,
    index_topk=8,
    linear_attn_config={"num_heads": 4, "head_dim": 64, "short_conv_kernel_size": 4,
                        "gate_lower_bound": -5.0},
)


def _cfg(quant=False):
    return TextConfig.from_dict(TINY_Q if quant else TINY)


def _quantize(module, group_size=64, bits=4):
    """Quantize every Linear / SwitchLinear under `module`, in place.

    MultiLinear (DSA embed_q/unembed_out) is left in fp: its head-axis split is
    a plain axis-0 slice with no group alignment to get wrong, so quantizing it
    would test nothing new here.
    """
    import mlx.nn as nn

    def maybe_q(child):
        if hasattr(child, "to_quantized") and type(child).__name__ in (
            "Linear", "SwitchLinear",
        ):
            return child.to_quantized(group_size=group_size, bits=bits)
        return None

    def walk(m):
        for name, child in list(m.children().items()):
            if isinstance(child, list):
                # Module lists (e.g. Glm5NextModel.layers) are plain lists
                for i, sub in enumerate(child):
                    q = maybe_q(sub)
                    if q is not None:
                        child[i] = q
                    elif hasattr(sub, "children"):
                        walk(sub)
                continue
            if isinstance(child, dict) or not hasattr(child, "children"):
                continue
            q = maybe_q(child)
            if q is not None:
                setattr(m, name, q)
            else:
                walk(child)

    walk(module)
    return module


def _rel(a, b):
    return float(mx.max(mx.abs(a - b)) / mx.maximum(mx.max(mx.abs(b)), 1e-6))


def _kda_cache():
    return ArraysCache(size=2)


def _prime_arrays(cfg, primed):
    """Built once and shared, so every rank starts from the SAME cache."""
    return (
        mx.random.normal((1, 1, primed, cfg.kv_lora_rank)),
        mx.random.normal((1, 1, primed, 2 * cfg.index_head_dim + 1)),
    )


def _dsa_cache(cfg, primed=0, arrays=None):
    c = CacheList(KVCache(), KVCache())
    if primed:
        lat, packed = arrays if arrays is not None else _prime_arrays(cfg, primed)
        c[0].state = (lat, lat)
        c[1].state = (packed, mx.zeros((1, 1, primed, 0)))
    return c


def validate(size: int = 2, seed: int = 0, verbose: bool = True, quant: bool = False) -> dict:
    mx.random.seed(seed)
    cfg = _cfg(quant)
    out = {}
    x = mx.random.normal((1, 1, cfg.hidden_size))

    # ---- KDA attention -----------------------------------------------------
    layer = Glm5NextDecoderLayer(cfg, 0)          # linear_attention
    ref_attn = _quantize(layer.self_attn) if quant else layer.self_attn
    ref = ref_attn(x, None, _kda_cache())
    parts = []
    for r in range(size):
        m = shard_kda(copy.deepcopy(ref_attn), r, size, lambda z: z)
        parts.append(m(x, None, _kda_cache()))
    out["kda_attn"] = _rel(sum(parts), ref)

    # ---- DSA attention, cache short enough that the indexer bypasses -------
    dl = Glm5NextDecoderLayer(cfg, 3)             # full attention
    ref_dsa = _quantize(dl.self_attn) if quant else dl.self_attn
    ref = ref_dsa(x, None, _dsa_cache(cfg, 0))
    parts = []
    for r in range(size):
        m = shard_dsa(copy.deepcopy(ref_dsa), r, size, lambda z: z)
        parts.append(m(x, None, _dsa_cache(cfg, 0)))
    out["dsa_attn_short"] = _rel(sum(parts), ref)

    # ---- DSA indexer: partial head sums must add up to the reference -------
    mx.random.seed(seed + 1)
    primed = 4 * cfg.index_topk
    prime = _prime_arrays(cfg, primed)
    cache_ref = _dsa_cache(cfg, primed, prime)
    mask = mx.ones((1, 1, 1, primed + 1), dtype=mx.bool_)
    ref_scores = {}

    def _capture(store):
        def f(z):
            store["v"] = z
            return z
        return f

    ref_dsa.indexer._tp_reduce = _capture(ref_scores)
    qr = ref_dsa.q_a_layernorm(ref_dsa.q_a_proj(x))
    ref_dsa.indexer(x, qr, mask, cache=cache_ref[1])
    ref_dsa.indexer._tp_reduce = None

    partial = []
    for r in range(size):
        m = shard_dsa(copy.deepcopy(ref_dsa), r, size, lambda z: z)
        got = {}
        m.indexer._tp_reduce = _capture(got)
        qr_r = m.q_a_layernorm(m.q_a_proj(x))
        m.indexer(x, qr_r, mask, cache=_dsa_cache(cfg, primed, prime)[1])
        partial.append(got.get("v"))
    if ref_scores.get("v") is not None and all(p is not None for p in partial):
        out["dsa_indexer_scores"] = _rel(sum(partial), ref_scores["v"])
        ref_top = mx.argsort(-ref_scores["v"], axis=-1)[..., :4]
        tp_top = mx.argsort(-sum(partial), axis=-1)[..., :4]
        out["dsa_indexer_topk_match"] = bool(mx.all(ref_top == tp_top).item())
    else:
        out["dsa_indexer_scores"] = "indexer bypassed (cache too short)"

    # ---- DSA attention with a long cache: indexer + gathered path ---------
    #      needs a real cross-rank reduce inside the indexer, so run rank 0
    #      first, stash its partial, and have rank 1 add it -- the indexer's
    #      inputs (q, pool_keys) do not depend on the reduced value, so the
    #      two-pass trick is exact here.
    stash = {}

    def _reduce_r0(z):
        stash["v"] = z
        return z

    def _reduce_r1(z):
        return z + stash["v"]

    parts = []
    for r, fn in ((0, _reduce_r0), (1, _reduce_r1)):
        m = shard_dsa(copy.deepcopy(ref_dsa), r, size, fn)
        parts.append(m(x, mask, _dsa_cache(cfg, primed, prime)))
    ref_long = ref_dsa(x, mask, _dsa_cache(cfg, primed, prime))
    out["dsa_attn_long_gathered"] = _rel(sum(parts), ref_long)

    # ---- dense MLP ---------------------------------------------------------
    ref_mlp = Glm5NextDecoderLayer(cfg, 0).mlp
    if quant:
        ref_mlp = _quantize(ref_mlp)
    ref = ref_mlp(x)
    parts = [shard_mlp(copy.deepcopy(ref_mlp), r, size)(x) for r in range(size)]
    out["dense_mlp"] = _rel(sum(parts), ref)

    # ---- MoE ---------------------------------------------------------------
    ref_moe = Glm5NextDecoderLayer(cfg, 3).mlp
    if quant:
        ref_moe = _quantize(ref_moe)
    ref = ref_moe(x)
    parts = [shard_moe(copy.deepcopy(ref_moe), r, size)(x) for r in range(size)]
    out["moe"] = _rel(sum(parts), ref)

    if verbose:
        print(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    print("--- fp ---")
    validate()
    print("--- quantized (4-bit, group 64) ---")
    validate(quant=True)
