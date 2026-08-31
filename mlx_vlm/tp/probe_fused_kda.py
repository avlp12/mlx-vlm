"""Does the fused KDA kernel actually run on a half head-shard?

The biggest stated risk to the TP=2 projection: the fused decode kernel takes
H as a compile-time template constant and grids ``B * H`` threadgroups, so a
TP shard asks it to compile a second pipeline at H=32.  Inspection says that is
fine (threadgroup memory is sized by D, not H), but inspection is not a run.

This needs one layer at the real dims, not the 320B tree.
"""

from __future__ import annotations

import copy
import json
import os

os.environ.setdefault("MLX_VLM_GLM5_FUSED_KDA", "1")

import mlx.core as mx
import mlx.nn as nn

from ..models.cache import ArraysCache
from ..models.glm5_next.config import TextConfig
from ..models.glm5_next.fused_kda import fused_kda_probe, fused_kda_supported
from ..models.glm5_next.language import Glm5NextLinearAttention
from .glm5_next import shard_kda

REAL = dict(
    model_type="glm5_next", vocab_size=512, hidden_size=4096, intermediate_size=64,
    moe_intermediate_size=32, num_hidden_layers=1, num_attention_heads=4,
    num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8,
    routed_scaling_factor=1.0, kv_lora_rank=32, q_lora_rank=32, qk_rope_head_dim=0,
    v_head_dim=16, qk_nope_head_dim=16, num_experts_per_tok=2, first_k_dense_replace=3,
    max_position_embeddings=1024, rms_norm_eps=1e-6, index_topk=8, index_head_dim=16,
    index_n_heads=2, layer_types=["linear_attention"], mlp_layer_types=["dense"],
    linear_attn_config={"num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4,
                        "gate_lower_bound": -5.0},
    index_kpool=4, mla_use_nope=True, hc_mult=4,
)
QUANT = ("q_proj", "k_proj", "v_proj", "b_proj", "g_a_proj", "g_b_proj", "o_proj")


def _build(dtype=mx.bfloat16):
    cfg = TextConfig.from_dict(REAL)
    attn = Glm5NextLinearAttention(cfg)
    # KDA projections are 8-bit group-64 in the shipped tree
    for name in QUANT:
        setattr(attn, name, nn.QuantizedLinear.from_linear(
            getattr(attn, name), group_size=64, bits=8))
    fg = attn.forget_gate
    fg.f_a_proj = nn.QuantizedLinear.from_linear(fg.f_a_proj, group_size=64, bits=8)
    fg.f_b_proj = nn.QuantizedLinear.from_linear(fg.f_b_proj, group_size=64, bits=8)
    attn.conv1d.weight = attn.conv1d.weight.astype(dtype)
    attn.o_norm.weight = attn.o_norm.weight.astype(dtype)
    return cfg, attn


def _cache(B, H, D, K, dtype):
    c = ArraysCache(size=2)
    c[0] = mx.zeros((B, K - 1, 3 * H * D), dtype=dtype)
    c[1] = mx.zeros((B, H, D, D), dtype=mx.float32)
    return c


def _run(attn, x, H, D, K, dtype, fused: bool):
    attn = copy.deepcopy(attn)
    attn._fused_kda = bool(fused)          # None -> probe; False -> force eager
    c = _cache(x.shape[0], H, D, K, dtype)
    out = attn(x, None, c)
    mx.eval(out)
    return out


def main():
    dtype = mx.bfloat16
    mx.random.seed(0)
    cfg, base = _build(dtype)
    H, D, K = cfg.linear_num_heads, cfg.linear_head_dim, cfg.linear_conv_kernel_dim
    x = mx.random.normal((1, 1, cfg.hidden_size)).astype(dtype)
    res = {"H_full": H, "head_dim": D, "conv_k": K}

    for size, tag in ((1, "H64_full"), (2, "H32_shard")):
        attn = copy.deepcopy(base)
        if size > 1:
            shard_kda(attn, 0, size, lambda z: z)
        h = attn.num_heads
        sup = fused_kda_supported(num_heads=h, head_dim=D, conv_kernel_size=K,
                                  lower_bound=attn.forget_gate.safe_gate_lower_bound)
        ty = fused_kda_probe(kind="base", num_heads=h, head_dim=D, conv_kernel_size=K,
                             dtype=dtype, state_dtype=mx.float32, bits=8, group_size=64)
        entry = {"num_heads_local": h, "supported": bool(sup), "probe_ty": ty,
                 "conv_shape": list(attn.conv1d.weight.shape),
                 "A_log": int(attn.forget_gate.A_log.size),
                 "dt_bias": int(attn.forget_gate.dt_bias.size)}
        if sup and ty is not None:
            eager = _run(attn, x, h, D, K, dtype, fused=False)
            fus = _run(attn, x, h, D, K, dtype, fused=True)
            diff = float(mx.max(mx.abs(fus.astype(mx.float32) - eager.astype(mx.float32))))
            scale = float(mx.max(mx.abs(eager.astype(mx.float32))))
            entry["fused_vs_eager_max_abs"] = diff
            entry["fused_vs_eager_rel"] = diff / max(scale, 1e-6)
            entry["bit_identical"] = bool(mx.array_equal(fus, eager).item())
        res[tag] = entry
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
