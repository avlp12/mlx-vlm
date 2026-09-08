import math
import os
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import CacheList, KVCache
from ..mla import MultiLinear
from ..mlp import DeepseekMLP as DeepseekV32MLP
from ..rope_utils import initialize_rope
from ..switch_layers import SwitchGLU
from .config import ModelConfig


class Indexer(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self.rope = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=True,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any] = None,
    ):
        b, s, _ = x.shape
        q = self.wq_b(qr)
        q = q.reshape(b, s, self.n_heads, self.head_dim).swapaxes(1, 2)
        k = self.wk(x)
        k = self.k_norm(k)
        k = mx.reshape(k, (b, 1, s, self.head_dim))

        offset = cache.offset if cache is not None else 0

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros([b, 1, s, 0]))
        if k.shape[2] <= self.index_topk:
            return None
        scores = q @ k.swapaxes(-1, -2)
        scores = mx.maximum(scores, 0)
        weights = self.weights_proj(x) * (self.n_heads**-0.5 * self.softmax_scale)
        weights = weights.swapaxes(-1, -2)[..., None]
        scores = scores * weights
        scores = scores.sum(axis=1, keepdims=True)
        if mask is not None:
            scores = mx.where(mask, scores, -float("inf"))
        return mx.argpartition(scores, kth=-self.index_topk, axis=-1)[
            ..., -self.index_topk :
        ]


class DeepseekV32Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.scale = self.q_head_dim**-0.5

        self.q_a_proj = nn.Linear(
            self.hidden_size, self.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
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

        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            if mscale_all_dim:
                scaling_factor = self.config.rope_scaling["factor"]
                if scaling_factor > 1:
                    s = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                    self.scale = self.scale * s * s

        self.indexer = Indexer(config)
        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=self.rope_theta,
            traditional=True,
            max_position_embeddings=self.max_position_embeddings,
            scaling_config=self.config.rope_scaling,
        )

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
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache[0].offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None] * 2

        topk_indices = self.indexer(x, qr, mask, cache=cache[1])
        if topk_indices is not None:
            if L == 1:
                idx = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk_indices, axis=-1)
            else:
                shape = list(topk_indices.shape)
                shape[-1] = kv_latent.shape[2]
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, topk_indices, mx.array(True), axis=-1
                )
                if mask is not None:
                    sparse_mask = sparse_mask & mask
                mask = sparse_mask
        if cache is not None and cache[0] is not None:
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


# ---------------------------------------------------------------------------
# M1/M2 (V5, 2026-09-08).  Prefill-only FLOP cuts: what "prefill" means here.
#
# Nothing in this stack carries a prefill/decode flag.  ``n_to_process`` stops at the
# server/generate layer, the model sees only shapes, and "S > 1" is NOT prefill: the
# speculative verify block is S = 8 (and adaptive-K varies S inside that window), which
# is precisely the shape I1310 showed is quality-sensitive.  So the condition used by
# both prefill-only levers is a ROW FLOOR on the sequence axis of the call:
#
#     prefill  :=  rows >= MLX_VLM_GLM5_PREFILL_TOPK_MIN_ROWS  (default 512)
#
# and the floor is shared by M1 (MoE top-k) and M2 (DSA indexer top-k) so a run can
# never have one of them think it is in prefill and the other not.  Properties:
#   * decode (S = 1) and every speculative verify block (S <= 8 by the decoder's own
#     compile gate, glm5_next/language.py) are excluded by two orders of magnitude;
#   * the shipped prefill chunk is 8,192 and the tail-merge floor is 1,024, so a normal
#     prefill is entirely above the floor.  A prompt whose LAST chunk is shorter than
#     512 processes that tail at the full, unmodified top-k -- the arm is then weaker,
#     never stronger, than advertised, which is the safe direction;
#   * a batched forward uses the same rule per forward (rows is the shared sequence
#     length), so B > 1 prefill is included and B > 1 decode is not.
_PREFILL_ROW_FLOOR_DEFAULT = 512


def prefill_row_floor() -> int:
    """``MLX_VLM_GLM5_PREFILL_TOPK_MIN_ROWS``, default 512.  <= 0 is refused."""
    raw = os.environ.get("MLX_VLM_GLM5_PREFILL_TOPK_MIN_ROWS", "").strip()
    if not raw:
        return _PREFILL_ROW_FLOOR_DEFAULT
    try:
        val = int(raw)
    except ValueError:
        return _PREFILL_ROW_FLOOR_DEFAULT
    return val if val > 0 else _PREFILL_ROW_FLOOR_DEFAULT


def is_prefill_rows(rows: int) -> bool:
    """Is a forward this wide a PREFILL forward, for the M1/M2 levers?"""
    return int(rows) >= prefill_row_floor()


def prefill_moe_top_k(configured_top_k: int, rows: int) -> int:
    """M1: ``MLX_VLM_GLM5_PREFILL_MOE_TOPK``.  Returns ``configured_top_k`` unless set.

    Only the number of experts changes.  ``group_expert_select`` renormalises the kept
    scores over exactly the k it is given (``scores / scores.sum(-1, keepdims=True)``
    when ``norm_topk_prob``), so k = 7 renormalises over 7 the same way the model
    renormalises over 8 -- there is no separate renormalisation to keep in step.

    ``group_expert_select`` is ``mx.compile``d and takes ``top_k`` as a python int;
    MLX hashes non-array arguments into the trace key (mlx python/src/transforms.cpp:
    489-499), so a second k value simply compiles a second trace -- the k = 8 trace,
    and therefore the decode path, is untouched.

    A value outside [1, configured_top_k] is a typo, not a lever: raise rather than
    silently clamp, which would make an arm vacuous and score a false PASS.
    """
    raw = os.environ.get("MLX_VLM_GLM5_PREFILL_MOE_TOPK", "").strip()
    if not raw or not is_prefill_rows(rows):
        return configured_top_k
    try:
        k = int(raw)
    except ValueError:
        raise ValueError(
            f"MLX_VLM_GLM5_PREFILL_MOE_TOPK={raw!r} is not an integer"
        ) from None
    if not 1 <= k <= configured_top_k:
        raise ValueError(
            f"MLX_VLM_GLM5_PREFILL_MOE_TOPK={k} is outside [1, {configured_top_k}] "
            "(the configured num_experts_per_tok); it can only ever REDUCE the top-k"
        )
    return k


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):

    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / denominator
    scores = scores * routed_scaling_factor

    return inds, scores


class MoEGate(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))
        assert config.topk_method == "noaux_tc", "Unsupported topk method."

    def __call__(self, x):
        return group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


# K5 (V5, 2026-09-08).  MLX_VLM_GLM5_MOE_COMBINE_MATMUL -- opt-in, default OFF.
#
# The shipped combine is ``(y * scores[..., None]).sum(-2)``.  ``y`` is the routed-expert
# output, [B, S, top_k, hidden] in bf16; ``scores`` comes out of ``group_expert_select``
# in FLOAT32 (the router is deliberately fp32 -- glm5_next/language.py's Glm5NextMoEGate
# keeps near-tie top-k membership matching the reference).  MLX promotes, so the product
# is a float32 tensor of the same shape as y: at chunk 8,192, top_k 8, hidden 4,096 that
# is 537 MB of bf16 read, 1,074 MB of fp32 written, 1,074 MB read back by the sum and
# 268 MB written -- ~3.0 GB per sparse layer, 42 layers per chunk, and the V5 SW
# decomposition puts it at 175 ms/step = 1.0 % of the step.
#
# The contraction is a batched matvec: out[b, s, :] = sum_k scores[b, s, k] * y[b, s, k, :].
# Expressed as ``scores[..., None, :] @ y`` MLX runs one GEMM whose accumulator is fp32
# inside the kernel, reads y once, and never materialises the fp32 product at all.
#
# NOT BIT-IDENTICAL, by construction and in two ways: the scores are rounded to y's dtype
# (bf16, ~2**-9 relative) before they multiply, and the reduction over top_k is the GEMM's
# accumulation order rather than mx.sum's.  This arm therefore goes through the KL gate,
# not through a fingerprint.  Where y is already float32 the cast is exact and only the
# accumulation order differs.
_MOE_COMBINE_ON = ("1", "true", "yes", "on")


def _moe_combine_matmul_enabled() -> bool:
    """``MLX_VLM_GLM5_MOE_COMBINE_MATMUL``; read per call (L40 EnvSpec ``per_call``).

    Read once per MoE layer per forward.  NOTE for the in-process arm switcher: at
    B == 1 and S <= 8 this call happens inside ``Glm5NextDecoderLayer._ffn_c``, an
    ``mx.compile``d trace, so the value is frozen into that per-shape trace on first
    use; prefill widths are not compiled and re-read it every forward.
    """
    return os.environ.get("MLX_VLM_GLM5_MOE_COMBINE_MATMUL", "").strip().lower() in _MOE_COMBINE_ON


def moe_combine(y: mx.array, scores: mx.array) -> mx.array:
    """Weighted sum over the top_k axis: ``sum_k scores[..., k] * y[..., k, :]``."""
    if _moe_combine_matmul_enabled():
        # [B, S, 1, k] @ [B, S, k, H] -> [B, S, 1, H]; no fp32 [B, S, k, H] anywhere.
        return (scores[..., None, :].astype(y.dtype) @ y).squeeze(-2)
    return (y * scores[..., None]).sum(axis=-2).astype(y.dtype)


class DeepseekV32MoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV32MLP(
                config=config, intermediate_size=intermediate_size
            )

        self.sharding_group = None

    def __call__(self, x):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = moe_combine(y, scores)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


class DeepseekV32DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DeepseekV32Attention(config)
        self.mlp = (
            DeepseekV32MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV32MLP(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class DeepseekV32Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepseekV32DecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self.num_layers = self.end_idx

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pipeline_rank = 0
        self.pipeline_size = 1

    def pipeline(self, group):
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        layers_per_rank = len(self.layers) // self.pipeline_size
        extra = len(self.layers) - layers_per_rank * self.pipeline_size
        if self.pipeline_rank < extra:
            layers_per_rank += 1
        self.start_idx = (self.pipeline_size - self.pipeline_rank - 1) * layers_per_rank
        self.end_idx = self.start_idx + layers_per_rank
        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx
        self.num_layers = len(self.layers) - self.start_idx

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * self.num_layers
        mask = create_attention_mask(
            h, cache[0][0] if cache[0] else None, return_array=True
        )

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for i in range(self.num_layers):
            h = self.layers[self.start_idx + i](h, mask, cache[i])

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV32Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        mpt_layer = self.args.num_hidden_layers
        new_weights = {}
        for k, v in weights.items():
            parts = k.split(".")
            if len(parts) >= 3 and parts[1] == "layers" and int(parts[2]) >= mpt_layer:
                continue
            new_weights[k] = v
        weights = new_weights

        def dequant(weight, scale_inv):
            dtype = mx.bfloat16
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            bs = 128
            m, n = weight.shape
            pad_bottom = (-m) % bs
            pad_side = (-n) % bs
            weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
            weight = weight.reshape(
                ((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs)
            )
            weight = (weight * scale_inv[:, None, :, None]).reshape(
                m + pad_bottom, n + pad_side
            )
            return weight[:m, :n].astype(dtype)

        new_weights = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                scale_inv = v
                wk = k.replace("_scale_inv", "")
                weight = weights[wk]
                weight = dequant(weight, scale_inv)
                new_weights[wk] = weight
            elif k not in new_weights:
                new_weights[k] = v
        weights = new_weights

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
            prefix = f"model.layers.{l}.self_attn"
            if f"{prefix}.kv_b_proj.weight" in weights:
                layer = self.model.layers[l].self_attn.embed_q
                quantized = f"{prefix}.kv_b_proj.scales" in weights
                v = weights.pop(f"{prefix}.kv_b_proj.weight")
                head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim

                if quantized:
                    dims = self.args.kv_lora_rank
                    scales = weights.pop(f"{prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{prefix}.kv_b_proj.biases")
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(
                        v, scales, biases, bits=bits, group_size=group_size
                    )
                num_heads = self.args.num_attention_heads
                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(
                    v[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
                )
                wv = mx.contiguous(v[:, self.args.qk_nope_head_dim :, :])
                if quantized:
                    wk, wk_scales, wk_biases = mx.quantize(
                        wk, bits=bits, group_size=group_size
                    )
                    wv, wv_scales, wv_biases = mx.quantize(
                        wv, bits=bits, group_size=group_size
                    )
                    weights[f"{prefix}.embed_q.scales"] = wk_scales
                    weights[f"{prefix}.unembed_out.scales"] = wv_scales
                    weights[f"{prefix}.embed_q.biases"] = wk_biases
                    weights[f"{prefix}.unembed_out.biases"] = wv_biases
                weights[f"{prefix}.embed_q.weight"] = wk
                weights[f"{prefix}.unembed_out.weight"] = wv

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()
        for layer in self.model.layers:
            layer.self_attn.q_b_proj = shard_linear(
                layer.self_attn.q_b_proj, "all-to-sharded", group=group
            )

            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.num_heads //= N
            num_heads = layer.self_attn.num_heads
            sh = rank * num_heads
            eh = sh + num_heads

            def shard_heads(w):
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

            if isinstance(layer.mlp, DeepseekV32MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

            else:
                layer.mlp.sharding_group = group = group
                shard_inplace(
                    layer.mlp.shared_experts.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_experts.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.shared_experts.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )

    @property
    def layers(self):
        return self.model.layers[self.model.start_idx : self.model.end_idx]

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    def make_cache(self):
        return [CacheList(KVCache(), KVCache()) for _ in self.layers]


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = DeepseekV32Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        out = self.model(inputs, cache)
        return LanguageModelOutput(logits=self.lm_head(out))

    def sanitize(self, weights):
        return Model.sanitize(self, weights)

    def shard(self, group=None):
        Model.shard(self, group)

    @property
    def cast_predicate(self):
        return Model.cast_predicate.fget(self)

    @property
    def layers(self):
        return self.model.layers[self.model.start_idx : self.model.end_idx]

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def make_cache(self):
        return [CacheList(KVCache(), KVCache()) for _ in self.layers]
