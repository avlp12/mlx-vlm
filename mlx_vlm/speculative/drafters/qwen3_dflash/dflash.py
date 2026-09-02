import os
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from ....models.activations import swiglu
from ....models.cache import BufferedRotatingKVCache, KVCache, RotatingKVCache
from ....models.rope_utils import initialize_rope
from .config import DFlashConfig


# Hoist the sliding-window context discard in front of ``fc``/``hidden_norm``.
#
# Every sliding layer throws away all but the last ``sliding_window - 1`` context
# positions inside DFlashAttention (see :meth:`DFlashAttention.__call__` below).
# ``fc`` and ``hidden_norm`` are position-wise, so running them on the whole
# prompt and then discarding 87% of the rows is pure waste: at a 16,384-token
# prompt with the shipped GLM-5.3-Flash DFlash2 geometry (5 target layers x 4096
# -> 4096) that is 2.75 TFLOP of which 2.41 TFLOP is thrown away.
#
# Deliberately NOT memoized: the adaptive-K knob is memoized on first call and
# every test that touches it has to reach into a module global to undo that.  One
# ``os.environ`` lookup happens once per draft round, next to a multi-GFLOP
# matmul, so the memo buys nothing and costs test ergonomics.
#
# NUMERICS, measured 2026-09-03 (mlx 0.32.1.dev20260902, M3 Ultra).  Bit-identical
# on CPU.  On Metal, bit-identity is a row-count-dependent property of the matmul
# kernel: MLX's quantized matmul changes reduction order at M = 1, 2, 15, 33 and
# ~100 rows and is stable for every M >= 100.  Pre-truncation only fires when
# S > sliding_window - 1, so the shipped GLM-5.3-Flash DFlash2 geometry (fc
# 5*4096 -> 4096, 8-bit affine, group 64, window 2048) only ever compares
# M_new = 2047 against M_ref = S >= 2048, both in the stable regime: measured
# 0 of 8,384,512 output elements differ at S in {2048, 2049, 3000, 4096, 5000,
# 8192, 12000, 16384}, q8 and bf16 alike.  A drafter with a small enough window
# to land in the low-M regimes (the toy geometries in
# mlx_vlm/tests/test_dflash_fc_pretrunc.py keep 7 rows) does differ, by <= 4.34
# bfloat16 ulp of the row maximum, with the drafter's argmax unchanged in 0 of
# 384 draft positions over 32 seeds.  Drafter-only either way: the target's
# forward is not on this path, so a draft that differs can only be accepted or
# rejected, never emitted unverified.  R29
# (/Users/gesicht/glm53flash/logs/sweep10/R29_VERDICT.md, 16,394-token prompt,
# ABAB x 3 on epsilon) ran this ON against the pre-truncation-free tree and
# measured accept/round 2.9231 -> 2.9231 and text sha1 18d1fe50b8beac71513a
# identical.  Hence DEFAULT ON.
def _fc_pretrunc_enabled() -> bool:
    return os.environ.get("MLX_VLM_DFLASH_FC_PRETRUNC", "1") not in ("0", "false", "False")


def _build_rope(config: DFlashConfig):
    # Qwen-family checkpoints normally use split-half (NeoX) pairing. DSpark
    # checkpoints expose rope_is_neox_style explicitly; MLX calls the
    # interleaved GPT-J pairing "traditional".
    traditional = not bool(getattr(config, "rope_is_neox_style", True))
    return initialize_rope(
        dims=config.head_dim,
        base=config.rope_theta,
        traditional=traditional,
        scaling_config=config.rope_scaling,
        max_position_embeddings=config.max_position_embeddings,
    )


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        layer_types = (
            config.layer_types or ["full_attention"] * config.num_hidden_layers
        )
        self.is_sliding = layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, x_ctx: mx.array, rope, cache: KVCache):
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        if self.is_sliding:
            if self.sliding_window is None:
                raise ValueError(
                    "DFlash draft config must define sliding_window for sliding layers."
                )
            keep_ctx = self.sliding_window - 1
            if S > keep_ctx:
                skip = S - keep_ctx
                x_ctx = x_ctx[:, skip:]
                S = x_ctx.shape[1]
                cache.offset += skip

        # Project context and proposal separately so only context KV
        queries = self.q_proj(x)
        ctx_keys = self.k_proj(x_ctx)
        ctx_values = self.v_proj(x_ctx)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        ctx_keys = self.k_norm(ctx_keys.reshape(B, S, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        ctx_values = ctx_values.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(prop_keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        prop_values = prop_values.reshape(B, L, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )
        queries = rope(queries, offset=cache.offset + S)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + S)
        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)
        # DFlash denoises the whole proposed block at once, so draft-block
        # self-attention is intentionally non-causal. Sliding layers already
        # limit resident prefix context through the rotating cache above.
        mask = None
        o = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class Qwen3MLP(nn.Module):
    """Qwen3-style gated MLP (matches mlx_lm.models.qwen3.MLP weights)."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x, x_ctx, rope, cache):
        h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlashDraftModel(nn.Module):
    layer_class = DFlashDecoderLayer

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        if not self.config.layer_types:
            self.config.layer_types = ["full_attention"] * self.config.num_hidden_layers
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [
            self.layer_class(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(config)
        self.embed_tokens = None
        self.embed_scale = 1.0
        self.lm_head = None
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def bind(self, target_model) -> "DFlashDraftModel":
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(
            target_model.model, "embed_tokens"
        ):
            inner = target_model.model
        elif (
            hasattr(target_model, "language_model")
            and hasattr(target_model.language_model, "model")
            and hasattr(target_model.language_model.model, "embed_tokens")
        ):
            inner = target_model.language_model.model
        else:
            raise AttributeError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )
        self.embed_tokens = inner.embed_tokens
        self.embed_scale = getattr(
            self.embed_tokens, "embed_scale", getattr(inner, "embed_scale", 1.0)
        )
        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self.embed_tokens.as_linear
        )
        return self

    def make_cache(self) -> List[KVCache]:
        window = getattr(self.config, "draft_window_size", None)
        if window is not None and int(window) > 0:
            return [
                BufferedRotatingKVCache(max_size=int(window), buffer_size=64)
                for _ in self.layers
            ]
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                if self.config.sliding_window is None:
                    raise ValueError(
                        "DFlash draft config must define sliding_window for sliding layers."
                    )
                caches.append(
                    RotatingKVCache(max_size=self.config.sliding_window - 1, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches

    def reset(self, target_model) -> List[KVCache]:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        return self.make_cache()

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache: List[KVCache],
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        mask_id = int(self.config.mask_token_id)
        if isinstance(last_bonus, int):
            block = mx.array(
                [[last_bonus] + [mask_id] * (block_size - 1)],
                dtype=token_dtype,
            )
        else:
            B = last_bonus.shape[0]
            masks = mx.full((B, block_size - 1), mask_id, dtype=token_dtype)
            block = mx.concatenate(
                [last_bonus[:, None].astype(token_dtype), masks], axis=1
            )
        draft_hidden = self._hidden(block, hidden, cache)
        draft_logits = self._logits(draft_hidden[:, 1:])
        return sampler(draft_logits)

    # ---------------------------------------------------------- prefill trim
    # The two methods below are the public half of the pre-truncation contract,
    # so a PREFILL can hand this drafter the trailing window instead of the whole
    # prompt.  They exist because the hoist inside ``_hidden`` only saves FLOPs:
    # the caller still had to build and retain a [B, S, 5*4096] context, which on
    # a 16k prompt is 0.67 GB held for the life of the request.
    #
    # Splitting the discard across the seam is bit-identical ONLY if the offset
    # bookkeeping crosses with it.  ``_pretruncate_ctx`` does two things -- drop
    # ``skip`` rows AND add ``skip`` to every layer cache's offset -- and the
    # second is what fixes the absolute RoPE positions of the surviving context
    # (``rope(ctx_keys, offset=cache.offset)`` and ``rope(queries,
    # offset=cache.offset + S)`` below).  A caller that trims without calling
    # :meth:`adopt_pretruncated_context` shifts every draft position by ``skip``.
    # RoPE is relative, so the attention scores are mathematically the same and
    # the emitted tokens cannot change (acceptance is resolved against the
    # target's argmax) -- but the arithmetic is not the same arithmetic, and the
    # acceptance rate is free to move.  Do both or neither.

    def prefill_context_keep(self) -> Optional[int]:
        """Trailing context rows a round-1 forward keeps, or ``None`` for all of them."""
        return self._uniform_ctx_keep()

    def adopt_pretruncated_context(self, cache: List[KVCache], skip: int) -> None:
        """Account for ``skip`` context rows a caller already dropped.

        Exactly the offset half of :meth:`_pretruncate_ctx`, applied to a fresh
        cache before the first draft round -- the same ``+= skip`` on the same
        caches, just earlier.  ``_pretruncate_ctx`` then finds nothing left to
        drop and is a no-op, so the round-1 computation is unchanged.
        """
        if skip <= 0:
            return
        # zip against self.layers: never touch a cache entry no layer owns.
        for _, c in zip(self.layers, cache):
            c.offset += skip

    def _uniform_ctx_keep(self) -> Optional[int]:
        """Context positions every layer will keep, or ``None`` if they differ.

        Only meaningful when EVERY layer is a sliding layer with the same
        window: a full-attention layer keeps the whole context, so a hoisted
        truncation would change what it sees.
        """
        types = self.config.layer_types or []
        if not types or any(t != "sliding_attention" for t in types):
            return None
        window = getattr(self.config, "sliding_window", None)
        if window is None or int(window) <= 1:
            return None
        return int(window) - 1

    def _pretruncate_ctx(
        self, target_hidden: mx.array, cache: List[KVCache]
    ) -> mx.array:
        """Drop the context rows every layer is about to drop anyway.

        Bit-identical to the per-layer discard for the rows that survive:
        ``fc`` is a matmul over the feature axis and ``hidden_norm`` an RMSNorm
        over the feature axis, so both are position-wise and row ``i`` of the
        result does not depend on how many other rows were handed in.  The
        offset bookkeeping is the same arithmetic the layer does -- it adds
        ``skip`` to its own cache before the two rope calls -- moved out one
        level so each cache is advanced exactly once.
        """
        keep = self._uniform_ctx_keep()
        if keep is None or not _fc_pretrunc_enabled():
            return target_hidden
        skip = target_hidden.shape[1] - keep
        if skip <= 0:
            return target_hidden
        # zip against self.layers: never touch a cache entry no layer owns.
        for _, c in zip(self.layers, cache):
            c.offset += skip
        return target_hidden[:, skip:]

    def _hidden(
        self,
        inputs: mx.array,
        target_hidden: mx.array,
        cache: List[KVCache],
    ) -> mx.array:
        h = self._embed_input_tokens(inputs)
        target_hidden = self._pretruncate_ctx(target_hidden, cache)
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, c in zip(self.layers, cache):
            h = layer(h, h_ctx, self.rope, c)
        return self.norm(h)

    def _embed_input_tokens(self, inputs: mx.array) -> mx.array:
        return self.embed_tokens(inputs) * self.embed_scale

    def _logits(self, hidden: mx.array) -> mx.array:
        logits = self.lm_head(hidden)
        if self.config.final_logit_softcapping is not None:
            softcap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / softcap) * softcap
        return logits

    def __call__(
        self,
        inputs: mx.array,
        target_hidden: mx.array,
        cache: List[KVCache],
    ) -> mx.array:
        return self._logits(self._hidden(inputs, target_hidden, cache))

    def sanitize(self, weights: dict) -> dict:
        out = {}
        for k, v in weights.items():
            if k.startswith("model."):
                k = k[len("model.") :]
            out[k] = v
        return out


DFlashKVCache = KVCache
