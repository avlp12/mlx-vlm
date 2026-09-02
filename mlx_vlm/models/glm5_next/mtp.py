import os
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import TextConfig
from .language import Glm5NextMoE, Glm5NextSparseAttention


_MTP_FFN_COMPILE_ENV: Optional[bool] = None


def _mtp_ffn_compile_enabled() -> bool:
    """Opt-OUT: ``MLX_VLM_GLM5_MTP_FFN_COMPILE=0`` runs the nextn FFN eager.

    Default-on is justified by a bit-identity receipt, not by a speed claim: the
    compiled and eager FFN halves are ``mx.array_equal`` at B=1 for every S the
    gate admits (see ``test_glm5_next_mtp_ffn_compile_is_bit_identical``), so the
    switch cannot move a single output bit and the only thing at risk is compile
    time.  The speedup itself is NOT measured here -- the target's own
    ``compile_ffn`` note records a mean +0.137 ms on a 33.9 ms step with the sign
    flipping, i.e. indistinguishable from zero at layer scale.  What is different
    for the nextn head is that it is ONE layer executed once or twice per
    speculative round on the latency-critical path, so the per-round dispatch
    saving is not diluted 45x.  Flip this off if a compile-time regression shows up.
    """
    global _MTP_FFN_COMPILE_ENV
    if _MTP_FFN_COMPILE_ENV is None:
        _MTP_FFN_COMPILE_ENV = os.environ.get(
            "MLX_VLM_GLM5_MTP_FFN_COMPILE", "1"
        ).lower() not in ("0", "false", "no", "off")
    return _MTP_FFN_COMPILE_ENV


class Glm5NextMTP(nn.Module):
    """Multi-token-prediction (nextn) head. Drafts token t+2 from the base model's
    hidden state h(t+1) and the embedding of the accepted token t+1. Structure mirrors
    the GLM-5-Next layer-45 nextn layer: enorm/hnorm -> eh_proj -> DSA+MoE decoder ->
    shared_head norm -> (shared) lm_head applied by the caller."""

    def __init__(self, config: TextConfig):
        super().__init__()
        h = config.hidden_size
        self.enorm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.hnorm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * h, h, bias=False)
        self.input_layernorm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.self_attn = Glm5NextSparseAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.mlp = Glm5NextMoE(config)
        self.shared_head_norm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        # Mirrors Glm5NextDecoderLayer.compile_ffn: the stateless FFN half compiles
        # cleanly at a fixed small decode shape, and mx.compile keeps a per-shape
        # cache so each S compiles once.  Restricted to B == 1 and S <= 8 for the
        # same reason as the target -- compiling the 288-expert MoE at prefill or
        # batched shapes spikes memory and can OOM.
        self.compile_ffn = _mtp_ffn_compile_enabled()
        self._ffn_c = None

    def __call__(
        self,
        hidden: mx.array,
        next_embed: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        x = self.eh_proj(
            mx.concatenate([self.enorm(next_embed), self.hnorm(hidden)], axis=-1)
        )
        x = x + self.self_attn(self.input_layernorm(x), mask, cache)
        if self.compile_ffn and x.shape[0] == 1 and x.shape[1] <= 8:
            if self._ffn_c is None:
                self._ffn_c = mx.compile(self._ffn_block)
            x = x + self._ffn_c(x)
        else:
            x = x + self._ffn_block(x)
        return self.shared_head_norm(x)

    def _ffn_block(self, x: mx.array) -> mx.array:
        # Stateless FFN half (no cache) -> compiles cleanly at a fixed decode shape.
        return self.mlp(self.post_attention_layernorm(x))


def load_mtp_weights(config: TextConfig, weights: dict, layer_idx: int = 45) -> dict:
    """Map raw layer-{layer_idx} checkpoint tensors -> Glm5NextMTP module tree.
    Absorbs raw kv_b_proj into embed_q/unembed_out (kept in the stored dtype; no
    re-quantization). Experts are already stacked into switch_mlp in the checkpoint."""
    pfx = f"language_model.model.layers.{layer_idx}."
    out = {}
    rename = {
        "shared_head.norm.": "shared_head_norm.",
    }
    for k, v in weights.items():
        if not k.startswith(pfx):
            continue
        rest = k[len(pfx) :]
        for a, b in rename.items():
            if rest.startswith(a):
                rest = b + rest[len(a) :]
        out[rest] = v

    kb = "self_attn.kv_b_proj.weight"
    if kb in out:
        v = out.pop(kb)
        nope, vhd = config.qk_nope_head_dim, config.v_head_dim
        nheads = config.num_attention_heads
        v = v.reshape(nheads, nope + vhd, -1)
        out["self_attn.embed_q.weight"] = mx.contiguous(v[:, :nope, :].swapaxes(-1, -2))
        out["self_attn.unembed_out.weight"] = mx.contiguous(v[:, nope:, :])
    return out
