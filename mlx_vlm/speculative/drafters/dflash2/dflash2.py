import os
from collections.abc import Callable, Mapping

import mlx.core as mx
import mlx.nn as nn

from ..compatibility import validate_dflash_target
from ..qwen3_dflash.dflash import DFlashDecoderLayer, DFlashDraftModel
from .config import DFlash2Config

_COMPILE_ENV = None


def _compile_enabled() -> bool:
    """MLX_VLM_DFLASH_COMPILE=1 -- fuse the drafter's small-op chains.

    A DFlash2 draft block is launch-bound, not bandwidth-bound: five layers of
    ~1.2 GB of q8 weights is ~2 ms of traffic, but the round spends several ms
    more in dispatch, most of it in two Python loops -- the two-offset
    _grouped_dynamic_convolve (four calls per layer, twenty per block) and the
    selector's per-position edge scoring.

    mx.compile is known on this model family to silently bake cache offsets into
    a trace, so only regions that are provably pure with respect to cache state
    are compiled here: the conv/norm halves around self_attn, and the selector
    step.  self_attn (which reads and writes the KV cache) and the MLP (whose
    silu carries a sigmoid, and JIT'd Metal swaps the precise exp for the fast
    approximation) are both deliberately left outside the boundary so the fused
    path stays bit-identical.

    ON by default since it was re-measured.  It shipped OFF because a live A/B
    put it at -15.6% (AIF I842), but that verdict was n=1 per arm, in a fixed
    order, at HEAD 3ac662d1 -- before 4bb3b97b generalised MLA absorption to
    L > 1 -- and both of its arms sat at a speculative multiple of 0.94x and
    1.11x, a regime that no longer exists.  Its own component receipt disagreed
    with it at the time: dflash2_compile_paired.json measured the draft block
    8.2% faster compiled over 60 paired samples.

    Re-measured in-process, one load, arms interleaved, three cycles, natural
    prompts at a 1024-token prime (logs/sweep3/R9_spec_width_r9.json):

        workload   compile-on vs compile-off, per cycle
        code       +1.03%  +0.33%  +0.31%
        prose      +0.49%  +0.58%  +0.70%

    Six paired comparisons, six positive; a sign test on that is p = 0.016.  The
    effect is small because the draft call is only ~6.2 ms of a ~66 ms round, so
    an 8% draft win is worth well under 1% end to end -- which is what shows up.

    Arithmetic is untouched, and the run says so rather than assuming it: rounds
    and accepted-per-round are byte-identical between the arms (79 / 2.2278 on
    code, 118 / 1.1695 on prose) and the decoded text matches.  Speculative
    decoding is exact by construction, and in the same session all seven width
    policies produced identical output for a given workload -- so this is a
    wall-clock lever with no identity question attached.
    """
    global _COMPILE_ENV
    if _COMPILE_ENV is None:
        _COMPILE_ENV = os.environ.get("MLX_VLM_DFLASH_COMPILE", "1").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    return _COMPILE_ENV


def _grouped_dynamic_convolve(
    hidden: mx.array,
    dynamic: mx.array,
    base: mx.array,
    group_size: int,
) -> mx.array:
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.reshape(batch, length, groups, group_size)
    dynamic = dynamic.reshape(batch, length, base.shape[0], groups, 1)
    output = mx.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = (
            blocks
            if offset == 0
            else mx.concatenate(
                [mx.zeros_like(blocks[:, :offset]), blocks[:, :-offset]], axis=1
            )
        )
        kernel = base[offset].reshape(1, 1, groups, group_size).astype(hidden.dtype)
        output = output + (kernel + dynamic[:, :, offset]) * values
    return output.reshape(hidden.shape)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        groups = hidden_size // group_size
        self.base_kernel = mx.zeros((2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * kernel_size * groups, bias=False
        )

    def prepare(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.kernel_size, groups
        )
        prepared = _grouped_dynamic_convolve(
            hidden,
            dynamic[..., 0, :, :],
            self.base_kernel[0],
            self.group_size,
        )
        return prepared, dynamic[..., 1, :, :]

    def finish(self, hidden: mx.array, dynamic: mx.array) -> mx.array:
        return _grouped_dynamic_convolve(
            hidden, dynamic, self.base_kernel[1], self.group_size
        )


class DFlash2DecoderLayer(DFlashDecoderLayer):
    def __init__(self, config: DFlash2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_conv = GroupedDynamicCausalConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size
        )
        self.mlp_conv = GroupedDynamicCausalConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size
        )
        self._fused = None

    # The four cache-free halves of the layer.  Each is a norm plus a projection
    # plus a two-offset dynamic convolution -- a dozen tiny dispatches that fuse
    # into one.  None of them reads or writes the KV cache, and none of them
    # contains an exp, a sigmoid or an RNG.
    def _pre_attn(self, x):
        return self.attention_conv.prepare(self.input_layernorm(x))

    def _post_attn(self, residual, attn_out, kernel):
        return residual + self.attention_conv.finish(attn_out, kernel)

    def _pre_mlp(self, x):
        return self.mlp_conv.prepare(self.post_attention_layernorm(x))

    def _post_mlp(self, residual, mlp_out, kernel):
        return residual + self.mlp_conv.finish(mlp_out, kernel)

    def _halves(self):
        if not _compile_enabled():
            return self._pre_attn, self._post_attn, self._pre_mlp, self._post_mlp
        if self._fused is None:
            # mx.compile keeps a per-shape cache, so the handful of live block
            # sizes (block_size, the adaptive floor, and 1 on the bonus step)
            # each trace once.
            self._fused = (
                mx.compile(self._pre_attn),
                mx.compile(self._post_attn),
                mx.compile(self._pre_mlp),
                mx.compile(self._post_mlp),
            )
        return self._fused

    def __call__(self, x, x_ctx, rope, cache):
        pre_attn, post_attn, pre_mlp, post_mlp = self._halves()
        residual = x
        x, kernel = pre_attn(x)
        # self_attn stays eager: it carries the rotating KV cache, and a trace
        # would freeze cache.offset.
        x = post_attn(residual, self.self_attn(x, x_ctx, rope, cache), kernel)
        residual = x
        x, kernel = pre_mlp(x)
        # self.mlp stays eager: swiglu -> nn.silu -> sigmoid -> exp, and JIT'd
        # Metal uses the fast exp approximation where the prebuilt path uses the
        # precise one.  Keeping it out is what makes the fused path bit-identical.
        return post_mlp(residual, self.mlp(x), kernel)


class CandidateSelector(nn.Module):
    def __init__(self, config: DFlash2Config):
        super().__init__()
        self.top_k = config.selector_top_k
        self.predecessor_codebook = nn.Embedding(
            config.vocab_size, config.selector_rank
        )
        self.successor_codebook = nn.Embedding(config.vocab_size, config.selector_rank)
        self.hidden_projection = nn.Linear(
            config.hidden_size, config.selector_rank, bias=False
        )
        self._fused_step = None

    def _scores(self, predecessor, hidden_pos, candidates_pos, unary_pos):
        edges = mx.sum(
            self.predecessor_codebook(predecessor)[:, None]
            * hidden_pos[:, None]
            * self.successor_codebook(candidates_pos),
            axis=-1,
        )
        return unary_pos + edges

    def _greedy_step(self, predecessor, hidden_pos, candidates_pos, unary_pos):
        """One position of the Viterbi walk, greedy.  Two codebook gathers, a
        three-way product, a reduction, an argmax and a gather -- six dispatches
        that fuse into one.  Pure: no cache, no exp, no RNG."""
        scores = self._scores(predecessor, hidden_pos, candidates_pos, unary_pos)
        selected = mx.argmax(scores, axis=-1).reshape(-1)
        return mx.take_along_axis(candidates_pos, selected[:, None], axis=-1)[:, 0]

    def select(
        self,
        hidden: mx.array,
        logits: mx.array,
        anchor_ids: mx.array,
        sampler: Callable[[mx.array], mx.array],
    ) -> mx.array:
        candidates = mx.argpartition(logits, -self.top_k, axis=-1)[..., -self.top_k :]
        unary = mx.take_along_axis(logits, candidates, axis=-1)
        hidden = self.hidden_projection(hidden)
        predecessor = anchor_ids.reshape(-1)
        path = []
        sample_proposal = getattr(sampler, "sample_proposal", None)
        # Only the greedy walk is fused.  A caller-supplied sample_proposal is an
        # opaque callable that may draw randomness, and mx.compile freezes an RNG
        # key into the trace, so that path stays eager.
        step = None
        if _compile_enabled() and not callable(sample_proposal):
            if self._fused_step is None:
                self._fused_step = mx.compile(self._greedy_step)
            step = self._fused_step
        for position in range(hidden.shape[1]):
            if step is not None:
                predecessor = step(
                    predecessor,
                    hidden[:, position],
                    candidates[:, position],
                    unary[:, position],
                )
                path.append(predecessor)
                continue
            scores = self._scores(
                predecessor, hidden[:, position], candidates[:, position],
                unary[:, position],
            )
            selected = (
                sample_proposal(scores)
                if callable(sample_proposal)
                else mx.argmax(scores, axis=-1)
            ).reshape(-1)
            predecessor = mx.take_along_axis(
                candidates[:, position], selected[:, None], axis=-1
            )[:, 0]
            path.append(predecessor)
        return mx.stack(path, axis=1)


class DFlash2DraftModel(DFlashDraftModel):
    layer_class = DFlash2DecoderLayer
    prefer_requested_block_size = False
    dflash_initial_block_size = 3
    dflash_min_block_size = 3

    def __init__(self, config: DFlash2Config):
        super().__init__(config)
        self.candidate_selector = CandidateSelector(config)

    def validate_target_compatibility(self, target_model) -> None:
        validate_dflash_target(self.config, target_model, "DFlash2")

    def bind(self, target_model) -> "DFlash2DraftModel":
        self.validate_target_compatibility(target_model)
        super().bind(target_model)
        return self

    def _embed_input_tokens(self, inputs: mx.array) -> mx.array:
        return (
            self.embed_tokens(inputs)
            * self.embed_scale
            * self.config.input_embedding_scale
        )

    def _logits(self, hidden: mx.array) -> mx.array:
        logits = self.lm_head(hidden) * self.config.output_multiplier
        if self.config.final_logit_softcapping is not None:
            softcap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / softcap) * softcap
        return logits

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler: Callable[[mx.array], mx.array],
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        proposal_length = int(block_size) - 1
        if proposal_length <= 0:
            batch = 1 if isinstance(last_bonus, int) else int(last_bonus.shape[0])
            return mx.zeros((batch, 0), dtype=token_dtype)
        anchor = (
            mx.array([last_bonus], dtype=token_dtype)
            if isinstance(last_bonus, int)
            else last_bonus.reshape(-1).astype(token_dtype)
        )
        masks = mx.full(
            (anchor.shape[0], proposal_length),
            int(self.config.mask_token_id),
            dtype=token_dtype,
        )
        draft_inputs = mx.concatenate([anchor[:, None], masks], axis=1)
        draft_hidden = self._hidden(draft_inputs, hidden, cache)[:, 1:]
        return self.candidate_selector.select(
            draft_hidden,
            self._logits(draft_hidden),
            anchor,
            sampler,
        ).astype(token_dtype)

    def sanitize(self, weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
        normalized = {}
        codebooks = {
            "candidate_selector.predecessor_codebook",
            "candidate_selector.successor_codebook",
        }
        for key, value in weights.items():
            key = key.removeprefix("model.")
            if key in codebooks:
                key = f"{key}.weight"
            if key in normalized:
                raise ValueError(
                    f"Duplicate DFlash2 weight key after sanitization: {key}"
                )
            normalized[key] = value
        return normalized


Model = DFlash2DraftModel


__all__ = [
    "CandidateSelector",
    "DFlash2DecoderLayer",
    "DFlash2DraftModel",
    "GroupedDynamicCausalConv",
    "Model",
    "_grouped_dynamic_convolve",
]
