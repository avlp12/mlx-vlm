from __future__ import annotations

import contextlib
import functools
import logging
import os
import sys
import time
import warnings
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm

from .. import apc as _apc
from .. import context_vault as _context_vault
from .. import harvest_provenance as _harvest_prov
from ..kv_quant import from_legacy as kv_quant_from_legacy
from ..models import cache
from ..prompt_utils import apply_chat_template
from ..sample_utils import make_logits_processors, make_sampler, top_p_sampling
from ..speculative.utils import (
    PrefillHiddenAccumulator,
    chunk_capture_kwargs_for,
    make_speculative_prompt_cache,
    prefill_capture_kwargs,
    prefill_context_keep,
    run_speculative_rounds,
    run_speculative_server_rounds,
    speculative_hidden_state,
    speculative_prefill_kwargs,
)
from ..turboquant import BatchTurboQuantKVCache, turboquant_enabled
from ..utils import group_images_by_shape, prepare_inputs, should_add_special_tokens
from .common import (
    DEFAULT_COMPLETION_BATCH_SIZE,
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_P,
    DEFAULT_PREFILL_BATCH_SIZE,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    _chunked_prefill_enabled,
    generation_stream,
    maybe_quantize_kv_cache,
    wired_limit,
)
from .types import GenerateKwargs, ProcessorLike, Unpack

logger = logging.getLogger("mlx_vlm.generate")

DEFAULT_TOP_N_SIGMA = 0.0
DEFAULT_BATCH_CACHE_EVAL_INTERVAL = 50


def _get_batch_cache_eval_interval() -> int:
    raw = os.environ.get("MLX_VLM_BATCH_CACHE_EVAL_INTERVAL")
    if raw is None:
        return DEFAULT_BATCH_CACHE_EVAL_INTERVAL
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Ignoring invalid MLX_VLM_BATCH_CACHE_EVAL_INTERVAL=%r", raw)
        return DEFAULT_BATCH_CACHE_EVAL_INTERVAL


def _position_seed(seed: int, row_id: int, position: int) -> int:
    x = (int(seed) ^ 0x9E3779B9) & 0xFFFFFFFF
    x = (x + (int(row_id) + 1) * 0x85EBCA6B) & 0xFFFFFFFF
    x = (x ^ ((int(position) + 1) * 0xC2B2AE35)) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    return int(x & 0xFFFFFFFF)


def _position_keys(seed: int, row_ids: List[int], positions: List[int]) -> mx.array:
    return mx.stack(
        [
            mx.random.key(_position_seed(seed, row, pos))
            for row, pos in zip(row_ids, positions)
        ]
    )


class _PositionedTargetSampler:
    """Sampler with stateless target draws keyed by generated-token position."""

    def __init__(self, *, temperature: float, top_p: float, seed: int):
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = int(seed)

    def __call__(self, logprobs: mx.array) -> mx.array:
        if self.top_p > 0 and self.top_p < 1.0:
            return top_p_sampling(logprobs, self.top_p, self.temperature)
        return mx.random.categorical(logprobs * (1 / self.temperature))

    def sample_target(
        self,
        logprobs: mx.array,
        *,
        row_ids: List[int],
        positions: List[int],
    ) -> mx.array:
        if logprobs.shape[0] != len(row_ids) or len(row_ids) != len(positions):
            raise ValueError("row_ids and positions must match logprobs batch size.")
        keys = _position_keys(self.seed, row_ids, positions)
        if self.top_p > 0 and self.top_p < 1.0:
            return mx.vmap(self._sample_top_p_one, in_axes=(0, 0))(logprobs, keys)
        return mx.vmap(self._sample_one, in_axes=(0, 0))(logprobs, keys)

    def sample_proposal(
        self,
        logprobs: mx.array,
        *,
        row_ids: List[int],
        positions: List[int],
    ) -> mx.array:
        keys = _position_keys(self.seed ^ 0x0DFA5202, row_ids, positions)
        return mx.vmap(self._sample_one, in_axes=(0, 0))(logprobs, keys)

    def _sample_one(self, logprobs: mx.array, key: mx.array) -> mx.array:
        return mx.random.categorical(logprobs * (1 / self.temperature), key=key)

    def _sample_top_p_one(self, logprobs: mx.array, key: mx.array) -> mx.array:
        if logprobs.dtype == mx.bfloat16:
            logprobs = logprobs.astype(mx.float32)
        probs = mx.softmax(logprobs / self.temperature, axis=-1)
        sorted_indices = mx.argsort(probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        top_probs = mx.where(
            cumulative_probs > 1 - self.top_p,
            sorted_probs,
            mx.zeros_like(sorted_probs),
        )
        sampled_pos = mx.random.categorical(mx.log(top_probs), key=key)
        return mx.take_along_axis(sorted_indices, sampled_pos[..., None], axis=-1)[0]


def _generate_module_override(name: str, fallback):
    generate_module = sys.modules.get("mlx_vlm.generate")
    return getattr(generate_module, name, fallback) if generate_module else fallback


def normalize_resize_shape(values):
    if values is None:
        return None
    if not (
        not isinstance(values, (str, bytes))
        and len(values) in (1, 2)
        and all(type(value) is int for value in values)
    ):
        raise ValueError("resize_shape must contain 1 or 2 integers")
    return (values[0], values[0]) if len(values) == 1 else tuple(values)


def generate_step(
    input_ids: mx.array,
    model: nn.Module,
    pixel_values,
    mask,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE,
    presence_penalty: Optional[float] = None,
    presence_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE,
    frequency_penalty: Optional[float] = None,
    frequency_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE,
    top_p: float = DEFAULT_TOP_P,
    min_p: float = DEFAULT_MIN_P,
    top_k: int = DEFAULT_TOP_K,
    top_n_sigma: float = DEFAULT_TOP_N_SIGMA,
    p_less: bool = False,
    typical_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
    prompt_cache: Optional[List[Any]] = None,
    max_kv_size: Optional[int] = None,
    kv_bits: Optional[float] = None,
    kv_key_bits: Optional[float] = None,
    kv_value_bits: Optional[float] = None,
    kv_key_scheme: Optional[str] = None,
    kv_value_scheme: Optional[str] = None,
    kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
    kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
    quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
    draft_model: Optional[nn.Module] = None,
    draft_kind: str = "dflash",
    draft_block_size: Optional[int] = None,
    prompt_cache_checkpoint: Optional[Callable[[int, List[Any]], None]] = None,
    prompt_cache_checkpoint_len: Optional[Union[int, Sequence[int]]] = None,
    warm_prefix: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
    **kwargs,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        input_ids (mx.array): The input prompt token ids.
        model (nn.Module): The model to use for generation.
        pixel_values: The pixel values for vision models (optional).
        mask: The attention mask (optional).
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): The temperature for sampling, if 0 the argmax is used.
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty.
        presence_penalty (float, optional): Additive penalty for tokens that
          already appeared in recent generated context.
        presence_context_size (int, optional): The number of tokens to
          consider for presence penalty.
        frequency_penalty (float, optional): Additive penalty scaled by token
          frequency in recent generated context.
        frequency_context_size (int, optional): The number of tokens to
          consider for frequency penalty.
        top_p (float, optional): Nucleus sampling, higher means model considers
          more less likely words.
        min_p (float, optional): Minimum probability threshold relative to the
          highest-probability token.
        top_k (int, optional): Restrict sampling to the top-k tokens.
        logit_bias (dictionary, optional): Additive logit bias.
        prompt_cache (list, optional): Pre-existing KV cache for the prompt.
        max_kv_size (int, optional): Maximum KV cache size.
        kv_bits (float, optional): Number of bits for KV cache quantization.
        kv_group_size (int): Group size for uniform KV cache quantization.
        kv_quant_scheme (str): KV cache quantization backend.
        quantized_kv_start (int): Start index for quantized KV cache.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits.
        prefill_step_size (int): Number of tokens to process per prefill step.
          Chunked prefill processes prompts in smaller chunks to reduce peak
          memory usage.
        draft_model (nn.Module, optional): A drafter for speculative decoding.
          When set, the decode loop is replaced by the drafter's speculative
          loop (e.g. DFlash block-diffusion). VLM prefill with image/audio
          is supported via the same ``get_input_embeddings`` path the normal
          decoder uses; decode itself is text-only. ``temperature`` and
          ``sampler`` are respected; ``logprobs`` is always ``None`` on the
          speculative path.
        draft_block_size (int, optional): Override the drafter's configured
          block size.

    Yields:
        Generator[Tuple[mx.array, mx.array], None, None]: A generator producing
          one token and a vector of log probabilities.
    """

    quantize_cache_fn = functools.partial(
        _generate_module_override("maybe_quantize_kv_cache", maybe_quantize_kv_cache),
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
        kv_quant_scheme=kv_quant_scheme,
        kv_key_bits=kv_key_bits,
        kv_value_bits=kv_value_bits,
        kv_key_scheme=kv_key_scheme,
        kv_value_scheme=kv_value_scheme,
    )

    sampler_is_greedy = sampler is None and temperature == 0
    if sampler is None:
        if (
            seed is not None
            and temperature > 0
            and min_p == DEFAULT_MIN_P
            and top_k == DEFAULT_TOP_K
            and top_n_sigma == DEFAULT_TOP_N_SIGMA
            and not p_less
            and typical_p == 1.0
        ):
            sampler = _PositionedTargetSampler(
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
        else:
            sampler = _generate_module_override("make_sampler", make_sampler)(
                temp=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                top_n_sigma=top_n_sigma,
                p_less=p_less,
                typical_p=typical_p,
            )

    processors = _generate_module_override(
        "make_logits_processors", make_logits_processors
    )(
        logit_bias,
        repetition_penalty,
        repetition_context_size,
        presence_penalty,
        presence_context_size,
        frequency_penalty,
        frequency_context_size,
    )
    if logits_processors is not None:
        processors.extend(logits_processors)

    y = input_ids
    tokens = mx.array([], dtype=input_ids.dtype)
    target_sample_position = 0

    thinking_budget_criteria = kwargs.pop("thinking_budget_criteria", None)

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model.language_model,
            max_kv_size=max_kv_size,
        )

    # Speculative decoding setup
    last_outputs = None
    speculative_prefill_capture_kwargs = {}
    if draft_model is not None:
        from ..speculative.drafters import validate_drafter_compatibility

        validate_drafter_compatibility(model, draft_model, draft_kind)
        speculative_prefill_capture_kwargs = speculative_prefill_kwargs(
            draft_kind, draft_model
        )
        # Reset stale mRoPE state from any previous generation.
        lm = model.language_model if hasattr(model, "language_model") else model
        if hasattr(lm, "_position_ids"):
            lm._position_ids = None
        if hasattr(lm, "_rope_deltas"):
            lm._rope_deltas = None

    # The chunk loop below must carry the SAME capture as the final forward, or
    # the drafter is handed a one-row context (issue #2096: ``chunk_kwargs`` was
    # built from ``kwargs`` only, so turning chunking on for a capturing drafter
    # silently dropped the prompt).  Only a per-layer capture, or MTP's
    # ``return_hidden`` with the server-priming window on, stitches back
    # together -- see ``chunk_capture_kwargs_for``.
    _prefill_capture_kwargs = (
        prefill_capture_kwargs(
            model.language_model if hasattr(model, "language_model") else model,
            speculative_prefill_capture_kwargs,
        )
        if speculative_prefill_capture_kwargs
        else {}
    )
    _chunk_capture_kwargs = chunk_capture_kwargs_for(_prefill_capture_kwargs)
    target_hidden_offset = 0
    _prefill_hidden = PrefillHiddenAccumulator(
        keep=(
            prefill_context_keep(draft_kind, draft_model)
            if _chunk_capture_kwargs
            else None
        )
    )

    def _step(y, inputs_embeds=None):
        nonlocal tokens, kwargs, last_outputs, target_sample_position

        step_kwargs = kwargs
        if speculative_prefill_capture_kwargs:
            # Prefill only -- with a drafter attached the loop below is unreachable
            # (the speculative branch returns first), so every _step that carries
            # these kwargs is a prefill forward.  Drop the rollback stash there.
            step_kwargs = {**kwargs, **_prefill_capture_kwargs}
        if getattr(model.language_model, "supports_logits_to_keep", False):
            step_kwargs = {**step_kwargs, "logits_to_keep": 1}

        with mx.stream(generation_stream):
            if "decoder_input_ids" in step_kwargs:
                outputs = model.language_model(
                    cache=prompt_cache,
                    **step_kwargs,
                )
            else:
                outputs = model.language_model(
                    y,
                    inputs_embeds=inputs_embeds,
                    cache=prompt_cache,
                    **step_kwargs,
                )

            last_outputs = outputs
            logits = outputs.logits[:, -1, :]

            if len(processors) > 0 and len(y) > 0:
                tokens = mx.concat([tokens, y.flatten()])

                for processor in processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)

            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            y = _sample_with_positions(
                sampler,
                logprobs,
                row_ids=[0] * logprobs.shape[0],
                positions=list(
                    range(
                        target_sample_position,
                        target_sample_position + logprobs.shape[0],
                    )
                ),
            )
            target_sample_position += logprobs.shape[0]

            if outputs.cross_attention_states is not None:
                kwargs = {"cross_attention_states": outputs.cross_attention_states}
            elif outputs.encoder_outputs is not None:
                kwargs = {"encoder_outputs": outputs.encoder_outputs}
            else:
                kwargs = {}

            return y, logprobs.squeeze(0) if logprobs.shape[0] == 1 else logprobs

    # Chunked prefill trims ``input_ids`` down to its last token below; the
    # prompt-lookup drafter needs the whole prompt to build its n-gram index, so
    # snapshot it before that happens.
    full_prompt_ids = input_ids

    with mx.stream(generation_stream):
        # Get input embeddings (handles both multimodal and text-only)
        embedding_output = model.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )

        inputs_embeds = embedding_output.inputs_embeds

        kwargs.update(
            {
                k: v
                for k, v in embedding_output.to_dict().items()
                if k != "inputs_embeds" and v is not None
            }
        )
        policy_kwargs = kwargs
        if speculative_prefill_capture_kwargs:
            policy_kwargs = {**kwargs, **speculative_prefill_capture_kwargs}
        if prefill_step_size is not None and not _chunked_prefill_enabled(
            model,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            prompt_cache=prompt_cache,
            draft_model=draft_model,
            draft_kind=draft_kind,
            prefill_kwargs=policy_kwargs,
        ):
            prefill_step_size = None
        # ``prompt_cache_checkpoint_len`` accepts an int (single checkpoint, the
        # original contract) or a sequence of ints (a Warm Context Vault boundary
        # ladder). Boundaries are consumed in ascending order; each one fires the
        # callback exactly once.
        if prompt_cache_checkpoint is None or prompt_cache_checkpoint_len is None:
            checkpoint_boundaries: List[int] = []
        elif isinstance(prompt_cache_checkpoint_len, (list, tuple, set)):
            checkpoint_boundaries = sorted(
                {int(b) for b in prompt_cache_checkpoint_len if int(b) > 0}
            )
        else:
            checkpoint_boundaries = [int(prompt_cache_checkpoint_len)]
        from ..context_vault import CheckpointLadder

        ladder = CheckpointLadder(checkpoint_boundaries, inputs_embeds.shape[1])
        should_chunk = (
            prefill_step_size is not None and inputs_embeds.shape[1] > prefill_step_size
        ) or bool(ladder) or (warm_prefix and inputs_embeds.shape[1] > 1)
        # ``warm_prefix`` marks a request resuming from an already-populated
        # prompt cache (vault restore / APC prefix hit). Without it, a tail
        # shorter than prefill_step_size skips the chunk loop entirely and the
        # WHOLE tail -- final token included -- is processed by a single _step
        # forward, while a cold prefill always splits the last token off into
        # its own _step. The two decompositions are mathematically equal but not
        # bit-identical: measured on GLM-5.3-Flash the KDA conv/recurrent state
        # then diverges by up to 3e-2 in 101 of 112 cache components, and greedy
        # decode splits from the cold reference at token 35 of 64. Forcing the
        # loop reproduces the cold path's exact tail split (n-1 chunked, 1
        # stepped) and restores token identity.
        if prefill_step_size is not None and should_chunk:
            # Chunked prefill with embeddings
            total_tokens = inputs_embeds.shape[1]
            processed_tokens = 0
            # Optional two-box layer-pipelined prefill: stage A runs layers
            # [0, split) here while stage B runs the rest on a peer box, one
            # chunk behind. Only the boundary activation crosses; stage B's
            # caches come back once at the end so decode stays single-box.
            # Disabled when an APC checkpoint is requested (the checkpoint
            # would capture a half-populated cache).
            pipeline = None
            # (vault merge fix: the old single ``checkpoint_len`` became the
            # boundary ladder -- pipeline stays disabled whenever any
            # checkpoint boundary is requested.)
            # ``_chunk_capture_kwargs`` disables it too: ``pipeline.prefill_chunk``
            # returns no output object, so a pipelined chunk's hidden capture
            # cannot be accumulated and the drafter would lose the prompt.
            if not ladder and not _chunk_capture_kwargs:
                from ..pipeline_runtime import maybe_open_pipeline

                pipeline = maybe_open_pipeline(model, total_tokens, verbose=verbose)
                if pipeline is not None:
                    pipeline.begin(total_tokens, prefill_step_size)
            with tqdm(
                total=total_tokens, desc="Prefill", unit="tok", disable=not verbose
            ) as pbar:
                while inputs_embeds.shape[1] > 1:
                    n_to_process = min(prefill_step_size, inputs_embeds.shape[1] - 1)
                    # Land exactly on the next boundary. Vault boundaries are
                    # multiples of prefill_step_size, so this clamp is a no-op
                    # for them and the chunk decomposition -- and thus
                    # bit-identity against a straight-through prefill -- is
                    # preserved. An unaligned caller-supplied boundary still
                    # works, but trades that guarantee away.
                    n_to_process = ladder.clamp(processed_tokens, n_to_process)
                    chunk_kwargs = kwargs
                    if getattr(model.language_model, "supports_logits_to_keep", False):
                        chunk_kwargs = {**kwargs, "logits_to_keep": 1}
                    if _chunk_capture_kwargs:
                        chunk_kwargs = {**chunk_kwargs, **_chunk_capture_kwargs}
                    if pipeline is not None:
                        pipeline.prefill_chunk(
                            model,
                            input_ids[:, :n_to_process],
                            inputs_embeds[:, :n_to_process],
                            prompt_cache,
                        )
                        # only stage A's caches exist on this box until finalize
                        mx.eval([c.state for c in pipeline.local_caches(prompt_cache)])
                    else:
                        chunk_out = model.language_model(
                            inputs=input_ids[:, :n_to_process],
                            inputs_embeds=inputs_embeds[:, :n_to_process],
                            cache=prompt_cache,
                            n_to_process=n_to_process,
                            **chunk_kwargs,
                        )
                        _prefill_hidden.append(chunk_out)
                        # Drop the chunk's logits (and any gdn stash) BEFORE the
                        # eval, so the vocab-wide projection is never materialised
                        # for a chunk nobody samples from.
                        chunk_out = None
                        quantize_cache_fn(prompt_cache)
                        mx.eval(
                            [c.state for c in prompt_cache] + _prefill_hidden.pending()
                        )
                    processed_tokens += n_to_process
                    for reached in ladder.reached(processed_tokens):
                        prompt_cache_checkpoint(reached, prompt_cache)
                    inputs_embeds = inputs_embeds[:, n_to_process:]
                    input_ids = input_ids[:, n_to_process:]
                    mx.clear_cache()
                    pbar.update(n_to_process)

            if pipeline is not None:
                # pull stage B's KDA/DSA caches back and install them, so the
                # last token and all of decode run locally over the full stack
                pipeline.finalize(prompt_cache)
                quantize_cache_fn(prompt_cache)
                pipeline.close()

            input_ids = input_ids[:, -1:]

        y, logprobs = _step(input_ids, inputs_embeds=inputs_embeds)
        if _chunk_capture_kwargs and last_outputs is not None:
            _prefill_hidden.append(last_outputs)
            stitched, target_hidden_offset = _prefill_hidden.finish()
            if stitched is not None:
                last_outputs.hidden_states = stitched

    mx.async_eval(y, logprobs)

    # Speculative decoding
    if draft_model is not None:
        yield from run_speculative_rounds(
            model,
            draft_model,
            prompt_cache,
            input_ids,
            y,
            logprobs,
            last_outputs,
            draft_kind=draft_kind,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            sampler_is_greedy=sampler_is_greedy,
            prompt_tokens=full_prompt_ids,
            target_hidden_offset=target_hidden_offset,
        )
        return

    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y[None])
            mx.async_eval(next_y, next_logprobs)
        if n == 0:
            mx.eval(y)
        if n == max_tokens:
            break

        yield y.item(), logprobs
        if n % 256 == 0:
            mx.clear_cache()

        if thinking_budget_criteria is not None:
            forced_token_id = thinking_budget_criteria.pop_forced_token_id()
            if forced_token_id is not None:
                next_y = mx.array([forced_token_id], dtype=next_y.dtype)
        y, logprobs = next_y, next_logprobs
        n += 1


@dataclass
class BatchGenerationResult:
    """
    Result of batch generation with optional image size tracking.

    Attributes:
        texts: Generated text for each sample
        tokens: Last generated token for each sample
        logprobs: Log probabilities for each sample
        prompt_tokens: Number of prompt tokens per sample
        generation_tokens: Number of generated tokens per sample
        total_tokens: Total tokens (prompt + generation) per sample
        prompt_tps: Prompt tokens per second per sample
        generation_tps: Generation tokens per second per sample
        peak_memory: Peak memory usage in GB
        image_sizes: Original (height, width) for each image (for tracking)
    """

    texts: List[str]
    tokens: List[Optional[int]]
    logprobs: List[Optional[List[float]]]
    prompt_tokens: List[int]
    generation_tokens: List[int]
    total_tokens: List[int]
    prompt_tps: List[float]
    generation_tps: List[float]
    peak_memory: float = 0.0
    image_sizes: Optional[List[Tuple[int, int]]] = None


def _left_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)

    return mx.array([[0] * (max_length - len(p)) + p for p in prompts])


def _right_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)

    return mx.array([list(p) + [0] * (max_length - len(p)) for p in prompts])


_SEQUENCE_ALIGNED_PROMPT_KWARGS = {
    "attention_mask",
    "decoder_inputs_embeds",
    "deepstack_visual_embeds",
    "visual_pos_masks",
    "per_layer_inputs",
    "full_text_row_masked_out_mask",
    "position_ids",
    "pos_hw",
    "mm_token_type_ids",
    "token_type_ids",
}

APC_PRIVATE_PROMPT_KEYS = (
    "_apc_tenant",
    "_apc_image_hash",
    "_apc_semantic_hash",
)


def _is_mrope_position_ids_prompt_kwarg(key: str, v: mx.array) -> bool:
    return key == "position_ids" and v.ndim == 3 and v.shape[0] == 3


def _prompt_kwarg_batch_size(key: str, v: mx.array) -> int:
    if _is_mrope_position_ids_prompt_kwarg(key, v):
        return v.shape[1]
    return v.shape[0] if v.ndim > 0 else 0


def _prompt_kwarg_row(key: str, v: mx.array, row_idx: int, batch_size: int) -> mx.array:
    if _is_mrope_position_ids_prompt_kwarg(key, v):
        if v.shape[1] == batch_size:
            return v[:, row_idx : row_idx + 1, :]
        return v[:, :1, :]
    if v.shape[0] == batch_size:
        return v[row_idx : row_idx + 1]
    return v[:1]


def _split_prompt_kwargs_per_row(prompt_kwargs: dict, batch_size: int) -> List[dict]:
    """Normalize batched prompt kwargs into one dict per batch row.

    ``model.get_input_embeddings()`` commonly returns batch-sized tensors
    (notably ``inputs_embeds``). ``BatchGenerator.insert()`` stores prompt
    kwargs per sequence, so passing the same batched dict for every row causes
    the prompt builder to concatenate those batched tensors ``batch_size``
    times, effectively squaring the batch dimension.
    """
    if batch_size <= 1:
        return [prompt_kwargs or {}]

    rows = [{} for _ in range(batch_size)]
    for k, v in (prompt_kwargs or {}).items():
        if isinstance(v, mx.array) and _prompt_kwarg_batch_size(k, v) >= 1:
            for i in range(batch_size):
                rows[i][k] = _prompt_kwarg_row(k, v, i, batch_size)
        else:
            for row in rows:
                row[k] = v
    return rows


def _is_sequence_aligned_prompt_kwarg(
    key: str, v: mx.array, sequence_length: int
) -> bool:
    if key not in _SEQUENCE_ALIGNED_PROMPT_KWARGS:
        return False
    if _is_mrope_position_ids_prompt_kwarg(key, v):
        return v.shape[2] == sequence_length
    return v.ndim >= 2 and v.shape[1] == sequence_length


def _pad_sequence_aligned_prompt_kwarg(
    key: str, v: mx.array, target_length: int, *, left: bool
) -> mx.array:
    sequence_axis = 2 if _is_mrope_position_ids_prompt_kwarg(key, v) else 1
    pad = target_length - v.shape[sequence_axis]
    if pad <= 0:
        return v
    pad_shape = tuple(
        pad if axis == sequence_axis else size for axis, size in enumerate(v.shape)
    )
    pad_v = mx.zeros(pad_shape, dtype=v.dtype)
    parts = [pad_v, v] if left else [v, pad_v]
    return mx.concatenate(parts, axis=sequence_axis)


def _slice_sequence_aligned_prompt_kwarg(
    key: str, v: mx.array, start: Optional[int] = None, stop: Optional[int] = None
) -> mx.array:
    sequence_axis = 2 if _is_mrope_position_ids_prompt_kwarg(key, v) else 1
    slices = [slice(None)] * v.ndim
    slices[sequence_axis] = slice(start, stop)
    return v[tuple(slices)]


def _mrope_position_ids_row(v: mx.array) -> mx.array:
    if _is_mrope_position_ids_prompt_kwarg("position_ids", v):
        return v
    if v.ndim == 2:
        return mx.broadcast_to(v[None, :, :], (3, v.shape[0], v.shape[1]))
    return v


def _concat_prompt_kwarg_rows(key: str, rows: List[mx.array]) -> mx.array:
    if key == "position_ids" and any(
        _is_mrope_position_ids_prompt_kwarg(key, row) for row in rows
    ):
        return mx.concatenate([_mrope_position_ids_row(row) for row in rows], axis=1)
    return mx.concatenate(rows, axis=0)


def _merge_prefill_prompt_kwargs(
    prompt_kwargs_list: List[Optional[dict]],
    input_ids: List[List[int]],
) -> Tuple[mx.array, dict]:
    """Batch per-row prompt kwargs for a left-padded prefill forward."""
    lengths = [len(ids) for ids in input_ids]
    max_length = max(lengths)

    row_embeds: List[mx.array] = []
    embed_dtype = None
    embed_dim = None
    for kw, length in zip(prompt_kwargs_list, lengths):
        if not kw or kw.get("inputs_embeds") is None:
            raise ValueError("inputs_embeds is required")
        embeds = kw["inputs_embeds"]  # [1, length, D]
        embed_dtype = embeds.dtype
        embed_dim = embeds.shape[-1]
        if length < max_length:
            pad = mx.zeros(
                (embeds.shape[0], max_length - length, embed_dim),
                dtype=embed_dtype,
            )
            embeds = mx.concatenate([pad, embeds], axis=1)
        row_embeds.append(embeds)
    inputs_embeds = mx.concatenate(row_embeds, axis=0)

    merged_kwargs: dict = {}
    per_row_keys: dict = {}
    batch_size = len(prompt_kwargs_list)
    for i, (kw, length) in enumerate(zip(prompt_kwargs_list, lengths)):
        if not kw:
            continue
        for k, v in kw.items():
            if k == "inputs_embeds" or k in APC_PRIVATE_PROMPT_KEYS:
                continue
            if isinstance(v, mx.array) and _prompt_kwarg_batch_size(k, v) >= 1:
                row_v = _prompt_kwarg_row(k, v, i, batch_size)
                if _is_sequence_aligned_prompt_kwarg(k, row_v, length):
                    row_v = _pad_sequence_aligned_prompt_kwarg(
                        k, row_v, max_length, left=True
                    )
                per_row_keys.setdefault(k, []).append(row_v)
            elif k not in merged_kwargs:
                merged_kwargs[k] = v
    for k, vs in per_row_keys.items():
        merged_kwargs[k] = _concat_prompt_kwarg_rows(k, vs)

    return inputs_embeds, merged_kwargs


def _is_batch_cache_entry(entry) -> bool:
    """Return whether a cache entry already owns a batch dimension."""
    if isinstance(entry, cache.CacheList):
        return all(_is_batch_cache_entry(child) for child in entry.caches)
    return callable(getattr(entry, "filter", None)) and callable(
        getattr(entry, "extend", None)
    )


def _extend_cache(cache_a, cache_b):
    """Extend cache_a with cache_b along the batch dimension."""
    if not cache_a:
        return cache_b
    if not cache_b:
        return cache_a
    extended = []
    for ca, cb in zip(cache_a, cache_b):
        if not _is_batch_cache_entry(ca) and hasattr(ca.__class__, "merge"):
            ca = ca.__class__.merge([ca])
        if not _is_batch_cache_entry(cb) and hasattr(cb.__class__, "merge"):
            cb = cb.__class__.merge([cb])
        ca.extend(cb)
        extended.append(ca)
    return extended


def _make_cache(
    model,
    left_padding,
    kv_bits=None,
    kv_key_bits=None,
    kv_value_bits=None,
    kv_key_scheme=None,
    kv_value_scheme=None,
    kv_group_size=64,
    kv_quant_scheme=DEFAULT_KV_QUANT_SCHEME,
    quantized_kv_start=0,
    prefill_length=0,
):
    """
    Convert a list of regular caches into their corresponding
    batch-aware caches.

    When *kv_bits* is set, a quantized batch cache is used instead of
    ``BatchKVCache`` so that KV states are quantized on-the-fly during
    generation, reducing memory usage for long sequences.

    *kv_quant_scheme* selects the quantization backend:
    - ``"uniform"`` → ``BatchQuantizedKVCache`` (``mx.quantize``)
    - ``"turboquant"`` or fractional *kv_bits* → ``BatchTurboQuantKVCache``
    """
    _batch_policy = kv_quant_from_legacy(
        kv_bits,
        kv_quant_scheme,
        kv_group_size,
        kv_key_bits,
        kv_value_bits,
        kv_key_scheme,
        kv_value_scheme,
    )
    if _batch_policy is not None and not _batch_policy.is_homogeneous:
        raise NotImplementedError(
            "mixed key/value KV quantization schemes are not supported on the "
            "batch path yet; run with a single --kv-quant-scheme or disable "
            "continuous batching"
        )

    use_turbo = kv_bits is not None and turboquant_enabled(kv_bits, kv_quant_scheme)

    defer_turbo = (
        use_turbo and quantized_kv_start > 0 and prefill_length < quantized_kv_start
    )

    def _make_quant_cache(lp):
        if use_turbo:
            if defer_turbo:
                return cache.BatchKVCache(lp)
            return BatchTurboQuantKVCache(
                lp, bits=kv_bits, key_bits=kv_key_bits, value_bits=kv_value_bits
            )
        return cache.BatchQuantizedKVCache(
            lp, group_size=kv_group_size, bits=int(kv_bits)
        )

    def to_batch_cache(c, quantize=True):
        # Caches that ship their own batch-conversion (e.g. MiniMax M3 sparse
        # index-key side cache) know how to build the correct batch cache.
        if hasattr(c, "to_batch") and not isinstance(c, cache.KVCache):
            return c.to_batch(left_padding)
        if isinstance(c, cache.KVCache):
            if kv_bits is not None and quantize:
                return _make_quant_cache(left_padding)
            return cache.BatchKVCache(left_padding)
        elif isinstance(c, cache.ChunkedKVCache):
            if kv_bits is not None and quantize:
                return _make_quant_cache(left_padding)
            return cache.BatchKVCache(left_padding)
        elif isinstance(c, cache.SimpleKVCache):
            if kv_bits is not None and quantize:
                return _make_quant_cache(left_padding)
            return cache.BatchKVCache(left_padding)
        elif isinstance(c, cache.ArraysCache):
            c.left_padding = mx.array(left_padding)
            return c
        elif isinstance(c, cache.PoolingCache):
            return cache.BatchPoolingCache(c.ratio, left_padding)
        elif isinstance(c, cache.RotatingKVCache):
            if c.keep > 0:
                raise ValueError("RotatingKVCache with keep tokens is not supported.")
            return cache.BatchRotatingKVCache(c.max_size, left_padding)
        elif isinstance(c, cache.CacheList):
            return cache.CacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
        elif isinstance(c, tuple):
            return cache.CacheList(*(to_batch_cache(sub_c) for sub_c in c))
        else:
            raise ValueError(f"{type(c)} does not yet support batching")

    if hasattr(model, "make_cache"):
        model_cache = model.make_cache()
        n = len(model_cache)
        return [
            to_batch_cache(c, quantize=cache.should_quantize_kv_layer(i, n))
            for i, c in enumerate(model_cache)
        ]
    else:
        if kv_bits is not None:
            n = len(model.layers)
            return [
                (
                    _make_quant_cache(left_padding)
                    if cache.should_quantize_kv_layer(i, n)
                    else cache.BatchKVCache(left_padding)
                )
                for i in range(n)
            ]
        return [cache.BatchKVCache(left_padding) for _ in model.layers]


# ---------------------------------------------------------------------------
# Right-padded prefill: the model capability, and the refusals it causes
# ---------------------------------------------------------------------------
#
# ``BatchGenerator._build_mixed_prompt_batch`` squares a mixed warm/cold batch
# off by RIGHT-padding every row's suffix to the longest one, and then rolls
# that padding into left padding in ``finalize()`` once the prefill forward is
# done.  That is sound for a cache whose state is a per-column K/V buffer --
# rolling the buffer IS the coordinate change -- and it is not sound for a cache
# whose state is RECURRENT.  A linear-attention layer (GLM-5's KDA, and every
# hybrid model in this tree that carries an ``ArraysCache``) folds the padded
# columns into a running state plus a short convolution window:
#
#   * ``ArraysCache.make_mask`` (``models/cache.py``) does have a ``lengths``
#     branch, but a right-padded batch never reaches it: ``PromptProcessingBatch``
#     sets ``left_padding = [0] * B`` for such a batch (see ``__init__`` below),
#     and ``left_padding`` wins that ``if``.  The padding is therefore attended.
#   * Even with the mask restored, the conv state is taken as the last K-1
#     columns of the padded input, and the forget gate is applied at every
#     column -- so a row with ``right_pad[i] > 0`` finishes prefill with a state
#     taken at the wrong column and decayed ``right_pad[i]`` steps too far.
#   * A recurrent state cannot be rolled back into place the way a K/V buffer
#     can: it does not carry the column it came from.
#
# So right padding is structurally incompatible with these layers UNLESS every
# row's prefill ends at the same column -- i.e. unless the padding is zero.
# The capability below is what the builder consults, and the policy it drives is
# "batch only rows whose suffix lengths are EQUAL".
#
# DERIVATION.  The capability is derived from ``model.make_cache()`` -- the
# presence of an ``ArraysCache`` (this tree's container for a recurrent/conv
# state) in the prototype cache -- rather than declared per model class, with an
# explicit class attribute ``supports_right_padded_prefill`` taken as an
# override when a model sets one.  Derivation is the default because the defect
# is a property of the STATE, not of the model: 24 model packages under
# ``mlx_vlm/models`` construct an ``ArraysCache`` today (baichuan_m1,
# bailing_moe_linear, falcon_h1, glm5_next, granitemoehybrid, inkling, jamba,
# kimi_k3, kimi_linear, lfm2, lfm2_vl, longcat_flash_ngram, mamba, mamba2,
# nemotron_h, nemotron_h_nano_omni, nemotron_voicechat, plamo2vl, qwen3_5,
# qwen3_next, qwen4_exp, recurrent_gemma, rwkv7, zaya1_vl), and a per-class
# declaration would silently omit the next one to land.  The explicit attribute
# exists so a model that knows something the prototype does not can still say
# so; glm5_next sets it to False for exactly that reason (documentation at the
# site of the KDA layers, not action -- its prototype already answers False).
_RECURRENT_STATE_CACHE_TYPES = (cache.ArraysCache,)


def _cache_entry_has_recurrent_state(entry) -> bool:
    """True if ``entry`` (one element of a prototype cache) holds recurrent state."""
    if isinstance(entry, _RECURRENT_STATE_CACHE_TYPES):
        return True
    subs = getattr(entry, "caches", None)
    if subs is None and isinstance(entry, (list, tuple)):
        subs = entry
    if subs:
        return any(_cache_entry_has_recurrent_state(sub) for sub in subs)
    return False


def model_supports_right_padded_prefill(model) -> bool:
    """Can this model's prefill end at different columns in different rows?

    ``False`` means a right-padded prefill would leave at least one layer's
    state at the wrong column with no way to roll it back, so the caller must
    not build such a batch.  Answering ``False`` costs throughput; answering
    ``True`` wrongly costs correctness, so every uncertain branch answers
    ``False``.
    """
    declared = getattr(model, "supports_right_padded_prefill", None)
    if declared is not None:
        return bool(declared)
    make_cache = getattr(model, "make_cache", None)
    if make_cache is None:
        # No prototype to inspect: this model gets a list of plain
        # ``BatchKVCache`` from ``_make_cache``, which rolls correctly.
        return True
    try:
        prototype = make_cache()
    except Exception:  # noqa: BLE001 - an unbuildable prototype is not a licence
        logger.warning(
            "right-padded prefill: %s.make_cache() raised; declining right "
            "padding for this model",
            type(model).__name__,
        )
        return False
    return not any(_cache_entry_has_recurrent_state(c) for c in prototype)


# Process-wide, because the ``BatchGenerator`` that counts these is a local of
# ``ResponseGenerator``'s loop and is rebuilt whenever the batch drains -- there
# is no long-lived object for the server's ``/health`` snapshot to read.  Same
# shape, and the same reason, as ``context_vault.session_skip_counts()``.
_PREFILL_BATCH_REFUSALS: Dict[str, int] = {}


def prefill_batch_refusal_counts() -> Dict[str, int]:
    """Prefill batches the admission policy refused to build, by reason.

    ``right_pad_kda`` counts the REFUSAL EVENTS (batches that would have been
    right-padded and were split instead); ``right_pad_kda_rows_deferred``
    counts the rows those events pushed back into the pending queue, which is
    the number the throughput cost is actually proportional to.
    """
    return dict(_PREFILL_BATCH_REFUSALS)


def reset_prefill_batch_refusal_counts() -> None:
    """Zero the counters (tests; not called on any serving path)."""
    _PREFILL_BATCH_REFUSALS.clear()


def _note_prefill_batch_refusal(reason: str, rows_deferred: int) -> None:
    _PREFILL_BATCH_REFUSALS[reason] = _PREFILL_BATCH_REFUSALS.get(reason, 0) + 1
    rows_key = f"{reason}_rows_deferred"
    _PREFILL_BATCH_REFUSALS[rows_key] = _PREFILL_BATCH_REFUSALS.get(
        rows_key, 0
    ) + int(rows_deferred)


@dataclass
class BatchStats:
    """
    An data object to hold generation stats.

    Args:
        prompt_tokens (int): The number of prompt tokens processed.
        prompt_tps (float): The prompt processing tokens-per-second.
        prompt_time (float): The time in seconds spent in prompt processing.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        generation_time (float): The time in seconds spent in generation .
        peak_memory (float): The peak memory used so far in GB.
    """

    prompt_tokens: int = 0
    prompt_tps: float = 0
    prompt_time: float = 0
    generation_tokens: int = 0
    generation_tps: float = 0
    generation_time: float = 0
    peak_memory: float = 0


@dataclass
class BatchResponse:
    """
    An data object to hold a batch generation response.

    Args:
        texts: (List[str]): The generated text for each prompt.
        stats (BatchStats): Statistics about the generation.
        image_sizes: (Optional[List[Tuple[int, int]]]): Original (height, width)
            for each image. Useful for tracking which images produced which responses
            and for debugging padding/batching behavior.
    """

    texts: List[str]
    stats: BatchStats
    image_sizes: Optional[List[Tuple[int, int]]] = None


@dataclass
class PromptProgress:
    """Per-request prompt processing metrics for continuous batching."""

    uid: int
    prompt_tokens: int
    prompt_tps: float = 0.0
    prompt_time: float = 0.0
    cached_tokens: int = 0
    # Width of the prefill batch this row's warm prefix was HARVESTED in, or
    # ``None`` when the prefix has no recorded provenance.  Rides next to
    # ``cached_tokens`` because that is the number it qualifies: 3,091 cached
    # tokens taken out of a B=2 prefill and 3,091 taken out of a B=1 prefill are
    # the same count and, measured, not the same cache (L1b-1).
    cached_from_width: Optional[int] = None


def _sample_with_positions(
    sampler: Callable[[mx.array], mx.array],
    logprobs: mx.array,
    *,
    row_ids: Optional[List[int]] = None,
    positions: Optional[List[int]] = None,
) -> mx.array:
    sample_target = getattr(sampler, "sample_target", None)
    if callable(sample_target) and row_ids is not None and positions is not None:
        return sample_target(logprobs, row_ids=row_ids, positions=positions)
    return sampler(logprobs)


class GenerationBatch:
    """
    Batched token generator with double-buffered pipelining.

    Manages the generation phase after prompt processing, with KV caches,
    sampling, and stop detection for multiple sequences. Uses async_eval
    to overlap GPU computation with CPU processing (decode-ahead pattern).
    """

    @dataclass
    class Response:
        uid: int
        token: int
        token_logprob: float
        finish_reason: Optional[str]
        top_logprobs: Optional[List[Tuple[int, float]]] = None

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        inputs: mx.array,
        prompt_cache: List[Any],
        sampler: Callable[[mx.array], mx.array],
        stop_criteria,
        max_tokens: List[int],
        top_logprobs_k: int = 0,
        greedy_sampling: bool = False,
        token_context: Optional[List[List[int]]] = None,
        logits_processors: Optional[
            List[Optional[List[Callable[[mx.array, mx.array], mx.array]]]]
        ] = None,
        thinking_budget_criteria: Optional[List[Any]] = None,
    ):
        self.model = model
        self._language_model = getattr(model, "language_model", model)
        self.uids = uids
        self.prompt_cache = prompt_cache
        self.sampler = sampler
        self.stop_criteria = stop_criteria
        self.max_tokens = max_tokens
        self._num_tokens = [0] * len(uids)
        self.compute_logprobs = True
        self.top_logprobs_k = top_logprobs_k
        self.greedy_sampling = greedy_sampling
        self.logits_processors = logits_processors or []
        self.thinking_budget_criteria = thinking_budget_criteria or []
        self.token_context = [list(ctx) for ctx in (token_context or [])]
        self._ensure_token_context()

        self._current_tokens = None
        self._current_lps = None
        self._next_tokens = inputs
        self._next_lps = None
        self._next_top_idx = None
        self._next_top_lp = None

        # Per-sequence MRoPE delta
        self._rope_deltas = None

    def __len__(self):
        return len(self.uids)

    def cache_states(self):
        return [c.state for c in self.prompt_cache if hasattr(c, "state")]

    def _ensure_logits_processor_slots(self, *, force: bool = False):
        if not (force or (self.logits_processors and any(self.logits_processors))):
            return
        if len(self.logits_processors) < len(self.uids):
            missing = len(self.uids) - len(self.logits_processors)
            self.logits_processors.extend([None] * missing)
        elif len(self.logits_processors) > len(self.uids):
            self.logits_processors = self.logits_processors[: len(self.uids)]

    def _ensure_token_context(self, *, force: bool = False):
        if not (force or (self.logits_processors and any(self.logits_processors))):
            if not self.logits_processors:
                self.token_context = []
            return
        if len(self.token_context) < len(self.uids):
            missing = len(self.uids) - len(self.token_context)
            self.token_context.extend([[] for _ in range(missing)])
        elif len(self.token_context) > len(self.uids):
            self.token_context = self.token_context[: len(self.uids)]

    def _fused_greedy_step(self, inputs: mx.array, fwd_kwargs: dict):
        if not self.greedy_sampling or self.compute_logprobs or self.top_logprobs_k > 0:
            return None

        fused_greedy_decode = getattr(self._language_model, "fused_greedy_decode", None)
        if not callable(fused_greedy_decode):
            return None

        decode_kwargs = dict(fwd_kwargs)
        if self.logits_processors and any(self.logits_processors):
            supports_processors = getattr(
                self._language_model, "supports_fused_greedy_logits_processors", None
            )
            if not callable(supports_processors) or not supports_processors(
                self.logits_processors
            ):
                return None
            decode_kwargs["logits_processors"] = self.logits_processors
        sampled = fused_greedy_decode(
            inputs[:, None],
            cache=self.prompt_cache,
            **decode_kwargs,
        )
        if sampled is None:
            return None
        if sampled.ndim == 2 and sampled.shape[1] == 1:
            sampled = sampled[:, 0]
        return sampled

    def _step(self):
        """Perform one generation step with double buffering."""
        self._current_tokens = self._next_tokens
        self._current_lps = self._next_lps
        inputs = self._current_tokens

        fwd_kwargs = {}
        if self._rope_deltas is not None:
            fwd_kwargs["rope_deltas"] = self._rope_deltas

        sampled = self._fused_greedy_step(inputs, fwd_kwargs)
        if sampled is not None:
            self._next_tokens = sampled
            self._next_lps = None
            self._next_top_idx = None
            self._next_top_lp = None
            mx.async_eval(self._next_tokens)
            mx.eval(inputs)
            return inputs.tolist(), None, None, None

        output = self._language_model(
            inputs[:, None], cache=self.prompt_cache, **fwd_kwargs
        )
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits[:, -1, :]

        if self.logits_processors and any(self.logits_processors):
            last_tokens = inputs.tolist()
            self._ensure_token_context()
            for i, token in enumerate(last_tokens):
                self.token_context[i].append(token)

            processed_logits = []
            for i in range(logits.shape[0]):
                sample_logits = logits[i : i + 1]
                processors = self.logits_processors[i] or []
                for processor in processors:
                    if hasattr(processor, "process_last_token"):
                        sample_logits = processor.process_last_token(
                            last_tokens[i], sample_logits
                        )
                    else:
                        sample_logits = processor(
                            mx.array(self.token_context[i]), sample_logits
                        )
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = _sample_with_positions(
            self.sampler,
            logprobs,
            row_ids=[0] * len(self.uids),
            positions=[n + 1 for n in self._num_tokens],
        )

        self._next_tokens = sampled
        prev_top_idx = self._next_top_idx
        prev_top_lp = self._next_top_lp

        eval_targets = [self._next_tokens]
        if self.compute_logprobs:
            self._next_lps = logprobs[mx.arange(sampled.shape[0]), sampled]
            eval_targets.append(self._next_lps)
        else:
            self._next_lps = None

        k = self.top_logprobs_k
        if k > 0:
            # argsort ascending; take last K columns and reverse for descending.
            sort_idx = mx.argsort(logprobs, axis=-1)
            top_idx = sort_idx[..., -k:][..., ::-1].astype(mx.int32)
            top_lp = mx.take_along_axis(logprobs, top_idx, axis=-1)
            self._next_top_idx = top_idx
            self._next_top_lp = top_lp
            eval_targets.extend([top_idx, top_lp])
        else:
            self._next_top_idx = None
            self._next_top_lp = None

        mx.async_eval(*eval_targets)

        if self._current_lps is not None:
            to_eval = [inputs, self._current_lps]
            if prev_top_idx is not None:
                to_eval.extend([prev_top_idx, prev_top_lp])
            mx.eval(*to_eval)
            top_idx_list = prev_top_idx.tolist() if prev_top_idx is not None else None
            top_lp_list = prev_top_lp.tolist() if prev_top_lp is not None else None
            return (
                inputs.tolist(),
                self._current_lps.tolist(),
                top_idx_list,
                top_lp_list,
            )
        else:
            mx.eval(inputs)
            return inputs.tolist(), None, None, None

    def _eval_pending_state(self):
        """Materialize lazy decode outputs before mutating batch-owned state."""
        targets = []

        def append_arrays(value):
            if isinstance(value, mx.array):
                targets.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    append_arrays(item)

        append_arrays(
            (
                self._current_tokens,
                self._current_lps,
                self._next_tokens,
                self._next_lps,
                self._next_top_idx,
                self._next_top_lp,
                self._rope_deltas,
            )
        )
        for c in self.prompt_cache:
            try:
                append_arrays(c.state)
            except (AttributeError, TypeError):
                pass

        if targets:
            mx.eval(*targets)

    def extend(self, other: "GenerationBatch"):
        """Extend this batch with another generation batch."""
        self_was_empty = len(self.uids) == 0
        if not self_was_empty and len(other.uids) > 0:
            self._eval_pending_state()
            other._eval_pending_state()

        self_has_processors = self.logits_processors and any(self.logits_processors)
        other_has_processors = other.logits_processors and any(other.logits_processors)
        if self_has_processors or other_has_processors:
            self._ensure_logits_processor_slots(force=bool(other_has_processors))
            other._ensure_logits_processor_slots(force=bool(self_has_processors))
            self._ensure_token_context(force=bool(other_has_processors))
            other._ensure_token_context(force=bool(self_has_processors))
        else:
            self.token_context = []
            other.token_context = []
            self.logits_processors = []
            other.logits_processors = []

        self.uids.extend(other.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, other.prompt_cache)
        self.max_tokens.extend(other.max_tokens)
        self._num_tokens.extend(other._num_tokens)
        self.token_context.extend(other.token_context)
        self.logits_processors.extend(other.logits_processors)
        self.thinking_budget_criteria.extend(other.thinking_budget_criteria)
        self._ensure_logits_processor_slots()
        self._ensure_token_context()

        if self._current_tokens is None:
            self._current_tokens = other._current_tokens
            self._current_lps = other._current_lps
        elif other._current_tokens is not None:
            self._current_tokens = mx.concatenate(
                [self._current_tokens, other._current_tokens]
            )
            if self._current_lps is not None and other._current_lps is not None:
                self._current_lps = mx.concatenate(
                    [self._current_lps, other._current_lps]
                )

        if self._next_tokens is None:
            self._next_tokens = other._next_tokens
            self._next_lps = other._next_lps
            self._next_top_idx = other._next_top_idx
            self._next_top_lp = other._next_top_lp
        elif other._next_tokens is not None:
            self._next_tokens = mx.concatenate([self._next_tokens, other._next_tokens])
            if self._next_lps is not None and other._next_lps is not None:
                self._next_lps = mx.concatenate([self._next_lps, other._next_lps])

            if (
                self._next_top_idx is not None
                and other._next_top_idx is not None
                and self._next_top_idx.shape[-1] == other._next_top_idx.shape[-1]
            ):
                self._next_top_idx = mx.concatenate(
                    [self._next_top_idx, other._next_top_idx]
                )
                self._next_top_lp = mx.concatenate(
                    [self._next_top_lp, other._next_top_lp]
                )
            else:
                self._next_top_idx = None
                self._next_top_lp = None

        if self_was_empty:
            self._rope_deltas = other._rope_deltas
        elif (self._rope_deltas is None) != (other._rope_deltas is None):
            raise RuntimeError(
                "extend() mixes MRoPE and non-MRoPE batches; both sides must "
                "carry rope_deltas or neither side may."
            )
        elif self._rope_deltas is not None:
            self._rope_deltas = mx.concatenate([self._rope_deltas, other._rope_deltas])

    def filter(self, keep: List[int]):
        """Filter the batch to keep only the specified indices."""
        if len(keep) < len(self.uids):
            self._eval_pending_state()

        self.uids = [self.uids[idx] for idx in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self._num_tokens = [self._num_tokens[idx] for idx in keep]
        if self.token_context:
            self.token_context = [self.token_context[idx] for idx in keep]
        if self.logits_processors:
            self.logits_processors = [self.logits_processors[idx] for idx in keep]
        if self.thinking_budget_criteria:
            self.thinking_budget_criteria = [
                self.thinking_budget_criteria[idx] for idx in keep
            ]

        if not keep:
            self.prompt_cache.clear()
            self._current_tokens = None
            self._current_lps = None
            self._next_tokens = None
            self._next_lps = None
            self._next_top_idx = None
            self._next_top_lp = None
            self._rope_deltas = None
            self.token_context = []
            self.logits_processors = []
            self.thinking_budget_criteria = []
        else:
            keep_arr = mx.array(keep, mx.int32)
            for c in self.prompt_cache:
                c.filter(keep_arr)
            if self._next_tokens is not None:
                self._next_tokens = self._next_tokens[keep_arr]
            if self._next_lps is not None:
                self._next_lps = self._next_lps[keep_arr]
            if self._next_top_idx is not None:
                self._next_top_idx = self._next_top_idx[keep_arr]
                self._next_top_lp = self._next_top_lp[keep_arr]
            if self._rope_deltas is not None:
                self._rope_deltas = self._rope_deltas[keep_arr]

    def next(self) -> List[Response]:
        """Generate the next batch of tokens."""
        if not self.uids:
            return []

        tokens, lp_list, top_idx_list, top_lp_list = self._step()

        keep = []
        responses = []
        forced_next_tokens = [None] * len(self.uids)
        for i in range(len(self.uids)):
            finish_reason = None
            self._num_tokens[i] += 1
            tok = tokens[i]
            if (
                i < len(self.thinking_budget_criteria)
                and self.thinking_budget_criteria[i] is not None
            ):
                criteria = self.thinking_budget_criteria[i]
                criteria(tok)
                forced_next_tokens[i] = criteria.pop_forced_token_id()

            if self.stop_criteria(tok):
                finish_reason = "stop"
            elif self._num_tokens[i] >= self.max_tokens[i]:
                finish_reason = "length"

            if finish_reason is None:
                keep.append(i)

            top_lp = None
            if top_idx_list is not None:
                top_lp = list(zip(top_idx_list[i], top_lp_list[i]))

            responses.append(
                self.Response(
                    uid=self.uids[i],
                    token=tok,
                    token_logprob=lp_list[i] if lp_list is not None else 0.0,
                    finish_reason=finish_reason,
                    top_logprobs=top_lp,
                )
            )

        has_forced_next_tokens = any(token is not None for token in forced_next_tokens)
        if has_forced_next_tokens:
            force_mask = mx.array(
                [token is not None for token in forced_next_tokens], dtype=mx.bool_
            )
            replacements = mx.array(
                [token if token is not None else 0 for token in forced_next_tokens],
                dtype=self._next_tokens.dtype,
            )
            self._next_tokens = mx.where(force_mask, replacements, self._next_tokens)

        if has_forced_next_tokens:
            mx.async_eval(self._next_tokens)

        if len(keep) < len(self.uids):
            self.filter(keep)

        return responses

    @classmethod
    def empty(
        cls,
        model,
        sampler,
        stop_criteria,
        compute_logprobs=True,
        top_logprobs_k=0,
        greedy_sampling: bool = False,
    ):
        """Create an empty generation batch."""
        batch = cls.__new__(cls)
        batch.model = model
        batch._language_model = getattr(model, "language_model", model)
        batch.uids = []
        batch.prompt_cache = []
        batch.sampler = sampler
        batch.stop_criteria = stop_criteria
        batch.max_tokens = []
        batch._num_tokens = []
        batch.compute_logprobs = compute_logprobs
        batch.top_logprobs_k = top_logprobs_k
        batch.greedy_sampling = greedy_sampling
        batch.token_context = []
        batch.logits_processors = []
        batch.thinking_budget_criteria = []
        batch._current_tokens = None
        batch._current_lps = None
        batch._next_tokens = None
        batch._next_lps = None
        batch._next_top_idx = None
        batch._next_top_lp = None
        batch._rope_deltas = None
        return batch


class SpeculativeGenerationBatch:
    """GenerationBatch-compatible wrapper for server-side MTP decode."""

    is_speculative = True
    Response = GenerationBatch.Response

    def __init__(
        self,
        model: nn.Module,
        draft_model: nn.Module,
        draft_kind: str,
        uids: List[int],
        first_tokens: mx.array,
        prompt_cache: List[Any],
        sampler: Callable[[mx.array], mx.array],
        stop_criteria,
        max_tokens: List[int],
        hidden: mx.array,
        shared_kv_states: Optional[dict],
        prompt_tokens: mx.array,
        *,
        draft_block_size: Optional[int] = None,
        token_dtype: mx.Dtype = mx.int32,
        greedy_sampling: bool = False,
        target_hidden_offset: int = 0,
    ):
        self.model = model
        self.draft_model = draft_model
        self.draft_kind = draft_kind
        self.uids = list(uids)
        self._all_uids = list(uids)
        self.first_tokens = first_tokens
        self.prompt_cache = prompt_cache
        self.sampler = sampler
        self.stop_criteria = stop_criteria
        self.max_tokens = list(max_tokens)
        self.hidden = hidden
        self.shared_kv_states = shared_kv_states
        self.prompt_tokens = prompt_tokens
        self.draft_block_size = draft_block_size
        self.token_dtype = token_dtype
        self.greedy_sampling = greedy_sampling
        # Rows the prefill trimmed off the front of the drafter's context.  The
        # drafter discards that prefix itself when it is handed the whole prompt,
        # and adds its width to every draft cache offset; when the prefill
        # discarded it first the offset has to be supplied here or the drafter's
        # absolute RoPE positions move.  Same contract the single-stream path
        # carries as ``target_hidden_offset`` (``generate_step``).
        self.target_hidden_offset = int(target_hidden_offset or 0)
        self._num_tokens = [0] * len(uids)
        self._finished = [False] * len(uids)
        self._sent_first = False
        self._rounds_iter = None

    def __len__(self):
        return sum(not done for done in self._finished)

    def _refresh_uids(self):
        self.uids = [
            uid for uid, done in zip(self._all_uids, self._finished) if not done
        ]

    def extend(self, other: "SpeculativeGenerationBatch"):
        if len(self) == 0:
            self.__dict__.update(other.__dict__)
            return
        raise RuntimeError("Cannot extend an active speculative generation batch.")

    def filter(self, keep: List[int]):
        keep_uids = {self.uids[idx] for idx in keep}
        for i, uid in enumerate(self._all_uids):
            if uid not in keep_uids:
                self._finished[i] = True
        self._refresh_uids()

    def cache_states(self):
        return [c.state for c in self.prompt_cache if hasattr(c, "state")]

    def _finish_reason(self, row: int, token: int) -> Optional[str]:
        if self.stop_criteria(token):
            return "stop"
        if self._num_tokens[row] >= self.max_tokens[row]:
            return "length"
        return None

    def _append_token_responses(
        self,
        responses: List[GenerationBatch.Response],
        tok_list: List[Optional[int]],
    ) -> None:
        for row, token in enumerate(tok_list):
            if token is None or self._finished[row]:
                continue
            token = int(token)
            self._num_tokens[row] += 1
            finish_reason = self._finish_reason(row, token)
            if finish_reason is not None:
                self._finished[row] = True
            responses.append(
                self.Response(
                    uid=self._all_uids[row],
                    token=token,
                    token_logprob=0.0,
                    finish_reason=finish_reason,
                )
            )

    def _start_rounds(self):
        if self._rounds_iter is not None:
            return

        def stop_check(seq_idx, token_id):
            return (
                self._finished[seq_idx]
                or self.stop_criteria(token_id)
                or self._num_tokens[seq_idx] >= self.max_tokens[seq_idx]
            )

        self._rounds_iter = run_speculative_server_rounds(
            self.model,
            self.draft_model,
            self.prompt_cache,
            self.hidden,
            draft_kind=self.draft_kind,
            first_bonus=self.first_tokens,
            max_tokens=max(self.max_tokens) if self.max_tokens else 0,
            sampler=self.sampler,
            draft_block_size=self.draft_block_size,
            token_dtype=self.token_dtype,
            stop_check=stop_check,
            greedy_sampling=self.greedy_sampling,
            shared_kv_states=self.shared_kv_states,
            eos_token_ids=None,
            prompt_tokens=self.prompt_tokens,
            row_ids=[0] * len(self._all_uids),
            target_hidden_offset=self.target_hidden_offset,
        )

    def next(self) -> List[GenerationBatch.Response]:
        if len(self) == 0:
            return []

        responses: List[GenerationBatch.Response] = []
        if not self._sent_first:
            self._sent_first = True
            mx.eval(self.first_tokens)
            for row, token in enumerate(self.first_tokens.tolist()):
                if self._finished[row]:
                    continue
                token = int(token)
                self._num_tokens[row] += 1
                finish_reason = self._finish_reason(row, token)
                if finish_reason is not None:
                    self._finished[row] = True
                responses.append(
                    self.Response(
                        uid=self._all_uids[row],
                        token=token,
                        token_logprob=0.0,
                        finish_reason=finish_reason,
                    )
                )
            self._refresh_uids()
            return responses

        self._start_rounds()
        try:
            tok_list, round_meta = next(self._rounds_iter)
        except StopIteration:
            for row, done in enumerate(self._finished):
                if not done:
                    self._finished[row] = True
                    responses.append(
                        self.Response(
                            uid=self._all_uids[row],
                            token=None,
                            token_logprob=0.0,
                            finish_reason="length",
                        )
                    )
            self._refresh_uids()
            return responses

        self._append_token_responses(responses, tok_list)
        while isinstance(round_meta, dict) and int(
            round_meta.get("round_pos", 0)
        ) + 1 < int(round_meta.get("round_len", 1)):
            try:
                tok_list, round_meta = next(self._rounds_iter)
            except StopIteration:
                break
            self._append_token_responses(responses, tok_list)

        self._refresh_uids()
        return responses


class PromptProcessingBatch:
    """
    Handles VLM prompt processing with inputs_embeds and chunked prefill.

    Processes prompt tokens incrementally (one chunk per step) to allow
    interleaving with generation for continuous batching. Transitions to
    a GenerationBatch when prompt processing is complete.
    """

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        input_ids: List[List[int]],
        max_tokens: List[int],
        inputs_embeds: mx.array,
        prompt_kwargs: dict,
        logits_processors: Optional[
            List[Optional[List[Callable[[mx.array, mx.array], mx.array]]]]
        ] = None,
        thinking_budget_criteria: Optional[List[Any]] = None,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
        kv_bits=None,
        kv_key_bits=None,
        kv_value_bits=None,
        kv_key_scheme=None,
        kv_value_scheme=None,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
        quantized_kv_start: int = 0,
        warm_cache: Optional[List[Any]] = None,
        apc_meta: Optional[List[dict]] = None,
        apc_manager: Optional["_apc.APCManager"] = None,
        right_pad_per_row: Optional[List[int]] = None,
        existing_left_padding: Optional[List[int]] = None,
        suffix_lens: Optional[List[int]] = None,
        apc_mode: Optional[str] = None,
        vault: Optional[Any] = None,
        draft_model: Optional[nn.Module] = None,
        draft_kind: Optional[str] = None,
        draft_block_size: Optional[int] = None,
        greedy_sampling: bool = False,
    ):
        self.model = model
        self.uids = uids
        self._prompt_uids = list(uids)
        self.max_tokens = max_tokens
        self.prefill_step_size = prefill_step_size
        self.draft_model = draft_model
        self.draft_kind = draft_kind
        self.draft_block_size = draft_block_size
        self.greedy_sampling = greedy_sampling

        lengths = [len(ids) for ids in input_ids]
        max_length = max(lengths)
        # ``input_ids`` here are the per-row prefill inputs — for warm-start
        # rows this is the suffix, for cold rows the full prompt. When
        # ``right_pad_per_row`` is set the rows are right-padded (used in
        # mixed warm/cold prefill so suffix RoPE positions align). Otherwise
        # we left-pad as before.
        self._right_pad_per_row = right_pad_per_row
        self._suffix_lens = suffix_lens or lengths
        self._left_padding_per_row: List[int]

        if right_pad_per_row is not None:
            # Right-pad each row to max_length (so the last `pad[i]` cells are
            # right-pad and need to be rolled into left-pad by finalize()).
            left_padding = [0] * len(input_ids)
            self._input_ids = _right_pad_prompts(input_ids, max_length=max_length)
        else:
            left_padding = [max_length - l for l in lengths]
            if existing_left_padding is not None:
                left_padding = [
                    pad + int(existing)
                    for pad, existing in zip(left_padding, existing_left_padding)
                ]
            self._input_ids = _left_pad_prompts(input_ids, max_length=max_length)
        self._left_padding_per_row = list(left_padding)
        # ``prompt_step`` consumes ``_input_ids`` one chunk at a time, so anything
        # that needs the WHOLE prompt -- ``prompt_tokens`` on the speculative round
        # loop, which is the prompt-lookup drafter's n-gram corpus -- must be handed
        # a snapshot taken before the loop starts.  Passing ``self._input_ids`` at
        # the round-loop call site instead hands over only the tail that survived
        # chunking.  ``generate_step`` keeps the same snapshot as ``full_prompt_ids``.
        #
        # This is only the PREFILL input, though: on a warm row it is the suffix, and
        # the prefix came from a cache.  Refined below once ``_apc_meta`` is known --
        # see ``_whole_prompt_ids_rows``.
        self._speculative_prompt_ids = self._input_ids
        self._total_prompt_tokens = sum(lengths)
        self._processed_prompt_columns = 0

        # Absolute column, in the padded prefill sequence, of each row's LAST
        # REAL token -- the only position whose logits are ever sampled.  A
        # right-padded row's real tokens stop BEFORE the sequence does, so once
        # the prefill chunks, that column can land in an earlier chunk than the
        # final forward; ``generate()`` used to index it off the final forward's
        # width alone, which goes negative for such a row.  ``prompt_step()``
        # now captures the row's ``[1, vocab]`` slice wherever it lands.
        # ``None`` -- the left-padded and unpadded cases, i.e. every batch that
        # is not a mixed warm/cold one -- keeps ``prompt_step`` on exactly the
        # code path it took before: there, every row's last real token is the
        # final column of the final forward.
        self._last_real_column: Optional[List[int]] = None
        self._captured_last_logits: List[Optional[mx.array]] = []
        if right_pad_per_row is not None and any(right_pad_per_row):
            width = self._input_ids.shape[1]
            self._last_real_column = [width - 1 - int(p) for p in right_pad_per_row]
            self._captured_last_logits = [None] * len(right_pad_per_row)

        self.logits_processors = logits_processors or []
        self.thinking_budget_criteria = thinking_budget_criteria or []
        self._token_context = (
            [list(ids) for ids in input_ids]
            if self.logits_processors and any(self.logits_processors)
            else []
        )
        self._inputs_embeds = inputs_embeds
        self._prompt_kwargs = prompt_kwargs or {}
        self._prompt_length_aware_keys: List[str] = []
        if self._prompt_kwargs and self._inputs_embeds is not None:
            prompt_batch = self._inputs_embeds.shape[0]
            prompt_len = self._inputs_embeds.shape[1]
            for k, v in self._prompt_kwargs.items():
                if (
                    isinstance(v, mx.array)
                    and _prompt_kwarg_batch_size(k, v) == prompt_batch
                    and _is_sequence_aligned_prompt_kwarg(k, v, prompt_len)
                ):
                    self._prompt_length_aware_keys.append(k)

        # APC metadata used for post-prefill block harvest (per-row).
        self._apc_meta = apc_meta or []
        self._apc_manager = apc_manager
        self._apc_mode = apc_mode
        # Warm Context Vault.  ``None`` unless the caller passed one, and every
        # vault branch below is gated on that, so with the vault off this class
        # takes exactly the code path it took before.
        self._vault = vault
        self._apc_harvest_enabled = True
        self._prompt_time_s = 0.0
        self._prompt_tokens_per_row: List[int] = []
        self._cached_tokens_per_row: List[int] = []
        self._cached_from_width_per_row: List[Optional[int]] = []
        for idx, suffix_len in enumerate(lengths):
            full_input_ids = None
            prefix_len = 0
            harvest_width = None
            if idx < len(self._apc_meta) and self._apc_meta[idx] is not None:
                full_input_ids = self._apc_meta[idx].get("full_input_ids")
                prefix_len = int(self._apc_meta[idx].get("prefix_len") or 0)
                harvest_width = _harvest_prov.batch_width_of(
                    self._apc_meta[idx].get("harvest_provenance")
                )
            self._prompt_tokens_per_row.append(
                len(full_input_ids) if full_input_ids is not None else suffix_len
            )
            self._cached_tokens_per_row.append(prefix_len)
            self._cached_from_width_per_row.append(
                harvest_width if prefix_len > 0 else None
            )

        if warm_cache is not None:
            self.prompt_cache = warm_cache
        elif draft_model is not None and draft_kind is not None:
            self.prompt_cache = make_speculative_prompt_cache(
                model,
                draft_kind=draft_kind,
                batch_size=len(input_ids),
                left_padding=left_padding,
                make_cache=lambda lm, lp: _make_cache(
                    lm,
                    lp,
                    kv_bits=kv_bits,
                    kv_key_bits=kv_key_bits,
                    kv_value_bits=kv_value_bits,
                    kv_key_scheme=kv_key_scheme,
                    kv_value_scheme=kv_value_scheme,
                    kv_group_size=kv_group_size,
                    kv_quant_scheme=kv_quant_scheme,
                    quantized_kv_start=quantized_kv_start,
                    prefill_length=max_length,
                ),
            )
        elif (
            len(input_ids) == 1
            and right_pad_per_row is None
            and kv_bits is None
            and hasattr(model, "make_cache")
        ):
            self.prompt_cache = cache.make_prompt_cache(model)
        else:
            self.prompt_cache = _make_cache(
                model,
                left_padding,
                kv_bits=kv_bits,
                kv_key_bits=kv_key_bits,
                kv_value_bits=kv_value_bits,
                kv_key_scheme=kv_key_scheme,
                kv_value_scheme=kv_value_scheme,
                kv_group_size=kv_group_size,
                kv_quant_scheme=kv_quant_scheme,
                quantized_kv_start=quantized_kv_start,
                prefill_length=max_length,
            )

        # Declare per-row right-padding on each cache so finalize() can roll
        # it into left-padding once the prefill forward pass is complete.
        if right_pad_per_row is not None and any(right_pad_per_row):
            for c in self.prompt_cache:
                prepare = getattr(c, "prepare", None)
                if not callable(prepare):
                    self._apc_harvest_enabled = False
                    self._release_apc_meta_blocks()
                    raise RuntimeError(
                        "APC mixed prefill requires a prompt cache with prepare()"
                    )
                prepare(right_padding=right_pad_per_row, lengths=self._suffix_lens)

        if self.prefill_step_size is not None:
            policy_kwargs = dict(self._prompt_kwargs)
            if draft_model is not None and draft_kind is not None:
                policy_kwargs.update(
                    speculative_prefill_kwargs(draft_kind, draft_model)
                )
            if not _chunked_prefill_enabled(
                self.model,
                input_ids=self._input_ids,
                inputs_embeds=self._inputs_embeds,
                prompt_cache=self.prompt_cache,
                draft_model=draft_model,
                draft_kind=draft_kind,
                prefill_kwargs=policy_kwargs,
            ):
                self.prefill_step_size = None

        # This is the THIRD chunked-prefill driver in the tree, and it had the same
        # defect the other two were fixed for: ``prompt_step`` built its kwargs from
        # ``self._prompt_kwargs`` alone, so a batched request with a hidden-reading
        # drafter chunked its prompt and then handed the drafter only the final
        # forward -- a one-chunk context -- while ``chunked_prefill_policy`` had
        # already admitted the chunking on the strength of the capture being asked
        # for.  Carry the same capture on every chunk and stitch the pieces back;
        # see ``generate_step`` and ``server/generation.py::
        # _run_chunked_speculative_prefill`` for the identical shape.
        self._prefill_capture_kwargs: dict = {}
        self._chunk_capture_kwargs: dict = {}
        self.target_hidden_offset = 0
        if draft_model is not None and draft_kind is not None:
            # Prefill leg: hidden captures yes, KDA rollback stash no.
            self._prefill_capture_kwargs = prefill_capture_kwargs(
                self.model,
                speculative_prefill_kwargs(draft_kind, draft_model),
            )
            # Only a per-layer capture (``capture_layer_ids``), or MTP's
            # ``return_hidden`` with the server-priming window on, survives
            # being split across chunks and stitched back on the time axis --
            # see ``chunk_capture_kwargs_for``.
            self._chunk_capture_kwargs = chunk_capture_kwargs_for(
                self._prefill_capture_kwargs
            )

        # ``prompt_tokens`` must be the WHOLE prompt, and on a warm row the prefill
        # input is only the suffix.  ``_build_mixed_prompt_batch`` records the whole
        # thing per row as ``apc_meta[i]["full_input_ids"]``; recover it from there.
        whole_rows = self._whole_prompt_ids_rows(input_ids, self._apc_meta)
        if whole_rows is not None:
            self._speculative_prompt_ids = _left_pad_prompts(whole_rows)

        # Two refusals, both of them policy rather than mechanism, and both of them
        # costing memory rather than correctness.  Each names the TRIM it refuses;
        # only the second one also refuses the CHUNKING.
        #
        # 1.  A RIGHT-PADDED batch (the mixed warm/cold driver,
        #     ``_build_mixed_prompt_batch``) is refused the TRIM ONLY.  The drafter's
        #     window is the TRAILING rows of the capture, and for a short right-padded
        #     row those rows are padding, so a trimmed context would hand that row a
        #     window of zeros.  That objection is about the trim and nothing else.
        #
        #     NARROWED 2026-09-03, on this merge.  The blanket version of this refusal
        #     also declined the chunking, for two reasons that no longer hold:
        #       * "a chunk boundary is not the same column of real content in every
        #         row, so the stitched pieces do not line up".  They do line up when
        #         ``keep is None``: ``PrefillHiddenAccumulator._prune`` is then a
        #         no-op and ``finish()`` concatenates every chunk on the time axis,
        #         which reproduces the full padded width column for column, exactly
        #         as an unchunked capture does.  Misalignment is a property of the
        #         trim (which counts back from a prompt end that differs per row),
        #         not of the stitch.  Measured on the section-9 fixture at chunk
        #         4/8/12/16/24/32: capture ``[2, 40, 256]``, offset 0, agreeing with
        #         the unchunked capture to <= 4.2e-07 -- the same KDA scan-split drift
        #         class this file already records for the left-padded batch path --
        #         and the prompt cache BIT-EQUAL to the unchunked arm in every one of
        #         those arms.
        #       * "chunking walks into the negative last-real-token index".  That
        #         defect is FIXED on this branch: ``_last_real_column`` /
        #         ``_capture_last_real_logits`` above capture each row's ``[1, vocab]``
        #         slice in whatever chunk its last real token lands in, so chunked
        #         greedy selection on a right-padded batch is exact (measured: the
        #         same tokens as the unchunked arm at every chunk size).
        # 2.  A warm row whose prefix cannot be named (no ``full_input_ids``) means
        #     the drafter cannot be handed the whole prompt, so it does not get a
        #     trimmed one either -- and this one KEEPS the blanket form (chunking
        #     declined as well).  Nothing was measured about that case here, and a
        #     refusal is not narrowed on an argument alone.
        #
        # In both cases the capture stays FULL WIDTH and untrimmed, and
        # ``capture_gdn_states=False`` still rides on every forward, so the
        # sequence-shaped KDA rollback stash is still never built.
        self._capture_refusal: Optional[str] = None
        self._capture_refusal_declines_chunking = False
        if self._chunk_capture_kwargs:
            if right_pad_per_row is not None:
                self._capture_refusal = (
                    "right-padded batch (mixed warm/cold prefill): the drafter's "
                    "window is the trailing rows, which are padding for a short row"
                )
            elif whole_rows is None:
                self._capture_refusal = (
                    "a warm row's prefix is not recoverable (apc_meta carries no "
                    "full_input_ids), so the drafter cannot be handed the whole "
                    "prompt"
                )
                self._capture_refusal_declines_chunking = True
        if self._capture_refusal is not None:
            logger.info(
                "speculative prefill: declining the trailing-context trim%s for "
                "this batch -- %s. The capture stays full width and %s; "
                "capture_gdn_states is still off.",
                (
                    " and the chunked prefill"
                    if self._capture_refusal_declines_chunking
                    else ""
                ),
                self._capture_refusal,
                (
                    "the prefill runs in one forward"
                    if self._capture_refusal_declines_chunking
                    else "the prefill still chunks"
                ),
            )
            if self._capture_refusal_declines_chunking:
                self.prefill_step_size = None
        self._prefill_hidden = PrefillHiddenAccumulator(
            keep=(
                prefill_context_keep(draft_kind, draft_model)
                if self._chunk_capture_kwargs and self._capture_refusal is None
                else None
            )
        )

    @staticmethod
    def _whole_prompt_ids_rows(input_ids, apc_meta):
        """The whole prompt per row, or ``None`` if a warm row's prefix is unnameable.

        A cold row's prefill input IS its whole prompt.  A warm row's is the suffix;
        its whole prompt is ``apc_meta[i]["full_input_ids"]``.  Returning ``None``
        rather than a best effort is deliberate: a caller that silently used the
        suffix would hand the prompt-lookup drafter an n-gram corpus missing
        everything the cache already held.
        """
        rows = []
        for i, ids in enumerate(input_ids):
            meta = apc_meta[i] if i < len(apc_meta) else None
            full = (meta or {}).get("full_input_ids")
            prefix_len = int((meta or {}).get("prefix_len", 0) or 0)
            if full is not None and len(full) >= len(ids):
                rows.append(list(full))
            elif prefix_len > 0:
                return None
            else:
                rows.append(list(ids))
        return rows

    def __len__(self):
        return len(self.uids)

    def _release_apc_meta_blocks(self):
        if self._apc_manager is None:
            return
        for meta in self._apc_meta:
            if meta is not None:
                self._apc_manager.release(meta.get("apc_blocks", []))

    def needs_processing(self):
        """True if prompt needs chunked processing before generate()."""
        if self._inputs_embeds is None or self.prefill_step_size is None:
            return self._next_apc_checkpoint_column() is not None
        if self._next_apc_checkpoint_column() is not None:
            return True
        return self._inputs_embeds.shape[1] > self.prefill_step_size

    def _apc_checkpoint_column_for_meta(
        self, batch_idx: int, meta: dict
    ) -> Optional[int]:
        checkpoint_len = int(meta.get("checkpoint_len") or 0)
        if (
            self._apc_mode != "exact"
            or checkpoint_len <= 0
            or meta.get("checkpoint_done")
        ):
            return None
        prefix_len = int(meta.get("prefix_len", 0) or 0)
        if checkpoint_len <= prefix_len:
            meta["checkpoint_done"] = True
            return None
        return self._checkpoint_column_for_len(batch_idx, meta, checkpoint_len)

    def _checkpoint_column_for_len(
        self, batch_idx: int, meta: dict, target_len: int
    ) -> Optional[int]:
        """Batch column at which row ``batch_idx`` reaches ``target_len`` tokens.

        Split out of ``_apc_checkpoint_column_for_meta`` verbatim so the vault's
        boundary ladder is placed by the same arithmetic the single APC
        checkpoint has always used, rather than by a second copy of it.
        """
        prefix_len = int(meta.get("prefix_len", 0) or 0)
        if target_len <= prefix_len:
            return None
        if self._right_pad_per_row is not None:
            suffix_checkpoint = target_len - prefix_len
            if suffix_checkpoint >= self._suffix_lens[batch_idx]:
                return None
            return suffix_checkpoint
        return self._left_padding_per_row[batch_idx] + target_len

    def _vault_checkpoint_columns_for_meta(
        self, batch_idx: int, meta: dict
    ) -> List[int]:
        out: List[int] = []
        for target in meta.get("vault_rungs") or ():
            col = self._checkpoint_column_for_len(batch_idx, meta, int(target))
            if col is not None:
                out.append(col)
        return out

    def _next_apc_checkpoint_column(self) -> Optional[int]:
        """Column the next chunk must stop on, over APC's checkpoint and the vault's ladder.

        With ``self._vault is None`` this reduces term for term to what it was:
        ``apc_on`` reproduces the old two-clause guard, the vault list is empty,
        and the min is taken over the same single column per row.
        """
        if not self._apc_meta or self._inputs_embeds is None:
            return None
        apc_on = self._apc_manager is not None and self._apc_mode == "exact"
        if not apc_on and self._vault is None:
            return None
        start = self._processed_prompt_columns
        end = start + self._inputs_embeds.shape[1]
        next_col: Optional[int] = None
        for batch_idx, meta in enumerate(self._apc_meta):
            if meta is None:
                continue
            cols: List[int] = []
            if apc_on:
                col = self._apc_checkpoint_column_for_meta(batch_idx, meta)
                if col is not None:
                    cols.append(col)
            if self._vault is not None:
                cols.extend(self._vault_checkpoint_columns_for_meta(batch_idx, meta))
            for col in cols:
                if col <= start or col >= end:
                    continue
                next_col = col if next_col is None else min(next_col, col)
        return next_col

    def _row_real_tokens_processed(self, batch_idx: int) -> int:
        meta = self._apc_meta[batch_idx]
        prefix_len = int(meta.get("prefix_len", 0) or 0)
        if self._right_pad_per_row is not None:
            suffix_done = min(
                self._suffix_lens[batch_idx],
                max(0, self._processed_prompt_columns),
            )
            return prefix_len + suffix_done
        real_done = (
            self._processed_prompt_columns - self._left_padding_per_row[batch_idx]
        )
        return prefix_len + min(self._suffix_lens[batch_idx], max(0, real_done))

    def _apc_prompt_cache_for_store(self, batch_idx: int) -> Optional[List[Any]]:
        return _apc.snapshot_prompt_cache_row(self.prompt_cache, batch_idx)

    def _harvest_provenance(self, batch_idx: int) -> dict:
        """Where row ``batch_idx``'s snapshot is being taken FROM.

        The width is ``len(self._prompt_uids)`` -- the width of the prefill
        batch as admitted -- and not ``len(self.uids)``, which the decode loop
        shortens as rows finish.  A snapshot's provenance is a property of the
        forward that produced it, so it must not move when a sibling row exits.

        L1b-1 measured this width to be the CARRIER of a bit difference in the
        KDA recurrent snapshot: an equal-suffix, zero-padding B=2 batch
        (``right_pad_per_row=[0, 0]``) poisons the entry to the same sha as a
        right-padded one, so the pads are recorded as evidence rather than as
        the cause.
        """
        right_pad = 0
        if self._right_pad_per_row is not None and batch_idx < len(
            self._right_pad_per_row
        ):
            right_pad = int(self._right_pad_per_row[batch_idx] or 0)
        left_pad = 0
        if batch_idx < len(self._left_padding_per_row):
            left_pad = int(self._left_padding_per_row[batch_idx] or 0)
        # ``_prompt_uids`` is absent on a hand-built batch (several tests build
        # one with ``__new__`` and set only what the method under test reads);
        # ``uids`` is the honest fallback there and identical before any row
        # finishes, which for a PREFILL batch is always.
        width = len(getattr(self, "_prompt_uids", None) or self.uids) or 1
        meta = (getattr(self, "_apc_meta", []) or [])
        row_meta = (meta[batch_idx] or {}) if batch_idx < len(meta) else {}
        return _harvest_prov.make(
            width,
            prefix_len=int(row_meta.get("prefix_len") or 0),
            parent=row_meta.get("harvest_provenance"),
            right_pad=right_pad,
            left_pad=left_pad,
        )

    def _store_apc_exact_checkpoints(self) -> None:
        if self._apc_manager is None or self._apc_mode != "exact":
            return
        for batch_idx, meta in enumerate(self._apc_meta):
            if meta is None or meta.get("checkpoint_done"):
                continue
            checkpoint_len = int(meta.get("checkpoint_len") or 0)
            if checkpoint_len <= 0:
                continue
            if self._row_real_tokens_processed(batch_idx) != checkpoint_len:
                continue
            prompt_cache = self._apc_prompt_cache_for_store(batch_idx)
            if prompt_cache is None:
                continue
            self._apc_manager.store_exact_cache(
                meta["full_input_ids"][:checkpoint_len],
                prompt_cache,
                extra_hash=meta.get("extra_hash", 0),
                harvest_provenance=self._harvest_provenance(batch_idx),
            )
            meta["checkpoint_done"] = True

    def _store_vault_checkpoints(self) -> None:
        """Store every vault rung this chunk landed on, per row.

        Unlike the APC exact checkpoint there are many boundaries per row, so a
        rung is dropped from the row's pending list once passed rather than
        marked with a single done flag.  Storing is best-effort: a vault fault
        must never fail a request, and a rung that fails to capture is simply
        not stored (``capture_fragments`` returns None rather than a partial
        ladder, and ``insert`` refuses None).
        """
        if self._vault is None or not self._apc_meta:
            return
        for batch_idx, meta in enumerate(self._apc_meta):
            if meta is None:
                continue
            rungs = meta.get("vault_rungs")
            if not rungs:
                continue
            done = self._row_real_tokens_processed(batch_idx)
            landed = [r for r in rungs if int(r) == done]
            remaining = [r for r in rungs if int(r) > done]
            if landed:
                row_cache = self._apc_prompt_cache_for_store(batch_idx)
                if row_cache is not None:
                    full_ids = meta.get("full_input_ids") or []
                    provenance = self._harvest_provenance(batch_idx)
                    for r in landed:
                        try:
                            _context_vault.insert_checkpoint(
                                self._vault,
                                full_ids,
                                int(r),
                                _context_vault.capture_fragments(row_cache, int(r)),
                                harvest_provenance=provenance,
                            )
                        except Exception:  # noqa: BLE001 - storing is best-effort
                            pass
            meta["vault_rungs"] = remaining

    def _prompt_kwargs_for_step(self, n: Optional[int] = None) -> dict:
        if n is None or not self._prompt_length_aware_keys:
            return self._prompt_kwargs
        out = dict(self._prompt_kwargs)
        for k in self._prompt_length_aware_keys:
            out[k] = _slice_sequence_aligned_prompt_kwarg(k, out[k], stop=n)
        return out

    def _rows_ending_in_chunk(self, n: int) -> List[int]:
        """Rows whose last real token lies in the next ``n`` columns.

        Empty unless the batch is right-padded, so a left-padded or unpadded
        batch never enters the capture branch below.
        """
        if self._last_real_column is None:
            return []
        start = self._processed_prompt_columns
        return [
            i
            for i, col in enumerate(self._last_real_column)
            if start <= col < start + n
        ]

    def _capture_last_real_logits(self, chunk_out, rows: List[int]) -> List[mx.array]:
        """Keep each named row's last-real-token logits out of this chunk.

        Only a ``[1, vocab]`` slice per row is retained -- never the chunk's
        ``[B, chunk, vocab]`` projection, which is dropped with ``chunk_out``
        as soon as this returns.  ``mx.contiguous`` so the kept row owns its
        buffer rather than viewing the chunk's.
        """
        logits = chunk_out.logits if hasattr(chunk_out, "logits") else chunk_out
        if logits is None:
            raise RuntimeError(
                "chunked prefill of a right-padded batch needs the chunk's logits"
            )
        start = self._processed_prompt_columns
        width = logits.shape[1]
        captured = []
        for i in rows:
            j = self._last_real_column[i] - start
            if not 0 <= j < width:
                raise RuntimeError(
                    f"row {i}: last real token at chunk column {j}, chunk is {width} wide"
                )
            row_logits = mx.contiguous(logits[i : i + 1, j : j + 1, :].squeeze(1))
            self._captured_last_logits[i] = row_logits
            captured.append(row_logits)
        return captured

    def prompt_step(self) -> int:
        """Process one chunk of the prompt. Returns tokens processed."""
        if not self.needs_processing():
            return 0

        step = self.prefill_step_size or self._inputs_embeds.shape[1]
        n = min(step, self._inputs_embeds.shape[1] - 1)
        checkpoint_col = self._next_apc_checkpoint_column()
        if checkpoint_col is not None:
            n = min(n, checkpoint_col - self._processed_prompt_columns)
        if n <= 0:
            return 0
        prompt_kwargs = self._prompt_kwargs_for_step(n)
        # Which rows END in this chunk (right-padded batches only).  Reading it
        # off the recorded absolute column adds NOTHING to the forward's
        # arguments, so on a batch with no hidden-reading drafter the model
        # still sees byte-for-byte the call it saw before and the prompt cache
        # this chunk writes cannot move.
        capture_rows = self._rows_ending_in_chunk(n)
        # The speculative capture, on the other hand, IS an argument, and it has
        # to ride on every chunk or the drafter is handed a one-chunk context.
        # The two are independent: this dict is empty without a hidden-reading
        # drafter, and ``capture_rows`` is empty without right padding.
        if self._chunk_capture_kwargs:
            prompt_kwargs = {**prompt_kwargs, **self._chunk_capture_kwargs}
        chunk_out = self.model(
            self._input_ids[:, :n],
            cache=self.prompt_cache,
            inputs_embeds=self._inputs_embeds[:, :n],
            n_to_process=n,
            **prompt_kwargs,
        )
        # BOTH readers take what they need out of the ONE bound output, in this
        # order, before it is released:
        #   1. the last-real-token logits of every row that ends in this chunk,
        #      kept as a ``[1, vocab]`` slice per row (right-padded batches only);
        #   2. this chunk's per-layer hidden, appended to the accumulator that
        #      stitches the whole-prompt context for a hidden-reading drafter.
        # Neither reader is aware of the other; both are no-ops when their own
        # precondition is absent.
        last_real_pending: List[mx.array] = []
        if capture_rows:
            last_real_pending = self._capture_last_real_logits(chunk_out, capture_rows)
        if self._chunk_capture_kwargs:
            self._prefill_hidden.append(chunk_out)
        # Only now drop the chunk's logits (and any gdn stash) -- BEFORE the eval
        # below, so neither the vocab-wide projection of a chunk nobody samples
        # from nor a sequence-shaped KDA rollback stash is ever materialised.
        # Without either reader this is exactly the old statement-expression call.
        chunk_out = None
        # Both pendings are empty unless their reader is active, so the plain
        # greedy no-drafter path schedules the same single argument it always did.
        # Scheduling them is not optional: an unevaluated capture is a graph node
        # that pins every intermediate behind it
        # (``PrefillHiddenAccumulator.pending``), and an unevaluated ``[1, vocab]``
        # slice would pin the chunk projection this statement just dropped.
        mx.async_eval(
            [c.state for c in self.prompt_cache]
            + last_real_pending
            + self._prefill_hidden.pending()
        )
        self._processed_prompt_columns += n
        self._store_apc_exact_checkpoints()
        self._store_vault_checkpoints()
        self._inputs_embeds = self._inputs_embeds[:, n:]
        self._input_ids = self._input_ids[:, n:]
        for k in self._prompt_length_aware_keys:
            self._prompt_kwargs[k] = _slice_sequence_aligned_prompt_kwarg(
                k, self._prompt_kwargs[k], start=n
            )
        mx.clear_cache()
        return n

    def record_prompt_time(self, elapsed_s: float) -> None:
        self._prompt_time_s += max(0.0, float(elapsed_s))

    def prompt_progress(self) -> List[PromptProgress]:
        if self._prompt_time_s <= 0:
            return []
        return [
            PromptProgress(
                uid=uid,
                prompt_tokens=prompt_tokens,
                prompt_tps=prompt_tokens / self._prompt_time_s,
                prompt_time=self._prompt_time_s,
                cached_tokens=cached_tokens,
                cached_from_width=cached_from_width,
            )
            for uid, prompt_tokens, cached_tokens, cached_from_width in zip(
                self._prompt_uids,
                self._prompt_tokens_per_row,
                self._cached_tokens_per_row,
                self._cached_from_width_per_row,
            )
        ]

    def generate(
        self, sampler, stop_criteria, compute_logprobs=True, top_logprobs_k=0
    ) -> GenerationBatch:
        """Process final tokens and transition to GenerationBatch."""
        call_kwargs = dict(self._prompt_kwargs)
        # Prefill leg: hidden captures yes, KDA rollback stash no.  Computed once in
        # ``__init__`` so this forward and every chunk before it carry byte-identical
        # capture kwargs -- the accumulator raises if the capture width moves.
        call_kwargs.update(self._prefill_capture_kwargs)

        output = self.model(
            self._input_ids,
            cache=self.prompt_cache,
            inputs_embeds=self._inputs_embeds,
            **call_kwargs,
        )
        if self._chunk_capture_kwargs:
            # Stitch this forward's capture onto the chunks' and hand the drafter one
            # whole-prompt context.  ``finish()`` also reports the rows it trimmed off
            # the front, which the drafter is owed as a RoPE offset.
            self._prefill_hidden.append(output)
            stitched, self.target_hidden_offset = self._prefill_hidden.finish()
            if stitched is not None:
                output.hidden_states = stitched
        logits = output.logits if hasattr(output, "logits") else output
        if self._right_pad_per_row is not None and any(self._right_pad_per_row):
            # Per-row last *real* token sits at absolute column
            # (width - 1 - right_pad[i]); subtracting the columns the chunk loop
            # already consumed puts it in THIS forward.  Unchunked, nothing has
            # been consumed and this is the (seq - 1 - right_pad[i]) it always
            # was.  A row whose last real token fell in an earlier chunk was
            # captured there; its take index is a placeholder that is discarded.
            start = self._processed_prompt_columns
            last_col = self._last_real_column
            captures = self._captured_last_logits
            if last_col is None:
                # ``right_pad_per_row`` was attached after construction, so no
                # chunk could have captured anything: the whole prompt is in
                # this forward and the original formula is exact.
                seq = logits.shape[1]
                last_col = [seq - 1 - p for p in self._right_pad_per_row]
                captures = [None] * len(last_col)
                start = 0
            take = [
                0 if capture is not None else col - start
                for col, capture in zip(last_col, captures)
            ]
            last_idx = mx.array(take, dtype=mx.int32)[:, None, None]
            last_idx = mx.broadcast_to(last_idx, (logits.shape[0], 1, logits.shape[-1]))
            logits = mx.take_along_axis(logits, last_idx, axis=1).squeeze(1)
            if any(c is not None for c in captures):
                logits = mx.concatenate(
                    [
                        capture if capture is not None else logits[i : i + 1]
                        for i, capture in enumerate(captures)
                    ],
                    axis=0,
                )
        else:
            logits = logits[:, -1, :]
        if self.logits_processors and any(self.logits_processors):
            processed_logits = []
            for i in range(logits.shape[0]):
                sample_logits = logits[i : i + 1]
                processors = self.logits_processors[i] or []
                for processor in processors:
                    sample_logits = processor(
                        mx.array(self._token_context[i]), sample_logits
                    )
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        first_tokens = _sample_with_positions(
            sampler,
            logprobs,
            row_ids=[0] * len(self.uids),
            positions=[0] * len(self.uids),
        )

        mx.async_eval(first_tokens)

        # Roll any right-padding into left-padding so the cache decoded by
        # GenerationBatch sees a canonical layout.
        if self._right_pad_per_row is not None and any(self._right_pad_per_row):
            for c in self.prompt_cache:
                finalize = getattr(c, "finalize", None)
                if not callable(finalize):
                    self._apc_harvest_enabled = False
                    self._release_apc_meta_blocks()
                    raise RuntimeError(
                        "APC mixed prefill requires a prompt cache with finalize()"
                    )
                finalize()
        if logger.isEnabledFor(logging.DEBUG) and os.environ.get("APC_DEBUG"):
            c0 = self.prompt_cache[0] if self.prompt_cache else None
            if c0 is not None:
                off = getattr(c0, "offset", None)
                lp = getattr(c0, "left_padding", None)
                logger.warning(
                    "post-prefill cache[0]: _idx=%s offset=%s left_padding=%s right_pad_per_row=%s suffix_lens=%s",
                    getattr(c0, "_idx", None),
                    off.tolist() if hasattr(off, "tolist") else off,
                    lp.tolist() if hasattr(lp, "tolist") else lp,
                    self._right_pad_per_row,
                    self._suffix_lens,
                )

        if self.draft_model is not None and self.draft_kind is not None:
            gen_batch = SpeculativeGenerationBatch(
                model=self.model,
                draft_model=self.draft_model,
                draft_kind=self.draft_kind,
                uids=list(self.uids),
                first_tokens=first_tokens,
                prompt_cache=self.prompt_cache,
                sampler=sampler,
                stop_criteria=stop_criteria,
                max_tokens=list(self.max_tokens),
                hidden=speculative_hidden_state(self.draft_kind, output),
                shared_kv_states=(
                    output.shared_kv_states if self.draft_kind == "mtp" else None
                ),
                prompt_tokens=self._speculative_prompt_ids,
                draft_block_size=self.draft_block_size,
                token_dtype=self._input_ids.dtype,
                greedy_sampling=self.greedy_sampling,
                target_hidden_offset=self.target_hidden_offset,
            )
            compute_logprobs = False
        else:
            gen_batch = GenerationBatch(
                model=self.model,
                uids=list(self.uids),
                inputs=first_tokens,
                prompt_cache=self.prompt_cache,
                sampler=sampler,
                stop_criteria=stop_criteria,
                max_tokens=list(self.max_tokens),
                top_logprobs_k=top_logprobs_k,
                greedy_sampling=self.greedy_sampling,
                token_context=[list(ctx) for ctx in self._token_context],
                logits_processors=list(self.logits_processors),
                thinking_budget_criteria=list(self.thinking_budget_criteria),
            )
        gen_batch.compute_logprobs = compute_logprobs

        if compute_logprobs and isinstance(gen_batch, GenerationBatch):
            gen_batch._next_lps = logprobs[
                mx.arange(first_tokens.shape[0]), first_tokens
            ]

        # Prime top-K buffers so the first token can emit top_logprobs too.
        if top_logprobs_k > 0 and isinstance(gen_batch, GenerationBatch):
            k = top_logprobs_k
            sort_idx = mx.argsort(logprobs, axis=-1)
            top_idx = sort_idx[..., -k:][..., ::-1].astype(mx.int32)
            top_lp = mx.take_along_axis(logprobs, top_idx, axis=-1)
            gen_batch._next_top_idx = top_idx
            gen_batch._next_top_lp = top_lp

        language_model = getattr(self.model, "language_model", self.model)
        rope_deltas = self._capture_rope_deltas_from_prompt_kwargs(
            call_kwargs, language_model, len(gen_batch.uids)
        )
        if rope_deltas is not None:
            gen_batch._rope_deltas = rope_deltas

        # Final prefill produces the first generated token and mutates the
        # prompt cache. Materialize that boundary before the decode loop so
        # the first decode step does not inherit the full lazy prefill graph.
        cache_states = []
        for c in self.prompt_cache:
            try:
                cache_states.append(c.state)
            except (AttributeError, TypeError):
                pass
        eval_targets = [first_tokens]
        if cache_states:
            eval_targets.append(cache_states)
        if compute_logprobs and isinstance(gen_batch, GenerationBatch):
            eval_targets.append(gen_batch._next_lps)
        if top_logprobs_k > 0 and isinstance(gen_batch, GenerationBatch):
            eval_targets.extend([gen_batch._next_top_idx, gen_batch._next_top_lp])
        if rope_deltas is not None:
            eval_targets.append(rope_deltas)
        mx.eval(*eval_targets)

        # APC: harvest the post-prefill K/V into hashed blocks. Done after the
        # final prefill forward but before the cache references are released
        # so the block tensors snapshot the prompt prefix.
        if (
            self._apc_manager is not None
            and self._apc_meta
            and self._apc_harvest_enabled
        ):
            try:
                for batch_idx, meta in enumerate(self._apc_meta):
                    if meta is None:
                        continue
                    provenance = self._harvest_provenance(batch_idx)
                    if self._apc_mode == "exact":
                        prompt_cache = self._apc_prompt_cache_for_store(batch_idx)
                        if prompt_cache is not None:
                            self._apc_manager.store_exact_cache(
                                meta["full_input_ids"],
                                prompt_cache,
                                extra_hash=meta.get("extra_hash", 0),
                                harvest_provenance=provenance,
                            )
                        self._apc_manager.release(meta.get("apc_blocks", []))
                    else:
                        _apc.commit_prefix_blocks(
                            self._apc_manager,
                            self.prompt_cache,
                            meta["full_input_ids"],
                            batch_idx=batch_idx,
                            extra_hash=meta.get("extra_hash", 0),
                            skip_first_n_tokens=meta.get("prefix_len", 0),
                            blocks_in_use=meta.get("apc_blocks", []),
                            harvest_provenance=provenance,
                        )
            except Exception as e:
                logger.warning("APC harvest failed during batched prefill: %s", e)
                # Best effort — release any acquired prefix blocks.
                for meta in self._apc_meta:
                    if meta is not None:
                        self._apc_manager.release(meta.get("apc_blocks", []))

        self.uids = []
        self.prompt_cache = []
        self._token_context = []
        self.logits_processors = []
        self._apc_meta = []
        self._captured_last_logits = []
        self._last_real_column = None
        return gen_batch

    @property
    def total_prompt_tokens(self):
        return self._total_prompt_tokens

    @staticmethod
    def _capture_rope_deltas(language_model, B: int):
        if not hasattr(language_model, "_rope_deltas"):
            return None
        rope_deltas = language_model._rope_deltas
        if rope_deltas is None:
            return mx.zeros((B, 1), dtype=mx.int32)
        return PromptProcessingBatch._normalize_rope_deltas(rope_deltas, B)

    @staticmethod
    def _capture_rope_deltas_from_prompt_kwargs(
        prompt_kwargs: dict, language_model, B: int
    ):
        rope_deltas = (prompt_kwargs or {}).get("rope_deltas")
        if isinstance(rope_deltas, mx.array):
            return PromptProcessingBatch._normalize_rope_deltas(rope_deltas, B)
        return PromptProcessingBatch._capture_rope_deltas(language_model, B)

    @staticmethod
    def _normalize_rope_deltas(rope_deltas: mx.array, B: int):
        if rope_deltas.ndim == 0:
            rope_deltas = rope_deltas.reshape(1, 1)
        elif rope_deltas.ndim == 1:
            rope_deltas = rope_deltas[:, None]
        # Falcon OCR emits a singleton meant to broadcast across rows.
        if rope_deltas.shape[0] == 1 and B > 1:
            rope_deltas = mx.broadcast_to(rope_deltas, (B, 1))
        if rope_deltas.shape[0] != B:
            if rope_deltas.shape[0] > B:
                rope_deltas = rope_deltas[:B]
            else:
                pad = B - rope_deltas.shape[0]
                rope_deltas = mx.concatenate(
                    [
                        rope_deltas,
                        mx.broadcast_to(rope_deltas[-1:], (pad, rope_deltas.shape[1])),
                    ],
                    axis=0,
                )
        return rope_deltas


def _vault_disk_deepen(vault, ids_list, hit, hit_tier, tiers, *,
                       require_harvest_width_1=False, min_depth=0):
    """Promote a deeper cold entry off disk, then let the RAM tier serve it.

    Module-level on purpose.  ``_vault_pick_for`` is called unbound against
    duck-typed generators (``test_session_restore._Gen`` is "only what
    _vault_pick_for touches"), so reaching for a new attribute on ``self`` would
    make every such caller fail on a feature that is off by default.

    The disk tier does NOT participate in the longest-strict-prefix contest in
    the caller.  It restores into the RAM vault through the ordinary ``insert``
    (ordinary byte accounting, ordinary eviction) and the ordinary lookup then
    serves the result, so every downstream path -- the trim, the suffix
    prefill, the identity refusals -- is byte-for-byte what it was.

    The cost of that two-step is that the FIRST request needing an entry pays
    the read: ~0.55-0.7 s for a 131k rung on the internal NVMe
    (sweep11/P2_VERDICT.md: 6.6-6.7 GB/s at >= 4 MiB reads) against the ~444 s
    of prefill it replaces.  Later requests pay nothing extra.
    """
    disk = getattr(vault, "disk", None)
    if disk is None:
        return hit, hit_tier
    have = max(int(hit.prefix_len) if hit is not None else 0, min_depth)
    for tier in tiers:
        try:
            policy = ({"require_harvest_width_1": True}
                      if require_harvest_width_1 else {})
            cand = disk.restore_into_vault(
                vault, list(ids_list), tier, min_depth=have, **policy)
        except Exception:  # noqa: BLE001 - a disk fault costs a cold prefill, no more
            continue
        if cand is None or (
            require_harvest_width_1 and not _harvest_prov.is_b1_eligible(
                getattr(cand, "harvest_provenance", None))
        ):
            continue
        if have < int(cand.prefix_len) < len(ids_list):
            hit, hit_tier = cand, tier
            have = int(cand.prefix_len)
    return hit, hit_tier


class BatchGenerator:
    """
    Continuous batching with separate prompt processing and generation phases.

    next() returns (prompt_responses, generation_responses) where:
    - prompt_responses contains completed prompt-batch timing stats
    - generation_responses is a list of GenerationBatch.Response objects
    """

    def __init__(
        self,
        model,
        processor,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stop_tokens: Optional[set] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        completion_batch_size: int = DEFAULT_COMPLETION_BATCH_SIZE,
        prefill_batch_size: int = DEFAULT_PREFILL_BATCH_SIZE,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
        existing_left_padding: Optional[List[int]] = None,
        prompt_cache=None,
        kv_bits=None,
        kv_key_bits=None,
        kv_value_bits=None,
        kv_key_scheme=None,
        kv_value_scheme=None,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
        quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
        compute_logprobs: bool = True,
        top_logprobs_k: int = 0,
        logits_processors: Optional[
            List[Callable[[mx.array, mx.array], mx.array]]
        ] = None,
        stream=None,
        apc_manager: Optional["_apc.APCManager"] = None,
        vault: Optional[Any] = None,
        draft_model: Optional[nn.Module] = None,
        draft_kind: Optional[str] = None,
        draft_block_size: Optional[int] = None,
        greedy_sampling: bool = False,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.processor = processor
        self.kv_bits = kv_bits
        self.kv_key_bits = kv_key_bits
        self.kv_value_bits = kv_value_bits
        self.kv_key_scheme = kv_key_scheme
        self.kv_value_scheme = kv_value_scheme
        self.kv_group_size = kv_group_size
        self.kv_quant_scheme = kv_quant_scheme
        self.quantized_kv_start = quantized_kv_start
        self.compute_logprobs = compute_logprobs
        self.top_logprobs_k = top_logprobs_k
        self.logits_processors = logits_processors or []
        self.draft_model = draft_model
        self.draft_kind = draft_kind
        self.draft_block_size = draft_block_size
        self.greedy_sampling = greedy_sampling or sampler is None
        if self.draft_model is not None:
            compute_logprobs = False
            top_logprobs_k = 0
            self.compute_logprobs = False
            self.top_logprobs_k = 0
        # APC mode detection: plain KV models use block APC;
        # mixed/custom cache models use exact prompt-cache snapshots.
        self.apc_mode = None
        if apc_manager is not None:
            self.apc_mode = _apc.model_apc_mode(model)
            if self.apc_mode is None:
                apc_manager = None
        self.apc_manager = apc_manager
        # Warm Context Vault, supplied by the caller (the server builds it from
        # the toggle; nothing here reads an env var).  The vault needs the same
        # whole-cache snapshot contract APC's "exact" mode needs -- this stack's
        # ArraysCache KDA state is CHECKPOINT-only -- so a model that cannot
        # offer it does not get a vault rather than getting a broken one.
        self.vault = vault
        # uid -> prompt ids + everything emitted for it so far.  Only populated
        # while the session-capture flag is on; empty dict otherwise, so the
        # feature costs one attribute when off.
        self._session_tokens: Dict[Any, List[int]] = {}
        if self.vault is not None:
            if self.apc_mode is None:
                self.apc_mode = _apc.model_apc_mode(model)
            if self.apc_mode != "exact":
                logger.info(
                    "vault: model apc_mode=%s is not 'exact'; vault disabled for "
                    "this generator", self.apc_mode
                )
                self.vault = None
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = completion_batch_size

        self._stream = stream or generation_stream

        self.tokenizer.stopping_criteria.add_eos_token_ids(stop_tokens)

        self._generation_batch = GenerationBatch.empty(
            self.model,
            self.sampler,
            self.tokenizer.stopping_criteria,
            compute_logprobs=self.compute_logprobs,
            top_logprobs_k=self.top_logprobs_k,
            greedy_sampling=self.greedy_sampling,
        )
        self._existing_left_padding = existing_left_padding
        self._prompt_batch: Optional[PromptProcessingBatch] = None
        self._unprocessed_sequences = []
        # Lazily filled by ``_supports_right_padded_prefill``; a prototype
        # ``make_cache()`` is cheap but this is on the admission path.
        self._right_pad_capability: Optional[bool] = None

        self._prompt_tokens_counter = 0
        self._prompt_time_counter = 0
        self._gen_tokens_counter = 0
        self._steps_counter = 0
        self._cache_eval_interval = _get_batch_cache_eval_interval()

        self._wire_stack = contextlib.ExitStack()
        self._wire_stack.enter_context(wired_limit(model, [self._stream]))

    # ---------------- APC integration helpers ----------------
    # Keys that are APC-only metadata; stripped from ``prompt_kwargs`` before
    # the merged kwargs are passed to the language model forward.
    _APC_PRIVATE_KEYS = APC_PRIVATE_PROMPT_KEYS

    def _apc_extra_hash(self, prompt_kwargs: dict) -> int:
        """Salt for the APC hash chain."""
        if self.apc_manager is None:
            return 0
        if prompt_kwargs is None:
            prompt_kwargs = {}
        precomputed = prompt_kwargs.get("_apc_semantic_hash")
        if precomputed is not None:
            return int(precomputed)
        img = prompt_kwargs.get("_apc_image_hash")
        if img is None:
            pixel_values = prompt_kwargs.get("pixel_values")
            img = _apc.hash_image_payload(pixel_values=pixel_values, image_ref=None)
        tenant = prompt_kwargs.get("_apc_tenant")
        return _apc.semantic_extra_hash(
            tenant=tenant,
            image_hash=img,
            media={
                "audio": prompt_kwargs.get("input_features"),
                "video": prompt_kwargs.get("pixel_values_videos"),
                "embeddings": prompt_kwargs.get("inputs_embeds"),
                "masks": prompt_kwargs.get("attention_mask"),
            },
            model=getattr(self, "model", None),
            processor=getattr(self, "processor", None),
        )

    def _apc_media_token_ids(self) -> set[int]:
        config = getattr(self.model, "config", None)
        if config is None:
            return set()
        return _apc.multimodal_token_ids_from_config(config)

    def _apc_safe_prefix_lookup_min(self, ids_list: List[int]) -> int:
        safe_min = _apc.media_safe_prefix_min(ids_list, self._apc_media_token_ids())
        return max(0, safe_min - 1)

    def _apc_suffix_is_text_only(self, ids_list: List[int], prefix_len: int) -> bool:
        return _apc.prefix_leaves_text_only_suffix(
            ids_list,
            prefix_len,
            self._apc_media_token_ids(),
        )

    def _apc_prefix_has_media_tokens(
        self, ids_list: List[int], prefix_len: int
    ) -> bool:
        return _apc.prefix_contains_media_tokens(
            ids_list,
            prefix_len,
            self._apc_media_token_ids(),
        )

    def _apc_exact_checkpoint_len(self, ids_list: List[int]) -> int:
        if self.apc_manager is None or getattr(self, "apc_mode", "block") != "exact":
            return 0
        return _apc.adjust_prefix_to_text_suffix_boundary(
            ids_list,
            len(ids_list) - self.apc_manager.exact_cache_guard_tokens,
            self._apc_media_token_ids(),
            max_prefix_tokens=len(ids_list) - 1,
        )

    def _apc_pick_for(self, sequence, serve_batch_width: int = 1) -> Optional[dict]:
        """Look up a warm prefix for ``sequence`` -- APC first, then the vault.

        Returns a plan dict with matched blocks + suffix metadata when there is
        a usable hit, else None.  The vault only ever *deepens* the answer: see
        ``_vault_pick_for``.

        ``serve_batch_width`` is the width of the window being admitted, which
        is what this generator can know BEFORE the picks decide the batch.  It
        is read only by the ``MLX_VLM_APC_SERVE_B1_FROM_B1_ONLY`` policy, and
        only when it is 1.  Note the one-sided approximation this makes and the
        direction it errs in: ``_apply_right_pad_policy`` can narrow a 2-row
        window to 1 row AFTER the lookups have run, so a request that ends up
        served alone may have been looked up as width 2 and may therefore have
        accepted a wider entry.  The reverse -- a width-1 window growing -- cannot
        happen.  So the knob is a guarantee about what a SOLO ADMISSION accepts,
        not about every prefill that happens to end up with one row in it.  The
        default of a keyword makes an unbound duck-typed caller
        (``test_session_restore._Gen``) keep working.
        """
        vault = getattr(self, "vault", None)
        if self.apc_manager is None and vault is None:
            return None
        uid, ids_list, max_toks, prompt_kwargs, lps, criteria = sequence
        if not ids_list or len(ids_list) < 2:
            return None
        pick = None
        if self.apc_manager is not None:
            pick = _apc.apc_lookup_plan(
                self.apc_manager,
                ids_list,
                extra_hash=self._apc_extra_hash(prompt_kwargs or {}),
                apc_mode=getattr(self, "apc_mode", "block"),
                safe_lookup_min=self._apc_safe_prefix_lookup_min(ids_list),
                serve_batch_width=int(serve_batch_width),
                suffix_is_text_only=lambda pl: self._apc_suffix_is_text_only(
                    ids_list, pl
                ),
                prefix_has_media=lambda pl: self._apc_prefix_has_media_tokens(
                    ids_list, pl
                ),
            )
        if vault is None:
            return pick
        return self._vault_pick_for(
            ids_list, prompt_kwargs, pick, serve_batch_width=serve_batch_width)

    def _vault_prefix_trim_is_safe(self) -> bool:
        """Refuse a vault warm start where trimming the prompt breaks RoPE.

        ``generate/dispatch.py`` primes ``_rope_deltas`` from the FULL prompt
        before it trims (``_prime_cached_prefix_rope_state``), because a
        Qwen-style mRoPE model cannot recompute the original delta from the
        suffix alone.  This path has no equivalent hook, and getting it wrong is
        silent rather than loud, so decline instead.  GLM-5-Next is NoPE and
        exposes no ``get_rope_index``, so this is True there.
        """
        return not callable(getattr(self.model, "get_rope_index", None))

    def _vault_pick_for(
        self, ids_list, prompt_kwargs, pick: Optional[dict], serve_batch_width: int = 1
    ) -> Optional[dict]:
        """Deepen (or supply) the warm start from the context vault.

        The vault competes only when it strictly beats what APC found: a
        shallower rung is worse than the hit already in hand, and taking it
        would also drop APC's block references. On a win the APC blocks are
        released, because the plan that replaces them will never reference them.
        """
        vault = getattr(self, "vault", None)
        have = int(pick.get("prefix_len", 0)) if pick else 0
        # Tiers compete under one rule: strictly deeper than what we already
        # have, and a strict prefix of the new prompt. That right-hand term is
        # the returning-turn condition -- the stored rung covers turn N, the new
        # prompt is that plus turn N+1's user message -- so the session tier
        # needs no separate guard. The SESSION tier is consulted only when the
        # flag is on, so with it off this reduces term for term to the prefill
        # lookup it replaces.
        _PREFILL_TIER = _context_vault.VaultTier.PREFILL
        tiers = [_PREFILL_TIER]
        if _context_vault.session_capture_enabled():
            tiers.append(_context_vault.VaultTier.SESSION)
        require_b1 = (serve_batch_width == 1 and _harvest_prov.serve_b1_from_b1_only())
        policy = {"require_harvest_width_1": True} if require_b1 else {}
        hit = None
        hit_tier = _PREFILL_TIER
        for tier in tiers:
            try:
                # PREFILL uses the two-argument call the vault has always had,
                # so a duck-typed vault (the server tests use one) keeps working
                # and this path stays byte-for-byte what it replaced. The tier
                # kwarg is only used where it is load-bearing.
                cand = (vault.lookup(list(ids_list), **policy) if tier is _PREFILL_TIER
                        else vault.lookup(list(ids_list), tier=tier, **policy))
            except Exception:  # noqa: BLE001 - a vault fault must never fail a request
                return pick
            if cand is None or (
                require_b1 and not _harvest_prov.is_b1_eligible(
                    getattr(cand, "harvest_provenance", None))
            ):
                continue
            if not (have < int(cand.prefix_len) < len(ids_list)):
                continue
            if hit is None or int(cand.prefix_len) > int(hit.prefix_len):
                hit, hit_tier = cand, tier
        hit, hit_tier = _vault_disk_deepen(
            vault, ids_list, hit, hit_tier, tiers,
            require_harvest_width_1=require_b1, min_depth=have)
        if hit is None:
            return pick
        if not self._vault_prefix_trim_is_safe():
            return pick
        fresh = cache.make_prompt_cache(self.model)
        try:
            # The tier is mandatory: restore_into refuses a mismatch by design,
            # so a session rung restored as PREFILL returns False and falls
            # through to a cold prefill rather than serving a wrong guarantee.
            restored = bool(
                vault.restore_into(fresh, hit) if hit_tier is _PREFILL_TIER
                else vault.restore_into(fresh, hit, tier=hit_tier))
        except Exception:  # noqa: BLE001
            restored = False
        if not restored:
            return pick
        if pick is not None and self.apc_manager is not None:
            self.apc_manager.release(pick.get("matched_blocks", []))
        return {
            "matched_blocks": [],
            "warm_cache": fresh,
            "prefix_len": int(hit.prefix_len),
            "extra_hash": self._apc_extra_hash(prompt_kwargs or {}),
            "full_input_ids": list(ids_list),
            "source": ("vault-session"
                       if hit_tier is _context_vault.VaultTier.SESSION else "vault"),
            # A vault rung carries its own harvest provenance (it travels ON the
            # checkpoint), so a vault-served warm start reports the width it was
            # captured at exactly as an APC-served one does.
            "harvest_provenance": _harvest_prov.normalise(
                getattr(hit, "harvest_provenance", None)
            ),
        }

    def _vault_rungs_for(self, ids_list, prefix_len: int) -> List[int]:
        """Absolute boundaries this row should checkpoint on the way past.

        Same geometric ladder ``dispatch.py`` uses, shifted past whatever prefix
        the row starts warm from -- a rung at or below the warm prefix is
        already stored by construction.
        """
        if getattr(self, "vault", None) is None:
            return []
        return [
            b
            for b in _context_vault.boundary_ladder(
                len(ids_list), step=self.prefill_step_size
            )
            if b > int(prefix_len)
        ]

    def _pending_after_admission(
        self, window: List[tuple], n: int, batch: "PromptProcessingBatch"
    ) -> List[tuple]:
        """The pending list after ``batch`` took some rows out of ``window``.

        The mixed builder may admit only SOME of the window: a model that cannot
        take right-padded prefill (see ``_apply_right_pad_policy``) batches only
        the rows whose suffix lengths are equal.  The rows it did not take go
        back at the FRONT of the pending list, keeping their queue position --
        they are older than everything behind them and the next pass must see
        them first, which is also what makes the head-anchored split
        starvation-free.
        """
        admitted = set(batch.uids)
        return [s for s in window if s[0] not in admitted] + (
            self._unprocessed_sequences[n:]
        )

    def _supports_right_padded_prefill(self) -> bool:
        """Memoised ``model_supports_right_padded_prefill`` for this generator.

        Memoised on the instance rather than on the model: an ``nn.Module`` here
        is a ``dict`` subclass, so it is neither hashable (no
        ``WeakKeyDictionary``) nor safe to hang a new attribute off (it would
        land in the module's own dict and be walked by ``parameters()``).
        """
        cached = getattr(self, "_right_pad_capability", None)
        if cached is None:
            cached = bool(model_supports_right_padded_prefill(self.model))
            self._right_pad_capability = cached
        return cached

    def _apply_right_pad_policy(
        self, sequences: List[tuple], picks: List[Optional[dict]]
    ) -> Tuple[Optional[List[tuple]], Optional[List[Optional[dict]]]]:
        """Drop rows that would force RIGHT padding on a model that refuses it.

        Returns the (possibly shortened) ``(sequences, picks)`` to admit, or
        ``(None, None)`` to tell the caller to decline the mixed batch entirely.

        The policy, when ``supports_right_padded_prefill`` is False (see
        ``model_supports_right_padded_prefill`` for why the capability exists):

          * rows whose suffix lengths are EQUAL need no right padding at all, so
            they batch together exactly as before -- the fast path is kept, not
            removed;
          * otherwise only the group of rows sharing the suffix length of the
            row at the HEAD of the queue is admitted, and the rest go back to
            the pending list.  Anchoring on the head, not on the largest group
            or on the first warm row, is what makes this starvation-free: the
            oldest pending row is admitted by SOME branch on every pass;
          * if that head group happens to contain no warm row, there is nothing
            for a warm batch to be built out of, so the mixed path declines and
            the caller's cold-only path admits the whole window LEFT-padded --
            correct, and the one case where a warm row loses its prefix reuse
            for this round.

        A batch of one warm row is fine and is the common shape after a split.
        Left-padded cold-only batching is untouched: the defect is right padding.
        """
        if self._supports_right_padded_prefill():
            return sequences, picks
        prefix_lens = [p["prefix_len"] if p else 0 for p in picks]
        suffix_lens = [
            len(s[1]) - prefix_lens[i] for i, s in enumerate(sequences)
        ]
        if len(set(suffix_lens)) <= 1:
            return sequences, picks  # no right padding would be built anyway

        head_len = suffix_lens[0]
        keep = [i for i, length in enumerate(suffix_lens) if length == head_len]
        kept = set(keep)
        deferred = [i for i in range(len(sequences)) if i not in kept]
        kept_picks = [picks[i] for i in keep]
        cold_fallback = not any(p is not None for p in kept_picks)
        _note_prefill_batch_refusal("right_pad_kda", len(deferred))
        logger.info(
            "prefill batch refusal right_pad_kda: %s declines right-padded "
            "prefill (recurrent/linear-attention state cannot be rolled); "
            "suffix lens %s -> %s %d row(s) at suffix_len=%d, deferring %d row(s) "
            "%s",
            type(self.model).__name__,
            suffix_lens,
            "cold-only fallback for" if cold_fallback else "admitting",
            len(sequences) if cold_fallback else len(keep),
            head_len,
            0 if cold_fallback else len(deferred),
            [] if cold_fallback else [suffix_lens[i] for i in deferred],
        )
        if cold_fallback:
            return None, None
        return [sequences[i] for i in keep], kept_picks

    def _build_mixed_prompt_batch(
        self, sequences: List[tuple]
    ) -> Optional["PromptProcessingBatch"]:
        """Build a multi-row PromptProcessingBatch admitting ``sequences``.

        Each row is independently looked up in APC. Warm rows have their
        suffixes prefilled against pre-populated K/V; cold rows prefill from
        scratch in the same batch. Right-padding aligns RoPE positions
        across rows with different prefix/suffix lengths.

        On a model that cannot take right-padded prefill -- one with recurrent
        (linear-attention) layers, see ``model_supports_right_padded_prefill``
        -- ``_apply_right_pad_policy`` first drops the rows that would force
        padding, so the batch this returns may cover only part of
        ``sequences``.  The caller consumes ``batch.uids``, not ``sequences``.

        Returns ``None`` if neither APC nor the vault can offer a warm row, or
        if the policy left no warm row in the admitted group (in which case the
        caller should use the cold-only, LEFT-padded path).
        """
        if self.apc_manager is None and getattr(self, "vault", None) is None:
            return None

        picks: List[Optional[dict]] = [
            self._apc_pick_for(s, serve_batch_width=len(sequences))
            for s in sequences
        ]
        any_warm = any(p is not None for p in picks)
        if not any_warm:
            return None  # caller falls back to cold-only path

        sequences, picks = self._apply_right_pad_policy(sequences, picks)
        if sequences is None:
            return None  # caller falls back to cold-only (LEFT-padded) path

        uids = [s[0] for s in sequences]
        full_ids = [list(s[1]) for s in sequences]
        max_tokens_list = [s[2] for s in sequences]
        prompt_kwargs_list = [s[3] for s in sequences]
        logits_processors = [s[4] for s in sequences]
        thinking_budget_criteria = [s[5] for s in sequences]

        # Per-row prefix length and suffix tokens
        prefix_lens = [p["prefix_len"] if p else 0 for p in picks]
        suffix_ids_list = [full_ids[i][prefix_lens[i] :] for i in range(len(sequences))]
        suffix_lens = [len(s) for s in suffix_ids_list]

        max_suffix_len = max(suffix_lens)
        right_pad_per_row = [max_suffix_len - s for s in suffix_lens]

        # Source inputs_embeds: every row's prompt_kwargs holds the full-prompt
        # embeddings. Slice to suffix per-row, right-pad to max_suffix_len, stack.
        suffix_embeds_per_row: List[mx.array] = []
        for i, kw in enumerate(prompt_kwargs_list):
            if kw is None or kw.get("inputs_embeds") is None:
                raise ValueError("APC mixed prefill requires precomputed inputs_embeds")
            full = kw["inputs_embeds"]  # [1, full_len, D]
            suff = full[:, prefix_lens[i] :, :]
            pad = right_pad_per_row[i]
            if pad > 0:
                pad_emb = mx.zeros(
                    (suff.shape[0], pad, suff.shape[-1]), dtype=suff.dtype
                )
                suff = mx.concatenate([suff, pad_emb], axis=1)
            suffix_embeds_per_row.append(suff)
        inputs_embeds = mx.concatenate(suffix_embeds_per_row, axis=0)

        # Merge prompt-side kwargs (excluding inputs_embeds, which we've just
        # rebuilt). Per-batch tensors get concatenated across rows; scalars
        # take the first row's value (matches the existing cold-only path).
        # APC-private keys (e.g. tenant salt) are dropped — they're consumed
        # in _apc_extra_hash, never forwarded to the model.
        merged_kwargs: dict = {}
        per_row_keys: dict = {}
        batch_size = len(prompt_kwargs_list)
        for i, kw in enumerate(prompt_kwargs_list):
            if not kw:
                continue
            full_len = len(full_ids[i])
            prefix_len = prefix_lens[i]
            right_pad = right_pad_per_row[i]
            for k, v in kw.items():
                if k == "inputs_embeds" or k in self._APC_PRIVATE_KEYS:
                    continue
                if isinstance(v, mx.array) and _prompt_kwarg_batch_size(k, v) >= 1:
                    row_v = _prompt_kwarg_row(k, v, i, batch_size)
                    if _is_sequence_aligned_prompt_kwarg(k, row_v, full_len):
                        row_v = _slice_sequence_aligned_prompt_kwarg(
                            k, row_v, start=prefix_len
                        )
                        row_v = _pad_sequence_aligned_prompt_kwarg(
                            k,
                            row_v,
                            max_suffix_len,
                            left=False,
                        )
                    per_row_keys.setdefault(k, []).append(row_v)
                elif k not in merged_kwargs:
                    merged_kwargs[k] = v
        for k, vs in per_row_keys.items():
            merged_kwargs[k] = _concat_prompt_kwarg_rows(k, vs)

        apc_mode = getattr(self, "apc_mode", "block")
        # bits + group_size + scheme so warm restore matches live _make_cache
        # backend (uniform BatchQuantized vs BatchTurboQuant).
        _quant_policy = kv_quant_from_legacy(
            self.kv_bits,
            self.kv_quant_scheme,
            self.kv_group_size,
            getattr(self, "kv_key_bits", None),
            getattr(self, "kv_value_bits", None),
            getattr(self, "kv_key_scheme", None),
            getattr(self, "kv_value_scheme", None),
        )
        _quant_cfg = _quant_policy.to_config() if _quant_policy is not None else None
        if apc_mode == "exact":
            row_caches = [
                p["warm_cache"] if p is not None else self.model.make_cache()
                for p in picks
            ]
            # Pass kv_quant_config so exact multi warm matches live _make_cache
            # layer types under --kv-bits (cold quant row + exact float join).
            warm_cache, _ = _apc.make_warm_batch_exact_cache_multi(
                row_caches,
                prefix_lens,
                kv_quant_config=_quant_cfg,
            )
            if warm_cache is None:
                return None
        else:
            # Build the multi-row warm cache (zeros for cold rows, K/V for warm).
            num_layers = (
                len(self.model.make_cache())
                if hasattr(self.model, "make_cache")
                else len(self.model.layers)
            )
            warm_cache, _ = _apc.make_warm_batch_kv_cache_multi(
                picks, num_layers=num_layers, kv_quant_config=_quant_cfg
            )

        apc_meta = [
            {
                "full_input_ids": full_ids[i],
                "prefix_len": prefix_lens[i],
                "extra_hash": (
                    picks[i]["extra_hash"]
                    if picks[i]
                    else self._apc_extra_hash(prompt_kwargs_list[i] or {})
                ),
                "apc_blocks": picks[i].get("matched_blocks", []) if picks[i] else [],
                "checkpoint_len": self._apc_exact_checkpoint_len(full_ids[i]),
                "vault_rungs": self._vault_rungs_for(full_ids[i], prefix_lens[i]),
                # Where the entry SERVING this row was harvested (not where this
                # row's own snapshot will be taken -- that is computed at store
                # time from the batch actually built).  Reported by
                # ``prompt_progress`` and thence by the server's prefill line.
                "harvest_provenance": (
                    picks[i].get("harvest_provenance") if picks[i] else None
                ),
            }
            for i in range(len(sequences))
        ]

        prompt_batch_cls = _generate_module_override(
            "PromptProcessingBatch", PromptProcessingBatch
        )
        return prompt_batch_cls(
            model=self.model,
            uids=uids,
            input_ids=suffix_ids_list,
            max_tokens=max_tokens_list,
            inputs_embeds=inputs_embeds,
            prompt_kwargs=merged_kwargs,
            logits_processors=logits_processors,
            thinking_budget_criteria=thinking_budget_criteria,
            prefill_step_size=self.prefill_step_size,
            kv_bits=self.kv_bits,
            kv_key_bits=getattr(self, "kv_key_bits", None),
            kv_value_bits=getattr(self, "kv_value_bits", None),
            kv_key_scheme=getattr(self, "kv_key_scheme", None),
            kv_value_scheme=getattr(self, "kv_value_scheme", None),
            kv_group_size=self.kv_group_size,
            kv_quant_scheme=self.kv_quant_scheme,
            quantized_kv_start=getattr(
                self, "quantized_kv_start", DEFAULT_QUANTIZED_KV_START
            ),
            warm_cache=warm_cache,
            apc_meta=apc_meta,
            apc_manager=self.apc_manager,
            vault=getattr(self, "vault", None),
            right_pad_per_row=right_pad_per_row,
            suffix_lens=suffix_lens,
            apc_mode=apc_mode,
            draft_model=getattr(self, "draft_model", None),
            draft_kind=getattr(self, "draft_kind", None),
            draft_block_size=getattr(self, "draft_block_size", None),
            greedy_sampling=getattr(self, "greedy_sampling", False),
        )

    def _build_apc_meta_for_cold(
        self,
        input_ids_list: List[List[int]],
        prompt_kwargs_list: List[Optional[dict]],
    ) -> Optional[List[Optional[dict]]]:
        """Build per-row harvest metadata for a cold-prefill batch so the
        produced K/V are added to APC after prefill, and so the vault stores its
        boundary ladder on the way past.

        With APC off, ``_apc_extra_hash`` returns 0 and
        ``_apc_exact_checkpoint_len`` returns 0, so the rows carry vault rungs
        and nothing else -- no APC work is scheduled by a vault-only run.
        """
        if self.apc_manager is None and getattr(self, "vault", None) is None:
            return None
        meta: List[Optional[dict]] = []
        for ids_list, kw in zip(input_ids_list, prompt_kwargs_list):
            extra_hash = self._apc_extra_hash(kw or {})
            meta.append(
                {
                    "full_input_ids": list(ids_list),
                    "prefix_len": 0,
                    "extra_hash": extra_hash,
                    "apc_blocks": [],
                    "checkpoint_len": self._apc_exact_checkpoint_len(list(ids_list)),
                    "vault_rungs": self._vault_rungs_for(list(ids_list), 0),
                }
            )
        return meta

    @property
    def stream(self):
        return self._stream

    def close(self):
        if self._wire_stack is not None:
            self._wire_stack.close()
            self._wire_stack = None

    def __del__(self):
        self.close()

    def insert(
        self,
        prompts,
        max_tokens: Union[List[int], int, None] = None,
        prompt_kwargs: Optional[List[dict]] = None,
        logits_processors: Optional[
            List[Optional[List[Callable[[mx.array, mx.array], mx.array]]]]
        ] = None,
        thinking_budget_criteria: Optional[List[Any]] = None,
    ):
        uids = []

        if max_tokens is None or isinstance(max_tokens, int):
            max_tokens = [max_tokens or self.max_tokens] * len(prompts)

        if prompt_kwargs is None:
            prompt_kwargs = [{}] * len(prompts)
        if logits_processors is None:
            logits_processors = [self.logits_processors] * len(prompts)
        elif len(logits_processors) != len(prompts):
            raise ValueError("Insufficient number of logits_processors provided")
        if thinking_budget_criteria is None:
            thinking_budget_criteria = [None] * len(prompts)
        elif len(thinking_budget_criteria) != len(prompts):
            raise ValueError("Insufficient number of thinking_budget_criteria provided")

        for p, m, kw, lp, tc in zip(
            prompts,
            max_tokens,
            prompt_kwargs,
            logits_processors,
            thinking_budget_criteria,
        ):
            self._unprocessed_sequences.append((self.uid_count, p, m, kw, lp, tc))
            if _context_vault.session_capture_enabled():
                # Seed the session key with the EXACT prompt ids the model will
                # see. Re-deriving them at completion from the request would
                # re-tokenise and could disagree by a token; these are the ones.
                self._session_tokens[self.uid_count] = list(p)
            uids.append(self.uid_count)
            self.uid_count += 1
        # Sort in ascending order of length
        self._unprocessed_sequences = sorted(
            self._unprocessed_sequences, key=lambda x: len(x[1])
        )
        return uids

    def note_generated(self, uid, tokens: Sequence[int]) -> None:
        """Append emitted tokens to ``uid``'s session key.  Never raises.

        Must be called EXACTLY once per emitted token, in order.  The session
        rung is keyed by the full token sequence, so a dropped or duplicated
        token does not make the rung wrong -- it makes it unreachable, which is
        a silent loss of the feature rather than a wrong answer.  Still: the
        caller owns that contract, and under speculative decoding a single
        emitted chunk can cover several tokens while naming only the last one.
        See ``docs/vault_session_restore.md``.
        """
        if not _context_vault.session_capture_enabled():
            return
        acc = self._session_tokens.get(uid)
        if acc is None:
            return
        try:
            acc.extend(int(t) for t in tokens)
        except Exception:  # noqa: BLE001 - bookkeeping must never fail a response
            self._session_tokens.pop(uid, None)

    def forget_session(self, uid) -> None:
        """Drop ``uid``'s accumulator.  Called on removal so a cancelled or
        completed request cannot leak its token list for the process lifetime."""
        self._session_tokens.pop(uid, None)

    def capture_session(
        self,
        uid,
        tokens: Optional[Sequence[int]] = None,
        *,
        session_id: str,
        ttl_s: Optional[float] = None,
    ) -> bool:
        """Store an end-of-turn session rung for ``uid``.  Never raises.

        Called BY THE SERVER when a response completes, not from inside the
        decode loop.  Two reasons, and they are the design:

        * the server is the layer that knows what a conversation is.  This
          object knows about rows and caches and has no idea which requests
          belong together, and ``session_id`` must be the server's conversation
          id -- never the token-derived fallback, which in production hashes the
          shared system prompt and collapses every conversation into one
          eviction group (see ``context_vault.derived_session_id_allowed``).

        * the row's cache is only meaningful while the row is still in the
          generation batch, i.e. in the window between ``finish_reason`` being
          emitted and ``remove()``.  Calling it from the server's completion
          handler lands inside that window; calling it later finds no row and
          returns False rather than storing something wrong.

        ``adopt=False`` deliberately: the row cache is extracted from a
        batch-shaped cache the other rows are still decoding against, so the
        buffers are not ours to take.  Adoption is only for a cache nobody will
        touch again.
        """
        # Every early return NAMES ITSELF. Live on ff9a3045 this method returned
        # False five different ways in silence, and the seven checks could only
        # report "nothing happened" -- which is indistinguishable from the
        # feature being switched off.
        if not _context_vault.session_capture_enabled():
            _context_vault.record_session_skip("flag_off")
            return False
        if getattr(self, "vault", None) is None:
            _context_vault.record_session_skip("generator_has_no_vault")
            return False
        if not session_id:
            _context_vault.record_session_skip("no_session_id_at_generator")
            return False
        try:
            gb = self._generation_batch
            if gb is None:
                _context_vault.record_session_skip("no_generation_batch")
                return False
            # ``uids`` is "rows still generating", NOT "rows in this batch":
            # SpeculativeGenerationBatch._refresh_uids rebuilds it from
            # ``_finished``, so a uid leaves it at the very moment finish_reason
            # is set -- which is exactly when we are called. The live diagnostic
            # on 32046983 named this gate (uid_gone_from_batch, twice, both
            # turns) and it is why the feature stored nothing.
            #
            # ``_all_uids`` is the stable list and is what the row indices are
            # aligned to: _append_token_responses attributes tokens via
            # _all_uids[row], and filter() -- the only thing that compacts the
            # prompt cache -- rewrites both together. Plain GenerationBatch has
            # no _all_uids and never prunes on finish, so uids is correct there.
            all_uids = getattr(gb, "_all_uids", None) or gb.uids
            if uid not in all_uids:
                _context_vault.record_session_skip("uid_gone_from_batch")
                return False
            row = all_uids.index(uid)
            row_cache = _apc.snapshot_prompt_cache_row(gb.prompt_cache, row)
            if not row_cache:
                _context_vault.record_session_skip("row_cache_unavailable")
                return False
            key = list(tokens) if tokens is not None else self._session_tokens.get(uid)
            if not key:
                _context_vault.record_session_skip("empty_token_key")
                return False
            return _context_vault.record_session_turn(
                self.vault,
                key,
                row_cache,
                completed=True,
                session_id=session_id,
                ttl_s=ttl_s,
                adopt=False,
            )
        except Exception:  # noqa: BLE001 - never fail a response over a rung
            _context_vault.record_session_skip("exception_in_capture_session")
            logger.warning("vault: session capture failed for uid=%s; the next "
                           "turn falls back to a cold prefill", uid, exc_info=True)
            return False

    def remove(self, uid) -> bool:
        """Remove a sequence from the batch by uid."""
        self.forget_session(uid)
        with mx.stream(self._stream):
            # Waiting in the queue.
            for i, (seq_uid, _, _, _, _, _) in enumerate(self._unprocessed_sequences):
                if seq_uid == uid:
                    self._unprocessed_sequences.pop(i)
                    return True

            # Being prefilled
            if self._prompt_batch is not None and uid in self._prompt_batch.uids:
                if len(self._prompt_batch.uids) == 1:
                    self._prompt_batch.uids = []
                    self._prompt_batch.prompt_cache = []
                    self._prompt_batch = None
                    mx.clear_cache()
                    return True

            # Already decoding.
            if uid in self._generation_batch.uids:
                idx = self._generation_batch.uids.index(uid)
                keep = [i for i in range(len(self._generation_batch.uids)) if i != idx]
                self._generation_batch.filter(keep)
                return True

            return False

    @property
    def unprocessed_prompts(self):
        """Backward-compatible alias for server flush logic."""
        return self._unprocessed_sequences

    @property
    def has_pending_prompts(self):
        """True if there are prompts waiting or being processed."""
        return len(self._unprocessed_sequences) > 0 or self._prompt_batch is not None

    @property
    def has_work(self):
        """True if there is any remaining work."""
        return (
            len(self._generation_batch) > 0
            or self._prompt_batch is not None
            or len(self._unprocessed_sequences) > 0
        )

    def stats(self):
        """Return accumulated batch statistics."""
        stats = BatchStats()
        stats.prompt_tokens = self._prompt_tokens_counter
        stats.prompt_time = self._prompt_time_counter
        stats.prompt_tps = (
            self._prompt_tokens_counter / self._prompt_time_counter
            if self._prompt_time_counter > 0
            else 0
        )
        stats.generation_tokens = self._gen_tokens_counter
        stats.peak_memory = mx.get_peak_memory() / 1e9
        return stats

    @staticmethod
    def _record_prompt_batch_time(prompt_batch, elapsed_s: float) -> None:
        recorder = getattr(prompt_batch, "record_prompt_time", None)
        if callable(recorder):
            recorder(elapsed_s)

    @staticmethod
    def _prompt_batch_progress(prompt_batch) -> List[PromptProgress]:
        progress = getattr(prompt_batch, "prompt_progress", None)
        if callable(progress):
            return progress()
        return []

    def _extend_generation_batch(self, gen_batch) -> None:
        if len(self._generation_batch) == 0:
            self._generation_batch = gen_batch
        else:
            self._generation_batch.extend(gen_batch)

    def _next(self, **kwargs):
        generation_responses = []
        prompt_responses = []

        # Decode-first: always emit a generation step before touching prefill.
        yield_after_decode = any(
            getattr(processor, "requires_immediate_decode_yield", False)
            for processors in getattr(self._generation_batch, "logits_processors", [])
            for processor in processors or []
        )
        if len(self._generation_batch) > 0:
            generation_responses = self._generation_batch.next()
            self._gen_tokens_counter += len(generation_responses)
            self._steps_counter += 1
            if (
                self._cache_eval_interval > 0
                and self._steps_counter % self._cache_eval_interval == 0
            ):
                cache_states = getattr(self._generation_batch, "cache_states", None)
                if callable(cache_states):
                    mx.eval(cache_states())
                else:
                    mx.eval([c.state for c in self._generation_batch.prompt_cache])
                mx.clear_cache()
            if yield_after_decode:
                return prompt_responses, generation_responses

        if (
            getattr(self._generation_batch, "is_speculative", False)
            and len(self._generation_batch) > 0
        ):
            return prompt_responses, generation_responses

        if len(self._generation_batch) >= self.completion_batch_size:
            return prompt_responses, generation_responses

        if self._prompt_batch is not None:
            if self._prompt_batch.needs_processing():
                tic = time.perf_counter()
                self._prompt_batch.prompt_step()
                elapsed = time.perf_counter() - tic
                self._prompt_time_counter += elapsed
                self._record_prompt_batch_time(self._prompt_batch, elapsed)
                return prompt_responses, generation_responses

            tic = time.perf_counter()
            gen_batch = self._prompt_batch.generate(
                self.sampler,
                self.tokenizer.stopping_criteria,
                compute_logprobs=self.compute_logprobs,
                top_logprobs_k=self.top_logprobs_k,
            )
            elapsed = time.perf_counter() - tic
            self._prompt_time_counter += elapsed
            self._record_prompt_batch_time(self._prompt_batch, elapsed)
            prompt_responses = self._prompt_batch_progress(self._prompt_batch)
            self._extend_generation_batch(gen_batch)
            self._prompt_batch = None
            mx.clear_cache()
            return prompt_responses, generation_responses

        num_active = len(self._generation_batch)
        num_to_add = self.completion_batch_size - num_active
        if self._unprocessed_sequences and num_to_add >= self.prefill_batch_size:
            # Take up to prefill_batch_size pending sequences. If APC is on
            # and at least one of them has a prefix hit, build a mixed
            # warm/cold PromptProcessingBatch with right-padded suffixes so
            # warm and cold rows prefill in a single forward pass.
            n = min(self.prefill_batch_size, len(self._unprocessed_sequences))
            sequences = self._unprocessed_sequences[:n]
            if logger.isEnabledFor(logging.DEBUG) and os.environ.get("APC_DEBUG"):
                logger.warning(
                    "APC admit n=%d (pending=%d)",
                    n,
                    len(self._unprocessed_sequences),
                )
            mixed = self._build_mixed_prompt_batch(sequences)
            if mixed is not None:
                self._unprocessed_sequences = self._pending_after_admission(
                    sequences, n, mixed
                )
                self._prompt_batch = mixed
                self._prompt_tokens_counter += self._prompt_batch.total_prompt_tokens
                if self._prompt_batch.needs_processing():
                    tic = time.perf_counter()
                    nstep = self._prompt_batch.prompt_step()
                    elapsed = time.perf_counter() - tic
                    self._prompt_time_counter += elapsed
                    self._record_prompt_batch_time(self._prompt_batch, elapsed)
                else:
                    tic = time.perf_counter()
                    gen_batch = self._prompt_batch.generate(
                        self.sampler,
                        self.tokenizer.stopping_criteria,
                        compute_logprobs=self.compute_logprobs,
                        top_logprobs_k=self.top_logprobs_k,
                    )
                    elapsed = time.perf_counter() - tic
                    self._prompt_time_counter += elapsed
                    self._record_prompt_batch_time(self._prompt_batch, elapsed)
                    prompt_responses = self._prompt_batch_progress(self._prompt_batch)
                    self._extend_generation_batch(gen_batch)
                    self._prompt_batch = None
                    mx.clear_cache()
                return prompt_responses, generation_responses

            self._unprocessed_sequences = self._unprocessed_sequences[n:]

            uids = [s[0] for s in sequences]
            input_ids = [s[1] for s in sequences]
            max_tokens_list = [s[2] for s in sequences]
            prompt_kwargs_list = [s[3] for s in sequences]
            logits_processors = [s[4] for s in sequences]
            thinking_budget_criteria = [s[5] for s in sequences]

            inputs_embeds, merged_kwargs = _merge_prefill_prompt_kwargs(
                prompt_kwargs_list, input_ids
            )

            # APC: also harvest cold-prefill prefixes so future requests hit.
            apc_meta = self._build_apc_meta_for_cold(input_ids, prompt_kwargs_list)

            prompt_batch_cls = _generate_module_override(
                "PromptProcessingBatch", PromptProcessingBatch
            )
            self._prompt_batch = prompt_batch_cls(
                model=self.model,
                uids=uids,
                existing_left_padding=getattr(self, "_existing_left_padding", None),
                input_ids=input_ids,
                max_tokens=max_tokens_list,
                inputs_embeds=inputs_embeds,
                prompt_kwargs=merged_kwargs,
                logits_processors=logits_processors,
                thinking_budget_criteria=thinking_budget_criteria,
                prefill_step_size=self.prefill_step_size,
                kv_bits=self.kv_bits,
                kv_key_bits=getattr(self, "kv_key_bits", None),
                kv_value_bits=getattr(self, "kv_value_bits", None),
                kv_key_scheme=getattr(self, "kv_key_scheme", None),
                kv_value_scheme=getattr(self, "kv_value_scheme", None),
                kv_group_size=self.kv_group_size,
                kv_quant_scheme=self.kv_quant_scheme,
                quantized_kv_start=getattr(
                    self, "quantized_kv_start", DEFAULT_QUANTIZED_KV_START
                ),
                apc_meta=apc_meta,
                apc_manager=self.apc_manager,
                vault=getattr(self, "vault", None),
                apc_mode=self.apc_mode,
                draft_model=getattr(self, "draft_model", None),
                draft_kind=getattr(self, "draft_kind", None),
                draft_block_size=getattr(self, "draft_block_size", None),
                greedy_sampling=getattr(self, "greedy_sampling", False),
            )
            self._prompt_tokens_counter += self._prompt_batch.total_prompt_tokens

            if self._prompt_batch.needs_processing():
                tic = time.perf_counter()
                n = self._prompt_batch.prompt_step()
                elapsed = time.perf_counter() - tic
                self._prompt_time_counter += elapsed
                self._record_prompt_batch_time(self._prompt_batch, elapsed)
            else:
                tic = time.perf_counter()
                gen_batch = self._prompt_batch.generate(
                    self.sampler,
                    self.tokenizer.stopping_criteria,
                    compute_logprobs=self.compute_logprobs,
                    top_logprobs_k=self.top_logprobs_k,
                )
                elapsed = time.perf_counter() - tic
                self._prompt_time_counter += elapsed
                self._record_prompt_batch_time(self._prompt_batch, elapsed)
                prompt_responses = self._prompt_batch_progress(self._prompt_batch)
                self._extend_generation_batch(gen_batch)
                self._prompt_batch = None
                mx.clear_cache()

            return prompt_responses, generation_responses

        return prompt_responses, generation_responses

    def next(self, **kwargs):
        with mx.stream(self._stream):
            return self._next(**kwargs)


def batch_generate(
    model: nn.Module,
    processor: ProcessorLike,
    images: Union[str, List[str], None] = None,
    audios: Union[str, List[str], None] = None,
    prompts: Optional[List[str]] = None,
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    group_by_shape: bool = True,
    track_image_sizes: bool = True,
    **kwargs: Unpack[GenerateKwargs],
) -> BatchResponse:
    """
    Generate responses for the given batch of prompts with variable-sized images.

    This function implements the transformers-style approach to batching:
    1. Group images with the same shape for efficient batch processing
    2. Process each group as a batch (no padding waste within groups)
    3. Track original image sizes for proper attention masking
    4. Restore results to original batch order

    Key insight: Instead of padding all images to the same spatial dimensions
    (which wastes computation and may hurt accuracy), we group same-sized
    images together so there's zero padding within each group.

    Args:
       model (nn.Module): The language model.
       processor (PreTrainedTokenizer): The tokenizer/processor.
       images (Union[str, List[str]]): Images (paths, URLs, or PIL images).
       audios (Union[str, List[str]]): Audio files (not yet supported for batching).
       prompts (List[str]): The input prompts.
       max_tokens (Union[int, List[int]]): Maximum number of output tokens. This
          can be per prompt if a list is provided.
       verbose (bool): If ``True``, print tokens and timing information.
       group_by_shape (bool): If ``True``, group same-shaped images for efficient
          batch processing.
       track_image_sizes (bool): If ``True``, track and return original image sizes.
       kwargs: The remaining options get passed to :obj:`BatchGenerator`.
          See :obj:`BatchGenerator` for more details.

    Returns:
        BatchResponse with generated texts, statistics, and optionally image_sizes.
    """
    from PIL import Image

    from ..utils import process_image

    processor.detokenizer.reset()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Handle single image case
    if isinstance(images, str):
        images = [images]

    # Handle no images case
    if images is None:
        texts, stats = _generate_batch(
            model, processor, prompts, None, max_tokens, verbose, **kwargs
        )
        return BatchResponse(texts, stats)

    # Load and preprocess images
    image_processor = (
        processor.image_processor if hasattr(processor, "image_processor") else None
    )

    processed_images = []
    image_sizes_original = []
    for img in images:
        if isinstance(img, str):
            pil_img = process_image(img, None, image_processor)
        elif isinstance(img, Image.Image):
            pil_img = img
        else:
            pil_img = img
        processed_images.append(pil_img)
        # Track original size
        if hasattr(pil_img, "height"):
            image_sizes_original.append((pil_img.height, pil_img.width))
        else:
            image_sizes_original.append((0, 0))

    # Group images by shape for efficient processing (no padding within groups)
    if group_by_shape and len(processed_images) > 1:
        grouped_images, grouped_indices = group_images_by_shape(processed_images)

        if verbose:
            print(f"[batch_generate] Found {len(grouped_images)} unique image shapes")
    else:
        # Single image or grouping disabled - treat as one group
        shape = (
            (processed_images[0].height, processed_images[0].width)
            if processed_images
            else (0, 0)
        )
        grouped_images = {shape: processed_images}
        grouped_indices = {shape: list(range(len(processed_images)))}

    # Process each shape group
    all_texts = [None] * len(prompts)
    all_image_sizes = [None] * len(prompts)
    total_stats = BatchStats()

    for shape, indices in grouped_indices.items():
        # Get images and prompts for this shape group
        group_images = [processed_images[i] for i in indices]
        group_prompts = [prompts[i] for i in indices]
        group_sizes = [image_sizes_original[i] for i in indices]

        # Handle per-sample max_tokens
        if isinstance(max_tokens, list):
            group_max_tokens = [max_tokens[i] for i in indices]
        else:
            group_max_tokens = max_tokens

        group_kwargs = dict(kwargs)
        logits_processors = group_kwargs.get("logits_processors")
        if logits_processors is not None and isinstance(logits_processors, list):
            if not logits_processors or all(callable(p) for p in logits_processors):
                group_kwargs["logits_processors"] = logits_processors
            else:
                group_kwargs["logits_processors"] = [
                    logits_processors[i] for i in indices
                ]

        # Process the entire group at once (same shape = no padding needed)
        chunk_texts, chunk_stats = _generate_batch(
            model,
            processor,
            group_prompts,
            group_images,
            group_max_tokens,
            **group_kwargs,
        )

        # Store results in original order
        for j, orig_idx in enumerate(indices):
            all_texts[orig_idx] = chunk_texts[j]
            all_image_sizes[orig_idx] = group_sizes[j]

        # Accumulate stats
        total_stats.prompt_tokens += chunk_stats.prompt_tokens
        total_stats.prompt_time += chunk_stats.prompt_time
        total_stats.generation_tokens += chunk_stats.generation_tokens
        total_stats.generation_time += chunk_stats.generation_time

    text_only_indices = list(range(len(processed_images), len(prompts)))
    if text_only_indices:
        group_prompts = [prompts[i] for i in text_only_indices]
        if isinstance(max_tokens, list):
            group_max_tokens = [max_tokens[i] for i in text_only_indices]
        else:
            group_max_tokens = max_tokens

        group_kwargs = dict(kwargs)
        logits_processors = group_kwargs.get("logits_processors")
        if logits_processors is not None and isinstance(logits_processors, list):
            if not logits_processors or all(callable(p) for p in logits_processors):
                group_kwargs["logits_processors"] = logits_processors
            else:
                group_kwargs["logits_processors"] = [
                    logits_processors[i] for i in text_only_indices
                ]

        chunk_texts, chunk_stats = _generate_batch(
            model,
            processor,
            group_prompts,
            None,
            group_max_tokens,
            **group_kwargs,
        )

        for j, orig_idx in enumerate(text_only_indices):
            all_texts[orig_idx] = chunk_texts[j]

        total_stats.prompt_tokens += chunk_stats.prompt_tokens
        total_stats.prompt_time += chunk_stats.prompt_time
        total_stats.generation_tokens += chunk_stats.generation_tokens
        total_stats.generation_time += chunk_stats.generation_time

    mx.clear_cache()

    # Compute final stats
    if total_stats.prompt_time > 0:
        total_stats.prompt_tps = total_stats.prompt_tokens / total_stats.prompt_time
    if total_stats.generation_time > 0:
        total_stats.generation_tps = (
            total_stats.generation_tokens / total_stats.generation_time
        )
    total_stats.peak_memory = mx.get_peak_memory() / 1e9

    if verbose:
        print(f"[batch_generate] Finished processing {len(prompts)} samples")
        print(
            f"[batch_generate] Prompt: {total_stats.prompt_tokens} tokens, {total_stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[batch_generate] Generation: {total_stats.generation_tokens} tokens, "
            f"{total_stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[batch_generate] Peak memory: {total_stats.peak_memory:.3f} GB")

    response = BatchResponse(all_texts, total_stats)
    if track_image_sizes:
        response.image_sizes = all_image_sizes
    return response


def _clone_or_share_logits_processor(processor):
    if hasattr(processor, "clone"):
        return processor.clone()
    warnings.warn(
        "Sharing logits processor across batch entries because it does not "
        "implement clone(). Stateful logits processors should implement clone() "
        "to avoid shared state across sequences.",
        RuntimeWarning,
        stacklevel=2,
    )
    return processor


def _generate_batch(
    model,
    processor,
    prompts: List[str],
    images: List = None,
    max_tokens: Union[int, List[int]] = 100,
    verbose: bool = False,
    **kwargs,
) -> Tuple[List[str], BatchStats]:

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    batch_size = len(prompts)
    logits_processors = kwargs.pop("logits_processors", None)

    num_images_list = [
        1 if i < (len(images) if images is not None else 0) else 0
        for i in range(len(prompts))
    ]
    formatted_prompts = [
        apply_chat_template(
            processor,
            model.config,
            p,
            num_images=num_images_list[i],
        )
        for i, p in enumerate(prompts)
    ]

    add_special_tokens = should_add_special_tokens(model.config.model_type, processor)

    resize_shape = normalize_resize_shape(kwargs.pop("resize_shape", None))
    image_token_index = getattr(model.config, "image_token_index", None)

    inputs = prepare_inputs(
        processor,
        images=images,
        audio=None,
        prompts=formatted_prompts,
        image_token_index=image_token_index,
        resize_shape=resize_shape,
        add_special_tokens=add_special_tokens,
        pad_to_uniform_size=False,  # Since images are pre-grouped by shape, they're already uniform size
    )
    input_ids = inputs.get("input_ids", None)
    pixel_values = inputs.get("pixel_values", None)
    mask = inputs.get("attention_mask", None)

    data_kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }

    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **data_kwargs
    )

    gen_kwargs = {
        **data_kwargs,
        **{k: v for k, v in embedding_output.to_dict().items() if v is not None},
    }

    if kwargs.get("prefill_step_size", DEFAULT_PREFILL_STEP_SIZE) is not None:
        policy_kwargs = dict(gen_kwargs)
        draft_model = kwargs.get("draft_model")
        draft_kind = kwargs.get("draft_kind")
        if draft_model is not None and draft_kind is not None:
            policy_kwargs.update(speculative_prefill_kwargs(draft_kind, draft_model))
        if not _chunked_prefill_enabled(
            model,
            input_ids=input_ids,
            inputs_embeds=embedding_output.inputs_embeds,
            draft_model=draft_model,
            draft_kind=draft_kind,
            prefill_kwargs=policy_kwargs,
        ):
            kwargs.pop("prefill_step_size", None)
            kwargs["prefill_step_size"] = None

    # Use batch_size for prefill and completion to ensure consistent processing
    existing_left_padding = None
    if mask is not None and getattr(mask, "ndim", 0) == 2:
        pads = [int(v) for v in (mask.shape[1] - mask.sum(axis=1)).tolist()]
        if any(pads):
            existing_left_padding = pads

    gen = BatchGenerator(
        model.language_model,
        processor,
        prefill_batch_size=batch_size,
        completion_batch_size=batch_size,
        compute_logprobs=False,
        existing_left_padding=existing_left_padding,
        **kwargs,
    )

    if logits_processors and all(
        callable(processor) for processor in logits_processors
    ):
        logits_processors = [
            [_clone_or_share_logits_processor(p) for p in logits_processors]
            for _ in range(batch_size)
        ]

    uids = gen.insert(
        input_ids.tolist(),
        max_tokens,
        prompt_kwargs=_split_prompt_kwargs_per_row(gen_kwargs, batch_size),
        logits_processors=logits_processors,
    )
    results = {uid: [] for uid in uids}

    tic = time.perf_counter()
    while gen.has_work:
        _, generation_responses = gen.next()
        for r in generation_responses:
            if r.finish_reason != "stop":
                results[r.uid].append(r.token)
    total_time = time.perf_counter() - tic

    gen.close()

    detokenizer = processor.detokenizer
    texts = []
    for uid in uids:
        detokenizer.reset()
        for t in results[uid]:
            detokenizer.add_token(t)
        detokenizer.finalize()
        texts.append(detokenizer.text)

    stats = gen.stats()
    stats.generation_time = total_time - stats.prompt_time
    if stats.generation_time > 0:
        stats.generation_tps = stats.generation_tokens / stats.generation_time
    return texts, stats
