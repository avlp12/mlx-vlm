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
from ..sampling_coupling import (
    PROPOSAL_KEY_XOR,
    sample_top_p_token_order,
    sampled_coupling_enabled,
)
from ..speculative.structured_ledger import resolve_structured_processor
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
    next_prefill_chunk,
    prefill_logits_keep_kwargs,
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

    def __init__(
        self,
        *,
        temperature: float,
        top_p: float,
        seed: int,
        coupled: Optional[bool] = None,
    ):
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = int(seed)
        # F2: shared-key Gumbel coupling between the proposal and the target.
        # Off by default -- unset env == the previous independent streams.
        self.coupled = (
            sampled_coupling_enabled() if coupled is None else bool(coupled)
        )

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
        return self._draw(logprobs, keys)

    def sample_proposal(
        self,
        logprobs: mx.array,
        *,
        row_ids: List[int],
        positions: List[int],
    ) -> mx.array:
        if self.coupled:
            # Same key, same nucleus mask, same token-id axis as the target:
            # mx.random.categorical is Gumbel-max, so the two draws agree
            # whenever the shared Gumbel argmax agrees -- close to the maximal
            # coupling sum_t min(p_t, q_t) instead of the collision
            # probability sum_t p_t q_t.
            keys = _position_keys(self.seed, row_ids, positions)
            return self._draw(logprobs, keys)
        keys = _position_keys(self.seed ^ PROPOSAL_KEY_XOR, row_ids, positions)
        return mx.vmap(self._sample_one, in_axes=(0, 0))(logprobs, keys)

    def _draw(self, logprobs: mx.array, keys: mx.array) -> mx.array:
        if self.top_p > 0 and self.top_p < 1.0:
            if self.coupled:
                return mx.vmap(self._sample_top_p_one_token_order, in_axes=(0, 0))(
                    logprobs, keys
                )
            return mx.vmap(self._sample_top_p_one, in_axes=(0, 0))(logprobs, keys)
        return mx.vmap(self._sample_one, in_axes=(0, 0))(logprobs, keys)

    def _sample_one(self, logprobs: mx.array, key: mx.array) -> mx.array:
        return mx.random.categorical(logprobs * (1 / self.temperature), key=key)

    def _sample_top_p_one_token_order(
        self, logprobs: mx.array, key: mx.array
    ) -> mx.array:
        return sample_top_p_token_order(logprobs, key, self.top_p, self.temperature)

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
        # L35(a), ON by default since I1437 (``...PREFILL_LOGITS_KEEP=0``
        # restores the old argument list).  This function serves BOTH the prefill
        # forward (called once, below the chunk loop) and every decode step.
        # Only the prefill forward is ever wider than one column, and only when
        # the prompt was NOT chunked -- the loop always hands this call a single
        # token.  So the kwarg is
        # decided from the actual width: decode steps and post-chunk prefills get
        # the argument list they always got, and a wide unchunked prefill stops
        # projecting the whole prompt into vocab space to read one row of it.
        _keep_width = (
            inputs_embeds.shape[1]
            if inputs_embeds is not None
            else (y.shape[-1] if y is not None and getattr(y, "ndim", 0) >= 1 else 1)
        )
        step_kwargs = {
            **step_kwargs,
            **prefill_logits_keep_kwargs(model.language_model, _keep_width),
        }

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
            from ..pipeline_runtime import maybe_open_pipeline, pipeline_bypass_reason

            bypass = (
                pipeline_bypass_reason(
                    ladder=bool(ladder),
                    capture=bool(_chunk_capture_kwargs),
                    warm=warm_prefix,
                    pixel_values=pixel_values,
                    mask=mask,
                    cache=prompt_cache,
                    input_ids=input_ids,
                    kv_quantized=any(
                        v is not None for v in (kv_bits, kv_key_bits, kv_value_bits)
                    ),
                )
                if os.environ.get("MLX_VLM_PIPELINE_HOSTS", "").strip()
                else "disabled"
            )
            if bypass is None:
                pipeline = maybe_open_pipeline(model, total_tokens, verbose=verbose)
            elif verbose and os.environ.get("MLX_VLM_PIPELINE_HOSTS"):
                print(f"[pipeline] bypass={bypass}", flush=True)
            try:
                if pipeline is not None:
                    pipeline.begin(total_tokens, prefill_step_size, input_ids=input_ids)
                with tqdm(
                    total=total_tokens, desc="Prefill", unit="tok", disable=not verbose
                ) as pbar:
                    while inputs_embeds.shape[1] > 1:
                        # L35(b) x PP (A11).  The tail merge is ON by default
                        # (I1437) and folds a short final tail into the chunk
                        # before it -- but a PIPELINED chunk's width is not this
                        # loop's to choose.  ``PrefillEnvelope.create`` derives
                        # the peer's schedule as ``min(C, depth - p)`` per chunk
                        # and ``PipelineClient.prefill_chunk`` RAISES on any
                        # other width, so a grown chunk would fail the request
                        # here (no fallback on this path) rather than merely
                        # cost it a re-prefill.  With the peer open the width is
                        # therefore exactly ``min(step, remaining)``, which is
                        # the envelope's own arithmetic; with no peer this is
                        # the unified single-box path unchanged.
                        n_to_process = next_prefill_chunk(
                            inputs_embeds.shape[1] - 1,
                            prefill_step_size,
                            batch=inputs_embeds.shape[0],
                            enabled=False if pipeline is not None else None,
                        )
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

            finally:
                if pipeline is not None:
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
        # R1.  ``_step`` above has already applied ``processors`` to the FIRST
        # bonus token; the list used to stop here, so every speculative token
        # after it decoded unconstrained.  It is now carried into the round
        # loop, which threads a grammar processor through (the rail is ON by
        # default since the LU panel) and refuses the shapes it cannot serve.
        # With MLX_VLM_SPEC_STRUCTURED=0 the gate returns before it looks at the
        # list, so this argument changes nothing and that path is base 71732451.
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
            logits_processors=processors,
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

    ``dflash_warm_multirow`` counts multi-row warm batches declined because a
    dflash drafter would read padded rows as its round-1 context; its
    ``_rows_deferred`` companion counts the rows served COLD instead (they are
    not pushed back, so that name is a slight abuse -- the cost is a cold
    prefill, not a deferral).
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


def _hidden_tail_join_refusal(
    tail: Sequence[mx.array],
    reference: Sequence[mx.array],
    prefix_len: int,
) -> Optional[str]:
    """Why a stored APC hidden tail cannot be joined onto ``reference``, or ``None``.

    ``reference`` is this prefill's per-layer capture -- the thing the tail is
    about to sit in front of, either to prime the drafter
    (``PromptProcessingBatch._prepend_apc_hidden_tail``) or to be re-stored as
    the deeper entry's tail (``_hidden_tail_for_store``).  Both need the same
    answer, so both ask here.

    A tail is captured for one drafter's target-layer set, at that drafter's
    dtype and hidden width.  A mismatch means the entry was harvested under a
    different drafter (or a different quantisation), and joining it would be
    silently WRONG rather than merely short -- so it is named and refused
    instead of being reshaped into agreement.
    """
    if not tail:
        return "no stored tail"
    if len(tail) != len(reference):
        return (
            f"the stored tail has {len(tail)} captured layers, this prefill "
            f"has {len(reference)}"
        )
    if any(
        int(t.shape[0]) != 1
        or int(t.shape[-1]) != int(h.shape[-1])
        or t.dtype != h.dtype
        for t, h in zip(tail, reference)
    ):
        return "the stored tail's shape or dtype does not match this prefill's capture"
    if len({int(t.shape[1]) for t in tail}) != 1:
        return "the stored tail's layers disagree on length"
    if int(tail[0].shape[1]) > int(prefix_len):
        return (
            f"the stored tail is {int(tail[0].shape[1])} rows but the cached "
            f"prefix is only {int(prefix_len)}"
        )
    return None


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

        # A row that just finished is filtered out of ``uids``/``prompt_cache``
        # at the TOP of the *next* call to next() (see next()), not at the end
        # of THIS one -- so it stays fully present, cache and all, for exactly
        # the window between this next() call returning and the next one
        # being made. That window is what generate/server generation.py's
        # _step() uses to call capture_session/note_generated for the row
        # that just finished (LW5, 2026-09-07: capture_session's own docstring
        # already claimed this window existed -- "between finish_reason being
        # emitted and remove()" -- but remove() is never called on the normal
        # completion path, and this class used to filter() immediately at the
        # end of next(), so the window was empty and every capture refused
        # with uid_gone_from_batch, every time, for every plain-GenerationBatch
        # row). SpeculativeGenerationBatch needed no equivalent change: its
        # _refresh_uids() only recomputes the visible uids list and never
        # compacts prompt_cache/_all_uids, so a finished row's cache is
        # already intact there until an explicit remove().
        self._pending_filter_keep: Optional[List[int]] = None

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
        # Apply the PREVIOUS call's deferred filter now, before doing any new
        # work -- see the comment on _pending_filter_keep in __init__ for why
        # this is deferred by exactly one call rather than applied at the end
        # of the call that computed it.
        if self._pending_filter_keep is not None:
            self.filter(self._pending_filter_keep)
            self._pending_filter_keep = None

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
            # NOT self.filter(keep) here -- deferred to the top of the NEXT
            # call (see __init__/_pending_filter_keep) so a row that just
            # finished (its uid and cache still fully present in
            # uids/prompt_cache) is visible to the caller's own
            # capture_session/note_generated for the response this call is
            # about to return, before it is ever compacted away.
            self._pending_filter_keep = keep

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
        batch._pending_filter_keep = None
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
        logits_processors: Optional[List[Any]] = None,
        thinking_budget_criteria: Optional[List[Any]] = None,
    ):
        self.model = model
        self.draft_model = draft_model
        self.draft_kind = draft_kind
        # R1: the batch path never handed these on at all -- ``PromptProcessingBatch``
        # simply did not pass ``logits_processors`` when it built this class.  They
        # are carried now, and used when the structured rail is on (the
        # default); with MLX_VLM_SPEC_STRUCTURED=0 they are ignored exactly as
        # they were before.
        self.logits_processors = logits_processors or []
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
        # One thinking-budget criteria per ROW, indexed like ``_all_uids`` (the
        # speculative loops call back with the original row index, never the
        # active slot).  ``None`` rows have no budget.
        criteria = list(thinking_budget_criteria or [])
        criteria.extend([None] * (len(uids) - len(criteria)))
        self.thinking_budget_criteria = criteria[: len(uids)]
        self._num_tokens = [0] * len(uids)
        self._finished = [False] * len(uids)
        self._sent_first = False
        self._rounds_iter = None
        # Refuse an unsupported STRUCTURED shape at construction rather than on
        # the first ``next()``, so the request fails where the batch was
        # admitted.  A request with no grammar processor, and any request at all
        # under MLX_VLM_SPEC_STRUCTURED=0, returns from the gate untouched.
        if self.logits_processors:
            resolve_structured_processor(
                self.logits_processors,
                batch_size=len(self._all_uids),
                draft_kind=draft_kind,
                call_site="SpeculativeGenerationBatch",
            )

    def _criteria_for(self, row: int):
        if row < 0 or row >= len(self.thinking_budget_criteria):
            return None
        return self.thinking_budget_criteria[row]

    def _emit_limit(self, row: int) -> Optional[int]:
        """Tokens the next round may emit for ``row`` (``None`` = uncapped).

        This is the whole thinking-budget mechanism on the speculative path: a
        budget never needs to change what the target would have produced, only
        to stop the accepted walk at an exact position -- and ``draft[:k]`` is
        the target's own token for every ``j < accepted``, so a truncated round
        is token-identical to the autoregressive reference cut at the same
        point.
        """
        criteria = self._criteria_for(row)
        if criteria is None:
            return None
        tokens_left = getattr(criteria, "tokens_before_budget_stop", None)
        if not callable(tokens_left):
            return None
        return tokens_left()

    def _forced_draft_ids(self, row: int) -> List[int]:
        criteria = self._criteria_for(row)
        if criteria is None:
            return []
        pending = getattr(criteria, "pending_forced_sequence", None)
        if not callable(pending):
            return []
        return list(pending())

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
            # The rounds call this once per EMITTED token, in order, per row --
            # the same contract the autoregressive loop gives the criteria, so
            # the criteria's own ``_forced_index`` ledger advances exactly once
            # per forced id as that id is emitted.
            criteria = self._criteria_for(seq_idx)
            if criteria is not None:
                criteria(int(token_id))
            return (
                self._finished[seq_idx]
                or self.stop_criteria(token_id)
                or self._num_tokens[seq_idx] >= self.max_tokens[seq_idx]
            )

        has_budget = any(c is not None for c in self.thinking_budget_criteria)
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
            logits_processors=self.logits_processors,
            emit_limit=self._emit_limit if has_budget else None,
            forced_draft_ids=self._forced_draft_ids if has_budget else None,
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

        # A FACTORY, not a statement, because the pipelined-prefill fallback has
        # to be able to build this cache a SECOND time: a mid-prefill peer
        # failure leaves half the layers written on this box and half never
        # written, and that cache can only be thrown away (see
        # ``_pipeline_fallback``).  Called immediately below, so with the
        # pipeline off nothing about the construction moved.
        def _build_prompt_cache():
            if draft_model is not None and draft_kind is not None:
                return make_speculative_prompt_cache(
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
            if (
                len(input_ids) == 1
                and right_pad_per_row is None
                and kv_bits is None
                and hasattr(model, "make_cache")
            ):
                return cache.make_prompt_cache(model)
            return _make_cache(
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

        self._build_prompt_cache = _build_prompt_cache
        self._kv_quantized = any(
            v is not None for v in (kv_bits, kv_key_bits, kv_value_bits)
        )
        # Two-box pipelined prefill (A4).  Decided ONCE per request, at the top
        # of the first chunk; ``_pipeline_declined`` latches that decision so no
        # later chunk pays the gate again.  A batch that never meets the gate --
        # every batch, with ``MLX_VLM_PIPELINE_HOSTS`` unset -- carries these
        # seven attributes and executes one ``os.environ.get``.
        self._pipeline = None
        self._pipeline_declined = False
        self._pipeline_slot = False
        self._pipeline_chunks: List[int] = []
        self._pipeline_chunks_done = 0
        self._pipeline_restore: Optional[dict] = None
        # A10-0: the per-request warm split (``warm_prefix_len``,
        # ``warm_suffix_len``, ``chunk_size``) for a request the warm arm of the
        # gate refused, or ``None``.  Instrumentation only -- nothing reads it
        # to make a decision -- and it is the per-request twin of the aggregate
        # ``pp_warm_suffix_tokens_hist`` an operator diffs out of ``/metrics``.
        self._pipeline_warm_stats: Optional[dict] = None
        # A5c: the per-request record of a SHORTENED schedule -- ``reason``
        # (``exact_column`` / ``vault_rung``), ``full_chunks``, ``chunks``,
        # ``chunk_size`` -- or ``None`` on every request whose schedule is the
        # full ``k``.  Per-request twin of ``pp_schedule_shortened``.
        self._pipeline_schedule_stats: Optional[dict] = None
        # A5: the vault ladder this request took out of the chunk loop's way,
        # kept so the post-finalize checkpoint knows which rows to store and so
        # a fallback can put the original rungs BACK (a peer that dies must not
        # turn a vault-on request into a silently vault-off one).
        self._pipeline_ladder: Optional[dict] = None

        if warm_cache is not None:
            self.prompt_cache = warm_cache
        else:
            self.prompt_cache = _build_prompt_cache()

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
        # Rows THIS PREFILL trimmed off the front of the drafter's context --
        # not rows APC never forwarded.  On an APC-warm row the capture covers
        # the SUFFIX only (the prefix was served from K/V and never went through
        # the target), so ``finish()`` reports 0 here and the dflash drafter
        # starts its context at absolute position 0 while the target's row-0
        # token actually sits at ``prefix_len``.  RoPE is relative, so the
        # drafter's attention scores over its own contiguous context are
        # unchanged and the emitted tokens cannot change (acceptance is resolved
        # against the target's argmax) -- but it is not the same arithmetic the
        # cold arm does, and acceptance is free to move.  Note the asymmetry:
        # the drafter's OWN trim (``_pretruncate_ctx``) drops rows AND advances
        # every layer cache's offset by the same amount, so the analogous value
        # for an APC-served prefix would be ``prefix_len``, not 0.  Left at 0
        # here because that is the status quo this dispatch change inherits
        # (MTP's warm rows have run this way and its head is NoPE-MLA, so it
        # cannot see the difference at all), and moving it is a change to the
        # drafter's absolute positions that cannot be settled without a live
        # acceptance measurement.
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
            # ``is not None`` alone is the WRONG test, and it was spuriously
            # refusing the commonest warm shape: a single warm request is
            # ``right_pad_per_row == [0]``, which is a batch with no padding in
            # it at all.  The refusal's whole argument is "the trailing rows are
            # padding for a short row", so it has to ask whether any row IS
            # short -- the same ``is not None and any(...)`` the two other
            # right-padding sites in this class already use (the ``prepare()``
            # declaration above and the last-real-token selection in
            # ``generate()``).
            if right_pad_per_row is not None and any(right_pad_per_row):
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

    def _next_apc_checkpoint_column(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> Optional[int]:
        """Column the next chunk must stop on, over APC's checkpoint and the vault's ladder.

        With ``self._vault is None`` this reduces term for term to what it was:
        ``apc_on`` reproduces the old two-clause guard, the vault list is empty,
        and the min is taken over the same single column per row.

        A11b.  ``start``/``end`` default to the window the loop is in right now,
        which is every historical caller.  The pipelined SCHEDULE asks about a
        window it has not reached yet -- the last chunk it may hand the peer --
        because the width of that chunk is the width this clamp will give it,
        and the schedule has to predict it exactly rather than approximately.
        """
        if not self._apc_meta or self._inputs_embeds is None:
            return None
        apc_on = self._apc_manager is not None and self._apc_mode == "exact"
        if not apc_on and self._vault is None:
            return None
        if start is None:
            start = self._processed_prompt_columns
        if end is None:
            end = self._processed_prompt_columns + self._inputs_embeds.shape[1]
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
                # A CALLABLE, not a list.  ``store_exact_cache`` rejects on
                # three conditions before it ever needs a tail (too few tokens,
                # an unclonable cache, the RAM LRU disabled), and building one
                # costs a per-layer copy plus an ``mx.eval`` sync in the middle
                # of the prefill.  Deferring it means a rejected store pays
                # nothing.  Bound to defaults so the closure cannot read a later
                # iteration's row.
                hidden_tail=(
                    lambda i=batch_idx, m=meta, c=checkpoint_len: (
                        self._hidden_tail_for_store(i, m, c)
                    )
                ),
            )
            meta["checkpoint_done"] = True

    def _hidden_tail_for_store(
        self, batch_idx: int, meta: dict, checkpoint_len: int
    ) -> Optional[list]:
        """The drafter's round-1 window over this checkpoint's tail, or ``None``.

        Stored beside the prompt-cache snapshot so a LATER warm request -- which
        forwards only its suffix and therefore has no hidden for the cached part
        of its prompt -- can still prime the drafter on the whole window instead
        of on the 16-token suffix (I1312).

        THE WIDTH IS ``min(keep, checkpoint_len)``, NOT ``keep``.  The
        accumulator counts CAPTURED COLUMNS, and a cold row in a left-padded B>1
        batch carries ``_left_padding_per_row[i]`` columns of padding in front of
        its first real token.  Bounding by ``keep`` alone therefore stored a tail
        whose leading rows were zero embeddings and whose width exceeded the
        prefix it claims to cover -- measured on a ragged (40, 24)-token batch
        with left pads (0, 16): both rows stored width 32 while the short row had
        only 24 real tokens.  ``checkpoint_len`` IS the row's real-token count at
        this boundary (it is what ``_row_real_tokens_processed`` was just checked
        against), and left padding is at the FRONT, so the last
        ``min(keep, checkpoint_len)`` captured columns are all real and the
        stored width can never exceed the stored prefix.

        CHAINING (the warm case).  A warm row's own capture covers only its
        suffix, so on its own it cannot furnish a window for the DEEPER entry it
        is about to store -- and because ``lookup_exact_cache`` prefers the
        deepest entry, a tail-less deeper entry silently retires the tail from
        turn 2 of every conversation.  So when the row arrived carrying a usable
        ``hidden_tail``, the new tail is that tail followed by this row's own
        captured rows, trimmed to ``keep`` from the end: rows
        ``checkpoint_len - w .. checkpoint_len - 1`` with
        ``w = min(keep, len(incoming) + suffix_rows)``.  ``w <= checkpoint_len``
        because ``len(incoming) <= prefix_len`` is checked before the join.

        The refusals, all of them fail-safe (storing ``None`` leaves that entry
        warming exactly as it does today):

        * ``self._chunk_capture_kwargs`` empty: no hidden-reading drafter, or a
          capture that does not ride the chunks, so the accumulator holds only
          the final forward's rows.
        * ``any(self._right_pad_per_row)``: a right-padded row's TRAILING
          capture columns are padding -- the same objection ``_capture_refusal``
          makes -- and the tail is exactly those columns.  ``[0]`` and
          ``[0, 0]`` are not padding and are allowed through.
        * the drafter declares no finite window (``prefill_context_keep`` is
          ``None``).  An unbounded tail is not a 16.8 MB MTP window but the
          whole prompt's activations kept alive for the life of the entry --
          gigabytes on a 131k prompt.
        * a warm row (``prefix_len > 0``) with no incoming tail, or one whose
          incoming tail cannot be joined (``_hidden_tail_join_refusal``), or a
          B > 1 warm row -- the tails are per row and the chain has to be built
          per row, which is the same limitation ``_prepend_apc_hidden_tail``
          records for the prepend.

        Every attribute is read through ``getattr``, as ``_harvest_provenance``
        does and for the same reason: several tests build this batch with
        ``__new__`` and set only what the method under test reads, so a store
        must not start depending on the drafter wiring being present.
        """
        if not getattr(self, "_chunk_capture_kwargs", None):
            return None
        right_pad = getattr(self, "_right_pad_per_row", None)
        if right_pad is not None and any(right_pad):
            return None
        keep = prefill_context_keep(
            getattr(self, "draft_kind", None), getattr(self, "draft_model", None)
        )
        if keep is None or keep <= 0:
            return None
        accumulator = getattr(self, "_prefill_hidden", None)
        if accumulator is None:
            return None
        checkpoint_len = int(checkpoint_len)
        prefix_len = int(meta.get("prefix_len", 0) or 0)
        own_rows = checkpoint_len - prefix_len
        if own_rows <= 0:
            return None
        own = accumulator.tail(batch_idx, min(keep, own_rows))
        if not own:
            return None
        if prefix_len == 0:
            return own

        # Warm row: chain the incoming tail in front of what this prefill saw.
        if len(getattr(self, "_apc_meta", []) or []) != 1:
            return None
        incoming = meta.get("hidden_tail") or []
        if _hidden_tail_join_refusal(incoming, own, prefix_len) is not None:
            return None
        need = keep - int(own[0].shape[1])
        if need <= 0:
            return own
        head_rows = min(need, int(incoming[0].shape[1]))
        chained = [
            mx.contiguous(mx.concatenate([t[:, -head_rows:], o], axis=1))
            for t, o in zip(incoming, own)
        ]
        # Evaluated for the same reason ``tail`` evaluates: this list is about
        # to sit on a cache entry that outlives the request, and a lazy array
        # there pins every intermediate of the prefill behind it.
        mx.eval(chained)
        return chained

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
            landed = [int(r) for r in rungs if int(r) == done]
            remaining = [r for r in rungs if int(r) > done]
            if landed:
                self._insert_vault_rungs(batch_idx, meta, landed)
            meta["vault_rungs"] = remaining

    def _insert_vault_rungs(self, batch_idx: int, meta: dict, rungs: List[int]) -> None:
        """Capture and store this row's cache at each of ``rungs``.

        The ONE place a vault rung is written from the batch path, so the
        pipelined full-depth checkpoint (A5) and the chunk loop's ladder store
        cannot drift apart in what they capture or how they name it: same
        ``capture_fragments`` over the same row snapshot, same
        ``insert_checkpoint``, same provenance.  Best-effort per rung -- a vault
        fault must never fail a request, and ``capture_fragments`` returns None
        rather than a partial ladder, which ``insert`` refuses.
        """
        row_cache = self._apc_prompt_cache_for_store(batch_idx)
        if row_cache is None:
            return
        full_ids = meta.get("full_input_ids") or []
        provenance = self._harvest_provenance(batch_idx)
        for r in rungs:
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

    # ------------------------------------------------- two-box pipelined prefill
    #
    # The server does NOT prefill through ``generate_step``: its GPU thread runs
    # ``BatchGenerator``, whose prefill is the chunk loop below.  The pipeline
    # was wired only into ``generate_step``, so every served request bypassed it
    # by construction.  This is the second call site, and it is gated on THIS
    # batch's own facts rather than on ``generate_step``'s arguments.
    #
    # WHAT THE PEER GETS.  ``prompt_step`` stops once the remainder fits in one
    # step and ``generate()`` runs that remainder as the final forward, so the
    # pipelined part is AT MOST the chunk loop's part: the first
    # ``k = ceil(T/C) - 1`` chunks of the plan the loop itself would run (A11b
    # -- the last of them may be L35's merged chunk or one clamped to a
    # checkpoint column), and ``k - 1`` of them when A5c shortens the schedule
    # to move a checkpoint column out of the peer's way
    # (``_pipeline_schedule_plan``).  Everything past the pipelined depth is
    # prefilled here AFTER ``finalize`` has installed all 45 layers -- one
    # remainder forward, or a chunk loop plus one, and in both cases the same
    # chunks a single box would have run -- so no token is ever forwarded by
    # half a stack, and decode is untouched single-box code.

    def _pipeline_warm_prefix_len(self) -> int:
        """How many columns of this prompt are already in the cache.

        Three ways they can be, and the handoff schema supports none of them: an
        APC/vault prefix hit (``prefix_len``), a resumed cache
        (``existing_left_padding``, which is why a non-zero left pad counts),
        and a chunk loop that has already run (defensive; the gate is only ever
        evaluated at column 0).  The DEEPEST of the three, because the question
        the caller asks is "how much of the prompt would a two-box prefill have
        to be handed that it cannot be", and one warm row is enough to refuse
        the batch.

        Note what the left-padding term also catches, unchanged from the
        predicate this replaces: a COLD batch of unequal-length rows is
        left-padded too, so ``max(...) > 0`` there as well.  That request is
        refused as warm rather than as ``batch_not_one`` only because warm is
        asked first, and it was before this split as well -- A10-0 moves no
        decision, only the name it is recorded under.
        """
        lengths = [int(self._processed_prompt_columns or 0)]
        lengths += [
            int((m or {}).get("prefix_len") or 0) for m in (self._apc_meta or [])
        ]
        lengths += [int(p or 0) for p in self._left_padding_per_row]
        return max(lengths)

    def _pipeline_warm_prefix(self) -> bool:
        """Is any part of this prompt already in the cache?"""
        return self._pipeline_warm_prefix_len() > 0

    def _pipeline_warm_facts(self):
        """``(reason, per-request stats)`` for the warm arm of the gate.

        ``(None, None)`` when nothing is cached.  Otherwise the reason is one of
        the two A10-0 names, decided by the uncached SUFFIX against the chunk
        size:

        * ``S`` is ``self._inputs_embeds.shape[1]``.  For a warm row that array
          is already the suffix -- ``input_ids`` here are the per-row prefill
          inputs, "for warm-start rows this is the suffix" (``__init__``) -- so
          the whole prompt is ``prefix_len + S`` and the number the chunk loop
          will actually work on is ``S``.  It is also exactly what
          ``_pipeline_chunk_schedule`` measures, which is the point: the split
          has to agree with the thing it predicts.
        * ``S <= C`` is ``warm_suffix_lt_chunk``.  ``ceil(S/C) - 1 == 0``, so
          there is no chunk to hand a peer at this C and no version of A10
          changes that -- this request stays single-box on arithmetic, not on
          policy.
        * ``S > C`` is ``warm_suffix_ge_chunk``: at least one chunk exists, so
          this is the shape A10-1..7 would pay for, and its frequency is the
          gate on building them at all.
        """
        prefix_len = self._pipeline_warm_prefix_len()
        if prefix_len <= 0:
            return None, None
        suffix_len = (
            int(self._inputs_embeds.shape[1])
            if self._inputs_embeds is not None
            else 0
        )
        chunk_size = int(self.prefill_step_size or 0)
        reason = (
            "warm_suffix_ge_chunk"
            if chunk_size > 0 and suffix_len > chunk_size
            else "warm_suffix_lt_chunk"
        )
        return reason, {
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "chunk_size": chunk_size,
        }

    def _pipeline_full_depth(self) -> int:
        """Tokens the pipelined part of this prefill will have written.

        The one depth at which a two-box prefill holds a COMPLETE cache before
        the request is over: ``finalize`` has just pulled stage B's caches back
        and installed them, the head owns all ``n_layers``, and the remainder
        has not been forwarded yet.  It is the ADMITTED depth and not ``k*C``:
        A5c may have given one chunk back to the head, in which case the
        remainder is ``C + r`` tokens rather than ``r``
        (``_pipeline_schedule_plan``).
        """
        return sum(self._pipeline_chunk_schedule())

    def _pipeline_plan_chunks(self, count: int) -> List[int]:
        """The FIRST ``count`` chunks of the plan a single box would run.

        A11b.  Two schedules used to meet in ``prompt_step`` and disagree: the
        loop's, which is L35's plan (a merged tail, a chunk clamped to a
        checkpoint column), and the peer's, which ``PrefillEnvelope.create``
        fixed at ``C`` per chunk.  A11 reconciled them by giving a chunk BACK to
        the head whenever they would differ -- which at the served ``C = 8192,
        tail_min = 1024`` is one prompt length in eight, and costs that request a
        whole extra ``C``-wide head chunk (+9-18 s of TTFT, measured 58.9 s vs
        48 s at 33k on 2026-09-07) to preserve a merge worth ~0.5 s.

        The envelope carries an explicit width list now, so the peer can run the
        loop's OWN chunks and the reconciliation is free.  What this returns is
        therefore a PREFIX of the single-box chunk sequence, which is the whole
        of the identity argument: chunk for chunk, the two-box prefill forwards
        the shapes the one-box prefill forwards, and everything past ``count``
        is the head's ordinary remainder.

        Only the LAST chunk may leave ``C``, and only in the two ways
        ``prompt_step`` leaves it:

        * L35's tail merge grows it to ``C + r - 1`` when what is left is one
          full chunk plus a short tail (``next_prefill_chunk``);
        * the APC/vault ladder clamps it to a checkpoint column that lands
          inside it (``_next_apc_checkpoint_column``), which is what the served
          config actually does -- the vault's deepest rung is
          ``((T-1)//stride)*stride``, i.e. exactly the ``kC`` the merge would
          have swallowed, so the served single-box loop never merges that chunk
          at all and A11's give-back was paid for a merge that does not happen.

        Every EARLIER chunk is exactly ``C``: the merge cannot fire before the
        last cell (``remaining >= 2C + r - 1 >= C + tail_min``), and a
        checkpoint column strictly inside one is refused by
        ``_pipeline_schedule_unserved`` rather than clamped, because a chunk
        boundary inside the pipelined part is a boundary at which half the KV
        is still on the peer.
        """
        step = int(self.prefill_step_size or 0)
        if count <= 0 or step <= 0:
            return []
        chunks = [step] * (count - 1)
        processed = step * (count - 1)
        total = int(
            self._inputs_embeds.shape[1] if self._inputs_embeds is not None else 0
        )
        width = next_prefill_chunk(
            total - 1 - processed,
            step,
            batch=int(self._inputs_embeds.shape[0]),
        )
        column = self._next_apc_checkpoint_column(start=processed, end=total)
        if column is not None:
            width = min(width, column - processed)
        if width <= 0:
            return []
        return chunks + [width]

    def _pipeline_schedule_unserved(self, chunks: List[int]) -> Optional[str]:
        """Which checkpoint a pipelined part ``depth`` tokens deep cannot pay.

        ``None`` when every rung and column this request owes is serveable at
        that depth.  Otherwise the NAME of the first one that is not, which is
        also A5c's shortening reason: ``exact_column`` or ``vault_rung``.

        PP cannot checkpoint MID-prefill: at a chunk boundary half the KV lives
        on the peer and the snapshot would be of a half-populated cache.  What
        it can do is take exactly one snapshot at ``depth``, immediately after
        ``finalize`` (A5), and then run everything past ``depth`` itself, over
        the full stack, as ordinary single-box ``prompt_step`` chunks (A5b).  So
        a requirement at column ``X`` is SERVEABLE at ``depth`` when:

        * ``X == depth`` -- the post-finalize snapshot IS it.  For the vault
          that is the collapse target; for APC exact, ``_pipeline_step`` calls
          ``_store_apc_exact_checkpoints`` after ``finalize`` and after the
          column advance, so the store sees all ``n_layers`` over a cache
          exactly ``depth`` deep.
        * ``depth < X < T`` -- it lands in the head's own remainder, where the
          chunk loop clamps to it exactly as it does on one box (``T=40, C=8``,
          column 36: the pipelined chunks, then a chunk clamped to 4 here, then
          ``generate()``'s final 4) and the same store fires from the same call
          site over the same complete cache.  This is A5b's argument, and A5c is
          nothing but the observation that ``depth`` is a CHOICE: one chunk
          fewer moves a column out of the pipelined part and into this case.
        * ``X < depth`` and ``X`` is a vault rung ON ONE OF THE PLAN'S OWN
          BOUNDARIES -- the collapse drops it and keeps the deepest serveable
          one instead (``_pipeline_collapse_ladder``).  That is A5's trade:
          early-divergence coverage for the peer, counted as
          ``pp_ladder_rungs_skipped``.  A11b asks the PLAN rather than
          ``rung % C``: the two agree on every uniform schedule, but the last
          chunk may be merged or clamped now, and a rung on a multiple of ``C``
          can then land strictly inside it -- where it is neither a boundary
          the collapse may drop nor a column a later chunk may stop on.

        and UNSERVEABLE otherwise:

        * an APC exact column strictly inside the pipelined part.  The entry is
          keyed on ``full_input_ids[:checkpoint_len]`` and stored at exactly
          that many tokens, so it can be neither moved nor dropped; a boundary
          there would snapshot a cache whose stage-B layers were never written.
          (A column the LAST chunk would run past is not this case: the plan
          ends that chunk on it, and ``X == depth`` above is the answer.)
        * a rung inside the pipelined part that is not one of the plan's
          boundaries, for the same reason.
        * a rung at or past the prompt length ``T``.  Left refused exactly as
          A5 refused it -- and unreachable in the served config, because
          ``align_boundaries`` admits only multiples of ``C`` strictly below the
          prompt length.

        The narrowing A5c makes to A5's vault clause is the middle case: a rung
        DEEPER than the pipelined part used to be refused with the unaligned
        ones, and it is the same request the APC exact column is (a column in
        the head's own remainder), so it is admitted on the same argument and
        stays PENDING through the collapse.
        """
        if not self._apc_meta:
            return None
        apc_on = self._apc_manager is not None and self._apc_mode == "exact"
        step = int(self.prefill_step_size or 0)
        total = (
            int(self._inputs_embeds.shape[1])
            if self._inputs_embeds is not None
            else 0
        )
        # A11b.  The BOUNDARIES, not ``% step``.  The two coincide while every
        # pipelined chunk is exactly ``C`` -- which is every schedule A5c could
        # produce -- but the last chunk may now be merged or clamped, and then
        # a rung on a multiple of ``C`` can land strictly INSIDE it, where it
        # is neither collapsible nor clampable.  Asking the plan directly is
        # the same question the old arithmetic asked, of a plan that can
        # answer it.
        depth = sum(chunks)
        boundaries = set()
        running = 0
        for n in chunks:
            running += n
            boundaries.add(running)
        for batch_idx, meta in enumerate(self._apc_meta):
            if meta is None:
                continue
            if apc_on:
                # The COLUMN, not the length: it is the column the chunk loop
                # would stop on, and it is what ``depth`` is measured in.  They
                # coincide on every request the rest of this gate admits (B=1,
                # no left pad, no right pad) and this keeps them from drifting
                # if one of those ever loosens.
                column = self._apc_checkpoint_column_for_meta(batch_idx, meta)
                if column is not None and column < depth:
                    return "exact_column"
            if self._vault is None:
                continue
            rungs = [int(r) for r in (meta.get("vault_rungs") or ())]
            if not rungs:
                continue
            if step <= 0:
                return "vault_rung"
            for rung in rungs:
                if rung < depth and rung not in boundaries:
                    return "vault_rung"
                if rung > depth and rung >= total:
                    return "vault_rung"
        return None

    def _pipeline_schedule_plan(self) -> dict:
        """The chunks this loop may hand the peer, and why they are that many.

        ONE derivation, read by everything that has to agree with it: the gate's
        refusal (``_pipeline_has_checkpoint_ladder``), the envelope
        (``_pipeline_open`` -> ``begin``), the loop's own schedule check
        (``_pipeline_step``), the collapse depth and the post-finalize
        checkpoint.  There used to be two -- the ladder predicate measured
        ``k*C`` while the schedule handed out ``k`` chunks -- and two is exactly
        the drift A5c would introduce by shortening one of them.

        ``needs_processing`` stops the loop while ``remaining > step``, so the
        FULL schedule is fixed the moment the batch is built: ``k = ceil(T/C) -
        1`` chunks, the first ``k-1`` of them exactly ``C`` and the last one
        whatever the single-box loop would run there -- ``C``, or L35's merged
        ``C + r - 1``, or a chunk clamped to a checkpoint column
        (``_pipeline_plan_chunks``).  A5c may spend one of them:

        * ``k`` chunks serve every checkpoint -> the schedule is ``k``, which is
          every request A5b already admitted, list for list.
        * they do not and ``k-1`` do -> the schedule is ``k-1`` (``shortened``,
          with the reason ``k`` failed on).  The head's post-finalize remainder
          is now ``C + r`` tokens and it splits them at the column exactly as
          one box splits them.  This is the case the first served smoke found:
          at ``C=8192`` the 32,780- and 131,084-token prompts have ``r = 12``,
          under ``APC_EXACT_PREFIX_GUARD_TOKENS``, so the exact column ``T-16``
          fell inside the last pipelined chunk and A5b refused BOTH.
        * ``k-1 <= 0`` -> no schedule at all, and the caller names it
          ``no_pipelined_chunks`` -- which is what it is, and is not a claim
          about this request's checkpoints.
        * ``k-1`` do not serve it either -> ``apc_checkpoint_ladder``, A5's
          refusal, now genuinely the last resort.
        * ``k-1`` serve it but are fewer than
          ``MLX_VLM_PIPELINE_MIN_PIPELINED_CHUNKS`` ->
          ``below_min_pipelined_chunks``.  The shortened head pays a whole extra
          chunk of ``C`` inside the TTFT (~``C/450`` s, +18 s at the served
          ``C=8192``), so the trade only pays while the pipelined part still
          dominates: two chunks, i.e. ``T >= ~3C``.  A POLICY refusal, named
          like ``below_min_tokens`` and not like a shape one.

        ``begin`` re-derives the chunk tuple from the ``depth + 1`` token ids it
        is handed and ``finalize`` refuses an envelope mismatch, so a drift
        between this arithmetic and the loop is a failure, never a silent
        divergence -- and the envelope the tail sees is the SHORTENED list,
        because ``_pipeline_open`` measures ``depth`` off ``chunks``.
        """
        step = int(self.prefill_step_size or 0)
        total = (
            int(self._inputs_embeds.shape[1])
            if self._inputs_embeds is not None
            else 0
        )
        plan = {
            "chunks": [],
            "full_chunks": 0,
            "chunk_size": step,
            "depth": 0,
            "shortened": False,
            "reason": None,
            "refusal": None,
        }
        if step <= 0 or total <= step:
            return plan
        full = -(-total // step) - 1
        plan["full_chunks"] = full
        chunks = self._pipeline_plan_chunks(full)
        if not chunks:
            return plan
        unserved = self._pipeline_schedule_unserved(chunks)
        if unserved is None:
            plan["chunks"] = chunks
            plan["depth"] = sum(chunks)
            return plan
        short = full - 1
        if short <= 0:
            # Nothing to shorten TO.  Leaving ``refusal`` unset is the point:
            # the request is refused for having no pipelined chunks, not for a
            # checkpoint the peer could have served at some other depth.
            return plan
        # A5c's fallback, and it stays UNIFORM.  The short schedule's last
        # chunk is not the loop's last cell, so the merge cannot fire in it
        # (``remaining >= 2C + r - 1``); and a checkpoint column strictly inside
        # it is a column the peer cannot serve at any width, because giving the
        # chunk back is precisely what moves that column into the head's own
        # remainder.  Clamping here would put it back on the boundary the
        # shortening exists to take it off.
        short_chunks = [step] * short
        if self._pipeline_schedule_unserved(short_chunks) is not None:
            plan["refusal"] = True
            return plan
        from ..pipeline_runtime import min_pipelined_chunks

        if short < min_pipelined_chunks():
            plan["refusal"] = "below_min_pipelined_chunks"
            return plan
        plan["chunks"] = short_chunks
        plan["depth"] = sum(short_chunks)
        plan["shortened"] = True
        plan["reason"] = unserved
        return plan

    def _pipeline_has_checkpoint_ladder(self, plan: Optional[dict] = None):
        """The refusal this request's checkpoints owe the gate, or ``False``.

        A name and not a bool now: ``apc_checkpoint_ladder`` (``True``) when no
        admissible depth serves them, ``below_min_pipelined_chunks`` when one
        does but the TTFT floor refuses to buy it.  ``pipeline_bypass_reason``
        takes either, the same way it takes ``capture`` and ``warm``.

        Derived from the plan the caller already computed, so the depth the gate
        judges is the depth the peer is actually sent.
        """
        return (plan or self._pipeline_schedule_plan())["refusal"] or False

    def _pipeline_collapse_ladder(self, depth: int) -> None:
        """Take the vault rungs out of the chunk loop's way for this request.

        Two things have to happen before the first pipelined chunk runs, and
        both are this method.  A rung the peer will run past must stop being
        PENDING, or ``_store_vault_checkpoints`` fires at the chunk boundary it
        lands on and captures a cache whose stage-B layers are empty -- a rung
        that restores to a fluent wrong answer, which is the one failure a cache
        change must not introduce.  And it must be REMEMBERED, because a peer
        that dies mid-prefill sends this request back to column 0 single-box,
        where the ladder is exactly as serveable as it was before the peer was
        ever dialled.

        A5c: only the rungs AT OR BELOW ``depth`` are the peer's business.  A
        rung deeper than the admitted depth -- which A5 refused and A5c admits,
        and which the default served ladder's ``k*C`` rung becomes the moment
        the schedule is shortened -- lands in the head's own post-finalize
        remainder, where the single-box loop clamps to it and stores it from a
        complete cache.  So it stays pending, is stored at ITS OWN depth rather
        than collapsed onto another, and is not counted as skipped.  It is still
        remembered: a fallback re-prefills from column 0 and re-assigns the
        whole original ladder, which must be the whole ladder whichever side of
        the depth each rung was on.
        """
        self._pipeline_ladder = None
        if self._vault is None or not self._apc_meta:
            return
        rows: dict = {}
        skipped = 0
        for batch_idx, meta in enumerate(self._apc_meta):
            if meta is None:
                continue
            rungs = [int(r) for r in (meta.get("vault_rungs") or ())]
            if not rungs:
                continue
            collapsed = [r for r in rungs if r <= depth]
            if not collapsed:
                # Nothing the peer would run past: the ladder is entirely in the
                # head's remainder and the chunk loop serves it unchanged.
                continue
            rows[batch_idx] = rungs
            skipped += sum(1 for r in collapsed if r != depth)
            meta["vault_rungs"] = [r for r in rungs if r > depth]
        if not rows:
            return
        self._pipeline_ladder = {"depth": int(depth), "rows": rows}
        from ..pipeline_runtime import note_pipeline_ladder_collapsed

        note_pipeline_ladder_collapsed(skipped)

    def _pipeline_restore_ladder(self) -> None:
        """Give the rungs back.  Runs when the peer failed and the request is
        about to be re-prefilled on one box, which can serve all of them."""
        ladder, self._pipeline_ladder = self._pipeline_ladder, None
        if not ladder:
            return
        for batch_idx, rungs in ladder["rows"].items():
            meta = self._apc_meta[batch_idx] if batch_idx < len(self._apc_meta) else None
            if meta is not None:
                meta["vault_rungs"] = list(rungs)

    def _pipeline_store_full_depth_checkpoint(self) -> None:
        """The one checkpoint a two-box prefill CAN take, taken.

        Called immediately after ``finalize`` and BEFORE the batch advances, so
        the preconditions are the ones the gate reasoned about and not a
        reconstruction of them: the head owns every layer, the cache holds
        exactly ``depth`` tokens, ``_processed_prompt_columns`` has not moved,
        and ``_apc_meta`` is intact.  The capture goes through
        ``_insert_vault_rungs`` -- the same ``capture_fragments`` +
        ``insert_checkpoint`` the single-box chunk loop stores a rung with -- so
        the entry's bytes, keys and provenance are what the vault expects and a
        loopback run is bit-identical to a single-box one at this depth.

        Never raises.  A fault here would otherwise reach ``_pipeline_step``'s
        handler, which would throw away a cache that is COMPLETE and correct and
        re-prefill the whole prompt -- paying the entire request over again for
        a best-effort store.
        """
        ladder, self._pipeline_ladder = self._pipeline_ladder, None
        if not ladder or self._vault is None:
            return
        try:
            depth = int(ladder["depth"])
            for batch_idx in ladder["rows"]:
                meta = (
                    self._apc_meta[batch_idx]
                    if batch_idx < len(self._apc_meta)
                    else None
                )
                if meta is not None:
                    self._insert_vault_rungs(batch_idx, meta, [depth])
        except Exception as exc:  # noqa: BLE001 - a store must not fail a prefill
            logger.warning(
                "pipeline: the post-finalize vault checkpoint failed (%r); the "
                "prefill itself is complete and the request continues", exc,
            )

    def _pipeline_capture_plan(self):
        """``(capture spec, refusal)`` for this batch's hidden-reading drafter.

        A6.  Until now ANY such drafter refused the pipeline outright
        (``speculative_hidden_capture``), which in the default served config --
        DFlash2 -- is every request, so the feature was unreachable twice over
        (A5 removed the other block).  What the drafter actually needs from a
        prefill is a bounded window of the target's own activations, and both
        boxes can produce their share of it:

        * ``layers``: a per-layer capture (``capture_layer_ids``; dflash and
          eagle3).  DFlash2's ``[5, 14, 24, 33, 42]`` STRADDLES the shipped
          split of 23, so the head keeps 5/14 and the tail returns 24/33/42.
          ``sorted(set(...))`` is not a normalisation of the drafter's list but
          a reproduction of the model's own order: ``Glm5NextModel.__call__``
          tests membership of a SET inside an ascending layer loop, so the
          single-box capture is ascending and deduplicated whatever order the
          drafter declared.
        * ``hidden``: MTP's ``return_hidden`` -- the pre-final-norm,
          mHC-collapsed hidden after the LAST layer, so the tail owns it whole.

        Two refusals survive, and they are different questions:

        * ``speculative_hidden_capture`` (the historical name, kept for the
          historical reason) when the drafter declares NO finite window.
          ``self._prefill_hidden.keep`` is read rather than
          ``prefill_context_keep`` recomputed, because the window that comes
          back has to be the window the accumulator would have trimmed to --
          including when ``_capture_refusal`` already forced it to ``None`` for
          a right-padded batch.  An unbounded window is not merely large, it is
          the whole prompt's activations on the wire: 3.2 GB at 131k for
          DFlash2, against 50.3 MB for the trailing 2047 rows.
        * ``capture_unsupported`` when the capture is a shape this rail has no
          merge for.  Additive: nothing produces it today (``chunk_capture_
          kwargs_for`` emits only the two forms above), which is exactly why it
          exists -- a third form added later must fall out of the pipeline by
          name rather than be handed a merge that was written for two.
        """
        kwargs = self._chunk_capture_kwargs
        if not kwargs:
            return None, None
        layer_ids = kwargs.get("capture_layer_ids")
        if layer_ids:
            kind, ids = "layers", sorted({int(i) for i in layer_ids})
        elif kwargs.get("return_hidden"):
            kind, ids = "hidden", []
        else:
            return None, "capture_unsupported"
        keep = getattr(self._prefill_hidden, "keep", None)
        if not keep or int(keep) <= 0:
            return None, "speculative_hidden_capture"
        return {"schema": 1, "kind": kind, "layers": ids, "keep": int(keep)}, None

    def _pipeline_adopt_capture(self, hidden, depth: int) -> None:
        """Seed the accumulator with the merged window, as ``depth`` appends would.

        The pipelined chunks never ran on this box, so the accumulator has
        nothing in it; what it gets instead is the one thing those chunks would
        have left behind -- the trailing ``keep`` rows of ``depth`` tokens,
        merged across the split -- plus the row count they stand for.  After
        this, ``generate()``'s post-finalize remainder forward appends its own
        capture exactly as it does on one box, and ``finish()`` returns the same
        arrays and the same ``target_hidden_offset``.

        Raises on anything unexpected, and the caller turns that into the
        single-box fallback: a drafter primed on a short or misordered context
        is a quietly worse answer, which is the failure mode this whole rail is
        built to avoid.
        """
        if not hidden:
            raise RuntimeError(
                "pipelined prefill: the peer returned no speculative capture"
            )
        self._prefill_hidden.adopt_window(hidden, rows_covered=depth)

    def _next_chunk_width(self, step: int) -> int:
        """Columns the next SINGLE-BOX prefill chunk takes -- L35's plan.

        A11b.  This used to answer for the pipelined chunk too, by turning the
        merge off while the peer held the request: the peer could not run a
        chunk of any width but ``C``, so the loop had to promise it would not
        ask for one.  The envelope carries an explicit width list now, so a
        pipelined chunk's width is READ FROM THE SCHEDULE that was handed to
        the peer (``prompt_step``) instead of re-derived here -- which is not
        merely tidier, it is the only thing that works: the collapse
        (``_pipeline_collapse_ladder``) drops the rungs the schedule may have
        clamped a chunk to, so a re-derivation after the collapse can no longer
        reproduce the width the peer was promised.

        What is left is the plain single-box width, which is what every caller
        of this method now is.
        """
        return next_prefill_chunk(
            self._inputs_embeds.shape[1] - 1,
            step,
            batch=self._inputs_embeds.shape[0],
        )

    def _pipeline_chunk_schedule(self) -> List[int]:
        """The chunks this loop will hand the peer, in order.

        The admitted schedule: the first ``k = ceil(T/C) - 1`` chunks of the
        plan a single box would run, unless A5c shortened it to ``k-1`` -- see
        ``_pipeline_schedule_plan``, which is where the decision is made and the
        only place it is made.
        """
        return list(self._pipeline_schedule_plan()["chunks"])

    def _pipeline_should_open(self) -> bool:
        return (
            self._pipeline is None
            and not self._pipeline_declined
            and self._processed_prompt_columns == 0
            and self.prefill_step_size is not None
        )

    def _pipeline_open(self) -> None:
        """Evaluate the admission gate once and, if it passes, take the lease."""
        from ..pipeline_runtime import (
            PipelineSettings,
            acquire_pipeline_slot,
            maybe_open_pipeline,
            note_pipeline_bypass,
            note_pipeline_schedule_shortened,
            note_pipeline_warm,
            pipeline_bypass_reason,
            pipeline_language_model,
        )

        self._pipeline_declined = True
        if not os.environ.get("MLX_VLM_PIPELINE_HOSTS", "").strip():
            # The feature is off.  Nothing above this line touched an mx array
            # and nothing below runs, so the statement sequence of the prefill
            # is the one it was before this call site existed.
            return
        total_tokens = int(self._inputs_embeds.shape[1])
        # A5b.  THE ROUTING RULE IS ASKED FIRST.  It used to be asked inside
        # ``maybe_open_pipeline``, i.e. after every request-shape reason below,
        # so a request the >= 16k policy never intended to route reported
        # whichever shape reason happened to fire -- and in the default served
        # config that was ``apc_checkpoint_ladder`` on EVERY request, at 8192
        # tokens as loudly as at 131072.  The histogram is the only thing an
        # operator reads to learn why the peer is idle, so a length refusal has
        # to be named as one: below the routing threshold this request was never
        # a pipeline candidate and its shape is not the reason it stayed home.
        settings = PipelineSettings.from_env()
        if settings is not None and total_tokens < settings.min_tokens:
            note_pipeline_bypass("below_min_tokens")
            return
        capture, capture_refusal = self._pipeline_capture_plan()
        # A10-0.  The warm arm now names ITSELF -- ``warm_suffix_lt_chunk`` or
        # ``warm_suffix_ge_chunk`` -- because only this call site knows the
        # suffix and the chunk size.  The gate is unchanged: it still asks the
        # ladder and the capture first, and a warm request either of those
        # refuses is still recorded under THEIR name, exactly as it was under
        # ``warm_prefix``.
        warm_reason, warm_facts = self._pipeline_warm_facts()
        # A5c.  Derived ONCE, here: the depth the gate judges below is the depth
        # the envelope carries and the depth the collapse and the post-finalize
        # checkpoint use.  Deriving it twice is how a shortened schedule and a
        # full-depth ladder test would disagree.
        schedule = self._pipeline_schedule_plan()
        reason = pipeline_bypass_reason(
            ladder=self._pipeline_has_checkpoint_ladder(schedule),
            capture=capture_refusal,
            warm=warm_reason,
            right_pad=(
                self._right_pad_per_row is not None
                and any(self._right_pad_per_row)
            ),
            pixel_values=self._prompt_kwargs.get("pixel_values"),
            mask=self._prompt_kwargs.get("mask"),
            cache=self.prompt_cache,
            input_ids=self._input_ids,
            kv_quantized=self._kv_quantized,
        )
        if reason is not None:
            if warm_reason is not None and reason == warm_reason:
                # Counted here and nowhere else, so ``pp_warm_requests`` is
                # reconcilable with the bypass histogram by construction: it is
                # the sum of the two warm names and nothing more.
                self._pipeline_warm_stats = note_pipeline_warm(
                    prefix_len=warm_facts["prefix_len"],
                    suffix_len=warm_facts["suffix_len"],
                    chunk_size=warm_facts["chunk_size"],
                )
            return
        lm = pipeline_language_model(self.model)
        if lm is None:
            note_pipeline_bypass("no_pipeline_hook")
            return
        if len(self.prompt_cache) != int(lm.pipeline_num_layers):
            # ``install_state`` refuses a cache that is not exactly n_layers
            # long, and it refuses it at FINALIZE -- after the whole prefill has
            # been paid for.  Refuse it here, where a fallback is still free.
            note_pipeline_bypass("cache_depth_mismatch")
            return
        chunks = list(schedule["chunks"])
        if not chunks:
            note_pipeline_bypass("no_pipelined_chunks")
            return
        if not acquire_pipeline_slot():
            return
        self._pipeline_slot = True
        try:
            pipeline = maybe_open_pipeline(self.model, total_tokens, verbose=False)
            if pipeline is None:
                self._pipeline_release()
                return
            self._pipeline = pipeline
            depth = sum(chunks)
            # ``begin`` hashes ``input_ids[:, :-1]``, so hand it the pipelined
            # prefix PLUS one token -- and ``chunks`` as well (A11b), because
            # the plan's last chunk may be merged or clamped and the envelope's
            # own uniform arithmetic would not derive it.
            pipeline.begin(
                depth + 1,
                int(self.prefill_step_size),
                input_ids=self._input_ids[:, : depth + 1],
                capture=capture,
                chunks=list(chunks),
            )
        except Exception as exc:  # noqa: BLE001 - a peer must not fail a request
            logger.warning(
                "pipeline: opening the two-box prefill failed (%r); this request "
                "prefills single-box", exc,
            )
            note_pipeline_bypass("pp_begin_failed")
            self._pipeline_release()
            return
        self._pipeline_chunks = chunks
        self._pipeline_chunks_done = 0
        # Counted HERE and nowhere else: the peer has the envelope, so this is
        # the first moment a shortening is a fact about a request that ran
        # rather than about one the gate was still thinking about.
        if schedule["shortened"]:
            self._pipeline_schedule_stats = note_pipeline_schedule_shortened(
                reason=schedule["reason"],
                full_chunks=schedule["full_chunks"],
                chunks=len(chunks),
                chunk_size=int(self.prefill_step_size),
            )
        # The gate has admitted the ladder; collapse it now, BEFORE the first
        # chunk, so no boundary the chunk loop is about to cross can fire a
        # capture over a half-populated cache.
        self._pipeline_collapse_ladder(depth)
        # Everything the fallback needs to start over.  ``_inputs_embeds`` is
        # the whole prompt's embedding and is held for the duration of the
        # pipelined prefill (1.07 GB at 131k) -- the price of being able to
        # re-prefill without re-embedding.
        self._pipeline_restore = {
            "input_ids": self._input_ids,
            "inputs_embeds": self._inputs_embeds,
            "columns": self._processed_prompt_columns,
            "prompt_kwargs": dict(self._prompt_kwargs),
        }

    def _pipeline_release(self) -> None:
        """End the lease.  Idempotent, and never raises: it runs in failure paths.

        ``close`` returns the connection to the pool if the request finalized and
        discards it (counting ``pp_failed``) if it did not, so the caller does
        not have to know which happened.
        """
        pipeline, self._pipeline = self._pipeline, None
        try:
            if pipeline is not None:
                pipeline.close()
        finally:
            if self._pipeline_slot:
                self._pipeline_slot = False
                from ..pipeline_runtime import release_pipeline_slot

                release_pipeline_slot()
            self._pipeline_chunks = []
            self._pipeline_chunks_done = 0

    def _pipeline_fallback(self, exc: BaseException) -> None:
        """Throw the half-filled cache away and restart this request single-box.

        ``PipelineHead.abort`` already refuses to reuse a partially filled cache
        and it is right to: layers ``[0, split)`` hold this box's writes for the
        chunks that got through and layers ``[split, n)`` hold nothing at all.
        There is no state to salvage and no way to tell the client, so the
        request pays the prefill again, from column 0, on a fresh cache -- which
        is what makes the fallback bit-identical to a run that never tried.
        """
        logger.warning(
            "pipeline: two-box prefill failed after %d/%d chunks (%r); discarding "
            "the partial cache and re-prefilling single-box",
            self._pipeline_chunks_done,
            len(self._pipeline_chunks),
            exc,
        )
        restore = self._pipeline_restore or {}
        self._pipeline_release()
        self._pipeline_restore = None
        # The re-prefill runs every chunk on this box and captures each one, so
        # the accumulator has to start where a request that never tried starts.
        # It is empty already on every path that reaches here today (the chunk
        # loop's ``append`` runs only in the single-box branch, and PP adopts
        # its window after the last chunk) -- reset anyway, because "empty
        # already" is a property of the caller and a stale window here would be
        # rows of a prompt prefix stitched in front of the whole prompt.
        self._prefill_hidden = PrefillHiddenAccumulator(keep=self._prefill_hidden.keep)
        # The single-box re-prefill can serve every rung, including the ones the
        # collapse dropped, so it gets the whole ladder back.
        self._pipeline_restore_ladder()
        self.prompt_cache = self._build_prompt_cache()
        if restore:
            self._input_ids = restore["input_ids"]
            self._inputs_embeds = restore["inputs_embeds"]
            self._processed_prompt_columns = restore["columns"]
            self._prompt_kwargs = dict(restore["prompt_kwargs"])
        mx.clear_cache()

    def _pipeline_step(self, n: int) -> Optional[int]:
        """One pipelined chunk, or ``None`` if the request fell back to one box.

        Advances the batch by exactly what the single-box branch advances it by;
        the two differ only in WHERE layers ``[split, n_layers)`` ran.
        """
        try:
            idx = self._pipeline_chunks_done
            if idx >= len(self._pipeline_chunks) or n != self._pipeline_chunks[idx]:
                raise RuntimeError(
                    f"pipeline chunk {idx} is {n} tokens, schedule says "
                    f"{self._pipeline_chunks[idx:idx + 1] or '(end)'}"
                )
            self._pipeline.prefill_chunk(
                self.model,
                self._input_ids[:, :n],
                self._inputs_embeds[:, :n],
                self.prompt_cache,
            )
            # Only stage A's caches exist on this box until finalize, so
            # scheduling the whole list would evaluate empty tail entries.
            mx.eval([c.state for c in self._pipeline.local_caches(self.prompt_cache)])
            self._pipeline_chunks_done += 1
            if self._pipeline_chunks_done == len(self._pipeline_chunks):
                # Pull stage B's KDA/DSA caches back and install them: from here
                # the remainder forward, the last token and all of decode run
                # locally over the full stack.
                self._pipeline.finalize(self.prompt_cache)
                # A6.  The drafter's context for the pipelined part, merged
                # from both halves and adopted BEFORE the lease is released --
                # inside this handler, so a window that did not arrive or does
                # not fit costs the request a single-box re-prefill rather than
                # a silently short drafter context.
                if self._chunk_capture_kwargs:
                    self._pipeline_adopt_capture(
                        self._pipeline.take_hidden(), sum(self._pipeline_chunks)
                    )
                # A5.  Here and nowhere else: all 45 layers are on this box,
                # the cache is exactly ``sum(chunks)`` tokens deep, and the
                # batch has not advanced past it yet.
                self._pipeline_store_full_depth_checkpoint()
                self._pipeline_restore = None
                self._pipeline_release()
        except Exception as exc:  # noqa: BLE001 - never surface a peer fault
            self._pipeline_fallback(exc)
            return None
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

    def prompt_step(self) -> int:
        """Process one chunk of the prompt. Returns tokens processed."""
        if not self.needs_processing():
            return 0

        step = self.prefill_step_size or self._inputs_embeds.shape[1]
        # A11.  THE GATE IS ASKED BEFORE THE WIDTH IS CHOSEN, because the width
        # DEPENDS on the answer: a pipelined chunk's width is the schedule's
        # (A11b), a single-box one's is ``_next_chunk_width``, and the two are
        # the same number only while the plan is a prefix of the loop's own.
        # Asking in the other order made the first chunk's width a prediction of
        # the gate's verdict, and a wrong prediction is not a slower request --
        # it is a ``pipeline chunk 0 is N tokens`` mismatch and a whole
        # re-prefill.  Nothing between here and the old call site touched an mx
        # array, and the lease this may take is released on every exit below
        # (``_pipeline_step`` -> fallback/release), so the move costs a request
        # that ends up single-box exactly what it cost before.
        if self._pipeline_should_open():
            self._pipeline_open()
        if self._pipeline is not None and self._pipeline_chunks_done < len(
            self._pipeline_chunks
        ):
            # A11b.  ONE derivation, and this is not it: the width of a
            # pipelined chunk was decided when ``_pipeline_schedule_plan`` built
            # the list the peer's envelope carries, and re-deriving it here
            # would have to reproduce a clamp whose rung the collapse has since
            # dropped.  The ENVELOPE is still the cross-check -- it was built
            # from this same list, and both ``PipelineHead.prefill_chunk`` and
            # the tail's receiver refuse a chunk whose width is not the one at
            # this index.
            n = int(self._pipeline_chunks[self._pipeline_chunks_done])
        else:
            n = self._next_chunk_width(step)
            checkpoint_col = self._next_apc_checkpoint_column()
            if checkpoint_col is not None:
                n = min(n, checkpoint_col - self._processed_prompt_columns)
        if n <= 0:
            return 0
        if self._pipeline is not None:
            processed = self._pipeline_step(n)
            if processed is not None:
                return processed
            # The peer failed mid-prefill: the batch is back at column 0 on a
            # fresh cache, so re-derive this chunk and run it on one box -- and
            # ``_pipeline`` is None now, so this is the L35 plan a request that
            # never tried would have taken.
            n = self._next_chunk_width(step)
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

    def _prepend_apc_hidden_tail(
        self, stitched: List[mx.array], offset: int
    ) -> Tuple[List[mx.array], int]:
        """Put the APC entry's stored hidden tail back in front of a warm capture.

        A warm row forwards only its SUFFIX, so ``stitched`` covers prompt
        positions ``prefix_len ..`` and a drafter primed off it sees 16 rows of a
        457-token prompt (I1312: 1.67 -> 1.25 accepted/round).  The entry that
        served this row carries the target hidden of the prefix's last
        ``len(tail)`` rows -- positions ``prefix_len - len(tail) .. prefix_len -
        1`` -- which is exactly the piece missing from the front.  Concatenate,
        then apply the drafter's window to the join.

        THE OFFSET.  ``target_hidden_offset`` keeps the meaning it has on the
        cold path and the one ``adopt_pretruncated_context`` implements: the
        number of leading rows of the WHOLE prompt's hidden that the caller
        already dropped, i.e. the absolute prompt position of row 0 of what the
        drafter is handed.  Cold and untrimmed that is 0; cold and trimmed to
        ``keep`` it is ``S - keep``; here row 0 sits at ``prefix_len - len(tail)``
        (plus whatever the window then trims off the join), so that is what this
        returns.  The two readings coincide because the drafter's ideal context
        is the whole prompt starting at position 0 in both cases.

        Refusals, each logged once in the ``_capture_refusal`` style and each
        leaving the batch EXACTLY as it is today:

        * ``B > 1``.  The tails are per row and generally differ in width, so
          there is no single time axis to concatenate on and no single offset to
          hand ``_dflash_rounds_batch`` (which applies one to every row's draft
          cache).  Fixing that means a per-row offset, which is a wider change
          than this one.
        * a tail that does not match the capture (layer count, feature width,
          dtype, batch dim), or one wider than the prefix it claims to cover.
          A tail is stored per drafter target-layer set; a mismatch means the
          entry was harvested under a different drafter and joining it would be
          silently wrong rather than merely short.
        * ``offset != 0``, i.e. the suffix capture alone already overflowed the
          drafter's window.  The tail would be trimmed away entirely, and
          prepending it before a capture that has already lost rows off its
          front would put a hole in the middle of the context.
        """
        metas = self._apc_meta or []
        if not any((m or {}).get("hidden_tail") for m in metas):
            return stitched, offset

        meta = metas[0] or {}
        tail = meta.get("hidden_tail") or []
        prefix_len = int(meta.get("prefix_len", 0) or 0)
        reason: Optional[str] = None
        if len(metas) != 1 or len(self.uids) != 1 or int(stitched[0].shape[0]) != 1:
            reason = (
                "the stored hidden tails are per row and differ in width, so a "
                "B > 1 batch has no single time axis to prepend on"
            )
        elif offset:
            reason = (
                f"the suffix capture already overflowed the drafter's window "
                f"(offset {offset}), so the stored tail would be trimmed away"
            )
        else:
            reason = _hidden_tail_join_refusal(tail, stitched, prefix_len)
        if reason is not None:
            logger.info(
                "speculative prefill: declining the stored APC hidden tail for "
                "this batch -- %s. The drafter is primed on the suffix only, as "
                "it was before the tail existed.",
                reason,
            )
            return stitched, offset

        tail_rows = int(tail[0].shape[1])
        joined = [mx.concatenate([t, h], axis=1) for t, h in zip(tail, stitched)]
        start = prefix_len - tail_rows
        keep = prefill_context_keep(self.draft_kind, self.draft_model)
        width = int(joined[0].shape[1])
        if keep is not None and 0 < keep < width:
            # Same trim ``finish()`` applies, on the same end, for the same
            # reason -- the drafter only ever reads its trailing ``keep`` rows.
            # ``mx.contiguous`` because a bare slice would pin the join.
            joined = [mx.contiguous(h[:, -keep:]) for h in joined]
            start += width - keep
        return joined, start

    def generate(
        self, sampler, stop_criteria, compute_logprobs=True, top_logprobs_k=0
    ) -> GenerationBatch:
        """Process final tokens and transition to GenerationBatch."""
        if self._pipeline is not None:
            # Unreachable by the schedule (``_pipeline_step`` finalizes and
            # releases the lease on the schedule's LAST chunk, whether that is
            # the loop's last chunk or -- A5c -- one before it), so this is a
            # guard and not a path: half the KV would be on the peer.  Falling
            # back restores the whole prompt, which the forward below then runs
            # unchunked on a fresh cache.
            self._pipeline_fallback(RuntimeError("prefill ended with the peer open"))
        call_kwargs = dict(self._prompt_kwargs)
        # Prefill leg: hidden captures yes, KDA rollback stash no.  Computed once in
        # ``__init__`` so this forward and every chunk before it carry byte-identical
        # capture kwargs -- the accumulator raises if the capture width moves.
        call_kwargs.update(self._prefill_capture_kwargs)

        # L35(a), ON by default since I1437: an UNCHUNKED batch prefill arrives
        # here with the whole prompt and reads exactly one row per sequence out of
        # the projection.  Withheld
        # when right padding is in play: there the row a sequence needs is at
        # ``width - 1 - right_pad[i]``, which a keep-1 slice would not contain.
        if not (self._right_pad_per_row is not None and any(self._right_pad_per_row)):
            call_kwargs.update(
                prefill_logits_keep_kwargs(self.model, self._inputs_embeds.shape[1])
            )
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
                stitched, self.target_hidden_offset = self._prepend_apc_hidden_tail(
                    stitched, self.target_hidden_offset
                )
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
                logits_processors=list(self.logits_processors),
                thinking_budget_criteria=list(self.thinking_budget_criteria),
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
        # uid -> len(prompt ids) at seed time, so capture_session's log line
        # can report the prompt/generated split (P/G) without having to
        # guess it back out of the combined accumulator.
        self._session_prompt_len: Dict[Any, int] = {}
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
        if _context_vault.session_tier_active():
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
                # Per-tier, not per-call: with session_tier_active() now
                # default ON (2026-09-07), SESSION is tried after PREFILL on
                # every lookup, including against duck-typed vault stand-ins
                # in tests that predate the tier kwarg and only implement
                # lookup(tokens). A fault -- or a missing kwarg -- on THAT
                # tier must not discard a hit PREFILL already found; only
                # this one tier's candidate is skipped.
                continue
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

        # LW4 (2026-09-07): the server log had nothing per-request to say
        # about whether the session tier ever had a chance, or where it gave
        # up -- this is that line, logged once per pick regardless of outcome
        # (hit, miss, or "not attempted because the flag is off"). Diagnostics
        # only cost a second trie walk (diagnose_lookup, not on lookup's own
        # hot path) and only run when the session tier was actually in play.
        _session_active = _context_vault.VaultTier.SESSION in tiers
        _diag = None
        if _session_active and hasattr(vault, "diagnose_lookup"):
            try:
                _diag = vault.diagnose_lookup(
                    list(ids_list), tier=_context_vault.VaultTier.SESSION
                )
            except Exception:  # noqa: BLE001 - diagnostics must never fail a pick
                _diag = None
        if hit is not None:
            _context_vault.record_session_pick(
                "hit" if hit_tier is _context_vault.VaultTier.SESSION
                else "hit_prefill"
            )
        elif _session_active:
            _context_vault.record_session_pick(
                "no_candidate" if not _diag or not _diag.get("candidates")
                else "diverged"
            )
        logger.info(
            "vault-pick: tier=%s prefix_len=%s of prompt_len=%d (session "
            "candidates=%s, longest common prefix=%s, first mismatch at "
            "position %s: got id %s expected %s)",
            (hit_tier.value if hit is not None else "none"),
            (int(hit.prefix_len) if hit is not None else have),
            len(ids_list),
            _diag.get("candidates") if _diag else ("n/a" if not _session_active else 0),
            _diag.get("longest_common_prefix") if _diag else "n/a",
            _diag.get("mismatch_position") if _diag else "n/a",
            _diag.get("got") if _diag else "n/a",
            _diag.get("expected") if _diag else "n/a",
        )

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

        # A warm multi-row batch with a dflash drafter is refused when -- and
        # ONLY when -- the batch is genuinely RIGHT-PADDED.
        #
        # 1. The hazard is the padding.  dflash CONSUMES the trailing rows of
        #    the prefill capture as its round-1 context (``_hidden`` -> ``fc``
        #    -> ``hidden_norm`` over the last ``sliding_window - 1`` rows), so a
        #    short row would be primed on the zero embeddings padded onto it.
        #    ``PromptProcessingBatch``'s ``_capture_refusal`` declines the
        #    trailing-context TRIM for this shape but nothing stops the drafter
        #    from reading those rows.  For MTP the whole question is moot --
        #    it primes from ``hidden_states[-1]`` and its head is NoPE-MLA.
        # 2. So the test is ``any(right_pad_per_row)``, not ``len(sequences) >
        #    1``.  Asking the coarser question BEFORE ``_apply_right_pad_policy``
        #    ran killed batches that had no padding in them at all: on a model
        #    that declines right-padded prefill (GLM-5-Next -- KDA recurrent
        #    state cannot be rolled, ``model_supports_right_padded_prefill`` is
        #    False) that policy has ALREADY either admitted only an
        #    equal-suffix-length group, or declined the mixed batch outright, so
        #    every surviving multi-row batch has ``right_pad_per_row`` all
        #    zeros.  Every dflash warm multi-row batch on the shipped model was
        #    being sent cold for a hazard that could not occur.
        # 3. ``make_speculative_prompt_cache`` being bypassed on a warm batch
        #    (``PromptProcessingBatch.__init__``, ``warm_cache is not None``
        #    wins over the speculative constructor) is NOT a dflash-specific
        #    hazard: that helper ignores ``draft_kind`` entirely and at B=1
        #    returns a plain ``make_prompt_cache``.
        #
        # Refusing rather than splitting keeps this to one branch; the rows are
        # not deferred, they are served cold in the caller's LEFT-padded path,
        # so nothing starves.  The APC blocks acquired by ``_apc_pick_for`` are
        # released on the way out, as ``_vault_pick_for`` does when it retires a
        # pick -- otherwise a refused batch leaks a reference per warm row.
        if (
            getattr(self, "draft_kind", None) == "dflash"
            and len(sequences) > 1
            and any(right_pad_per_row)
        ):
            _note_prefill_batch_refusal("dflash_warm_multirow", len(sequences))
            logger.info(
                "prefill batch refusal dflash_warm_multirow: a dflash drafter "
                "reads the TRAILING context rows of the prefill capture, which "
                "on a right-padded mixed warm/cold batch are padding for the "
                "short rows; declining the warm batch for %d row(s) "
                "(right_pad_per_row=%s) and serving them cold (left-padded). "
                "An unpadded warm batch -- B=1, or every row at the same suffix "
                "length -- is unaffected.",
                len(sequences),
                right_pad_per_row,
            )
            if self.apc_manager is not None:
                for p in picks:
                    if p is not None:
                        self.apc_manager.release(p.get("matched_blocks", []))
            return None

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
                # The drafter's window over the tail of the PREFIX this row is
                # being served from, when the entry carries one.  The prefill
                # below only covers the suffix, so without this the drafter is
                # primed on the suffix alone (I1312).  ``None`` on every cold
                # row and on any warm row whose entry predates the tail.
                "hidden_tail": (picks[i].get("hidden_tail") if picks[i] else None),
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

    @staticmethod
    def _release_prompt_batch_pipeline(prompt_batch) -> None:
        """Give back a pipelined prefill's lease, if it holds one.

        ``getattr`` rather than an attribute access because several tests build a
        ``PromptProcessingBatch`` with ``__new__`` and set only the fields the
        method under test reads.
        """
        release = getattr(prompt_batch, "_pipeline_release", None)
        if callable(release) and getattr(prompt_batch, "_pipeline", None) is not None:
            try:
                release()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.warning("pipeline: releasing a cancelled lease failed",
                               exc_info=True)

    def close(self):
        # ``__del__`` calls this, and it can run on a half-built generator.
        if getattr(self, "_prompt_batch", None) is not None:
            self._release_prompt_batch_pipeline(self._prompt_batch)
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
            if _context_vault.session_tier_active():
                # Seed the session key with the EXACT prompt ids the model will
                # see. Re-deriving them at completion from the request would
                # re-tokenise and could disagree by a token; these are the ones.
                self._session_tokens[self.uid_count] = list(p)
                prompt_len_map = getattr(self, "_session_prompt_len", None)
                if prompt_len_map is not None:
                    prompt_len_map[self.uid_count] = len(p)
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
        if not _context_vault.session_tier_active():
            return
        acc = self._session_tokens.get(uid)
        if acc is None:
            return
        try:
            acc.extend(int(t) for t in tokens)
        except Exception:  # noqa: BLE001 - bookkeeping must never fail a response
            self._session_tokens.pop(uid, None)
            getattr(self, "_session_prompt_len", {}).pop(uid, None)

    def forget_session(self, uid) -> None:
        """Drop ``uid``'s accumulator.  Called on removal so a cancelled or
        completed request cannot leak its token list for the process lifetime."""
        self._session_tokens.pop(uid, None)
        getattr(self, "_session_prompt_len", {}).pop(uid, None)

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
        # feature being switched off. Each one also gets an INFO log line
        # (LW4, 2026-09-07: the only thing the previous server log showed was
        # the two startup lines -- no per-request line said whether capture
        # even ran), in addition to the SESSION_SKIPS counter it already fed.
        def _refuse(reason: str) -> bool:
            _context_vault.record_session_skip(reason)
            logger.info("vault-session: refused uid=%s reason=%s", uid, reason)
            return False

        if not _context_vault.session_tier_active():
            return _refuse("flag_off")
        if getattr(self, "vault", None) is None:
            return _refuse("generator_has_no_vault")
        if not session_id:
            return _refuse("no_session_id_at_generator")
        try:
            gb = self._generation_batch
            if gb is None:
                return _refuse("no_generation_batch")
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
                return _refuse("uid_gone_from_batch")
            row = all_uids.index(uid)
            row_cache = _apc.snapshot_prompt_cache_row(gb.prompt_cache, row)
            if not row_cache:
                return _refuse("row_cache_unavailable")
            key = list(tokens) if tokens is not None else self._session_tokens.get(uid)
            if not key:
                return _refuse("empty_token_key")
            stored = _context_vault.record_session_turn(
                self.vault,
                key,
                row_cache,
                completed=True,
                session_id=session_id,
                ttl_s=ttl_s,
                adopt=False,
            )
            if stored:
                prompt_len = getattr(self, "_session_prompt_len", {}).get(uid)
                generated_len = (
                    len(key) - prompt_len if prompt_len is not None else None
                )
                logger.info(
                    "vault-session: captured uid=%s tokens=%d (prompt %s + "
                    "generated %s) session_id=%s",
                    uid, len(key),
                    prompt_len if prompt_len is not None else "?",
                    generated_len if generated_len is not None else "?",
                    session_id,
                )
            else:
                # record_session_turn names its own refusal via
                # record_session_skip; mirror it here as a log line too
                # rather than duplicating its reason-selection logic.
                logger.info(
                    "vault-session: not captured uid=%s tokens=%d -- see "
                    "session_skip_counts() for the reason", uid, len(key),
                )
            return stored
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
                    # A cancelled pipelined prefill still holds the peer's only
                    # slot and a pooled socket the peer thinks is mid-request;
                    # dropping the batch without saying so would strand both.
                    self._release_prompt_batch_pipeline(self._prompt_batch)
                    self._prompt_batch.uids = []
                    self._prompt_batch.prompt_cache = []
                    self._prompt_batch = None
                    mx.clear_cache()
                    return True

            # Already decoding. A plain GenerationBatch defers its own
            # end-of-turn filter by one next() call (see
            # GenerationBatch._pending_filter_keep) so capture_session has a
            # window to read a just-finished row's cache; flush that deferred
            # filter FIRST, before computing our own keep list, so both
            # filters apply against the same, current uids/prompt_cache
            # indexing rather than one going stale under the other. Always
            # safe here: remove() is only ever called for a DIFFERENT uid's
            # cancellation, reached from the server's outer loop strictly
            # after the _step() call that would have captured any row whose
            # filter is pending -- capture_session for that row has already
            # run by the time control could reach here.
            pending = getattr(self._generation_batch, "_pending_filter_keep", None)
            if pending is not None:
                self._generation_batch.filter(pending)
                self._generation_batch._pending_filter_keep = None
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
