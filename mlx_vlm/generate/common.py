from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce

from ..kv_quant import from_legacy as kv_quant_from_legacy
from ..models import cache
from ..turboquant import HybridQuantKVCache, TurboQuantKVCache, turboquant_enabled

DEFAULT_KV_GROUP_SIZE = 64
DEFAULT_KV_QUANT_SCHEME = "uniform"
DEFAULT_QUANTIZED_KV_START = 5000

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
DEFAULT_MIN_P = 0.0
DEFAULT_REPETITION_CONTEXT_SIZE = 20
# 8192 since 2026-09-05 (was 2048). Operator-approved default flip on the
# GLM-5.3-Flash serving campaign: chunked prefill at 8192 raised single-box
# prefill throughput +11.5 % (8k) / +6.9 % (32k) with peak memory +16 GiB
# (181 -> 197 GiB, under the 243 GiB cap), and a 16k teacher-forced quality
# gate against an UNCHUNKED reference showed 8192 is no farther from exact
# than 2048 was (mean KL 0.0250 vs 0.0275 nats, both under the 0.042075 cap).
# Receipts: bench/hwdossier/receipts/sweep11/L7B_PREFILL_CHUNK_20260905,
# L7B2_CHUNK_QUALITY_20260905, L7B3_UNCHUNKED_REFERENCE_20260905 (private
# campaign repo). Override per process with PREFILL_STEP_SIZE (server) or
# --prefill-step-size (CLI). TP=2 mode caps b*s per forward at
# MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD (default 8192), so only B=1 chunks
# fit there at this size -- see mlx_vlm/tp/README_TP_SERVING.md.
DEFAULT_PREFILL_STEP_SIZE = 8192
DEFAULT_COMPLETION_BATCH_SIZE = 32
DEFAULT_PREFILL_BATCH_SIZE = 8
DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH = 64
DEFAULT_DIFFUSION_MAX_DENOISING_STEPS = 48

# A stream on the default device just for generation
generation_stream = mx.new_thread_local_stream(mx.default_device())


def _policy_enabled(policy) -> bool:
    return bool(getattr(policy, "enabled", policy))


def _chunked_prefill_enabled(
    model,
    *,
    input_ids=None,
    inputs_embeds=None,
    prompt_cache=None,
    draft_model=None,
    draft_kind=None,
    prefill_kwargs=None,
) -> bool:
    prefill_kwargs = prefill_kwargs or {}
    candidates = [model]
    language_model = getattr(model, "language_model", None)
    if language_model is not None and language_model is not model:
        candidates.append(language_model)

    for candidate in candidates:
        policy = getattr(candidate, "chunked_prefill_policy", None)
        if callable(policy):
            return _policy_enabled(
                policy(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    prompt_cache=prompt_cache,
                    draft_model=draft_model,
                    draft_kind=draft_kind,
                    prefill_kwargs=prefill_kwargs,
                )
            )

    if any(getattr(candidate, "no_chunked_prefill", False) for candidate in candidates):
        return False

    # Hidden-state speculative prefill is model-contract dependent. Keep unknown
    # target models conservative unless they expose a chunked_prefill_policy.
    return draft_model is None


# --------------------------------------------------------------------------
# L35 -- two prefill wins (2026-09-07).  BOTH ARE ON BY DEFAULT since the
# rule-13 rail of 2026-09-07 (ledger I1437): on the DFlash2 natural gen1024
# panel plus a greedy arm, all 4/4 completion text sha256 were IDENTICAL to the
# off arm, speculative acceptance and rounds were identical (2.02/3.68/6.42/
# 4.57), and TTFT moved -1.5..-3 % (receipts L35_RAIL_20260907/).  Neither lever
# is bit-identical (see below), so the rail -- not an identity claim -- is what
# licences the default.
#
# SETTING EITHER ENV TO "0" RESTORES THE a6634a75 PATH EXACTLY, and
# ``MLX_VLM_GLM5_PREFILL_LOGITS_KEEP=0 MLX_VLM_GLM5_PREFILL_TAIL_MERGE=0``
# together restore it on every path (asserted in tests/test_prefill_chunk_plan.py
# ::TestBaseParityWhenBothAreOff).  That is the revert knob if the L40 KL gate,
# which is kept running as a check, ever fails.
#
# (a) PREFILL LM-HEAD SKIP (``MLX_VLM_GLM5_PREFILL_LOGITS_KEEP``, default 1).
#     The chunk LOOP already skips the vocab projection: every loop drops
#     ``chunk_out`` before the eval (ar.py, server/generation.py), and MLX never
#     computes an unreferenced graph -- measured here on CPU, a [1,2048,8192]
#     projection built-and-dropped costs 1.9 ms vs 32.9 ms when a slice of it is
#     evaluated.  What is NOT skipped is the FINAL forward of an *unchunked*
#     prefill: ``should_chunk`` is false whenever the prompt is <= the step, so
#     the whole prompt goes through one forward whose ``logits[:, -1, :]`` pulls
#     the full [B, S, vocab] matmul.  On GLM-5.3-Flash that is 309,760 B/token
#     (I1077) and, on the PFINAL receipt, the 8,192-token exact prefill's
#     out-of-forward time is 0.69-0.72 s against 0.12-0.22 s for the runs whose
#     final forward is one token wide.  Slicing the hidden BEFORE the projection
#     removes it.
#
#     Not bit-identical: narrowing the projection changes the GEMM's M
#     dimension, which moves the last ulp of the kept row (I1098 declined to mix
#     this into a correctness fix for exactly that reason).  On a near-tie it can
#     flip a token -- the rail's 4/4 text-sha match is the evidence that it does
#     not, on the natural panel.
#
# (b) TAIL-CHUNK MERGE (``MLX_VLM_GLM5_PREFILL_TAIL_MERGE``, default 1, with
#     ``..._TAIL_MIN`` 1024 and ``..._TAIL_MODE`` grow).
#     The last chunk of an N-token prompt is (N-1) mod step wide, and a short
#     chunk runs at a worse per-token rate: the as-fed 32k run's 537-token tail
#     cost 1.788 s = 3.33 ms/token against 2.38 ms/token for its 8,192-token
#     chunks (PFINAL_PREFILL_PATH_20260906).  Folding a short tail into the
#     previous chunk removes that penalty.
#
#     Also not bit-identical, and the numerical evidence against identity is
#     stronger than for (a): L7B measured "identity across chunk sizes: FALSE"
#     at both 8k and 32k (L7B_PREFILL_CHUNK_20260905), and L7B3 put a chunked
#     prefill's first divergence from an unchunked reference at token 35/45 with
#     mean KL 0.025.  Chunk decomposition is chaos-limited on this model, which
#     is why the promotion rests on the rule-13 rail (text sha + acceptance
#     parity) rather than on an identity assertion.
#     (The L23e receipts' "identity across chunk sizes: True" is vacuous -- those
#     arms ran a single chunk size.)


# An explicit "0"/"false"/"no"/"off" (any case) turns a default-ON lever off.
# Everything else -- including the empty string -- is ON, so a launcher that
# emits ``NAME=`` for a variable it did not set (the shape of the TP passthrough
# bug fixed in server/tp_mode.py::launch_worker) lands on the DEFAULT rather
# than silently disabling the lever on one rank only.
_ENV_OFF = ("0", "false", "no", "off")


def _env_default_on(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return True
    return raw.strip().lower() not in _ENV_OFF


def prefill_logits_keep_enabled() -> bool:
    """(a), default ON since I1437.  ``...LOGITS_KEEP=0`` restores a6634a75."""
    return _env_default_on("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP")


def prefill_logits_keep_kwargs(language_model, width: int) -> dict:
    """``{"num_logits_to_keep": 1}`` when this forward may skip the projection.

    ``width`` is the number of columns the forward will process.  At width 1 the
    slice is the identity and the kwarg is withheld, so decode steps keep the
    arguments -- and therefore the kernels -- they always had.

    TP=2 is safe: ``num_logits_to_keep`` is in ``server/tp_mode.py``'s
    ``_RANK0_ONLY_KWARGS`` ("slices replicated lm_head output"), so the announce
    guard does not refuse it and the collective sequence is unchanged.
    """
    if int(width) <= 1 or not prefill_logits_keep_enabled():
        return {}
    if not getattr(language_model, "supports_num_logits_to_keep", False):
        return {}
    return {"num_logits_to_keep": 1}


def prefill_keep_cache_enabled() -> bool:
    """R-cc (V5, 2026-09-08).  ``MLX_VLM_GLM5_PREFILL_KEEP_CACHE``, default OFF.

    Both chunk loops call ``mx.clear_cache()`` after every chunk (generate/ar.py,
    server/generation.py).  That returns the allocator's free pool to the OS, so
    the ~17 GB of per-layer transients the NEXT chunk allocates are fresh pages:
    the forward pays the re-allocation and the first-touch faults INSIDE the timed
    region.  With the flag on the loop keeps the pool and reuses it.

    Bit-identical by construction: this is an allocator hint and touches no array,
    no shape and no kernel.  The cost it trades against is peak RSS -- the pool is
    not returned between chunks -- which is exactly what the epsilon arm measures.

    Default OFF, and OFF means "clear as today": unset restores byte-for-byte the
    95bbe594 loop.  Read per call (L40 EnvSpec mode ``per_call``); it is read once
    per chunk, next to a call that used to unmap gigabytes.
    """
    raw = os.environ.get("MLX_VLM_GLM5_PREFILL_KEEP_CACHE")
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def prefill_tail_merge_enabled() -> bool:
    """(b), default ON since I1437.  ``...TAIL_MERGE=0`` restores a6634a75."""
    return _env_default_on("MLX_VLM_GLM5_PREFILL_TAIL_MERGE")


def prefill_tail_min() -> int:
    """Tails strictly shorter than this are merged.  0 disables the merge."""
    try:
        return max(0, int(os.environ.get("MLX_VLM_GLM5_PREFILL_TAIL_MIN", "1024")))
    except ValueError:
        return 1024


def prefill_tail_mode() -> str:
    mode = os.environ.get("MLX_VLM_GLM5_PREFILL_TAIL_MODE", "grow").strip().lower()
    return mode if mode in ("grow", "balance") else "grow"


def tp_forward_token_room() -> Optional[int]:
    """How many ``b * s`` columns one announced TP forward may carry, or None.

    None means "not serving TP", which is every single-box process: the check
    below then costs one ``os.environ.get`` and no import.

    In TP=2 the control plane rides the data collective as a fixed-width int32
    vector, so a forward's token ids must fit the payload:
    ``tp/worker.py::encode`` RAISES ``TPUnavailable`` when ``len(flat)`` exceeds
    ``MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD`` minus the ``ECHO_WORDS`` reserved
    for the shape agreement (worker.py:224).  A grown tail chunk is the one
    thing in this file that can make a forward WIDER than ``step``, so it is the
    one thing that could turn a working TP deployment into a raise.  It does not
    -- ``next_prefill_chunk`` falls back instead.
    """
    raw = os.environ.get("MLX_VLM_GLM5_TP_HOSTS", "")
    hosts = [h for h in raw.split(",") if h.strip()]
    if len(hosts) < 2:
        return None
    try:
        from ..tp.worker import ECHO_WORDS, _max_tok
    except Exception:  # pragma: no cover - TP extras absent; treat as single-box
        return None
    return max(0, _max_tok() - ECHO_WORDS)


def next_prefill_chunk(
    remaining: int,
    step: int,
    *,
    tail_min: Optional[int] = None,
    mode: Optional[str] = None,
    enabled: Optional[bool] = None,
    batch: int = 1,
    tp_room: Optional[int] = None,
) -> int:
    """Width of the next prefill chunk.  ``min(step, remaining)`` when disabled.

    ``remaining`` is the number of columns the chunk loop still has to consume
    (the loops already hold back the final token, so this is prompt_len - 1).

    The merge only ever touches the LAST GRID CELL: it fires when what is left
    is one full chunk plus a short tail (``step < remaining < step + tail_min``),
    which means every chunk before it stays exactly ``step`` wide.  That is the
    property the APC/vault ladder needs -- ``align_boundaries`` (context_vault.py
    :469) admits only multiples of ``step``, and the deepest admissible boundary
    is at most the start of this cell, so no boundary can fall inside a merged
    chunk except the one at ``processed + step``.  That one is still honoured:
    ``CheckpointLadder.clamp`` / ``_next_apc_checkpoint_column`` run AFTER this
    and can only shorten, so a checkpointing request silently gets the old plan
    rather than a missed rung.

    ``grow``    -- one chunk of ``remaining`` (<= step + tail_min - 1, and
                   ``tail_min`` is clamped to ``step``, so <= 2*step - 1).  Costs
                   up to (tail_min-1)/step more activation peak on one chunk.
    ``balance`` -- two chunks of ceil/floor ``remaining/2``, memory-neutral, but
                   L7B's per-chunk rate falls with chunk size (209 tok/s at 8192
                   vs 200 at 4096), so it is expected to be the slower arm; it
                   exists for the memory-capped case.

    TP=2 CAVEAT.  ``batch`` (default 1) is the forward's batch dimension, and in
    TP mode ``batch * width`` must fit the control vector's payload
    (:func:`tp_forward_token_room`; ``tp_room`` overrides it for tests).  A grown
    chunk is the only width here that can exceed ``step``, so when it would
    overflow the cap this returns the ``balance`` split instead -- and, if even
    that does not fit, ``min(step, balanced)``, which is never wider than the
    plan the unmerged loop would have produced.  The lever therefore cannot turn
    a TP deployment that worked into ``TPUnavailable``; a cap that is already too
    small for ``step`` itself stays exactly as (mis)configured as it was.
    """
    remaining = int(remaining)
    step = int(step)
    if step <= 0 or remaining <= 0:
        return max(0, remaining)
    if remaining <= step:
        return remaining
    if enabled is None:
        enabled = prefill_tail_merge_enabled()
    if not enabled:
        return step
    tail_min = prefill_tail_min() if tail_min is None else max(0, int(tail_min))
    # ``tail_min`` is an ABSOLUTE token count sized for the shipped 8192 step, so
    # it is clamped to ``step``: without this, any deployment whose step is below
    # the threshold stops chunking altogether (at step 64 a 1024 threshold merges
    # every prompt up to 1088 columns into ONE chunk), and the activation peak of
    # the merged cell is unbounded relative to the step it was sized against.
    # With it, a merged chunk is never wider than ``2 * step - 1``.  No-op at
    # every step >= tail_min, which includes the serving default (8192) and the
    # pipeline-parallel prefill step (2048), so the plans the I1437 rail measured
    # are bit-for-bit the plans this produces.
    tail_min = min(tail_min, step)
    if tail_min <= 0 or remaining >= step + tail_min:
        return step
    # step < remaining < step + tail_min: one full chunk plus a short tail.
    mode = prefill_tail_mode() if mode is None else mode
    balanced = (remaining + 1) // 2
    width = balanced if mode == "balance" else remaining
    batch = max(1, int(batch))
    room = tp_forward_token_room() if tp_room is None else int(tp_room)
    if room is not None and width * batch > room:
        width = balanced if balanced * batch <= room else min(step, balanced)
    return width


def plan_prefill_chunks(
    remaining: int,
    step: int,
    *,
    tail_min: Optional[int] = None,
    mode: Optional[str] = None,
    enabled: Optional[bool] = None,
    batch: int = 1,
    tp_room: Optional[int] = None,
) -> List[int]:
    """The whole chunk plan for ``remaining`` columns -- the loop, unrolled.

    Exists so the plan is testable without a model; the loops call
    :func:`next_prefill_chunk` one chunk at a time (they must, because the
    checkpoint ladder can shorten any chunk out from under a precomputed plan).
    """
    plan: List[int] = []
    left = int(remaining)
    while left > 0:
        n = next_prefill_chunk(
            left,
            step,
            tail_min=tail_min,
            mode=mode,
            enabled=enabled,
            batch=batch,
            tp_room=tp_room,
        )
        if n <= 0:
            raise ValueError(f"non-advancing chunk plan: remaining={left} step={step}")
        plan.append(n)
        left -= n
    return plan


def maybe_quantize_kv_cache(
    prompt_cache,
    quantized_kv_start,
    kv_group_size,
    kv_bits,
    kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
    kv_key_bits: Optional[float] = None,
    kv_value_bits: Optional[float] = None,
    kv_key_scheme: Optional[str] = None,
    kv_value_scheme: Optional[str] = None,
):
    if kv_bits is None:
        return

    policy = kv_quant_from_legacy(
        kv_bits,
        kv_quant_scheme,
        kv_group_size,
        kv_key_bits,
        kv_value_bits,
        kv_key_scheme,
        kv_value_scheme,
    )
    if policy is not None and not policy.is_homogeneous:

        def hybridize(entry):
            if isinstance(entry, (HybridQuantKVCache, cache.RotatingKVCache)):
                return entry
            if getattr(entry, "preserve_auxiliary_kv_state", False):
                return entry
            if isinstance(entry, cache.KVCache):
                if entry.offset >= quantized_kv_start or entry.offset == 0:
                    built = HybridQuantKVCache(policy)
                    if entry.offset:
                        built.update_and_fetch(*entry.state)
                    return built
                return entry
            if isinstance(entry, cache.CacheList):
                entry.caches = [hybridize(sub) for sub in entry.caches]
                return entry
            if isinstance(entry, list):
                for i, sub in enumerate(entry):
                    entry[i] = hybridize(sub)
                return entry
            if isinstance(entry, tuple):
                return tuple(hybridize(sub) for sub in entry)
            return entry

        last_idx = len(prompt_cache) - 1 if len(prompt_cache) > 2 else -1
        for index, layer_cache in enumerate(prompt_cache):
            if index == last_idx:
                continue
            prompt_cache[index] = hybridize(layer_cache)
        return

    if turboquant_enabled(kv_bits, kv_quant_scheme):

        def quantize_entry(entry):
            if isinstance(entry, TurboQuantKVCache):
                return entry
            if isinstance(entry, cache.RotatingKVCache):
                return entry
            if getattr(entry, "preserve_auxiliary_kv_state", False):
                return entry
            if isinstance(entry, cache.KVCache):
                if entry.offset == 0:
                    # Empty: replace so update_and_fetch quantizes on the fly
                    return TurboQuantKVCache(
                        bits=kv_bits,
                        key_bits=kv_key_bits,
                        value_bits=kv_value_bits,
                    )
                if entry.offset < quantized_kv_start:
                    return entry
                return TurboQuantKVCache.from_cache(
                    entry,
                    bits=kv_bits,
                    key_bits=kv_key_bits,
                    value_bits=kv_value_bits,
                )
            if isinstance(entry, cache.CacheList):
                entry.caches = [quantize_entry(sub_entry) for sub_entry in entry.caches]
                return entry
            if isinstance(entry, list):
                for i, sub_entry in enumerate(entry):
                    entry[i] = quantize_entry(sub_entry)
                return entry
            if isinstance(entry, tuple):
                return tuple(quantize_entry(sub_entry) for sub_entry in entry)
            return entry

        # Skip the last layer (before final norm/LM head); it is sensitive to
        # quantization in deep models.
        last_idx = len(prompt_cache) - 1 if len(prompt_cache) > 2 else -1
        for index, layer_cache in enumerate(prompt_cache):
            if index == last_idx:
                continue
            prompt_cache[index] = quantize_entry(layer_cache)
        return

    for index, layer_cache in enumerate(prompt_cache):
        if (
            hasattr(layer_cache, "to_quantized")
            and layer_cache.offset >= quantized_kv_start
        ):
            prompt_cache[index] = layer_cache.to_quantized(
                group_size=kv_group_size,
                bits=int(kv_bits),
            )


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """Temporarily set the wired memory limit for generation."""
    if not mx.metal.is_available():
        yield
        return

    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    max_rec_size = mx.device_info()["max_recommended_working_set_size"]
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        print(
            f"[WARNING] Generating with a model that requires {model_mb} MB "
            f"which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    old_limit = mx.set_wired_limit(max_rec_size)
    try:
        yield
    finally:
        if streams is not None:
            for stream in streams:
                mx.synchronize(stream)
        else:
            mx.synchronize()
        mx.set_wired_limit(old_limit)


@dataclass
class GenerationResult:
    text: str = ""
    token: Optional[int] = None
    logprobs: Optional[List[float]] = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    total_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0
    cached_tokens: int = 0
    # Width of the prefill batch the served warm prefix was HARVESTED in, or
    # ``None`` when the prefix has no recorded provenance (L1b-1).
    cached_from_width: Optional[int] = None
    finish_reason: Optional[str] = None
    diffusion_canvas_tokens: int = 0
    diffusion_denoising_steps: int = 0
    diffusion_work_tokens: int = 0
    diffusion_canvas_tps: float = 0.0
    diffusion_work_tps: float = 0.0
    is_draft: bool = False
    draft_text: str = ""
    text_already_printed: bool = False
    diffusion_step: int = 0
    diffusion_total_steps: int = 0
    diffusion_canvas_index: int = 0
    diffusion_block_complete: bool = False


class PromptCacheState:
    """Holds KV cache and token history across conversation turns."""

    def __init__(self):
        self.cache: Optional[List[Any]] = None
        self.token_ids: Optional[List[int]] = None

    def find_prefix_length(self, new_ids: list) -> int:
        """Return the number of leading tokens that match the cached ids."""
        if self.token_ids is None:
            return 0
        max_len = min(len(self.token_ids), len(new_ids))
        for i in range(max_len):
            if self.token_ids[i] != new_ids[i]:
                return i
        return max_len

    def update(self, token_ids: list, kv_cache: list):
        """Store the full token sequence and corresponding KV cache."""
        self.token_ids = list(token_ids)
        self.cache = kv_cache
