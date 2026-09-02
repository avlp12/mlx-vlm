import os
from typing import Any, Callable, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..models import cache
from .common import (
    _dflash_block_total,
    _format_speculative_stats,
    _speculative_walk,
    _speculative_walk_batch,
    _speculative_walk_batch_uniform_acceptance,
    speculative_stats_since,
    speculative_stats_snapshot,
)
from .dflash import (
    _dflash_committed_hidden_segments,
    _dflash_next_block_size,
    _dflash_rounds,
    _dflash_rounds_batch,
)
from .eagle3 import _eagle3_capture_layer_ids, _eagle3_rounds, _eagle3_rounds_batch
from .lookup import _lookup_rounds, _lookup_rounds_batch
from .mtp import (
    _buffer_mtp_target_cache,
    _effective_mtp_block_size,
    _mtp_draft_block_active,
    _mtp_draft_hidden,
    _mtp_next_block_size,
    _mtp_rounds,
    _mtp_rounds_batch,
    _mtp_shared_kv_from_prompt_cache,
    _mtp_verify_target,
    _MTPVerifyResult,
    _speculative_walk_batch_deferred_greedy,
    _speculative_walk_deferred_greedy,
)

__all__ = [
    "PrefillHiddenAccumulator",
    "_MTPVerifyResult",
    "_dflash_block_total",
    "_dflash_committed_hidden_segments",
    "_dflash_next_block_size",
    "_dflash_rounds",
    "_dflash_rounds_batch",
    "_effective_mtp_block_size",
    "_format_speculative_stats",
    "_lookup_rounds",
    "_lookup_rounds_batch",
    "_mtp_draft_block_active",
    "_mtp_draft_hidden",
    "_mtp_next_block_size",
    "_mtp_rounds",
    "_mtp_rounds_batch",
    "_mtp_shared_kv_from_prompt_cache",
    "_mtp_verify_target",
    "_speculative_walk",
    "_speculative_walk_batch",
    "_speculative_walk_batch_deferred_greedy",
    "_speculative_walk_batch_uniform_acceptance",
    "_speculative_walk_deferred_greedy",
    "format_speculative_stats",
    "get_speculative_rounds_batch",
    "make_speculative_prompt_cache",
    "prefill_capture_kwargs",
    "prefill_context_keep",
    "prefill_context_offset",
    "prefill_context_trim_enabled",
    "run_speculative_rounds",
    "run_speculative_server_rounds",
    "speculative_hidden_state",
    "speculative_prefill_kwargs",
    "speculative_stats_since",
    "speculative_stats_snapshot",
]


def format_speculative_stats(draft_model: nn.Module) -> Optional[str]:
    return _format_speculative_stats(draft_model)


def _validate_speculative_sampling(draft_model: nn.Module, greedy: bool) -> None:
    if getattr(draft_model, "requires_greedy_sampling", False) and not greedy:
        raise ValueError(
            f"{type(draft_model).__name__} supports greedy speculative decoding "
            "only; set temperature=0."
        )


def get_speculative_rounds_batch(draft_kind: str):
    if draft_kind == "lookup":
        return _lookup_rounds_batch
    if draft_kind == "eagle3":
        return _eagle3_rounds_batch
    if draft_kind == "mtp":
        return _mtp_rounds_batch
    if draft_kind == "dflash":
        return _dflash_rounds_batch
    raise ValueError(
        f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'eagle3', 'lookup', 'mtp']"
    )


def speculative_prefill_kwargs(draft_kind: str, drafter) -> dict:
    if draft_kind == "lookup":
        # Nothing to capture: the drafter reads token ids.  Prefill therefore
        # runs exactly as it does with no drafter attached.
        return {}
    if draft_kind == "mtp":
        return {"return_hidden": True, "return_shared_kv": True}
    if draft_kind == "eagle3":
        return {"capture_layer_ids": _eagle3_capture_layer_ids(drafter)}
    if draft_kind == "dflash":
        return {"capture_layer_ids": list(drafter.config.target_layer_ids)}
    raise ValueError(
        f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'eagle3', 'lookup', 'mtp']"
    )


def prefill_capture_kwargs(lm, capture_kwargs: dict) -> dict:
    """Prefill flavour of :func:`speculative_prefill_kwargs`.

    The prefill leg needs the *hidden* captures -- they are the drafter's context.
    It does not need the KDA rollback stash: rollback happens inside a speculative
    round, and every consumer of ``gdn_states`` in this tree reads it off a VERIFY
    forward, never off the object a prefill returns --

        speculative/dflash.py:861, :1065      (verify_out.gdn_states)
        speculative/lookup.py:109             (verify_out.gdn_states)
        speculative/mtp.py:175, :884, :1283   (verify_out / verify.gdn_states)
        speculative/eagle3.py:176-201         (first_out / tail_out / verify_out)
        speculative/utils.py:322-323          (prefill leg: hidden_states and
                                               shared_kv_states only)

    On a model that carries recurrent state the stash is sequence-shaped, so on a
    long prompt it is the dominant retained allocation of the whole request.  Ask
    the model not to build it -- but only if the model says it understands the
    request, because a model that forwards ``**kwargs`` into its decoder stack
    would raise on an unknown one.
    """
    if not capture_kwargs:
        return capture_kwargs
    if not getattr(lm, "supports_capture_gdn_states", False):
        return capture_kwargs
    if capture_kwargs.get("capture_layer_ids") is None:
        return capture_kwargs
    return {**capture_kwargs, "capture_gdn_states": False}


def prefill_context_trim_enabled() -> bool:
    """Kill switch for the trailing-context trim (see :class:`PrefillHiddenAccumulator`).

    Deliberately not memoized, for the reason recorded in
    ``drafters/qwen3_dflash/dflash.py``: a first-call memo is a test hazard and one
    ``os.environ`` lookup per *request* is free.
    """
    return os.environ.get("MLX_VLM_SPEC_PREFILL_CTX_TRIM", "1") not in (
        "0",
        "false",
        "False",
    )


def prefill_context_keep(draft_kind: str, drafter) -> Optional[int]:
    """Trailing context rows the drafter keeps from a round-1 hidden, or ``None``.

    ``None`` means "do not trim": either the drafter does not publish the
    contract, or its layers do not all discard the same prefix (a full-attention
    draft layer reads the whole context, so nothing may be hoisted in front of
    it).  The drafter is the only thing that knows this -- see
    ``DFlashDraftModel.prefill_context_keep``.
    """
    if draft_kind != "dflash" or drafter is None:
        return None
    if not prefill_context_trim_enabled():
        return None
    fn = getattr(drafter, "prefill_context_keep", None)
    if not callable(fn):
        return None
    keep = fn()
    return None if keep is None else int(keep)


def prefill_context_offset(outputs) -> int:
    """Rows a chunked prefill trimmed off the front of the drafter's context.

    Zero unless the prefill applied :func:`prefill_context_keep`.  It has to reach
    the round loop as ``target_hidden_offset`` -- see
    ``DFlashDraftModel.adopt_pretruncated_context``.
    """
    return int(getattr(outputs, "speculative_context_offset", 0) or 0)


class PrefillHiddenAccumulator:
    """Stitch a chunked prefill's per-layer hidden captures back into one list.

    An unchunked prefill hands the drafter ``out.hidden_states`` -- one
    ``[B, S, D]`` array per captured target layer.  A chunked prefill produces
    one such list per chunk, so the accumulator keeps a per-layer list of chunk
    pieces and concatenates them along the TIME axis at the end.

    Two things it deliberately does NOT do:

    * It never slices a chunk to its own trailing ``keep`` rows.  A chunk
      boundary is not the prompt end: the drafter's window is the last ``keep``
      rows of the WHOLE prompt, so trimming per chunk would keep the tail of
      every chunk and drop rows that belong in the window.  The trim is applied
      once, to the concatenation, at :meth:`finish`.
    * It never hands back a bare MLX slice.  ``mx`` slices are views that pin
      their parent buffer (measured: holding a 0.5 MB slice of a 204 MB parent
      keeps 204 MB live), so a bare ``h[:, -keep:]`` would retain the very
      full-prompt array this class exists to drop.  :meth:`finish` copies.

    Whole leading chunks *are* dropped as they age out (:meth:`_prune`), which is
    not the same operation: a chunk is only released once the pieces after it
    already cover ``keep`` rows, so no row of the final window is ever in it.
    """

    def __init__(self, keep: Optional[int] = None):
        self.keep = None if keep is None or int(keep) <= 0 else int(keep)
        self._layers: Optional[List[List[mx.array]]] = None
        self._widths: List[int] = []
        self.total_rows = 0
        self.dropped_rows = 0

    @property
    def active(self) -> bool:
        return self._layers is not None

    def append(self, outputs) -> None:
        """Collect one forward's captures.  A forward without captures is a no-op."""
        captured = getattr(outputs, "hidden_states", None)
        if not captured:
            return
        if self._layers is None:
            self._layers = [[] for _ in captured]
        if len(captured) != len(self._layers):
            raise RuntimeError(
                "chunked speculative prefill: capture width changed mid-prompt "
                f"({len(self._layers)} layers, then {len(captured)}). The capture "
                "kwargs must be identical on every chunk."
            )
        width = int(captured[0].shape[1])
        for slot, h in zip(self._layers, captured):
            if int(h.shape[1]) != width:
                raise RuntimeError(
                    "chunked speculative prefill: captured layers disagree on "
                    f"length ({width} vs {int(h.shape[1])})."
                )
            slot.append(h)
        self._widths.append(width)
        self.total_rows += width
        self._prune()

    def _prune(self) -> None:
        if self.keep is None or self._layers is None:
            return
        # Release the oldest chunk while what remains after it still covers the
        # window.  ``resident`` is the row count currently held.
        resident = self.total_rows - self.dropped_rows
        while len(self._widths) > 1 and resident - self._widths[0] >= self.keep:
            head = self._widths.pop(0)
            for slot in self._layers:
                slot.pop(0)
            self.dropped_rows += head
            resident -= head

    def pending(self) -> List[mx.array]:
        """The captures of the most recent chunk, for ``mx.eval``.

        Evaluating them is not optional: an unevaluated capture is a graph node
        that pins every intermediate behind it, so an accumulator of lazy chunk
        captures would hold the whole prefill's activations instead of 5 arrays.
        """
        if self._layers is None:
            return []
        return [slot[-1] for slot in self._layers if slot]

    def finish(self) -> Tuple[Optional[List[mx.array]], int]:
        """``(per-layer hidden, rows dropped off the front)``.

        The second element is what the drafter's own truncation would have added
        to each of its cache offsets had it been handed the untrimmed context --
        the caller must apply it (``target_hidden_offset``) or the drafter's RoPE
        positions move by that amount.
        """
        if self._layers is None:
            return None, 0
        keep = self.keep
        skip = 0
        out: List[mx.array] = []
        for slot in self._layers:
            h = slot[0] if len(slot) == 1 else mx.concatenate(slot, axis=1)
            if keep is not None and keep < int(h.shape[1]):
                skip = int(h.shape[1]) - keep
                h = mx.contiguous(h[:, -keep:])
            out.append(h)
        return out, self.dropped_rows + skip


def speculative_hidden_state(draft_kind: str, outputs):
    if draft_kind == "lookup":
        return None
    if draft_kind == "mtp":
        return outputs.hidden_states[-1]
    if draft_kind in ("dflash", "eagle3"):
        return mx.concatenate(outputs.hidden_states, axis=-1)
    raise ValueError(
        f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'eagle3', 'lookup', 'mtp']"
    )


def make_speculative_prompt_cache(
    lm,
    *,
    draft_kind: str,
    batch_size: int,
    left_padding,
    make_cache: Callable,
):
    if batch_size == 1:
        return cache.make_prompt_cache(lm)
    return make_cache(lm, left_padding)


def run_speculative_server_rounds(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    *,
    draft_kind: str,
    first_bonus: mx.array,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    stop_check: Optional[Callable[[int, int], bool]] = None,
    greedy_sampling: bool = False,
    shared_kv_states: Optional[dict] = None,
    eos_token_ids: Optional[set] = None,
    prompt_tokens: Optional[mx.array] = None,
    row_ids: Optional[List[int]] = None,
    target_hidden_offset: int = 0,
) -> Generator[Tuple[List[Optional[int]], None], None, None]:
    batch_size = int(first_bonus.shape[0]) if first_bonus.ndim > 0 else 1
    _validate_speculative_sampling(draft_model, greedy_sampling)

    if draft_kind == "lookup":
        if batch_size != 1:
            _lookup_rounds_batch()
        for tok, state in _lookup_rounds(
            model,
            draft_model,
            prompt_cache,
            prompt_tokens=prompt_tokens,
            first_bonus=int(first_bonus.reshape(-1).item()),
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=token_dtype,
            greedy_sampling=greedy_sampling,
        ):
            yield [tok], state
            if stop_check is not None and stop_check(0, tok):
                return
        return

    if draft_kind == "eagle3":
        if batch_size == 1:
            yield from (
                ([tok], state)
                for tok, state in _eagle3_rounds(
                    model,
                    draft_model,
                    prompt_cache,
                    hidden,
                    prompt_tokens=prompt_tokens,
                    first_bonus=int(first_bonus.reshape(-1).item()),
                    max_tokens=max_tokens,
                    sampler=sampler,
                    draft_block_size=draft_block_size,
                    token_dtype=token_dtype,
                    greedy_sampling=greedy_sampling,
                )
            )
            return

        yield from _eagle3_rounds_batch(
            model,
            draft_model,
            prompt_cache,
            hidden,
            prompt_tokens=prompt_tokens,
            first_bonus=first_bonus,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=token_dtype,
            stop_check=stop_check,
            eos_token_ids=eos_token_ids,
            greedy_sampling=greedy_sampling,
        )
        return

    if draft_kind == "mtp":
        yield from _mtp_rounds_batch(
            model,
            draft_model,
            prompt_cache,
            hidden,
            shared_kv_states,
            first_bonus=first_bonus,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=token_dtype,
            stop_check=stop_check,
            eos_token_ids=eos_token_ids,
            greedy_sampling=greedy_sampling,
            row_ids=row_ids,
        )
        return

    if draft_kind == "dflash":
        if batch_size == 1:
            for tok, state in _dflash_rounds(
                model,
                draft_model,
                prompt_cache,
                hidden,
                first_bonus=int(first_bonus.reshape(-1).item()),
                max_tokens=max_tokens,
                sampler=sampler,
                draft_block_size=draft_block_size,
                token_dtype=token_dtype,
                greedy_sampling=greedy_sampling,
                target_hidden_offset=target_hidden_offset,
            ):
                yield [tok], state
                if stop_check is not None and stop_check(0, tok):
                    return
            return

        yield from _dflash_rounds_batch(
            model,
            draft_model,
            prompt_cache,
            hidden,
            first_bonus=first_bonus,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=token_dtype,
            stop_check=stop_check,
            greedy_sampling=greedy_sampling,
            row_ids=row_ids,
            target_hidden_offset=target_hidden_offset,
        )
        return

    raise ValueError(
        f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'eagle3', 'lookup', 'mtp']"
    )


def run_speculative_rounds(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    input_ids: mx.array,
    first_token: mx.array,
    logprobs: mx.array,
    last_outputs: Any,
    *,
    draft_kind: str,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    sampler_is_greedy: bool = False,
    prompt_tokens: Optional[mx.array] = None,
    target_hidden_offset: int = 0,
) -> Generator[Tuple[Any, mx.array], None, None]:
    B = input_ids.shape[0]
    _validate_speculative_sampling(draft_model, sampler_is_greedy)

    if draft_kind == "lookup":
        if B != 1:
            _lookup_rounds_batch()
        mx.eval(first_token)
        bonus = first_token.item()
        yield bonus, logprobs
        # ``input_ids`` has been trimmed to its tail when prefill chunked, so the
        # caller passes the untrimmed prompt separately; fall back to input_ids
        # for callers that do not.
        yield from _lookup_rounds(
            model,
            draft_model,
            prompt_cache,
            prompt_tokens=prompt_tokens if prompt_tokens is not None else input_ids,
            first_bonus=bonus,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=input_ids.dtype,
            greedy_sampling=sampler_is_greedy,
        )
        return

    if draft_kind == "mtp":
        shared_kv_states = last_outputs.shared_kv_states
        hidden = last_outputs.hidden_states[-1]
        if B == 1:
            _buffer_mtp_target_cache(prompt_cache, draft_model, draft_block_size)
            mx.eval(first_token)
            bonus = first_token.item()
            yield bonus, logprobs
            yield from _mtp_rounds(
                model,
                draft_model,
                prompt_cache,
                hidden,
                shared_kv_states,
                prompt_tokens=input_ids,
                first_bonus=bonus,
                max_tokens=max_tokens,
                sampler=sampler,
                draft_block_size=draft_block_size,
                token_dtype=input_ids.dtype,
                greedy_sampling=sampler_is_greedy,
            )
        else:
            mx.eval(first_token)
            first_bonus = (
                first_token if first_token.ndim == 1 else first_token.reshape(-1)
            )
            yield first_bonus.tolist(), logprobs
            eos = getattr(model.config, "eos_token_id", None)
            if isinstance(eos, int):
                eos_set = {eos}
            elif eos is None:
                eos_set = None
            else:
                eos_set = set(int(x) for x in eos)
            yield from _mtp_rounds_batch(
                model,
                draft_model,
                prompt_cache,
                hidden,
                shared_kv_states,
                first_bonus=first_bonus,
                max_tokens=max_tokens,
                sampler=sampler,
                draft_block_size=draft_block_size,
                token_dtype=input_ids.dtype,
                eos_token_ids=eos_set,
                greedy_sampling=sampler_is_greedy,
            )
        return

    if draft_kind == "eagle3":
        hidden = mx.concatenate(last_outputs.hidden_states, axis=-1)
        if B == 1:
            mx.eval(first_token)
            bonus = first_token.item()
            yield bonus, logprobs
            yield from _eagle3_rounds(
                model,
                draft_model,
                prompt_cache,
                hidden,
                prompt_tokens=input_ids,
                first_bonus=bonus,
                max_tokens=max_tokens,
                sampler=sampler,
                draft_block_size=draft_block_size,
                token_dtype=input_ids.dtype,
                greedy_sampling=sampler_is_greedy,
            )
        else:
            mx.eval(first_token)
            first_bonus = first_token.squeeze(-1)
            yield first_bonus.tolist(), logprobs
            eos = getattr(model.config, "eos_token_id", None)
            if isinstance(eos, int):
                eos_set = {eos}
            elif eos is None:
                eos_set = None
            else:
                eos_set = set(int(x) for x in eos)
            yield from _eagle3_rounds_batch(
                model,
                draft_model,
                prompt_cache,
                hidden,
                prompt_tokens=input_ids,
                first_bonus=first_bonus,
                max_tokens=max_tokens,
                sampler=sampler,
                draft_block_size=draft_block_size,
                token_dtype=input_ids.dtype,
                eos_token_ids=eos_set,
                greedy_sampling=sampler_is_greedy,
            )
        return

    if draft_kind != "dflash":
        raise ValueError(
            f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'eagle3', 'lookup', 'mtp']"
        )

    hidden = mx.concatenate(last_outputs.hidden_states, axis=-1)
    if B == 1:
        mx.eval(first_token)
        bonus = first_token.item()
        yield bonus, logprobs
        yield from _dflash_rounds(
            model,
            draft_model,
            prompt_cache,
            hidden,
            first_bonus=bonus,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=input_ids.dtype,
            greedy_sampling=sampler_is_greedy,
            target_hidden_offset=target_hidden_offset,
        )
    else:
        mx.eval(first_token)
        first_bonus = first_token.squeeze(-1)
        yield first_bonus.tolist(), logprobs
        yield from _dflash_rounds_batch(
            model,
            draft_model,
            prompt_cache,
            hidden,
            first_bonus=first_bonus,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=input_ids.dtype,
            greedy_sampling=sampler_is_greedy,
            target_hidden_offset=target_hidden_offset,
        )
