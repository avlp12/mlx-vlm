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
    speculative_clamp_since,
    speculative_clamp_snapshot,
    speculative_stats_since,
    speculative_stats_snapshot,
)
from .dflash import (
    batched_draft_enabled,
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
    _mtp_round_timers_enabled,
    _mtp_rounds,
    _mtp_rounds_batch,
    _mtp_shared_kv_from_prompt_cache,
    _mtp_verify_target,
    _MTPVerifyResult,
    _speculative_walk_batch_deferred_greedy,
    _speculative_walk_deferred_greedy,
)
from .structured_ledger import StructuredLedger, resolve_structured_processor

__all__ = [
    "PrefillHiddenAccumulator",
    "batched_draft_enabled",
    "speculative_clamp_since",
    "speculative_clamp_snapshot",
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
    "_mtp_round_timers_enabled",
    "_mtp_rounds",
    "_mtp_rounds_batch",
    "_mtp_shared_kv_from_prompt_cache",
    "_mtp_verify_target",
    "_speculative_walk",
    "_speculative_walk_batch",
    "_speculative_walk_batch_deferred_greedy",
    "_speculative_walk_batch_uniform_acceptance",
    "_speculative_walk_deferred_greedy",
    "chunk_capture_kwargs_for",
    "format_speculative_stats",
    "get_speculative_rounds_batch",
    "make_speculative_prompt_cache",
    "mtp_prime_window",
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

    DFlash only.  See :func:`mtp_prime_window` for the analogous MTP knob --
    they are deliberately independent switches (:func:`prefill_context_keep`
    explains why).
    """
    return os.environ.get("MLX_VLM_SPEC_PREFILL_CTX_TRIM", "1") not in (
        "0",
        "false",
        "False",
    )


def mtp_prime_window() -> int:
    """Trailing prompt rows an MTP drafter is primed on, across chunk boundaries.

    ``0`` disables the carry: the chunk loop stops asking for ``return_hidden``
    on intermediate chunks, exactly as it did before server priming existed, so
    ``prefill_from_target_hidden`` only ever sees whatever a single (unchunked,
    or last-chunk-only) forward captured -- i.e. the pre-fix behaviour.

    Capped at 2048 by default rather than left unbounded, because
    ``Glm5NextMTPDraftModel._mask_for`` builds a DENSE ``[S, S]`` boolean
    attention mask (``create_attention_mask(..., return_array=True)`` ->
    ``create_causal_mask``) whenever it primes on S>1 tokens: the mask is
    O(S^2), so at the 2048 default it stays a single-digit-MB allocation, while
    an unbounded prime on a 131k-token prompt would try to build a mask with
    ~1.7*10^10 elements instead -- gigabytes for a boolean array, and the whole
    point of the cap is to never get there.

    Deliberately not memoized, for the same reason as
    :func:`prefill_context_trim_enabled`: a first-call memo is a test hazard and
    one ``os.environ`` lookup per *request* is free.
    """
    return int(os.environ.get("MLX_VLM_MTP_PRIME_WINDOW", "2048"))


def prefill_context_keep(draft_kind: str, drafter) -> Optional[int]:
    """Trailing context rows the drafter keeps from a round-1 hidden, or ``None``.

    ``None`` means "do not trim": either the drafter does not publish the
    contract, or its layers do not all discard the same prefix (a full-attention
    draft layer reads the whole context, so nothing may be hoisted in front of
    it).  The drafter is the only thing that knows this -- see
    ``DFlashDraftModel.prefill_context_keep``.

    ``mtp`` is handled separately from ``dflash`` and does NOT consult
    :func:`prefill_context_trim_enabled`.  That switch guards a
    correctness-preserving OPTIMIZATION for dflash: the drafter's own contract
    says it only ever reads its trailing K rows, so keeping just those rows
    changes nothing the drafter can see, and turning the switch off simply
    forgoes the memory saving.  For MTP there is no such contract -- the
    drafter has no declared window of its own, ``mtp_prime_window()`` chooses
    one for it -- so the window IS the feature (server priming across chunk
    boundaries), not an optional trim on top of an already-correct wider
    context.  Gating it on a dflash-focused kill switch would mean a deployment
    that flips ``MLX_VLM_SPEC_PREFILL_CTX_TRIM=0`` for dflash reasons silently
    also loses MTP prompt priming; ``MLX_VLM_MTP_PRIME_WINDOW=0`` is the
    dedicated off switch for that instead.
    """
    if drafter is None:
        return None
    if draft_kind == "mtp":
        window = mtp_prime_window()
        return window if window > 0 else None
    if draft_kind != "dflash":
        return None
    if not prefill_context_trim_enabled():
        return None
    fn = getattr(drafter, "prefill_context_keep", None)
    if not callable(fn):
        return None
    keep = fn()
    return None if keep is None else int(keep)


def chunk_capture_kwargs_for(prefill_capture_kwargs: dict) -> dict:
    """The capture kwargs an intermediate chunk of a chunked prefill must carry.

    Only two captures survive being split across chunk boundaries and stitched
    back together on the time axis by :class:`PrefillHiddenAccumulator`:

    * a per-layer capture (``capture_layer_ids`` -- dflash/eagle3): each chunk's
      per-layer ``[B, chunk_len, D]`` pieces concatenate cleanly along axis 1.
    * MTP's ``return_hidden``, but ONLY when :func:`mtp_prime_window` is on.
      When it is off, MTP's original (pre-priming) consumer wants the LAST
      prompt token's hidden only, which any single forward already gives it
      without paying for a capture on every chunk -- so the capture stays off
      the chunks in that case, exactly as before server priming existed.

    ``return_shared_kv`` never rides a chunk regardless: it is only consumed
    off the FINAL prefill forward (the live target KV state prefill hands to
    the round loop), and the model returns ``shared_kv_states = {}`` for MTP
    whether or not the kwarg rides a given forward (see
    ``Glm5NextModel.__call__`` in ``models/glm5_next/language.py``) -- so
    asking for it on a chunk would be a forward-kwarg for no benefit.
    """
    if not prefill_capture_kwargs:
        return {}
    if prefill_capture_kwargs.get("capture_layer_ids"):
        return prefill_capture_kwargs
    if prefill_capture_kwargs.get("return_hidden") and mtp_prime_window() > 0:
        return {"return_hidden": True}
    return {}


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
        self.append_layers(getattr(outputs, "hidden_states", None))

    def append_layers(self, captured) -> None:
        """:meth:`append` for a caller that holds the per-layer list itself.

        The two-box prefill does: its chunks are captured on the peer (and on
        this box's own half of the stack) into a plain list rather than into a
        ``LanguageModelOutput``, and the window that survives them is stitched
        by exactly this arithmetic -- so it must BE this arithmetic and not a
        second copy of it.
        """
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

    def tail(self, row: int, k: Optional[int] = None) -> List[mx.array]:
        """Per-layer copies of the last ``k`` captured rows of batch row ``row``.

        A READ taken while the prefill is still running -- at an APC exact
        checkpoint, so the copy can be stored alongside the prompt-cache
        snapshot and handed back to the drafter on a later warm request whose
        target forward only covers the suffix.  It is non-destructive: the chunk
        pieces it copies from are still owed to :meth:`finish`.

        ``k`` of ``None`` (or ``<= 0``) means "every row still resident"; the
        caller is expected to pass the drafter's own window
        (:func:`prefill_context_keep`), because the result is about to be stored
        on a cache entry that outlives the request.

        Same "never hand back a bare slice" rule as :meth:`finish`, and for the
        same reason twice over: an ``mx`` slice is a view that pins its parent
        buffer, and an unevaluated slice is a graph node that pins every
        intermediate behind it.  So each layer is copied with ``mx.contiguous``
        and the copies are evaluated before they leave -- otherwise a stored tail
        would retain the whole prefill it was cut from.

        Returns ``[]`` when nothing has been captured (no drafter, or a forward
        that carried no capture kwargs), which the caller reads as "no tail".

        ONE ROW, and the narrowing happens FIRST.  The naive order -- stitch the
        whole ``[B, S, D]`` batch on the time axis, then take one row and its
        last ``k`` columns -- materialises B times the bytes the caller asked
        for and O(S) of them where it wanted O(k).  So the row slice is applied
        to each chunk piece before the concatenation, and whole leading pieces
        that fall outside the window are skipped rather than concatenated and
        then thrown away (the same arithmetic :meth:`_prune` uses to release
        them).

        NOT bounded by the prompt.  ``k`` counts CAPTURED rows, and for a
        left-padded row the leading captured columns are padding, so a caller
        that wants real tokens must pass a ``k`` it has already bounded by the
        row's own real-token count -- see
        ``PromptProcessingBatch._hidden_tail_for_store``, which bounds by
        ``min(keep, checkpoint_len)``.  Left padding is at the FRONT, so a ``k``
        within the row's real-token count is all real.
        """
        if self._layers is None:
            return []
        want = None if k is None or int(k) <= 0 else int(k)
        start = 0
        if want is not None and self._widths:
            resident = sum(self._widths)
            while start < len(self._widths) - 1 and (
                resident - self._widths[start] >= want
            ):
                resident -= self._widths[start]
                start += 1
        out: List[mx.array] = []
        for slot in self._layers:
            if not slot:
                return []
            pieces = [p[row : row + 1] for p in slot[start:]] or [
                slot[-1][row : row + 1]
            ]
            h = pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=1)
            if want is not None and want < int(h.shape[1]):
                h = h[:, -want:]
            out.append(mx.contiguous(h))
        mx.eval(out)
        return out

    def adopt_window(self, layers: List[mx.array], *, rows_covered: int) -> None:
        """Seed the accumulator with a window ANOTHER box computed (two-box prefill).

        The pipelined part of a prefill runs ``k`` chunks across two machines, so
        the per-chunk captures this class normally collects never exist on the
        head at all.  What comes back instead is the one thing :meth:`finish`
        could still have used them for: the trailing ``keep`` rows of those ``k``
        chunks, already merged across the split.  Adopting it as a single piece
        and remembering how many rows it stands for reproduces the state ``k``
        real appends would have left -- and therefore reproduces :meth:`finish`
        exactly, including the offset it reports:

            single box   dropped_rows = k*C - resident,  skip = resident + r - keep
            two box      dropped_rows = k*C - W,         skip = W + r - keep
                         => dropped + skip = k*C + r - keep, both times.

        Refused (loudly, because a silently short context is a silently worse
        drafter) unless nothing has been appended yet, the layers agree on width,
        and the width is exactly the window the trim would have kept:
        ``min(keep, rows_covered)``.
        """
        if self._layers is not None:
            raise RuntimeError(
                "pipelined prefill: the hidden accumulator already holds chunks"
            )
        if not layers:
            raise RuntimeError("pipelined prefill: empty hidden window")
        rows_covered = int(rows_covered)
        width = int(layers[0].shape[1])
        if any(int(h.shape[1]) != width for h in layers):
            raise RuntimeError(
                "pipelined prefill: the merged hidden window's layers disagree "
                "on length"
            )
        want = rows_covered if self.keep is None else min(self.keep, rows_covered)
        if width != want or rows_covered <= 0:
            raise RuntimeError(
                f"pipelined prefill: hidden window is {width} rows over "
                f"{rows_covered}, expected {want}"
            )
        self._layers = [[h] for h in layers]
        self._widths = [width]
        self.total_rows = rows_covered
        self.dropped_rows = rows_covered - width

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


def _resolve_structured_ledger(
    logits_processors,
    structured_ledger,
    *,
    batch_size: int,
    draft_kind: Optional[str],
    call_site: str,
):
    """D1/D2/D3/D7 gate for one speculative entry point.

    ``structured_ledger`` is the test/caller injection hook: when a ledger is
    handed in directly it is used as is (a ``StubLedger`` in the CPU tests).
    Otherwise a request's ``logits_processors`` build one when the structured
    rail is on (the default since the LU panel), or refuse if the shape is
    unsupported.

    With MLX_VLM_SPEC_STRUCTURED=0 this returns ``None`` without inspecting the
    list, so the round loop runs exactly the code it ran before this feature
    existed -- including the pre-existing R1 behaviour of ignoring the
    processors.  That variable is the only thing that changes what a request
    does.
    """
    if structured_ledger is not None:
        return structured_ledger
    processor = resolve_structured_processor(
        logits_processors,
        batch_size=batch_size,
        draft_kind=draft_kind,
        call_site=call_site,
    )
    if processor is None:
        return None
    return StructuredLedger.from_processor(processor)


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
    logits_processors: Optional[List[Any]] = None,
    structured_ledger: Optional[Any] = None,
    emit_limit: Optional[Callable[[int], Optional[int]]] = None,
    forced_draft_ids: Optional[Callable[[int], List[int]]] = None,
    admission: Optional[Callable[[], Optional[dict]]] = None,
) -> Generator[Tuple[List[Optional[int]], None], None, None]:
    """Server-side speculative rounds for one batch.

    ``emit_limit(row)`` returns how many tokens the NEXT round may emit for that
    row, or ``None`` for uncapped; ``forced_draft_ids(row)`` returns the ids the
    round must place at the front of the draft when the cap is 0.  Together they
    are how a thinking budget rides the speculative loop instead of disabling
    it: a budget only has to stop the accepted walk at an exact position, and
    ``draft[:k]`` is the target's own continuation for every ``j < accepted``.
    Only the continuous-batching kinds (dflash, mtp) honour them; eagle3 and
    lookup ignore them, which is why the server still refuses a budget there.

    ``admission`` (V1b, ``MLX_VLM_SPEC_EXTEND_ACTIVE``) lets a LIVE batch grow:
    the round loop polls it at each round boundary for rows the server prefilled
    while this batch was decoding.  Only ``dflash`` implements it; every other
    kind refuses rather than silently dropping the rows, because a dropped
    admission is a request that never answers.  A B == 1 dflash batch that may
    grow is routed to the BATCH loop instead of the scalar one -- the scalar
    loop has no active-slot map to append to -- so under this flag a single-row
    request takes a different (equivalent, not bit-identical: see
    ``_adaptive_k_enabled``'s ON IDENTITY note) code path than it does today.
    """
    batch_size = int(first_bonus.shape[0]) if first_bonus.ndim > 0 else 1
    _validate_speculative_sampling(draft_model, greedy_sampling)
    structured_ledger = _resolve_structured_ledger(
        logits_processors,
        structured_ledger,
        batch_size=batch_size,
        draft_kind=draft_kind,
        call_site="run_speculative_server_rounds",
    )

    if admission is not None and draft_kind != "dflash":
        raise NotImplementedError(
            f"mid-stream row admission is implemented for dflash only, not "
            f"{draft_kind!r}; set MLX_VLM_SPEC_EXTEND_ACTIVE=0."
        )

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
            prompt_tokens=prompt_tokens,
            emit_limit=emit_limit,
            forced_draft_ids=forced_draft_ids,
        )
        return

    if draft_kind == "dflash":
        if batch_size == 1 and admission is None:
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
                structured_ledger=structured_ledger,
                emit_limit=emit_limit,
                forced_draft_ids=forced_draft_ids,
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
            emit_limit=emit_limit,
            forced_draft_ids=forced_draft_ids,
            admission=admission,
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
    logits_processors: Optional[List[Any]] = None,
    structured_ledger: Optional[Any] = None,
) -> Generator[Tuple[Any, mx.array], None, None]:
    B = input_ids.shape[0]
    _validate_speculative_sampling(draft_model, sampler_is_greedy)
    structured_ledger = _resolve_structured_ledger(
        logits_processors,
        structured_ledger,
        batch_size=B,
        draft_kind=draft_kind,
        call_site="run_speculative_rounds",
    )

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
            structured_ledger=structured_ledger,
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
