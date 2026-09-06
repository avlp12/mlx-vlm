"""Grammar ledger for structured output under speculative decoding (D4).

Why a ledger at all
-------------------
A logits processor is a *one position at a time* interface: it consumes the last
token, fills one next-token bitmask, and masks one row of logits.  A speculative
round asks a different question -- "what is legal at each of the W positions of
this drafted block, *assuming the drafted prefix*" -- and the answer has to come
back before the target's verify logits are sampled, without disturbing the
matcher's real state (the drafted prefix is a guess; most of it will be thrown
away).

``llguidance`` answers exactly that question:
``fill_next_token_bitmask_par_with_draft_tokens(executor, [(matcher, idx,
drafts)], bitmask)`` walks the draft path and fills ``len(drafts) + 1`` rows in
one call while leaving the matcher where it was.  So this module needs no
clone/advance/rollback bookkeeping: one live ``LLMatcher``, one ``LLExecutor``,
and a fresh bitmask per round.

Row ``i`` of the block mask constrains block position ``i``, whose predecessor
is ``b`` for ``i == 0`` and ``drafts[i-1]`` otherwise -- the same alignment as
``verify_input = [b] + drafts`` and ``verify_out.logits[:, i]``.

Rows that follow an *illegal* draft come back all-ones (vacuous).  That is
harmless: the target mask at the first illegal position already excludes the
drafted token, so the speculative walk stops there and the vacuous rows are
never consulted.  Drafter-side masking is therefore an acceptance-rate
optimisation, not a correctness requirement.

Thinking (R9)
-------------
``ThinkingAwareLogitsProcessor`` activates on the token that *is* the thinking
end token and constrains the **next** one; the end token itself is never fed to
the matcher.  ``StructuredLedger`` reproduces that off-by-one exactly: while
inactive it emits all-allow rows, watches every committed token for the end
token, and starts the grammar at the token immediately after it.  Because the
drafted block is known up front, activation *inside* a block is resolved too:
the predecessor chain says which row the grammar starts at.

R2 (empty legal set)
--------------------
An all-``-inf`` logits row is not an error to ``argmax`` (it returns 0) or to
``mx.random.categorical`` (it returns garbage), so a grammar dead end would emit
a silently wrong token.  Every grammar-filled row is popcount-checked and a zero
raises.

R3 (shared bitmask vs. lazy evaluation)
---------------------------------------
MLX evaluates lazily, so a bitmask buffer that is overwritten in place before
the round that referenced it has been evaluated would mask with the *next*
round's grammar state.  ``masks_for_block`` therefore returns a FRESH
``mx.array`` every round (155 KB at V=154856, W=8) instead of recycling one.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Iterable, List, Optional, Sequence, Set, Tuple

import mlx.core as mx
import numpy as np

logger = logging.getLogger("mlx_vlm.speculative.structured_ledger")

#: D1.  ``1`` is the default as of the LU-panel promotion: text sha identical to
#: the AR+mask reference on 4/4 prompts, 2.2x per-token, and the natural (no
#: response_format) panel unchanged with the rail on.  ``0`` restores the
#: pre-promotion behaviour exactly -- the server's original refusal message and
#: a round loop byte-for-byte the code at base 71732451.  Empty/whitespace counts
#: as unset; an unparseable value warns once per distinct value and falls back to
#: the default.
SPEC_STRUCTURED_ENV = "MLX_VLM_SPEC_STRUCTURED"
SPEC_STRUCTURED_DEFAULT = True

_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off")

# (variable, offending value) pairs already reported.  The round loop resolves
# the toggle per request, so an unguarded warning would print per request.
_WARNED: Set[Tuple[str, str]] = set()


def _reset_env_warnings() -> None:
    """Test hook: forget which invalid values have already been reported."""
    _WARNED.clear()


def _env_value(environ, name: str) -> Optional[str]:
    """An empty or whitespace-only value is the same as not setting it."""
    raw = environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def spec_structured_enabled(env: Optional[dict] = None) -> bool:
    """Is the structured x speculative rail switched on?  Default ON."""
    environ = os.environ if env is None else env
    raw = _env_value(environ, SPEC_STRUCTURED_ENV)
    if raw is None:
        return SPEC_STRUCTURED_DEFAULT
    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    key = (SPEC_STRUCTURED_ENV, raw)
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(
            "Ignoring invalid %s=%r; using %r. Valid values: 0/1.",
            SPEC_STRUCTURED_ENV,
            raw,
            "1" if SPEC_STRUCTURED_DEFAULT else "0",
        )
    return SPEC_STRUCTURED_DEFAULT


class StructuredSpeculationRefused(ValueError):
    """A structured request the speculative rail will not serve (D1/D3/D7/R1)."""


class StructuredLedgerError(RuntimeError):
    """The grammar ledger reached a state it must not silently decode from."""


# --------------------------------------------------------------------------
# bitmask helpers
# --------------------------------------------------------------------------
def bitmask_words(vocab_size: int) -> int:
    return (int(vocab_size) + 31) // 32


def unpack_bitmask(mask: mx.array, vocab_size: int) -> mx.array:
    """(rows, words) int32 bitmask -> (rows, vocab) bool "token is allowed".

    Arithmetic shift on a negative int32 keeps the sign bit, but ``& 1`` reads
    the bit we asked for either way, so an all-ones (vacuous) row unpacks to all
    True as intended.
    """
    if mask.ndim == 1:
        mask = mask[None, :]
    index = mx.arange(int(vocab_size), dtype=mx.int32)
    words = mx.take(mask, index // 32, axis=-1)
    return ((words >> (index % 32)) & 1).astype(mx.bool_)


def apply_block_mask_portable(logits: mx.array, mask: mx.array) -> mx.array:
    """The device-independent mask: one gather + one select over (rows, vocab)."""
    if logits.ndim == 1:
        logits = logits[None, :]
    allowed = unpack_bitmask(mask, int(logits.shape[-1]))
    return mx.where(allowed, logits, mx.array(-float("inf"), dtype=logits.dtype))


def metal_mask_kernel_enabled() -> bool:
    """Should the block mask go through ``structured._apply_llguidance_mask``?

    Only on the GPU: ``mx.fast.metal_kernel`` has no CPU implementation, so a
    CPU rail must take the portable path.  ``MLX_DEFAULT_DEVICE=cpu`` is the
    fleet's CPU-only signal and is honoured here explicitly, because this MLX
    build (0.32.1) does not act on it -- ``mx.default_device()`` still reports
    the GPU -- and a CPU-only session must not be dispatching Metal kernels.

    Read per call, not cached: a test that flips the environment has to see it.
    """
    raw = (os.environ.get("MLX_DEFAULT_DEVICE") or "").strip().lower()
    if raw == "cpu":
        return False
    try:
        if not mx.metal.is_available():
            return False
        return mx.default_device() == mx.Device(mx.DeviceType.gpu, 0)
    except Exception:
        return False


def apply_block_mask(logits: mx.array, mask: mx.array) -> mx.array:
    """Set every grammar-illegal logit to ``-inf``.

    ``logits`` is ``(rows, vocab)`` and ``mask`` is the matching
    ``(rows, words)`` bitmask.  On the GPU this is the same fused Metal kernel
    the single-token processor already uses (``structured._apply_llguidance_mask``,
    one thread per token, no full-vocab bool materialised); on CPU it falls back
    to portable MLX ops.  Either way the result is an ordinary lazily-evaluated
    node, so the R3 fresh-array rule still does the work it has to.
    """
    if logits.ndim == 1:
        logits = logits[None, :]
    if metal_mask_kernel_enabled():
        from ..structured import _apply_llguidance_mask

        return _apply_llguidance_mask(logits, mask)
    return apply_block_mask_portable(logits, mask)


def _pack_rows(rows: Sequence[Optional[Sequence[int]]], vocab_size: int) -> np.ndarray:
    """Build an llguidance-shaped bitmask from explicit legal sets.

    ``None`` means "all allow" (the vacuous row llguidance writes after an
    illegal draft, and the row a ledger emits while thinking is still open).
    """
    words = bitmask_words(vocab_size)
    buffer = np.zeros((len(rows), words), dtype=np.int32)
    for index, legal in enumerate(rows):
        if legal is None:
            buffer[index, :] = -1
            continue
        for token in legal:
            token = int(token)
            if not 0 <= token < vocab_size:
                raise StructuredLedgerError(
                    f"legal token {token} is outside the vocabulary "
                    f"(0..{vocab_size - 1})"
                )
            buffer[index, token >> 5] |= np.int32(1) << np.int32(token & 31)
    return buffer


def _row_popcount(buffer: np.ndarray, index: int) -> int:
    return int(np.unpackbits(buffer[index].view(np.uint8)).sum())


# --------------------------------------------------------------------------
# processor plumbing
# --------------------------------------------------------------------------
def is_grammar_processor(processor: Any) -> bool:
    """Duck-typed ``LLGuidanceLogitsProcessor`` (avoids importing llguidance)."""
    return hasattr(processor, "grammar") and hasattr(processor, "llg_tokenizer")


def _unwrap_thinking(processor: Any):
    """Return ``(inner, thinking_end_token_id_or_None)`` for one processor.

    A ``ThinkingAwareLogitsProcessor`` that has ALREADY activated (its prompt
    closed thinking during prefill) is indistinguishable from a bare grammar
    processor for our purposes, so it reports no end token.
    """
    inner = getattr(processor, "processor", None)
    if inner is None or not hasattr(processor, "thinking_end_token_id"):
        return processor, None
    if getattr(processor, "_active", True):
        return inner, None
    return inner, int(processor.thinking_end_token_id)


def flatten_logits_processors(logits_processors: Any) -> List[Any]:
    """Accept either a flat list or the batch path's list-of-per-row-lists."""
    if not logits_processors:
        return []
    flat: List[Any] = []
    for entry in logits_processors:
        if entry is None:
            continue
        if isinstance(entry, (list, tuple)):
            flat.extend(p for p in entry if p is not None)
        else:
            flat.append(entry)
    return flat


def resolve_structured_processor(
    logits_processors: Any,
    *,
    batch_size: int,
    draft_kind: Optional[str],
    call_site: str,
    env: Optional[dict] = None,
):
    """The single gate every speculative entry point calls.

    Returns the one grammar processor to build a ledger from, or ``None`` when
    there is nothing to thread.

    THE TOGGLE IS CHECKED FIRST, BEFORE THE PROCESSORS ARE EVEN LOOKED AT.
    With ``MLX_VLM_SPEC_STRUCTURED=0`` this returns ``None`` unconditionally and
    every speculative path is byte-for-byte the code at base 71732451: no ledger
    is constructed, nothing is inspected, and nothing new can raise.  R1's silent
    partial constraint is still there in that configuration, and deliberately so
    -- ``0`` is the escape hatch back to the pre-promotion server.

    With the rail on (the default), exactly one grammar processor is threaded and
    every other STRUCTURED shape refuses by name: a non-grammar processor
    alongside the grammar, more than one grammar, MTP, a non-DFlash drafter,
    B > 1.

    A request with NO grammar processor is not a structured request, so it is
    none of this feature's business: penalty-only processors return ``None`` and
    keep whatever the speculative path did with them before.  That carve-out is
    load-bearing now that the default is on -- without it every
    ``repetition_penalty`` request served with a draft model would start failing,
    which is not what the panel promoted.
    """
    # D1: the off path must not differ from base in ANY observable way, so the
    # environment decides before anything else happens.
    if not spec_structured_enabled(env):
        return None

    processors = flatten_logits_processors(logits_processors)
    if not processors:
        return None

    grammar = [p for p in processors if is_grammar_processor(_unwrap_thinking(p)[0])]
    if not grammar:
        return None

    extra = [p for p in processors if p not in grammar]
    if extra:
        raise StructuredSpeculationRefused(
            f"{call_site}: the structured speculative rail threads a grammar "
            "processor through the round loop, but this request also carries "
            f"{len(extra)} non-grammar logits processor(s) "
            f"({', '.join(sorted({type(p).__name__ for p in extra}))}); those "
            "have no block interface and would be silently dropped. Drop the "
            "repetition/presence/frequency/logit-bias penalties or the draft "
            f"model, or set {SPEC_STRUCTURED_ENV}=0 to refuse structured "
            "requests outright."
        )
    if len(grammar) != 1:
        raise StructuredSpeculationRefused(
            f"{call_site}: the structured speculative rail supports exactly one "
            f"grammar processor per sequence; got {len(grammar)}."
        )

    # D7: MTP's fast path emits tokens without target logits at every position,
    # so there is nothing to mask.  Named separately from the generic refusal
    # because it is the one people will hit.
    if draft_kind == "mtp":
        raise StructuredSpeculationRefused(
            f"{call_site}: structured response_format is not supported with MTP "
            "speculative decoding (the structured rail covers DFlash only in v1). "
            "MTP's draft fast path produces tokens without per-position target "
            "logits, so the grammar mask has nothing to apply to."
        )
    if draft_kind != "dflash":
        raise StructuredSpeculationRefused(
            f"{call_site}: the structured speculative rail supports "
            f"draft_kind='dflash' only in v1; got {draft_kind!r}."
        )

    # D3: the block ledger, the fast-forward draft and the position-0 drafter
    # mask are all written for one sequence.  Refuse loudly rather than
    # constrain row 0 and leave the rest of the batch free.
    if int(batch_size) != 1:
        raise StructuredSpeculationRefused(
            f"{call_site}: the structured speculative rail supports batch size 1 "
            f"only in v1; got batch_size={int(batch_size)}. Serve the structured "
            f"request on its own batch, or set {SPEC_STRUCTURED_ENV}=0."
        )

    return grammar[0]


# --------------------------------------------------------------------------
# ledgers
# --------------------------------------------------------------------------
class _LedgerBase:
    """Shared thinking gate, block geometry and commit bookkeeping."""

    def __init__(
        self,
        vocab_size: int,
        *,
        thinking_end_token_id: Optional[int] = None,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.thinking_end_token_id = (
            None if thinking_end_token_id is None else int(thinking_end_token_id)
        )
        #: The grammar is live once thinking has ended (or was never open).
        self.active = self.thinking_end_token_id is None
        self.stopped = False
        #: Predecessor of block position 0 -- the last committed token.
        self.last_token: Optional[int] = None
        #: How many tokens the ledger has fed to the grammar since activation.
        self.consumed = 0

    # -- thinking gate ------------------------------------------------------
    def _grammar_start_row(self, drafts: Sequence[int], rows: int) -> Optional[int]:
        """First row of this block the grammar constrains, or None.

        R9.  Row ``i``'s predecessor is ``last_token`` for ``i == 0`` and
        ``drafts[i-1]`` otherwise; the grammar starts at the row whose
        predecessor IS the thinking end token, and that end token is not fed to
        the matcher.  The rows before it stay all-allow.
        """
        if self.active:
            return 0
        predecessors = [self.last_token] + list(drafts)
        for row in range(rows):
            if row >= len(predecessors):
                break
            if predecessors[row] == self.thinking_end_token_id:
                return row
        return None

    # -- public API ---------------------------------------------------------
    def next_token_mask(self) -> mx.array:
        """The (1, words) mask for the very next token: ``masks_for_block([])``."""
        return self.masks_for_block([])

    def masks_for_block(self, drafts: Sequence[int]) -> mx.array:
        drafts = [int(token) for token in drafts]
        rows = len(drafts) + 1
        start = self._grammar_start_row(drafts, rows)
        if start is None:
            buffer = _pack_rows([None] * rows, self.vocab_size)
            return mx.array(buffer)
        buffer = self._fill(drafts, rows, start)
        for row in range(start, rows):
            if _row_popcount(buffer, row) == 0:
                raise StructuredLedgerError(
                    "the grammar allows no token at block position "
                    f"{row} (empty legal set). Decoding from an all -inf logits "
                    "row would emit a silently wrong token."
                )
        # R3: a fresh array every round, never a recycled shared buffer.
        return mx.array(buffer)

    def commit(self, new_tokens: Sequence[int]) -> None:
        tokens = [int(token) for token in new_tokens]
        if not tokens:
            return
        if not self.active:
            for index, token in enumerate(tokens):
                if token == self.thinking_end_token_id:
                    self.active = True
                    # R9: the end token itself is NOT consumed by the grammar;
                    # the token after it is the first constrained one.
                    self._consume(tokens[index + 1 :])
                    break
        else:
            self._consume(tokens)
        self.last_token = tokens[-1]

    def forced_tokens(self, limit: int) -> List[int]:
        if not self.active or self.stopped or int(limit) <= 0:
            return []
        return self._forced_tokens(int(limit))

    # -- subclass hooks -----------------------------------------------------
    def _fill(self, drafts: List[int], rows: int, start: int) -> np.ndarray:
        raise NotImplementedError

    def _consume(self, tokens: Sequence[int]) -> None:
        raise NotImplementedError

    def _forced_tokens(self, limit: int) -> List[int]:
        raise NotImplementedError


_EXECUTOR = None


def _shared_executor():
    """One process-wide ``LLExecutor``; it is a thread pool, not request state."""
    global _EXECUTOR
    if _EXECUTOR is None:
        from llguidance import LLExecutor

        _EXECUTOR = LLExecutor()
    return _EXECUTOR


class StructuredLedger(_LedgerBase):
    """The real ledger: one ``LLMatcher`` + the shared ``LLExecutor``."""

    def __init__(
        self,
        matcher,
        *,
        vocab_size: int,
        executor=None,
        thinking_end_token_id: Optional[int] = None,
    ) -> None:
        super().__init__(vocab_size, thinking_end_token_id=thinking_end_token_id)
        self.matcher = matcher
        self.executor = executor if executor is not None else _shared_executor()

    @classmethod
    def from_processor(cls, processor) -> "StructuredLedger":
        """Build a ledger from the request's logits processor.

        The matcher is created fresh rather than borrowed: on the speculative
        path the processor has been set up during prefill (it masked the first
        bonus token) but has consumed nothing, which is exactly a fresh
        matcher's state.  ``LLGuidanceLogitsProcessor.clone()`` is documented
        NOT to preserve state (``structured.py:71``), so borrowing would be a
        trap for a future caller anyway.
        """
        from llguidance import LLMatcher

        inner, thinking_end_token_id = _unwrap_thinking(processor)
        if not is_grammar_processor(inner):
            raise StructuredSpeculationRefused(
                f"{type(processor).__name__} is not an llguidance grammar "
                "processor; the speculative ledger has nothing to constrain."
            )
        matcher = LLMatcher(inner.llg_tokenizer, inner.grammar)
        error = matcher.get_error()
        if error:
            raise StructuredLedgerError(f"LLGuidance matcher error: {error}")
        return cls(
            matcher,
            vocab_size=int(inner.llg_tokenizer.vocab_size),
            thinking_end_token_id=thinking_end_token_id,
        )

    def _fill(self, drafts: List[int], rows: int, start: int) -> np.ndarray:
        import llguidance.numpy as llnp

        buffer = llnp.allocate_token_bitmask(rows, self.vocab_size)
        buffer = np.asarray(buffer, dtype=np.int32)
        if start:
            # positions still inside the thinking span: all-allow
            buffer[:start, :] = -1
        remaining = list(drafts[start:])
        if remaining:
            llnp.fill_next_token_bitmask_par_with_draft_tokens(
                self.executor,
                [(self.matcher, start, remaining)],
                buffer,
            )
        else:
            # The draft-path fill refuses an empty draft list; a single row is
            # just the matcher's own next-token mask.  This is the Tier B
            # ``next_token_mask()`` call and the block that activates on its
            # last row.
            llnp.fill_next_token_bitmask(self.matcher, buffer, start)
        self._check_error("filling the block bitmask")
        return buffer

    def _consume(self, tokens: Sequence[int]) -> None:
        for token in tokens:
            if self.stopped:
                return
            self.matcher.consume_token(int(token))
            self._check_error(f"consuming token {int(token)}")
            self.consumed += 1
            if self.matcher.is_stopped():
                self.stopped = True

    def _forced_tokens(self, limit: int) -> List[int]:
        forced = list(self.matcher.compute_ff_tokens())
        self._check_error("computing fast-forward tokens")
        return [int(token) for token in forced[:limit]]

    def _check_error(self, what: str) -> None:
        error = self.matcher.get_error()
        if error:
            raise StructuredLedgerError(
                f"LLGuidance matcher error while {what}: {error}"
            )


class StubLedger(_LedgerBase):
    """CPU-testable ledger driven by a scripted legal-set schedule.

    ``legal(prefix)`` is called with the tuple of tokens the grammar has
    consumed so far (the drafted path included) and returns the legal set at
    that point, or ``None`` for "anything goes".  An empty set is a grammar dead
    end and raises, exactly as a zero popcount does on the real ledger.
    """

    def __init__(
        self,
        vocab_size: int,
        legal: Callable[[Tuple[int, ...]], Optional[Iterable[int]]],
        *,
        thinking_end_token_id: Optional[int] = None,
    ) -> None:
        super().__init__(vocab_size, thinking_end_token_id=thinking_end_token_id)
        self._legal = legal
        self.history: List[int] = []

    def legal_at(self, prefix: Sequence[int]) -> Optional[List[int]]:
        legal = self._legal(tuple(int(t) for t in prefix))
        return None if legal is None else [int(token) for token in legal]

    def _fill(self, drafts: List[int], rows: int, start: int) -> np.ndarray:
        packed: List[Optional[List[int]]] = [None] * start
        path = list(self.history)
        vacuous = False
        for row in range(start, rows):
            if vacuous:
                # llguidance writes all-ones rows once the draft path left the
                # grammar; mirror that so acceptance stops at the first illegal
                # draft rather than here.
                packed.append(None)
                continue
            legal = self.legal_at(path)
            packed.append(legal)
            draft_index = row  # drafts[row] is the drafted token AT position row
            if draft_index < len(drafts):
                token = drafts[draft_index]
                if legal is not None and token not in legal:
                    vacuous = True
                else:
                    path.append(token)
        return _pack_rows(packed, self.vocab_size)

    def _consume(self, tokens: Sequence[int]) -> None:
        for token in tokens:
            legal = self.legal_at(self.history)
            if legal is not None and int(token) not in legal:
                raise StructuredLedgerError(
                    f"committed token {int(token)} is not in the legal set "
                    f"{sorted(legal)} after prefix {tuple(self.history)}"
                )
            self.history.append(int(token))
            self.consumed += 1

    def _forced_tokens(self, limit: int) -> List[int]:
        forced: List[int] = []
        path = list(self.history)
        while len(forced) < limit:
            legal = self.legal_at(path)
            if legal is None or len(legal) != 1:
                break
            forced.append(legal[0])
            path.append(legal[0])
        return forced


__all__ = [
    "SPEC_STRUCTURED_DEFAULT",
    "SPEC_STRUCTURED_ENV",
    "StructuredLedger",
    "StructuredLedgerError",
    "StructuredSpeculationRefused",
    "StubLedger",
    "apply_block_mask",
    "bitmask_words",
    "flatten_logits_processors",
    "is_grammar_processor",
    "resolve_structured_processor",
    "spec_structured_enabled",
    "unpack_bitmask",
]
