"""Prompt-lookup speculative round loop.

Structurally the DFlash loop minus the drafter forward: propose -> verify ->
walk -> rollback.  Two things differ and both matter.

*No target hidden.*  The drafter reads token ids, so the verify forward passes
``capture_layer_ids=[]`` rather than the drafter's layer list.  The empty list is
load-bearing, not cosmetic: ``Glm5NextForCausalLM.__call__`` allocates the
``gdn_sink`` that ``rollback_speculative_cache`` replays only when
``capture_layer_ids is not None``, so passing ``None`` would silently break
rollback on rejection.  An empty list keeps the KDA states and skips the
per-layer hidden capture.

*Abstention is a first-class outcome.*  When no n-gram matches, the drafter
returns a [1, 0] proposal and the round verifies a single token -- exactly a
plain decode step, with no rollback (``accepted == n_draft == 0``).  The DFlash
loop cannot do this: it breaks out of generation on ``bs <= 1``.  Here the loop
just keeps going, which is what makes a lookup drafter safe to leave on for a
workload that turns out not to quote.
"""

from typing import Any, Callable, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .common import (
    _record_speculative_round,
    _speculative_walk,
    generation_stream,
)


def _lookup_rounds(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    *,
    prompt_tokens: Optional[mx.array],
    first_bonus: int,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    greedy_sampling: bool = True,
) -> Generator[Tuple[int, None], None, None]:
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache. "
            "This target does not currently support speculative decoding."
        )
    if not greedy_sampling:
        raise ValueError(
            "prompt-lookup drafting is greedy-only: it has no draft distribution "
            "to correct against, so rejection sampling cannot be made exact. "
            "Set temperature=0."
        )

    block_total = int(
        draft_block_size
        if draft_block_size is not None
        else draft_model.config.block_size
    )

    draft_model.reset(model)
    if prompt_tokens is not None:
        ids = prompt_tokens.reshape(-1).tolist()
        draft_model.set_context([int(t) for t in ids])
    draft_model.observe([int(first_bonus)])

    b = int(first_bonus)
    emitted = 1  # the first bonus was already yielded by the caller

    while emitted < max_tokens:
        # Never propose past the caller's budget: a token drafted beyond
        # max_tokens is verify width spent on output that is thrown away.
        room = max_tokens - emitted
        bs = min(block_total, room + 1)
        draft_tokens = draft_model.draft_block(
            b, None, None, bs, None, token_dtype
        )
        n_draft = int(draft_tokens.shape[1])

        with mx.stream(generation_stream):
            verify_input = mx.array([[b]], dtype=token_dtype)
            if n_draft:
                verify_input = mx.concatenate([verify_input, draft_tokens], axis=1)
            # capture_layer_ids=[] -- empty, not None: it is what allocates the
            # gdn states rollback needs, while skipping the hidden capture.
            verify_out = lm(
                verify_input,
                cache=prompt_cache,
                capture_layer_ids=[],
                speculative_verify=True,
            )
            target_tokens = sampler(verify_out.logits)
        mx.async_eval(target_tokens)

        accepted, new_tokens = _speculative_walk(draft_tokens, target_tokens, room)
        _record_speculative_round(draft_model, accepted, n_draft)
        note = getattr(draft_model, "note_round", None)
        if callable(note):
            note(accepted)

        if accepted < n_draft:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache, verify_out.gdn_states, accepted, n_draft + 1
                )

        if not new_tokens:
            break
        draft_model.observe(new_tokens)
        b = new_tokens[-1]

        for tok in new_tokens:
            yield tok, None
            emitted += 1
            if emitted >= max_tokens:
                return

        verify_out = None


def _lookup_rounds_batch(*args, **kwargs):
    raise NotImplementedError(
        "prompt-lookup drafting is single-sequence for now: each row needs its "
        "own n-gram index and its own proposal length, so a batched round would "
        "have to pad every row to the widest proposal and then discard the "
        "padding -- which is the cost the drafter exists to avoid. Use "
        "draft_kind='dflash' for batched speculative decoding."
    )


__all__ = ["_lookup_rounds", "_lookup_rounds_batch"]
