from typing import Any, Callable, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

generation_stream = mx.new_thread_local_stream(mx.default_device())


def _copy_rng_state() -> List[mx.array]:
    return [mx.array(state) for state in mx.random.state]


def _restore_rng_state(state: List[mx.array]) -> None:
    for i, value in enumerate(state):
        mx.random.state[i][:] = value


def _append_arrays(value: Any, arrays: List[mx.array]) -> None:
    if isinstance(value, mx.array):
        arrays.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _append_arrays(item, arrays)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _append_arrays(item, arrays)


def _draft_sampler_state_arrays(draft_model: nn.Module) -> List[mx.array]:
    state_fn = getattr(draft_model, "draft_eval_state", None)
    if callable(state_fn):
        arrays: List[mx.array] = []
        _append_arrays(state_fn(), arrays)
        return arrays

    attrs = getattr(draft_model, "sampler_state_attrs", ("_seed_token",))
    if isinstance(attrs, str):
        attrs = (attrs,)

    arrays = []
    for attr in attrs:
        _append_arrays(getattr(draft_model, attr, None), arrays)
    return arrays


class _SpeculativeSamplerRNG:
    """Keep target and drafter sampler RNG streams independent."""

    def __init__(self, draft_model: nn.Module, *, enabled: bool):
        self.draft_model = draft_model
        self.enabled = bool(enabled)
        self._target_rng_state = _copy_rng_state() if self.enabled else None
        self._draft_rng_state = _copy_rng_state() if self.enabled else None

    def draft_call(
        self,
        fn: Callable,
        *args,
        **kwargs,
    ):
        if not self.enabled:
            result = fn(*args, **kwargs)
            arrays = []
            _append_arrays(result, arrays)
            arrays.extend(_draft_sampler_state_arrays(self.draft_model))
            if arrays:
                mx.async_eval(*arrays)
            return result

        self._target_rng_state = _copy_rng_state()
        _restore_rng_state(self._draft_rng_state)
        result = fn(*args, **kwargs)

        arrays = _draft_sampler_state_arrays(self.draft_model)
        arrays.extend(mx.random.state)
        if arrays:
            mx.async_eval(*arrays)

        self._draft_rng_state = _copy_rng_state()
        _restore_rng_state(self._target_rng_state)
        return result

    def draft_tokens(self, fn: Callable, *args, **kwargs):
        if not self.enabled:
            result = fn(*args, **kwargs)
            arrays: List[mx.array] = []
            _append_arrays(result, arrays)
            if arrays:
                mx.async_eval(*arrays)
            return result

        self._target_rng_state = _copy_rng_state()
        _restore_rng_state(self._draft_rng_state)
        result = fn(*args, **kwargs)

        arrays = []
        _append_arrays(result, arrays)
        arrays.extend(_draft_sampler_state_arrays(self.draft_model))
        arrays.extend(mx.random.state)
        if arrays:
            mx.async_eval(*arrays)

        self._draft_rng_state = _copy_rng_state()
        _restore_rng_state(self._target_rng_state)
        return result

    def target_sampled(self, *, sync_draft: bool = False) -> None:
        if self.enabled:
            self._target_rng_state = _copy_rng_state()
            if sync_draft:
                self._draft_rng_state = _copy_rng_state()

    def sync_draft_to_target(self) -> None:
        if self.enabled:
            self._draft_rng_state = _copy_rng_state()

    def target_eval(self, *values: Any) -> None:
        if not self.enabled:
            arrays: List[mx.array] = []
            for value in values:
                _append_arrays(value, arrays)
            if arrays:
                mx.async_eval(*arrays)
            return

        arrays = []
        for value in values:
            _append_arrays(value, arrays)
        arrays.extend(mx.random.state)
        if arrays:
            mx.async_eval(*arrays)
        self._target_rng_state = _copy_rng_state()


def _speculative_walk(
    draft_tokens: mx.array,
    target_tokens: mx.array,
    budget: int,
) -> Tuple[int, List[int]]:
    """Exact-greedy speculative-decoding walk.

    Accept drafted tokens up to the first mismatch with the target's
    greedy choice, then take the target's bonus at that position.
    Returns ``(accepted_count, new_tokens)`` with ``new_tokens``
    truncated to ``budget``.
    """
    n_draft = int(draft_tokens.shape[1])
    draft_row = draft_tokens.reshape(-1).tolist()[:n_draft]
    target_row = target_tokens.reshape(-1).tolist()

    accepted = n_draft
    for i, (draft_tok, target_tok) in enumerate(zip(draft_row, target_row)):
        if draft_tok != target_tok:
            accepted = i
            break

    new_tokens = draft_row[:accepted] + target_row[accepted : accepted + 1]
    return accepted, new_tokens[:budget]


def _speculative_walk_batch(
    draft_tokens: mx.array,
    target_tokens: mx.array,
    budgets: List[int],
) -> Tuple[List[int], List[List[int]]]:
    """Per-sequence speculative walk for B > 1.

    Returns ``(accepted_list, new_tokens_list)`` where each entry
    corresponds to one sequence in the batch.
    """
    B = int(draft_tokens.shape[0])
    n_draft = int(draft_tokens.shape[1])
    draft_rows = draft_tokens.tolist()
    target_rows = target_tokens.tolist()
    accepted_list = []
    new_tokens_list: List[List[int]] = []
    for i in range(B):
        accepted = n_draft
        for j, (draft_tok, target_tok) in enumerate(
            zip(draft_rows[i][:n_draft], target_rows[i])
        ):
            if draft_tok != target_tok:
                accepted = j
                break
        accepted_list.append(accepted)
        new_tokens = draft_rows[i][:accepted] + target_rows[i][accepted : accepted + 1]
        new_tokens_list.append(new_tokens[: budgets[i]])
    return accepted_list, new_tokens_list


def _speculative_walk_batch_uniform_acceptance(
    draft_tokens: mx.array,
    target_tokens: mx.array,
    accepted_list: List[int],
    budgets: List[int],
) -> Tuple[List[int], List[List[int]]]:
    """Clamp a batch to the earliest rejection with verifier-token fallback."""
    accepted = min(accepted_list)
    new_tokens_list: List[List[int]] = []
    for i, budget in enumerate(budgets):
        accepted_prefix = draft_tokens[i : i + 1, :accepted]
        bonus = target_tokens[i : i + 1, accepted : accepted + 1]
        new = (
            mx.concatenate([accepted_prefix, bonus], axis=1)[:, :budget]
            .reshape(-1)
            .tolist()
        )
        new_tokens_list.append(new)
    return [accepted] * len(accepted_list), new_tokens_list


def _requires_uniform_batch_acceptance(
    draft_model: nn.Module, target_model: Optional[nn.Module] = None
) -> bool:
    """Whether ragged per-row acceptance must be clamped to a uniform count.

    The flag is honored on either the drafter or the target model. A target's
    dedicated drafter (e.g. glm5_next -> glm5_next_dflash2) may not advertise
    it, yet the target's own ``rollback_speculative_cache`` may be unable to
    represent ragged accepts: glm5_next trims one shared KV length and replays
    one shared KDA prefix, so a row that accepted fewer tokens than the batch
    maximum keeps the real KV of tokens it rejected (issue #1962 upstream, and
    our own variant of it). The target therefore may require uniformity
    independently of the drafter.

    Named after upstream Blaizzy/mlx-vlm 3b8f727d so a later rebase is a
    no-conflict fast-forward.
    """
    if getattr(draft_model, "requires_uniform_batch_acceptance", False):
        return True
    return bool(getattr(target_model, "requires_uniform_batch_acceptance", False))


def _record_uniform_clamp(draft_model: nn.Module, clamped_tokens: int) -> None:
    """Record speculative tokens thrown away by the uniform-acceptance clamp.

    The clamp buys correctness with throughput: every row that accepted more
    than the batch minimum re-drafts the difference next round. ``accept_lens``
    already shows the *post*-clamp acceptance (``_record_speculative_round``
    runs after the clamp), so the mean-accept receipt reflects the cost; this
    counter makes the cost attributable rather than merely implied.
    """
    clamped_tokens = int(clamped_tokens)
    if clamped_tokens <= 0:
        return
    # Per-request (zeroed alongside ``accept_lens`` by _reset_uniform_clamp)
    # and monotonic lifetime, matching the two scopes the other speculative
    # counters already keep.
    draft_model.clamped_tokens = (
        getattr(draft_model, "clamped_tokens", 0) + clamped_tokens
    )
    draft_model.speculative_total_clamped = (
        getattr(draft_model, "speculative_total_clamped", 0) + clamped_tokens
    )


def _reset_uniform_clamp(draft_model: nn.Module) -> None:
    """Zero the per-request clamp counter.

    Called next to ``draft_model.reset()``, which is where ``accept_lens``
    restarts; without it the receipt would divide a lifetime give-back by one
    request's rounds.
    """
    draft_model.clamped_tokens = 0


def _record_speculative_round(
    draft_model: nn.Module, accepted: float, draft_count: int
) -> None:
    draft_model.accept_lens.append(accepted)
    if hasattr(draft_model, "draft_lens"):
        draft_model.draft_lens.append(int(draft_count))
    # Monotonic lifetime counters for per-request attribution. Unlike the
    # ``accept_lens`` history, these survive ``reset()`` between requests,
    # so callers can snapshot-and-diff across a request's lifetime.
    draft_model.speculative_total_rounds = (
        getattr(draft_model, "speculative_total_rounds", 0) + 1
    )
    draft_model.speculative_total_accepted = getattr(
        draft_model, "speculative_total_accepted", 0.0
    ) + float(accepted)
    draft_model.speculative_total_drafted = getattr(
        draft_model, "speculative_total_drafted", 0
    ) + int(draft_count)


def speculative_stats_snapshot(draft_model: nn.Module) -> Tuple[int, float, int]:
    """Capture the drafter's lifetime round counters for later diffing."""
    return (
        getattr(draft_model, "speculative_total_rounds", 0),
        getattr(draft_model, "speculative_total_accepted", 0.0),
        getattr(draft_model, "speculative_total_drafted", 0),
    )


def speculative_stats_since(
    draft_model: nn.Module, snapshot: Tuple[int, float, int]
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (rounds, accepted, drafted) recorded since ``snapshot``.

    Rounds are batch-wide, so the attribution is exact for batch size 1 and
    shared across concurrent requests otherwise.
    """
    rounds0, accepted0, drafted0 = snapshot
    rounds = getattr(draft_model, "speculative_total_rounds", 0) - rounds0
    if rounds <= 0:
        return None, None, None
    accepted = getattr(draft_model, "speculative_total_accepted", 0.0) - accepted0
    drafted = getattr(draft_model, "speculative_total_drafted", 0) - drafted0
    return rounds, int(round(accepted)), int(drafted)


def _dflash_block_total(
    draft_model: nn.Module,
    draft_block_size: Optional[int],
    ignore_runtime: bool = False,
) -> int:
    """Resolve the requested verify block total.

    ``ignore_runtime`` exists for the FIXED width policy, whose contract is to
    propose the checkpoint's TRAINED width.  ``runtime_block_size`` is a runtime
    NARROWING hint -- and for DFlash2 it is not even authored: the config loader
    injects ``min(5, block_size)`` whenever the checkpoint omits it
    (drafters/dflash2/config.py:139-140).  Our checkpoint advertises
    ``block_size: 8`` and no runtime value, so the loader supplies 5 and the
    shipped fixed-8 default silently resolved to 5 on the server path.

    An explicit ``draft_block_size`` (MLX_VLM_DRAFT_BLOCK_SIZE) still wins over
    everything, which is why the override is checked first and why the fixed
    policy cannot simply read config.block_size itself: only here can an
    explicit pin be told apart from an injected narrowing.
    """
    if draft_block_size is not None:
        return int(draft_block_size)

    configured = int(draft_model.config.block_size)
    runtime = getattr(draft_model.config, "runtime_block_size", None)
    if runtime is None or ignore_runtime:
        return configured
    return min(configured, max(1, int(runtime)))


def _batch_cache_left_padding(prompt_cache: List[Any]) -> Optional[mx.array]:
    for cache_entry in prompt_cache:
        left_padding = getattr(cache_entry, "left_padding", None)
        if left_padding is not None:
            return left_padding
    return None


def _format_speculative_stats(draft_model: nn.Module) -> Optional[str]:
    accepted_lens = getattr(draft_model, "accept_lens", None) or []
    if not accepted_lens:
        return None

    rounds = len(accepted_lens)
    accepted_drafts = sum(accepted_lens)
    mean_accept = accepted_drafts / rounds
    mean_accepted_tokens = (accepted_drafts + rounds) / rounds
    draft_lens = getattr(draft_model, "draft_lens", None) or []
    # Tokens a row had already accepted but gave back so the whole batch could
    # roll back to one uniform length. Reported so the correctness clamp's
    # throughput cost is visible in the receipt rather than only in a lower
    # mean accept.
    clamped = int(getattr(draft_model, "clamped_tokens", 0) or 0)
    clamp_note = f", clamped {clamped} tok" if clamped else ""
    if len(draft_lens) == rounds and sum(draft_lens) > 0:
        accept_rate = 100 * accepted_drafts / sum(draft_lens)
        mean_draft = sum(draft_lens) / rounds
        return (
            "Speculative decoding: "
            f"{mean_accepted_tokens:.2f} accepted tokens/round "
            f"({mean_accept:.2f} accepted drafts/round, "
            f"{accept_rate:.1f}% of drafted, "
            f"avg draft {mean_draft:.2f}{clamp_note}) over {rounds} rounds"
        )

    return (
        "Speculative decoding: "
        f"{mean_accepted_tokens:.2f} accepted tokens over {rounds} rounds"
        f"{clamp_note}"
    )
