"""Ragged batched DFlash2 acceptance must be clamped before the rollback.

The batched DFlash2 loop walks each row independently, so rows routinely accept
different numbers of drafted tokens. ``glm5_next``'s
``rollback_speculative_cache`` cannot represent that: it trims ONE shared KV
length (``max_a``), replays ONE shared KDA prefix and rolls the indexer pool to
ONE shared length. Handing it a ragged ``accepted`` therefore leaves every row
that accepted fewer tokens than the batch maximum holding the LIVE KV of tokens
it rejected, while the tokens that row emitted say otherwise.

These tests drive the real ``_dflash_rounds_batch`` loop over a real (tiny)
same-architecture Glm5Next target with a stub drafter, and check the only thing
that matters: after the round, each row's cache is what a fresh forward over
exactly that row's committed tokens produces.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.speculative import dflash as dflash_utils
from mlx_vlm.speculative.common import (
    _record_uniform_clamp,
    _requires_uniform_batch_acceptance,
    _reset_uniform_clamp,
)

# Verify block total: one bonus token + four drafted tokens.
BLOCK_TOTAL = 5
# Per-row drafted tokens. Distinct across rows so a row that keeps another
# row's trim length is visible in the cache, not merely in a counter.
DRAFTS = {0: [11, 12, 13, 14], 1: [21, 22, 23, 24]}
BONUS = [5, 7]
PROMPT = [2, 4, 6, 8]
# The ragged round under test: row 0 accepts 3 of its 4 drafts, row 1 accepts 1.
RAGGED = [3, 1]


def _tiny_glm5_next_target():
    """A 2-layer Glm5Next language model: one KDA layer, one sparse-MLA layer.

    Same shape as the fixture in test_models_dflash2.py, but float32 -- the
    per-row cache comparison below is numerical, and bfloat16 would drown the
    signal it is looking for.
    """
    from mlx_vlm.models import glm5_next
    from mlx_vlm.models.glm5_next.language import LanguageModel

    config = glm5_next.TextConfig(
        model_type="glm5_next_text",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=0,
        v_head_dim=8,
        qk_nope_head_dim=8,
        qk_head_dim=8,
        num_experts_per_tok=4,
        first_k_dense_replace=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        index_topk=6,
        index_head_dim=8,
        index_n_heads=2,
        index_kpool=3,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 2,
            "gate_lower_bound": -5.0,
        },
        hc_mult=4,
        num_nextn_predict_layers=1,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = LanguageModel(config)
    model.set_dtype(mx.float32)
    return model


class _StubDrafter:
    """Emits a fixed per-row draft block. Not a model -- just the surface the
    batched DFlash2 loop actually touches."""

    # Deliberately False: the point of the fix is that the TARGET's
    # requirement is honoured even when the drafter says nothing.
    requires_uniform_batch_acceptance = False
    dflash_deferred_walk = False

    def __init__(self):
        self.config = SimpleNamespace(
            block_size=BLOCK_TOTAL,
            runtime_block_size=BLOCK_TOTAL,
            target_layer_ids=[0],
        )
        self.accept_lens = []
        self.draft_lens = []
        self.calls = 0

    def reset(self, model=None):
        self.accept_lens = []
        self.draft_lens = []

    def make_cache(self):
        return []

    def draft_block(self, bonus, hidden, cache, bs, sampler, token_dtype):
        # The batch loop drafts row by row, in active-slot order.
        row = self.calls % 2
        self.calls += 1
        return mx.array([DRAFTS[row][: bs - 1]], dtype=token_dtype)


class _RollbackDone(Exception):
    """Stops the round loop the instant the rollback has been applied.

    The round's cache state is only observable between the rollback and the
    next round's drafting, and the generator does not yield there.
    """


def _run_one_ragged_round(model, accepted=RAGGED):
    """One real ``_dflash_rounds_batch`` round with ragged per-row acceptance.

    Returns (drafter, emitted_tokens_per_row, accepted_seen_by_rollback).
    """
    target = SimpleNamespace(language_model=model)
    cache = model.make_cache()
    model(mx.array([PROMPT, PROMPT], dtype=mx.int32), cache=cache)
    hidden = mx.zeros((2, 1, model.args.hidden_size), dtype=mx.float32)

    def ragged_walk(draft_tokens, target_tokens, budgets):
        rows = draft_tokens.tolist()
        # Same shape the real walk produces: the accepted draft prefix plus the
        # target's own token at the rejection point.
        return list(accepted), [
            (rows[i][:a] + [90 + i])[: budgets[i]] for i, a in enumerate(accepted)
        ]

    # The acceptance pattern is the thing under test, so it is dictated rather
    # than coaxed out of a random tiny model: everything else in the round --
    # the drafting, the batched verify forward, the emit loop and the rollback
    # itself -- is the real code path.
    original_walk = dflash_utils._speculative_walk_batch
    dflash_utils._speculative_walk_batch = ragged_walk

    seen = {}
    original_rollback = model.rollback_speculative_cache

    def spy(caches, gdn_states, accepted_arg, block_size):
        seen["accepted"] = [int(v) for v in accepted_arg.reshape(-1).tolist()]
        seen["block_size"] = int(block_size)
        original_rollback(caches, gdn_states, accepted_arg, block_size)
        raise _RollbackDone

    model.rollback_speculative_cache = spy
    drafter = _StubDrafter()
    emitted = [[], []]
    try:
        rounds = dflash_utils._dflash_rounds_batch(
            target,
            drafter,
            cache,
            hidden,
            first_bonus=mx.array(BONUS, dtype=mx.int32),
            max_tokens=64,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            greedy_sampling=True,
        )
        try:
            for tokens_out, _ in rounds:
                for i, token in enumerate(tokens_out):
                    if token is not None:
                        emitted[i].append(int(token))
        except _RollbackDone:
            pass
        else:  # pragma: no cover - the round always rolls back here
            pytest.fail("the round never reached rollback_speculative_cache")
    finally:
        model.rollback_speculative_cache = original_rollback
        dflash_utils._speculative_walk_batch = original_walk
    assert "accepted" in seen, "rollback was never called"
    return drafter, emitted, cache, seen


def _reference_cache(model, row, emitted_row):
    """A fresh forward over exactly what this row committed this round.

    ``emitted_row`` is [accepted drafts..., target's own token]; the tokens the
    cache must hold are the row's previous bonus plus those accepted drafts,
    i.e. ``[bonus] + emitted_row[:-1]``.
    """
    cache = model.make_cache()
    model(mx.array([PROMPT], dtype=mx.int32), cache=cache)
    committed = [BONUS[row]] + emitted_row[:-1]
    model(mx.array([committed], dtype=mx.int32), cache=cache)
    return cache, committed


def _row_signature(cache_entry):
    """(offset, tensor) pairs for one row of one layer's cache."""
    from mlx_vlm.models.cache import ArraysCache

    out = []
    if isinstance(cache_entry, ArraysCache):
        for i, arr in enumerate(cache_entry.cache):
            out.append((f"kda[{i}]", None, arr))
        return out
    for i, sub in enumerate(cache_entry.caches):
        keys = None if sub.keys is None else sub.keys[:, :, : sub.offset]
        values = None if sub.values is None else sub.values[:, :, : sub.offset]
        out.append((f"kv[{i}].keys", int(sub.offset), keys))
        out.append((f"kv[{i}].values", int(sub.offset), values))
    return out


def _worst_diff(got, ref):
    if got is None or ref is None:
        return 0.0
    if got.size == 0 and ref.size == 0:
        return 0.0
    assert got.shape == ref.shape, f"shape {got.shape} vs {ref.shape}"
    return float(mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))))


# Batched B=2 reductions and a B=1 reference do not agree bit for bit in
# float32, so the per-row check is a tight tolerance rather than a hash. The
# defect it is looking for is not subtle: with the ragged rollback the short
# row's KDA conv state is off by ~1.6 and its KV offset is off by two whole
# tokens.
_ROW_TOL = 1e-4


def test_dflash_batch_clamps_ragged_accepts_so_cache_matches_emitted_tokens():
    mx.random.seed(3)
    model = _tiny_glm5_next_target()
    mx.eval(model.parameters())

    drafter, emitted, cache, seen = _run_one_ragged_round(model)

    # The invariant: the cache each row is left with is exactly what a fresh
    # forward over that row's own committed tokens produces -- offset first,
    # then the tensors.
    for row in range(2):
        ref_cache, committed = _reference_cache(model, row, emitted[row])
        for layer, (got_entry, ref_entry) in enumerate(zip(cache, ref_cache)):
            got_sig = _row_signature(got_entry.extract(row))
            ref_sig = _row_signature(ref_entry)
            for (name, got_off, got_arr), (_, ref_off, ref_arr) in zip(
                got_sig, ref_sig
            ):
                assert got_off == ref_off, (
                    f"row {row} layer {layer} {name}: cache offset {got_off} "
                    f"but a forward over {committed} lands at {ref_off} -- the "
                    "row is holding the KV of tokens it rejected"
                )
                worst = _worst_diff(got_arr, ref_arr)
                assert worst <= _ROW_TOL, (
                    f"row {row} layer {layer} {name}: max abs diff {worst:.3e} "
                    f"against a fresh forward over {committed} -- the row's "
                    "state was rolled back to another row's accept count"
                )

    # And the two things that made it true: the rollback saw a uniform count,
    # and every row emitted exactly (uniform accepted + 1) tokens.
    assert seen["accepted"] == [min(RAGGED)] * 2, (
        "rollback_speculative_cache received ragged accepts "
        f"{seen['accepted']}; glm5_next trims by max(), so the short row keeps "
        "the KV of tokens it rejected"
    )
    assert [len(e) for e in emitted] == [min(RAGGED) + 1] * 2, emitted


def test_dflash_batch_clamp_is_counted_in_the_receipts():
    mx.random.seed(3)
    model = _tiny_glm5_next_target()
    mx.eval(model.parameters())

    drafter, _, _, _ = _run_one_ragged_round(model)

    # accept_lens records the post-clamp acceptance, so the mean-accept receipt
    # already carries the cost...
    assert drafter.accept_lens == [min(RAGGED)] * 2
    # ...and the give-back is attributable on its own.
    given_back = sum(a - min(RAGGED) for a in RAGGED)
    assert drafter.clamped_tokens == given_back  # per request
    assert drafter.speculative_total_clamped == given_back  # lifetime


def test_uniform_requirement_is_honored_on_either_side():
    neither = SimpleNamespace()
    drafter_only = SimpleNamespace(requires_uniform_batch_acceptance=True)
    target_only = SimpleNamespace(requires_uniform_batch_acceptance=True)

    assert not _requires_uniform_batch_acceptance(neither, neither)
    assert not _requires_uniform_batch_acceptance(neither, None)
    assert _requires_uniform_batch_acceptance(drafter_only, neither)
    assert _requires_uniform_batch_acceptance(neither, target_only)


def test_glm5_next_language_model_declares_the_requirement():
    from mlx_vlm.models.glm5_next.language import LanguageModel

    assert LanguageModel.requires_uniform_batch_acceptance is True


def test_uniform_clamp_counter_is_per_request_and_lifetime():
    drafter = SimpleNamespace()
    _record_uniform_clamp(drafter, 0)
    assert not hasattr(drafter, "speculative_total_clamped")
    _record_uniform_clamp(drafter, 3)
    _record_uniform_clamp(drafter, 2)
    assert drafter.clamped_tokens == 5
    assert drafter.speculative_total_clamped == 5
    # A new request zeroes the per-request half and leaves the lifetime half.
    _reset_uniform_clamp(drafter)
    assert drafter.clamped_tokens == 0
    assert drafter.speculative_total_clamped == 5


def test_glm5_next_rollback_refuses_a_ragged_batch():
    """A wrong precondition must fail loudly, not trim by the batch maximum."""
    mx.random.seed(3)
    model = _tiny_glm5_next_target()
    mx.eval(model.parameters())
    cache = model.make_cache()
    model(mx.array([PROMPT, PROMPT], dtype=mx.int32), cache=cache)

    with pytest.raises(RuntimeError, match="uniform per-row acceptance"):
        model.rollback_speculative_cache(
            cache, [], mx.array(RAGGED, dtype=mx.int32), BLOCK_TOTAL
        )

def test_glm5_next_rollback_still_accepts_a_uniform_batch_and_a_scalar():
    """The guard must not fire on the shapes the fixed path actually sends."""
    mx.random.seed(3)
    model = _tiny_glm5_next_target()
    mx.eval(model.parameters())
    for accepted in (1, [1, 1], mx.array([1, 1], dtype=mx.int32)):
        cache = model.make_cache()
        batch = 1 if isinstance(accepted, int) else 2
        model(mx.array([PROMPT] * batch, dtype=mx.int32), cache=cache)
        # No gdn_states are needed: with a single non-KDA layer touched the
        # guard is the only thing under test here, so drive the KV half only.
        kv_only = [None if i == 0 else c for i, c in enumerate(cache)]
        assert model.rollback_speculative_cache(
            kv_only, [], accepted, BLOCK_TOTAL
        ) == 1


def test_the_clamp_shows_up_in_the_speculative_receipt():
    """The give-back has to be readable off a run, not inferred from it."""
    from mlx_vlm.speculative.common import _format_speculative_stats

    lens = dict(accept_lens=[1, 0, 2], draft_lens=[2, 1, 2])
    assert _format_speculative_stats(SimpleNamespace(**lens)) == (
        "Speculative decoding: 2.00 accepted tokens/round "
        "(1.00 accepted drafts/round, 60.0% of drafted, "
        "avg draft 1.67) over 3 rounds"
    )
    assert _format_speculative_stats(SimpleNamespace(clamped_tokens=4, **lens)) == (
        "Speculative decoding: 2.00 accepted tokens/round "
        "(1.00 accepted drafts/round, 60.0% of drafted, "
        "avg draft 1.67, clamped 4 tok) over 3 rounds"
    )
