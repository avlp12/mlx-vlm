"""One batched draft call for the whole batch, instead of B sequential ones.

``_dflash_rounds_batch`` verifies B rows in ONE target forward and then drafts
them in B separate ``draft_block`` calls.  That asymmetry was a property of the
drafter's CACHE, not of its arithmetic: a scalar ``offset`` cannot hold B
different context lengths, and per-row rollback makes the per-row context
lengths differ every ragged round.

``BatchDFlashKVCache`` gives the drafter the representation the target's batch
caches already use -- per-row offsets, per-row left padding, all padding
contiguous at the left -- plus the one thing a drafter needs on top: a per-round
count of how many rows of the incoming block are REAL.

The contract these tests hold the batched path to is discrete, not statistical:
for the same inputs it must propose the SAME TOKEN IDS, per row, as the row-wise
path.  Draft tokens are integers, so a float difference that never flips an
argmax is not a behaviour change and a float difference that does flip one is a
failure.  On CPU float32 the sweeps below are exact.
"""

import os
import random
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.speculative import dflash as dflash_utils
from mlx_vlm.speculative.drafters.dflash2 import DFlash2DraftModel, ModelConfig
from mlx_vlm.speculative.drafters.qwen3_dflash.batched_cache import BatchDFlashKVCache

HIDDEN = 16
VOCAB = 32
TARGET_LAYER_IDS = [0, 1]
NUM_TARGET_LAYERS = 4
BLOCK_TOTAL = 8


# --------------------------------------------------------------------------
# fixtures: a real (tiny) DFlash2 drafter, sliding or full attention
# --------------------------------------------------------------------------
def _drafter_config(sliding_window, layers=2):
    return ModelConfig.from_dict(
        {
            "architectures": ["DFlash2DraftModel"],
            "model_type": "qwen3",
            "is_causal": False,
            "hidden_size": HIDDEN,
            "intermediate_size": 32,
            "num_hidden_layers": layers,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "vocab_size": VOCAB,
            "max_position_embeddings": 4096,
            "num_target_layers": NUM_TARGET_LAYERS,
            "layer_types": [
                "sliding_attention" if sliding_window else "full_attention"
            ]
            * layers,
            "sliding_window": sliding_window,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000},
            "dflash_config": {
                "block_size": BLOCK_TOTAL,
                "runtime_block_size": BLOCK_TOTAL,
                "conv_group_size": 4,
                "conv_kernel_size": 2,
                "mask_token_id": VOCAB - 1,
                "selector_rank": 4,
                "selector_top_k": 4,
                "target_layer_ids": TARGET_LAYER_IDS,
            },
        }
    )


class _EmbedOnlyTarget(nn.Module):
    """The only surface ``DFlashDraftModel.bind`` and the validator touch."""

    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(
            embed_tokens=nn.Embedding(VOCAB, HIDDEN),
            layers=[None] * NUM_TARGET_LAYERS,
        )
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.config = SimpleNamespace(
            hidden_size=HIDDEN,
            vocab_size=VOCAB,
            num_hidden_layers=NUM_TARGET_LAYERS,
        )

    def rollback_speculative_cache(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def _drafter(sliding_window, seed=0):
    mx.random.seed(seed)
    drafter = DFlash2DraftModel(_drafter_config(sliding_window))
    drafter.bind(_EmbedOnlyTarget())
    mx.eval(drafter.parameters())
    return drafter


def _greedy(logits):
    return mx.argmax(logits, axis=-1)


def _pad_block(segments):
    width = max(int(s.shape[1]) for s in segments)
    padded = [
        (
            s
            if int(s.shape[1]) == width
            else mx.concatenate(
                [s, mx.zeros((s.shape[0], width - s.shape[1], s.shape[2]), s.dtype)],
                axis=1,
            )
        )
        for s in segments
    ]
    return mx.concatenate(padded, axis=0)


def _drive_draft_rounds(drafter, contexts, bonus, block_size=BLOCK_TOTAL):
    """Draft ``len(contexts)`` rounds row-wise and batched; return both.

    ``contexts[r][j]`` is row ``j``'s committed target hidden for round ``r``.
    """
    batch = len(contexts[0])
    row_caches = [drafter.make_cache() for _ in range(batch)]
    row_wise = []
    for round_index, per_row in enumerate(contexts):
        row_wise.append(
            mx.concatenate(
                [
                    drafter.draft_block(
                        int(bonus[round_index][j]),
                        per_row[j],
                        row_caches[j],
                        block_size,
                        _greedy,
                        mx.int32,
                    )
                    for j in range(batch)
                ],
                axis=0,
            )
        )

    batched_caches = drafter.make_batched_cache(batch)
    batched = []
    for round_index, per_row in enumerate(contexts):
        lengths = [int(s.shape[1]) for s in per_row]
        for cache_entry in batched_caches:
            cache_entry.set_pending_lengths(lengths)
        batched.append(
            drafter.draft_block(
                mx.array(bonus[round_index], dtype=mx.int32),
                _pad_block(per_row),
                batched_caches,
                block_size,
                _greedy,
                mx.int32,
            )
        )
    return row_wise, batched, batched_caches


def _random_contexts(batch, rounds, prompt_len, feature_dim, rng, ragged=True):
    contexts = [[mx.random.normal((1, prompt_len, feature_dim)) for _ in range(batch)]]
    for _ in range(rounds):
        contexts.append(
            [
                mx.random.normal(
                    (1, rng.randint(1, BLOCK_TOTAL) if ragged else 3, feature_dim)
                )
                for _ in range(batch)
            ]
        )
    mx.eval([segment for row in contexts for segment in row])
    return contexts


# --------------------------------------------------------------------------
# the identity gate
# --------------------------------------------------------------------------
@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize(
    "sliding_window",
    [None, 64, 512],
    ids=["full-attention", "window-64", "window-512"],
)
def test_batched_draft_proposes_the_same_tokens_on_ragged_contexts(
    batch, sliding_window
):
    """The gate the default rides on: same token ids, per row, ragged contexts.

    ``window-64`` is the interesting one -- the context outgrows the resident
    window inside the run, so the sliding trim fires on both paths, and the
    batched path has to evict exactly the rows each scalar cache would have.
    """
    rng = random.Random(11 + batch)
    drafter = _drafter(sliding_window, seed=batch)
    feature_dim = len(TARGET_LAYER_IDS) * HIDDEN
    contexts = _random_contexts(batch, 10, 17, feature_dim, rng)
    bonus = [[(3 * j + 5 * r) % VOCAB for j in range(batch)] for r in range(11)]

    row_wise, batched, _ = _drive_draft_rounds(drafter, contexts, bonus)
    for round_index, (expected, got) in enumerate(zip(row_wise, batched)):
        assert expected.tolist() == got.tolist(), (
            f"round {round_index} diverged with window {sliding_window} at B={batch}"
        )


def test_batched_draft_is_identical_when_no_row_is_ragged():
    """With equal per-row lengths nothing is padded, so nothing is masked.

    ``context_mask`` returns None there, which means the batched path hands
    ``scaled_dot_product_attention`` exactly the argument the row-wise path
    hands it. This is the cheap half of the contract and it must never regress.
    """
    rng = random.Random(3)
    drafter = _drafter(64, seed=5)
    feature_dim = len(TARGET_LAYER_IDS) * HIDDEN
    contexts = _random_contexts(3, 8, 12, feature_dim, rng, ragged=False)
    bonus = [[(7 * j + r) % VOCAB for j in range(3)] for r in range(9)]
    row_wise, batched, caches = _drive_draft_rounds(drafter, contexts, bonus)
    assert all(pad == 0 for pad in caches[0].left_padding)
    assert caches[0].context_mask(4) is None
    for expected, got in zip(row_wise, batched):
        assert expected.tolist() == got.tolist()


def test_a_prompt_longer_than_the_window_pretruncates_identically():
    """Round 1 hands the drafter the whole prompt; both paths drop the same rows.

    ``_pretruncate_ctx`` (the ``fc`` pre-truncation contract) drops the prefix
    AND advances the offset. On the batched cache it also has to move the
    per-row real-row count, or the proposal's RoPE offset would sit past the
    context it actually attends.
    """
    rng = random.Random(19)
    drafter = _drafter(16, seed=2)
    feature_dim = len(TARGET_LAYER_IDS) * HIDDEN
    contexts = _random_contexts(2, 6, 40, feature_dim, rng)
    bonus = [[(5 * j + 2 * r) % VOCAB for j in range(2)] for r in range(7)]
    row_wise, batched, _ = _drive_draft_rounds(drafter, contexts, bonus)
    for expected, got in zip(row_wise, batched):
        assert expected.tolist() == got.tolist()


def test_a_ragged_block_wider_than_the_window_is_refused_loudly():
    """A per-row front discard is not representable here, so it raises.

    The resident window is 7 rows and the round block is 8, so the sliding
    discard would have to drop a DIFFERENT prefix from each row. Silently
    dropping a uniform one would give the rows a context neither path agrees on.
    """
    drafter = _drafter(8, seed=1)
    feature_dim = len(TARGET_LAYER_IDS) * HIDDEN
    caches = drafter.make_batched_cache(2)
    for cache_entry in caches:
        cache_entry.set_pending_lengths([8, 3])
    with pytest.raises(RuntimeError, match="RAGGED block"):
        drafter.draft_block(
            mx.array([1, 2], dtype=mx.int32),
            mx.random.normal((2, 8, feature_dim)),
            caches,
            BLOCK_TOTAL,
            _greedy,
            mx.int32,
        )


# --------------------------------------------------------------------------
# the cache's own invariants
# --------------------------------------------------------------------------
def _kv(batch, length, value=1.0):
    return mx.full((batch, 1, length, 4), value), mx.full((batch, 1, length, 4), value)


def test_cache_keeps_padding_left_and_stays_as_wide_as_the_longest_row():
    cache = BatchDFlashKVCache(3)
    cache.set_pending_lengths([4, 4, 4])
    cache.update_and_fetch(*_kv(3, 4))
    assert cache.left_padding == [0, 0, 0]
    assert cache.size() == 4

    cache.set_pending_lengths([3, 1, 2])
    keys, _ = cache.update_and_fetch(*_kv(3, 3))
    # widest row has 7 real rows, so the cache is 7 wide and the others are pad
    assert cache.size() == 7
    assert cache.left_padding == [0, 2, 1]
    assert cache.context_lengths == [7, 5, 6]
    assert keys.shape[2] == 7
    assert cache.offset.tolist() == [7, 5, 6]
    assert min(cache.left_padding) == 0


def test_cache_masks_exactly_the_pad_columns():
    cache = BatchDFlashKVCache(2)
    cache.set_pending_lengths([3, 1])
    cache.update_and_fetch(*_kv(2, 3))
    mask = cache.context_mask(2)
    assert mask.shape == (2, 1, 1, 5)
    assert mask[0, 0, 0].tolist() == [True, True, True, True, True]
    # row 1 has one real context row; the two pad columns are excluded and the
    # proposal columns are not
    assert mask[1, 0, 0].tolist() == [False, False, True, True, True]


def test_cache_eviction_matches_the_scalar_rotating_trim_per_row():
    """The load-bearing algebra: pad columns absorb the shared trim first.

    Row j loses ``max(0, real_j - max_size + 1)`` real rows, which is what its
    own ``RotatingKVCache`` would have computed from its own length -- so the
    window stays per-row exact after it saturates.
    """
    from mlx_vlm.models.cache import RotatingKVCache

    max_size = 5
    batched = BatchDFlashKVCache(2, max_size=max_size)
    scalar = [RotatingKVCache(max_size=max_size, keep=0) for _ in range(2)]
    lengths_per_round = [[3, 3], [3, 1], [2, 2], [4, 1]]
    for lengths in lengths_per_round:
        batched.set_pending_lengths(lengths)
        batched.update_and_fetch(*_kv(2, max(lengths)))
        for row, n in enumerate(lengths):
            scalar[row].update_and_fetch(*_kv(1, n))
    for row in range(2):
        resident = scalar[row].keys.shape[2]
        if scalar[row].offset < resident:
            resident = int(scalar[row].offset)
        assert batched.context_lengths[row] == resident, (
            f"row {row}: batched kept {batched.context_lengths[row]} rows, "
            f"the scalar rotating cache kept {resident}"
        )
        assert int(batched.offset[row].item()) == int(scalar[row].offset)


def test_cache_filter_drops_rows_and_recompacts():
    cache = BatchDFlashKVCache(3)
    cache.set_pending_lengths([4, 2, 2])
    cache.update_and_fetch(*_kv(3, 4))
    assert cache.left_padding == [0, 2, 2]
    cache.filter([1, 2])
    assert cache.batch_size == 2
    # both survivors were padded by 2, so those columns are pad for everyone
    assert cache.left_padding == [0, 0]
    assert cache.size() == 2


def test_cache_rejects_a_pending_length_vector_of_the_wrong_width():
    cache = BatchDFlashKVCache(2)
    with pytest.raises(ValueError):
        cache.set_pending_lengths([1, 2, 3])


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
def _reset_env_memo():
    dflash_utils._BATCHED_DRAFT_ENV = None


def test_the_knob_is_read_once_and_defaults_on(monkeypatch):
    _reset_env_memo()
    monkeypatch.delenv("MLX_VLM_DFLASH_BATCHED_DRAFT", raising=False)
    try:
        assert dflash_utils.batched_draft_enabled() is True
        monkeypatch.setenv("MLX_VLM_DFLASH_BATCHED_DRAFT", "0")
        assert dflash_utils.batched_draft_enabled() is True, "must be memoized"
        _reset_env_memo()
        assert dflash_utils.batched_draft_enabled() is False
    finally:
        _reset_env_memo()


def test_a_stochastic_unpositioned_sampler_keeps_the_row_wise_path(monkeypatch):
    """Batching reorders the draws of a single RNG stream, so it is refused.

    A positioned sampler keys its randomness by (row, position) and is order
    independent; greedy has no state at all. A plain temperature sampler has
    neither property, and quietly giving row 2 the draw row 0 would have taken is
    exactly the kind of silent behaviour change this tree does not ship.
    """
    _reset_env_memo()
    monkeypatch.setenv("MLX_VLM_DFLASH_BATCHED_DRAFT", "1")
    try:
        drafter = _drafter(64, seed=0)
        assert dflash_utils._use_batched_draft(drafter, True, False) is True
        assert dflash_utils._use_batched_draft(drafter, False, True) is True
        assert dflash_utils._use_batched_draft(drafter, False, False) is False
    finally:
        _reset_env_memo()


def test_a_drafter_without_a_batched_cache_keeps_the_row_wise_path(monkeypatch):
    _reset_env_memo()
    monkeypatch.setenv("MLX_VLM_DFLASH_BATCHED_DRAFT", "1")
    try:
        assert (
            dflash_utils._use_batched_draft(SimpleNamespace(), True, False) is False
        )
        # a buffered fixed-window drafter has no concat-cache counterpart
        drafter = _drafter(64, seed=0)
        drafter.config.draft_window_size = 128
        assert drafter.supports_batched_draft() is False
        assert dflash_utils._use_batched_draft(drafter, True, False) is False
    finally:
        _reset_env_memo()


# --------------------------------------------------------------------------
# end to end: the tokens the TARGET emits must not move
# --------------------------------------------------------------------------
E2E_PROMPT = [2, 4, 6, 8, 10, 12]


def _e2e_drafter(target, seed=0):
    """A real DFlash2 drafter shaped for the tiny Glm5Next target."""
    mx.random.seed(seed)
    config = _drafter_config(64)
    config.num_target_layers = 2
    config.target_layer_ids = [0, 1]
    drafter = DFlash2DraftModel(config)
    drafter.bind(target)
    mx.eval(drafter.parameters())
    return drafter


class _RaggedWalk:
    """Force ragged per-row acceptance, then walk the REAL drafts.

    A randomly initialised tiny drafter agrees with a randomly initialised tiny
    target essentially never, so every row would accept 0 and no round would be
    ragged -- the batched path's whole reason to exist would go untested. This
    dictates the acceptance COUNT per row and then builds each row's emitted
    tokens out of that row's own proposals, so a batched draft that proposed
    something different for a row shows up as different emitted tokens.
    """

    def __init__(self, pattern):
        self.pattern = pattern
        self.round = 0

    def __call__(self, draft_tokens, target_tokens, budgets):
        drafts = draft_tokens.tolist()
        targets = target_tokens.tolist()
        row_counts = self.pattern[self.round % len(self.pattern)]
        self.round += 1
        accepted = []
        emitted = []
        for row, (draft_row, target_row) in enumerate(zip(drafts, targets)):
            count = min(row_counts[row % len(row_counts)], len(draft_row))
            accepted.append(count)
            emitted.append((draft_row[:count] + [target_row[count]])[: budgets[row]])
        return accepted, emitted


def _run_batch_rounds(target_lm, drafter, batch, max_tokens, walk=None):
    from mlx_vlm.generate.ar import _make_cache

    model = SimpleNamespace(language_model=target_lm)
    cache = _make_cache(target_lm, [0] * batch)
    out = target_lm(
        mx.array([E2E_PROMPT] * batch, dtype=mx.int32),
        cache=cache,
        capture_layer_ids=[0, 1],
        speculative_verify=True,
    )
    hidden = mx.concatenate(out.hidden_states, axis=-1)[:, -1:, :]
    first_bonus = mx.argmax(out.logits[:, -1, :], axis=-1).astype(mx.int32)
    emitted = [[] for _ in range(batch)]
    original_walk = dflash_utils._speculative_walk_batch
    if walk is not None:
        # the eager walk is the one a caller can substitute; the packed greedy
        # walk resolves acceptance on device and has nothing to substitute into
        drafter.dflash_deferred_walk = False
        dflash_utils._speculative_walk_batch = walk
    try:
        for tokens_out, _ in dflash_utils._dflash_rounds_batch(
            model,
            drafter,
            cache,
            hidden,
            first_bonus=first_bonus,
            max_tokens=max_tokens,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            greedy_sampling=True,
        ):
            for row, token in enumerate(tokens_out):
                if token is not None:
                    emitted[row].append(int(token))
    finally:
        dflash_utils._speculative_walk_batch = original_walk
    return emitted


@pytest.mark.parametrize("batch", [2, 4])
def test_the_emitted_target_tokens_are_unchanged_by_the_batched_draft(
    batch, monkeypatch
):
    """The whole round loop, both arms, real target and real drafter.

    Speculative decoding emits the TARGET's tokens, so a drafter change can only
    move throughput -- unless it silently changes what the drafter proposes for a
    row, in which case the accepted prefix moves and the emitted text moves with
    it. This is the test that says it does not.
    """
    from mlx_vlm.tests.test_dflash_ragged_rollback import _tiny_glm5_next_target

    mx.random.seed(17)
    target_lm = _tiny_glm5_next_target()
    mx.eval(target_lm.parameters())

    arms = {}
    for arm, value in (("row-wise", "0"), ("batched", "1")):
        _reset_env_memo()
        monkeypatch.setenv("MLX_VLM_DFLASH_BATCHED_DRAFT", value)
        try:
            drafter = _e2e_drafter(target_lm, seed=3)
            arms[arm] = _run_batch_rounds(
                target_lm,
                drafter,
                batch,
                24,
                walk=_RaggedWalk([[3, 0, 5, 1], [0, 4, 2, 6], [6, 1, 0, 3]]),
            )
            arms[arm + "/kept"] = int(getattr(drafter, "per_row_kept_tokens", 0))
            arms[arm + "/clamped"] = int(getattr(drafter, "clamped_tokens", 0))
        finally:
            _reset_env_memo()

    assert arms["row-wise"] == arms["batched"], "the emitted target tokens moved"
    assert arms["row-wise/clamped"] == 0, "the tiny target should roll back per row"
    assert arms["row-wise/kept"] == arms["batched/kept"]
    # ragged rounds are the whole point; a run where no row ever accepted more
    # than another would pass the comparison above without testing anything.
    assert arms["row-wise/kept"] > 0, "no ragged round occurred; the test is vacuous"


def test_the_batch_round_counter_separates_rows_from_rounds(monkeypatch):
    """``rounds`` in the server log counts ROW-rounds; this is the denominator.

    Without it a log line cannot say whether eight rows shared one round or one
    row took eight, which is exactly the question ``rows/round`` answers.
    """
    from mlx_vlm.tests.test_dflash_ragged_rollback import _tiny_glm5_next_target

    mx.random.seed(23)
    target_lm = _tiny_glm5_next_target()
    mx.eval(target_lm.parameters())
    _reset_env_memo()
    monkeypatch.setenv("MLX_VLM_DFLASH_BATCHED_DRAFT", "1")
    try:
        drafter = _e2e_drafter(target_lm, seed=4)
        _run_batch_rounds(
            target_lm,
            drafter,
            4,
            16,
            walk=_RaggedWalk([[2, 0, 4, 1], [1, 3, 0, 5]]),
        )
    finally:
        _reset_env_memo()
    batch_rounds = drafter.speculative_total_batch_rounds
    row_rounds = drafter.speculative_total_row_rounds
    assert batch_rounds > 0
    assert row_rounds == len(drafter.accept_lens)
    assert 1.0 < row_rounds / batch_rounds <= 4.0


# --------------------------------------------------------------------------
# the receipts have to leave the process (law 23)
# --------------------------------------------------------------------------
def test_the_server_log_line_carries_the_clamp_and_per_row_receipts():
    """Both counters, the realised rows/round, and the knob -- in the log line.

    They were maintained on the served path and printed by nothing on it:
    ``_format_speculative_stats`` has one non-test caller and it is the LIBRARY
    generate path (``generate/dispatch.py``). Sweep11's L2 had to read them off
    the live drafter object with an in-process observer thread, which is not a
    receipt anyone repeating the measurement can obtain from a server log.
    """
    import inspect

    from mlx_vlm.server import generation

    src = inspect.getsource(generation)
    for field in (
        "clamped=%d",
        "per_row_kept=%d",
        "rows/round=%.2f",
        "batched_draft=%s",
    ):
        assert field in src, f"the speculative log line must report {field}"
    assert "speculative_clamp_since(drafter, clamp_snapshot)" in src, (
        "the counters must be diffed against a per-batch snapshot, not printed "
        "as the drafter's lifetime totals"
    )
    assert "(row_rounds / batch_rounds) if batch_rounds else" in src


def test_clamp_counters_diff_against_a_snapshot():
    from mlx_vlm.speculative.common import (
        speculative_clamp_since,
        speculative_clamp_snapshot,
        _record_batch_round,
        _record_per_row_rollback,
        _record_uniform_clamp,
    )

    drafter = SimpleNamespace()
    _record_uniform_clamp(drafter, 5)
    _record_per_row_rollback(drafter, 9)
    _record_batch_round(drafter, 4)
    snapshot = speculative_clamp_snapshot(drafter)

    _record_uniform_clamp(drafter, 2)
    _record_per_row_rollback(drafter, 3)
    _record_batch_round(drafter, 8)
    _record_batch_round(drafter, 6)
    clamped, kept, batch_rounds, row_rounds = speculative_clamp_since(
        drafter, snapshot
    )
    assert (clamped, kept, batch_rounds, row_rounds) == (2, 3, 2, 14)
    assert row_rounds / batch_rounds == 7.0


def test_the_runtime_snapshot_exposes_the_speculative_counters(monkeypatch):
    """``/health``-side receipt, so the observer thread is not needed at all."""
    import sys

    app_module = sys.modules.get("mlx_vlm.server.app")
    if app_module is None:  # pragma: no cover - import for standalone runs
        import importlib

        app_module = importlib.import_module("mlx_vlm.server.app")

    drafter = SimpleNamespace(
        speculative_total_rounds=40,
        speculative_total_batch_rounds=5,
        speculative_total_row_rounds=40,
        speculative_total_drafted=280,
        speculative_total_accepted=120.0,
        speculative_total_clamped=17,
        speculative_total_per_row_kept=23,
    )
    monkeypatch.setattr(
        app_module.runtime,
        "response_generator",
        SimpleNamespace(draft_model=drafter, draft_kind="dflash"),
        raising=False,
    )
    snapshot = app_module._speculative_stats_snapshot()
    assert snapshot["enabled"] is True
    assert snapshot["clamped"] == 17
    assert snapshot["per_row_kept"] == 23
    assert snapshot["rows_per_round"] == 8.0
    assert snapshot["width"] == 8.0
    assert snapshot["kind"] == "dflash"
    assert "batched_draft" in snapshot
    # the timing arm is opt-in; its absence must not read as zero
    assert "draft_seconds" not in snapshot

    drafter.speculative_draft_seconds = 1.5
    drafter.speculative_verify_seconds = 4.5
    timed = app_module._speculative_stats_snapshot()
    assert timed["draft_seconds"] == 1.5
    assert timed["verify_seconds"] == 4.5


def test_the_runtime_snapshot_says_disabled_without_a_drafter(monkeypatch):
    import sys

    app_module = sys.modules["mlx_vlm.server.app"]
    monkeypatch.setattr(app_module.runtime, "response_generator", None, raising=False)
    assert app_module._speculative_stats_snapshot() == {"enabled": False}


def test_the_round_timers_are_off_by_default_and_readable(monkeypatch):
    """The split the prereg reads is opt-in, and it perturbs the round it times.

    Each half is waited for, which the untimed path never does, so the arms are
    comparable to each other and not to an untimed run.
    """
    dflash_utils._ROUND_TIMERS_ENV = None
    monkeypatch.delenv("MLX_VLM_DFLASH_ROUND_TIMERS", raising=False)
    try:
        assert dflash_utils._round_timers_enabled() is False
        dflash_utils._ROUND_TIMERS_ENV = None
        monkeypatch.setenv("MLX_VLM_DFLASH_ROUND_TIMERS", "1")
        assert dflash_utils._round_timers_enabled() is True
    finally:
        dflash_utils._ROUND_TIMERS_ENV = None
