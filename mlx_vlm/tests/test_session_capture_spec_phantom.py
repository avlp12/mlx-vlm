"""V1d: a speculative round's session capture must not carry PHANTOM columns.

The L31 end-of-turn capture runs inside the window between ``finish_reason``
and ``remove()`` -- i.e. inside one of the round loop's ``yield``s
(``server/generation.py::_step`` -> ``BatchGenerator.capture_session``).  Until
this change the batch loops emitted the whole verify block FIRST and rolled the
cache back AFTER, so the snapshot the capture took held the KV of every DRAFTED
token in the block, including the ones the walk rejected and the row never
emitted.

The key is ``prompt + everything emitted``, and the last emitted token is never
fed back through the model, so a truthful rung has ``n == len(key) - 1``.  With
``phantom`` unemitted columns still live:

    n = len(key) - 1 + phantom,   phantom = block_total - 1 - p

(``p`` = the emit position the row finished at).  Measured on the tiny
glm5_next fixture at 1a589953, block_total 5, a 3-row batch, row 0 stopping on
the target's own token at the end of its round:

    accepted 2 -> n = 9, len(key) = 8  -> phantom 2 -> record_session_turn
                  refuses (`cache_longer_than_key`): the rung is silently lost
    accepted 3 -> n = 9, len(key) = 9  -> phantom 1 -> n == len(key), STORED,
                  with its last column labelled 13 while it holds the KV of the
                  rejected draft 14
    accepted 3, stop mid-block at p=1 -> n = 9, len(key) = 7 -> phantom 3

The fix is in two places and this file pins both: the batch loops roll back
BEFORE they emit (dflash/mtp/eagle3), and ``capture_session`` checks the
snapshot's own length against the key before it stores, giving the extra
columns back where the cache can and refusing where it cannot.

CPU only, real ``_dflash_rounds_batch``, real ``record_session_turn``, real
``ContextVault``; only the drafter and the accept walk are stubbed.
"""

from types import SimpleNamespace
from typing import List, Optional

import mlx.core as mx
import pytest

from mlx_vlm import apc as _apc
from mlx_vlm import context_vault as _cv
from mlx_vlm.generate import ar as ar_mod
from mlx_vlm.generate.ar import BatchGenerator, SpeculativeGenerationBatch, _make_cache
from mlx_vlm.speculative import dflash as dflash_utils
from mlx_vlm.tests.test_dflash_ragged_rollback import _tiny_glm5_next_target

BLOCK_TOTAL = 5
PROMPTS = {0: [2, 4, 6, 8], 1: [3, 5, 7, 9], 2: [10, 12, 14, 16]}
DRAFTS = {0: [11, 12, 13, 14], 1: [21, 22, 23, 24], 2: [17, 18, 19, 20]}
BONUS = [5, 7, 9]
SENTINEL = 28
_ROW_OF_BONUS = {5: 0, 7: 1, 9: 2}
_ROW_OF_FIRST_DRAFT = {11: 0, 21: 1, 17: 2}
UIDS = [101, 102, 103]
_TOL = 1e-4


@pytest.fixture(scope="module")
def model():
    mx.random.seed(3)
    m = _tiny_glm5_next_target()
    mx.eval(m.parameters())
    return m


@pytest.fixture(autouse=True)
def _session_on(monkeypatch):
    monkeypatch.setenv(_cv._ENV_SESSION, "1")
    monkeypatch.delenv("MLX_VLM_SPEC_EXTEND_ACTIVE", raising=False)
    _cv.reset_session_skips()


class _StubDrafter:
    requires_uniform_batch_acceptance = False
    dflash_deferred_walk = False

    def __init__(self):
        self.config = SimpleNamespace(
            block_size=BLOCK_TOTAL,
            runtime_block_size=BLOCK_TOTAL,
            target_layer_ids=[0],
        )
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def reset(self, model=None):
        self.accept_lens = []
        self.draft_lens = []

    def make_cache(self):
        return []

    def draft_block(self, bonus, hidden, cache, bs, sampler, token_dtype):
        row = _ROW_OF_BONUS.get(int(bonus))
        if row is None:
            row = int(bonus) - SENTINEL
        return mx.array([DRAFTS[row][: bs - 1]], dtype=token_dtype)


class _Capture:
    """What the capture saw, in the order the code sees it."""

    def __init__(self):
        self.snapshot_len: Optional[int] = None   # before capture_session's guard
        self.recorded_len: Optional[int] = None   # what reached the vault
        self.key: List[int] = []
        self.row_cache = None
        self.stored: Optional[bool] = None

    @property
    def phantom(self) -> int:
        return self.snapshot_len - (len(self.key) - 1)


def _drive(model, *, nrows, stops, rounds, capture_uid, vault=None):
    """Real rounds until ``capture_uid`` finishes; real capture at that instant.

    ``stops`` is a set of ``(round_number, token)`` pairs -- round-aware because
    every round ends every row on that row's SENTINEL.
    """
    remaining = list(rounds)
    round_no = {"n": 0}

    def walk(draft_tokens, target_tokens, budgets):
        accepted = remaining.pop(0)
        round_no["n"] += 1
        rows = draft_tokens.tolist()
        out = []
        for i, a in enumerate(accepted):
            orig = _ROW_OF_FIRST_DRAFT[int(rows[i][0])]
            out.append((rows[i][:a] + [SENTINEL + orig])[: budgets[i]])
        return list(accepted), out

    def walk1(draft_tokens, target_tokens, budget, **kw):
        a, out = walk(draft_tokens, target_tokens, [budget])
        return a[0], out[0]

    cache = _make_cache(model, [0] * nrows)
    model(
        mx.array([PROMPTS[r] for r in range(nrows)], dtype=mx.int32), cache=cache
    )
    uids = UIDS[:nrows]
    batch = SpeculativeGenerationBatch(
        model=SimpleNamespace(language_model=model),
        draft_model=_StubDrafter(),
        draft_kind="dflash",
        uids=list(uids),
        first_tokens=mx.array(BONUS[:nrows], dtype=mx.int32),
        prompt_cache=cache,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        stop_criteria=lambda token: (round_no["n"], int(token)) in stops,
        max_tokens=[64] * nrows,
        hidden=mx.zeros((nrows, 1, model.args.hidden_size), dtype=mx.float32),
        shared_kv_states=None,
        prompt_tokens=None,
        greedy_sampling=True,
    )
    session_tokens = {uid: list(PROMPTS[i]) for i, uid in enumerate(uids)}
    generator = SimpleNamespace(
        vault=vault if vault is not None else _cv.ContextVault(
            "v1d", budget_bytes=1 << 30
        ),
        _session_tokens=session_tokens,
        _generation_batch=batch,
    )
    seen = _Capture()

    real_snapshot = _apc.snapshot_prompt_cache_row
    real_record = _cv.record_session_turn

    def snapshot_spy(caches, batch_idx=0, **kw):
        row = real_snapshot(caches, batch_idx, **kw)
        seen.snapshot_len = _cv.prefix_len_from_cache(row) if row else None
        return row

    def record_spy(v, key, row_cache, **kw):
        seen.key = list(key)
        seen.recorded_len = _cv.prefix_len_from_cache(row_cache)
        seen.row_cache = row_cache
        seen.stored = real_record(v, key, row_cache, **kw)
        return seen.stored

    originals = (
        dflash_utils._speculative_walk_batch,
        dflash_utils._speculative_walk,
        ar_mod._apc.snapshot_prompt_cache_row,
        ar_mod._context_vault.record_session_turn,
    )
    dflash_utils._speculative_walk_batch = walk
    dflash_utils._speculative_walk = walk1
    ar_mod._apc.snapshot_prompt_cache_row = snapshot_spy
    ar_mod._context_vault.record_session_turn = record_spy
    try:
        for _ in range(12):
            if len(batch) == 0:
                break
            for response in batch.next():
                if response.token is not None:
                    session_tokens[response.uid].append(int(response.token))
                if response.finish_reason is None:
                    continue
                # Exactly where the server captures.
                ok = BatchGenerator.capture_session(
                    generator, response.uid, session_id="conv-1"
                )
                if response.uid == capture_uid:
                    if seen.stored is None:
                        seen.stored = ok
                        seen.key = list(session_tokens[response.uid])
                    return seen, generator.vault, cache
    finally:
        (
            dflash_utils._speculative_walk_batch,
            dflash_utils._speculative_walk,
            ar_mod._apc.snapshot_prompt_cache_row,
            ar_mod._context_vault.record_session_turn,
        ) = originals
    pytest.fail(f"uid {capture_uid} never finished")


def _reference_kv(model, tokens):
    """A single-row forward over exactly ``tokens``."""
    cache = model.make_cache()
    model(mx.array([list(tokens)], dtype=mx.int32), cache=cache)
    return cache


def _kv_diff(row_cache, ref_cache, n):
    """Worst |diff| over the first ``n`` KV columns; inf on a shape mismatch."""
    worst, compared = 0.0, 0
    for got_entry, ref_entry in zip(row_cache, ref_cache):
        got_subs = getattr(got_entry, "caches", None)
        ref_subs = getattr(ref_entry, "caches", None)
        if got_subs is None or ref_subs is None:
            continue
        for got, ref in zip(got_subs, ref_subs):
            gk, rk = getattr(got, "keys", None), getattr(ref, "keys", None)
            if gk is None or rk is None:
                continue
            a, b = gk[:, :, :n, :], rk[:, :, :n, :]
            if a.shape != b.shape:
                return float("inf")
            worst = max(
                worst,
                float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))),
            )
            compared += 1
    return worst if compared else float("inf")


# ==========================================================================
# The defect
# ==========================================================================
def test_a_rejected_draft_is_not_carried_into_the_session_rung(model):
    """Row 0 accepts 2 of its 4 drafts and stops on the target's own token.

    At 1a589953 the snapshot was taken with the whole block still live:
    n = 9 against a key of 8, so ``record_session_turn`` refused
    (``cache_longer_than_key``) and the turn lost its rung in silence.
    """
    seen, vault, _ = _drive(
        model,
        nrows=3,
        stops={(1, SENTINEL + 0)},
        rounds=[[2, 3, 3]],
        capture_uid=101,
    )
    assert seen.key == [2, 4, 6, 8, 5, 11, 12, 28], seen.key
    assert seen.snapshot_len == len(seen.key) - 1 == 7, (
        f"the snapshot holds {seen.snapshot_len} tokens for a key of "
        f"{len(seen.key)}: phantom {seen.phantom} columns of drafted tokens "
        "this row never emitted"
    )
    assert seen.stored is True, _cv.session_skip_counts()
    assert vault.stats.session_inserts == 1
    assert "cache_longer_than_key" not in _cv.session_skip_counts()
    cp = _cv.lookup_session(vault, seen.key + [99])
    assert cp is not None and cp.prefix_len == 7


def test_a_block_one_short_is_not_stored_with_a_mislabelled_last_column(model):
    """Row 0 accepts 3 of 4: the phantom == 1 case, which is the dangerous one.

    One unemitted column makes the cache length EQUAL the key length, so the
    ``n > len(toks)`` guard does not fire and the rung is stored -- with its
    last column holding the KV of the rejected draft 14 under the label 13.
    Here that column is gone, and the rung's KV is what a forward over
    ``key[:n]`` produces, column for column.
    """
    seen, vault, _ = _drive(
        model,
        nrows=3,
        stops={(1, SENTINEL + 0)},
        rounds=[[3, 3, 3]],
        capture_uid=101,
    )
    assert seen.key == [2, 4, 6, 8, 5, 11, 12, 13, 28], seen.key
    assert seen.snapshot_len == 8 == len(seen.key) - 1, seen.snapshot_len
    assert seen.stored is True, _cv.session_skip_counts()

    n = seen.recorded_len
    mine = _kv_diff(seen.row_cache, _reference_kv(model, seen.key[:n]), n)
    assert mine <= _TOL, (
        f"the stored rung differs from a forward over its own key by {mine:.3e}"
    )
    # ...and it is NOT the sequence the phantom column would have described.
    phantom_key = seen.key[:n] + [DRAFTS[0][3]]
    other = _kv_diff(seen.row_cache, _reference_kv(model, phantom_key), n + 1)
    assert other > _TOL, (
        "the rung still spans the rejected draft's column: that is the "
        "mislabelled-last-column store this test exists for"
    )


@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
def test_the_snapshot_length_is_the_emitted_length_for_every_accept_count(
    model, accepted
):
    """The arithmetic, swept: ``n == len(key) - 1``, phantom 0, for every
    acceptance the walk can return.  Before the fix the same sweep gave
    phantom = block_total - 1 - accepted = 4, 3, 2, 1."""
    seen, _, _ = _drive(
        model,
        nrows=3,
        stops={(1, SENTINEL + 0)},
        rounds=[[accepted, 3, 3]],
        capture_uid=101,
    )
    assert len(seen.key) == len(PROMPTS[0]) + 1 + accepted + 1
    assert seen.phantom == 0, (
        f"accepted {accepted}: snapshot {seen.snapshot_len}, key "
        f"{len(seen.key)} -> {seen.phantom} unemitted columns"
    )
    assert seen.stored is True, _cv.session_skip_counts()


def test_a_single_row_speculative_batch_is_covered_too(model, monkeypatch):
    """B == 1 is not exempt: with V1b's admission on, a one-row batch runs the
    BATCH loop (``run_speculative_server_rounds``: ``batch_size == 1 and
    admission is None`` is the only route to the scalar one), and the batch
    loop is where the capture landed before the rollback."""
    monkeypatch.setenv("MLX_VLM_SPEC_EXTEND_ACTIVE", "1")
    seen, vault, _ = _drive(
        model,
        nrows=1,
        stops={(1, SENTINEL + 0)},
        rounds=[[3]],
        capture_uid=101,
    )
    assert seen.snapshot_len == len(seen.key) - 1 == 8, seen.snapshot_len
    assert seen.stored is True, _cv.session_skip_counts()
    assert vault.stats.session_inserts == 1


def test_the_scalar_single_row_loop_already_rolled_back_before_it_emitted(model):
    """Stated honestly, because the V1c report said otherwise.

    ``_dflash_rounds`` (the B == 1 scalar loop, reached when admission is off)
    has ALWAYS rolled back before its ``yield`` -- speculative/dflash.py, the
    ``if accepted < bs - 1: rollback`` block sits above ``for tok in
    new_tokens: yield tok``.  So the pre-rollback phantom never applied there,
    and this test pins that it still does not.
    """
    seen, vault, _ = _drive(
        model,
        nrows=1,
        stops={(1, SENTINEL + 0)},
        rounds=[[2]],
        capture_uid=101,
    )
    assert seen.key == [2, 4, 6, 8, 5, 11, 12, 28], seen.key
    assert seen.snapshot_len == len(seen.key) - 1 == 7
    assert seen.stored is True, _cv.session_skip_counts()


def test_a_row_that_stopped_mid_block_is_refused_not_mislabelled(model):
    """The residual, and why it is a refusal rather than a trim.

    A row that stops on a stop token INSIDE the block leaves the round holding
    the KV of the tokens it committed after that one -- real, correctly
    attended tokens that the stream never carried, so they are not in the key.
    On this hybrid cache they cannot be given back: the KDA half is a recurrent
    state that has absorbed them and can only be replayed, never sliced.  The
    rung is refused by name.
    """
    seen, vault, _ = _drive(
        model,
        nrows=3,
        stops={(1, DRAFTS[0][1])},          # stop at emit position 1 of 3
        rounds=[[3, 3, 3]],
        capture_uid=101,
    )
    assert seen.key == [2, 4, 6, 8, 5, 11, 12], seen.key
    assert seen.snapshot_len == 8 > len(seen.key) - 1 == 6
    assert seen.stored is False
    assert _cv.session_skip_counts().get("cache_ahead_of_emitted") == 1
    assert vault.rungs == 0, "a refusal must not leave a rung behind"


# ==========================================================================
# The two mechanisms, alone
# ==========================================================================
def test_an_attention_only_snapshot_gives_the_unemitted_columns_back():
    """``trim_cache_to_prefix`` is exactly ``is_trimmable()``: a KV-only row
    gives the columns back, a hybrid row refuses."""
    from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache

    def kv_row(n):
        c = CacheList(KVCache(), KVCache())
        k = mx.arange(1 * 2 * n * 4, dtype=mx.float32).reshape(1, 2, n, 4)
        for sub in c.caches:
            sub.update_and_fetch(k, k)
        return [c]

    row = kv_row(8)
    assert _cv.trim_cache_to_prefix(row, 6) is True
    assert _cv.prefix_len_from_cache(row) == 6
    assert _cv.trim_cache_to_prefix(row, 6) is True, "already there is not an error"
    assert _cv.trim_cache_to_prefix(row, 9) is False, "cannot invent columns"

    hybrid = [ArraysCache(size=2)] + kv_row(8)
    hybrid[0][0] = mx.zeros((1, 2, 8, 4))
    hybrid[0][1] = mx.zeros((1, 2, 8, 8))
    assert _cv.trim_cache_to_prefix(hybrid, 6) is False, (
        "the recurrent half cannot be rewound by slicing, so the whole row "
        "has to refuse rather than trim one half of it"
    )
    assert _cv.prefix_len_from_cache(hybrid) == 8, "and it is left as it was"


def test_the_vault_refuses_a_caller_that_names_a_length_the_cache_lacks():
    """``prefix_len`` is a claim; the cache's own offset is the witness."""
    from mlx_vlm.models.cache import CacheList, KVCache

    c = CacheList(KVCache(), KVCache())
    k = mx.zeros((1, 2, 8, 4), dtype=mx.float32)
    for sub in c.caches:
        sub.update_and_fetch(k, k)
    v = _cv.ContextVault("v1d-claim", budget_bytes=1 << 30)
    toks = list(range(1, 13))
    assert _cv.record_session_turn(
        v, toks, [c], completed=True, session_id="c", adopt=False, prefix_len=6
    ) is False
    assert _cv.session_skip_counts().get("prefix_len_disagrees_with_cache") == 1
    assert v.rungs == 0
    assert _cv.record_session_turn(
        v, toks, [c], completed=True, session_id="c", adopt=False, prefix_len=8
    ) is True


# ==========================================================================
# What must not move
# ==========================================================================
def test_the_capture_instant_sees_the_committed_cache(model):
    """At the capture instant the row's cache is exactly what a fresh forward
    over that row's committed tokens produces -- for B == 1 as well as B == 3.

    ``[first bonus] + emitted[:-1]`` is the committed sequence (the last emitted
    token is the next round's input and has not been fed back).  This is the
    same invariant test_dflash_ragged_rollback / test_dflash_perrow_rollback
    assert for the cache the DECODE path is left with at the end of a round;
    what moved is only WHEN it becomes true within the round, and the emitted
    stream (asserted token by token through ``seen.key``) did not move at all.
    """
    for nrows in (1, 3):
        seen, _, cache = _drive(
            model,
            nrows=nrows,
            stops={(1, SENTINEL + 0)},
            rounds=[[3, 3, 3][:nrows]],
            capture_uid=101,
        )
        assert seen.key == [2, 4, 6, 8, 5, 11, 12, 13, 28], (
            f"B={nrows}: the emitted stream changed"
        )
        committed = seen.key[len(PROMPTS[0]):][:-1]
        ref = _reference_kv(model, PROMPTS[0] + committed)
        got = _apc.snapshot_prompt_cache_row(cache, 0)
        n = _cv.prefix_len_from_cache(ref)
        assert _cv.prefix_len_from_cache(got) == n, (
            f"B={nrows}: the row's cache offset moved"
        )
        worst = _kv_diff(got, ref, n)
        assert worst <= _TOL, f"B={nrows}: max abs diff {worst:.3e}"
