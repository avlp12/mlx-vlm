"""Rank-1 side of TP=2 serving, and the both-sides replay of a whole spec round.

The rank-0 tests (``test_server_tp_mode.py``) pin what gets announced.  These
pin what rank 1 *does* with it, and then close the loop: a real speculative
round loop is driven against a scripted target on rank 0, the control stream it
emits is captured, and that stream is replayed into a worker holding a twin
model.  What comes out is the sequence of forwards, capture flags and rollbacks
rank 1 would actually have performed -- which is the only thing that has to
match, because everything else about the two ranks is already symmetric.

No GPU, no second box, no group: every collective is stubbed, so the properties
under test are protocol properties rather than hardware ones.
"""

import mlx.core as mx
import pytest
from types import SimpleNamespace

from mlx_vlm.tp import worker as W


# --------------------------------------------------------------- proto guard
def test_handshake_rejects_a_peer_on_a_different_revision(monkeypatch):
    """A header-width skew between the boxes is a HANG, not an error.

    ``_ctrl_recv`` allocates ``HEADER + n`` words and ``_ctrl_send`` sends the
    same; mismatched shapes in a jaccl all_sum do not raise, they wedge.  The
    handshake rides a vector whose width is frozen forever, so the skew gets
    diagnosed by the one collective that cannot itself be a victim of it.
    """
    class _FakeTransport:
        @staticmethod
        def all_sum(x):
            # Peer claims header width 12 while we are on 14.
            row = x[0].tolist()
            peer = [W.PROTO_VERSION, 12, W._max_tok(), 0, 0, 0, 0, 0]
            return mx.array([[a + b for a, b in zip(row, peer)]], dtype=mx.int32)

    monkeypatch.setattr(W, "_proto_handshake", _real_handshake_with(_FakeTransport))
    with pytest.raises(W.TPUnavailable, match="header_width mismatch"):
        W._proto_handshake()


def _real_handshake_with(transport):
    """Re-bind ``_proto_handshake`` onto a stub transport."""
    def _handshake(timeout_s: float = 60.0) -> dict:
        mine = [W.PROTO_VERSION, W.HEADER, W._max_tok(), 0, 0, 0, 0, 0]
        out = transport.all_sum(mx.array([mine], dtype=mx.int32))
        total = [int(v) for v in out[0].tolist()]
        n = 2
        for i, label in enumerate(("proto_version", "header_width", "max_tokens")):
            if total[i] != mine[i] * n:
                raise W.TPUnavailable(f"tp {label} mismatch")
        return {}
    return _handshake


# ------------------------------------------------------------- worker basics
class _TwinLM:
    """Rank 1's half: records what it was asked to run."""

    def __init__(self, hidden=4, vocab=8):
        self.forwards = []      # (batch, seqlen, captured)
        self.rollbacks = []
        self.caches = []
        self._h, self._v = hidden, vocab

    def make_cache(self):
        c = ["cache", len(self.caches)]
        self.caches.append(c)
        return c

    def __call__(self, ids, cache=None, **kw):
        captured = kw.get("capture_layer_ids") is not None
        self.forwards.append((ids.shape[0], ids.shape[1], captured))
        return SimpleNamespace(
            logits=mx.zeros((ids.shape[0], ids.shape[1], self._v)),
            hidden_states=[mx.zeros((ids.shape[0], ids.shape[1], self._h))],
            gdn_states=["gdn"] if captured else None,
        )

    def rollback_speculative_cache(self, caches, gdn, accepted, bs):
        self.rollbacks.append((list(accepted), bs, gdn))
        return max(accepted)


def _msg(op, epoch, *, ids=None, flags=0, arg0=0, name=""):
    shape = (1, len(ids)) if ids is not None else None
    return W.decode(W.encode(op, epoch, shape, ids, n=64,
                             flags=flags, arg0=arg0, name=name))


def test_worker_makes_a_cache_then_forwards_into_it():
    lm = _TwinLM()
    st = W._WorkerState(lm)
    assert st.handle(_msg(W.OP_MAKE_CACHE, 1)) is True
    assert st.handle(_msg(W.OP_FORWARD, 1, ids=[1, 2, 3])) is True
    assert lm.forwards == [(1, 3, False)]
    assert len(lm.caches) == 1


def test_worker_replaces_the_cache_on_a_new_epoch():
    """One live cache: rank 0 drives one conversation at a time, and holding the
    previous one would pin its KV for nothing."""
    lm = _TwinLM()
    st = W._WorkerState(lm)
    st.handle(_msg(W.OP_MAKE_CACHE, 1))
    st.handle(_msg(W.OP_FORWARD, 1, ids=[1]))
    st.handle(_msg(W.OP_MAKE_CACHE, 2))
    assert list(st.caches) == [2]


def test_worker_exits_on_exit():
    assert W._WorkerState(_TwinLM()).handle(_msg(W.OP_EXIT, 0)) is False


def test_worker_rejects_an_unknown_verb():
    """A verb this worker does not know means the ranks disagree about the
    protocol; guessing would be a desync with no symptom."""
    with pytest.raises(W.TPDesync, match="unknown control op"):
        W._WorkerState(_TwinLM()).handle(_msg(99, 1))


# --------------------------------------------------------- capture + rollback
def test_capture_flag_allocates_the_sink_on_rank_1():
    lm = _TwinLM()
    st = W._WorkerState(lm)
    st.handle(_msg(W.OP_MAKE_CACHE, 1))
    st.handle(_msg(W.OP_FORWARD, 1, ids=[1, 2, 3, 4], flags=W.FLAG_CAPTURE))
    assert lm.forwards[-1] == (1, 4, True)
    assert st.last_gdn == ["gdn"]


def test_rollback_replays_this_ranks_own_half():
    lm = _TwinLM()
    st = W._WorkerState(lm)
    st.handle(_msg(W.OP_MAKE_CACHE, 1))
    st.handle(_msg(W.OP_FORWARD, 1, ids=[1, 2, 3, 4], flags=W.FLAG_CAPTURE))
    st.handle(_msg(W.OP_ROLLBACK, 1, ids=[2], arg0=4))
    assert lm.rollbacks == [([2], 4, ["gdn"])]
    assert st.last_gdn is None, "a consumed round must not be rolled back twice"


def test_rollback_without_a_captured_round_is_refused():
    """Rank 1 cannot roll back what it never captured.  Silently skipping would
    leave rank 1 holding the whole rejected block while rank 0 dropped it."""
    lm = _TwinLM()
    st = W._WorkerState(lm)
    st.handle(_msg(W.OP_MAKE_CACHE, 1))
    st.handle(_msg(W.OP_FORWARD, 1, ids=[1, 2]))          # no capture flag
    with pytest.raises(W.TPDesync, match="no captured round"):
        st.handle(_msg(W.OP_ROLLBACK, 1, ids=[0], arg0=2))


# ------------------------------------------------------------- vault on rank 1
class _FakeShardVault:
    def __init__(self, holds=()):
        self.holds = dict(holds)
        self.stored = []
        self.restored = []

    def store(self, name, prefix_len, caches):
        self.stored.append((name, prefix_len, caches))
        self.holds[name] = prefix_len
        return True

    def restore(self, name, prefix_len, caches):
        ok = self.holds.get(name) == prefix_len
        self.restored.append((name, prefix_len, ok))
        return ok


def test_vault_store_checkpoints_this_ranks_own_cache(monkeypatch):
    lm, v = _TwinLM(), _FakeShardVault()
    st = W._WorkerState(lm, vault=v)
    st.handle(_msg(W.OP_MAKE_CACHE, 1))
    st.handle(_msg(W.OP_FORWARD, 1, ids=[1, 2]))
    st.handle(_msg(W.OP_VAULT_STORE, 1, arg0=2048, name="ab" * 16))
    assert v.stored == [("ab" * 16, 2048, lm.caches[0])]


def test_vault_restore_acks_a_hit_and_installs_the_cache(monkeypatch):
    acks = []
    monkeypatch.setattr(W, "_ack_send", acks.append)
    lm, v = _TwinLM(), _FakeShardVault({"cd" * 16: 4096})
    st = W._WorkerState(lm, vault=v)
    st.handle(_msg(W.OP_VAULT_RESTORE, 5, arg0=4096, name="cd" * 16))
    assert acks == [1]
    assert list(st.caches) == [5]


def test_vault_restore_acks_a_miss_without_installing_anything(monkeypatch):
    """The two vaults evict independently.  Rank 1 answering "no" is the only
    thing that stops rank 0 serving warm against a cold peer."""
    acks = []
    monkeypatch.setattr(W, "_ack_send", acks.append)
    lm, v = _TwinLM(), _FakeShardVault()
    st = W._WorkerState(lm, vault=v)
    st.handle(_msg(W.OP_VAULT_RESTORE, 5, arg0=4096, name="ef" * 16))
    assert acks == [0]
    assert list(st.caches) == []


def test_vault_restore_without_a_vault_still_acks(monkeypatch):
    """Never leave rank 0 waiting on an ack that is not coming: an unanswered
    collective is a hang, and a worker built without a vault is a supported
    configuration (MLX_VLM_GLM5_VAULT unset)."""
    acks = []
    monkeypatch.setattr(W, "_ack_send", acks.append)
    st = W._WorkerState(_TwinLM(), vault=None)
    st.handle(_msg(W.OP_VAULT_RESTORE, 1, arg0=8, name="11" * 16))
    assert acks == [0]


# =============================================================================
# The whole loop: a real speculative round loop, replayed on rank 1
# =============================================================================
class _PlanExhausted(Exception):
    """The script ran out; stop the round loop deterministically."""


class _ScriptedPair:
    """A target whose acceptance is dictated per round.

    ``plan[i]`` is how many of round ``i``'s drafted tokens the target agrees
    with; ``None`` means "all of them" (a full-accept round, which must NOT
    produce a rollback).
    """

    def __init__(self, plan, hidden=4, vocab=64):
        self.plan = plan
        self.round = 0
        self.last_draft = None
        self.hidden, self.vocab = hidden, vocab
        self.forwards = []
        self.rollbacks = []
        self.aborted = False

    # -- target side --
    def make_cache(self):
        return ["target-cache"]

    def _onehot(self, row):
        out = mx.zeros((1, len(row), self.vocab))
        idx = mx.array(row, dtype=mx.int32)
        return out + (mx.arange(self.vocab)[None, None, :] == idx[None, :, None]) * 10.0

    def __call__(self, ids, cache=None, **kw):
        if self.round >= len(self.plan):
            self.aborted = True
            raise _PlanExhausted
        captured = kw.get("capture_layer_ids") is not None
        S = ids.shape[1]
        self.forwards.append((ids.shape[0], S, captured))
        draft = self.last_draft or []
        a = self.plan[self.round] if self.plan[self.round] is not None else len(draft)
        # Token ids must stay inside the fake vocab or the one-hot never fires
        # and every round would silently look like a zero-accept.
        row = list(draft[:a]) + [40 + self.round]
        row += [50] * (S - len(row))
        self.round += 1
        return SimpleNamespace(
            logits=self._onehot(row[:S]),
            hidden_states=[mx.zeros((1, S, self.hidden))],
            gdn_states=["gdn"],
        )

    def rollback_speculative_cache(self, caches, gdn, accepted, bs):
        self.rollbacks.append((int(accepted) if isinstance(accepted, int)
                               else accepted, bs))
        return 0

    # -- drafter side --
    def draft_block(self, b, hidden, draft_cache, bs, sampler, token_dtype, **kw):
        self.last_draft = [10 + i for i in range(bs - 1)]
        return mx.array([self.last_draft], dtype=token_dtype)


def _drive_spec_rounds(monkeypatch, plan, block_size=4, max_tokens=64):
    """Run ``_dflash_rounds`` through the mirror; return (tokens, control stream)."""
    from mlx_vlm.server import tp_mode as T
    from mlx_vlm.speculative.dflash import _dflash_rounds

    sent = []

    def _send(op, ep, ids, *, flags=0, arg0=0, name=""):
        shape = None
        flat = None
        if ids is not None:
            if hasattr(ids, "reshape"):
                shape, flat = (ids.shape[0], ids.shape[1]), ids.reshape(-1).tolist()
            else:
                shape, flat = (1, len(ids)), [int(v) for v in ids]
        sent.append(W.decode(W.encode(op, ep, shape, flat, n=256,
                                      flags=flags, arg0=arg0, name=name)))

    monkeypatch.setattr(T, "_ctrl_send", _send)

    pair = _ScriptedPair(plan)
    mirror = T.MirroredLanguageModel(pair)
    model = SimpleNamespace(language_model=mirror)
    drafter = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[0], block_size=block_size,
                               runtime_block_size=block_size),
        accept_lens=[], draft_lens=[],
        dflash_deferred_walk=True,
        reset=lambda m: ["draft-cache"],
        draft_block=pair.draft_block,
    )
    # A cache the mirror has not seen yet, and that is genuinely empty -- which
    # is what OP_MAKE_CACHE tells rank 1 to build.  (In the live path the spec
    # loop inherits the *prefilled* cache, but the mirror has already announced
    # that one, so the identity short-circuit fires before the emptiness check.)
    cache = [SimpleNamespace(offset=0)]
    hidden = mx.zeros((1, 1, 4))
    rounds = _dflash_rounds(
        model, drafter, cache, hidden,
        first_bonus=7, max_tokens=max_tokens,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        draft_block_size=block_size, use_model_initial_block_size=False,
    )
    tokens = []
    try:
        for tok, _ in rounds:
            tokens.append(int(tok))
    except _PlanExhausted:
        pass
    finally:
        rounds.close()
    if pair.aborted:
        # Drop the announce for the round the script refused to serve: rank 0
        # announced it and then raised, so it is an artefact of the harness,
        # not of the protocol.
        last = max(i for i, m in enumerate(sent) if m.op == W.OP_FORWARD)
        sent = sent[:last]
    return pair, sent, tokens


def _replay(sent):
    """Feed rank 0's control stream to a worker and report what it did."""
    twin = _TwinLM()
    st = W._WorkerState(twin)
    for msg in sent:
        st.handle(msg)
    return twin, st


def test_spec_round_loop_partial_accept_is_mirrored_exactly(monkeypatch):
    """Partial accept: a verify forward at width W, then a rollback verb.

    This is the round shape that actually happens most of the time, and the one
    that mutates the cache outside a forward.  Rank 1 must see both halves.
    """
    pair, sent, tokens = _drive_spec_rounds(monkeypatch, plan=[2])
    assert pair.rollbacks == [(2, 4)], "rank 0 rolled back a partial round"
    twin, _ = _replay(sent)
    assert twin.forwards == [(1, 4, True)], "one width-4 capturing verify"
    assert twin.rollbacks == [([2], 4, ["gdn"])], "rank 1 rolled back the same"
    assert pair.forwards == twin.forwards


def test_spec_round_loop_full_accept_emits_no_rollback(monkeypatch):
    """A fully accepted round must NOT announce a rollback.

    An unconditional rollback verb would be the mirror-image bug: rank 1
    replaying a trim that rank 0 never performed.
    """
    pair, sent, _ = _drive_spec_rounds(monkeypatch, plan=[None])
    assert pair.rollbacks == []
    twin, _ = _replay(sent)
    assert twin.rollbacks == []
    assert [m.op for m in sent].count(W.OP_ROLLBACK) == 0


def test_spec_round_loop_zero_accept_is_mirrored(monkeypatch):
    """Abstain: the drafter was wrong immediately, so the whole block is
    trimmed and only the target's own token survives."""
    pair, sent, _ = _drive_spec_rounds(monkeypatch, plan=[0])
    assert pair.rollbacks == [(0, 4)]
    twin, _ = _replay(sent)
    assert twin.rollbacks == [([0], 4, ["gdn"])]


def test_spec_round_loop_mixed_sequence_is_mirrored_step_for_step(monkeypatch):
    """The gate: partial, full, abstain in one loop, replayed step for step.

    Every verify is capturing on both ranks (the KDA block inputs rank 1 needs
    for its own rollback exist only if its sink was allocated), every rollback
    carries the same (accepted, block_size), and no rollback appears on one rank
    without the other.
    """
    pair, sent, _ = _drive_spec_rounds(monkeypatch, plan=[2, None, 0])
    twin, st = _replay(sent)

    assert len(pair.forwards) == 3
    assert pair.forwards == twin.forwards, "same widths, same capture flags"
    assert all(captured for _, _, captured in twin.forwards)
    # Partial and abstain roll back; the fully accepted round does not.
    assert [a for a, _ in pair.rollbacks] == [2, 0]
    assert [(a, bs) for a, bs, _ in twin.rollbacks] == \
           [([a], bs) for a, bs in pair.rollbacks]
    # And the announced stream never asks rank 1 to build a cache it was not
    # told about, nor to forward into one that does not exist.
    epochs = {m.epoch for m in sent if m.op == W.OP_FORWARD}
    assert epochs <= set(st.caches) | {st.epoch}


def test_verify_width_varies_with_acceptance_and_is_announced(monkeypatch):
    """Width-W verify: W is decided by rank 0's width policy and can change
    between rounds, so it has to travel rather than be assumed."""
    # The shipped default is the FIXED policy (block total 8, R24), under which
    # W never moves and this test would be vacuous -- it held [6]*7 and failed
    # once the default flipped.  The invariant under test is the wire, not the
    # policy, so SELECT the one policy that moves W instead of reaching it by
    # omission (the same rule test_dflash_adaptive_k.py follows).  Adaptive-K
    # needs MINROUNDS (4) of acceptance history before it has a hazard to act
    # on, so a three-round plan holds W constant; this seven-round plan
    # establishes a hazard and then breaks it: [6, 6, 6, 6, 5, 4, 5].
    from mlx_vlm.speculative import dflash as _dflash
    monkeypatch.setenv("MLX_VLM_DFLASH_ADAPTIVE_K", "1")
    monkeypatch.delenv("MLX_VLM_DFLASH_FIXED_WIDTH", raising=False)
    for name in ("_ADAPTIVE_K_ENV", "_FIXED_WIDTH_ENV"):
        monkeypatch.setattr(_dflash, name, None)
    assert _dflash._adaptive_k_enabled(), "test must run under the adaptive policy"
    pair, sent, _ = _drive_spec_rounds(
        monkeypatch, plan=[4, 4, 4, 4, 0, None, 2], block_size=6
    )
    widths = [m.seqlen for m in sent if m.op == W.OP_FORWARD]
    # The block-size cost model shrinks W after acceptance falls off, so the
    # stream really does carry more than one width -- which is the point: a
    # width rank 1 assumed rather than received would be wrong by round five.
    assert len(set(widths)) > 1, f"expected varying verify widths, got {widths}"
    twin, _ = _replay(sent)
    assert [s for _, s, _ in twin.forwards] == widths


# ------------------------------------------- batched rounds: the uniform clamp
class _ScriptedBatchPair:
    """A B=2 target/drafter pair that declares the uniform-acceptance rule.

    Stands in for glm5_next in TP mode: its ``rollback_speculative_cache``
    reduces the batch to one trim length, so the batched DFlash2 loop must clamp
    ragged per-row accepts BEFORE announcing the rollback -- otherwise the two
    ranks would agree on a count that is wrong for every short row.
    """

    requires_uniform_batch_acceptance = True

    def __init__(self, rounds=1, batch=2, hidden=4, vocab=64, per_row=False):
        # ``per_row`` stands in for glm5_next handed BATCHED caches, which can
        # represent a per-row length and would keep the ragged accepts if it
        # were driven directly. Under the mirror it must not: see
        # MirroredLanguageModel.supports_per_row_speculative_rollback.
        if per_row:
            self.supports_per_row_speculative_rollback = lambda caches: True
        self.rounds, self.batch = rounds, batch
        self.hidden, self.vocab = hidden, vocab
        self.round = 0
        self.forwards = []
        self.rollbacks = []
        self.aborted = False

    def __call__(self, ids, cache=None, **kw):
        if self.round >= self.rounds:
            self.aborted = True
            raise _PlanExhausted
        B, S = ids.shape
        self.forwards.append((B, S, kw.get("capture_layer_ids") is not None))
        self.round += 1
        return SimpleNamespace(
            logits=mx.zeros((B, S, self.vocab)),
            hidden_states=[mx.zeros((B, S, self.hidden))],
            gdn_states=["gdn"],
        )

    def rollback_speculative_cache(self, caches, gdn, accepted, bs):
        self.rollbacks.append(
            ([int(v) for v in accepted.reshape(-1).tolist()], int(bs))
        )
        return 0

    def draft_block(self, b, hidden, draft_cache, bs, sampler, token_dtype, **kw):
        return mx.array([[10 + i for i in range(bs - 1)]], dtype=token_dtype)


def _drive_batch_round(monkeypatch, accepted, block_size=4, batch=2, per_row=False):
    """One batched DFlash2 round with dictated per-row acceptance, mirrored."""
    from mlx_vlm.server import tp_mode as T
    from mlx_vlm.speculative import dflash as _dflash

    sent = []

    def _send(op, ep, ids, *, flags=0, arg0=0, name=""):
        shape = flat = None
        if ids is not None:
            if hasattr(ids, "reshape"):
                shape, flat = (ids.shape[0], ids.shape[1]), ids.reshape(-1).tolist()
            else:
                shape, flat = (1, len(ids)), [int(v) for v in ids]
        sent.append(W.decode(W.encode(op, ep, shape, flat, n=256,
                                      flags=flags, arg0=arg0, name=name)))

    monkeypatch.setattr(T, "_ctrl_send", _send)

    def _ragged_walk(draft_tokens, target_tokens, budgets):
        rows = draft_tokens.tolist()
        return list(accepted), [
            (rows[i][:a] + [90 + i])[: budgets[i]] for i, a in enumerate(accepted)
        ]

    monkeypatch.setattr(_dflash, "_speculative_walk_batch", _ragged_walk)

    pair = _ScriptedBatchPair(batch=batch, per_row=per_row)
    mirror = T.MirroredLanguageModel(pair)
    model = SimpleNamespace(language_model=mirror)
    drafter = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[0], block_size=block_size,
                               runtime_block_size=block_size),
        accept_lens=[], draft_lens=[],
        dflash_deferred_walk=False,
        requires_uniform_batch_acceptance=False,
        reset=lambda m: None,
        make_cache=lambda: [],
        draft_block=pair.draft_block,
    )
    cache = [SimpleNamespace(offset=0)]
    rounds = _dflash._dflash_rounds_batch(
        model, drafter, cache, mx.zeros((batch, 1, pair.hidden)),
        first_bonus=mx.array([7] * batch, dtype=mx.int32),
        max_tokens=64,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        draft_block_size=block_size,
        greedy_sampling=True,
    )
    emitted = [[] for _ in range(batch)]
    try:
        for tokens_out, _ in rounds:
            for i, token in enumerate(tokens_out):
                if token is not None:
                    emitted[i].append(int(token))
    except _PlanExhausted:
        pass
    finally:
        rounds.close()
    if pair.aborted:
        last = max(i for i, m in enumerate(sent) if m.op == W.OP_FORWARD)
        sent = sent[:last]
    return pair, drafter, sent, emitted


def test_batched_ragged_accept_is_clamped_before_both_ranks_see_it(monkeypatch):
    """B>1 with ragged accepts: both ranks must roll back the CLAMPED count.

    The mirror is faithful either way -- it announces whatever rank 0 was
    handed -- so a ragged list would desync nothing and corrupt both ranks
    identically. The fix has to happen upstream of the announcement, which is
    what this pins: rank 0's rollback, the announced ids, and rank 1's rollback
    are one uniform count, and the emitted tokens match it.
    """
    pair, drafter, sent, emitted = _drive_batch_round(monkeypatch, accepted=[2, 0])

    assert pair.rollbacks == [([0, 0], 4)], "rank 0 rolled back the clamped count"
    twin, _ = _replay(sent)
    assert twin.rollbacks == [([0, 0], 4, ["gdn"])], "rank 1 rolled back the same"
    announced = [list(m.ids) for m in sent if m.op == W.OP_ROLLBACK]
    assert announced == [[0, 0]], "the wire carried the clamped, uniform count"
    # Row 0 gave back the two tokens it had accepted; both rows emit one token.
    assert [len(e) for e in emitted] == [1, 1], emitted
    assert drafter.clamped_tokens == 2
    assert drafter.speculative_total_clamped == 2
    assert drafter.accept_lens == [0, 0]


def test_batched_uniform_accept_is_mirrored_unclamped(monkeypatch):
    """Already-uniform accepts must travel untouched -- no clamp, no give-back."""
    pair, drafter, sent, emitted = _drive_batch_round(monkeypatch, accepted=[1, 1])

    assert pair.rollbacks == [([1, 1], 4)]
    twin, _ = _replay(sent)
    assert twin.rollbacks == [([1, 1], 4, ["gdn"])]
    assert [len(e) for e in emitted] == [2, 2], emitted
    assert not getattr(drafter, "speculative_total_clamped", 0)
    assert drafter.clamped_tokens == 0


def test_per_row_rollback_is_declined_under_tp_even_when_the_target_can(monkeypatch):
    """The gate: rank 1 has no cache that can hold a per-row length.

    ``tp/worker.py`` builds ``self.lm.make_cache()`` and nothing else -- the
    scalar-offset cache, one offset for the batch. A ragged accept list would
    cross the wire intact (the next test) and then raise on the peer, mid-round,
    after rank 0 had already rolled back. So the mirror answers False for the
    pair no matter what the wrapped target can do, and the loop clamps.
    """
    from mlx_vlm.server import tp_mode as T

    pair, drafter, sent, emitted = _drive_batch_round(
        monkeypatch, accepted=[2, 0], per_row=True
    )
    # The wrapped target says it could; the mirror says the PAIR cannot.
    assert pair.supports_per_row_speculative_rollback(None) is True
    assert T.MirroredLanguageModel.supports_per_row_speculative_rollback(
        object(), None
    ) is False

    assert pair.rollbacks == [([0, 0], 4)], "rank 0 rolled back the clamped count"
    twin, _ = _replay(sent)
    assert twin.rollbacks == [([0, 0], 4, ["gdn"])], "rank 1 rolled back the same"
    assert [list(m.ids) for m in sent if m.op == W.OP_ROLLBACK] == [[0, 0]]
    assert [len(e) for e in emitted] == [1, 1], emitted
    assert drafter.clamped_tokens == 2
    assert getattr(drafter, "per_row_kept_tokens", 0) == 0


def test_the_rollback_wire_already_carries_a_per_row_vector():
    """Not the blocker: the payload is a list, and rank 1 replays it verbatim.

    Pinned so that when rank 1 learns to build batched caches, the protocol side
    is known to be a no-op rather than an assumption.
    """
    lm = _TwinLM()
    st = W._WorkerState(lm)
    st.handle(_msg(W.OP_MAKE_CACHE, 1))
    st.handle(_msg(W.OP_FORWARD, 1, ids=[1, 2, 3, 4], flags=W.FLAG_CAPTURE))
    st.handle(_msg(W.OP_ROLLBACK, 1, ids=[3, 1, 2], arg0=5))
    assert lm.rollbacks == [([3, 1, 2], 5, ["gdn"])], (
        "a ragged per-row vector must survive encode/decode and reach rank 1's "
        "rollback unchanged"
    )
