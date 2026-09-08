"""DFlash2 under TP=2 must keep emitting: the L3B path, both ranks, on CPU.

Regression guard for the 2026-09-08 D5 arm, where rank 0 stopped driving 140
tokens into a 256-token request while rank 1 sat parked in its control wait.
The question this file answers is narrow and mechanical: does the MTP/peer-gone
work change what DFlash2 announces, what rank 1 runs, or whether tokens reach
the queue the server streams from?

It runs the REAL ``_dflash_rounds`` through ``MirroredLanguageModel`` on one
thread and the REAL ``_WorkerState.handle(_ctrl_recv())`` loop on another, over
the positional two-rank wire in ``tp_wire``.  A round that fails to emit, an
announcement rank 1 does not receive, or a collective count that drifts all
surface here as a hang the wire bounds and names.
"""

import queue
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mlx_vlm.tp.worker as W
from mlx_vlm.server import tp_mode as T
from mlx_vlm.tests import tp_wire

mx.set_default_device(mx.cpu)

REDUCES_PER_FORWARD = 3     # stands in for the 101 of a real sharded step


class _PlanExhausted(Exception):
    pass


class _Target:
    """Rank 0's target: acceptance dictated per round, plus its own reduces."""

    def __init__(self, wire, plan, hidden=4, vocab=64):
        self.wire, self.plan, self.round = wire, plan, 0
        self.hidden, self.vocab = hidden, vocab
        self.last_draft = None
        self.forwards, self.rollbacks = [], []
        # The drafter reaches through the mirror for these on reset(); keeping
        # them here is what proves that path still resolves.
        self.model = SimpleNamespace(embed_tokens=lambda x: x, layers=[])

    def make_cache(self):
        return [SimpleNamespace(offset=0)]

    def _onehot(self, row):
        idx = mx.array(row, dtype=mx.int32)
        return (mx.zeros((1, len(row), self.vocab))
                + (mx.arange(self.vocab)[None, None, :] == idx[None, :, None]) * 10.0)

    def __call__(self, ids, cache=None, **kw):
        if self.round >= len(self.plan):
            raise _PlanExhausted
        S = ids.shape[1]
        self.forwards.append((ids.shape[0], S, kw.get("capture_layer_ids") is not None))
        for _ in range(REDUCES_PER_FORWARD):
            self.wire.data_reduce(0)
        draft = self.last_draft or []
        a = self.plan[self.round]
        a = len(draft) if a is None else a
        row = (list(draft[:a]) + [40 + self.round] + [50] * S)[:S]
        self.round += 1
        return SimpleNamespace(logits=self._onehot(row),
                               hidden_states=[mx.zeros((1, S, self.hidden))],
                               gdn_states=["gdn"])

    def rollback_speculative_cache(self, caches, gdn, accepted, bs):
        self.rollbacks.append((int(accepted), int(bs)))
        return 0

    def draft_block(self, b, hidden, draft_cache, bs, sampler, token_dtype, **kw):
        self.last_draft = [10 + i for i in range(bs - 1)]
        return mx.array([self.last_draft], dtype=token_dtype)


class _Rank1LM:
    """Rank 1's half: the same number of reduces, its own logits to evaluate."""

    def __init__(self, wire, vocab=64):
        self.wire, self.vocab = wire, vocab
        self.forwards, self.rollbacks = [], []

    def make_cache(self):
        return ["rank1-cache"]

    def __call__(self, ids, cache=None, **kw):
        self.forwards.append((ids.shape[0], ids.shape[1],
                              kw.get("capture_layer_ids") is not None))
        for _ in range(REDUCES_PER_FORWARD):
            self.wire.data_reduce(1)
        return SimpleNamespace(logits=mx.zeros((1, ids.shape[1], self.vocab)),
                               gdn_states=["gdn"])

    def rollback_speculative_cache(self, caches, gdn, accepted, bs):
        self.rollbacks.append((list(accepted), int(bs)))
        return 0


def _drive(monkeypatch, plan, block_size=4, max_tokens=24):
    """Both ranks, one DFlash2 request.  Returns everything worth asserting."""
    from mlx_vlm.speculative.dflash import _dflash_rounds

    emitted = queue.Queue()
    holder = {}

    def rank0(wire):
        target = holder["target"] = _Target(wire, plan)
        mirror = holder["mirror"] = T.MirroredLanguageModel(target)
        model = SimpleNamespace(language_model=mirror)
        drafter = SimpleNamespace(
            config=SimpleNamespace(target_layer_ids=[0], block_size=block_size,
                                   runtime_block_size=block_size),
            accept_lens=[], draft_lens=[],
            dflash_deferred_walk=True,
            reset=lambda m: ["draft-cache"],
            draft_block=target.draft_block,
        )
        rounds = _dflash_rounds(
            model, drafter, target.make_cache(), mx.zeros((1, 1, 4)),
            first_bonus=7, max_tokens=max_tokens,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            draft_block_size=block_size, use_model_initial_block_size=False,
        )
        try:
            for tok, _ in rounds:
                # THE ASSERTION SURFACE: this is the server's per-token hand-off
                # (server/generation.py puts each token on the queue the HTTP
                # stream reads).  A round that computes but never emits shows up
                # as a queue that stops growing.
                emitted.put(int(tok))
        except _PlanExhausted:
            pass
        finally:
            rounds.close()
            mirror.shutdown()          # OP_EXIT, so rank 1's loop terminates

    def rank1(wire, R1):
        lm = holder["rank1_lm"] = _Rank1LM(wire)
        state = R1._WorkerState(lm)
        while state.handle(R1._ctrl_recv()):
            pass

    e0, e1 = tp_wire.run_pair(monkeypatch, rank0, rank1)
    tokens = []
    while not emitted.empty():
        tokens.append(emitted.get())
    return e0, e1, tokens, holder


def test_dflash2_under_tp_emits_every_round(monkeypatch):
    """Full-accept rounds: tokens keep reaching the queue, both ranks agree."""
    e0, e1, tokens, h = _drive(monkeypatch, plan=[None] * 12, max_tokens=24)
    assert e0 is None, f"rank 0 stopped driving: {e0!r}"
    assert e1 is None, f"rank 1 refused: {e1!r}"
    # 23, not 24: generate_step yields the first bonus itself and the round
    # loop starts at emitted=1 (speculative/dflash.py).  The number that matters
    # is that it reaches the budget, not that it starts at zero.
    assert len(tokens) == 23, f"emitted {len(tokens)} of 23 round-loop tokens"
    assert h["target"].forwards == h["rank1_lm"].forwards, (
        "the ranks ran different forwards -- rank 1 mirrors ids and the capture "
        "flag, and nothing else may differ")


def test_dflash2_rejections_still_emit_and_roll_both_halves(monkeypatch):
    """Partial acceptance: the rollback is announced and rank 1 performs it."""
    e0, e1, tokens, h = _drive(monkeypatch, plan=[1, 2, None, 0, 1] * 4,
                               max_tokens=20)
    assert e0 is None and e1 is None, f"{e0!r} / {e1!r}"
    assert len(tokens) == 19
    assert h["target"].rollbacks, "the scripted plan must produce rejections"
    assert len(h["rank1_lm"].rollbacks) == len(h["target"].rollbacks), (
        "rank 1 rolled a different number of rounds back than rank 0")


def test_the_mirror_is_still_the_only_thing_that_forwards(monkeypatch):
    """No hidden extra forward: one OP_FORWARD per target call, no more.

    A verify announced twice (or a mirrored hook firing on the DFlash2 path)
    would leave rank 1 running a forward rank 0 never ran, which is a hang one
    verb later rather than a wrong answer.
    """
    e0, e1, tokens, h = _drive(monkeypatch, plan=[None] * 12, max_tokens=16)
    assert e0 is None and e1 is None
    assert len(h["rank1_lm"].forwards) == len(h["target"].forwards)
