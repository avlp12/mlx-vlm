"""MTP under TP=2: every verify must be announced (incident 2026-09-08, I1528).

WHAT HAPPENED.  TP=2 with the MTP drafter (block 3) handshook, loaded both
shards, prefilled 15 tokens, and then produced zero tokens.  Rank 1 died with

    TPDesync: TP shape disagreement on rank 1:
              batch: ranks differ by -1; seqlen: ranks differ by -15

and rank 0 spun at ~200% CPU with its transport socket in CLOSED, ignoring three
SIGTERMs and a POST /unload.  DFlash2 under the same TP=2 works.

WHY.  ``speculative/mtp.py`` does not call the language model.  It asks it for a
hook -- ``getattr(lm, "speculative_verify_hidden")`` at mtp.py:92, reached from
mtp.py:175/188 -- and ``MirroredLanguageModel.__getattr__`` handed back a bound
method of the RAW model.  That method (models/glm5_next/language.py:3612) runs
the whole sharded stack, 101 all_sums, with no OP_FORWARD announced.  Rank 1 was
sitting in its control wait; its control reduce paired with one of rank 0's data
reduces; the reserved agreement words at the tail of the control vector no
longer cancelled, and the numbers it reported (-1 batch, -15 seqlen) are exactly
the negation of the last shape it HAD agreed on: the b=1 s=15 prefill of that
run.  DFlash2 escapes because it verifies through ``lm(...)``
(speculative/dflash.py:1206), i.e. through ``__call__``, which announces.

The tests below are in two layers: a two-rank in-process transport that shows
the desync appearing and then not appearing, and a round-loop test that pins the
announcements ``_mtp_rounds`` now produces.
"""

import threading
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mlx_vlm.tp.worker as W
from mlx_vlm.server import tp_mode as T

mx.set_default_device(mx.cpu)

N = 64          # payload words; keeps the control vector small and readable
CTRL_WORDS = W.HEADER + N


# =============================================================================
# A two-rank transport, in one process
# =============================================================================
class _PeerNeverCame(Exception):
    """A collective whose other half never arrived: the hang, bounded."""


class _Wire:
    """Pair the i-th collective of rank 0 with the i-th of rank 1.

    That pairing IS the protocol: there is no side channel, so "rank 0 issued a
    collective rank 1 did not" cannot be detected by the transport -- it is only
    visible later, as the wrong two buffers meeting.  jaccl does not report a
    size mismatch either (worker.py records 8 elements against 256 completing
    silently and returning 3.0 to both ranks), so the model here is the
    conservative one: each rank gets its own buffer plus whatever prefix of the
    peer's overlaps it.  The property that matters is the one the echo check was
    built for -- the reserved words at the TAIL keep the contributor's own value
    when the peer's buffer is a different length, so they stop cancelling.
    """

    def __init__(self, timeout=10.0):
        self.slots = {}
        self.cv = threading.Condition()
        self.timeout = timeout
        self.pairs = []

    def _index(self, rank):
        return self.slots.setdefault(f"n{rank}", 0)

    def exchange(self, rank, row):
        with self.cv:
            i = self.slots.get(f"n{rank}", 0)
            self.slots[f"n{rank}"] = i + 1
            self.slots[(rank, i)] = list(row)
            self.cv.notify_all()
            peer = 1 - rank
            if not self.cv.wait_for(lambda: (peer, i) in self.slots, self.timeout):
                raise _PeerNeverCame(
                    f"rank {rank} waited {self.timeout}s for the peer's "
                    f"collective #{i}: this is the live hang, in a test")
            other = self.slots[(peer, i)]
        out = list(row)
        for k in range(min(len(out), len(other))):
            out[k] += other[k]
        return out


def _patch_transport(monkeypatch, wire):
    """One ``all_sum`` for both ranks; the caller is identified by thread name."""
    import mlx_vlm.tp.transport as X

    def all_sum(x):
        rank = 0 if threading.current_thread().name == "rank0" else 1
        row = x[0].tolist() if x.ndim == 2 else x.reshape(-1).tolist()
        return mx.array([wire.exchange(rank, [int(v) for v in row])],
                        dtype=mx.int32)

    monkeypatch.setattr(X, "all_sum", all_sum)
    monkeypatch.setattr(X, "driving", lambda: __import__("contextlib").nullcontext())
    monkeypatch.setattr(X, "set_epoch", lambda e: None)


def _rank1_worker_module():
    """A SECOND copy of tp.worker, so rank 1 has its own ``_LAST_SHAPE``.

    The echo agreement is a comparison of two per-rank module globals.  Running
    both ranks against one import would compare a value with itself and could
    never fail -- which is to say it would never reproduce anything.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_tp_worker_rank1", W.__file__)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "mlx_vlm.tp"          # so ``from ..tp.transport`` resolves
    spec.loader.exec_module(mod)
    return mod


def _data_reduce(rank, wire, width=32):
    """One collective from inside a sharded forward: not a control message."""
    wire.exchange(rank, [0] * width)


def _run_pair(monkeypatch, rank0_script, rank1_steps):
    """Drive both ranks to completion; return (rank0 error, rank1 error)."""
    monkeypatch.setenv(W.ENV_MAX_TOK, str(N))
    wire = _Wire()
    _patch_transport(monkeypatch, wire)
    R1 = _rank1_worker_module()
    W._LAST_SHAPE[:] = [0, 0, 0]
    R1._LAST_SHAPE[:] = [0, 0, 0]
    W.clear_peer_gone()
    R1.clear_peer_gone()
    out = {}

    def r0():
        try:
            rank0_script(wire)
        except BaseException as e:            # noqa: BLE001 - reported, not raised
            out["r0"] = e

    def r1():
        try:
            for _ in range(rank1_steps):
                msg = R1._ctrl_recv()
                if msg.op == W.OP_FORWARD:
                    for _ in range(3):        # rank 1's own share of the forward
                        _data_reduce(1, wire)
        except BaseException as e:            # noqa: BLE001
            out["r1"] = e

    t0 = threading.Thread(target=r0, name="rank0")
    t1 = threading.Thread(target=r1, name="rank1")
    t1.start(); t0.start()
    t0.join(20); t1.join(20)
    assert not t0.is_alive() and not t1.is_alive(), "a rank never finished"
    return out.get("r0"), out.get("r1")


def test_an_unannounced_verify_desyncs_rank_1():
    """The incident, reproduced: rank 0 runs a forward it did not announce."""
    mp = pytest.MonkeyPatch()
    try:
        def rank0(wire):
            W._ctrl_send(W.OP_MAKE_CACHE, 1, None)
            ids = mx.arange(15, dtype=mx.int32).reshape(1, 15)
            W._ctrl_send(W.OP_FORWARD, 1, ids)
            for _ in range(3):
                _data_reduce(0, wire)
            # THE BUG: mtp.py -> lm.speculative_verify_hidden -> the raw model.
            for _ in range(3):
                _data_reduce(0, wire)
            W._ctrl_send(W.OP_FORWARD, 1, mx.arange(4, dtype=mx.int32).reshape(1, 4))

        e0, e1 = _run_pair(mp, rank0, rank1_steps=3)
    finally:
        mp.undo()
    # ``TPDesync`` by name: rank 1 runs its own copy of the module (see
    # _rank1_worker_module), so it raises its own copy of the class.
    assert type(e1).__name__ == "TPDesync", f"rank 1 did not refuse: {e1!r}"
    # The live message, from the live prompt length.
    assert "batch: ranks differ by -1" in str(e1)
    assert "seqlen: ranks differ by -15" in str(e1)
    # The live run also carried "; epoch: ranks differ by -1"? It did NOT: it
    # named batch and seqlen only, so rank 0's contribution happened to cancel
    # the epoch word.  What a size-mismatched jaccl reduce leaves in any GIVEN
    # word is not reproducible in-process -- only that the reserved words stop
    # cancelling is -- so this asserts the two terms that name the shape and not
    # the third.
    # And the other half of the incident, in the same run: rank 0 is left
    # inside a collective whose peer has exited.  Live that is an unpreemptible
    # ~200% CPU spin; here the wire bounds it so the test can name it.
    assert isinstance(e0, _PeerNeverCame), (
        f"rank 0 should have been left waiting on the dead peer, got {e0!r}")


def test_an_announced_verify_keeps_the_ranks_paired():
    """The fix: the verify is an OP_FORWARD like any other, so nothing drifts."""
    mp = pytest.MonkeyPatch()
    try:
        def rank0(wire):
            W._ctrl_send(W.OP_MAKE_CACHE, 1, None)
            ids = mx.arange(15, dtype=mx.int32).reshape(1, 15)
            W._ctrl_send(W.OP_FORWARD, 1, ids)
            for _ in range(3):
                _data_reduce(0, wire)
            verify = mx.arange(4, dtype=mx.int32).reshape(1, 4)
            W._ctrl_send(W.OP_FORWARD, 1, verify, flags=W.FLAG_CAPTURE)
            for _ in range(3):
                _data_reduce(0, wire)
            W._ctrl_send(W.OP_FORWARD, 1, verify, flags=W.FLAG_CAPTURE)
            for _ in range(3):
                _data_reduce(0, wire)

        e0, e1 = _run_pair(mp, rank0, rank1_steps=4)
    finally:
        mp.undo()
    assert e0 is None, f"rank 0 refused: {e0!r}"
    assert e1 is None, f"rank 1 refused: {e1!r}"


# =============================================================================
# The round loop: what _mtp_rounds now announces
# =============================================================================
class _ScriptedMTPPair:
    """A target that answers the MTP hooks, with dictated per-round acceptance."""

    def __init__(self, plan, hidden=4, vocab=64):
        self.plan, self.round = plan, 0
        self.hidden, self.vocab = hidden, vocab
        self.last_draft = None
        self.verifies = []          # (batch, seqlen) per verify forward
        self.rollbacks = []
        self.model = SimpleNamespace(layers=[], norm=lambda h: h)

    # -- target side --
    def make_cache(self):
        return ["target-cache"]

    def _row(self, S):
        draft = self.last_draft or []
        a = self.plan[min(self.round, len(self.plan) - 1)]
        a = len(draft) if a is None else a
        row = list(draft[:a]) + [40 + self.round]
        return (row + [50] * S)[:S]

    def speculative_verify_hidden(self, inputs, cache):
        S = inputs.shape[1]
        self.verifies.append((inputs.shape[0], S))
        self._pending = self._row(S)
        self.round += 1
        return mx.zeros((1, S, self.hidden)), {}, ["gdn"]

    def speculative_logits_from_hidden(self, hidden):
        S = hidden.shape[1]
        row = mx.array(self._pending[:S], dtype=mx.int32)
        return (mx.zeros((1, S, self.vocab))
                + (mx.arange(self.vocab)[None, None, :] == row[None, :, None]) * 10.0)

    def speculative_argmax_from_hidden(self, hidden):
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def rollback_speculative_cache(self, caches, gdn, accepted, bs):
        self.rollbacks.append((int(accepted), int(bs)))
        return 0

    # -- drafter side --
    def draft_block(self, b, hidden, draft_cache, bs, sampler, token_dtype, **kw):
        self.last_draft = [10 + i for i in range(bs - 1)]
        return mx.array([self.last_draft], dtype=token_dtype)


def _drive_mtp_rounds(monkeypatch, plan, block_size=4, max_tokens=12):
    """Run the real ``_mtp_rounds`` through the mirror; return (pair, control)."""
    from mlx_vlm.speculative.mtp import _mtp_rounds

    sent = []

    def _send(op, ep, ids, *, flags=0, arg0=0, name=""):
        shape = flat = None
        if ids is not None:
            if hasattr(ids, "reshape") and hasattr(ids, "shape"):
                shape, flat = (ids.shape[0], ids.shape[1]), ids.reshape(-1).tolist()
            else:
                shape, flat = (1, len(ids)), [int(v) for v in ids]
        sent.append(W.decode(W.encode(op, ep, shape, flat, n=N, flags=flags,
                                      arg0=arg0, name=name)))

    monkeypatch.setattr(T, "_ctrl_send", _send)
    pair = _ScriptedMTPPair(plan)
    mirror = T.MirroredLanguageModel(pair)
    model = SimpleNamespace(language_model=mirror)
    drafter = SimpleNamespace(
        config=SimpleNamespace(block_size=block_size,
                               runtime_block_size=block_size),
        accept_lens=[], draft_lens=[],
        prefer_requested_block_size=True,
        reset=lambda m: None,
        set_shared_kv=lambda *a, **k: None,
        draft_block=pair.draft_block,
    )
    rounds = _mtp_rounds(
        model, drafter, [SimpleNamespace(offset=0)], mx.zeros((1, 1, pair.hidden)),
        {}, first_bonus=7, max_tokens=max_tokens,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        draft_block_size=block_size, greedy_sampling=True,
    )
    tokens = []
    try:
        for tok, _ in rounds:
            tokens.append(int(tok))
    finally:
        rounds.close()
        mirror.shutdown()
    return pair, sent, tokens


def test_every_mtp_verify_is_announced(monkeypatch):
    """One OP_FORWARD per verify, with the shape the target actually ran."""
    pair, sent, _ = _drive_mtp_rounds(monkeypatch, plan=[None] * 8)
    fwds = [m for m in sent if m.op == W.OP_FORWARD]
    assert pair.verifies, "the scripted target never verified"
    assert len(fwds) == len(pair.verifies), (
        f"{len(pair.verifies)} verify forwards but {len(fwds)} announcements -- "
        f"rank 1 would run a different number of collectives")
    assert [(m.batch, m.seqlen) for m in fwds] == pair.verifies


def test_the_verify_announcement_carries_capture(monkeypatch):
    """FLAG_CAPTURE or rank 1 has no captured round to roll back.

    ``tp/worker.py`` OP_ROLLBACK refuses outright when ``last_gdn`` is None, and
    only a capturing forward sets it.  A rejected MTP round therefore needs the
    flag on the forward that PRECEDED it, not just the rollback verb.
    """
    _, sent, _ = _drive_mtp_rounds(monkeypatch, plan=[1, 1, 1, None, None])
    fwds = [m for m in sent if m.op == W.OP_FORWARD]
    assert fwds and all(m.capture for m in fwds), \
        "a verify that is not announced as capturing makes OP_ROLLBACK refuse"
    rolls = [m for m in sent if m.op == W.OP_ROLLBACK]
    assert rolls, "a partially-accepted round must announce its rollback"


def test_the_mirror_refuses_a_direct_call_to_the_inner_stack():
    """``lm.model(...)`` is mtp.py's fallback verify (mtp.py:119,129)."""
    pair = _ScriptedMTPPair(plan=[None])
    mirror = T.MirroredLanguageModel(pair)
    assert mirror.model.layers == []          # reads still work
    with pytest.raises(T.TPDesync, match="direct call to language_model.model"):
        mirror.model(mx.zeros((1, 2), dtype=mx.int32))


def test_load_refuses_a_speculative_hook_it_cannot_mirror():
    """A hook nobody has checked is a named refusal, not a forty-minute hang."""
    class _Future:
        def speculative_verify_hidden(self, i, c):
            return None

        def speculative_something_new(self, x):
            return x

    with pytest.raises(T.TPUnavailable, match="speculative_something_new"):
        T._refuse_unmirrored_speculative_hooks(_Future())


def test_load_accepts_the_hooks_glm5_next_actually_has():
    class _Glm5Like:
        def speculative_verify_hidden(self, i, c): ...
        def speculative_verify_logits(self, i, c, s): ...
        def speculative_logits_from_hidden(self, h): ...
        def speculative_argmax_from_hidden(self, h): ...

    T._refuse_unmirrored_speculative_hooks(_Glm5Like())   # must not raise
