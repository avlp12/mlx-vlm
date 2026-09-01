"""TP=2 serving mode: toggle, control codec, mirror discipline, teardown.

None of these need a GPU or a second box.  The properties they pin are the ones
that decide whether the branch is safe to merge: that TP-off is untouched, that
the control message rank 1 decodes is exactly what rank 0 encoded, that the
mirror refuses inputs it cannot faithfully replay rather than silently letting
the ranks diverge, that every cache mutation is announced, and that shutdown
actually gives the memory back.
"""

import atexit
import gc
import os
import weakref

import pytest

from mlx_vlm.server import tp_mode as T


@pytest.fixture(autouse=True)
def _clean_env():
    keep = {k: os.environ.get(k) for k in
            (T.ENV_HOSTS, T.ENV_RANK, T.ENV_MAX_TOK, "KV_BITS")}
    for k in keep:
        os.environ.pop(k, None)
    yield
    for k, v in keep.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


# ------------------------------------------------------------------- toggle
def test_off_by_default():
    assert T.tp_enabled() is False
    assert T.tp_hosts() == []


@pytest.mark.parametrize("val,on", [
    ("", False), ("10.0.0.1", False), ("10.0.0.1,10.0.0.2", True),
    (" 10.0.0.1 , 10.0.0.2 ", True), (",,", False),
])
def test_toggle_needs_two_hosts(val, on):
    """A single host is not tensor parallelism; it must not half-enable."""
    os.environ[T.ENV_HOSTS] = val
    assert T.tp_enabled() is on


def test_disabled_load_returns_none_without_importing_transport():
    assert T.maybe_load_tp("/nonexistent") is None


# -------------------------------------------------------------------- codec
def test_codec_roundtrip():
    flat = [5, 6, 7, 8, 9, 10]
    row = T.encode(T.OP_FORWARD, 3, (2, 3), flat, n=64)
    assert len(row) == T.HEADER + 64
    c = T.decode(row)
    assert (c.op, c.epoch, c.batch, c.seqlen, c.ids) == (T.OP_FORWARD, 3, 2, 3, flat)
    assert (c.flags, c.arg0, c.name) == (0, 0, "")


def test_codec_control_only_message():
    c = T.decode(T.encode(T.OP_MAKE_CACHE, 7, None, None, n=32))
    assert (c.op, c.epoch, c.batch, c.seqlen, c.ids) == (T.OP_MAKE_CACHE, 7, 0, 0, None)


def test_codec_pads_and_does_not_leak_previous_payload():
    """Every message is the same width and fully zero-filled, so a short forward
    can never be read as a longer stale one."""
    row = T.encode(T.OP_FORWARD, 1, (1, 2), [11, 12], n=16)
    assert row[T.HEADER + 2:] == [0] * 14
    assert T.decode(row).ids == [11, 12]


def test_codec_rejects_oversized_forward():
    with pytest.raises(T.TPUnavailable, match="exceeds"):
        T.encode(T.OP_FORWARD, 1, (1, 100), list(range(100)), n=64)


def test_max_tok_env_is_honoured_and_floored():
    os.environ[T.ENV_MAX_TOK] = "128"
    assert T._max_tok() == 128
    os.environ[T.ENV_MAX_TOK] = "4"
    assert T._max_tok() == 64          # floor
    os.environ[T.ENV_MAX_TOK] = "junk"
    assert T._max_tok() == 8192        # default on garbage


# ------------------------------------------------- codec: the new verb fields
def test_codec_carries_flags_and_arg0():
    c = T.decode(T.encode(T.OP_ROLLBACK, 4, (1, 2), [3, 3], n=32,
                          flags=T.FLAG_CAPTURE, arg0=8))
    assert c.op == T.OP_ROLLBACK and c.arg0 == 8 and c.ids == [3, 3]
    assert c.capture is True


@pytest.mark.parametrize("name", [
    "0" * 32, "f" * 32, "0123456789abcdef0123456789abcdef",
    "00000000000000000000000000000001",
])
def test_boundary_name_roundtrip(name):
    """A 128-bit rung name survives the int32 vector exactly.

    Sixteen bits per word is not arbitrary: the vector is *summed*, and rank 1
    contributes zeros, so every word must stay far inside int32 for the sum to
    reproduce rank 0's word rather than wrap.
    """
    words = T.name_to_words(name)
    assert len(words) == T.NAME_WORDS
    assert all(0 <= w <= 0xFFFF for w in words)
    assert T.words_to_name(words) == name
    assert T.decode(T.encode(T.OP_VAULT_RESTORE, 1, None, None, n=16,
                             name=name)).name == name


def test_boundary_name_rejects_the_wrong_shape():
    """A truncated or non-hex name must not be silently zero-padded into a
    different rung: two different prompts would then share a checkpoint."""
    for bad in ("abc", "z" * 32, "0" * 31, "0" * 33):
        with pytest.raises(T.TPUnavailable, match="hex"):
            T.name_to_words(bad)


def test_all_verbs_fit_the_same_fixed_width():
    """Every message is the same length regardless of verb -- that is what makes
    the control plane safe to carry on a collective."""
    widths = {
        len(T.encode(op, 1, None, None, n=64, arg0=3, name="a" * 32))
        for op in (T.OP_EXIT, T.OP_MAKE_CACHE, T.OP_FORWARD, T.OP_ROLLBACK,
                   T.OP_VAULT_STORE, T.OP_VAULT_RESTORE)
    }
    assert widths == {T.HEADER + 64}


# ------------------------------------------------------------------- mirror
class _FakeLM:
    def __init__(self):
        self.calls = []
        self.config = "cfg"
        self.rollbacks = []

    def __call__(self, inputs=None, cache=None, **kw):
        self.calls.append((inputs, id(cache), kw.get("capture_layer_ids")))
        return "out"

    def make_cache(self):
        return []

    def rollback_speculative_cache(self, caches, gdn, accepted, bs):
        self.rollbacks.append((accepted, bs))
        return 0


class _Ids:
    def __init__(self, b, s):
        self.shape = (b, s)

    def reshape(self, *_):
        return self

    def tolist(self):
        return list(range(self.shape[0] * self.shape[1]))


def _mirror(monkeypatch, lm=None):
    sent = []

    def _send(op, ep, ids, *, flags=0, arg0=0, name=""):
        sent.append(T.Ctrl(op, ep, 0, 0, ids, flags, arg0, name))

    monkeypatch.setattr(T, "_ctrl_send", _send)
    return T.MirroredLanguageModel(lm or _FakeLM()), sent


def test_mirror_announces_cache_then_forward(monkeypatch):
    m, sent = _mirror(monkeypatch)
    m(_Ids(1, 4), cache=[])
    assert [s.op for s in sent] == [T.OP_MAKE_CACHE, T.OP_FORWARD]


def test_mirror_reannounces_only_when_the_cache_changes(monkeypatch):
    """Decode steps on one cache must not re-create it on rank 1; a new cache
    must, or rank 1 would keep decoding into the previous conversation."""
    m, sent = _mirror(monkeypatch)
    c1, c2 = [], []
    m(_Ids(1, 4), cache=c1)
    m(_Ids(1, 1), cache=c1)
    m(_Ids(1, 1), cache=c1)
    assert [s.op for s in sent] == [T.OP_MAKE_CACHE] + [T.OP_FORWARD] * 3
    sent.clear()
    m(_Ids(1, 6), cache=c2)
    assert [s.op for s in sent] == [T.OP_MAKE_CACHE, T.OP_FORWARD]


def test_mirror_epoch_increments_per_cache(monkeypatch):
    m, sent = _mirror(monkeypatch)
    c1, c2 = [], []                      # kept alive: see the id()-reuse note
    m(_Ids(1, 2), cache=c1)
    e1 = sent[0].epoch
    sent.clear()
    m(_Ids(1, 2), cache=c2)
    assert sent[0].epoch == e1 + 1


def test_mirror_survives_id_reuse(monkeypatch):
    """Regression: a freed cache can hand its address to the next one.  Matching
    on id() alone would skip MAKE_CACHE and leave rank 1 on the old cache."""
    m, sent = _mirror(monkeypatch)
    m(_Ids(1, 2), cache=[])              # first cache becomes garbage
    sent.clear()
    for _ in range(50):                  # force address reuse
        c = []
        m(_Ids(1, 2), cache=c)
    assert [s.op for s in sent].count(T.OP_MAKE_CACHE) == 50


@pytest.mark.parametrize("kw", [{"inputs_embeds": object()}])
def test_mirror_refuses_unreplayable_forwards(monkeypatch, kw):
    """Refusing is the point: silently running these would desynchronise the
    ranks, and a desynchronised collective hangs rather than errors."""
    m, _ = _mirror(monkeypatch)
    with pytest.raises(T.TPUnavailable):
        m(_Ids(1, 2), cache=[], **kw)


def test_mirror_refuses_when_inputs_missing(monkeypatch):
    m, _ = _mirror(monkeypatch)
    with pytest.raises(T.TPUnavailable):
        m(None, cache=[])


def test_mirror_delegates_attributes(monkeypatch):
    m, _ = _mirror(monkeypatch)
    assert m.config == "cfg"
    assert m.make_cache() == []


# ------------------------------------------------ the populated-cache refusal
class _Populated:
    """A cache entry that has already seen tokens."""

    def __init__(self, offset=7):
        self.offset = offset


class _ArraysLike:
    def __init__(self, filled):
        self.cache = [object() if filled else None, None]


def test_empty_cache_detection():
    assert T._cache_is_empty([]) is True
    assert T._cache_is_empty([_Populated(0), _ArraysLike(False)]) is True
    assert T._cache_is_empty([_Populated(7)]) is False
    assert T._cache_is_empty([_ArraysLike(True)]) is False


def test_unknown_cache_kind_is_treated_as_populated():
    """A cache type this file has never seen must not be assumed empty: guessing
    would turn a new upstream cache class into a silent desync."""
    assert T._cache_is_empty([object()]) is False


def test_mirror_refuses_a_cache_that_arrives_already_populated(monkeypatch):
    """The continuous-batching desync, caught structurally.

    ``generate/ar.py::_extend_cache`` merges a joining request's cache into the
    running batch in place and returns a NEW list.  The mirror sees an unknown
    cache and would announce OP_MAKE_CACHE -- which tells rank 1 to build an
    EMPTY one, while rank 0 carries the first request's history.  The two ranks
    would then sum halves of different computations and produce fluent nonsense
    (observed live on 2026-08-31: one concurrent request returned, the other
    timed out at 90 s).  Refuse instead.
    """
    m, sent = _mirror(monkeypatch)
    m(_Ids(1, 4), cache=[_Populated(0)])          # fresh: fine
    sent.clear()
    with pytest.raises(T.TPDesync, match="already populated"):
        m(_Ids(1, 1), cache=[_Populated(12)])     # merged batch: refused
    assert sent == []


# ------------------------------------------------------------- inputs_embeds
class _EmbLM(_FakeLM):
    """A model whose inputs_embeds really is embed_tokens(inputs)."""

    class _Inner:
        def __init__(self, out):
            self._out = out

        def embed_tokens(self, ids):
            return self._out

    def __init__(self, emb_out):
        super().__init__()
        self.model = self._Inner(emb_out)


def test_prefill_embeds_are_accepted_when_reconstructible(monkeypatch):
    """generate_step passes inputs_embeds on EVERY prefill, so refusing it
    outright would refuse every request.  For text-only it is exactly
    embed_tokens(inputs), which rank 1 recomputes from the broadcast ids."""
    import mlx.core as mx

    ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    emb = mx.zeros((1, 3, 8))
    m, sent = _mirror(monkeypatch, _EmbLM(emb))
    m(ids, cache=[], inputs_embeds=emb)
    assert [s.op for s in sent] == [T.OP_MAKE_CACHE, T.OP_FORWARD]


def test_prefill_embeds_refused_when_not_reconstructible(monkeypatch):
    """Multimodal prefill splices image embeddings in; rank 1 cannot rebuild
    that from ids, so it must be refused rather than silently desynced."""
    import mlx.core as mx

    ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    m, _ = _mirror(monkeypatch, _EmbLM(mx.zeros((1, 3, 8))))
    with pytest.raises(T.TPUnavailable, match="multimodal"):
        m(ids, cache=[], inputs_embeds=mx.ones((1, 3, 8)))


def test_embeds_verdict_is_not_cached(monkeypatch):
    """Regression: whether a prefill is multimodal is a property of the REQUEST.

    An earlier revision cached the verdict per model.  On a VLM checkpoint that
    means the first text-only request licenses every later image request --
    rank 1 would be handed ids it cannot turn back into the spliced embedding,
    and the ranks would sum halves of different forwards.  The check is cheap
    (one embedding gather against 45 MoE layers); paying it every prefill is the
    correct trade.
    """
    import mlx.core as mx

    ids = mx.array([[1, 2]], dtype=mx.int32)
    text_emb = mx.zeros((1, 2, 4))
    m, _ = _mirror(monkeypatch, _EmbLM(text_emb))
    cache = []
    m(ids, cache=cache, inputs_embeds=text_emb)                 # text: accepted
    with pytest.raises(T.TPUnavailable, match="multimodal"):
        m(ids, cache=cache, inputs_embeds=mx.ones((1, 2, 4)))   # image: refused


# --------------------------------------------------------- speculative verbs
def test_capture_forward_sets_the_capture_flag(monkeypatch):
    """A speculative verify must be capturing on BOTH ranks.

    Not for the hidden states -- only rank 0's drafter reads those -- but
    because ``gdn_sink is not None`` is what makes each KDA layer stash the
    block inputs its own half needs in order to roll back.  Rank 1 is told with
    a flag, not the layer id list: the ids belong to the drafter, which is
    rank-0-only by design.
    """
    m, sent = _mirror(monkeypatch)
    m(_Ids(1, 8), cache=[], capture_layer_ids=[5, 24, 42])
    fwd = [s for s in sent if s.op == T.OP_FORWARD]
    assert len(fwd) == 1 and fwd[0].capture is True
    m(_Ids(1, 1), cache=m._last_cache_obj)
    assert [s for s in sent if s.op == T.OP_FORWARD][-1].capture is False


def test_rollback_is_announced_before_it_is_applied(monkeypatch):
    """The rejected-round rollback mutates the cache OUTSIDE a forward.

    Attribute access falls through to the wrapped model, so an unintercepted
    ``rollback_speculative_cache`` would roll rank 0 back and leave rank 1
    holding the whole rejected block.  Nothing but the two integers crosses:
    the KDA recurrence is head-split and the DSA latent is replicated, so each
    rank already owns everything its own rollback needs.
    """
    lm = _FakeLM()
    m, sent = _mirror(monkeypatch, lm)
    cache = []
    m(_Ids(1, 8), cache=cache, capture_layer_ids=[5])
    sent.clear()
    m.rollback_speculative_cache(cache, ["gdn"], 3, 8)
    assert [s.op for s in sent] == [T.OP_ROLLBACK]
    assert sent[0].ids == [3] and sent[0].arg0 == 8
    assert lm.rollbacks == [(3, 8)], "the local rollback must still happen"


@pytest.mark.parametrize("accepted,expect", [
    (3, [3]), ([2, 3], [2, 3]),
])
def test_rollback_accepts_int_and_sequence(monkeypatch, accepted, expect):
    m, sent = _mirror(monkeypatch)
    cache = []
    m(_Ids(1, 4), cache=cache, capture_layer_ids=[5])
    sent.clear()
    m.rollback_speculative_cache(cache, [], accepted, 4)
    assert sent[0].ids == expect


def test_rollback_on_a_foreign_cache_is_refused(monkeypatch):
    m, _ = _mirror(monkeypatch)
    m(_Ids(1, 4), cache=[], capture_layer_ids=[5])
    with pytest.raises(T.TPDesync, match="not the announced one"):
        m.rollback_speculative_cache([], [], 1, 4)


# --------------------------------------------------------------- vault verbs
def test_vault_store_announces_the_name_and_depth(monkeypatch):
    m, sent = _mirror(monkeypatch)
    m.announce_vault_store("ab" * 16, 8192)
    assert sent[-1].op == T.OP_VAULT_STORE
    assert sent[-1].name == "ab" * 16 and sent[-1].arg0 == 8192


def test_vault_restore_believes_the_peer_ack(monkeypatch):
    """The two vaults evict independently, so "rank 0 has the rung" does not
    imply "rank 1 has it".  Rank 0 must ask, and act on the answer."""
    m, sent = _mirror(monkeypatch)
    monkeypatch.setattr(T, "_ack_recv", lambda: 1)
    cache = ["restored"]
    assert m.announce_vault_restore(cache, "cd" * 16, 4096) is True
    assert sent[-1].op == T.OP_VAULT_RESTORE
    # The restored cache is now the announced one, so the next forward must NOT
    # re-announce MAKE_CACHE and wipe rank 1's freshly restored half.
    sent.clear()
    m(_Ids(1, 3), cache=cache)
    assert [s.op for s in sent] == [T.OP_FORWARD]


def test_vault_restore_peer_miss_forces_a_cold_prefill(monkeypatch):
    m, sent = _mirror(monkeypatch)
    monkeypatch.setattr(T, "_ack_recv", lambda: 0)
    cache = ["would-be-restored"]
    assert m.announce_vault_restore(cache, "ef" * 16, 4096) is False
    sent.clear()
    # Forgotten epoch: the next forward starts both ranks from an empty cache.
    m(_Ids(1, 3), cache=[])
    assert [s.op for s in sent] == [T.OP_MAKE_CACHE, T.OP_FORWARD]


# ----------------------------------------------------------------- teardown
def test_shutdown_broadcasts_exit_and_swallows_transport_errors(monkeypatch):
    m, sent = _mirror(monkeypatch)
    m.shutdown()
    assert sent[-1].op == T.OP_EXIT

    def boom(*a, **k):
        raise RuntimeError("link down")
    monkeypatch.setattr(T, "_ctrl_send", boom)
    m2, _ = _mirror(monkeypatch)
    monkeypatch.setattr(T, "_ctrl_send", boom)
    m2.shutdown()          # teardown must not raise


def test_shutdown_is_idempotent(monkeypatch):
    m, sent = _mirror(monkeypatch)
    m.shutdown()
    m.shutdown()
    assert [s.op for s in sent].count(T.OP_EXIT) == 1


def test_shutdown_releases_the_wired_limit(monkeypatch):
    """``wired_limit`` is a context manager the loader ENTERS and must exit.

    Held as a bare local it is collected the moment the loader returns and the
    generator's ``finally`` quietly restores the old limit -- so the process
    that believes it wired the model has not.  Owning it makes both the raise
    and the release explicit, and the release ordered after the peer EXIT.
    """
    exits = []

    class _Wire:
        def __exit__(self, *a):
            exits.append(True)

    m, _ = _mirror(monkeypatch)
    m._wire = _Wire()
    m.shutdown()
    assert exits == [True]


def test_shutdown_drops_the_model_reference(monkeypatch):
    """The freeze test: after shutdown nothing here may still own the shard.

    On 2026-08-31 a server hung in shutdown with 183 GiB resident and the next
    load froze the box.  A mirror that keeps ``self._lm`` alive defeats the
    server's unload no matter how carefully the caches are popped.
    """
    lm = _FakeLM()
    m, _ = _mirror(monkeypatch, lm)
    ref = weakref.ref(lm)
    m(_Ids(1, 2), cache=[])
    m.shutdown()
    del lm
    gc.collect()
    assert ref() is None, "the mirror is still holding the model after shutdown"


def test_atexit_hook_does_not_pin_the_model(monkeypatch):
    """Regression: ``atexit.register(self.shutdown)`` stores a BOUND METHOD.

    That is a strong reference to the mirror, and through it to every weight
    tensor -- for the life of the interpreter.  Every unload the server performs
    would then free nothing at all.  The hook must hold a weakref.
    """
    lm = _FakeLM()
    m, _ = _mirror(monkeypatch, lm)
    ref = weakref.ref(m)
    hook = m._atexit_hook
    del m
    gc.collect()
    try:
        assert ref() is None, "atexit is pinning the mirror (and the model)"
        hook()                       # must be a no-op, not an AttributeError
    finally:
        atexit.unregister(hook)


def test_shutdown_unregisters_its_atexit_hook(monkeypatch):
    m, _ = _mirror(monkeypatch)
    hook = m._atexit_hook
    m.shutdown()
    assert m._atexit_hook is None
    # Re-registering would raise if the original were still queued twice.
    atexit.unregister(hook)


def test_shutdown_tp_is_safe_on_a_plain_model():
    assert T.shutdown_tp(object()) is False


def test_shutdown_tp_stops_the_mirror(monkeypatch):
    m, sent = _mirror(monkeypatch)

    class _Model:
        language_model = m

    assert T.shutdown_tp(_Model()) is True
    assert sent[-1].op == T.OP_EXIT


# ----------------------------------------------------------------- watchdog
def test_watchdog_fires_only_while_a_step_is_in_flight():
    fired = []
    w = T._Watchdog(timeout_s=0.01, poll_s=0.005,
                    on_timeout=lambda label, waited: fired.append(label))
    w.start()
    try:
        import time
        time.sleep(0.05)
        assert fired == [], "an idle server must not be aborted"
        w.arm("forward")
        time.sleep(0.08)
        assert "forward" in fired
        fired.clear()
        w.disarm()
        time.sleep(0.05)
        assert fired == []
    finally:
        w.stop()


# ---------------------------------------------------------------- env guards
def test_kv_quantization_is_refused_rather_than_silently_unmirrored():
    """``maybe_quantize_kv_cache`` rewrites the prompt cache BETWEEN forwards
    (generate/ar.py), and rank 1 is never told.  Refusing costs single-box
    speed; not refusing costs correctness with no symptom."""
    os.environ["KV_BITS"] = "8"
    with pytest.raises(T.TPUnavailable, match="KV cache quantization"):
        T._refuse_unmirrorable_env()


def test_no_kv_quantization_passes():
    T._refuse_unmirrorable_env()


# ----------------------------------------------------------------- fallback
def test_preflight_failure_falls_back_not_raises(monkeypatch):
    os.environ[T.ENV_HOSTS] = "10.0.0.1,10.0.0.2"
    monkeypatch.setattr(T, "launch_worker", lambda *a, **k: None)

    def boom(*a, **k):
        raise RuntimeError("no RDMA device")
    monkeypatch.setattr(T, "preflight", boom)
    assert T.maybe_load_tp("/some/model") is None


def test_worker_launch_failure_falls_back(monkeypatch):
    os.environ[T.ENV_HOSTS] = "10.0.0.1,10.0.0.2"

    def boom(*a, **k):
        raise OSError("ssh: connect failed")
    monkeypatch.setattr(T, "launch_worker", boom)
    assert T.maybe_load_tp("/some/model") is None


def test_a_busy_fleet_falls_back_instead_of_freezing_the_box(monkeypatch):
    """The 2026-08-31 freeze, made impossible to repeat by accident.

    A leftover 183 GiB resident plus a fresh 94 GiB shard does not fit in
    512 GiB, and macOS wires the model rather than swapping it, so the failure
    mode is a hard freeze rather than a slowdown.
    """
    os.environ[T.ENV_HOSTS] = "10.0.0.1,10.0.0.2"
    from mlx_vlm.tp import fleet

    def busy(*a, **k):
        raise fleet.HeavyRunActive("pid 999 183.0GB python -m mlx_vlm.server")
    monkeypatch.setattr(fleet, "require_quiet_fleet", busy)
    assert T.maybe_load_tp("/some/model") is None


def test_launch_worker_command_shape(monkeypatch):
    os.environ[T.ENV_HOSTS] = "10.0.0.1,10.0.0.2"
    seen = {}

    class P:
        def __init__(self, cmd):
            seen["cmd"] = cmd
    monkeypatch.setattr(T.subprocess, "Popen", P)
    T.launch_worker("/models/quasar", T.tp_hosts())
    cmd = seen["cmd"]
    assert cmd[0] == "ssh"
    assert cmd[3] == "m3ms@10.0.0.2"
    inner = cmd[4]
    assert "mlx_vlm.tp.worker" in inner
    assert f"{T.ENV_RANK}=1" in inner
    assert "/models/quasar" in inner


def test_watchdog_covers_the_forward_not_just_the_announce():
    """The 19-minute hang.

    A collective-count mismatch stalls INSIDE the forward's 101 reduces, not in
    the one-collective announcement before them.  A watchdog that disarms when
    the control send returns is disarmed for exactly the window that can hang.
    """
    import time

    fired = []
    w = T._Watchdog(timeout_s=0.02, poll_s=0.005,
                    on_timeout=lambda label, waited: fired.append(label))
    w.start()
    try:
        class _SlowLM(_FakeLM):
            def __call__(self, inputs=None, cache=None, **kw):
                time.sleep(0.12)          # a forward that never comes back
                return "out"

        sent = []
        import mlx_vlm.server.tp_mode as M
        real = M._ctrl_send
        M._ctrl_send = lambda op, ep, ids, **k: sent.append(op)
        try:
            m = T.MirroredLanguageModel(_SlowLM(), watchdog=w)
            m(_Ids(1, 4), cache=[])
        finally:
            M._ctrl_send = real
        assert any("forward" in f for f in fired), (
            f"watchdog must fire during the forward, got {fired}")
    finally:
        w.stop()


def test_guard_disarms_even_when_the_forward_raises():
    fired = []
    w = T._Watchdog(timeout_s=5, poll_s=0.01,
                    on_timeout=lambda label, waited: fired.append(label))
    m = T.MirroredLanguageModel(_FakeLM(), watchdog=w)
    try:
        with m._guard("x"):
            raise ValueError("boom")
    except ValueError:
        pass
    assert m._watchdog._inflight is None


def test_a_wedged_gpu_falls_back_instead_of_hanging(monkeypatch):
    """Memory is not the only way a box goes unusable.

    Measured 2026-09-01: a 4x4 mx.eval never returned with 302 GB free and
    nothing resident.  Every memory check passed.  Loading into that would have
    hung for the whole step timeout and then aborted, leaking a shard on the
    way out.
    """
    os.environ[T.ENV_HOSTS] = "10.0.0.1,10.0.0.2"
    from mlx_vlm.tp import fleet

    monkeypatch.setattr(fleet, "require_quiet_fleet", lambda *a, **k: {})
    monkeypatch.setattr(fleet, "gpu_responsive", lambda *a, **k: False)
    assert T.maybe_load_tp("/some/model") is None


def test_gpu_check_can_be_skipped(monkeypatch):
    os.environ[T.ENV_HOSTS] = "10.0.0.1,10.0.0.2"
    os.environ["MLX_VLM_GLM5_TP_SKIP_GPU_CHECK"] = "1"
    try:
        called = []
        from mlx_vlm.tp import fleet

        monkeypatch.setattr(fleet, "gpu_responsive",
                            lambda *a, **k: called.append(1) or True)
        T._require_live_gpus(["10.0.0.1", "10.0.0.2"])
        assert called == []
    finally:
        os.environ.pop("MLX_VLM_GLM5_TP_SKIP_GPU_CHECK", None)


def test_unset_passthrough_vars_are_not_forwarded_as_empty(monkeypatch):
    """Regression: ``NAME=`` in the worker's env is not the same as unset.

    glm5_next parses the gather gate with int(), so an empty string is
    int('') -> ValueError at import, rank 1 dies before it can join, and rank 0
    waits on a peer that never arrives until the watchdog fires.  Observed
    2026-09-01: every TP run that did not happen to set the gate was broken
    this way, and the lc arms hid it because they always set it.
    """
    os.environ[T.ENV_HOSTS] = "10.0.0.1,10.0.0.2"
    os.environ.pop("MLX_VLM_GLM5_GATHER_MIN_CONTEXT", None)
    seen = {}

    class P:
        def __init__(self, cmd):
            seen["cmd"] = cmd
    monkeypatch.setattr(T.subprocess, "Popen", P)

    T.launch_worker("/m", T.tp_hosts())
    assert "GATHER_MIN_CONTEXT" not in seen["cmd"][4]

    monkeypatch.setenv("MLX_VLM_GLM5_GATHER_MIN_CONTEXT", "65536")
    T.launch_worker("/m", T.tp_hosts())
    assert "MLX_VLM_GLM5_GATHER_MIN_CONTEXT=65536" in seen["cmd"][4]


def test_no_passthrough_var_is_ever_emitted_empty(monkeypatch):
    """The general form: nothing may reach the worker as NAME= ."""
    os.environ[T.ENV_HOSTS] = "10.0.0.1,10.0.0.2"
    for k in ("MLX_VLM_GLM5_TP_TRACE", "MLX_VLM_GLM5_TP_TRACE_DEEP",
              "MLX_VLM_GLM5_IDX_FAST", "MLX_VLM_GLM5_SYNC_TRACE",
              "MLX_VLM_GLM5_GATHER_MIN_CONTEXT", "MLX_VLM_GLM5_VAULT"):
        os.environ.pop(k, None)
    seen = {}

    class P:
        def __init__(self, cmd):
            seen["cmd"] = cmd
    monkeypatch.setattr(T.subprocess, "Popen", P)
    T.launch_worker("/m", T.tp_hosts())
    assert "= " not in seen["cmd"][4].replace("' ", "").replace("= '", "")
