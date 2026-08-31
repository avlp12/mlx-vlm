"""TP=2 serving mode: toggle, control codec, mirror discipline, and fallback.

None of these need a GPU or a second box.  The properties they pin are the ones
that decide whether the branch is safe to merge: that TP-off is untouched, that
the control message rank 1 decodes is exactly what rank 0 encoded, that the
mirror refuses inputs it cannot faithfully replay rather than silently letting
the ranks diverge, and that every failure path degrades to single-box instead of
taking the server down.
"""

import os

import pytest

from mlx_vlm.server import tp_mode as T


@pytest.fixture(autouse=True)
def _clean_env():
    keep = {k: os.environ.get(k) for k in
            (T.ENV_HOSTS, T.ENV_RANK, T.ENV_MAX_TOK)}
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
    op, ep, b, s, got = T.decode(row)
    assert (op, ep, b, s, got) == (T.OP_FORWARD, 3, 2, 3, flat)


def test_codec_control_only_message():
    row = T.encode(T.OP_MAKE_CACHE, 7, None, None, n=32)
    assert T.decode(row) == (T.OP_MAKE_CACHE, 7, 0, 0, None)


def test_codec_pads_and_does_not_leak_previous_payload():
    """Every message is the same width and fully zero-filled, so a short forward
    can never be read as a longer stale one."""
    row = T.encode(T.OP_FORWARD, 1, (1, 2), [11, 12], n=16)
    assert row[T.HEADER + 2:] == [0] * 14
    assert T.decode(row)[4] == [11, 12]


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


# ------------------------------------------------------------------- mirror
class _FakeLM:
    def __init__(self):
        self.calls = []
        self.config = "cfg"

    def __call__(self, inputs=None, cache=None, **kw):
        self.calls.append((inputs, id(cache)))
        return "out"

    def make_cache(self):
        return []


class _Ids:
    def __init__(self, b, s):
        self.shape = (b, s)

    def reshape(self, *_):
        return self

    def tolist(self):
        return list(range(self.shape[0] * self.shape[1]))


def _mirror(monkeypatch):
    sent = []
    monkeypatch.setattr(T, "_ctrl_send", lambda op, ep, ids: sent.append((op, ep, ids)))
    return T.MirroredLanguageModel(_FakeLM()), sent


def test_mirror_announces_cache_then_forward(monkeypatch):
    m, sent = _mirror(monkeypatch)
    c = []
    m(_Ids(1, 4), cache=c)
    assert [s[0] for s in sent] == [T.OP_MAKE_CACHE, T.OP_FORWARD]


def test_mirror_reannounces_only_when_the_cache_changes(monkeypatch):
    """Decode steps on one cache must not re-create it on rank 1; a new cache
    must, or rank 1 would keep decoding into the previous conversation."""
    m, sent = _mirror(monkeypatch)
    c1, c2 = [], []
    m(_Ids(1, 4), cache=c1)
    m(_Ids(1, 1), cache=c1)
    m(_Ids(1, 1), cache=c1)
    assert [s[0] for s in sent] == [T.OP_MAKE_CACHE] + [T.OP_FORWARD] * 3
    sent.clear()
    m(_Ids(1, 6), cache=c2)
    assert [s[0] for s in sent] == [T.OP_MAKE_CACHE, T.OP_FORWARD]


def test_mirror_epoch_increments_per_cache(monkeypatch):
    m, sent = _mirror(monkeypatch)
    c1, c2 = [], []                      # kept alive: see the id()-reuse note
    m(_Ids(1, 2), cache=c1)
    e1 = sent[0][1]
    sent.clear()
    m(_Ids(1, 2), cache=c2)
    assert sent[0][1] == e1 + 1


def test_mirror_survives_id_reuse(monkeypatch):
    """Regression: a freed cache can hand its address to the next one.  Matching
    on id() alone would skip MAKE_CACHE and leave rank 1 on the old cache."""
    m, sent = _mirror(monkeypatch)
    m(_Ids(1, 2), cache=[])              # first cache becomes garbage
    sent.clear()
    for _ in range(50):                  # force address reuse
        c = []
        m(_Ids(1, 2), cache=c)
    assert [s[0] for s in sent].count(T.OP_MAKE_CACHE) == 50


@pytest.mark.parametrize("kw", [
    {"inputs_embeds": object()},
    {"capture_layer_ids": [5, 24, 42]},
])
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


def test_shutdown_broadcasts_exit_and_swallows_transport_errors(monkeypatch):
    m, sent = _mirror(monkeypatch)
    m.shutdown()
    assert sent[-1][0] == T.OP_EXIT
    def boom(*a, **k):
        raise RuntimeError("link down")
    monkeypatch.setattr(T, "_ctrl_send", boom)
    m.shutdown()          # teardown must not raise


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


def test_launch_worker_command_shape(monkeypatch):
    os.environ[T.ENV_HOSTS] = "10.0.0.1,10.0.0.2"
    seen = {}
    class P:
        def __init__(self, cmd): seen["cmd"] = cmd
    monkeypatch.setattr(T.subprocess, "Popen", P)
    T.launch_worker("/models/quasar", T.tp_hosts())
    cmd = seen["cmd"]
    assert cmd[0] == "ssh"
    assert cmd[3] == "m3ms@10.0.0.2"
    inner = cmd[4]
    assert "mlx_vlm.server.tp_worker" in inner
    assert f"{T.ENV_RANK}=1" in inner
    assert "/models/quasar" in inner
