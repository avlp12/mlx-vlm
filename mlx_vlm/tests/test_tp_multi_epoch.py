from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.models.cache import KVCache
from mlx_vlm.server import tp_mode as T
from mlx_vlm.tp import worker as W


class _LM:
    def __init__(self):
        self.made = []

    def make_cache(self):
        cache = [KVCache()]
        self.made.append(cache)
        return cache

    def __call__(self, ids, cache=None, **kwargs):
        kv = mx.zeros((ids.shape[0], 1, ids.shape[1], 1))
        cache[0].update_and_fetch(kv, kv)
        return SimpleNamespace(loglaugits=mx.zeros((1, 1, 1)), logits=mx.zeros((1, 1, 1)))


def _msg(op, epoch, ids=None):
    shape = None if ids is None else (1, len(ids))
    return W.decode(W.encode(op, epoch, shape, ids, n=64))


def test_worker_preserves_a_across_b_prefill_then_a_decode():
    lm = _LM()
    state = W._WorkerState(lm)
    state.handle(_msg(W.OP_MAKE_CACHE, 1))
    state.handle(_msg(W.OP_FORWARD, 1, [1, 2]))
    a = state.caches[1]
    state.handle(_msg(W.OP_MAKE_CACHE, 2))
    state.handle(_msg(W.OP_FORWARD, 2, [3]))
    state.handle(_msg(W.OP_FORWARD, 1, [4]))
    assert state.caches[1] is a
    assert a[0].offset == 3
    assert state.caches[2][0].offset == 1


def test_release_unknown_double_and_exit_cleanup_are_fail_closed():
    state = W._WorkerState(_LM())
    state.handle(_msg(W.OP_MAKE_CACHE, 1))
    state.handle(_msg(W.OP_RELEASE_CACHE, 1))
    with pytest.raises(W.TPDesync, match="unknown epoch"):
        state.handle(_msg(W.OP_RELEASE_CACHE, 1))
    with pytest.raises(W.TPDesync, match="unknown cache epoch"):
        state.handle(_msg(W.OP_FORWARD, 1, [1]))
    state.handle(_msg(W.OP_MAKE_CACHE, 2))
    assert state.handle(_msg(W.OP_EXIT, 2)) is False
    assert state.caches == {}


def test_releasing_selected_a_after_b_clears_only_as_capture():
    lm = _LM()
    state = W._WorkerState(lm)
    state.handle(_msg(W.OP_MAKE_CACHE, 1))
    state.handle(_msg(W.OP_MAKE_CACHE, 2))
    state.gdn_by_epoch[1] = ["a"]
    state.last_gdn = state.gdn_by_epoch[1]
    state.handle(_msg(W.OP_FORWARD, 1, [4]))
    state.gdn_by_epoch[1] = ["a"]
    state.last_gdn = state.gdn_by_epoch[1]
    state.handle(_msg(W.OP_RELEASE_CACHE, 1))
    assert state.epoch == -1
    assert state.last_gdn is None and 1 not in state.gdn_by_epoch
    assert 2 in state.caches


def test_rank0_reselects_known_epoch_and_releases_explicitly(monkeypatch):
    sent = []
    monkeypatch.setattr(T, "_ctrl_send", lambda op, epoch, ids, **kw: sent.append((op, epoch)))
    mirror = T.MirroredLanguageModel(SimpleNamespace())
    a, b = [SimpleNamespace(offset=0)], [SimpleNamespace(offset=0)]
    mirror._ensure_epoch(a)
    mirror._ensure_epoch(b)
    mirror._ensure_epoch(a)
    assert sent == [(W.OP_MAKE_CACHE, 1), (W.OP_MAKE_CACHE, 2)]
    mirror.release_cache(a)
    assert sent[-1] == (W.OP_RELEASE_CACHE, 1)
    with pytest.raises(T.TPDesync, match="unknown cache"):
        mirror.release_cache(a)
    c = [SimpleNamespace(offset=0)]
    mirror._ensure_epoch(c)
    assert sent[-1] == (W.OP_MAKE_CACHE, 3)
