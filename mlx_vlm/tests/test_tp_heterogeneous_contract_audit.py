"""Desk audit: document the unresolved heterogeneous TP cache contract."""

import mlx.core as mx

from types import SimpleNamespace

import pytest

from mlx_vlm.models.cache import ArraysCache, BatchKVCache, CacheList, KVCache
from mlx_vlm.generate.ar import GenerationBatch
from mlx_vlm.server import tp_mode as T
from mlx_vlm.tp import worker as W


def test_rank0_batch_and_rank1_scalar_cache_masks_are_not_equivalent():
    """The current worker creates a scalar cache and cannot mirror left padding."""
    rank0 = BatchKVCache([2, 0])
    rank1 = KVCache()

    prefill0 = rank0.make_mask(3, return_array=True, window_size=None)
    prefill1 = rank1.make_mask(3, return_array=True, window_size=None)
    assert prefill0.shape == (2, 1, 3, 3)
    assert prefill1.shape == (3, 3)
    assert prefill0[0, 0].tolist() != prefill1.tolist()

    kv = mx.zeros((2, 1, 3, 4))
    rank0.update_and_fetch(kv, kv)
    rank1.update_and_fetch(kv, kv)
    decode0 = rank0.make_mask(1, return_array=True, window_size=None)
    decode1 = rank1.make_mask(1, return_array=True, window_size=None)
    assert rank0.offset.tolist() == [1, 3]
    assert rank1.offset == 3
    assert decode0.shape == (2, 1, 1, 4)
    assert decode1 is None


class _CacheLM:
    def make_cache(self):
        return [CacheList(KVCache(), KVCache())]


def _step(cache, width):
    leaf = cache[0].caches[0]
    mask = leaf.make_mask(width, return_array=True, window_size=None)
    kv = mx.zeros((2, 1, width, 1))
    for entry in cache[0].caches:
        entry.update_and_fetch(kv, kv)
    # Include both the mask and offsets in a deterministic stand-in output.
    return mask, leaf.offset.tolist()


def test_worker_reconstructs_rank0_padding_through_first_decode():
    rank0 = [CacheList(BatchKVCache([2, 0]), BatchKVCache([2, 0]))]
    state = W._WorkerState(_CacheLM())
    assert state.handle(W.decode(W.encode(
        W.OP_MAKE_CACHE, 1, (1, 2), [2, 0], n=64)))
    rank1 = state.caches[1]

    prefill0, offsets0 = _step(rank0, 3)
    prefill1, offsets1 = _step(rank1, 3)
    decode0, decode_offsets0 = _step(rank0, 1)
    decode1, decode_offsets1 = _step(rank1, 1)

    assert prefill0.tolist() == prefill1.tolist()
    assert offsets0 == offsets1 == [1, 3]
    assert decode0.tolist() == decode1.tolist()
    assert decode_offsets0 == decode_offsets1 == [2, 4]


def test_mixed_glm_cache_keeps_stable_row_signature_after_prefill():
    arrays = ArraysCache(2, left_padding=[2, 0])
    cache = [arrays, CacheList(BatchKVCache([2, 0]), BatchKVCache([2, 0]))]
    assert T._cache_left_padding(cache) == [2, 0]
    arrays.advance(3)
    assert arrays.left_padding.tolist() == [-1, -3]
    assert T._cache_left_padding(cache) == [2, 0]


def test_same_epoch_refuses_filtered_batch_cache(monkeypatch):
    monkeypatch.setattr(T, "_ctrl_send", lambda *a, **kw: None)
    mirror = T.MirroredLanguageModel(SimpleNamespace())
    cache = [BatchKVCache([2, 0])]
    mirror._ensure_epoch(cache)
    cache[0].filter(mx.array([1], mx.int32))
    with pytest.raises(T.TPDesync, match="row composition changed"):
        mirror._ensure_epoch(cache)


def test_mixed_batch_cache_padding_is_refused():
    cache = [CacheList(BatchKVCache([2, 0]), KVCache())]
    with pytest.raises(T.TPDesync, match="disagree"):
        T._cache_left_padding(cache)


@pytest.mark.parametrize(
    ("method", "args", "message"),
    [("filter", ([1, 0],), "filter/reorder"),
     ("extend", (SimpleNamespace(uids=[3]),), "extend")],
)
def test_generation_batch_refuses_unmirrored_row_changes(method, args, message):
    batch = GenerationBatch.__new__(GenerationBatch)
    batch._language_model = T.MirroredLanguageModel(SimpleNamespace())
    batch.uids = [1, 2]
    with pytest.raises(T.TPDesync, match=message):
        getattr(batch, method)(*args)
