import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx_vlm.models.cache import ArraysCache, BatchKVCache, CacheList, KVCache
from mlx_vlm.tp.cache_rows import commit_cache_rows, prepare_cache_rows


def kv():
    cache = BatchKVCache([2, 0, 1])
    data = mx.arange(12).reshape(3, 1, 4, 1)
    cache.update_and_fetch(data, data + 100)
    mx.eval(cache.keys, cache.values)
    return cache


def test_prepare_preserves_live_state_then_commit_reorders_and_compacts():
    cache = kv()
    cache._pool = cache._fpool = cache._ffpool = object()
    cache._no_pad = True
    cache._right_padding = mx.array([1, 0, 2])
    original = cache.__dict__
    old_keys = cache.keys.tolist()
    plan = prepare_cache_rows([cache], [2, 0], batch_size=3)
    assert cache.__dict__ is original
    assert cache.keys.tolist() == old_keys
    commit_cache_rows(plan)
    assert cache.keys[:, :, :3, :].tolist() == [[[[9], [10], [11]]], [[[1], [2], [3]]]]
    assert cache.values[:, :, :3, :].tolist() == [[[[109], [110], [111]]], [[[101], [102], [103]]]]
    assert cache.offset.tolist() == [3, 2]
    assert cache.left_padding.tolist() == [0, 1]
    assert cache._idx == 3
    assert cache._right_padding.tolist() == [2, 1]
    assert cache._pool is cache._fpool is cache._ffpool is None
    assert cache._no_pad is False


def test_recursive_arrays_advance_and_identity():
    arrays = ArraysCache(2, left_padding=[2, 0, 1])
    arrays[0] = mx.array([[10], [20], [30]])
    arrays.lengths = mx.array([5, 6, 7])
    arrays._left_padding_advance = 4
    arrays._lengths_advance = 2
    batch = kv()
    nested = CacheList(CacheList(batch), arrays)
    old = arrays.__dict__
    old_list = arrays.cache
    plan = prepare_cache_rows([nested], [2, 0], batch_size=3)
    assert arrays.__dict__ is old and arrays.cache is old_list
    assert arrays.left_padding.tolist() == [-2, -4, -3]
    commit_cache_rows(plan)
    assert nested[0][0] is batch and nested[1] is arrays
    assert arrays[0].tolist() == [[30], [10]] and arrays[1] is None
    assert arrays.left_padding.tolist() == [-3, -2]
    assert arrays.lengths.tolist() == [5, 3]
    assert arrays._left_padding_advance == arrays._lengths_advance == 0


@pytest.mark.parametrize("indices", [[], [True], [-1], [3], [1, 1], [0.5], [[0]], None])
def test_bad_indices_preserve_state(indices):
    cache = kv()
    state = cache.__dict__
    with pytest.raises(ValueError):
        prepare_cache_rows([cache], indices, batch_size=3)
    assert cache.__dict__ is state


@pytest.mark.parametrize("kind", ["unsupported", "subclass", "rows", "metadata", "alias"])
def test_validate_whole_tree_before_filter(monkeypatch, kind):
    first = kv()
    second = ArraysCache(2, left_padding=[0, 0, 0])
    if kind == "unsupported":
        second = KVCache()
    elif kind == "subclass":
        class Custom(ArraysCache):
            pass
        second = Custom(1)
    elif kind == "rows":
        second[0], second[1] = mx.zeros((3, 1)), mx.zeros((2, 1))
    elif kind == "metadata":
        second.lengths = mx.zeros((2,))
    else:
        second = first
    called = []
    monkeypatch.setattr(BatchKVCache, "filter", lambda *args: called.append(True))
    with pytest.raises(ValueError):
        prepare_cache_rows([CacheList(first, CacheList(second))], [2, 0], batch_size=3)
    assert called == []


def test_evaluation_failure_never_commits_earlier_leaf(monkeypatch):
    first, second = kv(), kv()
    states = first.__dict__, second.__dict__
    calls = []
    real_eval = mx.eval
    def fail_second(*args):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("injected evaluation failure")
        real_eval(*args)
    monkeypatch.setattr(mx, "eval", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        prepare_cache_rows([CacheList(first, second)], [2, 0], batch_size=3)
    assert first.__dict__ is states[0] and second.__dict__ is states[1]
