"""Desk audit: document the unresolved heterogeneous TP cache contract."""

import mlx.core as mx

from mlx_vlm.models.cache import BatchKVCache, KVCache


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
