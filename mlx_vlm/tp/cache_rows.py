"""Prepare TP cache row changes without changing live leaves.

Callers must serialize prepare/commit with cache use. This helper does not mirror
rows to a peer or update epoch, GDN, or scheduler metadata.
"""

import copy
from dataclasses import dataclass

import mlx.core as mx

from ..models.cache import ArraysCache, BatchKVCache, CacheList


@dataclass(frozen=True)
class PreparedCacheRows:
    _updates: tuple


def prepare_cache_rows(caches, keep_indices, *, batch_size):
    """Validate the complete tree, then eagerly stage unique ordered rows.

    ``caches`` is a list/tuple of caches; ``batch_size`` is explicit row identity
    metadata, not a padding signature. Keep indices are Python ints in a
    nonempty list/tuple. Empty batches must use the epoch release path.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(keep_indices, (list, tuple)) or not keep_indices:
        raise ValueError("keep_indices must be a nonempty list or tuple")
    keep_indices = tuple(keep_indices)
    if any(type(i) is not int or i < 0 or i >= batch_size for i in keep_indices):
        raise ValueError("keep_indices must contain in-range integers (not bool)")
    if len(set(keep_indices)) != len(keep_indices):
        raise ValueError("keep_indices must be unique")
    if type(caches) not in (list, tuple) or not caches:
        raise ValueError("caches must be a nonempty list or tuple")

    leaves = []
    seen = set()

    def row_array(value, *, vector=False, rank=None):
        if value is None:
            return
        if (
            not isinstance(value, mx.array)
            or value.ndim < 1
            or value.shape[0] != batch_size
            or (vector and value.ndim != 1)
            or (rank is not None and value.ndim != rank)
        ):
            raise ValueError("cache array has incompatible row shape")

    def validate(cache):
        if id(cache) in seen:
            raise ValueError("cache tree must not contain aliases or cycles")
        seen.add(id(cache))
        if type(cache) is CacheList:
            if not cache.caches:
                raise ValueError("empty CacheList is unsupported")
            for child in cache.caches:
                validate(child)
            return
        if type(cache) is BatchKVCache:
            if cache.left_padding is None or cache.offset is None:
                raise ValueError("BatchKVCache requires row metadata")
            row_array(cache.left_padding, vector=True)
            row_array(cache.offset, vector=True)
            row_array(cache._right_padding, vector=True)
            if (cache.keys is None) != (cache.values is None):
                raise ValueError("BatchKVCache requires paired keys and values")
            row_array(cache.keys, rank=4)
            row_array(cache.values, rank=4)
            if cache.keys is not None and cache.keys.shape[2] != cache.values.shape[2]:
                raise ValueError("BatchKVCache key/value lengths differ")
        elif type(cache) is ArraysCache:
            for value in cache.cache:
                row_array(value)
            row_array(cache._left_padding, vector=True)
            row_array(cache._lengths, vector=True)
        else:
            raise ValueError(f"unsupported TP cache leaf: {type(cache).__name__}")
        leaves.append(cache)

    for cache in caches:
        validate(cache)

    indices = mx.array(keep_indices, dtype=mx.int32)
    updates = []
    for leaf in leaves:
        staged = copy.copy(leaf)
        staged.filter(indices)
        if type(staged) is BatchKVCache:
            # Match GLM give_back_sparse_cache_rows invalidation semantics.
            staged._pool = staged._fpool = staged._ffpool = None
            staged._no_pad = False
            arrays = (staged.keys, staged.values, staged.offset,
                      staged.left_padding, staged._right_padding)
        else:
            arrays = (*staged.cache, staged._left_padding, staged._lengths)
        mx.eval(*(value for value in arrays if value is not None))
        updates.append((leaf, staged.__dict__))
    return PreparedCacheRows(tuple(updates))


def commit_cache_rows(prepared):
    """Install a prepared plan, preserving all live container/leaf identities.

    No MLX work is performed. The caller owns the serialized commit boundary
    and must not reuse a plan after further cache updates.
    """
    for leaf, state in prepared._updates:
        leaf.__dict__ = state
