"""CPU-only readiness audit for the frozen TP2 batch ladder lifecycle."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.generate.ar import GenerationBatch, _extend_cache
from mlx_vlm.models.cache import BatchKVCache
from mlx_vlm.server import tp_mode as T


def _mirror(monkeypatch):
    monkeypatch.setattr(T, "_ctrl_send", lambda *args, **kwargs: None)
    return T.MirroredLanguageModel(SimpleNamespace())


def _cache(padding):
    return [BatchKVCache(padding)]


def test_all_rows_finish_then_next_group_can_open_a_fresh_epoch(monkeypatch):
    mirror = _mirror(monkeypatch)
    first = _cache([2, 0])
    mirror._ensure_epoch(first)
    # A completed fixed batch performs no later forward on its old cache.
    # The next L26 group owns a distinct empty cache and gets a new epoch.
    second = _cache([1, 0, 3, 2])
    mirror._ensure_epoch(second)
    assert mirror._epoch == 2


def test_partial_finish_blocks_the_next_speculative_forward(monkeypatch):
    mirror = _mirror(monkeypatch)
    cache = _cache([3, 1, 0])
    mirror._ensure_epoch(cache)
    # dflash._dflash_rounds_batch filters target caches after a row finishes.
    cache[0].filter(mx.array([0, 2], mx.int32))
    with pytest.raises(T.TPDesync, match="row composition changed"):
        mirror._ensure_epoch(cache)


def test_extend_active_blocks_after_round_boundary_admission(monkeypatch):
    mirror = _mirror(monkeypatch)
    cache = _cache([2, 0])
    mirror._ensure_epoch(cache)
    cache[:] = _extend_cache(cache, _cache([1]))
    with pytest.raises(T.TPDesync, match="row composition changed"):
        mirror._ensure_epoch(cache)


def test_greedy_live_batch_merge_is_refused_before_mutation(monkeypatch):
    batch = GenerationBatch.__new__(GenerationBatch)
    batch._language_model = _mirror(monkeypatch)
    batch.uids = [1]
    donor = SimpleNamespace(uids=[2])
    with pytest.raises(T.TPDesync, match="extend"):
        batch.extend(donor)
