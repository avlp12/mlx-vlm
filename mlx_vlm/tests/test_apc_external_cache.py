"""Complete stream dispatch regressions with CPU tokens and a fake decode step.

The production stream_generate control flow runs unchanged; only model execution,
shared-cache storage, input preparation, and wired-limit accounting are stand-ins.
These tests verify cache identity, token boundaries, and capture metadata, not logits.
"""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx

mx.set_default_device(mx.cpu)

import pytest

from mlx_vlm import harvest_provenance as HP
from mlx_vlm.generate import dispatch
from mlx_vlm.generate.common import PromptCacheState
from mlx_vlm.models.cache import KVCache


def kv_cache(depth=0):
    cache = KVCache()
    if depth:
        values = mx.zeros((1, 1, depth, 2))
        cache.update_and_fetch(values, values)
    return [cache]


@pytest.fixture
def stream_case(monkeypatch):
    """Exercise the entire production stream function without a model load."""
    def run(*, external=None, mode="exact", state=None, parent=None):
        tokens = list(range(64))
        stores = []
        steps = []
        manager = SimpleNamespace(exact_cache_guard_tokens=16, release=Mock())

        def record(kind, ids, cache, provenance):
            stores.append((kind, len(ids), cache[0].offset, provenance))

        manager.store_exact_cache = lambda ids, cache, **kw: record(
            "exact", ids, cache, kw.get("harvest_provenance")
        )
        vault = SimpleNamespace(lookup=Mock(return_value=None))
        monkeypatch.setattr(dispatch._apc, "model_apc_mode", lambda model: mode)
        monkeypatch.setattr(dispatch._apc, "semantic_extra_hash", lambda **kw: 0)
        monkeypatch.setattr(dispatch._apc, "hash_image_payload", lambda **kw: 0)
        plan = None
        if parent is not None:
            plan = dict(prefix_len=16, warm_cache=kv_cache(16),
                        harvest_provenance=HP.make(parent))
        monkeypatch.setattr(dispatch._apc, "apc_lookup_plan", lambda *a, **kw: plan)
        monkeypatch.setattr(dispatch._apc, "commit_prefix_blocks",
                            lambda manager, cache, ids, **kw: record(
                                "block", ids, cache, kw.get("harvest_provenance")))
        monkeypatch.setattr(dispatch._vault_mod, "vault_enabled", lambda: True)
        monkeypatch.setattr(dispatch._vault_mod, "vault_identity_for_model", lambda m: "test")
        monkeypatch.setattr(dispatch._vault_mod, "get_vault", lambda key: vault)
        monkeypatch.setattr(dispatch._vault_mod, "boundary_ladder", lambda *a, **kw: [48])
        monkeypatch.setattr(dispatch._vault_mod, "capture_fragments", lambda cache, n: cache)
        monkeypatch.setattr(dispatch._vault_mod, "insert_checkpoint",
                            lambda vault, ids, n, fragments, **kw: record(
                                "vault", ids[:n], fragments, kw.get("harvest_provenance")))
        monkeypatch.setattr(dispatch, "should_add_special_tokens", lambda *a: False)
        monkeypatch.setattr(dispatch, "is_diffusion_model", lambda *a: False)
        monkeypatch.setattr(dispatch, "_prime_cached_prefix_rope_state", lambda *a: True)
        monkeypatch.setattr(dispatch, "wired_limit", lambda *a: nullcontext())
        monkeypatch.setattr(dispatch.cache, "make_prompt_cache", lambda *a, **kw: kv_cache())
        monkeypatch.setattr(dispatch, "make_streaming_detokenizer", lambda *a: SimpleNamespace(
            last_segment="x", add_token=lambda *a, **kw: None, finalize=lambda: None))
        monkeypatch.setattr(dispatch.mx, "get_peak_memory", lambda: 0)
        monkeypatch.setattr(dispatch.mx, "clear_cache", lambda: None)
        monkeypatch.setenv("MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH", "1")

        def step(ids, model, pixels, mask, **kw):
            cache = kw["prompt_cache"]
            steps.append(dict(cache=cache, tokens=ids.flatten().tolist(),
                              warm_prefix=kw["warm_prefix"],
                              checkpoint=kw["prompt_cache_checkpoint"]))
            points = kw["prompt_cache_checkpoint_len"]
            points = [] if points is None else points if isinstance(points, list) else [points]
            previous = 0
            for point in points:
                for layer in cache:
                    layer.offset = getattr(layer, "offset", 0) + point - previous
                kw["prompt_cache_checkpoint"](point, cache)
                previous = point
            for layer in cache:
                layer.offset = getattr(layer, "offset", 0) + ids.size - previous
            yield 99, None

        monkeypatch.setattr(dispatch, "generate_step", step)
        kwargs = dict(input_ids=mx.array([tokens]), apc_manager=manager)
        if external is not None:
            kwargs["prompt_cache"] = external
        if state is not None:
            kwargs["prompt_cache_state"] = state
        model = SimpleNamespace(config=SimpleNamespace(model_type="test"),
                                language_model=SimpleNamespace())
        processor = SimpleNamespace(stopping_criteria=lambda token: False)
        results = list(dispatch.stream_generate(model, processor, "", **kwargs))
        return SimpleNamespace(stores=stores, steps=steps, results=results,
                               manager=manager, tokens=tokens)
    return run


@pytest.mark.parametrize("kind", ["warm", "unknown", "opaque", "raising", "mixed"])
@pytest.mark.parametrize("mode", ["exact", "block"])
def test_external_unproven_cache_never_harvests(stream_case, kind, mode):
    if kind == "warm":
        external = kv_cache(16)
    elif kind == "unknown":
        external = [SimpleNamespace(offset=16)]
    elif kind == "opaque":
        external = [SimpleNamespace()]
    elif kind == "raising":
        external = [SimpleNamespace(offset=16, empty=Mock(side_effect=NotImplementedError))]
    else:
        external = kv_cache() + kv_cache(16)
    out = stream_case(external=external, mode=mode)
    assert out.steps[0]["cache"] is external, "caller cache must remain the generation cache"
    assert out.steps[0]["tokens"] == out.tokens
    assert not out.stores, f"unproven external cache was harvested: {out.stores}"
    assert out.steps[0]["checkpoint"] is None
    assert out.results[-1].generation_tokens == 1


@pytest.mark.parametrize("external", [False, True])
def test_cold_or_proven_empty_cache_captures_truthful_boundaries(stream_case, external):
    cache = kv_cache() if external else None
    out = stream_case(external=cache)
    if external:
        assert out.steps[0]["cache"] is cache
    assert [(kind, n, offset) for kind, n, offset, p in out.stores] == [
        ("exact", 48, 48), ("vault", 48, 48), ("exact", 64, 64)]
    assert all(HP.is_b1_eligible(p) and HP.may_persist(p) for *_, p in out.stores)


@pytest.mark.parametrize("mode", ["exact", "block"])
def test_known_prompt_cache_state_keeps_tokens_and_unknown_lineage(stream_case, mode):
    state = PromptCacheState()
    cached = kv_cache(16)
    state.update(list(range(16)), cached)
    out = stream_case(state=state, mode=mode)
    assert out.steps[0]["cache"] is cached
    assert out.steps[0]["tokens"] == list(range(16, 64))
    assert out.results[-1].cached_tokens == 16
    assert state.token_ids == list(range(64)) + [99]
    assert out.stores
    for kind, n, offset, p in out.stores:
        assert HP.batch_width_of(p) == 1, "block harvest dropped known capture width"
        assert HP.lineage_width_of(p) is None
        assert not HP.may_persist(p)
        if kind == "vault":
            assert n == offset == 48


@pytest.mark.parametrize("parent_width", [None, 1, 2])
def test_block_harvest_forwards_capture_and_parent_lineage(stream_case, parent_width):
    out = stream_case(mode="block", parent=parent_width)
    p = next(p for kind, _, _, p in out.stores if kind == "block")
    assert HP.batch_width_of(p) == 1, "block harvest dropped known capture width"
    assert HP.lineage_width_of(p) == (parent_width or 1)
    assert HP.may_persist(p) is (parent_width != 2)


@pytest.mark.parametrize("stale_alias", [False, True])
def test_external_unknown_cache_cannot_seed_prompt_cache_state(stream_case, stale_alias):
    state = PromptCacheState()
    external = kv_cache(16)
    if stale_alias:
        state.update(list(range(100, 116)), external)
    stream_case(external=external, state=state)
    assert state.token_ids is None, "external ancestry was recorded with incomplete token identity"
    assert state.cache is None
