"""Cross-path provenance regressions; tiny CPU caches, no model loading."""
import ast
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx_vlm import apc, context_vault as V, harvest_provenance as HP
from mlx_vlm.generate import ar, dispatch
from mlx_vlm.models.cache import KVCache


def filled(depth):
    c = KVCache()
    a = mx.zeros((1, 1, depth, 2))
    c.update_and_fetch(a, a)
    return [c]


def batch(parent, prefix_len=32):
    return SimpleNamespace(
        _prompt_uids=[0], uids=[0], _right_pad_per_row=[0],
        _left_padding_per_row=[0],
        _apc_meta=[{"prefix_len": prefix_len, "harvest_provenance": parent}],
    )


@pytest.mark.parametrize("parent", [None, 2])
def test_b1_reharvest_does_not_launder_warm_ancestry(parent, monkeypatch):
    monkeypatch.setenv("MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH", "1")
    p = None if parent is None else HP.make(parent)
    child = ar.PromptProcessingBatch._harvest_provenance(batch(p), 0)
    assert HP.batch_width_of(child) == 1, "capture width must stay truthful"
    assert not HP.may_persist(child), "warm B2/unknown ancestry became durable B1"


def generator(vault):
    g = ar.BatchGenerator.__new__(ar.BatchGenerator)
    g.vault = vault
    g.model = object()
    g._wire_stack = None
    g.apc_manager = apc.APCManager(num_blocks=8, block_size=16)
    g.apc_mode = "exact"
    g._apc_extra_hash = lambda kw: 0
    g._apc_safe_prefix_lookup_min = lambda ids: 0
    g._apc_media_token_ids = lambda: []
    return g


def test_vault_cannot_replace_filtered_apc_with_wider_rung(monkeypatch):
    monkeypatch.setenv("MLX_VLM_APC_SERVE_B1_FROM_B1_ONLY", "1")
    monkeypatch.setattr(ar.cache, "make_prompt_cache", lambda model: [KVCache()])
    tokens = list(range(160))
    vault = V.ContextVault("cross-path", budget_bytes=1 << 20)
    for depth, width in [(80, 1), (128, 2)]:
        vault.insert(tokens, depth, V.capture_fragments(filled(depth), depth),
                     harvest_provenance=HP.make(width))
    g = generator(vault)
    g.apc_manager.store_exact_cache(tokens[:64], filled(64),
                                    harvest_provenance=HP.make(1))
    pick = g._apc_pick_for((0, tokens, 1, {}, None, None))
    assert pick["prefix_len"] == 80, "vault bypassed B1 filter instead of deepest eligible fallback"


def test_apc_disk_legacy_restore_obeys_disk_policy_with_serve_knob_off(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_VLM_APC_SERVE_B1_FROM_B1_ONLY", "0")
    monkeypatch.setenv("MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH", "1")
    disk = apc.DiskBlockStore(tmp_path)
    path = tmp_path / "legacy.safetensors"
    disk._exact_index[7] = path
    monkeypatch.setattr(disk, "_open_shard_header", lambda path: ({}, {
        "layout": "exact_cache_v1", "token_ids": "0,1,2,3", "extra_hash": "0"
    }, 0))
    try:
        assert disk.find_exact_prefix(list(range(8))) is None, "legacy APC disk shard accepted under default durability policy"
    finally:
        disk.close()


def test_dispatch_vault_hit_reports_capture_width(monkeypatch):
    """Execute the actual vault-hit branch without unrelated model/tokenizer work."""
    tree = ast.parse(inspect.getsource(dispatch.stream_generate))
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "_vault is not None and reused_prefix_len == 0")
    vault = V.ContextVault("dispatch", budget_bytes=1 << 20)
    tokens = list(range(16))
    vault.insert(tokens, 8, V.capture_fragments(filled(8), 8),
                 harvest_provenance=HP.make(2))
    monkeypatch.setattr(dispatch.cache, "make_prompt_cache", lambda *a, **kw: [KVCache()])
    monkeypatch.setattr(dispatch, "_prime_cached_prefix_rope_state", lambda *a: True)
    monkeypatch.setenv("MLX_VLM_APC_SERVE_B1_FROM_B1_ONLY", "0")
    ns = dict(dispatch.__dict__, _vault=vault, full_input_ids_list=tokens,
              reused_prefix_len=0, cached_from_width=None, cached_provenance=None,
              input_ids=mx.array([tokens]), model=SimpleNamespace(language_model=object()),
              mask=None, kwargs={})
    exec(compile(ast.Module(body=[branch], type_ignores=[]), "<dispatch-vault-hit>", "exec"), ns)
    assert ns["reused_prefix_len"] == 8
    assert ns["cached_from_width"] == 2, "known vault hit still reports cached_from_width=-"


@pytest.mark.parametrize("parent_width, expected", [(None, None), (1, 1), (2, 2)])
def test_lineage_survives_reharvest_and_wire(parent_width, expected):
    from mlx_vlm.context_vault_wire import pack_fragments, unpack_fragments
    parent = None if parent_width is None else HP.make(parent_width)
    child = ar.PromptProcessingBatch._harvest_provenance(batch(parent), 0)
    grandchild = ar.PromptProcessingBatch._harvest_provenance(batch(child), 0)
    assert HP.batch_width_of(grandchild) == 1
    assert HP.lineage_width_of(grandchild) == expected
    fragments = V.capture_fragments(filled(8), 8)
    manifest, payload = pack_fragments(fragments, grandchild)
    vault = V.ContextVault("wire", budget_bytes=1 << 20)
    vault.insert(list(range(16)), 8, unpack_fragments(manifest, payload),
                 harvest_provenance=manifest["harvest_provenance"])
    back = vault.lookup(list(range(16))).harvest_provenance
    assert back == grandchild
    assert HP.is_b1_eligible(back) is (expected == 1)


def test_old_capture_record_keeps_reported_width_but_unknown_lineage(monkeypatch):
    p = HP.make(1)
    p.pop("harvest_lineage_max_width")
    assert HP.batch_width_of(p) == 1
    assert HP.batch_width_of(HP.normalise(p)) == 1
    monkeypatch.setenv("MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH", "1")
    assert not HP.is_b1_eligible(p)
    assert not HP.may_persist(HP.normalise(p))
    monkeypatch.setenv("MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH", "0")
    assert HP.may_persist(p)
    assert not HP.is_b1_eligible(p), "disk opt-out must not disable B1 serve policy"


@pytest.mark.parametrize("parent_width", [None, 2])
def test_apc_disk_roundtrip_preserves_ancestry_and_read_policy(tmp_path, monkeypatch, parent_width):
    p = None if parent_width is None else HP.make(parent_width)
    child = ar.PromptProcessingBatch._harvest_provenance(batch(p), 0)
    monkeypatch.setenv("MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH", "0")
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    disk = apc.DiskBlockStore(tmp_path)
    manager = apc.APCManager(num_blocks=1, block_size=16, disk=disk)
    tokens = list(range(40))
    try:
        assert manager.store_exact_cache(tokens, filled(40), harvest_provenance=child)
        disk._q.join()
        key = next(iter(disk._exact_index))
        assert disk.exact_harvest_provenance(key) == child
        assert disk.load_exact_cache(key) is not None
        assert manager.lookup_exact_cache(tokens + [99])[1] == 40
        assert manager.lookup_exact_cache(tokens + [99], require_harvest_width_1=True) == (None, 0)
        monkeypatch.setenv("MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH", "1")
        assert disk.load_exact_cache(key) is None
        assert manager.lookup_exact_cache(tokens + [99]) == (None, 0)
    finally:
        manager.close()


@pytest.mark.parametrize("disk_cap, serve_only", [(1, False), (0, True)])
def test_vault_disk_finds_deepest_eligible_without_ram_override(tmp_path, monkeypatch, disk_cap, serve_only):
    from mlx_vlm import vault_disk as VD
    monkeypatch.setenv("MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH", "0")
    disk = VD.DiskPrefixVault(tmp_path, "disk-path")
    source = V.ContextVault("disk-path", budget_bytes=1 << 20)
    target = V.ContextVault("disk-path", budget_bytes=1 << 20)
    tokens = list(range(160))
    try:
        for depth, width in [(64, 1), (80, 1), (128, 2)]:
            source.insert(tokens, depth, V.capture_fragments(filled(depth), depth),
                          harvest_provenance=HP.make(width))
            assert disk.save_async(tokens, source.lookup(tokens))
        disk.flush()
        # A deeper ineligible RAM rung must not replace the eligible disk pick.
        target.insert(tokens, 128, V.capture_fragments(filled(128), 128),
                      harvest_provenance=HP.make(2))
        monkeypatch.setenv("MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH", str(disk_cap))
        picked = disk.restore_into_vault(target, tokens, require_harvest_width_1=serve_only)
        # The disk-cap controls disk eligibility, not pre-existing RAM. For B1
        # serve policy, however, the final RAM lookup must enforce the same rule.
        assert picked.prefix_len == (80 if serve_only else 128)
        rec = disk.best_record(tokens, require_harvest_width_1=serve_only)
        assert rec["prefix_len"] == 80
    finally:
        disk.close()


def test_dispatch_solo_filter_falls_back_to_safe_vault_rung(monkeypatch):
    tree = ast.parse(inspect.getsource(dispatch.stream_generate))
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "_vault is not None and reused_prefix_len == 0")
    tokens = list(range(24))
    vault = V.ContextVault("dispatch-filter", budget_bytes=1 << 20)
    for depth, width in [(8, 1), (16, 2)]:
        vault.insert(tokens, depth, V.capture_fragments(filled(depth), depth),
                     harvest_provenance=HP.make(width))
    monkeypatch.setattr(dispatch.cache, "make_prompt_cache", lambda *a, **kw: [KVCache()])
    monkeypatch.setattr(dispatch, "_prime_cached_prefix_rope_state", lambda *a: True)
    monkeypatch.setenv("MLX_VLM_APC_SERVE_B1_FROM_B1_ONLY", "1")
    ns = dict(dispatch.__dict__, _vault=vault, full_input_ids_list=tokens,
              reused_prefix_len=0, cached_from_width=None, cached_provenance=None,
              input_ids=mx.array([tokens]), model=SimpleNamespace(language_model=object()),
              mask=None, kwargs={})
    exec(compile(ast.Module(body=[branch], type_ignores=[]), "<dispatch-vault-hit>", "exec"), ns)
    assert ns["reused_prefix_len"] == 8
    assert ns["cached_from_width"] == 1


def test_dispatch_warm_checkpoint_keeps_parent_ancestry(monkeypatch):
    tree = ast.parse(inspect.getsource(dispatch.stream_generate))
    callback = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == "_vault_checkpoint")
    vault = V.ContextVault("dispatch-child", budget_bytes=1 << 20)
    p = HP.make(2)
    ns = dict(dispatch.__dict__, _apc_cb=None, _apc_len=None, _vault_rungs={8},
              _base=8, reused_prefix_len=8, cached_provenance=p, _vault=vault,
              full_input_ids_list=list(range(24)))
    exec(compile(ast.Module(body=[callback], type_ignores=[]), "<dispatch-checkpoint>", "exec"), ns)
    ns["_vault_checkpoint"](8, filled(16))
    child = vault.lookup(list(range(24))).harvest_provenance
    assert HP.batch_width_of(child) == 1
    assert HP.lineage_width_of(child) == 2


def test_mirrored_vault_preserves_lineage_and_forwards_b1_filter():
    from mlx_vlm.tp.mirror_vault import MirroredVault
    tokens = list(range(24))
    base = V.ContextVault("mirror-policy", budget_bytes=1 << 20)
    vault = MirroredVault(base, Mock())
    child = ar.PromptProcessingBatch._harvest_provenance(batch(HP.make(2)), 0)
    V.insert_checkpoint(vault, tokens, 16, V.capture_fragments(filled(16), 16),
                        harvest_provenance=child)
    assert base.lookup(tokens).harvest_provenance == child
    V.insert_checkpoint(vault, tokens, 8, V.capture_fragments(filled(8), 8),
                        harvest_provenance=HP.make(1))
    assert vault.lookup(tokens, require_harvest_width_1=True).prefix_len == 8


def test_vault_disk_roundtrip_does_not_launder_lineage(tmp_path, monkeypatch):
    from mlx_vlm import vault_disk as VD
    monkeypatch.setenv("MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH", "0")
    tokens = list(range(24))
    source = V.ContextVault("lineage-disk", budget_bytes=1 << 20)
    target = V.ContextVault("lineage-disk", budget_bytes=1 << 20)
    disk = VD.DiskPrefixVault(tmp_path, "lineage-disk")
    child = ar.PromptProcessingBatch._harvest_provenance(batch(HP.make(2)), 0)
    try:
        source.insert(tokens, 16, V.capture_fragments(filled(16), 16),
                      harvest_provenance=child)
        assert disk.save_async(tokens, source.lookup(tokens))
        disk.flush()
        cp = disk.restore_into_vault(target, tokens)
        assert cp.harvest_provenance == child
        assert HP.batch_width_of(cp.harvest_provenance) == 1
        assert HP.lineage_width_of(cp.harvest_provenance) == 2
        monkeypatch.setenv("MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH", "1")
        assert not HP.may_persist(cp.harvest_provenance)
        assert disk.best_record(tokens) is None
    finally:
        disk.close()
