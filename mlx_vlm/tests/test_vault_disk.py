"""Disk-backed prefix vault (C15): identity, refusals, policy, layout, races.

Everything here runs on CPU against caches shaped like ``glm5_next.make_cache()``
-- ``ArraysCache(size=2)`` for the KDA linear layers, ``CacheList(KVCache, KVCache)``
for the DSA attention layers -- so the round-trip proof exercises the real
adapter and manifest path without loading 169 GB of weights.

The measurement these tests protect is in
``~/glm53flash/logs/sweep11/P2_VERDICT.md``: the internal NVMe reads at
6.6-6.7 GB/s in >= 4 MiB records and 3.2 GB/s at 1 MiB, and the external X10
collapses from 0.82 GB/s to 0.166 (worst rep 0.037) when the same bytes are read
in 28,180 B per-token records.  Two of the tests below exist purely to keep the
implementation on the right side of that table.
"""

import json
import os
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path

import mlx.core as mx

from mlx_vlm.apc_adapters import Capability, StateFragment
from mlx_vlm.context_vault import (
    ContextVault,
    VaultTier,
    capture_fragments,
    restore_fragments,
)
from mlx_vlm.context_vault_wire import pack_fragments, plan_fragments
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache
from mlx_vlm import vault_disk as VD

KDA_LAYERS = 3
DSA_LAYERS = 2
N_HEADS = 2
HEAD_DIM = 4


def make_glm5_shaped_cache():
    caches = [ArraysCache(size=2) for _ in range(KDA_LAYERS)]
    caches += [CacheList(KVCache(), KVCache()) for _ in range(DSA_LAYERS)]
    return caches


def drive(caches, n_tokens, seed=0):
    for i, c in enumerate(caches):
        if isinstance(c, ArraysCache):
            key = mx.random.key(seed * 131 + i)
            c[0] = mx.random.normal((1, N_HEADS, HEAD_DIM, 4), key=key, dtype=mx.float32)
            c[1] = mx.random.normal(
                (1, N_HEADS, HEAD_DIM, HEAD_DIM), key=key, dtype=mx.float32
            )
        else:
            for j, sub in enumerate(c.caches):
                key = mx.random.key(seed * 977 + i * 13 + j)
                k = mx.random.normal((1, N_HEADS, n_tokens, HEAD_DIM), key=key)
                v = mx.random.normal((1, N_HEADS, n_tokens, HEAD_DIM), key=key)
                sub.update_and_fetch(k, v)
    mx.eval([c.state for c in caches])
    return caches


def frags_at(n_tokens, seed=0):
    return capture_fragments(drive(make_glm5_shaped_cache(), n_tokens, seed=seed), n_tokens)


def tokens_for(n, base=1000):
    return list(range(base, base + n))


def flat_arrays(fragments):
    """Every array of a fragment list, in manifest order."""
    _manifest, flats = plan_fragments(fragments)
    mx.eval(flats)
    return flats


def state_signature(caches):
    """Flat list of concrete arrays for bit-comparison (as test_context_vault)."""
    out = []

    def walk(o):
        if isinstance(o, mx.array):
            out.append(o)
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)

    for c in caches:
        walk(c.state)
    return out


def assert_caches_bit_identical(tc, a, b, label=""):
    sa, sb = state_signature(a), state_signature(b)
    tc.assertEqual(len(sa), len(sb), f"{label}: component count differs")
    for i, (x, y) in enumerate(zip(sa, sb)):
        tc.assertEqual(x.shape, y.shape, f"{label}: shape differs at {i}")
        tc.assertEqual(x.dtype, y.dtype, f"{label}: dtype differs at {i}")
        # Bytes, not allclose: NaN != NaN and +0.0 == -0.0 both matter here.
        xb = mx.view(x.flatten(), mx.uint8)
        yb = mx.view(y.flatten(), mx.uint8)
        tc.assertTrue(bool(mx.all(xb == yb).item()), f"{label}: bytes differ at {i}")


def payload_bytes(fragments) -> bytes:
    """The packed payload as plain Python bytes.

    Used where a comparison crosses a thread boundary: MLX streams are
    thread-local, so an array built on a worker thread cannot be evaluated on
    the main thread after that worker has exited.  Materialising to bytes inside
    the worker is both the fix and the stronger assertion.
    """
    _manifest, payload = pack_fragments(fragments)
    return bytes(memoryview(payload))


class DiskVaultTestCase(unittest.TestCase):
    """Each case gets its own directory, its own stats, and no env leakage."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="vaultdisk-")
        self.root = Path(self._tmp.name)
        self.stats = VD.DiskVaultStats()
        self._saved_env = {
            k: os.environ.get(k)
            for k in (
                VD._ENV_DIR, VD._ENV_MAX_GB, VD._ENV_SAVE_ON_INSERT,
                VD._ENV_FSYNC, VD._ENV_CHUNK_MB, VD._ENV_NOCACHE,
                VD._ENV_STRICT_GIT,
            )
        }
        self._open = []

    def tearDown(self):
        for dv in self._open:
            dv.close(timeout=10.0)
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()

    def mkvault(self, **kw):
        kw.setdefault("stats", self.stats)
        kw.setdefault("chunk_bytes", 4 << 20)
        kw.setdefault("cap_bytes", 1 << 40)
        dv = VD.DiskPrefixVault(self.root, kw.pop("identity", "ident-A"), **kw)
        self._open.append(dv)
        return dv


# --------------------------------------------------------------------------
# 1. Round-trip identity
# --------------------------------------------------------------------------


class TestRoundTripIdentity(DiskVaultTestCase):
    def test_every_array_survives_the_blob_bit_for_bit(self):
        dv = self.mkvault()
        toks = tokens_for(24)
        vault = ContextVault("ident-A", budget_bytes=1 << 30)
        self.assertTrue(vault.insert(toks, 24, frags_at(24, seed=3)))
        cp = vault.lookup(toks)

        self.assertTrue(dv.save_async(toks, cp))
        self.assertTrue(dv.flush(30.0))

        loaded = dv.load_entry(dv.records()[0]["key"])
        self.assertIsNotNone(loaded, "a freshly written entry must load")
        header, frags, token_ids = loaded

        self.assertEqual(token_ids, toks, "token ids round-trip")
        self.assertEqual(header["prefix_len"], 24)
        self.assertEqual(header["token_count"], 24)

        want = flat_arrays(cp.fragments)
        got = flat_arrays(frags)
        self.assertEqual(len(want), len(got), "same array count")
        for i, (a, b) in enumerate(zip(want, got)):
            self.assertEqual(a.dtype, b.dtype, f"array {i} dtype")
            self.assertEqual(a.shape, b.shape, f"array {i} shape")
            self.assertTrue(bool(mx.array_equal(a, b).item()), f"array {i} bytes")

    def test_restored_cache_equals_the_ram_restore(self):
        dv = self.mkvault()
        toks = tokens_for(20, base=77)
        frags = frags_at(20, seed=5)
        vault = ContextVault("ident-A", budget_bytes=1 << 30)
        vault.insert(toks, 20, frags)
        cp = vault.lookup(toks)
        dv.save_async(toks, cp)
        dv.flush(30.0)
        _h, disk_frags, _t = dv.load_entry(dv.records()[0]["key"])

        from_ram, from_disk = make_glm5_shaped_cache(), make_glm5_shaped_cache()
        self.assertTrue(restore_fragments(from_ram, cp.fragments))
        self.assertTrue(restore_fragments(from_disk, disk_frags))
        assert_caches_bit_identical(self, from_ram, from_disk, "disk round-trip")

    def test_header_is_stable_across_two_writes_of_the_same_rung(self):
        """The identity block must not drift; only the timestamps may."""
        dv_a = self.mkvault()
        toks = tokens_for(16, base=5)
        v = ContextVault("ident-A", budget_bytes=1 << 30)
        v.insert(toks, 16, frags_at(16, seed=9))
        cp = v.lookup(toks)
        dv_a.save_async(toks, cp)
        dv_a.flush(30.0)
        key = dv_a.records()[0]["key"]
        h1 = VD.read_header(dv_a._entry_path(key))

        second = Path(self._tmp.name) / "second"
        dv_b = VD.DiskPrefixVault(second, "ident-A", stats=self.stats,
                                  chunk_bytes=4 << 20, cap_bytes=1 << 40)
        self._open.append(dv_b)
        dv_b.save_async(toks, cp)
        dv_b.flush(30.0)
        h2 = VD.read_header(dv_b._entry_path(key))

        volatile = {"created_at", "created_at_iso", "saved_because"}
        self.assertEqual(
            {k: v for k, v in h1.items() if k not in volatile},
            {k: v for k, v in h2.items() if k not in volatile},
            "identical rung -> identical header, including the array table",
        )
        self.assertEqual(h1["key"], h2["key"], "and the same content-addressed key")

    def test_disk_payload_is_the_wire_payload(self):
        """One format, two transports -- the blob is what the peer tier ships."""
        dv = self.mkvault()
        toks = tokens_for(18, base=31)
        v = ContextVault("ident-A", budget_bytes=1 << 30)
        v.insert(toks, 18, frags_at(18, seed=11))
        cp = v.lookup(toks)
        dv.save_async(toks, cp)
        dv.flush(30.0)
        key = dv.records()[0]["key"]
        header = VD.read_header(dv._entry_path(key))

        manifest, payload = pack_fragments(cp.fragments)
        self.assertEqual(header["arrays"], manifest["offsets"])
        self.assertEqual(header["payload_nbytes"], manifest["total_bytes"])
        raw = dv._entry_path(key).read_bytes()
        on_disk = raw[header["payload_offset"] : header["payload_offset"] + header["payload_nbytes"]]
        self.assertEqual(on_disk, bytes(memoryview(payload)), "byte-for-byte")


# --------------------------------------------------------------------------
# 2. Refusals, each with a named reason counter
# --------------------------------------------------------------------------


class TestRefusals(DiskVaultTestCase):
    def _write_one(self, identity="ident-A", **kw):
        dv = self.mkvault(identity=identity, **kw)
        toks = tokens_for(14, base=400)
        v = ContextVault(identity, budget_bytes=1 << 30)
        v.insert(toks, 14, frags_at(14, seed=13))
        cp = v.lookup(toks)
        dv.save_async(toks, cp)
        dv.flush(30.0)
        return dv, toks, dv.records()[0]["key"]

    def _patch_header(self, path, **fields):
        raw = path.read_bytes()
        (hlen,) = struct.unpack("<Q", raw[8:16])
        header = json.loads(raw[16 : 16 + hlen].decode())
        header.update(fields)
        blob = json.dumps(header, separators=(",", ":")).encode()
        self.assertLessEqual(
            16 + len(blob), int(header["tokens_offset"]),
            "test rewrite must fit the reserved header area",
        )
        out = bytearray(raw)
        out[8:16] = struct.pack("<Q", len(blob))
        out[16 : 16 + len(blob)] = blob
        # Zero the slack so a stale tail cannot be parsed.
        out[16 + len(blob) : int(header["tokens_offset"])] = bytes(
            int(header["tokens_offset"]) - 16 - len(blob)
        )
        path.write_bytes(bytes(out))

    def test_a_different_model_hash_is_refused(self):
        dv, _toks, key = self._write_one()
        self._patch_header(dv._entry_path(key), model_identity="a" * 32)
        dv.model_identity = "b" * 32
        self.assertIsNone(dv.load_entry(key))
        self.assertEqual(
            self.stats.snapshot()["disk_refusals"].get("model_identity_mismatch"), 1
        )

    def test_a_different_git_head_is_refused_under_the_strict_policy(self):
        dv, _toks, key = self._write_one(strict_git=True)
        self._patch_header(dv._entry_path(key), git_head="deadbeefcafe")
        self.assertIsNone(dv.load_entry(key))
        self.assertEqual(
            self.stats.snapshot()["disk_refusals"].get("git_head_mismatch"), 1
        )

    def test_the_git_head_policy_field_can_be_relaxed(self):
        """The refusal is a POLICY, and the policy is nameable and testable."""
        dv, _toks, key = self._write_one(strict_git=True)
        self._patch_header(dv._entry_path(key), git_head="deadbeefcafe")
        dv.strict_git = False
        self.assertIsNotNone(dv.load_entry(key), "relaxed policy accepts")
        self.assertEqual(self.stats.snapshot()["disk_refusals"], {})

    def test_a_dtype_that_does_not_match_its_shape_is_refused(self):
        dv, _toks, key = self._write_one()
        header = VD.read_header(dv._entry_path(key))
        arrays = [dict(a) for a in header["arrays"]]
        # float32 -> float16 with the same shape and nbytes: the bytes would be
        # reinterpreted, every value silently wrong, every shape still right.
        arrays[0]["dtype"] = "float16"
        self._patch_header(dv._entry_path(key), arrays=arrays)
        self.assertIsNone(dv.load_entry(key))
        self.assertEqual(self.stats.snapshot()["disk_refusals"].get("dtype_mismatch"), 1)

    def test_an_unknown_dtype_is_refused(self):
        dv, _toks, key = self._write_one()
        header = VD.read_header(dv._entry_path(key))
        arrays = [dict(a) for a in header["arrays"]]
        arrays[0]["dtype"] = "float8_e4m3"
        self._patch_header(dv._entry_path(key), arrays=arrays)
        self.assertIsNone(dv.load_entry(key))
        self.assertEqual(self.stats.snapshot()["disk_refusals"].get("dtype_unknown"), 1)

    def test_a_foreign_vault_identity_is_refused(self):
        dv, _toks, key = self._write_one()
        dv.identity = "ident-B"
        self.assertIsNone(dv.load_entry(key))
        self.assertEqual(
            self.stats.snapshot()["disk_refusals"].get("identity_mismatch"), 1
        )

    def test_a_changed_kernel_toggle_set_is_refused(self):
        dv, _toks, key = self._write_one()
        dv.flags_hash = VD._flags_hash({"MLX_VLM_GLM5_FUSED_KDA": "1"})
        self.assertIsNone(dv.load_entry(key))
        self.assertEqual(self.stats.snapshot()["disk_refusals"].get("flags_mismatch"), 1)

    def test_a_truncated_blob_is_refused(self):
        dv, _toks, key = self._write_one()
        path = dv._entry_path(key)
        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) - 64])
        self.assertIsNone(dv.load_entry(key))
        self.assertEqual(self.stats.snapshot()["disk_refusals"].get("truncated"), 1)

    def test_a_prompt_hash_that_is_not_the_one_asked_for_is_refused(self):
        dv, _toks, key = self._write_one()
        self.assertIsNone(dv.load_entry(key, expect_prompt_sha="0" * 64))
        self.assertEqual(
            self.stats.snapshot()["disk_refusals"].get("prompt_sha_mismatch"), 1
        )

    def test_a_session_entry_is_not_served_to_a_prefill_restore(self):
        dv, _toks, key = self._write_one()
        self.assertIsNone(dv.load_entry(key, tier=VaultTier.SESSION.value))
        self.assertEqual(self.stats.snapshot()["disk_refusals"].get("tier_mismatch"), 1)

    def test_a_refusal_never_raises_and_never_serves(self):
        """Refusal is the only failure mode a caller sees; it costs a prefill."""
        dv, toks, key = self._write_one()
        dv.identity = "ident-B"
        vault = ContextVault("ident-B", budget_bytes=1 << 30)
        self.assertIsNone(dv.restore_into_vault(vault, toks + [9999]))
        self.assertEqual(vault.rungs, 0, "nothing was inserted")
        self.assertGreaterEqual(self.stats.snapshot()["disk_misses"], 1)


# --------------------------------------------------------------------------
# 3. The policy loop: evict -> save -> restore through a real ContextVault
# --------------------------------------------------------------------------


class TestEvictSaveRestoreCycle(DiskVaultTestCase):
    def test_an_evicted_rung_comes_back_off_disk_bit_identical(self):
        dv = self.mkvault()
        one_rung = sum(int(a.nbytes) for a in flat_arrays(frags_at(12, seed=21)))
        # A cap that holds exactly one rung, so inserting the second evicts the
        # first -- the moment the disk tier exists for.
        vault = ContextVault("ident-A", budget_bytes=int(one_rung * 1.5))
        vault.disk = dv

        toks_a = tokens_for(12, base=100)
        toks_b = tokens_for(12, base=900)
        vault.insert(toks_a, 12, frags_at(12, seed=21))
        want = flat_arrays(vault.lookup(toks_a).fragments)
        vault.insert(toks_b, 12, frags_at(12, seed=22))

        self.assertEqual(vault.stats.evictions, 1, "the cap must have bitten")
        self.assertIsNone(vault.lookup(toks_a), "and A must be gone from RAM")
        self.assertTrue(dv.flush(30.0))
        self.assertEqual(len(dv.records()), 1, "the evicted rung was saved")

        # A longer prompt that starts with A: the disk entry is a strict prefix.
        query = toks_a + [4242, 4243]
        cp = dv.restore_into_vault(vault, query, VaultTier.PREFILL)
        self.assertIsNotNone(cp, "the evicted rung is restorable")
        self.assertEqual(int(cp.prefix_len), 12)
        got = flat_arrays(cp.fragments)
        for i, (a, b) in enumerate(zip(want, got)):
            self.assertTrue(bool(mx.array_equal(a, b).item()), f"array {i}")

        snap = self.stats.snapshot()
        self.assertEqual(snap["disk_hits"], 1)
        self.assertEqual(snap["disk_restores"], 1)
        self.assertGreater(snap["disk_bytes_read"], 0)
        self.assertGreater(snap["disk_restore_seconds"], 0.0)

    def test_the_restore_goes_through_the_normal_ram_accounting(self):
        dv = self.mkvault()
        vault = ContextVault("ident-A", budget_bytes=1 << 30)
        vault.disk = dv
        toks = tokens_for(12, base=100)
        vault.insert(toks, 12, frags_at(12, seed=23))
        cp = vault.lookup(toks)
        dv.save_async(toks, cp)
        dv.flush(30.0)

        fresh = ContextVault("ident-A", budget_bytes=1 << 30)
        fresh.disk = dv
        before = fresh.stats.inserts
        restored = dv.restore_into_vault(fresh, toks + [7], VaultTier.PREFILL)
        self.assertIsNotNone(restored)
        self.assertEqual(fresh.stats.inserts, before + 1, "ordinary insert")
        self.assertEqual(fresh.resident_bytes, restored.nbytes, "ordinary accounting")
        self.assertEqual(fresh.rungs, 1)
        # And the ordinary longest-strict-prefix lookup now serves it.
        self.assertIs(fresh.lookup(toks + [7]), restored)

    def test_a_deeper_ram_rung_is_not_replaced_by_a_shallower_disk_one(self):
        dv = self.mkvault()
        vault = ContextVault("ident-A", budget_bytes=1 << 30)
        vault.disk = dv
        toks = tokens_for(24, base=100)
        vault.insert(toks, 8, frags_at(8, seed=24))
        cp8 = vault.lookup(toks)
        dv.save_async(toks, cp8)
        dv.flush(30.0)
        self.assertIsNone(
            dv.restore_into_vault(vault, toks, VaultTier.PREFILL, min_depth=8),
            "min_depth is the RAM depth; a shallower disk entry is not a win",
        )

    def test_an_entry_covering_the_whole_prompt_is_not_a_prefix_hit(self):
        dv = self.mkvault()
        vault = ContextVault("ident-A", budget_bytes=1 << 30)
        toks = tokens_for(12, base=600)
        vault.insert(toks, 12, frags_at(12, seed=25))
        dv.save_async(toks, vault.lookup(toks))
        dv.flush(30.0)
        self.assertIsNone(dv.best_record(toks, VaultTier.PREFILL, strict=True))
        self.assertIsNotNone(dv.best_record(toks, VaultTier.PREFILL, strict=False))

    def test_save_on_insert_is_off_by_default_and_can_be_turned_on(self):
        dv = self.mkvault(save_on_insert=False)
        vault = ContextVault("ident-A", budget_bytes=1 << 30)
        vault.disk = dv
        toks = tokens_for(10, base=222)
        vault.insert(toks, 10, frags_at(10, seed=26))
        dv.flush(5.0)
        self.assertEqual(len(dv.records()), 0, "default policy: save on eviction only")

        dv.save_on_insert = True
        toks2 = tokens_for(10, base=333)
        vault.insert(toks2, 10, frags_at(10, seed=27))
        self.assertTrue(dv.flush(30.0))
        self.assertEqual(len(dv.records()), 1)

    def test_a_rung_already_on_disk_is_not_rewritten(self):
        dv = self.mkvault()
        vault = ContextVault("ident-A", budget_bytes=1 << 30)
        toks = tokens_for(10, base=444)
        vault.insert(toks, 10, frags_at(10, seed=28))
        cp = vault.lookup(toks)
        self.assertTrue(dv.save_async(toks, cp))
        dv.flush(30.0)
        self.assertFalse(dv.save_async(toks, cp), "second save is a skip")
        self.assertEqual(self.stats.snapshot()["disk_saves"], 1)
        self.assertEqual(self.stats.snapshot()["disk_save_skips"], 1)

    def test_the_index_survives_a_restart(self):
        dv = self.mkvault()
        toks = tokens_for(12, base=555)
        vault = ContextVault("ident-A", budget_bytes=1 << 30)
        vault.insert(toks, 12, frags_at(12, seed=29))
        dv.save_async(toks, vault.lookup(toks))
        dv.flush(30.0)
        dv.close()

        reopened = self.mkvault()
        self.assertEqual(len(reopened.records()), 1, "index reloaded")
        self.assertIsNotNone(
            reopened.best_record(toks + [1], VaultTier.PREFILL), "and is queryable"
        )

    def test_a_directory_with_no_index_is_rebuilt_from_the_headers(self):
        dv = self.mkvault()
        toks = tokens_for(12, base=666)
        vault = ContextVault("ident-A", budget_bytes=1 << 30)
        vault.insert(toks, 12, frags_at(12, seed=30))
        dv.save_async(toks, vault.lookup(toks))
        dv.flush(30.0)
        dv.close()
        (self.root / "index.json").unlink()

        reopened = self.mkvault()
        self.assertEqual(len(reopened.records()), 1, "headers are self-describing")


# --------------------------------------------------------------------------
# 4. Disk cap: LRU deletion driven by the index
# --------------------------------------------------------------------------


class TestDiskCapLRU(DiskVaultTestCase):
    def test_the_least_recently_used_entry_is_deleted_when_over_the_cap(self):
        probe = self.mkvault()
        toks0 = tokens_for(12, base=10)
        v0 = ContextVault("ident-A", budget_bytes=1 << 30)
        v0.insert(toks0, 12, frags_at(12, seed=41))
        probe.save_async(toks0, v0.lookup(toks0))
        probe.flush(30.0)
        entry_bytes = probe.records()[0]["bytes"]
        probe.close()
        for p in self.root.iterdir():
            p.unlink()

        # A cap that holds two entries but not three.
        dv = self.mkvault(cap_bytes=int(entry_bytes * 2.5))
        keys = []
        for i in range(3):
            toks = tokens_for(12, base=1000 * (i + 1))
            v = ContextVault("ident-A", budget_bytes=1 << 30)
            v.insert(toks, 12, frags_at(12, seed=50 + i))
            dv.save_async(toks, v.lookup(toks))
            self.assertTrue(dv.flush(30.0))
            keys.append(dv.records()[-1]["key"])
            time.sleep(0.02)  # distinct last_used stamps

        recs = {r["key"]: r for r in dv.records()}
        self.assertEqual(len(recs), 2, "the cap held")
        self.assertEqual(self.stats.snapshot()["disk_evictions"], 1)
        self.assertEqual(len(dv.entry_files()), 2, "the file went, not just the record")
        self.assertLessEqual(dv.disk_bytes, dv.cap_bytes)

    def test_a_restore_refreshes_recency_so_it_is_not_the_next_victim(self):
        probe = self.mkvault()
        t0 = tokens_for(12, base=10)
        v0 = ContextVault("ident-A", budget_bytes=1 << 30)
        v0.insert(t0, 12, frags_at(12, seed=61))
        probe.save_async(t0, v0.lookup(t0))
        probe.flush(30.0)
        entry_bytes = probe.records()[0]["bytes"]
        probe.close()
        for p in self.root.iterdir():
            p.unlink()

        dv = self.mkvault(cap_bytes=int(entry_bytes * 2.5))
        token_sets = []
        for i in range(2):
            toks = tokens_for(12, base=2000 * (i + 1))
            v = ContextVault("ident-A", budget_bytes=1 << 30)
            v.insert(toks, 12, frags_at(12, seed=70 + i))
            dv.save_async(toks, v.lookup(toks))
            dv.flush(30.0)
            token_sets.append(toks)
            time.sleep(0.02)

        # Touch the OLDEST entry, then push a third one in.
        target = ContextVault("ident-A", budget_bytes=1 << 30)
        self.assertIsNotNone(
            dv.restore_into_vault(target, token_sets[0] + [1], VaultTier.PREFILL)
        )
        time.sleep(0.02)
        toks3 = tokens_for(12, base=9000)
        v3 = ContextVault("ident-A", budget_bytes=1 << 30)
        v3.insert(toks3, 12, frags_at(12, seed=72))
        dv.save_async(toks3, v3.lookup(toks3))
        self.assertTrue(dv.flush(30.0))

        survivors = {r["prompt_sha256"] for r in dv.records()}
        self.assertIn(
            VD.prompt_prefix_sha(token_sets[0], 12), survivors,
            "the recently restored entry must not be the LRU victim",
        )
        self.assertNotIn(VD.prompt_prefix_sha(token_sets[1], 12), survivors)


# --------------------------------------------------------------------------
# 5. Layout: one contiguous blob, >= 4 MiB records -- NOT 28 KB per token
# --------------------------------------------------------------------------


def big_fragment(nbytes):
    """One fp32 array of about ``nbytes``, wrapped as a checkpoint fragment."""
    n = nbytes // 4
    return [
        StateFragment(
            Capability.CHECKPOINT,
            1,
            payload=[mx.zeros((n,), dtype=mx.float32) + 1.5],
        )
    ]


class TestBlobLayout(DiskVaultTestCase):
    def test_one_entry_is_one_file(self):
        dv = self.mkvault()
        for i in range(3):
            toks = tokens_for(12, base=3000 * (i + 1))
            v = ContextVault("ident-A", budget_bytes=1 << 30)
            v.insert(toks, 12, frags_at(12, seed=80 + i))
            dv.save_async(toks, v.lookup(toks))
            dv.flush(30.0)
        self.assertEqual(len(dv.entry_files()), 3, "3 entries -> 3 files, not 3 x N")
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir() if p.suffix != ".vault"),
            ["index.json"],
            "the only non-entry file is the index",
        )

    def test_the_file_is_the_header_plus_the_tokens_plus_one_payload(self):
        dv = self.mkvault()
        toks = tokens_for(64, base=4000)
        v = ContextVault("ident-A", budget_bytes=1 << 30)
        v.insert(toks, 64, frags_at(64, seed=90))
        cp = v.lookup(toks)
        dv.save_async(toks, cp)
        dv.flush(30.0)
        key = dv.records()[0]["key"]
        header = VD.read_header(dv._entry_path(key))
        size = dv._entry_path(key).stat().st_size

        payload_bytes = sum(int(a.nbytes) for a in flat_arrays(cp.fragments))
        self.assertEqual(header["payload_nbytes"], payload_bytes)
        self.assertEqual(header["tokens_nbytes"], 4 * 64)
        self.assertEqual(size, header["payload_offset"] + header["payload_nbytes"])
        # Everything that is not payload is header + token ids + <8 KiB of
        # alignment: the file is the entry, not an envelope around N records.
        self.assertLess(size - payload_bytes, header["tokens_offset"] + 4 * 64 + 8192)

    def test_writes_and_reads_are_at_least_four_MiB(self):
        """P2: 1 MiB reads cost the internal NVMe half its bandwidth (3.2 vs 6.6
        GB/s) and 28,180 B records cost the external tier 4.9x median / 22x
        worst.  So the record size is a tested property, not a comment."""
        chunk = 4 << 20
        dv = self.mkvault(chunk_bytes=chunk)
        payload = 9 << 20  # 9 MiB: two full records and a tail
        frags = big_fragment(payload)
        v = ContextVault("ident-A", budget_bytes=1 << 30)
        toks = tokens_for(8, base=5000)
        self.assertTrue(v.insert(toks, 8, frags))
        dv.save_async(toks, v.lookup(toks))
        self.assertTrue(dv.flush(60.0))

        writes = dv.last_write_records
        self.assertGreaterEqual(len(writes), 2)
        self.assertTrue(
            all(w >= chunk for w in writes[:-1]),
            f"every write but the tail must be >= 4 MiB, got {writes}",
        )
        self.assertLess(
            len(writes), payload // 28180,
            "a 28,180 B per-token record layout would need far more writes",
        )

        loaded = dv.load_entry(dv.records()[0]["key"])
        self.assertIsNotNone(loaded)
        reads = [r for r in dv.last_read_records if r > 4 * 8]  # ignore the token block
        self.assertTrue(
            all(r >= chunk for r in reads[:-1]),
            f"every read but the tail must be >= 4 MiB, got {reads}",
        )

    def test_the_chunk_size_cannot_be_configured_below_the_measured_knee(self):
        os.environ[VD._ENV_CHUNK_MB] = "1"
        self.assertEqual(VD.read_chunk_bytes(), VD.MIN_CHUNK_BYTES)
        os.environ[VD._ENV_CHUNK_MB] = "64"
        self.assertEqual(VD.read_chunk_bytes(), 64 << 20)

    @unittest.skipUnless(os.uname().sysname == "Darwin", "F_NOCACHE is macOS-only")
    def test_nocache_is_actually_applied(self):
        dv = self.mkvault(nocache=True)
        toks = tokens_for(12, base=6000)
        v = ContextVault("ident-A", budget_bytes=1 << 30)
        v.insert(toks, 12, frags_at(12, seed=95))
        dv.save_async(toks, v.lookup(toks))
        dv.flush(30.0)
        self.assertTrue(dv.last_nocache_applied, "write fd took F_NOCACHE")
        dv.load_entry(dv.records()[0]["key"])
        self.assertTrue(dv.last_nocache_applied, "read fd took F_NOCACHE")

    def test_a_crash_mid_write_leaves_no_visible_entry(self):
        dv = self.mkvault()
        stray = self.root / "entry_deadbeef.vault.partial"
        stray.write_bytes(b"half a rung")
        reopened = self.mkvault()
        self.assertFalse(stray.exists(), "partials are swept on open")
        self.assertEqual(reopened.records(), [])


# --------------------------------------------------------------------------
# 6. Concurrency: a save in flight while a lookup restores a different entry
# --------------------------------------------------------------------------


class TestConcurrency(DiskVaultTestCase):
    def test_a_restore_runs_while_another_entry_is_being_written(self):
        dv = self.mkvault()
        # Entry A is already durable; entry B is a large write we start first.
        toks_a = tokens_for(12, base=7000)
        va = ContextVault("ident-A", budget_bytes=1 << 30)
        va.insert(toks_a, 12, frags_at(12, seed=101))
        cp_a = va.lookup(toks_a)
        want_a = payload_bytes(cp_a.fragments)
        dv.save_async(toks_a, cp_a)
        self.assertTrue(dv.flush(30.0))

        toks_b = tokens_for(8, base=8000)
        vb = ContextVault("ident-A", budget_bytes=1 << 30)
        vb.insert(toks_b, 8, big_fragment(24 << 20))
        cp_b = vb.lookup(toks_b)

        errors = []
        restored = {}

        def do_restore():
            try:
                target = ContextVault("ident-A", budget_bytes=1 << 30)
                cp = dv.restore_into_vault(target, toks_a + [1], VaultTier.PREFILL)
                restored["ok"] = cp is not None
                restored["prefix_len"] = None if cp is None else int(cp.prefix_len)
                # Materialise on THIS thread: MLX streams are thread-local.
                restored["bytes"] = None if cp is None else payload_bytes(cp.fragments)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        self.assertTrue(dv.save_async(toks_b, cp_b), "B is queued")
        t = threading.Thread(target=do_restore)
        t.start()
        t.join(timeout=60)
        self.assertFalse(t.is_alive(), "the restore must not block on the writer")
        self.assertEqual(errors, [])
        self.assertTrue(restored.get("ok"), "A restored while B was in flight")
        self.assertEqual(restored["prefix_len"], 12)
        self.assertEqual(restored["bytes"], want_a, "and byte-for-byte the same A")

        self.assertTrue(dv.flush(60.0))
        self.assertEqual(len(dv.entry_files()), 2, "both entries landed")
        self.assertEqual(self.stats.snapshot()["disk_save_errors"], 0)

    def test_concurrent_restores_of_the_same_entry_agree(self):
        dv = self.mkvault()
        toks = tokens_for(12, base=8500)
        v = ContextVault("ident-A", budget_bytes=1 << 30)
        v.insert(toks, 12, frags_at(12, seed=103))
        dv.save_async(toks, v.lookup(toks))
        dv.flush(30.0)

        want = payload_bytes(v.lookup(toks).fragments)
        out = []
        lock = threading.Lock()

        def worker():
            target = ContextVault("ident-A", budget_bytes=1 << 30)
            cp = dv.restore_into_vault(target, toks + [3], VaultTier.PREFILL)
            blob = None if cp is None else payload_bytes(cp.fragments)
            with lock:
                out.append(blob)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(len(out), 4)
        self.assertTrue(all(b is not None for b in out), "every restore succeeded")
        self.assertTrue(all(b == want for b in out), "and all agree byte-for-byte")

    def test_a_save_never_blocks_the_calling_thread(self):
        dv = self.mkvault()
        v = ContextVault("ident-A", budget_bytes=1 << 30)
        toks = tokens_for(8, base=8800)
        v.insert(toks, 8, big_fragment(48 << 20))
        cp = v.lookup(toks)
        t0 = time.perf_counter()
        dv.save_async(toks, cp)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 0.25, "save_async only enqueues")
        self.assertTrue(dv.flush(120.0))


# --------------------------------------------------------------------------
# 7. Env / wiring surface
# --------------------------------------------------------------------------


class TestEnvAndWiring(DiskVaultTestCase):
    def test_the_tier_is_off_unless_the_directory_is_set(self):
        os.environ.pop(VD._ENV_DIR, None)
        self.assertIsNone(VD.disk_vault_dir())
        self.assertFalse(VD.disk_vault_enabled())
        self.assertIsNone(VD.attach_disk_vault(ContextVault("x", budget_bytes=1)))

        os.environ[VD._ENV_DIR] = str(self.root / "attached")
        self.assertTrue(VD.disk_vault_enabled())
        vault = ContextVault("x", budget_bytes=1 << 20)
        dv = VD.attach_disk_vault(vault)
        self.assertIsNotNone(dv)
        self._open.append(dv)
        self.assertIs(vault.disk, dv)

    def test_defaults_match_the_documented_policy(self):
        for k in (VD._ENV_MAX_GB, VD._ENV_SAVE_ON_INSERT, VD._ENV_FSYNC,
                  VD._ENV_NOCACHE, VD._ENV_STRICT_GIT, VD._ENV_CHUNK_MB):
            os.environ.pop(k, None)
        self.assertEqual(VD.disk_cap_bytes(), 200 * (1000**3))
        self.assertFalse(VD.save_on_insert_enabled())
        self.assertFalse(VD.fsync_enabled())
        self.assertTrue(VD.nocache_enabled())
        self.assertTrue(VD.strict_git_head())
        self.assertEqual(VD.read_chunk_bytes(), 4 << 20)

    def test_no_startup_benchmark_runs(self):
        """The design says: measure per restore, never self-test at load."""
        t0 = time.perf_counter()
        dv = self.mkvault()
        self.assertLess(time.perf_counter() - t0, 2.0)
        snap = dv.stats_dict()
        self.assertEqual(snap["disk_bytes_read"], 0, "opening reads no payload")
        self.assertEqual(snap["disk_restores"], 0)

    def test_the_counters_ride_with_the_session_skips(self):
        import importlib

        _app = importlib.import_module("mlx_vlm.server.app")
        snap = _app._vault_stats_snapshot()
        self.assertIn("session_skips", snap)
        self.assertIn("disk_vault", snap)
        for name in ("disk_hits", "disk_misses", "disk_restore_seconds",
                     "disk_bytes_read", "disk_refusals"):
            self.assertIn(name, snap["disk_vault"])

    def test_a_vault_with_no_disk_tier_behaves_exactly_as_before(self):
        vault = ContextVault("ident-A", budget_bytes=1)
        self.assertIsNone(vault.disk)
        toks = tokens_for(12, base=9100)
        vault.insert(toks, 12, frags_at(12, seed=110))
        # Budget of 1 byte: the rung is refused, nothing raises, no disk touched.
        self.assertEqual(vault.rungs, 0)

    def test_the_model_identity_hash_reads_config_and_the_weight_index(self):
        tree = self.root / "model"
        tree.mkdir()
        (tree / "config.json").write_text('{"model_type":"glm5_next"}')
        (tree / "model.safetensors.index.json").write_text('{"weight_map":{"a":"1.st"}}')
        h1 = VD.model_identity_hash(tree)
        self.assertTrue(h1)
        (tree / "model.safetensors.index.json").write_text('{"weight_map":{"a":"2.st"}}')
        self.assertNotEqual(h1, VD.model_identity_hash(tree), "a reshard is a new model")
        self.assertEqual(VD.model_identity_hash(self.root / "nope"), "")

    def test_token_reconstruction_off_the_trie_is_exact(self):
        """The disk tier can only name what the trie can reproduce."""
        vault = ContextVault("ident-A", budget_bytes=1 << 30)
        a = tokens_for(30, base=1)
        b = a[:10] + [777] * 20
        vault.insert(a, 20, frags_at(20, seed=120))
        vault.insert(b, 25, frags_at(25, seed=121))
        seen = []
        for node in vault._iter_nodes():
            if node.checkpoint is not None:
                toks = ContextVault._tokens_for_node(node)
                self.assertIsNotNone(toks)
                self.assertEqual(len(toks), node.checkpoint.prefix_len)
                seen.append(toks)
        self.assertIn(a[:20], seen)
        self.assertIn(b[:25], seen)


if __name__ == "__main__":
    unittest.main()
