"""Warm Context Vault: trie semantics, budget/LRU, and restore bit-identity.

These run on synthetic caches shaped exactly like ``glm5_next.make_cache()`` —
``ArraysCache(size=2)`` for KDA linear layers and ``CacheList(KVCache, KVCache)``
for DSA attention layers — so the parity proof exercises the real adapter path
without loading 169 GB of weights.
"""

import os
import unittest

import mlx.core as mx

from mlx_vlm.context_vault import (
    ContextVault,
    align_boundaries,
    boundary_ladder,
    capture_fragments,
    fragments_nbytes,
    get_vault,
    reset_vault,
    restore_fragments,
    vault_identity,
)
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache

KDA_LAYERS = 3
DSA_LAYERS = 2
N_HEADS = 2
HEAD_DIM = 4


def make_glm5_shaped_cache():
    """A miniature of the real 45-layer hybrid stack: 34 linear + 11 attention."""
    caches = []
    for _ in range(KDA_LAYERS):
        caches.append(ArraysCache(size=2))
    for _ in range(DSA_LAYERS):
        caches.append(CacheList(KVCache(), KVCache()))
    return caches


def drive(caches, n_tokens, seed=0):
    """Push ``n_tokens`` of deterministic state through every component."""
    for i, c in enumerate(caches):
        if isinstance(c, ArraysCache):
            # KDA: conv window + recurrent state, flat in sequence length, fp32.
            key = mx.random.key(seed * 131 + i)
            c[0] = mx.random.normal((1, N_HEADS, HEAD_DIM, 4), key=key, dtype=mx.float32)
            c[1] = mx.random.normal((1, N_HEADS, HEAD_DIM, HEAD_DIM), key=key, dtype=mx.float32)
        else:
            for j, sub in enumerate(c.caches):
                key = mx.random.key(seed * 977 + i * 13 + j)
                k = mx.random.normal((1, N_HEADS, n_tokens, HEAD_DIM), key=key)
                v = mx.random.normal((1, N_HEADS, n_tokens, HEAD_DIM), key=key)
                sub.update_and_fetch(k, v)
    mx.eval([c.state for c in caches])
    return caches


def state_signature(caches):
    """Flat list of concrete arrays for bit-comparison."""
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


def assert_bit_identical(testcase, a, b, label):
    sa, sb = state_signature(a), state_signature(b)
    testcase.assertEqual(len(sa), len(sb), f"{label}: component count differs")
    for i, (x, y) in enumerate(zip(sa, sb)):
        testcase.assertEqual(x.shape, y.shape, f"{label}: shape differs at {i}")
        testcase.assertEqual(x.dtype, y.dtype, f"{label}: dtype differs at {i}")
        # Byte-level equality, not allclose: fp32 KDA state must round-trip exactly.
        xb = mx.view(x.flatten(), mx.uint8)
        yb = mx.view(y.flatten(), mx.uint8)
        testcase.assertTrue(bool(mx.all(xb == yb).item()), f"{label}: bytes differ at {i}")


class TestCaptureRestoreParity(unittest.TestCase):
    def test_roundtrip_is_bit_identical(self):
        original = drive(make_glm5_shaped_cache(), 16, seed=1)
        frags = capture_fragments(original, 16)
        self.assertIsNotNone(frags, "glm5-shaped cache must be fully capturable")

        fresh = make_glm5_shaped_cache()
        self.assertTrue(restore_fragments(fresh, frags))
        assert_bit_identical(self, original, fresh, "restore")

    def test_restore_is_detached_from_source(self):
        original = drive(make_glm5_shaped_cache(), 8, seed=2)
        frags = capture_fragments(original, 8)
        # Mutating the source after capture must not disturb the stored rung.
        drive(original, 8, seed=99)
        fresh = make_glm5_shaped_cache()
        restore_fragments(fresh, frags)
        again = make_glm5_shaped_cache()
        restore_fragments(again, frags)
        assert_bit_identical(self, fresh, again, "detach")

    def test_kv_offset_restored(self):
        original = drive(make_glm5_shaped_cache(), 12, seed=3)
        frags = capture_fragments(original, 12)
        fresh = make_glm5_shaped_cache()
        restore_fragments(fresh, frags)
        for c in fresh[KDA_LAYERS:]:
            for sub in c.caches:
                self.assertEqual(sub.offset, 12, "KV offset must resume mid-prefix")

    def test_fragments_nbytes_positive(self):
        original = drive(make_glm5_shaped_cache(), 32, seed=4)
        frags = capture_fragments(original, 32)
        self.assertGreater(fragments_nbytes(frags), 0)


class TestRadixTrie(unittest.TestCase):
    def _vault(self, budget=1 << 30):
        return ContextVault("test-identity", budget_bytes=budget)

    def _store(self, vault, tokens, depth, seed=0):
        caches = drive(make_glm5_shaped_cache(), depth, seed=seed)
        return vault.insert(tokens, depth, capture_fragments(caches, depth))

    def test_longest_prefix_wins(self):
        v = self._vault()
        doc = list(range(1000, 1064))
        self.assertTrue(self._store(v, doc, 16, seed=1))
        self.assertTrue(self._store(v, doc, 32, seed=2))
        self.assertTrue(self._store(v, doc, 48, seed=3))
        hit = v.lookup(doc + [7, 7, 7])
        self.assertIsNotNone(hit)
        self.assertEqual(hit.prefix_len, 48, "must restore the deepest rung <= N")

    def test_partial_prefix_hit_on_new_suffix(self):
        """The workload the vault exists for: same doc, different suffix."""
        v = self._vault()
        doc = list(range(500, 628))
        self._store(v, doc, 128, seed=5)
        for suffix in ([1, 2, 3], [9], list(range(40, 90))):
            hit = v.lookup(doc + suffix)
            self.assertIsNotNone(hit, "stored document must hit under any suffix")
            self.assertEqual(hit.prefix_len, 128)

    def test_divergent_branch_falls_back_to_shared_rung(self):
        v = self._vault()
        shared = list(range(300, 332))
        a = shared + list(range(700, 732))
        b = shared + list(range(800, 832))
        self._store(v, shared, 32, seed=6)
        self._store(v, a, 64, seed=7)
        hit = v.lookup(b)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.prefix_len, 32, "diverged query falls back to shared prefix")

    def test_edge_split_preserves_both_branches(self):
        v = self._vault()
        base = list(range(10, 74))
        a = base + [1] * 32
        b = base + [2] * 32
        self._store(v, a, 96, seed=8)
        self._store(v, b, 96, seed=9)
        self._store(v, base, 32, seed=10)
        self.assertEqual(v.lookup(a).prefix_len, 96)
        self.assertEqual(v.lookup(b).prefix_len, 96)
        self.assertEqual(v.lookup(base + [3] * 8).prefix_len, 32)

    def test_miss_returns_none(self):
        v = self._vault()
        self._store(v, list(range(100, 132)), 32, seed=11)
        self.assertIsNone(v.lookup([999, 998, 997]))

    def test_shorter_query_than_boundary_misses_that_rung(self):
        v = self._vault()
        doc = list(range(200, 264))
        self._store(v, doc, 64, seed=12)
        # A query shorter than the boundary cannot use it.
        self.assertIsNone(v.lookup(doc[:10]))


class TestBudgetAndEviction(unittest.TestCase):
    def test_lru_eviction_respects_budget(self):
        caches = drive(make_glm5_shaped_cache(), 16, seed=20)
        rung_bytes = fragments_nbytes(capture_fragments(caches, 16))
        v = ContextVault("budget-test", budget_bytes=int(rung_bytes * 2.5))
        docs = [list(range(i * 1000, i * 1000 + 32)) for i in range(4)]
        for i, d in enumerate(docs):
            c = drive(make_glm5_shaped_cache(), 16, seed=30 + i)
            v.insert(d, 16, capture_fragments(c, 16))
        self.assertLessEqual(v.resident_bytes, v.budget, "must not exceed budget")
        self.assertGreater(v.stats.evictions, 0, "eviction must have fired")

    def test_oversized_rung_rejected(self):
        v = ContextVault("tiny", budget_bytes=16)
        c = drive(make_glm5_shaped_cache(), 16, seed=40)
        self.assertFalse(v.insert(list(range(32)), 16, capture_fragments(c, 16)))
        self.assertEqual(v.rungs, 0)

    def test_unsupported_capture_is_rejected_not_stored(self):
        v = ContextVault("unsupported", budget_bytes=1 << 30)
        self.assertFalse(v.insert(list(range(8)), 4, None))
        self.assertEqual(v.stats.rejected_unsupported, 1)

    def test_stats_dict_shape(self):
        v = ContextVault("stats", budget_bytes=1 << 30)
        c = drive(make_glm5_shaped_cache(), 16, seed=50)
        v.insert(list(range(32)), 16, capture_fragments(c, 16))
        v.lookup(list(range(32)))
        d = v.stats_dict()
        for k in ("hits", "misses", "hit_rate", "gb_resident", "budget_gb", "utilization"):
            self.assertIn(k, d)
        self.assertEqual(d["hits"], 1)


class TestBoundaryPolicy(unittest.TestCase):
    def test_align_rejects_unaligned_boundaries(self):
        self.assertEqual(align_boundaries([2048, 3000, 4096], 2048, 10000), [2048, 4096])

    def test_align_rejects_boundary_at_or_past_total(self):
        self.assertEqual(align_boundaries([4096, 8192], 2048, 8192), [4096])

    def test_geometric_ladder_is_aligned_and_halving(self):
        rungs = boundary_ladder(131_072, stride=8192, step=2048, max_rungs=8)
        self.assertEqual(rungs, [8192, 24576, 57344, 122880])
        self.assertTrue(all(r % 2048 == 0 and r % 8192 == 0 for r in rungs))
        self.assertEqual(rungs, sorted(rungs))
        self.assertLess(rungs[-1], 131_072)

    def test_geometric_is_cheaper_than_uniform(self):
        """The ladder policy must not re-pay the flat KDA term at every stride."""
        geo = boundary_ladder(131_072, 8192, 2048, 8, "geometric")
        uni = boundary_ladder(131_072, 8192, 2048, 8, "uniform")
        kda, dsa = 140.8 / 1024, 28182 / 1024**3
        cost = lambda r: sum(kda + L * dsa for L in r)
        self.assertLess(cost(geo), cost(uni) / 3)
        # ...while still covering a wider span of divergence points.
        self.assertLess(geo[0], uni[0])

    def test_uniform_mode_keeps_deepest(self):
        rungs = boundary_ladder(100_000, 8192, 2048, 3, "uniform")
        self.assertEqual(len(rungs), 3)
        self.assertEqual(rungs[-1], 8192 * (99_999 // 8192))

    def test_max_rungs_respected(self):
        self.assertLessEqual(len(boundary_ladder(131_072, 8192, 2048, 2)), 2)

    def test_ladder_empty_for_short_prompt(self):
        self.assertEqual(boundary_ladder(1000, stride=8192, step=2048), [])

    def test_ladder_empty_when_total_equals_stride(self):
        self.assertEqual(boundary_ladder(8192, stride=8192, step=2048), [])


class TestIdentityInvalidation(unittest.TestCase):
    def tearDown(self):
        reset_vault()
        os.environ.pop("MLX_VLM_GLM5_FUSED_KDA", None)

    def test_toggle_change_changes_identity(self):
        os.environ["MLX_VLM_GLM5_FUSED_KDA"] = "1"
        a = vault_identity("/models/glm53")
        os.environ["MLX_VLM_GLM5_FUSED_KDA"] = "0"
        b = vault_identity("/models/glm53")
        self.assertNotEqual(a, b, "a kernel toggle must invalidate stored state")

    def test_model_path_change_changes_identity(self):
        self.assertNotEqual(vault_identity("/models/a"), vault_identity("/models/b"))

    def test_get_vault_rebuilds_on_identity_change(self):
        v1 = get_vault("id-one")
        c = drive(make_glm5_shaped_cache(), 16, seed=60)
        v1.insert(list(range(32)), 16, capture_fragments(c, 16))
        self.assertEqual(v1.rungs, 1)
        v2 = get_vault("id-two")
        self.assertIsNot(v1, v2)
        self.assertEqual(v2.rungs, 0, "identity change must drop the store")

    def test_get_vault_is_stable_for_same_identity(self):
        self.assertIs(get_vault("same"), get_vault("same"))


if __name__ == "__main__":
    unittest.main()


class TestCheckpointLadder(unittest.TestCase):
    """Ordering contract of the boundary consumer used by the prefill loop."""

    def _run(self, boundaries, total, step):
        """Replay a chunked prefill, returning (fired, chunk_sizes)."""
        from mlx_vlm.context_vault import CheckpointLadder

        ladder = CheckpointLadder(boundaries, total)
        processed, fired, sizes = 0, [], []
        remaining = total
        while remaining > 1:
            n = min(step, remaining - 1)
            n = ladder.clamp(processed, n)
            sizes.append(n)
            processed += n
            remaining -= n
            fired.extend(ladder.reached(processed))
        return fired, sizes

    def test_fires_every_aligned_boundary(self):
        fired, _ = self._run([2048, 4096, 6144], 10000, 2048)
        self.assertEqual(fired, [2048, 4096, 6144])

    def test_aligned_boundaries_do_not_change_chunking(self):
        _, with_cp = self._run([2048, 4096], 10000, 2048)
        _, without = self._run([], 10000, 2048)
        self.assertEqual(
            with_cp, without, "aligned boundaries must not perturb chunk sizes"
        )

    def test_unaligned_boundary_splits_a_chunk(self):
        _, sizes = self._run([3000], 10000, 2048)
        self.assertIn(952, sizes, "unaligned boundary clamps the chunk")
        self.assertNotEqual(sizes, self._run([], 10000, 2048)[1])

    def test_boundary_at_or_past_total_is_dropped(self):
        fired, _ = self._run([10000, 20000], 10000, 2048)
        self.assertEqual(fired, [])

    def test_single_int_contract_preserved(self):
        from mlx_vlm.context_vault import CheckpointLadder

        ladder = CheckpointLadder([4096], 10000)
        self.assertEqual(ladder.clamp(2048, 2048), 2048)
        self.assertEqual(ladder.reached(4096), [4096])
        self.assertFalse(ladder)

    def test_duplicate_and_unsorted_boundaries_normalized(self):
        from mlx_vlm.context_vault import CheckpointLadder

        ladder = CheckpointLadder([4096, 2048, 4096, -5, 0], 10000)
        self.assertEqual(ladder.pending, [2048, 4096])
