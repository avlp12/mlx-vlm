"""Vault capture de-duplicates the DSA latent, which glm5_next stores twice.

WHY THERE IS ANYTHING TO SAVE
-----------------------------
``models/glm5_next/language.py:1416`` caches the MLA latent as both halves of a
KVCache::

    kv_latent, _ = cache[0].update_and_fetch(kv_latent, kv_latent)

and ``models/cache.py:353-354`` allocates two *separate* buffers for keys and
values, then writes each independently at :365-366.  So every stored token holds
the latent twice.  Against the vault's own measured constant of 2562
B/token/DSA-layer (context_vault.py:28-38) the split is::

    latent keys                      512 x 2 B = 1024
    latent values (byte-identical)   512 x 2 B = 1024   <-- 40.0%
    indexer packed keys      (128+128+1) x 2 B =  514
                                                 ----
                                                 2562

Collapsing the duplicate takes a full 131k session from 3.578 GiB to 2.203 GiB,
i.e. 65 -> 105 resident sessions in a 250 GB budget.

WHAT THESE TESTS PIN
--------------------
1. the duplicate is collapsed at capture and the rung gets smaller;
2. a restored cache is BIT-identical to the pre-capture one -- compared through
   an unsigned integer view, not ``==``, because value-equality is wrong in both
   directions on this data (NaN != NaN would hide a real duplicate, +0.0 == -0.0
   would merge two buffers that differ in bytes);
3. the restored keys and values are SEPARATE buffers.  KVCache writes in place,
   so aliasing them would corrupt the cache on the first decode step after a
   restore -- this is the test that makes the optimisation safe rather than
   merely small;
4. the off-switch reproduces the pre-dedup payload exactly.

CPU device only: the vault is memory machinery and these run on a contended box.
"""

import os
import unittest

import mlx.core as mx

from mlx_vlm import apc_adapters as A
from mlx_vlm.context_vault import (
    ContextVault,
    capture_fragments,
    fragments_nbytes,
    restore_fragments,
)
from mlx_vlm.context_vault_wire import pack_fragments, unpack_fragments
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache

N_HEADS, LATENT, TOKENS = 2, 16, 8

_PREV_DEVICE = None


def setUpModule():
    """Pin to the CPU for the duration of this module only.

    The vault is memory machinery, so nothing here needs Metal, and these run on
    a box that other lanes are timing on.  Saved and restored rather than set at
    import time, so importing this module cannot quietly move the rest of the
    suite off the GPU.
    """
    global _PREV_DEVICE
    _PREV_DEVICE = mx.default_device()
    mx.set_default_device(mx.cpu)


def tearDownModule():
    if _PREV_DEVICE is not None:
        mx.set_default_device(_PREV_DEVICE)


def _bits(x):
    """Bit pattern of ``x`` as uint, so NaN and signed zero compare honestly."""
    u = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}[x.dtype.size]
    return mx.view(mx.contiguous(x), u)


def assert_same_bits(tc, a, b, label):
    tc.assertEqual(a.shape, b.shape, f"{label}: shape")
    tc.assertEqual(a.dtype, b.dtype, f"{label}: dtype")
    tc.assertTrue(bool(mx.array_equal(_bits(a), _bits(b)).item()), f"{label}: bits")


def dsa_layer_like_glm5(n=TOKENS, seed=0):
    """A DSA layer built the way the model builds it: ONE latent, passed twice."""
    entry = CacheList(KVCache(), KVCache())
    latent = mx.random.normal((1, N_HEADS, n, LATENT),
                              key=mx.random.key(seed)).astype(mx.bfloat16)
    entry.caches[0].update_and_fetch(latent, latent)          # language.py:1416
    packed = mx.random.normal((1, 1, n, 2 * LATENT + 1),
                              key=mx.random.key(seed + 7)).astype(mx.bfloat16)
    entry.caches[1].update_and_fetch(packed, mx.zeros((1, 1, n, 0), packed.dtype))
    mx.eval(entry.state)
    return entry


def kv(entry, sub):
    """The LIVE view of a KVCache half, sliced to ``offset``.

    ``.keys`` is the raw buffer, over-allocated in steps of 256 (cache.py:338),
    so comparing it against a restored cache compares 256 rows against 8.
    ``.state`` is the slice the vault actually stores and restores.
    """
    return entry.caches[sub].state


def _payload_of(frag, sub):
    return frag.payload[sub].payload["state"]


class TestLatentDedup(unittest.TestCase):
    def setUp(self):
        os.environ.pop(A._ENV_DEDUP, None)

    # ---------------------------------------------------------------- 1. saves
    def test_duplicate_latent_is_collapsed_at_capture(self):
        entry = dsa_layer_like_glm5()
        frag = capture_fragments([entry], TOKENS)[0]
        state = _payload_of(frag, 0)
        self.assertIsInstance(state[0], mx.array, "keys must be stored for real")
        self.assertTrue(A._is_alias(state[1]),
                        "the byte-identical values half must become an alias")
        self.assertEqual(state[1][1], 0, "alias must point at the keys entry")

    def test_rung_is_smaller_than_without_dedup(self):
        entry = dsa_layer_like_glm5()
        on = fragments_nbytes(capture_fragments([entry], TOKENS))
        os.environ[A._ENV_DEDUP] = "0"
        off = fragments_nbytes(capture_fragments([entry], TOKENS))
        latent_bytes = 1 * N_HEADS * TOKENS * LATENT * 2
        self.assertEqual(off - on, latent_bytes,
                         "exactly one latent copy must disappear")
        self.assertLess(on, off)

    # ------------------------------------------------------- 2. + 3. identity
    def test_restore_is_bit_identical_and_buffers_are_separate(self):
        entry = dsa_layer_like_glm5(seed=3)
        pre_k, pre_v = kv(entry, 0)
        (pre_idx, _) = kv(entry, 1)

        frag = capture_fragments([entry], TOKENS)
        fresh = CacheList(KVCache(), KVCache())
        self.assertTrue(restore_fragments([fresh], frag))

        new_k, new_v = kv(fresh, 0)
        assert_same_bits(self, new_k, pre_k, "latent keys")
        assert_same_bits(self, new_v, pre_v, "latent values")
        assert_same_bits(self, kv(fresh, 1)[0], pre_idx, "indexer keys")
        self.assertEqual(fresh.caches[0].offset, TOKENS)

        # The buffers must be distinct: KVCache writes in place (cache.py:365).
        k, v = fresh.caches[0].keys, fresh.caches[0].values
        v_before = mx.array(v)
        k[..., 0, :] = mx.zeros((1, N_HEADS, LATENT), k.dtype)
        mx.eval(k, v)
        assert_same_bits(self, v, v_before, "values after mutating keys")
        self.assertFalse(bool(mx.array_equal(_bits(k), _bits(v)).item()),
                         "mutating keys must not have moved values")

    def test_two_restores_do_not_share_a_buffer(self):
        frag = capture_fragments([dsa_layer_like_glm5(seed=4)], TOKENS)
        a, b = CacheList(KVCache(), KVCache()), CacheList(KVCache(), KVCache())
        restore_fragments([a], frag)
        restore_fragments([b], frag)
        b_before = mx.array(b.caches[0].values)
        a.caches[0].values[:] = mx.zeros_like(a.caches[0].values)
        mx.eval(a.caches[0].values, b.caches[0].values)
        assert_same_bits(self, b.caches[0].values, b_before, "second restore")

    # ------------------------------------------------- the two wrong predicates
    def test_signed_zero_is_not_deduped(self):
        """+0.0 == -0.0 under ``==``; their bytes differ, so they must not merge."""
        a = mx.array([[0.0, 0.0]], dtype=mx.bfloat16)
        b = mx.array([[-0.0, -0.0]], dtype=mx.bfloat16)
        self.assertTrue(bool(mx.array_equal(a, b).item()), "premise: == says equal")
        self.assertFalse(A._bytes_identical(a, b), "bytes differ -> must not merge")

    def test_nan_identical_is_deduped(self):
        """NaN != NaN under ``==``; identical bytes must still merge."""
        a = mx.array([[float("nan")] * 4], dtype=mx.bfloat16)
        b = mx.array(a)
        self.assertFalse(bool(mx.array_equal(a, b).item()), "premise: == says unequal")
        self.assertTrue(A._bytes_identical(a, b), "identical bytes -> must merge")

    def test_distinct_halves_are_left_alone(self):
        entry = CacheList(KVCache(), KVCache())
        k = mx.random.normal((1, N_HEADS, TOKENS, LATENT), key=mx.random.key(1))
        v = mx.random.normal((1, N_HEADS, TOKENS, LATENT), key=mx.random.key(2))
        entry.caches[0].update_and_fetch(k, v)
        entry.caches[1].update_and_fetch(k, mx.zeros((1, N_HEADS, TOKENS, 0), k.dtype))
        mx.eval(entry.state)
        state = _payload_of(capture_fragments([entry], TOKENS)[0], 0)
        self.assertIsInstance(state[1], mx.array, "distinct halves must both be kept")

    # ------------------------------------------------------------ 4. off-switch
    def test_off_switch_reproduces_the_undeduped_payload(self):
        entry = dsa_layer_like_glm5(seed=5)
        os.environ[A._ENV_DEDUP] = "0"
        frag = capture_fragments([entry], TOKENS)
        state = _payload_of(frag[0], 0)
        self.assertIsInstance(state[1], mx.array, "off must store both halves")
        fresh = CacheList(KVCache(), KVCache())
        restore_fragments([fresh], frag)
        assert_same_bits(self, kv(fresh, 0)[1], kv(entry, 0)[1], "off")

    # --------------------------------------------------------- peer-tier wire
    def test_alias_survives_the_peer_wire(self):
        frag = capture_fragments([dsa_layer_like_glm5(seed=6)], TOKENS)
        manifest, blob = pack_fragments(frag)
        back = unpack_fragments(manifest, blob)
        fresh = CacheList(KVCache(), KVCache())
        self.assertTrue(restore_fragments([fresh], back))
        ref = CacheList(KVCache(), KVCache())
        restore_fragments([ref], frag)
        assert_same_bits(self, kv(fresh, 0)[0], kv(ref, 0)[0], "wire keys")
        assert_same_bits(self, kv(fresh, 0)[1], kv(ref, 0)[1], "wire vals")

    # ------------------------------------------------- full vault, mixed stack
    def test_full_glm5_shaped_stack_through_the_vault(self):
        caches = [ArraysCache(size=2), dsa_layer_like_glm5(seed=8),
                  dsa_layer_like_glm5(seed=9)]
        caches[0][0] = mx.random.normal((1, N_HEADS, LATENT, 4), dtype=mx.float32)
        caches[0][1] = mx.random.normal((1, N_HEADS, LATENT, LATENT), dtype=mx.float32)
        mx.eval([c.state for c in caches])
        toks = list(range(64))
        v = ContextVault("dedup-test", budget_bytes=1 << 30)
        self.assertTrue(v.insert(toks, TOKENS, capture_fragments(caches, TOKENS)))
        cp = v.lookup(toks)
        self.assertIsNotNone(cp)
        fresh = [ArraysCache(size=2), CacheList(KVCache(), KVCache()),
                 CacheList(KVCache(), KVCache())]
        self.assertTrue(v.restore_into(fresh, cp))
        for i in (1, 2):
            for half, name in enumerate(("keys", "values")):
                assert_same_bits(self, kv(fresh[i], 0)[half],
                                 kv(caches[i], 0)[half], f"L{i}.{name}")
        assert_same_bits(self, fresh[0][1], caches[0][1], "KDA state")


if __name__ == "__main__":
    unittest.main()
