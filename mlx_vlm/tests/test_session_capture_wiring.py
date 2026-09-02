"""End-of-turn session capture: policy layer + the BatchGenerator entry point.

Unit level, CPU, no model.  The live validation (a real server, a real second
turn) still needs a box; what is pinned here is everything that can be got wrong
without one -- the flag defaults, the required session id, the prefix_len the
rung is stored under, and the refusals.
"""

import os
import unittest

import mlx.core as mx

from mlx_vlm import context_vault as V
from mlx_vlm.generate.ar import BatchGenerator
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache

H, D = 2, 8
_PREV = None


def setUpModule():
    global _PREV
    _PREV = mx.default_device()
    mx.set_default_device(mx.cpu)


def tearDownModule():
    if _PREV is not None:
        mx.set_default_device(_PREV)


def row_cache(n=8):
    c = [ArraysCache(size=2), CacheList(KVCache(), KVCache())]
    c[0][0] = mx.zeros((1, H, D, 4), mx.float32)
    c[0][1] = mx.zeros((1, H, D, D), mx.float32)
    lat = mx.zeros((1, H, n, D), mx.bfloat16)
    c[1].caches[0].update_and_fetch(lat, lat)
    idx = mx.zeros((1, 1, n, 2 * D + 1), mx.bfloat16)
    c[1].caches[1].update_and_fetch(idx, mx.zeros((1, 1, n, 0), mx.bfloat16))
    mx.eval([e.state for e in c])
    return c


class _Batch:
    def __init__(self, uids, cache):
        self.uids = list(uids)
        self.prompt_cache = cache


class _Gen:
    """Only the attributes capture_session touches."""
    def __init__(self, vault, uids, cache):
        self.vault = vault
        self._generation_batch = _Batch(uids, cache)


def capture(gen, uid, tokens, session_id, ttl_s=None):
    return BatchGenerator.capture_session(
        gen, uid, tokens, session_id=session_id, ttl_s=ttl_s)


class TestFlagDefaults(unittest.TestCase):
    def setUp(self):
        for k in (V._ENV_SESSION, V._ENV_SESSION_DERIVED_ID):
            os.environ.pop(k, None)

    def test_capture_is_off_by_default(self):
        """It fires on the response path, so it stays off until validated."""
        self.assertFalse(V.session_capture_enabled())

    def test_derived_session_id_is_off_by_default(self):
        self.assertFalse(V.derived_session_id_allowed())

    def test_missing_session_id_is_refused_not_derived(self):
        v = V.ContextVault("wire", budget_bytes=1 << 30)
        self.assertFalse(V.record_session_turn(
            v, list(range(1, 9)), row_cache(8), completed=True, session_id=""))
        self.assertEqual(v.rungs, 0)

    def test_derived_id_is_available_only_behind_the_flag(self):
        os.environ[V._ENV_SESSION_DERIVED_ID] = "1"
        v = V.ContextVault("wire", budget_bytes=1 << 30)
        self.assertTrue(V.record_session_turn(
            v, list(range(1, 9)), row_cache(8), completed=True, session_id=""))


class TestPrefixLenFromCache(unittest.TestCase):
    def test_offset_is_taken_from_the_cache_not_the_token_count(self):
        """The last sampled token is not fed back, so len(tokens) != offset."""
        self.assertEqual(V.prefix_len_from_cache(row_cache(8)), 8)

    def test_disagreeing_offsets_refuse(self):
        c = row_cache(8)
        c[1].caches[1].update_and_fetch(
            mx.zeros((1, 1, 1, 2 * D + 1), mx.bfloat16),
            mx.zeros((1, 1, 1, 0), mx.bfloat16))
        self.assertIsNone(V.prefix_len_from_cache(c),
                          "a half-consistent cache must not be stored")

    def test_empty_cache_refuses(self):
        self.assertIsNone(V.prefix_len_from_cache([]))
        self.assertIsNone(V.prefix_len_from_cache(
            [ArraysCache(size=2)]), "no offset anywhere -> nothing to key on")

    def test_rung_is_stored_under_the_cache_offset(self):
        v = V.ContextVault("wire", budget_bytes=1 << 30)
        toks = list(range(1, 10))          # 9 tokens, 8 of them in the cache
        self.assertTrue(V.record_session_turn(
            v, toks, row_cache(8), completed=True, session_id="conv-1"))
        cp = V.lookup_session(v, toks)
        self.assertEqual(cp.prefix_len, 8,
                         "the un-fed final token must not be claimed as cached")

    def test_cache_longer_than_the_key_is_refused(self):
        v = V.ContextVault("wire", budget_bytes=1 << 30)
        self.assertFalse(V.record_session_turn(
            v, [1, 2, 3], row_cache(8), completed=True, session_id="conv-1"))
        self.assertEqual(v.rungs, 0)


class TestBatchGeneratorEntryPoint(unittest.TestCase):
    def setUp(self):
        for k in (V._ENV_SESSION, V._ENV_SESSION_DERIVED_ID):
            os.environ.pop(k, None)

    def _gen(self):
        v = V.ContextVault("wire", budget_bytes=1 << 30)
        return v, _Gen(v, ["u1"], row_cache(8))

    def test_no_op_while_the_flag_is_off(self):
        v, g = self._gen()
        self.assertFalse(capture(g, "u1", list(range(1, 10)), "conv-1"))
        self.assertEqual(v.rungs, 0)

    def test_stores_when_enabled(self):
        os.environ[V._ENV_SESSION] = "1"
        v, g = self._gen()
        self.assertTrue(capture(g, "u1", list(range(1, 10)), "conv-1"))
        self.assertEqual(v.stats.session_inserts, 1)

    def test_unknown_uid_stores_nothing(self):
        """Called after remove() the row is gone; refuse rather than guess."""
        os.environ[V._ENV_SESSION] = "1"
        v, g = self._gen()
        self.assertFalse(capture(g, "gone", list(range(1, 10)), "conv-1"))
        self.assertEqual(v.rungs, 0)

    def test_missing_session_id_stores_nothing(self):
        os.environ[V._ENV_SESSION] = "1"
        v, g = self._gen()
        self.assertFalse(capture(g, "u1", list(range(1, 10)), ""))

    def test_no_vault_is_a_no_op(self):
        os.environ[V._ENV_SESSION] = "1"
        g = _Gen(None, ["u1"], row_cache(8))
        self.assertFalse(capture(g, "u1", list(range(1, 10)), "conv-1"))

    def test_an_exploding_batch_never_raises(self):
        os.environ[V._ENV_SESSION] = "1"
        class Boom:
            @property
            def uids(self):
                raise RuntimeError("boom")
        v = V.ContextVault("wire", budget_bytes=1 << 30)
        g = _Gen(v, [], None)
        g._generation_batch = Boom()
        self.assertFalse(capture(g, "u1", [1, 2], "conv-1"))


if __name__ == "__main__":
    unittest.main()
