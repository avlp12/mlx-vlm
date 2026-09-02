"""Session-tier restore in `_vault_pick_for` (docs/vault_session_restore.md).

Fake store, no model: `cache.make_prompt_cache` is patched to hand back a cache
shaped like the one that was captured. Behind MLX_VLM_GLM5_VAULT_SESSION.
"""

import os
import unittest
from unittest import mock

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


def fresh():
    return [ArraysCache(size=2), CacheList(KVCache(), KVCache())]


def filled(n):
    c = fresh()
    c[0][0] = mx.zeros((1, H, D, 4), mx.float32)
    c[0][1] = mx.zeros((1, H, D, D), mx.float32)
    lat = mx.zeros((1, H, n, D), mx.bfloat16)
    c[1].caches[0].update_and_fetch(lat, lat)
    idx = mx.zeros((1, 1, n, 2 * D + 1), mx.bfloat16)
    c[1].caches[1].update_and_fetch(idx, mx.zeros((1, 1, n, 0), mx.bfloat16))
    mx.eval([e.state for e in c])
    return c


class _Gen:
    """Only what _vault_pick_for touches."""
    def __init__(self, vault):
        self.vault = vault
        self.model = object()
        self.apc_manager = None

    def _vault_prefix_trim_is_safe(self):
        return True

    def _apc_extra_hash(self, kw):
        return 0


def _isolate_session_env(testcase):
    """Restore the session flags after the test.

    Leaking MLX_VLM_GLM5_VAULT_SESSION=1 out of a module makes a LATER module's
    duck-typed vault see a tier kwarg it has no parameter for -- which is how it
    was found: test_vault_server_wiring passed alone and failed in the full run.
    """
    saved = {k: os.environ.get(k)
             for k in (V._ENV_SESSION, V._ENV_SESSION_DERIVED_ID)}

    def restore():
        for k, val in saved.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
    testcase.addCleanup(restore)
    for k in saved:
        os.environ.pop(k, None)


def pick_for(gen, ids, pick=None):
    with mock.patch("mlx_vlm.generate.ar.cache.make_prompt_cache",
                    side_effect=lambda m: fresh()):
        return BatchGenerator._vault_pick_for(gen, ids, {}, pick)


class TestSessionRestore(unittest.TestCase):
    def setUp(self):
        _isolate_session_env(self)
        self.v = V.ContextVault("restore", budget_bytes=1 << 30)
        self.ids = list(range(1, 33))

    def _store(self, tier, depth, session_id=""):
        self.v.insert(self.ids, depth, V.capture_fragments(filled(depth), depth),
                      tier=tier, session_id=session_id)

    def test_session_rung_is_ignored_while_the_flag_is_off(self):
        self._store(V.VaultTier.SESSION, 24, "c1")
        self.assertIsNone(pick_for(_Gen(self.v), self.ids),
                          "off must reduce to the prefill lookup exactly")

    def test_session_rung_is_taken_and_reported_as_vault_session(self):
        os.environ[V._ENV_SESSION] = "1"
        self._store(V.VaultTier.SESSION, 24, "c1")
        got = pick_for(_Gen(self.v), self.ids)
        self.assertIsNotNone(got)
        self.assertEqual(got["prefix_len"], 24)
        self.assertEqual(got["source"], "vault-session")

    def test_the_deeper_tier_wins(self):
        os.environ[V._ENV_SESSION] = "1"
        self._store(V.VaultTier.PREFILL, 8)
        self._store(V.VaultTier.SESSION, 24, "c1")
        self.assertEqual(pick_for(_Gen(self.v), self.ids)["source"], "vault-session")

        v2 = V.ContextVault("restore2", budget_bytes=1 << 30)
        v2.insert(self.ids, 24, V.capture_fragments(filled(24), 24))
        v2.insert(self.ids, 8, V.capture_fragments(filled(8), 8),
                  tier=V.VaultTier.SESSION, session_id="c1")
        self.assertEqual(pick_for(_Gen(v2), self.ids)["source"], "vault")

    def test_a_shallower_hit_than_apc_is_declined(self):
        os.environ[V._ENV_SESSION] = "1"
        self._store(V.VaultTier.SESSION, 8, "c1")
        held = {"prefix_len": 16, "matched_blocks": []}
        self.assertIs(pick_for(_Gen(self.v), self.ids, held), held)

    def test_exact_length_match_is_declined(self):
        """The strict < leaves at least one column for the generate loop."""
        os.environ[V._ENV_SESSION] = "1"
        self._store(V.VaultTier.SESSION, len(self.ids), "c1")
        self.assertIsNone(pick_for(_Gen(self.v), self.ids))

    def test_a_diverged_prompt_misses_and_leaves_the_store_untouched(self):
        os.environ[V._ENV_SESSION] = "1"
        self._store(V.VaultTier.SESSION, 24, "c1")
        before = self.v.resident_bytes
        diverged = self.ids[:12] + [999] * 20
        self.assertIsNone(pick_for(_Gen(self.v), diverged))
        self.assertEqual(self.v.resident_bytes, before, "a miss must not evict")
        self.assertEqual(self.v.rungs, 1)

    def test_restore_is_called_with_the_matching_tier(self):
        os.environ[V._ENV_SESSION] = "1"
        self._store(V.VaultTier.SESSION, 24, "c1")
        seen = {}
        real = self.v.restore_into

        def spy(caches, cp, tier=V.VaultTier.PREFILL):
            seen["tier"] = tier
            return real(caches, cp, tier=tier)
        self.v.restore_into = spy
        self.assertIsNotNone(pick_for(_Gen(self.v), self.ids))
        self.assertIs(seen["tier"], V.VaultTier.SESSION,
                      "a session rung restored as PREFILL is refused by design")


if __name__ == "__main__":
    unittest.main()
