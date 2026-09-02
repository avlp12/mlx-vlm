"""Resident session store: the implementation of the T1-T13 contract.

The contract and its reasoning live in ``test_session_store_design.py``; this
file is the executable half.  Every test here names the T-number it discharges.

Synthetic caches shaped like ``glm5_next.make_cache()``, CPU device, no model.
"""

import time
import unittest

import mlx.core as mx

from mlx_vlm.context_vault import (
    ContextVault,
    VaultTier,
    capture_fragments,
    lookup_session,
    record_session_turn,
    restore_session,
    session_id_for,
)
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


def _bits(x):
    u = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}[x.dtype.size]
    return mx.view(mx.contiguous(x), u)


def same_bits(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype
            and bool(mx.array_equal(_bits(a), _bits(b)).item()))


def fresh_cache():
    return [ArraysCache(size=2), CacheList(KVCache(), KVCache())]


def step(caches, token):
    """One deterministic decode step whose result DEPENDS on prior state.

    T5 is meaningless against a fixture that overwrites state, so this threads
    the previous value through: the recurrent half decays and accumulates, the
    attention half appends a row derived from the token.
    """
    t = float(token)
    kda = caches[0]
    prev = kda[1] if kda[1] is not None else mx.zeros((1, H, D, D), mx.float32)
    kda[0] = mx.full((1, H, D, 4), t, mx.float32)
    kda[1] = prev * 0.5 + t
    lat = mx.full((1, H, 1, D), t, mx.bfloat16)
    caches[1].caches[0].update_and_fetch(lat, lat)     # language.py:1416 shape
    idx = mx.full((1, 1, 1, 2 * D + 1), t, mx.bfloat16)
    caches[1].caches[1].update_and_fetch(idx, mx.zeros((1, 1, 1, 0), mx.bfloat16))
    mx.eval([c.state for c in caches])
    return caches


def drive(tokens):
    c = fresh_cache()
    for t in tokens:
        step(c, t)
    return c


def state_bits_equal(a, b):
    for ca, cb in zip(a, b):
        xa = ca.state if not isinstance(ca, CacheList) else [s.state for s in ca.caches]
        xb = cb.state if not isinstance(cb, CacheList) else [s.state for s in cb.caches]
        fa, fb = [], []

        def flat(o, out):
            if isinstance(o, mx.array):
                out.append(o)
            elif isinstance(o, (list, tuple)):
                for i in o:
                    flat(i, out)
        flat(xa, fa)
        flat(xb, fb)
        if len(fa) != len(fb) or not all(same_bits(x, y) for x, y in zip(fa, fb)):
            return False
    return True


class TestSessionCapture(unittest.TestCase):
    def _vault(self, budget=1 << 30):
        return ContextVault("session-test", budget_bytes=budget)

    def test_T1_capture_fires_when_a_response_completes(self):
        v = self._vault()
        toks = list(range(1, 33))
        self.assertTrue(record_session_turn(v, toks, drive(toks), completed=True,
                                     session_id="conv-A"))
        self.assertEqual(v.stats.session_inserts, 1)
        cp = lookup_session(v, toks + [99])
        self.assertIsNotNone(cp, "next turn must hit the end-of-turn rung")
        self.assertEqual(cp.prefix_len, len(toks),
                         "only the new user message should be left to prefill")
        self.assertEqual(cp.session_id, "conv-A",
                         "the server's conversation id, not a token-derived one")

    def test_T2_capture_is_skipped_when_the_stream_was_aborted(self):
        v = self._vault()
        toks = list(range(1, 17))
        self.assertFalse(record_session_turn(v, toks, drive(toks), completed=False,
                                      session_id="conv-A"))
        self.assertEqual(v.rungs, 0)
        self.assertIsNone(lookup_session(v, toks))

    def test_T3_adopt_does_not_copy_the_state_tree(self):
        """Adopt hands over buffers instead of duplicating them.

        Asserted by observing the aliasing directly -- which is also exactly why
        adopt is only ever used at end of turn, on a cache nobody will touch
        again.  The copying path is asserted to NOT alias, so the two modes are
        pinned against each other rather than one being taken on trust.
        """
        c = drive(range(1, 9))
        adopted = capture_fragments(c, 8, adopt=True)
        copied = capture_fragments(c, 8, adopt=False)
        kda_state = adopted[0].payload["state"]
        self.assertIs(kda_state[1], c[0][1], "adopt must store the caller's array")
        self.assertIsNot(copied[0].payload["state"][1], c[0][1],
                         "the default path must still copy")


class TestIdentityTiers(unittest.TestCase):
    def _vault(self):
        return ContextVault("tier-test", budget_bytes=1 << 30)

    def test_T4_session_rungs_never_serve_a_prefill_query(self):
        v = self._vault()
        toks = list(range(1, 33))
        record_session_turn(v, toks, drive(toks), completed=True,
                                     session_id="conv-A")
        self.assertIsNotNone(v.lookup(toks, tier=VaultTier.SESSION))
        self.assertIsNone(v.lookup(toks),
                          "a prefill lookup must not reach into the session tier")
        self.assertIsNone(v.lookup(toks, tier=VaultTier.PREFILL))

    def test_T5_restore_then_continue_is_bit_identical_to_never_stopping(self):
        v = self._vault()
        head, tail = list(range(1, 25)), list(range(100, 108))
        stopped = drive(head)
        record_session_turn(v, head, stopped, completed=True, adopt=False,
                            session_id="conv-A")
        resumed = fresh_cache()
        self.assertTrue(restore_session(v, resumed, lookup_session(v, head)))
        for t in tail:
            step(resumed, t)
        straight = drive(head + tail)
        self.assertTrue(state_bits_equal(resumed, straight),
                        "restore+continue must equal never having stopped")

    def test_T6_a_session_rung_is_refused_for_a_prefill_restore(self):
        v = self._vault()
        toks = list(range(1, 33))
        record_session_turn(v, toks, drive(toks), completed=True,
                                     session_id="conv-A")
        cp = lookup_session(v, toks)
        self.assertEqual(cp.tier, VaultTier.SESSION)
        self.assertFalse(v.restore_into(fresh_cache(), cp, tier=VaultTier.PREFILL),
                         "the cold-prefill guarantee must not be widened")
        self.assertEqual(v.stats.rejected_tier, 1)

    def test_T7_identity_change_invalidates_session_rungs_too(self):
        v = self._vault()
        toks = list(range(1, 33))
        record_session_turn(v, toks, drive(toks), completed=True,
                                     session_id="conv-A")
        cp = lookup_session(v, toks)
        other = ContextVault("a-different-build", budget_bytes=1 << 30)
        self.assertFalse(other.restore_into(fresh_cache(), cp, tier=VaultTier.SESSION))
        self.assertEqual(other.stats.rejected_foreign, 1)


class TestEviction(unittest.TestCase):
    def test_T8_deep_rungs_of_a_session_go_before_its_shallow_ones(self):
        v = ContextVault("deep-first", budget_bytes=1 << 30)
        toks = list(range(1, 65))
        sid = session_id_for(toks)
        for depth in (16, 32, 48):
            v.insert(toks, depth, capture_fragments(drive(toks[:depth]), depth),
                     tier=VaultTier.SESSION, session_id=sid)
        self.assertEqual(v.rungs, 3)
        v.budget = v.resident_bytes - 1
        v._evict_until(0)
        depths = sorted(n.checkpoint.prefix_len for n in v._iter_nodes()
                        if n.checkpoint is not None)
        self.assertNotIn(48, depths, "the deepest rung must go first")
        self.assertIn(16, depths, "shallow rungs must survive to keep the 8-12x")

    def test_T9_evicting_a_session_cannot_orphan_a_live_request(self):
        v = ContextVault("orphan", budget_bytes=1 << 30)
        toks = list(range(1, 33))
        record_session_turn(v, toks, drive(toks), completed=True, adopt=False,
                            session_id="conv-A")
        cp = lookup_session(v, toks)
        live = fresh_cache()
        restore_session(v, live, cp)
        # The mechanism, not just the symptom: restore expands into fresh
        # buffers (_expand_tree -> _copy_array), so the store and the request
        # never share memory and eviction cannot reach into a live cache.
        stored = cp.fragments[1].payload[0].payload["state"]
        self.assertIsNot(live[1].caches[0].keys, stored[0])
        before = [s.state[0] for s in live[1].caches]
        v.clear()
        step(live, 7)                       # keep using it after the store dropped it
        after = [s.state[0] for s in live[1].caches]
        for b, a in zip(before, after):
            self.assertTrue(same_bits(b, a[..., : b.shape[-2], :]),
                            "a live cache must not depend on the store")

    def test_T10_ttl_expiry_frees_bytes_and_is_observable(self):
        v = ContextVault("ttl", budget_bytes=1 << 30)
        toks = list(range(1, 33))
        record_session_turn(v, toks, drive(toks), completed=True, ttl_s=0.05,
                            session_id="conv-A")
        self.assertGreater(v.resident_bytes, 0)
        time.sleep(0.08)
        self.assertIsNone(lookup_session(v, toks), "an expired rung must not hit")
        self.assertEqual(v.resident_bytes, 0)
        self.assertEqual(v.stats.expired, 1)

    def test_T11_budget_is_never_exceeded_across_mixed_tiers(self):
        one = capture_fragments(drive(range(1, 17)), 16)
        from mlx_vlm.context_vault import fragments_nbytes
        v = ContextVault("mixed", budget_bytes=int(fragments_nbytes(one) * 2.5))
        for i in range(4):
            d = list(range(i * 1000, i * 1000 + 32))
            v.insert(d, 16, capture_fragments(drive(d[:16]), 16))
            record_session_turn(v, d, drive(d), completed=True,
                                session_id=f"conv-{i}")
        self.assertLessEqual(v.resident_bytes, v.budget)
        self.assertGreater(v.stats.evictions, 0)


class TestServerWiring(unittest.TestCase):
    def test_T12_session_store_is_a_no_op_when_the_vault_is_off(self):
        toks = list(range(1, 17))
        self.assertFalse(record_session_turn(None, toks, drive(toks), completed=True,
                                      session_id="conv-A"))
        self.assertIsNone(lookup_session(None, toks))
        self.assertFalse(restore_session(None, fresh_cache(), None))

    def test_T13_a_store_fault_never_fails_a_request(self):
        class Exploding:
            def insert(self, *a, **k):
                raise RuntimeError("boom")
        toks = list(range(1, 17))
        self.assertFalse(
            record_session_turn(Exploding(), toks, drive(toks), completed=True,
                                session_id="conv-A"))


if __name__ == "__main__":
    unittest.main()
