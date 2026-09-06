"""End-to-end proof of the LW3 multiturn fix (ledger 2026-09-07): a plain
multi-turn conversation with NO explicit session id (no X-Session-Id header,
no Responses previous_response_id chain -- the shape a stock OpenAI
chat.completions client sends) still gets its full turn-1 prompt+generated
span reused as a literal token prefix on turn 2, via the vault's SESSION
tier, keyed by an id ``context_vault.auto_session_id`` derives by default
(``MLX_VLM_APC_SAVE_SESSION``, default ON).

Fake store, no model, CPU-only: mirrors test_session_capture_wiring.py's
``_Gen``/``row_cache`` (what ``capture_session``/``note_generated`` touch) and
test_session_restore.py's ``_Gen``/``fresh`` (what ``_vault_pick_for``
touches) in one merged double, and exercises both call chains for real
against a real ``ContextVault``.

Deliberately does NOT set MLX_VLM_GLM5_VAULT_SESSION anywhere in this module:
proving the default-ON path works WITHOUT that (still off-by-default,
unmodified) flag is the entire point of this fix.
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


def row_cache(n):
    """A cache shaped like GLM-5-Next's (KDA ArraysCache + DSA CacheList),
    holding ``n`` tokens -- same shape test_session_capture_wiring.py's
    ``row_cache``/test_session_restore.py's ``filled`` use."""
    c = [ArraysCache(size=2), CacheList(KVCache(), KVCache())]
    c[0][0] = mx.zeros((1, H, D, 4), mx.float32)
    c[0][1] = mx.zeros((1, H, D, D), mx.float32)
    lat = mx.zeros((1, H, n, D), mx.bfloat16)
    c[1].caches[0].update_and_fetch(lat, lat)
    idx = mx.zeros((1, 1, n, 2 * D + 1), mx.bfloat16)
    c[1].caches[1].update_and_fetch(idx, mx.zeros((1, 1, n, 0), mx.bfloat16))
    mx.eval([e.state for e in c])
    return c


def fresh():
    return [ArraysCache(size=2), CacheList(KVCache(), KVCache())]


class _Batch:
    def __init__(self, uids, cache):
        self.uids = list(uids)
        self.prompt_cache = cache


class _Gen:
    """Merges what capture_session/note_generated touch (session_capture_
    wiring's _Gen) with what _vault_pick_for touches (session_restore's
    _Gen) -- this test drives BOTH the write side (end of turn 1) and the
    read side (start of turn 2) against one real ContextVault."""

    def __init__(self, vault, uid, prompt_cache):
        self.vault = vault
        self._generation_batch = _Batch([uid], prompt_cache)
        self._session_tokens = {}
        self.model = object()
        self.apc_manager = None

    def _vault_prefix_trim_is_safe(self):
        return True

    def _apc_extra_hash(self, kw):
        return 0


def pick_for(gen, ids):
    from unittest import mock

    with mock.patch(
        "mlx_vlm.generate.ar.cache.make_prompt_cache", side_effect=lambda m: fresh()
    ):
        return BatchGenerator._vault_pick_for(gen, ids, {}, None)


def _isolate_env(testcase):
    """Restore MLX_VLM_GLM5_VAULT_SESSION[_DERIVED_ID]/MLX_VLM_APC_SAVE_SESSION
    after the test. Unlike the sibling test modules' helper, this one does
    NOT force MLX_VLM_APC_SAVE_SESSION to "0" -- the whole point of this
    module is to prove the default (unset) state already works."""
    keys = (V._ENV_SESSION, V._ENV_SESSION_DERIVED_ID, V._ENV_APC_SAVE_SESSION)
    saved = {k: os.environ.get(k) for k in keys}

    def restore():
        for k, val in saved.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val

    testcase.addCleanup(restore)
    for k in keys:
        os.environ.pop(k, None)


class TestTwoTurnSessionRoundTrip(unittest.TestCase):
    def setUp(self):
        _isolate_env(self)
        self.v = V.ContextVault("e2e", budget_bytes=1 << 30)

    def _run_turn1(self, generated_tokens):
        """Simulate one completed turn 1: insert()'s session-key seed,
        note_generated()'s per-token accumulation, then capture_session() at
        finish_reason -- exactly the sequence generate/ar.py and
        server/generation.py run, at the points that matter for this test.
        """
        prompt = list(range(1, 21))  # turn-1 prompt, ending in a '<think>' stand-in
        uid = "u1"
        cache = row_cache(len(prompt) + len(generated_tokens))
        gen = _Gen(self.v, uid, cache)

        # insert(): seeding the session key is gated the same way
        # note_generated/capture_session are (session_tier_active()) --
        # replicate that gate here rather than special-casing the seed.
        self.assertTrue(
            V.session_tier_active(),
            "this whole test is pointless if the tier is not active by default",
        )
        gen._session_tokens[uid] = list(prompt)

        for tok in generated_tokens:
            BatchGenerator.note_generated(gen, uid, [tok])
        self.assertEqual(gen._session_tokens[uid], prompt + generated_tokens)

        # No explicit header/session_id anywhere in this test: this is
        # exactly what auto_session_id derives for a request that supplied
        # none (server/generation.py's admission loop does this same call).
        auto_id = V.auto_session_id(prompt)
        self.assertIsNotNone(auto_id, "auto id must be derivable by default")
        self.assertTrue(auto_id.startswith("auto:"))

        stored = BatchGenerator.capture_session(gen, uid, session_id=auto_id)
        self.assertTrue(stored, "capture_session must succeed with no env vars set")
        self.assertEqual(self.v.stats.session_inserts, 1)
        return prompt, prompt + generated_tokens

    def test_completed_turn_is_reused_on_turn_two(self):
        """A turn that closed its thinking span normally and produced
        visible content: generated = [reasoning...] + [close-marker] +
        [content...]. Session capture doesn't parse text -- these are just
        distinguishable token values -- but the SHAPE matches a real
        completed response.
        """
        reasoning = list(range(101, 111))  # 10 "reasoning" tokens
        close_marker = [154842]  # stand-in for '</think>'
        content = list(range(201, 206))  # 5 "content" tokens
        generated = reasoning + close_marker + content

        turn1_prompt, turn1_session = self._run_turn1(generated)

        # Turn 2's re-rendered prompt is turn 1's full session span PLUS
        # whatever the template appends after it (a synthesized '</think>'
        # if the fix needed one, '<|user|>', the next user turn, the next
        # '<think>' priming token, ...) -- never a prefix of it.
        turn2_suffix = [999, 998, 997, 996]
        turn2_ids = turn1_session + turn2_suffix

        plan = pick_for(_Gen(self.v, "u2", fresh()), turn2_ids)
        self.assertIsNotNone(plan, "turn 2 must hit the session rung")
        self.assertEqual(
            plan["prefix_len"],
            len(turn1_session),
            "cached prefix must equal len(turn1 prompt + generated) exactly, "
            "not merely len(turn1 prompt)",
        )
        self.assertEqual(plan["source"], "vault-session")
        self.assertEqual(self.v.stats.session_hits, 1)

    def test_truncated_thinking_turn_is_also_reused_on_turn_two(self):
        """A turn that hit max_tokens INSIDE the thinking span: no close
        marker, no content -- generated is reasoning tokens only. This is
        the code_ratelimiter_spec shape from the LW3 panel (reasoning_len
        nonzero, content_len == 0). capture_session's own contract already
        passes completed=True unconditionally regardless of finish_reason
        (see generate/ar.py capture_session: it fires whenever
        finish_reason is not None, "length" included) -- this test pins that
        a short, unclosed-thinking turn is captured and reused exactly like
        a completed one, not silently skipped for looking incomplete.
        """
        reasoning_only = list(range(301, 309))  # 8 tokens, never closes
        turn1_prompt, turn1_session = self._run_turn1(reasoning_only)

        turn2_suffix = [154842, 111, 222]  # synthesized close + next turn
        turn2_ids = turn1_session + turn2_suffix

        plan = pick_for(_Gen(self.v, "u2", fresh()), turn2_ids)
        self.assertIsNotNone(plan, "a truncated-thinking turn must still be reused")
        self.assertEqual(plan["prefix_len"], len(turn1_session))
        self.assertEqual(plan["source"], "vault-session")

    def test_single_turn_is_unaffected_when_no_session_rung_exists(self):
        """No regression on single-turn behaviour: a fresh conversation with
        nothing captured for it yet must miss cleanly (None), exactly as
        before this fix -- default-ON session tier participation must not
        conjure a hit, or error, out of an empty vault.
        """
        ids = list(range(1, 33))
        plan = pick_for(_Gen(self.v, "solo", fresh()), ids)
        self.assertIsNone(plan)
        self.assertEqual(self.v.stats.session_hits, 0)
        self.assertEqual(self.v.rungs, 0)

    def test_single_turn_with_an_unrelated_prefill_rung_is_unaffected(self):
        """No regression: an existing PREFILL-tier rung (the boundary-ladder
        mechanism this fix does not touch) is still served exactly as
        before -- as "vault", not "vault-session" -- when there is no
        session rung for this prompt.
        """
        ids = list(range(1, 65))
        self.v.insert(
            ids, 16, V.capture_fragments(row_cache(16), 16), tier=V.VaultTier.PREFILL
        )
        plan = pick_for(_Gen(self.v, "solo", fresh()), ids)
        self.assertIsNotNone(plan)
        self.assertEqual(plan["prefix_len"], 16)
        self.assertEqual(plan["source"], "vault")


if __name__ == "__main__":
    unittest.main()
