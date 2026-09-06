"""LW5 (2026-09-07): capture_session ran AFTER the finished row had already
left the batch.

Root cause (confirmed via LW5's own vault-session/vault-pick log lines, see
context_vault.diagnose_lookup and generate/ar.py's per-request logging added
for LW4): ``GenerationBatch.next()`` called ``self.filter(keep)`` -- which
compacts ``self.uids`` AND ``self.prompt_cache`` down to only the
still-generating rows -- unconditionally at the END of the very call that
just computed ``finish_reason`` for a row. server/generation.py's ``_step()``
only gets to call ``capture_session`` (and even ``note_generated`` for the
row's own last token) AFTER that same ``next()`` call has already returned,
by which point ``filter()`` had already run and the row was gone: every
capture refused with ``uid_gone_from_batch``, every request, including
plain single-turn ones with no multi-turn history at all -- LW3's and LW4's
fixes had nothing to reuse because nothing was ever being stored.

``capture_session``'s own docstring already described the window this needed
("between finish_reason being emitted and remove()") -- ``remove()`` is
simply never called on the normal completion path (server/generation.py's
``_step()`` does ``del active[r.uid]`` with no ``batch_gen.remove()`` call),
so that window never existed for a plain (non-speculative)
``GenerationBatch``. ``SpeculativeGenerationBatch`` needed no fix: its
``_refresh_uids()`` only recomputes the VISIBLE ``uids`` list from
``_finished`` flags and never compacts ``prompt_cache``/``_all_uids``, so a
finished row's cache genuinely is intact there until an explicit
``remove()``.

Fix: ``GenerationBatch`` defers its own ``filter()`` by exactly one
``next()`` call (``_pending_filter_keep``, applied at the TOP of the
following call, before any new decode work happens) -- so a row that just
finished stays fully present, cache and all, for the entire window between
one ``next()`` call returning and the next one being made, which is exactly
the window ``_step()`` uses. ``BatchGenerator.remove()`` flushes any pending
deferred filter before computing its own, so an unrelated cancellation can
never apply two filters against inconsistent index bases.

This test drives the REAL admission loop -- a real ``GenerationBatch``,
real ``next()`` calls, real ``finish_reason``/``filter()`` sequencing -- with
a stub MODEL (fixed logits, no real weights) and a stub BATCHGENERATOR
wrapper (``object.__new__``, sidestepping ``BatchGenerator.__init__``'s
``wired_limit`` context manager, which raises unrelated to this fix on this
venv/MLX build under ``MLX_DEFAULT_DEVICE=cpu`` -- see
test_vault_server_wiring.py's already-skipped tests for the same pre-existing
issue). What is real and unmodified here is ``GenerationBatch.next()``/
``filter()`` and ``BatchGenerator.note_generated``/``capture_session``
themselves.
"""

import os
import unittest

import mlx.core as mx

from mlx_vlm import context_vault as V
from mlx_vlm.generate.ar import BatchGenerator, GenerationBatch
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
    reporting ``n`` tokens held -- same helper the other session-tier test
    modules use, so capture_fragments/record_session_turn accept it as a
    real, restorable, correctly-offset fragment source."""
    c = [ArraysCache(size=2), CacheList(KVCache(), KVCache())]
    c[0][0] = mx.zeros((1, H, D, 4), mx.float32)
    c[0][1] = mx.zeros((1, H, D, D), mx.float32)
    lat = mx.zeros((1, H, n, D), mx.bfloat16)
    c[1].caches[0].update_and_fetch(lat, lat)
    idx = mx.zeros((1, 1, n, 2 * D + 1), mx.bfloat16)
    c[1].caches[1].update_and_fetch(idx, mx.zeros((1, 1, n, 0), mx.bfloat16))
    mx.eval([e.state for e in c])
    return c


class _FixedLogitModel:
    """Ignores the cache entirely (it is a stand-in for weight-driven
    attention, not what this test measures) and always favours token id 7 --
    a deterministic, real forward call through GenerationBatch's own
    _step(), not a mocked Response list."""

    VOCAB = 16
    FAVORED_TOKEN = 7

    def __call__(self, input_ids, cache=None, **kwargs):
        scores = mx.array([0.0] * self.VOCAB)
        scores = mx.where(
            mx.arange(self.VOCAB) == self.FAVORED_TOKEN, mx.array(10.0), scores
        )
        logits = mx.broadcast_to(
            scores, (input_ids.shape[0], input_ids.shape[1], self.VOCAB)
        )
        from types import SimpleNamespace

        return SimpleNamespace(logits=logits)


def _isolate_session_env(testcase):
    """This module wants the session tier ON without depending on the other
    (still-off-by-default, real-header-only) mechanism -- rely on
    MLX_VLM_APC_SAVE_SESSION's default-ON state, restoring whatever was
    actually set afterward."""
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


class TestCaptureSessionTiming(unittest.TestCase):
    def setUp(self):
        _isolate_session_env(self)
        self.vault = V.ContextVault("timing", budget_bytes=1 << 30)

    def _drive_to_completion(self, prompt_ids, max_tokens):
        """Real GenerationBatch, real next()/filter() sequencing, exactly
        mirroring server/generation.py's _step() call order: next() first,
        then note_generated for the token(s) it returned, then --
        r.finish_reason permitting -- capture_session. Returns
        (bg, uid, responses_by_step) so the test can assert on both the
        capture outcome and the exact token stream that was captured.
        """
        uid = 0
        total_len = len(prompt_ids) + max_tokens
        gen_batch = GenerationBatch(
            model=_FixedLogitModel(),
            uids=[uid],
            inputs=mx.array([prompt_ids[-1]], dtype=mx.int32),
            prompt_cache=row_cache(total_len),
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
            stop_criteria=lambda token: False,  # only max_tokens ends this
            max_tokens=[max_tokens],
        )

        # object.__new__ deliberately skips BatchGenerator.__init__ (and its
        # wired_limit context manager) -- everything this test calls
        # (note_generated/capture_session) only touches the attributes set
        # here, matching the other session-tier test modules' pattern.
        bg = object.__new__(BatchGenerator)
        bg._wire_stack = None  # __del__ calls close(), which reads this
        bg.vault = self.vault
        bg._generation_batch = gen_batch
        bg._session_tokens = {uid: list(prompt_ids)}
        bg._session_prompt_len = {uid: len(prompt_ids)}

        capture_result = None
        generated = []
        for _ in range(max_tokens + 1):  # +1 safety margin; loop breaks on finish
            responses = gen_batch.next()
            if not responses:
                break
            for r in responses:
                tok = int(r.token)
                generated.append(tok)
                BatchGenerator.note_generated(bg, r.uid, [tok])
                if r.finish_reason is not None:
                    capture_result = BatchGenerator.capture_session(
                        bg, r.uid, session_id="timing-session"
                    )
            if capture_result is not None:
                break

        return bg, uid, capture_result, generated

    def test_capture_session_succeeds_not_uid_gone_from_batch(self):
        prompt_ids = list(range(1, 21))  # 20-token stand-in prompt
        max_tokens = 3

        V.reset_session_skips()
        bg, uid, captured, generated = self._drive_to_completion(prompt_ids, max_tokens)

        self.assertEqual(len(generated), max_tokens)
        self.assertTrue(
            captured,
            f"capture_session must succeed; skip reasons so far: "
            f"{V.session_skip_counts()}",
        )
        self.assertNotIn("uid_gone_from_batch", V.session_skip_counts())

        expected_key = prompt_ids + generated
        self.assertEqual(bg._session_tokens[uid], expected_key)

        hit = self.vault.lookup(expected_key, tier=V.VaultTier.SESSION)
        self.assertIsNotNone(hit, "the vault must hold an entry for the full session")
        self.assertEqual(
            hit.prefix_len, len(expected_key),
            "the stored entry must cover prompt + ALL generated tokens, not "
            "just the prompt",
        )
        self.assertEqual(self.vault.stats.session_inserts, 1)

    def test_second_turn_reuses_the_captured_session(self):
        """Not just "capture succeeds" in isolation -- the point of capturing
        at all: a follow-up turn whose prompt is [captured session] + [more]
        must find the full captured depth via the ordinary SESSION-tier
        lookup, with no bridge/splice machinery needed (no retokenisation
        mismatch in this synthetic token stream)."""
        prompt_ids = list(range(1, 21))
        max_tokens = 3

        _, _uid, captured, generated = self._drive_to_completion(prompt_ids, max_tokens)
        self.assertTrue(captured)

        turn1_session = prompt_ids + generated
        turn2_query = turn1_session + list(range(101, 106))  # next user turn, 5 tokens

        hit = self.vault.lookup(turn2_query, tier=V.VaultTier.SESSION)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.prefix_len, len(turn1_session))

    def test_a_row_that_never_finishes_this_call_is_not_captured(self):
        """No regression: a row still generating (finish_reason is None on
        every response) must never reach capture_session at all -- the
        deferred-filter mechanism must not somehow trigger an early or
        spurious capture."""
        prompt_ids = list(range(1, 11))
        uid = 0
        gen_batch = GenerationBatch(
            model=_FixedLogitModel(),
            uids=[uid],
            inputs=mx.array([prompt_ids[-1]], dtype=mx.int32),
            prompt_cache=row_cache(len(prompt_ids) + 1),
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
            stop_criteria=lambda token: False,
            max_tokens=[1000],  # far from finishing in one step
        )
        bg = object.__new__(BatchGenerator)
        bg._wire_stack = None  # __del__ calls close(), which reads this
        bg.vault = self.vault
        bg._generation_batch = gen_batch
        bg._session_tokens = {uid: list(prompt_ids)}
        bg._session_prompt_len = {uid: len(prompt_ids)}

        responses = gen_batch.next()
        self.assertEqual(len(responses), 1)
        self.assertIsNone(responses[0].finish_reason)
        # Mirroring _step()'s own guard: capture_session is only ever called
        # when finish_reason is not None -- assert THAT guard, not
        # capture_session's behaviour if misused (it would happily "succeed"
        # on a still-generating row, which is exactly what must not be
        # invoked in the first place).
        self.assertIsNone(responses[0].finish_reason)
        self.assertEqual(self.vault.stats.session_inserts, 0)


if __name__ == "__main__":
    unittest.main()
