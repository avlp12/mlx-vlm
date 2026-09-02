"""End-to-end session-capture plumbing: request -> args -> active -> _step.

Fake generator, no model, no server. What is pinned is the chain that was
missing when the live checks were blocked: a conversation id reaching
GenerationArguments, surviving into active[uid], and producing exactly one
capture_session call at finish_reason -- and none without an id.
"""

import argparse
import importlib
import os
import types
import unittest

from mlx_vlm.server import request_normalization as _rn
from mlx_vlm.server.generation import ResponseGenerator

_app = importlib.import_module("mlx_vlm.server.app")


class _Req:
    """Minimal stand-in for a compatible API request."""
    max_tokens = 8
    temperature = 0.0
    top_p = 1.0
    stream = False
    model = "m"


class _Http:
    def __init__(self, **h):
        self.headers = {k.lower().replace("_", "-"): v for k, v in h.items()}


class TestArgsCarrySessionId(unittest.TestCase):
    def test_build_gen_args_carries_it(self):
        a = _rn._build_gen_args(_Req(), session_id="conv-7")
        self.assertEqual(a.session_id, "conv-7")

    def test_absent_means_do_not_capture(self):
        self.assertIsNone(_rn._build_gen_args(_Req()).session_id)

    def test_app_wrapper_passes_it_through(self):
        a = _app._build_gen_args(_Req(), session_id="conv-9")
        self.assertEqual(a.session_id, "conv-9")


class TestResolutionRules(unittest.TestCase):
    def setUp(self):
        _app._CONVERSATION_ROOTS.clear()

    def test_chat_completions_needs_the_header(self):
        self.assertIsNone(_app._read_session_id(_Http()))
        self.assertEqual(_app._read_session_id(_Http(X_Session_Id="s1")), "s1")

    def test_responses_chain_resolves_to_one_root(self):
        root = _app.resolve_conversation_id(None)
        _app.note_response_root("resp_a", root)
        self.assertEqual(_app.resolve_conversation_id("resp_a"), root)


class _Resp:
    def __init__(self, uid, token, finish_reason=None):
        self.uid, self.token, self.finish_reason = uid, token, finish_reason
        self.token_logprob = 0.0
        self.top_logprobs = None


class _BatchGen:
    def __init__(self, batches):
        self._batches = list(batches)
        self.captured = []
        self.noted = []

    def next(self, **kw):
        return [], self._batches.pop(0) if self._batches else []

    def note_generated(self, uid, toks):
        self.noted.append((uid, list(toks)))

    def capture_session(self, uid, tokens=None, *, session_id, ttl_s=None):
        self.captured.append((uid, session_id))
        return True


def _stub_self():
    """Only what _step touches on the decode path."""
    s = types.SimpleNamespace()
    s._log_prefill_progress = lambda *a, **k: None
    s._log_prefill_completed = lambda *a, **k: None
    s._stream_text = lambda info, tok, fr: "x"
    s._log_decode_progress = staticmethod(lambda *a, **k: 0.0)
    s.draft_model = None
    s.draft_kind = None
    return s


def _active(session_id):
    return {"u1": {"rqueue": types.SimpleNamespace(put=lambda *_: None),
                   "streamer": types.SimpleNamespace(finalize=lambda: ""),
                   "prompt_tps": None, "cached_tokens": 0,
                   "spec_snapshot": None, "session_id": session_id,
                   "request_id": "r1"}}


class TestStepCapture(unittest.TestCase):
    def _run(self, session_id, batches):
        bg = _BatchGen(batches)
        act = _active(session_id)
        for _ in batches:
            ResponseGenerator._step(_stub_self(), bg, act)
        return bg

    def test_capture_fires_once_at_finish(self):
        bg = self._run("conv-1", [[_Resp("u1", 10)], [_Resp("u1", 11)],
                                  [_Resp("u1", 12, finish_reason="stop")]])
        self.assertEqual(bg.captured, [("u1", "conv-1")],
                         "exactly one capture, on the finishing response")
        self.assertEqual(bg.noted, [("u1", [10]), ("u1", [11]), ("u1", [12])],
                         "one token appended per response, in order")

    def test_no_capture_without_a_session_id(self):
        bg = self._run(None, [[_Resp("u1", 10, finish_reason="stop")]])
        self.assertEqual(bg.captured, [], "no conversation id -> no capture")

    def test_no_capture_while_the_turn_is_still_running(self):
        bg = self._run("conv-1", [[_Resp("u1", 10)], [_Resp("u1", 11)]])
        self.assertEqual(bg.captured, [])

    def test_length_stop_still_captures(self):
        bg = self._run("conv-1", [[_Resp("u1", 9, finish_reason="length")]])
        self.assertEqual(len(bg.captured), 1,
                         "a length-capped turn is complete and reusable")


if __name__ == "__main__":
    unittest.main()


class TestVaultObservability(unittest.TestCase):
    """The seven live checks are observed through this surface.

    Four of them (capture fired, eviction order, resident bytes, session hits)
    have no other external witness: the vault lives inside the response
    generator. Exposing it is the same lesson as the APC default -- a cache
    nobody can observe is a cache nobody can tell is off.
    """

    def test_absent_generator_reports_disabled(self):
        old = getattr(_app.runtime, "response_generator", None)
        try:
            _app.runtime.response_generator = None
            snap = _app._vault_stats_snapshot()
            self.assertFalse(snap["enabled"])
            self.assertIn("session_skips", snap)
        finally:
            _app.runtime.response_generator = old

    def test_absent_vault_reports_disabled(self):
        old = getattr(_app.runtime, "response_generator", None)
        try:
            _app.runtime.response_generator = types.SimpleNamespace(vault=None)
            snap = _app._vault_stats_snapshot()
            self.assertFalse(snap["enabled"])
            self.assertIn("session_skips", snap)
        finally:
            _app.runtime.response_generator = old

    def test_counters_the_live_checks_need_are_present(self):
        from mlx_vlm.context_vault import ContextVault
        old = getattr(_app.runtime, "response_generator", None)
        try:
            v = ContextVault("obs", budget_bytes=1 << 20)
            _app.runtime.response_generator = types.SimpleNamespace(vault=v)
            snap = _app._vault_stats_snapshot()
            self.assertTrue(snap["enabled"])
            for k in ("session_inserts", "session_hits", "evictions",
                      "bytes_resident", "rungs_resident"):
                self.assertIn(k, snap, f"{k} is what a live check reads")
        finally:
            _app.runtime.response_generator = old

    def test_a_broken_vault_cannot_break_the_endpoint(self):
        class Boom:
            def stats_dict(self):
                raise RuntimeError("boom")
        old = getattr(_app.runtime, "response_generator", None)
        try:
            _app.runtime.response_generator = types.SimpleNamespace(vault=Boom())
            snap = _app._vault_stats_snapshot()
            self.assertTrue(snap["enabled"])
            self.assertTrue(snap["stats_error"])
        finally:
            _app.runtime.response_generator = old


class TestSkipReasonsAreNamed(unittest.TestCase):
    """The feature was inert live and said nothing. Now it says which gate.

    On unified ff9a3045 the seven live checks found session_inserts stuck at 0
    with an empty server log, because capture_session had five early returns and
    none of them spoke. A disabled feature and a broken one produced the same
    picture, which is the failure the APC default had -- rebuilt, by me, in my
    own code, while fixing theirs.
    """

    def setUp(self):
        from mlx_vlm import context_vault as V
        self.V = V
        V.reset_session_skips()
        saved = {k: os.environ.get(k) for k in (V._ENV_SESSION,)}

        def restore():
            for k, val in saved.items():
                if val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = val
        self.addCleanup(restore)
        os.environ.pop(V._ENV_SESSION, None)

    def test_no_vault_is_named(self):
        self.V.record_session_turn(None, [1, 2], [], completed=True, session_id="s")
        self.assertEqual(self.V.session_skip_counts().get("no_vault"), 1)

    def test_cancel_is_named_rather_than_silent(self):
        self.V.record_session_turn(object(), [1, 2], [], completed=False,
                                   session_id="s")
        self.assertEqual(self.V.session_skip_counts().get("not_completed"), 1)

    def test_missing_session_id_is_named(self):
        v = self.V.ContextVault("skips", budget_bytes=1 << 20)
        self.V.record_session_turn(v, [1, 2], [], completed=True, session_id="")
        self.assertEqual(self.V.session_skip_counts().get("no_session_id"), 1)

    def test_flag_off_is_named_at_the_generator(self):
        from mlx_vlm.generate.ar import BatchGenerator
        g = types.SimpleNamespace(vault=object(), _generation_batch=None,
                                  _session_tokens={})
        BatchGenerator.capture_session(g, "u1", session_id="s")
        self.assertEqual(self.V.session_skip_counts().get("flag_off"), 1)

    def test_uid_gone_is_distinguishable_from_flag_off(self):
        from mlx_vlm.generate.ar import BatchGenerator
        os.environ[self.V._ENV_SESSION] = "1"
        g = types.SimpleNamespace(
            vault=object(), _session_tokens={},
            _generation_batch=types.SimpleNamespace(uids=["other"]))
        BatchGenerator.capture_session(g, "u1", session_id="s")
        counts = self.V.session_skip_counts()
        self.assertEqual(counts.get("uid_gone_from_batch"), 1)
        self.assertIsNone(counts.get("flag_off"),
                          "the two must not collapse into one reason")

    def test_every_reason_is_a_distinct_string(self):
        """A counter that reuses a name cannot separate two causes."""
        import inspect
        from mlx_vlm.generate import ar
        src = inspect.getsource(ar.BatchGenerator.capture_session)
        names = [ln.split('record_session_skip("')[1].split('"')[0]
                 for ln in src.splitlines() if 'record_session_skip("' in ln]
        self.assertEqual(len(names), len(set(names)), f"duplicate reasons: {names}")
        self.assertGreaterEqual(len(names), 6)

    def test_the_skips_reach_the_stats_endpoint(self):
        self.V.record_session_turn(None, [1], [], completed=True, session_id="s")
        self.assertIn("no_vault",
                      _app._vault_stats_snapshot().get("session_skips", {}))


class TestStepDrivesTheRealCaptureSession(unittest.TestCase):
    """_step -> the REAL BatchGenerator.capture_session, against a real vault.

    The gap that let ff9a3045 ship inert: TestStepCapture fakes capture_session
    entirely, so it proves _step CALLS something and nothing about whether that
    something stores a rung. This drives the real method with real state and
    asserts the vault actually gains a rung.
    """

    def setUp(self):
        import mlx.core as mx
        from mlx_vlm import context_vault as V
        from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache
        self.V, self.mx = V, mx
        self._prev = mx.default_device()
        mx.set_default_device(mx.cpu)
        self.addCleanup(lambda: mx.set_default_device(self._prev))
        V.reset_session_skips()
        saved = os.environ.get(V._ENV_SESSION)

        def restore():
            if saved is None:
                os.environ.pop(V._ENV_SESSION, None)
            else:
                os.environ[V._ENV_SESSION] = saved
        self.addCleanup(restore)
        os.environ[V._ENV_SESSION] = "1"

        H, D, N = 2, 8, 6
        c = [ArraysCache(size=2), CacheList(KVCache(), KVCache())]
        c[0][0] = mx.zeros((1, H, D, 4), mx.float32)
        c[0][1] = mx.zeros((1, H, D, D), mx.float32)
        lat = mx.zeros((1, H, N, D), mx.bfloat16)
        c[1].caches[0].update_and_fetch(lat, lat)
        idx = mx.zeros((1, 1, N, 2 * D + 1), mx.bfloat16)
        c[1].caches[1].update_and_fetch(idx, mx.zeros((1, 1, N, 0), mx.bfloat16))
        mx.eval([e.state for e in c])
        self.cache = c
        self.vault = V.ContextVault("real-step", budget_bytes=1 << 30)

    def _generator(self):
        from mlx_vlm.generate.ar import BatchGenerator
        gb = types.SimpleNamespace(uids=["u1"], prompt_cache=self.cache)
        g = types.SimpleNamespace(vault=self.vault, _generation_batch=gb,
                                  _session_tokens={"u1": [1, 2, 3, 4, 5]})
        g.capture_session = lambda uid, tokens=None, *, session_id, ttl_s=None: \
            BatchGenerator.capture_session(g, uid, tokens,
                                           session_id=session_id, ttl_s=ttl_s)
        g.note_generated = lambda uid, toks: \
            BatchGenerator.note_generated(g, uid, toks)
        return g

    def test_a_finished_turn_actually_stores_a_rung(self):
        g = self._generator()
        bg = _BatchGen([[_Resp("u1", 9, finish_reason="stop")]])
        bg.capture_session = g.capture_session
        bg.note_generated = g.note_generated
        ResponseGenerator._step(_stub_self(), bg, _active("conv-A"))
        self.assertEqual(
            self.vault.stats.session_inserts, 1,
            f"no rung stored; skips={self.V.session_skip_counts()}")
        cp = self.V.lookup_session(self.vault, [1, 2, 3, 4, 5, 9])
        self.assertIsNotNone(cp, "the stored rung must be reachable by its key")
        self.assertEqual(cp.prefix_len, 6, "prefix_len must be the cache offset")

    def test_no_session_id_names_itself_rather_than_storing_nothing_silently(self):
        g = self._generator()
        bg = _BatchGen([[_Resp("u1", 9, finish_reason="stop")]])
        bg.capture_session = g.capture_session
        bg.note_generated = g.note_generated
        ResponseGenerator._step(_stub_self(), bg, _active(None))
        self.assertEqual(self.vault.stats.session_inserts, 0)
        self.assertEqual(
            self.V.session_skip_counts().get("no_session_id_on_request"), 1)


class TestFinishedRowIsStillFindable(unittest.TestCase):
    """The bug the live diagnostic named: uid_gone_from_batch.

    SpeculativeGenerationBatch._refresh_uids rebuilds .uids from ._finished, so
    a uid leaves .uids at the instant finish_reason is set -- which is precisely
    when capture_session runs. Looking the row up in .uids therefore always
    missed, and the feature stored nothing while reporting nothing.

    My earlier fake gave the batch a permanent uids=["u1"], which is why every
    unit test passed against an inert feature.
    """

    def setUp(self):
        from mlx_vlm import context_vault as V
        self.V = V
        V.reset_session_skips()
        saved = os.environ.get(V._ENV_SESSION)

        def restore():
            if saved is None:
                os.environ.pop(V._ENV_SESSION, None)
            else:
                os.environ[V._ENV_SESSION] = saved
        self.addCleanup(restore)
        os.environ[V._ENV_SESSION] = "1"

    def _batch(self, uids, all_uids):
        """A batch shaped like the real one after a row finished."""
        ns = types.SimpleNamespace(uids=list(uids), prompt_cache=[])
        if all_uids is not None:
            ns._all_uids = list(all_uids)
        return ns

    def test_a_finished_uid_is_still_found_via_all_uids(self):
        from mlx_vlm.generate.ar import BatchGenerator
        g = types.SimpleNamespace(
            vault=object(), _session_tokens={"u1": [1, 2]},
            _generation_batch=self._batch(uids=[], all_uids=["u1"]))
        BatchGenerator.capture_session(g, "u1", session_id="s")
        counts = self.V.session_skip_counts()
        self.assertIsNone(counts.get("uid_gone_from_batch"),
                          "a finished row must still be locatable")
        self.assertEqual(counts.get("row_cache_unavailable"), 1,
                         "it should get past the uid gate and stop on the cache")

    def test_a_genuinely_absent_uid_still_reports_gone(self):
        from mlx_vlm.generate.ar import BatchGenerator
        g = types.SimpleNamespace(
            vault=object(), _session_tokens={"u1": [1, 2]},
            _generation_batch=self._batch(uids=[], all_uids=["other"]))
        BatchGenerator.capture_session(g, "u1", session_id="s")
        self.assertEqual(self.V.session_skip_counts().get("uid_gone_from_batch"), 1)

    def test_plain_batch_without_all_uids_still_works(self):
        """GenerationBatch has no _all_uids and never prunes on finish."""
        from mlx_vlm.generate.ar import BatchGenerator
        g = types.SimpleNamespace(
            vault=object(), _session_tokens={"u1": [1, 2]},
            _generation_batch=self._batch(uids=["u1"], all_uids=None))
        BatchGenerator.capture_session(g, "u1", session_id="s")
        self.assertIsNone(self.V.session_skip_counts().get("uid_gone_from_batch"))
