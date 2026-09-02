"""End-to-end session-capture plumbing: request -> args -> active -> _step.

Fake generator, no model, no server. What is pinned is the chain that was
missing when the live checks were blocked: a conversation id reaching
GenerationArguments, surviving into active[uid], and producing exactly one
capture_session call at finish_reason -- and none without an id.
"""

import argparse
import importlib
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
            self.assertEqual(_app._vault_stats_snapshot(), {"enabled": False})
        finally:
            _app.runtime.response_generator = old

    def test_absent_vault_reports_disabled(self):
        old = getattr(_app.runtime, "response_generator", None)
        try:
            _app.runtime.response_generator = types.SimpleNamespace(vault=None)
            self.assertEqual(_app._vault_stats_snapshot(), {"enabled": False})
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
            self.assertEqual(_app._vault_stats_snapshot(),
                             {"enabled": True, "stats_error": True})
        finally:
            _app.runtime.response_generator = old
