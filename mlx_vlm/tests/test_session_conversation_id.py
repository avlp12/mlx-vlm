"""Conversation identity for the vault's session tier (coordinator ruling I981).

Chat Completions captures only with an explicit X-Session-Id header; the
Responses API mints an id at the root of the previous_response_id chain and
resolves follow-ups through an O(1) map written at response time. Deriving an id
from the prompt, or keying on tenant + prompt prefix, is rejected: both merge or
fragment eviction groups silently.
"""

import unittest

import importlib

# NB: `from mlx_vlm.server import app` binds the FastAPI *instance* exported by
# the package, not this module. Import it explicitly.
A = importlib.import_module("mlx_vlm.server.app")


class _Req:
    def __init__(self, **h):
        self.headers = {k.lower().replace("_", "-"): v for k, v in h.items()}


class TestValidation(unittest.TestCase):
    def test_rejects_non_strings_and_blanks(self):
        for bad in (None, 123, "", "   ", b"x"):
            self.assertIsNone(A._valid_session_id(bad))

    def test_rejects_overlong(self):
        self.assertIsNone(A._valid_session_id("x" * 129))
        self.assertEqual(A._valid_session_id("x" * 128), "x" * 128)

    def test_rejects_whitespace_and_control_characters(self):
        """It becomes a log field and a dict key; newlines are log injection."""
        for bad in ("a b", "a\nb", "a\tb", "a\rb", "a\x00b"):
            self.assertIsNone(A._valid_session_id(bad))

    def test_trims_and_accepts_opaque_ids(self):
        self.assertEqual(A._valid_session_id("  conv-9f2a  "), "conv-9f2a")


class TestChatCompletionsHeader(unittest.TestCase):
    def test_header_is_read(self):
        self.assertEqual(A._read_session_id(_Req(X_Session_Id="conv-1")), "conv-1")

    def test_no_header_means_no_capture(self):
        self.assertIsNone(A._read_session_id(_Req()))
        self.assertIsNone(A._read_session_id(None))

    def test_invalid_header_is_dropped_not_sanitised(self):
        self.assertIsNone(A._read_session_id(_Req(X_Session_Id="a b")))


class TestResponsesApiRoot(unittest.TestCase):
    def setUp(self):
        A._CONVERSATION_ROOTS.clear()

    def test_a_request_with_no_previous_id_is_a_new_root(self):
        a = A.resolve_conversation_id(None)
        b = A.resolve_conversation_id(None)
        self.assertTrue(a.startswith("conv-"))
        self.assertNotEqual(a, b, "each new root must be distinct")

    def test_follow_ups_resolve_to_the_same_root(self):
        root = A.resolve_conversation_id(None)
        A.note_response_root("resp-1", root)
        A.note_response_root("resp-2", root)
        self.assertEqual(A.resolve_conversation_id("resp-1"), root)
        self.assertEqual(A.resolve_conversation_id("resp-2"), root,
                         "a chain resolves in one lookup, not a walk")

    def test_unknown_previous_id_starts_a_new_root(self):
        """Server restart or an evicted map: one cold prefill beats silently
        merging two conversations."""
        got = A.resolve_conversation_id("resp-never-seen")
        self.assertTrue(got.startswith("conv-"))
        self.assertNotIn("resp-never-seen", A._CONVERSATION_ROOTS)

    def test_explicit_header_wins_for_a_new_root(self):
        self.assertEqual(
            A.resolve_conversation_id(None, _Req(X_Session_Id="mine")), "mine")

    def test_header_does_not_override_an_established_chain(self):
        root = A.resolve_conversation_id(None)
        A.note_response_root("resp-1", root)
        self.assertEqual(
            A.resolve_conversation_id("resp-1", _Req(X_Session_Id="other")), root)

    def test_note_response_root_ignores_invalid_ids(self):
        A.note_response_root("bad id", "root")
        A.note_response_root("resp", "bad root")
        self.assertEqual(A._CONVERSATION_ROOTS, {})


if __name__ == "__main__":
    unittest.main()
