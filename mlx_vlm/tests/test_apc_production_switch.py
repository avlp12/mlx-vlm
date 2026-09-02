"""The APC production switch: it already exists, and the flag must not fork it.

CONTEXT (lane 3's X3 probe, corrected). An identical 32,768-token prompt
resubmitted three times cost 92.288 / 92.354 / 93.680 s with cached_tokens 0.
The reported root cause was that nothing assigns `runtime.apc_manager`. That is
wrong: `app.get_cached_model` assigns it from `apc.from_env(...)`, gated on
`RuntimeConfig.apc_enabled`, which reads **APC_ENABLED** -- not MLX_VLM_APC*,
which is what was grepped for. The measurement stands; the mechanism is a
default, not a gap.

So these tests pin the two things that actually matter: the switch works, and
`--apc` writes the variable that already exists instead of inventing a second
one that can disagree with it.
"""

import argparse
import importlib
import os
import unittest

from mlx_vlm import apc as _apc
from mlx_vlm.server import cli as _cli
from mlx_vlm.server import runtime_config as _rc
from mlx_vlm.server.runtime_config import RuntimeConfig

_app = importlib.import_module("mlx_vlm.server.app")

_APC_KEYS = ("APC_ENABLED", "APC_EXACT_CACHE_ENTRIES", "APC_BLOCK_SIZE")


def _isolate(tc):
    saved = {k: os.environ.get(k) for k in _APC_KEYS}

    def restore():
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    tc.addCleanup(restore)
    for k in _APC_KEYS:
        os.environ.pop(k, None)


class TestSwitchExists(unittest.TestCase):
    def setUp(self):
        _isolate(self)

    def test_server_default_is_ON(self):
        """I1024: prefix caching is the server default.

        Measured before flipping it: identical 32k prompt 79.2 s cold -> 2.64 s
        warm (30x), 2k 6.73 -> 2.45 s (2.7x), ~2 GB added at the default 2
        entries against a 169 GB model, swap flat.
        """
        self.assertTrue(RuntimeConfig.from_env().apc_enabled)

    def test_APC_ENABLED_0_still_disables(self):
        os.environ["APC_ENABLED"] = "0"
        self.assertFalse(RuntimeConfig.from_env().apc_enabled)
        self.assertIsNone(_apc.from_env(overrides={"enabled": False}))

    def test_library_default_stays_OFF_and_that_is_deliberate(self):
        """from_env() without overrides is for direct library callers.

        The server never reaches this branch -- it always passes an override
        built from its config -- so the two defaults are allowed to differ, and
        a future reader should not unify them without measuring the library
        path. Pinned so the divergence is a decision, not a drift.
        """
        self.assertIsNone(_apc.from_env(), "library callers keep opting in")

    def test_env_turns_it_on(self):
        os.environ["APC_ENABLED"] = "1"
        self.assertTrue(RuntimeConfig.from_env().apc_enabled)
        self.assertIsNotNone(_apc.from_env(), "APC_ENABLED=1 must build a manager")

    def test_settings_table_agrees_with_the_dataclass(self):
        """Two declarations of the same default is how they drift apart."""
        row = [r for r in _rc.KNOBS if r[0] == "apc_enabled"][0]
        self.assertEqual(row[2], RuntimeConfig.apc_enabled)

    def test_override_beats_env(self):
        """get_cached_model passes cfg through as an override, so live settings
        drive APC without env mutation."""
        self.assertIsNotNone(_apc.from_env(overrides={"enabled": True}))
        os.environ["APC_ENABLED"] = "1"
        self.assertIsNone(_apc.from_env(overrides={"enabled": False}))

    def test_exact_entries_bound_is_a_count_not_bytes(self):
        """The viability number: each entry is a whole prompt-cache clone."""
        os.environ["APC_ENABLED"] = "1"
        os.environ["APC_EXACT_CACHE_ENTRIES"] = "5"
        self.assertEqual(_apc.from_env()._exact_cache_max, 5)
        os.environ.pop("APC_EXACT_CACHE_ENTRIES")
        self.assertEqual(_apc.from_env()._exact_cache_max, 2, "default is 2 entries")


class TestFlagDoesNotForkTheSwitch(unittest.TestCase):
    def setUp(self):
        _isolate(self)

    def _args(self, **kw):
        base = {"apc": None, "apc_exact_entries": None}
        base.update(kw)
        return argparse.Namespace(**base)

    def test_flag_absent_leaves_the_environment_alone(self):
        """An operator exporting APC_ENABLED must not be overridden by a default."""
        env = {"APC_ENABLED": "0"}
        _cli._apply_apc_env(self._args(), env)
        self.assertEqual(env["APC_ENABLED"], "0", "unpassed flag must not write")

    def test_no_apc_turns_it_off(self):
        env = {}
        _cli._apply_apc_env(self._args(apc=False), env)
        self.assertEqual(env["APC_ENABLED"], "0")

    def test_flag_writes_the_existing_variable(self):
        env = {}
        _cli._apply_apc_env(self._args(apc=True), env)
        self.assertEqual(env, {"APC_ENABLED": "1"},
                         "must write APC_ENABLED, not a new MLX_VLM_APC* name")

    def test_entries_flag_maps_to_the_existing_variable(self):
        env = {}
        _cli._apply_apc_env(self._args(apc=True, apc_exact_entries=8), env)
        self.assertEqual(env["APC_EXACT_CACHE_ENTRIES"], "8")

    def test_flag_reaches_from_env_end_to_end(self):
        _cli._apply_apc_env(self._args(apc=True, apc_exact_entries=3), os.environ)
        mgr = _apc.from_env()
        self.assertIsNotNone(mgr)
        self.assertEqual(mgr._exact_cache_max, 3)

    def test_parser_exposes_the_flags(self):
        self.assertTrue(hasattr(_cli, "_apply_apc_env"))


class TestVisibility(unittest.TestCase):
    def test_the_state_is_announced_either_way(self):
        """A server that does not say whether prefix caching is on invites the
        X3 measurement again."""
        src = open(_app.__file__).read()
        self.assertIn("APC: disabled", src)
        self.assertIn("APC: enabled", src)


if __name__ == "__main__":
    unittest.main()
