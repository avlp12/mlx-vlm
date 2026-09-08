"""R-cc -- ``MLX_VLM_GLM5_PREFILL_KEEP_CACHE``: keep the allocator pool across chunks.

Both prefill chunk loops end each chunk with ``mx.clear_cache()``, which returns the
allocator's free pool to the OS; the next chunk then re-allocates ~17 GB of per-layer
transients and pays the first-touch faults inside the timed forward.  The flag skips
that call.  It is an allocator hint: no array, shape or kernel changes, so the arm is
bit-identical to the default by construction and the only thing worth testing is that
the DEFAULT still clears, the ARM does not, and every OTHER clear_cache in the file is
untouched.
"""
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from mlx_vlm.generate import PromptProcessingBatch
from mlx_vlm.generate import ar as ar_module
from mlx_vlm.generate.common import prefill_keep_cache_enabled

KEY = "MLX_VLM_GLM5_PREFILL_KEEP_CACHE"


@pytest.fixture(autouse=True)
def _clean_env():
    saved = os.environ.get(KEY)
    os.environ.pop(KEY, None)
    yield
    if saved is None:
        os.environ.pop(KEY, None)
    else:
        os.environ[KEY] = saved


class TestFlag:
    @pytest.mark.parametrize("raw,want", [
        (None, False), ("", False), ("0", False), ("off", False), ("no", False),
        ("1", True), ("true", True), ("yes", True), ("on", True), (" On ", True),
    ])
    def test_parsing_defaults_off(self, raw, want):
        if raw is None:
            os.environ.pop(KEY, None)
        else:
            os.environ[KEY] = raw
        assert prefill_keep_cache_enabled() is want

    def test_read_per_call_not_latched(self):
        os.environ[KEY] = "1"
        assert prefill_keep_cache_enabled() is True
        os.environ[KEY] = "0"
        assert prefill_keep_cache_enabled() is False


def _batch(monkeypatch, clear_mock):
    monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", "0")
    monkeypatch.setattr(ar_module.mx, "eval", MagicMock())
    monkeypatch.setattr(ar_module.mx, "async_eval", MagicMock())
    monkeypatch.setattr(ar_module.mx, "clear_cache", clear_mock)
    return PromptProcessingBatch(
        model=MagicMock(),
        uids=[1],
        input_ids=[[1, 2, 3, 4, 5]],
        max_tokens=[1],
        inputs_embeds=mx.ones((1, 5, 4)),
        prompt_kwargs={},
        prefill_step_size=2,
        warm_cache=[SimpleNamespace(state=mx.array([1]))],
    )


class TestChunkLoopHonoursTheFlag:
    def test_default_clears_after_every_chunk(self, monkeypatch):
        clear = MagicMock()
        batch = _batch(monkeypatch, clear)
        assert batch.prompt_step() == 2
        assert clear.call_count == 1
        assert batch.prompt_step() == 2
        assert clear.call_count == 2

    def test_arm_never_clears(self, monkeypatch):
        monkeypatch.setenv(KEY, "1")
        clear = MagicMock()
        batch = _batch(monkeypatch, clear)
        assert batch.prompt_step() == 2
        assert batch.prompt_step() == 2
        clear.assert_not_called()

    def test_explicit_off_clears(self, monkeypatch):
        monkeypatch.setenv(KEY, "0")
        clear = MagicMock()
        batch = _batch(monkeypatch, clear)
        assert batch.prompt_step() == 2
        assert clear.call_count == 1


class TestOnlyTheChunkLoopsAreGated:
    """The decode loop's periodic clear and the teardown clears must stay unconditional."""

    def test_gated_call_sites(self):
        import inspect

        from mlx_vlm.server import generation as gen_module

        for mod in (ar_module, gen_module):
            src = inspect.getsource(mod)
            gated = src.count("if not prefill_keep_cache_enabled():")
            assert gated == (3 if mod is ar_module else 1), (
                f"{mod.__name__}: {gated} gated clear_cache sites"
            )
        # ar.py keeps its unconditional clears: the decode loop's every-256-tokens
        # call and the generator teardown.
        src = inspect.getsource(ar_module)
        assert src.count("mx.clear_cache()") > 4
