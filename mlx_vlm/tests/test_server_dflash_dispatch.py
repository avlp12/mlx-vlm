"""DFlash dispatch: which server loop a drafter kind decodes in.

``_run_speculative`` never constructs a ``BatchGenerator``, and
``BatchGenerator`` is the only object that holds ``apc_manager``/``vault`` -- so
for as long as the dispatch read ``draft_kind != "mtp"`` every dflash request
was a full cold prefill, and the prefill log's ``cached_tokens=0`` on that path
was a string literal rather than a measurement.  These tests pin the routing by
NAME (dflash moves, eagle3/lookup do not) and the toggle that restores the old
arm for an A/B.
"""

import contextlib
from threading import Event
from types import SimpleNamespace

import pytest

from mlx_vlm.generate import ar as ar_mod
from mlx_vlm.server import generation as server_generation

ENV = "MLX_VLM_DFLASH_CONTINUOUS_BATCHING"


# --------------------------------------------------------------------------
# The toggle itself
# --------------------------------------------------------------------------
class TestContinuousBatchingPredicate:
    def test_default_is_on_for_dflash(self, monkeypatch):
        monkeypatch.delenv(ENV, raising=False)
        assert server_generation.dflash_continuous_batching_enabled() is True
        assert server_generation._uses_continuous_batching_loop("dflash") is True

    def test_zero_restores_the_legacy_loop(self, monkeypatch):
        monkeypatch.setenv(ENV, "0")
        assert server_generation.dflash_continuous_batching_enabled() is False
        assert server_generation._uses_continuous_batching_loop("dflash") is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", ""])
    def test_only_zero_turns_it_off(self, monkeypatch, value):
        monkeypatch.setenv(ENV, value)
        assert server_generation._uses_continuous_batching_loop("dflash") is True

    @pytest.mark.parametrize("kind", ["eagle3", "lookup"])
    @pytest.mark.parametrize("value", ["1", "0"])
    def test_other_kinds_never_move(self, monkeypatch, kind, value):
        monkeypatch.setenv(ENV, value)
        assert server_generation._uses_continuous_batching_loop(kind) is False

    @pytest.mark.parametrize("kind", [None, "mtp"])
    @pytest.mark.parametrize("value", ["1", "0"])
    def test_mtp_and_no_drafter_are_unaffected(self, monkeypatch, kind, value):
        monkeypatch.setenv(ENV, value)
        assert server_generation._uses_continuous_batching_loop(kind) is True


# --------------------------------------------------------------------------
# The dispatch in _run_impl
# --------------------------------------------------------------------------
def _make_generator(monkeypatch, draft_kind):
    """A ResponseGenerator whose GPU loop exits on its first poll.

    Built with ``__new__`` and only the attributes ``_run_impl`` reads before
    the dispatch, in the style of the existing server tests.  ``wired_limit`` is
    neutralised because the fake model is not an ``nn.Module``.
    """
    gen = server_generation.ResponseGenerator.__new__(
        server_generation.ResponseGenerator
    )
    gen.draft_model = None
    gen.draft_kind = None
    gen._stop = False
    gen._ready = Event()
    gen._load_error = None

    draft_model = object()
    calls = {"collect": [], "speculative": 0, "batch_generator": 0}

    def fake_initialize_model():
        gen.model = SimpleNamespace(language_model=object())
        gen.processor = SimpleNamespace()
        gen.config = SimpleNamespace()
        gen.stop_tokens = set()
        gen.draft_model = draft_model
        gen.draft_kind = draft_kind
        gen.tokenizer = SimpleNamespace()

    def fake_collect(*, active, idle_timeout=0.1, coalesce_s=0.0, capacity=None):
        del idle_timeout, capacity
        calls["collect"].append((active, coalesce_s))
        return [], True

    def fake_speculative():
        calls["speculative"] += 1

    def fake_batch_generator(*args, **kwargs):
        calls["batch_generator"] += 1
        raise AssertionError("no request is ever admitted in these tests")

    gen._initialize_model = fake_initialize_model
    gen._collect_pending_requests = fake_collect
    gen._run_speculative = fake_speculative
    monkeypatch.setattr(
        server_generation, "wired_limit", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(server_generation, "BatchGenerator", fake_batch_generator)
    return gen, calls


class TestRunImplDispatch:
    def test_dflash_reaches_the_continuous_batching_loop_by_default(self, monkeypatch):
        monkeypatch.delenv(ENV, raising=False)
        gen, calls = _make_generator(monkeypatch, "dflash")

        gen._run_impl()

        assert calls["speculative"] == 0, (
            "dflash must not take _run_speculative: that loop never constructs a "
            "BatchGenerator, so apc_manager and the vault are never wired and "
            "every request is a cold prefill."
        )
        assert calls["collect"], "the continuous-batching loop must have polled"

    def test_dflash_toggle_off_restores_the_legacy_loop(self, monkeypatch):
        monkeypatch.setenv(ENV, "0")
        gen, calls = _make_generator(monkeypatch, "dflash")

        gen._run_impl()

        assert calls["speculative"] == 1
        assert calls["collect"] == [], (
            "the legacy loop does its own polling; the continuous-batching loop "
            "must not have run"
        )

    @pytest.mark.parametrize("kind", ["eagle3", "lookup"])
    def test_eagle3_and_lookup_keep_the_legacy_loop(self, monkeypatch, kind):
        monkeypatch.setenv(ENV, "1")
        gen, calls = _make_generator(monkeypatch, kind)

        gen._run_impl()

        assert calls["speculative"] == 1
        assert calls["collect"] == []

    def test_mtp_still_uses_the_continuous_batching_loop(self, monkeypatch):
        monkeypatch.setenv(ENV, "0")
        gen, calls = _make_generator(monkeypatch, "mtp")

        gen._run_impl()

        assert calls["speculative"] == 0
        assert calls["collect"]

    def test_dflash_gets_the_idle_coalescing_window(self, monkeypatch):
        """Parity with the legacy loop, which coalesced for every drafter kind."""
        monkeypatch.delenv(ENV, raising=False)
        monkeypatch.setenv("MLX_VLM_SPEC_BATCH_COALESCE_MS", "37")
        gen, calls = _make_generator(monkeypatch, "dflash")

        gen._run_impl()

        assert calls["collect"] == [(False, 0.037)]


# --------------------------------------------------------------------------
# The new admission refusal
# --------------------------------------------------------------------------
def _sequence(uid, ids):
    # (uid, input_ids, max_tokens, prompt_kwargs, logits_processors,
    #  thinking_budget_criteria) -- the tuple _build_mixed_prompt_batch unpacks.
    return (uid, list(ids), 8, {"inputs_embeds": object()}, None, None)


def _mixed_batch_generator(draft_kind):
    gen = ar_mod.BatchGenerator.__new__(ar_mod.BatchGenerator)
    gen.apc_manager = object()
    gen.vault = None
    gen.draft_kind = draft_kind
    gen.model = SimpleNamespace()
    # ``__del__`` -> ``close()`` reads this; a __new__-built generator has no
    # constructor to set it and the deallocator would raise at collection time.
    gen._wire_stack = None
    seen = {"right_pad_policy": 0}

    gen._apc_pick_for = lambda s, serve_batch_width=1: {
        "prefix_len": 4,
        "warm_cache": object(),
        "extra_hash": 0,
        "matched_blocks": [],
    }

    def fake_policy(sequences, picks):
        seen["right_pad_policy"] += 1
        return None, None  # stop the builder right after the refusal check

    gen._apply_right_pad_policy = fake_policy
    return gen, seen


class TestDflashWarmMultirowRefusal:
    def test_multirow_warm_batch_is_refused_for_dflash(self, caplog):
        ar_mod.reset_prefill_batch_refusal_counts()
        gen, seen = _mixed_batch_generator("dflash")
        sequences = [_sequence(1, [1, 2, 3, 4, 5]), _sequence(2, [1, 2, 3, 4, 5, 6])]

        with caplog.at_level("INFO", logger=ar_mod.logger.name):
            assert gen._build_mixed_prompt_batch(sequences) is None

        assert seen["right_pad_policy"] == 0, (
            "the refusal must fire BEFORE the right-pad policy: the hazard is "
            "the padding itself, not how the rows are grouped"
        )
        counts = ar_mod.prefill_batch_refusal_counts()
        assert counts.get("dflash_warm_multirow") == 1
        assert counts.get("dflash_warm_multirow_rows_deferred") == 2
        assert "dflash_warm_multirow" in caplog.text
        ar_mod.reset_prefill_batch_refusal_counts()

    def test_single_warm_row_is_still_admitted_for_dflash(self, caplog):
        ar_mod.reset_prefill_batch_refusal_counts()
        gen, seen = _mixed_batch_generator("dflash")

        with caplog.at_level("INFO", logger=ar_mod.logger.name):
            assert gen._build_mixed_prompt_batch([_sequence(1, [1, 2, 3, 4, 5])]) is None

        # None here comes from the stubbed right-pad policy, not the refusal:
        # B=1 means right_pad_per_row == [0] and there is no padding to read.
        assert seen["right_pad_policy"] == 1
        assert "dflash_warm_multirow" not in ar_mod.prefill_batch_refusal_counts()
        assert "dflash_warm_multirow" not in caplog.text

    @pytest.mark.parametrize("kind", ["mtp", "eagle3", None])
    def test_other_kinds_keep_multirow_warm_batches(self, kind):
        ar_mod.reset_prefill_batch_refusal_counts()
        gen, seen = _mixed_batch_generator(kind)
        sequences = [_sequence(1, [1, 2, 3, 4, 5]), _sequence(2, [1, 2, 3, 4, 5, 6])]

        assert gen._build_mixed_prompt_batch(sequences) is None
        assert seen["right_pad_policy"] == 1
        assert "dflash_warm_multirow" not in ar_mod.prefill_batch_refusal_counts()
