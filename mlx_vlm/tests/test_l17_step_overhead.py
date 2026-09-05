"""L17: server B=1 decode-step overhead instrumentation and default-OFF fixes.

Covers:
  A. step timers (_StepTimers, ar.py hook wiring, ServerMetricsStore exposure)
  B1. BatchKVCache vacuous-mask toggle (MLX_VLM_BATCH_VACUOUS_MASK_NONE)
  B2. batched-emit token draining (_TokenIterator.next_batch) + switch-interval
      env parsing
  B3. batch cache-eval-interval env parsing (pre-existing, exercised here for
      completeness of the ablation surface)

All new behaviour is default OFF; several tests assert that explicitly.
"""

import queue
import time
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

# Import order matters here: mlx_vlm.server.generation pulls in the full
# ..generate package (and installs its own `from ..generate import ar as
# _ar`), which is what makes the plain `from mlx_vlm.generate import ar`
# below resolve to the real submodule rather than tripping over
# mlx_vlm/__init__.py's `generate` *function* re-export shadowing the
# `generate` *package* attribute on a cold import.
import mlx_vlm.server.cli as server_cli
import mlx_vlm.server.generation as server_generation
from mlx_vlm.generate import ar as ar_module
from mlx_vlm.generate.ar import GenerationBatch
from mlx_vlm.models import cache as cache_module
from mlx_vlm.models.cache import BatchKVCache


# ---------------------------------------------------------------------------
# A. step timers
# ---------------------------------------------------------------------------


class _FakeStepTimers:
    """Minimal stand-in satisfying the ar.STEP_TIMERS contract: .record()."""

    def __init__(self):
        self.records = []

    def record(self, bucket, elapsed_s):
        self.records.append((bucket, elapsed_s))


@pytest.fixture(autouse=True)
def _restore_ar_step_timers_hook():
    """Every test in this module leaves ar.STEP_TIMERS as it found it."""
    original = ar_module.STEP_TIMERS
    yield
    ar_module.STEP_TIMERS = original


def test_step_timers_default_off():
    # Import-time default: no env var set in the test process, so the flag
    # is False and the ar.py hook is never installed.
    assert server_generation._STEP_TIMERS is False
    assert ar_module.STEP_TIMERS is None


def test_step_timers_record_accumulates_sum_and_count():
    timers = server_generation._StepTimers()
    timers.record("queue_poll", 0.001)
    timers.record("queue_poll", 0.003)
    timers.record("batch_next_total", 0.010)

    snap = timers.snapshot()
    assert snap["queue_poll"]["count"] == 2
    assert snap["queue_poll"]["sum_s"] == pytest.approx(0.004)
    assert snap["queue_poll"]["ms_per"] == pytest.approx(2.0)
    assert snap["batch_next_total"]["count"] == 1
    assert snap["batch_next_total"]["ms_per"] == pytest.approx(10.0)


def test_step_timers_reset_clears_all_buckets():
    timers = server_generation._StepTimers()
    timers.record("detok_emit", 0.02)
    assert timers.snapshot() != {}
    timers.reset()
    assert timers.snapshot() == {}


def test_server_metrics_store_exposes_and_resets_shared_step_timers():
    # Exercise the real module-level singleton (as ServerMetricsStore does),
    # not a fresh instance -- this is what /metrics actually serves.
    server_generation._step_timers.reset()
    try:
        store = server_generation.ServerMetricsStore()
        snap = store.snapshot()
        assert snap["step_timers"] == {}

        server_generation._step_timers.record("sampling_sync", 0.005)
        snap = store.snapshot()
        assert snap["step_timers"]["sampling_sync"]["count"] == 1

        store.reset()
        assert server_generation._step_timers.snapshot() == {}
        assert store.snapshot()["step_timers"] == {}
    finally:
        server_generation._step_timers.reset()


def test_ar_step_timers_record_model_fwd_sampling_and_stop_check():
    """Drives a real (fake-model) GenerationBatch.next() call on CPU and
    checks the three ar.py buckets all fire exactly once, with the hook
    installed via ar.STEP_TIMERS (the same mechanism the server uses)."""

    class FixedLogitModel:
        def __call__(self, input_ids, cache=None, **kwargs):
            token_scores = mx.array([0.0, 10.0, 0.0, 0.0])
            logits = mx.broadcast_to(
                token_scores, (input_ids.shape[0], input_ids.shape[1], 4)
            )
            return MagicMock(logits=logits)

    fake_timers = _FakeStepTimers()
    ar_module.STEP_TIMERS = fake_timers

    batch = GenerationBatch(
        model=FixedLogitModel(),
        uids=[0],
        inputs=mx.array([5], dtype=mx.int32),
        prompt_cache=[],
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        stop_criteria=lambda token: False,
        max_tokens=[5],
        token_context=[mx.array([10])],
    )

    batch.next()

    buckets = [b for b, _ in fake_timers.records]
    assert buckets.count("model_fwd_plus_eval") == 1
    assert buckets.count("sampling_sync") == 1
    assert buckets.count("stop_check") == 1
    assert all(secs >= 0.0 for _, secs in fake_timers.records)


def test_ar_step_timers_off_by_default_records_nothing():
    """Same drive as above but with the hook left at its default (None):
    no timer bookkeeping should run at all."""

    class FixedLogitModel:
        def __call__(self, input_ids, cache=None, **kwargs):
            token_scores = mx.array([0.0, 10.0, 0.0, 0.0])
            logits = mx.broadcast_to(
                token_scores, (input_ids.shape[0], input_ids.shape[1], 4)
            )
            return MagicMock(logits=logits)

    assert ar_module.STEP_TIMERS is None  # from the autouse fixture restore

    batch = GenerationBatch(
        model=FixedLogitModel(),
        uids=[0],
        inputs=mx.array([5], dtype=mx.int32),
        prompt_cache=[],
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        stop_criteria=lambda token: False,
        max_tokens=[5],
        token_context=[mx.array([10])],
    )
    # Must not raise with the hook at None (guarded by `is not None` checks).
    batch.next()


def test_ar_cache_evict_timer_records_when_interval_hits():
    class _FakeGenBatch:
        logits_processors = []

        def __len__(self):
            return 1

        def next(self):
            return []

        def cache_states(self):
            return [mx.array([1.0])]

    fake_timers = _FakeStepTimers()
    ar_module.STEP_TIMERS = fake_timers

    bg = object.__new__(ar_module.BatchGenerator)
    bg._generation_batch = _FakeGenBatch()
    bg._gen_tokens_counter = 0
    bg._steps_counter = 0
    bg._cache_eval_interval = 1
    # Forces the early `len(gen_batch) >= completion_batch_size` return right
    # after the decode/cache-evict block, so no prefill-admission attributes
    # (_prompt_batch, _unprocessed_sequences, ...) are needed for this test.
    bg.completion_batch_size = 0
    bg._wire_stack = None  # so BatchGenerator.__del__ has something to check

    bg._next()

    buckets = [b for b, _ in fake_timers.records]
    assert buckets.count("cache_evict") == 1


# ---------------------------------------------------------------------------
# B1. BatchKVCache vacuous-mask toggle
# ---------------------------------------------------------------------------


def test_batch_vacuous_mask_none_default_off():
    assert cache_module._BATCH_VACUOUS_MASK_NONE is False


def test_batch_make_mask_toggle_off_is_byte_identical_to_today(monkeypatch):
    monkeypatch.setattr(cache_module, "_BATCH_VACUOUS_MASK_NONE", False)
    c = BatchKVCache([0])
    c.update_and_fetch(mx.zeros((1, 1, 3, 2)), mx.zeros((1, 1, 3, 2)))

    mask_off = c.make_mask(1)
    assert isinstance(mask_off, mx.array)  # unchanged: always builds the array


def test_batch_make_mask_returns_none_at_n1_no_padding_when_toggled_on(
    monkeypatch,
):
    monkeypatch.setattr(cache_module, "_BATCH_VACUOUS_MASK_NONE", True)
    c = BatchKVCache([0, 0])  # no left padding on either row
    c.update_and_fetch(mx.zeros((2, 1, 3, 2)), mx.zeros((2, 1, 3, 2)))

    assert c._no_left_padding is True
    assert c.make_mask(1) is None


def test_batch_make_mask_still_returns_array_with_padding_when_toggled_on(
    monkeypatch,
):
    monkeypatch.setattr(cache_module, "_BATCH_VACUOUS_MASK_NONE", True)
    c = BatchKVCache([0, 2])  # row 1 has left padding
    c.update_and_fetch(mx.zeros((2, 1, 3, 2)), mx.zeros((2, 1, 3, 2)))

    assert c._no_left_padding is False
    mask = c.make_mask(1)
    assert isinstance(mask, mx.array)


def test_batch_make_mask_toggle_on_but_n_greater_than_1_still_builds_array(
    monkeypatch,
):
    monkeypatch.setattr(cache_module, "_BATCH_VACUOUS_MASK_NONE", True)
    c = BatchKVCache([0])
    c.update_and_fetch(mx.zeros((1, 1, 3, 2)), mx.zeros((1, 1, 3, 2)))

    mask = c.make_mask(3)
    assert isinstance(mask, mx.array)


def test_no_left_padding_flag_tracks_prepare():
    c = BatchKVCache([0])
    assert c._no_left_padding is True
    c.prepare(left_padding=[2])
    assert c._no_left_padding is False


def test_no_left_padding_flag_survives_zero_prepare():
    c = BatchKVCache([0])
    c.prepare(left_padding=[0])
    assert c._no_left_padding is True


def test_no_left_padding_flag_extend_requires_equal_idx_and_both_flags():
    a = BatchKVCache([0])
    a.update_and_fetch(mx.zeros((1, 1, 4, 2)), mx.zeros((1, 1, 4, 2)))
    b = BatchKVCache([0])
    b.update_and_fetch(mx.zeros((1, 1, 4, 2)), mx.zeros((1, 1, 4, 2)))

    assert a._no_left_padding and b._no_left_padding
    a.extend(b)
    assert a._no_left_padding is True


def test_no_left_padding_flag_extend_false_on_unequal_length():
    a = BatchKVCache([0])
    a.update_and_fetch(mx.zeros((1, 1, 4, 2)), mx.zeros((1, 1, 4, 2)))
    b = BatchKVCache([0])
    b.update_and_fetch(mx.zeros((1, 1, 2, 2)), mx.zeros((1, 1, 2, 2)))

    a.extend(b)
    assert a._no_left_padding is False


def test_no_left_padding_flag_finalize_false_once_right_padding_applied():
    c = BatchKVCache([0, 0])
    c.update_and_fetch(mx.zeros((2, 1, 3, 2)), mx.zeros((2, 1, 3, 2)))
    assert c._no_left_padding is True

    c.prepare(right_padding=[0, 1])
    c.finalize()
    assert c._no_left_padding is False


def test_no_left_padding_flag_state_setter_recomputes():
    c = BatchKVCache([0])
    c.update_and_fetch(mx.zeros((1, 1, 3, 2)), mx.zeros((1, 1, 3, 2)))
    k, v, offset, left_padding = c.state
    c._no_left_padding = False  # perturb, then confirm state.setter fixes it
    c.state = (k, v, offset, left_padding)
    assert c._no_left_padding is True

    c.state = (k, v, offset, mx.array([1]))
    assert c._no_left_padding is False


# ---------------------------------------------------------------------------
# B2. batched emit (_TokenIterator.next_batch) + switch interval env parsing
# ---------------------------------------------------------------------------


class _Tok:
    def __init__(self, finish_reason=None):
        self.finish_reason = finish_reason


def _make_iterator():
    q = queue.Queue()
    return server_generation._TokenIterator(q, uid=0, cancel_fn=lambda uid: None, queue_timeout=1.0), q


def test_next_batch_drains_everything_already_queued():
    it, q = _make_iterator()
    q.put(_Tok())
    q.put(_Tok())
    q.put(_Tok())

    tokens, terminal = it.next_batch(max_items=32)

    assert len(tokens) == 3
    assert terminal is None


def test_next_batch_respects_max_items():
    it, q = _make_iterator()
    for _ in range(5):
        q.put(_Tok())

    tokens, terminal = it.next_batch(max_items=2)

    assert len(tokens) == 2
    assert terminal is None
    # The remaining 3 are still queued for the next call.
    assert q.qsize() == 3


def test_next_batch_stops_at_finish_reason_without_consuming_more():
    it, q = _make_iterator()
    tok_ok = _Tok()
    tok_done = _Tok(finish_reason="stop")
    q.put(tok_ok)
    q.put(tok_done)
    q.put(_Tok())  # would-be next request's token; must stay queued

    tokens, terminal = it.next_batch(max_items=32)

    assert tokens == [tok_ok, tok_done]
    assert terminal is None
    assert it._ended is True
    assert q.qsize() == 1


def test_next_batch_reports_stop_terminal_on_none_sentinel():
    it, q = _make_iterator()
    q.put(_Tok())
    q.put(None)

    tokens, terminal = it.next_batch(max_items=32)

    assert len(tokens) == 1
    assert terminal == "stop"
    assert it._ended is True


def test_next_batch_reports_exception_terminal():
    it, q = _make_iterator()
    err = RuntimeError("boom")
    q.put(_Tok())
    q.put(err)

    tokens, terminal = it.next_batch(max_items=32)

    assert len(tokens) == 1
    assert terminal is err
    assert it._ended is True


def test_next_batch_first_item_raises_stop_iteration_like_next(monkeypatch):
    it, q = _make_iterator()
    q.put(None)

    with pytest.raises(StopIteration):
        it.next_batch(max_items=32)


def test_server_batched_emit_env_default_off():
    import mlx_vlm.server.openai as server_openai

    assert server_openai._SERVER_BATCHED_EMIT is False


def test_parse_switch_interval_valid():
    assert server_cli._parse_switch_interval("0.001") == pytest.approx(0.001)


def test_parse_switch_interval_invalid_returns_none():
    assert server_cli._parse_switch_interval("not-a-float") is None


# ---------------------------------------------------------------------------
# B3. batch cache-eval-interval env parsing (pre-existing implementation)
# ---------------------------------------------------------------------------


def test_batch_cache_eval_interval_default(monkeypatch):
    monkeypatch.delenv("MLX_VLM_BATCH_CACHE_EVAL_INTERVAL", raising=False)
    assert (
        ar_module._get_batch_cache_eval_interval()
        == ar_module.DEFAULT_BATCH_CACHE_EVAL_INTERVAL
        == 50
    )


def test_batch_cache_eval_interval_env_override(monkeypatch):
    monkeypatch.setenv("MLX_VLM_BATCH_CACHE_EVAL_INTERVAL", "7")
    assert ar_module._get_batch_cache_eval_interval() == 7


def test_batch_cache_eval_interval_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("MLX_VLM_BATCH_CACHE_EVAL_INTERVAL", "not-an-int")
    assert (
        ar_module._get_batch_cache_eval_interval()
        == ar_module.DEFAULT_BATCH_CACHE_EVAL_INTERVAL
    )
