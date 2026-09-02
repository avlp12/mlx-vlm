"""GPU keepalive (server/keepalive.py) -- PA796.

Measured basis for the feature, on epsilon 2026-09-02 (L3_keepwarm.json):
macOS drops the model's ~173 GB wired residency within ~2 s of GPU idle and the
next forward pays a flat ~1.19 s to rebuild it (+92% on a 512-token prefill,
+12.5% on a 4096-token one).  A 1 Hz tick removes 99.8% of that at ~0.3% duty.

These tests use a fake tick, so nothing here touches the GPU.
"""

from queue import Queue
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mlx_vlm.server as server
import mlx_vlm.server.generation as server_generation
from mlx_vlm.server.keepalive import (
    DEFAULT_KEEPALIVE_HZ,
    ENV_KEEPALIVE_HZ,
    GpuKeepalive,
    get_keepalive_hz,
)


def _fake_keepalive(hz=1.0):
    """A keepalive whose tick records instead of dispatching to the GPU."""
    ka = GpuKeepalive(hz=hz)
    ka.calls = []
    ka._do_tick = lambda: ka.calls.append(True)
    return ka


# --------------------------------------------------------------- configuration
class TestKeepaliveConfig:
    def test_default_is_one_hz(self, monkeypatch):
        monkeypatch.delenv(ENV_KEEPALIVE_HZ, raising=False)
        assert get_keepalive_hz() == DEFAULT_KEEPALIVE_HZ == 1.0
        assert GpuKeepalive().enabled is True

    def test_zero_disables(self, monkeypatch):
        monkeypatch.setenv(ENV_KEEPALIVE_HZ, "0")
        ka = GpuKeepalive()
        assert ka.enabled is False
        ka.arm()
        assert ka.tick_if_due(now=1e9) is False
        assert "disabled" in ka.describe()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(ENV_KEEPALIVE_HZ, "4")
        ka = GpuKeepalive()
        assert ka.hz == 4.0
        assert ka.interval_s == pytest.approx(0.25)

    def test_garbage_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(ENV_KEEPALIVE_HZ, "not-a-number")
        assert get_keepalive_hz() == DEFAULT_KEEPALIVE_HZ

    def test_explicit_hz_beats_env(self, monkeypatch):
        monkeypatch.setenv(ENV_KEEPALIVE_HZ, "0")
        assert GpuKeepalive(hz=2.0).enabled is True


# ----------------------------------------------------------------- tick timing
class TestKeepaliveTicking:
    def test_unarmed_never_ticks(self):
        """Must not run before the model has finished loading."""
        ka = _fake_keepalive()
        assert ka.armed is False
        assert ka.tick_if_due(now=1e9) is False
        assert ka.calls == []

    def test_arm_defers_the_first_tick_by_one_interval(self, monkeypatch):
        monkeypatch.setattr(server_generation.time, "monotonic", lambda: 100.0)
        ka = _fake_keepalive(hz=1.0)
        ka.arm()
        assert ka.tick_if_due(now=100.5) is False   # load just touched the GPU
        assert ka.tick_if_due(now=101.0) is True

    def test_respects_the_interval(self):
        ka = _fake_keepalive(hz=1.0)
        ka.arm()
        ka._last = 0.0
        assert ka.tick_if_due(now=0.5) is False
        assert ka.tick_if_due(now=1.0) is True
        assert ka.tick_if_due(now=1.5) is False
        assert ka.tick_if_due(now=2.0) is True
        assert ka.ticks == 2

    def test_close_stops_ticking_and_drops_operands(self):
        ka = _fake_keepalive()
        ka.arm()
        ka._a = ka._b = object()
        ka.close()
        assert ka.armed is False
        assert ka._a is None and ka._b is None
        assert ka.tick_if_due(now=1e9) is False

    def test_failing_tick_disables_itself_and_never_raises(self):
        ka = GpuKeepalive(hz=1.0)
        ka._do_tick = lambda: (_ for _ in ()).throw(RuntimeError("metal is sad"))
        ka.arm()
        ka._last = 0.0
        assert ka.tick_if_due(now=10.0) is False   # swallowed
        assert ka.enabled is False                 # and latched off
        assert ka.tick_if_due(now=20.0) is False

    def test_cap_wait(self):
        ka = _fake_keepalive(hz=1.0)
        assert ka.cap_wait(5.0) == 5.0             # unarmed: identity
        ka.arm()
        assert ka.cap_wait(5.0) == pytest.approx(1.0)
        assert ka.cap_wait(0.1) == pytest.approx(0.1)

    def test_cap_wait_identity_when_disabled(self, monkeypatch):
        monkeypatch.setenv(ENV_KEEPALIVE_HZ, "0")
        ka = GpuKeepalive()
        ka.arm()
        assert ka.cap_wait(5.0) == 5.0


# ------------------------------------------------- integration with the loop
def _bare_generator(keepalive):
    gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
    gen.requests = Queue()
    gen._stop = False
    gen._keepalive = keepalive
    return gen


class TestKeepaliveInGenerationLoop:
    def test_ticks_on_the_idle_path(self):
        ka = _fake_keepalive()
        ka.arm()
        ka._last = 0.0
        gen = _bare_generator(ka)

        pending, should_stop = gen._collect_pending_requests(
            active=False, idle_timeout=0.001
        )

        assert pending == [] and should_stop is False
        assert len(ka.calls) == 1

    def test_suppressed_while_a_request_is_in_flight(self):
        """active=True is the in-flight case; a tick there would contend."""
        ka = _fake_keepalive()
        ka.arm()
        ka._last = 0.0
        gen = _bare_generator(ka)

        for _ in range(5):
            gen._collect_pending_requests(active=True, idle_timeout=0.001)

        assert ka.calls == []

    def test_idle_path_still_returns_queued_work(self):
        ka = _fake_keepalive()
        ka.arm()
        ka._last = 0.0
        gen = _bare_generator(ka)
        item = object()
        gen.requests.put(item)

        pending, _ = gen._collect_pending_requests(active=False, idle_timeout=0.5)

        assert pending == [item]

    def test_disabled_keepalive_leaves_the_idle_timeout_alone(self, monkeypatch):
        monkeypatch.setenv(ENV_KEEPALIVE_HZ, "0")
        gen = _bare_generator(GpuKeepalive())
        # Record every call: after the blocking get raises, the collector drains
        # with get_nowait(), which re-enters this same override with timeout=None.
        # Asserting on a single slot would read the drain call, not the wait.
        seen = []

        class _Q(Queue):
            def get(self, block=True, timeout=None):
                seen.append(timeout)
                raise server_generation.QueueEmpty

        gen.requests = _Q()
        gen._collect_pending_requests(active=False, idle_timeout=7.5)
        assert seen[0] == 7.5, seen

    def test_accessor_tolerates_an_uninitialised_instance(self):
        """The suite builds ResponseGenerator via __new__ in ~22 places."""
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.requests = Queue()
        gen._stop = False
        assert gen.keepalive.armed is False
        pending, should_stop = gen._collect_pending_requests(
            active=False, idle_timeout=0.001
        )
        assert pending == [] and should_stop is False


class TestKeepaliveLifecycle:
    def test_armed_only_after_the_model_loads(self, monkeypatch):
        ka = _fake_keepalive()
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen._keepalive = ka
        gen._ready = server_generation.Event()
        gen._load_error = None
        gen.model = SimpleNamespace()
        order = []

        gen._initialize_model = lambda: order.append(
            ("load", ka.armed)
        )
        monkeypatch.setattr(server_generation, "is_diffusion_model", lambda _m: True)
        gen._run_diffusion = lambda: order.append(("serve", ka.armed))

        gen._run_impl()

        assert order == [("load", False), ("serve", True)]

    def test_not_armed_when_the_load_fails(self):
        ka = _fake_keepalive()
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen._keepalive = ka
        gen._ready = server_generation.Event()
        gen._load_error = None

        def boom():
            raise RuntimeError("no weights")

        gen._initialize_model = boom
        gen._run_impl()

        assert isinstance(gen._load_error, RuntimeError)
        assert ka.armed is False

    def test_closed_when_the_generation_thread_exits(self):
        ka = _fake_keepalive()
        ka.arm()
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen._keepalive = ka
        gen._run_impl = lambda: None
        gen._release_model_refs = lambda: None

        gen._run()

        assert ka.armed is False

    def test_closed_even_when_the_thread_dies_with_an_exception(self):
        ka = _fake_keepalive()
        ka.arm()
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen._keepalive = ka
        gen._release_model_refs = lambda: None

        def boom():
            raise RuntimeError("gpu wedged")

        gen._run_impl = boom
        with pytest.raises(RuntimeError):
            gen._run()

        assert ka.armed is False


# --------------------------------------------------- wired-limit hoist (PA796b)
class _LimitSpy:
    """Stands in for mx.set_wired_limit and remembers the current value."""

    def __init__(self, start=0):
        self.value = start
        self.calls = []

    def set(self, v):
        self.calls.append(v)
        old, self.value = self.value, v
        return old


MAXREC = 512 * 2**30


def _patch_mlx(monkeypatch, spy, raising=False):
    def _set(v):
        if raising:
            raise RuntimeError("wiring refused by the driver")
        return spy.set(v)

    monkeypatch.setattr(mx, "set_wired_limit", _set)
    monkeypatch.setattr(mx, "device_info",
                        lambda: {"max_recommended_working_set_size": MAXREC})
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "synchronize", lambda *a, **k: None)


def _loaded_generator(keepalive=None):
    gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
    gen._keepalive = keepalive or _fake_keepalive()
    gen._wired_stack = None
    gen._ready = server_generation.Event()
    gen._load_error = None
    gen.model = SimpleNamespace(language_model=SimpleNamespace())
    gen._initialize_model = lambda: None
    return gen


class TestWiredLimitHoist:
    def test_limit_is_non_zero_before_the_first_request(self, monkeypatch):
        """The window the keepalive alone cannot cover (L3_wirelimit.json arm C)."""
        spy = _LimitSpy(start=0)
        _patch_mlx(monkeypatch, spy)
        gen = _loaded_generator()
        seen = {}
        monkeypatch.setattr(server_generation, "is_diffusion_model", lambda _m: True)
        gen._run_diffusion = lambda: seen.update(limit_while_serving=spy.value)

        gen._run_impl()

        assert seen["limit_while_serving"] == MAXREC
        assert seen["limit_while_serving"] > 0

    def test_released_when_the_generation_thread_exits(self, monkeypatch):
        spy = _LimitSpy(start=0)
        _patch_mlx(monkeypatch, spy)
        gen = _loaded_generator()
        gen._release_model_refs = lambda: None
        monkeypatch.setattr(server_generation, "is_diffusion_model", lambda _m: True)
        gen._run_diffusion = lambda: None

        gen._run()

        assert spy.value == 0
        assert gen._wired_stack is None

    def test_inner_context_no_longer_drops_the_limit_to_zero(self, monkeypatch):
        """BatchGenerator.close() must restore max_rec, not the floor.

        This is the property that lets BatchGenerator keep its own wiring
        untouched: nesting is idempotent.
        """
        spy = _LimitSpy(start=0)
        _patch_mlx(monkeypatch, spy)
        gen = _loaded_generator()
        gen._hold_wired_limit([object()])
        assert spy.value == MAXREC

        with server_generation.wired_limit(SimpleNamespace(), [object()]):
            assert spy.value == MAXREC
        assert spy.value == MAXREC, "inner exit dropped the limit to the floor"

        gen._release_wired_limit()
        assert spy.value == 0

    def test_a_refused_wiring_never_stops_the_server(self, monkeypatch):
        spy = _LimitSpy(start=0)
        _patch_mlx(monkeypatch, spy, raising=True)
        gen = _loaded_generator()
        served = []
        monkeypatch.setattr(server_generation, "is_diffusion_model", lambda _m: True)
        gen._run_diffusion = lambda: served.append(True)

        gen._run_impl()          # must not raise

        assert served == [True], "serving did not continue after a wiring refusal"
        assert gen._wired_stack is None

    def test_no_model_is_a_no_op(self, monkeypatch):
        spy = _LimitSpy(start=0)
        _patch_mlx(monkeypatch, spy)
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen._wired_stack = None
        gen.model = None

        gen._hold_wired_limit([object()])

        assert spy.calls == []
        assert gen._wired_stack is None

    def test_env_toggle_off_skips_the_hoist(self, monkeypatch):
        """Benchmark harnesses need the un-hoisted behaviour to A/B against."""
        spy = _LimitSpy(start=0)
        _patch_mlx(monkeypatch, spy)
        monkeypatch.setenv(server_generation.ENV_HOIST_WIRED_LIMIT, "0")
        gen = _loaded_generator()

        gen._hold_wired_limit([object()])

        assert spy.calls == []
        assert gen._wired_stack is None

    def test_env_toggle_defaults_on(self, monkeypatch):
        spy = _LimitSpy(start=0)
        _patch_mlx(monkeypatch, spy)
        monkeypatch.delenv(server_generation.ENV_HOIST_WIRED_LIMIT, raising=False)
        gen = _loaded_generator()

        gen._hold_wired_limit([object()])

        assert spy.value == MAXREC
        gen._release_wired_limit()

    def test_release_is_idempotent(self, monkeypatch):
        spy = _LimitSpy(start=0)
        _patch_mlx(monkeypatch, spy)
        gen = _loaded_generator()
        gen._hold_wired_limit([object()])
        gen._release_wired_limit()
        gen._release_wired_limit()          # must not raise or re-set
        assert spy.value == 0
