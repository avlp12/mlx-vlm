"""The head keeps its connection; the peer's bad day is not the request's.

Before this, ``maybe_open_pipeline`` opened a socket per request and
``close()`` said ``bye`` -- which is why the cooperation driver had to
monkey-patch ``close`` to stop the tail from unloading between requests.  A
pooled connection changes three things that these tests pin: the socket
outlives the request, a dead peer costs a fallback instead of a failure, and
every refusal has a name and a count.
"""

import socket
import threading
import time
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm import pipeline_runtime as pr


@pytest.fixture(autouse=True)
def _clean_metrics():
    pr.METRICS.reset()
    pr.POOL.breaker.reset()
    yield
    pr.METRICS.reset()
    pr.POOL.breaker.reset()


class _FakeHead:
    """Stands in for a connected PipelineHead."""

    instances = []

    def __init__(self, settings=None, split=1, n_layers=3, alive=True, fail_ping=False):
        self.settings = settings
        self.split = split
        self.n_layers = n_layers
        self.sock = object() if alive else None
        self.stats = {}
        self.fail_ping = fail_ping
        self.pings = 0
        self.aborted = False
        self.byes = 0
        _FakeHead.instances.append(self)

    def ping(self):
        self.pings += 1
        if self.fail_ping:
            raise OSError("stale socket")
        return True

    def abort(self):
        self.aborted = True
        self.sock = None

    def close(self):
        self.byes += 1
        self.sock = None

    def finalize(self, cache):
        return {
            "wire_send_s": 0.5,
            "handoff": {"handoff_bytes": 4096, "handoff_wire_recv_s": 0.25},
        }

    def begin(self, tokens, chunk, *, input_ids):
        return None

    def local_caches(self, cache):
        return []

    def prefill_chunk(self, *a, **k):
        return None


def _settings(port=1):
    return pr.PipelineSettings(
        ("127.0.0.1", port),
        None,
        "1",
        1,
        "unused",
        "socket",
        model_sha256="a" * 64,
        source_revision="b" * 40,
        io_timeout=2.0,
    )


def _pool(monkeypatch, factory):
    pool = pr.PipelinePool(breaker=pr.CircuitBreaker(threshold=2, cooldown=100.0))
    monkeypatch.setattr(pr, "PipelineHead", factory)
    return pool


def test_one_connection_is_reused_across_requests(monkeypatch):
    made = []

    class Factory(_FakeHead):
        def __init__(self, settings, split, n_layers):
            super().__init__(settings, split, n_layers)
            made.append(self)

        def connect(self):
            return self

    pool = _pool(monkeypatch, Factory)
    s = _settings()
    key = pr._pool_key(s, 1, 3)
    for _ in range(3):
        head = pool.acquire(s, 1, 3)
        lease = pr.PooledPipelineHead(pool, head, key)
        lease.finalize(cache=None)
        lease.close()
    assert len(made) == 1, "the socket must outlive the request"
    assert made[0].pings == 2, "a reused socket is checked before it is trusted"
    assert made[0].byes == 0, "close() is the request boundary, not the connection's"
    assert pool.idle_count() == 1
    snap = pr.METRICS.snapshot()
    assert snap["pp_used"] == 3
    assert snap["pp_handoff_bytes"] == 3 * 4096
    assert snap["pp_wire_s"] == pytest.approx(3 * 0.75)

    pool.shutdown()
    assert made[0].byes == 1, "shutdown() is what says bye"
    assert pool.idle_count() == 0


def test_a_stale_pooled_socket_is_reconnected_not_used(monkeypatch):
    made = []

    class Factory(_FakeHead):
        def __init__(self, settings, split, n_layers):
            super().__init__(settings, split, n_layers, fail_ping=not made)
            made.append(self)

        def connect(self):
            return self

    pool = _pool(monkeypatch, Factory)
    s = _settings()
    key = pr._pool_key(s, 1, 3)
    first = pool.acquire(s, 1, 3)
    lease = pr.PooledPipelineHead(pool, first, key)
    lease.finalize(cache=None)  # a completed request: the socket goes back
    lease.close()
    second = pool.acquire(s, 1, 3)  # ping fails -> discard, dial again
    assert second is not first
    assert first.aborted
    assert len(made) == 2
    assert pr.METRICS.snapshot()["pp_reconnects"] == 1


def test_a_failed_request_discards_its_connection(monkeypatch):
    class Factory(_FakeHead):
        def __init__(self, settings, split, n_layers):
            super().__init__(settings, split, n_layers)

        def connect(self):
            return self

    pool = _pool(monkeypatch, Factory)
    s = _settings()
    key = pr._pool_key(s, 1, 3)
    head = pool.acquire(s, 1, 3)
    lease = pr.PooledPipelineHead(pool, head, key)
    lease.close()  # never finalized: the peer's state is undefined
    assert head.aborted and pool.idle_count() == 0
    assert pr.METRICS.snapshot()["pp_failed"] == 1


def test_close_never_raises_over_the_real_exception(monkeypatch):
    """``close`` runs in the caller's ``finally``; if it raised there it would
    replace the prefill failure with a bookkeeping one."""

    class Exploding(pr.PipelinePool):
        def release(self, head, key, ok):
            raise RuntimeError("bookkeeping blew up")

    pool = Exploding()
    lease = pr.PooledPipelineHead(pool, _FakeHead(), ("k",))
    lease.close()  # must not raise


def test_connect_failure_is_a_fallback_not_a_failure(monkeypatch):
    class Refusing(_FakeHead):
        def __init__(self, settings, split, n_layers):
            super().__init__(settings, split, n_layers)

        def connect(self):
            raise ConnectionRefusedError("tail is down")

    pool = _pool(monkeypatch, Refusing)
    s = _settings()
    assert pool.acquire(s, 1, 3) is None
    assert pr.METRICS.snapshot()["pp_bypass_reason"]["peer_unreachable"] == 1


def test_breaker_opens_after_n_failures_and_half_opens_after_t(monkeypatch):
    clock = {"t": 1000.0}
    breaker = pr.CircuitBreaker(threshold=2, cooldown=30.0, clock=lambda: clock["t"])
    assert breaker.state == "closed" and breaker.allow()
    breaker.record_failure()
    assert breaker.state == "closed", "one bad request is not an outage"
    breaker.record_failure()
    assert breaker.state == "open"
    assert not breaker.allow(), "an open breaker costs nothing per request"
    assert pr.METRICS.snapshot()["pp_breaker_state"] == "open"
    assert pr.METRICS.snapshot()["pp_breaker_trips"] == 1

    clock["t"] += 31
    assert breaker.state == "half_open"
    assert breaker.allow(), "exactly one probe"
    assert not breaker.allow(), "and only one"
    breaker.record_failure()
    assert breaker.state == "open", "a failed probe re-opens"
    clock["t"] += 31
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed" and breaker.allow()
    assert pr.METRICS.snapshot()["pp_breaker_state"] == "closed"


def test_an_open_breaker_bypasses_without_dialling(monkeypatch):
    dials = []

    class Refusing(_FakeHead):
        def __init__(self, settings, split, n_layers):
            super().__init__(settings, split, n_layers)

        def connect(self):
            dials.append(1)
            raise ConnectionRefusedError("tail is down")

    pool = _pool(monkeypatch, Refusing)
    s = _settings()
    assert pool.acquire(s, 1, 3) is None
    assert pool.acquire(s, 1, 3) is None
    assert pool.breaker.state == "open"
    assert pool.acquire(s, 1, 3) is None
    assert len(dials) == 2, "the third request must not pay a connect timeout"
    hist = pr.METRICS.snapshot()["pp_bypass_reason"]
    assert hist["peer_unreachable"] == 2 and hist["breaker_open"] == 1


def test_bypass_reasons_are_counted_by_name():
    from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache

    cold = [ArraysCache(2), CacheList(KVCache(), KVCache())]
    kw = dict(
        ladder=False,
        capture=False,
        warm=False,
        pixel_values=None,
        mask=None,
        cache=cold,
        input_ids=mx.zeros((1, 4), dtype=mx.int32),
        kv_quantized=False,
    )
    assert pr.pipeline_bypass_reason(**kw) is None
    assert pr.pipeline_bypass_reason(**{**kw, "warm": True}) == "warm_prefix"
    assert pr.pipeline_bypass_reason(**{**kw, "ladder": True}) == "apc_checkpoint_ladder"
    assert pr.pipeline_bypass_reason(**{**kw, "ladder": True}) == "apc_checkpoint_ladder"
    hist = pr.METRICS.snapshot()["pp_bypass_reason"]
    assert hist == {"warm_prefix": 1, "apc_checkpoint_ladder": 2}
    assert "None" not in hist and None not in hist


def test_maybe_open_pipeline_names_every_refusal(monkeypatch):
    monkeypatch.setattr(pr, "_CTX", None)
    monkeypatch.delenv("MLX_VLM_PIPELINE_HOSTS", raising=False)
    assert pr.maybe_open_pipeline(object(), 100000) is None
    assert pr.METRICS.snapshot()["pp_bypass_reason"]["disabled"] == 1

    monkeypatch.setattr(pr, "_CTX", None)
    monkeypatch.setenv("MLX_VLM_PIPELINE_HOSTS", "127.0.0.1:39210")
    monkeypatch.setenv("MLX_VLM_PIPELINE_MODEL_SHA256", "a" * 64)
    monkeypatch.setenv("MLX_VLM_PIPELINE_SOURCE_REVISION", "b" * 40)
    model = SimpleNamespace(language_model=SimpleNamespace())
    assert pr.maybe_open_pipeline(model, 100000) is None
    assert pr.METRICS.snapshot()["pp_bypass_reason"]["no_pipeline_hook"] == 1

    monkeypatch.setattr(pr, "_CTX", None)
    model2 = SimpleNamespace(
        language_model=SimpleNamespace(
            pipeline_prefill_head=lambda **k: None, pipeline_num_layers=3
        )
    )
    assert pr.maybe_open_pipeline(model2, 10) is None
    assert pr.METRICS.snapshot()["pp_bypass_reason"]["below_min_tokens"] == 1

    monkeypatch.setattr(pr, "_CTX", None)
    monkeypatch.setattr(
        pr, "resolve_split", lambda *a, **k: (_ for _ in ()).throw(OSError("no calib"))
    )
    assert pr.maybe_open_pipeline(model2, 100000) is None
    assert pr.METRICS.snapshot()["pp_bypass_reason"]["split_unresolved"] == 1


def test_metrics_snapshot_keys_are_stable():
    snap = pr.pipeline_metrics_snapshot()
    for key in (
        "pp_used",
        "pp_bypass_reason",
        "pp_handoff_bytes",
        "pp_wire_s",
        "pp_breaker_state",
        "pp_pool_idle",
        "pp_enabled",
    ):
        assert key in snap, key
    assert isinstance(snap["pp_bypass_reason"], dict)


def test_server_runtime_snapshot_carries_the_pipeline_block():
    import importlib

    app = importlib.import_module("mlx_vlm.server.app")
    block = app._pipeline_prefill_snapshot()
    import inspect

    src = inspect.getsource(app._server_runtime_snapshot)
    assert '"pipeline_prefill": _pipeline_prefill_snapshot()' in src
    assert block["pp_breaker_state"] in ("closed", "open", "half_open")
    assert "pp_bypass_reason" in block


def test_ping_rides_the_live_connection_end_to_end(monkeypatch):
    """A real tail, a real socket: ping must not disturb the run protocol."""
    from mlx_vlm import pipeline_prefill as pp
    from mlx_vlm.tests.test_pipeline_tail_daemon import (
        _FakeModel,
        _free_port,
        _head,
        _tail_args,
    )

    tail_model = _FakeModel()
    monkeypatch.setattr(
        pp,
        "load_stage",
        lambda *a: (tail_model, tail_model.make_cache(), [1, 2], 3, 0.0),
    )
    port = _free_port()
    args = _tail_args(port)
    box = {}
    errors = []

    def tail():
        try:
            pp.run_tail(args, on_ready=lambda d: box.__setitem__("daemon", d))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    th = threading.Thread(target=tail, daemon=True)
    th.start()
    deadline = time.monotonic() + 5
    while "daemon" not in box and time.monotonic() < deadline:
        time.sleep(0.01)
    head = _head(port)
    try:
        head.connect(timeout=5)
        assert head.ping() is True
        assert head.ping() is True
        model = _FakeModel()
        ids = mx.arange(4, dtype=mx.int32)[None, :]
        cache = model.make_cache()
        head.begin(4, 2, input_ids=ids)
        for start in (0, 2):
            head.prefill_chunk(model, ids[:, start : min(start + 2, 3)], None, cache)
        head.finalize(cache)
        assert head.ping() is True, "ping still works after a request"
        head.close()
    finally:
        head.abort()
        box["daemon"].request_shutdown("test")
        th.join(5)
    assert not errors


def test_unloading_the_server_says_bye_to_the_pooled_tails(monkeypatch):
    """A pooled socket outlives the request by design -- so it also outlives
    the model unless unload closes it, which would pin the peer's stage."""
    import importlib

    app = importlib.import_module("mlx_vlm.server.app")
    called = []
    monkeypatch.setattr(pr, "shutdown_pipeline_pool", lambda: called.append(1))
    app._shutdown_pipeline_pool()
    assert called == [1]
