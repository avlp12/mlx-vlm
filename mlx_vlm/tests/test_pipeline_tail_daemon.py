"""The tail is a service, not a one-shot bench process.

The shipped tail served exactly one connection and then unloaded tens of GB of
weights, so every request paid the load again.  These tests pin the service
behaviour that replaces it: accept is re-armed after a connection ends, a
``bye`` retires the connection and not the daemon, a peer that fails takes only
its own connection down, a deadline can only kill a tail that has never been
used, and SIGTERM unloads BEFORE the process exits (a bare kill of a process
holding MLX buffers leaks wired memory that only a reboot reclaims).
"""

import json
import socket
import threading
import time
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm import pipeline_prefill as pp
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _listener():
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    return srv, srv.getsockname()[1]


def _run(daemon):
    out = {}

    def go():
        try:
            out["status"] = daemon.serve_forever()
        except BaseException as exc:  # noqa: BLE001
            out["error"] = exc

    th = threading.Thread(target=go, daemon=True)
    th.start()
    return th, out


def test_accept_is_rearmed_after_every_connection():
    srv, port = _listener()
    seen = []

    def session(sock, addr):
        data = sock.recv(64)
        seen.append(data)
        sock.sendall(b"ok")
        return 1

    unloaded = []
    daemon = pp.TailDaemon(srv, session, unload=lambda: unloaded.append(1))
    th, out = _run(daemon)
    try:
        for i in range(2):
            c = socket.create_connection(("127.0.0.1", port), timeout=5)
            c.sendall(f"req{i}".encode())
            assert c.recv(8) == b"ok"
            c.close()
        deadline = time.monotonic() + 5
        while daemon.counters["connections"] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert seen == [b"req0", b"req1"]
        assert daemon.counters["requests"] == 2
        # still listening: the service outlives its connections
        assert th.is_alive() and daemon.state == "listening"
        assert unloaded == []
    finally:
        daemon.request_shutdown("test")
        th.join(5)
    assert not th.is_alive()
    assert unloaded == [1], "the stage must be released exactly once, on exit"
    assert out["status"]["shutdown_reason"] == "test"
    assert out["status"]["state"] == "stopped"


def test_a_failing_connection_does_not_take_the_service_down():
    srv, port = _listener()
    calls = []

    def session(sock, addr):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("peer went away mid-chunk")
        sock.sendall(b"ok")
        return 1

    daemon = pp.TailDaemon(srv, session)
    th, out = _run(daemon)
    try:
        socket.create_connection(("127.0.0.1", port), timeout=5).close()
        deadline = time.monotonic() + 5
        while daemon.counters["connection_errors"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert daemon.counters["connection_errors"] == 1
        assert "peer went away" in daemon.counters["last_error"]
        c = socket.create_connection(("127.0.0.1", port), timeout=5)
        assert c.recv(8) == b"ok"
        c.close()
        assert daemon.counters["requests"] == 1
    finally:
        daemon.request_shutdown("test")
        th.join(5)
    assert not th.is_alive() and "error" not in out


def test_connect_deadline_only_applies_before_the_first_head():
    srv, _ = _listener()
    daemon = pp.TailDaemon(srv, lambda s, a: 0, connect_timeout=0.3)
    th, out = _run(daemon)
    th.join(5)
    assert isinstance(out.get("error"), TimeoutError)

    # a tail that HAS served a head never dies of a quiet hour
    srv2, port2 = _listener()
    d2 = pp.TailDaemon(srv2, lambda s, a: 1, connect_timeout=0.3)
    th2, out2 = _run(d2)
    try:
        socket.create_connection(("127.0.0.1", port2), timeout=5).close()
        time.sleep(0.8)
        assert th2.is_alive() and "error" not in out2
    finally:
        d2.request_shutdown("test")
        th2.join(5)


def test_idle_timeout_is_a_clean_shutdown_not_an_error():
    srv, _ = _listener()
    daemon = pp.TailDaemon(srv, lambda s, a: 0, idle_timeout=0.3)
    th, out = _run(daemon)
    th.join(5)
    assert not th.is_alive() and "error" not in out
    assert out["status"]["shutdown_reason"] == "idle_timeout"


def test_sigterm_sets_the_flag_and_the_loop_unloads():
    import signal

    srv, _ = _listener()
    order = []
    daemon = pp.TailDaemon(srv, lambda s, a: 0, unload=lambda: order.append("unload"))
    previous = pp.install_tail_signal_handlers(daemon, signums=(signal.SIGTERM,))
    try:
        th, out = _run(daemon)
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)  # what the OS delivers
        th.join(5)
        assert not th.is_alive()
        assert order == ["unload"], "wired memory is only released if we unload"
        assert out["status"]["shutdown_reason"] == f"signal_{int(signal.SIGTERM)}"
    finally:
        signal.signal(signal.SIGTERM, previous[signal.SIGTERM])


def test_health_socket_answers_one_line_of_json():
    srv, _ = _listener()
    daemon = pp.TailDaemon(srv, lambda s, a: 0)
    th, _out = _run(daemon)
    port = _free_port()
    hs = pp.serve_health(daemon, port)
    try:
        c = socket.create_connection(("127.0.0.1", port), timeout=5)
        line = c.recv(4096).decode()
        c.close()
        status = json.loads(line)
        assert status["role"] == "tail"
        assert status["state"] in ("listening", "serving", "starting")
        assert "connections" in status and "uptime_s" in status
    finally:
        daemon.request_shutdown("test")
        th.join(5)
        if hs is not None:
            time.sleep(0.4)
    assert pp.serve_health(daemon, 0) is None


# --------------------------------------------------------------- end to end


class _FakeModel:
    """Three layers, two linear (KDA-shaped) and one attention (DSA-shaped)."""

    def __init__(self):
        layers = [
            SimpleNamespace(
                is_linear=i != 2,
                input_layernorm=SimpleNamespace(weight=mx.ones((2,))),
                self_attn=SimpleNamespace(
                    num_heads=1,
                    head_dim=2,
                    conv_kernel_size=3,
                    kv_lora_rank=2,
                    indexer=SimpleNamespace(head_dim=1),
                ),
            )
            for i in range(3)
        ]
        self.lm = SimpleNamespace(
            layers=layers,
            hc_mult=1,
            pipeline_forward=self.forward,
            pipeline_finish=lambda h: h[:, :, 0, :],
        )
        self.language_model = SimpleNamespace(
            model=self.lm, pipeline_prefill_head=self.head, _logits=lambda h: h
        )

    def make_cache(self):
        return [ArraysCache(2), ArraysCache(2), CacheList(KVCache(), KVCache())]

    def forward(self, h, cache, lo, hi, inputs=None):
        if h is None:
            h = mx.ones((1, inputs.shape[1], 1, 2), dtype=mx.bfloat16)
        for i in range(lo, hi):
            if i != 2:
                cache[i].state = [
                    mx.ones((1, 2, 6), dtype=mx.bfloat16),
                    mx.ones((1, 1, 2, 2)),
                ]
            else:
                n = h.shape[1]
                cache[i][0].update_and_fetch(
                    mx.ones((1, 1, n, 2), dtype=mx.bfloat16),
                    mx.ones((1, 1, n, 2), dtype=mx.bfloat16),
                )
                cache[i][1].update_and_fetch(
                    mx.ones((1, 1, n, 3), dtype=mx.bfloat16),
                    mx.zeros((1, 1, n, 0), dtype=mx.float32),
                )
        return h

    def head(self, inputs, inputs_embeds, cache, split):
        return self.forward(None, cache, 0, split, inputs)


def _tail_args(port):
    return SimpleNamespace(
        model="unused",
        split=1,
        layers=3,
        prune=False,
        bind="127.0.0.1",
        port=port,
        model_sha256="a" * 64,
        source_revision="b" * 40,
        connect_timeout=0.0,
        idle_timeout=0.0,
        io_timeout=5.0,
        depth=2,
        transport="socket",
        once=False,
        health_port=0,
        stop_file=None,
    )


def _head(port):
    from mlx_vlm.pipeline_runtime import PipelineHead, PipelineSettings

    return PipelineHead(
        PipelineSettings(
            ("127.0.0.1", port),
            None,
            "1",
            1,
            "unused",
            "socket",
            model_sha256="a" * 64,
            source_revision="b" * 40,
            io_timeout=5.0,
        ),
        1,
        3,
    )


def test_resident_tail_serves_two_heads_then_shuts_down_on_signal(monkeypatch):
    """Two SEPARATE connections, each with its own bye, against one loaded
    stage -- the whole point of the daemon: the weights are loaded once."""
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
            box["status"] = pp.run_tail(
                args, on_ready=lambda d: box.__setitem__("daemon", d)
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    th = threading.Thread(target=tail, daemon=True)
    th.start()
    deadline = time.monotonic() + 5
    while "daemon" not in box and time.monotonic() < deadline:
        time.sleep(0.01)
    daemon = box["daemon"]
    ids_seen = []
    try:
        for tokens in (4, 6):
            head = _head(port)
            head.connect(timeout=5)
            model = _FakeModel()
            ids = mx.arange(tokens, dtype=mx.int32)[None, :]
            cache = model.make_cache()
            head.begin(tokens, 2, input_ids=ids)
            for start in range(0, tokens - 1, 2):
                part = ids[:, start : min(start + 2, tokens - 1)]
                head.prefill_chunk(model, part, None, cache)
            stats = head.finalize(cache)
            assert cache[2][0].offset == tokens - 1
            ids_seen.append(stats["envelope"]["request_id"])
            head.close()  # bye: retires the CONNECTION
            assert th.is_alive(), "bye must not retire the service"
        assert len(set(ids_seen)) == 2
        assert daemon.counters["connections"] == 2
        assert daemon.counters["requests"] == 2
        assert daemon.counters["connection_errors"] == 0
    finally:
        daemon.request_shutdown("signal_15")
        th.join(5)
    assert not th.is_alive() and not errors
    assert box["status"]["shutdown_reason"] == "signal_15"
    assert box["status"]["state"] == "stopped"


def test_two_requests_ride_one_connection_and_the_cache_is_reset_between(monkeypatch):
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
        model = _FakeModel()
        for tokens in (4, 6):
            ids = mx.arange(tokens, dtype=mx.int32)[None, :]
            cache = model.make_cache()
            cache[2][1]._pool = "stale"
            head.begin(tokens, 2, input_ids=ids)
            for start in range(0, tokens - 1, 2):
                head.prefill_chunk(
                    model, ids[:, start : min(start + 2, tokens - 1)], None, cache
                )
            stats = head.finalize(cache)
            assert sum(stats["chunks"]) == tokens - 1
            assert not hasattr(cache[2][1], "_pool")
        head.close()
        assert box["daemon"].counters["requests"] == 2
        assert box["daemon"].counters["connections"] == 1
    finally:
        head.abort()
        box["daemon"].request_shutdown("test")
        th.join(5)
    assert not th.is_alive() and not errors


def test_shutdown_makes_the_stage_weights_unreachable(monkeypatch):
    """The rule that costs a reboot when it is broken: a tail must release the
    stage BEFORE the process exits.  The session closure holds the Stage and
    the Stage holds the model, so ``unload`` has to clear the Stage's own
    references -- clearing ``run_tail``'s locals is not enough."""
    import gc
    import weakref

    box_model = {"m": _FakeModel()}
    ref = weakref.ref(box_model["m"])
    monkeypatch.setattr(
        pp,
        "load_stage",
        lambda *a: (box_model["m"], box_model["m"].make_cache(), [1, 2], 3, 0.0),
    )
    args = _tail_args(_free_port())
    box = {}
    th = threading.Thread(
        target=lambda: pp.run_tail(
            args, on_ready=lambda d: box.__setitem__("daemon", d)
        ),
        daemon=True,
    )
    th.start()
    deadline = time.monotonic() + 5
    while "daemon" not in box and time.monotonic() < deadline:
        time.sleep(0.01)
    box["daemon"].request_shutdown("signal_15")
    th.join(5)
    assert not th.is_alive()
    box_model.clear()
    gc.collect()
    assert ref() is None, "the stage weights outlived the daemon"
