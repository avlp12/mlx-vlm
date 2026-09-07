"""The tail is a service, not a one-shot bench process.

The shipped tail served exactly one connection and then unloaded tens of GB of
weights, so every request paid the load again.  These tests pin the service
behaviour that replaces it: accept is re-armed after a connection ends, a
``bye`` retires the connection and not the daemon, a peer that fails takes only
its own connection down, a deadline can only kill a tail that has never been
used, and SIGTERM unloads BEFORE the process exits (a bare kill of a process
holding MLX buffers leaks wired memory that only a reboot reclaims).

A2b adds the half of that contract the B3 fallback drills found missing
(2026-09-07 15:25-16:02): SIGTERM only ever stopped a tail that happened to be
between connections, and a pooled head means a tail is never between
connections, so a signalled tail kept serving for eight minutes with
``shutdown_reason: "signal_15"`` on its own health line and its flock in its
hand.  The tests below pin the stop in EVERY state -- idle, idle-on-a-pooled-
connection, mid-request (drain), past the drain, and on a second signal -- and
pin that every one of those paths unloads exactly once.
"""

import json
import os
import signal
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


def _capture(daemon):
    """Every health line the daemon emits, parsed, in order."""
    lines = []
    daemon.log = lines.append
    return lines


def _statuses(lines):
    return [
        json.loads(ln.split(" ", 1)[1])
        for ln in lines
        if ln.startswith("[tail-health] ")
    ]


def _assert_never_listening_after_a_stop(lines):
    """THE regression shape: ``shutdown_reason`` set and ``state: listening``.

    That is what the operator read off the epsilon tail at 15:26 and again at
    15:35 while it went on serving PP requests; it must be unreachable now.
    """
    bad = [
        st
        for st in _statuses(lines)
        if st.get("shutdown_reason") and st.get("state") == "listening"
    ]
    assert bad == [], f"a stopping tail reported itself as listening: {bad}"


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


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


# ---------------------------------------------------- A2b: the stop contract


def test_the_gate_is_the_whole_semantics_table():
    """state x signal -> verdict, without a socket in sight."""
    now = [100.0]
    gate = pp.ShutdownGate(drain_timeout=30.0, clock=lambda: now[0])
    assert gate.stopping is False and gate.remaining() is None
    gate.check()  # nothing asked, nothing raised

    # idle: the first signal ends the wait immediately
    assert gate.request("signal_15") == "soft"
    assert (gate.stopping, gate.hard, gate.reason) == (True, False, "signal_15")
    with pytest.raises(InterruptedError, match="idle connection"):
        gate.check()

    # in a request: the drain window is honoured to the second
    gate.enter_request()
    gate.check()
    now[0] += 29.9
    gate.check()
    assert gate.remaining() == pytest.approx(0.1)
    now[0] += 0.1
    with pytest.raises(InterruptedError, match="drain timeout"):
        gate.check()
    assert gate.snapshot()["drain_remaining_s"] == 0.0

    # a second ask is the hard stop, whatever the drain had left
    fresh = pp.ShutdownGate(drain_timeout=30.0, clock=lambda: now[0])
    fresh.enter_request()
    assert fresh.request("signal_15") == "soft"
    fresh.check()
    assert fresh.request("signal_15") == "hard"
    with pytest.raises(InterruptedError, match="second signal"):
        fresh.check()
    assert fresh.snapshot()["hard_stop"] is True

    # and the stop FILE keeps its own meaning, unchanged, in every state
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        stop = os.path.join(d, "STOP")
        filed = pp.ShutdownGate(stop, drain_timeout=30.0)
        filed.enter_request()
        filed.check()
        open(stop, "w").close()
        with pytest.raises(InterruptedError, match="STOP"):
            filed.check()


def test_an_idle_sigterm_exits_within_a_second():
    """The real signal, really delivered -- the drill's D3/D4 precondition."""
    srv, _ = _listener()
    unloaded = []
    daemon = pp.TailDaemon(srv, lambda s, a: 0, unload=lambda: unloaded.append(1))
    lines = _capture(daemon)
    previous = pp.install_tail_signal_handlers(daemon, signums=(signal.SIGTERM,))
    try:
        th, out = _run(daemon)
        assert _wait(lambda: daemon.state == "listening")
        t0 = time.monotonic()
        os.kill(os.getpid(), signal.SIGTERM)
        th.join(5)
        elapsed = time.monotonic() - t0
    finally:
        signal.signal(signal.SIGTERM, previous[signal.SIGTERM])
    assert not th.is_alive()
    assert elapsed < 2.0, f"an idle tail took {elapsed:.2f}s to notice SIGTERM"
    assert unloaded == [1]
    assert out["status"]["state"] == "stopped"
    assert out["status"]["shutdown_reason"] == f"signal_{int(signal.SIGTERM)}"
    assert out["status"]["stopping"] is True and out["status"]["hard_stop"] is False
    _assert_never_listening_after_a_stop(lines)


def test_a_pooled_connection_does_not_outlive_a_stop():
    """The field failure itself: the tail is not between connections when the
    signal lands, because a pooled head keeps ONE connection across requests
    (pipeline_runtime.PipelinePool).  Before A2b the flag was only ever read
    between connections, so this daemon served on -- here it must not."""
    srv, port = _listener()
    gate = pp.ShutdownGate(drain_timeout=5.0)
    served = []
    unloaded = []

    def pooled_session(sock, addr):
        # request/reply forever, exactly like the shipped session loop
        peer = pp.StopAwareSocket(sock, gate, 20.0)
        n = 0
        while True:
            pp._check_stop(gate)  # idle between requests
            buf = bytearray(4)
            pp._recv_exact(peer, memoryview(buf), 4)
            gate.enter_request()
            try:
                peer.sendall(b"ok..")
                n += 1
                served.append(bytes(buf))
            finally:
                gate.leave_request()
            if gate.stopping:
                return n

    daemon = pp.TailDaemon(
        srv, pooled_session, gate=gate, unload=lambda: unloaded.append(1)
    )
    lines = _capture(daemon)
    th, out = _run(daemon)
    c = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        for i in range(2):
            c.sendall(f"rq{i:02d}".encode())
            assert c.recv(4) == b"ok.."
        assert _wait(lambda: daemon.state == "serving")
        t0 = time.monotonic()
        daemon.request_shutdown("signal_15")
        th.join(5)
        elapsed = time.monotonic() - t0
    finally:
        c.close()
    assert not th.is_alive(), "a signalled tail with a pooled head kept serving"
    assert elapsed < 2.0
    assert served == [b"rq00", b"rq01"], "no request may be served after the stop"
    assert unloaded == [1]
    assert out["status"]["shutdown_reason"] == "signal_15"
    assert out["status"]["state"] == "stopped"
    _assert_never_listening_after_a_stop(lines)
    # and the port is gone: the head's next attempt is peer_unreachable
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=2).close()


def test_a_stop_mid_request_drains_the_request_then_exits():
    """Graceful semantics: the request that was in flight is FINISHED and
    answered, and only then does the connection retire.  This is why the B3
    D2 expectation has to change -- a drained request is a correct request."""
    srv, port = _listener()
    gate = pp.ShutdownGate(drain_timeout=10.0)
    finished = []
    unloaded = []

    def session(sock, addr):
        peer = pp.StopAwareSocket(sock, gate, 20.0)
        buf = bytearray(2)
        pp._recv_exact(peer, memoryview(buf), 2)
        gate.enter_request()
        try:
            time.sleep(0.3)  # the request's own work, after the signal lands
            gate.check()  # a drain point inside the request: still inside 10 s
            peer.sendall(b"done")
            finished.append(1)
            return 1
        finally:
            gate.leave_request()

    daemon = pp.TailDaemon(srv, session, gate=gate, unload=lambda: unloaded.append(1))
    lines = _capture(daemon)
    th, out = _run(daemon)
    c = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        c.sendall(b"go")
        assert _wait(lambda: daemon.state == "serving")
        daemon.request_shutdown("signal_15")
        assert c.recv(4) == b"done", "a drained request must still be answered"
    finally:
        c.close()
    th.join(5)
    assert not th.is_alive()
    assert finished == [1] and daemon.counters["requests"] == 1
    assert daemon.counters["connection_errors"] == 0
    assert unloaded == [1]
    _assert_never_listening_after_a_stop(lines)


def test_a_request_that_outlasts_the_drain_is_aborted():
    """A head that has gone quiet mid-request does not get to hold the box: at
    the drain deadline the connection is cut, which is exactly the
    ``pp_failed`` -> single-box fallback the head documents."""
    srv, port = _listener()
    gate = pp.ShutdownGate(drain_timeout=0.3)
    unloaded = []

    def session(sock, addr):
        peer = pp.StopAwareSocket(sock, gate, 30.0)
        gate.enter_request()
        try:
            pp._recv_json(peer)  # the head never speaks again
            return 1
        finally:
            gate.leave_request()

    daemon = pp.TailDaemon(srv, session, gate=gate, unload=lambda: unloaded.append(1))
    lines = _capture(daemon)
    th, out = _run(daemon)
    c = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        assert _wait(lambda: daemon.state == "serving")
        t0 = time.monotonic()
        daemon.request_shutdown("signal_15")
        th.join(5)
        elapsed = time.monotonic() - t0
        # the head sees the connection go, not a hang
        c.settimeout(2.0)
        assert c.recv(16) == b""
    finally:
        c.close()
    assert not th.is_alive()
    assert 0.25 <= elapsed < 2.0, f"aborted after {elapsed:.2f}s, drain was 0.3s"
    assert "drain timeout" in daemon.counters["last_error"]
    assert daemon.counters["requests"] == 0
    assert unloaded == [1]
    _assert_never_listening_after_a_stop(lines)


def test_a_second_signal_aborts_the_request_in_flight_at_once():
    srv, port = _listener()
    gate = pp.ShutdownGate(drain_timeout=600.0)  # a drain nobody would wait out
    unloaded = []

    def session(sock, addr):
        peer = pp.StopAwareSocket(sock, gate, 30.0)
        gate.enter_request()
        try:
            pp._recv_json(peer)
            return 1
        finally:
            gate.leave_request()

    daemon = pp.TailDaemon(srv, session, gate=gate, unload=lambda: unloaded.append(1))
    lines = _capture(daemon)
    th, out = _run(daemon)
    c = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        assert _wait(lambda: daemon.state == "serving")
        assert daemon.request_shutdown("signal_15") == "soft"
        time.sleep(0.3)
        assert th.is_alive(), "the first signal drains, it does not cut"
        t0 = time.monotonic()
        assert daemon.request_shutdown("signal_15") == "hard"
        th.join(5)
        elapsed = time.monotonic() - t0
    finally:
        c.close()
    assert not th.is_alive() and elapsed < 2.0
    assert "second signal" in daemon.counters["last_error"]
    assert out["status"]["hard_stop"] is True
    assert unloaded == [1], "even a hard stop unloads: SIGKILL is never our verb"
    assert any("hard stop" in ln for ln in lines)
    _assert_never_listening_after_a_stop(lines)


def test_the_shipped_session_retires_a_pooled_connection_on_a_stop(monkeypatch):
    """End to end, through ``run_tail``/``tail_session_factory``/the real head:
    a pooled connection is left OPEN and idle after a request -- which is where
    the tail actually lives in production -- and a stop must still end the
    daemon inside a second.  This is the drill's D3/D4 precondition: the tail
    has to be GONE for the head to observe ``peer_unreachable``."""
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
    assert _wait(lambda: "daemon" in box)
    daemon = box["daemon"]
    lines = _capture(daemon)
    head = _head(port)
    head.connect(timeout=5)
    model = _FakeModel()
    ids = mx.arange(4, dtype=mx.int32)[None, :]
    cache = model.make_cache()
    head.begin(4, 2, input_ids=ids)
    for start in range(0, 3, 2):
        head.prefill_chunk(model, ids[:, start : min(start + 2, 3)], None, cache)
    head.finalize(cache)
    # the pool's behaviour: the request is over, the CONNECTION is not
    assert head.sock is not None
    assert daemon.state == "serving", (
        "the daemon must be inside the session, not at accept -- that is the "
        "state the signal was never observed in"
    )
    t0 = time.monotonic()
    daemon.request_shutdown("signal_15")
    th.join(5)
    elapsed = time.monotonic() - t0
    head.abort()
    assert not th.is_alive() and not errors
    assert elapsed < 2.0, f"a pooled tail took {elapsed:.2f}s to stop"
    assert box["status"]["state"] == "stopped"
    assert box["status"]["shutdown_reason"] == "signal_15"
    assert daemon.counters["connection_errors"] == 0, (
        "our own shutdown ending an idle connection is not the peer failing"
    )
    assert daemon.counters["requests"] == 1, (
        "a connection retired by the shutdown still reports what it served"
    )
    _assert_never_listening_after_a_stop(lines)


def test_the_health_thread_retires_with_the_daemon():
    srv, _ = _listener()
    daemon = pp.TailDaemon(srv, lambda s, a: 0)
    th, _out = _run(daemon)
    port = _free_port()
    health = pp.serve_health(daemon, port)
    assert health is not None and health.thread.is_alive()
    c = socket.create_connection(("127.0.0.1", port), timeout=5)
    assert json.loads(c.recv(4096).decode())["stopping"] is False
    c.close()
    daemon.request_shutdown("test")
    th.join(5)
    assert not th.is_alive()
    assert health.join(3.0), "the health thread outlived the daemon"
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=2).close()


class _Report:
    """An AdmissionReport double: release is idempotent, like the real one."""

    def __init__(self):
        self.released = 0

    def to_dict(self):
        return {"enabled": True, "admitted": True}

    def release(self):
        if not self.released:
            self.released = 1


def _admitted_tail(monkeypatch, report):
    monkeypatch.setattr(pp, "_admission_enabled", lambda args: True)
    monkeypatch.setattr(pp.admission, "admit", lambda args: report)


def test_the_lock_is_released_on_every_exit_path(monkeypatch):
    """A tail that stopped serving but still holds the flock locks the box out
    of its own restart -- the other half of what the B3 drills hit (every
    replacement tail: ``DIED_ON_START`` = admission flock refusal)."""
    model = _FakeModel()
    monkeypatch.setattr(
        pp, "load_stage", lambda *a: (model, model.make_cache(), [1, 2], 3, 0.0)
    )
    # 1. the ordinary path: stopped, unloaded, released
    report = _Report()
    _admitted_tail(monkeypatch, report)
    args = _tail_args(_free_port())
    box = {}
    th = threading.Thread(
        target=lambda: pp.run_tail(args, on_ready=lambda d: box.__setitem__("d", d)),
        daemon=True,
    )
    th.start()
    assert _wait(lambda: "d" in box)
    box["d"].request_shutdown("signal_15")
    th.join(5)
    assert not th.is_alive() and report.released == 1

    # 2. a refusal before any weights: released by run_tail's own guard
    refused = _Report()
    monkeypatch.setattr(pp, "load_stage", _boom)
    _admitted_tail(monkeypatch, refused)
    with pytest.raises(RuntimeError, match="no shard"):
        pp.run_tail(_tail_args(_free_port()))
    assert refused.released == 1

    # 3. an unload that itself fails must not swallow the release
    monkeypatch.setattr(
        pp, "load_stage", lambda *a: (model, model.make_cache(), [1, 2], 3, 0.0)
    )
    broken = _Report()
    _admitted_tail(monkeypatch, broken)
    monkeypatch.setattr(pp.mx, "clear_cache", _boom)
    args = _tail_args(_free_port())
    box = {}
    errors = []

    def go():
        try:
            pp.run_tail(args, on_ready=lambda d: box.__setitem__("d", d))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    th = threading.Thread(target=go, daemon=True)
    th.start()
    assert _wait(lambda: "d" in box)
    box["d"].request_shutdown("signal_15")
    th.join(5)
    assert not th.is_alive()
    assert errors and "no shard" in str(errors[0])
    assert broken.released == 1, "the lock outlived a failed unload"


def _boom(*a, **k):
    raise RuntimeError("no shard")


def test_an_aborted_request_still_makes_the_stage_unreachable(monkeypatch):
    """The weakref proof, on the abort path this time: a drain that expires
    must release the weights as thoroughly as a clean bye does."""
    import gc
    import weakref

    box_model = {"m": _FakeModel()}
    ref = weakref.ref(box_model["m"])
    monkeypatch.setattr(
        pp,
        "load_stage",
        lambda *a: (box_model["m"], box_model["m"].make_cache(), [1, 2], 3, 0.0),
    )
    port = _free_port()
    args = _tail_args(port)
    args.drain_timeout = 0.2
    box = {}
    th = threading.Thread(
        target=lambda: pp.run_tail(args, on_ready=lambda d: box.__setitem__("d", d)),
        daemon=True,
    )
    th.start()
    assert _wait(lambda: "d" in box)
    daemon = box["d"]
    # a head that connects, says hello, and then stops talking mid-request
    head = _head(port)
    head.connect(timeout=5)
    model = _FakeModel()
    ids = mx.arange(4, dtype=mx.int32)[None, :]
    cache = model.make_cache()
    head.begin(4, 2, input_ids=ids)
    head.prefill_chunk(model, ids[:, 0:2], None, cache)
    assert _wait(lambda: daemon.state == "serving")
    daemon.request_shutdown("signal_15")
    th.join(5)
    assert not th.is_alive()
    assert "drain timeout" in (daemon.counters["last_error"] or ""), (
        "this request was meant to die of the drain deadline"
    )
    head.abort()
    box_model.clear()
    gc.collect()
    assert ref() is None, "an aborted request left the stage weights resident"
