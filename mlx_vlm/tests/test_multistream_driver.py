"""Unit tests for the multi-stream decode driver. No model, no GPU model load.

These pin the PLUMBING and the invariants, not the physics. Whether overlap
actually materialises is a measurement (PREREG_streaming_overlap.json), and a
unit test on a fake model cannot answer it -- a fake step releases the GIL and
would report perfect overlap regardless, which is exactly the trap that made the
first co-scheduling harness report decode at 300-411 tok/s.
"""
import threading

import mlx.core as mx
import pytest

from mlx_vlm.serve.multistream import Channel, MultiStreamDriver


def _fake_step(n=64):
    """Constructs a small real graph. Deliberately does NOT call mx.eval --
    a step that evaluates itself defeats the driver's lag."""
    w = mx.ones((n, n))
    def step():
        return (w @ w).sum()
    return step


def _driver(k=2, n_ch=2, dev=mx.cpu):
    chans = [Channel(name=f"c{i}", step=_fake_step()) for i in range(n_ch)]
    return MultiStreamDriver(chans, lag=k, device=dev)


def test_streams_are_distinct_per_channel():
    d = _driver(n_ch=3)
    ids = {id(ch.stream) for ch in d.channels}
    assert len(ids) == 3, "channels must not share a stream"


def test_lag_is_respected_and_work_stays_outstanding():
    """THE INVARIANT THE DESIGN RESTS ON: after topping up, every channel has
    `lag` forwards in flight. If this drops to 1 the driver has degenerated into
    submit-one-collect-one, which is the serialising shape (law 18)."""
    for k in (1, 2, 4):
        d = _driver(k=k)
        for ch in d.channels:
            while len(ch.pending) < d.lag:
                d._submit(ch)
        assert all(len(ch.pending) == k for ch in d.channels)
        d.drain()
        assert all(len(ch.pending) == 0 for ch in d.channels)


def test_tick_submits_all_before_collecting_any():
    """Ordering is load-bearing: collecting channel 0 before submitting channel 1
    would leave the device with nothing to interleave."""
    d = _driver(k=2)
    order = []
    orig_sub, orig_col = d._submit, d._collect
    d._submit = lambda ch: (order.append(("s", ch.name)), orig_sub(ch))[1]
    d._collect = lambda ch: (order.append(("c", ch.name)), orig_col(ch))[1]
    d.tick()
    kinds = [k for k, _ in order]
    assert kinds.index("c") > max(i for i, k in enumerate(kinds) if k == "s"), \
        f"a collect happened before the last submit: {order}"


def test_completions_and_latencies_recorded():
    d = _driver(k=2)
    d.run(rounds=5)
    for ch in d.channels:
        assert ch.completed == 5
        assert len(ch.latencies) == 5
    s = d.stats()
    assert s["lag"] == 2
    for name, v in s["channels"].items():
        assert v["completed"] == 5
        assert v["median_ms"] is not None and v["p95_ms"] is not None


def test_reset_stats():
    d = _driver()
    d.run(rounds=3); d.drain(); d.reset_stats()
    assert all(ch.completed == 0 and not ch.latencies for ch in d.channels)


def test_lag_must_be_at_least_one():
    with pytest.raises(ValueError):
        _driver(k=0)


def test_refuses_to_be_driven_from_another_thread():
    """Streams are thread-OWNED. Driving from a second thread would raise deep
    inside MLX with a confusing message; refuse up front with the reason."""
    d = _driver()
    err = {}
    def worker():
        try:
            d.tick()
        except RuntimeError as e:
            err["e"] = str(e)
    t = threading.Thread(target=worker); t.start(); t.join(10)
    assert "thread-OWNED" in err.get("e", ""), err


def test_memo_scope_is_entered_per_channel():
    """Each channel must construct under its own memo key, or two streams share
    the shape-keyed compiled caches."""
    import mlx_vlm.models.glm5_next.language as glm5
    seen = []
    chans = [Channel(name=f"k{i}",
                     step=(lambda: (seen.append(glm5.stream_memo_key()),
                                    mx.ones((4,)).sum())[1]))
             for i in range(2)]
    d = MultiStreamDriver(chans, lag=1, device=mx.cpu)
    d.tick()
    assert set(seen) == {"k0", "k1"}, seen
    assert glm5.stream_memo_key() == "default", "scope leaked past the driver"
