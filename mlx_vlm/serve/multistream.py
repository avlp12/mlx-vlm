"""Multi-stream decode driver: one thread, N Metal streams, async_eval + lag.

WHAT THIS IS FOR
================
Two products, and the second survives even if the first dies.

1. THROUGHPUT (unproven).  Two verify/decode streams measured 1.257x aggregate --
   but behind ONE barrier per round, which is the pattern playbook law 18 says
   inflates overlap while destroying per-token latency.  Whether any of it
   survives a streaming-compatible shape is what the k-sweep decides.

2. SCHEDULING POLICY (already established, and the real product).  The server
   today alternates on a SINGLE stream: `BatchGenerator.next()` wraps everything
   in `with mx.stream(self._stream)`, and `_next()` runs one decode step OR one
   prefill chunk per call, decode-first (ar.py:3080).  That throttles a decoding
   user to one token per prefill chunk -- about 4.7 s at the default
   PREFILL_STEP_SIZE=2048.  Driving decode and prefill on separate streams with
   async_eval reallocates that wait: measured, decode kept 99.5-99.9% of its solo
   per-token rate while the prefill was the thing that waited.  Same total work,
   same wall clock, opposite latency allocation.  That is a policy choice about
   who waits, and it is shippable regardless of whether aggregate throughput moves.

WHY ONE THREAD
==============
MLX streams are thread-OWNED, not merely thread-local defaults: a stream created
on one thread raises "There is no Stream(gpu, N) in current thread" when used on
another (playbook law 12).  And glm5_next carries per-module memoised ARRAYS that
two threads would corrupt.  So concurrency comes from MLX's laziness -- one
thread CONSTRUCTS on N streams and submits with async_eval -- not from threads.

WHY THE LAG
===========
A blocking mx.eval per step drains the queue and serialises the streams: measured
0.2-1.7% of the available overlap captured, aggregate 1.0014-1.0032x (law 18).
For the device to have something to interleave, more than one forward must be
OUTSTANDING.  `lag` is how many each channel runs ahead before its oldest result
is collected.  lag=1 is submit-one-collect-one, which is the serialising shape;
lag>=2 leaves work in flight on every channel while one is being collected.

lag is not free: it is added latency to first token on each channel, so the k it
takes to win is part of the result, not an implementation detail.

AND FOR REAL DECODE, lag IS PINNED AT 1.  An autoregressive channel cannot run
ahead -- step N+1 needs step N's token, which does not exist until step N is
collected.  Submitting ahead does not fail loudly; it silently reuses a stale
token.  So concurrency for real serving comes from CHANNELS (users), not from
lag: N=2 at lag=1 already leaves channel 1's forward outstanding while channel 0
is being collected, which is the whole mechanism.  lag>1 exists only to
characterise non-autoregressive work and is not a serving configuration.

MEMO KEYING
===========
Three per-module memos are keyed by SHAPE, not stream -- the compiled FFN, the
compiled attention prologue, and the DSA indexer's pool-tail constants.  Two
streams sharing them either crash across streams or silently share a trace.  Each
channel therefore constructs inside `glm5_next.stream_memo_scope(channel.key)`.
This driver never exposes a path that skips it.  (Pool buffers are not affected:
they live on the cache, which is already per-request.)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Tuple

import mlx.core as mx


def _memo_scope(key: str):
    """glm5_next's per-stream memo scope, or a no-op for models without it."""
    try:
        from ..models.glm5_next.language import stream_memo_scope
    except Exception:                                   # pragma: no cover
        import contextlib
        return contextlib.nullcontext()
    return stream_memo_scope(key)


@dataclass
class Channel:
    """One request. ``step`` CONSTRUCTS the next forward and returns the array to
    evaluate -- it must not call mx.eval itself, or the driver's lag is defeated
    and the streams serialise (law 18).

    ``sequential`` (default True) means step N+1 depends on step N's RESULT, which
    is what autoregressive decode is: the next token does not exist until the
    previous forward has been COLLECTED. Such a channel CANNOT run ahead, and
    letting it try produces SILENTLY WRONG RESULTS rather than an error -- measured
    on this driver, a lag=2 autoregressive channel was fed tokens [0, 0, 1, 1]
    where the correct sequence is [0, 1, 2]. The driver refuses it.

    Set sequential=False only for work whose next input is genuinely independent
    of the previous output (a fixed-shape probe, a replayed trace). Real decode
    and real speculative verify are both sequential."""
    name: str
    step: Callable[[], mx.array]
    sequential: bool = True
    stream: Optional[object] = None
    pending: Deque[Tuple[mx.array, float]] = field(default_factory=deque)
    latencies: List[float] = field(default_factory=list)
    completed: int = 0
    #: the most recently COLLECTED (evaluated) output. A sequential channel's
    #: next step reads this -- it is the whole reason such a channel cannot run
    #: ahead, and it is only valid after the collect that produced it.
    last: Optional[mx.array] = None


class MultiStreamDriver:
    """Drive N channels on N streams from ONE thread with a completion lag.

    Streams are created in the calling thread and must stay there.
    """

    def __init__(self, channels: List[Channel], lag: int = 2, device=None):
        if lag < 1:
            raise ValueError("lag must be >= 1")
        bad = [c.name for c in channels if c.sequential and lag > 1]
        if bad:
            raise ValueError(
                f"lag={lag} with sequential channel(s) {bad}: an autoregressive "
                f"channel cannot run ahead -- step N+1 needs step N's RESULT, which "
                f"only exists after collection. Running ahead does not error, it "
                f"silently reuses a stale input (measured: tokens [0,0,1,1] instead "
                f"of [0,1,2]). Use lag=1 and add CHANNELS for concurrency, or mark "
                f"the channel sequential=False if its input really is independent."
            )
        self.lag = int(lag)
        self.channels = channels
        dev = device if device is not None else mx.gpu
        for ch in self.channels:
            # created HERE, in the driver's own thread, on purpose
            ch.stream = mx.new_stream(dev)
        self._owner_thread = _thread_id()

    # -- one round ---------------------------------------------------------
    def _submit(self, ch: Channel) -> None:
        # t0 IS TAKEN BEFORE CONSTRUCTION, on purpose. Recording it after
        # async_eval excludes the Python graph build AND whatever the submit
        # itself completed, so the reported per-token latency is only the residual
        # wait. Measured: that made the driver look 15x FASTER than the plain
        # decode loop (2.56 ms against 37.9 ms), which ARM_0's A2 gate then passed
        # because its rule was one-sided. A latency this API reports must be
        # comparable to `construct + eval` in an ordinary loop, or it is not a
        # latency at all.
        t0 = time.perf_counter()
        with _memo_scope(ch.name):
            with mx.stream(ch.stream):
                out = ch.step()
        mx.async_eval(out)                 # SUBMIT, do not block
        ch.pending.append((out, t0))

    def _collect(self, ch: Channel) -> None:
        out, t0 = ch.pending.popleft()
        mx.eval(out)                       # completes this one
        ch.latencies.append((time.perf_counter() - t0) * 1e3)
        ch.last = out
        ch.completed += 1

    def tick(self) -> None:
        """Top every channel up to `lag` outstanding, then collect one from each.

        Submitting ALL channels before collecting ANY is the whole point: it is
        what leaves work in flight on the other streams while one is drained.
        """
        self._assert_owner()
        for ch in self.channels:
            while len(ch.pending) < self.lag:
                self._submit(ch)
        for ch in self.channels:
            self._collect(ch)

    def run(self, rounds: int) -> None:
        for _ in range(rounds):
            self.tick()

    def drain(self) -> None:
        self._assert_owner()
        for ch in self.channels:
            while ch.pending:
                self._collect(ch)

    # -- reporting ---------------------------------------------------------
    def stats(self) -> dict:
        def pct(v, p):
            if not v:
                return None
            s = sorted(v)
            return round(s[min(len(s) - 1, int(len(s) * p))], 3)
        return {
            "lag": self.lag,
            "channels": {
                ch.name: {
                    "completed": ch.completed,
                    "median_ms": pct(ch.latencies, 0.5),
                    "p95_ms": pct(ch.latencies, 0.95),
                }
                for ch in self.channels
            },
        }

    def reset_stats(self) -> None:
        for ch in self.channels:
            ch.latencies.clear()
            ch.completed = 0

    def _assert_owner(self) -> None:
        if _thread_id() != self._owner_thread:
            raise RuntimeError(
                "MultiStreamDriver was built on another thread. MLX streams are "
                "thread-OWNED: using one off-thread raises 'There is no "
                "Stream(gpu, N) in current thread'. Construct and drive on the "
                "same thread (playbook law 12)."
            )


def _thread_id() -> int:
    import threading
    return threading.get_ident()
