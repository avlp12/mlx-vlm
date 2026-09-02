"""GPU keepalive: stop macOS un-wiring the model while the server sits idle.

WHY THIS EXISTS (measured on epsilon, M3 Ultra 512 GB, 2026-09-02; receipts in
``glm53flash/logs/sweep4/L3_keepwarm.json``)

macOS releases the model's entire wired residency -- about 173 GB for
GLM-5.3-Flash 4-bit -- within roughly two seconds of GPU idle, and the next
forward has to re-establish it.  That re-wiring costs a flat ~1.19 s of wall on
the first request after the gap:

    idle 0 s -> +6.4 ms      idle 2 s -> +1213 ms      idle 30 s -> +1206 ms
    idle 1 s -> +1122 ms     idle 3 s -> +1230 ms      idle 150 s -> +1189 ms

It is a step at 1-2 s, not a ramp, and it is dead flat out to 150 s.  The cost is
ABSOLUTE, not proportional, so it hurts short prompts worst -- the common chat
case:

    512-token prefill:   +1210 ms = +92.3%   (TTFT nearly doubles)
    4096-token prefill:  +1151 ms = +12.5%

The mechanism was measured directly rather than inferred: ``vm_stat`` around each
idle window shows wired falling by 172.5-177.0 GB and the cold forward putting
172.5-176.4 GB back.  Holding the MLX buffer pool across the gap (skipping
``mx.clear_cache()``) does NOT help -- 1215/2537/2427 ms against a control of
1196/2470/2445 -- so the allocator is not the mechanism.  A DVFS component on top
cannot be excluded; ``powermetrics`` needs sudo, which the box does not have.

THE CURE, AND WHY THE PERIOD IS NOT A FREE PARAMETER

Per-tick timing is what makes this safe to ship.  The summed duty cycle hides the
important fact:

    period   penalty removed   ms per tick after the first   duty cycle
    1 s      99.8%             2-3 ms                        ~0.3% steady state
    2 s      unreliable        2440-2564 ms                  46-57%
    5 s      99.8%             2434-2469 ms                  33%
    10 s     0.8%              2383-2472 ms                  16%

At 1 Hz residency never lapses and a tick is 2-3 ms.  At every period >= 2 s the
residency lapses *between* ticks, so each tick pays a full re-wire and the
keepalive burns a third of the GPU while "idle".  The 5 s arm scores well on the
probe only because it happens to pay that 2.45 s shortly before the probe runs.
So 1 Hz is not a tuning preference -- it is the only period that is both
effective and cheap.  Raising ``MLX_VLM_KEEPALIVE_HZ`` above 1 is harmless;
lowering it below 1 (other than to 0) is the worst of both worlds.

DESIGN NOTES

* The tick runs on the generation thread, which already owns all GPU work, and
  only on the idle branch of ``_collect_pending_requests``.  In-flight
  suppression is therefore structural rather than a flag that can race, and no
  second stream is ever created -- so the campaign's worker-thread rule
  ("a thread doing GPU work must clear its streams before exit") is satisfied by
  the generation thread's existing ``clear_mlx_streams()`` in ``_run``'s finally.
* It is armed only after the model has finished loading, so it can never run
  against a half-initialised model or perturb a load the fleet gate is watching.
* A failing tick disables itself and never propagates: a keepalive must not be
  able to take the server down.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

ENV_KEEPALIVE_HZ = "MLX_VLM_KEEPALIVE_HZ"
DEFAULT_KEEPALIVE_HZ = 1.0

# 256x256 bf16.  Big enough to be a real GPU dispatch, small enough that the
# operands are 256 KB total and a tick is 2-3 ms once residency is held.
_TICK_DIM = 256


def get_keepalive_hz() -> float:
    """Ticks per second from the environment.  0 (or negative) disables."""
    raw = os.environ.get(ENV_KEEPALIVE_HZ)
    if raw is None or raw == "":
        return DEFAULT_KEEPALIVE_HZ
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "%s=%r is not a number; using the default %s Hz",
            ENV_KEEPALIVE_HZ, raw, DEFAULT_KEEPALIVE_HZ,
        )
        return DEFAULT_KEEPALIVE_HZ


class GpuKeepalive:
    """Issues a tiny GPU op at most once per interval while the server is idle.

    Call :meth:`arm` once the model is loaded, :meth:`tick_if_due` from the idle
    path of the generation loop, and :meth:`close` on the way down.
    """

    def __init__(self, hz: Optional[float] = None):
        self.hz = get_keepalive_hz() if hz is None else float(hz)
        self._armed = False
        self._last = 0.0
        self._a = None
        self._b = None
        self._ticks = 0
        self._failed = False

    # ---------------------------------------------------------------- state
    @property
    def enabled(self) -> bool:
        return self.hz > 0 and not self._failed

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def interval_s(self) -> float:
        return 1.0 / self.hz if self.hz > 0 else float("inf")

    @property
    def ticks(self) -> int:
        return self._ticks

    def describe(self) -> str:
        if not self.enabled:
            return f"GPU keepalive disabled ({ENV_KEEPALIVE_HZ}=0)"
        return (
            f"GPU keepalive armed at {self.hz:g} Hz "
            f"({_TICK_DIM}x{_TICK_DIM} bf16 matmul on the generation thread while "
            f"idle) -- holds the model's wired residency, worth ~1.2 s on the first "
            f"request after a gap of 2 s or more"
        )

    def arm(self) -> None:
        """Enable ticking.  Called only after the model has finished loading."""
        self._armed = True
        # Treat arming as a tick: the load itself just touched the GPU, so the
        # first tick is not due for a full interval.
        self._last = time.monotonic()
        logger.info("%s", self.describe())

    def cap_wait(self, timeout: float) -> float:
        """Shorten an idle blocking wait so the tick cadence is actually met.

        The generation loop's idle wait is 0.1 s today, well under a 1 Hz
        interval, but a caller that ever passes a longer timeout would silently
        starve the keepalive.  Capping here keeps the guarantee local.
        """
        if not (self.enabled and self._armed):
            return timeout
        return min(timeout, self.interval_s)

    # ----------------------------------------------------------------- tick
    def _operands(self):
        import mlx.core as mx

        if self._a is None:
            self._a = mx.ones((_TICK_DIM, _TICK_DIM), dtype=mx.bfloat16)
            self._b = mx.ones((_TICK_DIM, _TICK_DIM), dtype=mx.bfloat16)
            mx.eval(self._a, self._b)
        return self._a, self._b

    def _do_tick(self) -> None:
        """The GPU work itself.  Overridden in tests."""
        import mlx.core as mx

        a, b = self._operands()
        mx.eval(mx.matmul(a, b))

    def tick_if_due(self, now: Optional[float] = None) -> bool:
        """Tick if armed, enabled and the interval has elapsed.  Never raises."""
        if not (self.enabled and self._armed):
            return False
        now = time.monotonic() if now is None else now
        if (now - self._last) < self.interval_s:
            return False
        self._last = now
        try:
            self._do_tick()
        except Exception:  # noqa: BLE001 - a keepalive must never kill the server
            self._failed = True
            self._a = self._b = None
            logger.exception(
                "GPU keepalive tick failed; disabling it for the life of this "
                "process. Serving continues, but the first request after an idle "
                "gap will pay the re-wiring cost again."
            )
            return False
        self._ticks += 1
        return True

    def close(self) -> None:
        """Drop the scratch operands and stop ticking."""
        self._armed = False
        self._a = self._b = None
