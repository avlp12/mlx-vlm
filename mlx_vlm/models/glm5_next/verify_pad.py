"""Pad the speculative verify block's dense projections past MLX's qmv/qmm crossover.

WHAT THIS IS
------------
MLX v0.32.x dispatches ``mx.quantized_matmul`` on the flattened row count ``M``::

    mlx/backend/metal/quantized.cpp:1804   vector_limit = transpose ? get_qmv_batch_limit(K, N, d) : 4
    mlx/backend/metal/quantized.cpp:1806   if (M >= vector_limit) -> qmm_splitk / qmm      (weight-reusing)
    mlx/backend/metal/quantized.cpp:1754   else if (K == 64 || K == 128) -> qmv_quad
    mlx/backend/metal/quantized.cpp:1761   else if (M >= 2) -> qmv_wide
    mlx/backend/metal/quantized.cpp:1765   else -> qmv / qmv_fast

Below ``vector_limit`` the time is LINEAR in M -- each of the M rows walks the
whole weight matrix (``kernels/quantized.h:791``: the weight pointer advances on
``out_row``, never on ``tid.x``, while ``:794-795`` advance x and y by ``tid.x``).
At ``vector_limit`` the time drops DISCONTINUOUSLY, because ``qmm`` reads each
weight tile once for all rows.  On ``applegpu_g15d``
(``get_qmv_batch_limit``, ``quantized.cpp:85-122``, ``arch_gen 15``, ``arch_size 'd'``)
that limit is 32 / 18 / 12 by shape, and the big projections of this checkpoint
all land on 12.

A DFlash2 verify block presents exactly ``M = S = block_total`` rows
(``speculative/dflash.py:842-846``: ``concat([[bonus]], draft_tokens)``, no pad,
no mask), which for the shipped fixed width is 8, and 9..11 for the widths just
above it.  Those are precisely the M values that sit in the linear regime one
step short of the crossover.  So: pad the activation to ``vector_limit`` rows
with zeros, run the matmul, slice the first M rows back off.  Three dummy rows
of arithmetic buy the weight-reusing kernel.

MEASUREMENT THIS IMPLEMENTS (sweep11 P-L1, epsilon, mlx 0.32.1, no model loaded;
``logs/sweep11/PL1_qmv_small_m_reread_RESULT.json``).  ``t(12) < t(11)`` at 9 of 9
shape/width curves in BOTH sweep orders (ratio 0.455-0.840) even though M=12 does
9 % more work, and ``t(12) < t(9)`` at 9 of 9 (0.501-0.919).  Reconstructing the
WHOLE dense projection set of GLM-5.3-Flash at its real shapes, real shipped bit
widths and real layer counts (259 matmuls, 7.617 GB of quantized weights)::

    M            1       2       4       8       9      12      16
    chain ms  10.945  11.191  12.877  23.843  31.665  24.209  25.641

so at M = 9 padding to 12 saves 7.456 ms = 9.7 % of the 76.585 ms verify step.

WIDTH SENSITIVITY, AND WHY THERE ARE TWO THRESHOLDS
---------------------------------------------------
At M = 8 the WHOLE chain padded to 12 LOSES 0.37 ms (23.843 -> 24.209): M=8 sits
on the true crossover, and the 4-bit and 6-bit modules lose what the 8-bit ones
gain.  Per module at M=8 (ms, isolated, [qmv at M=8, qmm at M=12]):

    kda_in_proj_fused b8  [0.4155, 0.3172]   -24 %
    kda_o_proj        b8  [0.1614, 0.1182]   -27 %
    kda_qkv_single    b8  [0.1671, 0.1122]   -33 %
    kda_in_proj_fused b4  [0.2697, 0.3003]   +11 %
    lily K4096 N8192  b4  [0.0955, 0.1061]   +11 %

The 8-bit shapes cross EARLIER than MLX's tuned constant, the sub-8-bit ones do
not.  Mechanism: at 8 bits the M=1 baseline is already byte-bound, so the
re-read costs proportionally more.  Hence a per-bit-width floor: pad from M >= 8
at 8 bits and wider, from M >= 9 below.  Both are env-tunable, because both are
two-point microbench extrapolations, not a fitted curve.

    MLX_VLM_VERIFY_PAD_M        pad target, DEFAULT 0 = OFF (see below; it was
                                12 until R-PL1b measured it losing).
    MLX_VLM_VERIFY_PAD_MIN      lowest M padded at bits < 8, default 9.
    MLX_VLM_VERIFY_PAD_MIN_Q8   lowest M padded at bits >= 8, default 8.

A call site is padded only when ``get_qmv_batch_limit(K, N) <= PAD_M``, i.e. only
when the pad actually crosses that shape's own route boundary.  At the default 12
that admits the 12-limit shapes (kda in/o proj, mla o/q_b, dense MLP, lm_head)
and declines the 18-limit ones (mla q_a/kv_a, indexer, shared expert), which the
probe never measured.  Set PAD_M=18 to admit them -- UNMEASURED as of this
writing.

MEASURED ON THE SERVED PATH, AND DECLINED: DEFAULT OFF
------------------------------------------------------
R-PL1b ran this on epsilon against the production server loop, one load, ABAB
x 3 cycles, W8 and W9, code and prose at 1024 (receipt
``logs/sweep11/RPL1b_p1024.json``, verdict ``RPL1b_VERDICT.md``).  Per-round wall
clock got WORSE in every one of the four pairs:

    arm         round ms, pad off -> pad on
    code W8       80.90 -> 91.40   +13.0 %
    code W9       89.96 -> 97.11   + 7.9 %
    prose W8      80.51 -> 89.32   +10.9 %
    prose W9      90.52 -> 99.62   +10.1 %

(The tok/s of the code W8 pair went UP 2.3 %, but only because the padded arm
decoded a different, easier stream and its accepted-per-round rose 3.83 -> 4.45.
Round time is the honest quantity here, and it is uniformly worse.)

WHY, measured directly (``logs/sweep11/PL1c_dependency_RESULT.json``,
``PL1d_ladder_RESULT.json``, same shapes, no model loaded): P-L1 issued its 259
matmuls with NO data dependencies, so the GPU could overlap them.  A real forward
cannot -- layer i+1's in-projection needs layer i's output -- and ``qmm_splitk``
is a TWO-kernel algorithm (partial products, then a reduction) whose second
kernel's latency hides completely under overlap and not at all in a serial chain.
Rerunning the M ladder both ways, on the same weights, in both sweep orders:

    kda_in_proj_fused, 34 layers, ms   M=1     M=8     M=9    M=11    M=12
      issued independently             5.73   10.92   13.57   18.75   10.12
      issued as a dependent chain      6.00    8.82    9.90   12.28   13.06

    kda_o_proj, 34 layers, ms          M=1     M=8     M=9    M=11    M=12
      issued independently             2.21    4.96    5.40    6.88    4.20
      issued as a dependent chain      2.63    3.64    4.11    4.93    5.37

Independently issued, t(12)/t(11) is 0.54 and 0.61 -- the discontinuity the lever
was built on.  In a dependent chain it is 1.06 and 1.09: THERE IS NO
DISCONTINUITY, the curve rises monotonically through the crossover and then goes
flat.  The pad and slice OPS are not the problem; they cost 1.5 us per call
(M8pad12 minus M12raw = 0.05 ms over 34 calls).  The problem is that ``qmm`` is
the slower kernel at these shapes once the calls are serialized.

The same probe halves the re-read penalty the skinny-kernel program is priced on:
k(M=9)/k(M=1) is 2.37 and 2.44 issued independently but only 1.65 and 1.56 in a
dependent chain.

So ``MLX_VLM_VERIFY_PAD_M`` DEFAULTS TO 0.  The code stays because the knob is
how the above was measured, because the module is the only place the
``get_qmv_batch_limit`` table is written down in Python, and because the same
scaffolding is what a later, better lever (a real weight-stationary kernel, or a
batched ``qmm`` that avoids the split-K reduction) would be A/B'd against.

NOT BIT-IDENTICAL, BY CONSTRUCTION
----------------------------------
``qmm_splitk`` and ``qmv_wide`` are different kernels with different fp32
reduction partitions.  The padded path is the same mathematics summed in a
different order; it is NOT bit-identical to the unpadded one and does not claim
to be.  R-PL1e measured it on epsilon with a determinism control that is exactly
max|delta| == 0.0 (``logs/sweep11/RPL1e_identity_RESULT.json``), 8 blocks at each
of S=8 and S=9, against BOTH references:

                                    S=8              S=9
    determinism, off vs off      0.0    64/64     0.0    72/72
    pad on vs pad off            2.44   64/64     3.50   72/72
    pad off vs PER-TOKEN greedy  2.69   64/64     2.13   70/72
    pad on  vs PER-TOKEN greedy  2.78   64/64     2.75   70/72
    (max|delta| on a logit scale of 27.6 / 28.9, then argmax agreement)

The pad flips ZERO argmaxes in 136 positions against the unpadded block, and the
two flips against per-token greedy at S=9 are ALREADY THERE in the shipped
unpadded path -- same block, same position, with and without the pad.  The
departure the pad adds is smaller than the departure S >= 2 already costs.

SCOPE
-----
Armed ONLY inside :func:`verify_window`, which ``LanguageModel.__call__`` opens
when the caller passes ``speculative_verify=True``.  S = 1 decode is untouched
(M=1 is below every floor anyway) and so is prefill -- a ragged prefill chunk can
land on M = 9..11 too, and padding it would break the receipted bit-identity of
the chunked speculative prefill (``logs/sweep11/R30_VERDICT.md`` K6) for a
saving nobody asked for.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, NamedTuple, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

__all__ = [
    "PAD_M_ENV",
    "PAD_MIN_ENV",
    "PAD_MIN_Q8_ENV",
    "config",
    "describe",
    "log_config_once",
    "counters",
    "observed_widths",
    "project",
    "quantized_matmul",
    "qmv_batch_limit",
    "reset_for_tests",
    "verify_window",
    "window_is_open",
]

PAD_M_ENV = "MLX_VLM_VERIFY_PAD_M"
PAD_MIN_ENV = "MLX_VLM_VERIFY_PAD_MIN"
PAD_MIN_Q8_ENV = "MLX_VLM_VERIFY_PAD_MIN_Q8"

DEFAULT_PAD_M = 0  # OFF -- R-PL1b measured the lever losing 8-13 % of round time
DEFAULT_PAD_MIN = 9
DEFAULT_PAD_MIN_Q8 = 8


class PadConfig(NamedTuple):
    pad_m: int
    pad_min: int
    pad_min_q8: int

    @property
    def enabled(self) -> bool:
        return self.pad_m > 0


_CONFIG: Optional[PadConfig] = None
_LOGGED = False
# M values actually presented to the padder, so the log can answer "which width
# did the box really run" without a second harness (law 23: the default is
# verified by the server's own log, not by the test that sets it).
_WIDTHS: dict = {}
# How many projection calls the armed window actually padded, versus declined.
# Only touched inside a verify window, so decode and prefill pay nothing.
_COUNTS: dict = {"padded": 0, "declined": 0}

_ACTIVE: ContextVar = ContextVar("glm5_next_verify_pad_active", default=False)


def _read_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


def config() -> PadConfig:
    """Resolve the three knobs ONCE per process."""
    global _CONFIG
    if _CONFIG is None:
        pad_m = _read_int(PAD_M_ENV, DEFAULT_PAD_M)
        if pad_m < 0:
            pad_m = 0
        _CONFIG = PadConfig(
            pad_m=pad_m,
            pad_min=max(2, _read_int(PAD_MIN_ENV, DEFAULT_PAD_MIN)),
            pad_min_q8=max(2, _read_int(PAD_MIN_Q8_ENV, DEFAULT_PAD_MIN_Q8)),
        )
    return _CONFIG


def reset_for_tests() -> None:
    """Drop the resolved config and the width histogram.  Tests only."""
    global _CONFIG, _LOGGED
    _CONFIG = None
    _LOGGED = False
    _WIDTHS.clear()
    _COUNTS["padded"] = 0
    _COUNTS["declined"] = 0
    _arch_table.cache_clear()
    _plan_static.cache_clear()


def describe() -> str:
    cfg = config()
    state = "ON" if cfg.enabled else "OFF"
    return (
        f"glm5_next verify-block dense pad {state}: "
        f"{PAD_M_ENV}={cfg.pad_m} {PAD_MIN_ENV}={cfg.pad_min} "
        f"{PAD_MIN_Q8_ENV}={cfg.pad_min_q8} arch={_architecture()!r}"
    )


def log_config_once() -> None:
    """Emit the resolved knobs to the SERVER log, exactly once."""
    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        logger.info("%s", describe())


def observed_widths() -> dict:
    """{M: count} over every verify forward this process declared."""
    return dict(_WIDTHS)


def counters() -> dict:
    """{'padded': n, 'declined': n} over projection calls inside verify windows."""
    return dict(_COUNTS)


# ---------------------------------------------------------------------------
# get_qmv_batch_limit, transcribed from mlx v0.32.1
# mlx/backend/metal/quantized.cpp:85-122.  arch_gen is the two digits before the
# trailing size letter of the architecture string (device.cpp:593-601), so
# "applegpu_g15d" -> gen 15, size 'd'.
# ---------------------------------------------------------------------------


def _architecture() -> str:
    """The GPU architecture string, e.g. ``applegpu_g15d``.

    Explicitly the GPU device: ``mx.device_info()`` reports the DEFAULT device,
    and the desk correctness gate pins that to CPU (where it answers ``arm64``
    and would silently select the wrong ``get_qmv_batch_limit`` branch).
    """
    info = None
    getter = getattr(mx, "device_info", None)
    if getter is not None:
        try:
            info = getter(mx.gpu)
        except Exception:
            info = None
    if info is None:
        try:
            info = mx.metal.device_info()
        except Exception:  # pragma: no cover - no Metal device at all
            return ""
    return str(info.get("architecture", ""))


@lru_cache(maxsize=None)
def _arch_table() -> Tuple[int, str]:
    arch = _architecture()
    if len(arch) >= 3:
        tens = ord(arch[-3]) - ord("0")
        ones = ord(arch[-2]) - ord("0")
        tens = tens if 0 <= tens < 10 else 0
        ones = ones if 0 <= ones < 10 else 0
        return tens * 10 + ones, arch[-1]
    # No Metal device (the desk CPU gate runs here).  This tree targets M3
    # Ultra; assume its table so the pad path is exercised, and say so.
    return 15, "d"


def qmv_batch_limit(K: int, N: int) -> int:
    """``M >= this`` takes the weight-reusing qmm route.  D = K, O = N."""
    arch_gen, arch_size = _arch_table()
    if arch_gen >= 17 and arch_size != "d":
        if K <= 2048 and N <= 2048:
            return 33
        if K <= 4096 and N <= 4096:
            return 25
        return 13
    if arch_gen >= 15 and arch_size != "d":
        if K <= 2048 and N <= 2048:
            return 13
        if K <= 4096 and N <= 4096:
            return 15
        return 13
    if arch_size == "d":
        if K <= 2048 and N <= 2048:
            return 32
        if K <= 4096 and N <= 4096:
            return 18
        return 12
    if arch_gen >= 13:
        if K <= 2048 and N <= 2048:
            return 14
        if K <= 4096 and N <= 4096:
            return 10
        return 6
    if K <= 2048 and N <= 2048:
        return 16
    if K <= 4096 and N <= 4096:
        return 8
    return 6


@lru_cache(maxsize=1024)
def _plan_static(K: int, N: int, bits: int, cfg: PadConfig) -> int:
    """Pad target for this shape, or 0 for "do not pad".  Memoized on shape."""
    if not cfg.enabled:
        return 0
    if K in (64, 128):
        # quantized.cpp:1754 routes these to qmv_quad, whose grid is already
        # (M, ., B) -- the pad would only buy more work.
        return 0
    limit = qmv_batch_limit(K, N)
    if limit > cfg.pad_m:
        # Padding to PAD_M would not cross this shape's route boundary.
        return 0
    target = max(cfg.pad_m, limit)
    return target


def _floor(bits: int, cfg: PadConfig) -> int:
    return cfg.pad_min_q8 if bits >= 8 else cfg.pad_min


def _plan(M: int, K: int, N: int, bits: int) -> int:
    """Pad target, or 0.  ``M`` is the FLATTENED row count MLX will dispatch on."""
    if not _ACTIVE.get():
        return 0
    cfg = config()
    target = _plan_static(int(K), int(N), int(bits), cfg)
    if target == 0 or M >= target or M < _floor(int(bits), cfg):
        _COUNTS["declined"] += 1
        return 0
    _COUNTS["padded"] += 1
    return target


def window_is_open() -> bool:
    return bool(_ACTIVE.get())


@contextmanager
def verify_window(enabled: bool = True, width: Optional[int] = None):
    """Arm the pad for the duration of one speculative verify forward.

    ``width`` is the FLATTENED row count M the projections will see (B * S), not
    S.  It is recorded whenever the caller declares a verify forward, whether or
    not the pad is on, so both arms of an A/B report the same M distribution --
    and the first sight of each M is logged, because the shipped block total has
    silently resolved to the wrong number before (defects I1052 / I1055).
    """
    cfg = config()
    if enabled and width is not None:
        m = int(width)
        seen = _WIDTHS.get(m, 0)
        _WIDTHS[m] = seen + 1
        if seen == 0:
            log_config_once()
            logger.info(
                "glm5_next verify block: M=%d (first sight), pad target=%s",
                m,
                "none" if not cfg.enabled else cfg.pad_m,
            )
    token = _ACTIVE.set(bool(enabled) and cfg.enabled)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


# ---------------------------------------------------------------------------
# The two entry points the model calls.
# ---------------------------------------------------------------------------


def _pad_rows(x: mx.array, M: int, target: int) -> mx.array:
    K = x.shape[-1]
    flat = x.reshape(M, K)
    return mx.pad(flat, ((0, target - M), (0, 0)))


def project(layer: Any, x: mx.array) -> mx.array:
    """``layer(x)``, with the verify-block row pad when this shape and M pay."""
    if not _ACTIVE.get():
        return layer(x)
    scales = getattr(layer, "scales", None)
    if scales is None or getattr(layer, "mode", "affine") != "affine":
        return layer(x)
    weight = layer.weight
    if weight.ndim != 2:
        return layer(x)
    K = x.shape[-1]
    M = x.size // K
    target = _plan(M, K, weight.shape[0], int(layer.bits))
    if target == 0:
        return layer(x)
    y = layer(_pad_rows(x, M, target))
    return y[:M].reshape(tuple(x.shape[:-1]) + (y.shape[-1],))


def quantized_matmul(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: Optional[mx.array] = None,
    *,
    transpose: bool = True,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    """``mx.quantized_matmul``, with the verify-block row pad when it pays."""
    target = 0
    K = x.shape[-1]
    M = x.size // K
    if _ACTIVE.get() and transpose and w.ndim == 2:
        target = _plan(M, K, w.shape[0], bits)
    if target == 0:
        return mx.quantized_matmul(
            x, w, scales, biases, transpose=transpose, group_size=group_size, bits=bits
        )
    y = mx.quantized_matmul(
        _pad_rows(x, M, target),
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
    )
    return y[:M].reshape(tuple(x.shape[:-1]) + (y.shape[-1],))
