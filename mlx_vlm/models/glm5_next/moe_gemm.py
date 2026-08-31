"""Segment-aligned expert GEMM for the glm5_next MoE prefill path (opt-in).

The problem
-----------
On a non-``nax`` GPU (M3/M2 Ultra) ``mx.gather_qmm(..., sorted_indices=True)`` with
``x`` of shape ``[N, 1, K]`` takes ``GatherQMM::eval_gpu``'s ``M == 1`` branch and
dispatches ``gather_qmm_rhs`` with ``bm=16, bn=32, bk=32, wm=1, wn=2``.  That kernel
walks the *distinct experts inside each 16-row block* and runs a **full** ``16 x 32 x K``
block-gemm per distinct expert, storing only that expert's row slice
(``store_result_slice``).  Every expert boundary that falls strictly inside a 16-row
block therefore costs one extra full block-gemm.

glm5_next prefill: E=288 routed experts, top_k=8, chunk 2048 -> 16384 sorted rows,
1024 blocks of 16, 287 interior boundaries -> ~1311 block-gemms for 1024 blocks of
useful work.  Measured on M3 Ultra / mlx 0.32.0 at the real shapes
(x[16384,4096] @ w[288,2048,4096] q4 g64):

    sorted gather_qmm, counts 36..78 (random top-8 route) 16.65 ms  16.5 TFLOPS
    same but every count a multiple of 16 (no interior boundary)
                                                         13.15 ms  20.9 TFLOPS
    dense quantized_matmul, same M/N/K (bm=32)           12.18 ms  22.6 TFLOPS
    bf16 matmul, same M/N/K                              11.92 ms  23.1 TFLOPS

i.e. the ~69% figure in ml-explore/mlx#4246 is *not* the bm=16 tile (that reaches 93%
of dense) -- it is the redundant boundary block-gemms.

The fix
-------
Pad every expert's row run up to a multiple of ``R`` with zero rows so no 16-row block
ever spans two experts.  Wasted rows drop from ~16 per boundary (a whole extra
block-gemm) to ``(R - c_e mod R) mod R`` (~R/2 per expert): 1.265x -> 1.132x at R=16.
The padding is folded into the sort gather that ``SwitchGLU`` already performs, so no
extra bulk traffic is added on the way in, and the unpad is folded into the unsort
gather on the way out.

``R = 16`` keeps ``M == 1`` and therefore keeps the ``gather_qmm_rhs`` (bm=16) kernel.
``R >= 32`` instead hands mlx ``x[T, R, K]``, which falls through to the general
``gather_qmm`` dispatch (bm=32 bn=32 wm=2 wn=2, the same tile the dense ``qmm_t``
kernel uses) -- a bigger tile but more padding.  Measured per-projection:

    stock              16.65 ms   16.5 TFLOPS
    R=16 flat  (bm16)  14.84 ms   18.5 TFLOPS   pad 1.132x
    R=32 tiled (bm32)  14.89 ms   18.5 TFLOPS   pad 1.227x
    R=64 tiled (bm32)  16.14 ms   17.0 TFLOPS   pad 1.328x

R=16 is the default: same throughput as R=32 with a smaller padded buffer, and its
padding factor is independent of the route's balance (it is E * R/2 / N whatever the
counts are).

Numerics
--------
The accumulation over K is unchanged (same ``BlockMMA``, same ``BK=32`` steps, same
fp32 accumulator, same order); only which rows share a threadgroup changes.  The
parity test asserts bit-identity against the stock path rather than assuming it.

Toggle: ``MLX_VLM_GLM5_MOE_GEMM=1`` (default OFF).  Sorted/prefill only -- the path is
skipped when ``indices.size < MLX_VLM_GLM5_MOE_GEMM_MIN`` (default 64, the threshold
``SwitchGLU`` itself uses to decide to sort), so decode keeps the stock kernel.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import mlx.core as mx

from ..switch_layers import SwitchGLU

MOE_GEMM_ENV = "MLX_VLM_GLM5_MOE_GEMM"
MOE_GEMM_ROWS_ENV = "MLX_VLM_GLM5_MOE_GEMM_ROWS"
MOE_GEMM_MIN_ENV = "MLX_VLM_GLM5_MOE_GEMM_MIN"

_DEFAULT_TILE_ROWS = "auto"
_DEFAULT_MIN_INDICES = 64

# gather_qmm throughput at the real gate/up shape (M3 Ultra, mlx 0.32.0, q4 g64,
# x[16384,4096] @ w[288,2048,4096]) once no tile spans two experts:
#   R=16 keeps the bm=16 `gather_qmm_rhs` kernel      21.00 TFLOPS
#   R>=32 falls through to the bm=32 `gather_qmm`     22.34 TFLOPS (= dense qmm_t)
# R=32 is only worth its extra padding when that padding costs less than the 1.064x
# the bigger tile buys back.
_BM32_SPEEDUP = 1.064


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def moe_gemm_enabled(config: Any = None) -> bool:
    """``config.moe_prefill_gemm`` (when set) wins over ``MLX_VLM_GLM5_MOE_GEMM``."""
    flag = getattr(config, "moe_prefill_gemm", None) if config is not None else None
    if flag is not None:
        return bool(flag)
    return _env_flag(MOE_GEMM_ENV)


def tile_rows() -> Optional[int]:
    """``R``, or ``None`` for the padding-aware R=16/R=32 choice (the default)."""
    raw = os.environ.get(MOE_GEMM_ROWS_ENV)
    if raw is None or raw.strip().lower() in ("", "auto"):
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 16 else None


def choose_tile_rows(counts: mx.array) -> int:
    """Pick R=16 or R=32 from the actual per-expert counts.

    Both padded row totals come from the ``counts`` array that ``segment_tile_plan``
    needs anyway, so this costs no extra device sync.  Against the measured
    whole-layer A/B (E=288, M3 Ultra) the rule picks the winner at 28.4, 42.7 and
    227.6 rows/expert and ties at 56.9.
    """
    rows16 = int(mx.sum((counts + 15) // 16).item()) * 16
    rows32 = int(mx.sum((counts + 31) // 32).item()) * 32
    return 32 if rows32 <= _BM32_SPEEDUP * rows16 else 16


def min_rows(num_experts: int, rows_per_tile: int) -> int:
    """Smallest routed-row count for which padding is a win.

    Padding costs ``~E * R/2`` wasted rows against the stock kernel's ``~(E-1) * 16``
    boundary rows, so on paper padding always wins.  It stops winning once the route
    is thin enough that most experts hold fewer than ``R`` rows: the padded layout
    then allocates a whole ``R``-row tile per *active* expert, and below ``4 * E``
    routed rows mlx does not even take the ``gather_qmm_rhs`` branch for the stock
    path (``B / E >= 4`` in ``GatherQMM::eval_gpu``), so the comparison changes
    kernel underneath.  Measured whole-layer A/B (E=288, R=16, M3 Ultra):

        3.6 rows/expert  0.696x   7.1  0.679x   10.7  1.286x   14.2  1.337x
       28.4              1.220x  42.7  1.150x   56.9  1.107x

    The crossover sits between 7.1 and 10.7, so the default gate is
    ``R * E * 3/4`` routed rows (>= 12 rows/expert on average; 3456 rows = 432 tokens
    at top-8 for E=288, R=16) -- past the crossover with margin and comfortably above
    the ``4 * E`` kernel-selection boundary.  Override with
    ``MLX_VLM_GLM5_MOE_GEMM_MIN`` (absolute routed-row count).
    """
    return _env_int(
        MOE_GEMM_MIN_ENV,
        max(_DEFAULT_MIN_INDICES, rows_per_tile * num_experts * 3 // 4),
    )


def segment_tile_plan(
    sorted_idx: mx.array, num_experts: int, rows_per_tile: Optional[int] = None
) -> Tuple[mx.array, mx.array, int, int]:
    """Plan a padded, segment-aligned tiling of an ascending-sorted expert index.

    Args:
        sorted_idx: ``[N]`` ascending expert id per routed row.
        num_experts: ``E``.
        rows_per_tile: ``R``; each expert's run is padded up to a multiple of ``R``.
            ``None`` -> chosen from the counts by :func:`choose_tile_rows`.

    Returns ``(slot, tile_expert, n_tiles, rows_per_tile)``:
        ``slot`` ``[N] int32`` destination row of each sorted row in the ``[T*R, ...]``
        padded buffer, ``tile_expert`` ``[T] uint32``, ``T``, and the ``R`` actually
        used (``rows_per_tile=None`` asks ``choose_tile_rows`` to pick it).

    ``T`` is data dependent, so this forces one device sync per call: one flush per
    MoE layer per prefill chunk (42 at chunk 2048) against ~50 ms of GPU GEMM per
    layer.
    """
    n = sorted_idx.shape[0]
    idx_i = sorted_idx.astype(mx.int32)
    counts = mx.zeros((num_experts,), mx.int32).at[idx_i].add(mx.ones((n,), mx.int32))
    if rows_per_tile is None:
        rows_per_tile = choose_tile_rows(counts)
    row_end = mx.cumsum(counts, axis=0)
    row_start = row_end - counts
    tiles_per_expert = (counts + rows_per_tile - 1) // rows_per_tile
    tile_end = mx.cumsum(tiles_per_expert, axis=0)
    tile_start = tile_end - tiles_per_expert

    pos = mx.arange(n, dtype=mx.int32) - row_start[idx_i]
    slot = (tile_start[idx_i] + pos // rows_per_tile) * rows_per_tile + (
        pos % rows_per_tile
    )
    n_tiles = int(tile_end[-1].item())  # <- the one sync
    tile_expert = mx.zeros((n_tiles,), mx.uint32)
    tile_expert[slot // rows_per_tile] = sorted_idx.astype(mx.uint32)
    return slot, tile_expert, n_tiles, rows_per_tile


class Glm5NextTiledSwitchGLU(SwitchGLU):
    """``SwitchGLU`` whose sorted (prefill) path runs segment-aligned padded tiles.

    Parameter names/shapes, the sort threshold and the ``activation(x_up, x_gate)``
    call order are identical to ``SwitchGLU``; only the row layout handed to
    ``gather_qmm`` changes.
    """

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        rows = tile_rows()
        num_experts = self.gate_proj.num_experts
        if indices.size < min_rows(num_experts, rows or 16):
            return super().__call__(x, indices)

        *_, top_k = indices.shape
        d_model = x.shape[-1]

        flat_idx = indices.flatten()
        order = mx.argsort(flat_idx)
        sorted_idx = flat_idx[order]
        slot, tile_expert, n_tiles, rows = segment_tile_plan(
            sorted_idx, num_experts, rows
        )
        if self.training:
            tile_expert = mx.stop_gradient(tile_expert)

        # Fold the pad into the sort gather: row `slot[i]` of the padded buffer reads
        # token row `order[i] // top_k`; unused rows read an appended zero row.
        n_rows = flat_idx.size
        n_tokens = x.size // d_model
        src = mx.full((n_tiles * rows,), n_tokens, dtype=mx.int32)
        src[slot] = (order // top_k).astype(mx.int32)
        x_pad = mx.concatenate(
            [x.reshape(n_tokens, d_model), mx.zeros((1, d_model), x.dtype)], axis=0
        )
        xp = x_pad[src]

        if rows == 16:
            # M == 1 keeps the gather_qmm_rhs (bm=16) kernel.
            xin = xp.reshape(n_tiles * rows, 1, d_model)
            ridx = mx.repeat(tile_expert, rows)
        else:
            # M == R > 1 falls through to the general gather_qmm (bm=32) kernel.
            xin = xp.reshape(n_tiles, rows, d_model)
            ridx = tile_expert

        x_up = self.up_proj(xin, ridx, sorted_indices=True)
        x_gate = self.gate_proj(xin, ridx, sorted_indices=True)
        y = self.down_proj(self.activation(x_up, x_gate), ridx, sorted_indices=True)

        # Fold the unpad into the unsort gather.
        inv_order = mx.argsort(order)
        y = y.reshape(n_tiles * rows, d_model)[slot[inv_order]]
        return mx.unflatten(y, 0, indices.shape)
