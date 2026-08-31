"""Boot-time weight warmer — touch every weight page once after load.

The pipeline campaign observed gesicht ramping 216 -> 380 tok/s prefill across a
cold run as the page cache filled: ``mx.load`` memory-maps safetensors, so the
first forward pass faults ~169 GB of weights in from disk *while it is being
timed*. A single cheap pass over every parameter pays that cost once, at boot,
where it is not on a user's first request.

The pass is deliberately not a forward pass: it never allocates activations,
never needs a prompt, and touches each array with a scalar reduction so MLX must
materialize the underlying pages without producing a large intermediate.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import mlx.core as mx

__all__ = ["warm_model", "warmer_enabled"]

_ENV_ENABLE = "MLX_VLM_WARM_WEIGHTS"


def warmer_enabled(default: str = "") -> bool:
    return os.environ.get(_ENV_ENABLE, default).strip().lower() in ("1", "true", "yes", "on")


def _iter_arrays(tree: Any) -> Iterable[mx.array]:
    if isinstance(tree, mx.array):
        yield tree
    elif isinstance(tree, dict):
        for v in tree.values():
            yield from _iter_arrays(v)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            yield from _iter_arrays(v)


def warm_model(model: Any, batch: int = 512, verbose: bool = False) -> Dict[str, float]:
    """Fault in every weight page of ``model``. Returns timing/size stats.

    ``batch`` bounds how many reductions are evaluated per ``mx.eval`` so peak
    scratch stays small on a 169 GB tree.
    """
    t0 = time.monotonic()
    params = model.parameters() if hasattr(model, "parameters") else model
    total_bytes = 0
    count = 0
    pending = []
    for arr in _iter_arrays(params):
        total_bytes += int(arr.nbytes)
        count += 1
        # A scalar reduction forces every page of ``arr`` to be read exactly once
        # without materializing anything the size of the weight itself.
        pending.append(mx.sum(arr, keepdims=False) if arr.size else mx.array(0))
        if len(pending) >= batch:
            mx.eval(pending)
            pending = []
    if pending:
        mx.eval(pending)
    mx.clear_cache()
    elapsed = time.monotonic() - t0
    stats = {
        "arrays": float(count),
        "bytes": float(total_bytes),
        "gib": total_bytes / (1024**3),
        "seconds": elapsed,
        "gib_per_s": (total_bytes / (1024**3) / elapsed) if elapsed > 0 else 0.0,
    }
    if verbose:
        print(
            f"[warm] {count} arrays, {stats['gib']:.1f} GiB in {elapsed:.1f}s "
            f"({stats['gib_per_s']:.2f} GiB/s)"
        )
    return stats
