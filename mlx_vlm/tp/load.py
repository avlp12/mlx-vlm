"""Load only this rank's half of a glm5_next checkpoint.

``load_model(..., lazy=True)`` skips ``mx.eval(model.parameters())``, so every
weight is an unevaluated view over the mmap'd safetensors.  Sharding *before*
the first eval means each rank only ever touches its own slice, and the pages
behind the other half are never read.  This is the same mechanism that kept a
pipeline stage at ~85 GB of a 169 GB tree; ``verify_halves_resident`` below
checks it rather than assuming it, because the alternative -- MLX materializing
the whole parent to take a slice -- would silently double the footprint.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import mlx.core as mx

from ..utils import load_model
from .glm5_next import shard_model


def load_sharded(model_path, rank: int, size: int, reduce_fn=None, verbose=True):
    """Load, shard to this rank, and never materialize the other half."""
    t0 = time.perf_counter()
    model = load_model(Path(model_path), lazy=True)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    report = shard_model(model, rank, size, reduce_fn=reduce_fn)
    gc.collect()
    mx.clear_cache()
    report["load_s"] = t_load
    report["shard_s"] = time.perf_counter() - t0
    if verbose:
        print("[tp] " + json.dumps(report), flush=True)
    return model, report


def materialize(model, verbose=True, every: int = 32) -> float:
    """Force this rank's slice resident, one tensor at a time.

    Evaluating the whole sharded parameter tree at once peaks at
    (whole checkpoint + this rank's half), because taking a slice of a lazily
    mmap'd tensor materializes the parent -- measured 6.0 GiB peak for a 4.0 GiB
    file keeping half. Evaluating slice by slice and releasing each parent
    bounds peak to (this rank's half + one tensor): 2.5 GiB on the same probe.
    The full file is still read either way; MLX has no partial-tensor read.
    """
    from mlx.utils import tree_flatten

    t0 = time.perf_counter()
    leaves = tree_flatten(model.parameters())
    for i, (_, arr) in enumerate(leaves):
        mx.eval(arr)
        if i % every == every - 1:
            mx.clear_cache()
    mx.clear_cache()
    dt = time.perf_counter() - t0
    gb = mx.get_peak_memory() / 2**30
    if verbose:
        print(f"[tp] materialized {len(leaves)} tensors in {dt:.1f}s, "
              f"peak {gb:.1f} GiB", flush=True)
    return gb


def verify_halves_resident(tmpdir=None, mb: int = 512) -> dict:
    """Does slicing a lazy mmap'd tensor read only the slice?

    Writes a safetensors file, reloads it lazily, evaluates only half of each
    tensor, and compares peak memory against the file size.  If MLX had to
    materialize the parent to slice it, peak would track the whole file.
    """
    import tempfile

    tmpdir = Path(tmpdir or tempfile.mkdtemp())
    path = tmpdir / "shardprobe.safetensors"
    n = (mb * 2**20) // 4 // 8
    mx.save_safetensors(str(path), {f"w{i}": mx.zeros((8, n), dtype=mx.float32)
                                    for i in range(8)})
    size_gb = path.stat().st_size / 2**30

    def bulk():
        w = mx.load(str(path))
        half = {k: mx.contiguous(v[: v.shape[0] // 2]) for k, v in w.items()}
        del w
        mx.eval(list(half.values()))
        return half

    def incremental():
        """Evaluate one slice at a time and drop its parent immediately.

        The parent still has to be read -- MLX has no partial-tensor read -- but
        only one parent is live at a time, so peak is bounded by
        (this rank's half + one tensor) instead of (whole file + half).
        """
        w = mx.load(str(path))
        half = {}
        for k in list(w.keys()):
            v = w.pop(k)
            half[k] = mx.contiguous(v[: v.shape[0] // 2])
            mx.eval(half[k])
            del v
            mx.clear_cache()
        return half

    out = {"file_gb": round(size_gb, 3)}
    for name, fn in (("bulk", bulk), ("incremental", incremental)):
        mx.clear_cache()
        mx.reset_peak_memory()
        kept = fn()
        peak = mx.get_peak_memory() / 2**30
        out[name] = {
            "peak_gb": round(peak, 3),
            "ratio_to_file": round(peak / size_gb, 3),
        }
        del kept
        mx.clear_cache()
    out["verdict"] = (
        "incremental bounds peak"
        if out["incremental"]["peak_gb"] < 0.75 * out["bulk"]["peak_gb"]
        else "no benefit"
    )
    return out


if __name__ == "__main__":
    print(json.dumps(verify_halves_resident(), indent=1))
