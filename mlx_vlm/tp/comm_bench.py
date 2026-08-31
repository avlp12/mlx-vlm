"""PHASE 1 gate: sustained all_sum at glm5_next decode cadence.

One "step" is the 101 all-reduces a TP=2 decode step of GLM-5.3-Flash needs
(34 KDA o_proj + 11 DSA x 2 + 45 MLP/MoE), each on a [1, 1, 4096] bf16
activation.  The reduces are *chained* -- each one consumes the previous
result -- because in a real step every all-reduce sits on the critical path of
the next layer; issuing them independently would let MLX pipeline them and
understate the cost.

Run (rank 1 first, it is not the coordinator)::

    MLX_RANK=1 python -m mlx_vlm.tp.comm_bench --hosts 10.0.0.1,10.0.0.2
    MLX_RANK=0 python -m mlx_vlm.tp.comm_bench --hosts 10.0.0.1,10.0.0.2 --steps 1000 --soak 600
"""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

from .transport import all_sum, backend, init_tp, tp_rank, tp_size


def _pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def step_chain(x0, n_ops, scale, eval_each=False):
    """One decode step's worth of chained all-reduces."""
    x = x0
    for _ in range(n_ops):
        x = all_sum(x) * scale
        if eval_each:
            mx.eval(x)
    if not eval_each:
        mx.eval(x)
    return x


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hosts", default="10.0.0.1,10.0.0.2")
    p.add_argument("--backend", default="jaccl", choices=["jaccl", "ring"])
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--ops", type=int, default=101, help="all-reduces per step")
    p.add_argument("--shape", default="1,1,4096")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--soak", type=float, default=0.0, help="soak seconds after the run")
    p.add_argument("--eval-each", action="store_true", help="pessimistic: sync per op")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    g = init_tp(a.hosts.split(","), a.rank, backend=a.backend, strict=True)
    r, n = tp_rank(), tp_size()
    shape = tuple(int(v) for v in a.shape.split(","))
    scale = 1.0 / n
    print(f"[tp] rank {r}/{n} backend={backend()} shape={shape} ops/step={a.ops}", flush=True)

    x0 = mx.ones(shape, dtype=mx.bfloat16) * mx.array(r + 1, dtype=mx.bfloat16)
    mx.eval(x0)

    # correctness: all_sum of (rank+1) over 2 ranks is 3, and the chain with
    # scale=1/n is a no-op fixed point only if every rank agrees -- so a wrong
    # or dropped reduce shows up as a value drift.
    probe = mx.ones((4,), dtype=mx.float32) * (r + 1)
    expect = float(sum(range(1, n + 1)))
    got = float(all_sum(probe)[0].item())
    assert abs(got - expect) < 1e-3, f"all_sum wrong: {got} != {expect}"
    print(f"[tp] all_sum correctness ok ({got})", flush=True)

    for _ in range(20):
        step_chain(x0, a.ops, scale, a.eval_each)

    times = []
    t_run = time.perf_counter()
    for _ in range(a.steps):
        t0 = time.perf_counter()
        step_chain(x0, a.ops, scale, a.eval_each)
        times.append(time.perf_counter() - t0)
    wall = time.perf_counter() - t_run

    res = {
        "rank": r,
        "size": n,
        "backend": backend(),
        "ops_per_step": a.ops,
        "shape": list(shape),
        "steps": a.steps,
        "eval_each": a.eval_each,
        "wall_s": wall,
        "step_ms": {
            "mean": 1e3 * sum(times) / len(times),
            "p50": 1e3 * _pct(times, 0.50),
            "p90": 1e3 * _pct(times, 0.90),
            "p99": 1e3 * _pct(times, 0.99),
            "max": 1e3 * max(times),
        },
        "per_op_us": 1e6 * _pct(times, 0.50) / a.ops,
    }

    if a.soak > 0:
        errs, checks, buckets = 0, 0, []
        t_end = time.time() + a.soak
        t_bucket, bt = time.time() + 60.0, []
        while time.time() < t_end:
            try:
                t0 = time.perf_counter()
                step_chain(x0, a.ops, scale, a.eval_each)
                bt.append(time.perf_counter() - t0)
                checks += 1
                if checks % 200 == 0:
                    v = float(all_sum(probe)[0].item())
                    if abs(v - expect) > 1e-3:
                        errs += 1
            except Exception as e:  # noqa: BLE001
                errs += 1
                buckets.append({"error": repr(e)})
            if time.time() >= t_bucket and bt:
                buckets.append(
                    {
                        "minute": len(buckets),
                        "steps": len(bt),
                        "p50_ms": 1e3 * _pct(bt, 0.50),
                        "p99_ms": 1e3 * _pct(bt, 0.99),
                        "max_ms": 1e3 * max(bt),
                    }
                )
                bt, t_bucket = [], time.time() + 60.0
        res["soak"] = {
            "seconds": a.soak,
            "steps": checks,
            "errors": errs,
            "per_minute": buckets,
        }

    if r == 0:
        print(json.dumps(res, indent=1), flush=True)
        if a.out:
            from pathlib import Path

            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=1))
    else:
        print(f"[tp] rank {r} done, p50 step {res['step_ms']['p50']:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
