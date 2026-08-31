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

RESULT (twin M3 Ultra, tbnet, jaccl): the gate FAILS. Expected ~4.5 ms of comm
per step, measured 45.4 ms.

The cause is not the network. An isolated all_sum of this shape is 20.4 us, and
101 of them chained with nothing in between is 8.2 us each -- MLX pipelines
them on the CPU stream. But ``[AllReduce::eval_gpu] has no GPU implementation``,
so every reduce placed between two pieces of GPU compute forces a CPU<->GPU
stream crossing. Measured with a real 2-rank group: 270.2 us/op compute-only vs
719.8 us/op interleaved = 449.6 us added per reduce.

That penalty is intrinsic to MLX, not to jaccl. With no collective anywhere,
swapping a trivial GPU-stream op for a trivial CPU-stream op inside a GPU chain
costs 215.3 us (301.7 -> 517.0 us/op), and the penalty stays flat at ~207 us
while the GPU block between crossings grows from 257 to 719 us -- it is hard
serialization, not a latency that more compute can hide.

Budget: TP=2 can save at most half the step (13.5 ms at B=1, 36.9 ms at B=8),
so it affords 30 crossings at B=1 and 82 at B=8. Megatron needs 101 and EP=2
needs 84. No scheme fits at either batch size.

Unblock: AllReduce::eval_gpu for the Metal backend. MLX already ships GPU
collectives for nccl/CUDA, so this is a backend gap. Receipt:
glm53flash/logs/tp2/phase1_gate.json
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
    p.add_argument(
        "--soak-steps",
        type=int,
        default=0,
        help="soak length in STEPS, not seconds: a wall-clock deadline makes the "
        "ranks stop at different times, and a jaccl collective whose peer has "
        "exited busy-spins forever with no timeout",
    )
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

    if a.soak_steps > 0:
        errs, buckets, bt = 0, [], []
        t_bucket = time.time() + 60.0
        for i in range(a.soak_steps):
            try:
                t0 = time.perf_counter()
                step_chain(x0, a.ops, scale, a.eval_each)
                bt.append(time.perf_counter() - t0)
                if i % 200 == 0:
                    v = float(all_sum(probe)[0].item())
                    if abs(v - expect) > 1e-3:
                        errs += 1
            except Exception as e:  # noqa: BLE001
                errs += 1
                buckets.append({"error": repr(e)[:160]})
            if time.time() >= t_bucket and bt:
                buckets.append(
                    {
                        "bucket": len(buckets),
                        "steps": len(bt),
                        "p50_ms": 1e3 * _pct(bt, 0.50),
                        "p99_ms": 1e3 * _pct(bt, 0.99),
                        "max_ms": 1e3 * max(bt),
                    }
                )
                bt, t_bucket = [], time.time() + 60.0
        res["soak"] = {
            "steps": a.soak_steps,
            "errors": errs,
            "wall_s": sum(bt) if not buckets else None,
            "per_bucket": buckets,
        }
        mx.eval(all_sum(probe))  # final barrier: both ranks leave together

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
