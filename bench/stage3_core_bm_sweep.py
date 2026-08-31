#!/usr/bin/env python3
"""PART 3: the real kernel-level prize -- sweep gather_qmm_rhs's bm in mlx core.

Requires the mlx-core-pr experimental build, which exposes MLX_GATHER_QMM_BM
(16/32/64) for the non-nax affine_gather_qmm_rhs dispatch.

Prediction from the boundary-pass model: raising bm makes the *waste* worse
(passes = ceil(N/bm) + boundaries, each pass costing bm rows), so bm=32/64 only
wins if the bigger tile's throughput more than covers it.  At E=288 / top-8 /
chunk 2048 (56.9 rows/expert):
    bm=16  1024 blocks + 287 bnd -> waste 1.280
    bm=32   512        + 287     -> waste 1.560
    bm=64   256        + 287     -> waste 2.121
so bm=32 should LOSE ~13% unless the tile buys back more than 1.22x.
This measures it instead of arguing about it.
"""

from __future__ import annotations

import argparse, json, math, os, subprocess, sys, time

HIDDEN, INTER, EXPERTS, TOP_K = 4096, 2048, 288, 8
GROUP, BITS = 64, 4

CHILD = r"""
import json, math, os, sys, time
import mlx.core as mx
HIDDEN, INTER, EXPERTS, TOP_K, GROUP, BITS = 4096, 2048, 288, 8, 64, 4
rows_per_expert = json.loads(sys.argv[1])
def timeit(fn, warm=3, iters=10):
    for _ in range(warm): mx.eval(fn())
    mx.synchronize(); t0=time.perf_counter()
    for _ in range(iters): mx.eval(fn())
    mx.synchronize(); return (time.perf_counter()-t0)/iters
def quant(shape):
    s=math.sqrt(1.0/shape[-1])
    w=mx.random.uniform(low=-s,high=s,shape=shape).astype(mx.bfloat16)
    q,sc,b=mx.quantize(w,group_size=GROUP,bits=BITS,mode="affine"); mx.eval(q,sc,b); return q,sc,b
q,sc,b = quant((EXPERTS,INTER,HIDDEN))
res={"bm":os.environ.get("MLX_GATHER_QMM_BM","16"),"mlx":mx.__file__,"rows":[]}
for rpe in rows_per_expert:
    n = rpe*EXPERTS
    idx = mx.concatenate([mx.full((rpe,),e,dtype=mx.uint32) for e in range(EXPERTS)])
    x = mx.random.normal((n,1,HIDDEN)).astype(mx.bfloat16); mx.eval(x, idx)
    dt = timeit(lambda: mx.gather_qmm(x,q,sc,b,rhs_indices=idx,transpose=True,
                 group_size=GROUP,bits=BITS,mode="affine",sorted_indices=True))
    res["rows"].append({"rows_per_expert":rpe,"n":n,"ms":dt*1e3,
                        "tflops":2.0*n*HIDDEN*INTER/dt/1e12})
    del x, idx
# realistic random top-8 route at chunk 2048 as well
mx.random.seed(0)
lg = mx.random.normal((2048,EXPERTS))
inds = mx.argpartition(-lg,TOP_K-1,axis=-1)[:,:TOP_K]
sidx = mx.sort(inds.flatten()).astype(mx.uint32)
n = sidx.size
x = mx.random.normal((n,1,HIDDEN)).astype(mx.bfloat16); mx.eval(x,sidx)
dt = timeit(lambda: mx.gather_qmm(x,q,sc,b,rhs_indices=sidx,transpose=True,
             group_size=GROUP,bits=BITS,mode="affine",sorted_indices=True))
res["realistic_2048"]={"n":n,"ms":dt*1e3,"tflops":2.0*n*HIDDEN*INTER/dt/1e12}
print("@@JSON@@"+json.dumps(res))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--python",
        default=os.path.expanduser("~/glm53flash/prep/dflash2-repro/venv/bin/python"),
    )
    ap.add_argument("--mlx-build", default="/Users/gesicht/src/mlx-core-pr/python")
    ap.add_argument("--bms", nargs="+", default=["16", "32", "64"])
    ap.add_argument(
        "--rows-per-expert", type=int, nargs="+", default=[16, 32, 57, 64, 128, 256]
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    all_res = {"mlx_build": a.mlx_build, "arms": []}
    for bm in a.bms:
        env = dict(os.environ, PYTHONPATH=a.mlx_build, MLX_GATHER_QMM_BM=bm)
        p = subprocess.run(
            [a.python, "-c", CHILD, json.dumps(a.rows_per_expert)],
            env=env,
            capture_output=True,
            text=True,
        )
        line = [l for l in p.stdout.splitlines() if l.startswith("@@JSON@@")]
        if not line:
            print(f"bm={bm} FAILED\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
            continue
        r = json.loads(line[0][len("@@JSON@@") :])
        all_res["arms"].append(r)
        print(f"--- MLX_GATHER_QMM_BM={bm} ({r['mlx']}) ---")
        for row in r["rows"]:
            print(
                f"   rows/expert={row['rows_per_expert']:<4d} "
                f"{row['ms']:8.3f} ms  {row['tflops']:6.2f} TFLOPS"
            )
        rr = r["realistic_2048"]
        print(
            f"   realistic chunk-2048 route (n={rr['n']}): "
            f"{rr['ms']:8.3f} ms  {rr['tflops']:6.2f} TFLOPS",
            flush=True,
        )

    base = next((x for x in all_res["arms"] if x["bm"] == "16"), None)
    if base:
        print("\n--- speedup vs bm=16 ---")
        for r in all_res["arms"]:
            if r["bm"] == "16":
                continue
            s = [
                f"{b['ms']/x['ms']:.3f}x@{x['rows_per_expert']}"
                for b, x in zip(base["rows"], r["rows"])
            ]
            rs = base["realistic_2048"]["ms"] / r["realistic_2048"]["ms"]
            print(f"  bm={r['bm']}: " + "  ".join(s) + f"   realistic {rs:.3f}x")
    with open(a.out, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
