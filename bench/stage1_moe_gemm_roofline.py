#!/usr/bin/env python3
"""Stage-1 gate for the glm5_next MoE prefill expert-GEMM.

Real shapes (GLM-5.3-Flash-vlm-q4-quasar, text_config):
  hidden=4096, moe_intermediate=2048, n_routed_experts=288, top_k=8,
  42 sparse layers (45 - first_k_dense_replace 3), switch_mlp = 4-bit g64 affine.
  Prefill chunk 2048 -> 16384 routed rows, mean 56.9 rows/expert.

What the stock path runs (mlx 0.32.0, M3 Ultra = non-nax):
  SwitchGLU -> mx.gather_qmm(x[N,1,K], w[E,O,K], sorted_indices=True)
  -> GatherQMM::eval_gpu takes the `M==1 && sorted && B/E>=4` branch
  -> gather_qmm_rhs()  bm=16 bn=32 bk=32 wm=1 wn=2   (64 threads/threadgroup)
  The kernel walks the DISTINCT experts inside each 16-row block and runs a FULL
  16x32xK gemm per distinct expert, storing only that expert's row slice.
  => every expert boundary that lands inside a 16-row block costs one extra
     full block-gemm.  That is the hypothesis this bench isolates.

Compared against:
  dense mx.quantized_matmul  -> qmm_t   bm=32 bn=32 wm=2 wn=2  (128 threads)
  tiled mx.gather_qmm(x[T,R,K], rhs_indices[T]) -> gather_qmm  bm=32 bn=32 wm=2 wn=2
        (a user-space, no-custom-kernel candidate: pad each expert's rows to a
         multiple of R so no tile ever spans two experts)
"""

from __future__ import annotations

import argparse
import json
import math
import time

import mlx.core as mx

HIDDEN = 4096
INTER = 2048
EXPERTS = 288
TOP_K = 8
SPARSE_LAYERS = 42
GROUP = 64
BITS = 4
BM_STOCK = 16


def timeit(fn, warm=3, iters=10):
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def make_q(shape, bits=BITS, group=GROUP):
    scale = math.sqrt(1.0 / shape[-1])
    w = mx.random.uniform(low=-scale, high=scale, shape=shape).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=group, bits=bits, mode="affine")
    mx.eval(q, s, b)
    del w
    return q, s, b


def idx_from_counts(counts):
    idx = mx.concatenate(
        [mx.full((c,), e, dtype=mx.uint32) for e, c in enumerate(counts) if c > 0]
    )
    mx.eval(idx)
    return idx


def balanced_counts(n_rows, experts):
    base, rem = divmod(n_rows, experts)
    return [base + (1 if e < rem else 0) for e in range(experts)]


def aligned_counts(n_rows, experts, bm=BM_STOCK):
    """Same total rows, every count a multiple of bm -> no block spans 2 experts."""
    assert n_rows % bm == 0
    blocks, rem = divmod(n_rows // bm, experts)
    return [(blocks + (1 if e < rem else 0)) * bm for e in range(experts)]


def realistic_counts(n_tokens, top_k, experts, seed=0):
    mx.random.seed(seed)
    logits = mx.random.normal((n_tokens, experts))
    inds = mx.argpartition(-logits, top_k - 1, axis=-1)[:, :top_k]
    flat = inds.flatten()
    c = mx.sum(
        (flat[:, None] == mx.arange(experts)[None, :]).astype(mx.int32), axis=0
    )
    mx.eval(c)
    return c.tolist()


def block_passes(counts, bm):
    """Number of full bm-row block-gemms the stock kernel performs."""
    n = sum(counts)
    n_blocks = (n + bm - 1) // bm
    # distinct experts per block, walking the sorted array
    bounds = []
    acc = 0
    for c in counts[:-1]:
        acc += c
        bounds.append(acc)
    extra = 0
    seen = set()
    for b in bounds:
        if b % bm != 0:                 # boundary strictly inside a block
            blk = b // bm
            extra += 1
            seen.add(blk)
    return n_blocks + extra


def tile_plan(counts, R):
    tiles = []
    for e, c in enumerate(counts):
        if c:
            tiles.extend([e] * ((c + R - 1) // R))
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--baseline-tok-s", type=float, default=450.0)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    n_rows = args.chunk * TOP_K
    it = args.iters
    useful = 2.0 * n_rows * HIDDEN * INTER          # FLOPs of ONE gate/up proj
    out = {"chunk": args.chunk, "n_rows": n_rows, "experts": EXPERTS,
           "hidden": HIDDEN, "inter": INTER, "bits": BITS, "group": GROUP,
           "device": mx.device_info().get("device_name"), "rows": []}
    try:
        import importlib.metadata as md
        out["mlx_version"] = md.version("mlx")
    except Exception:
        out["mlx_version"] = None

    c_bal = balanced_counts(n_rows, EXPERTS)
    c_ali = aligned_counts(n_rows, EXPERTS, BM_STOCK)
    c_real = realistic_counts(args.chunk, TOP_K, EXPERTS)
    assert sum(c_ali) == n_rows and sum(c_bal) == n_rows and sum(c_real) == n_rows

    for nm, c in (("balanced", c_bal), ("aligned16", c_ali), ("realistic", c_real)):
        bp = block_passes(c, BM_STOCK)
        out[f"counts_{nm}"] = {
            "min": min(c), "max": max(c), "mean": sum(c) / len(c),
            "predicted_block_passes_bm16": bp,
            "predicted_waste_bm16": bp * BM_STOCK / n_rows,
            "pad_factor_R32": sum((x + 31) // 32 for x in c) * 32 / n_rows,
            "pad_factor_R64": sum((x + 63) // 64 for x in c) * 64 / n_rows,
        }
    print(f"mlx {out['mlx_version']}  {out['device']}")
    for nm in ("balanced", "aligned16", "realistic"):
        d = out[f"counts_{nm}"]
        print(f"  counts[{nm:<9s}] {d['min']}..{d['max']} mean {d['mean']:.1f}  "
              f"predicted bm16 waste {d['predicted_waste_bm16']:.3f}  "
              f"padR32 {d['pad_factor_R32']:.3f}  padR64 {d['pad_factor_R64']:.3f}")
    print()

    def rec(name, dt, flops_useful, extra=None):
        r = {"name": name, "ms": dt * 1e3, "useful_tflops": flops_useful / dt / 1e12}
        if extra:
            r.update(extra)
        out["rows"].append(r)
        print(f"{name:<50s} {dt*1e3:8.3f} ms  {flops_useful/dt/1e12:6.2f} TFLOPS(useful)",
              flush=True)
        return dt

    # ---------------- gate/up shape ----------------
    print("=== gate/up:  x[16384,K=4096]  w[288,N=2048,4096]  q4 g64 ===")
    qg, sg, bg = make_q((EXPERTS, INTER, HIDDEN))
    x_flat = mx.random.normal((n_rows, 1, HIDDEN)).astype(mx.bfloat16)
    mx.eval(x_flat)

    times = {}
    for nm, c in (("balanced57", c_bal), ("aligned16", c_ali), ("realistic", c_real)):
        idx = idx_from_counts(c)
        times[nm] = rec(
            f"A  sorted gather_qmm [STOCK bm16]  {nm}",
            timeit(lambda: mx.gather_qmm(x_flat, qg, sg, bg, rhs_indices=idx,
                                         transpose=True, group_size=GROUP, bits=BITS,
                                         mode="affine", sorted_indices=True), iters=it),
            useful, {"counts": nm})
        del idx

    idx_real = idx_from_counts(c_real)
    times["unsorted"] = rec(
        "B  UNsorted gather_qmm (realistic)",
        timeit(lambda: mx.gather_qmm(x_flat, qg, sg, bg, rhs_indices=idx_real,
                                     transpose=True, group_size=GROUP, bits=BITS,
                                     mode="affine", sorted_indices=False),
               iters=max(3, it // 3)), useful)

    qd, sd, bd = make_q((INTER, HIDDEN))
    x2 = x_flat.reshape(n_rows, HIDDEN)
    times["dense"] = rec(
        "C  dense quantized_matmul [bm32] same M/N/K",
        timeit(lambda: mx.quantized_matmul(x2, qd, sd, bd, transpose=True,
                                           group_size=GROUP, bits=BITS,
                                           mode="affine"), iters=it), useful)
    del qd, sd, bd

    for R in (32, 64):
        tiles = tile_plan(c_real, R)
        T = len(tiles)
        ti = mx.array(tiles, dtype=mx.uint32)
        xp = mx.random.normal((T, R, HIDDEN)).astype(mx.bfloat16)
        mx.eval(xp, ti)
        dt = timeit(lambda: mx.gather_qmm(xp, qg, sg, bg, rhs_indices=ti,
                                          transpose=True, group_size=GROUP,
                                          bits=BITS, mode="affine",
                                          sorted_indices=True),
                    iters=max(3, it // 2))
        times[f"tiled{R}"] = dt
        rec(f"D  tiled gather_qmm x[{T},{R},4096] [bm32] pad "
            f"{T*R/n_rows:.3f}x", dt, useful,
            {"R": R, "tiles": T, "pad_factor": T * R / n_rows})
        del xp, ti

    a = mx.random.normal((n_rows, HIDDEN)).astype(mx.bfloat16)
    wb = mx.random.normal((INTER, HIDDEN)).astype(mx.bfloat16)
    mx.eval(a, wb)
    times["bf16"] = rec("E  bf16 dense matmul (compute roofline proxy)",
                        timeit(lambda: a @ wb.T, iters=it), useful)
    del a, wb

    nb = qg.nbytes + sg.nbytes + bg.nbytes
    dtbw = timeit(lambda: mx.sum(qg, axis=(1, 2)), iters=it)
    out["weight_read_gbs"] = nb / dtbw / 1e9
    out["weight_bytes_per_proj"] = nb
    print(f"{'F  weight-read bandwidth (1.36 GB q4+s+b)':<50s} {dtbw*1e3:8.3f} ms  "
          f"{nb/dtbw/1e9:6.1f} GB/s")
    print(f"   -> weight traffic is {nb/dtbw/ (times['realistic']) /1e9*0+0:.0f}"
          f"{'':s}", end="")
    print(f"   bandwidth-bound floor for one proj = "
          f"{nb/(out['weight_read_gbs']*1e9)*1e3:.2f} ms", flush=True)

    # sweep rows/expert -> shows the boundary-waste curve
    print("\n--- rows/expert sweep (balanced), stock sorted path ---")
    sweep = []
    for rpe in (8, 16, 24, 32, 48, 57, 64, 96, 128, 256):
        nr = rpe * EXPERTS
        c = [rpe] * EXPERTS
        idx = idx_from_counts(c)
        xs = mx.random.normal((nr, 1, HIDDEN)).astype(mx.bfloat16)
        mx.eval(xs)
        dt = timeit(lambda: mx.gather_qmm(xs, qg, sg, bg, rhs_indices=idx,
                                          transpose=True, group_size=GROUP,
                                          bits=BITS, mode="affine",
                                          sorted_indices=True),
                    iters=max(3, it // 2))
        fl = 2.0 * nr * HIDDEN * INTER
        pred = block_passes(c, BM_STOCK) * BM_STOCK / nr
        sweep.append({"rows_per_expert": rpe, "ms": dt * 1e3,
                      "tflops": fl / dt / 1e12, "predicted_waste": pred})
        print(f"  rpe={rpe:<4d} n={nr:<6d} {dt*1e3:8.3f} ms  {fl/dt/1e12:6.2f} TFLOPS"
              f"   predicted bm16 waste {pred:.3f}", flush=True)
        del xs, idx
    out["sweep"] = sweep
    del qg, sg, bg, x2

    # ---------------- full SwitchGLU layer (stock) ----------------
    print("\n=== full stock SwitchGLU forward, prefill shape ===")
    import sys
    sys.path.insert(0, "/Users/gesicht/src/mlx-vlm-moegemm")
    from mlx_vlm.models.switch_layers import SwitchGLU
    mx.random.seed(0)
    sw = SwitchGLU(HIDDEN, INTER, EXPERTS, bias=False)
    sw.gate_proj = sw.gate_proj.to_quantized(GROUP, BITS, mode="affine")
    sw.up_proj = sw.up_proj.to_quantized(GROUP, BITS, mode="affine")
    sw.down_proj = sw.down_proj.to_quantized(GROUP, BITS, mode="affine")
    mx.eval(sw.parameters())
    xt = mx.random.normal((1, args.chunk, HIDDEN)).astype(mx.bfloat16)
    mx.random.seed(1)
    lg = mx.random.normal((1, args.chunk, EXPERTS))
    inds = mx.argpartition(-lg, TOP_K - 1, axis=-1)[..., :TOP_K].astype(mx.uint32)
    mx.eval(xt, inds)
    t_layer = timeit(lambda: sw(xt, inds), iters=max(4, it // 2))
    moe_flops = 3.0 * useful
    print(f"{'G  SwitchGLU(x[1,2048,4096], top8) per layer':<50s} "
          f"{t_layer*1e3:8.3f} ms  {moe_flops/t_layer/1e12:6.2f} TFLOPS(useful)")
    out["switchglu_layer_ms"] = t_layer * 1e3
    out["switchglu_layer_tflops"] = moe_flops / t_layer / 1e12

    # ---------------- projection ----------------
    chunk_wall_ms = args.chunk / args.baseline_tok_s * 1e3
    stock_total = t_layer * 1e3 * SPARSE_LAYERS
    # ceiling if the 3 gemms ran at the dense-qmm rate
    dense_layer_ms = 3 * times["dense"] * 1e3          # gate/up/down ~ same FLOPs
    tiled_best = min((times[k], k) for k in times if k.startswith("tiled"))
    tiled_layer_ms = 3 * tiled_best[0] * 1e3
    out["projection"] = {
        "chunk_wall_ms_at_baseline": chunk_wall_ms,
        "stock_moe_ms_per_chunk_42L": stock_total,
        "stock_moe_share_of_prefill": stock_total / chunk_wall_ms,
        "dense_ceiling_ms_per_chunk_42L": dense_layer_ms * SPARSE_LAYERS,
        "dense_ceiling_recoverable_pct_e2e":
            (stock_total - dense_layer_ms * SPARSE_LAYERS) / chunk_wall_ms * 100,
        "tiled_best": tiled_best[1],
        "tiled_ms_per_chunk_42L": tiled_layer_ms * SPARSE_LAYERS,
        "tiled_recoverable_pct_e2e":
            (stock_total - tiled_layer_ms * SPARSE_LAYERS) / chunk_wall_ms * 100,
    }
    print(f"\n=== projection (42 sparse layers, {args.baseline_tok_s} tok/s, "
          f"{chunk_wall_ms:.0f} ms/chunk) ===")
    for k, v in out["projection"].items():
        print(f"  {k:<38s} {v if isinstance(v,str) else format(v, ',.4f')}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
