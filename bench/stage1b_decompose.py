#!/usr/bin/env python3
"""Stage-1b: decompose the stock MoE prefill layer and price the candidate fixes."""
from __future__ import annotations
import argparse, json, math, time
import mlx.core as mx

HIDDEN, INTER, EXPERTS, TOP_K, LAYERS = 4096, 2048, 288, 8, 42
GROUP, BITS = 64, 4


def timeit(fn, warm=3, iters=8):
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def make_q(shape):
    sc = math.sqrt(1.0 / shape[-1])
    w = mx.random.uniform(low=-sc, high=sc, shape=shape).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=GROUP, bits=BITS, mode="affine")
    mx.eval(q, s, b)
    del w
    return q, s, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()
    L, it = a.chunk, a.iters
    N = L * TOP_K
    useful = 2.0 * N * HIDDEN * INTER
    res = {"chunk": L, "n_rows": N, "rows": {}}

    mx.random.seed(0)
    x = mx.random.normal((1, L, HIDDEN)).astype(mx.bfloat16)
    lg = mx.random.normal((1, L, EXPERTS))
    inds = mx.argpartition(-lg, TOP_K - 1, axis=-1)[..., :TOP_K].astype(mx.uint32)
    mx.eval(x, inds)

    flat = inds.flatten()
    order = mx.argsort(flat)
    inv = mx.argsort(order)
    sidx = flat[order]
    xs = x.reshape(-1, HIDDEN)[order // TOP_K]
    mx.eval(order, inv, sidx, xs)

    counts = mx.zeros((EXPERTS,), mx.int32).at[sidx.astype(mx.int32)].add(
        mx.ones((N,), mx.int32))
    mx.eval(counts)
    cl = counts.tolist()
    res["counts"] = {"min": min(cl), "max": max(cl), "mean": sum(cl) / EXPERTS}
    for R in (16, 32, 64):
        res[f"pad_factor_R{R}"] = sum((c + R - 1) // R for c in cl) * R / N
    print(f"counts {min(cl)}..{max(cl)} mean {sum(cl)/EXPERTS:.1f}   "
          + "  ".join(f"padR{R}={res[f'pad_factor_R{R}']:.3f}" for R in (16, 32, 64)))

    def rec(name, dt, fl=None):
        res["rows"][name] = {"ms": dt * 1e3,
                             "tflops": (fl / dt / 1e12) if fl else None}
        tf = f"{fl/dt/1e12:6.2f} TF" if fl else "        "
        print(f"  {name:<46s} {dt*1e3:8.3f} ms  {tf}", flush=True)
        return dt

    qg, sg, bg = make_q((EXPERTS, INTER, HIDDEN))     # gate / up
    qd, sd, bd = make_q((EXPERTS, HIDDEN, INTER))     # down
    x3 = xs.reshape(N, 1, HIDDEN)
    h3 = mx.random.normal((N, 1, INTER)).astype(mx.bfloat16)
    mx.eval(x3, h3)

    print("\n--- stock sorted path, per-projection ---")
    t_gate = rec("gate/up gather_qmm sorted [stock]",
                 timeit(lambda: mx.gather_qmm(x3, qg, sg, bg, rhs_indices=sidx,
                        transpose=True, group_size=GROUP, bits=BITS, mode="affine",
                        sorted_indices=True), iters=it), useful)
    t_down = rec("down    gather_qmm sorted [stock]",
                 timeit(lambda: mx.gather_qmm(h3, qd, sd, bd, rhs_indices=sidx,
                        transpose=True, group_size=GROUP, bits=BITS, mode="affine",
                        sorted_indices=True), iters=it), useful)

    print("\n--- non-GEMM overhead of SwitchGLU ---")
    t_sort = rec("argsort(flat) + argsort(order)",
                 timeit(lambda: (mx.argsort(flat), mx.argsort(mx.argsort(flat))),
                        iters=it))
    t_gx = rec("gather x rows  x[2048,4096] -> [16384,4096]",
               timeit(lambda: x.reshape(-1, HIDDEN)[order // TOP_K], iters=it))
    y_big = mx.random.normal((N, HIDDEN)).astype(mx.bfloat16)
    mx.eval(y_big)
    t_un = rec("unsort y[16384,4096][inv]", timeit(lambda: y_big[inv], iters=it))
    u1 = mx.random.normal((N, 1, INTER)).astype(mx.bfloat16)
    u2 = mx.random.normal((N, 1, INTER)).astype(mx.bfloat16)
    mx.eval(u1, u2)
    import mlx.nn as nn
    t_act = rec("clamped swiglu on [16384,1,2048]",
                timeit(lambda: nn.silu(mx.clip(u2, None, 10.0))
                       * mx.clip(u1, -10.0, 10.0), iters=it))

    print("\n--- candidate A: pad each expert run to a multiple of 16, keep bm16 ---")
    pad_res = {}
    for R in (16, 32, 64):
        tpe = [(c + R - 1) // R for c in cl]
        T = sum(tpe)
        tstart = [0] * EXPERTS
        acc = 0
        for e in range(EXPERTS):
            tstart[e] = acc
            acc += tpe[e]
        rstart = [0] * EXPERTS
        acc = 0
        for e in range(EXPERTS):
            rstart[e] = acc
            acc += cl[e]
        rs = mx.array(rstart, mx.int32)
        ts = mx.array(tstart, mx.int32)
        si = sidx.astype(mx.int32)
        pos = mx.arange(N, dtype=mx.int32) - rs[si]
        slot = (ts[si] + pos // R) * R + (pos % R)
        tile_e = mx.zeros((T,), mx.uint32)
        tile_e[slot // R] = sidx.astype(mx.uint32)
        src = mx.full((T * R,), N, mx.int32)
        src[slot] = mx.arange(N, dtype=mx.int32)
        xs_pad = mx.concatenate([xs, mx.zeros((1, HIDDEN), xs.dtype)], 0)
        xp = xs_pad[src]
        mx.eval(slot, tile_e, src, xp)
        pf = T * R / N
        t_pad = timeit(lambda: xs_pad[src], iters=it)
        if R == 16:
            ridx = mx.repeat(tile_e, R)
            mx.eval(ridx)
            xin = xp.reshape(T * R, 1, HIDDEN)
            lbl = f"pad{R} FLAT sorted gather_qmm [bm16] pad {pf:.3f}x"
        else:
            ridx = tile_e
            xin = xp.reshape(T, R, HIDDEN)
            lbl = f"pad{R} TILED gather_qmm [bm32] pad {pf:.3f}x"
        dt = timeit(lambda: mx.gather_qmm(xin, qg, sg, bg, rhs_indices=ridx,
                    transpose=True, group_size=GROUP, bits=BITS, mode="affine",
                    sorted_indices=True), iters=it)
        rec(lbl, dt, useful)
        rec(f"   pad-gather cost R={R} ([{T*R},4096])", t_pad)
        # down shape
        hp = mx.random.normal((T * R, 1, INTER)).astype(mx.bfloat16)
        mx.eval(hp)
        hin = hp if R == 16 else hp.reshape(T, R, INTER)
        dtd = timeit(lambda: mx.gather_qmm(hin, qd, sd, bd, rhs_indices=ridx,
                     transpose=True, group_size=GROUP, bits=BITS, mode="affine",
                     sorted_indices=True), iters=it)
        rec(f"   down R={R}", dtd, useful)
        pad_res[R] = {"pad_factor": pf, "gate_ms": dt * 1e3, "down_ms": dtd * 1e3,
                      "pad_gather_ms": t_pad * 1e3, "T": T}
        del xp, src, hp
    res["pad"] = pad_res

    # ---- full-layer projections -------------------------------------------
    stock_layer = 2 * t_gate + t_down
    print("\n--- layer roll-up (3 GEMMs only) ---")
    print(f"  stock 3 GEMMs           {stock_layer*1e3:8.3f} ms")
    for R, d in pad_res.items():
        tot = (2 * d["gate_ms"] + d["down_ms"]) / 1e3
        print(f"  pad{R} 3 GEMMs           {tot*1e3:8.3f} ms   "
              f"({stock_layer/tot:.3f}x)   +pad gather {d['pad_gather_ms']:.2f} ms")
        d["three_gemm_ms"] = tot * 1e3
        d["gemm_speedup"] = stock_layer / tot
    res["stock_three_gemm_ms"] = stock_layer * 1e3
    res["overhead_ms"] = {"sort": t_sort * 1e3, "gather_x": t_gx * 1e3,
                          "unsort": t_un * 1e3, "act": t_act * 1e3}

    if a.json:
        json.dump(res, open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
