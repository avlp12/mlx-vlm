#!/usr/bin/env python3
"""Stage-1c: whole-layer A/B -- stock SwitchGLU vs Glm5NextTiledSwitchGLU,
at the real glm5_next prefill shape, in one process (same weights, same route)."""
from __future__ import annotations
import argparse, json, os, sys, time
import mlx.core as mx
sys.path.insert(0, "/Users/gesicht/src/mlx-vlm-moegemm")
from mlx_vlm.models.switch_layers import SwitchGLU
from mlx_vlm.models.glm5_next.moe_gemm import Glm5NextTiledSwitchGLU

HIDDEN, INTER, EXPERTS, TOP_K, LAYERS = 4096, 2048, 288, 8, 42


def timeit(fn, warm=3, iters=8):
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, nargs="+", default=[2048])
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--rows", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--baseline-tok-s", type=float, default=450.0)
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    mx.random.seed(0)
    ref = SwitchGLU(HIDDEN, INTER, EXPERTS, bias=False)
    for n in ("gate_proj", "up_proj", "down_proj"):
        setattr(ref, n, getattr(ref, n).to_quantized(64, 4, mode="affine"))
    mx.eval(ref.parameters())
    new = Glm5NextTiledSwitchGLU(HIDDEN, INTER, EXPERTS, bias=False)
    new.gate_proj, new.up_proj, new.down_proj = ref.gate_proj, ref.up_proj, ref.down_proj
    new.activation = ref.activation

    out = {"experts": EXPERTS, "hidden": HIDDEN, "inter": INTER, "top_k": TOP_K,
           "layers": LAYERS, "runs": []}
    for L in a.chunks:
        mx.random.seed(1)
        x = mx.random.normal((1, L, HIDDEN)).astype(mx.bfloat16)
        lg = mx.random.normal((1, L, EXPERTS))
        inds = mx.argpartition(-lg, TOP_K - 1, axis=-1)[..., :TOP_K].astype(mx.uint32)
        mx.eval(x, inds)
        t_ref = timeit(lambda: ref(x, inds), iters=a.iters)
        y_ref = ref(x, inds); mx.eval(y_ref)
        wall = L / a.baseline_tok_s * 1e3
        print(f"\n=== chunk {L} ({L*TOP_K} rows, {L*TOP_K/EXPERTS:.1f} rows/expert), "
              f"full-stack wall {wall:.0f} ms/chunk @ {a.baseline_tok_s} tok/s ===")
        print(f"  stock SwitchGLU                 {t_ref*1e3:8.3f} ms/layer  "
              f"-> {t_ref*1e3*LAYERS:8.1f} ms/chunk  "
              f"({t_ref*1e3*LAYERS/wall*100:5.1f}% of prefill)")
        rec = {"chunk": L, "stock_ms": t_ref * 1e3,
               "stock_chunk_ms": t_ref * 1e3 * LAYERS,
               "stock_share": t_ref * 1e3 * LAYERS / wall, "variants": []}
        for R in a.rows:
            os.environ["MLX_VLM_GLM5_MOE_GEMM_ROWS"] = str(R)
            y_new = new(x, inds); mx.eval(y_new)
            same = bool(mx.all(y_ref == y_new))
            d = mx.abs(y_ref.astype(mx.float32) - y_new.astype(mx.float32))
            scale = float(mx.mean(mx.abs(y_ref.astype(mx.float32))))
            t_new = timeit(lambda: new(x, inds), iters=a.iters)
            save = (t_ref - t_new) * 1e3 * LAYERS
            print(f"  tiled R={R:<3d}                     {t_new*1e3:8.3f} ms/layer  "
                  f"-> {t_new*1e3*LAYERS:8.1f} ms/chunk   speedup {t_ref/t_new:.4f}x   "
                  f"e2e {save/wall*100:+5.2f}%   bit-identical={same} "
                  f"maxabs={float(mx.max(d)):.3g} (mean|y|={scale:.3g})")
            rec["variants"].append({
                "rows": R, "ms": t_new * 1e3, "chunk_ms": t_new * 1e3 * LAYERS,
                "speedup": t_ref / t_new, "e2e_pct": save / wall * 100,
                "bit_identical": same, "max_abs": float(mx.max(d)),
                "mean_abs_ref": scale})
        out["runs"].append(rec)
        del x, lg, inds, y_ref
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
