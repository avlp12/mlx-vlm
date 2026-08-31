#!/usr/bin/env python3
"""Verify-width sweep: cost of a width-L forward vs width-1 on a warm cache.

Speculative speedup estimates rest on how a width-L verify compares to L serial
decodes. ratio(L)=t(L)/t(1) near 1.0 means the verify is nearly free and accept
rate is the only lever; if it climbs with L the drafter must clear a higher bar.

The cache is snapshot once and restored before every timed call, so each L sees
an identical starting state and identical cache offset.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--max-width", type=int, default=8)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mlx_vlm.context_vault import capture_fragments, restore_fragments
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import get_model_path, load_model

    path = get_model_path(args.model)
    model = load_model(path, lazy=False)
    lm = model.language_model
    tok = load_tokenizer(path)._tokenizer

    ids = tok.encode("The verification width sweep measures forward cost. " * 300,
                     add_special_tokens=False)[: args.ctx]
    lm(inputs=mx.array([ids]), cache=(cache := lm.make_cache()))
    mx.eval([c.state for c in cache])
    snap = capture_fragments(cache, len(ids))
    assert snap is not None

    rows = []
    for L in range(1, args.max_width + 1):
        step = mx.array([[ids[i % len(ids)] for i in range(L)]])
        times = []
        for _ in range(args.reps + 2):
            fresh = lm.make_cache()
            restore_fragments(fresh, snap)
            mx.eval([c.state for c in fresh])
            t0 = time.perf_counter()
            out = lm(inputs=step, cache=fresh)
            mx.eval(out)
            times.append(time.perf_counter() - t0)
        times = sorted(times)[1:-1]
        rows.append({"width": L, "median_s": statistics.median(times),
                     "min_s": min(times), "samples": len(times)})
        print(f"L={L}: {statistics.median(times)*1000:.3f} ms", flush=True)

    t1 = rows[0]["median_s"]
    for r in rows:
        r["ratio_vs_L1"] = r["median_s"] / t1
        r["per_token_ratio"] = (r["median_s"] / r["width"]) / t1

    out = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": args.model,
           "ctx": len(ids), "reps": args.reps, "rows": rows,
           "interpretation": ("ratio_vs_L1 = cost of a width-L verify relative to one "
                              "decode step; per_token_ratio < 1 is the headroom "
                              "speculative decoding converts into speedup once accept "
                              "rate is applied")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{'L':>3} {'ms':>9} {'ratio(L)':>9} {'per-tok':>9}")
    for r in rows:
        print(f"{r['width']:>3} {r['median_s']*1000:>9.3f} {r['ratio_vs_L1']:>9.3f} {r['per_token_ratio']:>9.3f}")


if __name__ == "__main__":
    main()
