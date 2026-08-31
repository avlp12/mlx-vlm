#!/usr/bin/env python3
"""Localize the restore divergence: is it the cache tensors, or downstream?

Compares, at the SAME total prefix length N:
  cold : one straight-through chunked prefill of tokens[0:N]
  warm : restore a snapshot taken at boundary B, then prefill tokens[B:N]

If the resulting cache tensors are bit-identical, restore+tail is exact and any
generation divergence must come from kernel dispatch during DECODE. If they
differ, the tail prefill over a restored cache is already taking a different
numeric path -- the likely amplifier being GLM-5.3-Flash's sparse DSA indexer,
whose top-k block selection turns a 1-ulp score difference into a different
attention support.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx


def _load(model_dir):
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import get_model_path, load_model

    p = get_model_path(model_dir)
    return load_model(p, lazy=False), load_tokenizer(p)


def prefill(lm, ids, cache, step):
    i = 0
    while i < len(ids):
        n = min(step, len(ids) - i)
        lm(inputs=mx.array([ids[i : i + n]]), cache=cache, n_to_process=n)
        mx.eval([c.state for c in cache])
        i += n


def flat(cache):
    out = []

    def w(o):
        if isinstance(o, mx.array):
            out.append(o)
        elif isinstance(o, (list, tuple)):
            for x in o:
                w(x)

    for c in cache:
        w(c.state)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=6144)
    ap.add_argument("--boundary", type=int, default=4096)
    ap.add_argument("--step", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mlx_vlm.context_vault import capture_fragments, restore_fragments

    model, w = _load(args.model)
    lm = model.language_model
    tok = w._tokenizer
    base = tok.encode("Sparse indexer selection amplifies tiny numeric drift. " * 400,
                      add_special_tokens=False)
    ids = (base * (args.n // max(len(base), 1) + 2))[: args.n]

    cold = lm.make_cache()
    prefill(lm, ids, cold, args.step)

    seed = lm.make_cache()
    prefill(lm, ids[: args.boundary], seed, args.step)
    snap = capture_fragments(seed, args.boundary)
    warm = lm.make_cache()
    restore_fragments(warm, snap)
    mx.eval([c.state for c in warm])
    prefill(lm, ids[args.boundary :], warm, args.step)

    fc, fw = flat(cold), flat(warm)
    rows, n_diff = [], 0
    for i, (x, y) in enumerate(zip(fc, fw)):
        same_shape = x.shape == y.shape
        if not same_shape:
            rows.append({"i": i, "shape_cold": list(x.shape), "shape_warm": list(y.shape),
                         "bit_equal": False, "note": "shape mismatch"})
            n_diff += 1
            continue
        xb = mx.view(x.flatten(), mx.uint8)
        yb = mx.view(y.flatten(), mx.uint8)
        eq = bool(mx.all(xb == yb).item())
        if not eq:
            d = mx.abs(x.astype(mx.float32) - y.astype(mx.float32))
            rows.append({"i": i, "shape": list(x.shape), "dtype": str(x.dtype),
                         "bit_equal": False,
                         "max_abs_diff": float(mx.max(d).item()),
                         "mean_abs_diff": float(mx.mean(d).item())})
            n_diff += 1
        else:
            rows.append({"i": i, "shape": list(x.shape), "bit_equal": True})

    out = {
        "model": args.model, "n": args.n, "boundary": args.boundary, "step": args.step,
        "n_components": len(fc), "n_bit_differing": n_diff,
        "all_bit_identical": n_diff == 0,
        "components": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"components={len(fc)} differing={n_diff} all_identical={n_diff==0}")
    for r in rows:
        if not r.get("bit_equal"):
            print(" DIFF", {k: v for k, v in r.items() if k != "shape"})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
