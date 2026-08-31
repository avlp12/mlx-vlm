#!/usr/bin/env python3
"""Isolate WHERE the restore token divergence enters: prefill state, or decode.

Bypasses dispatch entirely and drives the model directly, so the only thing that
differs between arms is how the cache was built.

  cold : chunked prefill of tokens[0:N]
  warm : restore a snapshot at B, then chunked prefill of tokens[B:N]

Step 1 re-confirms the caches are bit-identical (localize already showed this).
Step 2 then runs the SAME greedy decode loop from each and compares token ids.

  identical caches + identical tokens -> restore is exact end to end, and the
     gate's divergence must come from something dispatch does, not from restore
  identical caches + divergent tokens -> decode itself is not deterministic
     given identical state, i.e. the divergence is a kernel/dispatch artifact
     rather than a vault correctness bug
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


def bits_equal(a, b):
    if len(a) != len(b):
        return False, "component count"
    for i, (x, y) in enumerate(zip(a, b)):
        if x.shape != y.shape or x.dtype != y.dtype:
            return False, f"shape/dtype at {i}"
        xb = mx.view(x.flatten(), mx.uint8)
        yb = mx.view(y.flatten(), mx.uint8)
        if not bool(mx.all(xb == yb).item()):
            return False, f"bytes at {i}"
    return True, None


def decode(lm, cache, last_id, n_tokens):
    """Greedy argmax decode; returns token ids."""
    ids = []
    y = mx.array([[last_id]])
    for _ in range(n_tokens):
        out = lm(inputs=y, cache=cache, n_to_process=1)
        logits = getattr(out, "logits", None)
        if logits is None:
            logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = logits[:, -1, :]
        mx.eval(logits)
        t = int(mx.argmax(logits, axis=-1).item())
        ids.append(t)
        y = mx.array([[t]])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=15141)
    ap.add_argument("--boundary", type=int, default=14336)
    ap.add_argument("--step", type=int, default=2048)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mlx_vlm.context_vault import capture_fragments, restore_fragments

    model, w = _load(args.model)
    lm = model.language_model
    tok = w._tokenizer
    base = tok.encode("Sparse indexer selection amplifies tiny numeric drift. " * 400,
                      add_special_tokens=False)
    ids = (base * (args.n // max(len(base), 1) + 2))[: args.n]
    body, last = ids[:-1], ids[-1]

    cold = lm.make_cache()
    prefill(lm, body, cold, args.step)

    seed = lm.make_cache()
    prefill(lm, body[: args.boundary], seed, args.step)
    snap = capture_fragments(seed, args.boundary)
    warm = lm.make_cache()
    restore_fragments(warm, snap)
    mx.eval([c.state for c in warm])
    prefill(lm, body[args.boundary :], warm, args.step)

    same, why = bits_equal(flat(cold), flat(warm))

    t_cold = decode(lm, cold, last, args.decode_tokens)
    t_warm = decode(lm, warm, last, args.decode_tokens)

    # Determinism control: a THIRD cold cache, decoded the same way. If this
    # diverges from t_cold, decode is simply not run-to-run reproducible and the
    # warm/cold comparison cannot be read as a vault defect.
    ctrl = lm.make_cache()
    prefill(lm, body, ctrl, args.step)
    t_ctrl = decode(lm, ctrl, last, args.decode_tokens)

    def firstdiff(x, y):
        for i, (p, q) in enumerate(zip(x, y)):
            if p != q:
                return i
        return None if len(x) == len(y) else min(len(x), len(y))

    out = {
        "model": args.model, "n": args.n, "boundary": args.boundary,
        "step": args.step, "decode_tokens": args.decode_tokens,
        "cache_bit_identical": same, "cache_mismatch_reason": why,
        "tokens_cold": t_cold, "tokens_warm": t_warm, "tokens_cold_control": t_ctrl,
        "warm_vs_cold_first_divergence": firstdiff(t_warm, t_cold),
        "control_vs_cold_first_divergence": firstdiff(t_ctrl, t_cold),
        "warm_equals_cold": t_warm == t_cold,
        "control_equals_cold": t_ctrl == t_cold,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("tokens_")}, indent=1))


if __name__ == "__main__":
    main()
