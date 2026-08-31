#!/usr/bin/env python3
"""Bisect the dispatch divergence: compare dispatch's OWN cache at decode start.

Direct drive is proven exact, so the remaining question is whether dispatch's
restore arm reaches the decode loop holding the same state as dispatch's cold
arm. Everything upstream (kwargs, sampler, checkpoint wiring) was already shown
identical by vault_dispatch_probe.py.

Method: spy on generate_step to grab the live prompt_cache object per arm, run
with max_tokens=1 so the cache lands at the same offset, then bit-compare A vs C
and dump the first decode step's top-5 logits.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path
import mlx.core as mx


def _P(w):
    class P:
        def __init__(s):
            s.tokenizer = w._tokenizer; s.detokenizer = w.detokenizer
        def __call__(s, *a, **k): return s.tokenizer(*a, **k)
        def __getattr__(s, n): return getattr(w, n)
    return P()


def flat(cache):
    out = []
    def go(o):
        if isinstance(o, mx.array): out.append(o)
        elif isinstance(o, (list, tuple)):
            for x in o: go(x)
    for c in cache: go(c.state)
    return out


def offsets(cache):
    res = []
    for e in cache:
        subs = getattr(e, "caches", None)
        if subs is not None:
            res.append([int(getattr(s, "offset", -1)) for s in subs])
        else:
            res.append("arrays")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["MLX_VLM_GLM5_VAULT_STRIDE"] = "2048"
    os.environ.pop("MLX_VLM_GLM5_VAULT", None)

    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import StoppingCriteria, get_model_path, load_model
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.generate import stream_generate
    from mlx_vlm.generate import dispatch as dm
    from mlx_vlm.context_vault import reset_vault

    path = get_model_path(args.model)
    model = load_model(path, lazy=False)
    w = load_tokenizer(path)
    eos = getattr(model.config, "eos_token_id", None) or getattr(
        getattr(model.config, "text_config", None), "eos_token_id", None)
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = eos
    sc = StoppingCriteria(eos, w); w.stopping_criteria = sc; w._tokenizer.stopping_criteria = sc
    proc = _P(w)
    tok = proc.tokenizer

    sids = tok.encode("Recomputation is the dominant cost in long-context serving. ",
                      add_special_tokens=False)
    doc = tok.decode((sids * (args.prompt_tokens // max(len(sids), 1) + 4))[: args.prompt_tokens])
    prompt = apply_chat_template(proc, model.config,
                                 doc + "\n\nSummarize the document in one sentence.", num_images=0)

    grabbed = {}
    orig = dm.generate_step

    def spy(input_ids, model_, pixel_values=None, mask=None, **kw):
        grabbed["cache"] = kw.get("prompt_cache")
        grabbed["in_shape"] = list(input_ids.shape)
        return orig(input_ids, model_, pixel_values, mask, **kw)

    dm.generate_step = spy

    def run(tag):
        ids = []
        for r in stream_generate(model, proc, prompt, max_tokens=args.max_tokens,
                                 temperature=0.0, top_p=1.0):
            if getattr(r, "token", None) is not None:
                ids.append(int(r.token))
        c = grabbed["cache"]
        snap = [mx.array(a) for a in flat(c)]
        mx.eval(snap)
        return {"tag": tag, "tokens": ids, "in_shape": grabbed["in_shape"],
                "offsets_first3": offsets(c)[:3], "n_components": len(snap)}, snap

    reset_vault(); os.environ.pop("MLX_VLM_GLM5_VAULT", None)
    a_meta, a_snap = run("A_cold")
    os.environ["MLX_VLM_GLM5_VAULT"] = "1"; reset_vault()
    b_meta, b_snap = run("B_store")
    c_meta, c_snap = run("C_restore")

    def cmp(x, y):
        if len(x) != len(y): return {"equal": False, "why": "count"}
        diffs = []
        for i, (p, q) in enumerate(zip(x, y)):
            if p.shape != q.shape:
                diffs.append({"i": i, "why": "shape", "a": list(p.shape), "b": list(q.shape)})
                continue
            pb = mx.view(p.flatten(), mx.uint8); qb = mx.view(q.flatten(), mx.uint8)
            if not bool(mx.all(pb == qb).item()):
                d = mx.abs(p.astype(mx.float32) - q.astype(mx.float32))
                diffs.append({"i": i, "why": "bytes", "shape": list(p.shape),
                              "max_abs": float(mx.max(d).item())})
        return {"equal": not diffs, "n_diff": len(diffs), "first5": diffs[:5]}

    out = {"prompt_tokens": args.prompt_tokens, "max_tokens": args.max_tokens,
           "A": a_meta, "B": b_meta, "C": c_meta,
           "cache_C_vs_A": cmp(c_snap, a_snap),
           "cache_B_vs_A": cmp(b_snap, a_snap),
           "tokens_equal_C_vs_A": a_meta["tokens"] == c_meta["tokens"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:2500])


if __name__ == "__main__":
    main()
