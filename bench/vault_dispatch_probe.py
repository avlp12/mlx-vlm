#!/usr/bin/env python3
"""Capture exactly what dispatch hands generate_step in the cold vs restore arm.

Restore is already proven exact under direct drive (bit-identical cache AND
token-identical decode), yet the dispatch-level gate fails reproducibly. So the
difference is in what dispatch passes. This monkeypatches generate_step to
record its arguments for each arm without changing behaviour.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path
import mlx.core as mx


def _P(w):
    class P:
        def __init__(s):
            s.tokenizer = w._tokenizer; s.detokenizer = w.detokenizer; s._w = w
        def __call__(s, *a, **k): return s.tokenizer(*a, **k)
        def __getattr__(s, n): return getattr(w, n)
    return P()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["MLX_VLM_GLM5_VAULT_STRIDE"] = "2048"
    os.environ.pop("MLX_VLM_GLM5_VAULT", None)

    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import StoppingCriteria, get_model_path, load_model
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.generate import stream_generate
    from mlx_vlm.generate import ar as ar_mod
    from mlx_vlm.generate import dispatch as dispatch_mod
    from mlx_vlm.context_vault import reset_vault

    path = get_model_path(args.model)
    model = load_model(path, lazy=False)
    w = load_tokenizer(path)
    eos = getattr(model.config, "eos_token_id", None) or getattr(
        getattr(model.config, "text_config", None), "eos_token_id", None)
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = eos
    c = StoppingCriteria(eos, w); w.stopping_criteria = c; w._tokenizer.stopping_criteria = c
    processor = _P(w)

    tok = processor.tokenizer
    sids = tok.encode("Recomputation is the dominant cost in long-context serving. ",
                      add_special_tokens=False)
    doc = tok.decode((sids * (args.prompt_tokens // max(len(sids), 1) + 4))[: args.prompt_tokens])
    prompt = apply_chat_template(processor, model.config,
                                 doc + "\n\nSummarize the document in one sentence.", num_images=0)

    captured = []
    orig = dispatch_mod.generate_step

    def spy(input_ids, model_, pixel_values=None, mask=None, **kw):
        pc = kw.get("prompt_cache")
        offs = []
        if pc:
            for e in pc[:3]:
                subs = getattr(e, "caches", None)
                if subs is not None:
                    offs.append([int(getattr(s, "offset", -1)) for s in subs])
                else:
                    st = e.state
                    offs.append(f"arrays:{[None if a is None else list(a.shape) for a in st]}")
        caps = []
        if pc:
            for e in pc:
                subs = getattr(e, "caches", None)
                if subs is not None:
                    caps.append([None if s.keys is None else int(s.keys.shape[2]) for s in subs])
        captured.append({
            "input_ids_shape": list(input_ids.shape),
            "mask": None if mask is None else list(getattr(mask, "shape", ["nonarray"])),
            "pixel_values": pixel_values is not None,
            "n_cache_entries": len(pc) if pc else 0,
            "first3_offsets": offs,
            "kv_capacities_first5": caps[:5],
            "checkpoint_len": kw.get("prompt_cache_checkpoint_len"),
            "has_checkpoint_cb": kw.get("prompt_cache_checkpoint") is not None,
            "prefill_step_size": kw.get("prefill_step_size"),
            "kwargs_keys": sorted([k for k in kw if k != "prompt_cache"]),
        })
        return orig(input_ids, model_, pixel_values, mask, **kw)

    dispatch_mod.generate_step = spy

    def run(tag):
        ids = []
        for r in stream_generate(model, processor, prompt, max_tokens=args.max_tokens,
                                 temperature=0.0, top_p=1.0):
            if getattr(r, "token", None) is not None:
                ids.append(int(r.token))
        captured[-1]["arm"] = tag
        captured[-1]["tokens"] = ids
        return ids

    reset_vault(); os.environ.pop("MLX_VLM_GLM5_VAULT", None)
    a = run("A_cold_vault_off")
    os.environ["MLX_VLM_GLM5_VAULT"] = "1"; reset_vault()
    b = run("B_store")
    c2 = run("C_warm_restore")

    out = {"arms": captured,
           "A_eq_B": a == b, "A_eq_C": a == c2,
           "first_div_C_vs_A": next((i for i, (x, y) in enumerate(zip(c2, a)) if x != y), None)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    for r in captured:
        print(json.dumps({k: v for k, v in r.items() if k != "tokens"}, indent=1))
    print("A_eq_B", out["A_eq_B"], "A_eq_C", out["A_eq_C"], "first_div", out["first_div_C_vs_A"])


if __name__ == "__main__":
    main()
