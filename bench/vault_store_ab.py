#!/usr/bin/env python3
"""Vault gate (ii): paired in-process store-cost A/B, alternating arms.

Settles the 16k store anomaly (1.86x) against the 32k result (0.99x). Both arms
run in ONE process, alternating off/on/off/on..., so page-cache warmth, thermal
state, and any background campaign contention hit both arms equally -- the
failure mode of the earlier separate-process runs.

Each ON arm clears the vault first so it is always a genuine cold store, never
a hit. Each arm uses a DIFFERENT document (same length) so an OFF arm can never
be silently served by a rung left from a previous ON arm.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx


class _P:
    def __init__(self, w):
        self.tokenizer = w._tokenizer
        self.detokenizer = w.detokenizer
        self._w = w

    def __call__(self, *a, **k):
        return self.tokenizer(*a, **k)

    def __getattr__(self, n):
        return getattr(self._w, n)


def _load(model_dir):
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import StoppingCriteria, get_model_path, load_model

    path = get_model_path(model_dir)
    model = load_model(path, lazy=False)
    w = load_tokenizer(path)
    eos = getattr(model.config, "eos_token_id", None)
    if eos is None:
        tc = getattr(model.config, "text_config", None)
        eos = getattr(tc, "eos_token_id", None) if tc is not None else None
    if eos is None:
        eos = getattr(w, "eos_token_id", None)
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = eos
    c = StoppingCriteria(eos, w)
    w.stopping_criteria = c
    w._tokenizer.stopping_criteria = c
    return model, _P(w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--stride", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["MLX_VLM_GLM5_VAULT_STRIDE"] = str(args.stride)
    from mlx_vlm.context_vault import (
        boundary_ladder, get_vault, reset_vault, vault_identity_for_model,
    )
    from mlx_vlm.generate import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = _load(args.model)
    tok = processor.tokenizer

    def make_prompt(tag: str) -> str:
        seed = f"Document {tag}: recomputation dominates long-context serving cost. "
        sids = tok.encode(seed, add_special_tokens=False)
        doc = tok.decode((sids * (args.prompt_tokens // max(len(sids), 1) + 4))[: args.prompt_tokens])
        return apply_chat_template(
            processor, model.config, doc + "\n\nSummarize it.", num_images=0
        )

    def run(prompt):
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        r = generate(
            model, processor, prompt, verbose=False,
            max_tokens=args.max_tokens, temperature=0.0, top_p=1.0,
        )
        return {
            "wall_s": time.perf_counter() - t0,
            "prompt_tokens": r.prompt_tokens,
            "prompt_tps": r.prompt_tps,
            "generation_tps": r.generation_tps,
            "cached_tokens": r.cached_tokens,
            "peak_memory_gb": r.peak_memory,
        }

    rows = []
    # Untimed warmup so arm 1 does not absorb one-time graph/kernel compilation.
    os.environ.pop("MLX_VLM_GLM5_VAULT", None)
    run(make_prompt("warmup"))

    for i in range(args.reps):
        os.environ.pop("MLX_VLM_GLM5_VAULT", None)
        reset_vault()
        off = run(make_prompt(f"off{i}"))
        off["arm"], off["rep"] = "off", i
        rows.append(off)

        os.environ["MLX_VLM_GLM5_VAULT"] = "1"
        reset_vault()
        on = run(make_prompt(f"on{i}"))
        v = get_vault(vault_identity_for_model(model))
        on["arm"], on["rep"] = "on", i
        on["vault_stats"] = v.stats_dict()
        rows.append(on)
        print(f"rep {i}: off {off['wall_s']:.2f}s ({off['prompt_tps']:.1f} tok/s)  "
              f"on {on['wall_s']:.2f}s ({on['prompt_tps']:.1f} tok/s)  "
              f"rungs={on['vault_stats']['rungs_resident']}", flush=True)

    offs = [r["wall_s"] for r in rows if r["arm"] == "off"]
    ons = [r["wall_s"] for r in rows if r["arm"] == "on"]
    off_tps = [r["prompt_tps"] for r in rows if r["arm"] == "off"]
    on_tps = [r["prompt_tps"] for r in rows if r["arm"] == "on"]
    ntok = rows[0]["prompt_tokens"]
    out = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "prompt_tokens": ntok,
        "reps": args.reps,
        "stride": args.stride,
        "ladder": boundary_ladder(ntok, args.stride, 2048),
        "rows": rows,
        "median_off_wall_s": statistics.median(offs),
        "median_on_wall_s": statistics.median(ons),
        "median_off_prompt_tps": statistics.median(off_tps),
        "median_on_prompt_tps": statistics.median(on_tps),
        "store_overhead_x": statistics.median(ons) / statistics.median(offs),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in (
        "ladder", "median_off_wall_s", "median_on_wall_s",
        "median_off_prompt_tps", "median_on_prompt_tps", "store_overhead_x")}, indent=1))


if __name__ == "__main__":
    main()
