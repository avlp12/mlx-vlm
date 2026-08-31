#!/usr/bin/env python3
"""Model warmer A/B: first-request latency with and without a boot warm pass.

mx.load memory-maps the safetensors, so a cold process faults ~169 GB in during
its first forward pass. The pipeline campaign saw gesicht ramp 216 -> 380 tok/s
prefill across a cold run for exactly this reason. This measures whether paying
that cost at boot removes it from the first request.

Run twice in FRESH processes -- the page cache is the thing under test, so an
in-process A/B would measure nothing:

  python bench/warmer_bench.py --model <tree> --arm cold --out logs/warm_cold.json
  python bench/warmer_bench.py --model <tree> --arm warm --out logs/warm_warm.json

Between arms the page cache must be evicted or the second arm inherits the
first's warmth. `sudo purge` does it; without that, run the arms on different
boots or accept that `cold` is only cold the first time.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=["cold", "warm"], required=True)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mlx_vlm import generate
    from mlx_vlm.model_warmer import warm_model
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import get_model_path, load_model

    path = get_model_path(args.model)
    t0 = time.perf_counter()
    model = load_model(path, lazy=False)
    load_s = time.perf_counter() - t0

    wrapper = load_tokenizer(path)

    class _P:
        def __init__(self, w):
            self.tokenizer = w._tokenizer
            self.detokenizer = w.detokenizer
            self._w = w

        def __call__(self, *a, **k):
            return self.tokenizer(*a, **k)

        def __getattr__(self, n):
            return getattr(self._w, n)

    processor = _P(wrapper)

    warm_stats = None
    if args.arm == "warm":
        warm_stats = warm_model(model, verbose=True)

    ids = processor.tokenizer.encode("The system under test. ", add_special_tokens=False)
    doc = processor.tokenizer.decode((ids * (args.prompt_tokens // max(len(ids), 1) + 4))[: args.prompt_tokens])
    prompt = apply_chat_template(
        processor, model.config, doc, num_images=0, enable_thinking=False
    )

    mx.reset_peak_memory()
    t1 = time.perf_counter()
    r = generate(
        model, processor, prompt, verbose=False,
        max_tokens=args.max_tokens, temperature=0.0, top_p=1.0,
    )
    first_s = time.perf_counter() - t1

    # A second identical request: with the pages already faulted by request 1,
    # the cold arm should converge on the warm arm here. That convergence is the
    # control that says the delta really was page faults.
    t2 = time.perf_counter()
    r2 = generate(
        model, processor, prompt, verbose=False,
        max_tokens=args.max_tokens, temperature=0.0, top_p=1.0,
    )
    second_s = time.perf_counter() - t2

    out = {
        "arm": args.arm,
        "model": args.model,
        "load_seconds": load_s,
        "warm_stats": warm_stats,
        "first_request_s": first_s,
        "first_prompt_tps": getattr(r, "prompt_tps", None),
        "first_generation_tps": getattr(r, "generation_tps", None),
        "second_request_s": second_s,
        "second_prompt_tps": getattr(r2, "prompt_tps", None),
        "second_generation_tps": getattr(r2, "generation_tps", None),
        "prompt_tokens": getattr(r, "prompt_tokens", None),
        "peak_memory_gb": getattr(r, "peak_memory", None),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
