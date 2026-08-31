#!/usr/bin/env python3
"""Warm Context Vault live bench: warm-hit TTFT vs cold prefill.

Protocol mirrors the DFlash2 repro harness (greedy, temperature 0, fresh
process, mx.reset_peak_memory per arm) so numbers are comparable to the
receipts in ~/glm53flash/prep/dflash2-repro/logs/.

Arms per context length:
  cold        vault off  -- the baseline every current receipt measures
  store       vault on, first sighting -- lays the boundary ladder
  warm        vault on, same document + a NEW suffix -- the workload the
              vault exists for: only the suffix should prefill

Usage:
  MLX_VLM_GLM5_VAULT=1 python bench/vault_bench.py \
      --model ~/glm53flash/builds/GLM-5.3-Flash-vlm-q4-quasar \
      --prompt-tokens 16384 32768 --out logs/vault.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import mlx.core as mx


def _git(cwd: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


class _TextProcessor:
    """generate()/prepare_inputs need the HF tokenizer, not the wrapper."""

    def __init__(self, wrapper):
        self.tokenizer = wrapper._tokenizer
        self.detokenizer = wrapper.detokenizer
        self._wrapper = wrapper

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._wrapper, name)


def _load(model_dir: str):
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import get_model_path, load_model

    path = get_model_path(model_dir)
    t0 = time.perf_counter()
    model = load_model(path, lazy=False)
    load_s = time.perf_counter() - t0
    wrapper = load_tokenizer(path)
    return model, _TextProcessor(wrapper), load_s


def _doc_text(tokenizer, n_tokens: int, seed_text: str) -> str:
    ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode("The quick brown fox. ", add_special_tokens=False)
    reps = (n_tokens // max(len(ids), 1)) + 4
    return tokenizer.decode((ids * reps)[:n_tokens])


def _prompt(processor, doc: str, question: str) -> str:
    from mlx_vlm.prompt_utils import apply_chat_template

    return apply_chat_template(
        processor, None, doc + "\n\n" + question, num_images=0
    )


def _run(model, processor, prompt: str, max_tokens: int) -> Dict[str, Any]:
    from mlx_vlm import generate

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    r = generate(
        model,
        processor,
        prompt,
        verbose=False,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    wall = time.perf_counter() - t0
    return {
        "wall_s": wall,
        "prompt_tokens": getattr(r, "prompt_tokens", None),
        "generation_tokens": getattr(r, "generation_tokens", None),
        "prompt_tps": getattr(r, "prompt_tps", None),
        "generation_tps": getattr(r, "generation_tps", None),
        "cached_tokens": getattr(r, "cached_tokens", None),
        "peak_memory_gb": getattr(r, "peak_memory", None),
        "text_preview": (getattr(r, "text", "") or "").replace("\n", " ")[:160],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, nargs="+", default=[16384])
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--doc", default=None, help="seed text file for the document")
    ap.add_argument("--src", default=str(Path.home() / "src/mlx-vlm-vault"))
    args = ap.parse_args()

    from mlx_vlm.context_vault import get_vault, vault_enabled, vault_identity_for_model

    seed = "In a distributed serving system the cost of recomputation dominates. "
    if args.doc and Path(args.doc).is_file():
        seed = Path(args.doc).read_text()[:200_000]

    model, processor, load_s = _load(args.model)
    tok = processor.tokenizer

    out: Dict[str, Any] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "src": args.src,
        "head": _git(Path(args.src), "rev-parse", "HEAD"),
        "branch": _git(Path(args.src), "rev-parse", "--abbrev-ref", "HEAD"),
        "vault_enabled": vault_enabled(),
        "vault_env": {
            k: v for k, v in os.environ.items() if k.startswith("MLX_VLM_GLM5_VAULT")
        },
        "load_seconds": load_s,
        "protocol": {
            "greedy": True,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "metric": "mlx_vlm.generate prompt_tps / cached_tokens",
        },
        "runs": {},
    }

    vault = get_vault(vault_identity_for_model(model)) if vault_enabled() else None

    for n in args.prompt_tokens:
        doc = _doc_text(tok, n, seed)
        key = f"p{n}"
        out["runs"][key] = {}

        # store: first sighting, lays the ladder
        p_store = _prompt(processor, doc, "Summarize the document in one sentence.")
        out["runs"][key]["store"] = _run(model, processor, p_store, args.max_tokens)
        if vault is not None:
            out["runs"][key]["store"]["vault_stats"] = vault.stats_dict()

        # warm: same document, DIFFERENT question -> only the suffix may prefill
        p_warm = _prompt(processor, doc, "List three key themes from the document.")
        out["runs"][key]["warm"] = _run(model, processor, p_warm, args.max_tokens)
        if vault is not None:
            out["runs"][key]["warm"]["vault_stats"] = vault.stats_dict()

        # repeat the warm arm to show steady-state (second hit on the same rung)
        p_warm2 = _prompt(processor, doc, "Give the document a title.")
        out["runs"][key]["warm2"] = _run(model, processor, p_warm2, args.max_tokens)
        if vault is not None:
            out["runs"][key]["warm2"]["vault_stats"] = vault.stats_dict()

        print(json.dumps({key: out["runs"][key]}, indent=1)[:1200], flush=True)

    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
