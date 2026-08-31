#!/usr/bin/env python3
"""Vault gate (i): warm-restore must be token-for-token identical to cold prefill.

The vault is process-resident RAM, so a literal "restart the process" test would
just measure a cold miss -- there is no disk tier yet. The equivalent, and
stronger, in-process form:

  A  vault OFF                 -> 64 greedy tokens   (the reference)
  B  vault ON, first sighting  -> 64 greedy tokens   (stores the ladder)
  C  vault ON, same prompt     -> 64 greedy tokens   (restores + tail prefill)

Gate: token_ids(C) == token_ids(A), element by element. C is the path that
restores KDA recurrent state and DSA latents from a snapshot and re-prefills
only the tail, so equality is the real proof that boundary restore is exact.
B == A is reported too; it isolates whether merely *capturing* perturbs decode.
"""
from __future__ import annotations

import argparse
import json
import os
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


def _gen_ids(model, processor, prompt, max_tokens):
    """Greedy decode, collecting raw token ids (text can hide a divergence)."""
    from mlx_vlm.generate import stream_generate

    ids, tps, cached = [], None, None
    t0 = time.perf_counter()
    for r in stream_generate(
        model, processor, prompt, max_tokens=max_tokens, temperature=0.0, top_p=1.0
    ):
        if getattr(r, "token", None) is not None:
            ids.append(int(r.token))
        tps = getattr(r, "generation_tps", tps)
        cached = getattr(r, "cached_tokens", cached)
    return {
        "ids": ids,
        "n": len(ids),
        "wall_s": time.perf_counter() - t0,
        "generation_tps": tps,
        "cached_tokens": cached,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--stride", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.pop("MLX_VLM_GLM5_VAULT", None)
    os.environ["MLX_VLM_GLM5_VAULT_STRIDE"] = str(args.stride)

    from mlx_vlm.context_vault import get_vault, reset_vault, vault_identity_for_model
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = _load(args.model)
    tok = processor.tokenizer
    seed = "Recomputation is the dominant cost in long-context serving. "
    sids = tok.encode(seed, add_special_tokens=False)
    doc = tok.decode((sids * (args.prompt_tokens // max(len(sids), 1) + 4))[: args.prompt_tokens])
    prompt = apply_chat_template(
        processor, model.config, doc + "\n\nSummarize the document in one sentence.", num_images=0
    )

    out = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "prompt_tokens_target": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "stride": args.stride,
        "arms": {},
    }

    # A: reference, vault off
    reset_vault()
    os.environ.pop("MLX_VLM_GLM5_VAULT", None)
    out["arms"]["A_cold_vault_off"] = _gen_ids(model, processor, prompt, args.max_tokens)

    # B/C: vault on -- first sighting stores, second restores
    os.environ["MLX_VLM_GLM5_VAULT"] = "1"
    reset_vault()
    out["arms"]["B_store"] = _gen_ids(model, processor, prompt, args.max_tokens)
    v = get_vault(vault_identity_for_model(model))
    out["vault_after_store"] = v.stats_dict()
    out["arms"]["C_warm_restore"] = _gen_ids(model, processor, prompt, args.max_tokens)
    out["vault_after_warm"] = v.stats_dict()

    a = out["arms"]["A_cold_vault_off"]["ids"]
    b = out["arms"]["B_store"]["ids"]
    c = out["arms"]["C_warm_restore"]["ids"]

    def cmp(x, y):
        if len(x) != len(y):
            return {"equal": False, "reason": f"length {len(x)} vs {len(y)}"}
        bad = [i for i, (p, q) in enumerate(zip(x, y)) if p != q]
        return {
            "equal": not bad,
            "n": len(x),
            "first_divergence": (bad[0] if bad else None),
            "n_divergent": len(bad),
        }

    out["gate"] = {
        "C_vs_A_warm_restore_identity": cmp(c, a),
        "B_vs_A_capture_is_nonperturbing": cmp(b, a),
    }
    out["verdict"] = "PASS" if out["gate"]["C_vs_A_warm_restore_identity"]["equal"] else "FAIL"
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("gate", "verdict", "vault_after_warm")}, indent=1))
    print(f"cached_tokens C={c and out['arms']['C_warm_restore']['cached_tokens']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
