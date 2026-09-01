#!/usr/bin/env python3
"""MoE prefill GEMM toggle: paired REAL-TEXT prefill A/B.

Campaign law: never judge this toggle on a PAD template.  The previous round
measured -0.86% on a repeated PAD sentence and +2.14% on real code, and traced
the difference to routing -- PAD activates 158-253 of 288 experts per chunk
while real text activates 281-285, which is the regime the padding-aware tile
choice was designed for.  So the prompt here is real source code and prose, and
the PAD number is not the number.

One arm per invocation (the toggle is read at import); the caller interleaves.
"""
import json, os, sys, time

SRC = os.environ.get("MLXVLM_SRC", "/Users/gesicht/src/mlx-vlm-moegemm")
sys.path.insert(0, SRC)
os.environ.setdefault("MLX_VLM_GLM5_FUSED_KDA", "1")
os.environ.pop("MLX_VLM_GLM5_TP_HOSTS", None)

import mlx.core as mx
from mlx_vlm.generate import wired_limit
from mlx_vlm.models.glm5_next.moe_gemm import moe_gemm_enabled
from mlx_vlm.utils import get_model_path, load_model, load_tokenizer

M = os.environ.get("MODEL_PATH",
                   "/Users/gesicht/glm53flash/builds/GLM-5.3-Flash-vlm-q4-quasar")
TAG = os.environ.get("TAG") or time.strftime("%m%d_%H%M%S")
ARM = os.environ.get("ARM", "off")
OUT = os.environ.get("OUT", f"/Users/gesicht/glm53flash/logs/tp2/moegemm_realtext_{ARM}_{TAG}.json")
CHUNK = int(os.environ.get("CHUNK", "2048"))
NTOK = int(os.environ.get("PROMPT_TOKENS", "8192"))
REPS = int(os.environ.get("REPS", "4"))

# Real text: this repository's own source plus prose.  Not a template.
parts = []
for f in ("mlx_vlm/models/glm5_next/language.py",
          "mlx_vlm/models/glm5_next/moe_gemm.py",
          "mlx_vlm/utils.py"):
    try:
        parts.append(open(os.path.join(SRC, f), encoding="utf-8",
                          errors="replace").read())
    except OSError:
        pass
try:
    parts.append(open("/Users/gesicht/glm53flash/prep/dflash2-repro/natural_doc.txt",
                      encoding="utf-8", errors="replace").read())
except OSError:
    pass
text = "\n\n".join(parts)

tok = load_tokenizer(get_model_path(M))._tokenizer
ids = tok.encode(text, add_special_tokens=False)[:NTOK]
assert len(ids) >= NTOK, f"only {len(ids)} real tokens available, need {NTOK}"

model = load_model(get_model_path(M), lazy=False)
lm = model.language_model
wire = wired_limit(model, [mx.default_stream(mx.default_device())])
wire.__enter__()

res = {"arm": ARM, "tag": TAG, "toggle_enabled": bool(moe_gemm_enabled(lm.config)),
       "env_MOE_GEMM": os.environ.get("MLX_VLM_GLM5_MOE_GEMM"),
       "prompt_tokens": len(ids), "chunk": CHUNK, "reps": REPS,
       "text_kind": "real source code + prose (NOT a PAD template)",
       "started": time.strftime("%FT%T%z")}
print(f"[moegemm {ARM}] toggle_enabled={res['toggle_enabled']} "
      f"prompt={len(ids)} tok", flush=True)


def prefill_once():
    cache = lm.make_cache()
    mx.synchronize(); t = time.perf_counter()
    for off in range(0, len(ids), CHUNK):
        n = min(CHUNK, len(ids) - off)
        x = mx.array([ids[off:off + n]], dtype=mx.int32)
        out = lm(x, cache=cache)
        mx.eval(out.logits)
        mx.eval([c.state for c in cache])
    mx.synchronize()
    return time.perf_counter() - t


prefill_once()                      # warm: pipelines, not the compiler
runs = [prefill_once() for _ in range(REPS)]
runs.sort()
med = runs[len(runs) // 2]
res["run_s"] = [round(r, 4) for r in runs]
res["median_s"] = round(med, 4)
res["prefill_tok_s"] = round(len(ids) / med, 1)
res["spread_pct"] = round(100 * (runs[-1] - runs[0]) / runs[0], 2)
print(f"[moegemm {ARM}] median {med:.3f}s -> {res['prefill_tok_s']} tok/s "
      f"(spread {res['spread_pct']}%)", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("WROTE", OUT, flush=True)
try:
    wire.__exit__(None, None, None)
except Exception:
    pass
