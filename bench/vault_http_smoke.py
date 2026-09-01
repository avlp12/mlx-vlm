#!/usr/bin/env python3
"""Warm Context Vault over HTTP: does the SERVER's request path use it?

``bench/vault_bench.py`` measures the vault through ``generate()``, which is
the path that already had it.  This measures it through the server, which until
now did not: the continuous-batching ``BatchGenerator`` built its own prompt
cache and never consulted a vault.

FOUR arms, TWO server processes, paired.  The pairing is the whole point: the
server also has an Automatic Prefix Cache, and two prompts that share a 16k
prefix hit it whether or not a vault exists.  A first version of this harness
compared a fresh-process cold request against a same-process second request and
measured 117x -- almost all of which was APC.  So:

  process 1, vault OFF
    cold           first request.  The baseline.
    apc_only_warm  same document, NEW question.  THE CONTROL: whatever the
                   prefix cache alone can do.
  process 2, vault ON
    store          first request.  Lays the boundary ladder.
    warm           same document, NEW question.

The vault's contribution is ``apc_only_warm`` vs ``warm`` -- same process shape,
same request pair, one toggle apart.  ``cold`` vs ``warm`` is reported too, but
it is the number that flatters, and it is labelled as such.

Each pair shares one process on purpose: the vault is in-process, so a restart
would throw the ladder away and ``warm`` would measure ``store`` again.

Reported per arm: TTFT (streamed first token), total wall, and the prompt-token
count the server reports back, so a warm hit is visible in the accounting and
not only in the clock.

  MLX_VLM_GLM5_VAULT is set per arm by this script -- do not set it outside.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

SRC = os.environ.get("MLXVLM_SRC", "/Users/gesicht/src/mlx-vlm-tp2serve")
sys.path.insert(0, SRC)

from mlx_vlm.tp.fleet import (  # noqa: E402
    HeavyRunActive, memory_snapshot, phys_footprint_gb, require_quiet_fleet,
)

M = os.environ.get("MODEL_PATH",
                   "/Users/gesicht/glm53flash/builds/GLM-5.3-Flash-vlm-q4-quasar")
PORT = int(os.environ.get("PORT", "8137"))
BASE = f"http://127.0.0.1:{PORT}"
DOC_TOKENS = int(os.environ.get("DOC_TOKENS", "16384"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32"))
OUT_DIR = os.environ.get("OUT_DIR", "/Users/gesicht/glm53flash/logs/sprint2")
TAG = os.environ.get("TAG") or time.strftime("%m%d_%H%M%S")
OUT = f"{OUT_DIR}/vault_http_smoke_{TAG}.json"
LOG = f"{OUT_DIR}/vault_http_smoke_{TAG}_server.log"
GATE = os.environ.get("SKIP_GATE") != "1"

Q1 = "\n\nQuestion: in one sentence, what is the file above mostly about?"
Q2 = "\n\nQuestion: in one sentence, name one risk the text above describes."


def _doc(n_tokens: int) -> str:
    """Real source and prose, tokenized once, truncated to n_tokens.

    Never a repeated sentence: a degenerate prompt collapses MoE routing and
    would make every number here optimistic (campaign law, and the lc_curve
    harness broke it once already).
    """
    from mlx_vlm.utils import get_model_path, load_tokenizer

    tok = load_tokenizer(get_model_path(M))._tokenizer
    root = os.path.join(SRC, "mlx_vlm")
    files = []
    for dirpath, _d, names in os.walk(root):
        if "test" in dirpath or "__pycache__" in dirpath:
            continue
        files.extend(
            os.path.join(dirpath, nm) for nm in sorted(names)
            if nm.endswith((".py", ".md"))
        )
    files.sort()
    parts, total = [], 0
    for fp in files:
        try:
            parts.append(open(fp, encoding="utf-8", errors="replace").read())
        except OSError:
            continue
        total += len(parts[-1])
        if total > n_tokens * 6:
            break
    ids = tok.encode("\n\n".join(parts), add_special_tokens=False)[:n_tokens]
    return tok.decode(ids)


def stream(prompt: str, max_tokens: int) -> dict:
    body = {"model": M, "stream": True, "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, parts, usage, n = None, [], None, 0
    r = urllib.request.urlopen(req, timeout=1800)
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ch = json.loads(line[6:])
        if ch.get("usage"):
            usage = ch["usage"]
        d = (ch.get("choices") or [{}])[0].get("delta", {}) or {}
        piece = d.get("content") or d.get("reasoning_content") or ""
        if piece:
            if ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(piece)
            n += 1
    total = time.perf_counter() - t0
    return {"ttft_s": round(ttft if ttft is not None else total, 3),
            "total_s": round(total, 3), "chunks": n,
            "usage": usage, "text": "".join(parts)[:200]}


def serve(vault_on: bool, log_suffix: str):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    env["MLX_VLM_GLM5_FUSED_KDA"] = "1"
    env.pop("KV_BITS", None)
    env.pop("MLX_VLM_GLM5_TP_HOSTS", None)      # single box; TP vault stays off
    if vault_on:
        env["MLX_VLM_GLM5_VAULT"] = "1"
    else:
        env.pop("MLX_VLM_GLM5_VAULT", None)
    cmd = [sys.executable, "-m", "mlx_vlm.server", "--model", M,
           "--port", str(PORT)]
    lf = open(LOG + log_suffix, "w")
    p = subprocess.Popen(cmd, cwd=SRC, env=env, stdout=lf, stderr=lf)
    t0 = time.perf_counter()
    for _ in range(300):
        try:
            urllib.request.urlopen(f"{BASE}/v1/models", timeout=5).read()
            break
        except Exception:
            if p.poll() is not None:
                raise RuntimeError(f"server exited during startup, see {LOG}{log_suffix}")
            time.sleep(2)
    else:
        raise RuntimeError("server never came up")
    return p, round(time.perf_counter() - t0, 1), " ".join(cmd[2:])


def shutdown(p):
    """Clean verb first: SIGINT, then wait. A bare kill leaks wired memory."""
    if p.poll() is not None:
        return p.returncode
    p.send_signal(2)
    try:
        p.wait(timeout=180)
    except subprocess.TimeoutExpired:
        p.terminate()
        p.wait(timeout=60)
    return p.returncode


res = {"harness": "bench/vault_http_smoke.py", "tag": TAG,
       "started": time.strftime("%FT%T%z"), "model": M,
       "doc_tokens": DOC_TOKENS, "max_tokens": MAX_TOKENS,
       "argv": sys.argv, "hostname": os.uname().nodename,
       "git_head": subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip(),
       "mem_before": memory_snapshot(), "arms": {}}

if GATE:
    try:
        res["fleet_preflight"] = require_quiet_fleet(
            ["10.0.0.1", "10.0.0.2"], label="vault http smoke")
    except HeavyRunActive as e:
        res["error"] = str(e)
        json.dump(res, open(OUT, "w"), indent=1)
        print(json.dumps(res, indent=1))
        raise SystemExit(75)

doc = _doc(DOC_TOKENS)
res["doc_chars"] = len(doc)

srv = None


def _pair(vault_on, first_name, second_name, suffix):
    global srv
    srv, startup, cmdline = serve(vault_on, suffix)
    res.setdefault("cmd", cmdline)
    res["arms"][first_name] = {
        "vault": vault_on, "startup_s": startup,
        "footprint_gb": round(phys_footprint_gb(srv.pid), 2)}
    stream("Say ok.", 8)                                    # warm the kernels
    res["arms"][first_name]["run"] = stream(doc + Q1, MAX_TOKENS)
    print(f"[{first_name}]", json.dumps(res["arms"][first_name]["run"])[:180],
          flush=True)
    res["arms"][second_name] = {"vault": vault_on, "same_process_as": first_name}
    res["arms"][second_name]["run"] = stream(doc + Q2, MAX_TOKENS)
    print(f"[{second_name}]", json.dumps(res["arms"][second_name]["run"])[:180],
          flush=True)
    res["arms"][second_name]["exit"] = shutdown(srv)
    srv = None


try:
    _pair(False, "cold", "apc_only_warm", ".novault")
    time.sleep(5)
    _pair(True, "store", "warm", ".vault")
except Exception as e:                                       # noqa: BLE001
    import traceback
    res["error"] = f"{type(e).__name__}: {e}"[:400]
    res["traceback"] = traceback.format_exc()[-1500:]
    print(res["traceback"], flush=True)
finally:
    if srv is not None:
        res.setdefault("arms", {}).setdefault("aborted", {})["exit"] = shutdown(srv)
    time.sleep(3)
    res["mem_after"] = memory_snapshot()

def _t(name):
    return res["arms"].get(name, {}).get("run", {}).get("ttft_s")


c, a, st, w = _t("cold"), _t("apc_only_warm"), _t("store"), _t("warm")
if c and w and a:
    res["ttft_summary"] = {
        "cold_s": c, "apc_only_warm_s": a, "store_s": st, "warm_s": w,
        "THE_NUMBER_vault_vs_apc_only": round(a / w, 3),
        "flattering_warm_vs_cold": round(c / w, 2),
        "note": "THE_NUMBER is the only one that isolates the vault. "
                "warm_vs_cold conflates the vault with the prefix cache and "
                "with being the second request in a live process.",
    }
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res.get("ttft_summary", {"no_summary": True}), indent=1))
print("WROTE", OUT, flush=True)
