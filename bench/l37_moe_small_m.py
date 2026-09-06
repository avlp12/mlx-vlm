#!/usr/bin/env python3
"""L37: MoE small-M expert-GEMM dispatch policy -- the GPU micro arm.

Drives the REAL fork object (``mlx_vlm.models.switch_layers.SwitchGLU``) with
``MLX_VLM_MOE_SMALL_M_POLICY`` off and on, so what gets measured is the shipped
dispatch decision and not a re-implementation of it.  (This is why it lives in
the fork's ``bench/`` and not next to ``l28_prefill_gemm_efficiency.py``, which
deliberately re-implements the fork's gather convention in numpy so that it
never imports the fork.)

What it settles.  The kernel probe
(lls-kernels-run/docs/logs/glm53_kernels/{gesicht,epsilon}/moe_dispatch_probe.json)
established the cliff at rows/expert = 4 -- GLM-5.3-Flash E=288 top_k=8, so
tokens = 144 -- where MLX switches ``GatherQMM`` to ``gather_qmm_rhs`` and the
per-token cost JUMPS from 75.06 to 113.37 us.  Two cost models fit those same
points and disagree about where the policy should STOP:

  - pessimistic (each slab re-pays the full 288-expert weight traffic, i.e. the
    slab costs what k copies of the probe's own tokens=T/k point cost):
    crossover at rows/expert ~ 6.7
  - locality-aware (a contiguous slab of a SORTED batch spans ~1/k of the expert
    range, so the k slabs together touch each expert about once and only the k
    launches multiply): crossover at ~8.1-9.0

The shipped default is the pessimistic edge (MLX_VLM_MOE_SMALL_M_MAX_RPE=6.0).
This bench exists so the 6.0-8.0 stretch gets measured instead of modelled.

Run (GPU, on a box holding the measurement rail):

    PYTHONPATH=$PWD python3 bench/l37_moe_small_m.py --device gpu \
        --out /tmp/l37_moe_small_m

Run (CPU smoke -- tiny shapes; proves the harness, is NOT a measurement, and on
CPU there is no branch at all so every speedup should read ~1.0):

    PYTHONPATH=$PWD python3 bench/l37_moe_small_m.py --device cpu --quick \
        --out /tmp/l37_smoke
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import time
from typing import Optional

import mlx.core as mx

# GLM-5.3-Flash text_config (config.json; same dims the L28 shape table uses).
HIDDEN = 4096
MOE_INTER = 2048
EXPERTS = 288
TOP_K = 8
SPARSE_LAYERS = 42  # num_hidden_layers 45 - first_k_dense_replace 3

# Tokens straddling the band.  143/144 is the MLX branch flip; 216 is the default policy cut
# (rows/E = 6.0); 288 is rows/E = 8.0, the hard cap.  128 / 512 / 1024 are CONTROLS: the policy
# is inert there, so their off-vs-on spread is the harness noise floor.
DEFAULT_TOKENS = [128, 143, 144, 160, 192, 216, 224, 256, 288, 320, 512, 1024]
# QUICK: E=32 top_k=4 -> the band (4 <= rows/E < 8) is 32 <= tokens < 64.
QUICK = dict(hidden=128, inter=64, experts=32, top_k=4,
             tokens=[24, 32, 40, 56, 64, 96])


def _reload_switch_layers(policy: str, max_rpe: Optional[str]):
    os.environ["MLX_VLM_MOE_SMALL_M_POLICY"] = policy
    if max_rpe is None:
        os.environ.pop("MLX_VLM_MOE_SMALL_M_MAX_RPE", None)
    else:
        os.environ["MLX_VLM_MOE_SMALL_M_MAX_RPE"] = max_rpe
    import mlx_vlm.models.switch_layers as S

    return importlib.reload(S)


def _build(S, hidden, inter, experts, bits, group_size, mode):
    mx.random.seed(0)
    sw = S.SwitchGLU(hidden, inter, experts, bias=False)
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(sw, name, getattr(sw, name).to_quantized(
            group_size=group_size, bits=bits, mode=mode))
    mx.eval(sw.parameters())
    return sw


def _inputs(tokens, hidden, experts, top_k, dtype):
    mx.random.seed(1)
    x = mx.random.normal((1, tokens, hidden)).astype(dtype)
    # same routing distribution as the kernel probe: a fresh top_k draw per token
    idx = mx.argpartition(
        mx.random.normal((1, tokens, experts)), kth=top_k - 1, axis=-1
    )[..., :top_k].astype(mx.int32)
    mx.eval(x, idx)
    return x, idx


def _time(fn, warmup, iters):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    ap.add_argument("--quick", action="store_true",
                    help="tiny shapes; harness smoke test, not a measurement")
    ap.add_argument("--tokens", type=int, nargs="+", default=None)
    ap.add_argument("--max-rpe", default="8.0",
                    help="policy cap for the ON arm; 8.0 measures the whole k=2 regime, "
                         "including the 6.0-8.0 stretch the shipped default declines")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--mode", default="affine")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    if args.quick:
        hidden, inter = QUICK["hidden"], QUICK["inter"]
        experts, top_k = QUICK["experts"], QUICK["top_k"]
        tokens = args.tokens or QUICK["tokens"]
        dtype = mx.float32
    else:
        hidden, inter, experts, top_k = HIDDEN, MOE_INTER, EXPERTS, TOP_K
        tokens = args.tokens or DEFAULT_TOKENS
        dtype = mx.bfloat16

    records = []
    for policy, max_rpe in (("off", None), ("auto", args.max_rpe)):
        S = _reload_switch_layers(policy, max_rpe)
        sw = _build(S, hidden, inter, experts, args.bits, args.group_size, args.mode)
        for t in tokens:
            x, idx = _inputs(t, hidden, experts, top_k, dtype)
            rows = t * top_k
            k = S._small_m_slabs(rows, experts)
            per = _time(lambda: sw(x, idx), args.warmup, args.iters)
            records.append({
                "policy": policy,
                "max_rpe": max_rpe,
                "tokens": t,
                "rows": rows,
                "rows_per_expert": rows / experts,
                "slabs": k,
                "ms": per * 1000,
                "us_per_token": per * 1e6 / t,
                # what MLX dispatches on the WHOLE batch when we leave it alone
                # (mlx/backend/metal/quantized.cpp: M==1 && B>=16 && sorted && B/E>=4)
                "mlx_branch": ("gather_qmm_rhs"
                               if (rows >= 16 and rows // experts >= 4)
                               else "gather_qmv"),
                "est_forward_ms_42L": per * 1000 * SPARSE_LAYERS,
            })

    by = {(r["policy"], r["tokens"]): r for r in records}
    verdict = []
    for t in tokens:
        off, on = by[("off", t)], by[("auto", t)]
        verdict.append({
            "tokens": t,
            "rows_per_expert": off["rows_per_expert"],
            "slabs_on": on["slabs"],
            "off_ms": off["ms"],
            "on_ms": on["ms"],
            "speedup": off["ms"] / on["ms"],
            # PRE-REGISTERED: where slabs_on == 1 the two arms are the SAME code path, so
            # |speedup - 1| there is pure harness noise. A win inside the band only counts
            # if it is several times that floor.
            "is_control": on["slabs"] == 1,
        })
    controls = [abs(v["speedup"] - 1) for v in verdict if v["is_control"]]
    out = {
        "meta": {
            "mlx_version": mx.__version__,
            "device": str(mx.default_device()),
            "host": platform.node(),
            "quick": bool(args.quick),
            "config": {"hidden": hidden, "moe_inter": inter, "experts": experts,
                       "top_k": top_k, "bits": args.bits,
                       "group_size": args.group_size, "mode": args.mode},
            "iters": args.iters, "warmup": args.warmup,
            "max_rpe": args.max_rpe,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "records": records,
        "verdict": verdict,
        "control_noise_floor_pct": (max(controls) * 100) if controls else None,
    }
    if args.out:
        path = args.out if args.out.endswith(".json") else args.out + ".json"
        with open(path, "w") as fh:
            fh.write(json.dumps(out, indent=1))
        print(f"wrote {path}")
    for v in verdict:
        tag = "control" if v["is_control"] else f"k={v['slabs_on']}"
        print(f"  tokens={v['tokens']:>5} rpe={v['rows_per_expert']:>5.2f} {tag:>8} "
              f"off={v['off_ms']:>9.4f} ms  on={v['on_ms']:>9.4f} ms  "
              f"speedup={v['speedup']:.4f}")
    if controls:
        print(f"  control noise floor: {max(controls) * 100:.2f}%")


if __name__ == "__main__":
    main()
