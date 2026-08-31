#!/usr/bin/env python3
"""Live glm5_next MoE prefill probe: routing histograms + in-situ MoE share.

Two modes, one model load each.

``--mode hist``
    Record the per-layer, per-chunk expert-occupancy histogram that the real
    router produces during a real prefill, for several prompt kinds.  The
    padding win predicted in bench/README.md is
    ``(1 + 16*(A-1)/N) / (1 + sum_e ceil(c_e/R)*R / N - 1)`` where ``A`` is the
    number of *active* experts in the chunk -- both terms are read off the
    histogram, so this either confirms or kills the model on live data.

``--mode ablate``
    Four arms through the same ``generate()`` the A/B harness uses, so the live
    -0.94% can be decomposed:
      A  stock                          baseline
      B  stock + the plan/sync the tiled path forces, result discarded
                                        -> cost of the host syncs alone
      C  toggle ON (padded layout)      -> total
      D  routed experts -> zeros        -> in-situ routed-expert share
    D is a timing ablation only; its output is numerically meaningless.

Prompts: the harness's own repetitive PAD template (the exact A/B workload),
plus a real source file and a real prose document.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

HARNESS = Path(
    "/Users/gesicht/local-llm-serving/.claude/worktrees/frosty-engelbart-0997b4"
    "/ports/glm53flash-mlx/serving/repro_mlxvlm_dflash2.py"
)


def foreign_busy():
    """Any foreign heavy python (>8 GiB RSS) other than this process."""
    import os, subprocess

    me = os.getpid()
    try:
        out = subprocess.check_output(["ps", "-Ao", "pid,rss,args"], text=True)
    except Exception:
        return ""
    hits = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, rss, args = parts
        if int(pid) == me or "python" not in args.lower():
            continue
        if int(rss) > 8 * 1024 * 1024:
            hits.append(f"{pid}:{args.split()[0].rsplit('/', 1)[-1]}")
    return ",".join(hits)


def _load_harness():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_harness", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def counts_of(indices, num_experts):
    flat = indices.flatten().astype(mx.int32)
    return (
        mx.zeros((num_experts,), mx.int32).at[flat].add(mx.ones((flat.size,), mx.int32))
    )


def stock_passes(counts, bm=16):
    """Exact number of bm-row block-gemms affine_gather_qmm_rhs runs.

    Expert e occupying sorted rows [o, o+c) touches
    floor((o+c-1)/bm) - floor(o/bm) + 1 blocks, and the kernel re-runs a full
    block-gemm for each (block, expert) pair.  Validated to 3.3% mean abs error
    against a bm=16/32/64 core sweep (bench/stage3_core_bm.json); the
    "1 + bm/(2*rows_per_expert)" form is 16.9% and mispredicts every case where
    the counts are bm-aligned.
    """
    total, o = 0, 0
    for c in counts:
        if c:
            total += (o + c - 1) // bm - o // bm + 1
            o += c
    return total


def waste_model(counts, rows_per_tile):
    """(stock waste, padded waste, active experts, stock block-gemms)."""
    n = sum(counts)
    active = sum(1 for c in counts if c > 0)
    stock = stock_passes(counts, 16) * 16 / n
    padded = (
        sum((c + rows_per_tile - 1) // rows_per_tile for c in counts)
        * rows_per_tile
        / n
    )
    return stock, padded, active, stock_passes(counts, 16)


def _switch_class(model):
    for attr in ("language_model", "model"):
        m = getattr(model, attr, None)
        if m is None:
            continue
        layers = getattr(getattr(m, "model", m), "layers", None) or getattr(
            m, "layers", None
        )
        if layers:
            for l in layers:
                sm = getattr(getattr(l, "mlp", None), "switch_mlp", None)
                if sm is not None:
                    return type(sm).__name__
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["hist", "ablate"], required=True)
    ap.add_argument("--prompt-tokens", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--kinds", nargs="+", default=["pad", "code", "prose"])
    ap.add_argument(
        "--code-file",
        default=str(
            Path(__file__).parent.parent / "mlx_vlm/models/glm5_next/language.py"
        ),
    )
    ap.add_argument(
        "--prose-file", default=str(Path(__file__).parent.parent / "README.md")
    )
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--prefill-step-size", type=int, default=None)
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    H = _load_harness()
    import mlx_vlm.models.switch_layers as sl
    from mlx_vlm.models.glm5_next.moe_gemm import (
        Glm5NextTiledSwitchGLU,
        min_rows,
        segment_tile_plan,
    )

    print(f"LOAD {a.model}", flush=True)
    model, processor = H._load_text_only(a.model)
    tok = processor.tokenizer
    cfg = model.config
    print("LOAD ok", flush=True)

    docs = {}
    if "code" in a.kinds:
        docs["code"] = Path(a.code_file).read_text()
    if "prose" in a.kinds:
        docs["prose"] = Path(a.prose_file).read_text()

    def build(kind, ntok):
        if kind == "pad":
            return H._prompt(processor, cfg, tok, ntok)
        return H._natural_prompt(processor, cfg, tok, ntok, docs[kind])

    stock_call = sl.SwitchGLU.__call__
    out = {
        "model": a.model,
        "mode": a.mode,
        "max_tokens": a.max_tokens,
        "prefill_step_size": a.prefill_step_size,
        "mlx": __import__("importlib.metadata", fromlist=["x"]).version("mlx"),
        "mlx_module": mx.__file__,
        "switch_mlp_class": _switch_class(model),
        "records": [],
    }
    extra = {"prefill_step_size": a.prefill_step_size} if a.prefill_step_size else {}

    # ------------------------------------------------------------------ hist
    if a.mode == "hist":
        recs = []

        def hist_call(self, x, indices):
            if indices.size >= 1024:
                recs.append(
                    (indices.size, counts_of(indices, self.gate_proj.num_experts))
                )
            return stock_call(self, x, indices)

        sl.SwitchGLU.__call__ = hist_call
        for kind in a.kinds:
            for ntok in a.prompt_tokens:
                prompt = build(kind, ntok)
                recs.clear()
                H._run(model, processor, prompt, 1, **extra)
                mx.eval([c for _, c in recs])
                per = [(int(n), [int(v) for v in c.tolist()]) for n, c in recs]
                nlayers = 42
                chunks = [per[i : i + nlayers] for i in range(0, len(per), nlayers)]
                summary = []
                for ci, ch in enumerate(chunks):
                    if len(ch) < nlayers:
                        continue
                    rows = []
                    for li, (n, c) in enumerate(ch):
                        st, pd, act, bnd = waste_model(c, 16)
                        nz = sorted([v for v in c if v > 0], reverse=True)
                        top = sum(nz[: max(1, len(nz) // 10)])
                        rows.append(
                            {
                                "layer": li,
                                "n_rows": n,
                                "active_experts": act,
                                "boundaries": bnd,
                                "max": max(c),
                                "min_nonzero": min(nz),
                                "stock_waste": st,
                                "pad16_waste": pd,
                                "predicted_gain": st / pd,
                                "top10pct_share": top / n,
                            }
                        )
                    summary.append(
                        {
                            "chunk": ci,
                            "n_rows": rows[0]["n_rows"],
                            "active_experts_mean": sum(
                                r["active_experts"] for r in rows
                            )
                            / len(rows),
                            "boundaries_mean": sum(r["boundaries"] for r in rows)
                            / len(rows),
                            "stock_waste_mean": sum(r["stock_waste"] for r in rows)
                            / len(rows),
                            "pad16_waste_mean": sum(r["pad16_waste"] for r in rows)
                            / len(rows),
                            "predicted_gain_mean": sum(
                                r["predicted_gain"] for r in rows
                            )
                            / len(rows),
                            "top10pct_share_mean": sum(
                                r["top10pct_share"] for r in rows
                            )
                            / len(rows),
                            "max_count_mean": sum(r["max"] for r in rows) / len(rows),
                            "layers": rows,
                        }
                    )
                out["records"].append(
                    {"kind": kind, "target_tokens": ntok, "chunks": summary}
                )
                for s in summary:
                    print(
                        f"  {kind:<6s} t={ntok:<6d} chunk{s['chunk']}: "
                        f"n={s['n_rows']} active={s['active_experts_mean']:.1f}/288 "
                        f"bnd={s['boundaries_mean']:.1f} "
                        f"stock_waste={s['stock_waste_mean']:.3f} "
                        f"pad16={s['pad16_waste_mean']:.3f} "
                        f"predicted_gain={s['predicted_gain_mean']:.3f} "
                        f"top10%={s['top10pct_share_mean']:.3f} "
                        f"max={s['max_count_mean']:.0f}",
                        flush=True,
                    )
                Path(a.out).write_text(json.dumps(out, indent=2))
        sl.SwitchGLU.__call__ = stock_call

    # ---------------------------------------------------------------- ablate
    else:

        def plan_only_call(self, x, indices):
            e = self.gate_proj.num_experts
            if indices.size >= min_rows(e, 16):
                # Only the host syncs the tiled path forces: counts (order
                # independent), then the three .item() calls
                # (choose_tile_rows x2 + tile_end[-1]).  No extra sort: the
                # tiled path sorts exactly once, as stock already does.
                c = counts_of(indices, e)
                int(mx.sum((c + 15) // 16).item())
                int(mx.sum((c + 31) // 32).item())
                int(mx.cumsum((c + 15) // 16, axis=0)[-1].item())
            return stock_call(self, x, indices)

        def zero_call(self, x, indices):
            e = self.gate_proj.num_experts
            if indices.size >= min_rows(e, 16):
                return mx.zeros((*indices.shape, x.shape[-1]), x.dtype)
            return stock_call(self, x, indices)

        # Arm C swaps the *module instances* (what the real toggle does) rather
        # than the class method: Glm5NextTiledSwitchGLU.__call__ falls back via
        # zero-arg super(), which cannot be rebound onto a plain SwitchGLU.
        import mlx.nn as nn

        def find_moes(root):
            found = []
            for name, m in root.named_modules():
                sm = m.get("switch_mlp") if isinstance(m, dict) else None
                if sm is None:
                    sm = getattr(m, "switch_mlp", None)
                if isinstance(sm, sl.SwitchGLU):
                    found.append((name, m))
            return found

        named = find_moes(model)
        moes = [m for _, m in named]
        assert moes, "no MoE modules found -- arm C would be a no-op"
        print(f"  found {len(moes)} MoE modules, e.g. {named[0][0]}", flush=True)

        def as_tiled(sm):
            new = Glm5NextTiledSwitchGLU.__new__(Glm5NextTiledSwitchGLU)
            nn.Module.__init__(new)
            new.gate_proj, new.up_proj, new.down_proj = (
                sm.gate_proj,
                sm.up_proj,
                sm.down_proj,
            )
            new.activation = sm.activation
            return new

        originals = [m.switch_mlp for m in moes]
        tiled = [as_tiled(sm) for sm in originals]

        arms = {
            "A": ("stock", stock_call),
            "B": ("stock+plan/sync", plan_only_call),
            "C": ("tiled ON", None),
            "D": ("routed experts -> zeros", zero_call),
        }
        for kind in a.kinds:
            for ntok in a.prompt_tokens:
                prompt = build(kind, ntok)
                H._run(model, processor, prompt, 2, **extra)  # warm this shape
                for rep in range(a.repeats):
                    for arm in a.arms:
                        label, fn = arms[arm]
                        busy_before = foreign_busy()
                        if arm == "C":
                            for m, t in zip(moes, tiled):
                                m.switch_mlp = t
                        else:
                            sl.SwitchGLU.__call__ = fn
                        r = H._run(model, processor, prompt, a.max_tokens, **extra)
                        sl.SwitchGLU.__call__ = stock_call
                        for m, o in zip(moes, originals):
                            m.switch_mlp = o
                        rec = {
                            "kind": kind,
                            "target_tokens": ntok,
                            "rep": rep,
                            "arm": arm,
                            "label": label,
                            "foreign": busy_before,
                            "prompt_tokens": r["prompt_tokens"],
                            "prompt_tps": r["prompt_tps"],
                            "generation_tps": r["generation_tps"],
                            "peak_memory_gb": r["peak_memory_gb"],
                        }
                        out["records"].append(rec)
                        print(
                            f"  {kind:<6s} t={ntok:<6d} rep{rep} arm {arm} "
                            f"{label:<26s} prompt_tps={r['prompt_tps']:8.2f} "
                            f"peak={r['peak_memory_gb']:.1f}GB",
                            flush=True,
                        )
                        Path(a.out).write_text(json.dumps(out, indent=2))

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"WROTE {a.out}", flush=True)


if __name__ == "__main__":
    main()
