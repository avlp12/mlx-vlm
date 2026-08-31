#!/usr/bin/env python3
"""Stage 3 probe: ship-and-restore round trip for one context checkpoint.

Two modes:

``--mode local``  (no peer, no model, safe under GPU contention)
    Measures the serialization half of the round trip: pack -> bytes ->
    unpack -> restore, at a realistic checkpoint size. This bounds the CPU/GPU
    cost that the 4.6 GB/s tbnet transport is added to.

``--mode ring``   (requires mx.distributed on both boxes)
    Rank 0 packs and sends, rank 1 receives, unpacks and restores. Launch with
    the tbnet env-only ring the pipeline campaign uses.

Checkpoint geometry is the measured GLM-5.3-Flash shape: 34 KDA layers holding
flat fp32 state and 11 DSA layers holding 2562 B/tok/layer of latent, so a 32k
rung is ~1.00 GiB and a 131k rung ~3.58 GiB.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

KDA_LAYERS = 34
DSA_LAYERS = 11
KDA_BYTES_PER_LAYER = int(4.14 * 1024**2)
DSA_BYTES_PER_TOK_LAYER = 2562


def synth_fragments(n_tokens: int, scale: float = 1.0):
    """A cache-shaped fragment list at the measured byte geometry.

    ``scale`` shrinks every tensor proportionally so the probe can run under
    memory pressure and be extrapolated linearly (the wire cost is linear in
    bytes by construction -- one contiguous buffer).
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mlx_vlm.apc_adapters import Capability, StateFragment

    frags = []
    kda_elems = max(int(KDA_BYTES_PER_LAYER * scale) // 4, 1)
    for i in range(KDA_LAYERS):
        st = [
            mx.random.normal((kda_elems // 2,), key=mx.random.key(i)).astype(mx.float32),
            mx.random.normal((kda_elems // 2,), key=mx.random.key(i + 1000)).astype(mx.float32),
        ]
        frags.append(
            StateFragment(Capability.CHECKPOINT, n_tokens, payload={"state": st, "meta_state": ""})
        )
    dsa_elems = max(int(n_tokens * DSA_BYTES_PER_TOK_LAYER * scale) // 2 // 2, 1)
    for i in range(DSA_LAYERS):
        subs = []
        for j in range(2):
            k = mx.random.normal((dsa_elems,), key=mx.random.key(i * 7 + j)).astype(mx.float16)
            v = mx.random.normal((dsa_elems,), key=mx.random.key(i * 7 + j + 99)).astype(mx.float16)
            subs.append(
                StateFragment(Capability.CHECKPOINT, n_tokens, payload={"state": [k, v], "meta_state": ""})
            )
        frags.append(StateFragment(Capability.COMPOSITE, n_tokens, payload=subs))
    mx.eval([a for f in frags for a in f.eval_targets()])
    return frags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "ring"], default="local")
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mlx_vlm.context_vault_wire import pack_fragments, unpack_fragments

    frags = synth_fragments(args.tokens, args.scale)

    rows = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        manifest, payload = pack_fragments(frags)
        mx.eval(payload)
        t1 = time.perf_counter()
        rebuilt = unpack_fragments(manifest, payload)
        mx.eval([a for f in rebuilt for a in f.eval_targets()])
        t2 = time.perf_counter()
        nbytes = manifest["total_bytes"]
        rows.append(
            {
                "bytes": nbytes,
                "gib": nbytes / 1024**3,
                "pack_s": t1 - t0,
                "unpack_s": t2 - t1,
                "pack_gbps": nbytes / (t1 - t0) / 1e9,
                "unpack_gbps": nbytes / (t2 - t1) / 1e9,
            }
        )
        del manifest, payload, rebuilt
        mx.clear_cache()

    med = sorted(rows, key=lambda r: r["pack_s"])[len(rows) // 2]
    full_scale_bytes = med["bytes"] / args.scale
    out = {
        "mode": args.mode,
        "tokens": args.tokens,
        "scale": args.scale,
        "reps": rows,
        "median": med,
        "extrapolated_full_gib": full_scale_bytes / 1024**3,
        "tbnet_measured_gbps": 4.6,
        "est_ship_s_at_4.6GBps": full_scale_bytes / 4.6e9,
        "note": (
            "serialization is one contiguous buffer, so cost is linear in bytes; "
            "ring transport time is additive on top of pack+unpack"
        ),
    }
    if args.mode == "ring":
        out["ring"] = "not run -- requires mx.distributed launch on both boxes"
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
