#!/usr/bin/env python3
"""Analytic vault sizing for GLM-5.3-Flash 320B-A18B q4. No GPU required.

Constants are the measured per-layer cache footprints from the pipeline
campaign (~/glm53flash/logs/pipeline/):

  KDA  (linear layers)   4.14 MiB/layer, flat in sequence length, x34 layers
  DSA  (attention)       2562 B/tok/layer,  linear in length,     x11 layers

The flat KDA term is the whole reason ladder policy matters: every rung re-pays
140.8 MiB no matter how shallow it is.
"""
import sys

KDA_MIB_PER_LAYER = 4.14
KDA_LAYERS = 34
DSA_B_PER_TOK_LAYER = 2562
DSA_LAYERS = 11

KDA_FLAT_B = KDA_MIB_PER_LAYER * KDA_LAYERS * 1024**2
DSA_B_PER_TOK = DSA_B_PER_TOK_LAYER * DSA_LAYERS
GiB = 1024**3


def rung_bytes(length: int) -> float:
    return KDA_FLAT_B + length * DSA_B_PER_TOK


def main() -> None:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from mlx_vlm.context_vault import boundary_ladder

    print(f"KDA flat term : {KDA_FLAT_B/1024**2:8.1f} MiB per rung "
          f"({KDA_MIB_PER_LAYER} MiB x {KDA_LAYERS} layers)")
    print(f"DSA per token : {DSA_B_PER_TOK:8d} B "
          f"({DSA_B_PER_TOK_LAYER} B x {DSA_LAYERS} layers)")
    print()
    print("Single full-context checkpoint:")
    for n in (16384, 32768, 65536, 131072):
        print(f"  {n:>7} tok -> {rung_bytes(n)/GiB:6.2f} GiB")
    print()

    step = 2048
    print(f"{'ctx':>8} {'mode':<10} {'rungs':>5} {'GiB':>7} {'worst tail':>11}  boundaries")
    for total in (16384, 32768, 65536, 131072):
        for mode in ("geometric", "uniform"):
            r = boundary_ladder(total, 8192, step, 8, mode)
            gib = sum(rung_bytes(x) for x in r) / GiB
            tail = total - r[-1] if r else total
            print(f"{total:>8} {mode:<10} {len(r):>5} {gib:>7.2f} {tail:>11} {r}")
    print()

    for budget_gb in (128, 256, 340):
        budget = budget_gb * GiB
        print(f"Budget {budget_gb} GiB:")
        for total in (16384, 32768, 131072):
            geo = boundary_ladder(total, 8192, step, 8, "geometric")
            per_doc = sum(rung_bytes(x) for x in geo)
            flat = rung_bytes(total)
            print(
                f"  {total:>7} tok  ladder {per_doc/GiB:5.2f} GiB/doc -> "
                f"{int(budget//per_doc):>4} docs   |   "
                f"deepest-rung-only {flat/GiB:5.2f} GiB/doc -> "
                f"{int(budget//flat):>4} docs"
            )
        print()


if __name__ == "__main__":
    main()
