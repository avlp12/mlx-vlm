# glm5_next MoE prefill GEMM benchmarks

Exploratory harness behind `MLX_VLM_GLM5_MOE_GEMM` (see
`mlx_vlm/models/glm5_next/README.md` -> "Segment-aligned routed-expert GEMM").
The shipped, self-contained micro-bench is
`python -m mlx_vlm.tests.test_glm5_next_moe_gemm [chunk ...]`; these scripts are the
wider sweeps used to gate the work.

All numbers were taken on an M3 Ultra (512 GB) with mlx 0.32.0, at the real
GLM-5.3-Flash-vlm-q4-quasar routed-expert geometry (hidden 4096, moe_intermediate
2048, 288 routed experts, top-8, 42 sparse layers, 4-bit g64 affine), in windows
verified idle (no python process > 2 GB RSS and GPU device utilization < 25%).

| script                         | what it answers                                                          |
| ------------------------------ | ------------------------------------------------------------------------ |
| `stage1_moe_gemm_roofline.py`  | sorted `gather_qmm` vs dense `quantized_matmul` vs bf16 matmul vs weight-read bandwidth, plus a rows/expert sweep that isolates the boundary-pass waste |
| `stage1b_decompose.py`         | per-projection and non-GEMM (sort / gather / unsort / activation) split, and the padded variants priced against the stock path |
| `stage1c_switchglu_ab.py`      | whole-layer A/B, stock `SwitchGLU` vs `Glm5NextTiledSwitchGLU`, with a bit-identity check per point |

`*.json` next to each script are the receipts. `stage1c_small.json` and the chunk-1536
row of `stage1c_switchglu_ab.json` were taken in a contended window and are superseded
by `stage1c_small_clean.json` / `stage1c_clean_sweep.json`.

## Round 2 (2026-08-31): live rejection, explained

The synthetic +5% did not survive end to end (-0.94% / -2.41% on the harness A/B).
These scripts found out why; the verdict lives in
`mlx_vlm/models/glm5_next/README.md` -> "Live verdict".

| script                      | what it answers                                                        |
| --------------------------- | ---------------------------------------------------------------------- |
| `probe_live_moe.py --mode hist`   | per-layer, per-chunk live expert-occupancy histograms for the harness's PAD template vs real code vs real prose |
| `probe_live_moe.py --mode ablate` | 4-arm in-situ decomposition: stock / stock+the-syncs-only / toggle ON / routed-experts-ablated-to-zeros, i.e. the live MoE share and the sync tax |
| `stage3_core_bm_sweep.py`   | sweeps `affine_gather_qmm_rhs`'s row tile via the `MLX_GATHER_QMM_BM` experiment build at `~/src/mlx-core-pr` -- the real kernel-level ceiling |
| `part2_chunk_sweep.sh`      | `--prefill-step-size` {2048,4096,8192} x prompt {8192,32768} end to end, with peak memory |

Receipts also copied here: `moegemm_routing_hist.json`, `moegemm_ablate2.json`,
`moegemm_ablate_R16_clean.json`, `stage3_core_bm.json`. The live harness receipts are
`~/glm53flash/prep/serving/logs/moegemm_*.json` and `chunksweep_step*_rep*.json`.

Headline corrections to round 1:
- live routed-expert share is **44.9-46.8%**, not 53.6%;
- the plan's host syncs cost a flat **-0.8 to -1.2%** and cannot be hoisted;
- the padding prize is **routing dependent**: 1.111x on real text, 1.041-1.083x on the
  harness's repetitive PAD prompt -- the A/B measured the worst case;
- the boundary-pass waste model is confirmed (3.3% mean abs error over an 18-point core
  `bm` sweep) against the `1 + bm/(2g)` row-padding form (16.9%);
- `bm=16` is already optimal on M3 non-nax -- no core-side prize;
- `prefill_step_size=2048` is already optimal -- 8192 costs ~5% and +19-30 GB peak.

Measurement hygiene: this machine runs several agents' GPU jobs concurrently. Every
number above was taken behind `/tmp/strict_gate.sh` (no foreign python >8 GiB RSS AND
GPU device utilization <25%, confirmed twice 10 s apart); `probe_live_moe.py` also
records a per-measurement `foreign` field. Contended runs are labelled and excluded --
contention here is worth 30-40%, far larger than every effect being measured.
