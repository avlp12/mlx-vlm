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
