#!/bin/zsh
# PART 2: prefill_step_size e2e sweep on the serving branch.
#   toggles: serving set ON; buffer env (MLX_MAX_MB_PER_BUFFER / _OPS_) deliberately
#   OFF -- it is a decode lever and the campaign's prefill finding says OFF.
set -u
H=/Users/gesicht/local-llm-serving/.claude/worktrees/frosty-engelbart-0997b4/ports/glm53flash-mlx/serving/repro_mlxvlm_dflash2.py
V=~/glm53flash/prep/dflash2-repro/venv/bin/python
M=~/glm53flash/builds/GLM-5.3-Flash-vlm-q4-quasar
D=~/glm53flash/builds/GLM-5.3-Flash-DFlash2
SRC=~/src/mlx-vlm-moegemm   # serve branch (~/src/mlx-vlm-serve @6e93bc08) is currently
              # broken: NameError checkpoint_len in generate/ar.py. This worktree is
              # glm5-serve-unified@999ba476 + the (OFF-by-default) MoE patch.
LOG=~/glm53flash/prep/serving/logs
REP=${1:-0}
ORDER=${2:-"2048 4096 8192"}
for STEP in ${=ORDER}; do
  OUT=$LOG/chunksweep_step${STEP}_rep${REP}.json
  echo "=== step=$STEP rep=$REP -> $OUT  $(date +%T)"
  env -u MLX_MAX_MB_PER_BUFFER -u MLX_MAX_OPS_PER_BUFFER \
      -u MLX_VLM_GLM5_MOE_GEMM \
      MLX_VLM_GLM5_FUSED_KDA=1 MLX_VLM_GLM5_FUSED_KDA_QPROJ=1 \
      MLX_VLM_GLM5_IDX_FAST=1 \
      PYTHONPATH=$SRC \
    $V $H --model $M --draft-model $D --src $SRC --skip-spec \
      --prompt-tokens 8192 32768 --max-tokens 8 \
      --prefill-step-size $STEP --out $OUT 2>&1 | grep -E "baseline|LOAD ok|WROTE|Error|error"
done
