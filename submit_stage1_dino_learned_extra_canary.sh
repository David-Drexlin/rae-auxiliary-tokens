#!/bin/bash
set -euo pipefail

REPO="RAE_ROOT_PLACEHOLDER"
CFG_NAME="DINO_decB_Patch+Register+CLS_prepend_TruePlusLearnedExtra.yaml"
MODEL_NAME="${CFG_NAME%.yaml}"
SEED="${SEED:-0}"

train_job=$(sbatch --parsable \
  --job-name="RAE_s1_dino_in100_${MODEL_NAME}_seed${SEED}" \
  --export=ALL,CONFIG_NAME="${CFG_NAME}",GLOBAL_SEED="${SEED}",EXP_PREFIX="DINO_IN100_" \
  "${REPO}/run_stage1_dino_imagenet100_array.sh")

eval_out=$("${REPO}/submit_stage1_dino_seeded_eval_one.sh" "${MODEL_NAME}" "${SEED}" "${train_job}")
eval_job=$(printf '%s\n' "${eval_out}" | tail -n 1 | awk '{print $NF}')

printf 'train=%s\neval=%s\n' "${train_job}" "${eval_job}"
