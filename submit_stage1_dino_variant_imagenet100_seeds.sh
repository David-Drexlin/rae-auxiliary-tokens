#!/bin/bash
set -euo pipefail

cd RAE_ROOT_PLACEHOLDER
mkdir -p logs

WORKER="${WORKER:-RAE_ROOT_PLACEHOLDER/run_stage1_dino_imagenet100_array.sh}"
JOB_PREFIX="${JOB_PREFIX:-RAE_s1_dino_variant_in100_}"
EXP_PREFIX="${EXP_PREFIX:-DINO_VARIANT_IN100_}"

CONFIGS=(
  "DINOv2NoReg_decB.yaml"
  "DINOv2NoReg_decB_Patch+CLS_prepend.yaml"
  "DINOv2NoReg_decB_Patch+CLS_CA.yaml"
  "DINOv3ViTB16_decB.yaml"
  "DINOv3ViTB16_decB_Patch+CLS_prepend.yaml"
  "DINOv3ViTB16_decB_Patch+Register_prepend.yaml"
  "DINOv3ViTB16_decB_Patch+Register+CLS_prepend.yaml"
)

SEEDS=(0 1 2)

for cfg_name in "${CONFIGS[@]}"; do
  base_name="$(basename "${cfg_name}" .yaml)"
  for seed in "${SEEDS[@]}"; do
    job_name="${JOB_PREFIX}${base_name}_seed${seed}"
    sbatch \
      --job-name="${job_name}" \
      --export=ALL,CONFIG_NAME="${cfg_name}",GLOBAL_SEED="${seed}",EXP_PREFIX="${EXP_PREFIX}" \
      "${WORKER}"
  done
done
