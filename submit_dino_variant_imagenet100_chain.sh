#!/bin/bash
set -euo pipefail

cd RAE_ROOT_PLACEHOLDER
mkdir -p logs

STATS_WORKER="${STATS_WORKER:-RAE_ROOT_PLACEHOLDER/run_dino_variant_imagenet100_stats_array.sh}"
TRAIN_WORKER="${TRAIN_WORKER:-RAE_ROOT_PLACEHOLDER/run_stage1_dino_imagenet100_array.sh}"
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

dino2_stats_job="$(sbatch --parsable --array=0-1 --job-name=dino2_noreg_stats_in100 "${STATS_WORKER}")"
dino3_stats_job="$(sbatch --parsable --array=2-5 --job-name=dinov3_stats_in100 "${STATS_WORKER}")"
echo "dino2_stats=${dino2_stats_job}"
echo "dino3_stats=${dino3_stats_job}"

for cfg_name in "${CONFIGS[@]}"; do
  base_name="$(basename "${cfg_name}" .yaml)"
  for seed in "${SEEDS[@]}"; do
    job_name="${JOB_PREFIX}${base_name}_seed${seed}"
    dependency="${dino3_stats_job}"
    if [[ "${cfg_name}" == DINOv2NoReg_* ]]; then
      dependency="${dino2_stats_job}"
    fi
    train_job="$(
      sbatch \
        --parsable \
        --dependency="afterok:${dependency}" \
        --job-name="${job_name}" \
        --export=ALL,CONFIG_NAME="${cfg_name}",GLOBAL_SEED="${seed}",EXP_PREFIX="${EXP_PREFIX}" \
        "${TRAIN_WORKER}"
    )"
    echo "train ${cfg_name} seed${seed}=${train_job}"
  done
done
