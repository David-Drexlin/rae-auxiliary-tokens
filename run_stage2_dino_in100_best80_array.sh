#!/bin/bash
#SBATCH --job-name=rae_s2_dino_best80
#SBATCH --partition=gpu-2d
#SBATCH --constraint=40gb
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-3
#SBATCH --output=RAE_ROOT_PLACEHOLDER/logs/%x-%A_%a.out
#SBATCH --error=RAE_ROOT_PLACEHOLDER/logs/%x-%A_%a.err

set -euo pipefail

REPO="RAE_ROOT_PLACEHOLDER"
mkdir -p "$REPO/logs"
DATA="${DATA:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/train}"
RESULTS_DIR="${RESULTS_DIR:-$REPO/ckpts/stage2_best80}"

CONFIGS=(
  "$REPO/configs/stage2/training/ImageNet256/long80/DiTDH-S_DINO_IN100_patch_80ep.yaml"
  "$REPO/configs/stage2/training/ImageNet256/long80/LightningDiT-S_DINO_IN100_PatchRegCls_AdaLN_80ep.yaml"
  "$REPO/configs/stage2/training/ImageNet256/long80/LightningDiT-S_DINO_IN100_PatchRegCls_prepend_80ep.yaml"
  "$REPO/configs/stage2/training/ImageNet256/long80/LightningDiT-S_DINO_IN100_PatchRegCls_CA_80ep.yaml"
)

EXPERIMENT_NAMES=(
  "in100_s2_dino_patch80_ddt"
  "in100_s2_dino_patchregcls_adaln80_lightningdit"
  "in100_s2_dino_patchregcls_prepend80_lightningdit"
  "in100_s2_dino_patchregcls_ca80_lightningdit"
)

export CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
export EXPERIMENT_NAME="${EXPERIMENT_NAMES[$SLURM_ARRAY_TASK_ID]}"
export DATA
export RESULTS_DIR
export NPROC=2
export PRECISION=bf16
export IMAGE_SIZE=256

bash "$REPO/train_ImageNet_DiT.sh"
