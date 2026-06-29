#!/bin/bash
set -euo pipefail

cd RAE_ROOT_PLACEHOLDER

CONFIG_NAME="DINO_decB_Patch+PatchMAP5_prepend.yaml"
TRAIN_EXP_NAME="DINO_IN100_DINO_decB_Patch+PatchMAP5_prepend_seed0"
CKPT_DIR="RAE_ROOT_PLACEHOLDER/ckpts/${TRAIN_EXP_NAME}/checkpoints"
SAVE_FOLDER="eval_${TRAIN_EXP_NAME}"
CONFIG_PATH="RAE_ROOT_PLACEHOLDER/configs/stage1/training/ImageNet/${CONFIG_NAME}"

train_job=$(
  sbatch --parsable \
    --job-name="RAE_s1_dino_in100_DINO_decB_Patch+PatchMAP5_prepend_seed0" \
    --export=ALL,CONFIG_NAME="${CONFIG_NAME}",GLOBAL_SEED=0,EXP_PREFIX="DINO_IN100_" \
    RAE_ROOT_PLACEHOLDER/run_stage1_dino_imagenet100_array.sh
)

eval_job=$(
  sbatch --parsable \
    --job-name="eval_DINO_IN100_DINO_decB_Patch+PatchMAP5_prepend_seed0" \
    --dependency="afterok:${train_job}" \
    --export=ALL,CONFIG="${CONFIG_PATH}",CKPT_DIR="${CKPT_DIR}",TRAIN_EXP_NAME="${TRAIN_EXP_NAME}",SAVE_FOLDER="${SAVE_FOLDER}",PREFER_FULL_MODEL_CKPT=1 \
    RAE_ROOT_PLACEHOLDER/stage1_eval_ImageNet_DINO_seeded_single.sh
)

printf 'train=%s\neval=%s\n' "${train_job}" "${eval_job}"
