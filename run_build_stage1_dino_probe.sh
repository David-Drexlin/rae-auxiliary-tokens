#!/bin/bash
#SBATCH --partition=gpu-2h
#SBATCH --constraint=40gb
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=RAE_ROOT_PLACEHOLDER/logs/%x-%j.out
#SBATCH --error=RAE_ROOT_PLACEHOLDER/logs/%x-%j.err

set -euo pipefail

cd RAE_ROOT_PLACEHOLDER
mkdir -p logs

CONTAINER="${CONTAINER:-RAE_ROOT_PLACEHOLDER/container.sif}"
TRAIN_DATA="${TRAIN_DATA:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/train}"
VAL_DATA="${VAL_DATA:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/val}"
OUTPUT_DIR="${OUTPUT_DIR:-RAE_ROOT_PLACEHOLDER/assets/analysis/stage1_semantic_retention_dino/probe}"
ENCODER_PATH="${ENCODER_PATH:-facebook/dinov2-with-registers-base}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
ENCODER_INPUT_SIZE="${ENCODER_INPUT_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-bf16}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-}"

ARGS=(
  --train-data "${TRAIN_DATA}"
  --val-data "${VAL_DATA}"
  --output-dir "${OUTPUT_DIR}"
  --encoder-path "${ENCODER_PATH}"
  --image-size "${IMAGE_SIZE}"
  --encoder-input-size "${ENCODER_INPUT_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --precision "${PRECISION}"
)

if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  ARGS+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_VAL_SAMPLES}" ]]; then
  ARGS+=(--max-val-samples "${MAX_VAL_SAMPLES}")
fi

apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  python3 RAE_ROOT_PLACEHOLDER/src/build_stage1_dino_probe.py "${ARGS[@]}"
