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
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
VAL_DATA="${VAL_DATA:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/val}"
PROBE_DIR="${PROBE_DIR:-RAE_ROOT_PLACEHOLDER/assets/analysis/stage1_semantic_retention_dino/probe}"
OUTPUT_ROOT="${OUTPUT_ROOT:-RAE_ROOT_PLACEHOLDER/assets/analysis/stage1_semantic_retention_dino/runs}"
SAVE_FOLDER="${SAVE_FOLDER:-}"
MODEL_LABEL="${MODEL_LABEL:-}"
FAMILY_OVERRIDE="${FAMILY_OVERRIDE:-}"
STRESS_LABEL="${STRESS_LABEL:-}"
SELECTED_INDICES_PATH="${SELECTED_INDICES_PATH:-}"
CSV_OUT="${CSV_OUT:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-bf16}"
MAX_IMAGES="${MAX_IMAGES:-}"

ARGS=(
  --run-dir "${RUN_DIR}"
  --val-data "${VAL_DATA}"
  --probe-dir "${PROBE_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --precision "${PRECISION}"
)

if [[ -n "${SAVE_FOLDER}" ]]; then
  ARGS+=(--save-folder "${SAVE_FOLDER}")
fi
if [[ -n "${MODEL_LABEL}" ]]; then
  ARGS+=(--model-label "${MODEL_LABEL}")
fi
if [[ -n "${FAMILY_OVERRIDE}" ]]; then
  ARGS+=(--family-override "${FAMILY_OVERRIDE}")
fi
if [[ -n "${STRESS_LABEL}" ]]; then
  ARGS+=(--stress-label "${STRESS_LABEL}")
fi
if [[ -n "${SELECTED_INDICES_PATH}" ]]; then
  ARGS+=(--selected-indices-path "${SELECTED_INDICES_PATH}")
fi
if [[ -n "${CSV_OUT}" ]]; then
  ARGS+=(--csv-out "${CSV_OUT}")
fi
if [[ -n "${MAX_IMAGES}" ]]; then
  ARGS+=(--max-images "${MAX_IMAGES}")
fi

apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  python3 RAE_ROOT_PLACEHOLDER/src/eval_stage1_semantic_retention.py "${ARGS[@]}"
