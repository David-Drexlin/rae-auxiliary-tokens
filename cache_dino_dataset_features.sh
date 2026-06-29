#!/bin/bash
#SBATCH --partition=gpu-2d
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
DATA_PATH="${DATA_PATH:?DATA_PATH is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
SPLIT="${SPLIT:-val}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
SOURCE_INPUT_SIZE="${SOURCE_INPUT_SIZE:-224}"
TARGET_INPUT_SIZE="${TARGET_INPUT_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SHARD_SIZE="${SHARD_SIZE:-2048}"
CACHE_DTYPE="${CACHE_DTYPE:-bf16}"
PRECISION="${PRECISION:-bf16}"
OVERWRITE="${OVERWRITE:-false}"
ENCODER_MODEL="${ENCODER_MODEL:-facebook/dinov2-with-registers-base}"

mkdir -p "${OUTPUT_ROOT}"
TMP_CONFIG="$(mktemp /tmp/dino_cache_cfg.XXXXXX.yaml)"

cat > "${TMP_CONFIG}" <<EOF
experiment:
  name: "ood_dino_self_cache"
  seed: 0
  output_root: "${OUTPUT_ROOT}"

data:
  image_size: ${IMAGE_SIZE}
  batch_size: ${BATCH_SIZE}
  num_workers: ${NUM_WORKERS}
  train_path: "${DATA_PATH}"
  val_path: "${DATA_PATH}"
  max_samples: 0
  max_samples_train: 0
  max_samples_val: 0

encoders:
  source:
    name: "dinov2_patch_tokens"
    encoder_cls: "Dinov2withNorm"
    model_name: "${ENCODER_MODEL}"
    input_size: ${SOURCE_INPUT_SIZE}
    normalize: true
    cache_global_tokens: true
  target:
    name: "dinov2_regcls_tokens"
    model_name: "${ENCODER_MODEL}"
    input_size: ${TARGET_INPUT_SIZE}

cache:
  dir: "${OUTPUT_ROOT}/cache"
  dtype: "${CACHE_DTYPE}"
  precision: "${PRECISION}"
  shard_size: ${SHARD_SIZE}
  overwrite: ${OVERWRITE}
EOF

echo "=== DINO DATASET CACHE ==="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "DATA_PATH: ${DATA_PATH}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "SPLIT: ${SPLIT}"
echo "TMP_CONFIG: ${TMP_CONFIG}"
echo "=========================="

test -f "${CONTAINER}" || { echo "[error] missing CONTAINER: ${CONTAINER}"; exit 1; }
test -d "${DATA_PATH}" || { echo "[error] missing DATA_PATH: ${DATA_PATH}"; exit 1; }

apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  python3 RAE_ROOT_PLACEHOLDER/src/register_prediction/cache_features.py \
    --config "${TMP_CONFIG}" \
    --split "${SPLIT}"

rm -f "${TMP_CONFIG}"

echo "[done] cache ready at ${OUTPUT_ROOT}"
echo "End: $(date)"
