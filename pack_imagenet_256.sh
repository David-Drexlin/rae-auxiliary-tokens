#!/bin/bash
#SBATCH --job-name=pack_imagenet_val_256
#SBATCH --partition=cpu-2d
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd RAE_ROOT_PLACEHOLDER
mkdir -p logs
mkdir -p RAE_ROOT_PLACEHOLDER/assets/datasets

CONTAINER="${CONTAINER:-RAE_ROOT_PLACEHOLDER/container.sif}"
DATA="${DATA:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/val}"
OUT_DIR="${OUT_DIR:-RAE_ROOT_PLACEHOLDER/assets/datasets}"
REAL_NPZ="${REAL_NPZ:-RAE_ROOT_PLACEHOLDER/assets/datasets/imagenet100_val_256.npz}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
TMP_NPZ="${TMP_NPZ:-${OUT_DIR}/val.npz}"

echo "=== PACK JOB INFO ==="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "DATA: ${DATA}"
echo "OUT_DIR: ${OUT_DIR}"
echo "REAL_NPZ: ${REAL_NPZ}"
echo "TMP_NPZ: ${TMP_NPZ}"
echo "IMAGE_SIZE: ${IMAGE_SIZE}"
echo "====================="

test -d "${DATA}" || { echo "[error] missing DATA dir: ${DATA}"; exit 1; }
test -f "${CONTAINER}" || { echo "[error] missing CONTAINER: ${CONTAINER}"; exit 1; }
mkdir -p "${OUT_DIR}"

apptainer exec \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd RAE_ROOT_PLACEHOLDER

    python RAE_ROOT_PLACEHOLDER/pack_images.py \
      '${DATA}' \
      ${IMAGE_SIZE} \
      '${OUT_DIR}'
  "

echo "=== CHECK OUTPUT ==="
ls -lah "${OUT_DIR}" || true

test -f "${TMP_NPZ}" || { echo "[error] missing packed file: ${TMP_NPZ}"; exit 1; }
mv -f "${TMP_NPZ}" "${REAL_NPZ}"
test -f "${REAL_NPZ}" || { echo "[error] copy failed: ${REAL_NPZ}"; exit 1; }

echo "[done] wrote ${REAL_NPZ}"
echo "[done] finished at $(date)"
