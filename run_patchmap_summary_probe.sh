#!/bin/bash
#SBATCH --job-name=patchmap_probe
#SBATCH --partition=gpu-2d
#SBATCH --qos=SLURM_QOS_PLACEHOLDER
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
CONFIG="${CONFIG:?CONFIG is required}"
CKPT="${CKPT:?CKPT is required}"
DATA="${DATA:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/val}"
OUTDIR="${OUTDIR:?OUTDIR is required}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-bf16}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
PROBE_SPLITS="${PROBE_SPLITS:-5}"
PROBE_TEST_SIZE="${PROBE_TEST_SIZE:-0.3}"
COLOR_BINS="${COLOR_BINS:-16}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"

test -f "${CONTAINER}" || { echo "[error] missing CONTAINER: ${CONTAINER}"; exit 1; }
test -f "${CONFIG}" || { echo "[error] missing CONFIG: ${CONFIG}"; exit 1; }
test -f "${CKPT}" || { echo "[error] missing CKPT: ${CKPT}"; exit 1; }
test -d "${DATA}" || { echo "[error] missing DATA: ${DATA}"; exit 1; }

mkdir -p "${OUTDIR}"

apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  python3 RAE_ROOT_PLACEHOLDER/src/analyze_patchmap_summary_imagenet100.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --data-path "${DATA}" \
    --outdir "${OUTDIR}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --precision "${PRECISION}" \
    --image-size "${IMAGE_SIZE}" \
    --probe-splits "${PROBE_SPLITS}" \
    --probe-test-size "${PROBE_TEST_SIZE}" \
    --color-bins "${COLOR_BINS}" \
    --ridge-alpha "${RIDGE_ALPHA}"
