#!/bin/bash
#SBATCH --partition=gpu-2d
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
DATA="${DATA:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/val}"
OUTDIR="${OUTDIR:-RAE_ROOT_PLACEHOLDER/assets/analysis/global_color_hist_probe}"
ENCODER_KIND="${ENCODER_KIND:-dino}"
ENCODER_PATH="${ENCODER_PATH:-facebook/dinov2-with-registers-base}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-bf16}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
ENCODER_INPUT_SIZE="${ENCODER_INPUT_SIZE:-224}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PROBE_SPLITS="${PROBE_SPLITS:-5}"
PROBE_TEST_SIZE="${PROBE_TEST_SIZE:-0.3}"
COLOR_BINS="${COLOR_BINS:-16}"
DCT_INPUT_SIZE="${DCT_INPUT_SIZE:-32}"
DCT_KEEP="${DCT_KEEP:-8}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"

test -f "${CONTAINER}" || { echo "[error] missing CONTAINER: ${CONTAINER}"; exit 1; }
test -d "${DATA}" || { echo "[error] missing DATA: ${DATA}"; exit 1; }

mkdir -p "${OUTDIR}"

apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  python3 RAE_ROOT_PLACEHOLDER/src/analyze_global_color_hist_imagenet100.py \
    --data-path "${DATA}" \
    --outdir "${OUTDIR}" \
    --encoder-kind "${ENCODER_KIND}" \
    --encoder-path "${ENCODER_PATH}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --precision "${PRECISION}" \
    --image-size "${IMAGE_SIZE}" \
    --encoder-input-size "${ENCODER_INPUT_SIZE}" \
    --max-samples "${MAX_SAMPLES}" \
    --probe-splits "${PROBE_SPLITS}" \
    --probe-test-size "${PROBE_TEST_SIZE}" \
    --color-bins "${COLOR_BINS}" \
    --dct-input-size "${DCT_INPUT_SIZE}" \
    --dct-keep "${DCT_KEEP}" \
    --ridge-alpha "${RIDGE_ALPHA}"
