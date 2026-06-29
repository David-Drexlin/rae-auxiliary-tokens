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
OUTDIR="${OUTDIR:-RAE_ROOT_PLACEHOLDER/assets/analysis/dino_imagenet100_cls_register}"
ENCODER_PATH="${ENCODER_PATH:-facebook/dinov2-with-registers-base}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-bf16}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
ENCODER_INPUT_SIZE="${ENCODER_INPUT_SIZE:-224}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
KMEANS_REPEATS="${KMEANS_REPEATS:-5}"
KMEANS_N_INIT="${KMEANS_N_INIT:-20}"
ONE_NN_BOOTSTRAP_REPS="${ONE_NN_BOOTSTRAP_REPS:-50}"
PROBE_SPLITS="${PROBE_SPLITS:-5}"
PROBE_TEST_SIZE="${PROBE_TEST_SIZE:-0.3}"
PROBE_MAX_ITER="${PROBE_MAX_ITER:-2000}"
COLOR_BINS="${COLOR_BINS:-16}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"

test -f "${CONTAINER}" || { echo "[error] missing CONTAINER: ${CONTAINER}"; exit 1; }
test -d "${DATA}" || { echo "[error] missing DATA: ${DATA}"; exit 1; }

mkdir -p "${OUTDIR}"

apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  python3 RAE_ROOT_PLACEHOLDER/src/analyze_dino_global_tokens_imagenet100.py \
    --data-path "${DATA}" \
    --outdir "${OUTDIR}" \
    --encoder-path "${ENCODER_PATH}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --precision "${PRECISION}" \
    --image-size "${IMAGE_SIZE}" \
    --encoder-input-size "${ENCODER_INPUT_SIZE}" \
    --max-samples "${MAX_SAMPLES}" \
    --kmeans-repeats "${KMEANS_REPEATS}" \
    --kmeans-n-init "${KMEANS_N_INIT}" \
    --one-nn-bootstrap-reps "${ONE_NN_BOOTSTRAP_REPS}" \
    --probe-splits "${PROBE_SPLITS}" \
    --probe-test-size "${PROBE_TEST_SIZE}" \
    --probe-max-iter "${PROBE_MAX_ITER}" \
    --color-bins "${COLOR_BINS}" \
    --ridge-alpha "${RIDGE_ALPHA}"
