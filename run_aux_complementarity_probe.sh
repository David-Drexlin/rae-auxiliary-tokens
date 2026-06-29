#!/bin/bash
#SBATCH --partition=gpu-2h
#SBATCH --constraint=40gb
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=RAE_ROOT_PLACEHOLDER/logs/%x-%j.out
#SBATCH --error=RAE_ROOT_PLACEHOLDER/logs/%x-%j.err

set -euo pipefail

cd RAE_ROOT_PLACEHOLDER
mkdir -p logs

CONTAINER="${CONTAINER:-RAE_ROOT_PLACEHOLDER/container.sif}"
EMBEDDINGS="${EMBEDDINGS:?EMBEDDINGS is required}"
OUTDIR="${OUTDIR:?OUTDIR is required}"
PROBE_SPLITS="${PROBE_SPLITS:-5}"
PROBE_TEST_SIZE="${PROBE_TEST_SIZE:-0.3}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
RIDGE_ALPHAS="${RIDGE_ALPHAS:-0.001,0.01,0.1,1,10,100,1000,10000,100000}"
RIDGE_CV_FOLDS="${RIDGE_CV_FOLDS:-3}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
THUMBNAIL_SIZE="${THUMBNAIL_SIZE:-8}"
COLOR_BINS="${COLOR_BINS:-16}"
DCT_INPUT_SIZE="${DCT_INPUT_SIZE:-32}"
DCT_KEEP="${DCT_KEEP:-8}"
SEED="${SEED:-0}"

test -f "${CONTAINER}" || { echo "[error] missing CONTAINER: ${CONTAINER}"; exit 1; }
test -f "${EMBEDDINGS}" || { echo "[error] missing EMBEDDINGS: ${EMBEDDINGS}"; exit 1; }

mkdir -p "${OUTDIR}"

apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  python3 RAE_ROOT_PLACEHOLDER/src/analyze_aux_complementarity_from_embeddings.py \
    --embeddings "${EMBEDDINGS}" \
    --outdir "${OUTDIR}" \
    --seed "${SEED}" \
    --probe-splits "${PROBE_SPLITS}" \
    --probe-test-size "${PROBE_TEST_SIZE}" \
    --ridge-alpha "${RIDGE_ALPHA}" \
    --ridge-alphas "${RIDGE_ALPHAS}" \
    --ridge-cv-folds "${RIDGE_CV_FOLDS}" \
    --image-size "${IMAGE_SIZE}" \
    --thumbnail-size "${THUMBNAIL_SIZE}" \
    --color-bins "${COLOR_BINS}" \
    --dct-input-size "${DCT_INPUT_SIZE}" \
    --dct-keep "${DCT_KEEP}"
