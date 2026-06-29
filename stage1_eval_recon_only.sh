#!/bin/bash
#SBATCH --partition=gpu-2d
#SBATCH --qos=SLURM_QOS_PLACEHOLDER
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
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
CSV_OUT="${CSV_OUT:?CSV_OUT is required}"
FID_REAL_NPZ="${FID_REAL_NPZ:?FID_REAL_NPZ is required}"
FD_REAL_IMAGEFOLDER="${FD_REAL_IMAGEFOLDER:?FD_REAL_IMAGEFOLDER is required}"
GUIDED_EVAL="${GUIDED_EVAL:-HOME_PLACEHOLDER/guided-diffusion/evaluations/evaluator.py}"
FD_SCRIPT="${FD_SCRIPT:-HOME_PLACEHOLDER/FD-DINOv2/src/pytorch_fd/fd_score.py}"
DEVICE="${DEVICE:-cuda:0}"
FD_BATCH_SIZE="${FD_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export TF_ENABLE_ONEDNN_OPTS=0
export TOKENIZERS_PARALLELISM=false

echo "=== JOB INFO ==="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "RUN_DIR: ${RUN_DIR}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "CSV_OUT: ${CSV_OUT}"
echo "FID_REAL_NPZ: ${FID_REAL_NPZ}"
echo "FD_REAL_IMAGEFOLDER: ${FD_REAL_IMAGEFOLDER}"
echo "FD_BATCH_SIZE: ${FD_BATCH_SIZE}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "================"

echo "=== GPU INFO ==="
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id --format=csv,noheader || true
echo "================"

test -f "${CONTAINER}" || { echo "[error] missing CONTAINER: ${CONTAINER}"; exit 1; }
test -d "${RUN_DIR}" || { echo "[error] missing RUN_DIR: ${RUN_DIR}"; exit 1; }
test -f "${FID_REAL_NPZ}" || { echo "[error] missing FID_REAL_NPZ: ${FID_REAL_NPZ}"; exit 1; }
test -d "${FD_REAL_IMAGEFOLDER}" || { echo "[error] missing FD_REAL_IMAGEFOLDER: ${FD_REAL_IMAGEFOLDER}"; exit 1; }
test -f "${GUIDED_EVAL}" || { echo "[error] missing GUIDED_EVAL: ${GUIDED_EVAL}"; exit 1; }
test -f "${FD_SCRIPT}" || { echo "[error] missing FD_SCRIPT: ${FD_SCRIPT}"; exit 1; }
test -f "${RUN_DIR}/samples.npz" || { echo "[error] missing ${RUN_DIR}/samples.npz"; exit 1; }

apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  env XFORMERS_DISABLED=1 \
  python3 RAE_ROOT_PLACEHOLDER/src/eval_stage1_recon.py \
    --run-dir "${RUN_DIR}" \
    --model-name "${MODEL_NAME}" \
    --csv-out "${CSV_OUT}" \
    --fid-real-npz "${FID_REAL_NPZ}" \
    --fid-evaluator "${GUIDED_EVAL}" \
    --fd-dinov2-script "${FD_SCRIPT}" \
    --fd-real-imagefolder "${FD_REAL_IMAGEFOLDER}" \
    --device "${DEVICE}" \
    --fd-batch-size "${FD_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}"

echo "[DONE] ${MODEL_NAME} DATE=$(date)"
