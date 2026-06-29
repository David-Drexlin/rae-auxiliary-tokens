#!/bin/bash
#SBATCH --job-name=RAE_s1_dino_in1k_8gpu
#SBATCH --partition=gpu-2d
#SBATCH --qos=SLURM_QOS_PLACEHOLDER
#SBATCH --constraint=80gb
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=RAE_ROOT_PLACEHOLDER/logs/%x-%j.out
#SBATCH --error=RAE_ROOT_PLACEHOLDER/logs/%x-%j.err

set -euo pipefail

mkdir -p RAE_ROOT_PLACEHOLDER/logs
cd RAE_ROOT_PLACEHOLDER

CONTAINER="${CONTAINER:-RAE_ROOT_PLACEHOLDER/container.sif}"
CONFIG_DIR="${CONFIG_DIR:-RAE_ROOT_PLACEHOLDER/configs/stage1/training/ImageNet1k}"
CONFIG_NAME="${CONFIG_NAME:-DINO_decB_Patch+Register+CLS_prepend.yaml}"
DATA="${DATA:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/data/train}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-RAE_ROOT_PLACEHOLDER/src/train_stage1.py}"
RESULTS_DIR="${RESULTS_DIR:-RAE_ROOT_PLACEHOLDER/ckpts}"
NPROC="${NPROC:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
PRECISION="${PRECISION:-bf16}"
EXP_PREFIX="${EXP_PREFIX:-DINO_IN1K_}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export TF_ENABLE_ONEDNN_OPTS=0

CONFIG="${CONFIG_DIR}/${CONFIG_NAME}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Missing config: ${CONFIG}"
  exit 1
fi
if [[ ! -d "${DATA}" ]]; then
  echo "[ERROR] Missing dataset directory: ${DATA}"
  exit 1
fi

base_name="$(basename "${CONFIG_NAME}" .yaml)"
export EXPERIMENT_NAME="${EXP_PREFIX}${base_name}_seed${GLOBAL_SEED}"
export RESULTS_DIR

TMP_CONFIG="$(mktemp /tmp/dino_in1k_8gpu.XXXXXX.yaml)"
trap 'rm -f "${TMP_CONFIG}"' EXIT

echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] CONFIG=${CONFIG}"
echo "[JOB] GLOBAL_SEED=${GLOBAL_SEED}"
echo "[JOB] GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "[JOB] DATA=${DATA}"
echo "[JOB] IMAGE_SIZE=${IMAGE_SIZE}"
echo "[JOB] PRECISION=${PRECISION}"
echo "[JOB] NPROC=${NPROC}"
echo "[JOB] EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "[JOB] RESULTS_DIR=${RESULTS_DIR}"
echo "============================================================"

echo "=== GPU INFO ==="
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id --format=csv,noheader || true
echo "================"

apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd RAE_ROOT_PLACEHOLDER

    python3 - <<'PY'
from omegaconf import OmegaConf
cfg = OmegaConf.load('${CONFIG}')
cfg.training.global_batch_size = int('${GLOBAL_BATCH_SIZE}')
OmegaConf.save(cfg, '${TMP_CONFIG}')
print('[tmp-config] wrote ${TMP_CONFIG}')
print('[tmp-config] training.global_batch_size=', cfg.training.global_batch_size)
PY

    export EXPERIMENT_NAME='${EXPERIMENT_NAME}'
    export RESULTS_DIR='${RESULTS_DIR}'
    export OMP_NUM_THREADS='${OMP_NUM_THREADS}'
    export MKL_NUM_THREADS='${MKL_NUM_THREADS}'
    export NUMEXPR_NUM_THREADS='${NUMEXPR_NUM_THREADS}'
    export PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF}'
    export PYTHONDONTWRITEBYTECODE='${PYTHONDONTWRITEBYTECODE}'
    export TF_ENABLE_ONEDNN_OPTS='${TF_ENABLE_ONEDNN_OPTS}'

    torchrun --standalone --nproc_per_node='${NPROC}' \
      '${TRAIN_SCRIPT}' \
      --config '${TMP_CONFIG}' \
      --data-path '${DATA}' \
      --image-size '${IMAGE_SIZE}' \
      --precision '${PRECISION}' \
      --global-seed '${GLOBAL_SEED}'
  "

echo "[DONE] ${EXPERIMENT_NAME} DATE=$(date)"
