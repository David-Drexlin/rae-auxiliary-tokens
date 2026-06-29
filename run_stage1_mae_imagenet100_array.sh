#!/bin/bash
#SBATCH --job-name=RAE_s1_mae_in100
#SBATCH --partition=gpu-2d
#SBATCH --constraint=40gb
#SBATCH --array=0-11
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

mkdir -p logs
cd RAE_ROOT_PLACEHOLDER

CONTAINER="${CONTAINER:-RAE_ROOT_PLACEHOLDER/container.sif}"
CONFIG_DIR="${CONFIG_DIR:-RAE_ROOT_PLACEHOLDER/configs/stage1/training/ImageNet}"
DATA="${DATA:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/train}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-RAE_ROOT_PLACEHOLDER/src/train_stage1.py}"
RESULTS_DIR="${RESULTS_DIR:-RAE_ROOT_PLACEHOLDER/ckpts}"
NPROC="${NPROC:-2}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
PRECISION="${PRECISION:-bf16}"
EXP_PREFIX="${EXP_PREFIX:-MAE_IN100_}"
CONFIG_NAME="${CONFIG_NAME:-}"
GLOBAL_SEED="${GLOBAL_SEED:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export TF_ENABLE_ONEDNN_OPTS=0

CONFIGS=(
  "MAE_decB.yaml"
  "MAE_decB_Patch+CLS_AdaLN.yaml"
  "MAE_decB_Patch+CLS_prepend.yaml"
  "MAE_decB_Patch+CLS_CA.yaml"
)

if [[ -n "${CONFIG_NAME}" ]]; then
  cfg_name="${CONFIG_NAME}"
  if [[ -z "${GLOBAL_SEED}" ]]; then
    GLOBAL_SEED=0
  fi
else
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "[ERROR] CONFIG_NAME is not set and SLURM_ARRAY_TASK_ID is unavailable."
    exit 1
  fi
  num_seeds=3
  total_runs=$(( ${#CONFIGS[@]} * num_seeds ))
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= total_runs )); then
    echo "[ERROR] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is out of range 0-$(( total_runs - 1 ))."
    exit 1
  fi
  config_idx=$(( SLURM_ARRAY_TASK_ID / num_seeds ))
  GLOBAL_SEED=$(( SLURM_ARRAY_TASK_ID % num_seeds ))
  cfg_name="${CONFIGS[$config_idx]}"
fi

CONFIG="${CONFIG_DIR}/${cfg_name}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Missing config: ${CONFIG}"
  exit 1
fi

if [[ ! -d "${DATA}" ]]; then
  echo "[ERROR] Missing dataset directory: ${DATA}"
  exit 1
fi

base_name="$(basename "${cfg_name}" .yaml)"
seed_suffix="seed${GLOBAL_SEED}"
export EXPERIMENT_NAME="${EXP_PREFIX}${base_name}_${seed_suffix}"
export RESULTS_DIR

echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-N/A}"
echo "[JOB] CONFIG=${CONFIG}"
echo "[JOB] GLOBAL_SEED=${GLOBAL_SEED}"
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
      --config '${CONFIG}' \
      --data-path '${DATA}' \
      --image-size '${IMAGE_SIZE}' \
      --precision '${PRECISION}' \
      --global-seed '${GLOBAL_SEED}'
  "

echo "[DONE] ${EXPERIMENT_NAME} DATE=$(date)"
