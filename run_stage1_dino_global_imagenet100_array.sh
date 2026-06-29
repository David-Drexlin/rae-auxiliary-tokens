#!/bin/bash
#SBATCH --job-name=RAE_s1_dino_global_in100
#SBATCH --partition=gpu-2d
#SBATCH --array=0-2
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
IMAGE_SIZE="${IMAGE_SIZE:-256}"
PRECISION="${PRECISION:-bf16}"
EXP_PREFIX="${EXP_PREFIX:-DINO_IN100_GLOBAL_}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export TF_ENABLE_ONEDNN_OPTS=0

CONFIGS=(
  "DINO_decB_GlobalCLS.yaml"
  "DINO_decB_GlobalRegister.yaml"
  "DINO_decB_GlobalCLSRegister.yaml"
)

cfg_name="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
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
export EXPERIMENT_NAME="${EXP_PREFIX}${base_name}"
export RESULTS_DIR

echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-N/A}"
echo "[JOB] CONFIG=${CONFIG}"
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
      --precision '${PRECISION}'
  "

echo "[DONE] ${EXPERIMENT_NAME} DATE=$(date)"
