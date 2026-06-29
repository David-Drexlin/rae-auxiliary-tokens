#!/bin/bash
#SBATCH --job-name=siglip2_stats_in100
#SBATCH --partition=gpu-2d
#SBATCH --constraint=40gb
#SBATCH --array=0-2
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

mkdir -p logs
cd RAE_ROOT_PLACEHOLDER

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1

CONTAINER="${CONTAINER:-RAE_ROOT_PLACEHOLDER/container.sif}"
CFG_DIR="${CFG_DIR:-RAE_ROOT_PLACEHOLDER/configs/stage1/training/ImageNet}"
DATA="${DATA:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/train}"
STATS_ROOT="${STATS_ROOT:-RAE_ROOT_PLACEHOLDER/assets/stats/ImageNet100}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
STATS_SUBDIR="${STATS_SUBDIR:-crop${IMAGE_SIZE}}"
NPROC="${NPROC:-1}"
PRECISION="${PRECISION:-fp32}"
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-}"

if [[ -n "${STATS_SUBDIR}" ]]; then
  OUT_DIR="${STATS_ROOT}/${STATS_SUBDIR}"
else
  OUT_DIR="${STATS_ROOT}"
fi
mkdir -p "${OUT_DIR}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    CONFIG="${CFG_DIR}/SigLIP2_decB.yaml"
    OUT_PT="${OUT_DIR}/siglip2_base.pt"
    ;;
  1)
    CONFIG="${CFG_DIR}/SigLIP2_decB_Patch+Pooler_AdaLN.yaml"
    OUT_PT="${OUT_DIR}/siglip2_pooler_pooled.pt"
    ;;
  2)
    CONFIG="${CFG_DIR}/SigLIP2_decB_Patch+Pooler_prepend.yaml"
    OUT_PT="${OUT_DIR}/siglip2_pooler_tokens.pt"
    ;;
  *)
    echo "[ERROR] Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Missing config: ${CONFIG}"
  exit 1
fi

if [[ ! -d "${DATA}" ]]; then
  echo "[ERROR] Missing dataset directory: ${DATA}"
  exit 1
fi

echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-N/A}"
echo "[JOB] CONFIG=${CONFIG}"
echo "[JOB] DATA=${DATA}"
echo "[JOB] IMAGE_SIZE=${IMAGE_SIZE}"
echo "[JOB] PRECISION=${PRECISION}"
echo "[JOB] PER_PROC_BATCH_SIZE=${PER_PROC_BATCH_SIZE}"
echo "[JOB] OUT_PT=${OUT_PT}"
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

    extra_args=()
    if [[ -n '${NUM_SAMPLES}' ]]; then
      extra_args+=(--num-samples '${NUM_SAMPLES}')
    fi

    torchrun --standalone --nproc_per_node='${NPROC}' \
      src/calculate_stat.py \
      --config '${CONFIG}' \
      --data-path '${DATA}' \
      --output-pt '${OUT_PT}' \
      --sample-dir '${STATS_ROOT}' \
      --image-size '${IMAGE_SIZE}' \
      --per-proc-batch-size '${PER_PROC_BATCH_SIZE}' \
      --num-workers '${NUM_WORKERS}' \
      --precision '${PRECISION}' \
      --aux-stats-mode auto \
      --tf32 \
      \${extra_args[@]}
  "

echo "[DONE] wrote ${OUT_PT} DATE=$(date)"
