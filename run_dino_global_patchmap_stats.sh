#!/bin/bash
#SBATCH --job-name=dino_global_patchmap_stats
#SBATCH --partition=gpu-2d
#SBATCH --qos=SLURM_QOS_PLACEHOLDER
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

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
CONFIG="${CONFIG:-RAE_ROOT_PLACEHOLDER/configs/stage1/training/ImageNet/DINO_decB_GlobalPatchMAP5.yaml}"
DATA="${DATA:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/train}"
OUT_PT="${OUT_PT:-RAE_ROOT_PLACEHOLDER/assets/stats/ImageNet100/crop256/dino_global_patchmap5_latent.pt}"
STATS_ROOT="${STATS_ROOT:-RAE_ROOT_PLACEHOLDER/assets/stats/ImageNet100}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NPROC="${NPROC:-1}"
PRECISION="${PRECISION:-fp32}"
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Missing config: ${CONFIG}"
  exit 1
fi

if [[ ! -d "${DATA}" ]]; then
  echo "[ERROR] Missing dataset directory: ${DATA}"
  exit 1
fi

mkdir -p "$(dirname "${OUT_PT}")"

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
      --aux-stats-mode none \
      --tf32 \
      \${extra_args[@]}
  "
