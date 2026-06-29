#!/bin/bash
#SBATCH --job-name=in100_s2_dino
#SBATCH --partition=gpu-2d
#SBATCH --qos=SLURM_QOS_PLACEHOLDER
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

echo "=== GPU INFO ==="
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id --format=csv,noheader || true
echo "================"

CONTAINER="RAE_ROOT_PLACEHOLDER/container.sif"
REPO="RAE_ROOT_PLACEHOLDER"
PY_SCRIPT="${REPO}/src/train.py"
CONFIG="${CONFIG:-${REPO}/configs/stage2/training/ImageNet256/DiTDH-S_DINO_IN100_from_ckpt.yaml}"
DATA="${DATA:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/train}"
RESULTS_DIR="${RESULTS_DIR:-${REPO}/ckpts/stage2}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
PRECISION="${PRECISION:-bf16}"
NPROC="${NPROC:-2}"
GLOBAL_SEED="${GLOBAL_SEED:-}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-imagenet100_dino_decb_stage2_ddths}"
CKPT_ARG=()
if [[ -n "${CKPT:-}" ]]; then
  CKPT_ARG+=(--ckpt "$CKPT")
fi
SEED_ARG=()
if [[ -n "${GLOBAL_SEED}" ]]; then
  SEED_ARG+=(--global-seed "$GLOBAL_SEED")
fi

echo "=== RUN INFO ==="
echo "CONFIG=${CONFIG}"
echo "DATA=${DATA}"
if [[ -n "${GLOBAL_SEED}" ]]; then
  echo "GLOBAL_SEED=${GLOBAL_SEED}"
fi
echo "================"

apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "$CONTAINER" \
  torchrun --standalone --nproc_per_node="${NPROC}" \
    "$PY_SCRIPT" \
    --config "$CONFIG" \
    --data-path "$DATA" \
    --results-dir "$RESULTS_DIR" \
    --image-size "$IMAGE_SIZE" \
    --precision "$PRECISION" \
    "${SEED_ARG[@]}" \
    "${CKPT_ARG[@]}"
