#!/bin/bash
#SBATCH --job-name=in100_s2_sample_single
#SBATCH --partition=gpu-2d
#SBATCH --qos=SLURM_QOS_PLACEHOLDER
#SBATCH --constraint=40gb
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

echo "=== JOB INFO ==="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "================"

echo "=== GPU INFO ==="
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id --format=csv,noheader || true
echo "================"

REPO="RAE_ROOT_PLACEHOLDER"
CONTAINER="${CONTAINER:-${REPO}/container.sif}"
SRC="${REPO}/src"
NPROC="${NPROC:-2}"
CONFIG="${CONFIG:?CONFIG is required}"

export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="${SRC}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0

echo "=== PATHS ==="
echo "CONTAINER=${CONTAINER}"
echo "REPO=${REPO}"
echo "SRC=${SRC}"
echo "CONFIG=${CONFIG}"
echo "NPROC=${NPROC}"
echo "================"

apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${SRC}'
    torchrun --standalone --nproc_per_node=${NPROC} -m sample_ddp --sample-config '${CONFIG}'
  "

echo "End: $(date)"
echo "Done."
