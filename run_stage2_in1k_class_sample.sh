#!/bin/bash
#SBATCH --job-name=in1k_s2_class_sample
#SBATCH --partition=gpu-2h
#SBATCH --qos=SLURM_QOS_PLACEHOLDER
#SBATCH --constraint=40gb
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
SAMPLE_CONFIG="${SAMPLE_CONFIG:?SAMPLE_CONFIG is required}"
CLASS_CSV="${CLASS_CSV:?CLASS_CSV is required}"
Y_VOCAB_JSON="${Y_VOCAB_JSON:?Y_VOCAB_JSON is required}"
SAVE_FOLDER="${SAVE_FOLDER:?SAVE_FOLDER is required}"

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
echo "SAMPLE_CONFIG=${SAMPLE_CONFIG}"
echo "CLASS_CSV=${CLASS_CSV}"
echo "Y_VOCAB_JSON=${Y_VOCAB_JSON}"
echo "SAVE_FOLDER=${SAVE_FOLDER}"
echo "NPROC=${NPROC}"
echo "================"

apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${SRC}'
    torchrun --standalone --nproc_per_node=${NPROC} -m sample_ddp \
      --sample-config '${SAMPLE_CONFIG}' \
      --cond-mode actual_class \
      --csv-path '${CLASS_CSV}' \
      --count-level slide \
      --id-field sample_id \
      --label-field label \
      --y-vocab-json '${Y_VOCAB_JSON}' \
      --save-folder '${SAVE_FOLDER}'
  "

echo "End: $(date)"
echo "Done."
