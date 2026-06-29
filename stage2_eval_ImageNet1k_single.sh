#!/bin/bash
#SBATCH --job-name=stage2_in1k_eval
#SBATCH --partition=gpu-2d
#SBATCH --qos=SLURM_QOS_PLACEHOLDER
#SBATCH --constraint=40gb
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="RAE_ROOT_PLACEHOLDER"
cd "${REPO}"
mkdir -p logs

CONTAINER="${CONTAINER:-${REPO}/container.sif}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
MODEL_NAME="${MODEL_NAME:-$(basename "${RUN_DIR}")}"
CSV_OUT="${CSV_OUT:-${REPO}/samples_stage2_imagenet1k/eval_metrics_stage2_imagenet1k.csv}"

FID_REAL_NPZ="${FID_REAL_NPZ:-${REPO}/assets/datasets/imagenet1k_val_256.npz}"
FID_EVALUATOR="${FID_EVALUATOR:-HOME_PLACEHOLDER/guided-diffusion/evaluations/evaluator.py}"
FD_DINOV2_SCRIPT="${FD_DINOV2_SCRIPT:-HOME_PLACEHOLDER/FD-DINOv2/src/pytorch_fd/fd_score.py}"
FD_REAL_IMAGEFOLDER="${FD_REAL_IMAGEFOLDER:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/data/val}"
DEVICE="${DEVICE:-cuda:0}"
FD_BATCH_SIZE="${FD_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-50000}"

echo "=== JOB INFO ==="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "RUN_DIR: ${RUN_DIR}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "CSV_OUT: ${CSV_OUT}"
echo "MAX_SAMPLES: ${MAX_SAMPLES}"
echo "================"

echo "=== GPU INFO ==="
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id --format=csv,noheader || true
echo "================"

test -d "${RUN_DIR}" || { echo "[error] missing RUN_DIR: ${RUN_DIR}"; exit 1; }
test -f "${FID_REAL_NPZ}" || { echo "[error] missing FID_REAL_NPZ: ${FID_REAL_NPZ}"; exit 1; }
test -f "${FID_EVALUATOR}" || { echo "[error] missing FID_EVALUATOR: ${FID_EVALUATOR}"; exit 1; }
test -f "${FD_DINOV2_SCRIPT}" || { echo "[error] missing FD_DINOV2_SCRIPT: ${FD_DINOV2_SCRIPT}"; exit 1; }
test -d "${FD_REAL_IMAGEFOLDER}" || { echo "[error] missing FD_REAL_IMAGEFOLDER: ${FD_REAL_IMAGEFOLDER}"; exit 1; }

mkdir -p "$(dirname "${CSV_OUT}")"
LOCK_FILE="${CSV_OUT}.lock"

SAMPLES_NPZ="${RUN_DIR}/samples.npz"
GENERATED_NPZ="${RUN_DIR}/$(basename "${RUN_DIR}").npz"
FALLBACK_NPZ="${RUN_DIR}.npz"

if [ ! -f "${SAMPLES_NPZ}" ]; then
  if [ -f "${GENERATED_NPZ}" ]; then
    mv "${GENERATED_NPZ}" "${SAMPLES_NPZ}"
  elif [ -f "${FALLBACK_NPZ}" ]; then
    mv "${FALLBACK_NPZ}" "${SAMPLES_NPZ}"
  else
    existing_npz=$(find "${RUN_DIR}" -maxdepth 1 -type f -name '*.npz' | head -n 1 || true)
    if [ -n "${existing_npz}" ]; then
      mv "${existing_npz}" "${SAMPLES_NPZ}"
    else
      echo "[pack] samples.npz missing, packing PNGs in ${RUN_DIR}"
      apptainer exec --nv \
        -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
        -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
        "${CONTAINER}" \
        bash -lc "
          set -euo pipefail
          python3 '${REPO}/pack_images.py' '${RUN_DIR}' 256 '${RUN_DIR}'
        "
      if [ -f "${GENERATED_NPZ}" ]; then
        mv "${GENERATED_NPZ}" "${SAMPLES_NPZ}"
      elif [ -f "${FALLBACK_NPZ}" ]; then
        mv "${FALLBACK_NPZ}" "${SAMPLES_NPZ}"
      else
        existing_npz=$(find "${RUN_DIR}" -maxdepth 1 -type f -name '*.npz' | head -n 1 || true)
        if [ -n "${existing_npz}" ]; then
          mv "${existing_npz}" "${SAMPLES_NPZ}"
        fi
      fi
    fi
  fi
fi

test -f "${SAMPLES_NPZ}" || { echo "[error] missing ${RUN_DIR}/samples.npz after pack"; exit 1; }

flock "${LOCK_FILE}" \
  apptainer exec --nv \
    -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
    -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
    "${CONTAINER}" \
    python3 "${REPO}/src/stage2/eval_stage_1.py" \
      --run-dir "${RUN_DIR}" \
      --model-name "${MODEL_NAME}" \
      --csv-out "${CSV_OUT}" \
      --fid-real-npz "${FID_REAL_NPZ}" \
      --fid-evaluator "${FID_EVALUATOR}" \
      --fd-dinov2-script "${FD_DINOV2_SCRIPT}" \
      --fd-real-imagefolder "${FD_REAL_IMAGEFOLDER}" \
      --device "${DEVICE}" \
      --fd-batch-size "${FD_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --max-samples "${MAX_SAMPLES}"

echo "End: $(date)"
echo "Done."
