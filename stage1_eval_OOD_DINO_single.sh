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
CONFIG="${CONFIG:?CONFIG is required}"
CKPT_DIR="${CKPT_DIR:?CKPT_DIR is required}"
DATA="${DATA:?DATA is required}"
OUTDIR="${OUTDIR:-RAE_ROOT_PLACEHOLDER/assets/recon_samples}"
SAVE_FOLDER="${SAVE_FOLDER:?SAVE_FOLDER is required}"
MODEL_NAME="${MODEL_NAME:-${SAVE_FOLDER}}"
GUIDED_EVAL="${GUIDED_EVAL:-HOME_PLACEHOLDER/guided-diffusion/evaluations/evaluator.py}"
FD_SCRIPT="${FD_SCRIPT:-HOME_PLACEHOLDER/FD-DINOv2/src/pytorch_fd/fd_score.py}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-bf16}"
NUM_SAMPLES="${NUM_SAMPLES:-99999999}"
CSV_OUT="${CSV_OUT:-RAE_ROOT_PLACEHOLDER/assets/recon_samples/eval_metrics_ood_dino.csv}"
FD_BATCH_SIZE="${FD_BATCH_SIZE:-128}"
PREFER_FULL_MODEL_CKPT="${PREFER_FULL_MODEL_CKPT:-1}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export TF_ENABLE_ONEDNN_OPTS=0
export TOKENIZERS_PARALLELISM=false

echo "=== JOB INFO ==="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "CONFIG: ${CONFIG}"
echo "CKPT_DIR: ${CKPT_DIR}"
echo "DATA: ${DATA}"
echo "SAVE_FOLDER: ${SAVE_FOLDER}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "NUM_SAMPLES: ${NUM_SAMPLES}"
echo "================"

echo "=== GPU INFO ==="
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id --format=csv,noheader || true
echo "================"

test -f "${CONTAINER}" || { echo "[error] missing CONTAINER: ${CONTAINER}"; exit 1; }
test -f "${CONFIG}" || { echo "[error] missing CONFIG: ${CONFIG}"; exit 1; }
test -d "${CKPT_DIR}" || { echo "[error] missing CKPT_DIR: ${CKPT_DIR}"; exit 1; }
test -d "${DATA}" || { echo "[error] missing DATA: ${DATA}"; exit 1; }
test -f "${GUIDED_EVAL}" || { echo "[error] missing GUIDED_EVAL: ${GUIDED_EVAL}"; exit 1; }
test -f "${FD_SCRIPT}" || { echo "[error] missing FD_SCRIPT: ${FD_SCRIPT}"; exit 1; }

EMA_CKPT=$(find "${CKPT_DIR}" -maxdepth 1 -type f -name 'decoder_ema_ep-*.pt' | sort | tail -n 1)
if [[ -z "${EMA_CKPT}" ]]; then
  EMA_CKPT=$(find "${CKPT_DIR}" -maxdepth 1 -type f -name 'decoder_ep-*.pt' | sort | tail -n 1)
fi
if [[ -z "${EMA_CKPT}" ]]; then
  echo "[error] could not find decoder checkpoint in ${CKPT_DIR}"
  exit 1
fi
echo "EMA_CKPT=${EMA_CKPT}"

FULL_MODEL_CKPT=""
if [[ "${PREFER_FULL_MODEL_CKPT}" == "1" ]]; then
  FULL_MODEL_CKPT=$(find "${CKPT_DIR}" -maxdepth 1 -type f \( -name 'ep-*.pt' -o -name 'ep-last.pt' \) | sort | tail -n 1)
fi
if [[ -n "${FULL_MODEL_CKPT}" ]]; then
  echo "FULL_MODEL_CKPT=${FULL_MODEL_CKPT}"
fi

TMP_CONFIG=$(mktemp /tmp/dino_ood_eval_cfg.XXXXXX.yaml)
cleanup() {
  rm -f "${TMP_CONFIG}"
}
trap cleanup EXIT

apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  python3 - <<PY
from omegaconf import OmegaConf

cfg = OmegaConf.load("${CONFIG}")
full_ckpt = "${FULL_MODEL_CKPT}".strip()
if full_ckpt:
    cfg.stage_1.ckpt = full_ckpt
    cfg.stage_1.params.pretrained_decoder_path = None
else:
    if "ckpt" in cfg.stage_1:
        del cfg.stage_1["ckpt"]
    cfg.stage_1.params.pretrained_decoder_path = "${EMA_CKPT}"
OmegaConf.save(cfg, "${TMP_CONFIG}")
print("[tmp-config] wrote ${TMP_CONFIG}")
if full_ckpt:
    print(f"[tmp-config] stage_1.ckpt={full_ckpt}")
else:
    print("[tmp-config] pretrained_decoder_path=${EMA_CKPT}")
PY

RECON_DIR="${OUTDIR}/${SAVE_FOLDER}"

echo "=== STAGE 1: RECONSTRUCT OOD DATA ==="
apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  env SAVE_FOLDER="${SAVE_FOLDER}" \
  torchrun --standalone --nproc_per_node=1 \
    RAE_ROOT_PLACEHOLDER/src/stage1_sample_ddp.py \
      --config "${TMP_CONFIG}" \
      --sample-dir "${OUTDIR}" \
      --data-path "${DATA}" \
      --image-size "${IMAGE_SIZE}" \
      --per-proc-batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --precision "${PRECISION}" \
      --num-samples "${NUM_SAMPLES}" \
      --pack-npz \
      --pack-real-npz \
      --save-pngs

test -d "${RECON_DIR}" || { echo "[error] missing RECON_DIR: ${RECON_DIR}"; exit 1; }
test -f "${RECON_DIR}/samples.npz" || { echo "[error] missing fake samples.npz"; exit 1; }
test -f "${RECON_DIR}/real_samples.npz" || { echo "[error] missing real_samples.npz"; exit 1; }

echo "=== STAGE 2: EVALUATE OOD RECONSTRUCTIONS ==="
apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  env XFORMERS_DISABLED=1 \
  python3 RAE_ROOT_PLACEHOLDER/src/eval_stage1_recon.py \
    --run-dir "${RECON_DIR}" \
    --model-name "${MODEL_NAME}" \
    --csv-out "${CSV_OUT}" \
    --fid-real-npz "${RECON_DIR}/real_samples.npz" \
    --fid-evaluator "${GUIDED_EVAL}" \
    --fd-dinov2-script "${FD_SCRIPT}" \
    --fd-real-imagefolder "${DATA}" \
    --device cuda:0 \
    --fd-batch-size "${FD_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}"

echo "[DONE] ${MODEL_NAME} DATE=$(date)"
