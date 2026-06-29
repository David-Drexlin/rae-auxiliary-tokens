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
DATA="${DATA:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/val}"
OUTDIR="${OUTDIR:-RAE_ROOT_PLACEHOLDER/assets/recon_samples}"
SAVE_FOLDER="${SAVE_FOLDER:?SAVE_FOLDER is required}"
REAL_PNG_DIR="${REAL_PNG_DIR:-RAE_ROOT_PLACEHOLDER/assets/datasets/imagenet100_val_flat}"
REAL_NPZ="${REAL_NPZ:-RAE_ROOT_PLACEHOLDER/assets/datasets/imagenet100_val_256.npz}"
GUIDED_EVAL="${GUIDED_EVAL:-HOME_PLACEHOLDER/guided-diffusion/evaluations/evaluator.py}"
FD_ROOT="${FD_ROOT:-HOME_PLACEHOLDER/FD-DINOv2}"
FD_SCRIPT="${FD_SCRIPT:-${FD_ROOT}/src/pytorch_fd/fd_score.py}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-bf16}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
TRAIN_EXP_NAME="${TRAIN_EXP_NAME:-unknown}"
METRICS_CSV="${METRICS_CSV:-RAE_ROOT_PLACEHOLDER/assets/recon_samples/eval_metrics_seeded_dino.csv}"
PREFER_FULL_MODEL_CKPT="${PREFER_FULL_MODEL_CKPT:-1}"
AUX_EVAL_OVERRIDE="${AUX_EVAL_OVERRIDE:-}"
AUX_EVAL_SIGMA="${AUX_EVAL_SIGMA:-}"
AUX_EVAL_SCALE="${AUX_EVAL_SCALE:-}"
AUX_EVAL_ALPHA="${AUX_EVAL_ALPHA:-}"
AUX_EVAL_SHUFFLE_OFFSET="${AUX_EVAL_SHUFFLE_OFFSET:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export TF_ENABLE_ONEDNN_OPTS=0
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTDIR}" "$(dirname "${METRICS_CSV}")"

echo "=== JOB INFO ==="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "CONFIG: ${CONFIG}"
echo "CKPT_DIR: ${CKPT_DIR}"
echo "TRAIN_EXP_NAME: ${TRAIN_EXP_NAME}"
echo "SAVE_FOLDER: ${SAVE_FOLDER}"
echo "AUX_EVAL_OVERRIDE: ${AUX_EVAL_OVERRIDE:-none}"
echo "AUX_EVAL_SIGMA: ${AUX_EVAL_SIGMA:-}"
echo "AUX_EVAL_SCALE: ${AUX_EVAL_SCALE:-}"
echo "AUX_EVAL_ALPHA: ${AUX_EVAL_ALPHA:-}"
echo "AUX_EVAL_SHUFFLE_OFFSET: ${AUX_EVAL_SHUFFLE_OFFSET:-}"
echo "================"

echo "=== GPU INFO ==="
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id --format=csv,noheader || true
echo "================"

test -f "${CONTAINER}" || { echo "[error] missing CONTAINER: ${CONTAINER}"; exit 1; }
test -f "${CONFIG}" || { echo "[error] missing CONFIG: ${CONFIG}"; exit 1; }
test -d "${CKPT_DIR}" || { echo "[error] missing CKPT_DIR: ${CKPT_DIR}"; exit 1; }
test -d "${DATA}" || { echo "[error] missing DATA: ${DATA}"; exit 1; }
test -f "${REAL_NPZ}" || { echo "[error] missing REAL_NPZ: ${REAL_NPZ}"; exit 1; }
test -d "${REAL_PNG_DIR}" || { echo "[error] missing REAL_PNG_DIR: ${REAL_PNG_DIR}"; exit 1; }
test -f "${GUIDED_EVAL}" || { echo "[error] missing GUIDED_EVAL: ${GUIDED_EVAL}"; exit 1; }
test -d "${FD_ROOT}" || { echo "[error] missing FD_ROOT: ${FD_ROOT}"; exit 1; }
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

TMP_CONFIG=$(mktemp /tmp/dino_seeded_eval_cfg.XXXXXX.yaml)

apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  python3 - <<PY
import os
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
override = os.environ.get("AUX_EVAL_OVERRIDE", "").strip()
if override:
    cfg.stage_1.params.aux_eval_override = override
sigma = os.environ.get("AUX_EVAL_SIGMA", "").strip()
if sigma:
    cfg.stage_1.params.aux_eval_sigma = float(sigma)
scale = os.environ.get("AUX_EVAL_SCALE", "").strip()
if scale:
    cfg.stage_1.params.aux_eval_scale = float(scale)
alpha = os.environ.get("AUX_EVAL_ALPHA", "").strip()
if alpha:
    cfg.stage_1.params.aux_eval_alpha = float(alpha)
shuffle_offset = os.environ.get("AUX_EVAL_SHUFFLE_OFFSET", "").strip()
if shuffle_offset:
    cfg.stage_1.params.aux_eval_shuffle_offset = int(shuffle_offset)
OmegaConf.save(cfg, "${TMP_CONFIG}")
print("[tmp-config] wrote ${TMP_CONFIG}")
if full_ckpt:
    print(f"[tmp-config] stage_1.ckpt={full_ckpt}")
else:
    print("[tmp-config] pretrained_decoder_path=${EMA_CKPT}")
if override:
    print(f"[tmp-config] aux_eval_override={override}")
if sigma:
    print(f"[tmp-config] aux_eval_sigma={sigma}")
if scale:
    print(f"[tmp-config] aux_eval_scale={scale}")
if alpha:
    print(f"[tmp-config] aux_eval_alpha={alpha}")
if shuffle_offset:
    print(f"[tmp-config] aux_eval_shuffle_offset={shuffle_offset}")
PY

RECON_DIR="${OUTDIR}/${SAVE_FOLDER}"
FAKE_NPZ="${RECON_DIR}/samples.npz"
FID_LOG="${RECON_DIR}/fid.log"
FD_DINO_LOG="${RECON_DIR}/fd_dinov2.log"

SAMPLE_ARGS=()
if [[ -n "${NUM_SAMPLES}" ]]; then
  SAMPLE_ARGS+=(--num-samples "${NUM_SAMPLES}")
fi

echo "=== STAGE 1: RECONSTRUCT ==="
apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  torchrun --standalone --nproc_per_node=1 \
    RAE_ROOT_PLACEHOLDER/src/stage1_sample_ddp.py \
      --config "${TMP_CONFIG}" \
      --sample-dir "${OUTDIR}" \
      --data-path "${DATA}" \
      --image-size "${IMAGE_SIZE}" \
      --per-proc-batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --precision "${PRECISION}" \
      --pack-npz \
      --no-pack-real-npz \
      --save-pngs \
      "${SAMPLE_ARGS[@]}"

echo
echo "=== STAGE 2: STANDARD FID (NPZ) ==="
apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  env EVALUATOR_PATH="${GUIDED_EVAL}" REAL_NPZ_PATH="${REAL_NPZ}" FAKE_NPZ_PATH="${FAKE_NPZ}" \
  python3 - <<'PY' 2>&1 | tee "${FID_LOG}"
import os
import runpy
import numpy as np
import sys

if not hasattr(np, "bool"):
    np.bool = np.bool_

evaluator_path = os.environ["EVALUATOR_PATH"]
real_npz = os.environ["REAL_NPZ_PATH"]
fake_npz = os.environ["FAKE_NPZ_PATH"]

sys.argv = [evaluator_path, real_npz, fake_npz]
runpy.run_path(evaluator_path, run_name="__main__")
PY

echo
echo "=== STAGE 3: FD-DINOv2 (PNG dirs) ==="
apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  env RECON_DIR="${RECON_DIR}" REAL_PNG_DIR="${REAL_PNG_DIR}" FD_ROOT="${FD_ROOT}" FD_SCRIPT="${FD_SCRIPT}" XFORMERS_DISABLED=1 \
  bash -lc '
    set -euo pipefail
    cd "$FD_ROOT"
    export PYTHONPATH="$FD_ROOT:$FD_ROOT/src:${PYTHONPATH:-}"
    python "$FD_SCRIPT" \
      --batch-size 128 \
      --num-workers 8 \
      --device cuda:0 \
      "$REAL_PNG_DIR" \
      "$RECON_DIR"
  ' 2>&1 | tee "${FD_DINO_LOG}"

FID_VALUE="$(grep -E '^FID:' "${FID_LOG}" 2>/dev/null | tail -n1 | awk '{print $2}')"
FD_DINO_VALUE="$(grep -E '^FD-DINOv2:' "${FD_DINO_LOG}" 2>/dev/null | tail -n1 | awk '{print $2}')"
STATUS="ok"
if [[ -z "${FID_VALUE}" || -z "${FD_DINO_VALUE}" ]]; then
  STATUS="missing_metric"
fi

mkdir -p "$(dirname "${METRICS_CSV}")"
{
  flock 9
  if [[ ! -f "${METRICS_CSV}" ]]; then
    echo "timestamp,slurm_job_id,train_exp_name,save_folder,config,ckpt,fid,fd_dinov2,fid_log,fd_dino_log,status" > "${METRICS_CSV}"
  fi
  TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '"%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s"\n' \
    "${TIMESTAMP}" \
    "${SLURM_JOB_ID:-N/A}" \
    "${TRAIN_EXP_NAME}" \
    "${SAVE_FOLDER}" \
    "${CONFIG}" \
    "${EMA_CKPT}" \
    "${FID_VALUE}" \
    "${FD_DINO_VALUE}" \
    "${FID_LOG}" \
    "${FD_DINO_LOG}" \
    "${STATUS}" >> "${METRICS_CSV}"
} 9>>"${METRICS_CSV}.lock"

rm -f "${TMP_CONFIG}"

echo
echo "=== DONE ==="
echo "FID=${FID_VALUE}"
echo "FD-DINOv2=${FD_DINO_VALUE}"
echo "METRICS_CSV=${METRICS_CSV}"
echo "End: $(date)"
