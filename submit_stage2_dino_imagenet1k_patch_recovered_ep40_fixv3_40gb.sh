#!/bin/bash
set -euo pipefail

REPO="RAE_ROOT_PLACEHOLDER"
cd "${REPO}"

CONTAINER="${CONTAINER:-${REPO}/container.sif}"
WATCH_WORKER="${WATCH_WORKER:-${REPO}/watch_stage2_dino_imagenet1k_patch_recovered_ep40_fixv3_40gb.sh}"

VAL_DATA="${VAL_DATA:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/data/val}"
REAL_NPZ="${REAL_NPZ:-${REPO}/assets/datasets/imagenet1k_val_256.npz}"
TARGET_CKPT="${TARGET_CKPT:-${REPO}/ckpts/stage2_imagenet1k_patch_fixv3_40gb/in1k_s2_dino_patch80_ddt_b64_fixv3_40gb/checkpoints/ep-0000040.pt}"

PATCH_SAMPLE_CONFIG="${PATCH_SAMPLE_CONFIG:-${REPO}/configs/stage2/sampling/ImageNet256/imagenet1k/DINO_IN1K_B256_patch40_ddt_equal_fixv3_40gb.yaml}"
RECOVERED_SAMPLE_CONFIG="${RECOVERED_SAMPLE_CONFIG:-${REPO}/configs/stage2/sampling/ImageNet256/imagenet1k/recovered_aux/DINO_IN1K_B256_patch40_ddt_mhap_to_prepend_equal_fixv3_40gb.yaml}"

PATCH_RUN_DIR="${PATCH_RUN_DIR:-${REPO}/samples_stage2_imagenet1k_fixv3_40gb/in1k_s2_dino_patch40_ddt_equal_fixv3_40gb_cfg1}"
RECOVERED_RUN_DIR="${RECOVERED_RUN_DIR:-${REPO}/samples_stage2_imagenet1k_recovered_aux_fixv3_40gb/in1k_s2_dino_patch40_ddt_mhap_to_prepend_equal_fixv3_40gb_cfg1}"
EVAL_CSV="${EVAL_CSV:-${REPO}/samples_stage2_imagenet1k_fixv3_40gb/eval_metrics_stage2_imagenet1k_ep40_fixv3_40gb.csv}"

PATCH_MODEL_NAME="${PATCH_MODEL_NAME:-DINO_IN1K_patch40_ddt_fixv3_40gb}"
RECOVERED_MODEL_NAME="${RECOVERED_MODEL_NAME:-DINO_IN1K_patch40_ddt_mhap_fixv3_40gb}"
TRAIN_JOB_ID="${TRAIN_JOB_ID:-4311763}"
SBATCH_QOS="${SBATCH_QOS:-SLURM_QOS_PLACEHOLDER}"
POLL_SECONDS="${POLL_SECONDS:-300}"
POST_DETECT_SLEEP="${POST_DETECT_SLEEP:-180}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}"

mkdir -p "${REPO}/logs" "$(dirname "${EVAL_CSV}")"

watch_job=$(sbatch --parsable \
  --qos="${SBATCH_QOS}" \
  --job-name=in1k_s2_ep40_watch_fixv3_40gb \
  --export=ALL,CONTAINER="${CONTAINER}",VAL_DATA="${VAL_DATA}",REAL_NPZ="${REAL_NPZ}",TARGET_CKPT="${TARGET_CKPT}",PATCH_SAMPLE_CONFIG="${PATCH_SAMPLE_CONFIG}",RECOVERED_SAMPLE_CONFIG="${RECOVERED_SAMPLE_CONFIG}",PATCH_RUN_DIR="${PATCH_RUN_DIR}",RECOVERED_RUN_DIR="${RECOVERED_RUN_DIR}",EVAL_CSV="${EVAL_CSV}",PATCH_MODEL_NAME="${PATCH_MODEL_NAME}",RECOVERED_MODEL_NAME="${RECOVERED_MODEL_NAME}",TRAIN_JOB_ID="${TRAIN_JOB_ID}",SBATCH_QOS="${SBATCH_QOS}",POLL_SECONDS="${POLL_SECONDS}",POST_DETECT_SLEEP="${POST_DETECT_SLEEP}",MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS}" \
  "${WATCH_WORKER}")

echo "[watch] ${watch_job}"
echo "[target ckpt] ${TARGET_CKPT}"
echo "[patch run dir] ${PATCH_RUN_DIR}"
echo "[recovered run dir] ${RECOVERED_RUN_DIR}"
echo "[eval csv] ${EVAL_CSV}"
