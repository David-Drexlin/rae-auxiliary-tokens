#!/bin/bash
set -euo pipefail

REPO="RAE_ROOT_PLACEHOLDER"
cd "${REPO}"

CONTAINER="${CONTAINER:-${REPO}/container.sif}"
TRAIN_WORKER="${TRAIN_WORKER:-${REPO}/train_ImageNet_DiT.sh}"
EPOCH40_WATCH_WORKER="${EPOCH40_WATCH_WORKER:-${REPO}/watch_stage2_dino_imagenet1k_patch_recovered_ep40_fixv3_40gb.sh}"
NAN_WATCH_WORKER="${NAN_WATCH_WORKER:-${REPO}/watch_stage2_dino_imagenet1k_patch80_nan_recover_fixv3_40gb.sh}"

TRAIN_DATA="${TRAIN_DATA:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/data/train}"
VAL_DATA="${VAL_DATA:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/data/val}"
REAL_NPZ="${REAL_NPZ:-${REPO}/assets/datasets/imagenet1k_val_256.npz}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO}/configs/stage2/training/ImageNet256/imagenet1k/DiTDH-S_DINO_IN1K_B256_patch_80ep_b64.yaml}"
RESULTS_DIR="${RESULTS_DIR:-${REPO}/ckpts/stage2_imagenet1k_patch_fixv3_40gb}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-in1k_s2_dino_patch80_ddt_b64_fixv3_40gb}"
GLOBAL_SEED="${GLOBAL_SEED:-43}"
CKPT="${CKPT:-${RESULTS_DIR}/${EXPERIMENT_NAME}/checkpoints/ep-0000020.pt}"
SBATCH_QOS="${SBATCH_QOS:-SLURM_QOS_PLACEHOLDER}"

PATCH_SAMPLE_CONFIG="${PATCH_SAMPLE_CONFIG:-${REPO}/configs/stage2/sampling/ImageNet256/imagenet1k/DINO_IN1K_B256_patch40_ddt_equal_fixv3_40gb.yaml}"
RECOVERED_SAMPLE_CONFIG="${RECOVERED_SAMPLE_CONFIG:-${REPO}/configs/stage2/sampling/ImageNet256/imagenet1k/recovered_aux/DINO_IN1K_B256_patch40_ddt_mhap_to_prepend_equal_fixv3_40gb.yaml}"
PATCH_RUN_DIR="${PATCH_RUN_DIR:-${REPO}/samples_stage2_imagenet1k_fixv3_40gb/in1k_s2_dino_patch40_ddt_equal_fixv3_40gb_cfg1}"
RECOVERED_RUN_DIR="${RECOVERED_RUN_DIR:-${REPO}/samples_stage2_imagenet1k_recovered_aux_fixv3_40gb/in1k_s2_dino_patch40_ddt_mhap_to_prepend_equal_fixv3_40gb_cfg1}"
EVAL_CSV="${EVAL_CSV:-${REPO}/samples_stage2_imagenet1k_fixv3_40gb/eval_metrics_stage2_imagenet1k_ep40_fixv3_40gb.csv}"

PATCH_MODEL_NAME="${PATCH_MODEL_NAME:-DINO_IN1K_patch40_ddt_fixv3_40gb}"
RECOVERED_MODEL_NAME="${RECOVERED_MODEL_NAME:-DINO_IN1K_patch40_ddt_mhap_fixv3_40gb}"
TRAIN_JOB_NAME="${TRAIN_JOB_NAME:-in1k_s2_patch80_fixv3_40gb}"
EPOCH40_WATCH_JOB_NAME="${EPOCH40_WATCH_JOB_NAME:-in1k_s2_ep40_watch_fixv3_40gb}"
NAN_WATCH_JOB_NAME="${NAN_WATCH_JOB_NAME:-in1k_s2_nan_watch_fixv3_40gb}"
TARGET_CKPT="${TARGET_CKPT:-${RESULTS_DIR}/${EXPERIMENT_NAME}/checkpoints/ep-0000040.pt}"

NAN_POLL_SECONDS="${NAN_POLL_SECONDS:-60}"
EPOCH40_POLL_SECONDS="${EPOCH40_POLL_SECONDS:-300}"
EPOCH40_POST_DETECT_SLEEP="${EPOCH40_POST_DETECT_SLEEP:-180}"
EPOCH40_MAX_WAIT_SECONDS="${EPOCH40_MAX_WAIT_SECONDS:-0}"
RECOVERY_DEPTH="${RECOVERY_DEPTH:-0}"

mkdir -p "${REPO}/logs" "$(dirname "${EVAL_CSV}")"

if [[ ! -f "${CKPT}" ]]; then
  echo "[error] resume checkpoint not found: ${CKPT}" >&2
  exit 1
fi

train_job=$(sbatch --parsable \
  --qos="${SBATCH_QOS}" \
  --job-name="${TRAIN_JOB_NAME}" \
  --partition=gpu-7d \
  --constraint=40gb \
  --gpus-per-node=4 \
  --ntasks-per-node=1 \
  --cpus-per-task=16 \
  --mem=128G \
  --export=ALL,CONTAINER="${CONTAINER}",CONFIG="${TRAIN_CONFIG}",DATA="${TRAIN_DATA}",RESULTS_DIR="${RESULTS_DIR}",EXPERIMENT_NAME="${EXPERIMENT_NAME}",PRECISION=bf16,NPROC=4,IMAGE_SIZE=256,GLOBAL_SEED="${GLOBAL_SEED}",CKPT="${CKPT}" \
  "${TRAIN_WORKER}")

epoch40_watch_job=$(sbatch --parsable \
  --qos="${SBATCH_QOS}" \
  --job-name="${EPOCH40_WATCH_JOB_NAME}" \
  --export=ALL,CONTAINER="${CONTAINER}",VAL_DATA="${VAL_DATA}",REAL_NPZ="${REAL_NPZ}",TARGET_CKPT="${TARGET_CKPT}",PATCH_SAMPLE_CONFIG="${PATCH_SAMPLE_CONFIG}",RECOVERED_SAMPLE_CONFIG="${RECOVERED_SAMPLE_CONFIG}",PATCH_RUN_DIR="${PATCH_RUN_DIR}",RECOVERED_RUN_DIR="${RECOVERED_RUN_DIR}",EVAL_CSV="${EVAL_CSV}",PATCH_MODEL_NAME="${PATCH_MODEL_NAME}",RECOVERED_MODEL_NAME="${RECOVERED_MODEL_NAME}",TRAIN_JOB_ID="${train_job}",SBATCH_QOS="${SBATCH_QOS}",POLL_SECONDS="${EPOCH40_POLL_SECONDS}",POST_DETECT_SLEEP="${EPOCH40_POST_DETECT_SLEEP}",MAX_WAIT_SECONDS="${EPOCH40_MAX_WAIT_SECONDS}" \
  "${EPOCH40_WATCH_WORKER}")

nan_watch_job=$(sbatch --parsable \
  --qos="${SBATCH_QOS}" \
  --job-name="${NAN_WATCH_JOB_NAME}" \
  --export=ALL,CONTAINER="${CONTAINER}",TRAIN_JOB_ID="${train_job}",TRAIN_JOB_NAME="${TRAIN_JOB_NAME}",EPOCH40_WATCH_JOB_ID="${epoch40_watch_job}",RESULTS_DIR="${RESULTS_DIR}",EXPERIMENT_NAME="${EXPERIMENT_NAME}",SUBMIT_HELPER="${REPO}/submit_stage2_dino_imagenet1k_patch80_recovery_chain_fixv3_40gb.sh",TRAIN_WORKER="${TRAIN_WORKER}",EPOCH40_WATCH_WORKER="${EPOCH40_WATCH_WORKER}",NAN_WATCH_WORKER="${NAN_WATCH_WORKER}",TRAIN_DATA="${TRAIN_DATA}",VAL_DATA="${VAL_DATA}",REAL_NPZ="${REAL_NPZ}",TRAIN_CONFIG="${TRAIN_CONFIG}",GLOBAL_SEED="${GLOBAL_SEED}",SBATCH_QOS="${SBATCH_QOS}",PATCH_SAMPLE_CONFIG="${PATCH_SAMPLE_CONFIG}",RECOVERED_SAMPLE_CONFIG="${RECOVERED_SAMPLE_CONFIG}",PATCH_RUN_DIR="${PATCH_RUN_DIR}",RECOVERED_RUN_DIR="${RECOVERED_RUN_DIR}",EVAL_CSV="${EVAL_CSV}",PATCH_MODEL_NAME="${PATCH_MODEL_NAME}",RECOVERED_MODEL_NAME="${RECOVERED_MODEL_NAME}",TARGET_CKPT="${TARGET_CKPT}",POLL_SECONDS="${NAN_POLL_SECONDS}",RECOVERY_DEPTH="${RECOVERY_DEPTH}" \
  "${NAN_WATCH_WORKER}")

printf 'train=%s\nepoch40_watch=%s\nnan_watch=%s\nresume_ckpt=%s\n' \
  "${train_job}" "${epoch40_watch_job}" "${nan_watch_job}" "${CKPT}"
