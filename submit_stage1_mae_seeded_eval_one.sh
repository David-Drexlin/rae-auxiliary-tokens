#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 MODEL_NAME SEED TRAIN_JOB_ID" >&2
  exit 1
fi

MODEL="$1"
SEED="$2"
TRAIN_JOB_ID="$3"

CONTAINER="${CONTAINER:-RAE_ROOT_PLACEHOLDER/container.sif}"
YAML="RAE_ROOT_PLACEHOLDER/configs/stage1/training/ImageNet/${MODEL}.yaml"
SEEDED_TRAIN_EXP_NAME="MAE_IN100_${MODEL}_seed${SEED}"
CKPT_DIR="RAE_ROOT_PLACEHOLDER/ckpts/${SEEDED_TRAIN_EXP_NAME}/checkpoints"
TRAIN_EXP_NAME="${SEEDED_TRAIN_EXP_NAME}"
SAVE_FOLDER="eval_${SEEDED_TRAIN_EXP_NAME}"
FINAL_EMA="${CKPT_DIR}/decoder_ema_ep-0000039.pt"

SUBMIT_ARGS=(
  sbatch
  --partition="${EVAL_PARTITION:-gpu-2h}"
  --time="${EVAL_TIME:-02:00:00}"
  --job-name="${SAVE_FOLDER}"
  --export=ALL,CONTAINER="${CONTAINER}",CONFIG="${YAML}",CKPT_DIR="${CKPT_DIR}",DATA=DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/val,OUTDIR=RAE_ROOT_PLACEHOLDER/assets/recon_samples,SAVE_FOLDER="${SAVE_FOLDER}",REAL_PNG_DIR=RAE_ROOT_PLACEHOLDER/assets/datasets/imagenet100_val_flat,REAL_NPZ=RAE_ROOT_PLACEHOLDER/assets/datasets/imagenet100_val_224.npz,GUIDED_EVAL=HOME_PLACEHOLDER/guided-diffusion/evaluations/evaluator.py,FD_ROOT=HOME_PLACEHOLDER/FD-DINOv2,FD_SCRIPT=HOME_PLACEHOLDER/FD-DINOv2/src/pytorch_fd/fd_score.py,IMAGE_SIZE=224,BATCH_SIZE=16,NUM_WORKERS=8,PRECISION=bf16,NUM_SAMPLES=,TRAIN_EXP_NAME="${TRAIN_EXP_NAME}",METRICS_CSV=RAE_ROOT_PLACEHOLDER/assets/recon_samples/eval_metrics_seeded_mae.csv
)

if [[ -f "${FINAL_EMA}" ]]; then
  echo "[submit-now] ${TRAIN_EXP_NAME}"
else
  echo "[submit-afterok:${TRAIN_JOB_ID}] ${TRAIN_EXP_NAME}"
  SUBMIT_ARGS+=(--dependency="afterok:${TRAIN_JOB_ID}")
fi

SUBMIT_ARGS+=(RAE_ROOT_PLACEHOLDER/stage1_eval_ImageNet_DINO_seeded_single.sh)
"${SUBMIT_ARGS[@]}"
