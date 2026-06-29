#!/bin/bash
set -euo pipefail

cd RAE_ROOT_PLACEHOLDER
mkdir -p logs

CONTAINER="${CONTAINER:-RAE_ROOT_PLACEHOLDER/container.sif}"
CFG_DIR="${CFG_DIR:-RAE_ROOT_PLACEHOLDER/configs/stage1/training/ImageNet}"
CKPT_ROOT="${CKPT_ROOT:-RAE_ROOT_PLACEHOLDER/ckpts}"
IMAGENET_VAL="${IMAGENET_VAL:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/val}"
OUTDIR="${OUTDIR:-RAE_ROOT_PLACEHOLDER/assets/recon_samples}"
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
METRICS_CSV="${METRICS_CSV:-RAE_ROOT_PLACEHOLDER/assets/recon_samples/eval_metrics_seeded_dino.csv}"
WORKER="${WORKER:-RAE_ROOT_PLACEHOLDER/stage1_eval_ImageNet_DINO_seeded_single.sh}"

MODELS=(
  "DINO_decB"
  "DINO_decB_Patch+CLS_AdaLN"
  "DINO_decB_Patch+CLS_CA"
  "DINO_decB_Patch+CLS_prepend"
  "DINO_decB_Patch+Register_AdaLN"
  "DINO_decB_Patch+Register_CA"
  "DINO_decB_Patch+Register_prepend"
  "DINO_decB_Patch+Register+CLS_AdaLN"
  "DINO_decB_Patch+Register+CLS_CA"
  "DINO_decB_Patch+Register+CLS_prepend"
)

TRAIN_JOB_IDS=(
  "4236678 4236679 4236680"
  "4236682 4236681 4236683"
  "4236684 4236685 4236687"
  "4236686 4236688 4236689"
  "4236690 4236691 4236692"
  "4236693 4236694 4236695"
  "4236697 4236696 4236698"
  "4236699 4236700 4236702"
  "4236701 4236703 4236704"
  "4236705 4236707 4236706"
)

for idx in "${!MODELS[@]}"; do
  model="${MODELS[$idx]}"
  read -r -a seed_jobs <<< "${TRAIN_JOB_IDS[$idx]}"
  for seed in 0 1 2; do
    yaml="${CFG_DIR}/${model}.yaml"
    train_exp_name="DINO_IN100_${model}_seed${seed}"
    ckpt_dir="${CKPT_ROOT}/${train_exp_name}/checkpoints"
    save_folder="eval_${train_exp_name}"
    recon_dir="${OUTDIR}/${save_folder}"
    fid_log="${recon_dir}/fid.log"
    fd_log="${recon_dir}/fd_dinov2.log"
    train_job_id="${seed_jobs[$seed]}"
    final_ema="${ckpt_dir}/decoder_ema_ep-0000039.pt"

    if [[ -f "${fid_log}" && -f "${fd_log}" ]]; then
      echo "[skip] existing eval logs for ${train_exp_name}"
      continue
    fi

    submit_args=(
      sbatch
      --job-name="${save_folder}"
      --export=ALL,CONTAINER="${CONTAINER}",CONFIG="${yaml}",CKPT_DIR="${ckpt_dir}",DATA="${IMAGENET_VAL}",OUTDIR="${OUTDIR}",SAVE_FOLDER="${save_folder}",REAL_PNG_DIR="${REAL_PNG_DIR}",REAL_NPZ="${REAL_NPZ}",GUIDED_EVAL="${GUIDED_EVAL}",FD_ROOT="${FD_ROOT}",FD_SCRIPT="${FD_SCRIPT}",IMAGE_SIZE="${IMAGE_SIZE}",BATCH_SIZE="${BATCH_SIZE}",NUM_WORKERS="${NUM_WORKERS}",PRECISION="${PRECISION}",NUM_SAMPLES="${NUM_SAMPLES}",TRAIN_EXP_NAME="${train_exp_name}",METRICS_CSV="${METRICS_CSV}"
    )

    if [[ -f "${final_ema}" ]]; then
      echo "[submit-now] ${train_exp_name}"
    else
      echo "[submit-afterok:${train_job_id}] ${train_exp_name}"
      submit_args+=(--dependency="afterok:${train_job_id}")
    fi

    submit_args+=("${WORKER}")
    "${submit_args[@]}"
  done
done
