#!/bin/bash
set -euo pipefail

REPO="RAE_ROOT_PLACEHOLDER"
VAL_DATA="DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/val"
OUTDIR="${REPO}/assets/recon_samples"
REAL_PNG_DIR="${REPO}/assets/datasets/imagenet100_val_flat"
REAL_NPZ_224="${REPO}/assets/datasets/imagenet100_val_224.npz"
REAL_NPZ_256="${REPO}/assets/datasets/imagenet100_val_256.npz"
GUIDED_EVAL="HOME_PLACEHOLDER/guided-diffusion/evaluations/evaluator.py"
FD_ROOT="HOME_PLACEHOLDER/FD-DINOv2"
FD_SCRIPT="${FD_ROOT}/src/pytorch_fd/fd_score.py"

mae_cfg="MAE_decB_Patch+RecoveredDINORegCls_prepend.yaml"
mae_model="${mae_cfg%.yaml}"
mae_seed=0
mae_exp="MAE_IN100_${mae_model}_seed${mae_seed}"
mae_train=$(sbatch --parsable \
  --array=0-0 \
  --job-name="RAE_s1_mae_in100_${mae_model}_seed${mae_seed}" \
  --export=ALL,CONFIG_NAME="${mae_cfg}",GLOBAL_SEED="${mae_seed}" \
  "${REPO}/run_stage1_mae_imagenet100_array.sh")
mae_eval=$(sbatch --parsable \
  --job-name="eval_${mae_exp}" \
  --dependency="afterok:${mae_train}" \
  --export=ALL,CONFIG="${REPO}/configs/stage1/training/ImageNet/${mae_cfg}",CKPT_DIR="${REPO}/ckpts/${mae_exp}/checkpoints",DATA="${VAL_DATA}",OUTDIR="${OUTDIR}",SAVE_FOLDER="eval_${mae_exp}",REAL_PNG_DIR="${REAL_PNG_DIR}",REAL_NPZ="${REAL_NPZ_224}",GUIDED_EVAL="${GUIDED_EVAL}",FD_ROOT="${FD_ROOT}",FD_SCRIPT="${FD_SCRIPT}",IMAGE_SIZE=224,BATCH_SIZE=16,NUM_WORKERS=8,PRECISION=bf16,TRAIN_EXP_NAME="${mae_exp}",METRICS_CSV="${OUTDIR}/eval_metrics_seeded_mae.csv" \
  "${REPO}/stage1_eval_ImageNet_DINO_seeded_single.sh")

sig_cfg="SigLIP2_decB_Patch+RecoveredDINORegCls_prepend.yaml"
sig_model="${sig_cfg%.yaml}"
sig_seed=0
sig_exp="SIGLIP2_IN100_${sig_model}_seed${sig_seed}"
sig_train=$(sbatch --parsable \
  --array=0-0 \
  --job-name="RAE_s1_siglip2_in100_${sig_model}_seed${sig_seed}" \
  --export=ALL,CONFIG_NAME="${sig_cfg}",GLOBAL_SEED="${sig_seed}" \
  "${REPO}/run_stage1_siglip2_imagenet100_array.sh")
sig_eval=$(sbatch --parsable \
  --job-name="eval_${sig_exp}" \
  --dependency="afterok:${sig_train}" \
  --export=ALL,CONFIG="${REPO}/configs/stage1/training/ImageNet/${sig_cfg}",CKPT_DIR="${REPO}/ckpts/${sig_exp}/checkpoints",DATA="${VAL_DATA}",OUTDIR="${OUTDIR}",SAVE_FOLDER="eval_${sig_exp}",REAL_PNG_DIR="${REAL_PNG_DIR}",REAL_NPZ="${REAL_NPZ_256}",GUIDED_EVAL="${GUIDED_EVAL}",FD_ROOT="${FD_ROOT}",FD_SCRIPT="${FD_SCRIPT}",IMAGE_SIZE=256,BATCH_SIZE=16,NUM_WORKERS=8,PRECISION=bf16,TRAIN_EXP_NAME="${sig_exp}",METRICS_CSV="${OUTDIR}/eval_metrics_seeded_siglip2.csv" \
  "${REPO}/stage1_eval_ImageNet_DINO_seeded_single.sh")

printf 'mae_train=%s\nmae_eval=%s\nsiglip2_train=%s\nsiglip2_eval=%s\n' \
  "${mae_train}" "${mae_eval}" "${sig_train}" "${sig_eval}"
