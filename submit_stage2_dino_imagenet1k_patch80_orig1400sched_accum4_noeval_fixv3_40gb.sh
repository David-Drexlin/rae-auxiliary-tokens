#!/usr/bin/env bash
set -euo pipefail

cd RAE_ROOT_PLACEHOLDER

sbatch --parsable \
  --qos=SLURM_QOS_PLACEHOLDER \
  --job-name=in1k_s2_patch80_orig1400sched_accum4_noeval_40gb \
  --partition=gpu-7d \
  --constraint=40gb \
  --gpus-per-node=4 \
  --ntasks-per-node=1 \
  --cpus-per-task=16 \
  --mem=128G \
  --export=ALL,CONTAINER=RAE_ROOT_PLACEHOLDER/container.sif,CONFIG=RAE_ROOT_PLACEHOLDER/configs/stage2/training/ImageNet256/imagenet1k/DiTDH-S_DINO_IN1K_patch_80ep_orig1400sched_accum4_micro64_noeval.yaml,DATA=DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/data/train,RESULTS_DIR=RAE_ROOT_PLACEHOLDER/ckpts/stage2_imagenet1k_patch_fixv3_40gb,EXPERIMENT_NAME=in1k_s2_dino_patch80_ddt_orig1400sched_accum4_noeval_fixv3_40gb,PRECISION=bf16,NPROC=4,IMAGE_SIZE=256,GLOBAL_SEED=42,CKPT=RAE_ROOT_PLACEHOLDER/ckpts/stage2_imagenet1k_patch_fixv3_40gb/in1k_s2_dino_patch80_ddt_b64_fixv3_40gb/checkpoints/ep-0000000.pt \
  RAE_ROOT_PLACEHOLDER/train_ImageNet_DiT.sh
