#!/bin/bash
set -euo pipefail

REPO="RAE_ROOT_PLACEHOLDER"
CONTAINER="${REPO}/container.sif"
TRAIN_SCRIPT="${REPO}/run_stage1_dino_imagenet100_array.sh"
IN_DOMAIN_WORKER="${REPO}/stage1_eval_ImageNet_DINO_seeded_single.sh"
OOD_WORKER="${REPO}/stage1_eval_OOD_DINO_single.sh"

CFG_DIR="${REPO}/configs/stage1/training/ImageNet"
CKPT_ROOT="${REPO}/ckpts"
OUTDIR="${REPO}/assets/recon_samples"
LOG_DIR="${REPO}/logs"
mkdir -p "${LOG_DIR}" "${OUTDIR}"

IN100_VAL="DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/val"
REAL_PNG_DIR="${REPO}/assets/datasets/imagenet100_val_flat"
REAL_NPZ_256="${REPO}/assets/datasets/imagenet100_val_256.npz"
GUIDED_EVAL="HOME_PLACEHOLDER/guided-diffusion/evaluations/evaluator.py"
FD_ROOT="HOME_PLACEHOLDER/FD-DINOv2"
FD_SCRIPT="${FD_ROOT}/src/pytorch_fd/fd_score.py"

NINCO_ROOT="DATASETS_ROOT_PLACEHOLDER/ninco/NINCO_OOD_classes"
OOD_CSV="${OUTDIR}/eval_metrics_ood_dino.csv"
IN_DOMAIN_CSV="${OUTDIR}/eval_metrics_seeded_dino.csv"

QOS="SLURM_QOS_PLACEHOLDER"
OOD_CONSTRAINT="40gb|80gb|h100"

submit_in_domain_eval() {
  local model="$1"
  local seed="$2"
  local dep="${3:-}"
  local config="${CFG_DIR}/${model}.yaml"
  local exp="DINO_IN100_${model}_seed${seed}"
  local ckpt_dir="${CKPT_ROOT}/${exp}/checkpoints"
  local save_folder="eval_${exp}"
  local fid_log="${OUTDIR}/${save_folder}/fid.log"
  local fd_log="${OUTDIR}/${save_folder}/fd_dinov2.log"

  if [[ -f "${fid_log}" && -f "${fd_log}" ]] && grep -Fq "\"${exp}\"" "${IN_DOMAIN_CSV}" 2>/dev/null; then
    echo "[skip in-domain] ${exp} already has fid+fd logs"
    return
  fi

  local args=(
    sbatch --parsable
    --qos="${QOS}"
    --job-name="${save_folder}"
    --constraint="${OOD_CONSTRAINT}"
  )
  if [[ -n "${dep}" ]]; then
    args+=(--dependency="afterok:${dep}")
  fi
  args+=(
    --export=ALL,CONTAINER="${CONTAINER}",CONFIG="${config}",CKPT_DIR="${ckpt_dir}",DATA="${IN100_VAL}",OUTDIR="${OUTDIR}",SAVE_FOLDER="${save_folder}",REAL_PNG_DIR="${REAL_PNG_DIR}",REAL_NPZ="${REAL_NPZ_256}",GUIDED_EVAL="${GUIDED_EVAL}",FD_ROOT="${FD_ROOT}",FD_SCRIPT="${FD_SCRIPT}",IMAGE_SIZE=256,BATCH_SIZE=16,NUM_WORKERS=8,PRECISION=bf16,TRAIN_EXP_NAME="${exp}",METRICS_CSV="${IN_DOMAIN_CSV}"
    "${IN_DOMAIN_WORKER}"
  )

  local job_id
  job_id="$("${args[@]}")"
  echo "[submit in-domain] ${exp} eval=${job_id}"
}

submit_ninco_ood_eval() {
  local model="$1"
  local seed="$2"
  local dep="${3:-}"
  local config="${CFG_DIR}/${model}.yaml"
  local exp="DINO_IN100_${model}_seed${seed}"
  local ckpt_dir="${CKPT_ROOT}/${exp}/checkpoints"
  local model_name="ninco_${model}_seed${seed}"
  local save_folder="eval_ood_ninco_${model}_seed${seed}"
  local run_dir="${OUTDIR}/${save_folder}"
  local sample_npz="${run_dir}/samples.npz"
  local real_npz="${run_dir}/real_samples.npz"

  if [[ -f "${sample_npz}" && -f "${real_npz}" ]] && grep -Fq "${model_name}" "${OOD_CSV}" 2>/dev/null; then
    echo "[skip ninco] ${model_name} already has packed npzs"
    return
  fi

  local args=(
    sbatch --parsable
    --qos="${QOS}"
    --job-name="${model_name}"
    --constraint="${OOD_CONSTRAINT}"
    --export=ALL,CONTAINER="${CONTAINER}",CONFIG="${config}",CKPT_DIR="${ckpt_dir}",DATA="${NINCO_ROOT}",OUTDIR="${OUTDIR}",SAVE_FOLDER="${save_folder}",MODEL_NAME="${model_name}",GUIDED_EVAL="${GUIDED_EVAL}",FD_SCRIPT="${FD_SCRIPT}",IMAGE_SIZE=256,BATCH_SIZE=16,NUM_WORKERS=8,PRECISION=bf16,NUM_SAMPLES=99999999,CSV_OUT="${OOD_CSV}"
  )
  if [[ -n "${dep}" ]]; then
    args+=(--dependency="afterok:${dep}")
  fi
  args+=("${OOD_WORKER}")

  local job_id
  job_id="$("${args[@]}")"
  echo "[submit ninco] ${model_name} eval=${job_id}"
}

submit_recovered_train_seed() {
  local seed="$1"
  local model="DINO_decB_Patch+RecoveredDINORegCls_prepend"
  local exp="DINO_IN100_${model}_seed${seed}"
  local ckpt_dir="${CKPT_ROOT}/${exp}/checkpoints"

  if [[ -f "${ckpt_dir}/decoder_ema_ep-0000039.pt" ]]; then
    echo "[skip train] ${exp} checkpoint already exists"
    return 1
  fi

  local train_job
  train_job="$(sbatch --parsable \
    --qos="${QOS}" \
    --job-name="RAE_s1_dino_in100_${model}_seed${seed}" \
    --export=ALL,CONFIG_NAME="${model}.yaml",GLOBAL_SEED="${seed}",EXP_PREFIX="DINO_IN100_" \
    "${TRAIN_SCRIPT}")"

  echo "[submit train] ${exp} train=${train_job}" >&2
  echo "${train_job}"
  return 0
}

echo "=== Recovered-Aux In-Domain + NINCO Seed Bundle ==="

# Seed0 recovered-aux in-domain eval was previously miswired; relaunch it cleanly.
submit_in_domain_eval "DINO_decB_Patch+RecoveredDINORegCls_prepend" 0

# Baseline / true-aux seeded NINCO OOD: fill missing seeds 1 and 2 only.
for seed in 1 2; do
  submit_ninco_ood_eval "DINO_decB" "${seed}"
  submit_ninco_ood_eval "DINO_decB_Patch+Register+CLS_prepend" "${seed}"
done

# Recovered-aux seeds 1 and 2: train, then in-domain eval and NINCO OOD eval.
for seed in 1 2; do
  dep=""
  if train_job="$(submit_recovered_train_seed "${seed}")"; then
    dep="${train_job##*=}"
  else
    dep=""
  fi
  submit_in_domain_eval "DINO_decB_Patch+RecoveredDINORegCls_prepend" "${seed}" "${dep}"
  submit_ninco_ood_eval "DINO_decB_Patch+RecoveredDINORegCls_prepend" "${seed}" "${dep}"
done

echo "=== Done ==="
