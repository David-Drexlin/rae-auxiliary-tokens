#!/bin/bash
set -euo pipefail

REPO="RAE_ROOT_PLACEHOLDER"
CONTAINER="${REPO}/container.sif"
OOD_WORKER="${REPO}/stage1_eval_OOD_DINO_single.sh"
CFG_DIR="${REPO}/configs/stage1/training/ImageNet"
CKPT_ROOT="${REPO}/ckpts"
OUTDIR="${REPO}/assets/recon_samples"
OOD_CSV="${OUTDIR}/eval_metrics_ood_dino.csv"
GUIDED_EVAL="HOME_PLACEHOLDER/guided-diffusion/evaluations/evaluator.py"
FD_SCRIPT="HOME_PLACEHOLDER/FD-DINOv2/src/pytorch_fd/fd_score.py"

QOS="${QOS:-SLURM_QOS_PLACEHOLDER}"
RECOV_SEED1_DEP="${RECOV_SEED1_DEP:-4276022}"
RECOV_SEED2_DEP="${RECOV_SEED2_DEP:-4276031}"

IMAGENET_R_ROOT="DATASETS_ROOT_PLACEHOLDER/imagenet-r"
PLACES_ROOT="DATASETS_ROOT_PLACEHOLDER/places365/val"

submit_ood_eval() {
  local dataset_tag="$1"
  local data_root="$2"
  local model="$3"
  local seed="$4"
  local dep="${5:-}"
  local save_suffix="${6:-}"

  local config="${CFG_DIR}/${model}.yaml"
  local exp="DINO_IN100_${model}_seed${seed}"
  local ckpt_dir="${CKPT_ROOT}/${exp}/checkpoints"
  local model_name="${dataset_tag}_${model}_seed${seed}${save_suffix}"
  local save_folder="eval_ood_${dataset_tag}_${model}_seed${seed}${save_suffix}"

  local args=(
    sbatch --parsable
    --qos="${QOS}"
    --job-name="${model_name}"
    --export=ALL,CONTAINER="${CONTAINER}",CONFIG="${config}",CKPT_DIR="${ckpt_dir}",DATA="${data_root}",OUTDIR="${OUTDIR}",SAVE_FOLDER="${save_folder}",MODEL_NAME="${model_name}",GUIDED_EVAL="${GUIDED_EVAL}",FD_SCRIPT="${FD_SCRIPT}",IMAGE_SIZE=256,BATCH_SIZE=16,NUM_WORKERS=8,PRECISION=bf16,NUM_SAMPLES=99999999,CSV_OUT="${OOD_CSV}"
  )
  if [[ -n "${dep}" ]]; then
    args+=(--dependency="afterok:${dep}")
  fi
  args+=("${OOD_WORKER}")

  local job_id
  job_id="$("${args[@]}")"
  echo "[submit] ${model_name} eval=${job_id}"
}

recovered_dep_for_seed() {
  local seed="$1"
  local ckpt="${CKPT_ROOT}/DINO_IN100_DINO_decB_Patch+RecoveredDINORegCls_prepend_seed${seed}/checkpoints/decoder_ema_ep-0000039.pt"
  if [[ -f "${ckpt}" ]]; then
    echo ""
  elif [[ "${seed}" == "1" ]]; then
    echo "${RECOV_SEED1_DEP}"
  elif [[ "${seed}" == "2" ]]; then
    echo "${RECOV_SEED2_DEP}"
  else
    echo ""
  fi
}

echo "=== Seeded Places365 + ImageNet-R OOD Bundle ==="

# ImageNet-R seed0 refixes: previous runs only landed FID-side outputs, not the
# full FD-DINOv2 / CSV rows.
for model in \
  "DINO_decB" \
  "DINO_decB_Patch+Register+CLS_prepend" \
  "DINO_decB_Patch+RecoveredDINORegCls_prepend"; do
  submit_ood_eval "imagenet_r" "${IMAGENET_R_ROOT}" "${model}" 0 "" "_refix"
done

# Seeded Places365 and ImageNet-R: seeds 1/2 for baseline, true aux, recovered aux.
for seed in 1 2; do
  recov_dep="$(recovered_dep_for_seed "${seed}")"

  for dataset_tag in "places365" "imagenet_r"; do
    if [[ "${dataset_tag}" == "places365" ]]; then
      data_root="${PLACES_ROOT}"
    else
      data_root="${IMAGENET_R_ROOT}"
    fi

    submit_ood_eval "${dataset_tag}" "${data_root}" "DINO_decB" "${seed}"
    submit_ood_eval "${dataset_tag}" "${data_root}" "DINO_decB_Patch+Register+CLS_prepend" "${seed}"
    submit_ood_eval "${dataset_tag}" "${data_root}" "DINO_decB_Patch+RecoveredDINORegCls_prepend" "${seed}" "${recov_dep}"
  done
done

echo "=== Done ==="
