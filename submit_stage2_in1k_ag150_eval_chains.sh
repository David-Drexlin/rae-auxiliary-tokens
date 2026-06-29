#!/bin/bash
set -euo pipefail

REPO="RAE_ROOT_PLACEHOLDER"
cd "${REPO}"

CONTAINER="${CONTAINER:-${REPO}/container.sif}"
SAMPLE_WORKER="${SAMPLE_WORKER:-${REPO}/sample_stage2_imagenet100_single.sh}"
RECOVERED_SAMPLE_WORKER="${RECOVERED_SAMPLE_WORKER:-${REPO}/sample_recovered_aux_imagenet100_single.sh}"
FID_WORKER="${FID_WORKER:-${REPO}/stage2_eval_ImageNet1k_fid_only_single.sh}"
FD_WORKER="${FD_WORKER:-${REPO}/stage2_eval_ImageNet1k_fd_only_single.sh}"
SBATCH_QOS="${SBATCH_QOS:-SLURM_QOS_PLACEHOLDER}"

submit_chain() {
  local tag="$1"
  local sample_worker="$2"
  local config="$3"
  local run_dir="$4"
  local model_name="$5"
  local fid_csv="$6"
  local fd_csv="$7"
  local sample_job_name="$8"
  local fid_job_name="$9"
  local fd_job_name="${10}"

  local sample_job
  local fid_job
  local fd_job

  sample_job=$(sbatch --parsable \
    --qos="${SBATCH_QOS}" \
    --job-name="${sample_job_name}" \
    --export=ALL,CONTAINER="${CONTAINER}",CONFIG="${config}",NPROC=2 \
    "${sample_worker}")

  fid_job=$(sbatch --parsable \
    --qos="${SBATCH_QOS}" \
    --job-name="${fid_job_name}" \
    --dependency=afterok:${sample_job} \
    --export=ALL,CONTAINER="${CONTAINER}",RUN_DIR="${run_dir}",MODEL_NAME="${model_name}",CSV_OUT="${fid_csv}" \
    "${FID_WORKER}")

  fd_job=$(sbatch --parsable \
    --qos="${SBATCH_QOS}" \
    --job-name="${fd_job_name}" \
    --dependency=afterok:${sample_job} \
    --export=ALL,CONTAINER="${CONTAINER}",RUN_DIR="${run_dir}",MODEL_NAME="${model_name}",CSV_OUT="${fd_csv}" \
    "${FD_WORKER}")

  printf '%s sample=%s fid=%s fd=%s\n' "${tag}" "${sample_job}" "${fid_job}" "${fd_job}"
}

submit_chain \
  "patch40_ag150" \
  "${SAMPLE_WORKER}" \
  "${REPO}/configs/stage2/sampling/ImageNet256/imagenet1k/DINO_IN1K_patch40final_ag150_equal_orig80_accum4_noeval_50k.yaml" \
  "${REPO}/samples_stage2_imagenet1k_fixv3_40gb/in1k_s2_dino_patch40final_ag150_equal_orig80_accum4_noeval_50k_cfg1" \
  "DINO_IN1K_patch40final_ag150" \
  "${REPO}/samples_stage2_imagenet1k_fixv3_40gb/eval_metrics_stage2_imagenet1k_patch40final_ag150_50k_fid_only.csv" \
  "${REPO}/samples_stage2_imagenet1k_fixv3_40gb/eval_metrics_stage2_imagenet1k_patch40final_ag150_50k_fdd_only.csv" \
  "in1k_s2_samp_patch40final_ag150_50k" \
  "in1k_s2_fid_patch40final_ag150_50k" \
  "in1k_s2_fdd_patch40final_ag150_50k"

submit_chain \
  "mhap40_ag150" \
  "${RECOVERED_SAMPLE_WORKER}" \
  "${REPO}/configs/stage2/sampling/ImageNet256/imagenet1k/recovered_aux/DINO_IN1K_patch40final_ag150_mhap_to_prepend_equal_orig80_accum4_noeval_50k.yaml" \
  "${REPO}/samples_stage2_imagenet1k_recovered_aux_fixv3_40gb/in1k_s2_dino_patch40final_ag150_mhap_to_prepend_equal_orig80_accum4_noeval_50k_cfg1" \
  "DINO_IN1K_patch40final_mhap_ag150" \
  "${REPO}/samples_stage2_imagenet1k_fixv3_40gb/eval_metrics_stage2_imagenet1k_patch40final_mhap_ag150_50k_fid_only.csv" \
  "${REPO}/samples_stage2_imagenet1k_fixv3_40gb/eval_metrics_stage2_imagenet1k_patch40final_mhap_ag150_50k_fdd_only.csv" \
  "in1k_s2_samp_mhap40final_ag150_50k" \
  "in1k_s2_fid_mhap40final_ag150_50k" \
  "in1k_s2_fdd_mhap40final_ag150_50k"

submit_chain \
  "patch80_ag150" \
  "${SAMPLE_WORKER}" \
  "${REPO}/configs/stage2/sampling/ImageNet256/imagenet1k/DINO_IN1K_patch80final_ag150_equal_orig1400sched_accum4_noeval_50k.yaml" \
  "${REPO}/samples_stage2_imagenet1k_fixv3_40gb/in1k_s2_dino_patch80final_ag150_equal_orig1400sched_accum4_noeval_50k_cfg1" \
  "DINO_IN1K_patch80final_ag150" \
  "${REPO}/samples_stage2_imagenet1k_fixv3_40gb/eval_metrics_stage2_imagenet1k_patch80final_ag150_50k_fid_only.csv" \
  "${REPO}/samples_stage2_imagenet1k_fixv3_40gb/eval_metrics_stage2_imagenet1k_patch80final_ag150_50k_fdd_only.csv" \
  "in1k_s2_samp_patch80final_ag150_50k" \
  "in1k_s2_fid_patch80final_ag150_50k" \
  "in1k_s2_fdd_patch80final_ag150_50k"

submit_chain \
  "mhap80_ag150" \
  "${RECOVERED_SAMPLE_WORKER}" \
  "${REPO}/configs/stage2/sampling/ImageNet256/imagenet1k/recovered_aux/DINO_IN1K_patch80final_ag150_mhap_to_prepend_equal_orig1400sched_accum4_noeval_50k.yaml" \
  "${REPO}/samples_stage2_imagenet1k_recovered_aux_fixv3_40gb/in1k_s2_dino_patch80final_ag150_mhap_to_prepend_equal_orig1400sched_accum4_noeval_50k_cfg1" \
  "DINO_IN1K_patch80final_mhap_ag150" \
  "${REPO}/samples_stage2_imagenet1k_fixv3_40gb/eval_metrics_stage2_imagenet1k_patch80final_mhap_ag150_50k_fid_only.csv" \
  "${REPO}/samples_stage2_imagenet1k_fixv3_40gb/eval_metrics_stage2_imagenet1k_patch80final_mhap_ag150_50k_fdd_only.csv" \
  "in1k_s2_samp_mhap80final_ag150_50k" \
  "in1k_s2_fid_mhap80final_ag150_50k" \
  "in1k_s2_fdd_mhap80final_ag150_50k"
