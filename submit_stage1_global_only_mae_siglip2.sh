#!/bin/bash
set -euo pipefail

cd RAE_ROOT_PLACEHOLDER

submit_mae() {
  local seed="$1"
  local cfg="MAE_decB_GlobalCLS.yaml"
  local model="MAE_decB_GlobalCLS"
  local train_job

  train_job=$(
    sbatch --parsable \
      --array=0-0 \
      --job-name="RAE_s1_mae_globalcls_seed${seed}" \
      --gpus-per-node=1 \
      --export=ALL,CONFIG_NAME="${cfg}",GLOBAL_SEED="${seed}",EXP_PREFIX=MAE_IN100_,NPROC=1 \
      RAE_ROOT_PLACEHOLDER/run_stage1_mae_imagenet100_array.sh
  )
  bash RAE_ROOT_PLACEHOLDER/submit_stage1_mae_seeded_eval_one.sh "${model}" "${seed}" "${train_job}"
  printf 'mae seed=%s train=%s\n' "${seed}" "${train_job}"
}

submit_siglip2() {
  local seed="$1"
  local cfg="SigLIP2_decB_GlobalPooler.yaml"
  local model="SigLIP2_decB_GlobalPooler"
  local train_job

  train_job=$(
    sbatch --parsable \
      --array=0-0 \
      --job-name="RAE_s1_sig_globalpool_seed${seed}" \
      --gpus-per-node=1 \
      --export=ALL,CONFIG_NAME="${cfg}",GLOBAL_SEED="${seed}",EXP_PREFIX=SIGLIP2_IN100_,NPROC=1 \
      RAE_ROOT_PLACEHOLDER/run_stage1_siglip2_imagenet100_array.sh
  )
  bash RAE_ROOT_PLACEHOLDER/submit_stage1_siglip2_seeded_eval_one.sh "${model}" "${seed}" "${train_job}"
  printf 'siglip2 seed=%s train=%s\n' "${seed}" "${train_job}"
}

for seed in 0 1 2; do
  submit_mae "${seed}"
done

for seed in 0 1 2; do
  submit_siglip2 "${seed}"
done
