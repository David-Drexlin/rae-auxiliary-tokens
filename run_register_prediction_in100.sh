#!/bin/bash
#SBATCH --partition=gpu-2d
#SBATCH --qos=SLURM_QOS_PLACEHOLDER
#SBATCH --constraint=40gb
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=RAE_ROOT_PLACEHOLDER/logs/%x-%j.out
#SBATCH --error=RAE_ROOT_PLACEHOLDER/logs/%x-%j.err

set -euo pipefail

cd RAE_ROOT_PLACEHOLDER
mkdir -p logs

CONTAINER="${CONTAINER:-RAE_ROOT_PLACEHOLDER/container.sif}"
CONFIG="${CONFIG:-RAE_ROOT_PLACEHOLDER/configs/register_prediction/siglip2_to_dino_regcls_in100.yaml}"
PHASE="${PHASE:-all}"
MODEL="${MODEL:-all}"
SPLIT="${SPLIT:-val}"
OVERWRITE="${OVERWRITE:-0}"
CPU="${CPU:-0}"
NO_RETRIEVAL="${NO_RETRIEVAL:-0}"
SAVE_ATTN_GRIDS="${SAVE_ATTN_GRIDS:-}"

test -f "${CONTAINER}" || { echo "[error] missing CONTAINER: ${CONTAINER}"; exit 1; }
test -f "${CONFIG}" || { echo "[error] missing CONFIG: ${CONFIG}"; exit 1; }

PY_ARGS=()
if [[ "${CPU}" == "1" ]]; then
  PY_ARGS+=(--cpu)
fi

TRAIN_ARGS=()
if [[ "${MODEL}" != "" ]]; then
  TRAIN_ARGS+=(--model "${MODEL}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  TRAIN_ARGS+=(--overwrite)
fi
if [[ "${CPU}" == "1" ]]; then
  TRAIN_ARGS+=(--cpu)
fi

EVAL_ARGS=(--split "${SPLIT}")
if [[ "${MODEL}" != "" && "${MODEL}" != "all" ]]; then
  EVAL_ARGS+=(--model "${MODEL}")
fi
if [[ "${NO_RETRIEVAL}" == "1" ]]; then
  EVAL_ARGS+=(--no-retrieval)
fi
if [[ "${SAVE_ATTN_GRIDS}" != "" ]]; then
  EVAL_ARGS+=(--save-attn-grids "${SAVE_ATTN_GRIDS}")
fi
if [[ "${CPU}" == "1" ]]; then
  EVAL_ARGS+=(--cpu)
fi

run_py() {
  apptainer exec --nv \
    -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
    -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
    "${CONTAINER}" \
    env PYTHONPATH=RAE_ROOT_PLACEHOLDER/src:${PYTHONPATH:-} \
    "$@"
}

case "${PHASE}" in
  cache_train)
    run_py python3 -m register_prediction.cache_features --config "${CONFIG}" --split train "${PY_ARGS[@]}"
    ;;
  cache_val)
    run_py python3 -m register_prediction.cache_features --config "${CONFIG}" --split val "${PY_ARGS[@]}"
    ;;
  train)
    run_py python3 -m register_prediction.train --config "${CONFIG}" "${TRAIN_ARGS[@]}"
    ;;
  eval)
    run_py python3 -m register_prediction.eval --config "${CONFIG}" "${EVAL_ARGS[@]}"
    ;;
  all)
    run_py python3 -m register_prediction.cache_features --config "${CONFIG}" --split train "${PY_ARGS[@]}"
    run_py python3 -m register_prediction.cache_features --config "${CONFIG}" --split val "${PY_ARGS[@]}"
    run_py python3 -m register_prediction.train --config "${CONFIG}" "${TRAIN_ARGS[@]}"
    run_py python3 -m register_prediction.eval --config "${CONFIG}" "${EVAL_ARGS[@]}"
    ;;
  *)
    echo "[error] unknown PHASE=${PHASE}; expected cache_train|cache_val|train|eval|all" >&2
    exit 2
    ;;
esac
