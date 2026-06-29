#!/bin/bash
#SBATCH --job-name=cls_reg_fusion_probe
#SBATCH --partition=gpu-2h
#SBATCH --constraint=40gb
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=RAE_ROOT_PLACEHOLDER/logs/%x-%j.out
#SBATCH --error=RAE_ROOT_PLACEHOLDER/logs/%x-%j.err

set -euo pipefail

cd RAE_ROOT_PLACEHOLDER
mkdir -p logs

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"

CONTAINER="${CONTAINER:-RAE_ROOT_PLACEHOLDER/container.sif}"
EMBEDDINGS="${EMBEDDINGS:-RAE_ROOT_PLACEHOLDER/assets/analysis/dino_imagenet100_token_probe_v2/embeddings.pt}"
OUTDIR="${OUTDIR:-RAE_ROOT_PLACEHOLDER/assets/analysis/dino_imagenet100_token_probe_v2/cls_register_fusion}"
VARIANTS="${VARIANTS:-all}"
SPLITS="${SPLITS:-5}"
TEST_SIZE="${TEST_SIZE:-0.3}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2048}"
HIDDEN_DIM="${HIDDEN_DIM:-1024}"
ATTN_HEADS="${ATTN_HEADS:-12}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
GATE_INIT="${GATE_INIT:-1e-3}"
GEOMETRY_LAMBDA="${GEOMETRY_LAMBDA:-0.0}"
SEED="${SEED:-0}"
CPU="${CPU:-0}"

test -f "${CONTAINER}" || { echo "[error] missing CONTAINER: ${CONTAINER}"; exit 1; }
test -f "${EMBEDDINGS}" || { echo "[error] missing EMBEDDINGS: ${EMBEDDINGS}"; exit 1; }

CPU_ARGS=()
if [[ "${CPU}" == "1" ]]; then
  CPU_ARGS+=(--cpu)
fi

mkdir -p "${OUTDIR}"

apptainer exec --nv \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=RAE_ROOT_PLACEHOLDER/src:${PYTHONPATH:-} \
  python3 RAE_ROOT_PLACEHOLDER/src/analyze_cls_register_fusion_probe.py \
    --embeddings "${EMBEDDINGS}" \
    --outdir "${OUTDIR}" \
    --variants "${VARIANTS}" \
    --splits "${SPLITS}" \
    --test-size "${TEST_SIZE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --attn-heads "${ATTN_HEADS}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --label-smoothing "${LABEL_SMOOTHING}" \
    --grad-clip "${GRAD_CLIP}" \
    --gate-init "${GATE_INIT}" \
    --geometry-lambda "${GEOMETRY_LAMBDA}" \
    --seed "${SEED}" \
    "${CPU_ARGS[@]}"
