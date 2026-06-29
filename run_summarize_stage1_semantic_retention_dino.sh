#!/bin/bash
#SBATCH --partition=cpu-2d
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=RAE_ROOT_PLACEHOLDER/logs/%x-%j.out
#SBATCH --error=RAE_ROOT_PLACEHOLDER/logs/%x-%j.err

set -euo pipefail

cd RAE_ROOT_PLACEHOLDER
mkdir -p logs

CONTAINER="${CONTAINER:-RAE_ROOT_PLACEHOLDER/container.sif}"
INPUT_ROOT="${INPUT_ROOT:-RAE_ROOT_PLACEHOLDER/assets/analysis/stage1_semantic_retention_dino/runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-RAE_ROOT_PLACEHOLDER/assets/analysis/stage1_semantic_retention_dino/summary}"
METRICS_CSV="${METRICS_CSV:-RAE_ROOT_PLACEHOLDER/assets/recon_samples/eval_metrics_seeded_dino.csv}"
INCLUDE_STRESS="${INCLUDE_STRESS:-0}"
INCLUDE_NONCANONICAL="${INCLUDE_NONCANONICAL:-0}"

ARGS=(
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --metrics-csv "${METRICS_CSV}"
)

if [[ "${INCLUDE_STRESS}" == "1" ]]; then
  ARGS+=(--include-stress)
fi
if [[ "${INCLUDE_NONCANONICAL}" == "1" ]]; then
  ARGS+=(--include-noncanonical)
fi

apptainer exec \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  python3 RAE_ROOT_PLACEHOLDER/src/summarize_stage1_semantic_retention_dino.py "${ARGS[@]}"
