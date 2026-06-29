#!/bin/bash
#SBATCH --job-name=stage2_in1k_fd_resume
#SBATCH --partition=gpu-5h
#SBATCH --qos=SLURM_QOS_PLACEHOLDER
#SBATCH --constraint=40gb
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="RAE_ROOT_PLACEHOLDER"
cd "${REPO}"
mkdir -p logs

CONTAINER="${CONTAINER:-${REPO}/container.sif}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
FD_REAL_IMAGEFOLDER="${FD_REAL_IMAGEFOLDER:-DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/data/val}"
MAX_SAMPLES="${MAX_SAMPLES:-50000}"
USE_SYMLINKS="${USE_SYMLINKS:-1}"

REAL_CACHE_DIR="${RUN_DIR}/_fd_real_png_cache"
export REAL_CACHE_DIR FD_REAL_IMAGEFOLDER MAX_SAMPLES USE_SYMLINKS

echo "=== RESUME FD CACHE JOB INFO ==="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "RUN_DIR: ${RUN_DIR}"
echo "REAL_CACHE_DIR: ${REAL_CACHE_DIR}"
echo "FD_REAL_IMAGEFOLDER: ${FD_REAL_IMAGEFOLDER}"
echo "MAX_SAMPLES: ${MAX_SAMPLES}"
echo "================"

test -d "${RUN_DIR}" || { echo "[error] missing RUN_DIR: ${RUN_DIR}"; exit 1; }
test -d "${FD_REAL_IMAGEFOLDER}" || { echo "[error] missing FD_REAL_IMAGEFOLDER: ${FD_REAL_IMAGEFOLDER}"; exit 1; }

mkdir -p "${REAL_CACHE_DIR}"

echo "[resume] continuing real-image cache population without clearing existing entries"
apptainer exec \
  -B HOME_PLACEHOLDER:HOME_PLACEHOLDER \
  -B SPACE_ROOT_PLACEHOLDER:SPACE_ROOT_PLACEHOLDER \
  "${CONTAINER}" \
  python3 - <<'PY'
import json
import os
from pathlib import Path
from torchvision.datasets import ImageFolder

real_imagefolder = Path(os.environ["FD_REAL_IMAGEFOLDER"])
out_dir = Path(os.environ["REAL_CACHE_DIR"])
max_samples = int(os.environ["MAX_SAMPLES"])
use_symlinks = os.environ.get("USE_SYMLINKS", "1") != "0"

out_dir.mkdir(parents=True, exist_ok=True)
ds = ImageFolder(str(real_imagefolder))
all_paths = [Path(p) for p, _ in ds.samples]
chosen_paths = all_paths[:max_samples]

for i, src_path in enumerate(chosen_paths):
    dst_path = out_dir / f"{i:06d}{src_path.suffix.lower()}"
    if dst_path.exists() or dst_path.is_symlink():
        continue
    if use_symlinks:
        try:
            os.symlink(src_path, dst_path)
            continue
        except OSError:
            pass
    import shutil
    shutil.copy2(src_path, dst_path)

cache_info = {
    "real_imagefolder": str(real_imagefolder.resolve()),
    "num_total_imagefolder_samples": len(all_paths),
    "num_selected": len(chosen_paths),
    "selected_indices_used": False,
    "max_samples": int(max_samples),
    "use_symlinks": bool(use_symlinks),
}
(out_dir / "cache_info.json").write_text(json.dumps(cache_info, indent=2, sort_keys=True))
print(f"completed real cache entries: {len(chosen_paths)}")
PY

echo "[resume] handing off to standard FD-only evaluator"
export CONTAINER RUN_DIR FD_REAL_IMAGEFOLDER MAX_SAMPLES
bash "${REPO}/stage2_eval_ImageNet1k_fd_only_single.sh"

echo "End: $(date)"
echo "Done."
