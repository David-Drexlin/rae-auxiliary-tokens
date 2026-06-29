#!/usr/bin/env python3
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate a stage-1 reconstruction run with distributional metrics only.

What this script computes
-------------------------
1) FID
   - Uses the guided-diffusion evaluator on:
       real_npz vs fake_samples_npz

2) FD-DINOv2
   - Uses the FD-DINOv2 script on PNG directories:
       real_png_dir vs fake_png_dir

What this script intentionally does NOT compute
-----------------------------------------------
- PSNR
- SSIM
- LPIPS

Those pairwise metrics require 1:1 image alignment. FID / FD-DINOv2 do not.

Expected run directory
----------------------
The reconstruction run directory should usually contain:
- samples.npz
- optional selected_indices.npy
- many recon PNGs, e.g. 000000.png, 000001.png, ...

Typical usage
-------------
python eval_stage1_recon.py \
    --run-dir RAE_ROOT_PLACEHOLDER/assets/recon_samples/UNI2_Patch_Register_CLS_recon \
    --fid-real-npz RAE_ROOT_PLACEHOLDER/assets/datasets/tcga_adm_128.npz \
    --fid-evaluator HOME_PLACEHOLDER/guided-diffusion/evaluations/evaluator.py \
    --fd-dinov2-script HOME_PLACEHOLDER/FD-DINOv2/src/pytorch_fd/fd_score.py \
    --fd-real-imagefolder RAE_ROOT_PLACEHOLDER/assets/datasets/tcga_imagefolder_all \
    --device cuda:0 \
    --num-workers 8 \
    --fd-batch-size 128 \
    --model-name UNI2_Patch_Register_CLS \
    --csv-out RAE_ROOT_PLACEHOLDER/assets/eval_results/stage1_recon_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from torchvision.datasets import ImageFolder


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".pgm", ".png", ".ppm", ".tif", ".tiff", ".webp"}


class FilteredImageFolder(ImageFolder):
    """ImageFolder variant that ignores helper directories like _dino_latents."""

    def find_classes(self, directory: str):
        classes = sorted(
            entry.name
            for entry in os.scandir(directory)
            if entry.is_dir() and not entry.name.startswith("_")
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def count_pngs(path: Path) -> int:
    return len(sorted(path.glob("*.png")))


def list_fake_pngs(run_dir: Path) -> List[Path]:
    pngs = sorted(p for p in run_dir.glob("*.png") if p.is_file())
    return pngs


def maybe_load_selected_indices(path: Optional[Path]) -> Optional[List[int]]:
    if path is None or not path.exists():
        return None
    import numpy as np
    arr = np.load(path)
    return [int(x) for x in arr.tolist()]


def clear_directory_files(path: Path) -> None:
    if not path.exists():
        return
    for p in path.iterdir():
        if p.is_file() or p.is_symlink():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)


def extract_metric_from_text(text: str, preferred_keys: List[str]) -> float:
    """
    Try to extract a metric value from stdout/stderr.

    Strategy:
    1) search reversed lines for any preferred key and a float
    2) otherwise take the last float found in the full text
    """
    float_pattern = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

    lines = text.splitlines()
    for line in reversed(lines):
        line_lower = line.lower()
        if any(k.lower() in line_lower for k in preferred_keys):
            vals = re.findall(float_pattern, line)
            if vals:
                return float(vals[-1])

    vals = re.findall(float_pattern, text)
    if vals:
        return float(vals[-1])

    raise RuntimeError("Could not parse metric value from subprocess output.")


def run_subprocess(
    cmd: List[str],
    log_path: Path,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> str:
    ensure_dir(log_path.parent)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"See log: {log_path}"
        )
    return proc.stdout


# -----------------------------------------------------------------------------
# Real PNG cache for FD-DINOv2
# -----------------------------------------------------------------------------
def build_real_png_cache(
    real_imagefolder: Path,
    out_dir: Path,
    selected_indices: Optional[List[int]] = None,
    max_samples: Optional[int] = None,
    use_symlinks: bool = True,
) -> Path:
    """
    Build a flat PNG directory for FD-DINOv2 from an ImageFolder tree.

    Ordering is ImageFolder ordering, which matches torchvision ImageFolder
    used by the reconstruction pipeline.
    """
    ensure_dir(out_dir)

    ds = FilteredImageFolder(str(real_imagefolder))
    all_paths = [Path(p) for p, _ in ds.samples]

    if selected_indices is None:
        chosen_paths = all_paths
    else:
        chosen_paths = [all_paths[i] for i in selected_indices]

    if max_samples is not None:
        chosen_paths = chosen_paths[: int(max_samples)]

    cache_info = {
        "real_imagefolder": str(real_imagefolder.resolve()),
        "num_total_imagefolder_samples": len(all_paths),
        "num_selected": len(chosen_paths),
        "selected_indices_used": selected_indices is not None,
        "max_samples": None if max_samples is None else int(max_samples),
        "use_symlinks": bool(use_symlinks),
    }
    info_path = out_dir / "cache_info.json"

    reuse_ok = False
    if info_path.exists():
        try:
            old_info = load_json(info_path)
            existing_pngs = count_pngs(out_dir)
            reuse_ok = (old_info == cache_info) and (existing_pngs == len(chosen_paths))
        except Exception:
            reuse_ok = False

    if reuse_ok:
        return out_dir

    clear_directory_files(out_dir)

    for i, src_path in enumerate(chosen_paths):
        dst_path = out_dir / f"{i:06d}{src_path.suffix.lower()}"
        if use_symlinks:
            try:
                os.symlink(src_path, dst_path)
                continue
            except OSError:
                pass
        shutil.copy2(src_path, dst_path)

    save_json(cache_info, info_path)
    return out_dir


def build_fake_png_cache(
    fake_dir: Path,
    out_dir: Path,
    max_samples: Optional[int] = None,
    use_symlinks: bool = True,
) -> Path:
    """
    Optional flat cache for fake PNGs.
    Needed only if you want to truncate fake PNGs with max_samples.
    """
    ensure_dir(out_dir)

    fake_pngs = list_fake_pngs(fake_dir)
    if max_samples is not None:
        fake_pngs = fake_pngs[: int(max_samples)]

    cache_info = {
        "fake_dir": str(fake_dir.resolve()),
        "num_selected": len(fake_pngs),
        "max_samples": None if max_samples is None else int(max_samples),
        "use_symlinks": bool(use_symlinks),
    }
    info_path = out_dir / "cache_info.json"

    reuse_ok = False
    if info_path.exists():
        try:
            old_info = load_json(info_path)
            existing_pngs = count_pngs(out_dir)
            reuse_ok = (old_info == cache_info) and (existing_pngs == len(fake_pngs))
        except Exception:
            reuse_ok = False

    if reuse_ok:
        return out_dir

    clear_directory_files(out_dir)

    for i, src_path in enumerate(fake_pngs):
        dst_path = out_dir / f"{i:06d}{src_path.suffix.lower()}"
        if use_symlinks:
            try:
                os.symlink(src_path, dst_path)
                continue
            except OSError:
                pass
        shutil.copy2(src_path, dst_path)

    save_json(cache_info, info_path)
    return out_dir


# -----------------------------------------------------------------------------
# Metric runners
# -----------------------------------------------------------------------------
def compute_fid(
    fid_evaluator: Path,
    real_npz: Path,
    fake_npz: Path,
    log_path: Path,
) -> float:
    if not fid_evaluator.exists():
        raise FileNotFoundError(f"FID evaluator not found: {fid_evaluator}")
    if not real_npz.exists():
        raise FileNotFoundError(f"Real NPZ not found: {real_npz}")
    if not fake_npz.exists():
        raise FileNotFoundError(f"Fake NPZ not found: {fake_npz}")

    cmd = [sys.executable, str(fid_evaluator), str(real_npz), str(fake_npz)]
    out = run_subprocess(cmd, log_path=log_path, cwd=fid_evaluator.parent)
    float_pattern = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    for line in reversed(out.splitlines()):
        lower = line.lower().strip()
        if lower.startswith("fid:") or lower.startswith("[result] fid:"):
            vals = re.findall(float_pattern, line)
            if vals:
                return float(vals[-1])
    return extract_metric_from_text(out, preferred_keys=["fid", "frechet"])


def compute_fd_dinov2(
    fd_score_py: Path,
    real_dir: Path,
    fake_dir: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    log_path: Path,
) -> float:
    if not fd_score_py.exists():
        raise FileNotFoundError(f"FD-DINOv2 script not found: {fd_score_py}")
    if not real_dir.exists():
        raise FileNotFoundError(f"FD-DINOv2 real dir not found: {real_dir}")
    if not fake_dir.exists():
        raise FileNotFoundError(f"FD-DINOv2 fake dir not found: {fake_dir}")

    cmd = [
        sys.executable,
        str(fd_score_py),
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--device", device,
        str(real_dir),
        str(fake_dir),
    ]
    fd_env = os.environ.copy()
    fd_env.setdefault("XFORMERS_DISABLED", "1")
    out = run_subprocess(cmd, log_path=log_path, cwd=fd_score_py.parent, env=fd_env)
    return extract_metric_from_text(out, preferred_keys=["fd-dinov2", "fd", "dinov2"])


# -----------------------------------------------------------------------------
# CSV writing
# -----------------------------------------------------------------------------
def append_row_to_csv(csv_path: Path, row: Dict) -> None:
    ensure_dir(csv_path.parent)

    if csv_path.exists():
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames or []
    else:
        existing_rows = []
        existing_fields = []

    new_fields = list(existing_fields)
    for k in row.keys():
        if k not in new_fields:
            new_fields.append(k)

    if existing_rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=new_fields)
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)
            writer.writerow(row)
    else:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=new_fields)
            writer.writeheader()
            writer.writerow(row)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate stage-1 reconstructions with FID and FD-DINOv2.")

    p.add_argument("--run-dir", type=str, required=True, help="Reconstruction run directory containing PNGs and usually samples.npz.")
    p.add_argument("--model-name", type=str, default=None, help="Optional human-readable model name for CSV.")
    p.add_argument("--csv-out", type=str, required=True, help="CSV file to append one result row to.")

    p.add_argument("--fid-real-npz", type=str, required=True, help="Real NPZ for FID.")
    p.add_argument("--fid-fake-npz", type=str, default=None, help="Fake NPZ for FID. Defaults to <run-dir>/samples.npz.")
    p.add_argument("--fid-evaluator", type=str, required=True, help="Path to guided-diffusion evaluator.py.")

    p.add_argument("--fd-dinov2-script", type=str, required=True, help="Path to FD-DINOv2 fd_score.py.")
    p.add_argument("--fd-real-imagefolder", type=str, required=True, help="Root ImageFolder directory for real images.")
    p.add_argument("--selected-indices", type=str, default=None, help="Optional explicit selected_indices.npy. Defaults to <run-dir>/selected_indices.npy if present.")

    p.add_argument("--device", type=str, default="cuda:0", help="Device for FD-DINOv2.")
    p.add_argument("--fd-batch-size", type=int, default=128, help="Batch size for FD-DINOv2.")
    p.add_argument("--num-workers", type=int, default=8, help="Workers for FD-DINOv2.")

    p.add_argument("--max-samples", type=int, default=None, help="Optional truncation for both real/fake PNG sets. Default: use all.")
    p.add_argument("--copy-real-pngs", action="store_true", help="Copy real PNGs instead of symlinking them into the flat cache.")
    p.add_argument("--copy-fake-pngs", action="store_true", help="Copy fake PNGs instead of symlinking them into a truncated cache when max-samples is used.")

    p.add_argument("--skip-fid", action="store_true", help="Skip FID.")
    p.add_argument("--skip-fd-dinov2", action="store_true", help="Skip FD-DINOv2.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    fake_npz = Path(args.fid_fake_npz) if args.fid_fake_npz is not None else (run_dir / "samples.npz")
    real_npz = Path(args.fid_real_npz)
    fid_evaluator = Path(args.fid_evaluator)

    fd_dinov2_script = Path(args.fd_dinov2_script)
    fd_real_imagefolder = Path(args.fd_real_imagefolder)

    selected_indices_path = None
    if args.selected_indices is not None:
        selected_indices_path = Path(args.selected_indices)
    else:
        default_sel = run_dir / "selected_indices.npy"
        if default_sel.exists():
            selected_indices_path = default_sel

    selected_indices = maybe_load_selected_indices(selected_indices_path)
    fake_pngs = list_fake_pngs(run_dir)
    if len(fake_pngs) == 0 and not args.skip_fd_dinov2:
        raise RuntimeError(f"No fake PNGs found in run dir: {run_dir}")

    if args.max_samples is not None and args.max_samples <= 0:
        max_samples = None
    else:
        max_samples = args.max_samples

    metrics: Dict[str, Optional[float]] = {
        "fid": None,
        "fd_dinov2": None,
    }

    # -------------------------
    # FID
    # -------------------------
    if not args.skip_fid:
        metrics["fid"] = compute_fid(
            fid_evaluator=fid_evaluator,
            real_npz=real_npz,
            fake_npz=fake_npz,
            log_path=run_dir / "fid.log",
        )

    # -------------------------
    # FD-DINOv2
    # -------------------------
    real_fd_cache_dir = run_dir / "_fd_real_png_cache"

    if max_samples is None:
        fake_fd_dir = run_dir
    else:
        fake_fd_dir = run_dir / "_fd_fake_png_cache"
        build_fake_png_cache(
            fake_dir=run_dir,
            out_dir=fake_fd_dir,
            max_samples=max_samples,
            use_symlinks=not args.copy_fake_pngs,
        )

    if not args.skip_fd_dinov2:
        build_real_png_cache(
            real_imagefolder=fd_real_imagefolder,
            out_dir=real_fd_cache_dir,
            selected_indices=selected_indices,
            max_samples=max_samples,
            use_symlinks=not args.copy_real_pngs,
        )

        metrics["fd_dinov2"] = compute_fd_dinov2(
            fd_score_py=fd_dinov2_script,
            real_dir=real_fd_cache_dir,
            fake_dir=fake_fd_dir,
            device=args.device,
            batch_size=int(args.fd_batch_size),
            num_workers=int(args.num_workers),
            log_path=run_dir / "fd_dinov2.log",
        )

    # -------------------------
    # Summary row
    # -------------------------
    fake_png_count = len(fake_pngs) if max_samples is None else min(len(fake_pngs), int(max_samples))
    real_png_count = count_pngs(real_fd_cache_dir) if real_fd_cache_dir.exists() else None

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name if args.model_name is not None else run_dir.name,
        "run_dir": str(run_dir),
        "fake_npz": str(fake_npz),
        "real_npz": str(real_npz),
        "fd_real_imagefolder": str(fd_real_imagefolder),
        "selected_indices_path": str(selected_indices_path) if selected_indices_path is not None else "",
        "num_fake_pngs_used": fake_png_count,
        "num_real_pngs_used": real_png_count,
        "max_samples": "" if max_samples is None else int(max_samples),
        "fid": metrics["fid"],
        "fd_dinov2": metrics["fd_dinov2"],
    }

    csv_out = Path(args.csv_out)
    append_row_to_csv(csv_out, row)

    print("Evaluation finished.")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
