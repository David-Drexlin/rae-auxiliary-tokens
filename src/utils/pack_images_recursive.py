#!/usr/bin/env python3
"""
Recursively pack images into an FID-ready .npz:
- Output format: npz with key "images"
- dtype uint8
- shape (N, H, W, 3)  (NHWC)
- Images are resized to --image-size x --image-size and converted to RGB.

This matches common ADM-suite expectations and works with torch-fidelity.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images_recursive(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    files.sort()
    return files


def load_and_preprocess(path: Path, image_size: int) -> np.ndarray:
    # Robust load -> RGB -> resize -> uint8
    with Image.open(path) as im:
        im = im.convert("RGB")
        if image_size is not None:
            # bicubic resize like most vision eval pipelines
            im = im.resize((image_size, image_size), resample=Image.BICUBIC)
        arr = np.asarray(im, dtype=np.uint8)  # HWC, uint8
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unexpected image shape after RGB conversion: {arr.shape} for {path}")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=str, required=True, help="Root folder to scan recursively for images.")
    ap.add_argument("--output-npz", type=str, required=True, help="Path to write the packed .npz.")
    ap.add_argument("--image-size", type=int, default=256, help="Resize images to this square size.")
    ap.add_argument("--max-images", type=int, default=0, help="If >0, pack at most this many images (after sorting).")
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="If set, abort on first unreadable image. Otherwise skip unreadable images.",
    )
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_npz = Path(args.output_npz).expanduser().resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    files = iter_images_recursive(in_dir)
    if args.max_images and args.max_images > 0:
        files = files[: args.max_images]

    if not files:
        raise SystemExit(f"No images found under: {in_dir}")

    N = len(files)
    H = W = int(args.image_size)

    # Use a temporary memmap to avoid holding N*H*W*3 in RAM
    tmp_mmap_path = out_npz.with_suffix(".mmap.npy")
    mmap = np.memmap(tmp_mmap_path, dtype=np.uint8, mode="w+", shape=(N, H, W, 3))

    kept_paths: List[Path] = []
    write_idx = 0

    for p in tqdm(files, desc="Packing images"):
        try:
            img = load_and_preprocess(p, H)
            mmap[write_idx] = img
            kept_paths.append(p)
            write_idx += 1
        except Exception as e:
            if args.fail_fast:
                raise
            # skip unreadable images
            print(f"[WARN] Skipping {p}: {e}")

    mmap.flush()

    if write_idx == 0:
        # cleanup
        try:
            os.remove(tmp_mmap_path)
        except OSError:
            pass
        raise SystemExit("No images were successfully packed.")

    # If some images were skipped, trim by copying into a smaller array (still memmap-backed -> npz)
    if write_idx != N:
        trimmed = np.memmap(
            tmp_mmap_path, dtype=np.uint8, mode="r", shape=(N, H, W, 3)
        )[:write_idx].copy()
        np.savez_compressed(out_npz, images=trimmed)
    else:
        # Load memmap as view and write
        np.savez_compressed(out_npz, images=np.asarray(mmap))

    # Optional: write an index file for traceability
    index_path = out_npz.with_suffix(".paths.txt")
    with open(index_path, "w") as f:
        for p in kept_paths:
            f.write(str(p) + "\n")

    # cleanup memmap backing file
    try:
        os.remove(tmp_mmap_path)
    except OSError:
        pass

    print(f"Packed {write_idx} images -> {out_npz}")
    print(f"Paths written -> {index_path}")


if __name__ == "__main__":
    main()
