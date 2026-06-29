#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def find_images_under(cancer_dir: Path) -> List[Path]:
    return sorted([p for p in cancer_dir.rglob("*") if p.is_file() and is_image(p)])

def infer_slide_id_from_path(p: Path) -> Optional[str]:
    # .../<cancer_type>/<slide_submitter_id>/<tile_file>
    return p.parent.name if p.parent is not None else None

def load_and_resize_rgb(img_path: Path, image_size: int) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((image_size, image_size), resample=Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)  # (H,W,3)

def build_metadata_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    if "slide_submitter_id" not in df.columns:
        raise ValueError("CSV must contain column 'slide_submitter_id'")
    lookup = {}
    for _, row in df.iterrows():
        sid = str(row["slide_submitter_id"])
        lookup[sid] = row.to_dict()
    return lookup

def pack_one_cancer(
    cancer_dir: Path,
    out_npz: Path,
    image_size: int,
    filter_slide_ids: Optional[set],
    meta_lookup: Optional[Dict[str, Dict[str, object]]],
    extra_meta_cols: Optional[List[str]],
    store_paths_relative_to: Optional[Path],
    adm_only: bool,
    write_meta_sidecar: bool,
) -> None:
    imgs = find_images_under(cancer_dir)
    if not imgs:
        return

    kept_imgs: List[Path] = []
    slide_ids: List[str] = []
    meta_rows: List[Dict[str, object]] = []

    for p in imgs:
        sid = infer_slide_id_from_path(p)
        if sid is None:
            continue
        if filter_slide_ids is not None and sid not in filter_slide_ids:
            continue

        kept_imgs.append(p)
        slide_ids.append(sid)
        if meta_lookup is not None:
            meta_rows.append(meta_lookup.get(sid, {}))

    if not kept_imgs:
        return

    # Load images -> uint8 NHWC
    arr = np.empty((len(kept_imgs), image_size, image_size, 3), dtype=np.uint8)
    paths_out: List[str] = []

    for i, p in enumerate(tqdm(kept_imgs, desc=f"Packing {cancer_dir.name}", unit="img")):
        arr[i] = load_and_resize_rgb(p, image_size)
        if store_paths_relative_to is not None:
            try:
                paths_out.append(str(p.relative_to(store_paths_relative_to)))
            except ValueError:
                paths_out.append(str(p))
        else:
            paths_out.append(str(p))

    out_npz.parent.mkdir(parents=True, exist_ok=True)

    if adm_only:
        # Strict ADM-compatible payload: ONLY arr_0
        np.savez_compressed(out_npz, arr_0=arr)
    else:
        # Your richer payload (kept here if you still want it)
        payload = {
            "images": arr,
            "paths": np.array(paths_out, dtype=object),
            "slide_submitter_id": np.array(slide_ids, dtype=object),
            "cancer_type": np.array([cancer_dir.name] * len(kept_imgs), dtype=object),
        }
        if meta_lookup is not None and extra_meta_cols:
            for col in extra_meta_cols:
                payload[col] = np.array([mr.get(col, None) for mr in meta_rows], dtype=object)
        np.savez_compressed(out_npz, **payload)

    if write_meta_sidecar:
        meta_payload = {
            "paths": np.array(paths_out, dtype=object),
            "slide_submitter_id": np.array(slide_ids, dtype=object),
            "cancer_type": np.array([cancer_dir.name] * len(kept_imgs), dtype=object),
        }
        if meta_lookup is not None and extra_meta_cols:
            for col in extra_meta_cols:
                meta_payload[col] = np.array([mr.get(col, None) for mr in meta_rows], dtype=object)

        meta_path = out_npz.with_suffix(".meta.npz")
        np.savez_compressed(meta_path, **meta_payload)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcga-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--image-size", type=int, default=256)

    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--split-col", type=str, default=None)
    ap.add_argument("--split-val", type=str, default=None)
    ap.add_argument("--meta-cols", type=str, default=None)

    ap.add_argument("--paths-relative-to", type=str, default=None)

    # NEW:
    ap.add_argument("--adm-only", action="store_true",
                    help="Write ADM-compatible NPZ with ONLY arr_0 (uint8 NHWC).")
    ap.add_argument("--write-meta-sidecar", action="store_true",
                    help="Also write <name>.meta.npz with paths/slide ids/meta cols.")

    args = ap.parse_args()

    tcga_root = Path(args.tcga_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_lookup = None
    filter_slide_ids = None
    extra_meta_cols = None

    if args.csv is not None:
        df = pd.read_csv(args.csv)
        if args.split_col and args.split_val is not None:
            df = df[df[args.split_col].astype(str) == str(args.split_val)]
        meta_lookup = build_metadata_lookup(df)
        filter_slide_ids = set(meta_lookup.keys())
        if args.meta_cols:
            extra_meta_cols = [c.strip() for c in args.meta_cols.split(",") if c.strip()]

    rel_root = Path(args.paths_relative_to).expanduser().resolve() if args.paths_relative_to else None

    cancer_dirs = sorted([p for p in tcga_root.iterdir() if p.is_dir()])
    for cdir in cancer_dirs:
        out_npz = out_dir / f"{cdir.name}.npz"
        pack_one_cancer(
            cancer_dir=cdir,
            out_npz=out_npz,
            image_size=args.image_size,
            filter_slide_ids=filter_slide_ids,
            meta_lookup=meta_lookup,
            extra_meta_cols=extra_meta_cols,
            store_paths_relative_to=rel_root,
            adm_only=args.adm_only,
            write_meta_sidecar=args.write_meta_sidecar,
        )

    print(f"Done. Wrote NPZ files under: {out_dir}")

if __name__ == "__main__":
    main()
