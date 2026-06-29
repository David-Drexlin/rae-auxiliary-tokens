#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from stage1_semantic_retention_dino_utils import (
    FrozenDINOFeatureExtractor,
    IndexedImageFolder,
    PairedReconDataset,
    build_transform,
    canonical_dino_family,
    cosine_per_row,
    ensure_dir,
    extract_seed,
    load_alignment_indices,
    load_pickle,
    mean_std,
    save_json,
    summarize_token_cosines,
    topk_accuracy_from_proba,
)


DEFAULT_VAL = "DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/val"
DEFAULT_PROBE = "RAE_ROOT_PLACEHOLDER/assets/analysis/stage1_semantic_retention_dino/probe"
DEFAULT_OUT = "RAE_ROOT_PLACEHOLDER/assets/analysis/stage1_semantic_retention_dino/runs"


TOKEN_ORDER = ["cls", "reg1", "reg2", "reg3", "reg4", "reg_mean", "reg_centroid", "patch_mean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DINO decode-reencode semantic retention for Stage-1 reconstructions.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, default=Path(DEFAULT_VAL))
    parser.add_argument("--probe-dir", type=Path, default=Path(DEFAULT_PROBE))
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--save-folder", type=str, default=None)
    parser.add_argument("--model-label", type=str, default=None)
    parser.add_argument("--family-override", type=str, default=None)
    parser.add_argument("--selected-indices-path", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--stress-label", type=str, default=None)
    return parser.parse_args()


def infer_stress_label(save_folder: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if "AUXSENS" not in save_folder and not any(tag in save_folder for tag in ["_gauss", "_interpo", "_shuffle", "_zero"]):
        return "clean"
    for prefix in ["gauss", "interpo"]:
        marker = f"_{prefix}"
        if marker in save_folder:
            return f"{prefix}{save_folder.rsplit(marker, 1)[-1]}"
    if save_folder.endswith("_shuffle"):
        return "shuffle"
    if save_folder.endswith("_zero"):
        return "zero"
    return "stress"


def append_summary_csv(row: Dict[str, object], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = list(row.keys())
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    artifact = load_pickle(args.probe_dir / "probe.pkl")
    probe = artifact["probe"]
    class_names = list(artifact["class_names"])
    image_size = int(artifact["image_size"])
    encoder_path = str(artifact["encoder_path"])
    encoder_input_size = int(artifact["encoder_input_size"])

    save_folder = args.save_folder or args.run_dir.name
    out_dir = args.output_root / save_folder
    ensure_dir(out_dir)

    transform = build_transform(image_size)
    val_dataset = IndexedImageFolder(str(args.val_data), transform=transform)
    selected_indices = load_alignment_indices(args.run_dir, len(val_dataset), args.selected_indices_path)
    recon_paths = sorted(p for p in args.run_dir.glob("*.png") if p.is_file())
    if len(selected_indices) != len(recon_paths):
        raise ValueError(
            f"Alignment mismatch: {len(selected_indices)} selected indices vs {len(recon_paths)} PNGs in {args.run_dir}"
        )

    if args.max_images is not None:
        keep = int(args.max_images)
        selected_indices = selected_indices[:keep]
        recon_paths = recon_paths[:keep]

    paired = PairedReconDataset(val_dataset, selected_indices, recon_paths, image_size=image_size)
    loader = DataLoader(
        paired,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    extractor = FrozenDINOFeatureExtractor(
        encoder_path=encoder_path,
        encoder_input_size=encoder_input_size,
        device=args.device,
        precision=args.precision,
    )

    real_views: Dict[str, List[np.ndarray]] = {k: [] for k in ["cls", "reg1", "reg2", "reg3", "reg4", "patch_mean"]}
    recon_views: Dict[str, List[np.ndarray]] = {k: [] for k in ["cls", "reg1", "reg2", "reg3", "reg4", "patch_mean"]}
    labels_all: List[np.ndarray] = []
    ds_indices_all: List[np.ndarray] = []
    real_paths_all: List[str] = []
    recon_paths_all: List[str] = []

    for batch in tqdm(loader, desc=f"semantic_eval:{save_folder}", leave=False):
        real_imgs, recon_imgs, labels, ds_indices, real_paths, recon_batch_paths = batch
        real_encoded = extractor.encode_views(real_imgs)
        recon_encoded = extractor.encode_views(recon_imgs)
        for key in real_views:
            real_views[key].append(real_encoded[key].numpy())
            recon_views[key].append(recon_encoded[key].numpy())
        labels_all.append(labels.numpy().astype(np.int64, copy=False))
        ds_indices_all.append(ds_indices.numpy().astype(np.int64, copy=False))
        real_paths_all.extend(list(real_paths))
        recon_paths_all.extend(list(recon_batch_paths))

    arrays_real = {key: np.concatenate(value, axis=0) for key, value in real_views.items()}
    arrays_recon = {key: np.concatenate(value, axis=0) for key, value in recon_views.items()}
    labels = np.concatenate(labels_all, axis=0)
    ds_indices = np.concatenate(ds_indices_all, axis=0)

    real_proba = probe.predict_proba(arrays_real["cls"])
    recon_proba = probe.predict_proba(arrays_recon["cls"])
    probe_classes = np.asarray(getattr(probe, "classes_", probe.steps[-1][1].classes_))
    real_pred = probe.predict(arrays_real["cls"]).astype(np.int64, copy=False)
    recon_pred = probe.predict(arrays_recon["cls"]).astype(np.int64, copy=False)

    token_cosines: Dict[str, np.ndarray] = {}
    for key in ["cls", "reg1", "reg2", "reg3", "reg4", "patch_mean"]:
        token_cosines[key] = cosine_per_row(arrays_real[key], arrays_recon[key])
    stacked_regs = np.stack([token_cosines[f"reg{i}"] for i in range(1, 5)], axis=1)
    token_cosines["reg_mean"] = stacked_regs.mean(axis=1)
    reg_centroid_real = np.mean(np.stack([arrays_real[f"reg{i}"] for i in range(1, 5)], axis=1), axis=1)
    reg_centroid_recon = np.mean(np.stack([arrays_recon[f"reg{i}"] for i in range(1, 5)], axis=1), axis=1)
    token_cosines["reg_centroid"] = cosine_per_row(reg_centroid_real, reg_centroid_recon)

    probe_top1_real = float(np.mean(real_pred == labels))
    probe_top1_recon = float(np.mean(recon_pred == labels))
    probe_top5_real = topk_accuracy_from_proba(real_proba, labels, probe_classes, k=min(5, len(probe_classes)))
    probe_top5_recon = topk_accuracy_from_proba(recon_proba, labels, probe_classes, k=min(5, len(probe_classes)))
    agreement = real_pred == recon_pred

    per_image_path = out_dir / "per_image_metrics.csv"
    with per_image_path.open("w", newline="") as f:
        fieldnames = [
            "dataset_index",
            "label",
            "label_name",
            "real_path",
            "recon_path",
            "real_pred",
            "real_pred_name",
            "recon_pred",
            "recon_pred_name",
            "real_correct",
            "recon_correct",
            "prediction_agreement",
            "cls_cosine",
            "reg1_cosine",
            "reg2_cosine",
            "reg3_cosine",
            "reg4_cosine",
            "reg_mean_cosine",
            "reg_centroid_cosine",
            "patch_mean_cosine",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(labels)):
            writer.writerow(
                {
                    "dataset_index": int(ds_indices[i]),
                    "label": int(labels[i]),
                    "label_name": class_names[int(labels[i])],
                    "real_path": real_paths_all[i],
                    "recon_path": recon_paths_all[i],
                    "real_pred": int(real_pred[i]),
                    "real_pred_name": class_names[int(real_pred[i])],
                    "recon_pred": int(recon_pred[i]),
                    "recon_pred_name": class_names[int(recon_pred[i])],
                    "real_correct": int(real_pred[i] == labels[i]),
                    "recon_correct": int(recon_pred[i] == labels[i]),
                    "prediction_agreement": int(agreement[i]),
                    "cls_cosine": float(token_cosines["cls"][i]),
                    "reg1_cosine": float(token_cosines["reg1"][i]),
                    "reg2_cosine": float(token_cosines["reg2"][i]),
                    "reg3_cosine": float(token_cosines["reg3"][i]),
                    "reg4_cosine": float(token_cosines["reg4"][i]),
                    "reg_mean_cosine": float(token_cosines["reg_mean"][i]),
                    "reg_centroid_cosine": float(token_cosines["reg_centroid"][i]),
                    "patch_mean_cosine": float(token_cosines["patch_mean"][i]),
                }
            )

    latent_summary_path = out_dir / "latent_retention_by_token.csv"
    with latent_summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["token", "mean", "std", "min", "max"])
        writer.writeheader()
        for token_name in TOKEN_ORDER:
            stats = mean_std(token_cosines[token_name].tolist())
            writer.writerow({"token": token_name, **stats})

    family = args.family_override
    if family is None:
        family = canonical_dino_family(save_folder)
    if family is None and args.model_label is not None:
        family = canonical_dino_family(args.model_label)

    summary = {
        "run_dir": str(args.run_dir),
        "save_folder": save_folder,
        "model_label": args.model_label or save_folder,
        "family": family,
        "seed": extract_seed(save_folder),
        "stress_label": infer_stress_label(save_folder, args.stress_label),
        "num_images_evaluated": int(len(labels)),
        "probe_top1_real": probe_top1_real,
        "probe_top1_recon": probe_top1_recon,
        "probe_accuracy_drop": float(probe_top1_real - probe_top1_recon),
        "probe_top5_real": probe_top5_real,
        "probe_top5_recon": probe_top5_recon,
        "probe_top5_drop": float(probe_top5_real - probe_top5_recon),
        "prediction_agreement_real_vs_recon": float(np.mean(agreement)),
    }
    summary.update(summarize_token_cosines(token_cosines))
    save_json(summary, out_dir / "summary.json")

    if args.csv_out is not None:
        append_summary_csv(summary, args.csv_out)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
