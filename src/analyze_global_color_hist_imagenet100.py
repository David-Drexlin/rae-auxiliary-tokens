#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(REPO_ROOT / "src"))

from analyze_dino_global_tokens_imagenet100 import (  # noqa: E402
    bootstrap_one_nn_accuracy,
    center_crop_arr,
    compute_rgb_hist_batch,
    compute_lowfreq_dct_batch,
    ensure_dir,
    l2_normalize,
    make_errorbar_barplot,
    prefixed_stats,
    repeated_kmeans_metrics,
    run_repeated_target_regression,
    save_json,
    split_channelwise_dc_ac,
)
from stage1.encoders.dinov2 import Dinov2withNorm  # noqa: E402
from stage1.encoders.mae import MAEwNorm  # noqa: E402
from stage1.encoders.siglip2 import SigLIP2wNorm  # noqa: E402


class IndexedImageFolder(ImageFolder):
    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        return image, int(label), int(index)


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.ToTensor(),
        ]
    )


def build_encoder(kind: str, model_path: str) -> torch.nn.Module:
    if kind == "dino":
        return Dinov2withNorm(dinov2_path=model_path, normalize=True)
    if kind == "mae":
        return MAEwNorm(model_name=model_path)
    if kind == "siglip2":
        return SigLIP2wNorm(model_name=model_path)
    raise ValueError(f"Unsupported encoder kind '{kind}'.")


def preprocess_for_encoder(
    images: torch.Tensor,
    input_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    if images.shape[-1] != input_size or images.shape[-2] != input_size:
        images = F.interpolate(images, size=(input_size, input_size), mode="bicubic", align_corners=False)
    return (images - mean) / std


def view_dict_for_kind(
    kind: str,
    patch_mean_np: np.ndarray,
    global_np: np.ndarray,
) -> Dict[str, np.ndarray]:
    if kind == "dino":
        regs_np = global_np[:, 1:]
        return {
            "cls": global_np[:, 0],
            "patch_mean": patch_mean_np,
            "reg1": regs_np[:, 0],
            "reg2": regs_np[:, 1],
            "reg3": regs_np[:, 2],
            "reg4": regs_np[:, 3],
            "reg_mean": regs_np.mean(axis=1),
            "reg_concat": regs_np.reshape(regs_np.shape[0], -1),
        }
    if kind == "mae":
        return {
            "cls": global_np[:, 0],
            "patch_mean": patch_mean_np,
        }
    if kind == "siglip2":
        return {
            "pooler": global_np[:, 0],
            "patch_mean": patch_mean_np,
        }
    raise ValueError(f"Unsupported encoder kind '{kind}'.")


def ordered_views_for_kind(kind: str) -> List[str]:
    if kind == "dino":
        return ["cls", "patch_mean", "reg1", "reg2", "reg3", "reg4", "reg_mean", "reg_concat"]
    if kind == "mae":
        return ["cls", "patch_mean"]
    if kind == "siglip2":
        return ["pooler", "patch_mean"]
    raise ValueError(f"Unsupported encoder kind '{kind}'.")


def compute_thumbnail_batch(images: torch.Tensor, size: int) -> np.ndarray:
    thumbs = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    return thumbs.flatten(start_dim=1).detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--encoder-kind", type=str, required=True, choices=["dino", "mae", "siglip2"])
    parser.add_argument("--encoder-path", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--encoder-input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--kmeans-repeats", type=int, default=5)
    parser.add_argument("--kmeans-n-init", type=int, default=20)
    parser.add_argument("--one-nn-bootstrap-reps", type=int, default=30)
    parser.add_argument("--probe-splits", type=int, default=5)
    parser.add_argument("--probe-test-size", type=float, default=0.3)
    parser.add_argument("--color-bins", type=int, default=16)
    parser.add_argument("--dct-input-size", type=int, default=32)
    parser.add_argument("--dct-keep", type=int, default=8)
    parser.add_argument("--thumbnail-size", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"
    ensure_dir(plots_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_autocast = device.type == "cuda" and args.precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    transform = build_transform(args.image_size)
    dataset = IndexedImageFolder(args.data_path, transform=transform)
    if args.max_samples > 0:
        keep = min(args.max_samples, len(dataset))
        dataset.samples = dataset.samples[:keep]
        dataset.targets = dataset.targets[:keep]
    class_names = list(dataset.classes)
    image_paths = [sample[0] for sample in dataset.samples]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    processor = AutoImageProcessor.from_pretrained(args.encoder_path)
    encoder_mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    encoder_std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1).to(device)

    encoder = build_encoder(args.encoder_kind, args.encoder_path).to(device).eval()

    labels_all: List[int] = []
    patch_mean_all: List[torch.Tensor] = []
    global_all: List[torch.Tensor] = []
    color_hist_all: List[np.ndarray] = []
    lowfreq_dct_all: List[np.ndarray] = []
    thumbnail_all: List[np.ndarray] = []

    with torch.inference_mode():
        for images, labels, _indices in tqdm(loader, desc="extract"):
            images = images.to(device, non_blocking=True)
            labels_all.extend(labels.tolist())
            color_hist_all.append(compute_rgb_hist_batch(images, bins=args.color_bins))
            lowfreq_dct_all.append(compute_lowfreq_dct_batch(images, input_size=args.dct_input_size, keep=args.dct_keep))
            thumbnail_all.append(compute_thumbnail_batch(images, size=args.thumbnail_size))
            images = preprocess_for_encoder(images, args.encoder_input_size, encoder_mean, encoder_std)
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    patch_tokens, global_tokens = encoder.forward_with_global(images)
            else:
                patch_tokens, global_tokens = encoder.forward_with_global(images)
            patch_mean_all.append(patch_tokens.mean(dim=1).float().cpu())
            global_all.append(global_tokens.float().cpu())

    labels_np = np.asarray(labels_all, dtype=np.int64)
    patch_mean_np = torch.cat(patch_mean_all, dim=0).numpy()
    global_np = torch.cat(global_all, dim=0).numpy()
    color_hist_np = np.concatenate(color_hist_all, axis=0)
    lowfreq_dct_np = np.concatenate(lowfreq_dct_all, axis=0)
    thumbnail_np = np.concatenate(thumbnail_all, axis=0)
    lowfreq_dct_dc_np, lowfreq_dct_ac_np = split_channelwise_dc_ac(lowfreq_dct_np, keep=args.dct_keep)
    views = view_dict_for_kind(args.encoder_kind, patch_mean_np, global_np)

    torch.save(
        {
            "labels": labels_np,
            "class_names": class_names,
            "image_paths": image_paths,
            "patch_mean": patch_mean_np,
            "global_tokens": global_np,
            "views": views,
            "color_hist": color_hist_np,
            "lowfreq_dct": lowfreq_dct_np,
            "encoder_kind": args.encoder_kind,
            "encoder_path": args.encoder_path,
        },
        outdir / "embeddings.pt",
    )

    metric_rows = []
    color_rows = []
    dct_rows = []
    thumbnail_rows = []
    for view_name, feat in views.items():
        feat_norm = l2_normalize(feat.astype(np.float32, copy=False))
        kmeans_row, _assignments, _kmeans = repeated_kmeans_metrics(
            feat_norm,
            labels_np,
            n_clusters=len(class_names),
            base_seed=args.seed,
            repeats=args.kmeans_repeats,
            n_init=args.kmeans_n_init,
        )
        one_nn_stats = bootstrap_one_nn_accuracy(
            feat_norm,
            labels_np,
            reps=args.one_nn_bootstrap_reps,
            seed=args.seed + 1000,
        )
        metric_rows.append(
            {
                "view": view_name,
                "dim": int(feat_norm.shape[1]),
                **kmeans_row,
                "one_nn_acc": float(one_nn_stats["base"]),
                "one_nn_mean": float(one_nn_stats["mean"]),
                "one_nn_std": float(one_nn_stats["std"]),
                "one_nn_ci_low": float(one_nn_stats["ci_low"]),
                "one_nn_ci_high": float(one_nn_stats["ci_high"]),
                "one_nn_bootstrap_reps": int(one_nn_stats["reps"]),
            }
        )
        color_stats = run_repeated_target_regression(
            feat_norm,
            color_hist_np,
            labels_np,
            seed=args.seed,
            splits=args.probe_splits,
            test_size=args.probe_test_size,
            alpha=args.ridge_alpha,
        )
        color_rows.append(
            {
                "view": view_name,
                "dim": int(feat_norm.shape[1]),
                **color_stats,
            }
        )
        dct_stats = run_repeated_target_regression(
            feat_norm,
            lowfreq_dct_np,
            labels_np,
            seed=args.seed,
            splits=args.probe_splits,
            test_size=args.probe_test_size,
            alpha=args.ridge_alpha,
        )
        dct_dc_stats = run_repeated_target_regression(
            feat_norm,
            lowfreq_dct_dc_np,
            labels_np,
            seed=args.seed,
            splits=args.probe_splits,
            test_size=args.probe_test_size,
            alpha=args.ridge_alpha,
        )
        dct_ac_stats = run_repeated_target_regression(
            feat_norm,
            lowfreq_dct_ac_np,
            labels_np,
            seed=args.seed,
            splits=args.probe_splits,
            test_size=args.probe_test_size,
            alpha=args.ridge_alpha,
        )
        dct_rows.append(
            {
                "view": view_name,
                "dim": int(feat_norm.shape[1]),
                "target_dim": int(lowfreq_dct_np.shape[1]),
                "dc_target_dim": int(lowfreq_dct_dc_np.shape[1]),
                "ac_target_dim": int(lowfreq_dct_ac_np.shape[1]),
                **dct_stats,
                **prefixed_stats(dct_dc_stats, "dc"),
                **prefixed_stats(dct_ac_stats, "ac"),
            }
        )
        thumbnail_stats = run_repeated_target_regression(
            feat_norm,
            thumbnail_np,
            labels_np,
            seed=args.seed,
            splits=args.probe_splits,
            test_size=args.probe_test_size,
            alpha=args.ridge_alpha,
        )
        thumbnail_rows.append(
            {
                "view": view_name,
                "dim": int(feat_norm.shape[1]),
                "target_dim": int(thumbnail_np.shape[1]),
                **thumbnail_stats,
            }
        )

    metric_rows.sort(key=lambda row: row["view"])
    with open(outdir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    color_rows.sort(key=lambda row: row["view"])
    with open(outdir / "linear_probe_color_hist.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(color_rows[0].keys()))
        writer.writeheader()
        writer.writerows(color_rows)

    dct_rows.sort(key=lambda row: row["view"])
    with open(outdir / "linear_probe_lowfreq_dct.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dct_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dct_rows)

    thumbnail_rows.sort(key=lambda row: row["view"])
    with open(outdir / "linear_probe_thumbnail.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(thumbnail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(thumbnail_rows)

    ordered_names = ordered_views_for_kind(args.encoder_kind)
    color_plot_rows = [row for name in ordered_names for row in color_rows if row["view"] == name]
    dct_plot_rows = [row for name in ordered_names for row in dct_rows if row["view"] == name]
    thumbnail_plot_rows = [row for name in ordered_names for row in thumbnail_rows if row["view"] == name]
    make_errorbar_barplot(
        color_plot_rows,
        [row["view"] for row in color_plot_rows],
        metric_key="r2_mean",
        err_key="r2_std",
        out_path=plots_dir / "linear_probe_color_hist_r2.png",
        title=f"{args.encoder_kind.upper()} color histogram regression by token view",
        ylabel="R^2",
    )
    make_errorbar_barplot(
        dct_plot_rows,
        [row["view"] for row in dct_plot_rows],
        metric_key="r2_mean",
        err_key="r2_std",
        out_path=plots_dir / "linear_probe_lowfreq_dct_r2.png",
        title=f"{args.encoder_kind.upper()} low-frequency DCT regression by token view",
        ylabel="R^2",
    )
    make_errorbar_barplot(
        dct_plot_rows,
        [row["view"] for row in dct_plot_rows],
        metric_key="ac_r2_mean",
        err_key="ac_r2_std",
        out_path=plots_dir / "linear_probe_lowfreq_dct_ac_r2.png",
        title=f"{args.encoder_kind.upper()} low-frequency DCT AC regression by token view",
        ylabel="R^2",
    )
    make_errorbar_barplot(
        dct_plot_rows,
        [row["view"] for row in dct_plot_rows],
        metric_key="dc_r2_mean",
        err_key="dc_r2_std",
        out_path=plots_dir / "linear_probe_lowfreq_dct_dc_r2.png",
        title=f"{args.encoder_kind.upper()} low-frequency DCT DC regression by token view",
        ylabel="R^2",
    )
    make_errorbar_barplot(
        thumbnail_plot_rows,
        [row["view"] for row in thumbnail_plot_rows],
        metric_key="r2_mean",
        err_key="r2_std",
        out_path=plots_dir / "linear_probe_thumbnail_r2.png",
        title=f"{args.encoder_kind.upper()} {args.thumbnail_size}x{args.thumbnail_size} thumbnail regression by token view",
        ylabel="R^2",
    )
    metric_plot_rows = [row for name in ordered_names for row in metric_rows if row["view"] == name]
    make_errorbar_barplot(
        metric_plot_rows,
        [row["view"] for row in metric_plot_rows],
        metric_key="one_nn_mean",
        err_key="one_nn_std",
        out_path=plots_dir / "one_nn_accuracy_bootstrap.png",
        title=f"{args.encoder_kind.upper()} 1-NN accuracy by token view",
        ylabel="1-NN accuracy",
    )

    summary = {
        "data_path": args.data_path,
        "num_samples": len(labels_np),
        "num_classes": len(class_names),
        "class_names": class_names,
        "encoder_kind": args.encoder_kind,
        "encoder_path": args.encoder_path,
        "image_size": args.image_size,
        "encoder_input_size": args.encoder_input_size,
        "patch_mean_definition": "Mean of encoder patch-token embeddings computed per image before fitting the appearance probes.",
        "metrics": metric_rows,
        "linear_probe_color_hist": color_rows,
        "linear_probe_lowfreq_dct": dct_rows,
        "linear_probe_thumbnail": thumbnail_rows,
        "best_view_by_r2": max(color_rows, key=lambda row: row["r2_mean"]),
        "best_view_by_lowfreq_dct_r2": max(dct_rows, key=lambda row: row["r2_mean"]),
        "best_view_by_lowfreq_dct_ac_r2": max(dct_rows, key=lambda row: row["ac_r2_mean"]),
        "best_view_by_thumbnail_r2": max(thumbnail_rows, key=lambda row: row["r2_mean"]),
        "artifacts": {
            "embeddings": str(outdir / "embeddings.pt"),
            "metrics": str(outdir / "metrics.csv"),
            "linear_probe_color_hist": str(outdir / "linear_probe_color_hist.csv"),
            "linear_probe_lowfreq_dct": str(outdir / "linear_probe_lowfreq_dct.csv"),
            "linear_probe_thumbnail": str(outdir / "linear_probe_thumbnail.csv"),
            "plot_color_hist_r2": str(plots_dir / "linear_probe_color_hist_r2.png"),
            "plot_lowfreq_dct_r2": str(plots_dir / "linear_probe_lowfreq_dct_r2.png"),
            "plot_lowfreq_dct_dc_r2": str(plots_dir / "linear_probe_lowfreq_dct_dc_r2.png"),
            "plot_lowfreq_dct_ac_r2": str(plots_dir / "linear_probe_lowfreq_dct_ac_r2.png"),
            "plot_thumbnail_r2": str(plots_dir / "linear_probe_thumbnail_r2.png"),
            "plot_one_nn_accuracy": str(plots_dir / "one_nn_accuracy_bootstrap.png"),
        },
    }
    save_json(summary, outdir / "summary.json")


if __name__ == "__main__":
    main()
