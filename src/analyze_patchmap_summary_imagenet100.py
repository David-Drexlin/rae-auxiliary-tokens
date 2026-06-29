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
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(REPO_ROOT / "src"))

from analyze_dino_global_tokens_imagenet100 import (  # noqa: E402
    bootstrap_one_nn_accuracy,
    compute_rgb_hist_batch,
    ensure_dir,
    l2_normalize,
    make_errorbar_barplot,
    repeated_kmeans_metrics,
    run_repeated_hist_regression,
    save_json,
)
from stage1_semantic_retention_dino_utils import (  # noqa: E402
    IndexedImageFolder,
    build_transform,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe learned PatchMAP summary tokens on ImageNet100.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kmeans-repeats", type=int, default=5)
    parser.add_argument("--kmeans-n-init", type=int, default=20)
    parser.add_argument("--one-nn-bootstrap-reps", type=int, default=30)
    parser.add_argument("--probe-splits", type=int, default=5)
    parser.add_argument("--probe-test-size", type=float, default=0.3)
    parser.add_argument("--color-bins", type=int, default=16)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


def load_rae(config_path: Path, ckpt_path: Path, device: torch.device):
    cfg = OmegaConf.load(str(config_path))
    OmegaConf.update(cfg, "stage_1.ckpt", str(ckpt_path), force_add=True)
    rae = instantiate_from_config(cfg.stage_1).to(device).eval()
    rae.requires_grad_(False)
    return rae


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"
    ensure_dir(plots_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_autocast = device.type == "cuda" and args.precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    dataset = IndexedImageFolder(str(args.data_path), transform=build_transform(args.image_size))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    rae = load_rae(args.config, args.ckpt, device=device)

    labels_all: List[int] = []
    image_paths: List[str] = []
    color_hist_all: List[np.ndarray] = []
    patch_mean_all: List[torch.Tensor] = []
    cls_all: List[torch.Tensor] = []
    reg_mean_all: List[torch.Tensor] = []
    summary_all: List[torch.Tensor] = []

    with torch.inference_mode():
        for images, labels, indices in tqdm(loader, desc="extract_patchmap"):
            images = images.to(device, non_blocking=True)
            labels_all.extend(labels.tolist())
            color_hist_all.append(compute_rgb_hist_batch(images.detach().cpu(), bins=int(args.color_bins)))
            for idx in indices.tolist():
                image_paths.append(str(dataset.samples[int(idx)][0]))

            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    patch_tokens, global_tokens = rae._encode_tokens(images, need_aux=True)
                    summary_tokens = rae._get_patch_map_aux_tokens(patch_tokens)
            else:
                patch_tokens, global_tokens = rae._encode_tokens(images, need_aux=True)
                summary_tokens = rae._get_patch_map_aux_tokens(patch_tokens)

            patch_mean_all.append(patch_tokens.mean(dim=1).float().cpu())
            cls_all.append(global_tokens[:, 0].float().cpu())
            reg_mean_all.append(global_tokens[:, 1:].mean(dim=1).float().cpu())
            summary_all.append(summary_tokens.float().cpu())

    labels_np = np.asarray(labels_all, dtype=np.int64)
    color_hist_np = np.concatenate(color_hist_all, axis=0)
    patch_mean_np = torch.cat(patch_mean_all, dim=0).numpy()
    cls_np = torch.cat(cls_all, dim=0).numpy()
    reg_mean_np = torch.cat(reg_mean_all, dim=0).numpy()
    summary_np = torch.cat(summary_all, dim=0).numpy()

    views: Dict[str, np.ndarray] = {
        "patch_mean": patch_mean_np,
        "cls": cls_np,
        "reg_mean": reg_mean_np,
        "summary_mean": summary_np.mean(axis=1),
        "summary_concat": summary_np.reshape(summary_np.shape[0], -1),
    }
    for idx in range(summary_np.shape[1]):
        views[f"summary{idx + 1}"] = summary_np[:, idx]

    ordered_views = ["patch_mean", "cls", "reg_mean", "summary_mean"]
    ordered_views.extend([f"summary{idx + 1}" for idx in range(summary_np.shape[1])])
    ordered_views.append("summary_concat")

    metric_rows = []
    color_rows = []
    for view_name in ordered_views:
        feat = views[view_name]
        feat_norm = l2_normalize(feat.astype(np.float32, copy=False))
        kmeans_row, _assignments, _kmeans = repeated_kmeans_metrics(
            feat_norm,
            labels_np,
            n_clusters=len(dataset.classes),
            base_seed=int(args.seed),
            repeats=int(args.kmeans_repeats),
            n_init=int(args.kmeans_n_init),
        )
        one_nn_stats = bootstrap_one_nn_accuracy(
            feat_norm,
            labels_np,
            reps=int(args.one_nn_bootstrap_reps),
            seed=int(args.seed) + 1000,
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
        color_stats = run_repeated_hist_regression(
            feat_norm,
            color_hist_np,
            labels_np,
            seed=int(args.seed),
            splits=int(args.probe_splits),
            test_size=float(args.probe_test_size),
            alpha=float(args.ridge_alpha),
        )
        color_rows.append(
            {
                "view": view_name,
                "dim": int(feat_norm.shape[1]),
                **color_stats,
            }
        )

    metrics_csv = outdir / "view_metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    color_csv = outdir / "color_hist_metrics.csv"
    with color_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(color_rows[0].keys()))
        writer.writeheader()
        writer.writerows(color_rows)

    display_rows = [row for row in metric_rows if row["view"] in ordered_views]
    display_color_rows = [row for row in color_rows if row["view"] in ordered_views]
    make_errorbar_barplot(
        display_rows,
        ordered_views,
        metric_key="one_nn_mean",
        err_key="one_nn_std",
        out_path=plots_dir / "one_nn.png",
        title="PatchMAP summary views: 1-NN accuracy",
        ylabel="1-NN accuracy",
    )
    make_errorbar_barplot(
        display_rows,
        ordered_views,
        metric_key="kmeans_purity_mean",
        err_key="kmeans_purity_std",
        out_path=plots_dir / "kmeans_purity.png",
        title="PatchMAP summary views: KMeans purity",
        ylabel="purity",
    )
    make_errorbar_barplot(
        display_color_rows,
        ordered_views,
        metric_key="r2_mean",
        err_key="r2_std",
        out_path=plots_dir / "color_hist_r2.png",
        title="PatchMAP summary views: color histogram prediction",
        ylabel=r"$R^2$",
    )

    torch.save(
        {
            "labels": labels_np,
            "class_names": list(dataset.classes),
            "image_paths": image_paths,
            "color_hist": color_hist_np,
            "summary_tokens": summary_np,
            "views": views,
            "metrics": metric_rows,
            "color_metrics": color_rows,
        },
        outdir / "embeddings.pt",
    )

    summary = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "data_path": str(args.data_path),
        "num_samples": int(labels_np.shape[0]),
        "num_classes": int(len(dataset.classes)),
        "ordered_views": ordered_views,
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "color_csv": str(color_csv),
            "embeddings": str(outdir / "embeddings.pt"),
        },
    }
    save_json(summary, outdir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
