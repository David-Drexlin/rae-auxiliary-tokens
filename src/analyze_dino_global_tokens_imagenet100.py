#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, adjusted_rand_score, mean_absolute_error, normalized_mutual_info_score, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from transformers import AutoImageProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(REPO_ROOT / "src"))

from stage1.encoders.dinov2 import Dinov2withNorm  # noqa: E402


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


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


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def mean_std_ci(values: Sequence[float], ci_alpha: float = 0.95) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    lo_q = (1.0 - ci_alpha) / 2.0
    hi_q = 1.0 - lo_q
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "ci_low": float(np.quantile(arr, lo_q)),
        "ci_high": float(np.quantile(arr, hi_q)),
    }


def weighted_cluster_purity(assignments: np.ndarray, labels: np.ndarray, n_clusters: int) -> float:
    total = len(labels)
    correct = 0
    for cluster_id in range(n_clusters):
        mask = assignments == cluster_id
        if not np.any(mask):
            continue
        counts = np.bincount(labels[mask])
        correct += int(counts.max())
    return float(correct) / float(total)


def topk_cluster_classes(assignments: np.ndarray, labels: np.ndarray, class_names: Sequence[str], n_clusters: int, topk: int = 3):
    rows = []
    for cluster_id in range(n_clusters):
        mask = assignments == cluster_id
        size = int(mask.sum())
        if size == 0:
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "size": 0,
                    "purity": 0.0,
                    "top_classes": [],
                }
            )
            continue
        counts = Counter(labels[mask].tolist())
        dominant_label, dominant_count = counts.most_common(1)[0]
        top = [
            {
                "class_idx": int(cls_idx),
                "class_name": class_names[int(cls_idx)],
                "count": int(count),
                "fraction": float(count) / float(size),
            }
            for cls_idx, count in counts.most_common(topk)
        ]
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "size": size,
                "purity": float(dominant_count) / float(size),
                "top_classes": top,
            }
        )
    rows.sort(key=lambda row: (-row["size"], row["cluster_id"]))
    return rows


def compute_one_nn_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    nn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute")
    nn.fit(features)
    indices = nn.kneighbors(return_distance=False)
    neighbors = indices[:, 1]
    return float(np.mean(labels[neighbors] == labels))


def bootstrap_one_nn_accuracy(features: np.ndarray, labels: np.ndarray, reps: int, seed: int) -> Dict[str, float]:
    if reps <= 0:
        value = compute_one_nn_accuracy(features, labels)
        return {"base": value, "mean": value, "std": 0.0, "ci_low": value, "ci_high": value, "reps": 0}
    rng = np.random.default_rng(seed)
    values = []
    n = len(labels)
    for _ in range(reps):
        idx = rng.integers(0, n, size=n, endpoint=False)
        values.append(compute_one_nn_accuracy(features[idx], labels[idx]))
    stats = mean_std_ci(values)
    stats["base"] = compute_one_nn_accuracy(features, labels)
    stats["reps"] = int(reps)
    return stats


def repeated_kmeans_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    base_seed: int,
    repeats: int,
    n_init: int,
) -> Tuple[Dict[str, float], np.ndarray, KMeans]:
    repeats = max(int(repeats), 1)
    purity_values = []
    nmi_values = []
    ari_values = []
    best_assignments = None
    best_kmeans = None
    best_purity = -1.0
    for rep in range(repeats):
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=base_seed + rep)
        assignments = kmeans.fit_predict(features)
        purity = weighted_cluster_purity(assignments, labels, n_clusters)
        nmi = normalized_mutual_info_score(labels, assignments)
        ari = adjusted_rand_score(labels, assignments)
        purity_values.append(purity)
        nmi_values.append(nmi)
        ari_values.append(ari)
        if purity > best_purity:
            best_purity = purity
            best_assignments = assignments
            best_kmeans = kmeans
    assert best_assignments is not None and best_kmeans is not None
    purity_stats = mean_std_ci(purity_values)
    nmi_stats = mean_std_ci(nmi_values)
    ari_stats = mean_std_ci(ari_values)
    row = {
        "kmeans_purity": float(best_purity),
        "kmeans_purity_mean": purity_stats["mean"],
        "kmeans_purity_std": purity_stats["std"],
        "kmeans_purity_ci_low": purity_stats["ci_low"],
        "kmeans_purity_ci_high": purity_stats["ci_high"],
        "nmi": float(normalized_mutual_info_score(labels, best_assignments)),
        "nmi_mean": nmi_stats["mean"],
        "nmi_std": nmi_stats["std"],
        "nmi_ci_low": nmi_stats["ci_low"],
        "nmi_ci_high": nmi_stats["ci_high"],
        "ari": float(adjusted_rand_score(labels, best_assignments)),
        "ari_mean": ari_stats["mean"],
        "ari_std": ari_stats["std"],
        "ari_ci_low": ari_stats["ci_low"],
        "ari_ci_high": ari_stats["ci_high"],
        "kmeans_repeats": int(repeats),
    }
    return row, best_assignments, best_kmeans


def compute_rgb_hist_batch(images: torch.Tensor, bins: int) -> np.ndarray:
    images_np = images.detach().cpu().numpy()
    edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
    out = np.zeros((images_np.shape[0], 3 * bins), dtype=np.float32)
    for i in range(images_np.shape[0]):
        parts = []
        for c in range(3):
            hist, _ = np.histogram(images_np[i, c].reshape(-1), bins=edges)
            hist = hist.astype(np.float32)
            hist /= max(float(hist.sum()), 1.0)
            parts.append(hist)
        out[i] = np.concatenate(parts, axis=0)
    return out


def build_dct_basis(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(size, device=device, dtype=dtype)
    freqs = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
    basis = torch.cos((math.pi / float(size)) * (positions + 0.5) * freqs)
    basis[0] *= math.sqrt(1.0 / float(size))
    if size > 1:
        basis[1:] *= math.sqrt(2.0 / float(size))
    return basis


def compute_lowfreq_dct_batch(images: torch.Tensor, input_size: int, keep: int) -> np.ndarray:
    if keep <= 0 or keep > input_size:
        raise ValueError(f"Expected 0 < keep <= input_size, got keep={keep}, input_size={input_size}.")
    resized = F.interpolate(images, size=(input_size, input_size), mode="bicubic", align_corners=False)
    basis = build_dct_basis(input_size, resized.device, resized.dtype)
    coeffs = torch.einsum("ki,bcij,lj->bckl", basis, resized, basis)
    coeffs = coeffs[:, :, :keep, :keep].contiguous()
    return coeffs.reshape(coeffs.shape[0], -1).detach().cpu().numpy().astype(np.float32, copy=False)


def split_channelwise_dc_ac(targets: np.ndarray, keep: int) -> Tuple[np.ndarray, np.ndarray]:
    coeffs_per_channel = keep * keep
    dc_indices = [channel_idx * coeffs_per_channel for channel_idx in range(3)]
    mask = np.ones(targets.shape[1], dtype=bool)
    mask[dc_indices] = False
    return targets[:, dc_indices], targets[:, mask]


def run_repeated_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    splits: int,
    test_size: float,
    max_iter: int,
) -> Dict[str, float]:
    splitter = StratifiedShuffleSplit(n_splits=splits, test_size=test_size, random_state=seed)
    acc_values = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(features, labels)):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=max_iter,
                solver="lbfgs",
                multi_class="multinomial",
                random_state=seed + split_id,
            ),
        )
        clf.fit(features[train_idx], labels[train_idx])
        pred = clf.predict(features[test_idx])
        acc_values.append(accuracy_score(labels[test_idx], pred))
    stats = mean_std_ci(acc_values)
    stats["splits"] = int(splits)
    stats["test_size"] = float(test_size)
    return stats


def cosine_similarity_mean(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), eps)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), eps)
    return float(np.mean(np.sum(a_norm * b_norm, axis=1)))


def run_repeated_target_regression(
    features: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    seed: int,
    splits: int,
    test_size: float,
    alpha: float,
) -> Dict[str, float]:
    splitter = StratifiedShuffleSplit(n_splits=splits, test_size=test_size, random_state=seed)
    r2_values = []
    mae_values = []
    cos_values = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(features, labels)):
        reg = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha, random_state=seed + split_id),
        )
        reg.fit(features[train_idx], targets[train_idx])
        pred = reg.predict(features[test_idx])
        true = targets[test_idx]
        r2_values.append(r2_score(true, pred, multioutput="variance_weighted"))
        mae_values.append(mean_absolute_error(true, pred))
        cos_values.append(cosine_similarity_mean(true, pred))
    r2_stats = mean_std_ci(r2_values)
    mae_stats = mean_std_ci(mae_values)
    cos_stats = mean_std_ci(cos_values)
    return {
        "r2_mean": r2_stats["mean"],
        "r2_std": r2_stats["std"],
        "r2_ci_low": r2_stats["ci_low"],
        "r2_ci_high": r2_stats["ci_high"],
        "mae_mean": mae_stats["mean"],
        "mae_std": mae_stats["std"],
        "mae_ci_low": mae_stats["ci_low"],
        "mae_ci_high": mae_stats["ci_high"],
        "cosine_mean": cos_stats["mean"],
        "cosine_std": cos_stats["std"],
        "cosine_ci_low": cos_stats["ci_low"],
        "cosine_ci_high": cos_stats["ci_high"],
        "splits": int(splits),
        "test_size": float(test_size),
    }


def run_repeated_hist_regression(
    features: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    seed: int,
    splits: int,
    test_size: float,
    alpha: float,
) -> Dict[str, float]:
    return run_repeated_target_regression(
        features=features,
        targets=targets,
        labels=labels,
        seed=seed,
        splits=splits,
        test_size=test_size,
        alpha=alpha,
    )


def make_scatter(points: np.ndarray, colors: np.ndarray, out_path: Path, title: str, cmap_name: str = "hsv") -> None:
    ensure_dir(out_path.parent)
    plt.figure(figsize=(10, 8))
    plt.scatter(points[:, 0], points[:, 1], c=colors, cmap=cmap_name, s=6, alpha=0.8, linewidths=0)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_errorbar_barplot(
    rows: Sequence[Dict[str, float]],
    names: Sequence[str],
    metric_key: str,
    err_key: str,
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    ensure_dir(out_path.parent)
    values = [float(row[metric_key]) for row in rows]
    errs = [float(row[err_key]) for row in rows]
    x = np.arange(len(names))
    plt.figure(figsize=(12, 4.8))
    plt.bar(x, values, yerr=errs, capsize=4, color="#4C72B0")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def prefixed_stats(stats: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def maybe_umap(features: np.ndarray, seed: int, n_neighbors: int, min_dist: float):
    try:
        import umap.umap_ as umap
    except Exception:
        return None

    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        transform_seed=seed,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric="cosine",
    )
    return reducer.fit_transform(features).astype(np.float32)


def pca_2d(features: np.ndarray, seed: int) -> np.ndarray:
    reducer = PCA(n_components=2, svd_solver="randomized", random_state=seed)
    return reducer.fit_transform(features).astype(np.float32)


def image_from_path(path: str, vis_size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = center_crop_arr(img, vis_size)
    return img


def draw_label(img: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + 18), color=(255, 255, 255))
    canvas.paste(img, (0, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((2, 2), text, fill=(0, 0, 0))
    return canvas


def build_nn_sheet(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    image_paths: Sequence[str],
    query_indices: Sequence[int],
    out_path: Path,
    title: str,
    vis_size: int,
    neighbors_k: int,
) -> None:
    ensure_dir(out_path.parent)
    nn = NearestNeighbors(n_neighbors=neighbors_k + 1, metric="cosine", algorithm="brute")
    nn.fit(features)
    indices = nn.kneighbors(features[np.asarray(query_indices)], return_distance=False)

    row_gap = 8
    col_gap = 6
    cell_w = vis_size
    cell_h = vis_size + 18
    rows = len(query_indices)
    cols = neighbors_k + 1
    canvas = Image.new(
        "RGB",
        (cols * cell_w + (cols - 1) * col_gap, rows * cell_h + (rows - 1) * row_gap + 28),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill=(0, 0, 0))

    for row_idx, query_idx in enumerate(query_indices):
        query_label = class_names[int(labels[query_idx])]
        row_indices = [int(query_idx)] + [int(i) for i in indices[row_idx, 1:]]
        for col_idx, sample_idx in enumerate(row_indices):
            img = image_from_path(image_paths[sample_idx], vis_size)
            cls_name = class_names[int(labels[sample_idx])]
            prefix = "Q" if col_idx == 0 else f"N{col_idx}"
            panel = draw_label(img, f"{prefix}: {cls_name}")
            x = col_idx * (cell_w + col_gap)
            y = 28 + row_idx * (cell_h + row_gap)
            if col_idx == 0:
                panel = ImageOps.expand(panel, border=2, fill=(220, 20, 60))
            canvas.paste(panel, (x, y))
        draw.text((4, 28 + row_idx * (cell_h + row_gap) + cell_h - 14), f"query class: {query_label}", fill=(0, 0, 0))

    canvas.save(out_path)


def build_cluster_sheet(
    features: np.ndarray,
    assignments: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    image_paths: Sequence[str],
    cluster_rows: Sequence[Dict],
    out_path: Path,
    title: str,
    vis_size: int,
    exemplars_per_cluster: int,
) -> None:
    ensure_dir(out_path.parent)
    top_clusters = [row for row in cluster_rows if row["size"] > 0][:6]
    row_gap = 10
    col_gap = 6
    cell_w = vis_size
    cell_h = vis_size + 18
    cols = exemplars_per_cluster
    header_h = 32
    canvas = Image.new(
        "RGB",
        (cols * cell_w + (cols - 1) * col_gap, len(top_clusters) * (cell_h + header_h + row_gap) + 26),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill=(0, 0, 0))

    for row_idx, row in enumerate(top_clusters):
        cluster_id = int(row["cluster_id"])
        member_idx = np.flatnonzero(assignments == cluster_id)
        if member_idx.size == 0:
            continue
        feats = features[member_idx]
        centroid = centroids[cluster_id : cluster_id + 1]
        dists = ((feats - centroid) ** 2).sum(axis=1)
        exemplar_local = np.argsort(dists)[:exemplars_per_cluster]
        exemplar_idx = member_idx[exemplar_local]

        header_y = 26 + row_idx * (header_h + cell_h + row_gap)
        top_classes = ", ".join(
            f"{entry['class_name']}:{entry['count']}" for entry in row["top_classes"][:3]
        )
        header = f"cluster {cluster_id} | n={row['size']} | purity={row['purity']:.3f} | {top_classes}"
        draw.text((4, header_y), header, fill=(0, 0, 0))

        for col_idx, sample_idx in enumerate(exemplar_idx.tolist()):
            img = image_from_path(image_paths[sample_idx], vis_size)
            cls_name = class_names[int(labels[sample_idx])]
            panel = draw_label(img, cls_name)
            x = col_idx * (cell_w + col_gap)
            y = header_y + header_h
            canvas.paste(panel, (x, y))

    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--encoder-path", type=str, default="facebook/dinov2-with-registers-base")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--encoder-input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--vis-size", type=int, default=96)
    parser.add_argument("--nn-queries", type=int, default=10)
    parser.add_argument("--nn-k", type=int, default=6)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--kmeans-repeats", type=int, default=5)
    parser.add_argument("--kmeans-n-init", type=int, default=20)
    parser.add_argument("--one-nn-bootstrap-reps", type=int, default=50)
    parser.add_argument("--probe-splits", type=int, default=5)
    parser.add_argument("--probe-test-size", type=float, default=0.3)
    parser.add_argument("--probe-max-iter", type=int, default=2000)
    parser.add_argument("--color-bins", type=int, default=16)
    parser.add_argument("--dct-input-size", type=int, default=32)
    parser.add_argument("--dct-keep", type=int, default=8)
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
        indices = np.arange(min(args.max_samples, len(dataset)), dtype=np.int64)
        dataset.samples = [dataset.samples[i] for i in indices.tolist()]
        dataset.targets = [dataset.targets[i] for i in indices.tolist()]
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

    encoder = Dinov2withNorm(dinov2_path=args.encoder_path, normalize=True).to(device).eval()

    labels_all: List[int] = []
    cls_all: List[torch.Tensor] = []
    regs_all: List[torch.Tensor] = []
    patch_mean_all: List[torch.Tensor] = []
    color_hist_all: List[np.ndarray] = []
    lowfreq_dct_all: List[np.ndarray] = []

    with torch.inference_mode():
        for images, labels, _indices in tqdm(loader, desc="extract"):
            images = images.to(device, non_blocking=True)
            labels_all.extend(labels.tolist())
            color_hist_all.append(compute_rgb_hist_batch(images, bins=args.color_bins))
            lowfreq_dct_all.append(compute_lowfreq_dct_batch(images, input_size=args.dct_input_size, keep=args.dct_keep))
            if images.shape[-1] != args.encoder_input_size or images.shape[-2] != args.encoder_input_size:
                images = torch.nn.functional.interpolate(
                    images,
                    size=(args.encoder_input_size, args.encoder_input_size),
                    mode="bicubic",
                    align_corners=False,
                )
            images = (images - encoder_mean) / encoder_std

            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    patch_tokens, global_tokens = encoder.forward_with_global(images)
            else:
                patch_tokens, global_tokens = encoder.forward_with_global(images)

            patch_mean_all.append(patch_tokens.mean(dim=1).float().cpu())
            cls_all.append(global_tokens[:, 0].float().cpu())
            regs_all.append(global_tokens[:, 1:].float().cpu())

    labels_np = np.asarray(labels_all, dtype=np.int64)
    patch_mean_np = torch.cat(patch_mean_all, dim=0).numpy()
    cls_np = torch.cat(cls_all, dim=0).numpy()
    regs_np = torch.cat(regs_all, dim=0).numpy()
    reg_mean_np = regs_np.mean(axis=1)
    reg_concat_np = regs_np.reshape(regs_np.shape[0], -1)
    color_hist_np = np.concatenate(color_hist_all, axis=0)
    lowfreq_dct_np = np.concatenate(lowfreq_dct_all, axis=0)
    lowfreq_dct_dc_np, lowfreq_dct_ac_np = split_channelwise_dc_ac(lowfreq_dct_np, keep=args.dct_keep)

    views: Dict[str, np.ndarray] = {
        "cls": cls_np,
        "patch_mean": patch_mean_np,
        "reg_mean": reg_mean_np,
        "reg_concat": reg_concat_np,
        "reg1": regs_np[:, 0],
        "reg2": regs_np[:, 1],
        "reg3": regs_np[:, 2],
        "reg4": regs_np[:, 3],
    }

    torch.save(
        {
            "labels": labels_np,
            "class_names": class_names,
            "image_paths": image_paths,
            "patch_mean": patch_mean_np,
            "cls": cls_np,
            "regs": regs_np,
            "reg_mean": reg_mean_np,
            "reg_concat": reg_concat_np,
            "color_hist": color_hist_np,
            "lowfreq_dct": lowfreq_dct_np,
        },
        outdir / "embeddings.pt",
    )

    metric_rows = []
    probe_rows = []
    color_rows = []
    dct_rows = []
    cluster_tables = {}
    pca_plots = {}
    umap_plots = {}
    query_indices = np.random.default_rng(args.seed).choice(len(labels_np), size=min(args.nn_queries, len(labels_np)), replace=False)
    plot_views = {"cls", "patch_mean", "reg_mean", "reg1", "reg2", "reg3", "reg4"}

    for view_name, feat in views.items():
        feat_norm = l2_normalize(feat.astype(np.float32, copy=False))
        kmeans_row, assignments, kmeans = repeated_kmeans_metrics(
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

        row = {
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
        metric_rows.append(row)

        probe_stats = run_repeated_linear_probe(
            feat_norm,
            labels_np,
            seed=args.seed,
            splits=args.probe_splits,
            test_size=args.probe_test_size,
            max_iter=args.probe_max_iter,
        )
        probe_rows.append(
            {
                "view": view_name,
                "dim": int(feat_norm.shape[1]),
                "acc_mean": float(probe_stats["mean"]),
                "acc_std": float(probe_stats["std"]),
                "acc_ci_low": float(probe_stats["ci_low"]),
                "acc_ci_high": float(probe_stats["ci_high"]),
                "splits": int(probe_stats["splits"]),
                "test_size": float(probe_stats["test_size"]),
            }
        )

        color_stats = run_repeated_hist_regression(
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

        cluster_rows = topk_cluster_classes(assignments, labels_np, class_names, len(class_names), topk=3)
        cluster_tables[view_name] = cluster_rows
        with open(outdir / f"cluster_summary_{view_name}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cluster_id", "size", "purity", "top1", "top2", "top3"])
            for cluster_row in cluster_rows:
                top = cluster_row["top_classes"]
                cells = []
                for idx in range(3):
                    if idx < len(top):
                        entry = top[idx]
                        cells.append(f"{entry['class_name']}:{entry['count']}")
                    else:
                        cells.append("")
                writer.writerow([cluster_row["cluster_id"], cluster_row["size"], cluster_row["purity"], *cells])

        if view_name in plot_views:
            pca_points = pca_2d(feat_norm, seed=args.seed)
            pca_plots[view_name] = str(plots_dir / f"{view_name}_pca_by_class.png")
            make_scatter(pca_points, labels_np, plots_dir / f"{view_name}_pca_by_class.png", f"{view_name} PCA colored by class")
            make_scatter(pca_points, assignments, plots_dir / f"{view_name}_pca_by_cluster.png", f"{view_name} PCA colored by kmeans cluster", cmap_name="tab20")

            if view_name in {"cls", "patch_mean", "reg_mean"}:
                umap_points = maybe_umap(feat_norm, seed=args.seed, n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist)
            else:
                umap_points = None
            if umap_points is not None:
                umap_plots[view_name] = str(plots_dir / f"{view_name}_umap_by_class.png")
                make_scatter(umap_points, labels_np, plots_dir / f"{view_name}_umap_by_class.png", f"{view_name} UMAP colored by class")
                make_scatter(umap_points, assignments, plots_dir / f"{view_name}_umap_by_cluster.png", f"{view_name} UMAP colored by kmeans cluster", cmap_name="tab20")

            build_nn_sheet(
                feat_norm,
                labels_np,
                class_names,
                image_paths,
                query_indices,
                plots_dir / f"{view_name}_nearest_neighbors.png",
                title=f"{view_name}: nearest neighbors",
                vis_size=args.vis_size,
                neighbors_k=args.nn_k,
            )
            build_cluster_sheet(
                feat_norm,
                assignments,
                kmeans.cluster_centers_,
                labels_np,
                class_names,
                image_paths,
                cluster_rows,
                plots_dir / f"{view_name}_cluster_exemplars.png",
                title=f"{view_name}: largest KMeans clusters",
                vis_size=args.vis_size,
                exemplars_per_cluster=args.nn_k + 1,
            )

    metric_rows.sort(key=lambda row: row["view"])
    with open(outdir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    probe_rows.sort(key=lambda row: row["view"])
    with open(outdir / "linear_probe_classification.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(probe_rows[0].keys()))
        writer.writeheader()
        writer.writerows(probe_rows)

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

    ordered_views = ["cls", "patch_mean", "reg1", "reg2", "reg3", "reg4", "reg_mean", "reg_concat"]
    probe_plot_rows = [row for name in ordered_views for row in probe_rows if row["view"] == name]
    color_plot_rows = [row for name in ordered_views for row in color_rows if row["view"] == name]
    dct_plot_rows = [row for name in ordered_views for row in dct_rows if row["view"] == name]
    nn_plot_rows = [row for name in ordered_views for row in metric_rows if row["view"] == name]
    names = [row["view"] for row in probe_plot_rows]
    make_errorbar_barplot(
        probe_plot_rows,
        names,
        metric_key="acc_mean",
        err_key="acc_std",
        out_path=plots_dir / "linear_probe_classification_accuracy.png",
        title="Linear probe accuracy by token view",
        ylabel="Top-1 accuracy",
    )
    make_errorbar_barplot(
        color_plot_rows,
        names,
        metric_key="r2_mean",
        err_key="r2_std",
        out_path=plots_dir / "linear_probe_color_hist_r2.png",
        title="Color histogram regression by token view",
        ylabel="R^2",
    )
    make_errorbar_barplot(
        dct_plot_rows,
        names,
        metric_key="r2_mean",
        err_key="r2_std",
        out_path=plots_dir / "linear_probe_lowfreq_dct_r2.png",
        title="Low-frequency DCT regression by token view",
        ylabel="R^2",
    )
    make_errorbar_barplot(
        dct_plot_rows,
        names,
        metric_key="ac_r2_mean",
        err_key="ac_r2_std",
        out_path=plots_dir / "linear_probe_lowfreq_dct_ac_r2.png",
        title="Low-frequency DCT AC regression by token view",
        ylabel="R^2",
    )
    make_errorbar_barplot(
        dct_plot_rows,
        names,
        metric_key="dc_r2_mean",
        err_key="dc_r2_std",
        out_path=plots_dir / "linear_probe_lowfreq_dct_dc_r2.png",
        title="Low-frequency DCT DC regression by token view",
        ylabel="R^2",
    )
    make_errorbar_barplot(
        nn_plot_rows,
        [row["view"] for row in nn_plot_rows],
        metric_key="one_nn_mean",
        err_key="one_nn_std",
        out_path=plots_dir / "one_nn_accuracy_bootstrap.png",
        title="1-NN accuracy with bootstrap spread",
        ylabel="1-NN accuracy",
    )

    summary = {
        "data_path": args.data_path,
        "num_samples": len(labels_np),
        "num_classes": len(class_names),
        "class_names": class_names,
        "encoder_path": args.encoder_path,
        "image_size": args.image_size,
        "encoder_input_size": args.encoder_input_size,
        "metrics": metric_rows,
        "linear_probe_classification": probe_rows,
        "linear_probe_color_hist": color_rows,
        "linear_probe_lowfreq_dct": dct_rows,
        "best_view_by_lowfreq_dct_r2": max(dct_rows, key=lambda row: row["r2_mean"]),
        "best_view_by_lowfreq_dct_ac_r2": max(dct_rows, key=lambda row: row["ac_r2_mean"]),
        "plots": {
            "pca": pca_plots,
            "umap": umap_plots,
            "nn_queries": query_indices.tolist(),
        },
    }
    save_json(summary, outdir / "summary.json")


if __name__ == "__main__":
    main()
