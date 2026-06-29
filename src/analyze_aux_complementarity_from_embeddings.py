#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

from analyze_dino_global_tokens_imagenet100 import (  # noqa: E402
    center_crop_arr,
    compute_rgb_hist_batch,
    compute_lowfreq_dct_batch,
    ensure_dir,
    l2_normalize,
    make_errorbar_barplot,
    mean_std_ci,
)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def cosine_similarity_mean(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), eps)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), eps)
    return float(np.mean(np.sum(a_norm * b_norm, axis=1)))


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.ToTensor(),
        ]
    )


def compute_thumbnail_targets(image_paths: Sequence[str], image_size: int, thumbnail_size: int) -> np.ndarray:
    transform = build_transform(image_size)
    rows: List[np.ndarray] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = transform(image).unsqueeze(0)
        thumbs = F.interpolate(tensor, size=(thumbnail_size, thumbnail_size), mode="bilinear", align_corners=False)
        rows.append(thumbs.flatten(start_dim=1).squeeze(0).numpy().astype(np.float32, copy=False))
    return np.stack(rows, axis=0)


def compute_color_hist_targets(image_paths: Sequence[str], image_size: int, color_bins: int) -> np.ndarray:
    transform = build_transform(image_size)
    rows: List[np.ndarray] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = transform(image).unsqueeze(0)
        rows.append(compute_rgb_hist_batch(tensor, bins=color_bins)[0].astype(np.float32, copy=False))
    return np.stack(rows, axis=0)


def compute_lowfreq_dct_targets(image_paths: Sequence[str], image_size: int, dct_input_size: int, dct_keep: int) -> np.ndarray:
    transform = build_transform(image_size)
    rows: List[np.ndarray] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = transform(image).unsqueeze(0)
        rows.append(
            compute_lowfreq_dct_batch(
                tensor,
                input_size=dct_input_size,
                keep=dct_keep,
            )[0].astype(np.float32, copy=False)
        )
    return np.stack(rows, axis=0)


def split_channelwise_dc_ac(targets: np.ndarray, keep: int) -> Tuple[np.ndarray, np.ndarray]:
    coeffs_per_channel = keep * keep
    dc_indices = [channel_idx * coeffs_per_channel for channel_idx in range(3)]
    mask = np.ones(targets.shape[1], dtype=bool)
    mask[dc_indices] = False
    return targets[:, dc_indices], targets[:, mask]


def build_feature_specs(views: Dict[str, np.ndarray], encoder_kind: str) -> List[Dict[str, object]]:
    normalized_views = {
        key: l2_normalize(np.asarray(value, dtype=np.float32))
        for key, value in views.items()
    }
    patch = normalized_views["patch_mean"]

    def concat(*parts: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(part, dtype=np.float32) for part in parts], axis=1)

    if encoder_kind == "dino":
        cls = normalized_views["cls"]
        reg4 = normalized_views["reg4"]
        reg_concat = normalized_views["reg_concat"]
        cls_reg_concat = concat(cls, reg_concat)
        return [
            {"name": "patch_mean", "features": patch, "contains_patch": True, "aux_source": None},
            {"name": "cls", "features": cls, "contains_patch": False, "aux_source": "cls"},
            {"name": "reg4", "features": reg4, "contains_patch": False, "aux_source": "reg4"},
            {"name": "reg_concat", "features": reg_concat, "contains_patch": False, "aux_source": "reg_concat"},
            {"name": "cls_reg_concat", "features": cls_reg_concat, "contains_patch": False, "aux_source": "cls_reg_concat"},
            {
                "name": "patch_plus_cls",
                "features": concat(patch, cls),
                "contains_patch": True,
                "aux_source": "cls",
                "patch_block": patch,
                "aux_block": cls,
            },
            {
                "name": "patch_plus_reg4",
                "features": concat(patch, reg4),
                "contains_patch": True,
                "aux_source": "reg4",
                "patch_block": patch,
                "aux_block": reg4,
            },
            {
                "name": "patch_plus_reg_concat",
                "features": concat(patch, reg_concat),
                "contains_patch": True,
                "aux_source": "reg_concat",
                "patch_block": patch,
                "aux_block": reg_concat,
            },
            {
                "name": "patch_plus_cls_reg_concat",
                "features": concat(patch, cls_reg_concat),
                "contains_patch": True,
                "aux_source": "cls_reg_concat",
                "patch_block": patch,
                "aux_block": cls_reg_concat,
            },
        ]
    if encoder_kind == "mae":
        cls = normalized_views["cls"]
        return [
            {"name": "patch_mean", "features": patch, "contains_patch": True, "aux_source": None},
            {"name": "cls", "features": cls, "contains_patch": False, "aux_source": "cls"},
            {
                "name": "patch_plus_cls",
                "features": concat(patch, cls),
                "contains_patch": True,
                "aux_source": "cls",
                "patch_block": patch,
                "aux_block": cls,
            },
        ]
    if encoder_kind == "siglip2":
        pooler = normalized_views["pooler"]
        return [
            {"name": "patch_mean", "features": patch, "contains_patch": True, "aux_source": None},
            {"name": "pooler", "features": pooler, "contains_patch": False, "aux_source": "pooler"},
            {
                "name": "patch_plus_pooler",
                "features": concat(patch, pooler),
                "contains_patch": True,
                "aux_source": "pooler",
                "patch_block": patch,
                "aux_block": pooler,
            },
        ]
    raise ValueError(f"Unsupported encoder kind '{encoder_kind}'.")


def parse_alpha_grid(spec: str) -> List[float]:
    parts = [part for part in re.split(r"[\s,;:]+", spec.strip()) if part]
    return [float(part) for part in parts]


def fit_ridge_pipeline(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    alphas: Sequence[float],
    cv_folds: int,
):
    reg = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.asarray(alphas, dtype=np.float64), cv=cv_folds),
    )
    reg.fit(train_x, train_y)
    return reg


def evaluate_feature_set(
    features: np.ndarray,
    targets: np.ndarray,
    split_indices: Sequence[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    alphas: Sequence[float],
    cv_folds: int,
) -> Dict[str, object]:
    r2_values = []
    mae_values = []
    cos_values = []
    alpha_values = []
    for split_id, (train_idx, test_idx) in enumerate(split_indices):
        reg = fit_ridge_pipeline(
            features[train_idx],
            targets[train_idx],
            alphas=alphas,
            cv_folds=cv_folds,
        )
        pred = reg.predict(features[test_idx])
        true = targets[test_idx]
        alpha_values.append(float(reg.named_steps["ridgecv"].alpha_))
        r2_values.append(float(r2_score(true, pred, multioutput="variance_weighted")))
        mae_values.append(float(mean_absolute_error(true, pred)))
        cos_values.append(float(cosine_similarity_mean(true, pred)))
    r2_stats = mean_std_ci(r2_values)
    mae_stats = mean_std_ci(mae_values)
    cos_stats = mean_std_ci(cos_values)
    return {
        "r2_values": r2_values,
        "mae_values": mae_values,
        "cosine_values": cos_values,
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
        "alpha_values": alpha_values,
        "alpha_mean": float(np.mean(alpha_values)),
        "alpha_std": float(np.std(alpha_values, ddof=0)),
    }


def evaluate_residualized_pair(
    patch_features: np.ndarray,
    aux_features: np.ndarray,
    targets: np.ndarray,
    split_indices: Sequence[Tuple[np.ndarray, np.ndarray]],
    alphas: Sequence[float],
    cv_folds: int,
) -> Dict[str, Dict[str, object]]:
    residual_only_r2 = []
    residual_only_mae = []
    residual_only_cos = []
    residual_only_alpha = []

    patch_plus_resid_r2 = []
    patch_plus_resid_mae = []
    patch_plus_resid_cos = []
    patch_plus_resid_alpha = []

    residualizer_alpha = []

    for train_idx, test_idx in split_indices:
        patch_train = patch_features[train_idx]
        patch_test = patch_features[test_idx]
        aux_train = aux_features[train_idx]
        aux_test = aux_features[test_idx]
        target_train = targets[train_idx]
        target_test = targets[test_idx]

        residualizer = fit_ridge_pipeline(
            patch_train,
            aux_train,
            alphas=alphas,
            cv_folds=cv_folds,
        )
        residualizer_alpha.append(float(residualizer.named_steps["ridgecv"].alpha_))
        aux_train_resid = aux_train - residualizer.predict(patch_train)
        aux_test_resid = aux_test - residualizer.predict(patch_test)

        resid_model = fit_ridge_pipeline(
            aux_train_resid,
            target_train,
            alphas=alphas,
            cv_folds=cv_folds,
        )
        resid_pred = resid_model.predict(aux_test_resid)
        residual_only_alpha.append(float(resid_model.named_steps["ridgecv"].alpha_))
        residual_only_r2.append(float(r2_score(target_test, resid_pred, multioutput="variance_weighted")))
        residual_only_mae.append(float(mean_absolute_error(target_test, resid_pred)))
        residual_only_cos.append(float(cosine_similarity_mean(target_test, resid_pred)))

        joint_train = np.concatenate([patch_train, aux_train_resid], axis=1)
        joint_test = np.concatenate([patch_test, aux_test_resid], axis=1)
        patch_plus_resid_model = fit_ridge_pipeline(
            joint_train,
            target_train,
            alphas=alphas,
            cv_folds=cv_folds,
        )
        patch_plus_resid_pred = patch_plus_resid_model.predict(joint_test)
        patch_plus_resid_alpha.append(float(patch_plus_resid_model.named_steps["ridgecv"].alpha_))
        patch_plus_resid_r2.append(float(r2_score(target_test, patch_plus_resid_pred, multioutput="variance_weighted")))
        patch_plus_resid_mae.append(float(mean_absolute_error(target_test, patch_plus_resid_pred)))
        patch_plus_resid_cos.append(float(cosine_similarity_mean(target_test, patch_plus_resid_pred)))

    def summarize(r2_values, mae_values, cos_values, alpha_values, residualizer_values=None):
        r2_stats = mean_std_ci(r2_values)
        mae_stats = mean_std_ci(mae_values)
        cos_stats = mean_std_ci(cos_values)
        out = {
            "r2_values": r2_values,
            "mae_values": mae_values,
            "cosine_values": cos_values,
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
            "alpha_values": alpha_values,
            "alpha_mean": float(np.mean(alpha_values)),
            "alpha_std": float(np.std(alpha_values, ddof=0)),
        }
        if residualizer_values is not None:
            out["residualizer_alpha_values"] = residualizer_values
            out["residualizer_alpha_mean"] = float(np.mean(residualizer_values))
            out["residualizer_alpha_std"] = float(np.std(residualizer_values, ddof=0))
        return out

    return {
        "aux_resid": summarize(
            residual_only_r2,
            residual_only_mae,
            residual_only_cos,
            residual_only_alpha,
            residualizer_values=residualizer_alpha,
        ),
        "patch_plus_aux_resid": summarize(
            patch_plus_resid_r2,
            patch_plus_resid_mae,
            patch_plus_resid_cos,
            patch_plus_resid_alpha,
            residualizer_values=residualizer_alpha,
        ),
    }


def add_delta_stats(values: Sequence[float], prefix: str) -> Dict[str, float]:
    stats = mean_std_ci(values)
    return {
        f"{prefix}_mean": stats["mean"],
        f"{prefix}_std": stats["std"],
        f"{prefix}_ci_low": stats["ci_low"],
        f"{prefix}_ci_high": stats["ci_high"],
    }


def ordered_fieldnames(rows: Sequence[Dict[str, object]]) -> List[str]:
    seen = set()
    names: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-splits", type=int, default=5)
    parser.add_argument("--probe-test-size", type=float, default=0.3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-alphas", type=str, default="0.001,0.01,0.1,1,10,100,1000,10000,100000")
    parser.add_argument("--ridge-cv-folds", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--thumbnail-size", type=int, default=8)
    parser.add_argument("--color-bins", type=int, default=16)
    parser.add_argument("--dct-input-size", type=int, default=32)
    parser.add_argument("--dct-keep", type=int, default=8)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"
    ensure_dir(plots_dir)

    payload = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    labels_np = np.asarray(payload["labels"], dtype=np.int64)
    views = {key: np.asarray(value, dtype=np.float32) for key, value in payload["views"].items()}
    image_paths = payload["image_paths"]
    encoder_kind = str(payload["encoder_kind"])

    color_hist_np = payload.get("color_hist")
    if color_hist_np is None:
        color_hist_np = compute_color_hist_targets(image_paths, image_size=args.image_size, color_bins=args.color_bins)
    else:
        color_hist_np = np.asarray(color_hist_np, dtype=np.float32)

    lowfreq_dct_np = payload.get("lowfreq_dct")
    if lowfreq_dct_np is None:
        lowfreq_dct_np = compute_lowfreq_dct_targets(
            image_paths,
            image_size=args.image_size,
            dct_input_size=args.dct_input_size,
            dct_keep=args.dct_keep,
        )
    else:
        lowfreq_dct_np = np.asarray(lowfreq_dct_np, dtype=np.float32)
    lowfreq_dct_dc_np, lowfreq_dct_ac_np = split_channelwise_dc_ac(lowfreq_dct_np, keep=args.dct_keep)

    thumbnail_np = compute_thumbnail_targets(
        image_paths,
        image_size=args.image_size,
        thumbnail_size=args.thumbnail_size,
    )

    feature_specs = build_feature_specs(views, encoder_kind=encoder_kind)
    alpha_grid = parse_alpha_grid(args.ridge_alphas)

    split_indices = list(
        StratifiedShuffleSplit(
            n_splits=args.probe_splits,
            test_size=args.probe_test_size,
            random_state=args.seed,
        ).split(np.zeros(len(labels_np), dtype=np.int64), labels_np)
    )

    targets = {
        "color_hist": color_hist_np,
        "lowfreq_dct": lowfreq_dct_np,
        "lowfreq_dct_dc": lowfreq_dct_dc_np,
        "lowfreq_dct_ac": lowfreq_dct_ac_np,
        "thumbnail": thumbnail_np,
    }

    all_rows: List[Dict[str, object]] = []
    summary: Dict[str, object] = {
        "encoder_kind": encoder_kind,
        "embeddings": args.embeddings,
        "targets": {},
        "feature_sets": [spec["name"] for spec in feature_specs],
        "probe_splits": args.probe_splits,
        "probe_test_size": args.probe_test_size,
        "ridge_alpha": args.ridge_alpha,
        "ridge_alphas": alpha_grid,
        "ridge_cv_folds": args.ridge_cv_folds,
    }

    for target_name, target_values in targets.items():
        eval_by_name: Dict[str, Dict[str, object]] = {}
        for spec in feature_specs:
            eval_by_name[spec["name"]] = evaluate_feature_set(
                np.asarray(spec["features"], dtype=np.float32),
                target_values,
                split_indices=split_indices,
                seed=args.seed,
                alphas=alpha_grid,
                cv_folds=args.ridge_cv_folds,
            )

        patch_eval = eval_by_name["patch_mean"]
        target_rows: List[Dict[str, object]] = []
        residual_pair_cache: Dict[str, Dict[str, Dict[str, object]]] = {}
        for spec in feature_specs:
            name = str(spec["name"])
            result = eval_by_name[name]
            row: Dict[str, object] = {
                "target": target_name,
                "feature_set": name,
                "dim": int(np.asarray(spec["features"]).shape[1]),
                "contains_patch": bool(spec["contains_patch"]),
                "aux_source": spec["aux_source"] if spec["aux_source"] is not None else "",
                "r2_mean": float(result["r2_mean"]),
                "r2_std": float(result["r2_std"]),
                "r2_ci_low": float(result["r2_ci_low"]),
                "r2_ci_high": float(result["r2_ci_high"]),
                "mae_mean": float(result["mae_mean"]),
                "mae_std": float(result["mae_std"]),
                "mae_ci_low": float(result["mae_ci_low"]),
                "mae_ci_high": float(result["mae_ci_high"]),
                "cosine_mean": float(result["cosine_mean"]),
                "cosine_std": float(result["cosine_std"]),
                "cosine_ci_low": float(result["cosine_ci_low"]),
                "cosine_ci_high": float(result["cosine_ci_high"]),
                "alpha_mean": float(result["alpha_mean"]),
                "alpha_std": float(result["alpha_std"]),
                "splits": int(args.probe_splits),
                "test_size": float(args.probe_test_size),
            }

            delta_vs_patch = np.asarray(result["r2_values"]) - np.asarray(patch_eval["r2_values"])
            row.update(add_delta_stats(delta_vs_patch.tolist(), "delta_vs_patch_r2"))

            aux_source = spec["aux_source"]
            if bool(spec["contains_patch"]) and aux_source:
                aux_eval = eval_by_name[str(aux_source)]
                unique_aux = np.asarray(result["r2_values"]) - np.asarray(patch_eval["r2_values"])
                unique_patch = np.asarray(result["r2_values"]) - np.asarray(aux_eval["r2_values"])
                shared = np.asarray(patch_eval["r2_values"]) + np.asarray(aux_eval["r2_values"]) - np.asarray(result["r2_values"])
                conditional_gain_fraction = unique_aux / np.maximum(1e-12, 1.0 - np.asarray(patch_eval["r2_values"]))
                row.update(add_delta_stats(unique_aux.tolist(), "unique_aux_r2"))
                row.update(add_delta_stats(unique_patch.tolist(), "unique_patch_r2"))
                row.update(add_delta_stats(shared.tolist(), "shared_r2"))
                row.update(add_delta_stats(conditional_gain_fraction.tolist(), "conditional_gain_fraction"))
                row["base_feature_dim"] = int(np.asarray(spec["patch_block"]).shape[1])
                row["added_feature_dim"] = int(np.asarray(spec["aux_block"]).shape[1])

                if str(aux_source) not in residual_pair_cache:
                    residual_pair_cache[str(aux_source)] = evaluate_residualized_pair(
                        np.asarray(spec["patch_block"], dtype=np.float32),
                        np.asarray(spec["aux_block"], dtype=np.float32),
                        target_values,
                        split_indices=split_indices,
                        alphas=alpha_grid,
                        cv_folds=args.ridge_cv_folds,
                    )
            target_rows.append(row)
            all_rows.append(row)

        for aux_source, residual_bundle in residual_pair_cache.items():
            aux_dim = None
            for spec in feature_specs:
                if spec["aux_source"] == aux_source and bool(spec.get("contains_patch")):
                    aux_dim = int(np.asarray(spec["aux_block"]).shape[1])
                    break
            if aux_dim is None:
                continue

            aux_only_result = residual_bundle["aux_resid"]
            aux_only_row: Dict[str, object] = {
                "target": target_name,
                "feature_set": f"{aux_source}_resid_from_patch",
                "dim": aux_dim,
                "contains_patch": False,
                "aux_source": aux_source,
                "residualized_against": "patch_mean",
                "r2_mean": float(aux_only_result["r2_mean"]),
                "r2_std": float(aux_only_result["r2_std"]),
                "r2_ci_low": float(aux_only_result["r2_ci_low"]),
                "r2_ci_high": float(aux_only_result["r2_ci_high"]),
                "mae_mean": float(aux_only_result["mae_mean"]),
                "mae_std": float(aux_only_result["mae_std"]),
                "mae_ci_low": float(aux_only_result["mae_ci_low"]),
                "mae_ci_high": float(aux_only_result["mae_ci_high"]),
                "cosine_mean": float(aux_only_result["cosine_mean"]),
                "cosine_std": float(aux_only_result["cosine_std"]),
                "cosine_ci_low": float(aux_only_result["cosine_ci_low"]),
                "cosine_ci_high": float(aux_only_result["cosine_ci_high"]),
                "alpha_mean": float(aux_only_result["alpha_mean"]),
                "alpha_std": float(aux_only_result["alpha_std"]),
                "residualizer_alpha_mean": float(aux_only_result["residualizer_alpha_mean"]),
                "residualizer_alpha_std": float(aux_only_result["residualizer_alpha_std"]),
                "splits": int(args.probe_splits),
                "test_size": float(args.probe_test_size),
            }
            aux_only_row.update(
                add_delta_stats(
                    (np.asarray(aux_only_result["r2_values"]) - np.asarray(patch_eval["r2_values"])).tolist(),
                    "delta_vs_patch_r2",
                )
            )
            target_rows.append(aux_only_row)
            all_rows.append(aux_only_row)

            joint_resid_result = residual_bundle["patch_plus_aux_resid"]
            unique_aux_resid = np.asarray(joint_resid_result["r2_values"]) - np.asarray(patch_eval["r2_values"])
            conditional_gain_fraction = unique_aux_resid / np.maximum(1e-12, 1.0 - np.asarray(patch_eval["r2_values"]))
            joint_resid_row: Dict[str, object] = {
                "target": target_name,
                "feature_set": f"patch_plus_{aux_source}_resid",
                "dim": int(np.asarray(feature_specs[0]["features"]).shape[1]) + aux_dim,
                "contains_patch": True,
                "aux_source": aux_source,
                "residualized_against": "patch_mean",
                "r2_mean": float(joint_resid_result["r2_mean"]),
                "r2_std": float(joint_resid_result["r2_std"]),
                "r2_ci_low": float(joint_resid_result["r2_ci_low"]),
                "r2_ci_high": float(joint_resid_result["r2_ci_high"]),
                "mae_mean": float(joint_resid_result["mae_mean"]),
                "mae_std": float(joint_resid_result["mae_std"]),
                "mae_ci_low": float(joint_resid_result["mae_ci_low"]),
                "mae_ci_high": float(joint_resid_result["mae_ci_high"]),
                "cosine_mean": float(joint_resid_result["cosine_mean"]),
                "cosine_std": float(joint_resid_result["cosine_std"]),
                "cosine_ci_low": float(joint_resid_result["cosine_ci_low"]),
                "cosine_ci_high": float(joint_resid_result["cosine_ci_high"]),
                "alpha_mean": float(joint_resid_result["alpha_mean"]),
                "alpha_std": float(joint_resid_result["alpha_std"]),
                "residualizer_alpha_mean": float(joint_resid_result["residualizer_alpha_mean"]),
                "residualizer_alpha_std": float(joint_resid_result["residualizer_alpha_std"]),
                "splits": int(args.probe_splits),
                "test_size": float(args.probe_test_size),
                "base_feature_dim": int(np.asarray(feature_specs[0]["features"]).shape[1]),
                "added_feature_dim": aux_dim,
            }
            joint_resid_row.update(add_delta_stats(unique_aux_resid.tolist(), "unique_aux_r2"))
            joint_resid_row.update(add_delta_stats(conditional_gain_fraction.tolist(), "conditional_gain_fraction"))
            target_rows.append(joint_resid_row)
            all_rows.append(joint_resid_row)

        target_rows.sort(key=lambda r: r["feature_set"])
        with open(outdir / f"complementarity_{target_name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ordered_fieldnames(target_rows), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(target_rows)

        combo_rows = [
            row
            for row in target_rows
            if str(row["feature_set"]).startswith("patch_plus_")
            and not str(row["feature_set"]).endswith("_resid")
        ]
        if combo_rows:
            combo_rows.sort(key=lambda row: row["feature_set"])
            make_errorbar_barplot(
                combo_rows,
                [str(row["feature_set"]) for row in combo_rows],
                metric_key="unique_aux_r2_mean",
                err_key="unique_aux_r2_std",
                out_path=plots_dir / f"unique_aux_{target_name}.png",
                title=f"{encoder_kind.upper()} unique auxiliary contribution for {target_name}",
                ylabel="Unique auxiliary $\\Delta R^2$ over patch baseline",
            )

        residual_combo_rows = [row for row in target_rows if str(row["feature_set"]).endswith("_resid") and str(row["feature_set"]).startswith("patch_plus_")]
        if residual_combo_rows:
            residual_combo_rows.sort(key=lambda row: row["feature_set"])
            make_errorbar_barplot(
                residual_combo_rows,
                [str(row["feature_set"]) for row in residual_combo_rows],
                metric_key="unique_aux_r2_mean",
                err_key="unique_aux_r2_std",
                out_path=plots_dir / f"unique_aux_resid_{target_name}.png",
                title=f"{encoder_kind.upper()} residualized auxiliary contribution for {target_name}",
                ylabel="Residualized auxiliary $\\Delta R^2$ over patch baseline",
            )

        summary["targets"][target_name] = {
            "best_feature_set_by_r2": max(target_rows, key=lambda row: float(row["r2_mean"])),
            "rows": target_rows,
        }

    with open(outdir / "complementarity_all_targets.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fieldnames(all_rows), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    summary["artifacts"] = {
        "all_targets_csv": str(outdir / "complementarity_all_targets.csv"),
        "per_target_csvs": {name: str(outdir / f"complementarity_{name}.csv") for name in targets.keys()},
        "plots_dir": str(plots_dir),
    }
    save_json(summary, outdir / "summary.json")


if __name__ == "__main__":
    main()
