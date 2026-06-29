#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from stage1_semantic_retention_dino_utils import (
    FrozenDINOFeatureExtractor,
    IndexedImageFolder,
    build_transform,
    ensure_dir,
    save_json,
    save_pickle,
    topk_accuracy_from_proba,
)


DEFAULT_TRAIN = "DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/train"
DEFAULT_VAL = "DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/val"
DEFAULT_OUT = "RAE_ROOT_PLACEHOLDER/assets/analysis/stage1_semantic_retention_dino/probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frozen DINO CLS linear probe for ImageNet100.")
    parser.add_argument("--train-data", type=Path, default=Path(DEFAULT_TRAIN))
    parser.add_argument("--val-data", type=Path, default=Path(DEFAULT_VAL))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--encoder-path", type=str, default="facebook/dinov2-with-registers-base")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--encoder-input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--c", type=float, default=1.0)
    return parser.parse_args()


def _stratified_prefix_indices(dataset: IndexedImageFolder, max_samples: int) -> List[int]:
    """Pick a class-diverse subset so smoke runs don't collapse to one class."""
    class_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        class_to_indices[int(label)].append(idx)

    ordered_classes = sorted(class_to_indices.keys())
    selected: List[int] = []
    cursor = 0
    while len(selected) < max_samples:
        progressed = False
        for cls in ordered_classes:
            cls_indices = class_to_indices[cls]
            if cursor < len(cls_indices):
                selected.append(cls_indices[cursor])
                progressed = True
                if len(selected) >= max_samples:
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def maybe_subset(dataset: IndexedImageFolder, max_samples: int | None) -> Subset | IndexedImageFolder:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, _stratified_prefix_indices(dataset, int(max_samples)))


def collect_split(
    dataset,
    extractor: FrozenDINOFeatureExtractor,
    batch_size: int,
    num_workers: int,
    split_name: str,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=extractor.device.type == "cuda",
        drop_last=False,
    )
    cls_rows: List[np.ndarray] = []
    labels_rows: List[np.ndarray] = []
    indices_rows: List[np.ndarray] = []

    for batch in tqdm(loader, desc=f"encode_{split_name}", leave=False):
        images, labels, indices = batch
        views = extractor.encode_views(images)
        cls_rows.append(views["cls"].numpy())
        labels_rows.append(labels.numpy().astype(np.int64, copy=False))
        indices_rows.append(indices.numpy().astype(np.int64, copy=False))

    return {
        "cls": np.concatenate(cls_rows, axis=0),
        "labels": np.concatenate(labels_rows, axis=0),
        "indices": np.concatenate(indices_rows, axis=0),
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    transform = build_transform(args.image_size)
    train_dataset_full = IndexedImageFolder(str(args.train_data), transform=transform)
    val_dataset_full = IndexedImageFolder(str(args.val_data), transform=transform)
    train_dataset = maybe_subset(train_dataset_full, args.max_train_samples)
    val_dataset = maybe_subset(val_dataset_full, args.max_val_samples)

    extractor = FrozenDINOFeatureExtractor(
        encoder_path=args.encoder_path,
        encoder_input_size=args.encoder_input_size,
        device=args.device,
        precision=args.precision,
    )

    train = collect_split(train_dataset, extractor, args.batch_size, args.num_workers, "train")
    val = collect_split(val_dataset, extractor, args.batch_size, args.num_workers, "val")

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=args.max_iter,
            solver="lbfgs",
            multi_class="multinomial",
            C=float(args.c),
            random_state=0,
        ),
    )
    probe.fit(train["cls"], train["labels"])

    val_pred = probe.predict(val["cls"])
    val_proba = probe.predict_proba(val["cls"])
    probe_classes = np.asarray(getattr(probe, "classes_", probe.steps[-1][1].classes_))
    val_top1 = float(np.mean(val_pred == val["labels"]))
    val_top5 = topk_accuracy_from_proba(val_proba, val["labels"], probe_classes, k=min(5, len(probe_classes)))

    artifact = {
        "probe": probe,
        "class_names": list(train_dataset_full.classes),
        "encoder_path": args.encoder_path,
        "image_size": int(args.image_size),
        "encoder_input_size": int(args.encoder_input_size),
        "precision": args.precision,
        "feature_view": "cls",
        "train_count": int(train["cls"].shape[0]),
        "val_count": int(val["cls"].shape[0]),
    }
    save_pickle(artifact, args.output_dir / "probe.pkl")

    metadata = {
        "encoder_path": args.encoder_path,
        "image_size": int(args.image_size),
        "encoder_input_size": int(args.encoder_input_size),
        "precision": args.precision,
        "feature_view": "cls",
        "train_data": str(args.train_data),
        "val_data": str(args.val_data),
        "train_count": int(train["cls"].shape[0]),
        "val_count": int(val["cls"].shape[0]),
        "train_source_count": int(len(train_dataset_full)),
        "val_source_count": int(len(val_dataset_full)),
        "max_train_samples": None if args.max_train_samples is None else int(args.max_train_samples),
        "max_val_samples": None if args.max_val_samples is None else int(args.max_val_samples),
        "class_names": list(train_dataset_full.classes),
        "num_classes": int(len(train_dataset_full.classes)),
        "val_top1": val_top1,
        "val_top5": val_top5,
    }
    save_json(metadata, args.output_dir / "metadata.json")

    feature_info = {
        "train_shape": list(train["cls"].shape),
        "val_shape": list(val["cls"].shape),
        "dtype": str(train["cls"].dtype),
    }
    save_json(feature_info, args.output_dir / "feature_info.json")

    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
