#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(REPO_ROOT / "src"))

from stage1.encoders.dinov2 import Dinov2withNorm  # noqa: E402


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def save_pickle(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.ToTensor(),
        ]
    )


class IndexedImageFolder(ImageFolder):
    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        return image, int(label), int(index)


class PairedReconDataset(Dataset):
    def __init__(
        self,
        val_dataset: IndexedImageFolder,
        selected_indices: Sequence[int],
        recon_paths: Sequence[Path],
        image_size: int,
    ):
        if len(selected_indices) != len(recon_paths):
            raise ValueError(
                f"selected_indices length {len(selected_indices)} does not match "
                f"recon count {len(recon_paths)}"
            )
        self.val_dataset = val_dataset
        self.selected_indices = [int(x) for x in selected_indices]
        self.recon_paths = list(recon_paths)
        self.transform = build_transform(image_size)

    def __len__(self) -> int:
        return len(self.selected_indices)

    def __getitem__(self, pos: int):
        ds_idx = int(self.selected_indices[pos])
        real_img, label, _ = self.val_dataset[ds_idx]
        recon_img = Image.open(self.recon_paths[pos]).convert("RGB")
        recon_img = self.transform(recon_img)
        return (
            real_img,
            recon_img,
            int(label),
            int(ds_idx),
            self.val_dataset.samples[ds_idx][0],
            str(self.recon_paths[pos]),
        )


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def cosine_per_row(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for cosine: {a.shape} vs {b.shape}")
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), eps)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), eps)
    return np.sum(a_norm * b_norm, axis=1)


def topk_accuracy_from_proba(proba: np.ndarray, labels: np.ndarray, classes: np.ndarray, k: int) -> float:
    topk = np.argsort(proba, axis=1)[:, -k:]
    class_values = classes[topk]
    return float(np.mean(np.any(class_values == labels[:, None], axis=1)))


def canonical_dino_family(save_folder: str) -> Optional[str]:
    if "Patch+Register+CLS_prepend" in save_folder:
        return "Patch+Register+CLS_prepend"
    if "Patch+Register_prepend" in save_folder:
        return "Patch+Register_prepend"
    if "Patch+CLS_prepend" in save_folder:
        return "Patch+CLS_prepend"
    if "DINO_decB_seed" in save_folder or save_folder.endswith("DINO_decB"):
        return "DINO_decB"
    return None


def extract_seed(save_folder: str) -> Optional[int]:
    marker = "_seed"
    if marker not in save_folder:
        return None
    suffix = save_folder.rsplit(marker, 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return None


class FrozenDINOFeatureExtractor:
    def __init__(
        self,
        encoder_path: str = "facebook/dinov2-with-registers-base",
        encoder_input_size: int = 224,
        device: Optional[str] = None,
        precision: str = "bf16",
    ):
        self.encoder_path = str(encoder_path)
        self.encoder_input_size = int(encoder_input_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.precision = str(precision)
        self.use_autocast = self.device.type == "cuda" and self.precision in {"bf16", "fp16"}
        self.autocast_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16

        processor = AutoImageProcessor.from_pretrained(self.encoder_path)
        self.encoder_mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        self.encoder_std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)

        self.encoder = Dinov2withNorm(dinov2_path=self.encoder_path, normalize=True).to(self.device).eval()

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, non_blocking=True)
        if images.shape[-2:] != (self.encoder_input_size, self.encoder_input_size):
            images = F.interpolate(
                images,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic",
                align_corners=False,
            )
        images = (images - self.encoder_mean) / self.encoder_std
        return images

    @torch.inference_mode()
    def encode_views(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        images = self._prepare_images(images)
        if self.use_autocast:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                patch_tokens, global_tokens = self.encoder.forward_with_global(images)
        else:
            patch_tokens, global_tokens = self.encoder.forward_with_global(images)

        regs = global_tokens[:, 1:]
        out = {
            "cls": global_tokens[:, 0].float().cpu(),
            "regs": regs.float().cpu(),
            "reg_mean": regs.mean(dim=1).float().cpu(),
            "patch_mean": patch_tokens.mean(dim=1).float().cpu(),
        }
        for i in range(regs.shape[1]):
            out[f"reg{i + 1}"] = regs[:, i].float().cpu()
        return out


def load_alignment_indices(
    run_dir: Path,
    val_dataset_len: int,
    explicit_selected_indices: Optional[Path] = None,
) -> np.ndarray:
    selected_path = explicit_selected_indices
    if selected_path is None:
        candidate = run_dir / "selected_indices.npy"
        selected_path = candidate if candidate.exists() else None

    recon_pngs = sorted(p for p in run_dir.glob("*.png") if p.is_file())
    if not recon_pngs:
        raise FileNotFoundError(f"No PNG reconstructions found in {run_dir}")

    if selected_path is not None:
        selected = np.load(selected_path).astype(np.int64)
        if selected.shape[0] != len(recon_pngs):
            raise ValueError(
                f"selected_indices length {selected.shape[0]} does not match "
                f"reconstruction count {len(recon_pngs)} in {run_dir}"
            )
        return selected

    if len(recon_pngs) != val_dataset_len:
        raise ValueError(
            "selected_indices.npy is missing and reconstruction count does not match the "
            f"full validation set: {len(recon_pngs)} vs {val_dataset_len}"
        )
    return np.arange(val_dataset_len, dtype=np.int64)


def summarize_token_cosines(per_token_cosines: Dict[str, np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for token_name, values in per_token_cosines.items():
        stats = mean_std(values.tolist())
        out[f"{token_name}_cosine"] = stats["mean"]
        out[f"{token_name}_cosine_std"] = stats["std"]
    return out

