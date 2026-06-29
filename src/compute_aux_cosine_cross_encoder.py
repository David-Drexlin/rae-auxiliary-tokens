#!/usr/bin/env python3
"""Compute cosine alignment between source global tokens and DINO aux tokens.

We compare MAE / SigLIP2 global tokens (CLS / pooler) against DINO CLS and
register statistics on the same ImageNet100 val images.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage1.encoders.dinov2 import Dinov2withNorm
from stage1.encoders.mae import MAEwNorm
from stage1.encoders.siglip2 import SigLIP2wNorm


def center_crop_arr(pil_image, image_size: int):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=pil_image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=pil_image.BICUBIC)

    arr = torch.from_numpy(__import__("numpy").array(pil_image))
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    arr = arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
    return __import__("PIL").Image.fromarray(arr.numpy())


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.ToTensor(),
        ]
    )


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def cosine_stats(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    cos = (a * b).sum(dim=-1)
    return float(cos.mean()), float(cos.std(unbiased=False))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path("DATASETS_ROOT_PLACEHOLDER/imagenet_torchvision/imagenet100/val")
    image_size = 256

    transform = build_transform(image_size)
    dataset = ImageFolder(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=(device.type == "cuda"))

    dino = Dinov2withNorm(dinov2_path="facebook/dinov2-with-registers-base", normalize=True).to(device).eval()
    mae = MAEwNorm(model_name="facebook/vit-mae-base").to(device).eval()
    siglip = SigLIP2wNorm(model_name="google/siglip2-base-patch16-256").to(device).eval()

    stats = {
        "mae_vs_dino_cls": [],
        "mae_vs_dino_reg_mean": [],
        "mae_vs_dino_aux_mean": [],
        "siglip_vs_dino_cls": [],
        "siglip_vs_dino_reg_mean": [],
        "siglip_vs_dino_aux_mean": [],
    }

    with torch.inference_mode():
        for images, _ in tqdm(loader, desc="cosine-aux"):
            images = images.to(device, non_blocking=True)

            _, dino_aux = dino.forward_with_global(images)  # [B,5,C]
            dino_cls = dino_aux[:, :1, :].squeeze(1)
            dino_regs = dino_aux[:, 1:, :].mean(dim=1)
            dino_aux_mean = dino_aux.mean(dim=1)

            _, mae_global = mae.forward_with_global(images)  # [B,1,C]
            mae_cls = mae_global.squeeze(1)

            _, sig_global = siglip.forward_with_global(images)  # [B,1,C]
            sig_cls = sig_global.squeeze(1)

            # normalize
            dino_cls = l2_norm(dino_cls)
            dino_regs = l2_norm(dino_regs)
            dino_aux_mean = l2_norm(dino_aux_mean)
            mae_cls = l2_norm(mae_cls)
            sig_cls = l2_norm(sig_cls)

            stats["mae_vs_dino_cls"].append((mae_cls * dino_cls).sum(dim=-1).cpu())
            stats["mae_vs_dino_reg_mean"].append((mae_cls * dino_regs).sum(dim=-1).cpu())
            stats["mae_vs_dino_aux_mean"].append((mae_cls * dino_aux_mean).sum(dim=-1).cpu())
            stats["siglip_vs_dino_cls"].append((sig_cls * dino_cls).sum(dim=-1).cpu())
            stats["siglip_vs_dino_reg_mean"].append((sig_cls * dino_regs).sum(dim=-1).cpu())
            stats["siglip_vs_dino_aux_mean"].append((sig_cls * dino_aux_mean).sum(dim=-1).cpu())

    results: Dict[str, Dict[str, float]] = {}
    for key, vals in stats.items():
        all_vals = torch.cat(vals, dim=0)
        results[key] = {"mean": float(all_vals.mean()), "std": float(all_vals.std(unbiased=False))}

    out_path = Path("RAE_ROOT_PLACEHOLDER/assets/analysis/aux_cosine_cross_encoder.json")
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
