from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from transformers import AutoImageProcessor

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from stage1.encoders.dinov2 import Dinov2withNorm  # noqa: E402
from stage1.encoders.mae import MAEwNorm  # noqa: E402
from stage1.encoders.siglip2 import SigLIP2wNorm  # noqa: E402
from register_prediction.metrics import TOKEN_NAMES, ensure_dir, save_json  # noqa: E402


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


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


def load_cfg(path: str) -> Dict:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def cache_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported cache dtype '{name}'.")


def processor_stats(model_name: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    processor = AutoImageProcessor.from_pretrained(model_name)
    mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    return mean, std


def infer_source_encoder_cls(source_cfg: Dict) -> str:
    explicit = source_cfg.get("encoder_cls", None)
    if explicit is not None:
        return str(explicit)

    text = f"{source_cfg.get('name', '')} {source_cfg.get('model_name', '')}".lower()
    if "siglip" in text:
        return "SigLIP2wNorm"
    if "mae" in text:
        return "MAEwNorm"
    if "dino" in text:
        return "Dinov2withNorm"
    raise ValueError(
        "Could not infer source encoder class. Set encoders.source.encoder_cls "
        "to one of: SigLIP2wNorm, MAEwNorm, Dinov2withNorm."
    )


def build_source_encoder(source_cfg: Dict) -> torch.nn.Module:
    encoder_cls = infer_source_encoder_cls(source_cfg)
    model_name = str(source_cfg["model_name"])
    if encoder_cls == "SigLIP2wNorm":
        return SigLIP2wNorm(model_name=model_name)
    if encoder_cls == "MAEwNorm":
        return MAEwNorm(model_name=model_name)
    if encoder_cls == "Dinov2withNorm":
        return Dinov2withNorm(
            dinov2_path=model_name,
            normalize=bool(source_cfg.get("normalize", True)),
        )
    raise ValueError(
        f"Unsupported source encoder class '{encoder_cls}'. "
        "Choose from: SigLIP2wNorm, MAEwNorm, Dinov2withNorm."
    )


def preprocess_for_encoder(
    images: torch.Tensor,
    input_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    if images.shape[-1] != input_size or images.shape[-2] != input_size:
        images = F.interpolate(images, size=(input_size, input_size), mode="bicubic", align_corners=False)
    return (images - mean) / std


def split_data_path(cfg: Dict, split: str) -> str:
    key = f"{split}_path"
    if key not in cfg["data"]:
        raise KeyError(f"Config data section is missing '{key}'.")
    return str(cfg["data"][key])


def split_max_samples(cfg: Dict, split: str) -> int:
    data_cfg = cfg.get("data", {})
    return int(data_cfg.get(f"max_samples_{split}", data_cfg.get("max_samples", 0)))


def prepare_split_dir(split_dir: Path, overwrite: bool) -> None:
    if split_dir.exists() and (split_dir / "manifest.json").exists():
        if not overwrite:
            raise FileExistsError(
                f"Cache manifest already exists at {split_dir / 'manifest.json'}. "
                "Set cache.overwrite=true or choose a new output root."
            )
        for path in split_dir.glob("shard_*.pt"):
            path.unlink()
        for name in ["manifest.json", "stats.pt", "cache_config.json"]:
            target = split_dir / name
            if target.exists():
                target.unlink()
    ensure_dir(split_dir)


def update_stats(
    source: torch.Tensor,
    source_global: Optional[torch.Tensor],
    target: torch.Tensor,
    source_sum: Optional[torch.Tensor],
    source_sum_sq: Optional[torch.Tensor],
    source_global_sum: Optional[torch.Tensor],
    source_global_sum_sq: Optional[torch.Tensor],
    target_sum: Optional[torch.Tensor],
    target_sum_sq: Optional[torch.Tensor],
    source_count: int,
    source_global_count: int,
    target_count: int,
):
    source = source.float().cpu()
    source_global = None if source_global is None else source_global.float().cpu()
    target = target.float().cpu()
    if source_sum is None:
        source_sum = torch.zeros(source.shape[-1], dtype=torch.float64)
        source_sum_sq = torch.zeros(source.shape[-1], dtype=torch.float64)
        target_sum = torch.zeros(target.shape[1], target.shape[2], dtype=torch.float64)
        target_sum_sq = torch.zeros(target.shape[1], target.shape[2], dtype=torch.float64)
    if source_global is not None and source_global_sum is None:
        source_global_sum = torch.zeros(source_global.shape[-1], dtype=torch.float64)
        source_global_sum_sq = torch.zeros(source_global.shape[-1], dtype=torch.float64)
    assert source_sum_sq is not None and target_sum is not None and target_sum_sq is not None
    source_sum += source.double().sum(dim=(0, 1))
    source_sum_sq += source.double().square().sum(dim=(0, 1))
    if source_global is not None:
        assert source_global_sum is not None and source_global_sum_sq is not None
        source_global_sum += source_global.double().sum(dim=(0, 1))
        source_global_sum_sq += source_global.double().square().sum(dim=(0, 1))
        source_global_count += int(source_global.shape[0] * source_global.shape[1])
    target_sum += target.double().sum(dim=0)
    target_sum_sq += target.double().square().sum(dim=0)
    source_count += int(source.shape[0] * source.shape[1])
    target_count += int(target.shape[0])
    return (
        source_sum,
        source_sum_sq,
        source_global_sum,
        source_global_sum_sq,
        target_sum,
        target_sum_sq,
        source_count,
        source_global_count,
        target_count,
    )


def finalize_stats(
    source_sum: torch.Tensor,
    source_sum_sq: torch.Tensor,
    source_global_sum: Optional[torch.Tensor],
    source_global_sum_sq: Optional[torch.Tensor],
    target_sum: torch.Tensor,
    target_sum_sq: torch.Tensor,
    source_count: int,
    source_global_count: int,
    target_count: int,
) -> Dict[str, torch.Tensor]:
    source_mean = source_sum / max(source_count, 1)
    source_var = (source_sum_sq / max(source_count, 1) - source_mean.square()).clamp_min(1.0e-12)
    target_mean = target_sum / max(target_count, 1)
    target_var = (target_sum_sq / max(target_count, 1) - target_mean.square()).clamp_min(1.0e-12)
    stats = {
        "source_mean": source_mean.float(),
        "source_std": torch.sqrt(source_var).float(),
        "target_mean": target_mean.float(),
        "target_std": torch.sqrt(target_var).float(),
        "source_count": torch.tensor(source_count, dtype=torch.long),
        "target_count": torch.tensor(target_count, dtype=torch.long),
        "token_names": TOKEN_NAMES,
    }
    if source_global_sum is not None and source_global_sum_sq is not None:
        source_global_mean = source_global_sum / max(source_global_count, 1)
        source_global_var = (
            source_global_sum_sq / max(source_global_count, 1) - source_global_mean.square()
        ).clamp_min(1.0e-12)
        stats["source_global_mean"] = source_global_mean.float()
        stats["source_global_std"] = torch.sqrt(source_global_var).float()
        stats["source_global_count"] = torch.tensor(source_global_count, dtype=torch.long)
    return stats


def run(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.config)
    split = str(args.split)
    output_root = Path(cfg["experiment"]["output_root"])
    cache_root = Path(cfg.get("cache", {}).get("dir", output_root / "cache"))
    split_dir = cache_root / split
    prepare_split_dir(split_dir, overwrite=bool(cfg.get("cache", {}).get("overwrite", False)))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_autocast = device.type == "cuda" and str(cfg.get("cache", {}).get("precision", cfg.get("cache", {}).get("dtype", "bf16"))).lower() in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if str(cfg.get("cache", {}).get("precision", cfg.get("cache", {}).get("dtype", "bf16"))).lower() == "bf16" else torch.float16

    data_path = split_data_path(cfg, split)
    transform = build_transform(int(cfg["data"].get("image_size", 256)))
    dataset = IndexedImageFolder(data_path, transform=transform)
    max_samples = split_max_samples(cfg, split)
    if max_samples > 0:
        keep = list(range(min(max_samples, len(dataset))))
        dataset.samples = [dataset.samples[i] for i in keep]
        dataset.targets = [dataset.targets[i] for i in keep]

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["data"].get("batch_size", 128)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 8)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    source_cfg = cfg["encoders"]["source"]
    target_cfg = cfg["encoders"]["target"]
    source_model_name = str(source_cfg.get("model_name", "google/siglip2-base-patch16-256"))
    target_model_name = str(target_cfg.get("model_name", "facebook/dinov2-with-registers-base"))
    source_input_size = int(source_cfg.get("input_size", 128))
    target_input_size = int(target_cfg.get("input_size", 224))

    source_mean, source_std = processor_stats(source_model_name, device)
    target_mean, target_std = processor_stats(target_model_name, device)
    source_encoder = build_source_encoder(source_cfg).to(device).eval().requires_grad_(False)
    target_encoder = Dinov2withNorm(dinov2_path=target_model_name, normalize=True).to(device).eval().requires_grad_(False)

    shard_size = int(cfg.get("cache", {}).get("shard_size", 2048))
    dtype = cache_dtype(str(cfg.get("cache", {}).get("dtype", "bf16")))
    cache_global_tokens = bool(source_cfg.get("cache_global_tokens", False))
    source_buf: List[torch.Tensor] = []
    source_global_buf: List[torch.Tensor] = []
    target_buf: List[torch.Tensor] = []
    labels_buf: List[torch.Tensor] = []
    indices_buf: List[torch.Tensor] = []
    paths_buf: List[str] = []
    shards = []
    shard_id = 0
    total = 0

    source_sum = source_sum_sq = source_global_sum = source_global_sum_sq = target_sum = target_sum_sq = None
    source_count = 0
    source_global_count = 0
    target_count = 0

    def flush() -> None:
        nonlocal shard_id, total, source_buf, source_global_buf, target_buf, labels_buf, indices_buf, paths_buf
        if not source_buf:
            return
        source_cat = torch.cat(source_buf, dim=0)
        source_global_cat = torch.cat(source_global_buf, dim=0) if source_global_buf else None
        target_cat = torch.cat(target_buf, dim=0)
        labels_cat = torch.cat(labels_buf, dim=0)
        indices_cat = torch.cat(indices_buf, dim=0)
        shard_name = f"shard_{shard_id:05d}.pt"
        shard_payload = {
            "source_tokens": source_cat.to(dtype),
            "target_tokens": target_cat.to(dtype),
            "labels": labels_cat,
            "indices": indices_cat,
            "image_paths": paths_buf,
            "token_names": TOKEN_NAMES,
        }
        if source_global_cat is not None:
            shard_payload["source_global_tokens"] = source_global_cat.to(dtype)
        torch.save(shard_payload, split_dir / shard_name)
        shards.append(
            {
                "file": shard_name,
                "num_samples": int(source_cat.shape[0]),
                "source_shape": list(source_cat.shape[1:]),
                "target_shape": list(target_cat.shape[1:]),
                "source_global_shape": list(source_global_cat.shape[1:]) if source_global_cat is not None else None,
            }
        )
        total += int(source_cat.shape[0])
        shard_id += 1
        source_buf = []
        source_global_buf = []
        target_buf = []
        labels_buf = []
        indices_buf = []
        paths_buf = []

    with torch.inference_mode():
        for images, labels, indices in tqdm(loader, desc=f"cache {split}"):
            images = images.to(device, non_blocking=True)
            source_images = preprocess_for_encoder(images, source_input_size, source_mean, source_std)
            target_images = preprocess_for_encoder(images, target_input_size, target_mean, target_std)
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    if cache_global_tokens and hasattr(source_encoder, "forward_with_global"):
                        source_tokens, source_global_tokens = source_encoder.forward_with_global(source_images)
                    else:
                        source_tokens = source_encoder(source_images)
                        source_global_tokens = None
                    _, target_tokens = target_encoder.forward_with_global(target_images)
            else:
                if cache_global_tokens and hasattr(source_encoder, "forward_with_global"):
                    source_tokens, source_global_tokens = source_encoder.forward_with_global(source_images)
                else:
                    source_tokens = source_encoder(source_images)
                    source_global_tokens = None
                _, target_tokens = target_encoder.forward_with_global(target_images)
            if target_tokens.shape[1] != len(TOKEN_NAMES):
                raise ValueError(
                    f"Expected {len(TOKEN_NAMES)} DINO global tokens {TOKEN_NAMES}, got {target_tokens.shape[1]}."
                )
            source_cpu = source_tokens.float().cpu()
            source_global_cpu = None if source_global_tokens is None else source_global_tokens.float().cpu()
            target_cpu = target_tokens.float().cpu()
            (
                source_sum,
                source_sum_sq,
                source_global_sum,
                source_global_sum_sq,
                target_sum,
                target_sum_sq,
                source_count,
                source_global_count,
                target_count,
            ) = update_stats(
                source_cpu,
                source_global_cpu,
                target_cpu,
                source_sum,
                source_sum_sq,
                source_global_sum,
                source_global_sum_sq,
                target_sum,
                target_sum_sq,
                source_count,
                source_global_count,
                target_count,
            )
            source_buf.append(source_cpu)
            if source_global_cpu is not None:
                source_global_buf.append(source_global_cpu)
            target_buf.append(target_cpu)
            labels_buf.append(labels.long().cpu())
            indices_buf.append(indices.long().cpu())
            paths_buf.extend([dataset.samples[int(i)][0] for i in indices.tolist()])
            if sum(t.shape[0] for t in source_buf) >= shard_size:
                flush()
    flush()

    if source_sum is None or source_sum_sq is None or target_sum is None or target_sum_sq is None:
        raise RuntimeError(f"No samples were cached for split '{split}' from {data_path}.")
    stats = finalize_stats(
        source_sum,
        source_sum_sq,
        source_global_sum,
        source_global_sum_sq,
        target_sum,
        target_sum_sq,
        source_count,
        source_global_count,
        target_count,
    )
    torch.save(stats, split_dir / "stats.pt")
    manifest = {
        "split": split,
        "data_path": data_path,
        "num_samples": total,
        "class_names": list(dataset.classes),
        "source_model": source_model_name,
        "target_model": target_model_name,
        "source_input_size": source_input_size,
        "target_input_size": target_input_size,
        "cache_global_tokens": cache_global_tokens,
        "token_names": TOKEN_NAMES,
        "dtype": str(dtype).replace("torch.", ""),
        "shards": shards,
    }
    save_json(manifest, split_dir / "manifest.json")
    save_json(cfg, split_dir / "cache_config.json")
    print(f"Cached {total} {split} samples into {split_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Cache frozen source patch tokens and DINO CLS/register targets.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val"])
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
