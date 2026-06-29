#!/usr/bin/env python3
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Runs distributed reconstructions with a pre-trained stage-1 model.

- Input source:
  * ImageFolder: --data-path
  * HEST H5 patches: --hest-root

- If a sample limit is provided (e.g. --num-samples 50000):
  * we also pack the *REAL* inputs into ADM-style NPZ (real_samples.npz),
    aligned 1:1 with the recon outputs (samples.npz),
  * and we save selected_indices.npy (mapping position->dataset_index).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from omegaconf import OmegaConf

# repo imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1 import RAE
from utils.model_utils import instantiate_from_config
from utils.hest_h5_dataset import HESTPatchH5Dataset


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class IndexedImageFolder(ImageFolder):
    """ImageFolder that also returns the dataset index."""
    def find_classes(self, directory: str):
        classes = sorted(
            entry.name
            for entry in os.scandir(directory)
            if entry.is_dir() and not entry.name.startswith("_")
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image, int(index)


class IndexMapDataset(Dataset):
    """
    Dataset over positions [0..N-1] mapping to base_dataset[indices[pos]].

    Returns: (img_tensor, pos_int)
    """
    def __init__(self, base_dataset: Dataset, indices: np.ndarray):
        self.base = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, pos: int):
        ds_idx = int(self.indices[int(pos)])
        out = self.base[ds_idx]
        # base returns (img, something). We only need img.
        img = out[0] if isinstance(out, (tuple, list)) else out
        return img, int(pos)


def to_uint8_nhwc(images_chw: torch.Tensor) -> np.ndarray:
    """
    images_chw: (B,3,H,W) float in [0,1] -> uint8 NHWC
    """
    x = images_chw.detach().clamp(0, 1).mul(255).to(dtype=torch.uint8)
    x = x.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return x


def open_memmap_uint8(path: Path, shape: Tuple[int, int, int, int], mode: str):
    ensure_dir(path.parent)
    return np.lib.format.open_memmap(str(path), mode=mode, dtype=np.uint8, shape=shape)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling with DDP requires at least one GPU.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    # seed
    seed = int(args.global_seed) * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # autocast
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)

    if args.hest_root is None and args.data_path is None:
        raise ValueError("Provide either --data-path (ImageFolder) or --hest-root (HEST).")

    # -------------------------
    # Build stage-1 (RAE)
    # -------------------------
    cfg = OmegaConf.load(args.config)
    rae_config = cfg.get("stage_1", None)
    if rae_config is None:
        raise ValueError("Config must provide a stage_1 section.")

    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
    ])

    # -------------------------
    # Build base dataset
    # -------------------------
    using_hest = (args.hest_root is not None)
    if using_hest:
        base_dataset = HESTPatchH5Dataset(
            root=args.hest_root,
            patches_subdir="patches",
            transform=transform,
            return_index=True,
            block_size=int(args.hest_block_size),
            cache_blocks=int(args.hest_cache_blocks),
        )
        dataset_tag = "hest"
    else:
        base_dataset = IndexedImageFolder(args.data_path, transform=transform)
        dataset_tag = "imgfolder"

    total_available = len(base_dataset)
    if total_available <= 0:
        raise RuntimeError("Dataset has 0 items.")

    # -------------------------
    # Determine requested count (cap) + whether to pack real
    # -------------------------
    requested = total_available
    if args.num_samples is not None:
        requested = min(requested, int(args.num_samples))
    if using_hest and args.hest_limit is not None:
        requested = min(requested, int(args.hest_limit))

    if requested <= 0:
        raise RuntimeError("requested <= 0 (check --num-samples / --hest-limit).")

    limit_is_set = (args.num_samples is not None) or (using_hest and args.hest_limit is not None)
    pack_real = bool(args.pack_real_npz) and limit_is_set

    # subset mode
    subset_mode = str(args.subset_mode).lower()
    if subset_mode == "auto":
        subset_mode = "random" if (using_hest and args.num_samples is not None) else "sequential"
    if subset_mode not in {"sequential", "random"}:
        raise ValueError("--subset-mode must be one of: auto|sequential|random")

    # -------------------------
    # Output folder
    # -------------------------
    model_target = rae_config.get("target", "stage1")
    ckpt_path = rae_config.get("ckpt")
    ckpt_name = "pretrained" if not ckpt_path else os.path.splitext(os.path.basename(str(ckpt_path)))[0]

    folder_components: List[str] = [
        str(model_target).split(".")[-1].replace(os.sep, "-"),
        str(ckpt_name).replace(os.sep, "-"),
        f"{dataset_tag}",
        f"{subset_mode}",
        f"n{requested}",
        f"bs{args.per_proc_batch_size}",
        args.precision,
    ]
    folder_name = "-".join(folder_components)
    if os.environ.get("SAVE_FOLDER", None):
        folder_name = os.environ["SAVE_FOLDER"]

    out_root = Path(args.sample_dir)
    out_dir = out_root / folder_name
    if rank == 0:
        ensure_dir(out_dir)
        print(f"[out] {out_dir}")
    dist.barrier()

    # -------------------------
    # Select subset indices (position -> dataset_index)
    # -------------------------
    if subset_mode == "sequential":
        selected_ds_indices = np.arange(requested, dtype=np.int64)
    else:
        rng = np.random.default_rng(int(args.global_seed))
        selected_ds_indices = rng.choice(total_available, size=requested, replace=False).astype(np.int64)

    # Save selected indices ONLY when limit is set (your requirement)
    if rank == 0 and limit_is_set:
        np.save(out_dir / "selected_indices.npy", selected_ds_indices)
        info = {
            "dataset_tag": dataset_tag,
            "using_hest": bool(using_hest),
            "data_path": args.data_path,
            "hest_root": args.hest_root,
            "total_available": int(total_available),
            "requested": int(requested),
            "subset_mode": subset_mode,
            "global_seed": int(args.global_seed),
            "world_size": int(world_size),
            "per_proc_batch_size": int(args.per_proc_batch_size),
            "precision": str(args.precision),
            "image_size": int(args.image_size),
            "pack_real_npz": bool(pack_real),
            "pack_recon_npz": bool(args.pack_npz),
            "save_pngs": bool(args.save_pngs),
        }
        save_json(info, out_dir / "subset_info.json")
    dist.barrier()

    # -------------------------
    # Wrap into position-indexed dataset
    # -------------------------
    map_ds = IndexMapDataset(base_dataset, selected_ds_indices)

    # shard positions across ranks: positions = [rank, rank+world_size, ...]
    pos_all = np.arange(requested, dtype=np.int64)
    rank_pos = pos_all[rank::world_size].tolist()
    subset = Subset(map_ds, rank_pos)

    loader = DataLoader(
        subset,
        batch_size=int(args.per_proc_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
        persistent_workers=(int(args.num_workers) > 0),
    )

    # -------------------------
    # Prepare memmaps for NPZ packing
    # -------------------------
    H = W = int(args.image_size)
    shape = (int(requested), H, W, 3)

    recon_mm = None
    real_mm = None
    recon_mm_path = out_dir / "recon_arr.npy"
    real_mm_path = out_dir / "real_arr.npy"

    if args.pack_npz:
        if rank == 0:
            _ = open_memmap_uint8(recon_mm_path, shape=shape, mode="w+")
            del _
        dist.barrier()
        recon_mm = open_memmap_uint8(recon_mm_path, shape=shape, mode="r+")
    else:
        dist.barrier()

    if pack_real:
        if rank == 0:
            _ = open_memmap_uint8(real_mm_path, shape=shape, mode="w+")
            del _
        dist.barrier()
        real_mm = open_memmap_uint8(real_mm_path, shape=shape, mode="r+")
    else:
        dist.barrier()

    # -------------------------
    # Recon loop
    # -------------------------
    it = tqdm(loader, desc="Stage1 recon", total=math.ceil(len(rank_pos) / args.per_proc_batch_size)) if rank == 0 else loader

    with torch.inference_mode():
        for images, pos in it:
            if images.numel() == 0:
                continue

            # pos are subset positions (0..requested-1), used for filenames + memmap row indices
            pos_np = pos.detach().cpu().numpy().astype(np.int64)

            images = images.to(device, non_blocking=True)

            # write REAL (input) if requested
            if real_mm is not None:
                real_u8 = to_uint8_nhwc(images)
                real_mm[pos_np] = real_u8

            # encode/decode
            with autocast(**autocast_kwargs):
                recon = rae(images)

            recon = recon.clamp(0, 1)

            # write RECON memmap (for samples.npz)
            recon_u8 = to_uint8_nhwc(recon)
            if recon_mm is not None:
                recon_mm[pos_np] = recon_u8

            # optional PNGs (recon)
            if args.save_pngs:
                for sample_u8, p in zip(recon_u8, pos_np):
                    Image.fromarray(sample_u8).save(out_dir / f"{int(p):06d}.png")

    dist.barrier()

    # -------------------------
    # Finalize NPZs (rank0)
    # -------------------------
    if rank == 0:
        if recon_mm is not None:
            recon_mm.flush()
            np.savez_compressed(out_dir / "samples.npz", arr_0=recon_mm)
            print(f"[npz] wrote {out_dir / 'samples.npz'}")

            if not args.keep_memmaps:
                try:
                    del recon_mm
                except Exception:
                    pass
                try:
                    os.remove(recon_mm_path)
                except Exception:
                    pass

        if real_mm is not None:
            real_mm.flush()
            np.savez_compressed(out_dir / "real_samples.npz", arr_0=real_mm)
            print(f"[npz] wrote {out_dir / 'real_samples.npz'}")

            if not args.keep_memmaps:
                try:
                    del real_mm
                except Exception:
                    pass
                try:
                    os.remove(real_mm_path)
                except Exception:
                    pass

        print("Done.")

    dist.barrier()
    dist.destroy_process_group()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--config", type=str, required=True, help="Path to the stage-1 config file.")
    p.add_argument("--sample-dir", type=str, default="samples", help="Output root dir (subfolder will be created).")

    p.add_argument("--per-proc-batch-size", type=int, default=4, help="Images processed per GPU step.")
    p.add_argument("--num-samples", type=int, default=None, help="Cap N samples. If set, REAL NPZ packing can be enabled.")
    p.add_argument("--image-size", type=int, default=256, help="Target crop size before feeding images to the model.")
    p.add_argument("--num-workers", type=int, default=4, help="Dataloader workers per process.")
    p.add_argument("--global-seed", type=int, default=0, help="Base seed (adjusted per rank).")

    p.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Autocast precision.")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="Enable TF32 matmuls (Ampere+).")

    # subset policy
    p.add_argument(
        "--subset-mode",
        type=str,
        default="auto",
        choices=["auto", "sequential", "random"],
        help="How to select the subset when a cap is set. auto=random for HEST+num_samples, else sequential.",
    )

    # output toggles
    p.add_argument("--pack-npz", action=argparse.BooleanOptionalAction, default=True, help="Pack recon samples.npz.")
    p.add_argument(
        "--pack-real-npz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a limit is set, also pack real_samples.npz (REAL inputs).",
    )
    p.add_argument("--save-pngs", action=argparse.BooleanOptionalAction, default=True, help="Save recon PNGs.")
    p.add_argument("--keep-memmaps", action=argparse.BooleanOptionalAction, default=False, help="Keep *.npy memmaps.")

    # ---- Data source (choose exactly one) ----
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-path", type=str, default=None, help="ImageFolder root (class subdirs).")
    src.add_argument("--hest-root", type=str, default=None, help="HEST root, e.g. DATASETS_ROOT_PLACEHOLDER/hest1k")

    # Optional HEST controls (ignored if --data-path is used)
    p.add_argument("--hest-limit", type=int, default=None, help="Extra cap for HEST (in addition to --num-samples).")
    p.add_argument("--hest-block-size", type=int, default=256, help="H5 block size (patches per read).")
    p.add_argument("--hest-cache-blocks", type=int, default=2, help="LRU cache size in blocks (0 disables).")

    args = p.parse_args()
    main(args)
