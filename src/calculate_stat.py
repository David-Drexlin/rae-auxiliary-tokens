#!/usr/bin/env python3
"""
Distributed latent statistics estimation for a pre-trained stage-1 RAE.

Supports:
  - baseline / discard
  - pooled scale-shift conditioning ("adaln_pool")
  - token conditioning ("prepend", "cross_attn")

Outputs a normalization_stats.pt with keys:
  {
      "mean": mean_tensor_cpu,
      "var": var_tensor_cpu,
      "global_mean": global_mean_tensor_cpu,   # optional, pooled aux stats
      "global_var": global_var_tensor_cpu,     # optional
      "aux_mean": aux_mean_tensor_cpu,         # optional, token-level aux stats
      "aux_var": aux_var_tensor_cpu,           # optional
  }

Notes:
- This script computes stats from RAW encoder outputs / latents.
- Any preloaded normalization in the RAE is disabled during stats computation.
"""

import argparse
import os
import sys
from itertools import chain
from typing import List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from stage1 import RAE
from utils.model_utils import instantiate_from_config


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
    """ImageFolder that also returns the dataset index."""
    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image, index


def sanitize_component(component: str) -> str:
    return component.replace(os.sep, "-")


class TensorMomentAccumulator(nn.Module):
    """
    Exact moment accumulator over batch dimension only.

    For inputs x of shape [B, *feature_shape], accumulates:
      sum_x   = sum over batch and all processed batches
      sum_x2  = sum of squares
      count   = number of samples seen

    Final stats:
      mean = sum_x / count
      var  = sum_x2 / count - mean^2
    """
    def __init__(self, feature_shape, dtype=torch.float64):
        super().__init__()
        self.feature_shape = tuple(feature_shape)
        self.acc_dtype = dtype

        self.register_buffer("sum_x", torch.zeros(*self.feature_shape, dtype=self.acc_dtype))
        self.register_buffer("sum_x2", torch.zeros(*self.feature_shape, dtype=self.acc_dtype))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.ndim != 1 + len(self.feature_shape):
            raise ValueError(
                f"Expected x shape [B, {self.feature_shape}], got {tuple(x.shape)}"
            )
        x = x.to(dtype=self.acc_dtype)
        self.sum_x += x.sum(dim=0)
        self.sum_x2 += (x * x).sum(dim=0)
        self.count += x.shape[0]

    @torch.no_grad()
    def sync_across_ranks(self) -> None:
        if not dist.is_initialized():
            return
        dist.all_reduce(self.sum_x, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.sum_x2, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.count, op=dist.ReduceOp.SUM)

    @torch.no_grad()
    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if int(self.count.item()) <= 0:
            raise RuntimeError("Accumulator saw zero samples.")
        denom = self.count.to(dtype=self.acc_dtype)
        mean = self.sum_x / denom
        var = self.sum_x2 / denom - mean * mean
        var = torch.clamp(var, min=0.0)
        return mean, var


def _disable_loaded_normalization(rae: RAE) -> None:
    """
    When computing stats, we want RAW outputs, not already-normalized outputs.
    """
    if hasattr(rae, "do_normalization"):
        rae.do_normalization = False
    if hasattr(rae, "do_global_normalization"):
        rae.do_global_normalization = False
    if hasattr(rae, "do_aux_normalization"):
        rae.do_aux_normalization = False


def _resolve_aux_stats_mode(rae: RAE, requested_mode: str) -> str:
    """
    requested_mode:
      - auto
      - none
      - pooled
      - tokens
      - both
    """
    requested_mode = str(requested_mode)

    if requested_mode != "auto":
        return requested_mode

    decoder_aux_mode = getattr(rae, "decoder_aux_mode", None)

    if decoder_aux_mode == "adaln_pool":
        return "pooled"
    if decoder_aux_mode in {"prepend", "cross_attn"}:
        return "tokens"
    return "none"


@torch.no_grad()
def _extract_stats_targets(
    rae: RAE,
    images: torch.Tensor,
    aux_stats_mode: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns:
      z:          [B, ...]         patch latent
      pooled:     [B, C] or None   pooled aux stats target
      aux_tokens: [B, K, C] or None token-level aux stats target
    """
    aux_stats_mode = str(aux_stats_mode)

    if aux_stats_mode == "none":
        z = rae.encode(images)
        return z, None, None

    # Prefer single-pass extraction through internal helpers to avoid double encoding.
    if all(
        hasattr(rae, attr) for attr in
        ["_encode_tokens", "_patch_tokens_to_latent", "_select_aux_tokens", "_pool_aux_tokens"]
    ):
        patch_tokens, global_tokens = rae._encode_tokens(images, need_aux=True)
        z = rae._patch_tokens_to_latent(patch_tokens)
        aux_tokens = rae._select_aux_tokens(global_tokens)

        pooled = None
        if aux_stats_mode in {"pooled", "both"}:
            pooled = rae._pool_aux_tokens(aux_tokens)

        token_stats = None
        if aux_stats_mode in {"tokens", "both"}:
            token_stats = aux_tokens

        return z, pooled, token_stats

    # Fallback path using public API.
    if aux_stats_mode == "pooled":
        z, pooled = rae.encode_with_cond(images, normalize_global_cond=False)
        return z, pooled, None

    if aux_stats_mode == "tokens":
        z, aux_tokens = rae.encode_with_aux_tokens(images, normalize_aux_tokens=False)
        return z, None, aux_tokens

    if aux_stats_mode == "both":
        # This fallback does two encoder passes.
        z, pooled = rae.encode_with_cond(images, normalize_global_cond=False)
        _z2, aux_tokens = rae.encode_with_aux_tokens(images, normalize_aux_tokens=False)
        return z, pooled, aux_tokens

    raise ValueError(f"Unsupported aux_stats_mode='{aux_stats_mode}'")


def _make_accumulator(x: torch.Tensor) -> TensorMomentAccumulator:
    return TensorMomentAccumulator(x.shape[1:]).to(x.device)


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This script assumes CUDA + nccl DDP.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but this CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)

    cfg = OmegaConf.load(args.config)
    rae_config = cfg.get("stage_1", None)
    if rae_config is None:
        raise ValueError("Config must provide a stage_1 section.")

    # Stats must be computed from raw encoder outputs, so do not even load
    # any precomputed normalization file from the config at construction time.
    if OmegaConf.select(cfg, "stage_1.params.normalization_stat_path") is not None:
        OmegaConf.update(cfg, "stage_1.params.normalization_stat_path", None, force_add=True)
        rae_config = cfg.get("stage_1", None)

    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    _disable_loaded_normalization(rae)

    dataset = IndexedImageFolder(
        args.data_path,
        transform=transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
        ]),
    )

    total_available = len(dataset)
    if total_available == 0:
        raise ValueError(f"No images found at {args.data_path}.")

    requested = total_available if args.num_samples is None else min(args.num_samples, total_available)
    if requested <= 0:
        raise ValueError("Number of samples to process must be positive.")

    base_ds = dataset if requested == total_available else Subset(dataset, list(range(requested)))

    sampler = DistributedSampler(
        base_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    loader = DataLoader(
        base_ds,
        batch_size=args.per_proc_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    if len(loader) == 0:
        raise RuntimeError(
            f"Rank {rank} received an empty loader. requested={requested}, world_size={world_size}."
        )

    if rank == 0 and args.output_pt is None:
        os.makedirs(args.sample_dir, exist_ok=True)

    model_target = rae_config.get("target", "stage1")
    ckpt_path = rae_config.get("ckpt")
    ckpt_name = "pretrained" if not ckpt_path else os.path.splitext(os.path.basename(str(ckpt_path)))[0]

    folder_components: List[str] = [
        sanitize_component(str(model_target).split(".")[-1]),
        sanitize_component(ckpt_name),
        f"bs{args.per_proc_batch_size}",
        args.precision,
    ]
    base_folder = "-".join(folder_components)
    folder_name = os.environ.get("SAVE_FOLDER", base_folder)
    out_dir = os.path.join(args.sample_dir, folder_name)

    if rank == 0:
        if args.output_pt is not None:
            os.makedirs(os.path.dirname(args.output_pt), exist_ok=True)
            print(f"[path] saving stats directly to: {args.output_pt}")
        else:
            os.makedirs(out_dir, exist_ok=True)
            print(f"[path] saving stats under: {out_dir}")
        print(f"[init] world_size={world_size} requested_samples={requested}")
        print(f"[init] dataset_crop_size={args.image_size} encoder_input_size={getattr(rae, 'encoder_input_size', 'n/a')}")
        print(f"[norm] disabled preloaded normalization inside RAE during stats computation")
    dist.barrier()

    aux_stats_mode = _resolve_aux_stats_mode(rae, args.aux_stats_mode)

    loader_iter = iter(loader)
    first_batch = next(loader_iter, None)
    if first_batch is None:
        raise RuntimeError("Empty loader on this rank.")

    images0, _indices0 = first_batch
    images0 = images0.to(device, non_blocking=True)

    with autocast(**autocast_kwargs):
        z0, pooled0, aux0 = _extract_stats_targets(rae, images0, aux_stats_mode=aux_stats_mode)

    z_acc = _make_accumulator(z0)
    pooled_acc = _make_accumulator(pooled0) if pooled0 is not None else None
    aux_acc = _make_accumulator(aux0) if aux0 is not None else None

    if rank == 0:
        print(f"[mode] aux_stats_mode={aux_stats_mode}")
        print(f"[z]     shape={tuple(z0.shape)}")
        if pooled0 is not None:
            print(f"[pooled] shape={tuple(pooled0.shape)}")
        else:
            print("[pooled] skipped")
        if aux0 is not None:
            print(f"[aux]    shape={tuple(aux0.shape)}")
        else:
            print("[aux]    skipped")
    dist.barrier()

    iterator = tqdm(
        chain([first_batch], loader_iter),
        desc="Latent stats",
        total=len(loader),
        disable=(rank != 0),
    )

    for images, _indices in iterator:
        images = images.to(device, non_blocking=False)

        with autocast(**autocast_kwargs):
            z, pooled, aux_tokens = _extract_stats_targets(rae, images, aux_stats_mode=aux_stats_mode)

        z_acc.update(z)

        if pooled_acc is not None:
            if pooled is None:
                raise RuntimeError("pooled_acc was initialized, but pooled became None during iteration.")
            pooled_acc.update(pooled)

        if aux_acc is not None:
            if aux_tokens is None:
                raise RuntimeError("aux_acc was initialized, but aux_tokens became None during iteration.")
            aux_acc.update(aux_tokens)

        torch.cuda.synchronize()

    z_acc.sync_across_ranks()
    mean, var = z_acc.finalize()

    global_mean = None
    global_var = None
    if pooled_acc is not None:
        pooled_acc.sync_across_ranks()
        global_mean, global_var = pooled_acc.finalize()

    aux_mean = None
    aux_var = None
    if aux_acc is not None:
        aux_acc.sync_across_ranks()
        aux_mean, aux_var = aux_acc.finalize()

    dist.barrier()

    if rank == 0:
        payload = {
            "mean": mean.cpu().float(),
            "var": var.cpu().float(),
        }

        if global_mean is not None and global_var is not None:
            payload["global_mean"] = global_mean.cpu().float()
            payload["global_var"] = global_var.cpu().float()

        if aux_mean is not None and aux_var is not None:
            payload["aux_mean"] = aux_mean.cpu().float()
            payload["aux_var"] = aux_var.cpu().float()

        if args.output_pt is not None:
            out_path = args.output_pt
        else:
            out_path = os.path.join(out_dir, "normalization_stats.pt")

        torch.save(payload, out_path)

        print(f"[done] wrote: {out_path}")
        print(f"[mean]        shape={tuple(payload['mean'].shape)} dtype={payload['mean'].dtype}")
        print(f"[var]         shape={tuple(payload['var'].shape)} dtype={payload['var'].dtype}")

        if "global_mean" in payload:
            print(f"[global_mean] shape={tuple(payload['global_mean'].shape)} dtype={payload['global_mean'].dtype}")
            print(f"[global_var]  shape={tuple(payload['global_var'].shape)} dtype={payload['global_var'].dtype}")

        if "aux_mean" in payload:
            print(f"[aux_mean]    shape={tuple(payload['aux_mean'].shape)} dtype={payload['aux_mean'].dtype}")
            print(f"[aux_var]     shape={tuple(payload['aux_var'].shape)} dtype={payload['aux_var'].dtype}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the stage-1 config file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to an ImageFolder directory.")
    parser.add_argument("--sample-dir", type=str, default="stats/", help="Base directory to store stats output.")
    parser.add_argument(
        "--output-pt",
        type=str,
        default=None,
        help="If set, write the stats payload directly to this .pt file.",
    )
    parser.add_argument("--per-proc-batch-size", type=int, default=256, help="Images processed per GPU step.")
    parser.add_argument("--num-samples", type=int, default=None, help="How many images to use (default: all).")
    parser.add_argument("--image-size", type=int, default=256, help="Center crop size before feeding the model.")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers per process.")
    parser.add_argument("--global-seed", type=int, default=0, help="Base seed (adjusted per rank).")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Autocast precision.")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="Enable TF32 matmuls.")
    parser.add_argument(
        "--aux-stats-mode",
        type=str,
        choices=["auto", "none", "pooled", "tokens", "both"],
        default="auto",
        help=(
            "Which non-patch stats to compute. "
            "'auto' follows rae.decoder_aux_mode: "
            "discard->none, adaln_pool->pooled, prepend/cross_attn->tokens."
        ),
    )
    args = parser.parse_args()
    main(args)
