#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast
from tqdm import tqdm
from omegaconf import OmegaConf

REPO_SRC = Path(__file__).resolve().parent
sys.path.append(str(REPO_SRC))

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from utils.condition_sampler import (
    ConditionPool,
    compute_total_samples_rounded,
    shard_pool_for_rank,
)
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.state_utils import (
    duplicate_state_for_guidance,
    final_state_from_trajectory,
    infer_aux_state_spec,
    make_initial_sample_state,
    split_guided_state,
    state_float,
)
from stage2.transport import Sampler, create_transport
from register_prediction.eval import build_model_for_eval, cache_root, train_stats_path
from register_prediction.metrics import load_stats, standardize_source, unstandardize_target


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self._f = open(self.path, "a", buffering=1)

    def write(self, rec: Dict[str, Any]) -> None:
        self._f.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        try:
            self._f.flush()
        finally:
            self._f.close()


def write_manifest_info(out_folder: Path, info: Dict[str, Any]) -> None:
    ensure_dir(out_folder)
    with open(out_folder / "manifest_info.json", "w") as f:
        json.dump(info, f, indent=2, sort_keys=True)


def merge_rank_manifests(out_folder: Path, world_size: int) -> None:
    merged_jsonl = out_folder / "manifest.jsonl"
    merged_csv = out_folder / "manifest.csv"
    header = [
        "index",
        "filename",
        "y",
        "cond_mode",
        "rank",
        "seed",
    ]
    with open(merged_jsonl, "w") as f_jsonl, open(merged_csv, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=header)
        writer.writeheader()
        total_rows = 0
        for rank in range(world_size):
            rank_path = out_folder / f"manifest_rank{rank:03d}.jsonl"
            if not rank_path.exists():
                continue
            with open(rank_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    f_jsonl.write(line + "\n")
                    rec = json.loads(line)
                    writer.writerow(
                        {
                            "index": int(rec["index"]),
                            "filename": str(rec["filename"]),
                            "y": int(rec["y"]),
                            "cond_mode": str(rec.get("cond_mode", "")),
                            "rank": int(rec.get("rank", -1)),
                            "seed": int(rec.get("seed", -1)),
                        }
                    )
                    total_rows += 1
    print(f"[manifest] merged {total_rows} rows -> {merged_csv}")


def create_npz_from_sample_folder(sample_dir: Path, num: int, out_name: str = "samples.npz") -> Path:
    pngs = sorted(sample_dir.glob("*.png"))
    if len(pngs) == 0:
        raise RuntimeError(f"No PNGs found in {sample_dir}")
    pngs = pngs[:num] if num is not None else pngs
    first = np.array(Image.open(pngs[0]).convert("RGB"), dtype=np.uint8)
    h, w, _ = first.shape
    arr = np.empty((len(pngs), h, w, 3), dtype=np.uint8)
    arr[0] = first
    for idx, path in enumerate(tqdm(pngs[1:], desc="Packing NPZ", unit="img"), start=1):
        arr[idx] = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    out_path = sample_dir / out_name
    np.savez_compressed(out_path, arr_0=arr)
    print(f"[npz] wrote {out_path} shape={arr.shape} dtype={arr.dtype}")
    return out_path


def save_latent_shard(
    out_folder: Path,
    *,
    latent_subdir: str,
    rank: int,
    step_idx: int,
    indices,
    y,
    state,
) -> None:
    latent_dir = out_folder / latent_subdir
    ensure_dir(latent_dir)
    payload: Dict[str, Any] = {
        "indices": torch.as_tensor(indices, dtype=torch.int64),
        "y": torch.as_tensor(y, dtype=torch.int64),
    }
    if torch.is_tensor(state):
        payload["z"] = state.detach().to("cpu", dtype=torch.float16)
    else:
        z, aux = state
        payload["z"] = z.detach().to("cpu", dtype=torch.float16)
        payload["aux"] = aux.detach().to("cpu", dtype=torch.float16)
    shard_path = latent_dir / f"rank{rank:03d}_step{step_idx:06d}.pt"
    torch.save(payload, shard_path)


def load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    with open(p, "r") as f:
        return json.load(f)


def load_stage2_weights(model: torch.nn.Module, ckpt_path: str, use_ema: bool, rank: int) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["ema"] if use_ema else ckpt["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if rank == 0:
        print(f"[ckpt] loaded {ckpt_path} (ema={use_ema}) missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[ckpt] missing (first 10): {missing[:10]}")
        if unexpected:
            print(f"[ckpt] unexpected (first 10): {unexpected[:10]}")
    model.eval()
    return ckpt


def require(cfg: Dict[str, Any], key_path: str):
    cur: Any = cfg
    for key in key_path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing required sample-config key: '{key_path}'")
        cur = cur[key]
    return cur


def get_guidance_params(guidance_cfg: Dict[str, Any]):
    def guidance_value(key: str, default: float):
        if key in guidance_cfg:
            return guidance_cfg[key]
        dashed_key = key.replace("_", "-")
        return guidance_cfg.get(dashed_key, default)

    scale = float(guidance_cfg.get("scale", 1.0))
    aux_scale = float(guidance_cfg.get("aux_scale", guidance_cfg.get("aux-scale", 0.0)))
    method = str(guidance_cfg.get("method", "cfg"))
    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))
    return scale, aux_scale, method, t_min, t_max


def build_stage1_from_source_config(
    source_config_path: str,
    device: torch.device,
    *,
    decoder_ckpt: str = "auto",
) -> RAE:
    wrapper_cfg = OmegaConf.create(
        {
            "stage1_source": {
                "config_path": str(source_config_path),
                "decoder_ckpt": str(decoder_ckpt),
            }
        }
    )
    (_, rae_config, _, _, _, _, _, _, _) = parse_configs(wrapper_cfg)
    if rae_config is None:
        raise ValueError(f"Could not resolve stage_1 config from source: {source_config_path}")
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    return rae


def convert_patch_latent_between_raes(
    z_normalized: torch.Tensor,
    *,
    source_rae: RAE,
    target_rae: RAE,
) -> torch.Tensor:
    z_raw = source_rae._denormalize_patch_latent(z_normalized.float())
    return target_rae._normalize_patch_latent(z_raw)


def latent_to_source_tokens(z_normalized: torch.Tensor, source_rae: RAE) -> torch.Tensor:
    z_raw = source_rae._denormalize_patch_latent(z_normalized.float())
    if z_raw.ndim == 4:
        bsz, channels, height, width = z_raw.shape
        return z_raw.view(bsz, channels, height * width).transpose(1, 2).contiguous()
    if z_raw.ndim == 3:
        return z_raw
    raise ValueError(f"Unsupported latent shape for predictor input: {tuple(z_raw.shape)}")


def load_predictor_bundle(
    *,
    config_path: str,
    ckpt_path: str,
    device: torch.device,
    model_name: Optional[str] = None,
):
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError(f"Predictor config must resolve to a dict: {config_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    resolved_model_name = model_name or str(ckpt.get("model_name", cfg.get("model", {}).get("name", "mhap")))
    model = build_model_for_eval(
        resolved_model_name,
        cfg,
        ckpt,
        cache_root(cfg) / "val",
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    stats = load_stats(train_stats_path(cfg))
    return resolved_model_name, cfg, model, stats


def build_equal_class_pool(
    *,
    num_classes: int,
    total_samples: int,
    num_requested: int,
    seed: int,
    null_label: int,
) -> ConditionPool:
    valid = [idx for idx in range(int(num_classes)) if idx != int(null_label)]
    if len(valid) == 0:
        raise RuntimeError("No valid class ids remain for equal_class.")
    if num_requested % len(valid) != 0:
        raise ValueError(f"equal_class requires num_samples divisible by #valid_classes ({len(valid)}).")
    reps = num_requested // len(valid)
    base = np.repeat(np.array(valid, dtype=np.int64), reps)
    rng = np.random.default_rng(seed)
    rng.shuffle(base)
    if total_samples > len(base):
        tail = rng.choice(np.array(valid, dtype=np.int64), size=(total_samples - len(base)), replace=True)
        y_pool = np.concatenate([base, tail], axis=0)
    else:
        y_pool = base[:total_samples]
    return ConditionPool(y=y_pool.astype(np.int64), meta=None)


def infer_y_vocab(ckpt_path: str, ckpt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    y_vocab = ckpt.get("y_vocab", None)
    if y_vocab is not None:
        return y_vocab
    ckpt_p = Path(ckpt_path).resolve()
    candidates = [
        ckpt_p.parent / "y_vocab.json",
        ckpt_p.parent.parent / "y_vocab.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return load_json(str(candidate))
    return None


def main(cfg: Dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling with DDP requires at least one GPU.")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    stage2_yaml = str(require(cfg, "stage2.config"))
    ckpt_path = str(require(cfg, "stage2.ckpt"))
    use_ema = bool(require(cfg, "stage2.use_ema"))

    predictor_cfg_path = str(require(cfg, "predictor.config"))
    predictor_ckpt_path = str(require(cfg, "predictor.ckpt"))
    predictor_model_name = cfg.get("predictor", {}).get("model", None)

    target_stage1_cfg = str(require(cfg, "target_stage1.config_path"))
    target_stage1_decoder_ckpt = str(cfg.get("target_stage1", {}).get("decoder_ckpt", "auto"))

    sample_dir = Path(str(require(cfg, "output.sample_dir")))
    save_folder = str(require(cfg, "output.save_folder"))
    output_cfg = cfg.get("output", {})
    pack_npz = bool(output_cfg.get("pack_npz", False))
    save_latent_shards = bool(output_cfg.get("save_latent_shards", False))
    latent_shard_dir = str(output_cfg.get("latent_shard_dir", "latent_shards"))

    num_samples = int(require(cfg, "run.num_samples"))
    per_proc_batch_size = int(require(cfg, "run.per_proc_batch_size"))
    precision = str(require(cfg, "run.precision")).lower()
    tf32 = bool(require(cfg, "run.tf32"))
    global_seed = int(require(cfg, "run.global_seed"))

    cond_mode = str(cfg.get("conditioning", {}).get("mode", "equal_class"))
    if cond_mode not in {"equal_class", "random_class", "uncond"}:
        raise ValueError("conditioning.mode must be one of: equal_class, random_class, uncond")

    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_grad_enabled(False)

    use_bf16 = precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 but device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)

    stage2_cfg = OmegaConf.load(stage2_yaml)
    (
        _data_config,
        source_rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        _training_config,
        _eval_config,
    ) = parse_configs(stage2_cfg)
    if source_rae_config is None or model_config is None:
        raise ValueError("Stage-2 YAML must provide stage_1 and stage_2 sections.")

    misc = {} if misc_config is None else dict(OmegaConf.to_container(misc_config, resolve=True))
    transport_cfg = {} if transport_config is None else dict(OmegaConf.to_container(transport_config, resolve=True))
    sampler_cfg = {} if sampler_config is None else dict(OmegaConf.to_container(sampler_config, resolve=True))
    guidance_cfg = {} if guidance_config is None else dict(OmegaConf.to_container(guidance_config, resolve=True))

    guidance_scale, guidance_scale_aux, guidance_method, t_min, t_max = get_guidance_params(guidance_cfg)
    using_cfg = guidance_scale > 1.0

    guid_model_forward = None
    if using_cfg and guidance_method == "autoguidance":
        guid_model_config = guidance_cfg.get("guidance_model")
        if guid_model_config is None:
            raise ValueError("guidance_model must be provided for autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guid_model_config).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward

    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = int(misc.get("time_dist_shift_dim", math.prod(latent_size)))
    shift_base = int(misc.get("time_dist_shift_base", 4096))
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))

    source_rae: RAE = instantiate_from_config(source_rae_config).to(device)
    source_rae.eval()
    aux_state_spec = infer_aux_state_spec(source_rae, latent_size=latent_size)
    if aux_state_spec.enabled:
        raise ValueError("Recovered-aux sampler expects a patch-only Stage-2 baseline (aux disabled).")

    target_rae = build_stage1_from_source_config(
        target_stage1_cfg,
        device,
        decoder_ckpt=target_stage1_decoder_ckpt,
    )
    if str(getattr(target_rae, "decoder_aux_mode", "discard")) not in {"prepend", "cross_attn"}:
        raise ValueError(
            "Recovered-aux target decoder must consume aux tokens via prepend or cross_attn. "
            f"Got {target_rae.decoder_aux_mode!r}."
        )

    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    model.eval()
    ckpt = load_stage2_weights(model, ckpt_path, use_ema=use_ema, rank=rank)

    predictor_name, predictor_cfg, predictor, predictor_stats = load_predictor_bundle(
        config_path=predictor_cfg_path,
        ckpt_path=predictor_ckpt_path,
        device=device,
        model_name=str(predictor_model_name) if predictor_model_name is not None else None,
    )

    transport_params = dict(transport_cfg.get("params", {}))
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)
    transport_sampler = Sampler(transport)
    sampler_mode = str(sampler_cfg.get("mode", "ODE")).upper()
    sampler_params = dict(sampler_cfg.get("params", {}))
    if sampler_mode != "ODE":
        raise ValueError("Recovered-aux sampler currently supports only sampler.mode == ODE.")
    sample_fn = transport_sampler.sample_ode(**sampler_params)

    ensure_dir(sample_dir)
    out_folder = sample_dir / save_folder
    if rank == 0:
        ensure_dir(out_folder)
    dist.barrier()

    global_bs = per_proc_batch_size * world_size
    total_samples = compute_total_samples_rounded(num_samples, world_size, per_proc_batch_size)
    per_rank_samples = total_samples // world_size
    iterations = per_rank_samples // per_proc_batch_size

    if cond_mode == "equal_class":
        pool = build_equal_class_pool(
            num_classes=num_classes,
            total_samples=total_samples,
            num_requested=num_samples,
            seed=global_seed,
            null_label=null_label,
        )
    elif cond_mode == "random_class":
        rng = np.random.default_rng(global_seed)
        valid = [idx for idx in range(num_classes) if idx != null_label]
        y_pool = rng.choice(np.array(valid, dtype=np.int64), size=total_samples, replace=True)
        pool = ConditionPool(y=y_pool.astype(np.int64), meta=None)
    else:
        y_pool = np.full((total_samples,), int(null_label), dtype=np.int64)
        pool = ConditionPool(y=y_pool, meta=None)

    rank_view = shard_pool_for_rank(
        pool,
        rank=rank,
        world_size=world_size,
        per_rank_samples=per_rank_samples,
        batch_size=per_proc_batch_size,
    )
    rank_view_y = rank_view.y

    already_tensor = torch.zeros(1, device=device, dtype=torch.long)
    if rank == 0:
        existing_pngs = list(out_folder.glob("*.png"))
        already = (len(existing_pngs) // global_bs) * global_bs
        already_tensor[0] = already
        if already > 0:
            print(f"[resume] found {len(existing_pngs)} pngs, using already={already}")
    dist.broadcast(already_tensor, src=0)
    already = int(already_tensor.item())
    skip_iters = already // global_bs
    if skip_iters > iterations:
        raise RuntimeError(f"[resume] skip_iters={skip_iters} > iterations={iterations}")

    if rank == 0:
        y_vocab = infer_y_vocab(ckpt_path, ckpt)
        info = {
            "stage2_yaml": stage2_yaml,
            "ckpt_path": ckpt_path,
            "use_ema": use_ema,
            "target_stage1_config": target_stage1_cfg,
            "target_stage1_decoder_ckpt": target_stage1_decoder_ckpt,
            "predictor_name": predictor_name,
            "predictor_config": predictor_cfg_path,
            "predictor_ckpt": predictor_ckpt_path,
            "num_samples_requested": num_samples,
            "total_samples_rounded": total_samples,
            "per_proc_batch_size": per_proc_batch_size,
            "world_size": world_size,
            "precision": precision,
            "tf32": tf32,
            "global_seed": global_seed,
            "cond_mode": cond_mode,
            "num_classes": num_classes,
            "null_label": null_label,
            "sampler_mode": sampler_mode,
            "sampler_params": sampler_params,
            "guidance_scale": guidance_scale,
            "guidance_aux_scale": guidance_scale_aux,
            "guidance_method": guidance_method,
            "cfg_interval": [t_min, t_max],
            "latent_size": list(latent_size),
            "time_dist_shift": float(time_dist_shift),
            "source_stage1_config": str(source_rae_config.get("target", "stage1.RAE")),
            "source_decoder_aux_mode": str(getattr(source_rae, "decoder_aux_mode", "discard")),
            "target_decoder_aux_mode": str(getattr(target_rae, "decoder_aux_mode", "discard")),
            "y_vocab": y_vocab,
        }
        write_manifest_info(out_folder, info)

    dist.barrier()
    manifest_rank_path = out_folder / f"manifest_rank{rank:03d}.jsonl"
    writer = JsonlWriter(manifest_rank_path)

    start_index_base = already
    total_written = start_index_base
    progress = tqdm(range(skip_iters, iterations), desc="Sampling", unit="iter") if rank == 0 else range(skip_iters, iterations)

    try:
        for step_idx in progress:
            with autocast(**autocast_kwargs):
                state = make_initial_sample_state(
                    n=per_proc_batch_size,
                    latent_size=latent_size,
                    aux_state_spec=aux_state_spec,
                    device=device,
                )
                y_cond = rank_view_y[step_idx].to(device, non_blocking=True)

                model_fn = model.forward
                model_kwargs: Dict[str, Any] = {"y": y_cond}

                if using_cfg:
                    state = duplicate_state_for_guidance(state)
                    y_null = torch.full((per_proc_batch_size,), int(null_label), device=device, dtype=y_cond.dtype)
                    y_in = torch.cat([y_cond, y_null], dim=0)
                    model_kwargs = {
                        "y": y_in,
                        "cfg_scale": float(guidance_scale),
                        "cfg_scale_aux": float(guidance_scale_aux),
                        "cfg_interval": (float(t_min), float(t_max)),
                    }
                    if guidance_method == "autoguidance":
                        if guid_model_forward is None:
                            raise RuntimeError("Guidance model forward is not initialized.")
                        model_kwargs["additional_model_forward"] = guid_model_forward
                        model_fn = model.forward_with_autoguidance
                    else:
                        model_fn = model.forward_with_cfg

                sampled_state = final_state_from_trajectory(sample_fn(state, model_fn, **model_kwargs))
                if using_cfg:
                    sampled_state = split_guided_state(sampled_state)
                sampled_state = state_float(sampled_state)
                if not torch.is_tensor(sampled_state):
                    raise ValueError("Patch-only Stage-2 baseline should sample a tensor state, not a tuple.")

                source_tokens = latent_to_source_tokens(sampled_state, source_rae)
                source_tokens_std = standardize_source(source_tokens, predictor_stats).to(device, non_blocking=True)
                recovered_aux_std = predictor(source_tokens_std)
                recovered_aux = unstandardize_target(recovered_aux_std, predictor_stats)
                recovered_aux_norm = target_rae.normalize_aux_tokens(recovered_aux)

                target_z = convert_patch_latent_between_raes(
                    sampled_state,
                    source_rae=source_rae,
                    target_rae=target_rae,
                )
                imgs = target_rae.decode(
                    target_z,
                    aux_tokens=recovered_aux_norm,
                    aux_tokens_are_normalized=True,
                ).clamp(0, 1)
                imgs = imgs.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            y_cpu = y_cond.detach().cpu().long().numpy()
            indices = [total_written + local_idx * world_size + rank for local_idx in range(len(imgs))]
            if save_latent_shards:
                save_latent_shard(
                    out_folder,
                    latent_subdir=latent_shard_dir,
                    rank=rank,
                    step_idx=step_idx,
                    indices=indices,
                    y=y_cpu,
                    state=sampled_state,
                )

            for local_idx, sample in enumerate(imgs):
                index = indices[local_idx]
                filename = f"{index:06d}.png"
                Image.fromarray(sample).save(out_folder / filename)
                writer.write(
                    {
                        "index": int(index),
                        "filename": filename,
                        "y": int(y_cpu[local_idx]),
                        "cond_mode": cond_mode,
                        "rank": int(rank),
                        "seed": int(seed),
                    }
                )

            total_written += global_bs
            if rank == 0 and hasattr(progress, "set_postfix"):
                progress.set_postfix({"written": total_written})
    finally:
        writer.close()

    dist.barrier()

    if rank == 0:
        merge_rank_manifests(out_folder, world_size)
        if pack_npz:
            create_npz_from_sample_folder(out_folder, num=num_samples, out_name="samples.npz")
        print(f"[done] wrote recovered-aux samples to {out_folder}")

    dist.barrier()
    dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Sample a patch-only Stage-2 baseline, recover aux with MHAP, and decode with an aux-aware Stage-1 decoder.")
    parser.add_argument("--sample-config", type=str, required=True)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.sample_config), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError("sample-config must resolve to a dict.")
    main(cfg)
