#!/usr/bin/env python3
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
Extended to support optional joint stage-2 state:
    - patch latent z
    - pooled auxiliary conditioning vectors
    - token auxiliary conditioning banks
while remaining backward compatible with patch-only training.
"""

import argparse
import json
import logging
import math
import os
from collections import defaultdict, OrderedDict
from copy import deepcopy
from glob import glob
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image, make_grid
from omegaconf import OmegaConf

##### model imports
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport, Sampler
from stage2.state_utils import (
    Stage2State,
    decode_stage2_state,
    duplicate_state_for_guidance,
    final_state_from_trajectory,
    infer_aux_state_spec,
    make_initial_sample_state,
    normalize_stage2_state,
    split_guided_state,
    state_float,
)

##### general utils
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import *
from utils.wandb_utils import *
from utils.dist_utils import *

##### Eval utils
from eval import evaluate_generation_distributed


def _unwrap_dataset(ds):
    """
    Peel off common wrapper datasets (e.g. Subset) until we reach the base dataset.
    """
    visited = set()
    while True:
        ds_id = id(ds)
        if ds_id in visited:
            break
        visited.add(ds_id)

        if hasattr(ds, "dataset"):
            ds = ds.dataset
            continue
        if hasattr(ds, "base"):
            ds = ds.base
            continue
        break
    return ds

def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def _iter_float_tensors(state):
    if torch.is_tensor(state):
        if state.is_floating_point():
            yield state
        return

    if isinstance(state, dict):
        for value in state.values():
            yield from _iter_float_tensors(value)
        return

    if isinstance(state, (tuple, list)):
        for value in state:
            yield from _iter_float_tensors(value)


def _summarize_float_state(state: Any) -> Dict[str, float]:
    tensors = [t.detach().float() for t in _iter_float_tensors(state)]
    if not tensors:
        return {}

    numel = 0
    sum_v = 0.0
    sum_sq = 0.0
    absmax = 0.0
    nonfinite = 0.0
    for tensor in tensors:
        numel += tensor.numel()
        sum_v += tensor.sum().item()
        sum_sq += (tensor * tensor).sum().item()
        absmax = max(absmax, tensor.abs().max().item())
        nonfinite += (~torch.isfinite(tensor)).sum().item()

    mean = sum_v / max(1, numel)
    var = max(sum_sq / max(1, numel) - mean * mean, 0.0)
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "absmax": absmax,
        "nonfinite_frac": nonfinite / max(1, numel),
    }


def _collect_grad_debug(model: DDP) -> Dict[str, Any]:
    total_sq = 0.0
    max_abs = 0.0
    nonfinite_count = 0
    bad_names = []

    for name, param in model.module.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        grad_f = grad.detach().float()
        if not torch.isfinite(grad_f).all():
            nonfinite_count += (~torch.isfinite(grad_f)).sum().item()
            if len(bad_names) < 8:
                bad_names.append(name)
        total_sq += grad_f.pow(2).sum().item()
        max_abs = max(max_abs, grad_f.abs().max().item())

    return {
        "grad_norm": math.sqrt(total_sq) if total_sq > 0 else 0.0,
        "grad_absmax": max_abs,
        "grad_nonfinite_count": int(nonfinite_count),
        "grad_bad_param_names": bad_names,
    }


def _tensor_dict_to_floats(stats: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in stats.items():
        if torch.is_tensor(value):
            out[key] = float(value.detach().float().mean().item())
        else:
            out[key] = value
    return out


def _write_debug_snapshot(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 transport model on RAE latents.")
    parser.add_argument("--config", type=str, required=True, help="YAML config containing stage_1 and stage_2 sections.")
    parser.add_argument("--data-path", type=Path, required=True, help="Directory with ImageFolder structure for training.")
    parser.add_argument("--results-dir", type=str, default="ckpts", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=256, help="Input image resolution.")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32", help="Compute precision for training.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--compile", action="store_true", help="Use torch compile (for rae.encode / rae.encode_for_stage2 and model.forward).")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")
    args = parser.parse_args()
    return args


def main():
    """Trains a new SiT model using config-driven hyperparameters."""
    from contextlib import nullcontext

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")

    rank, world_size, device = setup_distributed()

    full_cfg = OmegaConf.load(args.config)

    # NOTE: parse_configs now returns data_config FIRST
    (
        data_config,
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
        eval_config,
    ) = parse_configs(full_cfg)

    if rae_config is None or model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 sections.")

    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)

    data_cfg = to_dict(data_config)
    misc = to_dict(misc_config)
    transport_cfg = to_dict(transport_config)
    sampler_cfg = to_dict(sampler_config)
    guidance_cfg = to_dict(guidance_config)
    training_cfg = to_dict(training_config)

    # -------------------------
    # Misc / sizes / seeds
    # -------------------------
    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")

    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None

    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    num_epochs = int(training_cfg.get("epochs", 1400))

    global_batch_size = training_cfg.get("global_batch_size", None)
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
    else:
        batch_size = int(training_cfg.get("batch_size", 16))
        global_batch_size = batch_size * world_size * grad_accum_steps

    num_workers = int(training_cfg.get("num_workers", 4))
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 2500))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4))
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    default_seed = int(training_cfg.get("global_seed", 0))
    debug_monitor_cfg = training_cfg.get("debug_monitor", {}) or {}
    debug_monitor_enabled = bool(debug_monitor_cfg.get("enabled", False))
    debug_monitor_log_interval = int(debug_monitor_cfg.get("log_interval", log_interval if log_interval > 0 else 1))
    debug_stop_on_nonfinite = bool(debug_monitor_cfg.get("stop_on_nonfinite", True))
    debug_capture_dir = None
    profiler_cfg = training_cfg.get("profiler", {}) or {}
    profiler_enabled = bool(profiler_cfg.get("enabled", False))
    profiler_wait = int(profiler_cfg.get("wait", 1))
    profiler_warmup = int(profiler_cfg.get("warmup", 1))
    profiler_active = int(profiler_cfg.get("active", 3))
    profiler_repeat = int(profiler_cfg.get("repeat", 1))
    profiler_record_shapes = bool(profiler_cfg.get("record_shapes", True))
    profiler_profile_memory = bool(profiler_cfg.get("profile_memory", True))
    profiler_with_stack = bool(profiler_cfg.get("with_stack", False))
    profiler_with_flops = bool(profiler_cfg.get("with_flops", False))
    profiler_trace_dir_cfg = profiler_cfg.get("trace_dir", None)
    profiler = None
    profiler_trace_dir = None

    # -------------------------
    # Eval config
    # -------------------------
    if eval_config:
        do_eval = bool(eval_config.get("do_eval", True))
        eval_interval = int(eval_config.get("eval_interval", 5000))
        eval_model = bool(eval_config.get("eval_model", False))
        eval_data = eval_config.get("data_path", None)
        reference_npz_path = eval_config.get("reference_npz_path", None)
        if do_eval:
            assert eval_data, "eval.data_path must be specified to enable evaluation."
            assert reference_npz_path, "eval.reference_npz_path must be specified to enable evaluation."
    else:
        do_eval = False
        eval_interval = 0
        eval_model = False
        eval_data = None
        reference_npz_path = None

    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)

    # AMP
    scaler, autocast_kwargs = get_autocast_scaler(args)

    # -------------------------
    # Guidance config
    # -------------------------
    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    if cfg_scale_override is not None:
        guidance_scale = float(cfg_scale_override)

    guidance_method = str(guidance_cfg.get("method", "cfg"))

    def guidance_value(key: str, default: float) -> float:
        if key in guidance_cfg:
            return float(guidance_cfg[key])
        dashed_key = key.replace("_", "-")
        return float(guidance_cfg.get(dashed_key, default))

    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))

    use_guidance = guidance_scale > 1.0

    # -------------------------
    # Experiment dirs / logger
    # -------------------------
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    if rank == 0 and debug_monitor_enabled:
        debug_capture_dir = os.path.join(experiment_dir, "debug_monitor")
        os.makedirs(debug_capture_dir, exist_ok=True)
    if profiler_enabled and rank == 0:
        profiler_trace_dir = str(profiler_trace_dir_cfg) if profiler_trace_dir_cfg else os.path.join(experiment_dir, "torch_profiler")
        os.makedirs(profiler_trace_dir, exist_ok=True)
        profiler_activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=profiler_activities,
            schedule=torch.profiler.schedule(
                wait=profiler_wait,
                warmup=profiler_warmup,
                active=profiler_active,
                repeat=profiler_repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                profiler_trace_dir,
                worker_name=f"rank{rank}",
            ),
            record_shapes=profiler_record_shapes,
            profile_memory=profiler_profile_memory,
            with_stack=profiler_with_stack,
            with_flops=profiler_with_flops,
        )

    # -------------------------
    # Model init
    # -------------------------
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    aux_state_spec = infer_aux_state_spec(rae, latent_size=latent_size)

    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    model_predict_aux = bool(getattr(model, "predict_aux", False))
    model_predict_aux_tokens = bool(getattr(model, "predict_aux_tokens", False))
    uses_joint_state = aux_state_spec.enabled

    if aux_state_spec.mode == "pooled" and not model_predict_aux:
        raise ValueError(
            "Stage-1 uses pooled auxiliary state, but Stage-2 model does not enable predict_aux=True."
        )
    if aux_state_spec.mode == "tokens" and not model_predict_aux_tokens:
        raise ValueError(
            "Stage-1 uses token auxiliary state, but Stage-2 model does not enable predict_aux_tokens=True."
        )

    if args.compile:
        try:
            if uses_joint_state and hasattr(rae, "encode_for_stage2"):
                rae.encode_for_stage2 = torch.compile(rae.encode_for_stage2)
            else:
                rae.encode = torch.compile(rae.encode)
        except Exception:
            if rank == 0:
                print("RAE encode compile failed; continuing without compile for RAE encode path")
        try:
            model.forward = torch.compile(model.forward)
        except Exception:
            if rank == 0:
                print("MODEL FORWARD compile failed; continuing without compile for model.forward")

    ema_model = deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()

    model.requires_grad_(True)
    transport_params_for_ddp = dict(transport_cfg.get("params", {})) if transport_cfg is not None else {}
    component_mode = str(transport_params_for_ddp.get("component_mode", "joint"))
    find_unused_parameters = bool(training_cfg.get("ddp_find_unused_parameters", False)) or component_mode == "aux_only"
    ddp_model = DDP(
        model,
        device_ids=[device.index],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters,
    )
    ddp_model.train()

    if rank == 0:
        model_param_count = sum(p.numel() for p in ddp_model.module.parameters() if p.requires_grad)
        logger.info(f"Stage-2 Model Parameters: {model_param_count/1e6:.2f}M")
        rae_param_count = sum(p.numel() for p in rae.parameters())
        logger.info(f"Stage-1 RAE parameters: {rae_param_count/1e6:.2f}M")
        logger.info(f"Stage-1 Stage-2 aux mode: {aux_state_spec.mode}")
        if uses_joint_state:
            logger.info(f"Stage-2 will use joint state: z + aux{aux_state_spec.shape}")
        else:
            logger.info("Stage-2 will use patch-latent-only state.")
            logger.info(
                "Patch-only Stage-2 targets will use full-precision rae.encode(images) "
                "for parity with the upstream LightningDiT training path."
            )
        if find_unused_parameters:
            if component_mode == "aux_only":
                logger.info("DDP unused-parameter detection enabled for component_mode='aux_only'.")
            else:
                logger.info("DDP unused-parameter detection enabled via training.ddp_find_unused_parameters=True.")
        if debug_monitor_enabled:
            logger.info(
                "Debug monitor enabled: "
                f"log_interval={debug_monitor_log_interval}, "
                f"stop_on_nonfinite={debug_stop_on_nonfinite}, "
                f"capture_dir={debug_capture_dir}"
            )
        if profiler_enabled:
            if profiler is not None:
                logger.info(
                    "Torch profiler enabled: "
                    f"trace_dir={profiler_trace_dir}, wait={profiler_wait}, warmup={profiler_warmup}, "
                    f"active={profiler_active}, repeat={profiler_repeat}, "
                    f"record_shapes={profiler_record_shapes}, profile_memory={profiler_profile_memory}, "
                    f"with_stack={profiler_with_stack}, with_flops={profiler_with_flops}"
                )
            else:
                logger.info("Torch profiler enabled in config but only rank 0 writes traces.")

    # -------------------------
    # Optim / sched
    # -------------------------
    optimizer, optim_msg = build_optimizer([p for p in ddp_model.module.parameters() if p.requires_grad], training_cfg)

    loader = None
    sampler = None
    scheduler = None
    sched_msg = None

    # -------------------------
    # Data init (YAML-governed via data_cfg)
    # -------------------------
    stage2_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    loader, sampler = prepare_dataloader(
        args.data_path,
        micro_batch_size,
        num_workers,
        rank,
        world_size,
        transform=stage2_transform,
        data_cfg=data_cfg,
    )

    # ---- dump y_vocab at experiment top-level (rank0 only) ----
    if rank == 0:
        exp_dir = Path(experiment_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)

        base_ds = _unwrap_dataset(loader.dataset)

        y_vocab = None

        if hasattr(base_ds, "class_to_idx") and isinstance(base_ds.class_to_idx, dict):
            y_vocab = {str(k): int(v) for k, v in base_ds.class_to_idx.items()}
        elif hasattr(base_ds, "y_vocab") and isinstance(base_ds.y_vocab, dict):
            y_vocab = {str(k): int(v) for k, v in base_ds.y_vocab.items()}
        elif hasattr(base_ds, "label_to_idx") and isinstance(base_ds.label_to_idx, dict):
            y_vocab = {str(k): int(v) for k, v in base_ds.label_to_idx.items()}

        if y_vocab is None:
            logger.warning("[vocab] could not find class_to_idx / y_vocab / label_to_idx; not writing y_vocab.json")
        else:
            out_path = exp_dir / "y_vocab.json"
            out_path.write_text(json.dumps(y_vocab, indent=2, sort_keys=True))
            logger.info(f"[vocab] wrote y_vocab -> {out_path} (n={len(y_vocab)})")

        meta_vocabs = None

        if hasattr(base_ds, "meta_vocabs") and isinstance(base_ds.meta_vocabs, dict):
            meta_vocabs = base_ds.meta_vocabs
        elif hasattr(base_ds, "meta_value_to_idx") and isinstance(base_ds.meta_value_to_idx, dict):
            meta_vocabs = base_ds.meta_value_to_idx
        elif hasattr(base_ds, "meta_field_to_vocab") and isinstance(base_ds.meta_field_to_vocab, dict):
            meta_vocabs = base_ds.meta_field_to_vocab

        if meta_vocabs is not None:
            out_path = exp_dir / "meta_vocabs.json"
            out_path.write_text(json.dumps(meta_vocabs, indent=2, sort_keys=True))
            logger.info(f"[vocab] wrote meta_vocabs -> {out_path}")
        else:
            logger.warning("[vocab] could not find meta vocabs; not writing meta_vocabs.json")

    dist.barrier()
    if do_eval:
        eval_dataset = ImageFolder(
            str(eval_data),
            transform=transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
                transforms.ToTensor(),
            ])
        )
        if rank == 0:
            logger.info(f"Evaluation dataset loaded from {eval_data}, containing {len(eval_dataset)} images.")
    else:
        eval_dataset = None

    # Current evaluator assumes patch-only latent sampling / decode.
    # Keep patch-only runs working; disable auto-eval for joint-state runs until the evaluator is updated.
    if do_eval and uses_joint_state:
        if rank == 0:
            logger.warning(
                "Automatic generation eval is currently patch-only. "
                "Disabling do_eval for joint-state runs until eval/evaluate_generation_distributed is updated."
            )
        do_eval = False
        eval_interval = 0
        eval_model = False
        eval_dataset = None

    loader_batches = len(loader)
    if loader_batches % grad_accum_steps != 0:
        raise ValueError("Number of loader batches must be divisible by grad_accum_steps when drop_last=True.")
    steps_per_epoch = loader_batches // grad_accum_steps
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")

    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)

    if rank == 0:
        logger.info(optim_msg)
        logger.info(sched_msg if sched_msg else "No LR scheduler.")
        logger.info(
            f"Training for {num_epochs} epochs, micro_batch_size {micro_batch_size} per GPU, "
            f"grad_accum_steps {grad_accum_steps}."
        )
        logger.info(
            f"Dataset contains {len(loader.dataset)} samples, loader_batches {loader_batches}, "
            f"optimizer_steps/epoch {steps_per_epoch}."
        )
        if clip_grad is not None:
            logger.info(f"Clipping gradients to max norm {clip_grad}.")
        else:
            logger.info("Not clipping gradients.")

    # -------------------------
    # Transport init
    # -------------------------
    transport_params = dict(transport_cfg.get("params", {}))
    transport_params.pop("time_dist_shift", None)

    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = Sampler(transport)

    sampler_mode = str(sampler_cfg.get("mode", "ODE")).upper()
    sampler_params = dict(sampler_cfg.get("params", {}))

    if sampler_mode == "ODE":
        eval_sampler = transport_sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        eval_sampler = transport_sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")

    # -------------------------
    # Guidance init
    # -------------------------
    guid_model_forward = None
    if use_guidance and guidance_method == "autoguidance":
        guidance_model_cfg = guidance_cfg.get("guidance_model")
        if guidance_model_cfg is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guidance_model_cfg).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward

    if use_guidance:
        sample_model_kwargs_base = dict(cfg_scale=guidance_scale, cfg_interval=(t_min, t_max))
        if guidance_method == "autoguidance":
            if guid_model_forward is None:
                raise RuntimeError("Guidance model forward is not initialized.")
            sample_model_kwargs_base["additional_model_forward"] = guid_model_forward
            ema_model_fn = ema_model.forward_with_autoguidance
            model_fn = ddp_model.module.forward_with_autoguidance
        else:
            ema_model_fn = ema_model.forward_with_cfg
            model_fn = ddp_model.module.forward_with_cfg
    else:
        sample_model_kwargs_base = dict()
        ema_model_fn = ema_model.forward
        model_fn = ddp_model.module.forward

    # -------------------------
    # Resume / checkpointing
    # -------------------------
    start_epoch = 0
    global_step = 0  # micro-step counter
    optim_step = 0   # optimizer-step counter

    explicit_ckpt_path = Path(args.ckpt).expanduser().resolve() if args.ckpt else None
    maybe_resume_ckpt_path = find_resume_checkpoint(experiment_dir)
    resume_ckpt_path = explicit_ckpt_path if explicit_ckpt_path is not None else maybe_resume_ckpt_path

    if resume_ckpt_path is not None:
        if explicit_ckpt_path is not None:
            logger.info(f"Explicit resume checkpoint provided: {resume_ckpt_path}")
        else:
            logger.info(f"Experiment resume checkpoint found at {resume_ckpt_path}, automatically resuming...")
        ckpt_path = Path(resume_ckpt_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        start_epoch, global_step = load_checkpoint(
            ckpt_path,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
        )
        optim_step = global_step // grad_accum_steps
        logger.info(
            f"[Rank {rank}] Resumed from {ckpt_path} "
            f"(epoch={start_epoch}, micro_step={global_step}, optim_step={optim_step})."
        )
    else:
        if rank == 0:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")

    # -------------------------
    # Training loop
    # -------------------------
    running_loss = 0.0
    terminate_training = False
    debug_snapshot_written = False

    dist.barrier()
    if profiler is not None:
        profiler.start()
    for epoch in range(start_epoch, num_epochs):
        if terminate_training:
            break
        ddp_model.train()
        sampler.set_epoch(epoch)

        epoch_metrics = defaultdict(lambda: torch.zeros(1, device=device))
        running_component_sums = defaultdict(float)
        num_batches = 0

        # epoch checkpointing (epoch-based)
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0 and rank == 0:
            logger.info(f"Saving checkpoint at epoch {epoch}...")
            ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt"
            save_checkpoint(
                ckpt_path,
                global_step,
                epoch,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
            )

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            # --- unpack batch (supports: images | (images,y) | (images,y,meta)) ---
            images = labels = meta = None
            if torch.is_tensor(batch):
                images = batch
            elif isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    images, labels = batch
                elif len(batch) == 3:
                    images, labels, meta = batch
                else:
                    raise ValueError(f"Unexpected batch len={len(batch)}")
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")

            if images is None:
                raise RuntimeError("No images in batch")

            images = images.to(device, non_blocking=True)
            if labels is not None:
                labels = labels.to(device, non_blocking=True)
            if meta is not None:
                meta = meta.to(device, non_blocking=True)

            # encode frozen stage-1 into stage-2 state
            with torch.profiler.record_function("train/stage1_encode"):
                with torch.no_grad():
                    if uses_joint_state:
                        with autocast(**autocast_kwargs):
                            encoded_state = rae.encode_for_stage2(images, normalize_global_cond=True)
                            transport_state = normalize_stage2_state(encoded_state)
                    else:
                        # Keep patch-only Stage-2 numerically aligned with the upstream path.
                        encoded_state = rae.encode(images)
                        transport_state = encoded_state

            # model kwargs
            model_kwargs = {}
            if labels is not None:
                model_kwargs["y"] = labels
            if meta is not None:
                model_kwargs["meta"] = meta

            step_debug_stats = {}
            if debug_monitor_enabled:
                encoded_summary = _summarize_float_state(encoded_state)
                transport_summary = _summarize_float_state(transport_state)
                step_debug_stats.update({f"encoded_{k}": v for k, v in encoded_summary.items()})
                step_debug_stats.update({f"transport_{k}": v for k, v in transport_summary.items()})

            # DDP no_sync for accumulation micro-steps
            is_update_step = ((step + 1) % grad_accum_steps == 0)
            sync_ctx = nullcontext() if is_update_step else ddp_model.no_sync()

            with sync_ctx:
                with autocast(**autocast_kwargs):
                    with torch.profiler.record_function("train/transport_loss"):
                        loss_terms = transport.training_losses(
                            ddp_model,
                            transport_state,
                            model_kwargs,
                            debug_monitor=debug_monitor_enabled,
                        )
                        loss = loss_terms["loss"].mean()
                        if debug_monitor_enabled and "debug" in loss_terms:
                            step_debug_stats.update(_tensor_dict_to_floats(loss_terms["debug"]))

                loss_float = loss.float()
                loss_scaled = loss_float / grad_accum_steps

                if debug_monitor_enabled:
                    step_debug_stats["loss"] = float(loss_float.item())
                    if "loss_patch" in loss_terms:
                        step_debug_stats["loss_patch"] = float(loss_terms["loss_patch"].mean().float().item())
                    if "loss_aux" in loss_terms:
                        step_debug_stats["loss_aux"] = float(loss_terms["loss_aux"].mean().float().item())

                if debug_monitor_enabled and not torch.isfinite(loss_float).all():
                    if rank == 0:
                        payload = {
                            "reason": "nonfinite_loss_pre_backward",
                            "epoch": int(epoch),
                            "step": int(global_step),
                            "optim_step": int(optim_step),
                            "is_update_step": bool(is_update_step),
                            "stats": step_debug_stats,
                        }
                        snapshot_path = os.path.join(debug_capture_dir, f"nonfinite_loss_step{global_step:07d}.json")
                        _write_debug_snapshot(snapshot_path, payload)
                        logger.error(
                            f"Non-finite loss detected before backward at epoch={epoch} step={global_step}. "
                            f"Snapshot written to {snapshot_path}"
                        )
                        ckpt_path = os.path.join(debug_capture_dir, f"nonfinite_loss_step{global_step:07d}.pt")
                        save_checkpoint(
                            ckpt_path,
                            global_step,
                            epoch,
                            ddp_model,
                            ema_model,
                            optimizer,
                            scheduler,
                        )
                    debug_snapshot_written = True
                    terminate_training = debug_stop_on_nonfinite
                    if debug_stop_on_nonfinite:
                        break

                with torch.profiler.record_function("train/backward"):
                    if scaler is not None:
                        scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()

            # clip and step only on update step
            if is_update_step:
                if clip_grad is not None:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                elif scaler is not None:
                    scaler.unscale_(optimizer)

                if debug_monitor_enabled:
                    step_debug_stats.update(_collect_grad_debug(ddp_model))
                    grad_norm = step_debug_stats["grad_norm"]
                    grad_nonfinite_count = int(step_debug_stats["grad_nonfinite_count"])
                    if (not math.isfinite(grad_norm)) or grad_nonfinite_count > 0:
                        if rank == 0 and not debug_snapshot_written:
                            payload = {
                                "reason": "nonfinite_grad_pre_step",
                                "epoch": int(epoch),
                                "step": int(global_step),
                                "optim_step": int(optim_step),
                                "is_update_step": True,
                                "stats": step_debug_stats,
                            }
                            snapshot_path = os.path.join(debug_capture_dir, f"nonfinite_grad_step{global_step:07d}.json")
                            _write_debug_snapshot(snapshot_path, payload)
                            logger.error(
                                f"Non-finite gradients detected before optimizer step at epoch={epoch} step={global_step}. "
                                f"Snapshot written to {snapshot_path}"
                            )
                            ckpt_path = os.path.join(debug_capture_dir, f"nonfinite_grad_step{global_step:07d}.pt")
                            save_checkpoint(
                                ckpt_path,
                                global_step,
                                epoch,
                                ddp_model,
                                ema_model,
                                optimizer,
                                scheduler,
                            )
                        debug_snapshot_written = True
                        terminate_training = debug_stop_on_nonfinite
                        if debug_stop_on_nonfinite:
                            break

                with torch.profiler.record_function("train/optimizer_step"):
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    if scheduler is not None:
                        scheduler.step()

                    update_ema(ema_model, ddp_model.module, decay=ema_decay)

                    optimizer.zero_grad(set_to_none=True)
                    optim_step += 1

            running_loss += loss_float.item()
            epoch_metrics["loss"] += loss_float.detach()
            for key in ("loss_patch", "loss_aux", "t_base_mean", "t_patch_mean", "t_aux_mean"):
                if key in loss_terms:
                    value = loss_terms[key].mean().float()
                    epoch_metrics[key] += value.detach()
                    running_component_sums[key] += value.item()
            if hasattr(ddp_model.module, "get_monitor_stats"):
                try:
                    with torch.profiler.record_function("train/model_monitor_stats"):
                        model_monitor_stats = ddp_model.module.get_monitor_stats()
                except Exception:
                    model_monitor_stats = {}
                for key, value in model_monitor_stats.items():
                    if isinstance(value, (int, float)):
                        epoch_metrics[key] += float(value)
                        running_component_sums[key] += float(value)

            if debug_monitor_enabled and debug_monitor_log_interval > 0 and global_step % debug_monitor_log_interval == 0 and rank == 0:
                logger.info(
                    f"[DEBUG Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.6f}" for k, v in sorted(step_debug_stats.items()) if isinstance(v, (int, float)))
                )

            # logging keyed on micro-steps
            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                avg_loss = running_loss / max(1, log_interval)
                stats = {
                    "train/loss": avg_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/optim_step": float(optim_step),
                }
                for key, total in running_component_sums.items():
                    stats[f"train/{key}"] = total / max(1, log_interval)
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.6f}" for k, v in stats.items())
                )
                if args.wandb:
                    wandb_utils.log(stats, step=global_step)
                running_loss = 0.0
                running_component_sums = defaultdict(float)

            # sampling keyed on micro-steps
            if sample_every > 0 and global_step % sample_every == 0:
                if labels is not None:
                    ddp_model.eval()
                    logger.info("Generating EMA samples...")

                    with torch.no_grad():
                        n_vis = min(8, images.shape[0])
                        state_vis = make_initial_sample_state(
                            n=n_vis,
                            latent_size=latent_size,
                            aux_state_spec=aux_state_spec,
                            device=device,
                        )

                        sample_kwargs = dict(sample_model_kwargs_base)

                        # build y/meta for CFG (cond + null)
                        y_vis = labels[:n_vis]
                        meta_vis = meta[:n_vis] if meta is not None else None

                        if use_guidance:
                            state_in = duplicate_state_for_guidance(state_vis)

                            y_null = torch.full((n_vis,), null_label, device=device, dtype=y_vis.dtype)
                            y_in = torch.cat([y_vis, y_null], dim=0)
                            sample_kwargs["y"] = y_in

                            if meta_vis is not None:
                                # convention: 0 = MISSING/null per field
                                meta_null = torch.zeros_like(meta_vis)
                                meta_in = torch.cat([meta_vis, meta_null], dim=0)
                                sample_kwargs["meta"] = meta_in
                        else:
                            state_in = state_vis
                            sample_kwargs["y"] = y_vis
                            if meta_vis is not None:
                                sample_kwargs["meta"] = meta_vis

                        with autocast(**autocast_kwargs):
                            sampled_state = final_state_from_trajectory(
                                eval_sampler(state_in, ema_model_fn, **sample_kwargs)
                            )

                        if use_guidance:
                            sampled_state = split_guided_state(sampled_state)

                        sampled_state = state_float(sampled_state)
                        imgs = decode_stage2_state(rae, sampled_state).clamp(0, 1).cpu().float()

                        dist.barrier()

                        if rank == 0:
                            sample_dir = Path(experiment_dir) / "ema_samples"
                            sample_dir.mkdir(parents=True, exist_ok=True)

                            nrow = min(4, n_vis)
                            grid = make_grid(imgs[:n_vis], nrow=nrow)
                            grid_path = sample_dir / f"epoch{epoch:04d}_step{global_step:07d}.png"
                            save_image(grid, grid_path)

                            logger.info(f"Saved EMA sample grid to {grid_path}")

                            if args.wandb:
                                wandb_utils.log_image(imgs, global_step)

                    logger.info("Generating EMA samples done.")
                    ddp_model.train()

            # evaluation keyed on micro-steps
            if do_eval and eval_interval > 0 and global_step % eval_interval == 0:
                logger.info("Starting evaluation...")
                ddp_model.eval()

                eval_models = [(ema_model_fn, "ema")]
                if eval_model:
                    eval_models.append((model_fn, "model"))

                for fn, mod_name in eval_models:
                    eval_stats = evaluate_generation_distributed(
                        fn,
                        eval_sampler,
                        latent_size,
                        sample_model_kwargs_base,
                        use_guidance,
                        null_label,
                        rae,
                        eval_dataset,
                        len(eval_dataset),
                        rank=rank,
                        world_size=world_size,
                        device=device,
                        batch_size=micro_batch_size,
                        experiment_dir=experiment_dir,
                        global_step=global_step,
                        autocast_kwargs=autocast_kwargs,
                        reference_npz_path=reference_npz_path,
                    )
                    eval_stats = {f"eval_{mod_name}/{k}": v for k, v in eval_stats.items()} if eval_stats else {}
                    if args.wandb:
                        wandb_utils.log(eval_stats, step=global_step)

                ddp_model.train()
                logger.info("Evaluation done.")

            global_step += 1
            num_batches += 1
            if profiler is not None:
                profiler.step()

        if terminate_training:
            if rank == 0:
                logger.warning(f"Stopping training early at epoch={epoch} due to debug monitor non-finite detection.")
            break

        # epoch logging
        if rank == 0 and num_batches > 0:
            avg_loss = (epoch_metrics["loss"].item() / num_batches)
            epoch_stats = {"epoch/loss": avg_loss, "epoch/optim_steps": float(optim_step)}
            if "loss_patch" in epoch_metrics:
                epoch_stats["epoch/loss_patch"] = epoch_metrics["loss_patch"].item() / num_batches
            if "loss_aux" in epoch_metrics:
                epoch_stats["epoch/loss_aux"] = epoch_metrics["loss_aux"].item() / num_batches
            if "t_base_mean" in epoch_metrics:
                epoch_stats["epoch/t_base_mean"] = epoch_metrics["t_base_mean"].item() / num_batches
            if "t_patch_mean" in epoch_metrics:
                epoch_stats["epoch/t_patch_mean"] = epoch_metrics["t_patch_mean"].item() / num_batches
            if "t_aux_mean" in epoch_metrics:
                epoch_stats["epoch/t_aux_mean"] = epoch_metrics["t_aux_mean"].item() / num_batches
            for key in (
                "cross_gate_patch_absmean",
                "cross_gate_aux_absmean",
                "patch_stream_absmean",
                "aux_stream_absmean",
                "patch_output_absmean",
                "aux_output_absmean",
            ):
                if key in epoch_metrics:
                    epoch_stats[f"epoch/{key}"] = epoch_metrics[key].item() / num_batches
            logger.info(
                f"[Epoch {epoch}] " + ", ".join(f"{k}: {v:.6f}" for k, v in epoch_stats.items())
            )
            if args.wandb:
                wandb_utils.log(epoch_stats, step=global_step)

    # save final ckpt
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-last.pt"
        save_checkpoint(
            ckpt_path,
            global_step,
            num_epochs,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
        )

    if profiler is not None:
        profiler.stop()

    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
