from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from register_prediction.metrics import (  # noqa: E402
    infer_shapes,
    iter_feature_batches,
    load_json,
    load_stats,
    save_json,
    standardize_source,
    standardize_target,
)
from register_prediction.models import build_model  # noqa: E402


def load_cfg(path: str) -> Dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cache_root(cfg: Dict) -> Path:
    output_root = Path(cfg["experiment"]["output_root"])
    return Path(cfg.get("cache", {}).get("dir", output_root / "cache"))


def train_stats_path(cfg: Dict) -> Path:
    return cache_root(cfg) / "train" / "stats.pt"


def get_model_names(cfg: Dict, override: Optional[str]) -> List[str]:
    if override and override != "all":
        return [override]
    names = cfg.get("training", {}).get("model_names")
    if names is None:
        names = [cfg.get("model", {}).get("name", "mhap")]
    return [str(name) for name in names]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict, steps_per_epoch: int) -> LambdaLR:
    training_cfg = cfg.get("training", {})
    sched_cfg = training_cfg.get("scheduler", {})
    schedule_type = str(sched_cfg.get("type", "cosine")).lower()
    epochs = int(training_cfg.get("epochs", 100))
    total_steps = max(epochs * max(steps_per_epoch, 1), 1)
    warmup_steps = int(float(sched_cfg.get("warmup_epochs", 0)) * max(steps_per_epoch, 1))
    base_lr = float(training_cfg.get("lr", training_cfg.get("optimizer", {}).get("lr", 1.0e-4)))
    final_lr = float(sched_cfg.get("final_lr", training_cfg.get("final_lr", base_lr * 0.1)))
    final_ratio = final_lr / base_lr if base_lr > 0 else 1.0

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if schedule_type == "linear":
            denom = max(total_steps - warmup_steps, 1)
            progress = min(max(step - warmup_steps, 0) / denom, 1.0)
            return 1.0 - (1.0 - final_ratio) * progress
        if schedule_type == "cosine":
            denom = max(total_steps - warmup_steps, 1)
            progress = min(max(step - warmup_steps, 0) / denom, 1.0)
            return final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        if schedule_type in {"none", "constant"}:
            return 1.0
        raise ValueError(f"Unsupported scheduler type '{schedule_type}'.")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def autocast_context(device: torch.device, precision: str):
    precision = str(precision).lower()
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def estimate_steps(cache_dir: Path, batch_size: int) -> int:
    manifest = load_json(cache_dir / "manifest.json")
    return int(math.ceil(int(manifest["num_samples"]) / float(batch_size)))


def compute_training_loss(pred: torch.Tensor, target: torch.Tensor, cfg: Dict) -> torch.Tensor:
    training_cfg = cfg.get("training", {})
    loss_cfg = training_cfg.get("loss", {})
    mse_weight = float(loss_cfg.get("mse_weight", 1.0))
    cosine_weight = float(loss_cfg.get("cosine_weight", 0.0))

    pred_f = pred.float()
    target_f = target.float()

    mse = F.mse_loss(pred_f, target_f)
    if cosine_weight <= 0.0:
        return mse_weight * mse

    cosine_mode = str(loss_cfg.get("cosine_mode", "per_slot")).lower()
    if cosine_mode == "per_slot":
        cosine_term = 1.0 - F.cosine_similarity(pred_f, target_f, dim=-1).mean()
    elif cosine_mode in {"flat", "flattened", "global"}:
        cosine_term = 1.0 - F.cosine_similarity(
            pred_f.reshape(pred_f.shape[0], -1),
            target_f.reshape(target_f.shape[0], -1),
            dim=-1,
        ).mean()
    else:
        raise ValueError(f"Unsupported cosine_mode '{cosine_mode}'.")

    return mse_weight * mse + cosine_weight * cosine_term


def build_predictor(model_name: str, cfg: Dict, train_cache: Path) -> Tuple[torch.nn.Module, Dict]:
    source_dim, target_dim, num_slots, num_source_tokens = infer_shapes(train_cache)
    model_cfg = dict(cfg.get("model", {}))
    cfg_slots = model_cfg.get("num_slots")
    if cfg_slots is not None and int(cfg_slots) != num_slots:
        raise ValueError(f"Config model.num_slots={cfg_slots} but cache target has {num_slots} slots.")
    model_kwargs = {
        "source_dim": source_dim,
        "target_dim": target_dim,
        "num_slots": num_slots,
        "num_source_tokens": num_source_tokens,
        "model_cfg": model_cfg,
    }
    model = build_model(
        model_name=model_name,
        source_dim=source_dim,
        target_dim=target_dim,
        num_slots=num_slots,
        model_cfg=model_cfg,
    )
    return model, model_kwargs


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    cfg: Dict,
    cache_dir: Path,
    stats: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    precision: str,
    seed: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for batch in iter_feature_batches(cache_dir, batch_size=batch_size, shuffle=False, seed=seed):
        source = standardize_source(batch["source_tokens"], stats).to(device, non_blocking=True)
        target = standardize_target(batch["target_tokens"], stats).to(device, non_blocking=True)
        with autocast_context(device, precision):
            pred = model(source)
            loss = compute_training_loss(pred, target, cfg)
        total_loss += float(loss.item()) * int(source.shape[0])
        total_count += int(source.shape[0])
    return total_loss / max(total_count, 1)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    model_name: str,
    model_kwargs: Dict,
    epoch: int,
    step: int,
    cfg: Dict,
    best_val_loss: float,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "epoch": int(epoch),
            "step": int(step),
            "config": cfg,
            "best_val_loss": float(best_val_loss),
        },
        path,
    )


def append_train_row(path: Path, row: Dict) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def maybe_resume_training(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
):
    if not ckpt_path.exists():
        return 0, 0, float("inf")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    global_step = int(ckpt.get("step", 0))
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    return start_epoch, global_step, best_val_loss


def train_one_model(model_name: str, cfg: Dict, args: argparse.Namespace) -> Path:
    training_cfg = cfg.get("training", {})
    output_root = Path(cfg["experiment"]["output_root"])
    run_dir = output_root / "runs" / model_name
    ckpt_dir = run_dir / "checkpoints"
    if run_dir.exists() and (args.overwrite or bool(training_cfg.get("overwrite", False))):
        shutil.rmtree(run_dir)
    ensure_dir(ckpt_dir)
    OmegaConf.save(OmegaConf.create(cfg), run_dir / "config.yaml")

    seed = int(training_cfg.get("seed", cfg.get("experiment", {}).get("seed", 0)))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = bool(training_cfg.get("tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(training_cfg.get("tf32", True))

    train_cache = cache_root(cfg) / "train"
    val_cache = cache_root(cfg) / "val"
    stats = load_stats(train_stats_path(cfg))
    model, model_kwargs = build_predictor(model_name, cfg, train_cache)
    model = model.to(device)

    lr = float(training_cfg.get("lr", training_cfg.get("optimizer", {}).get("lr", 1.0e-4)))
    betas = tuple(float(v) for v in training_cfg.get("betas", training_cfg.get("optimizer", {}).get("betas", [0.9, 0.95])))
    weight_decay = float(training_cfg.get("weight_decay", training_cfg.get("optimizer", {}).get("weight_decay", 0.01)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    batch_size = int(training_cfg.get("batch_size", 256))
    epochs = int(training_cfg.get("epochs", 100))
    precision = str(training_cfg.get("precision", "bf16"))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    log_interval = int(training_cfg.get("log_interval", 50))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 10))
    val_interval = int(training_cfg.get("val_interval", 1))
    steps_per_epoch = estimate_steps(train_cache, batch_size)
    scheduler = make_scheduler(optimizer, cfg, steps_per_epoch=steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and precision == "fp16"))

    last_ckpt_path = ckpt_dir / "last.pt"
    start_epoch, global_step, best_val_loss = maybe_resume_training(
        last_ckpt_path,
        model,
        optimizer,
        scheduler,
    )
    if start_epoch > 0:
        print(
            f"[{model_name}] resuming from {last_ckpt_path} "
            f"at epoch={start_epoch + 1}/{epochs} step={global_step} best={best_val_loss:.6f}",
            flush=True,
        )
    elif last_ckpt_path.exists():
        print(f"[{model_name}] found checkpoint at {last_ckpt_path} but resume state was empty; starting fresh", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for step_in_epoch, batch in enumerate(
            iter_feature_batches(train_cache, batch_size=batch_size, shuffle=True, seed=seed + epoch, drop_last=False),
            start=1,
        ):
            source = standardize_source(batch["source_tokens"], stats).to(device, non_blocking=True)
            target = standardize_target(batch["target_tokens"], stats).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                pred = model(source)
                loss = compute_training_loss(pred, target, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for {model_name} at epoch={epoch}, step={step_in_epoch}: {loss.item()}")
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bsz = int(source.shape[0])
            train_loss_sum += float(loss.item()) * bsz
            train_count += bsz
            global_step += 1
            if log_interval > 0 and global_step % log_interval == 0:
                print(
                    f"[{model_name}] epoch={epoch + 1}/{epochs} step={global_step} "
                    f"loss={loss.item():.6f} lr={optimizer.param_groups[0]['lr']:.6e}",
                    flush=True,
                )

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = float("nan")
        if val_cache.exists() and val_interval > 0 and ((epoch + 1) % val_interval == 0):
            val_loss = evaluate_loss(
                model,
                cfg,
                val_cache,
                stats,
                device=device,
                batch_size=int(training_cfg.get("eval_batch_size", batch_size)),
                precision=precision,
                seed=seed,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, model_name, model_kwargs, epoch, global_step, cfg, best_val_loss)
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, model_name, model_kwargs, epoch, global_step, cfg, best_val_loss)
        if checkpoint_interval > 0 and ((epoch + 1) % checkpoint_interval == 0):
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch + 1:04d}.pt",
                model,
                optimizer,
                scheduler,
                model_name,
                model_kwargs,
                epoch,
                global_step,
                cfg,
                best_val_loss,
            )
        row = {
            "model": model_name,
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_train_row(run_dir / "train_log.csv", row)
        print(
            f"[{model_name}] epoch={epoch + 1}/{epochs} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} best={best_val_loss:.6f}",
            flush=True,
        )

    best_path = ckpt_dir / "best.pt"
    selected_ckpt = best_path if best_path.exists() else ckpt_dir / "last.pt"
    summary = {
        "model": model_name,
        "run_dir": str(run_dir),
        "last_ckpt": str(ckpt_dir / "last.pt"),
        "best_ckpt": str(selected_ckpt),
        "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
        "epochs": epochs,
        "global_step": global_step,
    }
    save_json(summary, run_dir / "summary.json")
    return selected_ckpt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Train SigLIP2-to-DINO register predictors from cached features.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, help="Model to train: mhap, mean_mlp, or all. Defaults to config training.model_names.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_cfg(args.config)
    ckpts = {}
    for model_name in get_model_names(cfg, args.model):
        ckpts[model_name] = str(train_one_model(model_name, cfg, args))
    save_json(ckpts, Path(cfg["experiment"]["output_root"]) / "trained_checkpoints.json")


if __name__ == "__main__":
    main()
