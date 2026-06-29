from __future__ import annotations

import argparse
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from register_prediction.metrics import (  # noqa: E402
    TOKEN_NAMES,
    RegisterMetricAccumulator,
    infer_shapes,
    iter_feature_batches,
    load_stats,
    load_target_bank,
    save_json,
    standardize_source,
    standardize_target,
    unstandardize_target,
    write_metrics_csv,
)
from register_prediction.models import build_model  # noqa: E402


def load_cfg(path: str) -> Dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def cache_root(cfg: Dict) -> Path:
    output_root = Path(cfg["experiment"]["output_root"])
    return Path(cfg.get("cache", {}).get("dir", output_root / "cache"))


def train_stats_path(cfg: Dict) -> Path:
    return cache_root(cfg) / "train" / "stats.pt"


def autocast_context(device: torch.device, precision: str):
    precision = str(precision).lower()
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def model_names_from_cfg(cfg: Dict, override: Optional[str]) -> List[str]:
    if override:
        return [override]
    names = cfg.get("training", {}).get("model_names")
    if names is None:
        names = [cfg.get("model", {}).get("name", "mhap")]
    return [str(name) for name in names]


def build_model_for_eval(model_name: str, cfg: Dict, ckpt: Optional[Dict], cache_dir: Path) -> torch.nn.Module:
    if ckpt is not None and "model_kwargs" in ckpt:
        kwargs = ckpt["model_kwargs"]
        return build_model(
            model_name=model_name,
            source_dim=int(kwargs["source_dim"]),
            target_dim=int(kwargs["target_dim"]),
            num_slots=int(kwargs["num_slots"]),
            model_cfg=dict(kwargs.get("model_cfg", cfg.get("model", {}))),
        )
    source_dim, target_dim, num_slots, _num_source_tokens = infer_shapes(cache_dir)
    return build_model(
        model_name=model_name,
        source_dim=source_dim,
        target_dim=target_dim,
        num_slots=num_slots,
        model_cfg=dict(cfg.get("model", {})),
    )


def maybe_save_attention_maps(attn_maps: List[torch.Tensor], image_paths: List[str], out_dir: Path, num_samples: int) -> int:
    if num_samples <= 0 or not attn_maps:
        return 0
    last = attn_maps[-1].detach().float().cpu()
    if last.ndim != 4:
        return 0
    attn = last.mean(dim=1)  # [B, K, N]
    num_tokens = int(attn.shape[-1])
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        return 0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for sample_idx in range(min(num_samples, attn.shape[0])):
        fig, axes = plt.subplots(1, attn.shape[1], figsize=(3 * attn.shape[1], 3), squeeze=False)
        for slot_idx, token_name in enumerate(TOKEN_NAMES[: attn.shape[1]]):
            ax = axes[0][slot_idx]
            ax.imshow(attn[sample_idx, slot_idx].reshape(side, side), cmap="magma")
            ax.set_title(token_name)
            ax.axis("off")
        if image_paths:
            fig.suptitle(Path(image_paths[sample_idx]).name)
        fig.tight_layout()
        fig.savefig(out_dir / f"attn_{sample_idx:03d}.png", dpi=150)
        plt.close(fig)
        saved += 1
    return saved


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    model_name: str,
    cfg: Dict,
    ckpt_path: Path,
    split: str,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict:
    eval_cfg = cfg.get("eval", {})
    training_cfg = cfg.get("training", {})
    cache_dir = cache_root(cfg) / split
    stats = load_stats(train_stats_path(cfg))
    batch_size = int(eval_cfg.get("batch_size", training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 256))))
    precision = str(eval_cfg.get("precision", training_cfg.get("precision", "bf16")))
    retrieval = bool(eval_cfg.get("retrieval", True)) and not args.no_retrieval
    target_bank = bank_indices = None
    if retrieval:
        target_bank, bank_indices = load_target_bank(cache_dir, device=device)

    model.eval()
    accumulator = RegisterMetricAccumulator(TOKEN_NAMES)
    total_std_loss = 0.0
    total_count = 0
    saved_attn = 0
    requested_attn = int(args.save_attn_grids if args.save_attn_grids is not None else eval_cfg.get("save_attn_grids", 0))
    attn_out_dir = ckpt_path.parent.parent / "attention_maps" / split

    for batch in iter_feature_batches(cache_dir, batch_size=batch_size, shuffle=False, seed=int(eval_cfg.get("seed", 0))):
        source = standardize_source(batch["source_tokens"], stats).to(device, non_blocking=True)
        target_std = standardize_target(batch["target_tokens"], stats).to(device, non_blocking=True)
        with autocast_context(device, precision):
            if requested_attn > saved_attn:
                pred_std, aux = model(source, return_attn=True)
            else:
                pred_std = model(source)
                aux = {"attn": []}
            loss = F.mse_loss(pred_std.float(), target_std.float(), reduction="none").mean(dim=(1, 2))
        total_std_loss += float(loss.sum().item())
        total_count += int(loss.numel())
        pred = unstandardize_target(pred_std, stats)
        target = batch["target_tokens"].float().to(device, non_blocking=True)
        accumulator.update(
            pred=pred,
            target=target,
            indices=batch["indices"].to(device),
            target_bank=target_bank,
            bank_indices=bank_indices,
        )
        if requested_attn > saved_attn and aux.get("attn"):
            saved_attn += maybe_save_attention_maps(
                aux["attn"],
                list(batch.get("image_paths", [])),
                attn_out_dir,
                requested_attn - saved_attn,
            )

    rows = accumulator.rows()
    run_dir = ckpt_path.parent.parent
    metrics_path = run_dir / f"eval_metrics_{split}.csv"
    write_metrics_csv(rows, metrics_path)
    summary = {
        "model": model_name,
        "ckpt": str(ckpt_path),
        "split": split,
        "standardized_mse": total_std_loss / max(total_count, 1),
        "metrics_csv": str(metrics_path),
        "attention_maps_saved": saved_attn,
        "rows": rows,
    }
    save_json(summary, run_dir / f"eval_summary_{split}.json")
    print(f"[{model_name}] {split} standardized_mse={summary['standardized_mse']:.6f} metrics={metrics_path}")
    return summary


def evaluate_checkpoint(ckpt_path: Path, model_name: Optional[str], cfg: Dict, args: argparse.Namespace) -> Dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    resolved_model_name = model_name or str(ckpt.get("model_name", cfg.get("model", {}).get("name", "mhap")))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model_for_eval(resolved_model_name, cfg, ckpt, cache_root(cfg) / args.split).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    return evaluate_model(model, resolved_model_name, cfg, ckpt_path, args.split, device, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Evaluate SigLIP2-to-DINO register predictors.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint to evaluate. If omitted, evaluate best.pt for configured models.")
    parser.add_argument("--model", type=str, default=None, help="Model name for --ckpt or model subset when --ckpt is omitted.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--save-attn-grids", type=int, default=None)
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_cfg(args.config)
    summaries = []
    if args.ckpt:
        summaries.append(evaluate_checkpoint(Path(args.ckpt), args.model, cfg, args))
    else:
        output_root = Path(cfg["experiment"]["output_root"])
        for model_name in model_names_from_cfg(cfg, args.model):
            ckpt_path = output_root / "runs" / model_name / "checkpoints" / "best.pt"
            if not ckpt_path.exists():
                ckpt_path = output_root / "runs" / model_name / "checkpoints" / "last.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"No best.pt or last.pt checkpoint found for {model_name} under {ckpt_path.parent}")
            summaries.append(evaluate_checkpoint(ckpt_path, model_name, cfg, args))
    save_json({"summaries": summaries}, Path(cfg["experiment"]["output_root"]) / f"eval_summary_{args.split}.json")


if __name__ == "__main__":
    main()
