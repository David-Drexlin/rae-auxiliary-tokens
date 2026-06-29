#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def mean_std_ci(values: Sequence[float], ci_alpha: float = 0.95) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    lo_q = (1.0 - ci_alpha) / 2.0
    hi_q = 1.0 - lo_q
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "ci_low": float(np.quantile(arr, lo_q)),
        "ci_high": float(np.quantile(arr, hi_q)),
    }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_embeddings(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    cls = torch.as_tensor(data["cls"], dtype=torch.float32)
    regs = torch.as_tensor(data["regs"], dtype=torch.float32)
    labels = torch.as_tensor(data["labels"], dtype=torch.long)
    class_names = [str(x) for x in data.get("class_names", [])]
    if regs.ndim != 3:
        raise ValueError(f"Expected regs to have shape [N, K, D], got {tuple(regs.shape)}")
    if cls.ndim != 2:
        raise ValueError(f"Expected cls to have shape [N, D], got {tuple(cls.shape)}")
    if regs.shape[0] != cls.shape[0] or labels.shape[0] != cls.shape[0]:
        raise ValueError("cls, regs, and labels have inconsistent first dimensions.")
    if regs.shape[2] != cls.shape[1]:
        raise ValueError("CLS and register hidden dimensions do not match.")
    return cls, regs, labels, class_names


def l2_normalize_tokens(cls: torch.Tensor, regs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return F.normalize(cls, dim=-1), F.normalize(regs, dim=-1)


def stratified_splits(labels: torch.Tensor, splits: int, test_size: float, seed: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    labels_np = labels.cpu().numpy()
    classes = np.unique(labels_np)
    for split_id in range(int(splits)):
        rng = np.random.default_rng(seed + split_id)
        train_idx: List[int] = []
        test_idx: List[int] = []
        for class_id in classes:
            idx = np.flatnonzero(labels_np == class_id)
            idx = idx.copy()
            rng.shuffle(idx)
            n_test = int(round(float(test_size) * len(idx)))
            n_test = min(max(n_test, 1), max(len(idx) - 1, 1))
            test_idx.extend(idx[:n_test].tolist())
            train_idx.extend(idx[n_test:].tolist())
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        yield np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def mlp(dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim),
    )


class ProbeOutput:
    def __init__(self, logits: torch.Tensor, fused: torch.Tensor | None = None, gates: torch.Tensor | None = None):
        self.logits = logits
        self.fused = fused
        self.gates = gates


class LinearViewProbe(nn.Module):
    def __init__(self, view: str, dim: int, num_regs: int, num_classes: int):
        super().__init__()
        self.view = view
        in_dim = dim * num_regs if view == "reg_concat_linear" else dim
        self.head = nn.Linear(in_dim, num_classes)

    def select(self, cls: torch.Tensor, regs: torch.Tensor) -> torch.Tensor:
        if self.view == "cls_linear":
            return cls
        if self.view == "reg_mean_linear":
            return F.normalize(regs.mean(dim=1), dim=-1)
        if self.view == "reg_concat_linear":
            return regs.reshape(regs.shape[0], -1)
        raise ValueError(f"Unsupported linear view: {self.view}")

    def forward(self, cls: torch.Tensor, regs: torch.Tensor) -> ProbeOutput:
        x = self.select(cls, regs)
        fused = x if x.shape[-1] == cls.shape[-1] else None
        return ProbeOutput(self.head(x), fused=fused)


class ResidualMLPProbe(nn.Module):
    def __init__(self, source: str, dim: int, hidden_dim: int, num_classes: int, gate_init: float):
        super().__init__()
        self.source = source
        self.delta = mlp(dim, hidden_dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.head = nn.Linear(dim, num_classes)

    def source_tokens(self, cls: torch.Tensor, regs: torch.Tensor) -> torch.Tensor:
        if self.source == "cls":
            return cls
        if self.source == "reg_mean":
            return F.normalize(regs.mean(dim=1), dim=-1)
        raise ValueError(f"Unsupported residual source: {self.source}")

    def forward(self, cls: torch.Tensor, regs: torch.Tensor) -> ProbeOutput:
        delta = self.delta(self.source_tokens(cls, regs))
        fused = F.normalize(cls + self.gate * delta, dim=-1)
        return ProbeOutput(self.head(fused), fused=fused, gates=self.gate.detach().reshape(1))


class ResidualGatedRegisterProbe(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_regs: int, num_classes: int, gate_init: float):
        super().__init__()
        self.transforms = nn.ModuleList([mlp(dim, hidden_dim) for _ in range(num_regs)])
        self.gates = nn.Parameter(torch.full((num_regs,), float(gate_init)))
        self.head = nn.Linear(dim, num_classes)

    def forward(self, cls: torch.Tensor, regs: torch.Tensor) -> ProbeOutput:
        delta = torch.zeros_like(cls)
        for idx, transform in enumerate(self.transforms):
            delta = delta + self.gates[idx] * transform(regs[:, idx])
        fused = F.normalize(cls + delta, dim=-1)
        return ProbeOutput(self.head(fused), fused=fused, gates=self.gates.detach())


class CLSQueryRegisterAttentionProbe(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_classes: int, gate_init: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.reg_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.head = nn.Linear(dim, num_classes)

    def forward(self, cls: torch.Tensor, regs: torch.Tensor) -> ProbeOutput:
        q = self.query_norm(cls).unsqueeze(1)
        kv = self.reg_norm(regs)
        delta, _ = self.attn(q, kv, kv, need_weights=False)
        fused = F.normalize(cls + self.gate * delta.squeeze(1), dim=-1)
        return ProbeOutput(self.head(fused), fused=fused, gates=self.gate.detach().reshape(1))


def build_model(
    variant: str,
    dim: int,
    num_regs: int,
    num_classes: int,
    hidden_dim: int,
    attn_heads: int,
    gate_init: float,
) -> nn.Module:
    if variant in {"cls_linear", "reg_mean_linear", "reg_concat_linear"}:
        return LinearViewProbe(variant, dim=dim, num_regs=num_regs, num_classes=num_classes)
    if variant == "cls_mlp_control":
        return ResidualMLPProbe("cls", dim=dim, hidden_dim=hidden_dim, num_classes=num_classes, gate_init=gate_init)
    if variant == "cls_resid_regmean":
        return ResidualMLPProbe("reg_mean", dim=dim, hidden_dim=hidden_dim, num_classes=num_classes, gate_init=gate_init)
    if variant == "cls_resid_regtokens_gated":
        return ResidualGatedRegisterProbe(dim=dim, hidden_dim=hidden_dim, num_regs=num_regs, num_classes=num_classes, gate_init=gate_init)
    if variant == "cls_query_reg_attn":
        return CLSQueryRegisterAttentionProbe(dim=dim, num_heads=attn_heads, num_classes=num_classes, gate_init=gate_init)
    raise ValueError(f"Unknown variant '{variant}'.")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    cls: torch.Tensor,
    regs: torch.Tensor,
    labels: torch.Tensor,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float | str]:
    model.eval()
    correct = 0
    total = 0
    fused_chunks: List[torch.Tensor] = []
    cls_chunks: List[torch.Tensor] = []
    last_gates: torch.Tensor | None = None
    for start in range(0, len(indices), batch_size):
        idx = torch.as_tensor(indices[start : start + batch_size], dtype=torch.long)
        cls_b = cls[idx].to(device, non_blocking=True)
        regs_b = regs[idx].to(device, non_blocking=True)
        labels_b = labels[idx].to(device, non_blocking=True)
        out = model(cls_b, regs_b)
        pred = out.logits.argmax(dim=1)
        correct += int((pred == labels_b).sum().item())
        total += int(labels_b.numel())
        if out.fused is not None:
            fused_chunks.append(out.fused.detach().cpu())
            cls_chunks.append(cls_b.detach().cpu())
        if out.gates is not None:
            last_gates = out.gates.detach().cpu().float()
    metrics: Dict[str, float | str] = {"acc": float(correct) / max(float(total), 1.0)}
    if fused_chunks:
        fused = torch.cat(fused_chunks, dim=0).float()
        cls_ref = torch.cat(cls_chunks, dim=0).float()
        cos = F.cosine_similarity(fused, cls_ref, dim=-1)
        residual = fused - cls_ref
        rel_norm = residual.norm(dim=-1) / cls_ref.norm(dim=-1).clamp_min(1.0e-12)
        metrics["cls_cosine_mean"] = float(cos.mean().item())
        metrics["cls_cosine_std"] = float(cos.std(unbiased=False).item())
        metrics["residual_rel_norm_mean"] = float(rel_norm.mean().item())
        metrics["residual_rel_norm_std"] = float(rel_norm.std(unbiased=False).item())
    else:
        metrics["cls_cosine_mean"] = float("nan")
        metrics["cls_cosine_std"] = float("nan")
        metrics["residual_rel_norm_mean"] = float("nan")
        metrics["residual_rel_norm_std"] = float("nan")
    metrics["gate_values"] = "" if last_gates is None else " ".join(f"{x:.6g}" for x in last_gates.tolist())
    return metrics


def train_one(
    variant: str,
    split_id: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cls: torch.Tensor,
    regs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float | int | str]:
    variant_offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(variant))
    seed_all(args.seed + split_id * 1000 + variant_offset)
    dim = int(cls.shape[1])
    num_regs = int(regs.shape[1])
    model = build_model(
        variant,
        dim=dim,
        num_regs=num_regs,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        attn_heads=args.attn_heads,
        gate_init=args.gate_init,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long)
    best_state = None
    best_train_loss = math.inf
    last_train_loss = math.inf
    for epoch in range(args.epochs):
        model.train()
        perm = train_idx_t[torch.randperm(train_idx_t.numel())]
        loss_sum = 0.0
        seen = 0
        for start in range(0, perm.numel(), args.batch_size):
            idx = perm[start : start + args.batch_size]
            cls_b = cls[idx].to(device, non_blocking=True)
            regs_b = regs[idx].to(device, non_blocking=True)
            labels_b = labels[idx].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(cls_b, regs_b)
            loss = criterion(out.logits, labels_b)
            if args.geometry_lambda > 0 and out.fused is not None and variant.startswith("cls_") and variant != "cls_linear":
                geom = 1.0 - F.cosine_similarity(out.fused, cls_b, dim=-1).mean()
                loss = loss + args.geometry_lambda * geom
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for variant={variant} split={split_id} epoch={epoch}: {loss.item()}")
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_n = int(labels_b.numel())
            loss_sum += float(loss.item()) * batch_n
            seen += batch_n
        last_train_loss = loss_sum / max(float(seen), 1.0)
        if last_train_loss < best_train_loss:
            best_train_loss = last_train_loss
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    train_metrics = evaluate(model, cls, regs, labels, train_idx, device, args.eval_batch_size)
    test_metrics = evaluate(model, cls, regs, labels, test_idx, device, args.eval_batch_size)
    row: Dict[str, float | int | str] = {
        "variant": variant,
        "split": int(split_id),
        "dim": dim if variant != "reg_concat_linear" else dim * num_regs,
        "num_params": count_parameters(model),
        "train_loss": float(last_train_loss),
        "best_train_loss": float(best_train_loss),
        "train_acc": float(train_metrics["acc"]),
        "test_acc": float(test_metrics["acc"]),
        "test_cls_cosine_mean": float(test_metrics["cls_cosine_mean"]),
        "test_cls_cosine_std": float(test_metrics["cls_cosine_std"]),
        "test_residual_rel_norm_mean": float(test_metrics["residual_rel_norm_mean"]),
        "test_residual_rel_norm_std": float(test_metrics["residual_rel_norm_std"]),
        "gate_values": str(test_metrics["gate_values"]),
    }
    return row


def aggregate_rows(rows: Sequence[Dict[str, float | int | str]]) -> List[Dict[str, float | int | str]]:
    out = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        group = [row for row in rows if str(row["variant"]) == variant]
        acc_stats = mean_std_ci([float(row["test_acc"]) for row in group])
        cos_stats = mean_std_ci([float(row["test_cls_cosine_mean"]) for row in group])
        resid_stats = mean_std_ci([float(row["test_residual_rel_norm_mean"]) for row in group])
        out.append(
            {
                "variant": variant,
                "dim": int(group[0]["dim"]),
                "num_params": int(group[0]["num_params"]),
                "acc_mean": acc_stats["mean"],
                "acc_std": acc_stats["std"],
                "acc_ci_low": acc_stats["ci_low"],
                "acc_ci_high": acc_stats["ci_high"],
                "cls_cosine_mean": cos_stats["mean"],
                "cls_cosine_std": cos_stats["std"],
                "residual_rel_norm_mean": resid_stats["mean"],
                "residual_rel_norm_std": resid_stats["std"],
                "splits": len(group),
            }
        )
    order = {
        "cls_linear": 0,
        "reg_mean_linear": 1,
        "reg_concat_linear": 2,
        "cls_mlp_control": 3,
        "cls_resid_regmean": 4,
        "cls_resid_regtokens_gated": 5,
        "cls_query_reg_attn": 6,
    }
    out.sort(key=lambda row: order.get(str(row["variant"]), 100))
    return out


def write_csv(rows: Sequence[Dict[str, float | int | str]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(rows: Sequence[Dict[str, float | int | str]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = [str(row["variant"]) for row in rows]
    values = [float(row["acc_mean"]) for row in rows]
    errs = [float(row["acc_std"]) for row in rows]
    x = np.arange(len(names))
    ensure_dir(out_path.parent)
    plt.figure(figsize=(12, 4.8))
    plt.bar(x, values, yerr=errs, capsize=4, color="#3F6C8A")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("Top-1 accuracy")
    plt.title("CLS-anchored register fusion probe")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def parse_variants(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return [
            "cls_linear",
            "reg_mean_linear",
            "reg_concat_linear",
            "cls_mlp_control",
            "cls_resid_regmean",
            "cls_resid_regtokens_gated",
            "cls_query_reg_attn",
        ]
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser("Probe whether registers help when fused back into CLS geometry.")
    parser.add_argument("--embeddings", type=str, default="RAE_ROOT_PLACEHOLDER/assets/analysis/dino_imagenet100_token_probe_v2/embeddings.pt")
    parser.add_argument("--outdir", type=str, default="RAE_ROOT_PLACEHOLDER/assets/analysis/dino_imagenet100_token_probe_v2/cls_register_fusion")
    parser.add_argument("--variants", type=str, default="all")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--attn-heads", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gate-init", type=float, default=1.0e-3)
    parser.add_argument("--geometry-lambda", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    seed_all(args.seed)
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cls, regs, labels, class_names = load_embeddings(Path(args.embeddings))
    cls, regs = l2_normalize_tokens(cls, regs)
    num_classes = int(labels.max().item()) + 1 if not class_names else len(class_names)
    variants = parse_variants(args.variants)

    all_rows: List[Dict[str, float | int | str]] = []
    for split_id, (train_idx, test_idx) in enumerate(stratified_splits(labels, args.splits, args.test_size, args.seed)):
        for variant in variants:
            print(f"[probe] split={split_id} variant={variant}", flush=True)
            row = train_one(
                variant,
                split_id,
                train_idx,
                test_idx,
                cls,
                regs,
                labels,
                num_classes,
                args,
                device,
            )
            all_rows.append(row)
            write_csv(all_rows, outdir / "cls_register_fusion_probe_by_split.csv")

    summary_rows = aggregate_rows(all_rows)
    write_csv(summary_rows, outdir / "cls_register_fusion_probe.csv")
    maybe_plot(summary_rows, outdir / "plots" / "cls_register_fusion_probe_accuracy.png")
    save_json(
        {
            "embeddings": str(Path(args.embeddings)),
            "outdir": str(outdir),
            "num_samples": int(labels.numel()),
            "num_classes": num_classes,
            "num_registers": int(regs.shape[1]),
            "hidden_dim": int(cls.shape[1]),
            "variants": variants,
            "args": vars(args),
            "rows": summary_rows,
        },
        outdir / "cls_register_fusion_summary.json",
    )
    print(f"[done] wrote {outdir / 'cls_register_fusion_probe.csv'}", flush=True)


if __name__ == "__main__":
    main()
