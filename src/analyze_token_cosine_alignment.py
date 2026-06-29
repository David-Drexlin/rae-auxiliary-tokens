#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.nn import functional as F


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_tensor(data, key: str) -> torch.Tensor:
    return torch.as_tensor(data[key], dtype=torch.float32)


def safe_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1.0e-12)


def cosine_values(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (safe_norm(a) * safe_norm(b)).sum(dim=-1)


def summarize(values: torch.Tensor) -> Dict[str, float]:
    values_np = values.detach().cpu().numpy().astype(np.float64)
    return {
        "mean": float(values_np.mean()),
        "std": float(values_np.std(ddof=0)),
        "ci_low": float(np.quantile(values_np, 0.025)),
        "ci_high": float(np.quantile(values_np, 0.975)),
    }


def build_views(data) -> Dict[str, torch.Tensor]:
    cls = to_tensor(data, "cls")
    patch_mean = to_tensor(data, "patch_mean")
    regs = to_tensor(data, "regs")
    if regs.ndim != 3 or regs.shape[1] < 1:
        raise ValueError(f"Expected regs shape [N, K, D], got {tuple(regs.shape)}")
    reg_mean = regs.mean(dim=1)
    views: Dict[str, torch.Tensor] = {
        "cls": cls,
        "patch_mean": patch_mean,
        "reg_mean": reg_mean,
    }
    for idx in range(regs.shape[1]):
        views[f"reg{idx + 1}"] = regs[:, idx]

    # Simple same-dimensional composites. These are not learned fusion results;
    # they only quantify where naive CLS+register directions sit geometrically.
    views["cls_plus_reg_mean"] = safe_norm(safe_norm(cls) + safe_norm(reg_mean))
    views["cls_plus_reg_sum"] = safe_norm(safe_norm(cls) + safe_norm(regs).sum(dim=1))
    return views


def centered_views(views: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: value - value.mean(dim=0, keepdim=True) for name, value in views.items()}


def shuffled_summary(a: torch.Tensor, b: torch.Tensor, reps: int, seed: int) -> Dict[str, float]:
    del reps, seed
    # Exact expectation for random cross-image pairing after per-sample normalization:
    # E_i,j[cos(a_i, b_j)] = mean(norm(a), dim=0) dot mean(norm(b), dim=0).
    a_norm = safe_norm(a)
    b_norm = safe_norm(b)
    expected = float((a_norm.mean(dim=0) * b_norm.mean(dim=0)).sum().item())
    return {
        "shuffle_mean": expected,
        "shuffle_std": 0.0,
        "shuffle_ci_low": expected,
        "shuffle_ci_high": expected,
    }


def pair_rows(
    views: Dict[str, torch.Tensor],
    names: Sequence[str],
    mode: str,
    shuffle_reps: int,
    seed: int,
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            a = views[name_a]
            b = views[name_b]
            if a.shape != b.shape:
                continue
            stats = summarize(cosine_values(a, b))
            shuffle = shuffled_summary(a, b, shuffle_reps, seed + i * 1009 + j * 9173)
            row = {
                "mode": mode,
                "view_a": name_a,
                "view_b": name_b,
                "shuffle_method": "analytic_independent_pair_expectation",
                "dim": int(a.shape[-1]),
                "paired_mean": stats["mean"],
                "paired_std": stats["std"],
                "paired_ci_low": stats["ci_low"],
                "paired_ci_high": stats["ci_high"],
                **shuffle,
                "paired_minus_shuffle": float(stats["mean"] - shuffle["shuffle_mean"]),
            }
            rows.append(row)
    return rows


def write_csv(rows: Sequence[Dict[str, float | str]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(rows: Sequence[Dict[str, float | str]], names: Sequence[str], mode: str, out_path: Path) -> None:
    row_map = {(str(row["view_a"]), str(row["view_b"])): float(row["paired_mean"]) for row in rows if row["mode"] == mode}
    ensure_dir(out_path.parent)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["view", *names])
        for name_a in names:
            writer.writerow([name_a, *[row_map.get((name_a, name_b), float("nan")) for name_b in names]])


def maybe_heatmap(rows: Sequence[Dict[str, float | str]], names: Sequence[str], mode: str, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    row_map = {(str(row["view_a"]), str(row["view_b"])): float(row["paired_mean"]) for row in rows if row["mode"] == mode}
    mat = np.asarray([[row_map.get((a, b), np.nan) for b in names] for a in names], dtype=np.float32)
    ensure_dir(out_path.parent)
    plt.figure(figsize=(8.8, 7.2))
    im = plt.imshow(mat, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="paired cosine")
    plt.xticks(np.arange(len(names)), names, rotation=45, ha="right")
    plt.yticks(np.arange(len(names)), names)
    plt.title(f"Token cosine alignment ({mode})")
    for y in range(len(names)):
        for x in range(len(names)):
            value = mat[y, x]
            if np.isfinite(value):
                plt.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser("Compute same-image cosine alignment among DINO token views.")
    parser.add_argument("--embeddings", type=str, default="RAE_ROOT_PLACEHOLDER/assets/analysis/dino_imagenet100_token_probe_v2/embeddings.pt")
    parser.add_argument("--outdir", type=str, default="RAE_ROOT_PLACEHOLDER/assets/analysis/dino_imagenet100_token_probe_v2/token_cosine_alignment")
    parser.add_argument("--shuffle-reps", type=int, default=0, help="Kept for CLI compatibility; shuffled baseline is analytic.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    data = load_torch(Path(args.embeddings))
    views = build_views(data)
    names = ["cls", "patch_mean", "reg_mean", "reg1", "reg2", "reg3", "reg4", "cls_plus_reg_mean", "cls_plus_reg_sum"]
    names = [name for name in names if name in views]

    rows = []
    rows.extend(pair_rows(views, names, mode="raw", shuffle_reps=args.shuffle_reps, seed=args.seed))
    rows.extend(pair_rows(centered_views(views), names, mode="centered", shuffle_reps=args.shuffle_reps, seed=args.seed + 100000))
    write_csv(rows, outdir / "token_cosine_alignment_pairs.csv")
    for mode in ["raw", "centered"]:
        write_matrix(rows, names, mode=mode, out_path=outdir / f"token_cosine_alignment_matrix_{mode}.csv")
        maybe_heatmap(rows, names, mode=mode, out_path=outdir / "plots" / f"token_cosine_alignment_{mode}.png")

    highlight_pairs = [
        ("cls", "patch_mean"),
        ("cls", "reg_mean"),
        ("cls", "reg1"),
        ("cls", "reg2"),
        ("cls", "reg3"),
        ("cls", "reg4"),
        ("patch_mean", "reg_mean"),
        ("patch_mean", "cls_plus_reg_mean"),
        ("reg_mean", "cls_plus_reg_mean"),
        ("cls", "cls_plus_reg_mean"),
        ("cls", "cls_plus_reg_sum"),
    ]
    highlights = [
        row for row in rows
        if row["mode"] in {"raw", "centered"} and (row["view_a"], row["view_b"]) in highlight_pairs
    ]
    write_csv(highlights, outdir / "token_cosine_alignment_highlights.csv")
    save_json(
        {
            "embeddings": str(Path(args.embeddings)),
            "outdir": str(outdir),
            "num_samples": int(to_tensor(data, "cls").shape[0]),
            "views": names,
            "shuffle_reps_requested": int(args.shuffle_reps),
            "shuffle_method": "analytic_independent_pair_expectation",
            "seed": int(args.seed),
            "outputs": {
                "pairs": str(outdir / "token_cosine_alignment_pairs.csv"),
                "highlights": str(outdir / "token_cosine_alignment_highlights.csv"),
                "raw_matrix": str(outdir / "token_cosine_alignment_matrix_raw.csv"),
                "centered_matrix": str(outdir / "token_cosine_alignment_matrix_centered.csv"),
            },
        },
        outdir / "token_cosine_alignment_summary.json",
    )
    print(f"[done] wrote {outdir / 'token_cosine_alignment_pairs.csv'}", flush=True)


if __name__ == "__main__":
    main()
