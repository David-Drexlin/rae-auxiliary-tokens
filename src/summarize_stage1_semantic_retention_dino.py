#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from stage1_semantic_retention_dino_utils import canonical_dino_family, ensure_dir, mean_std, save_json


DEFAULT_INPUT = "RAE_ROOT_PLACEHOLDER/assets/analysis/stage1_semantic_retention_dino/runs"
DEFAULT_OUTPUT = "RAE_ROOT_PLACEHOLDER/assets/analysis/stage1_semantic_retention_dino/summary"
DEFAULT_METRICS = "RAE_ROOT_PLACEHOLDER/assets/recon_samples/eval_metrics_seeded_dino.csv"

CANONICAL_FAMILIES = [
    "DINO_decB",
    "Patch+CLS_prepend",
    "Patch+Register_prepend",
    "Patch+Register+CLS_prepend",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DINO Stage-1 semantic-retention evaluations.")
    parser.add_argument("--input-root", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--metrics-csv", type=Path, default=Path(DEFAULT_METRICS))
    parser.add_argument("--include-stress", action="store_true")
    parser.add_argument("--include-noncanonical", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def load_rfid_map(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    out: Dict[str, float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            save_folder = row.get("save_folder", "")
            fid_value = row.get("fid", "")
            if save_folder and fid_value:
                try:
                    out[save_folder] = float(fid_value)
                except ValueError:
                    pass
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_probe_bars(agg_rows: List[Dict[str, object]], out_path: Path) -> None:
    labels = [row["family"] for row in agg_rows]
    means = [row["probe_top1_recon_mean"] for row in agg_rows]
    stds = [row["probe_top1_recon_std"] for row in agg_rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, means, yerr=stds, capsize=4, color="#2f6c8f")
    ax.set_ylabel("Probe Top-1 on Recon")
    ax.set_ylim(0.0, min(1.0, max(means) + 0.1))
    ax.set_title("DINO Semantic Retention by Decoder Family")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_latent_bars(agg_rows: List[Dict[str, object]], out_path: Path) -> None:
    labels = [row["family"] for row in agg_rows]
    metric_names = ["cls_cosine_mean", "reg_mean_cosine_mean", "patch_mean_cosine_mean"]
    pretty = ["CLS", "REG mean", "Patch mean"]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (metric, label) in enumerate(zip(metric_names, pretty)):
        means = [row[metric] for row in agg_rows]
        stds = [row[metric.replace("_mean", "_std")] for row in agg_rows]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=4, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Cosine Retention")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Decode-Reencode Latent Retention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_rfid_scatter(rows: List[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    families = list(dict.fromkeys(row["family"] for row in rows))
    cmap = plt.get_cmap("tab10")
    for i, family in enumerate(families):
        subset = [row for row in rows if row["family"] == family]
        ax.scatter(
            [row["rFID"] for row in subset],
            [row["probe_top1_recon"] for row in subset],
            color=cmap(i % 10),
            label=family,
            s=50,
            alpha=0.85,
        )
    ax.set_xlabel("rFID")
    ax.set_ylabel("Probe Top-1 on Recon")
    ax.set_title("rFID vs Semantic Retention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    rfid_map = load_rfid_map(args.metrics_csv)

    summary_paths = sorted(args.input_root.glob("*/summary.json"))
    per_run_rows: List[Dict[str, object]] = []
    for summary_path in summary_paths:
        row = load_json(summary_path)
        row.setdefault("family", canonical_dino_family(str(row.get("save_folder", ""))))
        row["summary_path"] = str(summary_path)
        row["rFID"] = rfid_map.get(str(row.get("save_folder", "")))
        per_run_rows.append(row)

    filtered_rows = []
    for row in per_run_rows:
        if not args.include_stress and row.get("stress_label", "clean") != "clean":
            continue
        if row.get("seed") is None:
            continue
        if not args.include_noncanonical and row.get("family") not in CANONICAL_FAMILIES:
            continue
        filtered_rows.append(row)

    write_csv(args.output_root / "per_run_filtered.csv", filtered_rows)

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in filtered_rows:
        grouped[str(row["family"])].append(row)

    agg_rows: List[Dict[str, object]] = []
    for family in (CANONICAL_FAMILIES if not args.include_noncanonical else sorted(grouped.keys())):
        rows = grouped.get(family, [])
        if not rows:
            continue
        agg = {"family": family, "n_runs": len(rows)}
        for metric in [
            "probe_top1_recon",
            "prediction_agreement_real_vs_recon",
            "cls_cosine",
            "reg_mean_cosine",
            "patch_mean_cosine",
        ]:
            stats = mean_std([float(row[metric]) for row in rows])
            agg[f"{metric}_mean"] = stats["mean"]
            agg[f"{metric}_std"] = stats["std"]
        rfids = [float(row["rFID"]) for row in rows if row.get("rFID") is not None]
        if rfids:
            stats = mean_std(rfids)
            agg["rFID_mean"] = stats["mean"]
            agg["rFID_std"] = stats["std"]
        agg_rows.append(agg)

    write_csv(args.output_root / "canonical_family_summary.csv", agg_rows)
    save_json(
        {
            "num_run_summaries_found": len(per_run_rows),
            "num_runs_used": len(filtered_rows),
            "families": agg_rows,
        },
        args.output_root / "summary.json",
    )

    if agg_rows:
        plot_probe_bars(agg_rows, args.output_root / "probe_top1_by_family.png")
        plot_latent_bars(agg_rows, args.output_root / "latent_retention_by_family.png")
    scatter_rows = [row for row in filtered_rows if row.get("rFID") is not None]
    if scatter_rows:
        plot_rfid_scatter(scatter_rows, args.output_root / "rfid_vs_probe_top1.png")

    stress_rows = [row for row in per_run_rows if row.get("stress_label", "clean") != "clean"]
    if stress_rows:
        stress_grouped: Dict[tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
        for row in stress_rows:
            family = row.get("family") or row.get("model_label") or "unknown"
            stress_grouped[(str(family), str(row.get("stress_label")))].append(row)
        stress_summary_rows: List[Dict[str, object]] = []
        for (family, stress_label), rows in sorted(stress_grouped.items()):
            agg = {"family": family, "stress_label": stress_label, "n_runs": len(rows)}
            for metric in [
                "probe_top1_recon",
                "prediction_agreement_real_vs_recon",
                "cls_cosine",
                "reg_mean_cosine",
                "patch_mean_cosine",
            ]:
                stats = mean_std([float(row[metric]) for row in rows])
                agg[f"{metric}_mean"] = stats["mean"]
                agg[f"{metric}_std"] = stats["std"]
            stress_summary_rows.append(agg)
        write_csv(args.output_root / "stress_summary.csv", stress_summary_rows)

    print(json.dumps({"families": agg_rows, "num_runs_used": len(filtered_rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
