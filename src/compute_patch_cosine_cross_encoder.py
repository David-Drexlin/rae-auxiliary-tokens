#!/usr/bin/env python3
"""Compute cosine similarity between mean-pooled patch tokens across encoders.

Uses cached patch-token shards from register-prediction experiments to avoid
re-encoding images. Outputs mean/std cosine for DINO vs MAE vs SigLIP2.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch


def load_means(cache_root: Path) -> Dict[int, torch.Tensor]:
    shards = sorted(cache_root.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.pt files found in {cache_root}")
    means: Dict[int, torch.Tensor] = {}
    for shard in shards:
        data = torch.load(shard, map_location="cpu")
        toks = data["source_tokens"]  # [B, N, C]
        idxs = data["indices"]
        pooled = toks.mean(dim=1).float()
        pooled = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-12)
        for i, idx in enumerate(idxs.tolist()):
            means[int(idx)] = pooled[i]
    return means


def cosine_stats(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    cos = (a * b).sum(dim=1)
    return float(cos.mean()), float(cos.std(unbiased=False))


def main() -> None:
    roots = {
        "dino": Path("RAE_ROOT_PLACEHOLDER/assets/analysis/dino_patch_to_dino_regcls_in100/cache/val"),
        "mae": Path("RAE_ROOT_PLACEHOLDER/assets/analysis/mae_to_dino_regcls_in100/cache/val"),
        "siglip2": Path("RAE_ROOT_PLACEHOLDER/assets/analysis/siglip2_to_dino_regcls_in100/cache/val"),
    }

    means = {k: load_means(v) for k, v in roots.items()}
    common = set.intersection(*(set(m.keys()) for m in means.values()))
    keys = sorted(common)
    if not keys:
        raise RuntimeError("No common indices across caches.")

    stacked = {k: torch.stack([means[k][i] for i in keys], dim=0) for k in means}

    pairs = [("dino", "mae"), ("dino", "siglip2"), ("mae", "siglip2")]
    results = {"num_common": len(keys), "pairs": {}}
    for a, b in pairs:
        mean, std = cosine_stats(stacked[a], stacked[b])
        results["pairs"][f"{a}_vs_{b}"] = {"mean": mean, "std": std}
        print(f"{a} vs {b}: mean={mean:.4f} std={std:.4f}")

    out_path = Path("RAE_ROOT_PLACEHOLDER/assets/analysis/patch_cosine_cross_encoder.json")
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
