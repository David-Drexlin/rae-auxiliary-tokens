from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

TOKEN_NAMES = ["cls", "reg1", "reg2", "reg3", "reg4"]
EPS = 1.0e-6


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def save_json(obj: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def list_shards(cache_dir: Path) -> List[Path]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    shards = [cache_dir / item["file"] for item in manifest.get("shards", [])]
    if not shards:
        raise ValueError(f"No shards listed in {manifest_path}")
    return shards


def load_stats(path: Path) -> Dict[str, torch.Tensor]:
    stats = torch.load(path, map_location="cpu")
    required = ["source_mean", "source_std", "target_mean", "target_std"]
    missing = [key for key in required if key not in stats]
    if missing:
        raise KeyError(f"Stats file {path} is missing keys: {missing}")
    return {key: value.float() if torch.is_tensor(value) else value for key, value in stats.items()}


def standardize_source(x: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    mean = stats["source_mean"].to(device=x.device, dtype=torch.float32).view(1, 1, -1)
    std = stats["source_std"].to(device=x.device, dtype=torch.float32).view(1, 1, -1).clamp_min(EPS)
    return (x.float() - mean) / std


def standardize_source_global(x: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    mean = stats.get("source_global_mean", stats["source_mean"]).to(device=x.device, dtype=torch.float32).view(1, 1, -1)
    std = stats.get("source_global_std", stats["source_std"]).to(device=x.device, dtype=torch.float32).view(1, 1, -1).clamp_min(EPS)
    return (x.float() - mean) / std


def standardize_target(x: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    mean = stats["target_mean"].to(device=x.device, dtype=torch.float32).unsqueeze(0)
    std = stats["target_std"].to(device=x.device, dtype=torch.float32).unsqueeze(0).clamp_min(EPS)
    return (x.float() - mean) / std


def unstandardize_target(x: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    mean = stats["target_mean"].to(device=x.device, dtype=torch.float32).unsqueeze(0)
    std = stats["target_std"].to(device=x.device, dtype=torch.float32).unsqueeze(0).clamp_min(EPS)
    return x.float() * std + mean


def infer_shapes(cache_dir: Path) -> Tuple[int, int, int, int]:
    first = torch.load(list_shards(cache_dir)[0], map_location="cpu")
    source = first["source_tokens"]
    target = first["target_tokens"]
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError(
            f"Expected source [B,N,C] and target [B,K,C], got {tuple(source.shape)} and {tuple(target.shape)}"
        )
    return int(source.shape[-1]), int(target.shape[-1]), int(target.shape[1]), int(source.shape[1])


def iter_feature_batches(
    cache_dir: Path,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_last: bool = False,
) -> Iterator[Dict[str, object]]:
    shards = list_shards(cache_dir)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    if shuffle:
        order = torch.randperm(len(shards), generator=generator).tolist()
        shards = [shards[i] for i in order]

    for shard_path in shards:
        shard = torch.load(shard_path, map_location="cpu")
        source = shard["source_tokens"]
        target = shard["target_tokens"]
        labels = shard["labels"].long()
        indices = shard["indices"].long()
        image_paths = list(shard.get("image_paths", []))
        n = int(source.shape[0])
        sample_order = torch.randperm(n, generator=generator).tolist() if shuffle else list(range(n))
        for start in range(0, n, batch_size):
            batch_indices = sample_order[start : start + batch_size]
            if drop_last and len(batch_indices) < batch_size:
                continue
            idx = torch.tensor(batch_indices, dtype=torch.long)
            yield {
                "source_tokens": source.index_select(0, idx),
                "target_tokens": target.index_select(0, idx),
                "labels": labels.index_select(0, idx),
                "indices": indices.index_select(0, idx),
                "image_paths": [image_paths[i] for i in batch_indices] if image_paths else [],
                "source_global_tokens": (
                    shard["source_global_tokens"].index_select(0, idx)
                    if "source_global_tokens" in shard
                    else None
                ),
            }


def load_target_bank(cache_dir: Path, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    targets: List[torch.Tensor] = []
    indices: List[torch.Tensor] = []
    for shard_path in list_shards(cache_dir):
        shard = torch.load(shard_path, map_location="cpu")
        targets.append(shard["target_tokens"].float())
        indices.append(shard["indices"].long())
    return torch.cat(targets, dim=0).to(device), torch.cat(indices, dim=0).to(device)


class RegisterMetricAccumulator:
    def __init__(self, token_names: Optional[List[str]] = None):
        self.token_names = token_names or TOKEN_NAMES
        self.num_slots = len(self.token_names)
        self.slot_mse_sum = torch.zeros(self.num_slots, dtype=torch.float64)
        self.slot_cos_sum = torch.zeros(self.num_slots, dtype=torch.float64)
        self.slot_count = torch.zeros(self.num_slots, dtype=torch.float64)
        self.slot_retrieval_correct = torch.zeros(self.num_slots, dtype=torch.float64)
        self.slot_retrieval_count = torch.zeros(self.num_slots, dtype=torch.float64)
        self.all_mse_sum = 0.0
        self.all_mse_count = 0.0
        self.all_cos_sum = 0.0
        self.all_cos_count = 0.0
        self.all_retrieval_correct = 0.0
        self.all_retrieval_count = 0.0

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        target_bank: Optional[torch.Tensor] = None,
        bank_indices: Optional[torch.Tensor] = None,
    ) -> None:
        pred = pred.float()
        target = target.float()
        if pred.shape != target.shape:
            raise ValueError(f"Prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        bsz, slots, dim = pred.shape
        if slots != self.num_slots:
            raise ValueError(f"Expected {self.num_slots} slots, got {slots}")

        slot_mse = (pred - target).square().mean(dim=-1)
        slot_cos = F.cosine_similarity(pred, target, dim=-1)
        self.slot_mse_sum += slot_mse.detach().cpu().double().sum(dim=0)
        self.slot_cos_sum += slot_cos.detach().cpu().double().sum(dim=0)
        self.slot_count += bsz

        flat_pred = pred.reshape(bsz, -1)
        flat_target = target.reshape(bsz, -1)
        all_mse = (flat_pred - flat_target).square().mean(dim=-1)
        all_cos = F.cosine_similarity(flat_pred, flat_target, dim=-1)
        self.all_mse_sum += float(all_mse.sum().item())
        self.all_mse_count += float(bsz)
        self.all_cos_sum += float(all_cos.sum().item())
        self.all_cos_count += float(bsz)

        if indices is not None and target_bank is not None and bank_indices is not None:
            indices = indices.to(pred.device)
            bank_indices = bank_indices.to(pred.device)
            bank = target_bank.float().to(pred.device)
            for slot in range(slots):
                pred_norm = F.normalize(pred[:, slot], dim=-1)
                bank_norm = F.normalize(bank[:, slot], dim=-1)
                nearest = torch.argmax(pred_norm @ bank_norm.t(), dim=-1)
                correct = bank_indices[nearest].eq(indices)
                correct_count = float(correct.float().sum().item())
                self.slot_retrieval_correct[slot] += correct_count
                self.slot_retrieval_count[slot] += bsz
                self.all_retrieval_correct += correct_count
                self.all_retrieval_count += bsz

    def rows(self) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for idx, name in enumerate(self.token_names):
            count = max(float(self.slot_count[idx].item()), 1.0)
            mse = float(self.slot_mse_sum[idx].item()) / count
            retrieval_count = float(self.slot_retrieval_count[idx].item())
            retrieval = (
                float(self.slot_retrieval_correct[idx].item()) / retrieval_count
                if retrieval_count > 0
                else float("nan")
            )
            rows.append(
                {
                    "slot": name,
                    "mse": mse,
                    "rmse": math.sqrt(max(mse, 0.0)),
                    "cosine": float(self.slot_cos_sum[idx].item()) / count,
                    "retrieval_top1": retrieval,
                    "count": count,
                }
            )
        all_count = max(self.all_mse_count, 1.0)
        all_mse = self.all_mse_sum / all_count
        rows.append(
            {
                "slot": "all",
                "mse": all_mse,
                "rmse": math.sqrt(max(all_mse, 0.0)),
                "cosine": self.all_cos_sum / max(self.all_cos_count, 1.0),
                "retrieval_top1": self.all_retrieval_correct / self.all_retrieval_count
                if self.all_retrieval_count > 0
                else float("nan"),
                "count": all_count,
            }
        )
        return rows


def write_metrics_csv(rows: Iterable[Dict[str, float]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No metric rows to write.")
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
