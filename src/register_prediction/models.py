from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionPoolBlock(nn.Module):
    """Updates learned output slots by attending over source patch tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.slot_norm = nn.LayerNorm(hidden_dim)
        self.source_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, int(hidden_dim * mlp_ratio), dropout=dropout)

    def forward(
        self,
        slots: torch.Tensor,
        source: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q = self.slot_norm(slots)
        kv = self.source_norm(source)
        attn_out, attn_weights = self.attn(
            q,
            kv,
            kv,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        slots = slots + attn_out
        slots = slots + self.mlp(self.mlp_norm(slots))
        return slots, attn_weights if return_attn else None


class MHAPRegisterPredictor(nn.Module):
    """Multi-head attention pooling predictor for DINO CLS/register tokens."""

    def __init__(
        self,
        source_dim: int,
        target_dim: int = 768,
        hidden_dim: int = 768,
        num_slots: int = 5,
        depth: int = 2,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.source_dim = int(source_dim)
        self.target_dim = int(target_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.depth = int(depth)

        self.source_proj = nn.Linear(source_dim, hidden_dim)
        self.source_norm = nn.LayerNorm(hidden_dim)
        self.slot_queries = nn.Parameter(torch.zeros(1, num_slots, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                CrossAttentionPoolBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, target_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.slot_queries, std=0.02)
        nn.init.xavier_uniform_(self.source_proj.weight)
        nn.init.zeros_(self.source_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if x.ndim != 3:
            raise ValueError(f"Expected source tokens [B, N, C], got {tuple(x.shape)}")
        source = self.source_norm(self.source_proj(x))
        slots = self.slot_queries.expand(x.size(0), -1, -1)
        attn_maps: List[torch.Tensor] = []
        for block in self.blocks:
            slots, attn = block(slots, source, return_attn=return_attn)
            if attn is not None:
                attn_maps.append(attn)
        out = self.out_proj(self.out_norm(slots))
        if return_attn:
            return out, {"attn": attn_maps}
        return out


class MeanMLPRegisterPredictor(nn.Module):
    """Mean-pool baseline that predicts all DINO global tokens from one vector."""

    def __init__(
        self,
        source_dim: int,
        target_dim: int = 768,
        hidden_dim: int = 768,
        num_slots: int = 5,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.source_dim = int(source_dim)
        self.target_dim = int(target_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)

        layers = [nn.LayerNorm(source_dim), nn.Linear(source_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(int(depth) - 1, 0)):
            layers.extend(
                [
                    nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, num_slots * target_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if x.ndim != 3:
            raise ValueError(f"Expected source tokens [B, N, C], got {tuple(x.shape)}")
        pooled = x.mean(dim=1)
        out = self.net(pooled).view(x.size(0), self.num_slots, self.target_dim)
        if return_attn:
            return out, {"attn": []}
        return out


def build_model(
    model_name: str,
    source_dim: int,
    target_dim: int,
    num_slots: int,
    model_cfg: Dict[str, Any],
) -> nn.Module:
    name = str(model_name).lower()
    hidden_dim = int(model_cfg.get("hidden_dim", target_dim))
    depth = int(model_cfg.get("depth", 2))
    mlp_ratio = float(model_cfg.get("mlp_ratio", 4.0))
    dropout = float(model_cfg.get("dropout", 0.0))

    if name in {"mhap", "attention", "mha"}:
        return MHAPRegisterPredictor(
            source_dim=source_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            depth=depth,
            num_heads=int(model_cfg.get("num_heads", 12)),
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
    if name in {"mean_mlp", "mean", "mlp"}:
        return MeanMLPRegisterPredictor(
            source_dim=source_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
    raise ValueError(f"Unknown register predictor model '{model_name}'.")
