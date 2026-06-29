from __future__ import annotations

from typing import Dict, Optional

import torch
from omegaconf import OmegaConf
from torch import nn

from . import register_encoder
from .dinov2 import Dinov2withNorm
from .mae import MAEwNorm
from .siglip2 import SigLIP2wNorm
from register_prediction.eval import build_model_for_eval, cache_root, train_stats_path
from register_prediction.metrics import load_stats, standardize_source, unstandardize_target


class ResidualTokenBankAdapter(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_tokens: int,
        bottleneck_dim: int = 192,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.norm_slots = nn.LayerNorm(self.token_dim)
        self.norm_mlp = nn.LayerNorm(self.token_dim)
        self.slot_delta = nn.Parameter(torch.zeros(self.num_tokens, self.num_tokens))
        self.slot_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.fc1 = nn.Linear(self.token_dim, int(bottleneck_dim))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(bottleneck_dim), self.token_dim)
        self.mlp_scale = nn.Parameter(torch.tensor(float(init_scale)))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens
        slot_delta = torch.einsum("bkd,kl->bld", self.norm_slots(x), self.slot_delta)
        x = x + self.slot_scale * slot_delta
        mlp_delta = self.fc2(self.act(self.fc1(self.norm_mlp(x))))
        x = x + self.mlp_scale * mlp_delta
        return x


def _load_predictor_cfg(path: str) -> Dict:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError(f"Predictor config must resolve to a dict: {path}")
    return cfg


class _RecoveredDINORegCLSBase(nn.Module):
    def __init__(
        self,
        predictor_config_path: str,
        predictor_ckpt_path: str,
        predictor_model_name: Optional[str] = None,
    ):
        super().__init__()
        self.predictor_config_path = str(predictor_config_path)
        self.predictor_ckpt_path = str(predictor_ckpt_path)
        self.predictor_cfg = _load_predictor_cfg(self.predictor_config_path)

        ckpt = torch.load(self.predictor_ckpt_path, map_location="cpu")
        resolved_model_name = predictor_model_name or str(
            ckpt.get("model_name", self.predictor_cfg.get("model", {}).get("name", "mhap"))
        )
        self.predictor = build_model_for_eval(
            resolved_model_name,
            self.predictor_cfg,
            ckpt,
            cache_root(self.predictor_cfg) / "val",
        )
        self.predictor.load_state_dict(ckpt["model"], strict=True)
        self.predictor.eval()
        self.predictor.requires_grad_(False)
        self.predictor_stats = load_stats(train_stats_path(self.predictor_cfg))

        # Recovered target bank is DINO CLS + 4 registers.
        self.num_register_tokens = 4
        self.num_reg_tokens = self.num_register_tokens
        self.num_prefix_tokens = 1 + self.num_register_tokens
        self.adapter: nn.Module = nn.Identity()

    @torch.no_grad()
    def _predict_dino_aux(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        patch_tokens_std = standardize_source(patch_tokens, self.predictor_stats)
        pred_std = self.predictor(patch_tokens_std)
        return unstandardize_target(pred_std, self.predictor_stats)

    def _init_adapter(
        self,
        *,
        token_dim: int,
        adapter_bottleneck_dim: int = 192,
        adapter_init_scale: float = 1.0,
    ) -> None:
        self.adapter = ResidualTokenBankAdapter(
            token_dim=token_dim,
            num_tokens=self.num_prefix_tokens,
            bottleneck_dim=adapter_bottleneck_dim,
            init_scale=adapter_init_scale,
        )

    def _apply_adapter(self, recovered_aux: torch.Tensor) -> torch.Tensor:
        return self.adapter(recovered_aux)


@register_encoder()
class Dinov2WithRecoveredDINORegCLS(_RecoveredDINORegCLSBase):
    def __init__(
        self,
        source_model_name: str,
        predictor_config_path: str,
        predictor_ckpt_path: str,
        predictor_model_name: Optional[str] = None,
    ):
        source_encoder = Dinov2withNorm(dinov2_path=source_model_name, normalize=True)
        super().__init__(
            predictor_config_path=predictor_config_path,
            predictor_ckpt_path=predictor_ckpt_path,
            predictor_model_name=predictor_model_name,
        )
        self.source_encoder = source_encoder
        self.source_encoder.eval()
        self.source_encoder.requires_grad_(False)
        self.hidden_size = self.source_encoder.hidden_size
        self.patch_size = self.source_encoder.patch_size

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.source_encoder(images)

    @torch.no_grad()
    def forward_with_global(self, images: torch.Tensor):
        patch_tokens = self.source_encoder(images)
        recovered_aux = self._predict_dino_aux(patch_tokens)
        return patch_tokens, recovered_aux


@register_encoder()
class MAEWithRecoveredDINORegCLS(_RecoveredDINORegCLSBase):
    def __init__(
        self,
        source_model_name: str,
        predictor_config_path: str,
        predictor_ckpt_path: str,
        predictor_model_name: Optional[str] = None,
    ):
        source_encoder = MAEwNorm(model_name=source_model_name)
        super().__init__(
            predictor_config_path=predictor_config_path,
            predictor_ckpt_path=predictor_ckpt_path,
            predictor_model_name=predictor_model_name,
        )
        self.source_encoder = source_encoder
        self.source_encoder.eval()
        self.source_encoder.requires_grad_(False)
        self.hidden_size = self.source_encoder.hidden_size
        self.patch_size = self.source_encoder.patch_size

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.source_encoder(images)

    @torch.no_grad()
    def forward_with_global(self, images: torch.Tensor):
        patch_tokens = self.source_encoder(images)
        recovered_aux = self._predict_dino_aux(patch_tokens)
        return patch_tokens, recovered_aux


@register_encoder()
class SigLIP2WithRecoveredDINORegCLS(_RecoveredDINORegCLSBase):
    def __init__(
        self,
        source_model_name: str,
        predictor_config_path: str,
        predictor_ckpt_path: str,
        predictor_model_name: Optional[str] = None,
    ):
        source_encoder = SigLIP2wNorm(model_name=source_model_name)
        super().__init__(
            predictor_config_path=predictor_config_path,
            predictor_ckpt_path=predictor_ckpt_path,
            predictor_model_name=predictor_model_name,
        )
        self.source_encoder = source_encoder
        self.source_encoder.eval()
        self.source_encoder.requires_grad_(False)
        self.hidden_size = self.source_encoder.hidden_size
        self.patch_size = self.source_encoder.patch_size

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.source_encoder(images)

    @torch.no_grad()
    def forward_with_global(self, images: torch.Tensor):
        patch_tokens = self.source_encoder(images)
        recovered_aux = self._predict_dino_aux(patch_tokens)
        return patch_tokens, recovered_aux


@register_encoder()
class SigLIP2WithPoolerAndRecoveredDINORegs(_RecoveredDINORegCLSBase):
    def __init__(
        self,
        source_model_name: str,
        predictor_config_path: str,
        predictor_ckpt_path: str,
        predictor_model_name: Optional[str] = None,
    ):
        source_encoder = SigLIP2wNorm(model_name=source_model_name)
        super().__init__(
            predictor_config_path=predictor_config_path,
            predictor_ckpt_path=predictor_ckpt_path,
            predictor_model_name=predictor_model_name,
        )
        self.source_encoder = source_encoder
        self.source_encoder.eval()
        self.source_encoder.requires_grad_(False)
        self.hidden_size = self.source_encoder.hidden_size
        self.patch_size = self.source_encoder.patch_size
        self.num_register_tokens = 4
        self.num_reg_tokens = self.num_register_tokens
        self.num_prefix_tokens = 1 + self.num_register_tokens

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.source_encoder(images)

    @torch.no_grad()
    def forward_with_global(self, images: torch.Tensor):
        patch_tokens, siglip2_global = self.source_encoder.forward_with_global(images)
        recovered_aux = self._predict_dino_aux(patch_tokens)
        recovered_regs = recovered_aux[:, 1:, :]
        mixed_global_tokens = torch.cat([siglip2_global, recovered_regs], dim=1)
        return patch_tokens, mixed_global_tokens


@register_encoder()
class MAEWithRecoveredDINORegCLSAdapter(_RecoveredDINORegCLSBase):
    def __init__(
        self,
        source_model_name: str,
        predictor_config_path: str,
        predictor_ckpt_path: str,
        predictor_model_name: Optional[str] = None,
        adapter_bottleneck_dim: int = 192,
        adapter_init_scale: float = 1.0,
    ):
        source_encoder = MAEwNorm(model_name=source_model_name)
        super().__init__(
            predictor_config_path=predictor_config_path,
            predictor_ckpt_path=predictor_ckpt_path,
            predictor_model_name=predictor_model_name,
        )
        self.source_encoder = source_encoder
        self.source_encoder.eval()
        self.source_encoder.requires_grad_(False)
        self.hidden_size = self.source_encoder.hidden_size
        self.patch_size = self.source_encoder.patch_size
        self._init_adapter(
            token_dim=self.hidden_size,
            adapter_bottleneck_dim=adapter_bottleneck_dim,
            adapter_init_scale=adapter_init_scale,
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.source_encoder(images)

    def forward_with_global(self, images: torch.Tensor):
        with torch.no_grad():
            patch_tokens = self.source_encoder(images)
            recovered_aux = self._predict_dino_aux(patch_tokens)
        adapted_aux = self._apply_adapter(recovered_aux)
        return patch_tokens, adapted_aux


@register_encoder()
class SigLIP2WithRecoveredDINORegCLSAdapter(_RecoveredDINORegCLSBase):
    def __init__(
        self,
        source_model_name: str,
        predictor_config_path: str,
        predictor_ckpt_path: str,
        predictor_model_name: Optional[str] = None,
        adapter_bottleneck_dim: int = 192,
        adapter_init_scale: float = 1.0,
    ):
        source_encoder = SigLIP2wNorm(model_name=source_model_name)
        super().__init__(
            predictor_config_path=predictor_config_path,
            predictor_ckpt_path=predictor_ckpt_path,
            predictor_model_name=predictor_model_name,
        )
        self.source_encoder = source_encoder
        self.source_encoder.eval()
        self.source_encoder.requires_grad_(False)
        self.hidden_size = self.source_encoder.hidden_size
        self.patch_size = self.source_encoder.patch_size
        self._init_adapter(
            token_dim=self.hidden_size,
            adapter_bottleneck_dim=adapter_bottleneck_dim,
            adapter_init_scale=adapter_init_scale,
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.source_encoder(images)

    def forward_with_global(self, images: torch.Tensor):
        with torch.no_grad():
            patch_tokens = self.source_encoder(images)
            recovered_aux = self._predict_dino_aux(patch_tokens)
        adapted_aux = self._apply_adapter(recovered_aux)
        return patch_tokens, adapted_aux
