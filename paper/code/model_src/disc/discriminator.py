from __future__ import annotations

import torch
import torch.nn as nn

from .dinodisc import DinoDisc


class DinoDiscriminator(DinoDisc):
    

    def __init__(self, device: torch.device, aux_regression: dict | None = None, **kwargs):
        super().__init__(device=device, **kwargs)
        self.aux_regression_head: nn.Module | None = None
        self.aux_regression_num_tokens = 0
        self.aux_regression_token_dim = 0
        if aux_regression is not None and bool(aux_regression.get("enabled", True)):
            self.aux_regression_num_tokens = int(aux_regression.get("num_tokens", 5))
            self.aux_regression_token_dim = int(aux_regression.get("token_dim", 768))
            hidden_dim = int(aux_regression.get("hidden_dim", 1024))
            in_dim = int(self.dino_proxy[0].embed_dim) * len(self.heads)
            out_dim = self.aux_regression_num_tokens * self.aux_regression_token_dim
            self.aux_regression_head = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def classify(self, img: torch.Tensor) -> torch.Tensor:
        return super().forward(img)

    def _aux_from_activations(self, activations: list[torch.Tensor]) -> torch.Tensor:
        if self.aux_regression_head is None:
            raise RuntimeError("return_aux=True requires gan.disc.arch.aux_regression.enabled=true.")
        pooled = torch.cat([act.mean(dim=-1) for act in activations], dim=1)
        pred = self.aux_regression_head(pooled)
        return pred.view(-1, self.aux_regression_num_tokens, self.aux_regression_token_dim)

    def classify_with_aux(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        activations = self.extract_activations(img, grad_ckpt=False)
        logits = self.logits_from_activations(activations, img.shape[0])
        aux = self._aux_from_activations(activations)
        return logits, aux

    def forward(self, fake: torch.Tensor, real: torch.Tensor | None = None, return_aux: bool = False):
        if not return_aux:
            logits_fake = self.classify(fake)
            logits_real = self.classify(real) if real is not None else None
            return logits_fake, logits_real

        logits_fake, aux_fake = self.classify_with_aux(fake)
        if real is None:
            return logits_fake, None, aux_fake, None
        logits_real, aux_real = self.classify_with_aux(real)
        return logits_fake, logits_real, aux_fake, aux_real
