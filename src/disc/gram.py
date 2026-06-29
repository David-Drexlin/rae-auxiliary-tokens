from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    # x: [B, C, H, W]
    b, c, h, w = x.shape
    f = x.view(b, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2))
    return g / (c * h * w)


class VGGFeatures(nn.Module):
    def __init__(self, layer_ids: Sequence[int] = (3, 8, 15)):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.layer_ids = tuple(layer_ids)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, x: torch.Tensor):
        # x expected in [0,1]
        x = (x - self.mean) / self.std
        outs = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_ids:
                outs.append(x)
            if i >= self.layer_ids[-1]:
                break
        return outs


class GramLoss(nn.Module):
    def __init__(
        self,
        layer_ids: Sequence[int] = (3, 8, 15),
        layer_weights: Sequence[float] = (1.0, 1.0, 1.0),
        loss_type: str = "l1",
    ):
        super().__init__()
        if len(layer_ids) != len(layer_weights):
            raise ValueError("layer_ids and layer_weights must match in length")
        self.feat = VGGFeatures(layer_ids)
        self.layer_weights = tuple(float(w) for w in layer_weights)
        self.loss_type = loss_type.lower()

    def forward(self, real: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        real = real.clamp(0.0, 1.0)
        recon = recon.clamp(0.0, 1.0)

        with torch.no_grad():
            real_feats = self.feat(real)
        recon_feats = self.feat(recon)

        loss = 0.0
        for w, fr, ff in zip(self.layer_weights, real_feats, recon_feats):
            gr = gram_matrix(fr.float())
            gf = gram_matrix(ff.float())
            if self.loss_type in ("l2", "mse"):
                loss = loss + w * F.mse_loss(gf, gr)
            else:
                loss = loss + w * F.l1_loss(gf, gr)
        return loss