from __future__ import annotations

import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config

from . import register_encoder


@register_encoder()
class UNI2Encoder(nn.Module):
    def __init__(
        self,
        model_name: str = "hf_hub:MahmoodLab/UNI2-h",
        pretrained: bool = True,
        img_size: int = 224,
        patch_size: int = 14,
        depth: int = 24,
        num_heads: int = 24,
        init_values: float = 1e-5,
        embed_dim: int = 1536,
        mlp_ratio: float = 2.66667 * 2,
        num_classes: int = 0,
        no_embed_class: bool = True,
        reg_tokens: int = 8,
        dynamic_img_size: bool = True,
        drop_prefix_tokens: bool = True,
    ):
        super().__init__()

        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            patch_size=patch_size,
            depth=depth,
            num_heads=num_heads,
            init_values=init_values,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            no_embed_class=no_embed_class,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=reg_tokens,
            dynamic_img_size=dynamic_img_size,
        )
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        cfg = resolve_data_config(self.encoder.pretrained_cfg, model=self.encoder)
        self.register_buffer(
            "image_mean",
            torch.tensor(cfg["mean"], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(cfg["std"], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        ps = getattr(self.encoder.patch_embed, "patch_size", patch_size)
        self.patch_size = ps[0] if isinstance(ps, tuple) else int(ps)
        self.hidden_size = int(getattr(self.encoder, "embed_dim", embed_dim))

        self.num_prefix_tokens = int(getattr(self.encoder, "num_prefix_tokens", 0))
        self.num_reg_tokens = int(getattr(self.encoder, "num_reg_tokens", 0))
        self.drop_prefix_tokens = bool(drop_prefix_tokens)

    def _split_tokens(self, z: torch.Tensor):
        if self.num_prefix_tokens > 0:
            global_tokens = z[:, : self.num_prefix_tokens]   # CLS + registers
            patch_tokens = z[:, self.num_prefix_tokens :]
        else:
            global_tokens = z[:, :0]
            patch_tokens = z
        return patch_tokens, global_tokens

    @torch.no_grad()
    def uni2_forward_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder.forward_features(x)
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"Expected tensor from UNI2 forward_features, got {type(z)}")
        return z

    @torch.no_grad()
    def forward_with_global(self, x: torch.Tensor):
        z = self.uni2_forward_features(x)
        patch_tokens, global_tokens = self._split_tokens(z)
        return patch_tokens, global_tokens

    @torch.no_grad()
    def uni2_forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.uni2_forward_features(x)

        if self.drop_prefix_tokens and self.num_prefix_tokens > 0:
            z = z[:, self.num_prefix_tokens:]

        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.uni2_forward(x)