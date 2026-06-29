# disc/uni_perceptual.py
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config


def _to_01(x: torch.Tensor, assume_minus1_1: bool) -> torch.Tensor:
    if assume_minus1_1:
        return ((x + 1.0) * 0.5).clamp(0.0, 1.0)
    # fallback: accept [0,1] or [-1,1]
    if x.min() < 0:
        x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0)


def _norm_tokens(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return t / (t.norm(dim=-1, keepdim=True) + eps)


class UNIPerceptualLoss(nn.Module):
    """
    UNI feature loss with AMP + early exit.
    Collects ViT block outputs at selected indices and stops after max(layers).
    """

    def __init__(
        self,
        model_id: str = "hf-hub:MahmoodLab/uni",
        init_values: float = 1e-5,
        dynamic_img_size: bool = True,
        layers: Optional[Sequence[int]] = None,
        drop_tokens: int = 1,
        force_fp32: bool = False,
        amp_dtype: str = "bf16",         # "bf16" or "fp16"
        normalize_tokens: bool = True,
        assume_minus1_1: bool = True,
    ) -> None:
        super().__init__()

        self.model = timm.create_model(
            model_id,
            pretrained=True,
            init_values=init_values,
            dynamic_img_size=dynamic_img_size,
        )
        self.model.eval()
        self.model.requires_grad_(False)

        cfg = resolve_data_config(self.model.pretrained_cfg, model=self.model)
        self.register_buffer("mean", torch.tensor(cfg["mean"], dtype=torch.float32)[None, :, None, None])
        self.register_buffer("std", torch.tensor(cfg["std"], dtype=torch.float32)[None, :, None, None])

        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise RuntimeError("UNI perceptual: expected timm ViT with `.blocks`.")

        self.n_blocks = len(blocks)

        if layers is None:
            # default: early/mid (avoid last block)
            cand = {max(0, self.n_blocks // 6),
                    max(0, self.n_blocks // 3),
                    max(0, self.n_blocks // 2)}
            self.layers = sorted(cand)
        else:
            self.layers = sorted({int(i) for i in layers})

        if any(i < 0 or i >= self.n_blocks for i in self.layers):
            raise ValueError(f"UNI perceptual: layers {self.layers} out of range (n_blocks={self.n_blocks}).")

        self.max_layer = max(self.layers)
        self.drop_tokens = int(drop_tokens)
        self.force_fp32 = bool(force_fp32)
        self.normalize_tokens = bool(normalize_tokens)
        self.assume_minus1_1 = bool(assume_minus1_1)

        if amp_dtype == "bf16":
            self._amp_dtype = torch.bfloat16
        elif amp_dtype == "fp16":
            self._amp_dtype = torch.float16
        else:
            raise ValueError("amp_dtype must be 'bf16' or 'fp16'.")

        # speed niceties
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_01(x, self.assume_minus1_1)
        # buffers already on same device as module; avoid .to(x.device)
        return (x - self.mean) / self.std

    def _forward_upto(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Manual timm ViT forward up to max_layer, collecting features at self.layers.
        """
        m = self.model

        x = m.patch_embed(x)
        if hasattr(m, "_pos_embed"):
            x = m._pos_embed(x)  # handles cls + pos + interpolation
        else:
            # fallback (rare)
            B = x.shape[0]
            if hasattr(m, "cls_token") and m.cls_token is not None:
                cls = m.cls_token.expand(B, -1, -1)
                x = torch.cat((cls, x), dim=1)
            if hasattr(m, "pos_embed") and m.pos_embed is not None:
                x = x + m.pos_embed
            if hasattr(m, "pos_drop"):
                x = m.pos_drop(x)

        if hasattr(m, "norm_pre") and m.norm_pre is not None:
            x = m.norm_pre(x)

        feats: List[torch.Tensor] = []
        for i, blk in enumerate(m.blocks):
            x = blk(x)
            if i in self.layers:
                feats.append(x)
            if i == self.max_layer:
                break

        if not feats:
            raise RuntimeError("UNI perceptual: no activations collected.")
        return feats

    def forward(self, real: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        real = self._prep(real)
        recon = self._prep(recon)

        # real feats: no grad
        with torch.inference_mode(), torch.autocast("cuda", dtype=self._amp_dtype):
            real_feats = self._forward_upto(real)

        # recon feats: keep grad w.r.t recon
        with torch.autocast("cuda", dtype=self._amp_dtype):
            recon_feats = self._forward_upto(recon)

        losses = []
        for rf, ff in zip(real_feats, recon_feats):
            # [B, N, C]
            if rf.shape[1] > self.drop_tokens:
                rf = rf[:, self.drop_tokens:, :]
            if ff.shape[1] > self.drop_tokens:
                ff = ff[:, self.drop_tokens:, :]

            if self.normalize_tokens:
                rf = _norm_tokens(rf)
                ff = _norm_tokens(ff)

            if self.force_fp32:
                rf = rf.float()
                ff = ff.float()

            diff = (ff.float() - rf.float())
            losses.append((diff * diff).mean())

        return torch.stack(losses).mean()
