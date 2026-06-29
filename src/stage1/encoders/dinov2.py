from typing import Optional

from transformers import AutoModel
from torch import nn
import torch
from . import register_encoder


def _load_auto_model(model_name: str, trust_remote_code: bool) -> nn.Module:
    """Load from a local HF cache first, then allow the usual HF fallback."""
    errors = (OSError, ValueError, AttributeError, KeyError)
    try:
        return AutoModel.from_pretrained(
            model_name,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
    except errors:
        try:
            return AutoModel.from_pretrained(
                model_name,
                local_files_only=False,
                trust_remote_code=trust_remote_code,
            )
        except errors:
            if "with-registers" not in str(model_name).lower():
                raise
            from transformers import Dinov2WithRegistersModel

            try:
                return Dinov2WithRegistersModel.from_pretrained(
                    model_name,
                    local_files_only=True,
                )
            except errors:
                return Dinov2WithRegistersModel.from_pretrained(
                    model_name,
                    local_files_only=False,
                )


def _maybe_disable_final_norm_affine(model: nn.Module) -> None:
    for name in ("layernorm", "norm"):
        layer = getattr(model, name, None)
        if isinstance(layer, nn.LayerNorm):
            layer.elementwise_affine = False
            layer.weight = None
            layer.bias = None


def _infer_num_register_tokens(config, model_name: str) -> int:
    value = getattr(config, "num_register_tokens", None)
    if value is not None:
        return int(value)

    model_name_l = str(model_name).lower()
    if "dinov2-with-registers" in model_name_l or "with-registers" in model_name_l:
        return 4
    if "dinov3" in model_name_l:
        return 4
    return 0


@register_encoder(name="DINOwithNorm")
@register_encoder()
class Dinov2withNorm(nn.Module):
    def __init__(
        self,
        dinov2_path: Optional[str] = None,
        model_name: Optional[str] = None,
        normalize: bool = True,
        num_register_tokens: Optional[int] = None,
        trust_remote_code: bool = False,
    ):
        super().__init__()

        model_name = model_name or dinov2_path
        if model_name is None:
            raise ValueError("Dinov2withNorm requires either 'dinov2_path' or 'model_name'.")

        self.model_name = str(model_name)
        self.encoder = _load_auto_model(
            self.model_name,
            trust_remote_code=bool(trust_remote_code),
        )

        self.encoder.requires_grad_(False)

        if normalize:
            _maybe_disable_final_norm_affine(self.encoder)

        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size

        # ViT-style DINO outputs expose:
        #   1 CLS token + optional register tokens + patch tokens.
        # DINOv2 without registers has 0 register tokens; DINOv2-with-registers
        # and DINOv3 ViT checkpoints use 4 register tokens.
        if num_register_tokens is None:
            num_register_tokens = _infer_num_register_tokens(self.encoder.config, self.model_name)
        self.num_register_tokens = int(num_register_tokens)
        if self.num_register_tokens < 0:
            raise ValueError(f"num_register_tokens must be >= 0, got {self.num_register_tokens}")
        self.num_prefix_tokens = 1 + self.num_register_tokens

    @torch.no_grad()
    def _forward_all_tokens(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x, output_hidden_states=True)
        return outputs.last_hidden_state

    @torch.no_grad()
    def dinov2_forward(self, x: torch.Tensor) -> torch.Tensor:
        all_tokens = self._forward_all_tokens(x)
        patch_tokens = all_tokens[:, self.num_prefix_tokens:]
        return patch_tokens

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dinov2_forward(x)

    @torch.no_grad()
    def forward_with_global(self, x: torch.Tensor):
        """
        Returns:
            patch_tokens:  [B, N, C]
            global_tokens: [B, 1 + num_register_tokens, C]
        """
        all_tokens = self._forward_all_tokens(x)
        global_tokens = all_tokens[:, :self.num_prefix_tokens]
        patch_tokens = all_tokens[:, self.num_prefix_tokens:]
        return patch_tokens, global_tokens
