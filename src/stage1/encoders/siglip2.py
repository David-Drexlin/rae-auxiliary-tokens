from torch import nn
import torch
from . import register_encoder
from transformers import SiglipModel


@register_encoder()
class SigLIP2wNorm(nn.Module):
    def __init__(self, model_name: str, num_tokens: int = 256):
        super().__init__()
        self.model_name = model_name
        self.num_tokens = num_tokens
        self.model = SiglipModel.from_pretrained(self.model_name).vision_model

        # remove the affine of final layernorm
        self.model.post_layernorm.elementwise_affine = False
        self.model.post_layernorm.weight = None
        self.model.post_layernorm.bias = None

        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size

        # Expose a DINO-like prefix-token interface so RAE can reuse its
        # existing aux-token path with SigLIP2 pooler output.
        self.num_prefix_tokens = 1
        self.num_register_tokens = 0
        self.num_reg_tokens = 0

    @torch.no_grad()
    def _forward_outputs(self, images: torch.Tensor):
        return self.model(
            images,
            output_hidden_states=False,
            interpolate_pos_encoding=True,
        )

    @torch.no_grad()  # encoder is always frozen
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns dense visual tokens [B, N, C].
        """
        outputs = self._forward_outputs(images)
        return outputs.last_hidden_state

    @torch.no_grad()
    def forward_with_global(self, images: torch.Tensor):
        """
        Returns:
            patch_tokens:  [B, N, C]
            global_tokens: [B, 1, C]

        For SigLIP2, the single exposed global token is the model's pooled
        image embedding (`pooler_output`). This is not a literal CLS token in
        the sequence, but it behaves like a one-token global conditioning
        signal for AdaLN / prepend / cross-attention ablations.
        """
        outputs = self._forward_outputs(images)
        patch_tokens = outputs.last_hidden_state
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            raise AttributeError(
                f"{self.__class__.__name__} expected SigLIP2 to return pooler_output, "
                "but it was missing from the backend outputs."
            )
        global_tokens = pooled.unsqueeze(1)
        return patch_tokens, global_tokens
