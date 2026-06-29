# RAE_ROOT_PLACEHOLDER/src/stage2/models/unet2d_condition_domain.py
from __future__ import annotations

from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn

from diffusers import UNet2DConditionModel as HFUNet2DConditionModel


def _infer_vocab_size(values: List[Any]) -> int:
    """
    YAML usually gives values like [0,1,2]. Safer than len(values) in case values are not consecutive.
    """
    if len(values) == 0:
        return 1
    if all(isinstance(v, (int, float)) for v in values):
        return int(max(values)) + 1
    return len(values)


class UNet2DConditionModelWithDomains(HFUNet2DConditionModel):
    """
    Diffusers UNet2DConditionModel + additive domain embeddings.

    - domain_embeds: dict(domain_name -> list of allowed ids) used only for vocab sizing
    - forward(..., domain_labels=...) where domain_labels is dict(domain_name -> LongTensor[B])

    Missing domain_labels are treated as 0 (null token).
    """

    def __init__(self, domain_embeds: Optional[Dict[str, List[Any]]] = None, **kwargs):
        super().__init__(**kwargs)

        self.domain_embeddings: Optional[nn.ModuleDict] = None
        if domain_embeds is not None and len(domain_embeds) > 0:
            # time embedding dim is the output dim of the timestep embedding MLP
            # (works across diffusers versions)
            time_embed_dim = int(self.time_embedding.linear_1.out_features)

            self.domain_embeddings = nn.ModuleDict()
            for name, values in domain_embeds.items():
                vocab = _infer_vocab_size(list(values))
                self.domain_embeddings[name] = nn.Embedding(vocab, time_embed_dim)
                nn.init.normal_(self.domain_embeddings[name].weight, std=0.02)

        # used to pass domain_labels into get_class_embed without rewriting the whole forward
        self._last_domain_labels: Optional[Dict[str, torch.Tensor]] = None

    def _domain_embed_sum(self, sample: torch.Tensor) -> Optional[torch.Tensor]:
        if self.domain_embeddings is None:
            return None

        B = sample.shape[0]
        device = sample.device

        dom = self._last_domain_labels or {}
        out = None
        for name, emb in self.domain_embeddings.items():
            if name in dom and dom[name] is not None:
                ids = dom[name].to(device=device)
                if ids.dtype != torch.long:
                    ids = ids.long()
            else:
                # missing domain => null token 0
                ids = torch.zeros((B,), device=device, dtype=torch.long)

            e = emb(ids).to(dtype=sample.dtype)  # (B, D)
            out = e if out is None else (out + e)

        return out

    # key trick: override get_class_embed so super().forward will automatically add our domain embedding
    def get_class_embed(self, sample: torch.Tensor, class_labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        class_emb = super().get_class_embed(sample=sample, class_labels=class_labels)

        dom_emb = self._domain_embed_sum(sample)
        if dom_emb is None:
            return class_emb

        # If no class embedding is used, still return dom_emb so it gets added into `emb`.
        if class_emb is None:
            return dom_emb

        # Normal additive case
        return class_emb + dom_emb

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        domain_labels: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        # stash domain labels for get_class_embed() during this forward pass
        self._last_domain_labels = domain_labels

        try:
            return super().forward(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                class_labels=class_labels,
                **kwargs,
            )
        finally:
            self._last_domain_labels = None