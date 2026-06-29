# RAE_ROOT_PLACEHOLDER/src/stage2/models/unet_backend.py
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from utils.model_utils import instantiate_from_config


def _has_arg(mod: nn.Module, name: str) -> bool:
    try:
        return name in inspect.signature(mod.forward).parameters
    except Exception:
        return False


@dataclass
class MetaSpec:
    """
    Describes how to interpret `meta` coming from the dataloader.

    meta can be:
      - dict[str, Tensor] already (domain_name -> tensor)
      - Tensor(B,M) where columns follow meta_fields order
    """
    meta_fields: Sequence[str]
    meta_types: Sequence[str]  # "categorical" | "positional"
    meta_null: Dict[str, Union[int, float]]  # per-field null value for CFG/uncond
    # Optional: rename columns to model domain keys
    domain_alias: Optional[Dict[str, str]] = None


class UNetBackend(nn.Module):
    """
    Meta-native pixel backend.

    External API (used by pixel_train.py):
      forward(x, t, y=None, meta=None) -> Tensor(B,C,H,W) or (B,2C,H,W) if learn_sigma
      forward_with_cfg(...)
      forward_with_autoguidance(...)

    Internally maps:
      y    -> class_labels
      meta -> domain_labels dict (and optional positional domains)
    """

    def __init__(
        self,
        unet: Dict[str, Any],
        in_channels: int = 3,
        learn_sigma: bool = False,
        cond: Optional[Dict[str, Any]] = None,
        # For UNet2DConditionModel-style models that REQUIRE encoder_hidden_states
        null_context_tokens: int = 1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.learn_sigma = bool(learn_sigma)
        self.out_channels = self.in_channels * (2 if self.learn_sigma else 1)

        self.unet: nn.Module = instantiate_from_config(unet)

        cond = cond or {}
        self.num_classes = int(cond.get("num_classes", 0))
        self.null_label = int(cond.get("null_label", self.num_classes))  # common: null = num_classes

        # --- Meta spec ---
        ms = cond.get("meta", None)
        if ms is None:
            self.meta_spec: Optional[MetaSpec] = None
        else:
            self.meta_spec = MetaSpec(
                meta_fields=list(ms.get("fields", [])),
                meta_types=list(ms.get("types", [])),
                meta_null=dict(ms.get("null", {})),
                domain_alias=dict(ms.get("domain_alias", {})) if ms.get("domain_alias", None) else None,
            )
            if len(self.meta_spec.meta_fields) != len(self.meta_spec.meta_types):
                raise ValueError("cond.meta.fields and cond.meta.types must have same length")

        # underlying signature capabilities
        self._wants_encoder_hidden_states = _has_arg(self.unet, "encoder_hidden_states")
        self._supports_class_labels = _has_arg(self.unet, "class_labels")
        self._supports_domain_labels = _has_arg(self.unet, "domain_labels")

        # For condition UNets, create a learnable null context token so cross-attn blocks work
        self.null_context_tokens = int(null_context_tokens)
        self.null_context: Optional[nn.Parameter] = None
        if self._wants_encoder_hidden_states:
            # infer cross_attention_dim
            cross_dim = None
            if hasattr(self.unet, "config") and hasattr(self.unet.config, "cross_attention_dim"):
                cd = self.unet.config.cross_attention_dim
                cross_dim = cd[0] if isinstance(cd, (tuple, list)) else int(cd)
            if cross_dim is None:
                # fallback: use a reasonable default (often equals time_embed_dim in your UNet2DModel)
                cross_dim = int(cond.get("cond_dim", 1024))
            self.null_context = nn.Parameter(torch.zeros(1, self.null_context_tokens, cross_dim))
            nn.init.normal_(self.null_context, std=0.02)

    # -------------------------
    # meta parsing
    # -------------------------
    def _meta_to_domain_labels(
        self, meta: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Optional[Dict[str, torch.Tensor]]:
        if meta is None:
            return None
        if isinstance(meta, dict):
            # assume already correct
            return meta

        if self.meta_spec is None:
            raise ValueError(
                "Dataloader returned meta as a tensor but cond.meta is not configured in YAML."
            )

        if meta.ndim != 2 or meta.shape[1] != len(self.meta_spec.meta_fields):
            raise ValueError(
                f"meta must be (B,{len(self.meta_spec.meta_fields)}) but got {tuple(meta.shape)}"
            )

        out: Dict[str, torch.Tensor] = {}
        for j, (name, typ) in enumerate(zip(self.meta_spec.meta_fields, self.meta_spec.meta_types)):
            dom_key = self.meta_spec.domain_alias.get(name, name) if self.meta_spec.domain_alias else name
            col = meta[:, j]
            if typ == "categorical":
                out[dom_key] = col.long()
            elif typ == "positional":
                out[dom_key] = col.float()
            else:
                raise ValueError(f"Unknown meta type '{typ}' for field '{name}'")
        return out

    def _make_uncond_domain_labels(
        self, domain_labels: Optional[Dict[str, torch.Tensor]]
    ) -> Optional[Dict[str, torch.Tensor]]:
        if domain_labels is None:
            return None
        if self.meta_spec is None:
            # if meta came as dict without spec, default nulls to 0
            return {k: torch.zeros_like(v) for k, v in domain_labels.items()}

        nulls: Dict[str, torch.Tensor] = {}
        for k, v in domain_labels.items():
            # map back through alias if provided
            base_key = k
            null_val = self.meta_spec.meta_null.get(base_key, self.meta_spec.meta_null.get(k, None))
            if null_val is None:
                # safe fallback, but you SHOULD set this in YAML for CFG correctness
                null_val = 0.0 if v.dtype.is_floating_point else 0
            nulls[k] = torch.full_like(v, null_val)
        return nulls

    def _encoder_hidden_states(self, batch: int, device, dtype) -> torch.Tensor:
        assert self.null_context is not None
        ctx = self.null_context.to(device=device, dtype=dtype)
        return ctx.expand(batch, -1, -1).contiguous()

    # -------------------------
    # forward
    # -------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        meta: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        domain_labels = self._meta_to_domain_labels(meta)

        unet_kwargs: Dict[str, Any] = {}
        if self._supports_class_labels and y is not None:
            unet_kwargs["class_labels"] = y.long()
        if self._supports_domain_labels and domain_labels is not None:
            unet_kwargs["domain_labels"] = domain_labels

        if self._wants_encoder_hidden_states:
            # learned null context so you don't need text encoders
            unet_kwargs["encoder_hidden_states"] = self._encoder_hidden_states(
                batch=x.shape[0], device=x.device, dtype=x.dtype
            )

        # your UNets accept (sample, timestep, ...) and return .sample
        out = self.unet(x, t, **unet_kwargs, **kwargs)
        if hasattr(out, "sample"):
            out = out.sample

        if out.shape[1] != self.out_channels:
            raise ValueError(
                f"UNetBackend expected out_channels={self.out_channels}, got {out.shape[1]} "
                f"(learn_sigma={self.learn_sigma}, in_channels={self.in_channels})"
            )
        return out

    # -------------------------
    # guidance
    # -------------------------
    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
        meta: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        **kwargs,
    ) -> torch.Tensor:
        """
        Standard CFG: uses cond/uncond pair, returns guided eps.
        Expects x to be full batch; internally duplicates half-batch.
        """
        B = x.shape[0]
        half = B // 2
        if half == 0:
            raise ValueError("Batch too small for CFG (need B>=2).")

        x_half = x[:half]
        x_in = torch.cat([x_half, x_half], dim=0)

        # cond
        y_cond = y[:half].long()

        # domain labels
        dom = self._meta_to_domain_labels(meta)
        dom_cond = None if dom is None else {k: v[:half] for k, v in dom.items()}
        dom_uncond = self._make_uncond_domain_labels(dom_cond)

        # uncond class
        y_uncond = torch.full_like(y_cond, self.null_label)

        y_in = torch.cat([y_cond, y_uncond], dim=0)
        dom_in = None
        if dom_cond is not None:
            dom_in = {k: torch.cat([dom_cond[k], dom_uncond[k]], dim=0) for k in dom_cond.keys()}

        out = self.forward(x_in, t, y=y_in, meta=dom_in, **kwargs)

        eps = out[:, : self.in_channels]
        rest = out[:, self.in_channels :] if self.learn_sigma else None

        eps_c, eps_u = torch.split(eps, half, dim=0)

        t_half = t[:half]
        t_min, t_max = cfg_interval
        gate = ((t_half >= t_min) & (t_half <= t_max)).view(-1, *[1] * (eps_c.ndim - 1))
        guided = torch.where(gate, eps_u + cfg_scale * (eps_c - eps_u), eps_c)

        eps_out = torch.cat([guided, guided], dim=0)
        if rest is None:
            return eps_out
        return torch.cat([eps_out, rest], dim=1)

    def forward_with_autoguidance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
        additional_model_forward,
        meta: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        **kwargs,
    ) -> torch.Tensor:
        """
        Autoguidance: blend main model eps with aux model eps in [t_min,t_max].
        additional_model_forward must have SAME external signature: (x,t,y,meta)->out
        """
        out = self.forward(x, t, y=y, meta=meta, **kwargs)
        aux = additional_model_forward(x, t, y=y, meta=meta, **kwargs)

        if hasattr(aux, "sample"):
            aux = aux.sample

        eps = out[:, : self.in_channels]
        aux_eps = aux[:, : self.in_channels]

        t_min, t_max = cfg_interval
        gate = ((t >= t_min) & (t <= t_max)).view(-1, *[1] * (eps.ndim - 1))
        eps = torch.where(gate, aux_eps + cfg_scale * (eps - aux_eps), eps)

        if not self.learn_sigma:
            return eps
        return torch.cat([eps, out[:, self.in_channels :]], dim=1)
