from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .DDT import DiTwDDTHead
from .model_utils import GaussianFourierEmbedding


class DiTwDDTHeadV3(DiTwDDTHead):
    """
    Shared-token DDT variant with explicit patch/aux timestep embeddings.

    Relative to DiTwDDTHead:
    - keeps the shared encoder token stream (aux prefix + optional registers + patches)
    - accepts structured time input `(t_patch, t_aux)` or `{"patch": ..., "aux": ...}`
    - uses separate timestep embedders for patch and aux tokens inside the shared encoder

    If `t` is a plain tensor, the model falls back to the same time for both token families.
    """

    def __init__(self, *args, use_dual_time_embed: bool = True, use_token_type_embed: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_dual_time_embed = bool(use_dual_time_embed)
        self.use_token_type_embed = bool(use_token_type_embed)
        self.aux_t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)
        torch.nn.init.normal_(self.aux_t_embedder.mlp[0].weight, std=0.02)
        torch.nn.init.normal_(self.aux_t_embedder.mlp[2].weight, std=0.02)
        if self.use_token_type_embed:
            self.enc_patch_type_embed = torch.nn.Parameter(torch.zeros(1, 1, self.encoder_hidden_size))
            self.enc_aux_type_embed = torch.nn.Parameter(torch.zeros(1, 1, self.encoder_hidden_size))
            self.enc_reg_type_embed = torch.nn.Parameter(torch.zeros(1, 1, self.encoder_hidden_size))
            self.dec_patch_type_embed = torch.nn.Parameter(torch.zeros(1, 1, self.decoder_hidden_size))
            self.dec_reg_type_embed = torch.nn.Parameter(torch.zeros(1, 1, self.decoder_hidden_size))
            torch.nn.init.normal_(self.enc_patch_type_embed, std=0.02)
            torch.nn.init.normal_(self.enc_aux_type_embed, std=0.02)
            torch.nn.init.normal_(self.enc_reg_type_embed, std=0.02)
            torch.nn.init.normal_(self.dec_patch_type_embed, std=0.02)
            torch.nn.init.normal_(self.dec_reg_type_embed, std=0.02)
        else:
            self.enc_patch_type_embed = None
            self.enc_aux_type_embed = None
            self.enc_reg_type_embed = None
            self.dec_patch_type_embed = None
            self.dec_reg_type_embed = None

    @staticmethod
    def _extract_times(t, batch_size: int, device: torch.device):
        if torch.is_tensor(t):
            return t, t
        if isinstance(t, (tuple, list)):
            if len(t) != 2:
                raise ValueError(f"Expected time tuple/list of length 2, got {len(t)}")
            return t[0], t[1]
        if isinstance(t, dict):
            if "patch" not in t:
                raise ValueError("Time dict must provide key 'patch'")
            patch_t = t["patch"]
            aux_t = t.get("aux", patch_t)
            return patch_t, aux_t
        raise TypeError(f"Unsupported time input type: {type(t)}")

    @staticmethod
    def _primary_time(t):
        if torch.is_tensor(t):
            return t
        if isinstance(t, (tuple, list)):
            return t[0]
        if isinstance(t, dict):
            return t["patch"]
        raise TypeError(f"Unsupported time input type: {type(t)}")

    @staticmethod
    def _slice_time(t, slc):
        if torch.is_tensor(t):
            return t[slc]
        if isinstance(t, tuple):
            return tuple(v[slc] for v in t)
        if isinstance(t, list):
            return [v[slc] for v in t]
        if isinstance(t, dict):
            return {k: v[slc] for k, v in t.items()}
        raise TypeError(f"Unsupported time input type: {type(t)}")

    @staticmethod
    def _slice_meta(meta, slc):
        if meta is None:
            return None
        if isinstance(meta, dict):
            return {k: v[slc] for k, v in meta.items()}
        return meta[slc]

    def _build_encoder_tokenwise_condition(
        self,
        patch_cond: torch.Tensor,
        aux_cond: torch.Tensor,
        patch_token_count: int,
        aux_prefix_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        if aux_prefix_len > 0:
            parts.append(aux_cond.unsqueeze(1).expand(-1, aux_prefix_len, -1))
        if self.num_register_tokens > 0:
            parts.append(patch_cond.unsqueeze(1).expand(-1, self.num_register_tokens, -1))
        parts.append(patch_cond.unsqueeze(1).expand(-1, patch_token_count, -1))
        return torch.cat(parts, dim=1).to(dtype=dtype)

    def _add_encoder_type_embeddings(
        self,
        tokens: torch.Tensor,
        patch_token_count: int,
        aux_prefix_len: int,
    ) -> torch.Tensor:
        if not self.use_token_type_embed:
            return tokens
        pieces: List[torch.Tensor] = []
        cursor = 0
        if aux_prefix_len > 0:
            aux_tokens = tokens[:, cursor : cursor + aux_prefix_len, :]
            pieces.append(aux_tokens + self.enc_aux_type_embed.to(dtype=tokens.dtype))
            cursor += aux_prefix_len
        if self.num_register_tokens > 0:
            reg_tokens = tokens[:, cursor : cursor + self.num_register_tokens, :]
            pieces.append(reg_tokens + self.enc_reg_type_embed.to(dtype=tokens.dtype))
            cursor += self.num_register_tokens
        patch_tokens = tokens[:, cursor : cursor + patch_token_count, :]
        pieces.append(patch_tokens + self.enc_patch_type_embed.to(dtype=tokens.dtype))
        return torch.cat(pieces, dim=1)

    def _add_decoder_type_embeddings(self, tokens: torch.Tensor, has_register_prefix: bool) -> torch.Tensor:
        if not self.use_token_type_embed:
            return tokens
        if not has_register_prefix or self.num_register_tokens <= 0:
            return tokens + self.dec_patch_type_embed.to(dtype=tokens.dtype)
        reg_tokens = tokens[:, : self.num_register_tokens, :] + self.dec_reg_type_embed.to(dtype=tokens.dtype)
        patch_tokens = tokens[:, self.num_register_tokens :, :] + self.dec_patch_type_embed.to(dtype=tokens.dtype)
        return torch.cat([reg_tokens, patch_tokens], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        t,
        y: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        meta: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        aux: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, aux = self._extract_inputs(x, aux)

        batch_size = x.shape[0]
        device = x.device
        patch_t, aux_t = self._extract_times(t, batch_size=batch_size, device=device)
        patch_t = patch_t.to(device=device, dtype=x.dtype)
        aux_t = aux_t.to(device=device, dtype=x.dtype)

        t_patch_emb = self.t_embedder(patch_t)
        t_aux_emb = self.aux_t_embedder(aux_t if self.use_dual_time_embed else patch_t)
        y_emb = self.y_embedder(y, self.training)
        shared = y_emb

        if self.meta_embedders is not None:
            if meta is None:
                meta = torch.zeros((batch_size, len(self.meta_embedders)), device=device, dtype=torch.long)
            shared = shared + self._meta_embed(meta).to(dtype=shared.dtype)

        c_patch = F.silu(t_patch_emb + shared)
        c_aux = F.silu(t_aux_emb + shared)
        aux_out = None

        if s is None:
            s = self.s_embedder(x)
            if self.use_pos_embed and self.pos_embed is not None:
                s = s + self.pos_embed

            patch_token_count = s.shape[1]
            aux_prefix_len = 0
            enc_feat_rope = self.enc_feat_rope

            if self.predict_aux:
                if aux is None:
                    aux = torch.zeros((batch_size, self.aux_dim), device=device, dtype=x.dtype)
                if aux.ndim != 2:
                    raise ValueError(f"aux must be (B, aux_dim), got {tuple(aux.shape)}")
                if aux.shape[1] != self.aux_dim:
                    raise ValueError(f"aux has dim={aux.shape[1]} but expected aux_dim={self.aux_dim}")

                aux_prefix = self.aux_embedder(aux.to(dtype=s.dtype)).unsqueeze(1) + self.aux_pos_embed.to(dtype=s.dtype)
                s = torch.cat([aux_prefix, s], dim=1)
                aux_prefix_len = 1
            elif self.predict_aux_tokens:
                if aux is None:
                    aux = torch.zeros(
                        (batch_size, self.num_aux_tokens, self.aux_token_dim),
                        device=device,
                        dtype=x.dtype,
                    )
                if aux.ndim != 3:
                    raise ValueError(f"aux tokens must be (B, K, aux_token_dim), got {tuple(aux.shape)}")
                if aux.shape[1] != self.num_aux_tokens:
                    raise ValueError(f"aux has K={aux.shape[1]} but expected num_aux_tokens={self.num_aux_tokens}")
                if aux.shape[2] != self.aux_token_dim:
                    raise ValueError(f"aux has dim={aux.shape[2]} but expected aux_token_dim={self.aux_token_dim}")

                aux_prefix = self.aux_tokens_embedder(aux.to(dtype=s.dtype)) + self.aux_tokens_pos_embed.to(dtype=s.dtype)
                s = torch.cat([aux_prefix, s], dim=1)
                aux_prefix_len = self.num_aux_tokens

            if self.num_register_tokens > 0:
                enc_registers = self.enc_register_tokens.expand(batch_size, -1, -1).to(dtype=s.dtype)
                s = torch.cat([s[:, :aux_prefix_len, :], enc_registers, s[:, aux_prefix_len:, :]], dim=1)

            s = self._add_encoder_type_embeddings(
                s,
                patch_token_count=patch_token_count,
                aux_prefix_len=aux_prefix_len,
            )
            enc_feat_rope = self._wrap_rope_with_prefix(self.enc_feat_rope, aux_prefix_len + self.num_register_tokens)
            enc_c = self._build_encoder_tokenwise_condition(
                patch_cond=c_patch,
                aux_cond=c_aux,
                patch_token_count=patch_token_count,
                aux_prefix_len=aux_prefix_len,
                dtype=s.dtype,
            )
            for i in range(self.num_encoder_blocks):
                s = self.blocks[i](s, enc_c, feat_rope=enc_feat_rope)

            enc_t = self._build_encoder_tokenwise_condition(
                patch_cond=t_patch_emb,
                aux_cond=t_aux_emb,
                patch_token_count=patch_token_count,
                aux_prefix_len=aux_prefix_len,
                dtype=s.dtype,
            )
            s = F.silu(enc_t + s)

            if self.predict_aux:
                s_aux = s[:, :1, :]
                s = s[:, 1:, :]
                aux_out = self.aux_final_layer(s_aux, c_aux)
            elif self.predict_aux_tokens:
                s_aux = s[:, : self.num_aux_tokens, :]
                s = s[:, self.num_aux_tokens :, :]
                aux_out = self.aux_tokens_final_layer(s_aux, c_aux)

        elif self.predict_aux or self.predict_aux_tokens:
            raise ValueError("External encoder state s=... is not supported when auxiliary prediction is enabled.")

        s = self.s_projector(s)

        x_tok = self.x_embedder(x)
        if self.use_pos_embed and self.x_pos_embed is not None:
            x_tok = x_tok + self.x_pos_embed

        dec_feat_rope = self.dec_feat_rope
        has_register_prefix = False
        if self.num_register_tokens > 0:
            dec_registers = self.dec_register_tokens.expand(batch_size, -1, -1).to(dtype=x_tok.dtype)
            x_tok = torch.cat([dec_registers, x_tok], dim=1)
            dec_feat_rope = self._wrap_rope_with_prefix(self.dec_feat_rope, self.num_register_tokens)
            has_register_prefix = True

        x_tok = self._add_decoder_type_embeddings(x_tok, has_register_prefix=has_register_prefix)

        for i in range(self.num_encoder_blocks, self.num_blocks):
            x_tok = self.blocks[i](x_tok, s, feat_rope=dec_feat_rope)

        if self.num_register_tokens > 0:
            x_tok = x_tok[:, self.num_register_tokens :, :]
            s = s[:, self.num_register_tokens :, :]

        x_tok = self.final_layer(x_tok, s)
        x_img = self.unpatchify(x_tok)
        if aux_out is not None:
            return x_img, aux_out
        return x_img

    def forward_with_cfg(
        self,
        x,
        t,
        y: torch.Tensor,
        cfg_scale: float,
        cfg_scale_aux: float = 0.0,
        meta: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        aux: Optional[torch.Tensor] = None,
        cfg_interval: Union[Tuple[float, float], List[Tuple[float, float]]] = (0.0, 1.0),
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, aux = self._extract_inputs(x, aux)
        half_n = len(x) // 2
        half_x = x[:half_n]
        combined = torch.cat([half_x, half_x], dim=0)

        combined_aux = None
        if self.predict_aux or self.predict_aux_tokens:
            if aux is None:
                raise ValueError("forward_with_cfg requires aux when auxiliary prediction is enabled")
            half_aux = aux[:half_n]
            combined_aux = torch.cat([half_aux, half_aux], dim=0)

        model_out = self.forward(combined, t, y, meta=meta, aux=combined_aux, **kwargs)
        x_out, aux_out = self._split_outputs(model_out)

        eps, rest = x_out[:, : self.in_channels], x_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        if aux_out is not None:
            cond_aux, uncond_aux = torch.split(aux_out, len(aux_out) // 2, dim=0)

        t_half = self._primary_time(t)[:half_n]
        in_mask = self._interval_mask(t_half, cfg_interval, default_all=not self._is_interval_list(cfg_interval))
        half_eps = torch.where(
            self._mask_like(in_mask, cond_eps),
            uncond_eps + float(cfg_scale) * (cond_eps - uncond_eps),
            cond_eps,
        )

        eps = torch.cat([half_eps, half_eps], dim=0)
        x_return = torch.cat([eps, rest], dim=1)

        if aux_out is not None:
            half_aux = torch.where(
                self._mask_like(in_mask, cond_aux),
                uncond_aux + float(cfg_scale_aux) * (cond_aux - uncond_aux),
                cond_aux,
            )
            aux_return = torch.cat([half_aux, half_aux], dim=0)
            return x_return, aux_return

        return x_return

    def forward_with_autoguidance(
        self,
        x,
        t,
        y: torch.Tensor,
        cfg_scale: float,
        additional_model_forward,
        cfg_scale_aux: float = 0.0,
        meta: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        aux: Optional[torch.Tensor] = None,
        cfg_interval: Union[Tuple[float, float], List[Tuple[float, float]]] = (0.0, 1.0),
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, aux = self._extract_inputs(x, aux)
        half_n = len(x) // 2
        half = x[:half_n]
        t_half = self._slice_time(t, slice(0, half_n))
        y_half = y[:half_n]
        meta_half = self._slice_meta(meta, slice(0, half_n))
        aux_half = aux[:half_n] if aux is not None else None

        model_out = self.forward(half, t_half, y_half, meta=meta_half, aux=aux_half, **kwargs)
        try:
            ag_model_out = additional_model_forward(half, t_half, y_half, meta=meta_half, aux=aux_half, **kwargs)
        except TypeError:
            ag_model_out = additional_model_forward(half, t_half, y_half)

        x_out, aux_out = self._split_outputs(model_out)
        ag_x_out, ag_aux_out = self._split_outputs(ag_model_out)

        eps = x_out[:, : self.in_channels]
        ag_eps = ag_x_out[:, : self.in_channels]

        primary_half_t = self._primary_time(t_half)
        in_mask = self._interval_mask(primary_half_t, cfg_interval, default_all=not self._is_interval_list(cfg_interval))
        out = torch.where(
            self._mask_like(in_mask, eps),
            ag_eps + float(cfg_scale) * (eps - ag_eps),
            eps,
        )

        x_ret = torch.cat([out, out], dim=0)

        if aux_out is not None:
            if ag_aux_out is None:
                raise ValueError("additional_model_forward must also return aux output when auxiliary prediction is enabled")
            aux_ret_half = torch.where(
                self._mask_like(in_mask, aux_out),
                ag_aux_out + float(cfg_scale_aux) * (aux_out - ag_aux_out),
                aux_out,
            )
            aux_ret = torch.cat([aux_ret_half, aux_ret_half], dim=0)
            return x_ret, aux_ret

        return x_ret
