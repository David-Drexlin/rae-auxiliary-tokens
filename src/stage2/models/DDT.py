# RAE_ROOT_PLACEHOLDER/src/stage2/models/DDT.py
"""
DDT head model for stage-2.

- Provides DiTwDDTHead (the symbol imported by stage2/__init__.py).
- Includes optional MeDi-style meta conditioning:
    * meta_num_classes: list of vocab sizes per meta field
    * meta_dropout_prob: dropout applied to summed meta embedding
    * meta_fields: optional list of field names (enables dict meta input)

Meta behavior:
- If meta_num_classes is None/empty: meta is accepted but ignored (no crash).
- If meta is provided and meta_num_classes is set: meta embedding is added into the encoder conditioning.
- forward/forward_with_cfg/forward_with_autoguidance accept meta and **kwargs for trainer compatibility.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Mlp

from .model_utils import (
    VisionRotaryEmbeddingFast,
    RMSNorm,
    SwiGLUFFN,
    GaussianFourierEmbedding,
    LabelEmbedder,
    NormAttention,
    get_2d_sincos_pos_embed,
)


def DDTModulate(x: torch.Tensor, shift: Optional[torch.Tensor], scale: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Per-segment modulation:
      x:     (B, Lx, D)
      shift: (B, L,  D) or None
      scale: (B, L,  D) or None

    Returns:
      x * (1 + scale) + shift  (with shift/scale repeated along length if needed).
    """
    if shift is None or scale is None:
        return x

    B, Lx, D = x.shape
    _, L, _ = shift.shape
    if Lx % L != 0:
        raise ValueError(f"Lx ({Lx}) must be divisible by L ({L})")
    rep = Lx // L
    if rep != 1:
        shift = shift.repeat_interleave(rep, dim=1)
        scale = scale.repeat_interleave(rep, dim=1)
    return x * (1 + scale) + shift


def DDTGate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Per-segment gating:
      x:    (B, Lx, D)
      gate: (B, L,  D)

    Returns:
      x * gate  (with gate repeated along length if needed).
    """
    B, Lx, D = x.shape
    _, L, _ = gate.shape
    if Lx % L != 0:
        raise ValueError(f"Lx ({Lx}) must be divisible by L ({L})")
    rep = Lx // L
    if rep != 1:
        gate = gate.repeat_interleave(rep, dim=1)
    return x * gate


class LightningDDTBlock(nn.Module):
    """
    DDT block:
    - Attention + MLP with AdaLN modulation.
    - Uses DDTModulate/DDTGate to allow per-segment modulation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        **block_kwargs,
    ):
        super().__init__()

        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0.0,
            )

        # AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True))
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.wo_shift = wo_shift

    def forward(self, x: torch.Tensor, c: torch.Tensor, feat_rope=None) -> torch.Tensor:
        # Ensure c broadcastable: (B, D) -> (B, 1, D)
        if c.ndim < x.ndim:
            c = c.unsqueeze(1)

        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        x = x + DDTGate(self.attn(DDTModulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope), gate_msa)
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)
        return x


class DDTFinalLayer(nn.Module):
    """Final projection layer for DDT."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, use_rmsnorm: bool = False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)

        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if c.ndim < x.ndim:
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DDTAuxFinalLayer(nn.Module):
    """Final projection layer for pooled auxiliary state."""

    def __init__(self, hidden_size: int, aux_dim: int, use_rmsnorm: bool = False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)

        self.linear = nn.Linear(hidden_size, aux_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if c.ndim < x.ndim:
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x.squeeze(1)


class DDTAuxTokensFinalLayer(nn.Module):
    """Final projection layer for token auxiliary state."""

    def __init__(self, hidden_size: int, aux_token_dim: int, use_rmsnorm: bool = False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)

        self.linear = nn.Linear(hidden_size, aux_token_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if c.ndim < x.ndim:
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTwDDTHead(nn.Module):
    """
    Two-stage DiT-with-DDT head.

    Notes on inputs:
    - This model expects "x" to have channel dimension equal to x_channel_per_token
      (= in_channels * x_patch_size^2).
      That’s the original design: you feed a token-grid where each token carries an in_channels patch.

    Meta conditioning (MeDi-style):
    - Provide meta_num_classes=[...] to enable; otherwise meta is ignored but accepted.
    - meta can be:
        * Tensor(B, M) long
        * dict[field->Tensor(B,)] long if meta_fields provided
    """

    def __init__(
        self,
        input_size: int = 1,
        patch_size: Union[List[int], int] = 1,
        in_channels: int = 768,
        hidden_size: Sequence[int] = (1152, 2048),
        depth: Sequence[int] = (28, 2),
        num_heads: Union[Sequence[int], int] = (16, 16),
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        use_pos_embed: bool = True,

        # --- MeDi / meta conditioning (optional) ---
        meta_num_classes: Optional[Sequence[int]] = None,  # e.g. [3, 3] for CelebA (0=null,1/2 bin)
        meta_dropout_prob: float = 0.0,
        meta_fields: Optional[Sequence[str]] = None,
        predict_aux: bool = False,
        aux_dim: Optional[int] = None,
        predict_aux_tokens: bool = False,
        aux_token_dim: Optional[int] = None,
        num_aux_tokens: Optional[int] = None,
        num_register_tokens: int = 0,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.predict_aux = bool(predict_aux)
        self.aux_dim = int(aux_dim) if aux_dim is not None else int(in_channels)
        self.predict_aux_tokens = bool(predict_aux_tokens)
        self.aux_token_dim = int(aux_token_dim) if aux_token_dim is not None else int(in_channels)
        self.num_aux_tokens = int(num_aux_tokens) if num_aux_tokens is not None else 0
        self.num_register_tokens = int(num_register_tokens)

        if self.predict_aux and self.predict_aux_tokens:
            raise ValueError("predict_aux and predict_aux_tokens are mutually exclusive.")
        if self.predict_aux and self.aux_dim <= 0:
            raise ValueError("aux_dim must be positive when predict_aux=True")
        if self.predict_aux_tokens and self.num_aux_tokens <= 0:
            raise ValueError("num_aux_tokens must be positive when predict_aux_tokens=True")
        if self.predict_aux_tokens and self.aux_token_dim <= 0:
            raise ValueError("aux_token_dim must be positive when predict_aux_tokens=True")
        if self.num_register_tokens < 0:
            raise ValueError("num_register_tokens must be non-negative")

        hidden_size = list(hidden_size)
        depth = list(depth)
        if len(hidden_size) != 2 or len(depth) != 2:
            raise ValueError("hidden_size and depth must be length-2 sequences: [encoder, decoder]")

        self.encoder_hidden_size = int(hidden_size[0])
        self.decoder_hidden_size = int(hidden_size[1])

        if isinstance(num_heads, int):
            self.num_heads = [int(num_heads), int(num_heads)]
        else:
            nh = list(num_heads)
            if len(nh) != 2:
                raise ValueError("num_heads must be int or length-2 sequence: [enc, dec]")
            self.num_heads = [int(nh[0]), int(nh[1])]

        self.num_encoder_blocks = int(depth[0])
        self.num_decoder_blocks = int(depth[1])
        self.num_blocks = self.num_encoder_blocks + self.num_decoder_blocks

        # patch sizes: [s_patch_size, x_patch_size]
        if isinstance(patch_size, (int, float)):
            patch_size = [int(patch_size), int(patch_size)]
        patch_size = list(patch_size)
        if len(patch_size) != 2:
            raise ValueError(f"patch_size must be int or [s_patch_size, x_patch_size], got {patch_size}")
        self.s_patch_size = int(patch_size[0])
        self.x_patch_size = int(patch_size[1])

        self.s_channel_per_token = self.in_channels * self.s_patch_size * self.s_patch_size
        self.x_channel_per_token = self.in_channels * self.x_patch_size * self.x_patch_size

        # Embedders
        self.s_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=self.s_patch_size,
            in_chans=self.s_channel_per_token,
            embed_dim=self.encoder_hidden_size,
            bias=True,
        )
        self.x_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=self.x_patch_size,
            in_chans=self.x_channel_per_token,
            embed_dim=self.decoder_hidden_size,
            bias=True,
        )

        self.s_projector = (
            nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
            if self.encoder_hidden_size != self.decoder_hidden_size
            else nn.Identity()
        )

        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, self.encoder_hidden_size, class_dropout_prob)

        # --- optional meta embedding ---
        self.meta_fields = list(meta_fields) if meta_fields is not None else None
        self.meta_num_classes = list(meta_num_classes) if meta_num_classes is not None else None
        if self.meta_num_classes is not None and len(self.meta_num_classes) > 0:
            self.meta_embedders = nn.ModuleList(
                [nn.Embedding(int(nc), self.encoder_hidden_size) for nc in self.meta_num_classes]
            )
            self.meta_dropout = nn.Dropout(float(meta_dropout_prob)) if meta_dropout_prob and meta_dropout_prob > 0 else nn.Identity()
        else:
            self.meta_embedders = None
            self.meta_dropout = None

        # final layer predicts per-token patch (patch_size=1 because embedder already tokenizes)
        self.final_layer = DDTFinalLayer(
            hidden_size=self.decoder_hidden_size,
            patch_size=1,
            out_channels=self.x_channel_per_token,
            use_rmsnorm=use_rmsnorm,
        )
        if self.predict_aux:
            self.aux_embedder = nn.Linear(self.aux_dim, self.encoder_hidden_size, bias=True)
            self.aux_pos_embed = nn.Parameter(torch.zeros(1, 1, self.encoder_hidden_size))
            self.aux_final_layer = DDTAuxFinalLayer(
                hidden_size=self.encoder_hidden_size,
                aux_dim=self.aux_dim,
                use_rmsnorm=use_rmsnorm,
            )
        else:
            self.aux_embedder = None
            self.aux_pos_embed = None
            self.aux_final_layer = None

        if self.predict_aux_tokens:
            self.aux_tokens_embedder = nn.Linear(self.aux_token_dim, self.encoder_hidden_size, bias=True)
            self.aux_tokens_pos_embed = nn.Parameter(torch.zeros(1, self.num_aux_tokens, self.encoder_hidden_size))
            self.aux_tokens_final_layer = DDTAuxTokensFinalLayer(
                hidden_size=self.encoder_hidden_size,
                aux_token_dim=self.aux_token_dim,
                use_rmsnorm=use_rmsnorm,
            )
        else:
            self.aux_tokens_embedder = None
            self.aux_tokens_pos_embed = None
            self.aux_tokens_final_layer = None

        if self.num_register_tokens > 0:
            # ViT-style learned compute tokens: participate in attention but are discarded before output.
            self.enc_register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, self.encoder_hidden_size))
            self.dec_register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, self.decoder_hidden_size))
        else:
            self.enc_register_tokens = None
            self.dec_register_tokens = None

        # Positional embeddings
        self.use_pos_embed = bool(use_pos_embed)
        if self.use_pos_embed:
            num_patches = self.s_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.encoder_hidden_size), requires_grad=False)
            self.x_pos_embed = None  # optional; keep for backward-compat
        else:
            self.pos_embed = None
            self.x_pos_embed = None

        # RoPE
        self.use_rope = bool(use_rope)
        if self.use_rope:
            enc_half_head_dim = self.encoder_hidden_size // self.num_heads[0] // 2
            enc_hw = int(math.sqrt(self.s_embedder.num_patches))
            self.enc_feat_rope = VisionRotaryEmbeddingFast(dim=enc_half_head_dim, pt_seq_len=enc_hw)

            dec_half_head_dim = self.decoder_hidden_size // self.num_heads[1] // 2
            dec_hw = int(math.sqrt(self.x_embedder.num_patches))
            self.dec_feat_rope = VisionRotaryEmbeddingFast(dim=dec_half_head_dim, pt_seq_len=dec_hw)
        else:
            self.enc_feat_rope = None
            self.dec_feat_rope = None

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                LightningDDTBlock(
                    hidden_size=(self.encoder_hidden_size if i < self.num_encoder_blocks else self.decoder_hidden_size),
                    num_heads=(self.num_heads[0] if i < self.num_encoder_blocks else self.num_heads[1]),
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_rmsnorm=use_rmsnorm,
                    use_swiglu=use_swiglu,
                    wo_shift=wo_shift,
                )
                for i in range(self.num_blocks)
            ]
        )

        self.initialize_weights()

    def _meta_embed(self, meta: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        meta -> (B, D_enc)

        Supported:
          - Tensor(B,M) long
          - dict[field -> Tensor(B,)] long  (requires meta_fields)
        """
        # meta accepted but ignored
        if self.meta_embedders is None:
            if isinstance(meta, dict):
                v0 = next(iter(meta.values()))
                return torch.zeros(v0.shape[0], self.encoder_hidden_size, device=v0.device, dtype=torch.float32)
            return torch.zeros(meta.shape[0], self.encoder_hidden_size, device=meta.device, dtype=torch.float32)

        if isinstance(meta, dict):
            if self.meta_fields is None:
                raise ValueError("meta is a dict but meta_fields was not provided in DiTwDDTHead.__init__")
            cols = []
            for k in self.meta_fields:
                if k not in meta:
                    raise KeyError(f"meta dict missing key '{k}' (meta_fields={self.meta_fields})")
                cols.append(meta[k].long())
            meta_t = torch.stack(cols, dim=1)  # (B,M)
        else:
            meta_t = meta.long() if meta.dtype != torch.long else meta

        if meta_t.ndim != 2:
            raise ValueError(f"meta must be (B,M), got {tuple(meta_t.shape)}")
        if meta_t.shape[1] != len(self.meta_embedders):
            raise ValueError(f"meta has M={meta_t.shape[1]} but meta_num_classes has {len(self.meta_embedders)}")

        embs = [emb(meta_t[:, i]) for i, emb in enumerate(self.meta_embedders)]  # list (B,D)
        out = torch.stack(embs, dim=0).sum(dim=0)  # (B,D)
        out = self.meta_dropout(out) if self.meta_dropout is not None else out
        return out

    def initialize_weights(self, xavier_uniform_init: bool = False):
        if xavier_uniform_init:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

        # PatchEmbed (Conv2d) like Linear init
        for pe in [self.x_embedder, self.s_embedder]:
            w = pe.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if pe.proj.bias is not None:
                nn.init.constant_(pe.proj.bias, 0)

        # label embedding init (support both styles)
        if hasattr(self.y_embedder, "embedding_tables"):
            for emb in self.y_embedder.embedding_tables:
                nn.init.normal_(emb.weight, std=0.02)
        else:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        if self.predict_aux:
            nn.init.normal_(self.aux_embedder.weight, std=0.02)
            nn.init.constant_(self.aux_embedder.bias, 0)
            nn.init.normal_(self.aux_pos_embed, std=0.02)

        if self.predict_aux_tokens:
            nn.init.normal_(self.aux_tokens_embedder.weight, std=0.02)
            nn.init.constant_(self.aux_tokens_embedder.bias, 0)
            nn.init.normal_(self.aux_tokens_pos_embed, std=0.02)

        if self.num_register_tokens > 0:
            nn.init.normal_(self.enc_register_tokens, std=0.02)
            nn.init.normal_(self.dec_register_tokens, std=0.02)

        # meta embedding init (optional)
        if self.meta_embedders is not None:
            for emb in self.meta_embedders:
                nn.init.normal_(emb.weight, std=0.02)

        # fixed sin-cos pos embed
        if self.use_pos_embed and self.pos_embed is not None:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.s_embedder.num_patches ** 0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # zero AdaLN mods
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # zero output
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.predict_aux:
            nn.init.constant_(self.aux_final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.aux_final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.aux_final_layer.linear.weight, 0)
            nn.init.constant_(self.aux_final_layer.linear.bias, 0)

        if self.predict_aux_tokens:
            nn.init.constant_(self.aux_tokens_final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.aux_tokens_final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.aux_tokens_final_layer.linear.weight, 0)
            nn.init.constant_(self.aux_tokens_final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, T, patch_size**2 * C_token) where patch_size is 1 here
        imgs: (B, C_token, H, W)  (C_token = x_channel_per_token)
        """
        c = self.x_channel_per_token
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        if h * w != x.shape[1]:
            raise ValueError("Token count is not a square; cannot unpatchify.")
        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape((x.shape[0], c, h * p, h * p))
        return imgs

    @staticmethod
    def _is_interval_list(cfg_interval) -> bool:
        return isinstance(cfg_interval, (list, tuple)) and len(cfg_interval) > 0 and isinstance(cfg_interval[0], (list, tuple))

    @staticmethod
    def _interval_mask(t: torch.Tensor, cfg_interval, default_all: bool = False) -> torch.Tensor:
        if cfg_interval is None:
            return torch.ones_like(t, dtype=torch.bool) if default_all else torch.zeros_like(t, dtype=torch.bool)

        if DiTwDDTHead._is_interval_list(cfg_interval):
            mask = torch.zeros_like(t, dtype=torch.bool)
            for (a, b) in cfg_interval:
                mask |= (t >= float(a)) & (t <= float(b))
            return mask

        if isinstance(cfg_interval, (list, tuple)) and len(cfg_interval) == 2:
            a, b = float(cfg_interval[0]), float(cfg_interval[1])
            return (t >= a) & (t <= b)

        return torch.ones_like(t, dtype=torch.bool) if default_all else torch.zeros_like(t, dtype=torch.bool)

    def _wrap_rope_with_prefix(self, rope_module, num_prefix_tokens: int):
        if rope_module is None:
            return None
        if num_prefix_tokens == 0:
            return rope_module

        def _rope(x):
            prefix = x[:, :, :num_prefix_tokens, :]
            tokens = x[:, :, num_prefix_tokens:, :]
            if tokens.shape[2] == 0:
                return x
            tokens = rope_module(tokens)
            return torch.cat([prefix, tokens], dim=2)

        return _rope

    def _split_outputs(self, model_out):
        if self.predict_aux or self.predict_aux_tokens:
            if not isinstance(model_out, (tuple, list)) or len(model_out) != 2:
                raise ValueError("Expected model output to be (x_out, aux_out) when auxiliary prediction is enabled")
            return model_out[0], model_out[1]
        return model_out, None

    @staticmethod
    def _mask_like(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return mask.view(-1, *([1] * (ref.ndim - 1)))

    def _extract_inputs(self, x, aux: Optional[torch.Tensor]):
        if isinstance(x, (tuple, list)):
            if len(x) != 2:
                raise ValueError(f"Expected tuple/list input of length 2, got {len(x)}")
            x, state_aux = x
            if aux is None:
                aux = state_aux
        return x, aux

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        meta: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        aux: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x:    (B, x_channel_per_token, H, W) or tuple((B, x_channel_per_token, H, W), aux_state)
        t:    (B,) float or tensor
        y:    (B,) long
        meta: optional (B,M) long or dict
        """
        x, aux = self._extract_inputs(x, aux)

        # Conditioning vector in encoder space
        t_emb = self.t_embedder(t)                   # (B, D_enc)
        y_emb = self.y_embedder(y, self.training)    # (B, D_enc)
        c = t_emb + y_emb


        if self.meta_embedders is not None:
            if meta is None:
                B = x.shape[0]
                M = len(self.meta_embedders)
                meta = torch.zeros((B, M), device=x.device, dtype=torch.long)
            c = c + self._meta_embed(meta).to(dtype=c.dtype)

        c = F.silu(c)
        aux_out = None

        if s is None:
            # Encode s from x (original behavior)
            s = self.s_embedder(x)  # (B, Ls, D_enc)
            if self.use_pos_embed and self.pos_embed is not None:
                s = s + self.pos_embed

            aux_prefix_len = 0
            enc_feat_rope = self.enc_feat_rope

            if self.predict_aux:
                if aux is None:
                    aux = torch.zeros((x.shape[0], self.aux_dim), device=x.device, dtype=x.dtype)
                if aux.ndim != 2:
                    raise ValueError(f"aux must be (B, aux_dim), got {tuple(aux.shape)}")
                if aux.shape[1] != self.aux_dim:
                    raise ValueError(f"aux has dim={aux.shape[1]} but expected aux_dim={self.aux_dim}")

                aux_prefix = self.aux_embedder(aux.to(dtype=s.dtype)).unsqueeze(1) + self.aux_pos_embed.to(dtype=s.dtype)
                s = torch.cat([aux_prefix, s], dim=1)
                aux_prefix_len = 1
                enc_feat_rope = self._wrap_rope_with_prefix(self.enc_feat_rope, aux_prefix_len)
            elif self.predict_aux_tokens:
                if aux is None:
                    aux = torch.zeros(
                        (x.shape[0], self.num_aux_tokens, self.aux_token_dim),
                        device=x.device,
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
                enc_registers = self.enc_register_tokens.expand(x.shape[0], -1, -1).to(dtype=s.dtype)
                s = torch.cat([s[:, :aux_prefix_len, :], enc_registers, s[:, aux_prefix_len:, :]], dim=1)
            enc_feat_rope = self._wrap_rope_with_prefix(self.enc_feat_rope, aux_prefix_len + self.num_register_tokens)

            for i in range(self.num_encoder_blocks):
                s = self.blocks[i](s, c, feat_rope=enc_feat_rope)

            # Broadcast timestep embedding to tokens and gate
            t_tok = t_emb.unsqueeze(1).repeat(1, s.shape[1], 1)
            s = F.silu(t_tok + s)

            if self.predict_aux:
                s_aux = s[:, :1, :]
                s = s[:, 1:, :]
                aux_out = self.aux_final_layer(s_aux, c)
            elif self.predict_aux_tokens:
                s_aux = s[:, : self.num_aux_tokens, :]
                s = s[:, self.num_aux_tokens :, :]
                aux_out = self.aux_tokens_final_layer(s_aux, c)

        elif self.predict_aux or self.predict_aux_tokens:
            raise ValueError("External encoder state s=... is not supported when auxiliary prediction is enabled.")

        # project encoder->decoder hidden
        s = self.s_projector(s)  # (B, Ls, D_dec)

        # Decode on token grid from x
        x_tok = self.x_embedder(x)  # (B, Lx, D_dec)
        if self.use_pos_embed and self.x_pos_embed is not None:
            x_tok = x_tok + self.x_pos_embed

        dec_feat_rope = self.dec_feat_rope
        if self.num_register_tokens > 0:
            dec_registers = self.dec_register_tokens.expand(x.shape[0], -1, -1).to(dtype=x_tok.dtype)
            x_tok = torch.cat([dec_registers, x_tok], dim=1)
            dec_feat_rope = self._wrap_rope_with_prefix(self.dec_feat_rope, self.num_register_tokens)

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
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
        cfg_scale_aux: float = 0.0,
        meta: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        aux: Optional[torch.Tensor] = None,
        cfg_interval: Union[Tuple[float, float], List[Tuple[float, float]]] = (0.0, 1.0),
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Classic CFG wrapper.
        Convention: caller passes y (and meta, if used) as length 2n:
          y = cat([y_cond, y_null])
          meta = cat([meta_cond, meta_null])  (optional)
        and x has length 2n.

        We duplicate x[:n] into combined=(2n) and run forward(combined, t, y, meta).
        """
        x, aux = self._extract_inputs(x, aux)
        half_x = x[: len(x) // 2]
        combined = torch.cat([half_x, half_x], dim=0)

        combined_aux = None
        if self.predict_aux or self.predict_aux_tokens:
            if aux is None:
                raise ValueError("forward_with_cfg requires aux when auxiliary prediction is enabled")
            half_aux = aux[: len(aux) // 2]
            combined_aux = torch.cat([half_aux, half_aux], dim=0)

        model_out = self.forward(combined, t, y, meta=meta, aux=combined_aux, **kwargs)
        x_out, aux_out = self._split_outputs(model_out)

        eps, rest = x_out[:, : self.in_channels], x_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        if aux_out is not None:
            cond_aux, uncond_aux = torch.split(aux_out, len(aux_out) // 2, dim=0)

        t_half = t[: len(t) // 2]
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
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
        additional_model_forward,
        cfg_scale_aux: float = 0.0,
        meta: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        aux: Optional[torch.Tensor] = None,
        cfg_interval: Union[Tuple[float, float], List[Tuple[float, float]]] = (0.0, 1.0),
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Autoguidance wrapper: blend eps with auxiliary model's eps inside cfg_interval.
        """
        x, aux = self._extract_inputs(x, aux)
        half = x[: len(x) // 2]
        t_half = t[: len(t) // 2]
        y_half = y[: len(y) // 2]
        meta_half = meta[: len(meta) // 2] if meta is not None else None
        aux_half = aux[: len(aux) // 2] if aux is not None else None

        model_out = self.forward(half, t_half, y_half, meta=meta_half, aux=aux_half, **kwargs)
        # be defensive: aux model may not accept meta/kwargs
        try:
            ag_model_out = additional_model_forward(half, t_half, y_half, meta=meta_half, aux=aux_half, **kwargs)
        except TypeError:
            ag_model_out = additional_model_forward(half, t_half, y_half)

        x_out, aux_out = self._split_outputs(model_out)
        ag_x_out, ag_aux_out = self._split_outputs(ag_model_out)

        eps = x_out[:, : self.in_channels]
        ag_eps = ag_x_out[:, : self.in_channels]

        in_mask = self._interval_mask(t_half, cfg_interval, default_all=not self._is_interval_list(cfg_interval))
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


__all__ = ["DiTwDDTHead"]
