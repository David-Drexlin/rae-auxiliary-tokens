import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Mlp
from typing import Optional, Sequence, Union, Tuple, List

from .model_utils import (
    VisionRotaryEmbeddingFast,
    SwiGLUFFN,
    RMSNorm,
    NormAttention,
    LabelEmbedder,
    get_2d_sincos_pos_embed,
    GaussianFourierEmbedding,
    modulate,
)


class LightningDiTBlock(nn.Module):
    """
    Lightning DiT Block.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        **block_kwargs
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
            **block_kwargs
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )

        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, feat_rope=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class LightningFinalLayer(nn.Module):
    """
    Final patch-token output layer.
    """
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LightningAuxFinalLayer(nn.Module):
    """
    Final pooled-aux output layer.

    Input:
      x: (B, 1, D) aux token
      c: (B, D) conditioning vector
    Output:
      aux: (B, aux_dim)
    """
    def __init__(self, hidden_size, aux_dim, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, aux_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x.squeeze(1)


class LightningAuxTokensFinalLayer(nn.Module):
    """
    Final token-aux output layer.

    Input:
      x: (B, K, D) aux tokens
      c: (B, D) conditioning vector
    Output:
      aux_tokens: (B, K, aux_token_dim)
    """
    def __init__(self, hidden_size, aux_token_dim, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, aux_token_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LightningDiT(nn.Module):
    """
    DiT backbone (latent diffusion / transport).

    Supports three modes:

    1) Backward-compatible patch-only mode:
       forward(...) -> x_out

    2) Joint latent + pooled-aux mode:
       - pass predict_aux=True and aux_dim=...
       - forward(..., aux=aux_t) -> (x_out, aux_out)

    3) Joint latent + token-aux mode:
       - pass predict_aux_tokens=True, aux_token_dim=..., num_aux_tokens=...
       - forward(..., aux=aux_tokens_t) -> (x_out, aux_tokens_out)

    Notes:
      - meta conditioning remains optional
      - pooled aux is represented as a single learned aux token prepended to the patch sequence
      - token aux is represented as K learned aux tokens prepended to the patch sequence
      - RoPE is applied only to patch tokens, not to the aux prefix tokens
    """
    def __init__(
        self,
        input_size=16,
        patch_size=1,
        in_channels=768,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_gembed: bool = True,

        # optional metadata conditioning
        meta_num_classes: Optional[Sequence[int]] = None,
        meta_dropout_prob: float = 0.0,

        # NEW: pooled aux prediction
        predict_aux: bool = False,
        aux_dim: Optional[int] = None,
        predict_aux_tokens: bool = False,
        aux_token_dim: Optional[int] = None,
        num_aux_tokens: Optional[int] = None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2

        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_gembed = use_gembed

        self.predict_aux = bool(predict_aux)
        self.aux_dim = int(aux_dim) if aux_dim is not None else int(in_channels)
        if self.predict_aux and self.aux_dim <= 0:
            raise ValueError("aux_dim must be positive when predict_aux=True")

        self.predict_aux_tokens = bool(predict_aux_tokens)
        self.aux_token_dim = int(aux_token_dim) if aux_token_dim is not None else int(in_channels)
        self.num_aux_tokens = int(num_aux_tokens) if num_aux_tokens is not None else 0
        if self.predict_aux and self.predict_aux_tokens:
            raise ValueError("predict_aux and predict_aux_tokens are mutually exclusive.")
        if self.predict_aux_tokens and self.num_aux_tokens <= 0:
            raise ValueError("num_aux_tokens must be positive when predict_aux_tokens=True")
        if self.predict_aux_tokens and self.aux_token_dim <= 0:
            raise ValueError("aux_token_dim must be positive when predict_aux_tokens=True")

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = GaussianFourierEmbedding(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.meta_num_classes = list(meta_num_classes) if meta_num_classes is not None else None
        if self.meta_num_classes is not None and len(self.meta_num_classes) > 0:
            self.meta_embedders = nn.ModuleList([nn.Embedding(int(nc), hidden_size) for nc in self.meta_num_classes])
            self.meta_dropout = nn.Dropout(float(meta_dropout_prob)) if meta_dropout_prob and meta_dropout_prob > 0 else nn.Identity()
        else:
            self.meta_embedders = None
            self.meta_dropout = None

        if self.predict_aux:
            self.aux_embedder = nn.Linear(self.aux_dim, hidden_size, bias=True)
            self.aux_pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.aux_final_layer = LightningAuxFinalLayer(hidden_size, self.aux_dim, use_rmsnorm=use_rmsnorm)
        else:
            self.aux_embedder = None
            self.aux_pos_embed = None
            self.aux_final_layer = None

        if self.predict_aux_tokens:
            self.aux_tokens_embedder = nn.Linear(self.aux_token_dim, hidden_size, bias=True)
            self.aux_tokens_pos_embed = nn.Parameter(torch.zeros(1, self.num_aux_tokens, hidden_size))
            self.aux_tokens_final_layer = LightningAuxTokensFinalLayer(
                hidden_size,
                self.aux_token_dim,
                use_rmsnorm=use_rmsnorm,
            )
        else:
            self.aux_tokens_embedder = None
            self.aux_tokens_pos_embed = None
            self.aux_tokens_final_layer = None

        self.ssl_supervise = False

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
                wo_shift=wo_shift,
            ) for _ in range(depth)
        ])
        self.final_layer = LightningFinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if hasattr(self.y_embedder, "embedding_tables"):
            for emb in self.y_embedder.embedding_tables:
                nn.init.normal_(emb.weight, std=0.02)
        else:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        if self.meta_embedders is not None:
            for emb in self.meta_embedders:
                nn.init.normal_(emb.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.predict_aux:
            nn.init.normal_(self.aux_embedder.weight, std=0.02)
            nn.init.constant_(self.aux_embedder.bias, 0)
            nn.init.normal_(self.aux_pos_embed, std=0.02)

        if self.predict_aux_tokens:
            nn.init.normal_(self.aux_tokens_embedder.weight, std=0.02)
            nn.init.constant_(self.aux_tokens_embedder.bias, 0)
            nn.init.normal_(self.aux_tokens_pos_embed, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

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

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _meta_embed(self, meta: torch.Tensor) -> torch.Tensor:
        """
        meta: (B, M) long
        returns: (B, D)
        """
        if self.meta_embedders is None:
            return torch.zeros(meta.shape[0], self.hidden_size, device=meta.device, dtype=torch.float32)

        if meta.dtype != torch.long:
            meta = meta.long()
        if meta.ndim != 2:
            raise ValueError(f"meta must be (B,M), got {tuple(meta.shape)}")

        M = meta.shape[1]
        if M != len(self.meta_embedders):
            raise ValueError(f"meta has M={M} fields but meta_num_classes has {len(self.meta_embedders)}")

        embs = [emb(meta[:, i]) for i, emb in enumerate(self.meta_embedders)]
        out = torch.stack(embs, dim=0).sum(dim=0)
        out = self.meta_dropout(out) if self.meta_dropout is not None else out
        return out

    @staticmethod
    def _is_interval_list(cfg_interval) -> bool:
        return isinstance(cfg_interval, (list, tuple)) and len(cfg_interval) > 0 and isinstance(cfg_interval[0], (list, tuple))

    @staticmethod
    def _interval_mask(t: torch.Tensor, cfg_interval, default_all: bool = False) -> torch.Tensor:
        if cfg_interval is None:
            return torch.ones_like(t, dtype=torch.bool) if default_all else torch.zeros_like(t, dtype=torch.bool)

        if LightningDiT._is_interval_list(cfg_interval):
            mask = torch.zeros_like(t, dtype=torch.bool)
            for (a, b) in cfg_interval:
                mask |= (t >= float(a)) & (t < float(b))
            return mask

        if isinstance(cfg_interval, (list, tuple)) and len(cfg_interval) == 2:
            a, b = float(cfg_interval[0]), float(cfg_interval[1])
            if a <= -1e3 and b <= -1e3:
                return torch.ones_like(t, dtype=torch.bool) if default_all else torch.zeros_like(t, dtype=torch.bool)
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

    def forward(self, x, t=None, y=None, meta: Optional[torch.Tensor] = None, aux: Optional[torch.Tensor] = None, **kwargs):
        """
        x:    (N, C, H, W) or tuple((N, C, H, W), aux_state)
        t:    (N,)
        y:    (N,)
        meta: optional (N, M) long
        aux:  optional (N, aux_dim) float or (N, K, aux_token_dim) float
        """
        x, aux = self._extract_inputs(x, aux)
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y

        if self.meta_embedders is not None:
            if meta is None:
                B = x.shape[0]
                M = len(self.meta_embedders)
                meta = torch.zeros((B, M), device=x.device, dtype=torch.long)
            c = c + self._meta_embed(meta).to(dtype=c.dtype)

        if self.predict_aux:
            if aux is None:
                aux = torch.zeros((x.shape[0], self.aux_dim), device=x.device, dtype=x.dtype)
            if aux.ndim != 2:
                raise ValueError(f"aux must be (B, aux_dim), got {tuple(aux.shape)}")
            if aux.shape[1] != self.aux_dim:
                raise ValueError(f"aux has dim={aux.shape[1]} but expected aux_dim={self.aux_dim}")

            aux_token = self.aux_embedder(aux.to(dtype=x.dtype)).unsqueeze(1) + self.aux_pos_embed.to(dtype=x.dtype)
            x = torch.cat([aux_token, x], dim=1)
            feat_rope = self._wrap_rope_with_prefix(self.feat_rope, num_prefix_tokens=1)
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

            aux_tokens = self.aux_tokens_embedder(aux.to(dtype=x.dtype)) + self.aux_tokens_pos_embed.to(dtype=x.dtype)
            x = torch.cat([aux_tokens, x], dim=1)
            feat_rope = self._wrap_rope_with_prefix(self.feat_rope, num_prefix_tokens=self.num_aux_tokens)
        else:
            feat_rope = self.feat_rope

        for block in self.blocks:
            x = block(x, c, feat_rope=feat_rope)

        if self.predict_aux:
            aux_token = x[:, :1, :]
            patch_tokens = x[:, 1:, :]

            x_out = self.final_layer(patch_tokens, c)
            x_out = self.unpatchify(x_out)

            aux_out = self.aux_final_layer(aux_token, c)

            if self.learn_sigma:
                x_out, _ = x_out.chunk(2, dim=1)
            return x_out, aux_out

        if self.predict_aux_tokens:
            aux_tokens = x[:, : self.num_aux_tokens, :]
            patch_tokens = x[:, self.num_aux_tokens :, :]

            x_out = self.final_layer(patch_tokens, c)
            x_out = self.unpatchify(x_out)
            aux_out = self.aux_tokens_final_layer(aux_tokens, c)

            if self.learn_sigma:
                x_out, _ = x_out.chunk(2, dim=1)
            return x_out, aux_out

        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(
        self,
        x,
        t,
        y,
        cfg_scale,
        meta: Optional[torch.Tensor] = None,
        aux: Optional[torch.Tensor] = None,
        cfg_interval: Union[Tuple[float, float], List[Tuple[float, float]]] = (-1e4, -1e4),
        interval_cfg: float = 0.0,
        **kwargs
    ):
        """
        CFG wrapper.

        Convention assumed:
          - caller passes y (and meta if used) already as [cond, null] of length 2n
          - x has length 2n
          - if auxiliary prediction is enabled, aux also has length 2n and is duplicated state
        """
        x, aux = self._extract_inputs(x, aux)
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        combined_aux = None
        if self.predict_aux or self.predict_aux_tokens:
            if aux is None:
                raise ValueError("forward_with_cfg requires aux when auxiliary prediction is enabled")
            half_aux = aux[: len(aux) // 2]
            combined_aux = torch.cat([half_aux, half_aux], dim=0)

        model_out = self.forward(combined, t, y, meta=meta, aux=combined_aux, **kwargs)
        if self.ssl_supervise:
            model_out = model_out[0]

        x_out, aux_out = self._split_outputs(model_out)

        eps, rest = x_out[:, : self.in_channels], x_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        if aux_out is not None:
            cond_aux, uncond_aux = torch.split(aux_out, len(aux_out) // 2, dim=0)

        t_half = t[: len(t) // 2]

        if self._is_interval_list(cfg_interval):
            in_mask = self._interval_mask(t_half, cfg_interval, default_all=False)
            if interval_cfg and interval_cfg > 1.0:
                eps_in = uncond_eps + float(interval_cfg) * (cond_eps - uncond_eps)
                if aux_out is not None:
                    aux_in = uncond_aux + float(interval_cfg) * (cond_aux - uncond_aux)
            else:
                eps_in = cond_eps
                if aux_out is not None:
                    aux_in = cond_aux

                eps_out = uncond_eps + float(cfg_scale) * (cond_eps - uncond_eps)
                half_eps = torch.where(self._mask_like(in_mask, cond_eps), eps_in, eps_out)

                if aux_out is not None:
                    aux_out_guided = uncond_aux + float(cfg_scale) * (cond_aux - uncond_aux)
                    half_aux = torch.where(self._mask_like(in_mask, cond_aux), aux_in, aux_out_guided)
        else:
            in_mask = self._interval_mask(t_half, cfg_interval, default_all=True)
            guided = uncond_eps + float(cfg_scale) * (cond_eps - uncond_eps)
            half_eps = torch.where(self._mask_like(in_mask, cond_eps), guided, cond_eps)

            if aux_out is not None:
                aux_guided = uncond_aux + float(cfg_scale) * (cond_aux - uncond_aux)
                half_aux = torch.where(self._mask_like(in_mask, cond_aux), aux_guided, cond_aux)

        eps = torch.cat([half_eps, half_eps], dim=0)
        x_return = torch.cat([eps, rest], dim=1)

        if aux_out is not None:
            aux_return = torch.cat([half_aux, half_aux], dim=0)
            return x_return, aux_return

        return x_return

    def forward_with_autoguidance(
        self,
        x,
        t,
        y,
        cfg_scale,
        additional_model_forward,
        meta: Optional[torch.Tensor] = None,
        aux: Optional[torch.Tensor] = None,
        cfg_interval: Union[Tuple[float, float], List[Tuple[float, float]]] = (-1e4, -1e4),
        interval_cfg: float = 0.0,
        **kwargs
    ):
        """
        Autoguidance wrapper.

        Convention: input x has length 2n; we only run both models on first half (n),
        then replicate back to 2n.
        """
        x, aux = self._extract_inputs(x, aux)
        half = x[: len(x) // 2]
        t_half = t[: len(t) // 2]
        y_half = y[: len(y) // 2]
        meta_half = meta[: len(meta) // 2] if meta is not None else None
        aux_half = aux[: len(aux) // 2] if aux is not None else None

        model_out = self.forward(half, t_half, y_half, meta=meta_half, aux=aux_half, **kwargs)

        try:
            ag_model_out = additional_model_forward(half, t_half, y_half, meta=meta_half, aux=aux_half, **kwargs)
        except TypeError:
            ag_model_out = additional_model_forward(half, t_half, y_half)

        x_out, aux_out = self._split_outputs(model_out)
        ag_x_out, ag_aux_out = self._split_outputs(ag_model_out)

        eps = x_out[:, : self.in_channels]
        ag_eps = ag_x_out[:, : self.in_channels]

        if self._is_interval_list(cfg_interval):
            in_mask = self._interval_mask(t_half, cfg_interval, default_all=False)
            if interval_cfg and interval_cfg > 1.0:
                eps_in = ag_eps + float(interval_cfg) * (eps - ag_eps)
            else:
                eps_in = eps
            eps_out = ag_eps + float(cfg_scale) * (eps - ag_eps)
            out = torch.where(self._mask_like(in_mask, eps), eps_in, eps_out)

            if aux_out is not None:
                if ag_aux_out is None:
                    raise ValueError("additional_model_forward must also return aux output when auxiliary prediction is enabled")
                if interval_cfg and interval_cfg > 1.0:
                    aux_in = ag_aux_out + float(interval_cfg) * (aux_out - ag_aux_out)
                else:
                    aux_in = aux_out
                aux_out_guided = ag_aux_out + float(cfg_scale) * (aux_out - ag_aux_out)
                aux_ret_half = torch.where(self._mask_like(in_mask, aux_out), aux_in, aux_out_guided)
        else:
            in_mask = self._interval_mask(t_half, cfg_interval, default_all=True)
            guided = ag_eps + float(cfg_scale) * (eps - ag_eps)
            out = torch.where(self._mask_like(in_mask, eps), guided, eps)

            if aux_out is not None:
                if ag_aux_out is None:
                    raise ValueError("additional_model_forward must also return aux output when auxiliary prediction is enabled")
                aux_guided = ag_aux_out + float(cfg_scale) * (aux_out - ag_aux_out)
                aux_ret_half = torch.where(self._mask_like(in_mask, aux_out), aux_guided, aux_out)

        x_ret = torch.cat([out, out], dim=0)

        if aux_out is not None:
            aux_ret = torch.cat([aux_ret_half, aux_ret_half], dim=0)
            return x_ret, aux_ret

        return x_ret


__all__ = ["LightningDiT"]
