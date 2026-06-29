from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from .DDT import (
    LightningDDTBlock,
    DDTFinalLayer,
    DDTAuxFinalLayer,
    DDTAuxTokensFinalLayer,
)
from .model_utils import (
    VisionRotaryEmbeddingFast,
    RMSNorm,
    GaussianFourierEmbedding,
    LabelEmbedder,
    get_2d_sincos_pos_embed,
)


class CrossAttentionAsymmetric(nn.Module):
    """Cross-attention where query and context widths may differ.

    The context is projected into the query width so the attention head dimension
    is defined by the query stream. This lets the patch tower and aux tower use
    different hidden sizes while still exchanging information.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        use_rmsnorm: bool = False,
        fused_attn: bool = True,
    ):
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        self.query_dim = int(query_dim)
        self.context_dim = int(context_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.query_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = bool(fused_attn)

        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.q_proj = nn.Linear(self.query_dim, self.query_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(self.context_dim, 2 * self.query_dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(self.query_dim, self.query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        with torch.profiler.record_function("dualv2/cross_attention"):
            b, n, c = x.shape
            _, m, _ = context.shape
            assert c == self.query_dim, f"query dim mismatch: got {c}, expected {self.query_dim}"

            q = self.q_proj(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv_proj(context).reshape(b, m, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)

            if self.fused_attn:
                q = q.to(v.dtype)
                k = k.to(v.dtype)
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            else:
                q = q * self.scale
                attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
                out = attn @ v

            out = out.transpose(1, 2).reshape(b, n, c)
            return self.proj(out)


class CrossStreamFusionBlockV2(nn.Module):
    """Bidirectional fusion between patch and aux towers with asymmetric widths."""

    def __init__(
        self,
        patch_hidden_size: int,
        aux_hidden_size: int,
        patch_num_heads: int,
        aux_num_heads: int,
        use_qknorm: bool = False,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        patch_norm = RMSNorm if use_rmsnorm else lambda d: nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        aux_norm = RMSNorm if use_rmsnorm else lambda d: nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)

        self.patch_norm = patch_norm(patch_hidden_size)
        self.aux_norm = aux_norm(aux_hidden_size)
        self.patch_hidden_size = int(patch_hidden_size)
        self.aux_hidden_size = int(aux_hidden_size)

        self.aux_to_patch = CrossAttentionAsymmetric(
            query_dim=patch_hidden_size,
            context_dim=aux_hidden_size,
            num_heads=patch_num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
        )
        self.patch_to_aux = CrossAttentionAsymmetric(
            query_dim=aux_hidden_size,
            context_dim=patch_hidden_size,
            num_heads=aux_num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
        )
        self.patch_gate = nn.Sequential(nn.SiLU(), nn.Linear(patch_hidden_size, patch_hidden_size, bias=True))
        self.aux_gate = nn.Sequential(nn.SiLU(), nn.Linear(aux_hidden_size, aux_hidden_size, bias=True))

    def initialize_zero_gates(self):
        nn.init.constant_(self.patch_gate[-1].weight, 0)
        nn.init.constant_(self.patch_gate[-1].bias, 0)
        nn.init.constant_(self.aux_gate[-1].weight, 0)
        nn.init.constant_(self.aux_gate[-1].bias, 0)

    def _finalize(
        self,
        patch_tokens: torch.Tensor,
        aux_tokens: torch.Tensor,
        patch_gate: torch.Tensor,
        aux_gate: torch.Tensor,
        patch_delta: torch.Tensor,
        aux_delta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        patch_tokens = patch_tokens + patch_gate * patch_delta
        aux_tokens = aux_tokens + aux_gate * aux_delta
        monitor = {
            "cross_gate_patch_absmean": patch_gate.detach().abs().mean(),
            "cross_gate_aux_absmean": aux_gate.detach().abs().mean(),
            "patch_stream_absmean": patch_tokens.detach().abs().mean(),
            "aux_stream_absmean": aux_tokens.detach().abs().mean(),
        }
        return patch_tokens, aux_tokens, monitor

    def forward(
        self,
        patch_tokens: torch.Tensor,
        aux_tokens: torch.Tensor,
        c_patch: torch.Tensor,
        c_aux: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.profiler.record_function("dualv2/fusion_block"):
            patch_gate = self.patch_gate(c_patch).unsqueeze(1)
            aux_gate = self.aux_gate(c_aux).unsqueeze(1)

            patch_ctx = self.patch_norm(patch_tokens)
            aux_ctx = self.aux_norm(aux_tokens)
            patch_delta = self.aux_to_patch(patch_ctx, aux_ctx)
            aux_delta = self.patch_to_aux(aux_ctx, patch_ctx)
            return self._finalize(patch_tokens, aux_tokens, patch_gate, aux_gate, patch_delta, aux_delta)

    def forward_parallel(
        self,
        patch_tokens: torch.Tensor,
        aux_tokens: torch.Tensor,
        c_patch: torch.Tensor,
        c_aux: torch.Tensor,
        patch_stream: torch.cuda.Stream,
        aux_stream: torch.cuda.Stream,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.profiler.record_function("dualv2/fusion_block_parallel"):
            cur = torch.cuda.current_stream(patch_tokens.device)

            patch_gate = self.patch_gate(c_patch).unsqueeze(1)
            aux_gate = self.aux_gate(c_aux).unsqueeze(1)
            patch_ctx = self.patch_norm(patch_tokens)
            aux_ctx = self.aux_norm(aux_tokens)

            patch_stream.wait_stream(cur)
            aux_stream.wait_stream(cur)

            patch_ctx.record_stream(patch_stream)
            aux_ctx.record_stream(patch_stream)
            patch_ctx.record_stream(aux_stream)
            aux_ctx.record_stream(aux_stream)

            with torch.cuda.stream(patch_stream):
                with torch.profiler.record_function("dualv2/fusion_aux_to_patch"):
                    patch_delta = self.aux_to_patch(patch_ctx, aux_ctx)
            with torch.cuda.stream(aux_stream):
                with torch.profiler.record_function("dualv2/fusion_patch_to_aux"):
                    aux_delta = self.patch_to_aux(aux_ctx, patch_ctx)

            cur.wait_stream(patch_stream)
            cur.wait_stream(aux_stream)
            patch_delta.record_stream(cur)
            aux_delta.record_stream(cur)

            return self._finalize(patch_tokens, aux_tokens, patch_gate, aux_gate, patch_delta, aux_delta)


class DualStreamDDTV2(nn.Module):
    """Dual-stream Stage-2 model with an optional smaller aux tower and stream overlap."""

    def __init__(
        self,
        input_size: int = 1,
        patch_size: Union[List[int], int] = 1,
        in_channels: int = 768,
        hidden_size: Sequence[int] = (384, 2048),
        depth: Sequence[int] = (12, 2),
        num_heads: Union[Sequence[int], int] = (6, 16),
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        use_pos_embed: bool = True,
        meta_num_classes: Optional[Sequence[int]] = None,
        meta_dropout_prob: float = 0.0,
        meta_fields: Optional[Sequence[str]] = None,
        predict_aux: bool = False,
        aux_dim: Optional[int] = None,
        predict_aux_tokens: bool = False,
        aux_token_dim: Optional[int] = None,
        num_aux_tokens: Optional[int] = None,
        use_dual_time_embed: bool = True,
        aux_hidden_size: Optional[int] = None,
        aux_num_heads: Optional[int] = None,
        aux_depth: Optional[int] = None,
        aux_mlp_ratio: Optional[float] = None,
        fusion_interval: int = 1,
        overlap_streams: bool = False,
        overlap_fusion: bool = False,
        monitor_mode: str = "tensors",
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.predict_aux = bool(predict_aux)
        self.predict_aux_tokens = bool(predict_aux_tokens)
        self.aux_dim = int(aux_dim) if aux_dim is not None else int(in_channels)
        self.aux_token_dim = int(aux_token_dim) if aux_token_dim is not None else int(in_channels)
        self.num_aux_tokens = int(num_aux_tokens) if num_aux_tokens is not None else 0
        self.use_dual_time_embed = bool(use_dual_time_embed)
        if isinstance(monitor_mode, bool):
            normalized_monitor_mode = "tensors" if monitor_mode else "off"
        elif monitor_mode is None:
            normalized_monitor_mode = "off"
        else:
            normalized_monitor_mode = str(monitor_mode).strip().lower()
            if normalized_monitor_mode == "false":
                normalized_monitor_mode = "off"
            elif normalized_monitor_mode == "true":
                normalized_monitor_mode = "tensors"
        self.monitor_mode = normalized_monitor_mode
        self.overlap_streams = bool(overlap_streams)
        self.overlap_fusion = bool(overlap_fusion)
        self._patch_stream: Optional[torch.cuda.Stream] = None
        self._aux_stream: Optional[torch.cuda.Stream] = None
        self._fuse_patch_stream: Optional[torch.cuda.Stream] = None
        self._fuse_aux_stream: Optional[torch.cuda.Stream] = None
        self._stream_device_index: Optional[int] = None

        if self.monitor_mode not in {"off", "tensors", "python"}:
            raise ValueError(f"Unsupported monitor_mode={monitor_mode!r}")
        if self.predict_aux and self.predict_aux_tokens:
            raise ValueError("predict_aux and predict_aux_tokens are mutually exclusive.")
        if not self.predict_aux and not self.predict_aux_tokens:
            raise ValueError("DualStreamDDTV2 requires an auxiliary prediction mode.")
        if self.predict_aux and self.aux_dim <= 0:
            raise ValueError("aux_dim must be positive when predict_aux=True")
        if self.predict_aux_tokens and self.num_aux_tokens <= 0:
            raise ValueError("num_aux_tokens must be positive when predict_aux_tokens=True")
        if self.predict_aux_tokens and self.aux_token_dim <= 0:
            raise ValueError("aux_token_dim must be positive when predict_aux_tokens=True")

        hidden_size = list(hidden_size)
        depth = list(depth)
        if len(hidden_size) != 2 or len(depth) != 2:
            raise ValueError("hidden_size and depth must be length-2 sequences: [encoder, decoder]")
        self.patch_hidden_size = int(hidden_size[0])
        self.decoder_hidden_size = int(hidden_size[1])
        self.num_patch_encoder_blocks = int(depth[0])
        self.num_decoder_blocks = int(depth[1])

        if isinstance(num_heads, int):
            patch_heads, dec_heads = int(num_heads), int(num_heads)
        else:
            heads = list(num_heads)
            if len(heads) != 2:
                raise ValueError("num_heads must be int or length-2 sequence: [enc, dec]")
            patch_heads, dec_heads = int(heads[0]), int(heads[1])
        self.patch_num_heads = patch_heads
        self.decoder_num_heads = dec_heads
        self.aux_hidden_size = int(aux_hidden_size) if aux_hidden_size is not None else self.patch_hidden_size
        self.aux_num_heads = int(aux_num_heads) if aux_num_heads is not None else self.patch_num_heads
        self.num_aux_encoder_blocks = int(aux_depth) if aux_depth is not None else self.num_patch_encoder_blocks
        self.aux_mlp_ratio = float(aux_mlp_ratio) if aux_mlp_ratio is not None else float(mlp_ratio)
        self.fusion_interval = max(1, int(fusion_interval))

        if isinstance(patch_size, (int, float)):
            patch_size = [int(patch_size), int(patch_size)]
        patch_size = list(patch_size)
        if len(patch_size) != 2:
            raise ValueError(f"patch_size must be int or [s_patch_size, x_patch_size], got {patch_size}")
        self.s_patch_size = int(patch_size[0])
        self.x_patch_size = int(patch_size[1])

        self.s_channel_per_token = self.in_channels * self.s_patch_size * self.s_patch_size
        self.x_channel_per_token = self.in_channels * self.x_patch_size * self.x_patch_size

        self.s_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=self.s_patch_size,
            in_chans=self.s_channel_per_token,
            embed_dim=self.patch_hidden_size,
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
            nn.Linear(self.patch_hidden_size, self.decoder_hidden_size)
            if self.patch_hidden_size != self.decoder_hidden_size
            else nn.Identity()
        )

        self.t_embedder = GaussianFourierEmbedding(self.patch_hidden_size)
        self.aux_t_embedder = GaussianFourierEmbedding(self.aux_hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, self.patch_hidden_size, class_dropout_prob)
        self.shared_to_aux = (
            nn.Linear(self.patch_hidden_size, self.aux_hidden_size, bias=True)
            if self.patch_hidden_size != self.aux_hidden_size
            else nn.Identity()
        )

        self.meta_fields = list(meta_fields) if meta_fields is not None else None
        self.meta_num_classes = list(meta_num_classes) if meta_num_classes is not None else None
        if self.meta_num_classes is not None and len(self.meta_num_classes) > 0:
            self.meta_embedders = nn.ModuleList(
                [nn.Embedding(int(nc), self.patch_hidden_size) for nc in self.meta_num_classes]
            )
            self.meta_dropout = (
                nn.Dropout(float(meta_dropout_prob))
                if meta_dropout_prob and meta_dropout_prob > 0
                else nn.Identity()
            )
        else:
            self.meta_embedders = None
            self.meta_dropout = None

        if self.predict_aux:
            self.aux_embedder = nn.Linear(self.aux_dim, self.aux_hidden_size, bias=True)
            self.aux_pos_embed = nn.Parameter(torch.zeros(1, 1, self.aux_hidden_size))
            self.aux_final_layer = DDTAuxFinalLayer(
                hidden_size=self.aux_hidden_size,
                aux_dim=self.aux_dim,
                use_rmsnorm=use_rmsnorm,
            )
            self.aux_tokens_embedder = None
            self.aux_tokens_pos_embed = None
            self.aux_tokens_final_layer = None
        else:
            self.aux_embedder = None
            self.aux_pos_embed = None
            self.aux_final_layer = None
            self.aux_tokens_embedder = nn.Linear(self.aux_token_dim, self.aux_hidden_size, bias=True)
            self.aux_tokens_pos_embed = nn.Parameter(torch.zeros(1, self.num_aux_tokens, self.aux_hidden_size))
            self.aux_tokens_final_layer = DDTAuxTokensFinalLayer(
                hidden_size=self.aux_hidden_size,
                aux_token_dim=self.aux_token_dim,
                use_rmsnorm=use_rmsnorm,
            )

        self.use_pos_embed = bool(use_pos_embed)
        if self.use_pos_embed:
            num_patches = self.s_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.patch_hidden_size), requires_grad=False)
            self.x_pos_embed = None
        else:
            self.pos_embed = None
            self.x_pos_embed = None

        self.use_rope = bool(use_rope)
        if self.use_rope:
            patch_half_head_dim = self.patch_hidden_size // self.patch_num_heads // 2
            patch_hw = int(math.sqrt(self.s_embedder.num_patches))
            self.enc_feat_rope = VisionRotaryEmbeddingFast(dim=patch_half_head_dim, pt_seq_len=patch_hw)

            dec_half_head_dim = self.decoder_hidden_size // self.decoder_num_heads // 2
            dec_hw = int(math.sqrt(self.x_embedder.num_patches))
            self.dec_feat_rope = VisionRotaryEmbeddingFast(dim=dec_half_head_dim, pt_seq_len=dec_hw)
        else:
            self.enc_feat_rope = None
            self.dec_feat_rope = None

        self.patch_encoder_blocks = nn.ModuleList(
            [
                LightningDDTBlock(
                    hidden_size=self.patch_hidden_size,
                    num_heads=self.patch_num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_rmsnorm=use_rmsnorm,
                    use_swiglu=use_swiglu,
                    wo_shift=wo_shift,
                )
                for _ in range(self.num_patch_encoder_blocks)
            ]
        )
        self.aux_encoder_blocks = nn.ModuleList(
            [
                LightningDDTBlock(
                    hidden_size=self.aux_hidden_size,
                    num_heads=self.aux_num_heads,
                    mlp_ratio=self.aux_mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_rmsnorm=use_rmsnorm,
                    use_swiglu=use_swiglu,
                    wo_shift=wo_shift,
                )
                for _ in range(self.num_aux_encoder_blocks)
            ]
        )
        self.fusion_blocks = nn.ModuleList(
            [
                CrossStreamFusionBlockV2(
                    patch_hidden_size=self.patch_hidden_size,
                    aux_hidden_size=self.aux_hidden_size,
                    patch_num_heads=self.patch_num_heads,
                    aux_num_heads=self.aux_num_heads,
                    use_qknorm=use_qknorm,
                    use_rmsnorm=use_rmsnorm,
                )
                for _ in range(self.num_patch_encoder_blocks)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                LightningDDTBlock(
                    hidden_size=self.decoder_hidden_size,
                    num_heads=self.decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_rmsnorm=use_rmsnorm,
                    use_swiglu=use_swiglu,
                    wo_shift=wo_shift,
                )
                for _ in range(self.num_decoder_blocks)
            ]
        )
        self.final_layer = DDTFinalLayer(
            hidden_size=self.decoder_hidden_size,
            patch_size=1,
            out_channels=self.x_channel_per_token,
            use_rmsnorm=use_rmsnorm,
        )

        self._last_monitor_stats: Dict[str, Union[float, torch.Tensor]] = {}
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.pos_embed is not None:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.s_embedder.num_patches ** 0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        for pe in [self.s_embedder, self.x_embedder]:
            w = pe.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(pe.proj.bias, 0)

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
        nn.init.normal_(self.aux_t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.aux_t_embedder.mlp[2].weight, std=0.02)

        if self.predict_aux:
            nn.init.normal_(self.aux_embedder.weight, std=0.02)
            nn.init.constant_(self.aux_embedder.bias, 0)
            nn.init.normal_(self.aux_pos_embed, std=0.02)
        if self.predict_aux_tokens:
            nn.init.normal_(self.aux_tokens_embedder.weight, std=0.02)
            nn.init.constant_(self.aux_tokens_embedder.bias, 0)
            nn.init.normal_(self.aux_tokens_pos_embed, std=0.02)

        for block in self.patch_encoder_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.aux_encoder_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.decoder_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.fusion_blocks:
            block.initialize_zero_gates()

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
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def _meta_embed(self, meta: torch.Tensor) -> torch.Tensor:
        if self.meta_embedders is None:
            return torch.zeros(meta.shape[0], self.patch_hidden_size, device=meta.device, dtype=torch.float32)
        if meta.dtype != torch.long:
            meta = meta.long()
        if meta.ndim != 2:
            raise ValueError(f"meta must be (B,M), got {tuple(meta.shape)}")
        if meta.shape[1] != len(self.meta_embedders):
            raise ValueError(f"meta has M={meta.shape[1]} fields but meta_num_classes has {len(self.meta_embedders)}")
        embs = [emb(meta[:, i]) for i, emb in enumerate(self.meta_embedders)]
        out = torch.stack(embs, dim=0).sum(dim=0)
        return self.meta_dropout(out) if self.meta_dropout is not None else out

    @staticmethod
    def _is_interval_list(cfg_interval) -> bool:
        return isinstance(cfg_interval, (list, tuple)) and len(cfg_interval) > 0 and isinstance(cfg_interval[0], (list, tuple))

    @staticmethod
    def _mask_like(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return mask.view(-1, *([1] * (ref.ndim - 1)))

    @staticmethod
    def _extract_inputs(x, aux: Optional[torch.Tensor]):
        if isinstance(x, (tuple, list)):
            if len(x) != 2:
                raise ValueError(f"Expected tuple/list input of length 2, got {len(x)}")
            x, state_aux = x
            if aux is None:
                aux = state_aux
        return x, aux

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

    def _split_outputs(self, model_out):
        if self.predict_aux or self.predict_aux_tokens:
            if not isinstance(model_out, (tuple, list)) or len(model_out) != 2:
                raise ValueError("Expected model output to be (x_out, aux_out) when auxiliary prediction is enabled")
            return model_out[0], model_out[1]
        return model_out, None

    def _interval_mask(self, t: torch.Tensor, cfg_interval, default_all: bool = False) -> torch.Tensor:
        if cfg_interval is None:
            return torch.ones_like(t, dtype=torch.bool) if default_all else torch.zeros_like(t, dtype=torch.bool)
        if self._is_interval_list(cfg_interval):
            mask = torch.zeros_like(t, dtype=torch.bool)
            for a, b in cfg_interval:
                mask |= (t >= float(a)) & (t < float(b))
            return mask
        if isinstance(cfg_interval, (list, tuple)) and len(cfg_interval) == 2:
            a, b = float(cfg_interval[0]), float(cfg_interval[1])
            if a <= -1e3 and b <= -1e3:
                return torch.ones_like(t, dtype=torch.bool) if default_all else torch.zeros_like(t, dtype=torch.bool)
            return (t >= a) & (t <= b)
        return torch.ones_like(t, dtype=torch.bool) if default_all else torch.zeros_like(t, dtype=torch.bool)

    def _encode_aux(self, aux: Optional[torch.Tensor], dtype: torch.dtype, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.predict_aux:
            if aux is None:
                aux = torch.zeros((batch_size, self.aux_dim), device=device, dtype=dtype)
            if aux.ndim != 2:
                raise ValueError(f"aux must be (B, aux_dim), got {tuple(aux.shape)}")
            if aux.shape[1] != self.aux_dim:
                raise ValueError(f"aux has dim={aux.shape[1]} but expected aux_dim={self.aux_dim}")
            return self.aux_embedder(aux.to(dtype=dtype)).unsqueeze(1) + self.aux_pos_embed.to(dtype=dtype)

        if aux is None:
            aux = torch.zeros((batch_size, self.num_aux_tokens, self.aux_token_dim), device=device, dtype=dtype)
        if aux.ndim != 3:
            raise ValueError(f"aux tokens must be (B, K, aux_token_dim), got {tuple(aux.shape)}")
        if aux.shape[1] != self.num_aux_tokens:
            raise ValueError(f"aux has K={aux.shape[1]} but expected num_aux_tokens={self.num_aux_tokens}")
        if aux.shape[2] != self.aux_token_dim:
            raise ValueError(f"aux has dim={aux.shape[2]} but expected aux_token_dim={self.aux_token_dim}")
        return self.aux_tokens_embedder(aux.to(dtype=dtype)) + self.aux_tokens_pos_embed.to(dtype=dtype)

    def _ensure_streams(self, device: torch.device):
        if not self.overlap_streams or device.type != "cuda":
            return
        if self._stream_device_index == device.index and self._patch_stream is not None:
            return
        self._patch_stream = torch.cuda.Stream(device=device)
        self._aux_stream = torch.cuda.Stream(device=device)
        if self.overlap_fusion:
            self._fuse_patch_stream = torch.cuda.Stream(device=device)
            self._fuse_aux_stream = torch.cuda.Stream(device=device)
        else:
            self._fuse_patch_stream = None
            self._fuse_aux_stream = None
        self._stream_device_index = device.index

    def _run_encoder_pair(
        self,
        patch_block: LightningDDTBlock,
        aux_block: Optional[LightningDDTBlock],
        patch_tokens: torch.Tensor,
        aux_tokens: torch.Tensor,
        c_patch: torch.Tensor,
        c_aux: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if aux_block is None or (not self.overlap_streams) or (not patch_tokens.is_cuda):
            with torch.profiler.record_function("dualv2/patch_block"):
                patch_tokens = patch_block(patch_tokens, c_patch, feat_rope=self.enc_feat_rope)
            if aux_block is not None:
                with torch.profiler.record_function("dualv2/aux_block"):
                    aux_tokens = aux_block(aux_tokens, c_aux, feat_rope=None)
            return patch_tokens, aux_tokens

        cur = torch.cuda.current_stream(patch_tokens.device)
        ps = self._patch_stream
        aus = self._aux_stream
        assert ps is not None and aus is not None

        ps.wait_stream(cur)
        aus.wait_stream(cur)
        patch_tokens.record_stream(ps)
        aux_tokens.record_stream(aus)
        c_patch.record_stream(ps)
        c_aux.record_stream(aus)

        with torch.cuda.stream(ps):
            with torch.profiler.record_function("dualv2/patch_block_parallel"):
                patch_next = patch_block(patch_tokens, c_patch, feat_rope=self.enc_feat_rope)
        with torch.cuda.stream(aus):
            with torch.profiler.record_function("dualv2/aux_block_parallel"):
                aux_next = aux_block(aux_tokens, c_aux, feat_rope=None)

        cur.wait_stream(ps)
        cur.wait_stream(aus)
        patch_next.record_stream(cur)
        aux_next.record_stream(cur)
        return patch_next, aux_next

    def _run_fusion(
        self,
        fusion_block: CrossStreamFusionBlockV2,
        patch_tokens: torch.Tensor,
        aux_tokens: torch.Tensor,
        c_patch: torch.Tensor,
        c_aux: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if (not self.overlap_fusion) or (not patch_tokens.is_cuda):
            return fusion_block(patch_tokens, aux_tokens, c_patch, c_aux)
        assert self._fuse_patch_stream is not None and self._fuse_aux_stream is not None
        return fusion_block.forward_parallel(
            patch_tokens,
            aux_tokens,
            c_patch,
            c_aux,
            self._fuse_patch_stream,
            self._fuse_aux_stream,
        )

    def _should_fuse(self, idx: int) -> bool:
        return ((idx + 1) % self.fusion_interval == 0) or (idx == self.num_patch_encoder_blocks - 1)

    def _store_monitor_stats(self, monitor_accum: Dict[str, List[torch.Tensor]], x_img: torch.Tensor, aux_out: torch.Tensor):
        if self.monitor_mode == "off":
            self._last_monitor_stats = {}
            return

        stats: Dict[str, Union[torch.Tensor, float]] = {}
        for key, values in monitor_accum.items():
            if not values:
                continue
            mean_value = torch.stack([value.float() for value in values]).mean()
            stats[key] = float(mean_value.item()) if self.monitor_mode == "python" else mean_value

        aux_abs = aux_out.detach().abs().mean()
        patch_abs = x_img.detach().abs().mean()
        stats["aux_output_absmean"] = float(aux_abs.item()) if self.monitor_mode == "python" else aux_abs
        stats["patch_output_absmean"] = float(patch_abs.item()) if self.monitor_mode == "python" else patch_abs
        self._last_monitor_stats = stats

    def get_monitor_stats(self, as_tensors: bool = False) -> Dict[str, Union[float, torch.Tensor]]:
        if self.monitor_mode == "off":
            return {}
        out: Dict[str, Union[float, torch.Tensor]] = {}
        for key, value in self._last_monitor_stats.items():
            if as_tensors:
                if torch.is_tensor(value):
                    out[key] = value.detach()
                else:
                    device = next(self.parameters()).device
                    out[key] = torch.tensor(float(value), device=device)
            else:
                if torch.is_tensor(value):
                    out[key] = float(value.detach().float().mean().item())
                else:
                    out[key] = float(value)
        return out

    def forward(self, x, t=None, y=None, meta: Optional[torch.Tensor] = None, aux: Optional[torch.Tensor] = None, **kwargs):
        with torch.profiler.record_function("dualv2/forward"):
            x, aux = self._extract_inputs(x, aux)
            batch_size = x.shape[0]
            device = x.device
            self._ensure_streams(device)

            with torch.profiler.record_function("dualv2/time_and_cond"):
                patch_t, aux_t = self._extract_times(t, batch_size=batch_size, device=device)
                patch_t = patch_t.to(device=device, dtype=x.dtype)
                aux_t = aux_t.to(device=device, dtype=x.dtype)

                t_patch_emb = self.t_embedder(patch_t)
                t_aux_emb = self.aux_t_embedder(aux_t if self.use_dual_time_embed else patch_t)
                y_patch = self.y_embedder(y, self.training)
                shared_patch = y_patch
                if self.meta_embedders is not None:
                    if meta is None:
                        meta = torch.zeros((batch_size, len(self.meta_embedders)), device=device, dtype=torch.long)
                    shared_patch = shared_patch + self._meta_embed(meta).to(dtype=shared_patch.dtype)

                shared_aux = self.shared_to_aux(shared_patch)
                c_patch = F.silu(t_patch_emb + shared_patch)
                c_aux = F.silu(t_aux_emb + shared_aux)

            with torch.profiler.record_function("dualv2/input_embed"):
                patch_tokens = self.s_embedder(x)
                if self.use_pos_embed and self.pos_embed is not None:
                    patch_tokens = patch_tokens + self.pos_embed
                aux_tokens = self._encode_aux(aux, dtype=patch_tokens.dtype, batch_size=batch_size, device=device)

            monitor_accum: Dict[str, List[torch.Tensor]] = {
                "cross_gate_patch_absmean": [],
                "cross_gate_aux_absmean": [],
                "patch_stream_absmean": [],
                "aux_stream_absmean": [],
            }

            with torch.profiler.record_function("dualv2/encoder"):
                for idx, patch_block in enumerate(self.patch_encoder_blocks):
                    aux_block = self.aux_encoder_blocks[idx] if idx < self.num_aux_encoder_blocks else None
                    with torch.profiler.record_function(f"dualv2/encoder_layer_{idx:02d}"):
                        patch_tokens, aux_tokens = self._run_encoder_pair(
                            patch_block, aux_block, patch_tokens, aux_tokens, c_patch, c_aux
                        )
                        if self._should_fuse(idx):
                            patch_tokens, aux_tokens, block_stats = self._run_fusion(
                                self.fusion_blocks[idx], patch_tokens, aux_tokens, c_patch, c_aux
                            )
                            if self.monitor_mode != "off":
                                for key, value in block_stats.items():
                                    monitor_accum[key].append(value.detach())

            with torch.profiler.record_function("dualv2/decode"):
                projected_patch = self.s_projector(patch_tokens)
                x_tok = self.x_embedder(x)
                if self.use_pos_embed and self.x_pos_embed is not None:
                    x_tok = x_tok + self.x_pos_embed

                for block in self.decoder_blocks:
                    x_tok = block(x_tok, projected_patch, feat_rope=self.dec_feat_rope)

                x_tok = self.final_layer(x_tok, projected_patch)
                x_img = self.unpatchify(x_tok)

            with torch.profiler.record_function("dualv2/aux_head"):
                if self.predict_aux:
                    aux_out = self.aux_final_layer(aux_tokens, c_aux)
                else:
                    aux_out = self.aux_tokens_final_layer(aux_tokens, c_aux)

            self._store_monitor_stats(monitor_accum, x_img, aux_out)
            return x_img, aux_out

    def forward_with_cfg(
        self,
        x,
        t,
        y,
        cfg_scale,
        cfg_scale_aux: float = 0.0,
        meta: Optional[torch.Tensor] = None,
        aux: Optional[torch.Tensor] = None,
        cfg_interval: Union[Tuple[float, float], List[Tuple[float, float]]] = (0.0, 1.0),
        **kwargs,
    ):
        x, aux = self._extract_inputs(x, aux)
        half_x = x[: len(x) // 2]
        combined = torch.cat([half_x, half_x], dim=0)

        combined_aux = None
        if aux is not None:
            half_aux = aux[: len(aux) // 2]
            combined_aux = torch.cat([half_aux, half_aux], dim=0)

        model_out = self.forward(combined, t, y, meta=meta, aux=combined_aux, **kwargs)
        x_out, aux_out = self._split_outputs(model_out)

        eps, rest = x_out[:, : self.in_channels], x_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        cond_aux, uncond_aux = torch.split(aux_out, len(aux_out) // 2, dim=0)

        t_half = self._primary_time(t)[: len(cond_eps)]
        in_mask = self._interval_mask(t_half, cfg_interval, default_all=not self._is_interval_list(cfg_interval))

        half_eps = torch.where(
            self._mask_like(in_mask, cond_eps),
            uncond_eps + float(cfg_scale) * (cond_eps - uncond_eps),
            cond_eps,
        )
        half_aux = torch.where(
            self._mask_like(in_mask, cond_aux),
            uncond_aux + float(cfg_scale_aux) * (cond_aux - uncond_aux),
            cond_aux,
        )

        eps = torch.cat([half_eps, half_eps], dim=0)
        x_return = torch.cat([eps, rest], dim=1)
        aux_return = torch.cat([half_aux, half_aux], dim=0)
        return x_return, aux_return

    def forward_with_autoguidance(
        self,
        x,
        t,
        y,
        cfg_scale,
        additional_model_forward,
        cfg_scale_aux: float = 0.0,
        meta: Optional[torch.Tensor] = None,
        aux: Optional[torch.Tensor] = None,
        cfg_interval: Union[Tuple[float, float], List[Tuple[float, float]]] = (0.0, 1.0),
        **kwargs,
    ):
        x, aux = self._extract_inputs(x, aux)
        half = x[: len(x) // 2]
        aux_half = aux[: len(aux) // 2] if aux is not None else None

        if torch.is_tensor(t):
            t_half = t[: len(half)]
        elif isinstance(t, (tuple, list)):
            t_half = tuple(ti[: len(half)] for ti in t)
        elif isinstance(t, dict):
            t_half = {k: ti[: len(half)] for k, ti in t.items()}
        else:
            raise TypeError(f"Unsupported time input type: {type(t)}")

        y_half = y[: len(half)]
        meta_half = meta[: len(half)] if meta is not None else None

        model_out = self.forward(half, t_half, y_half, meta=meta_half, aux=aux_half, **kwargs)
        try:
            ag_model_out = additional_model_forward(half, t_half, y_half, meta=meta_half, aux=aux_half, **kwargs)
        except TypeError:
            ag_model_out = additional_model_forward(half, t_half, y_half)

        x_out, aux_out = self._split_outputs(model_out)
        ag_x_out, ag_aux_out = self._split_outputs(ag_model_out)

        eps = x_out[:, : self.in_channels]
        ag_eps = ag_x_out[:, : self.in_channels]
        primary_t_half = self._primary_time(t_half)
        in_mask = self._interval_mask(primary_t_half, cfg_interval, default_all=not self._is_interval_list(cfg_interval))

        out = torch.where(
            self._mask_like(in_mask, eps),
            ag_eps + float(cfg_scale) * (eps - ag_eps),
            eps,
        )
        aux_ret_half = torch.where(
            self._mask_like(in_mask, aux_out),
            ag_aux_out + float(cfg_scale_aux) * (aux_out - ag_aux_out),
            aux_out,
        )
        x_ret = torch.cat([out, out], dim=0)
        aux_ret = torch.cat([aux_ret_half, aux_ret_half], dim=0)
        return x_ret, aux_ret


__all__ = ["DualStreamDDTV2"]
