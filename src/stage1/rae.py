import torch
import torch.nn as nn
from .decoders import GeneralDecoder
from .encoders import ARCHS
from transformers import AutoConfig, AutoImageProcessor
from typing import Optional, Protocol
from math import sqrt


class PatchMAPSummaryBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, summary_tokens: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(summary_tokens),
            self.context_norm(patch_tokens),
            self.context_norm(patch_tokens),
            need_weights=False,
        )
        summary_tokens = summary_tokens + attn_out
        summary_tokens = summary_tokens + self.mlp(self.mlp_norm(summary_tokens))
        return summary_tokens


class PatchMAPSummary(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_tokens: int,
        num_heads: int,
        num_layers: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.summary_queries = nn.Parameter(torch.zeros(1, num_tokens, hidden_size))
        nn.init.trunc_normal_(self.summary_queries, std=0.02)
        self.layers = nn.ModuleList(
            [
                PatchMAPSummaryBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        summary_tokens = self.summary_queries.expand(patch_tokens.size(0), -1, -1)
        for layer in self.layers:
            summary_tokens = layer(summary_tokens, patch_tokens)
        return self.final_norm(summary_tokens)


class Stage1Protocal(Protocol):
    patch_size: int
    hidden_size: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def forward_with_global(self, x: torch.Tensor):
        ...


class RAE(nn.Module):
    def __init__(
        self,
        # ---- encoder configs ----
        encoder_cls: str = "Dinov2withNorm",
        encoder_config_path: str = "facebook/dinov2-base",
        encoder_input_size: int = 224,
        encoder_params: dict = {},
        # ---- decoder configs ----
        decoder_config_path: str = "vit_mae-base",
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # ---- legacy pooled-conditioning args (kept for backward compatibility) ----
        use_global_cond: bool = False,
        global_token_source: str = "cls+register",   # legacy alias of aux_token_source
        global_cond_pool: str = "mean",              # legacy alias of aux_pool
        global_cond_dropout_prob: float = 0.0,       # legacy alias of aux_dropout_prob
        adaln_zero_init: bool = True,
        # ---- new explicit aux-token API ----
        decoder_aux_mode: Optional[str] = None,      # "discard" | "adaln_pool" | "prepend" | "cross_attn"
        aux_token_source: Optional[str] = None,      # "cls" | "register" | "cls+register"
        aux_pool: Optional[str] = None,              # "mean" | "cls"
        aux_dropout_prob: Optional[float] = None,
        aux_dropout_mode: str = "zero",             # "zero" | "learned_null"
        aux_noise_sigma: float = 0.0,
        aux_eval_override: str = "none",            # "none" | "zero" | "learned_null" | "shuffle" | "gaussian" | "scale" | "interp_zero" | "interp_other"
        aux_eval_sigma: float = 0.0,
        aux_eval_scale: float = 1.0,
        aux_eval_alpha: float = 0.0,
        aux_eval_shuffle_offset: int = 1,
        num_aux_tokens: Optional[int] = None,        # required or inferred for prepend / cross_attn
        patch_map_num_heads: Optional[int] = None,
        patch_map_num_layers: int = 1,
        patch_map_mlp_ratio: float = 4.0,
        patch_map_dropout: float = 0.0,
        aux_ablation_mode: str = "none",            # "none" | "gaussian_replace" | "shuffle" | "true_plus_gaussian_extra" | "true_plus_learned_extra"
        aux_ablation_sigma: float = 1.0,
        aux_ablation_extra_tokens: int = 0,
        aux_shuffle_offset: int = 1,
        # ---- noising, reshaping and normalization ----
        noise_tau: float = 0.8,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        aux_normalization_stat_path: Optional[str] = None,
        latent_source: str = "patch",
        global_latent_adapter: str = "interp1d",
        eps: float = 1e-5,
    ):
        super().__init__()
        encoder_cls = ARCHS[encoder_cls]
        self.encoder: Stage1Protocal = encoder_cls(**encoder_params)
        print(f"encoder_config_path: {encoder_config_path}")

        self.eps = eps

        if hasattr(self.encoder, "image_mean") and hasattr(self.encoder, "image_std"):
            self.encoder_mean = self.encoder.image_mean
            self.encoder_std = self.encoder.image_std
        else:
            proc = AutoImageProcessor.from_pretrained(encoder_config_path)
            self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
            self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)

        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0, (
            f"encoder_input_size {self.encoder_input_size} must be divisible by "
            f"encoder_patch_size {self.encoder_patch_size}"
        )
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2

        # ------------------------------------------------------------------
        # Normalize legacy args -> explicit aux-token API
        # ------------------------------------------------------------------
        if decoder_aux_mode is None:
            decoder_aux_mode = "adaln_pool" if use_global_cond else "discard"
        if aux_token_source is None:
            aux_token_source = global_token_source
        if aux_pool is None:
            aux_pool = global_cond_pool
        if aux_dropout_prob is None:
            aux_dropout_prob = global_cond_dropout_prob

        self.decoder_aux_mode = str(decoder_aux_mode)
        self.aux_token_source = str(aux_token_source)
        self.aux_pool = str(aux_pool)
        self.aux_dropout_prob = float(aux_dropout_prob)
        self.aux_dropout_mode = str(aux_dropout_mode)
        self.aux_noise_sigma = float(aux_noise_sigma)
        self.aux_eval_override = str(aux_eval_override)
        self.aux_eval_sigma = float(aux_eval_sigma)
        self.aux_eval_scale = float(aux_eval_scale)
        self.aux_eval_alpha = float(aux_eval_alpha)
        self.aux_eval_shuffle_offset = int(aux_eval_shuffle_offset)
        self.adaln_zero_init = bool(adaln_zero_init)
        self.aux_ablation_mode = str(aux_ablation_mode)
        self.aux_ablation_sigma = float(aux_ablation_sigma)
        self.aux_ablation_extra_tokens = int(aux_ablation_extra_tokens)
        self.aux_shuffle_offset = int(aux_shuffle_offset)
        self.latent_source = str(latent_source)
        self.global_latent_adapter = str(global_latent_adapter)

        if self.decoder_aux_mode not in {"discard", "adaln_pool", "prepend", "cross_attn"}:
            raise ValueError(
                "decoder_aux_mode must be one of "
                "['discard', 'adaln_pool', 'prepend', 'cross_attn'], "
                f"got {self.decoder_aux_mode}"
            )
        if self.aux_token_source not in {"cls", "register", "cls+register", "learned", "patch_map"}:
            raise ValueError(
                f"aux_token_source must be one of ['cls', 'register', 'cls+register', 'learned', 'patch_map'], "
                f"got {self.aux_token_source}"
            )
        if self.aux_pool not in {"mean", "cls"}:
            raise ValueError(
                f"aux_pool must be one of ['mean', 'cls'], got {self.aux_pool}"
            )
        if self.aux_dropout_mode not in {"zero", "learned_null"}:
            raise ValueError(
                f"aux_dropout_mode must be one of ['zero', 'learned_null'], "
                f"got {self.aux_dropout_mode}"
            )
        if self.aux_eval_override not in {
            "none",
            "zero",
            "learned_null",
            "shuffle",
            "gaussian",
            "scale",
            "interp_zero",
            "interp_other",
        }:
            raise ValueError(
                "aux_eval_override must be one of "
                "['none', 'zero', 'learned_null', 'shuffle', 'gaussian', "
                "'scale', 'interp_zero', 'interp_other'], "
                f"got {self.aux_eval_override}"
            )
        if self.aux_eval_sigma < 0:
            raise ValueError(f"aux_eval_sigma must be >= 0, got {self.aux_eval_sigma}")
        if not (0.0 <= self.aux_eval_alpha <= 1.0):
            raise ValueError(
                f"aux_eval_alpha must be in [0, 1], got {self.aux_eval_alpha}"
            )
        if self.aux_noise_sigma < 0:
            raise ValueError(
                f"aux_noise_sigma must be >= 0, got {self.aux_noise_sigma}"
            )
        if self.aux_ablation_mode not in {
            "none",
            "gaussian_replace",
            "shuffle",
            "true_plus_gaussian_extra",
            "true_plus_learned_extra",
        }:
            raise ValueError(
                "aux_ablation_mode must be one of "
                "['none', 'gaussian_replace', 'shuffle', 'true_plus_gaussian_extra', 'true_plus_learned_extra'], "
                f"got {self.aux_ablation_mode}"
            )
        if self.aux_ablation_sigma < 0:
            raise ValueError(f"aux_ablation_sigma must be >= 0, got {self.aux_ablation_sigma}")
        if self.latent_source not in {"patch", "global"}:
            raise ValueError(
                f"latent_source must be one of ['patch', 'global'], got {self.latent_source}"
            )
        if self.global_latent_adapter not in {"interp1d"}:
            raise ValueError(
                "global_latent_adapter must be one of ['interp1d'], "
                f"got {self.global_latent_adapter}"
            )
        if self.latent_source == "global":
            if not hasattr(self.encoder, "forward_with_global"):
                raise ValueError(
                    f"latent_source='global' requires encoder {self.encoder.__class__.__name__} "
                    "to implement forward_with_global()."
                )
            if self.decoder_aux_mode != "discard":
                raise ValueError(
                    "latent_source='global' currently supports only decoder_aux_mode='discard' "
                    "to keep the ablation free of extra conditioning pathways."
                )
        if self.aux_token_source == "learned" and self.decoder_aux_mode not in {"prepend", "cross_attn"}:
            raise ValueError(
                "aux_token_source='learned' currently supports only token-conditioning "
                "decoders ('prepend' or 'cross_attn')."
            )
        if (
            self.aux_token_source == "patch_map"
            and self.decoder_aux_mode not in {"prepend", "cross_attn"}
            and self.latent_source != "global"
        ):
            raise ValueError(
                "aux_token_source='patch_map' currently supports only token-conditioning "
                "decoders ('prepend' or 'cross_attn') or latent_source='global'."
            )
        if self.aux_ablation_mode != "none" and self.decoder_aux_mode == "discard":
            raise ValueError(
                f"aux_ablation_mode='{self.aux_ablation_mode}' requires a decoder aux pathway, "
                "but decoder_aux_mode='discard'."
            )
        if self.aux_ablation_mode in {"true_plus_gaussian_extra", "true_plus_learned_extra"}:
            if self.decoder_aux_mode not in {"prepend", "cross_attn"}:
                raise ValueError(
                    f"aux_ablation_mode='{self.aux_ablation_mode}' is only meaningful for "
                    "token-conditioning decoders ('prepend' or 'cross_attn')."
                )
            if self.aux_ablation_extra_tokens <= 0:
                raise ValueError(
                    f"aux_ablation_mode='{self.aux_ablation_mode}' requires "
                    "aux_ablation_extra_tokens > 0."
                )
        elif self.aux_ablation_extra_tokens != 0:
            raise ValueError(
                "aux_ablation_extra_tokens is only used when "
                "aux_ablation_mode is one of "
                "['true_plus_gaussian_extra', 'true_plus_learned_extra']."
            )

        # legacy compatibility flag: True only for pooled AdaLN decoder conditioning
        self.use_global_cond = (self.decoder_aux_mode == "adaln_pool")

        # Infer number of selected aux tokens if not explicitly given
        self.base_num_aux_tokens = (
            num_aux_tokens if num_aux_tokens is not None else self._infer_num_aux_tokens()
        )
        if self.decoder_aux_mode in {"prepend", "cross_attn"} and self.base_num_aux_tokens is None:
            raise ValueError(
                f"decoder_aux_mode='{self.decoder_aux_mode}' requires num_aux_tokens. "
                "Could not infer it from the encoder, so please pass num_aux_tokens explicitly."
            )
        self.num_aux_tokens = self.base_num_aux_tokens
        if (
            self.decoder_aux_mode in {"prepend", "cross_attn"}
            and self.aux_ablation_mode in {"true_plus_gaussian_extra", "true_plus_learned_extra"}
        ):
            self.num_aux_tokens = int(self.base_num_aux_tokens) + self.aux_ablation_extra_tokens

        self.learned_null_cond = None
        self.learned_null_aux_tokens = None
        self.learned_extra_aux_tokens = None
        self.learned_input_aux_tokens = None
        self.patch_map_aux_tokens = None
        if self.aux_dropout_mode == "learned_null":
            if self.decoder_aux_mode == "adaln_pool":
                self.learned_null_cond = nn.Parameter(torch.zeros(1, self.latent_dim))
            elif self.decoder_aux_mode in {"prepend", "cross_attn"}:
                if self.num_aux_tokens is None:
                    raise ValueError(
                        "aux_dropout_mode='learned_null' for token-conditioning decoders "
                        "requires num_aux_tokens to be defined."
                    )
                self.learned_null_aux_tokens = nn.Parameter(
                    torch.zeros(1, int(self.num_aux_tokens), self.latent_dim)
                )
        if self.aux_ablation_mode == "true_plus_learned_extra":
            self.learned_extra_aux_tokens = nn.Parameter(
                torch.zeros(1, int(self.aux_ablation_extra_tokens), self.latent_dim)
            )
            nn.init.trunc_normal_(self.learned_extra_aux_tokens, std=0.02)
        if self.aux_token_source == "learned":
            if self.num_aux_tokens is None:
                raise ValueError(
                    "aux_token_source='learned' requires num_aux_tokens to be set explicitly."
                )
            self.learned_input_aux_tokens = nn.Parameter(
                torch.zeros(1, int(self.num_aux_tokens), self.latent_dim)
            )
            nn.init.trunc_normal_(self.learned_input_aux_tokens, std=0.02)
        if self.aux_token_source == "patch_map":
            if self.num_aux_tokens is None:
                raise ValueError(
                    "aux_token_source='patch_map' requires num_aux_tokens to be set explicitly or inferable."
                )
            if patch_map_num_heads is None:
                patch_map_num_heads = max(1, self.latent_dim // 64)
            if self.latent_dim % int(patch_map_num_heads) != 0:
                raise ValueError(
                    f"patch_map_num_heads={patch_map_num_heads} must divide latent_dim={self.latent_dim}."
                )
            self.patch_map_aux_tokens = PatchMAPSummary(
                hidden_size=self.latent_dim,
                num_tokens=int(self.num_aux_tokens),
                num_heads=int(patch_map_num_heads),
                num_layers=int(patch_map_num_layers),
                mlp_ratio=float(patch_map_mlp_ratio),
                dropout=float(patch_map_dropout),
            )

        # ------------------------------------------------------------------
        # Decoder config
        # ------------------------------------------------------------------
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches))

        # pooled-cond / AdaLN path
        decoder_config.use_global_cond = self.use_global_cond
        decoder_config.global_cond_dim = self.latent_dim
        decoder_config.adaln_zero_init = self.adaln_zero_init

        # explicit aux-token path
        decoder_config.decoder_aux_mode = self.decoder_aux_mode
        decoder_config.aux_token_source = self.aux_token_source
        decoder_config.aux_pool = self.aux_pool
        decoder_config.num_aux_tokens = int(self.num_aux_tokens) if self.num_aux_tokens is not None else 0

        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)

        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location="cpu")
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(f"Missing keys when loading pretrained decoder: {keys.missing_keys}")

        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d

        # patch-latent stats
        self.latent_mean = None
        self.latent_var = None
        self.do_normalization = False

        # pooled global-cond stats (adaln_pool)
        self.global_mean = None
        self.global_var = None
        self.do_global_normalization = False

        # per-token aux stats (prepend / cross_attn)
        self.aux_mean = None
        self.aux_var = None
        self.do_aux_normalization = False

        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location="cpu")

            self.latent_mean = stats.get("mean", None)
            self.latent_var = stats.get("var", None)
            self.do_normalization = True

            self.global_mean = stats.get("global_mean", None)
            self.global_var = stats.get("global_var", None)
            self.do_global_normalization = (self.global_mean is not None) or (self.global_var is not None)

            self.aux_mean = stats.get("aux_mean", None)
            self.aux_var = stats.get("aux_var", None)
            self.do_aux_normalization = (self.aux_mean is not None) or (self.aux_var is not None)

            print(f"Loaded normalization stats from {normalization_stat_path}")
            if self.do_global_normalization:
                print("Loaded pooled global conditioning stats from normalization file.")
            if self.do_aux_normalization:
                print("Loaded per-token aux stats from normalization file.")
        if aux_normalization_stat_path is not None:
            aux_stats = torch.load(aux_normalization_stat_path, map_location="cpu")
            self.global_mean = aux_stats.get("global_mean", self.global_mean)
            self.global_var = aux_stats.get("global_var", self.global_var)
            self.do_global_normalization = (self.global_mean is not None) or (self.global_var is not None)

            self.aux_mean = aux_stats.get("aux_mean", self.aux_mean)
            self.aux_var = aux_stats.get("aux_var", self.aux_var)
            self.do_aux_normalization = (self.aux_mean is not None) or (self.aux_var is not None)

            print(f"Loaded aux normalization stats from {aux_normalization_stat_path}")
            if self.do_global_normalization:
                print("Loaded pooled global conditioning stats from aux normalization file.")
            if self.do_aux_normalization:
                print("Loaded per-token aux stats from aux normalization file.")
        if self.aux_ablation_mode != "none":
            print(
                "Using aux ablation: "
                f"mode={self.aux_ablation_mode}, "
                f"sigma={self.aux_ablation_sigma}, "
                f"extra_tokens={self.aux_ablation_extra_tokens}, "
                f"shuffle_offset={self.aux_shuffle_offset}"
            )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _infer_num_aux_tokens(self) -> Optional[int]:
        """
        Best-effort inference of number of selected encoder prefix tokens.
        """
        num_reg = getattr(self.encoder, "num_register_tokens", None)
        if num_reg is None:
            num_reg = getattr(self.encoder, "num_reg_tokens", None)

        num_prefix = getattr(self.encoder, "num_prefix_tokens", None)

        if self.aux_token_source == "cls":
            return 1

        if self.aux_token_source == "register":
            return int(num_reg) if num_reg is not None else None

        if self.aux_token_source == "cls+register":
            if num_reg is not None:
                return 1 + int(num_reg)
            if num_prefix is not None:
                return int(num_prefix)
            return None
        if self.aux_token_source == "learned":
            return None
        if self.aux_token_source == "patch_map":
            if num_reg is not None:
                return 1 + int(num_reg)
            if num_prefix is not None:
                return int(num_prefix)
            return None

        return None

    def _get_learned_aux_tokens(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.learned_input_aux_tokens is None:
            raise ValueError(
                "aux_token_source='learned' requested, but learned_input_aux_tokens is not initialized."
            )
        return self.learned_input_aux_tokens.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    def _get_patch_map_aux_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if self.patch_map_aux_tokens is None:
            raise ValueError(
                "aux_token_source='patch_map' requested, but patch_map_aux_tokens is not initialized."
            )
        return self.patch_map_aux_tokens(patch_tokens)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(
                x,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic",
                align_corners=False,
            )
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        return x

    def _encode_tokens(self, x: torch.Tensor, need_aux: bool = False):
        """
        Returns:
            patch_tokens: [B, N, C]
            global_tokens: [B, P, C] or None
        """
        x = self._normalize_input(x)

        if need_aux:
            if not hasattr(self.encoder, "forward_with_global"):
                raise AttributeError(
                    f"Encoder {self.encoder.__class__.__name__} does not implement "
                    f"forward_with_global(), but aux tokens were requested."
                )
            patch_tokens, global_tokens = self.encoder.forward_with_global(x)
            return patch_tokens, global_tokens

        patch_tokens = self.encoder(x)
        return patch_tokens, None

    def _select_aux_tokens(self, global_tokens: torch.Tensor) -> torch.Tensor:
        if global_tokens is None:
            raise ValueError("global_tokens is None, but aux tokens were requested.")
        if global_tokens.ndim != 3:
            raise ValueError(
                f"Expected global_tokens to have shape [B, P, C], got {tuple(global_tokens.shape)}"
            )

        p = global_tokens.shape[1]
        if p == 0:
            raise ValueError("Encoder returned no prefix/global tokens.")

        if self.aux_token_source == "cls":
            return global_tokens[:, :1]

        if self.aux_token_source == "register":
            if p <= 1:
                raise ValueError(
                    "Requested register-only conditioning, but no register tokens were found."
                )
            return global_tokens[:, 1:]

        if self.aux_token_source == "cls+register":
            return global_tokens

        raise ValueError(f"Unsupported aux_token_source: {self.aux_token_source}")

    def _pool_aux_tokens(self, aux_tokens: torch.Tensor) -> torch.Tensor:
        if aux_tokens.ndim != 3:
            raise ValueError(
                f"Expected aux_tokens to have shape [B, K, C], got {tuple(aux_tokens.shape)}"
            )

        if self.aux_pool == "mean":
            return aux_tokens.mean(dim=1)
        if self.aux_pool == "cls":
            return aux_tokens[:, 0]

        raise ValueError(f"Unsupported aux_pool: {self.aux_pool}")

    def _maybe_drop_aux(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Sample-wise dropout for pooled cond or token cond.
        Zeroes per sample rather than returning None.
        """
        if x is None:
            return None
        if not self.training:
            return x
        if self.aux_dropout_prob <= 0.0:
            return x

        if x.ndim == 2:
            keep = (torch.rand(x.size(0), 1, device=x.device) >= self.aux_dropout_prob).to(x.dtype)
        elif x.ndim == 3:
            keep = (torch.rand(x.size(0), 1, 1, device=x.device) >= self.aux_dropout_prob).to(x.dtype)
        else:
            raise ValueError(f"Unsupported aux tensor rank for dropout: ndim={x.ndim}")

        if self.aux_dropout_mode == "learned_null":
            if x.ndim == 2:
                if self.learned_null_cond is None:
                    raise ValueError(
                        "learned null pooled conditioning requested, but learned_null_cond "
                        "is not initialized."
                    )
                null_x = self.learned_null_cond.to(device=x.device, dtype=x.dtype)
            else:
                if self.learned_null_aux_tokens is None:
                    raise ValueError(
                        "learned null token conditioning requested, but "
                        "learned_null_aux_tokens is not initialized."
                    )
                if self.learned_null_aux_tokens.size(1) != x.size(1):
                    raise ValueError(
                        "learned_null_aux_tokens shape mismatch: "
                        f"expected K={self.learned_null_aux_tokens.size(1)}, got K={x.size(1)}"
                    )
                null_x = self.learned_null_aux_tokens.to(device=x.device, dtype=x.dtype)
            return x * keep + null_x.expand_as(x) * (1 - keep)

        return x * keep

    def _apply_aux_eval_override(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Eval-only override for robustness tests.
        Lets us compare the same checkpoint with full aux vs perturbed aux while
        keeping the real patch latent fixed.
        """
        if x is None:
            return None
        if self.training:
            return x
        if self.aux_eval_override == "none":
            return x
        if self.aux_eval_override == "zero":
            return torch.zeros_like(x)
        if self.aux_eval_override == "shuffle":
            if x.size(0) <= 1:
                return x
            return torch.roll(x, shifts=self.aux_eval_shuffle_offset, dims=0)
        if self.aux_eval_override == "gaussian":
            x_norm = self._normalize_aux_like(x)
            x_norm = x_norm + self.aux_eval_sigma * torch.randn_like(x_norm)
            return self._denormalize_aux_like(x_norm, x)
        if self.aux_eval_override == "scale":
            x_norm = self._normalize_aux_like(x)
            x_norm = self.aux_eval_scale * x_norm
            return self._denormalize_aux_like(x_norm, x)
        if self.aux_eval_override == "interp_zero":
            x_norm = self._normalize_aux_like(x)
            x_norm = (1.0 - self.aux_eval_alpha) * x_norm
            return self._denormalize_aux_like(x_norm, x)
        if self.aux_eval_override == "interp_other":
            if x.size(0) <= 1:
                return x
            x_norm = self._normalize_aux_like(x)
            other_norm = torch.roll(x_norm, shifts=self.aux_eval_shuffle_offset, dims=0)
            mixed = (1.0 - self.aux_eval_alpha) * x_norm + self.aux_eval_alpha * other_norm
            return self._denormalize_aux_like(mixed, x)

        if x.ndim == 2:
            if self.learned_null_cond is None:
                raise ValueError(
                    "aux_eval_override='learned_null' requested, but learned_null_cond "
                    "is not initialized for this checkpoint/config."
                )
            null_x = self.learned_null_cond.to(device=x.device, dtype=x.dtype)
        elif x.ndim == 3:
            if self.learned_null_aux_tokens is None:
                raise ValueError(
                    "aux_eval_override='learned_null' requested, but "
                    "learned_null_aux_tokens is not initialized for this checkpoint/config."
                )
            if self.learned_null_aux_tokens.size(1) != x.size(1):
                raise ValueError(
                    "learned_null_aux_tokens shape mismatch during eval override: "
                    f"expected K={self.learned_null_aux_tokens.size(1)}, got K={x.size(1)}"
                )
            null_x = self.learned_null_aux_tokens.to(device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unsupported aux tensor rank for eval override: ndim={x.ndim}")

        return null_x.expand_as(x)

    def _normalize_aux_like(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return self.normalize_global_cond(x)
        if x.ndim == 3:
            return self.normalize_aux_tokens(x)
        raise ValueError(f"Unsupported aux tensor rank for normalization: ndim={x.ndim}")

    def _denormalize_aux_like(self, x_norm: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if ref.ndim == 2:
            return self.denormalize_global_cond(x_norm)
        if ref.ndim == 3:
            return self.denormalize_aux_tokens(x_norm)
        raise ValueError(f"Unsupported aux tensor rank for denormalization: ndim={ref.ndim}")

    def _sample_gaussian_aux(
        self,
        aux_tokens: torch.Tensor,
        num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        if aux_tokens.ndim != 3:
            raise ValueError(
                f"Expected aux_tokens to have shape [B, K, C], got {tuple(aux_tokens.shape)}"
            )

        b, k, c = aux_tokens.shape
        target_k = k if num_tokens is None else int(num_tokens)
        mean = aux_tokens.mean(dim=(0, 1), keepdim=True)
        var = aux_tokens.var(dim=(0, 1), keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        noise = torch.randn(b, target_k, c, device=aux_tokens.device, dtype=aux_tokens.dtype)
        return mean + self.aux_ablation_sigma * std * noise

    def _apply_aux_ablation(
        self,
        aux_tokens: Optional[torch.Tensor],
        allow_extra_tokens: bool,
    ) -> Optional[torch.Tensor]:
        if aux_tokens is None or self.aux_ablation_mode == "none":
            return aux_tokens

        if aux_tokens.ndim != 3:
            raise ValueError(
                f"Expected aux_tokens to have shape [B, K, C], got {tuple(aux_tokens.shape)}"
            )

        if self.aux_ablation_mode == "shuffle":
            if aux_tokens.size(0) <= 1:
                return aux_tokens
            return torch.roll(aux_tokens, shifts=self.aux_shuffle_offset, dims=0)

        if self.aux_ablation_mode == "gaussian_replace":
            return self._sample_gaussian_aux(aux_tokens)

        if self.aux_ablation_mode == "true_plus_gaussian_extra":
            if not allow_extra_tokens:
                raise ValueError(
                    "aux_ablation_mode='true_plus_gaussian_extra' requires a token-conditioning "
                    "decoder path ('prepend' or 'cross_attn')."
                )
            gaussian_extra = self._sample_gaussian_aux(
                aux_tokens,
                num_tokens=self.aux_ablation_extra_tokens,
            )
            return torch.cat([aux_tokens, gaussian_extra], dim=1)

        if self.aux_ablation_mode == "true_plus_learned_extra":
            if not allow_extra_tokens:
                raise ValueError(
                    "aux_ablation_mode='true_plus_learned_extra' requires a token-conditioning "
                    "decoder path ('prepend' or 'cross_attn')."
                )
            if self.learned_extra_aux_tokens is None:
                raise ValueError(
                    "aux_ablation_mode='true_plus_learned_extra' requires learned_extra_aux_tokens "
                    "to be initialized."
                )
            learned_extra = self.learned_extra_aux_tokens.to(
                device=aux_tokens.device,
                dtype=aux_tokens.dtype,
            ).expand(aux_tokens.size(0), -1, -1)
            return torch.cat([aux_tokens, learned_extra], dim=1)

        raise ValueError(f"Unsupported aux_ablation_mode: {self.aux_ablation_mode}")

    def _patch_tokens_to_latent(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        z = patch_tokens

        if self.training and self.noise_tau > 0:
            z = self.noising(z)

        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            if h * w != n:
                raise ValueError(f"Patch token count {n} is not a perfect square.")
            z = z.transpose(1, 2).view(b, c, h, w)

        z = self._normalize_patch_latent(z)
        return z

    def _selected_globals_to_patch_grid(self, selected_tokens: torch.Tensor) -> torch.Tensor:
        if selected_tokens.ndim != 3:
            raise ValueError(
                f"Expected selected_tokens to have shape [B, K, C], got {tuple(selected_tokens.shape)}"
            )

        _, k, _ = selected_tokens.shape
        if k <= 0:
            raise ValueError("Selected global token sequence is empty.")

        if self.global_latent_adapter != "interp1d":
            raise ValueError(
                f"Unsupported global_latent_adapter: {self.global_latent_adapter}"
            )

        if k == self.base_patches:
            return selected_tokens

        if k == 1:
            return selected_tokens.expand(-1, self.base_patches, -1)

        x = selected_tokens.transpose(1, 2)
        x = nn.functional.interpolate(
            x,
            size=self.base_patches,
            mode="linear",
            align_corners=False,
        )
        return x.transpose(1, 2).contiguous()

    def _select_global_latent_tokens(
        self,
        patch_tokens: torch.Tensor,
        global_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.aux_token_source == "patch_map":
            return self._get_patch_map_aux_tokens(patch_tokens)
        if self.aux_token_source == "learned":
            return self._get_learned_aux_tokens(
                batch_size=patch_tokens.size(0),
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            )
        return self._select_aux_tokens(global_tokens)

    def _global_tokens_to_latent(
        self,
        patch_tokens: torch.Tensor,
        global_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        selected_tokens = self._select_global_latent_tokens(patch_tokens, global_tokens)
        patch_like_tokens = self._selected_globals_to_patch_grid(selected_tokens)
        return self._patch_tokens_to_latent(patch_like_tokens)

    @torch.no_grad()
    def _encode_global_latent(self, x: torch.Tensor) -> torch.Tensor:
        patch_tokens, global_tokens = self._encode_tokens(x, need_aux=True)
        return self._global_tokens_to_latent(patch_tokens, global_tokens)

    def _normalize_patch_latent(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z

    def _denormalize_patch_latent(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        return z

    def normalize_global_cond(self, cond: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if cond is None:
            return None
        if self.do_global_normalization:
            global_mean = self.global_mean.to(cond.device) if self.global_mean is not None else 0
            global_var = self.global_var.to(cond.device) if self.global_var is not None else 1
            cond = (cond - global_mean) / torch.sqrt(global_var + self.eps)
        return cond

    def denormalize_global_cond(self, cond: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if cond is None:
            return None
        if self.do_global_normalization:
            global_mean = self.global_mean.to(cond.device) if self.global_mean is not None else 0
            global_var = self.global_var.to(cond.device) if self.global_var is not None else 1
            cond = cond * torch.sqrt(global_var + self.eps) + global_mean
        return cond

    def normalize_aux_tokens(self, aux_tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if aux_tokens is None:
            return None
        if self.do_aux_normalization:
            aux_mean = self.aux_mean.to(aux_tokens.device) if self.aux_mean is not None else 0
            aux_var = self.aux_var.to(aux_tokens.device) if self.aux_var is not None else 1
            aux_tokens = (aux_tokens - aux_mean) / torch.sqrt(aux_var + self.eps)
        return aux_tokens

    def denormalize_aux_tokens(self, aux_tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if aux_tokens is None:
            return None
        if self.do_aux_normalization:
            aux_mean = self.aux_mean.to(aux_tokens.device) if self.aux_mean is not None else 0
            aux_var = self.aux_var.to(aux_tokens.device) if self.aux_var is not None else 1
            aux_tokens = aux_tokens * torch.sqrt(aux_var + self.eps) + aux_mean
        return aux_tokens

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand(
            (x.size(0),) + (1,) * (len(x.shape) - 1),
            device=x.device,
        )
        noise = noise_sigma * torch.randn_like(x)
        return x + noise

    def _maybe_noise_aux_cond(self, cond: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if cond is None:
            return None
        if not self.training or self.aux_noise_sigma <= 0.0:
            return cond
        cond_norm = self.normalize_global_cond(cond)
        cond_norm = cond_norm + self.aux_noise_sigma * torch.randn_like(cond_norm)
        return self.denormalize_global_cond(cond_norm)

    def _maybe_noise_aux_tokens(self, aux_tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if aux_tokens is None:
            return None
        if not self.training or self.aux_noise_sigma <= 0.0:
            return aux_tokens
        aux_norm = self.normalize_aux_tokens(aux_tokens)
        aux_norm = aux_norm + self.aux_noise_sigma * torch.randn_like(aux_norm)
        return self.denormalize_aux_tokens(aux_norm)

    # ------------------------------------------------------------------
    # Encode paths
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.latent_source == "global":
            return self._encode_global_latent(x)
        patch_tokens, _ = self._encode_tokens(x, need_aux=False)
        return self._patch_tokens_to_latent(patch_tokens)

    def encode_with_cond(self, x: torch.Tensor, normalize_global_cond: bool = False):
        with torch.no_grad():
            patch_tokens, global_tokens = self._encode_tokens(x, need_aux=True)
            z = self._patch_tokens_to_latent(patch_tokens)

        aux_tokens = self._select_aux_tokens(global_tokens)
        aux_tokens = self._apply_aux_ablation(aux_tokens, allow_extra_tokens=False)
        cond = self._pool_aux_tokens(aux_tokens)
        cond = self._maybe_noise_aux_cond(cond)
        cond = self._maybe_drop_aux(cond)
        cond = self._apply_aux_eval_override(cond)

        if normalize_global_cond:
            cond = self.normalize_global_cond(cond)

        return z, cond

    def encode_with_aux_tokens(self, x: torch.Tensor, normalize_aux_tokens: bool = False):
        if self.aux_token_source == "learned":
            with torch.no_grad():
                patch_tokens, _ = self._encode_tokens(x, need_aux=False)
                z = self._patch_tokens_to_latent(patch_tokens)
            aux_tokens = self._get_learned_aux_tokens(
                batch_size=patch_tokens.size(0),
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            )
        elif self.aux_token_source == "patch_map":
            with torch.no_grad():
                patch_tokens, _ = self._encode_tokens(x, need_aux=False)
                z = self._patch_tokens_to_latent(patch_tokens)
            aux_tokens = self._get_patch_map_aux_tokens(patch_tokens)
        else:
            with torch.no_grad():
                patch_tokens, global_tokens = self._encode_tokens(x, need_aux=True)
                z = self._patch_tokens_to_latent(patch_tokens)
            aux_tokens = self._select_aux_tokens(global_tokens)

        aux_tokens = self._apply_aux_ablation(aux_tokens, allow_extra_tokens=True)
        aux_tokens = self._maybe_noise_aux_tokens(aux_tokens)
        aux_tokens = self._maybe_drop_aux(aux_tokens)
        aux_tokens = self._apply_aux_eval_override(aux_tokens)

        if normalize_aux_tokens:
            aux_tokens = self.normalize_aux_tokens(aux_tokens)

        return z, aux_tokens

    @torch.no_grad()
    def encode_for_stage2(
        self,
        x: torch.Tensor,
        return_aux: str = "auto",              # "auto" | "none" | "pooled" | "tokens"
        normalize_global_cond: bool = True,
        normalize_aux_tokens: bool = True,
    ):
        """
        Returns tuple-based stage-2 state:
          (z, None)
          (z, pooled_cond)
          (z, aux_tokens)
        """
        if return_aux == "auto":
            if self.decoder_aux_mode == "adaln_pool":
                return_aux = "pooled"
            elif self.decoder_aux_mode in {"prepend", "cross_attn"}:
                return_aux = "tokens"
            else:
                return_aux = "none"

        if return_aux == "none":
            z = self.encode(x)
            return z, None

        if return_aux == "pooled":
            z, cond = self.encode_with_cond(x, normalize_global_cond=normalize_global_cond)
            return z, cond

        if return_aux == "tokens":
            z, aux_tokens = self.encode_with_aux_tokens(x, normalize_aux_tokens=normalize_aux_tokens)
            return z, aux_tokens

        raise ValueError(f"Unsupported return_aux='{return_aux}'")

    # ------------------------------------------------------------------
    # Decode path
    # ------------------------------------------------------------------
    def decode(
        self,
        z: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        aux_tokens: Optional[torch.Tensor] = None,
        cond_is_normalized: bool = False,
        aux_tokens_are_normalized: bool = False,
    ) -> torch.Tensor:
        z = self._denormalize_patch_latent(z)

        if self.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)

        if cond is not None:
            if cond_is_normalized:
                cond = self.denormalize_global_cond(cond)
            cond = cond.to(z.dtype)

        if aux_tokens is not None:
            if aux_tokens_are_normalized:
                aux_tokens = self.denormalize_aux_tokens(aux_tokens)
            aux_tokens = aux_tokens.to(z.dtype)

        if self.decoder_aux_mode == "adaln_pool":
            output = self.decoder(
                z,
                drop_cls_token=False,
                cond=cond,
                aux_tokens=None,
            ).logits
        elif self.decoder_aux_mode in {"prepend", "cross_attn"}:
            output = self.decoder(
                z,
                drop_cls_token=False,
                cond=None,
                aux_tokens=aux_tokens,
            ).logits
        else:
            output = self.decoder(
                z,
                drop_cls_token=False,
                cond=None,
                aux_tokens=None,
            ).logits

        x_rec = self.decoder.unpatchify(output)
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        return x_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.latent_source == "global":
            z = self.encode(x)
            return self.decode(z)

        if self.decoder_aux_mode == "discard":
            z = self.encode(x)
            return self.decode(z)

        if self.decoder_aux_mode == "adaln_pool":
            z, cond = self.encode_with_cond(x, normalize_global_cond=False)
            return self.decode(z, cond=cond, cond_is_normalized=False)

        if self.decoder_aux_mode in {"prepend", "cross_attn"}:
            z, aux_tokens = self.encode_with_aux_tokens(x, normalize_aux_tokens=False)
            return self.decode(z, aux_tokens=aux_tokens, aux_tokens_are_normalized=False)

        raise ValueError(f"Unsupported decoder_aux_mode: {self.decoder_aux_mode}")
