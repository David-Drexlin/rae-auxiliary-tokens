import torch
import torch.nn as nn
from .decoders import GeneralDecoder
from .encoders import ARCHS
from transformers import AutoConfig, AutoImageProcessor
from typing import Optional, Protocol, Tuple
from math import sqrt

class Stage1Protocal(Protocol):
    patch_size: int
    hidden_size: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def forward_with_global(self, x: torch.Tensor):
        ...

class RAE(nn.Module):

    def __init__(self, encoder_cls: str='Dinov2withNorm', encoder_config_path: str='facebook/dinov2-base', encoder_input_size: int=224, encoder_params: dict={}, decoder_config_path: str='vit_mae-base', decoder_patch_size: int=16, decoder_aux_mode: Optional[str]='prepend', aux_token_source: Optional[str]='register', aux_dropout_prob: Optional[float]=0.0, aux_dropout_mode: str='zero', aux_noise_sigma: float=0.0, num_aux_tokens: Optional[int]=4, noise_tau: float=0.0, reshape_to_2d: bool=True, normalization_stat_path: Optional[str]=None, aux_normalization_stat_path: Optional[str]=None, eps: float=1e-05):
        super().__init__()
        if aux_dropout_mode != 'zero':
            raise ValueError('Decoder training configuration required: aux_dropout_mode')
        if aux_dropout_prob != 0.0:
            raise ValueError('Decoder training configuration required: aux_dropout_prob')
        if aux_noise_sigma != 0.0:
            raise ValueError('Decoder training configuration required: aux_noise_sigma')
        if aux_token_source != 'register':
            raise ValueError('Decoder training configuration required: aux_token_source')
        if (decoder_aux_mode, num_aux_tokens) not in {('discard', 0), ('prepend', 4)}:
            raise ValueError('Expected discard with zero auxiliary tokens or prepend with four registers')
        if noise_tau != 0.0:
            raise ValueError('Decoder training configuration required: noise_tau')
        if reshape_to_2d != True:
            raise ValueError('Decoder training configuration required: reshape_to_2d')
        encoder_cls = ARCHS[encoder_cls]
        self.encoder: Stage1Protocal = encoder_cls(**encoder_params)
        self.eps = eps
        if hasattr(self.encoder, 'image_mean') and hasattr(self.encoder, 'image_std'):
            self.encoder_mean = self.encoder.image_mean
            self.encoder_std = self.encoder.image_std
        else:
            proc = AutoImageProcessor.from_pretrained(encoder_config_path)
            self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
            self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0, f'encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}'
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2
        self.decoder_aux_mode = str(decoder_aux_mode)
        self.aux_token_source = str('register')
        self.latent_source = str('patch')
        self.base_num_aux_tokens = int(num_aux_tokens)
        self.num_aux_tokens = int(num_aux_tokens)
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches))
        decoder_config.decoder_aux_mode = self.decoder_aux_mode
        decoder_config.aux_token_source = 'register'
        decoder_config.num_aux_tokens = self.num_aux_tokens
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)
        self.noise_tau = 0.0
        self.reshape_to_2d = True
        self.latent_mean = None
        self.latent_var = None
        self.do_normalization = False
        self.aux_mean = None
        self.aux_var = None
        self.do_aux_normalization = False
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location='cpu')
            self.latent_mean = stats.get('mean', None)
            self.latent_var = stats.get('var', None)
            self.do_normalization = True
            self.aux_mean = stats.get('aux_mean', None)
            self.aux_var = stats.get('aux_var', None)
            self.do_aux_normalization = self.aux_mean is not None or self.aux_var is not None
        if aux_normalization_stat_path is not None:
            aux_stats = torch.load(aux_normalization_stat_path, map_location='cpu')
            self.aux_mean = aux_stats.get('aux_mean', self.aux_mean)
            self.aux_var = aux_stats.get('aux_var', self.aux_var)
            self.do_aux_normalization = self.aux_mean is not None or self.aux_var is not None

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size), mode='bicubic', align_corners=False)
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        return x

    def _encode_tokens(self, x: torch.Tensor, need_aux: bool=False):
        x = self._normalize_input(x)
        if need_aux:
            if not hasattr(self.encoder, 'forward_with_global'):
                raise AttributeError(f'Encoder {self.encoder.__class__.__name__} does not implement forward_with_global(), but aux tokens were requested.')
            patch_tokens, global_tokens = self.encoder.forward_with_global(x)
            return (patch_tokens, global_tokens)
        patch_tokens = self.encoder(x)
        return (patch_tokens, None)

    def _select_aux_tokens_from_source(self, global_tokens: torch.Tensor) -> torch.Tensor:
        if global_tokens is None:
            raise ValueError('global_tokens is None, but aux tokens were requested.')
        if global_tokens.ndim != 3:
            raise ValueError(f'Expected global_tokens to have shape [B, P, C], got {tuple(global_tokens.shape)}')
        p = global_tokens.shape[1]
        if p == 0:
            raise ValueError('Encoder returned no prefix/global tokens.')
        if p <= 1:
            raise ValueError('Requested register-only conditioning, but no register tokens were found.')
        return global_tokens[:, 1:]

    def _select_aux_tokens(self, global_tokens: torch.Tensor) -> torch.Tensor:
        return self._select_aux_tokens_from_source(global_tokens)

    def _patch_tokens_to_latent(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        z = patch_tokens
        b, n, c = z.shape
        h = w = int(sqrt(n))
        if h * w != n:
            raise ValueError(f'Patch token count {n} is not a perfect square.')
        z = z.transpose(1, 2).view(b, c, h, w)
        z = self._normalize_patch_latent(z)
        return z

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            patch_tokens, _ = self._encode_tokens(x, need_aux=False)
        return self._patch_tokens_to_latent(patch_tokens)

    def encode_with_aux_tokens(self, x: torch.Tensor, normalize_aux_tokens: bool=False):
        if self.decoder_aux_mode == 'discard':
            return (self.encode(x), None)
        with torch.no_grad():
            patch_tokens, global_tokens = self._encode_tokens(x, need_aux=True)
        z = self._patch_tokens_to_latent(patch_tokens)
        aux_tokens = self._select_aux_tokens(global_tokens)
        if normalize_aux_tokens:
            aux_tokens = self.normalize_aux_tokens(aux_tokens)
        return (z, aux_tokens)

    def decode(self, z: torch.Tensor, cond: Optional[torch.Tensor]=None, aux_tokens: Optional[torch.Tensor]=None, cond_is_normalized: bool=False, aux_tokens_are_normalized: bool=False) -> torch.Tensor:
        z = self._denormalize_patch_latent(z)
        b, c, h, w = z.shape
        n = h * w
        z = z.view(b, c, n).transpose(1, 2)
        if aux_tokens is not None:
            if aux_tokens_are_normalized:
                aux_tokens = self.denormalize_aux_tokens(aux_tokens)
            aux_tokens = aux_tokens.to(z.dtype)
        decoder_aux = aux_tokens if self.decoder_aux_mode == 'prepend' else None
        output = self.decoder(z, drop_cls_token=False, cond=None, aux_tokens=decoder_aux).logits
        x_rec = self.decoder.unpatchify(output)
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        return x_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.decoder_aux_mode == 'discard':
            return self.decode(self.encode(x))
        z, aux_tokens = self.encode_with_aux_tokens(x, normalize_aux_tokens=False)
        return self.decode(z, aux_tokens=aux_tokens, aux_tokens_are_normalized=False)
