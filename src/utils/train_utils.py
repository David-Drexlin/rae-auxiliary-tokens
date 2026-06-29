from omegaconf import OmegaConf, DictConfig
from typing import List, Tuple, Union
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from copy import deepcopy
import yaml
from .dist_utils import setup_distributed
from utils.tcga_dataset import TCGAPatchDataset 
from utils.celeba_dataset import CelebAMetaDataset
from torch.utils.data.dataloader import default_collate
from torch.cuda.amp import GradScaler
import torch.nn.functional as F


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def sobel_grad_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Sobel gradient mismatch on luminance.
    x, y: [B,3,H,W] in [0,1]
    returns scalar
    """
    # luminance to focus on structure, not stain hue noise
    xg = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
    yg = 0.2989 * y[:, 0:1] + 0.5870 * y[:, 1:2] + 0.1140 * y[:, 2:3]

    kx = x.new_tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]]).view(1, 1, 3, 3) / 4.0
    ky = x.new_tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]]).view(1, 1, 3, 3) / 4.0

    gx_x = F.conv2d(xg, kx, padding=1)
    gy_x = F.conv2d(xg, ky, padding=1)
    gx_y = F.conv2d(yg, kx, padding=1)
    gy_y = F.conv2d(yg, ky, padding=1)

    return (gx_y - gx_x).abs().mean() + (gy_y - gy_x).abs().mean()

def get_autocast_scaler(args):
    prec = str(getattr(args, "precision", "fp32")).lower()

    if prec == "fp32":
        return None, {"enabled": False}

    if prec == "fp16":
        scaler = GradScaler(enabled=True)
        return scaler, {"enabled": True, "dtype": torch.float16}

    if prec == "bf16":
        # GradScaler is not used for bf16
        return None, {"enabled": True, "dtype": torch.bfloat16}

    raise ValueError(f"Unknown precision: {prec} (expected fp32|fp16|bf16)")


def _sanitize_yaml_loaded_object(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_yaml_loaded_object(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_yaml_loaded_object(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_yaml_loaded_object(v) for v in obj]
    return obj


def _load_config_with_unsafe_yaml_fallback(config_path: Union[str, Path]) -> DictConfig:
    config_path = Path(config_path).expanduser()
    try:
        return OmegaConf.load(str(config_path))
    except Exception:
        # Stage-1 experiment configs saved by OmegaConf may contain Python YAML tags
        # in cmd_args (e.g. pathlib.PosixPath). These are trusted local files.
        with open(config_path, "r") as f:
            data = yaml.unsafe_load(f)
        return OmegaConf.create(_sanitize_yaml_loaded_object(data))


def _resolve_stage1_decoder_checkpoint(
    source_config_path: Path,
    decoder_ckpt: str,
    existing_pretrained_path,
) -> Union[str, None]:
    mode = str(decoder_ckpt).strip().lower()
    if mode in {"keep", "none"}:
        return existing_pretrained_path

    if mode == "auto":
        checkpoint_dir = source_config_path.parent / "checkpoints"
        for pattern in ("decoder_ema_ep-*.pt", "decoder_ep-*.pt"):
            matches = sorted(checkpoint_dir.glob(pattern))
            if matches:
                return str(matches[-1])
        if existing_pretrained_path is not None:
            return existing_pretrained_path
        raise FileNotFoundError(
            f"Could not auto-resolve a decoder checkpoint next to {source_config_path}. "
            f"Looked under {checkpoint_dir} for decoder_ema_ep-*.pt / decoder_ep-*.pt."
        )

    explicit_path = Path(decoder_ckpt).expanduser()
    if not explicit_path.is_absolute():
        explicit_path = (source_config_path.parent / explicit_path).resolve()
    if not explicit_path.exists():
        raise FileNotFoundError(f"Requested decoder checkpoint does not exist: {explicit_path}")
    return str(explicit_path)


def _resolve_stage1_param_path(value, source_config_path: Path):
    if value is None or not isinstance(value, (str, Path)):
        return value

    path_value = Path(value).expanduser()
    if path_value.is_absolute():
        return str(path_value)

    repo_candidate = (_repo_root() / path_value).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)

    source_candidate = (source_config_path.parent / path_value).resolve()
    if source_candidate.exists():
        return str(source_candidate)

    return str(value)


def _resolve_stage1_source(config):
    stage1_source = config.get("stage1_source", None)
    if stage1_source is None:
        stage1_source = config.get("stage_1_source", None)
    if stage1_source is None:
        return config

    if isinstance(stage1_source, DictConfig):
        stage1_source = OmegaConf.to_container(stage1_source, resolve=True)

    if isinstance(stage1_source, (str, Path)):
        source_config_path = Path(stage1_source).expanduser()
        decoder_ckpt = "auto"
    elif isinstance(stage1_source, dict):
        source_cfg = (
            stage1_source.get("config_path")
            or stage1_source.get("config")
            or stage1_source.get("path")
        )
        if source_cfg is None:
            raise ValueError(
                "stage1_source must provide one of: config_path, config, or path."
            )
        source_config_path = Path(source_cfg).expanduser()
        decoder_ckpt = str(stage1_source.get("decoder_ckpt", stage1_source.get("decoder_checkpoint", "auto")))
    else:
        raise TypeError(f"Unsupported stage1_source type: {type(stage1_source)}")

    source_config = _load_config_with_unsafe_yaml_fallback(source_config_path)
    source_stage1 = source_config.get("stage_1", None)
    if source_stage1 is None:
        raise ValueError(
            f"Stage-1 source config does not define a stage_1 section: {source_config_path}"
        )

    inline_stage1 = config.get("stage_1", None)
    if inline_stage1 is None:
        merged_stage1 = deepcopy(source_stage1)
    else:
        merged_stage1 = OmegaConf.merge(deepcopy(source_stage1), deepcopy(inline_stage1))

    params = merged_stage1.get("params", None)
    if params is not None:
        for key in ("encoder_config_path", "decoder_config_path", "normalization_stat_path", "pretrained_decoder_path"):
            if key in params:
                params[key] = _resolve_stage1_param_path(params.get(key), source_config_path)

        existing_pretrained_path = params.get("pretrained_decoder_path", None)
        resolved_pretrained_path = _resolve_stage1_decoder_checkpoint(
            source_config_path=source_config_path,
            decoder_ckpt=decoder_ckpt,
            existing_pretrained_path=existing_pretrained_path,
        )
        if resolved_pretrained_path is not None:
            merged_stage1.params.pretrained_decoder_path = resolved_pretrained_path

    config.stage_1 = merged_stage1
    return config
    
def parse_configs(config):
    if isinstance(config, (str, Path)):
        config = _load_config_with_unsafe_yaml_fallback(config)

    config = _resolve_stage1_source(config)

    data_config = config.get("data", None)
    rae_config = config.get("stage_1", None)
    stage2_config = config.get("stage_2", None)
    transport_config = config.get("transport", None)
    sampler_config = config.get("sampler", None)
    guidance_config = config.get("guidance", None)
    misc = config.get("misc", None)
    training_config = config.get("training", None)
    eval_config = config.get("eval", None)

    return (data_config, rae_config, stage2_config, transport_config,
            sampler_config, guidance_config, misc, training_config, eval_config)

def tcga_collate(batch):
    """
    Batch items are either:
      (img, y, meta) where meta is Tensor(M,) OR None
    Returns:
      images: Tensor(B,C,H,W)
      y     : Tensor(B,)
      meta  : Tensor(B,M) OR None
    """
    imgs, ys, metas = zip(*batch)
    imgs = default_collate(imgs)
    ys = default_collate(ys)

    # robust: if ANY sample has meta=None, return meta=None
    if any(m is None for m in metas):
        return imgs, ys, None

    metas = default_collate(metas)
    return imgs, ys, metas

def none_or_str(value):
    if value == 'None':
        return None
    return value

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def prepare_dataloader(
    data_path: Path,
    batch_size: int,
    workers: int,
    rank: int,
    world_size: int,
    transform=None,
    data_cfg=None,
):
    if data_cfg is None:
        kind = "imagefolder"
    else:
        kind = str(data_cfg.get("kind", "imagefolder"))

    collate_fn = None

    if kind == "imagefolder":
        dataset = ImageFolder(str(data_path), transform=transform)

    elif kind == "tcga_csv":
        root = Path(data_cfg["root"])
        csv_path = Path(data_cfg["csv_path"])
        meta_fields = list(data_cfg.get("meta_fields", []))
        slide_sep = str(data_cfg.get("slide_sep", "__"))

        dataset = TCGAPatchDataset(
            root=root,
            csv_path=csv_path,
            meta_fields=meta_fields,
            transform=transform,
            slide_sep=slide_sep,
        )
        collate_fn = tcga_collate  # <-- important for meta=None compatibility

    elif kind in {"celeba", "celeba_torchvision"}:
        from utils.celeba_dataset import CelebAMetaDataset

        root = str(data_cfg.get("root", "DATASETS_ROOT_PLACEHOLDER"))
        split = str(data_cfg.get("split", "train"))
        label_type = str(data_cfg.get("label_type", "attr"))
        label_attr = str(data_cfg.get("label_attr", "Smiling"))
        meta_fields = list(data_cfg.get("meta_fields", []))
        download = bool(data_cfg.get("download", False))
        remap_identity = bool(data_cfg.get("remap_identity", True))

        dataset = CelebAMetaDataset(
            root=root,
            split=split,
            label_type=label_type,
            label_attr=label_attr,
            meta_fields=meta_fields,
            transform=transform,
            download=download,
            remap_identity=remap_identity,
        )

    else:
        raise ValueError(f"Unknown data.kind={kind}")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    if workers > 0:
        prefetch_factor = 4
        if data_cfg is not None:
            prefetch_factor = int(data_cfg.get("prefetch_factor", prefetch_factor))
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(**loader_kwargs)
    return loader, sampler
