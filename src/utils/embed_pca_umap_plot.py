#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import matplotlib.pyplot as plt
from tqdm import tqdm


# -------------------------
# utils
# -------------------------
MISSING = "__MISSING__"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, p: Path) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))


def l2n(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _clean_tcga_join_key(x) -> str:
    """
    Mirror the join normalization used by TCGAPatchDataset:
    - missing / '--' / 'nan' / '' -> MISSING
    - otherwise lowercase + strip
    """
    if pd.isna(x):
        return MISSING
    s = str(x).strip()
    if s == "":
        return MISSING
    s_low = s.lower()
    if s_low in {"--", "'--", '"--', "nan", "none", "null"}:
        return MISSING
    return s_low


def effective_k(n: int, num_samples: int | None) -> int:
    """Interpret num_samples<=0 as 'all'."""
    if num_samples is None:
        return n
    if int(num_samples) <= 0:
        return n
    return min(int(num_samples), n)


def sample_indices_random(n: int, k: int, seed: int) -> np.ndarray:
    if k >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=int(k), replace=False)).astype(np.int64)


def block_shuffled_indices(n: int, block_len: int, seed: int) -> np.ndarray:
    """
    Permute indices by shuffling blocks (length block_len) then shuffling within each block.
    Good compromise for H5: less seeky than full random.
    """
    block_len = max(1, int(block_len))
    rng = np.random.default_rng(seed)

    starts = np.arange(0, n, block_len, dtype=np.int64)
    rng.shuffle(starts)

    out = np.empty(n, dtype=np.int64)
    w = 0
    for s in starts:
        e = min(s + block_len, n)
        blk = np.arange(s, e, dtype=np.int64)
        rng.shuffle(blk)
        out[w : w + len(blk)] = blk
        w += len(blk)
    return out


def _is_numeric_series(s: pd.Series) -> bool:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.notna().mean()) >= 0.95


def plot_numeric(E2: np.ndarray, s: pd.Series, title: str, out_png: Path) -> None:
    ensure_dir(out_png.parent)
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32)

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(E2[:, 0], E2[:, 1], c=x, s=2, alpha=0.75)
    plt.colorbar(sc, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_categorical(E2: np.ndarray, s: pd.Series, title: str, out_png: Path, topk: int = 12) -> None:
    ensure_dir(out_png.parent)
    s = s.astype("string").fillna("__NA__")
    vc = s.value_counts(dropna=False)

    # low-cardinality: color everything
    if len(vc) <= 25:
        cats = vc.index.tolist()
        lut = {c: i for i, c in enumerate(cats)}
        cidx = s.map(lut).to_numpy()

        plt.figure(figsize=(10, 8))
        plt.scatter(E2[:, 0], E2[:, 1], c=cidx, s=2, alpha=0.70)
        plt.title(f"{title} (k={len(cats)})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        return

    # high-cardinality: top-k explicit, rest gray
    keep = set(vc.head(topk).index.tolist())
    is_keep = s.isin(list(keep)).to_numpy()

    plt.figure(figsize=(10, 8))
    plt.scatter(E2[~is_keep, 0], E2[~is_keep, 1], s=2, alpha=0.12)

    cats = vc.head(topk).index.tolist()
    for c in cats:
        m = (s == c).to_numpy()
        plt.scatter(E2[m, 0], E2[m, 1], s=2, alpha=0.78, label=str(c))

    plt.title(f"{title} (top{topk} shown; total={len(vc)})")
    plt.axis("off")
    plt.legend(markerscale=4, fontsize=8, loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -------------------------
# PCA / UMAP
# -------------------------
def pca_2d(Z: np.ndarray, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        from sklearn.decomposition import PCA
    except Exception as e:
        raise RuntimeError(f"sklearn not available for PCA: {e}")

    pca = PCA(n_components=2, svd_solver="randomized", random_state=seed)
    E = pca.fit_transform(Z).astype(np.float32)
    info = {
        "pca2_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "pca2_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return E, info


def pca_pre_reduce(Z: np.ndarray, pca_dim: int, seed: int, ipca_batch: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Optional PCA pre-reduction for UMAP speed (NOT the PCA plot)."""
    if pca_dim <= 0 or pca_dim >= Z.shape[1]:
        return Z.astype(np.float32, copy=False), {"pca_pre": "skipped"}

    try:
        from sklearn.decomposition import PCA, IncrementalPCA
    except Exception as e:
        raise RuntimeError(f"sklearn not available for PCA: {e}")

    N = Z.shape[0]
    if N <= 200_000:
        pca = PCA(n_components=int(pca_dim), svd_solver="randomized", random_state=seed)
        Zp = pca.fit_transform(Z).astype(np.float32)
        info = {
            "pca_pre": "PCA(randomized)",
            "pca_pre_dim": int(pca_dim),
            "pca_pre_var_sum": float(np.sum(pca.explained_variance_ratio_)),
        }
        return Zp, info

    ipca = IncrementalPCA(n_components=int(pca_dim), batch_size=int(ipca_batch))
    for s in range(0, N, ipca_batch):
        e = min(s + ipca_batch, N)
        ipca.partial_fit(Z[s:e])

    Zp = np.empty((N, int(pca_dim)), dtype=np.float32)
    for s in range(0, N, ipca_batch):
        e = min(s + ipca_batch, N)
        Zp[s:e] = ipca.transform(Z[s:e]).astype(np.float32)

    info = {"pca_pre": "IncrementalPCA", "pca_pre_dim": int(pca_dim), "ipca_batch": int(ipca_batch)}
    return Zp, info


def umap_2d(Z_in: np.ndarray, seed: int, n_neighbors: int, min_dist: float, metric: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        import umap.umap_ as umap
    except Exception as e:
        raise RuntimeError(f"UMAP import failed: {e}\nInstall: pip install umap-learn")

    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        transform_seed=seed,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
    )
    E = reducer.fit_transform(Z_in).astype(np.float32)
    info = {"umap_n_neighbors": int(n_neighbors), "umap_min_dist": float(min_dist), "umap_metric": str(metric)}
    return E, info


# -------------------------
# Encoders (optional)
# -------------------------
class EncoderBase:
    def embed(self, imgs_01: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SigLIP2BaselineEncoder(EncoderBase):
    """Embedding = pooled SigLIP2 tokens (mean/cls), L2 normalized."""
    def __init__(self, model_name: str, pool: str, has_cls_token: bool, device: torch.device, precision: str):
        self.model_name = model_name
        self.pool = pool
        self.has_cls_token = bool(has_cls_token)
        self.device = device
        self.precision = precision.lower()

        from transformers import AutoImageProcessor, AutoModel

        self.proc = AutoImageProcessor.from_pretrained(model_name)
        self.mean = torch.tensor(self.proc.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.proc.image_std, device=device).view(1, 3, 1, 1)
        self.encoder = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.inference_mode()
    def embed(self, imgs_01: torch.Tensor) -> torch.Tensor:
        xb = (imgs_01 - self.mean) / self.std
        use_autocast = (self.device.type == "cuda" and self.precision in {"bf16", "fp16"})
        dtype = torch.bfloat16 if self.precision == "bf16" else (torch.float16 if self.precision == "fp16" else torch.float32)

        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = self.encoder(pixel_values=xb)
        else:
            out = self.encoder(pixel_values=xb)

        tok = getattr(out, "last_hidden_state", out[0])  # [B,N,C]
        if self.pool == "mean":
            pooled = tok.mean(dim=1)
        else:
            pooled = tok[:, 0] if self.has_cls_token else tok.mean(dim=1)

        return l2n(pooled.float())


class MeDiCLIPImageEncoder(EncoderBase):
    """Embedding = MeDiCLIP image tower output from a train_align.py checkpoint."""
    def __init__(self, ckpt_path: str, device: torch.device, precision: str):
        from transformers import AutoImageProcessor
        from mediclip.models.siglip2 import SigLIP2wNorm
        from mediclip.models.image_tower import SigLIP2PatchImageTower
        from mediclip.models.meta_encoder import CategoricalTabTransformerEncoder
        from mediclip.models.medi_clip import MeDiCLIPModel

        self.device = device
        self.precision = precision.lower()

        ckpt = torch.load(ckpt_path, map_location="cpu")
        r = ckpt["runtime"]

        model_name = str(r["model_name"])
        image_size = int(r["image_size"])
        embed_dim = int(r["embed_dim"])
        pool = str(r["img_pool"])
        has_cls = bool(r["img_has_cls_token"])
        proj_dropout = float(r["img_proj_dropout"])

        proc = AutoImageProcessor.from_pretrained(model_name)
        self.enc_input_size = image_size

        siglip_backbone = SigLIP2wNorm(model_name=model_name)
        image_tower = SigLIP2PatchImageTower(
            encoder=siglip_backbone,
            encoder_input_size=image_size,
            encoder_mean=torch.tensor(proc.image_mean).view(1, 3, 1, 1),
            encoder_std=torch.tensor(proc.image_std).view(1, 3, 1, 1),
            hidden_dim=siglip_backbone.hidden_size,
            out_dim=embed_dim,
            pool=pool,
            has_cls_token=has_cls,
            proj_dropout=proj_dropout,
            l2_normalize=True,
            freeze_backbone=True,
        )

        meta_tower = CategoricalTabTransformerEncoder(
            field_dims=(2,),
            d_model=int(r["meta_d_model"]),
            out_dim=int(r["meta_out_dim"]),
            n_heads=int(r["meta_heads"]),
            n_layers=int(r["meta_layers"]),
            ff_mult=int(r["meta_ff_mult"]),
            dropout=float(r["meta_dropout"]),
            emb_dropout=float(r["meta_emb_dropout"]),
            pool=str(r["meta_pool"]),
            use_cls_token=bool(r["meta_use_cls_token"]),
            use_field_embeddings=bool(r["meta_use_field_embeddings"]),
            project=bool(r["meta_project"]),
            l2_normalize=bool(r["meta_l2_normalize"]),
            field_dropout_p=0.0,
            cfg_null_id=int(r["meta_cfg_null_id"]),
        )

        self.model = MeDiCLIPModel(
            image_tower=image_tower,
            meta_tower=meta_tower,
            learnable_logit_scale=bool(r["learnable_logit_scale"]),
            force_l2_normalize=bool(r["force_l2_normalize"]),
        )
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.model.to(device).eval()

    @torch.inference_mode()
    def embed(self, imgs_01: torch.Tensor) -> torch.Tensor:
        xb = imgs_01
        if xb.shape[-1] != self.enc_input_size or xb.shape[-2] != self.enc_input_size:
            xb = torch.nn.functional.interpolate(
                xb, size=(self.enc_input_size, self.enc_input_size), mode="bicubic", align_corners=False
            )

        use_autocast = (self.device.type == "cuda" and self.precision in {"bf16", "fp16"})
        dtype = torch.bfloat16 if self.precision == "bf16" else (torch.float16 if self.precision == "fp16" else torch.float32)

        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=dtype):
                z = self.model.image_tower(xb)
        else:
            z = self.model.image_tower(xb)

        return l2n(z.float())


# -------------------------
# Pixel baseline (no encoder)
# -------------------------
@torch.inference_mode()
def pixel_embed(imgs_01: torch.Tensor, pixel_size: int, mode: str) -> torch.Tensor:
    x = imgs_01
    if int(pixel_size) > 0 and (x.shape[-1] != pixel_size or x.shape[-2] != pixel_size):
        x = torch.nn.functional.interpolate(x, size=(pixel_size, pixel_size), mode="bilinear", align_corners=False)

    if mode == "flatten":
        z = x.flatten(1)
    elif mode == "stats":
        mean = x.mean(dim=(2, 3))
        std = x.std(dim=(2, 3))
        mn = x.amin(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        z = torch.cat([mean, std, mn, mx], dim=1)
    else:
        raise ValueError(f"Unknown pixel-mode: {mode}")

    return l2n(z.float())


# -------------------------
# Dataset glue
# -------------------------
def build_transforms(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ]
    )


def tcga_collate(batch):
    n = len(batch[0])
    if n == 5:
        imgs, ys, metas, slide_ids, paths = zip(*batch)
    elif n == 4:
        imgs, ys, slide_ids, paths = zip(*batch)
        metas = [None] * len(batch)
    elif n == 3:
        imgs, ys, metas = zip(*batch)
        slide_ids, paths = [None] * len(batch), [None] * len(batch)
    else:
        raise ValueError(f"Unexpected TCGA item length={n}")

    imgs = torch.utils.data.dataloader.default_collate(imgs)
    ys = torch.utils.data.dataloader.default_collate(ys)

    meta_out = None
    if metas is not None and not all(m is None for m in metas):
        meta_out = torch.utils.data.dataloader.default_collate(metas)

    return imgs, ys, meta_out, list(slide_ids), list(paths)


@torch.inference_mode()
def embed_tcga(ds, indices: np.ndarray, encoder, device: torch.device, bs: int, nw: int, pixel_size: int, pixel_mode: str) -> Tuple[np.ndarray, pd.DataFrame]:
    subset = Subset(ds, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=tcga_collate,
    )

    it0 = iter(loader)
    imgs0, *_ = next(it0)
    imgs0 = imgs0.to(device, non_blocking=True)

    if encoder is None:
        z0 = pixel_embed(imgs0, pixel_size=pixel_size, mode=pixel_mode)
    else:
        z0 = encoder.embed(imgs0)
    D = int(z0.shape[1])

    Z = np.empty((len(indices), D), dtype=np.float32)
    rows = []
    w = 0

    for imgs, _y, _meta, slide_ids, paths in tqdm(loader, desc="embed(tcga)", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        if encoder is None:
            z = pixel_embed(imgs, pixel_size=pixel_size, mode=pixel_mode)
        else:
            z = encoder.embed(imgs)
        z = z.cpu().numpy().astype(np.float32)

        b = z.shape[0]
        Z[w : w + b] = z
        w += b
        for sid, p in zip(slide_ids, paths):
            rows.append({"slide_id": _clean_tcga_join_key(sid), "path": str(p)})

    meta_df = pd.DataFrame(rows)
    return Z, meta_df


def load_tcga_fields(csv_path: Path, id_field: str, fields: List[str]) -> pd.DataFrame:
    """
    Load only the fields needed for plotting, but normalize the join key the
    same way as TCGAPatchDataset so slide_id mapping works.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    if id_field not in df.columns:
        raise KeyError(f"TCGA CSV missing id_field='{id_field}'")
    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise KeyError(f"TCGA CSV missing requested fields: {missing}")

    df = df[[id_field] + fields].copy()
    df[id_field] = df[id_field].map(_clean_tcga_join_key)
    df = df[df[id_field] != MISSING].copy()
    df = df.drop_duplicates(subset=[id_field], keep="first")
    return df.set_index(id_field, drop=False)


@torch.inference_mode()
def embed_hest(ds, indices: np.ndarray, encoder, device: torch.device, bs: int, nw: int, pixel_size: int, pixel_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    subset = Subset(ds, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    it0 = iter(loader)
    b0 = next(it0)
    imgs0 = b0[0].to(device, non_blocking=True)
    if encoder is None:
        z0 = pixel_embed(imgs0, pixel_size=pixel_size, mode=pixel_mode)
    else:
        z0 = encoder.embed(imgs0)
    D = int(z0.shape[1])

    F = len(ds.meta_dims())
    Z = np.empty((len(indices), D), dtype=np.float32)
    metas = np.empty((len(indices), F), dtype=np.int64)

    w = 0
    for imgs, _y, meta in tqdm(loader, desc="embed(hest)", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        if encoder is None:
            z = pixel_embed(imgs, pixel_size=pixel_size, mode=pixel_mode)
        else:
            z = encoder.embed(imgs)
        z = z.cpu().numpy().astype(np.float32)

        b = z.shape[0]
        Z[w : w + b] = z
        metas[w : w + b] = meta.detach().cpu().numpy().astype(np.int64)
        w += b

    return Z, metas


def invert_vocabs(meta_vocabs: Dict[str, Dict[str, int]]) -> Dict[str, Dict[int, str]]:
    return {f: {v: k for k, v in vocab.items()} for f, vocab in meta_vocabs.items()}


def build_encoder(args, device: torch.device):
    if args.encoder == "none":
        return None
    if args.encoder == "siglip2":
        return SigLIP2BaselineEncoder(
            model_name=args.siglip_model,
            pool=args.pool,
            has_cls_token=args.has_cls_token,
            device=device,
            precision=args.precision,
        )
    if args.encoder == "mediclip":
        if args.mediclip_ckpt is None:
            raise ValueError("--encoder mediclip requires --mediclip-ckpt")
        return MeDiCLIPImageEncoder(args.mediclip_ckpt, device=device, precision=args.precision)
    raise ValueError("encoder must be: none | siglip2 | mediclip")


# -------------------------
# main (argparse)
# -------------------------
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)

    common.add_argument("--experiment", type=str, required=True)
    common.add_argument("--outdir", type=str, required=True)
    common.add_argument("--color-fields", type=str, nargs="+", required=True)

    common.add_argument("--image-size", type=int, default=128)
    common.add_argument("--batch-size", type=int, default=64)
    common.add_argument("--num-workers", type=int, default=8)
    common.add_argument("--device", type=str, default="cuda")
    common.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    common.add_argument("--seed", type=int, default=42)

    common.add_argument("--num-samples", type=int, default=50000, help="<=0 means all")
    common.add_argument("--sampling", type=str, default="random", choices=["random", "block_shuffle"])
    common.add_argument("--block-len", type=int, default=4096)

    common.add_argument("--encoder", type=str, default="none", choices=["none", "siglip2", "mediclip"])
    common.add_argument("--siglip-model", type=str, default="google/siglip2-base-patch16-256")
    common.add_argument("--mediclip-ckpt", type=str, default=None)
    common.add_argument("--pool", type=str, default="mean", choices=["mean", "cls"])
    common.add_argument("--has-cls-token", action="store_true")

    common.add_argument("--pixel-size", type=int, default=32)
    common.add_argument("--pixel-mode", type=str, default="flatten", choices=["flatten", "stats"])

    common.add_argument("--topk", type=int, default=12)

    common.add_argument("--umap-pca-pre", type=int, default=64)
    common.add_argument("--ipca-batch", type=int, default=8192)
    common.add_argument("--umap-n-neighbors", type=int, default=15)
    common.add_argument("--umap-min-dist", type=float, default=0.1)
    common.add_argument("--umap-metric", type=str, default="cosine", choices=["cosine", "euclidean", "manhattan"])

    ap = argparse.ArgumentParser("Embeddings -> PCA(2D) + UMAP(2D) plots for TCGA or HEST")
    sub = ap.add_subparsers(dest="dataset", required=True)

    ap_tcga = sub.add_parser("tcga", parents=[common])
    ap_tcga.add_argument("--tcga-root", type=str, required=True)
    ap_tcga.add_argument("--tcga-csv", type=str, required=True)
    ap_tcga.add_argument("--id-field", type=str, default="slide_submitter_id")
    ap_tcga.add_argument("--slide-sep", type=str, default="__")

    ap_hest = sub.add_parser("hest", parents=[common])
    ap_hest.add_argument("--hest-root", type=str, required=True)
    ap_hest.add_argument("--case-metadata-csv", type=str, required=True)
    ap_hest.add_argument("--hest-patches-subdir", type=str, default="patches")
    ap_hest.add_argument("--hest-block-size", type=int, default=256)
    ap_hest.add_argument("--hest-cache-blocks", type=int, default=1)

    return ap


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    out_root = Path(args.outdir) / args.experiment
    plots_dir = out_root / "plots"
    ensure_dir(plots_dir)
    save_json(vars(args), out_root / "run_args.json")

    encoder = build_encoder(args, device=device)
    tfm = build_transforms(args.image_size)

    if args.dataset == "tcga":
        from utils.tcga_dataset import TCGAPatchDataset

        ds = TCGAPatchDataset(
            root=Path(args.tcga_root),
            csv_path=Path(args.tcga_csv),
            meta_fields=[],
            transform=tfm,
            slide_sep=args.slide_sep,
            return_meta_none_if_empty=True,
            return_ids=True,
        )
        N = len(ds)
        K = effective_k(N, args.num_samples)

        if args.sampling == "block_shuffle":
            perm = block_shuffled_indices(N, block_len=args.block_len, seed=args.seed)
            sel = perm[:K].astype(np.int64)
        else:
            sel = sample_indices_random(N, K, seed=args.seed)

        Z, base_meta = embed_tcga(
            ds, sel, encoder, device=device, bs=args.batch_size, nw=args.num_workers,
            pixel_size=args.pixel_size, pixel_mode=args.pixel_mode
        )

        mapping = load_tcga_fields(Path(args.tcga_csv), args.id_field, args.color_fields)
        joined = pd.DataFrame({"dataset": "TCGA", "index": sel})
        joined["slide_id"] = base_meta["slide_id"].map(_clean_tcga_join_key).to_numpy()
        joined["path"] = base_meta["path"].to_numpy()

        for f in args.color_fields:
            joined[f] = joined["slide_id"].map(mapping[f])
            miss_rate = float(joined[f].isna().mean())
            nunique = int(joined[f].nunique(dropna=True))
            print(f"[tcga] field={f} matched={(1.0 - miss_rate):.2%} missing={miss_rate:.2%} unique_non_na={nunique}")

    else:
        from utils.hest_clip_dataset import HESTMetaPatchDataset

        ds = HESTMetaPatchDataset(
            hest_root=Path(args.hest_root),
            case_metadata_csv=Path(args.case_metadata_csv),
            meta_fields=list(args.color_fields),
            transform=tfm,
            patches_subdir=args.hest_patches_subdir,
            hest_block_size=args.hest_block_size,
            hest_cache_blocks=args.hest_cache_blocks,
            use_cfg_null_for_missing=False,
        )
        N = len(ds)
        K = effective_k(N, args.num_samples)

        if args.sampling == "block_shuffle":
            perm = block_shuffled_indices(N, block_len=args.block_len, seed=args.seed)
            sel = perm[:K].astype(np.int64)
        else:
            sel = sample_indices_random(N, K, seed=args.seed)

        Z, meta_ids = embed_hest(
            ds, sel, encoder, device=device, bs=args.batch_size, nw=args.num_workers,
            pixel_size=args.pixel_size, pixel_mode=args.pixel_mode
        )

        inv = invert_vocabs(ds.meta_vocabs)
        joined = pd.DataFrame({"dataset": "HEST", "index": sel})
        for j, f in enumerate(args.color_fields):
            joined[f] = [inv[f].get(int(v), "__UNK__") for v in meta_ids[:, j]]

    Z = Z.astype(np.float32)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    E_pca2, pca2_info = pca_2d(Z, seed=args.seed)
    np.save(out_root / "coords_pca2.npy", E_pca2)

    Z_umap_in, pca_pre_info = pca_pre_reduce(Z, pca_dim=args.umap_pca_pre, seed=args.seed, ipca_batch=args.ipca_batch)
    E_umap2, umap_info = umap_2d(
        Z_umap_in,
        seed=args.seed,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
    )
    np.save(out_root / "coords_umap2.npy", E_umap2)

    joined.to_csv(out_root / "joined_metadata.csv", index=False)

    save_json(
        {
            "n_points": int(Z.shape[0]),
            "embed_dim": int(Z.shape[1]),
            "encoder_used": args.encoder,
            **pca2_info,
            **pca_pre_info,
            **umap_info,
        },
        out_root / "run_stats.json",
    )

    for f in args.color_fields:
        s = joined[f]
        if _is_numeric_series(s):
            plot_numeric(E_pca2, s, title=f"{args.dataset.upper()} PCA(2D) colored by {f}", out_png=plots_dir / f"pca2__{f}.png")
            plot_numeric(E_umap2, s, title=f"{args.dataset.upper()} UMAP(2D) colored by {f}", out_png=plots_dir / f"umap2__{f}.png")
        else:
            plot_categorical(E_pca2, s, title=f"{args.dataset.upper()} PCA(2D) colored by {f}", out_png=plots_dir / f"pca2__{f}.png", topk=args.topk)
            plot_categorical(E_umap2, s, title=f"{args.dataset.upper()} UMAP(2D) colored by {f}", out_png=plots_dir / f"umap2__{f}.png", topk=args.topk)

    print(f"[done] outputs -> {out_root}")


if __name__ == "__main__":
    main()