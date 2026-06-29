# RAE_ROOT_PLACEHOLDER/src/utils/hest_clip_dataset.py
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.hest_h5_dataset import HESTPatchH5Dataset

CFG_NULL = "__CFG_NULL__"
UNK = "__UNK__"


@dataclass
class HESTDatasetStats:
    num_patches: int
    num_cases: int
    meta_fields: List[str]
    meta_cardinalities: List[int]


def _norm_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    return s


def _infer_hest_id_from_h5_path(p: Path) -> str:
    """
    HEST patch banks are typically per-sample ID (e.g. TENX149.h5).
    Be robust: try to match a leading token like 'TENX149', 'MISC18', 'NCBI123', etc.
    """
    stem = p.stem
    m = re.match(r"^([A-Za-z]+[0-9]+)", stem)
    return m.group(1) if m else stem


def _build_categorical_vocab(values: Sequence[str]) -> Dict[str, int]:
    """
    0 reserved for CFG null (not used for missing data),
    1 reserved for UNK (used for missing/unknown),
    then observed categories from training split.
    """
    vocab: Dict[str, int] = {CFG_NULL: 0, UNK: 1}
    for v in values:
        v = v.strip()
        if not v:
            continue
        key = v.lower()
        if key not in vocab:
            vocab[key] = len(vocab)
    return vocab


class HESTMetaPatchDataset(Dataset):
    """
    Returns (img_tensor, y_dummy, meta_int_tensor[F]).

    - patches come from HESTPatchH5Dataset (H5 streaming)
    - case-level metadata comes from case_metadata.csv (keyed by 'id')
    - each H5 file is assumed to correspond to one sample id; all patches in that file share metadata
    """

    def __init__(
        self,
        hest_root: Union[str, Path],
        case_metadata_csv: Union[str, Path],
        meta_fields: List[str],
        transform=None,
        ids: Optional[List[str]] = None,
        # H5 streaming knobs
        patches_subdir: str = "patches",
        hest_block_size: int = 256,
        hest_cache_blocks: int = 2,
        # vocab sharing
        meta_vocabs: Optional[Dict[str, Dict[str, int]]] = None,
        # missing handling
        use_cfg_null_for_missing: bool = False,  # default False: missing -> UNK, keep CFG_NULL reserved
    ):
        self.hest_root = Path(hest_root)
        self.case_metadata_csv = Path(case_metadata_csv)
        self.meta_fields = list(meta_fields)
        self.transform = transform
        self.use_cfg_null_for_missing = bool(use_cfg_null_for_missing)

        if not self.case_metadata_csv.exists():
            raise FileNotFoundError(f"case_metadata_csv not found: {self.case_metadata_csv}")
        if len(self.meta_fields) == 0:
            raise ValueError("meta_fields must be non-empty")

        # Load case metadata table
        df = pd.read_csv(self.case_metadata_csv, low_memory=False)
        if "id" not in df.columns:
            raise ValueError("case_metadata_csv must contain an 'id' column")
        df["id"] = df["id"].astype(str)

        # Keep only relevant columns + id (avoid dragging 300+ gene columns around)
        keep_cols = ["id"] + [c for c in self.meta_fields if c in df.columns]
        missing_cols = [c for c in self.meta_fields if c not in df.columns]
        if missing_cols:
            raise ValueError(f"These meta_fields are not present in case_metadata.csv: {missing_cols}")
        df = df[keep_cols].copy()

        # Build underlying patch dataset (H5 scan); can restrict by ids
        self.patch_ds = HESTPatchH5Dataset(
            root=self.hest_root,
            ids=ids,
            patches_subdir=patches_subdir,
            transform=transform,
            return_index=True,
            block_size=hest_block_size,
            cache_blocks=hest_cache_blocks,
        )

        # Map each H5 file/spec -> sample id
        # NOTE: we rely on HESTPatchH5Dataset internals (stable enough for our repo usage).
        specs = getattr(self.patch_ds, "_specs", None)
        if specs is None:
            raise RuntimeError("HESTPatchH5Dataset does not expose _specs; update the loader or this dataset.")
        self._spec_ids: List[str] = [_infer_hest_id_from_h5_path(s.path) for s in specs]

        # Build meta_vocabs from TRAIN split if not provided
        if meta_vocabs is None:
            meta_vocabs = {}
            for f in self.meta_fields:
                vals = df[f].map(_norm_str).tolist()
                meta_vocabs[f] = _build_categorical_vocab([v for v in vals if v])
        else:
            # shallow copy to avoid accidental mutation
            meta_vocabs = {k: dict(v) for k, v in meta_vocabs.items()}

        self.meta_vocabs = meta_vocabs

        # Precompute encoded meta per spec (fast lookup during __getitem__)
        df_idx = df.set_index("id", drop=False)
        F = len(self.meta_fields)
        meta_per_spec = np.empty((len(self._spec_ids), F), dtype=np.int64)

        for si, sid in enumerate(self._spec_ids):
            if sid not in df_idx.index:
                # no metadata row: all UNK (or CFG_NULL if you insist)
                fill = 0 if self.use_cfg_null_for_missing else 1
                meta_per_spec[si, :] = fill
                continue

            row = df_idx.loc[sid]
            for j, f in enumerate(self.meta_fields):
                raw = _norm_str(row[f])
                if raw == "":
                    meta_per_spec[si, j] = 0 if self.use_cfg_null_for_missing else 1
                else:
                    key = raw.lower()
                    meta_per_spec[si, j] = self.meta_vocabs[f].get(key, 1)  # default UNK

        self._meta_per_spec = meta_per_spec  # (S,F)
        self.y_vocab = {"__DUMMY__": 0}       # to stay compatible with your checkpoint format

        self.stats = HESTDatasetStats(
            num_patches=len(self.patch_ds),
            num_cases=len(set(self._spec_ids)),
            meta_fields=self.meta_fields,
            meta_cardinalities=[len(self.meta_vocabs[f]) for f in self.meta_fields],
        )

    def meta_dims(self) -> List[int]:
        return [len(self.meta_vocabs[f]) for f in self.meta_fields]

    def __len__(self) -> int:
        return len(self.patch_ds)

    def __getitem__(self, idx: int):
        img, gidx = self.patch_ds[idx]  # img already transformed to tensor
        # find which H5 spec this idx came from (again rely on patch_ds internal method)
        spec_idx, _local_idx = self.patch_ds._find_spec(int(gidx))  # type: ignore[attr-defined]
        meta = torch.from_numpy(self._meta_per_spec[int(spec_idx)].copy()).long()
        y = torch.tensor(0, dtype=torch.long)
        return img, y, meta