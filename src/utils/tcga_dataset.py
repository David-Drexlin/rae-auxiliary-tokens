# tcga_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


MISSING = "__MISSING__"
UNK = "__UNK__"
CFG_NULL = "__CFG_NULL__" 

def _load_exclude_ids(
    exclude_csv_paths: Optional[Sequence[Union[str, Path]]],
    id_field: str,
) -> set[str]:
    """
    Load slide IDs from one or more CSVs to exclude from the current dataset.
    IDs are cleaned with _clean to match join logic.
    """
    exclude_ids: set[str] = set()
    if not exclude_csv_paths:
        return exclude_ids

    for p in exclude_csv_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"exclude_csv_path does not exist: {p}")
        ex = pd.read_csv(p)
        if id_field not in ex.columns:
            raise ValueError(f"Exclude CSV missing id_field '{id_field}': {p}")
        ex_ids = ex[id_field].astype(str).map(_clean)
        exclude_ids.update(ex_ids.tolist())

    # drop MISSING if it sneaks in
    exclude_ids.discard(MISSING)
    return exclude_ids

def _clean(x) -> str:
    """
    Normalize raw CSV cell values into a stable categorical token.
    - missing / '--' / 'nan' / '' -> MISSING
    - otherwise: lowercased stripped string
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


def build_vocab(
    values: Sequence[str],
    reserve_null: bool = False,
    reserve_missing: bool = True,
    reserve_unk: bool = True,
) -> Dict[str, int]:
    """
    Stable vocab with deterministic ordering.

    Typical usage:
    - labels: reserve_null=False, reserve_missing=False, reserve_unk=True
    - meta:   reserve_null=True,  reserve_missing=False, reserve_unk=True
              (after mapping MISSING -> UNK in the dataframe)

    ID convention (for meta, with reserve_null=True):
      0 -> CFG_NULL (reserved, not emitted by dataset rows)
      1 -> UNK      (missing/unknown CSV values)
      2+ -> real categories
    """
    vocab: Dict[str, int] = {}

    if reserve_null:
        vocab[CFG_NULL] = len(vocab)
    if reserve_missing:
        vocab[MISSING] = len(vocab)
    if reserve_unk:
        vocab[UNK] = len(vocab)

    # deterministic
    for v in sorted(set(values)):
        # never duplicate reserved tokens
        if v in vocab:
            continue
        vocab[v] = len(vocab)

    return vocab

@dataclass(frozen=True)
class TCGADatasetStats:
    num_images: int
    num_slides: int
    num_classes: int
    meta_fields: Tuple[str, ...]
    meta_cardinalities: Tuple[int, ...]


class TCGAPatchDataset(Dataset):
    """
    ImageFolder-like dataset for TCGA patches + metadata join via slide_submitter_id.

    Expected image naming (your current convention):
      <slide_submitter_id>__<anything>.jpg
      e.g. TCGA-OR-A5J1-01Z-00-DX1__0_0_506.jpg

    root:
      /path/to/tcga_imagefolder_traincsv/
        <cancer_type_folder_1>/*.jpg
        <cancer_type_folder_2>/*.jpg
        ...

    csv_path:
      train_metadata_df_complex.csv

    Returns:
      (img, y, meta)
        img  : Float tensor (C,H,W) after transform
        y    : Long tensor scalar (class id)
        meta : Long tensor (M,) with M=len(meta_fields),
               OR None if meta_fields empty and return_meta_none_if_empty=True.
    """

    def __init__(
        self,
        root: Union[str, Path],
        csv_path: Union[str, Path],
        meta_fields: Sequence[str],
        transform=None,
        slide_sep: str = "__",
        image_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".webp"),
        id_field: str = "slide_submitter_id",
        label_field: str = "cancer_type",
        drop_missing_label: bool = True,
        strict_join: bool = False,
        return_ids: bool = False,  # debugging: also return slide_id and path
        return_meta_none_if_empty: bool = True,
        max_images: Optional[int] = None,  # debugging / smoke tests

        # ---------------- NEW ----------------
        y_vocab: Optional[Dict[str, int]] = None,
        meta_vocabs: Optional[Dict[str, Dict[str, int]]] = None,
        exclude_csv_paths: Optional[Sequence[Union[str, Path]]] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.slide_sep = str(slide_sep)
        self.image_exts = tuple(e.lower() for e in image_exts)
        self.id_field = str(id_field)
        self.label_field = str(label_field)
        self.drop_missing_label = bool(drop_missing_label)
        self.strict_join = bool(strict_join)
        self.return_ids = bool(return_ids)
        self.return_meta_none_if_empty = bool(return_meta_none_if_empty)
        self.max_images = max_images

        self.meta_fields = tuple(meta_fields)

        # explicit IDs for meta conditioning convention
        self.meta_null_id = 0  # reserved for CFG/unconditional meta token
        self.meta_unk_id = 1   # emitted by dataset for missing/unknown meta values

        if not self.root.exists():
            raise FileNotFoundError(f"root does not exist: {self.root}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"csv_path does not exist: {self.csv_path}")

        # ---- load + validate CSV ----
        df = pd.read_csv(self.csv_path)

        required = {self.id_field, self.label_field}
        missing_cols = sorted(list(required - set(df.columns)))
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")

        for f in self.meta_fields:
            if f not in df.columns:
                raise ValueError(f"CSV missing meta field column: '{f}'")

        # normalize columns
        df[self.id_field] = df[self.id_field].astype(str).map(_clean)
        df[self.label_field] = df[self.label_field].map(_clean)

        for f in self.meta_fields:
            df[f] = df[f].map(_clean)
            df[f] = df[f].replace({MISSING: UNK})

        # drop rows with missing slide id (cannot join)
        df = df[df[self.id_field] != MISSING].copy()

        # drop rows with missing label if requested
        if self.drop_missing_label:
            df = df[df[self.label_field] != MISSING].copy()

        # ---------------- NEW: exclude slide IDs from other CSVs ----------------
        exclude_ids = _load_exclude_ids(exclude_csv_paths, id_field=self.id_field)
        if exclude_ids:
            df = df[~df[self.id_field].isin(exclude_ids)].copy()

        if len(df) == 0:
            raise RuntimeError(
                "After cleaning/filtering/excluding, metadata CSV has 0 usable rows. "
                "Check label_field / id_field / exclude_csv_paths / cleaning rules."
            )

        # de-duplicate slides (keep first occurrence)
        dup_count = int(df.duplicated(self.id_field).sum())
        if dup_count > 0:
            df = df.drop_duplicates(self.id_field, keep="first").copy()

        self.by_slide = df.set_index(self.id_field, drop=False)

        # ---- build OR reuse vocabs (deterministic) ----
        if y_vocab is not None:
            # must contain UNK for safe fallback
            if UNK not in y_vocab:
                raise ValueError("Provided y_vocab must contain UNK token.")
            self.y_vocab = y_vocab
        else:
            self.y_vocab = build_vocab(
                self.by_slide[self.label_field].tolist(),
                reserve_null=False,
                reserve_missing=False,
                reserve_unk=True,
            )

        if meta_vocabs is not None:
            # validate provided vocabs cover required fields
            for f in self.meta_fields:
                if f not in meta_vocabs:
                    raise ValueError(f"Provided meta_vocabs missing field '{f}'.")
                v = meta_vocabs[f]
                if v.get(CFG_NULL, None) != self.meta_null_id:
                    raise ValueError(f"Provided meta_vocabs['{f}'] must have CFG_NULL id=0.")
                if v.get(UNK, None) != self.meta_unk_id:
                    raise ValueError(f"Provided meta_vocabs['{f}'] must have UNK id=1.")
            self.meta_vocabs = meta_vocabs
        else:
            self.meta_vocabs = {
                f: build_vocab(
                    self.by_slide[f].tolist(),
                    reserve_null=True,
                    reserve_missing=False,
                    reserve_unk=True,
                )
                for f in self.meta_fields
            }

        # sanity check meta ID convention (also for built vocabs)
        for f in self.meta_fields:
            vocab = self.meta_vocabs[f]
            if vocab.get(CFG_NULL, None) != self.meta_null_id:
                raise RuntimeError(f"Meta vocab for '{f}' does not reserve CFG_NULL at id 0.")
            if vocab.get(UNK, None) != self.meta_unk_id:
                raise RuntimeError(f"Meta vocab for '{f}' does not reserve UNK at id 1.")

        # slide -> integer row index (dense)
        self.slide_ids: List[str] = list(self.by_slide.index.tolist())
        self.slide_id_to_idx: Dict[str, int] = {sid: i for i, sid in enumerate(self.slide_ids)}

        # pre-encode slide-level targets for speed
        self.slide_y: List[int] = []
        self.slide_meta: List[List[int]] = []

        for sid in self.slide_ids:
            row = self.by_slide.loc[sid]

            y_str = row[self.label_field]
            y_id = self.y_vocab.get(y_str, self.y_vocab[UNK])
            self.slide_y.append(int(y_id))

            if len(self.meta_fields) > 0:
                mids = []
                for f in self.meta_fields:
                    v = row[f]  # cleaned, MISSING already mapped to UNK
                    mids.append(int(self.meta_vocabs[f].get(v, self.meta_vocabs[f][UNK])))
                self.slide_meta.append(mids)
            else:
                self.slide_meta.append([])

        # ---- enumerate images + join to slides ----
        self.samples: List[Tuple[Path, int]] = []  # (img_path, slide_idx)

        all_files: List[Path] = []
        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in self.image_exts:
                all_files.append(p)
        all_files = sorted(all_files)

        if self.max_images is not None:
            all_files = all_files[: int(self.max_images)]

        missed = 0
        for p in all_files:
            name = p.name
            slide_id = name.split(self.slide_sep, 1)[0]
            slide_id = _clean(slide_id)

            idx = self.slide_id_to_idx.get(slide_id, None)
            if idx is None:
                missed += 1
                if self.strict_join:
                    raise KeyError(
                        f"Image '{p}' slide_id '{slide_id}' not found in CSV index '{self.id_field}'."
                    )
                continue

            self.samples.append((p, idx))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images matched metadata join under {self.root}. "
                f"(Checked {len(all_files)} files; missed={missed})"
            )

        self._stats = TCGADatasetStats(
            num_images=len(self.samples),
            num_slides=len(self.slide_ids),
            num_classes=len(self.y_vocab),
            meta_fields=self.meta_fields,
            meta_cardinalities=tuple(len(self.meta_vocabs[f]) for f in self.meta_fields),
        )
    @property
    def stats(self) -> TCGADatasetStats:
        return self._stats

    def num_classes(self) -> int:
        return self._stats.num_classes

    def meta_dims(self) -> Tuple[int, ...]:
        return self._stats.meta_cardinalities

    # --- dataset API ---
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, slide_idx = self.samples[idx]

        img = default_loader(str(path))
        if self.transform is not None:
            img = self.transform(img)

        y = torch.tensor(self.slide_y[slide_idx], dtype=torch.long)

        if len(self.meta_fields) == 0 and self.return_meta_none_if_empty:
            meta = None
        else:
            meta = torch.tensor(self.slide_meta[slide_idx], dtype=torch.long)

        if self.return_ids:
            slide_id = self.slide_ids[slide_idx]
            return img, y, meta, slide_id, str(path)

        return img, y, meta

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}(\n"
            f"  root={str(self.root)!r},\n"
            f"  csv_path={str(self.csv_path)!r},\n"
            f"  id_field={self.id_field!r}, label_field={self.label_field!r},\n"
            f"  meta_fields={list(self.meta_fields)!r},\n"
            f"  num_images={self._stats.num_images}, num_slides={self._stats.num_slides},\n"
            f"  num_classes={self._stats.num_classes},\n"
            f"  meta_cardinalities={list(self._stats.meta_cardinalities)!r},\n"
            f")"
        )
        return s
