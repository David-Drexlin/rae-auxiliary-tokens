# RAE_ROOT_PLACEHOLDER/src/utils/condition_sampler.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

# Keep conventions consistent with utils/tcga_dataset.py
MISSING = "__MISSING__"
UNK = "__UNK__"
CFG_NULL = "__CFG_NULL__"


def _clean(x) -> str:
    """
    Mirror tcga_dataset._clean() semantics for stable categorical tokens.
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


def count_images_per_slide(
    image_root: Union[str, Path],
    slide_sep: str = "__",
    exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".webp"),
) -> Dict[str, int]:
    """
    Count patch images per slide in an ImageFolder-like directory.

    Assumes filename convention:
      <slide_submitter_id>__<anything>.<ext>
    """
    image_root = Path(image_root)
    exts = tuple(e.lower() for e in exts)

    counts: Dict[str, int] = {}
    for p in image_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        slide_id = p.name.split(slide_sep, 1)[0]
        slide_id = _clean(slide_id)
        if slide_id == MISSING:
            continue
        counts[slide_id] = counts.get(slide_id, 0) + 1
    return counts


def _build_vocab_inverse(vocab: Dict[str, int]) -> Dict[int, str]:
    inv: Dict[int, str] = {}
    for k, v in vocab.items():
        inv[int(v)] = str(k)
    return inv


def _valid_id_list(
    vocab: Dict[str, int],
    *,
    skip_unk: bool = True,
    skip_cfg_null: bool = True,
) -> List[int]:
    """
    Return a list of valid IDs to use for conditioning from a vocab mapping token->id.
    Conventions:
      - UNK token id should exist (often 0 or 1 depending on vocab type)
      - CFG_NULL token id should exist for meta vocabs (0)
    """
    ids = sorted(set(int(i) for i in vocab.values()))
    if skip_unk and UNK in vocab:
        unk_id = int(vocab[UNK])
        ids = [i for i in ids if i != unk_id]
    if skip_cfg_null and CFG_NULL in vocab:
        null_id = int(vocab[CFG_NULL])
        ids = [i for i in ids if i != null_id]
    return ids


@dataclass
class ConditionSpec:
    """
    Holds the *encoded* unique tuples and optional weights.

    tuples_y:     (T,) int64
    tuples_meta:  (T,F) int64 or None
    weights:      (T,) float64  (used by 'actual' sampling)
    """
    tuples_y: np.ndarray
    tuples_meta: Optional[np.ndarray]
    weights: np.ndarray
    meta_fields: List[str]
    num_classes: int
    null_label: int
    y_vocab: Dict[str, int]
    meta_vocabs: Dict[str, Dict[str, int]]


def build_condition_spec_from_tcga_csv(
    *,
    csv_path: Union[str, Path],
    image_root: Optional[Union[str, Path]],
    slide_sep: str,
    id_field: str,
    label_field: str,
    meta_fields: Sequence[str],
    y_vocab: Dict[str, int],
    meta_vocabs: Dict[str, Dict[str, int]],
    count_level: str = "image",  # "image" or "slide"
    skip_unk_y: bool = True,
    drop_rows_with_any_meta_unk: bool = False,
) -> ConditionSpec:
    """
    Build unique observed tuples and tuple weights from a TCGA metadata CSV.

    - Uses slide-level dedup (first occurrence per slide_submitter_id), consistent with TCGAPatchDataset.
    - For count_level='image', weights are #patch images per slide (computed from image_root).
    - For count_level='slide', weights are 1 per slide.
    - Encodes labels and meta using provided vocabs (NO reindexing).

    Returns ConditionSpec for joint tuple sampling.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"csv_path does not exist: {csv_path}")

    meta_fields = list(meta_fields)

    df = pd.read_csv(csv_path)
    if id_field not in df.columns:
        raise ValueError(f"CSV missing id_field '{id_field}': {csv_path}")
    if label_field not in df.columns:
        raise ValueError(f"CSV missing label_field '{label_field}': {csv_path}")
    for f in meta_fields:
        if f not in df.columns:
            raise ValueError(f"CSV missing meta field '{f}': {csv_path}")

    # clean
    df[id_field] = df[id_field].astype(str).map(_clean)
    df[label_field] = df[label_field].map(_clean)
    for f in meta_fields:
        df[f] = df[f].map(_clean).replace({MISSING: UNK})

    # filter usable rows
    df = df[df[id_field] != MISSING].copy()
    df = df[df[label_field] != MISSING].copy()
    if len(df) == 0:
        raise RuntimeError("After cleaning/filtering, 0 usable rows in CSV.")

    # slide-level dedup (same as dataset)
    df = df.drop_duplicates(id_field, keep="first").copy()

    # weights per slide
    if count_level == "image":
        if image_root is None:
            raise ValueError("count_level='image' requires image_root.")
        slide_counts = count_images_per_slide(image_root, slide_sep=slide_sep)
        df["_w"] = df[id_field].map(lambda sid: int(slide_counts.get(sid, 0))).astype(int)
        df = df[df["_w"] > 0].copy()
        if len(df) == 0:
            raise RuntimeError(
                "No slides from CSV matched any patch images under image_root. "
                "Check image_root and slide_sep."
            )
    elif count_level == "slide":
        df["_w"] = 1
    else:
        raise ValueError("count_level must be 'image' or 'slide'.")

    # encode y
    if UNK not in y_vocab:
        raise ValueError("y_vocab must contain UNK token.")
    y_unk_id = int(y_vocab[UNK])

    df["_y"] = df[label_field].map(lambda s: int(y_vocab.get(s, y_unk_id))).astype(int)
    if skip_unk_y:
        df = df[df["_y"] != y_unk_id].copy()

    # encode meta
    meta_cols_encoded: List[str] = []
    if meta_fields:
        for f in meta_fields:
            if f not in meta_vocabs:
                raise ValueError(f"meta_vocabs missing field '{f}'.")
            vocab = meta_vocabs[f]
            if UNK not in vocab:
                raise ValueError(f"meta_vocabs['{f}'] missing UNK token.")
            unk_id = int(vocab[UNK])
            col = f"_m_{f}"
            df[col] = df[f].map(lambda v: int(vocab.get(v, unk_id))).astype(int)
            meta_cols_encoded.append(col)

        if drop_rows_with_any_meta_unk:
            # for meta vocabs, UNK id is typically 1 (and CFG_NULL id is 0)
            unk_ids = [int(meta_vocabs[f][UNK]) for f in meta_fields]
            mask = np.ones(len(df), dtype=bool)
            for col, uid in zip(meta_cols_encoded, unk_ids):
                mask &= (df[col].to_numpy() != uid)
            df = df[mask].copy()

    if len(df) == 0:
        raise RuntimeError("After encoding/filtering, 0 usable rows remain.")

    # group by tuple -> sum weights
    if meta_fields:
        key_cols = ["_y"] + meta_cols_encoded
        grp = df.groupby(key_cols)["_w"].sum().reset_index()
        tuples_y = grp["_y"].to_numpy(np.int64)
        tuples_meta = grp[meta_cols_encoded].to_numpy(np.int64)
        weights = grp["_w"].to_numpy(np.float64)
    else:
        grp = df.groupby(["_y"])["_w"].sum().reset_index()
        tuples_y = grp["_y"].to_numpy(np.int64)
        tuples_meta = None
        weights = grp["_w"].to_numpy(np.float64)

    num_classes = int(max(tuples_y.max(initial=0) + 1, len(set(y_vocab.values()))))
    null_label = int(max(num_classes, num_classes))  # default: caller can override

    return ConditionSpec(
        tuples_y=tuples_y,
        tuples_meta=tuples_meta,
        weights=weights,
        meta_fields=meta_fields,
        num_classes=num_classes,
        null_label=null_label,
        y_vocab=y_vocab,
        meta_vocabs=meta_vocabs,
    )


def build_cartesian_condition_spec(
    *,
    y_vocab: Dict[str, int],
    meta_vocabs: Dict[str, Dict[str, int]],
    meta_fields: Sequence[str],
    num_classes: int,
    null_label: int,
    skip_unk_y: bool = True,
    skip_unk_meta: bool = True,
    skip_cfg_null_meta: bool = True,
    max_tuples: Optional[int] = None,
    seed: int = 0,
) -> ConditionSpec:
    """
    Build a ConditionSpec over the full Cartesian product:
      y × meta_1 × ... × meta_F

    Guardrails:
      - skip_unk_* removes UNK ids from each field
      - skip_cfg_null_meta removes CFG_NULL=0 from meta fields
      - max_tuples caps total tuples (random subset if needed)

    Weights are uniform.
    """
    rng = np.random.default_rng(seed)
    meta_fields = list(meta_fields)

    if UNK not in y_vocab:
        raise ValueError("y_vocab must contain UNK token.")

    # valid y ids
    y_ids = sorted(set(int(v) for v in y_vocab.values()))
    if skip_unk_y and UNK in y_vocab:
        y_ids = [i for i in y_ids if i != int(y_vocab[UNK])]
    # do not include null_label as a "class"
    y_ids = [i for i in y_ids if i != int(null_label)]
    if len(y_ids) == 0:
        raise RuntimeError("No valid y ids remain after filtering (check vocabs and flags).")

    # valid meta ids per field
    meta_id_lists: List[List[int]] = []
    for f in meta_fields:
        if f not in meta_vocabs:
            raise ValueError(f"meta_vocabs missing field '{f}'")
        ids = _valid_id_list(
            meta_vocabs[f],
            skip_unk=skip_unk_meta,
            skip_cfg_null=skip_cfg_null_meta,
        )
        if len(ids) == 0:
            raise RuntimeError(f"No valid ids remain for meta field '{f}' after filtering.")
        meta_id_lists.append(ids)

    # cartesian sizes
    sizes = [len(y_ids)] + [len(ids) for ids in meta_id_lists]
    total = int(np.prod(sizes))
    if total <= 0:
        raise RuntimeError("Cartesian product size is zero.")

    # If no meta fields, cartesian reduces to y only.
    if len(meta_fields) == 0:
        tuples_y = np.array(y_ids, dtype=np.int64)
        tuples_meta = None
        weights = np.ones_like(tuples_y, dtype=np.float64)
        if max_tuples is not None and len(tuples_y) > int(max_tuples):
            idx = rng.choice(len(tuples_y), size=int(max_tuples), replace=False)
            tuples_y = tuples_y[idx]
            weights = np.ones_like(tuples_y, dtype=np.float64)
        return ConditionSpec(
            tuples_y=tuples_y,
            tuples_meta=tuples_meta,
            weights=weights,
            meta_fields=[],
            num_classes=num_classes,
            null_label=null_label,
            y_vocab=y_vocab,
            meta_vocabs=meta_vocabs,
        )

    # Create tuples as an index space without materializing full product if huge:
    # We'll sample a subset if max_tuples is set.
    if max_tuples is not None and total > int(max_tuples):
        K = int(max_tuples)
        # sample K random indices in [0,total)
        flat_idx = rng.integers(0, total, size=K, endpoint=False)
    else:
        flat_idx = np.arange(total, dtype=np.int64)

    # decode flat indices into coordinates
    # coord order: y, m1, m2, ...
    strides = []
    acc = 1
    for s in reversed(sizes[1:]):
        strides.append(acc)
        acc *= s
    strides = list(reversed(strides))  # length F, for meta dims
    # For y stride: product of all meta sizes
    y_stride = int(np.prod(sizes[1:]))

    y_coords = (flat_idx // y_stride).astype(np.int64)
    rem = (flat_idx % y_stride).astype(np.int64)

    # map coords to ids
    tuples_y = np.array([y_ids[i] for i in y_coords.tolist()], dtype=np.int64)

    tuples_meta = np.empty((len(flat_idx), len(meta_fields)), dtype=np.int64)
    for j, ids in enumerate(meta_id_lists):
        # stride for meta j
        stride = int(np.prod(sizes[j + 2 :])) if (j + 2) <= len(sizes) - 1 else 1
        coord = (rem // stride).astype(np.int64)
        rem = (rem % stride).astype(np.int64)
        tuples_meta[:, j] = np.array([ids[i] for i in coord.tolist()], dtype=np.int64)

    weights = np.ones((len(flat_idx),), dtype=np.float64)

    return ConditionSpec(
        tuples_y=tuples_y,
        tuples_meta=tuples_meta,
        weights=weights,
        meta_fields=meta_fields,
        num_classes=num_classes,
        null_label=null_label,
        y_vocab=y_vocab,
        meta_vocabs=meta_vocabs,
    )


@dataclass
class ConditionPool:
    """
    A global pool of conditions, sharded by rank for DDP sampling.

    y:    (total_samples,) int64
    meta: (total_samples,F) int64 or None
    """
    y: np.ndarray
    meta: Optional[np.ndarray]


def build_condition_pool(
    spec: ConditionSpec,
    *,
    mode: str,  # "actual" | "uniform_observed" | "cartesian_uniform"
    total_samples: int,
    seed: int,
) -> ConditionPool:
    """
    Build a global pool of length total_samples.
    - actual: sample tuples proportional to weights
    - uniform_observed: sample tuples uniformly from observed tuples
    - cartesian_uniform: requires spec already built from cartesian product; sampled uniformly
    """
    rng = np.random.default_rng(seed)

    T = int(spec.tuples_y.shape[0])
    if T <= 0:
        raise RuntimeError("ConditionSpec contains 0 tuples.")

    if mode == "actual":
        w = spec.weights.astype(np.float64)
        if np.any(w < 0):
            raise ValueError("weights contain negative values.")
        s = float(w.sum())
        if s <= 0:
            raise ValueError("weights sum to 0.")
        p = w / s
        idx = rng.choice(T, size=int(total_samples), replace=True, p=p)

    elif mode == "uniform_observed":
        idx = rng.integers(0, T, size=int(total_samples), endpoint=False)

    elif mode == "cartesian_uniform":
        # spec tuples are already cartesian; uniform sampling is same as uniform over tuples
        idx = rng.integers(0, T, size=int(total_samples), endpoint=False)

    else:
        raise ValueError("mode must be one of: actual | uniform_observed | cartesian_uniform")

    y = spec.tuples_y[idx].astype(np.int64)
    if spec.tuples_meta is None:
        return ConditionPool(y=y, meta=None)
    meta = spec.tuples_meta[idx].astype(np.int64)
    return ConditionPool(y=y, meta=meta)


@dataclass
class RankConditionView:
    """
    Rank-local view reshaped for stepping.
    y:    (iterations, batch_size) LongTensor
    meta: (iterations, batch_size, F) LongTensor or None
    """
    y: torch.Tensor
    meta: Optional[torch.Tensor]


def shard_pool_for_rank(
    pool: ConditionPool,
    *,
    rank: int,
    world_size: int,
    per_rank_samples: int,
    batch_size: int,
) -> RankConditionView:
    """
    Take a contiguous shard [rank * per_rank_samples : (rank+1)*per_rank_samples]
    and reshape to (iterations, batch_size).
    """
    start = int(rank * per_rank_samples)
    end = int(start + per_rank_samples)

    y = pool.y[start:end]
    if y.shape[0] != per_rank_samples:
        raise RuntimeError("Pool too small for requested per-rank slice.")
    y_t = torch.from_numpy(y).view(-1, batch_size).long()

    if pool.meta is None:
        return RankConditionView(y=y_t, meta=None)

    meta = pool.meta[start:end]
    if meta.shape[0] != per_rank_samples:
        raise RuntimeError("Pool meta too small for requested per-rank slice.")
    meta_t = torch.from_numpy(meta).view(-1, batch_size, meta.shape[1]).long()
    return RankConditionView(y=y_t, meta=meta_t)


def compute_total_samples_rounded(
    num_requested: int,
    world_size: int,
    per_proc_batch_size: int,
) -> int:
    """
    Match your sampling script convention: round up to multiple of (world_size * batch_size).
    """
    global_bs = int(world_size * per_proc_batch_size)
    total = int(np.ceil(num_requested / global_bs) * global_bs)
    return total


def class_frequency_from_condition_spec(spec: ConditionSpec) -> Tuple[np.ndarray, np.ndarray]:
    """
    From a spec built on observed tuples, produce class-level frequencies.
    Returns (class_ids, probs), where probs sum to 1.
    """
    y = spec.tuples_y
    w = spec.weights.astype(np.float64)
    # aggregate weights by y
    uniq, inv = np.unique(y, return_inverse=True)
    agg = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(agg, inv, w)
    agg_sum = agg.sum()
    if agg_sum <= 0:
        raise RuntimeError("Class frequency weights sum to 0.")
    probs = agg / agg_sum
    return uniq.astype(np.int64), probs


def build_class_only_pool(
    *,
    num_classes: int,
    total_samples: int,
    seed: int,
    mode: str,  # "random" | "actual"
    class_ids: Optional[np.ndarray] = None,
    class_probs: Optional[np.ndarray] = None,
    skip_ids: Optional[Sequence[int]] = None,
) -> ConditionPool:
    """
    Build a class-only condition pool.
    - random: uniform over classes (optionally excluding skip_ids)
    - actual: sample from provided (class_ids, class_probs)

    Returns ConditionPool(y, meta=None).
    """
    rng = np.random.default_rng(seed)
    skip_ids = set(int(x) for x in (skip_ids or []))

    if mode == "random":
        valid = [i for i in range(int(num_classes)) if i not in skip_ids]
        if len(valid) == 0:
            raise RuntimeError("No valid class ids remain after skipping.")
        y = rng.choice(np.array(valid, dtype=np.int64), size=int(total_samples), replace=True)
        return ConditionPool(y=y.astype(np.int64), meta=None)

    if mode == "actual":
        if class_ids is None or class_probs is None:
            raise ValueError("actual class-only sampling requires class_ids and class_probs.")
        class_ids = class_ids.astype(np.int64)
        class_probs = class_probs.astype(np.float64)
        if class_probs.sum() <= 0:
            raise ValueError("class_probs sum to 0.")
        class_probs = class_probs / class_probs.sum()
        y = rng.choice(class_ids, size=int(total_samples), replace=True, p=class_probs)
        return ConditionPool(y=y.astype(np.int64), meta=None)

    raise ValueError("mode must be random | actual")