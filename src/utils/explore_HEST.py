#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------
def _pick_latest_hest_csv(root: Path) -> Path:
    cands = sorted(root.glob("HEST_v*.csv"))
    if not cands:
        raise FileNotFoundError(f"No HEST_v*.csv found in {root}")
    return cands[-1]


def _safe_name(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", path.stem)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _size_mb(p: Optional[Path]) -> float:
    if p is None or (not p.exists()) or (not p.is_file()):
        return 0.0
    return float(p.stat().st_size) / (1024.0 * 1024.0)


def _to_str(p: Optional[Path]) -> str:
    return str(p) if p is not None else ""


def _try_import_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None


def _iter_files(dirs: Sequence[Path], patterns: Sequence[str], recursive: bool = True) -> Iterable[Path]:
    for d in dirs:
        if not d.exists():
            continue
        if recursive:
            for pat in patterns:
                yield from d.rglob(pat)
        else:
            for pat in patterns:
                yield from d.glob(pat)


# ----------------------------
# Metadata missingness + cardinality report
# ----------------------------
def _meta_report(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    if n == 0:
        raise ValueError("metadata table has 0 rows")

    miss_cnt = df.isna().sum(axis=0)
    miss_frac = miss_cnt / float(n)
    nun = df.nunique(dropna=True)
    dtypes = df.dtypes.astype(str)

    out = pd.DataFrame(
        {
            "dtype": dtypes,
            "n_rows": n,
            "missing_count": miss_cnt,
            "missing_frac": miss_frac,
            "nunique": nun,
        }
    )

    # numeric summary
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if num_cols:
        num = df[num_cols]
        num_stats = pd.DataFrame(
            {
                "mean": num.mean(axis=0, skipna=True).astype("float64"),
                "std": num.std(axis=0, skipna=True).astype("float64"),
                "min": num.min(axis=0, skipna=True).astype("float64"),
                "max": num.max(axis=0, skipna=True).astype("float64"),
            }
        )
        qs = num.quantile([0.05, 0.50, 0.95], axis=0, interpolation="linear")
        qs.index = ["q05", "q50", "q95"]
        out = out.join(num_stats, how="left")
        out = out.join(qs.T, how="left")

    # top values for non-numeric
    top1_value: List[str] = []
    top1_count: List[float] = []
    top1_frac: List[float] = []
    top3: List[str] = []

    for c in out.index:
        if c in num_cols:
            top1_value.append("")
            top1_count.append(np.nan)
            top1_frac.append(np.nan)
            top3.append("")
            continue
        s = df[c]
        vc = s.value_counts(dropna=True)
        if vc.shape[0] == 0:
            top1_value.append("")
            top1_count.append(0.0)
            top1_frac.append(0.0)
            top3.append("")
        else:
            v1 = str(vc.index[0])
            c1 = float(vc.iloc[0])
            top1_value.append(v1)
            top1_count.append(c1)
            top1_frac.append(c1 / float(n))
            pairs = [f"{str(vc.index[i])}:{int(vc.iloc[i])}" for i in range(min(3, vc.shape[0]))]
            top3.append(" | ".join(pairs))

    out["top1_value"] = top1_value
    out["top1_count"] = top1_count
    out["top1_frac"] = top1_frac
    out["top3"] = top3

    # candidate heuristics
    is_obj = out["dtype"].astype(str).str.contains("object|string|category", regex=True, na=False)
    out["is_categorical_candidate"] = (
        is_obj
        & (out["missing_frac"] <= 0.30)
        & (out["nunique"] >= 2)
        & (out["nunique"] <= 200)
    )
    out["is_numeric_candidate"] = (
        out["dtype"].astype(str).str.contains("int|float|bool", regex=True, na=False)
        & (out["missing_frac"] <= 0.30)
        & (out["nunique"] >= 5)
    )

    out = out.reset_index().rename(columns={"index": "column"})
    out = out.sort_values(["missing_frac", "nunique"], ascending=[True, True], kind="stable")
    return out


def _write_meta_reports(
    df: pd.DataFrame,
    out_dir: Path,
    tag: str,
    core_cols: Optional[List[str]] = None,
) -> Dict[str, str]:
    _ensure_dir(out_dir)

    rep_all = _meta_report(df)
    p_all = out_dir / f"{tag}__meta_report__all_columns.csv"
    rep_all.to_csv(p_all, index=False)

    p_core = None
    if core_cols is not None:
        keep = [c for c in core_cols if c in df.columns]
        if keep:
            rep_core = _meta_report(df[keep].copy())
            p_core = out_dir / f"{tag}__meta_report__core_columns.csv"
            rep_core.to_csv(p_core, index=False)

    summary = {
        "tag": tag,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "all_columns_report": str(p_all),
        "core_columns_report": str(p_core) if p_core else None,
        "categorical_candidates_all": rep_all.loc[rep_all["is_categorical_candidate"], "column"].head(100).tolist(),
        "numeric_candidates_all": rep_all.loc[rep_all["is_numeric_candidate"], "column"].head(100).tolist(),
    }
    p_sum = out_dir / f"{tag}__meta_report__summary.json"
    p_sum.write_text(json.dumps(summary, indent=2))

    return {
        "csv_all": str(p_all),
        "csv_core": str(p_core) if p_core else "",
        "summary_json": str(p_sum),
    }


# ----------------------------
# Asset manifest (fast)
# ----------------------------
@dataclass
class ScanConfig:
    wsis_dirs: List[Path]
    st_dirs: List[Path]
    patches_dirs: List[Path]
    transcripts_dirs: List[Path]
    tissue_seg_dirs: List[Path]


def _default_scan_dirs(root: Path) -> ScanConfig:
    # Typical HEST layout (top-level dirs)
    return ScanConfig(
        wsis_dirs=[root / "wsis"],
        st_dirs=[root / "st"],
        patches_dirs=[root / "patches"],
        transcripts_dirs=[root / "transcripts"],
        tissue_seg_dirs=[root / "tissue_seg"],
    )


def _build_id_set(index_df: pd.DataFrame) -> set[str]:
    if "id" not in index_df.columns:
        raise ValueError("index CSV must contain column 'id'")
    ids = set(index_df["id"].astype(str).tolist())
    # remove empty-ish
    ids.discard("")
    ids.discard("nan")
    return ids


def _collect_one_per_id_by_stem(
    dirs: Sequence[Path],
    patterns: Sequence[str],
    ids: set[str],
    recursive: bool = True,
) -> Tuple[Dict[str, Path], Dict[str, int]]:
    """
    Maps id -> first matching path where path.stem == id.
    Also returns counts[id] = number of matches observed.
    """
    first: Dict[str, Path] = {}
    counts: Dict[str, int] = {}

    tqdm = _try_import_tqdm()
    files = list(_iter_files(dirs, patterns, recursive=recursive))
    it = tqdm(files, desc="scan(stem)", unit="file") if tqdm else files

    for p in it:
        if not p.is_file():
            continue
        stem = p.stem
        if stem not in ids:
            continue
        counts[stem] = counts.get(stem, 0) + 1
        if stem not in first:
            first[stem] = p
    return first, counts


def _collect_one_per_id_by_prefix(
    dirs: Sequence[Path],
    patterns: Sequence[str],
    ids: set[str],
    recursive: bool = True,
) -> Tuple[Dict[str, Path], Dict[str, int]]:
    """
    Maps id -> first matching path where id == stem.split('_')[0].
    Also returns counts[id].
    Useful for things like TENX149_*.h5, TENX149_*.parquet, TENX149_mask.jpg, ...
    """
    first: Dict[str, Path] = {}
    counts: Dict[str, int] = {}

    tqdm = _try_import_tqdm()
    files = list(_iter_files(dirs, patterns, recursive=recursive))
    it = tqdm(files, desc="scan(prefix)", unit="file") if tqdm else files

    for p in it:
        if not p.is_file():
            continue
        stem = p.stem
        pref = stem.split("_", 1)[0]
        if pref not in ids:
            continue
        counts[pref] = counts.get(pref, 0) + 1
        if pref not in first:
            first[pref] = p
    return first, counts


def build_asset_manifest(
    root: Path,
    index_df: pd.DataFrame,
    out_dir: Path,
    limit_ids: Optional[int] = None,
    recursive_scan: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fast approach:
      1) scan each modality dir ONCE
      2) map id -> first file (+count)
      3) build manifest table by joining on ids
    """
    ids = sorted(_build_id_set(index_df))
    if limit_ids is not None:
        ids = ids[: int(limit_ids)]

    scan = _default_scan_dirs(root)

    # Make it explicit what we will scan
    scan_info = {
        "wsis_dirs": [str(p) for p in scan.wsis_dirs if p.exists()],
        "st_dirs": [str(p) for p in scan.st_dirs if p.exists()],
        "patches_dirs": [str(p) for p in scan.patches_dirs if p.exists()],
        "transcripts_dirs": [str(p) for p in scan.transcripts_dirs if p.exists()],
        "tissue_seg_dirs": [str(p) for p in scan.tissue_seg_dirs if p.exists()],
    }

    ids_set = set(ids)

    # modality scans
    wsi_map, wsi_cnt = _collect_one_per_id_by_stem(scan.wsis_dirs, ["*.tif", "*.tiff"], ids_set, recursive=recursive_scan)
    st_map, st_cnt = _collect_one_per_id_by_stem(scan.st_dirs, ["*.h5ad"], ids_set, recursive=recursive_scan)
    patches_map, patches_cnt = _collect_one_per_id_by_prefix(scan.patches_dirs, ["*.h5"], ids_set, recursive=recursive_scan)
    transcripts_map, transcripts_cnt = _collect_one_per_id_by_prefix(scan.transcripts_dirs, ["*.parquet"], ids_set, recursive=recursive_scan)

    # tissue masks: often {id}_mask.jpg / {id}_mask.pkl
    mask_jpg_map, mask_jpg_cnt = _collect_one_per_id_by_prefix(scan.tissue_seg_dirs, ["*_mask.jpg", "*_mask.jpeg", "*_mask.png"], ids_set, recursive=recursive_scan)
    mask_pkl_map, mask_pkl_cnt = _collect_one_per_id_by_prefix(scan.tissue_seg_dirs, ["*_mask.pkl"], ids_set, recursive=recursive_scan)

    rows: List[Dict[str, Any]] = []
    for id_ in ids:
        wsi = wsi_map.get(id_)
        st = st_map.get(id_)
        patches = patches_map.get(id_)
        tr = transcripts_map.get(id_)
        mj = mask_jpg_map.get(id_)
        mp = mask_pkl_map.get(id_)

        r: Dict[str, Any] = {"id": id_}

        r["wsi_path"] = _to_str(wsi)
        r["st_path"] = _to_str(st)
        r["patches_path"] = _to_str(patches)
        r["transcripts_path"] = _to_str(tr)
        r["tissue_mask_jpg_path"] = _to_str(mj)
        r["tissue_mask_pkl_path"] = _to_str(mp)

        r["has_wsi"] = wsi is not None
        r["has_st"] = st is not None
        r["has_patches"] = patches is not None
        r["has_transcripts"] = tr is not None
        r["has_tissue_mask_jpg"] = mj is not None
        r["has_tissue_mask_pkl"] = mp is not None

        r["wsi_count"] = int(wsi_cnt.get(id_, 0))
        r["st_count"] = int(st_cnt.get(id_, 0))
        r["patches_count"] = int(patches_cnt.get(id_, 0))
        r["transcripts_count"] = int(transcripts_cnt.get(id_, 0))
        r["tissue_mask_jpg_count"] = int(mask_jpg_cnt.get(id_, 0))
        r["tissue_mask_pkl_count"] = int(mask_pkl_cnt.get(id_, 0))

        r["wsi_mb"] = _size_mb(wsi)
        r["st_mb"] = _size_mb(st)
        r["patches_mb"] = _size_mb(patches)
        r["transcripts_mb"] = _size_mb(tr)

        rows.append(r)

    man = pd.DataFrame(rows)

    summary = {
        "root": str(root),
        "scan_info": scan_info,
        "n_ids_requested": int(len(ids)),
        "availability_frac": {
            "has_wsi": float(man["has_wsi"].mean()) if len(man) else 0.0,
            "has_st": float(man["has_st"].mean()) if len(man) else 0.0,
            "has_patches": float(man["has_patches"].mean()) if len(man) else 0.0,
            "has_transcripts": float(man["has_transcripts"].mean()) if len(man) else 0.0,
            "has_tissue_mask_jpg": float(man["has_tissue_mask_jpg"].mean()) if len(man) else 0.0,
            "has_tissue_mask_pkl": float(man["has_tissue_mask_pkl"].mean()) if len(man) else 0.0,
        },
        "duplicates": {
            "wsi_ids_with_multiple_matches": int((man["wsi_count"] > 1).sum()),
            "st_ids_with_multiple_matches": int((man["st_count"] > 1).sum()),
            "patches_ids_with_multiple_matches": int((man["patches_count"] > 1).sum()),
            "transcripts_ids_with_multiple_matches": int((man["transcripts_count"] > 1).sum()),
            "mask_jpg_ids_with_multiple_matches": int((man["tissue_mask_jpg_count"] > 1).sum()),
            "mask_pkl_ids_with_multiple_matches": int((man["tissue_mask_pkl_count"] > 1).sum()),
        },
        "sizes_gb_total": {
            "wsi_total_gb": float(man["wsi_mb"].sum() / 1024.0),
            "st_total_gb": float(man["st_mb"].sum() / 1024.0),
            "patches_total_gb": float(man["patches_mb"].sum() / 1024.0),
            "transcripts_total_gb": float(man["transcripts_mb"].sum() / 1024.0),
        },
    }

    _ensure_dir(out_dir)
    p_man = out_dir / "hest_asset_manifest.csv"
    man.to_csv(p_man, index=False)

    p_sum = out_dir / "hest_asset_manifest_summary.json"
    p_sum.write_text(json.dumps(summary, indent=2))

    return man, summary


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="HEST root, e.g. DATASETS_ROOT_PLACEHOLDER/hest1k")
    ap.add_argument("--index_csv", type=str, default=None, help="Optional path to HEST_v*.csv (default: latest in root)")
    ap.add_argument(
        "--case_metadata_csv",
        type=str,
        default=None,
        help="Optional path to metadata_total/case_metadata.csv (default: root/metadata_total/case_metadata.csv if exists)",
    )
    ap.add_argument("--out_dir", type=str, default="hest_reports", help="Output directory")
    ap.add_argument("--limit_ids", type=int, default=None, help="Limit IDs for manifest scan (debug)")
    ap.add_argument("--no_manifest", action="store_true", help="Skip asset manifest scan")
    ap.add_argument(
        "--recursive_scan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan subdirectories recursively under wsis/st/patches/transcripts/tissue_seg (default: true)",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    index_csv = Path(args.index_csv) if args.index_csv else _pick_latest_hest_csv(root)
    index_df = pd.read_csv(index_csv, low_memory=False)

    # case metadata optional
    case_csv = Path(args.case_metadata_csv) if args.case_metadata_csv else (root / "metadata_total" / "case_metadata.csv")
    case_df = None
    if case_csv.exists():
        case_df = pd.read_csv(case_csv, low_memory=False)

    core_cols = list(index_df.columns)

    # meta reports
    meta_outputs: Dict[str, Any] = {}
    meta_outputs["index"] = _write_meta_reports(index_df, out_dir, tag=_safe_name(index_csv), core_cols=core_cols)
    if case_df is not None:
        meta_outputs["case_metadata"] = _write_meta_reports(case_df, out_dir, tag=_safe_name(case_csv), core_cols=core_cols)
    else:
        meta_outputs["case_metadata"] = None

    manifest_outputs = None
    if not args.no_manifest:
        man, man_summary = build_asset_manifest(
            root=root,
            index_df=index_df,
            out_dir=out_dir,
            limit_ids=args.limit_ids,
            recursive_scan=bool(args.recursive_scan),
        )
        manifest_outputs = {
            "manifest_csv": str(out_dir / "hest_asset_manifest.csv"),
            "manifest_summary_json": str(out_dir / "hest_asset_manifest_summary.json"),
            "summary": man_summary,
        }

    global_summary = {
        "root": str(root),
        "index_csv": str(index_csv),
        "case_metadata_csv": str(case_csv) if case_csv.exists() else None,
        "outputs": {
            "meta_reports": meta_outputs,
            "asset_manifest": manifest_outputs,
        },
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(global_summary, indent=2))
    print(json.dumps(global_summary, indent=2))


if __name__ == "__main__":
    main()