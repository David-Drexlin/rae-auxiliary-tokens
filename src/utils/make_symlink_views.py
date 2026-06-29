import argparse
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_groups(groups: List[str]) -> List[dict]:
    """
    Supported group specs:
      - "class"
      - "meta:<field>"            e.g. meta:race
      - "joint:class,<field>,..." e.g. joint:class,race,gender
    """
    out = []
    for g in groups:
        g = g.strip()
        if g == "class":
            out.append({"type": "class"})
        elif g.startswith("meta:"):
            field = g.split(":", 1)[1].strip()
            out.append({"type": "meta", "field": field})
        elif g.startswith("joint:"):
            fields = [x.strip() for x in g.split(":", 1)[1].split(",") if x.strip()]
            if not fields:
                raise ValueError(f"Bad joint group spec: {g}")
            out.append({"type": "joint", "fields": fields})
        else:
            raise ValueError(f"Unknown group spec: {g}")
    return out


def rel_symlink(src: Path, dst: Path) -> None:
    """
    Create relative symlink dst -> src.
    """
    if dst.exists() or dst.is_symlink():
        return
    ensure_dir(dst.parent)
    rel = os.path.relpath(str(src), start=str(dst.parent))
    os.symlink(rel, str(dst))


def main():
    ap = argparse.ArgumentParser("Create symlink views from a sampling manifest.")
    ap.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv (recommended) or manifest.jsonl")
    ap.add_argument("--images-dir", type=str, default=None, help="Directory that contains the PNGs (default: manifest parent)")
    ap.add_argument("--out-dir", type=str, required=True, help="Where to create symlink views (e.g. samples/views)")
    ap.add_argument(
        "--group",
        action="append",
        required=True,
        help="Grouping spec (repeatable). Examples: --group class --group meta:race --group joint:class,race",
    )
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False,
                    help="If true, deletes out-dir before creating.")
    ap.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    images_dir = Path(args.images_dir) if args.images_dir else manifest_path.parent
    out_dir = Path(args.out_dir)

    if args.overwrite and out_dir.exists():
        # safe delete of symlink tree
        import shutil
        shutil.rmtree(out_dir)

    ensure_dir(out_dir)

    # Load manifest
    if manifest_path.suffix.lower() == ".csv":
        df = pd.read_csv(manifest_path)
    else:
        # jsonl
        df = pd.read_json(manifest_path, lines=True)

    # Required columns
    if "filename" not in df.columns:
        raise ValueError("manifest must contain a 'filename' column.")
    if "y" not in df.columns:
        raise ValueError("manifest must contain a 'y' column (class id).")

    group_specs = parse_groups(args.group)

    # Prepare meta column naming: we expect meta_<field> columns if meta exists
    # Example: meta_race, meta_gender, meta_ajcc_pathologic_stage
    def meta_col(field: str) -> str:
        return f"meta_{field}"

    # Iterate rows
    n = len(df)
    for i, row in df.iterrows():
        fn = str(row["filename"])
        src = images_dir / fn
        if not src.exists():
            # allow missing (e.g. trimmed)
            continue

        y = int(row["y"])

        for spec in group_specs:
            if spec["type"] == "class":
                view = out_dir / "by_class" / f"y_{y:03d}" / fn

            elif spec["type"] == "meta":
                field = spec["field"]
                col = meta_col(field)
                if col not in df.columns:
                    raise ValueError(f"manifest missing column '{col}' for meta field '{field}'")
                v = int(row[col])
                view = out_dir / f"by_{field}" / f"{field}_{v:03d}" / fn

            else:
                # joint
                fields = spec["fields"]
                parts = []
                for f in fields:
                    if f == "class":
                        parts.append(f"y_{y:03d}")
                    else:
                        col = meta_col(f)
                        if col not in df.columns:
                            raise ValueError(f"manifest missing column '{col}' for joint field '{f}'")
                        v = int(row[col])
                        parts.append(f"{f}_{v:03d}")
                view = out_dir / ("by_" + "_".join(fields)) / Path(*parts) / fn

            if args.dry_run:
                continue
            rel_symlink(src, view)

        if (i + 1) % 5000 == 0:
            print(f"[views] processed {i+1}/{n}")

    print(f"[views] done -> {out_dir}")


if __name__ == "__main__":
    main()