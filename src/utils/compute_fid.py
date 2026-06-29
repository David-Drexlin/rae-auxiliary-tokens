#!/usr/bin/env python3
"""
Compute FID between:
- two folders of images (recursively), OR
- two .npz files containing uint8 images under a given key (default: "images") with shape (N,H,W,C)

Uses torch-fidelity's Inception-based FID.

Examples:
  # folders
  python compute_fid.py --input1 /path/real_pngs --input2 /path/recon_pngs --cuda --batch-size 32

  # npz (expects npz["images"] as uint8 NHWC)
  python compute_fid.py --input1 real.npz --input2 recon.npz --npz-key images --cuda --batch-size 32
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset


class NpzImagesDataset(Dataset):
    """
    torch-fidelity accepts an instance of torch.utils.data.Dataset as input.
    We stream samples from an NPZ via memory mapping to avoid loading everything into RAM.

    Expected array:
      - dtype: uint8 (recommended)
      - shape: (N, H, W, C)  where C is 1 or 3 (for histology, typically 3)
    """

    def __init__(self, npz_path: Union[str, Path], key: str = "images"):
        self.npz_path = str(Path(npz_path).expanduser())
        self.key = key

        # mmap_mode='r' => do not load full array into memory
        self._npz = np.load(self.npz_path, mmap_mode="r", allow_pickle=False)

        if key not in self._npz:
            raise KeyError(
                f"Key '{key}' not found in {self.npz_path}. Available keys: {list(self._npz.keys())}"
            )

        self.arr = self._npz[key]
        if self.arr.ndim != 4:
            raise ValueError(f"Expected 4D array (N,H,W,C). Got shape {self.arr.shape} in {self.npz_path}")
        if self.arr.dtype != np.uint8:
            # It *can* work with other dtypes, but uint8 is the intended format.
            # Keeping it strict avoids accidental 0..1 floats, etc.
            raise ValueError(
                f"Expected dtype uint8. Got {self.arr.dtype} in {self.npz_path}. "
                "Re-pack as uint8 NHWC to be safe."
            )

    def __len__(self) -> int:
        return int(self.arr.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.arr[idx]  # HWC uint8
        # Convert to CHW uint8 tensor (torch-fidelity handles uint8 images)
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        return x


def _as_torch_fidelity_input(path_str: str, npz_key: str) -> Union[str, Dataset]:
    """
    torch-fidelity input descriptor "input" can be:
      - string path to a directory of images
      - torch.utils.data.Dataset
    """
    p = Path(path_str).expanduser()
    if p.suffix.lower() == ".npz":
        return NpzImagesDataset(p, key=npz_key)
    return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input1", type=str, required=True, help="Folder of images OR .npz for dataset A (e.g., real).")
    ap.add_argument("--input2", type=str, required=True, help="Folder of images OR .npz for dataset B (e.g., recon/gen).")
    ap.add_argument("--npz-key", type=str, default="images", help="Key inside .npz (default: images).")
    ap.add_argument("--cuda", action="store_true", help="Use CUDA if available.")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for feature extraction.")
    ap.add_argument("--verbose", action="store_true", help="More logs from torch-fidelity.")
    args = ap.parse_args()

    try:
        import torch_fidelity
    except ImportError as e:
        raise SystemExit(
            "torch-fidelity is not installed in this environment.\n"
            "Install it inside your container, e.g.:\n"
            "  pip install --no-cache-dir torch-fidelity\n"
        ) from e

    input1 = _as_torch_fidelity_input(args.input1, npz_key=args.npz_key)
    input2 = _as_torch_fidelity_input(args.input2, npz_key=args.npz_key)

    metrics = torch_fidelity.calculate_metrics(
        input1=input1,
        input2=input2,
        cuda=args.cuda,
        fid=True,
        isc=False,
        kid=False,
        prc=False,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    fid = metrics.get("frechet_inception_distance", metrics.get("fid", None))

    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    if fid is not None:
        print(f"\nFID: {fid}")


if __name__ == "__main__":
    main()
