# RAE_ROOT_PLACEHOLDER/src/utils/hest_h5_dataset.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import h5py
from PIL import Image
from torch.utils.data import Dataset


# ----------------------------
# Helpers: discover image tensor inside H5
# ----------------------------
def _iter_datasets(h5obj, prefix=""):
    # recursively yield (path, dataset)
    for k in h5obj.keys():
        v = h5obj[k]
        p = f"{prefix}/{k}" if prefix else k
        if isinstance(v, h5py.Dataset):
            yield p, v
        elif isinstance(v, h5py.Group):
            yield from _iter_datasets(v, prefix=p)


def _is_image_like(ds: h5py.Dataset) -> bool:
    if ds.ndim < 3 or ds.ndim > 4:
        return False
    if ds.size == 0:
        return False
    # accept uint8 or float/uint16 (will convert)
    if str(ds.dtype) not in ("uint8", "uint16", "float32", "float16", "float64"):
        return False

    shape = ds.shape
    # common: (N,H,W,3) or (N,3,H,W) or (H,W,3)
    if ds.ndim == 4:
        n, a, b, c = shape
        if c == 3 and a >= 16 and b >= 16:
            return True  # NHWC
        if a == 3 and b >= 16 and c >= 16:
            return True  # NCHW
        return False
    else:
        a, b, c = shape
        return (c == 3 and a >= 16 and b >= 16)


def _pick_image_dataset(h5f: h5py.File) -> str:
    """
    Heuristic:
      - prefer 4D datasets (many patches) over single 3D images
      - prefer uint8
      - prefer dataset names containing 'img', 'image', 'patch'
    Returns dataset path (key).
    """
    candidates: List[Tuple[int, int, int, str]] = []
    # score tuple: (ndim_priority, dtype_priority, name_priority, path)
    for path, ds in _iter_datasets(h5f):
        if not _is_image_like(ds):
            continue

        ndim_priority = 2 if ds.ndim == 4 else 1  # prefer batched patches
        dtype_priority = 2 if ds.dtype == np.uint8 else 1
        name_l = path.lower()
        name_priority = 0
        for tok in ("patch", "image", "img", "tile", "rgb"):
            if tok in name_l:
                name_priority += 1

        candidates.append((ndim_priority, dtype_priority, name_priority, path))

    if not candidates:
        raise KeyError("Could not find an image-like dataset in this H5 file.")

    candidates.sort(reverse=True)
    return candidates[0][3]


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Accepts:
      - NHWC uint8
      - NCHW uint8
      - float [0,1] or [0,255]
    Returns NHWC uint8.
    """
    if arr.ndim == 3:
        # HWC
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected HWC with 3 channels, got shape={arr.shape}")
        out = arr
    elif arr.ndim == 4:
        # either NHWC or NCHW
        if arr.shape[-1] == 3:
            out = arr
        elif arr.shape[1] == 3:
            out = np.transpose(arr, (0, 2, 3, 1))
        else:
            raise ValueError(f"Expected NHWC or NCHW with 3 channels, got shape={arr.shape}")
    else:
        raise ValueError(f"Unexpected ndim={arr.ndim} for image array")

    if out.dtype == np.uint8:
        return out

    out_f = out.astype(np.float32)
    # guess range
    mx = float(np.nanmax(out_f)) if out_f.size else 1.0
    if mx <= 1.5:
        out_f = out_f * 255.0
    out_f = np.clip(out_f, 0.0, 255.0)
    return out_f.astype(np.uint8)


# ----------------------------
# Dataset
# ----------------------------
@dataclass
class H5Spec:
    path: Path
    key: str
    length: int


class HESTPatchH5Dataset(Dataset):
    """
    Fast-ish HEST patch loader for *.h5 patch banks.

    - scans ROOT/patches/**/*.h5 by default
    - per-worker lazy-open file handles
    - block cache to amortize HDF5 slice overhead

    Returns: (PIL.Image, global_index) if return_index=True else PIL.Image
    """

    def __init__(
        self,
        root: Union[str, Path],
        h5_paths: Optional[List[Union[str, Path]]] = None,
        ids: Optional[List[str]] = None,
        patches_subdir: str = "patches",
        transform=None,
        return_index: bool = True,
        block_size: int = 256,
        cache_blocks: int = 2,
    ):
        self.root = Path(root)
        self.transform = transform
        self.return_index = bool(return_index)

        self.block_size = int(block_size)
        self.cache_blocks = int(cache_blocks)

        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.cache_blocks < 0:
            raise ValueError("cache_blocks must be >= 0")

        patch_root = self.root / patches_subdir
        if patch_root.exists():
            scan_root = patch_root
        else:
            scan_root = self.root

        if h5_paths is not None and len(h5_paths) > 0:
            paths = [Path(p) for p in h5_paths]
        else:
            if ids is not None and len(ids) > 0:
                # try to find {id}.h5 anywhere under scan_root
                wanted = set(map(str, ids))
                paths = []
                for p in scan_root.rglob("*.h5"):
                    if p.stem in wanted or p.name.startswith(tuple(wanted)):
                        paths.append(p)
            else:
                paths = list(scan_root.rglob("*.h5"))

        paths = sorted({p.resolve() for p in paths if p.is_file()})
        if not paths:
            raise FileNotFoundError(f"No .h5 files found under: {scan_root}")

        self._specs: List[H5Spec] = []
        self._offsets: List[int] = [0]

        # Pre-scan each file ONCE to discover key + length.
        for p in paths:
            with h5py.File(p, "r") as h5f:
                key = _pick_image_dataset(h5f)
                ds = h5f[key]
                if ds.ndim == 4:
                    length = int(ds.shape[0])
                elif ds.ndim == 3:
                    length = 1
                else:
                    raise RuntimeError("Unexpected image dataset shape.")
            if length <= 0:
                continue
            self._specs.append(H5Spec(path=p, key=key, length=length))
            self._offsets.append(self._offsets[-1] + length)

        if not self._specs:
            raise RuntimeError("Found .h5 files but none contained usable image datasets.")

        self._n = self._offsets[-1]

        # per-process (i.e., per DataLoader worker) state
        self._handles: Dict[int, h5py.File] = {}
        # cache: list of (spec_idx, block_start, block_arr_uint8_NHWC)
        self._cache: List[Tuple[int, int, np.ndarray]] = []

    def __len__(self) -> int:
        return self._n

    def _find_spec(self, global_idx: int) -> Tuple[int, int]:
        # binary search in offsets
        # offsets: [0, len(file0), len(file0)+len(file1), ...]
        lo, hi = 0, len(self._offsets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._offsets[mid + 1] <= global_idx:
                lo = mid + 1
            else:
                hi = mid
        spec_idx = lo
        local_idx = global_idx - self._offsets[spec_idx]
        return spec_idx, int(local_idx)

    def _get_handle(self, spec_idx: int) -> h5py.File:
        # one handle per spec per worker process
        h = self._handles.get(spec_idx, None)
        if h is not None:
            return h
        p = self._specs[spec_idx].path
        # swmr=True can help sometimes; libhdf5 must support; safe to omit if issues
        h = h5py.File(p, "r")
        self._handles[spec_idx] = h
        return h

    def _read_block(self, spec_idx: int, block_start: int) -> np.ndarray:
        spec = self._specs[spec_idx]
        h5f = self._get_handle(spec_idx)
        ds = h5f[spec.key]

        if ds.ndim == 3:
            arr = ds[...][None, ...]  # (1,H,W,3) or (1,3,H,W)
            return _to_uint8_rgb(arr)

        block_end = min(block_start + self.block_size, spec.length)
        arr = ds[block_start:block_end]
        return _to_uint8_rgb(arr)

    def _get_from_cache(self, spec_idx: int, local_idx: int) -> np.ndarray:
        if self.cache_blocks == 0:
            block_start = (local_idx // self.block_size) * self.block_size
            block = self._read_block(spec_idx, block_start)
            return block[local_idx - block_start]

        block_start = (local_idx // self.block_size) * self.block_size

        # search cache (tiny)
        for i, (sidx, bstart, barr) in enumerate(self._cache):
            if sidx == spec_idx and bstart == block_start:
                # move-to-front (LRU)
                if i != 0:
                    self._cache.insert(0, self._cache.pop(i))
                return barr[local_idx - block_start]

        # miss: read + insert
        block = self._read_block(spec_idx, block_start)
        self._cache.insert(0, (spec_idx, block_start, block))
        if len(self._cache) > self.cache_blocks:
            self._cache.pop(-1)
        return block[local_idx - block_start]

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = self._n + idx
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)

        spec_idx, local_idx = self._find_spec(int(idx))
        patch = self._get_from_cache(spec_idx, local_idx)  # HWC uint8

        img = Image.fromarray(patch, mode="RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.return_index:
            return img, int(idx)
        return img

    def close(self):
        # optional explicit cleanup
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles = {}
        self._cache = []