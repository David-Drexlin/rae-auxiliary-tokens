# utils/celeba_dataset.py
from __future__ import annotations

from typing import List, Optional, Sequence
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CelebA


class CelebAMetaDataset(Dataset):
    """
    Returns:
      - (img, y) OR (img, y, meta)

    y:
      - if label_type="attr": a single binary attribute (0/1)
      - if label_type="identity": remapped identity index (0..K-1)

    meta:
      - optional list of binary attributes (0/1), one per meta field name
    """

    def __init__(
        self,
        root: str,
        split: str = "train",                 # train | valid | test | all
        label_type: str = "attr",             # "attr" | "identity"
        label_attr: str = "Smiling",          # used if label_type="attr"
        meta_fields: Optional[Sequence[str]] = None,  # list of attribute names (CelebA attrs)
        transform=None,
        download: bool = False,
        remap_identity: bool = True,
    ):
        super().__init__()
        self.root = str(root)
        self.split = str(split)
        self.label_type = str(label_type)
        self.label_attr = str(label_attr)
        self.meta_fields = list(meta_fields) if meta_fields else []
        self.transform = transform
        self.download = bool(download)
        self.remap_identity = bool(remap_identity)

        # We always need attrs if meta_fields requested, or if label_type is attr.
        target_types: List[str] = []
        if self.label_type == "identity":
            target_types.append("identity")
        else:
            target_types.append("attr")
        if self.meta_fields and "attr" not in target_types:
            target_types.append("attr")

        target_type = target_types[0] if len(target_types) == 1 else target_types

        self.ds = CelebA(
            root=self.root,
            split=self.split,
            target_type=target_type,
            transform=self.transform,
            download=self.download,
        )

        # attr name -> index
        self.attr_names = list(self.ds.attr_names)
        self.attr_idx = {name: i for i, name in enumerate(self.attr_names)}

        if self.label_type == "attr":
            if self.label_attr not in self.attr_idx:
                raise ValueError(f"Unknown CelebA attribute '{self.label_attr}'.")
            self.label_attr_idx = self.attr_idx[self.label_attr]
            self.num_classes = 2
            self._id_map = None
        elif self.label_type == "identity":
            # torchvision stores identity per-sample; avoid scanning images
            identity = getattr(self.ds, "identity", None)
            if identity is None:
                raise RuntimeError("torchvision CelebA dataset does not expose identity labels on this version.")
            identity = identity.view(-1).tolist()
            if self.remap_identity:
                uniq = sorted(set(int(x) for x in identity))
                self._id_map = {old: new for new, old in enumerate(uniq)}
                self.num_classes = len(uniq)
            else:
                self._id_map = None
                self.num_classes = int(max(identity)) + 1
        else:
            raise ValueError(f"label_type must be 'attr' or 'identity', got: {self.label_type}")

        # meta attribute indices
        self.meta_attr_indices = []
        for f in self.meta_fields:
            if f not in self.attr_idx:
                raise ValueError(f"Unknown CelebA meta attribute '{f}'.")
            self.meta_attr_indices.append(self.attr_idx[f])

    def __len__(self) -> int:
        return len(self.ds)

    @staticmethod
    def _to01(v: torch.Tensor) -> torch.Tensor:
        # CelebA attrs can be {-1, +1} or {0,1} depending on torchvision version
        return (v > 0).to(torch.long)

    def __getitem__(self, idx: int):
        img, _ = self.ds[idx]  # targets accessible via ds.attr / ds.identity

        if self.label_type == "attr":
            a = self.ds.attr[idx]
            y = self._to01(a[self.label_attr_idx])
        else:
            raw = int(self.ds.identity[idx].item())
            y = self._id_map[raw] if self._id_map is not None else raw
            y = torch.tensor(y, dtype=torch.long)

        if self.meta_attr_indices:
            a = self.ds.attr[idx]
            meta = self._to01(a[self.meta_attr_indices])
            return img, y, meta

        return img, y
