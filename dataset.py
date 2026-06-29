"""
src/utils/dataset.py
PyTorch Dataset for normalised iris strips.
  - Lazy loading from .npy files
  - CLAHE enhancement
  - Optional albumentations augmentation
  - build_dataloaders() factory returning train/val/test loaders
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from normalization import apply_clahe
from augmentation  import get_train_transform, get_val_transform


STRIP_H = 64
STRIP_W = 512


class IrisStripDataset(Dataset):
    """
    Lazy-loading dataset of normalised iris strips (.npy files).

    Args:
        records:      list of dicts, each with key "strip" → path to .npy
        augment:      apply training augmentation when True
        use_clahe:    apply CLAHE enhancement
    """

    def __init__(self,
                 records:   List[dict],
                 augment:   bool = False,
                 use_clahe: bool = True):
        self.records   = records
        self.augment   = augment
        self.use_clahe = use_clahe
        self.transform = get_train_transform() if augment else get_val_transform()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path  = Path(self.records[idx]["strip"])
        strip = np.load(str(path)).astype(np.float32)

        # Resize if spatial dims differ from expected
        if strip.shape != (STRIP_H, STRIP_W):
            strip = cv2.resize(strip, (STRIP_W, STRIP_H),
                               interpolation=cv2.INTER_LINEAR)

        if self.use_clahe:
            strip = apply_clahe(strip)

        if self.augment:
            uint8 = (strip * 255).clip(0, 255).astype(np.uint8)
            strip = self.transform(image=uint8)["image"].astype(np.float32) / 255.0

        # (1, H, W) float32 tensor
        return torch.from_numpy(strip).unsqueeze(0)


def build_dataloaders(
    records_file: Path,
    train_ratio:  float = 0.80,
    val_ratio:    float = 0.10,
    batch_size:   int   = 32,
    num_workers:  int   = 4,
    seed:         int   = 42,
    use_clahe:    bool  = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split records into train/val/test and return three DataLoaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    with open(records_file) as f:
        all_records = json.load(f)

    random.seed(seed)
    random.shuffle(all_records)

    n       = len(all_records)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_recs = all_records[:n_train]
    val_recs   = all_records[n_train: n_train + n_val]
    test_recs  = all_records[n_train + n_val:]

    print(f"  Dataset split — train: {len(train_recs)} | "
          f"val: {len(val_recs)} | test: {len(test_recs)}")

    train_ds = IrisStripDataset(train_recs, augment=True,  use_clahe=use_clahe)
    val_ds   = IrisStripDataset(val_recs,   augment=False, use_clahe=use_clahe)
    test_ds  = IrisStripDataset(test_recs,  augment=False, use_clahe=use_clahe)

    g = torch.Generator(); g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        generator=g, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader
