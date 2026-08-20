import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing.normalization import apply_clahe
from src.preprocessing.augmentation import get_train_transform, get_val_transform


STRIP_H = 64
STRIP_W = 512


def _subject_id_from_record(record: dict) -> str:
    """Return a subject identifier from the normalized strip filename.
    Examples: 001L_1_norm.npy -> '001', 015R_2_norm.npy -> '015'.
    """
    strip_path = str(record.get("strip", ""))
    stem = Path(strip_path).stem
    match = re.search(r"(\d+)", stem)
    if match:
        return match.group(1)
    return stem


def subject_disjoint_split(records: List[dict],
                          train_ratio: float = 0.80,
                          val_ratio: float = 0.10,
                          seed: int = 42) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split by subject ID so the same iris never leaks across train/val/test."""
    grouped = defaultdict(list)
    for record in records:
        grouped[_subject_id_from_record(record)].append(record)

    subject_ids = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(subject_ids)

    n_subjects = len(subject_ids)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)
    n_val = min(n_val, max(0, n_subjects - n_train))

    train_ids = subject_ids[:n_train]
    val_ids = subject_ids[n_train:n_train + n_val]
    test_ids = subject_ids[n_train + n_val:]

    train_recs = [r for sid in train_ids for r in grouped[sid]]
    val_recs = [r for sid in val_ids for r in grouped[sid]]
    test_recs = [r for sid in test_ids for r in grouped[sid]]

    return train_recs, val_recs, test_recs


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
        path = Path(self.records[idx]["strip"])
        if not path.exists():
            repo_root = Path(__file__).resolve().parents[2]
            alt_path = repo_root / "data" / "normalized" / path.name
            if alt_path.exists():
                path = alt_path

        if not path.exists():
            raise FileNotFoundError(
                f"Normalized strip not found: {path}\n"
                f"Expected file in data/normalized or a valid strip path in records."
            )

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

    train_recs, val_recs, test_recs = subject_disjoint_split(
        all_records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

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
