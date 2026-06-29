"""
scripts/input.py
Input pipeline utility for the iris anomaly detection project.

Handles three input modes:
  1. Single image  — one raw iris image file  (jpg / png / bmp)
  2. Directory     — folder of raw iris images
  3. Records file  — existing normalization_records.json

In modes 1 & 2 the image(s) are normalised on-the-fly (rubber-sheet
unwrapping is skipped here; images are assumed to already be iris strips).
CLAHE enhancement and resizing are applied to match the training pipeline.

Usage:
    # Single image → tensor ready for inference
    python input.py --image path/to/iris.png

    # Directory of images → tensor batch
    python input.py --dir path/to/iris_folder/

    # Existing records file → DataLoaders (same as train.py)
    python input.py --records data/normalized/normalization_records.json

    # Override strip size or batch size
    python input.py --image iris.png --strip-h 64 --strip-w 512 --batch-size 8
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config      import load_config, get_device
from normalization import apply_clahe
from augmentation  import get_val_transform

# ──────────────────────────────────────────────
# Constants (match dataset.py defaults)
# ──────────────────────────────────────────────
STRIP_H = 64
STRIP_W = 512
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ──────────────────────────────────────────────
# Image loading & preprocessing
# ──────────────────────────────────────────────

def load_image(path: Path, strip_h: int = STRIP_H,
               strip_w: int = STRIP_W) -> np.ndarray:
    """
    Load a single iris image from disk and preprocess it into a
    float32 strip of shape (strip_h, strip_w) in range [0, 1].

    Accepts:
      - Grayscale .npy  (already a normalised strip)
      - Any OpenCV-supported image format (png, jpg, bmp, tif)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() == ".npy":
        strip = np.load(str(path)).astype(np.float32)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        strip = img.astype(np.float32) / 255.0

    # Resize to expected strip dimensions if needed
    if strip.shape != (strip_h, strip_w):
        strip = cv2.resize(strip, (strip_w, strip_h),
                           interpolation=cv2.INTER_LINEAR)

    return strip


def preprocess_strip(strip: np.ndarray,
                     use_clahe: bool = True) -> torch.Tensor:
    """
    Apply CLAHE (optional) and convert to a (1, H, W) float32 tensor.
    Matches the exact preprocessing in IrisStripDataset.__getitem__.
    """
    if use_clahe:
        strip = apply_clahe(strip)

    return torch.from_numpy(strip).unsqueeze(0)   # (1, H, W)


# ──────────────────────────────────────────────
# Dataset for arbitrary image lists
# ──────────────────────────────────────────────

class RawIrisDataset(Dataset):
    """
    Lightweight dataset that wraps a list of image paths.
    No augmentation — inference / evaluation only.
    """

    def __init__(self, image_paths: list,
                 strip_h: int = STRIP_H,
                 strip_w: int = STRIP_W,
                 use_clahe: bool = True):
        self.paths     = image_paths
        self.strip_h   = strip_h
        self.strip_w   = strip_w
        self.use_clahe = use_clahe

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        strip = load_image(self.paths[idx], self.strip_h, self.strip_w)
        return preprocess_strip(strip, self.use_clahe)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def image_to_tensor(image_path: Path,
                    strip_h:   int  = STRIP_H,
                    strip_w:   int  = STRIP_W,
                    use_clahe: bool = True) -> torch.Tensor:
    """
    Load a single image and return a (1, 1, H, W) batch tensor.
    Ready to pass directly into a model.

    Example:
        x = image_to_tensor("iris.png")
        score = model.anomaly_score(x.to(device))
    """
    strip  = load_image(image_path, strip_h, strip_w)
    tensor = preprocess_strip(strip, use_clahe)
    return tensor.unsqueeze(0)   # add batch dim → (1, 1, H, W)


def dir_to_dataloader(image_dir:  Path,
                      batch_size: int  = 32,
                      num_workers: int = 4,
                      strip_h:   int   = STRIP_H,
                      strip_w:   int   = STRIP_W,
                      use_clahe: bool  = True) -> DataLoader:
    """
    Scan a directory for iris images and return a DataLoader.
    No augmentation applied — suitable for inference / evaluation.

    Example:
        loader = dir_to_dataloader("data/test_images/")
        for batch in loader:
            scores = model.anomaly_score(batch.to(device))
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")

    paths = sorted([
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS or p.suffix.lower() == ".npy"
    ])

    if not paths:
        raise FileNotFoundError(
            f"No supported images found in {image_dir}\n"
            f"Supported: {SUPPORTED_EXTS | {'.npy'}}"
        )

    print(f"  Found {len(paths)} image(s) in {image_dir}")
    ds = RawIrisDataset(paths, strip_h, strip_w, use_clahe)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def records_to_dataloader(records_file: Path,
                          batch_size:  int  = 32,
                          num_workers: int  = 4,
                          use_clahe:   bool = True) -> DataLoader:
    """
    Build a DataLoader from an existing normalization_records.json.
    Loads ALL records (no train/val split) — suitable for bulk inference.

    Example:
        loader = records_to_dataloader("data/normalized/normalization_records.json")
    """
    from dataset import IrisStripDataset

    records_file = Path(records_file)
    if not records_file.exists():
        raise FileNotFoundError(
            f"Records file not found: {records_file}\n"
            "Run  python prepare_data.py  first."
        )

    with open(records_file) as f:
        records = json.load(f)

    print(f"  Loaded {len(records)} record(s) from {records_file}")
    ds = IrisStripDataset(records, augment=False, use_clahe=use_clahe)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Iris input pipeline — load, preprocess, and inspect inputs."
    )

    # Input source (mutually exclusive)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",   type=Path,
                     help="Path to a single iris image or .npy strip")
    src.add_argument("--dir",     type=Path,
                     help="Directory of iris images")
    src.add_argument("--records", type=Path,
                     help="Path to normalization_records.json")

    # Preprocessing options
    parser.add_argument("--strip-h",    type=int,  default=STRIP_H, dest="strip_h")
    parser.add_argument("--strip-w",    type=int,  default=STRIP_W, dest="strip_w")
    parser.add_argument("--no-clahe",   action="store_true",
                        help="Disable CLAHE enhancement")
    parser.add_argument("--batch-size", type=int,  default=32, dest="batch_size")
    parser.add_argument("--workers",    type=int,  default=4)

    return parser.parse_args()


def main():
    args      = _parse_args()
    use_clahe = not args.no_clahe
    device    = get_device()

    print(f"\n  Device    : {device}")
    print(f"  Strip size: {args.strip_h} x {args.strip_w}")
    print(f"  CLAHE     : {'on' if use_clahe else 'off'}\n")

    # ── Single image ──────────────────────────────────────
    if args.image:
        tensor = image_to_tensor(args.image, args.strip_h,
                                 args.strip_w, use_clahe)
        print(f"  Image     : {args.image}")
        print(f"  Tensor    : shape={tuple(tensor.shape)}  "
              f"dtype={tensor.dtype}  "
              f"range=[{tensor.min():.3f}, {tensor.max():.3f}]")
        print("\n  Ready for inference — example usage:")
        print("      model.eval()")
        print("      score = model.anomaly_score(tensor.to(device))")
        return tensor

    # ── Directory ─────────────────────────────────────────
    if args.dir:
        loader = dir_to_dataloader(
            args.dir, args.batch_size, args.workers,
            args.strip_h, args.strip_w, use_clahe
        )
        print(f"  Batches   : {len(loader)}")
        # Inspect first batch
        first = next(iter(loader))
        print(f"  Batch[0]  : shape={tuple(first.shape)}  "
              f"dtype={first.dtype}  "
              f"range=[{first.min():.3f}, {first.max():.3f}]")
        print("\n  DataLoader ready — pass to your model loop.")
        return loader

    # ── Records file ──────────────────────────────────────
    if args.records:
        loader = records_to_dataloader(
            args.records, args.batch_size, args.workers, use_clahe
        )
        print(f"  Batches   : {len(loader)}")
        first = next(iter(loader))
        print(f"  Batch[0]  : shape={tuple(first.shape)}  "
              f"dtype={first.dtype}  "
              f"range=[{first.min():.3f}, {first.max():.3f}]")
        print("\n  DataLoader ready — pass to your model loop.")
        return loader


if __name__ == "__main__":
    main()