"""
src/models/patch_ae.py
PatchCore-inspired feature memory bank.
Extracts patch-level features from a pretrained EfficientNet-B0,
builds a coreset memory bank from normal training images,
scores test images by nearest-neighbour distance in feature space.
No gradient-based training required.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm


# ──────────────────────────────────────────────
# Feature extractor
# ──────────────────────────────────────────────

class PatchFeatureExtractor(nn.Module):
    """
    EfficientNet-B0 backbone truncated after 'features.4'.
    Patch features: spatial feature map averaged across channels
    then flattened per spatial location.
    """

    FEATURE_NODE = "features.4"

    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.extractor = create_feature_extractor(
            backbone, return_nodes={self.FEATURE_NODE: "feat"}
        )
        for p in self.extractor.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, H, W) grayscale strip
        Returns: (B, C, h, w) feature map
        """
        # EfficientNet expects 3-channel input
        x3 = x.repeat(1, 3, 1, 1)
        return self.extractor(x3)["feat"]


# ──────────────────────────────────────────────
# Coreset subsampling (greedy farthest-point)
# ──────────────────────────────────────────────

def greedy_coreset(features: np.ndarray, n: int) -> np.ndarray:
    """
    Greedy farthest-point sampling to select n representative
    feature vectors from a large pool.
    features: (N, D)
    Returns indices of selected samples.
    """
    N = features.shape[0]
    if N <= n:
        return np.arange(N)

    selected  = [np.random.randint(N)]
    distances = np.full(N, np.inf)

    for _ in range(n - 1):
        last    = features[selected[-1]]                # (D,)
        d       = np.linalg.norm(features - last, axis=1)
        distances = np.minimum(distances, d)
        selected.append(int(np.argmax(distances)))

    return np.array(selected)


# ──────────────────────────────────────────────
# PatchCore anomaly detector
# ──────────────────────────────────────────────

class PatchCoreDetector:
    """
    Training-free anomaly detector based on patch feature memory bank.

    Usage::
        detector = PatchCoreDetector(coreset_size=1000)
        detector.fit(train_loader, device)          # builds memory bank
        scores = detector.score_batch(test_batch, device)
        detector.save("checkpoints/patchcore.npz")
        detector.load("checkpoints/patchcore.npz")
    """

    def __init__(self,
                 coreset_size: int = 2000,
                 device: Optional[torch.device] = None):
        self.coreset_size = coreset_size
        self.device       = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.extractor  = PatchFeatureExtractor().to(self.device).eval()
        self.memory_bank: Optional[np.ndarray] = None   # (M, D)

    # ── Fit ───────────────────────────────────

    def fit(self, train_loader, verbose: bool = True) -> None:
        """Build memory bank from all training (normal) images."""
        all_patches: List[np.ndarray] = []

        for batch in tqdm(train_loader, desc="  Building memory bank",
                          disable=not verbose):
            x = batch.to(self.device)
            feat = self.extractor(x)           # (B, C, h, w)
            B, C, h, w = feat.shape
            # Reshape: each spatial location = one patch vector
            patches = feat.permute(0, 2, 3, 1).reshape(-1, C)  # (B*h*w, C)
            # L2-normalise
            patches = F.normalize(patches, dim=1).cpu().numpy()
            all_patches.append(patches)

        all_patches = np.concatenate(all_patches, axis=0)  # (N, C)
        print(f"  Total patches: {len(all_patches):,}  → coreset {self.coreset_size}")

        idx = greedy_coreset(all_patches, self.coreset_size)
        self.memory_bank = all_patches[idx]   # (M, C)
        print(f"  Memory bank: {self.memory_bank.shape}")

    # ── Score ─────────────────────────────────

    @torch.no_grad()
    def score_batch(self, x: torch.Tensor) -> np.ndarray:
        """
        Compute image-level anomaly scores for a batch.
        Score = max patch NN-distance across the image.
        Returns numpy array (B,).
        """
        if self.memory_bank is None:
            raise RuntimeError("Call fit() before score_batch().")

        x    = x.to(self.device)
        feat = self.extractor(x)          # (B, C, h, w)
        B, C, h, w = feat.shape
        patches = feat.permute(0, 2, 3, 1).reshape(B, h * w, C)  # (B, P, C)
        patches = F.normalize(patches, dim=2).cpu().numpy()

        mb  = torch.tensor(self.memory_bank, dtype=torch.float32)  # (M, C)
        scores = np.zeros(B, dtype=np.float32)

        for i in range(B):
            p = torch.tensor(patches[i], dtype=torch.float32)  # (P, C)
            # Distances: (P, M) → min over memory bank → max over patches
            dists = torch.cdist(p, mb)              # (P, M)
            nn_dist = dists.min(dim=1).values       # (P,)
            scores[i] = float(nn_dist.max())

        return scores

    # ── Persistence ───────────────────────────

    def save(self, path: str) -> None:
        if self.memory_bank is None:
            raise RuntimeError("Nothing to save — call fit() first.")
        np.savez_compressed(path, memory_bank=self.memory_bank)
        print(f"  Memory bank saved → {path}")

    def load(self, path: str) -> None:
        data = np.load(path)
        self.memory_bank = data["memory_bank"]
        print(f"  Memory bank loaded ← {path}  shape={self.memory_bank.shape}")
