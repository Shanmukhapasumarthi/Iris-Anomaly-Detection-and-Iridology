"""
src/training/losses.py
Loss functions:
  - SSIMLoss          : differentiable SSIM (1 - SSIM)
  - ReconstructionLoss: α·MSE + β·SSIM
  - vae_loss          : ReconstructionLoss + γ·KL
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# SSIM
# ──────────────────────────────────────────────

class SSIMLoss(nn.Module):
    """
    Structural Similarity loss (1 − SSIM) for single-channel images.
    Window: 11×11 Gaussian (Wang et al. 2004).
    """

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", self._gaussian_window(window_size))

    @staticmethod
    def _gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g     /= g.sum()
        return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w  = self.window.to(pred.device)
        p  = self.window_size // 2
        C1, C2 = 0.01 ** 2, 0.03 ** 2

        mu_x  = F.conv2d(pred,   w, padding=p)
        mu_y  = F.conv2d(target, w, padding=p)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sig_x  = F.conv2d(pred   * pred,   w, padding=p) - mu_x2
        sig_y  = F.conv2d(target * target, w, padding=p) - mu_y2
        sig_xy = F.conv2d(pred   * target, w, padding=p) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2)

        return 1.0 - (num / (den + 1e-8)).mean()


# ──────────────────────────────────────────────
# Reconstruction loss
# ──────────────────────────────────────────────

class ReconstructionLoss(nn.Module):
    """Combined α·MSE + β·SSIM loss."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.ssim  = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse  = F.mse_loss(pred, target)
        ssim = self.ssim(pred, target)
        return self.alpha * mse + self.beta * ssim


# ──────────────────────────────────────────────
# VAE loss
# ──────────────────────────────────────────────

def vae_loss(recon:    torch.Tensor,
             target:   torch.Tensor,
             mu:       torch.Tensor,
             logvar:   torch.Tensor,
             recon_fn: ReconstructionLoss,
             gamma:    float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Total VAE loss = recon_loss + γ·KL.

    Returns:
        total_loss : scalar tensor
        parts      : dict with individual loss values for logging
    """
    recon_l = recon_fn(recon, target)
    kl      = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()
                      ).sum(dim=1).mean()
    total   = recon_l + gamma * kl
    return total, {
        "recon": float(recon_l),
        "kl":    float(kl),
        "total": float(total),
    }
