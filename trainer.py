"""
src/training/trainer.py
Unified trainer for ConvAutoencoder and ConvVAE.
  - Epoch loop with train / val
  - Gradient clipping
  - Warmup-cosine LR schedule
  - Best-checkpoint saving (by val loss)
  - Reconstruction image logging every N epochs
  - Training curve saved to outputs/images/
  - Returns loss history dict
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before pyplot import

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import AdamW          # changed: Adam → AdamW (matches train.py)
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses    import ReconstructionLoss, vae_loss
from scheduler import build_scheduler


# ──────────────────────────────────────────────
# Paths  (mirrors train.py layout)
# ──────────────────────────────────────────────
_ROOT          = Path(__file__).resolve().parent
OUTPUT_IMG_DIR = _ROOT / "outputs" / "images"   # ← matches train.py OUTPUT_IMG_DIR


# ──────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────
DEFAULTS = dict(
    lr            = 1e-4,
    weight_decay  = 0.05,   # changed: 1e-5 → 0.05  (AdamW default in train.py)
    epochs        = 150,
    warmup_epochs = 10,
    grad_clip     = 1.0,
    gamma_kl      = 1.0,    # VAE only
    log_every     = 10,     # save recon images every N epochs
    alpha_mse     = 0.5,
    beta_ssim     = 0.5,
)


# ──────────────────────────────────────────────
# Shared utility: training curve plot
# ──────────────────────────────────────────────

def _save_training_curves(history: Dict, title: str, save_path: Path) -> None:
    """Save train/val loss curves to outputs/images/ (mirrors train.py helper)."""
    try:
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(epochs, history["train_loss"], label="Train loss", linewidth=1.5)
        ax.plot(epochs, history["val_loss"],   label="Val loss",   linewidth=1.5,
                linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150)
        plt.close(fig)
        print(f"  Curve saved → {save_path}")
    except Exception as e:
        print(f"  [warn] Could not save training curve: {e}")


# ──────────────────────────────────────────────
# Reconstruction visualisation helper
# ──────────────────────────────────────────────

def _log_reconstructions(model, loader: DataLoader,
                          epoch: int, save_dir: Path,
                          device: torch.device,
                          is_vae: bool = False,
                          n: int = 4) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    batch = next(iter(loader))[:n].to(device)
    with torch.no_grad():
        recon = model(batch)[0]

    fig, axes = plt.subplots(n, 2, figsize=(12, n * 1.8))
    for i in range(n):
        axes[i, 0].imshow(batch[i, 0].cpu(), cmap="gray", aspect="auto")
        axes[i, 0].set_title("Input",  fontsize=8); axes[i, 0].axis("off")
        axes[i, 1].imshow(recon[i, 0].cpu(), cmap="gray", aspect="auto")
        axes[i, 1].set_title("Recon.", fontsize=8); axes[i, 1].axis("off")

    plt.suptitle(f"Epoch {epoch}", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_dir / f"recon_{epoch:04d}.png",
                dpi=110, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────
# AE trainer
# ──────────────────────────────────────────────

def train_autoencoder(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    checkpoint_dir: Path,
    log_dir:        Path,
    device:         torch.device,
    **kwargs,
) -> Dict:
    """Train a ConvAutoencoder. Returns loss history."""
    cfg = {**DEFAULTS, **kwargs}

    model.to(device)
    criterion = ReconstructionLoss(cfg["alpha_mse"], cfg["beta_ssim"])
    optim     = AdamW(model.parameters(),             # changed: Adam → AdamW
                      lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched     = build_scheduler(optim, cfg["warmup_epochs"], cfg["epochs"])

    history   = {"train_loss": [], "val_loss": []}
    best_val  = np.inf
    recon_dir = OUTPUT_IMG_DIR / "recon_ae"           # changed: log_dir → OUTPUT_IMG_DIR

    for epoch in range(1, cfg["epochs"] + 1):
        # ── train ──
        model.train()
        t_losses = []
        for batch in tqdm(train_loader,
                          desc=f"Ep {epoch}/{cfg['epochs']} [AE train]",
                          leave=False):
            x = batch.to(device)
            optim.zero_grad()                         # changed: moved before forward pass
            loss = criterion(model(x)[0], x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optim.step()
            t_losses.append(loss.item())

        # ── val ──
        model.eval()
        v_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                v_losses.append(criterion(model(x)[0], x).item())

        t_loss = np.mean(t_losses)
        v_loss = np.mean(v_losses)
        sched.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        print(f"  Ep {epoch:4d} | train {t_loss:.5f} | val {v_loss:.5f} "
              f"| lr {optim.param_groups[0]['lr']:.2e}")

        if v_loss < best_val:
            best_val = v_loss
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "state": model.state_dict(),
                        "val_loss": v_loss},
                       checkpoint_dir / "best_ae.pth")

        if epoch % cfg["log_every"] == 0 or epoch == 1:
            _log_reconstructions(model, val_loader, epoch,
                                  recon_dir, device, is_vae=False)

    # changed: save training curve to outputs/images/ (matches train.py)
    _save_training_curves(history, title="ConvAE training",
                          save_path=OUTPUT_IMG_DIR / "ae_curves.png")
    return history


# ──────────────────────────────────────────────
# VAE trainer
# ──────────────────────────────────────────────

def train_vae(
    model:          nn.Module,
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    checkpoint_dir: Path,
    log_dir:        Path,
    device:         torch.device,
    **kwargs,
) -> Dict:
    """Train a ConvVAE. Returns loss history."""
    cfg = {**DEFAULTS, **kwargs}

    model.to(device)
    recon_fn  = ReconstructionLoss(cfg["alpha_mse"], cfg["beta_ssim"])
    optim     = AdamW(model.parameters(),             # changed: Adam → AdamW
                      lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched     = build_scheduler(optim, cfg["warmup_epochs"], cfg["epochs"])

    history  = {"train_loss": [], "val_loss": [],
                "train_kl":   [], "val_kl":   []}
    best_val = np.inf
    recon_dir = OUTPUT_IMG_DIR / "recon_vae"          # changed: log_dir → OUTPUT_IMG_DIR

    for epoch in range(1, cfg["epochs"] + 1):
        # ── train ──
        model.train()
        t_losses, t_kls = [], []
        for batch in tqdm(train_loader,
                          desc=f"Ep {epoch}/{cfg['epochs']} [VAE train]",
                          leave=False):
            x = batch.to(device)
            optim.zero_grad()                         # changed: moved before forward pass
            recon, mu, logvar = model(x)
            loss, parts = vae_loss(recon, x, mu, logvar,
                                   recon_fn, cfg["gamma_kl"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optim.step()
            t_losses.append(parts["total"])
            t_kls.append(parts["kl"])

        # ── val ──
        model.eval()
        v_losses, v_kls = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                recon, mu, logvar = model(x)
                _, parts = vae_loss(recon, x, mu, logvar,
                                    recon_fn, cfg["gamma_kl"])
                v_losses.append(parts["total"])
                v_kls.append(parts["kl"])

        t_loss = np.mean(t_losses); v_loss = np.mean(v_losses)
        sched.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_kl"].append(np.mean(t_kls))
        history["val_kl"].append(np.mean(v_kls))

        print(f"  Ep {epoch:4d} | train {t_loss:.5f} | val {v_loss:.5f} "
              f"| KL {np.mean(v_kls):.4f} "
              f"| lr {optim.param_groups[0]['lr']:.2e}")

        if v_loss < best_val:
            best_val = v_loss
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "state": model.state_dict(),
                        "val_loss": v_loss},
                       checkpoint_dir / "best_vae.pth")

        if epoch % cfg["log_every"] == 0 or epoch == 1:
            _log_reconstructions(model, val_loader, epoch,
                                  recon_dir, device, is_vae=True)

    # changed: save training curve to outputs/images/ (matches train.py)
    _save_training_curves(history, title="ConvVAE training",
                          save_path=OUTPUT_IMG_DIR / "vae_curves.png")
    return history