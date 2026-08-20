"""
scripts/train.py
Train ConvAE, ConvVAE, PatchCore, or ViT-MAE on normalised iris strips.

Usage:
    python scripts/train.py                        # default: ViT-MAE (recommended)
    python scripts/train.py --model mae            # ViT Masked Autoencoder (best)
    python scripts/train.py --model ae             # ConvAutoencoder
    python scripts/train.py --model vae            # ConvVAE
    python scripts/train.py --model patchcore      # PatchCore (no grad)
    python scripts/train.py --epochs 200 --lr 5e-5
    python scripts/train.py --model mae --mask-ratio 0.75

Why ViT-MAE?
    CNN autoencoders (AE/VAE) overfit easily on small iris datasets because
    their inductive biases (local convolutions, pooling) memorise textures.
    ViT-MAE forces the model to reconstruct 75 % of randomly masked patches
    from global context, acting as strong self-supervised regularisation that
    generalises better to unseen normal irises and produces sharper anomaly
    contrast for truly anomalous samples.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOT))

from src.Models.autoencoder import ConvAutoencoder
from src.Models.vae import ConvVAE
from src.Models.patch_ae import PatchCoreDetector
from src.training.trainer import train_autoencoder, train_vae
from src.utils.dataset import build_dataloaders
from src.utils.config import load_config, get_device
from src.evaluation.visualize import plot_training_curves


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
NORM_REC_FILE  = REPO_ROOT / "data" / "normalized" / "normalization_records.json"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
LOG_DIR        = REPO_ROOT / "logs"
OUTPUT_IMG_DIR = REPO_ROOT / "outputs" / "images"          # ← NEW dedicated image folder


def _ensure_dirs():
    for d in [CHECKPOINT_DIR, LOG_DIR, OUTPUT_IMG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════
# ViT-MAE implementation (self-contained)
# ══════════════════════════════════════════════

class PatchEmbed(nn.Module):
    """Split a 1-channel iris strip into non-overlapping patches."""
    def __init__(self, img_h, img_w, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.n_h = img_h // patch_size
        self.n_w = img_w // patch_size
        self.num_patches = self.n_h * self.n_w
        self.proj = nn.Conv2d(1, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 1, H, W)  →  (B, num_patches, embed_dim)
        x = self.proj(x)                          # (B, E, n_h, n_w)
        x = x.flatten(2).transpose(1, 2)          # (B, N, E)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1  = nn.LayerNorm(embed_dim)
        self.attn   = nn.MultiheadAttention(embed_dim, num_heads,
                                            dropout=drop, batch_first=True)
        self.norm2  = nn.LayerNorm(embed_dim)
        hidden_dim  = int(embed_dim * mlp_ratio)
        self.mlp    = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ViTMAE(nn.Module):
    """
    Vision Transformer Masked Autoencoder for iris anomaly detection.

    Architecture
    ────────────
    Encoder : ViT that sees only the (1-mask_ratio) visible patches.
    Decoder : shallow ViT that reconstructs ALL patches from encoder tokens
              + learned mask tokens.
    Anomaly : per-image MSE between input and reconstruction on masked patches.

    Reference: He et al., "Masked Autoencoders Are Scalable Vision Learners",
               CVPR 2022  (adapted for 1-channel, variable aspect-ratio strips).
    """

    def __init__(
        self,
        img_h=64, img_w=512,
        patch_size=16,
        encoder_embed_dim=256,
        encoder_depth=6,
        encoder_num_heads=8,
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        mask_ratio=0.75,
        drop=0.1,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        # ── Encoder ──────────────────────────────────────────
        self.patch_embed   = PatchEmbed(img_h, img_w, patch_size, encoder_embed_dim)
        num_patches        = self.patch_embed.num_patches
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.enc_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, encoder_embed_dim))
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(encoder_embed_dim, encoder_num_heads, mlp_ratio, drop)
            for _ in range(encoder_depth)
        ])
        self.enc_norm = nn.LayerNorm(encoder_embed_dim)

        # ── Decoder ──────────────────────────────────────────
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.dec_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, drop)
            for _ in range(decoder_depth)
        ])
        self.dec_norm = nn.LayerNorm(decoder_embed_dim)
        # Predict pixel values for each patch
        self.dec_pred = nn.Linear(decoder_embed_dim,
                                  patch_size * patch_size * 1)

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token,     std=0.02)
        nn.init.trunc_normal_(self.mask_token,    std=0.02)
        nn.init.trunc_normal_(self.enc_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Masking ───────────────────────────────────────────────
    def random_masking(self, x):
        """
        x : (B, N, E)
        Returns visible tokens, mask (1=masked), restore indices.
        """
        B, N, E = x.shape
        keep    = int(N * (1 - self.mask_ratio))

        noise   = torch.rand(B, N, device=x.device)
        ids_shuffle  = torch.argsort(noise, dim=1)
        ids_restore  = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :keep]
        x_vis    = torch.gather(x, 1,
                                ids_keep.unsqueeze(-1).expand(-1, -1, E))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_vis, mask, ids_restore

    # ── Encoder forward ───────────────────────────────────────
    def encode(self, x, mask_ratio=None):
        if mask_ratio is not None:
            old = self.mask_ratio
            self.mask_ratio = mask_ratio

        tokens = self.patch_embed(x)
        tokens = tokens + self.enc_pos_embed[:, 1:, :]
        tokens, mask, ids_restore = self.random_masking(tokens)

        cls    = self.cls_token + self.enc_pos_embed[:, :1, :]
        cls    = cls.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        for blk in self.encoder_blocks:
            tokens = blk(tokens)
        tokens = self.enc_norm(tokens)

        if mask_ratio is not None:
            self.mask_ratio = old
        return tokens, mask, ids_restore

    # ── Decoder forward ───────────────────────────────────────
    def decode(self, latent, ids_restore):
        tokens = self.decoder_embed(latent)

        B, N_full = ids_restore.shape
        n_vis     = tokens.shape[1] - 1   # exclude CLS

        mask_tokens = self.mask_token.expand(B, N_full - n_vis, -1)
        full        = torch.cat([tokens[:, 1:, :], mask_tokens], dim=1)
        full        = torch.gather(
            full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        )
        full = torch.cat([tokens[:, :1, :], full], dim=1)
        full = full + self.dec_pos_embed

        for blk in self.decoder_blocks:
            full = blk(full)
        full = self.dec_norm(full)
        pred = self.dec_pred(full[:, 1:, :])   # remove CLS  →  (B, N, patch²)
        return pred

    # ── Patchify / Unpatchify ─────────────────────────────────
    def patchify(self, imgs):
        """(B,1,H,W) → (B, N, patch²)"""
        p  = self.patch_size
        h  = imgs.shape[2] // p
        w  = imgs.shape[3] // p
        x  = imgs.reshape(imgs.shape[0], 1, h, p, w, p)
        x  = x.permute(0, 2, 4, 3, 5, 1).reshape(
                 imgs.shape[0], h * w, p * p * 1)
        return x

    def unpatchify(self, x, img_h, img_w):
        """(B, N, patch²) → (B,1,H,W)"""
        p  = self.patch_size
        h  = img_h // p
        w  = img_w // p
        x  = x.reshape(x.shape[0], h, w, p, p, 1)
        x  = x.permute(0, 5, 1, 3, 2, 4).reshape(
                 x.shape[0], 1, img_h, img_w)
        return x

    # ── Loss ─────────────────────────────────────────────────
    def forward(self, imgs):
        """Returns (loss, pred_pixels, mask)."""
        latent, mask, ids_restore = self.encode(imgs)
        pred  = self.decode(latent, ids_restore)
        target = self.patchify(imgs)

        loss = (pred - target) ** 2           # (B, N, patch²)
        loss = loss.mean(dim=-1)              # (B, N)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)   # only masked patches
        return loss, pred, mask

    # ── Anomaly scoring ──────────────────────────────────────
    @torch.no_grad()
    def anomaly_score(self, imgs):
        """
        Per-image anomaly score = mean reconstruction error over masked patches.
        Higher score → more anomalous.
        """
        _, pred, mask = self.forward(imgs)
        target = self.patchify(imgs)
        err    = ((pred - target) ** 2).mean(dim=-1)   # (B, N)
        score  = (err * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        return score


# ══════════════════════════════════════════════
# ViT-MAE trainer
# ══════════════════════════════════════════════

def train_mae(args, train_loader, val_loader, device):
    print("\n  Model: ViT-MAE  (Vision Transformer Masked Autoencoder)")
    print(f"  img_h={args.img_h}  img_w={args.img_w}  "
          f"patch={args.patch_size}  mask_ratio={args.mask_ratio}")

    model = ViTMAE(
        img_h=args.img_h,
        img_w=args.img_w,
        patch_size=args.patch_size,
        encoder_embed_dim=args.enc_dim,
        encoder_depth=args.enc_depth,
        encoder_num_heads=args.enc_heads,
        decoder_embed_dim=args.dec_dim,
        decoder_depth=args.dec_depth,
        decoder_num_heads=args.dec_heads,
        mask_ratio=args.mask_ratio,
        drop=args.drop,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        t_loss = 0.0
        for batch in train_loader:
            x = batch.to(device)
            optimiser.zero_grad()
            loss, _, _ = model(x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)
        scheduler.step()

        # ── validate ──
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                loss, _, _ = model(x)
                v_loss += loss.item()
        v_loss /= len(val_loader)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{args.epochs}  "
                  f"train={t_loss:.5f}  val={v_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if v_loss < best_val:
            best_val = v_loss
            torch.save(
                {"epoch": epoch, "val_loss": v_loss,
                 "state": model.state_dict()},
                str(CHECKPOINT_DIR / "best_mae.pth"),
            )

    print(f"\n  Best val loss: {best_val:.5f}")
    _save_training_curves(history, title="ViT-MAE training",
                          save_path=OUTPUT_IMG_DIR / "mae_curves.png")
    return history


# ══════════════════════════════════════════════
# Legacy model trainers (unchanged logic, new output paths)
# ══════════════════════════════════════════════

def train_ae(args, train_loader, val_loader, device):
    print("\n  Model: ConvAutoencoder")
    model   = ConvAutoencoder(latent_dim=args.latent_dim)
    history = train_autoencoder(
        model, train_loader, val_loader,
        checkpoint_dir=CHECKPOINT_DIR,
        log_dir=LOG_DIR,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        warmup_epochs=args.warmup,
        gamma_kl=args.gamma,
    )
    plot_training_curves(
        history, title="ConvAE training",
        save_path=OUTPUT_IMG_DIR / "ae_curves.png",       # → outputs/images/
    )
    return history


def train_vae_model(args, train_loader, val_loader, device):
    print("\n  Model: ConvVAE")
    model   = ConvVAE(latent_dim=args.latent_dim)
    history = train_vae(
        model, train_loader, val_loader,
        checkpoint_dir=CHECKPOINT_DIR,
        log_dir=LOG_DIR,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        warmup_epochs=args.warmup,
        gamma_kl=args.gamma,
    )
    plot_training_curves(
        history, title="ConvVAE training",
        save_path=OUTPUT_IMG_DIR / "vae_curves.png",      # → outputs/images/
    )
    return history


def train_patchcore(args, train_loader, device):
    print("\n  Model: PatchCore (no gradient training)")
    detector = PatchCoreDetector(
        coreset_size=args.coreset_size,
        device=device,
    )
    detector.fit(train_loader)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    detector.save(str(CHECKPOINT_DIR / "patchcore.npz"))
    print("  PatchCore memory bank saved.")


# ══════════════════════════════════════════════
# Shared utility: training curve plot
# ══════════════════════════════════════════════

def _save_training_curves(history, title, save_path):
    """Matplotlib training/validation loss curves saved to outputs/images/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Iris Anomaly — Train")

    # ── Model choice ──────────────────────────
    parser.add_argument("--model", default="mae",
                        choices=["ae", "vae", "patchcore", "mae"],
                        help="Model to train (default: mae — ViT-MAE, recommended)")

    # ── Shared hyper-params ───────────────────
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int,   default=16, dest="batch_size")
    parser.add_argument("--workers",    type=int,   default=4)

    # ── CNN AE / VAE ─────────────────────────
    parser.add_argument("--latent-dim", type=int,   default=64, dest="latent_dim")
    parser.add_argument("--gamma",      type=float, default=1.0,
                        help="KL weight (VAE only)")
    parser.add_argument("--warmup",     type=int,   default=10)

    # ── PatchCore ────────────────────────────
    parser.add_argument("--coreset-size", type=int, default=2000,
                        dest="coreset_size")

    # ── ViT-MAE architecture ─────────────────
    parser.add_argument("--img-h",      type=int,   default=64,  dest="img_h",
                        help="Iris strip height (pixels)")
    parser.add_argument("--img-w",      type=int,   default=512, dest="img_w",
                        help="Iris strip width  (pixels)")
    parser.add_argument("--patch-size", type=int,   default=16,  dest="patch_size",
                        help="ViT patch size; must divide img-h and img-w")
    parser.add_argument("--mask-ratio", type=float, default=0.75, dest="mask_ratio",
                        help="Fraction of patches masked during training (default 0.75)")
    parser.add_argument("--enc-dim",    type=int,   default=256, dest="enc_dim",
                        help="Encoder embedding dimension")
    parser.add_argument("--enc-depth",  type=int,   default=6,   dest="enc_depth")
    parser.add_argument("--enc-heads",  type=int,   default=8,   dest="enc_heads")
    parser.add_argument("--dec-dim",    type=int,   default=128, dest="dec_dim",
                        help="Decoder embedding dimension (lighter than encoder)")
    parser.add_argument("--dec-depth",  type=int,   default=4,   dest="dec_depth")
    parser.add_argument("--dec-heads",  type=int,   default=8,   dest="dec_heads")
    parser.add_argument("--drop",       type=float, default=0.1,
                        help="Dropout rate (regularisation)")
    parser.add_argument("--wd",         type=float, default=0.05,
                        help="AdamW weight decay (regularisation)")

    # ── Data ─────────────────────────────────
    parser.add_argument("--records",    type=Path,
                        default=NORM_REC_FILE)

    args = parser.parse_args()

    # ── Sanity checks for MAE ─────────────────
    if args.model == "mae":
        assert args.img_h % args.patch_size == 0, \
            f"img-h ({args.img_h}) must be divisible by patch-size ({args.patch_size})"
        assert args.img_w % args.patch_size == 0, \
            f"img-w ({args.img_w}) must be divisible by patch-size ({args.patch_size})"

    _ensure_dirs()
    device = get_device()
    print(f"  Device       : {device}")
    print(f"  Model        : {args.model}")
    print(f"  Output images: {OUTPUT_IMG_DIR}")

    if not args.records.exists():
        print(f"\n  ERROR: {args.records} not found.")
        print("  Run  python scripts/prepare_data.py  first.")
        sys.exit(1)

    train_loader, val_loader, _ = build_dataloaders(
        records_file=args.records,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    t0 = time.time()
    if args.model == "ae":
        train_ae(args, train_loader, val_loader, device)
    elif args.model == "vae":
        train_vae_model(args, train_loader, val_loader, device)
    elif args.model == "patchcore":
        train_patchcore(args, train_loader, device)
    else:   # mae (default)
        train_mae(args, train_loader, val_loader, device)

    elapsed = (time.time() - t0) / 60
    print(f"\n✓ Training complete in {elapsed:.1f} min.")
    print(f"  Checkpoints  → {CHECKPOINT_DIR}/")
    print(f"  Output images→ {OUTPUT_IMG_DIR}/")


if __name__ == "__main__":
    main()