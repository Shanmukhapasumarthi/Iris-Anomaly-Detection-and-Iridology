"""
src/models/vae.py
Convolutional Variational Autoencoder (VAE).
Same backbone as ConvAutoencoder but outputs (μ, log σ²).
Reparameterisation trick enables differentiable latent sampling.
Anomaly score = reconstruction loss + β·KL divergence.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _deconv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ConvVAE(nn.Module):
    """
    Variational Autoencoder for iris texture anomaly detection.

    Usage::
        model = ConvVAE(latent_dim=256)
        recon, mu, logvar = model(x)      # training forward
        score = model.anomaly_score(x)    # inference
    """

    FLAT_DIM = 256 * 4 * 32

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            _conv_block(1,   32),
            _conv_block(32,  64),
            _conv_block(64,  128),
            _conv_block(128, 256),
        )
        self.fc_mu     = nn.Linear(self.FLAT_DIM, latent_dim)
        self.fc_logvar = nn.Linear(self.FLAT_DIM, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.FLAT_DIM)

        self.decoder = nn.Sequential(
            _deconv_block(256, 128),
            _deconv_block(128, 64),
            _deconv_block(64,  32),
            _deconv_block(32,   1),
            nn.Sigmoid(),
        )

    # ── Encoder ───────────────────────────────

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (μ, log σ²)."""
        feat = self.encoder(x).flatten(1)
        return self.fc_mu(feat), self.fc_logvar(feat)

    def reparameterise(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """z = μ + ε·σ  (deterministic at inference)."""
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + torch.randn_like(std) * std
        return mu

    # ── Decoder ───────────────────────────────

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        feat = self.fc_decode(z).view(-1, 256, 4, 32)
        return self.decoder(feat)

    # ── Forward ───────────────────────────────

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, μ, log σ²)."""
        mu, logvar = self.encode(x)
        z          = self.reparameterise(mu, logvar)
        recon      = self.decode(z)
        return recon, mu, logvar

    # ── Anomaly score ─────────────────────────

    def anomaly_score(self, x: torch.Tensor,
                      beta: float = 1.0) -> torch.Tensor:
        """
        Per-image anomaly score = MSE(recon, x) + β·KL.
        Shape: (B,)
        """
        recon, mu, logvar = self.forward(x)
        recon_err = (x - recon).pow(2).mean(dim=[1, 2, 3])
        kl        = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()
                            ).sum(dim=1)
        return recon_err + beta * kl
