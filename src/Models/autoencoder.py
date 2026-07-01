"""
src/models/autoencoder.py
Convolutional Autoencoder baseline.
Input/Output: (B, 1, 64, 512)
Encoder: 1→32→64→128→256  (Conv2d, stride=2 each)
Bottleneck: flatten → FC(latent_dim) → FC(256·4·32) → reshape
Decoder: 256→128→64→32→1  (ConvTranspose2d, stride=2 each)
"""

from typing import Tuple
import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3,
                  stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _deconv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                           stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for iris texture anomaly detection.

    Usage::
        model = ConvAutoencoder(latent_dim=256)
        recon, z = model(x)          # x: (B, 1, 64, 512)
        z        = model.encode(x)   # bottleneck features
        recon    = model.decode(z)   # reconstruction
    """

    FLAT_DIM = 256 * 4 * 32   # feature map size after encoder

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            _conv_block(1,   32),    # → (B, 32,  32, 256)
            _conv_block(32,  64),    # → (B, 64,  16, 128)
            _conv_block(64,  128),   # → (B, 128,  8,  64)
            _conv_block(128, 256),   # → (B, 256,  4,  32)
        )
        self.fc_enc = nn.Linear(self.FLAT_DIM, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.FLAT_DIM)

        self.decoder = nn.Sequential(
            _deconv_block(256, 128),  # → (B, 128,  8,  64)
            _deconv_block(128, 64),   # → (B, 64,  16, 128)
            _deconv_block(64,  32),   # → (B, 32,  32, 256)
            _deconv_block(32,   1),   # → (B,  1,  64, 512)
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x).flatten(1)
        return self.fc_enc(feat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        feat = self.fc_dec(z).view(-1, 256, 4, 32)
        return self.decoder(feat)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z     = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-image mean squared reconstruction error. Shape: (B,)"""
        recon, _ = self.forward(x)
        return (x - recon).pow(2).mean(dim=[1, 2, 3])
