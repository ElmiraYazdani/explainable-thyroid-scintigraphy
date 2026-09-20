"""
Class-conditional DCGAN for 128x128 grayscale thyroid scintigraphy images.

A single generator/discriminator pair is conditioned on the 4-class group
label (see LABEL_GROUP_MAP in classification_different_models_kfold_oversampling.py)
rather than training four separate GANs. Pooling all classes into one model
lets the shared convolutional backbone learn general scintigraphy texture and
uptake-pattern statistics from the full training set, while the class
embedding steers the minority classes -- this tends to be much more stable
than training a class-specific GAN on as few as ~300 images.

Architecture notes:
- Spectral normalization on every discriminator conv layer for Lipschitz
  control, which is the standard stabilizer for small-batch, small-dataset
  GAN training.
- Hinge loss (rather than the original DCGAN BCE loss), which empirically
  gives better gradient behavior at low sample counts.
- Label conditioning: the generator concatenates a learned class embedding to
  the latent noise vector before the first transposed convolution; the
  discriminator projects the class embedding to a spatial map and
  concatenates it as an extra input channel (standard conditional-DCGAN
  conditioning-by-concatenation).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 128, num_classes: int = 4, embed_dim: int = 50, base_channels: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        in_dim = latent_dim + embed_dim

        c = base_channels
        self.net = nn.Sequential(
            # 1x1 -> 4x4
            nn.ConvTranspose2d(in_dim, c * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(True),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(c, c // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c // 2),
            nn.ReLU(True),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(c // 2, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_vec = self.label_embed(labels)
        x = torch.cat([z, label_vec], dim=1).unsqueeze(-1).unsqueeze(-1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes: int = 4, base_channels: int = 64, image_size: int = 128):
        super().__init__()
        self.image_size = image_size
        self.label_embed = nn.Embedding(num_classes, image_size * image_size)

        c = base_channels
        self.net = nn.Sequential(
            # in_channels = 1 (image) + 1 (label map) = 2
            spectral_norm(nn.Conv2d(2, c, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64
            spectral_norm(nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32
            spectral_norm(nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            spectral_norm(nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            spectral_norm(nn.Conv2d(c * 8, c * 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4
            spectral_norm(nn.Conv2d(c * 8, 1, kernel_size=4, stride=1, padding=0)),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_map = self.label_embed(labels).view(-1, 1, self.image_size, self.image_size)
        x = torch.cat([x, label_map], dim=1)
        out = self.net(x)
        return out.view(-1)


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
