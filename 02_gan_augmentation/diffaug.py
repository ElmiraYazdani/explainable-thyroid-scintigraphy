"""
Differentiable augmentation for GAN training on small datasets.

Implements the core policies from Zhao et al., "Differentiable Augmentation for
Data-Efficient GAN Training" (NeurIPS 2020): color jitter, translation, and
cutout. Applying the *same* random augmentation to both real and generated
images before they reach the discriminator lets the discriminator's gradient
still flow back through the augmentation into the generator, which
substantially stabilizes GAN training when only a few hundred images per
class are available (our case: 308-822 training images per group).

No external dependency is required -- this is a compact, self-contained
reimplementation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    magnitude = (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)
    return x + magnitude


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    magnitude = (torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5)
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - x_mean) * magnitude + x_mean


def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    shift_x = int(x.size(2) * ratio + 0.5)
    shift_y = int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=(x.size(0), 1), device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=(x.size(0), 1), device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(x.size(2), device=x.device),
        torch.arange(x.size(3), device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + translation_x.unsqueeze(-1) + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y.unsqueeze(-1) + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1)[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=(x.size(0), 1), device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=(x.size(0), 1), device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(cutout_size[0], device=x.device),
        torch.arange(cutout_size[1], device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + offset_x.unsqueeze(-1) - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y.unsqueeze(-1) - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


AUGMENT_FNS = {
    "color": [rand_brightness, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


def diff_augment(x: torch.Tensor, policy: str = "color,translation,cutout") -> torch.Tensor:
    if policy:
        for p in policy.split(","):
            for fn in AUGMENT_FNS[p]:
                x = fn(x)
        x = x.contiguous()
    return x
