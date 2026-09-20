"""
Utilities for measuring output diversity of the conditional generator, and for
detecting the specific failure mode you ran into: the generator ignoring the
latent noise vector and collapsing to (near) one output image per class,
regardless of z. This is "mode collapse" and is a well-known failure mode for
GANs trained on small per-class datasets (your smallest class has 308 real
training images) without an explicit diversity-preserving term in the loss.

We measure diversity as the mean pairwise L1 pixel distance among a batch of
images (either real or generated) that share the same class label. Reporting
this as a *ratio against the real training data's own diversity* is more
informative than an absolute number, since it says "the GAN is producing only
X% as much within-class variation as the real photographs have" -- 0% is
complete collapse to one canonical image, 100%+ means the GAN is at least as
diverse as the real data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


def pairwise_mean_l1(images: torch.Tensor) -> float:
    """Mean pairwise L1 distance per pixel among a batch of images (N, C, H, W)."""
    n = images.size(0)
    if n < 2:
        return 0.0
    flat = images.reshape(n, -1).float()
    d = torch.cdist(flat, flat, p=1) / flat.size(1)
    iu = torch.triu_indices(n, n, offset=1)
    return d[iu[0], iu[1]].mean().item()


@torch.no_grad()
def measure_generator_diversity(
    netG: torch.nn.Module,
    latent_dim: int,
    num_classes: int,
    device: torch.device,
    n_samples: int = 24,
    seed: int | None = None,
) -> Dict[int, float]:
    """Returns {class_idx: mean pairwise L1 diversity} for freshly sampled
    generator output, one class at a time."""
    was_training = netG.training
    netG.eval()
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    scores = {}
    for class_idx in range(num_classes):
        z = torch.randn(n_samples, latent_dim, device=device, generator=generator if device.type == "cpu" else None)
        labels = torch.full((n_samples,), class_idx, dtype=torch.long, device=device)
        images = netG(z, labels)
        scores[class_idx] = pairwise_mean_l1(images)
    if was_training:
        netG.train()
    return scores


def measure_real_diversity(
    df: pd.DataFrame,
    image_dir: Path,
    group_col: str,
    group_to_idx: Dict[int, int],
    image_size: int = 128,
    n_samples: int = 24,
    seed: int = 0,
) -> Dict[int, float]:
    """Returns {class_idx: mean pairwise L1 diversity} computed on real
    training images, as the reference baseline for 'healthy' diversity."""
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])]
    )
    rng = np.random.default_rng(seed)
    scores = {}
    for group, class_idx in group_to_idx.items():
        sub = df[df[group_col] == group]
        if len(sub) == 0:
            continue
        n = min(n_samples, len(sub))
        chosen = sub.sample(n=n, random_state=int(rng.integers(0, 1_000_000)))
        imgs = []
        for fname in chosen["image_file"]:
            img = Image.open(Path(image_dir) / str(fname)).convert("L")
            imgs.append(transform(img))
        images = torch.stack(imgs, dim=0)
        scores[class_idx] = pairwise_mean_l1(images)
    return scores


def format_diversity_report(gen_scores: Dict[int, float], real_scores: Dict[int, float], idx_to_group: Dict[int, int]) -> str:
    lines = ["class_idx  group  real_diversity  gen_diversity  ratio(gen/real)"]
    ratios = []
    for class_idx in sorted(gen_scores):
        real_d = real_scores.get(class_idx)
        gen_d = gen_scores[class_idx]
        group = idx_to_group.get(class_idx, "?")
        if real_d is not None and real_d > 1e-8:
            ratio = gen_d / real_d
            ratios.append(ratio)
            lines.append(f"{class_idx:<10} {group:<6} {real_d:<15.5f} {gen_d:<14.5f} {ratio:.3f}")
        else:
            lines.append(f"{class_idx:<10} {group:<6} {'n/a':<15} {gen_d:<14.5f} n/a")
    return "\n".join(lines), ratios


def is_collapsed(ratio: float, threshold: float = 0.15) -> bool:
    """A generator producing less than `threshold` fraction of the real
    within-class pixel diversity is considered collapsed for practical
    purposes -- below this, generated images for a class are visually
    near-duplicates of each other."""
    return ratio < threshold
