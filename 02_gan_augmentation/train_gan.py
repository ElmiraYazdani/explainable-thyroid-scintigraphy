#!/usr/bin/env python3
"""
Train a class-conditional DCGAN on the thyroid scintigraphy training split,
to be used afterwards by generate_augmented_dataset.py to synthesize
additional minority-class images.

This script only ever reads from cf_dataset_preprocessed_split/train -- the
held-out test split and the original dataset directory are never touched, so
there is no risk of test-set leakage into the generator and no risk of
overwriting existing data.

Usage:
    python train_gan.py \
        --data-root ../cf_dataset_preprocessed_split/train \
        --output-dir ./gan_checkpoints \
        --epochs 800

Run `python train_gan.py --help` for all options.

IMPORTANT -- diversity, not just image quality: the sample grids in
<output-dir>/samples/ can *look* like plausible scintigraphy scans while
still being mode-collapsed -- i.e. the generator producing (near) the same
one image per class regardless of the random noise vector it's given. This
is easy to miss by eye at a quick glance (each row can look "fine" on its
own) and is a very common failure mode on small per-class datasets like this
one (as few as 308 real images for the smallest class). This script now:
  1. Adds a mode-seeking regularization term to the generator loss (Mao et
     al., "Mode Seeking Generative Adversarial Networks for Diverse Image
     Synthesis", CVPR 2019) that directly penalizes the generator for
     producing similar outputs from dissimilar latent codes.
  2. Logs a quantitative diversity score every --sample-every epochs
     (mean pairwise pixel distance among same-class generated images,
     reported as a ratio against the real training data's own diversity),
     printed to the console and saved into training_history.json, so
     collapse shows up as a number over training time, not just something
     you have to eyeball in an image grid.
  3. Saves a *versioned* checkpoint at every --sample-every interval
     (checkpoint_epoch_XXXX.pt) instead of only overwriting a single
     checkpoint_latest.pt -- if the model collapses late in training, you
     can still fall back to an earlier, healthier checkpoint instead of
     having to retrain from scratch.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from diffaug import diff_augment
from diversity import format_diversity_report, is_collapsed, measure_generator_diversity, measure_real_diversity
from gan_models import Discriminator, Generator, hinge_d_loss, hinge_g_loss, weights_init

# Same grouping used by classification_different_models_kfold_oversampling.py
# 1 -> Diffuse Goiter group, 2 -> Tumoral group, 3 -> Thyroiditis group (label 4 only),
# 4 -> Normal group. Kept identical so downstream scripts require no changes.
LABEL_GROUP_MAP = {1: 1, 3: 1, 2: 2, 5: 2, 6: 2, 4: 3, 7: 4, 8: 4}
GROUP_IDS = sorted(set(LABEL_GROUP_MAP.values()))  # [1, 2, 3, 4]
GROUP_TO_IDX = {g: i for i, g in enumerate(GROUP_IDS)}  # 1->0, 2->1, 3->2, 4->3
IDX_TO_GROUP = {i: g for g, i in GROUP_TO_IDX.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ThyroidGanDataset(Dataset):
    """Loads images from the training split labels CSV, grouped into the
    4-class scheme, for GAN training."""

    def __init__(self, labels_csv: Path, image_dir: Path, image_size: int = 128):
        df = pd.read_csv(labels_csv)
        df["label_raw"] = df["label"].astype(float).astype(int)
        unknown = sorted(set(df["label_raw"].unique()) - set(LABEL_GROUP_MAP.keys()))
        if unknown:
            raise ValueError(f"Unknown raw labels in {labels_csv}: {unknown}")
        df["group"] = df["label_raw"].map(LABEL_GROUP_MAP)
        df["group_idx"] = df["group"].map(GROUP_TO_IDX)
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),  # -> [-1, 1] to match Tanh output
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(self.image_dir / str(row["image_file"])).convert("L")
        image = self.transform(image)
        return image, int(row["group_idx"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a conditional DCGAN for thyroid scintigraphy GAN augmentation.")
    p.add_argument("--data-root", type=Path, default=Path("../cf_dataset_preprocessed_split/train"),
                    help="Directory containing images/ and train_labels.csv (the training split only).")
    p.add_argument("--labels-csv", type=Path, default=None,
                    help="Override path to the labels CSV. Defaults to <data-root>/train_labels.csv.")
    p.add_argument("--images-dir", type=Path, default=None,
                    help="Override path to the images directory. Defaults to <data-root>/images.")
    p.add_argument("--output-dir", type=Path, default=Path("./gan_checkpoints"))
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--diffaug-policy", type=str, default="color,translation,cutout")
    p.add_argument("--lambda-ms", type=float, default=1.0,
                    help="Weight of the mode-seeking diversity loss on the generator. "
                         "Set to 0 to disable (not recommended -- this is what fixes collapse). "
                         "Raise it (e.g. 2-3) if diversity_ratio in the logs stays low.")
    p.add_argument("--sample-every", type=int, default=25,
                    help="Epochs between saved sample grids, versioned checkpoints, and diversity checks.")
    p.add_argument("--collapse-warning-ratio", type=float, default=0.15,
                    help="If a class's generated diversity falls below this fraction of the real "
                         "data's diversity, a collapse warning is printed for that class.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    labels_csv = args.labels_csv or (args.data_root / "train_labels.csv")
    images_dir = args.images_dir or (args.data_root / "images")

    output_dir = args.output_dir
    samples_dir = output_dir / "samples"
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    dataset = ThyroidGanDataset(labels_csv, images_dir, image_size=args.image_size)
    print(f"Loaded {len(dataset)} training images across {len(GROUP_IDS)} classes.")
    group_counts = dataset.df["group"].value_counts().sort_index()
    print("Per-group counts (raw group id -> count):")
    print(group_counts.to_string())

    # Reference diversity baseline from the real data, computed once, used to
    # judge whether the generator's diversity is collapsing during training.
    real_diversity = measure_real_diversity(
        dataset.df, images_dir, group_col="group", group_to_idx=GROUP_TO_IDX,
        image_size=args.image_size, n_samples=32, seed=args.seed,
    )
    print("Real-data per-class diversity baseline (mean pairwise L1 per pixel):")
    for idx, score in sorted(real_diversity.items()):
        print(f"  class {idx} (group {IDX_TO_GROUP[idx]}): {score:.5f}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    netG = Generator(latent_dim=args.latent_dim, num_classes=len(GROUP_IDS)).to(device)
    netD = Discriminator(num_classes=len(GROUP_IDS), image_size=args.image_size).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    optG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    # Fixed noise + fixed labels (one per class, repeated) for consistent sample grids across epochs.
    fixed_n_per_class = 8
    fixed_noise = torch.randn(fixed_n_per_class * len(GROUP_IDS), args.latent_dim, device=device)
    fixed_labels = torch.arange(len(GROUP_IDS), device=device).repeat_interleave(fixed_n_per_class)

    ms_eps = 1e-5
    history = []
    for epoch in range(1, args.epochs + 1):
        d_losses, g_losses, ms_losses = [], [], []
        for real_images, labels in loader:
            real_images = real_images.to(device)
            labels = labels.to(device)
            bs = real_images.size(0)

            # --- Discriminator step ---
            netD.zero_grad()
            z = torch.randn(bs, args.latent_dim, device=device)
            fake_images = netG(z, labels)

            real_aug = diff_augment(real_images, policy=args.diffaug_policy)
            fake_aug = diff_augment(fake_images.detach(), policy=args.diffaug_policy)

            real_logits = netD(real_aug, labels)
            fake_logits = netD(fake_aug, labels)
            d_loss = hinge_d_loss(real_logits, fake_logits)
            d_loss.backward()
            optD.step()

            # --- Generator step ---
            netG.zero_grad()
            z1 = torch.randn(bs, args.latent_dim, device=device)
            z2 = torch.randn(bs, args.latent_dim, device=device)
            fake1 = netG(z1, labels)
            fake2 = netG(z2, labels)

            fake_aug = diff_augment(fake1, policy=args.diffaug_policy)
            fake_logits = netD(fake_aug, labels)
            g_adv_loss = hinge_g_loss(fake_logits)

            # Mode-seeking regularization: penalize the generator when two
            # different latent codes (z1, z2) produce near-identical images
            # for the same class. This is the direct fix for the collapse
            # you saw -- without it, the generator has no incentive to let
            # z influence the output at all once it finds one image per
            # class that reliably fools the discriminator.
            img_dist = (fake1 - fake2).abs().mean(dim=[1, 2, 3])
            z_dist = (z1 - z2).abs().mean(dim=1)
            loss_ms = (z_dist / (img_dist + ms_eps)).mean()

            g_loss = g_adv_loss + args.lambda_ms * loss_ms
            g_loss.backward()
            optG.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_adv_loss.item())
            ms_losses.append(loss_ms.item())

        mean_d, mean_g, mean_ms = float(np.mean(d_losses)), float(np.mean(g_losses)), float(np.mean(ms_losses))
        epoch_record = {"epoch": epoch, "d_loss": mean_d, "g_loss": mean_g, "ms_loss": mean_ms}
        print(f"Epoch {epoch:4d}/{args.epochs} | D loss: {mean_d:.4f} | G loss: {mean_g:.4f} | MS loss: {mean_ms:.4f}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            netG.eval()
            with torch.no_grad():
                samples = netG(fixed_noise, fixed_labels).cpu()
            vutils.save_image(
                samples, samples_dir / f"epoch_{epoch:04d}.png",
                nrow=fixed_n_per_class, normalize=True, value_range=(-1, 1),
            )

            gen_diversity = measure_generator_diversity(
                netG, args.latent_dim, len(GROUP_IDS), device, n_samples=32, seed=args.seed + epoch,
            )
            report, ratios = format_diversity_report(gen_diversity, real_diversity, IDX_TO_GROUP)
            print(f"Diversity check at epoch {epoch}:\n{report}")
            epoch_record["diversity"] = {
                "gen": gen_diversity,
                "real": real_diversity,
                "mean_ratio": float(np.mean(ratios)) if ratios else None,
            }
            collapsed_classes = [
                IDX_TO_GROUP[i] for i, r in zip(sorted(gen_diversity), ratios)
                if is_collapsed(r, args.collapse_warning_ratio)
            ]
            if collapsed_classes:
                print(
                    f"  WARNING: group(s) {collapsed_classes} show collapsed diversity "
                    f"(< {args.collapse_warning_ratio:.0%} of real-data diversity) at epoch {epoch}."
                )

            netG.train()

            # Store args as plain JSON-safe types (str, int, float) rather than pickling
            # Path objects, so the checkpoint can be reloaded with torch.load(weights_only=True)
            # under PyTorch >= 2.6 without needing to trust/allowlist extra globals.
            safe_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
            ckpt_payload = {
                "epoch": epoch,
                "generator_state_dict": netG.state_dict(),
                "discriminator_state_dict": netD.state_dict(),
                "args": safe_args,
                "group_ids": GROUP_IDS,
            }
            # Versioned checkpoint -- kept permanently, so a later collapse doesn't destroy
            # an earlier, healthier state the way overwriting a single file would.
            torch.save(ckpt_payload, ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt")
            # Convenience pointer to the most recent checkpoint (matches the previous
            # script version's default path expected by generate_augmented_dataset.py).
            torch.save(ckpt_payload, output_dir / "checkpoint_latest.pt")

        history.append(epoch_record)

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Versioned checkpoints saved to {ckpt_dir}, sample grids to {samples_dir}")
    print(
        "Before generating the augmented dataset, check training_history.json (or the console log "
        "above) for the epoch with the best 'mean_ratio' diversity score -- it is not guaranteed to "
        "be the final epoch. Pass that specific checkpoint to generate_augmented_dataset.py via "
        "--checkpoint <output-dir>/checkpoints/checkpoint_epoch_XXXX.pt if it differs from the latest."
    )


if __name__ == "__main__":
    main()
