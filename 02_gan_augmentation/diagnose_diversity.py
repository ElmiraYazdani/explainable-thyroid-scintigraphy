#!/usr/bin/env python3
"""
Standalone utility: check whether a GAN checkpoint has collapsed (i.e. the
generator ignoring its latent noise input and producing near-identical
images per class), without running the full augmentation pipeline.

Usage:
    python diagnose_diversity.py \
        --checkpoint ./gan_checkpoints/checkpoints/checkpoint_epoch_0400.pt \
        --data-root ../cf_dataset_preprocessed_split/train

Useful for comparing multiple saved checkpoints (every train_gan.py run now
keeps one per --sample-every interval in <output-dir>/checkpoints/) to find
the best one to hand to generate_augmented_dataset.py, e.g.:

    for f in gan_checkpoints/checkpoints/checkpoint_epoch_*.pt; do
        echo "== $f =="
        python diagnose_diversity.py --checkpoint "$f" --data-root ../cf_dataset_preprocessed_split/train
    done
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from diversity import format_diversity_report, is_collapsed, measure_generator_diversity, measure_real_diversity
from gan_models import Generator

LABEL_GROUP_MAP = {1: 1, 3: 1, 2: 2, 5: 2, 6: 2, 4: 3, 7: 4, 8: 4}
GROUP_IDS = sorted(set(LABEL_GROUP_MAP.values()))
GROUP_TO_IDX = {g: i for i, g in enumerate(GROUP_IDS)}
IDX_TO_GROUP = {i: g for g, i in GROUP_TO_IDX.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check a GAN checkpoint for mode collapse.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("../cf_dataset_preprocessed_split/train"),
                    help="Directory containing images/ and train_labels.csv (used as the real-data diversity baseline).")
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--collapse-warning-ratio", type=float, default=0.15)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    gan_args = ckpt["args"]
    netG = Generator(latent_dim=gan_args["latent_dim"], num_classes=len(GROUP_IDS)).to(device)
    netG.load_state_dict(ckpt["generator_state_dict"])
    netG.eval()
    print(f"Loaded {args.checkpoint} (trained for {ckpt['epoch']} epochs).")

    train_df = pd.read_csv(args.data_root / "train_labels.csv")
    train_df["label_raw"] = train_df["label"].astype(float).astype(int)
    train_df["group"] = train_df["label_raw"].map(LABEL_GROUP_MAP)

    real_diversity = measure_real_diversity(
        train_df, args.data_root / "images", group_col="group", group_to_idx=GROUP_TO_IDX,
        image_size=gan_args["image_size"], n_samples=args.n_samples, seed=0,
    )
    gen_diversity = measure_generator_diversity(
        netG, gan_args["latent_dim"], len(GROUP_IDS), device, n_samples=args.n_samples, seed=0,
    )
    report, ratios = format_diversity_report(gen_diversity, real_diversity, IDX_TO_GROUP)
    print(report)

    collapsed = [IDX_TO_GROUP[i] for i, r in zip(sorted(gen_diversity), ratios) if is_collapsed(r, args.collapse_warning_ratio)]
    if collapsed:
        print(f"\nCOLLAPSED: group(s) {collapsed} are below {args.collapse_warning_ratio:.0%} of real-data diversity.")
    else:
        print("\nOK: no collapsed classes detected.")


if __name__ == "__main__":
    main()
