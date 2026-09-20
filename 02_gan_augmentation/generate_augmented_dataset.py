#!/usr/bin/env python3
"""
Use a trained conditional DCGAN checkpoint to build a GAN-augmented copy of
the thyroid scintigraphy dataset in a brand-new directory. The original
dataset (cf_dataset_preprocessed, cf_dataset_preprocessed_split, the
k-fold directories, etc.) is never modified -- this script only reads from
the existing training split and writes to a new output directory.

What it does:
  1. Copies the real training images and the real, untouched test split into
     the new directory (so the new directory is self-contained and can be
     pointed to directly from the existing classification script).
  2. For each of the 4 diagnostic groups, generates enough synthetic images
     to bring that group's training count up to the current majority class
     count (matching what oversample_to_majority() targets today, but with
     GAN-synthesized images instead of duplicated real ones).
  3. Assigns each synthetic image one of the original 8 raw diagnostic
     labels by sampling *within* its group in the same proportion as the
     real training data (e.g. group 1 = {label 1, label 3} at roughly a 3:1
     ratio in the real data, so synthetic group-1 images are labelled 1 or 3
     in that same ratio). This means the new train_labels.csv is readable by
     the existing LABEL_GROUP_MAP-based pipeline with zero code changes --
     it only needs --image-root / --data-root pointed at the new folder.
  4. Writes a `source` column (real / synthetic) into the labels CSV so any
     downstream analysis can filter synthetic images back out if needed, and
     a JSON summary of exactly how many images were added per class.

Usage:
    python generate_augmented_dataset.py \
        --checkpoint ./gan_checkpoints/checkpoint_latest.pt \
        --data-root ../cf_dataset_preprocessed_split \
        --output-dir ../cf_dataset_preprocessed_gan_augmented

Before generating the full dataset, this script runs a diversity pre-flight
check against the real training data (see diversity.py) and refuses to
proceed if any class looks collapsed -- i.e. the generator producing
(near-)identical images regardless of the random noise it's given. This is
what caused every synthetic image in a class to look the same in an earlier
run of this pipeline; generating thousands of duplicate images from a
collapsed checkpoint would silently poison the augmented dataset. Retrain
with train_gan.py (which now includes an anti-collapse loss term and reports
this same diversity metric during training) and pick a checkpoint that
passes the check, or pass --force to proceed anyway at your own risk (e.g.
for debugging).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from diversity import format_diversity_report, is_collapsed, measure_generator_diversity, measure_real_diversity
from gan_models import Generator

LABEL_GROUP_MAP = {1: 1, 3: 1, 2: 2, 5: 2, 6: 2, 4: 3, 7: 4, 8: 4}
GROUP_IDS = sorted(set(LABEL_GROUP_MAP.values()))
GROUP_TO_IDX = {g: i for i, g in enumerate(GROUP_IDS)}
IDX_TO_GROUP = {i: g for g, i in GROUP_TO_IDX.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a GAN-augmented copy of the training split.")
    p.add_argument("--checkpoint", type=Path, default=Path("./gan_checkpoints/checkpoint_latest.pt"))
    p.add_argument("--data-root", type=Path, default=Path("../cf_dataset_preprocessed_split"),
                    help="Directory containing train/ (images + train_labels.csv) and test/ (images + test_labels.csv).")
    p.add_argument("--output-dir", type=Path, default=Path("../cf_dataset_preprocessed_gan_augmented"),
                    help="New directory to create. Must not already exist, to avoid overwriting a previous run.")
    p.add_argument("--target-per-class", type=int, default=None,
                    help="If set, every class is topped up to this exact count instead of the current majority class count.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--collapse-warning-ratio", type=float, default=0.15,
                    help="Minimum acceptable ratio of generated to real within-class diversity. "
                         "Below this, the checkpoint is treated as collapsed.")
    p.add_argument("--force", action="store_true",
                    help="Proceed even if the diversity pre-flight check finds a collapsed class.")
    return p.parse_args()


def load_generator(checkpoint_path: Path, device: torch.device) -> tuple[Generator, dict, int]:
    # weights_only=True is the safe default from PyTorch 2.6 onward; our checkpoints only
    # ever contain tensors and plain Python types, so this always succeeds.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = ckpt["args"]
    netG = Generator(latent_dim=args["latent_dim"], num_classes=len(GROUP_IDS)).to(device)
    netG.load_state_dict(ckpt["generator_state_dict"])
    netG.eval()
    return netG, args, ckpt["epoch"]


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    # img_tensor is in [-1, 1] (Tanh output), single channel, shape (1, H, W)
    arr = img_tensor.squeeze(0).clamp(-1, 1).cpu().numpy()
    arr = ((arr + 1.0) * 127.5).round().astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.output_dir.exists():
        raise FileExistsError(
            f"{args.output_dir} already exists. Refusing to overwrite -- choose a new "
            f"--output-dir (e.g. append a version suffix) or remove it manually first."
        )

    device = torch.device(args.device)
    netG, gan_args, trained_epochs = load_generator(args.checkpoint, device)
    image_size = gan_args["image_size"]
    latent_dim = gan_args["latent_dim"]
    print(f"Loaded generator from {args.checkpoint} (trained for {trained_epochs} epochs).")

    train_root = args.data_root / "train"
    test_root = args.data_root / "test"
    train_labels_path = train_root / "train_labels.csv"
    test_labels_path = test_root / "test_labels.csv"

    train_df = pd.read_csv(train_labels_path)
    train_df["label_raw"] = train_df["label"].astype(float).astype(int)
    train_df["group"] = train_df["label_raw"].map(LABEL_GROUP_MAP)
    train_df["source"] = "real"

    group_counts = train_df["group"].value_counts()
    target_count = args.target_per_class or int(group_counts.max())
    print(f"Target count per class: {target_count} (current counts: {group_counts.sort_index().to_dict()})")

    # --- Diversity pre-flight check ---
    # Confirms the generator actually responds to its latent noise input before we spend time
    # generating (and the user spends time training on) potentially thousands of near-duplicate
    # images per class.
    print("\nRunning diversity pre-flight check against the real training data...")
    real_diversity = measure_real_diversity(
        train_df, train_root / "images", group_col="group", group_to_idx=GROUP_TO_IDX,
        image_size=image_size, n_samples=32, seed=args.seed,
    )
    gen_diversity = measure_generator_diversity(
        netG, latent_dim, len(GROUP_IDS), device, n_samples=32, seed=args.seed,
    )
    report, ratios = format_diversity_report(gen_diversity, real_diversity, IDX_TO_GROUP)
    print(report)
    collapsed_classes = [
        IDX_TO_GROUP[i] for i, r in zip(sorted(gen_diversity), ratios)
        if is_collapsed(r, args.collapse_warning_ratio)
    ]
    if collapsed_classes and not args.force:
        raise SystemExit(
            f"\nAborting: group(s) {collapsed_classes} show collapsed diversity "
            f"(< {args.collapse_warning_ratio:.0%} of real-data diversity) -- the generator is producing "
            f"(near-)identical images for these classes regardless of input noise. Generating from this "
            f"checkpoint would fill the augmented dataset with near-duplicate images.\n"
            f"Retrain with train_gan.py (now includes a mode-seeking anti-collapse loss term), or pick a "
            f"different, earlier checkpoint from <output-dir>/checkpoints/ where diversity was still healthy "
            f"(check training_history.json). Pass --force to proceed anyway at your own risk."
        )
    elif collapsed_classes:
        print(f"\n--force set: proceeding despite collapsed diversity in group(s) {collapsed_classes}.")
    else:
        print("\nDiversity check passed for all classes.")

    # Within-group raw-label proportions from the real training data, used to assign
    # a plausible raw sub-label to each synthetic image so it flows through the
    # existing LABEL_GROUP_MAP unmodified.
    within_group_label_probs: dict[int, tuple[list[int], list[float]]] = {}
    for group in GROUP_IDS:
        sub = train_df[train_df["group"] == group]["label_raw"]
        counts = sub.value_counts().sort_index()
        labels = counts.index.tolist()
        probs = (counts / counts.sum()).tolist()
        within_group_label_probs[group] = (labels, probs)

    # --- Set up output directory ---
    out_train_images = args.output_dir / "train" / "images"
    out_test_images = args.output_dir / "test" / "images"
    out_train_images.mkdir(parents=True, exist_ok=True)
    out_test_images.mkdir(parents=True, exist_ok=True)

    print("Copying real training images (unmodified) into the new directory...")
    for fname in train_df["image_file"]:
        shutil.copy2(train_root / "images" / fname, out_train_images / fname)

    print("Copying the untouched test split into the new directory (no synthetic images are added to test)...")
    shutil.copytree(test_root / "images", out_test_images, dirs_exist_ok=True)
    shutil.copy2(test_labels_path, args.output_dir / "test" / "test_labels.csv")

    # --- Generate synthetic images per group ---
    synthetic_rows = []
    summary = []
    next_id = 0
    with torch.no_grad():
        for group in GROUP_IDS:
            current_count = int(group_counts.get(group, 0))
            n_needed = max(0, target_count - current_count)
            labels_pool, probs_pool = within_group_label_probs[group]
            generated = 0
            group_idx = GROUP_TO_IDX[group]
            while generated < n_needed:
                bs = min(args.batch_size, n_needed - generated)
                z = torch.randn(bs, latent_dim, device=device)
                labels_t = torch.full((bs,), group_idx, dtype=torch.long, device=device)
                fakes = netG(z, labels_t)
                for i in range(bs):
                    raw_label = int(rng.choice(labels_pool, p=probs_pool))
                    fname = f"GAN_g{group}_{next_id:05d}.png"
                    next_id += 1
                    tensor_to_pil(fakes[i]).save(out_train_images / fname)
                    synthetic_rows.append(
                        {
                            "image_file": fname,
                            "label": float(raw_label),
                            "center": np.nan,
                            "age": np.nan,
                            "sex": np.nan,
                            "label_raw": raw_label,
                            "group": group,
                            "source": "synthetic",
                        }
                    )
                generated += bs
            summary.append(
                {
                    "group": group,
                    "real_count": current_count,
                    "synthetic_added": n_needed,
                    "final_count": current_count + n_needed,
                }
            )
            print(f"Group {group}: {current_count} real + {n_needed} synthetic -> {current_count + n_needed}")

    synthetic_df = pd.DataFrame(synthetic_rows)
    combined_df = pd.concat([train_df, synthetic_df], ignore_index=True) if len(synthetic_df) else train_df
    combined_df = combined_df.drop(columns=["label_raw"])  # keep same columns as original + group/source
    combined_df.to_csv(args.output_dir / "train" / "train_labels.csv", index=False)

    with open(args.output_dir / "augmentation_summary.json", "w") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "target_per_class": target_count,
                "per_group_summary": summary,
                "total_real": int(len(train_df)),
                "total_synthetic": int(len(synthetic_df)),
                "total_final": int(len(combined_df)),
            },
            f,
            indent=2,
        )

    print(f"\nDone. New GAN-augmented dataset written to: {args.output_dir}")
    print(f"  train/images/          ({len(combined_df)} images: {len(train_df)} real + {len(synthetic_df)} synthetic)")
    print(f"  train/train_labels.csv (adds 'group' and 'source' columns; 'label' column still usable as-is)")
    print(f"  test/  (unchanged copy of the real test split -- no synthetic images)")
    print(f"  augmentation_summary.json")


if __name__ == "__main__":
    main()
