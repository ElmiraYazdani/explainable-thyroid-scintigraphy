#!/usr/bin/env python3
"""Build a 5-fold CV split of the GAN-augmented train/ pool.

Validation folds contain real images only (source == "real"); synthetic
rows are added to every fold's training set unconditionally. The held-out
cf_dataset_preprocessed_gan_augmented/test/ split is never read here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "cf_dataset_preprocessed_gan_augmented"
TRAIN_CSV = SRC_DIR / "train" / "train_labels.csv"
TRAIN_IMAGES_DIR = SRC_DIR / "train" / "images"
TEST_CSV = SRC_DIR / "test" / "test_labels.csv"
OUT_DIR = ROOT / "cf_dataset_preprocessed_gan_augmented_5fold"
N_SPLITS = 5
SEED = 42
COLUMNS = ["image_file", "label", "center", "age", "sex", "group", "source"]


def main() -> None:
    df = pd.read_csv(TRAIN_CSV, dtype=str)
    df["group"] = df["group"].astype(int)

    real_df = df[df["source"] == "real"].reset_index(drop=True)
    synthetic_df = df[df["source"] == "synthetic"].reset_index(drop=True)
    assert len(real_df) + len(synthetic_df) == len(df)

    # filename-collision check (naming convention says no overlap; verify it)
    overlap = set(real_df["image_file"]) & set(synthetic_df["image_file"])
    if overlap:
        raise ValueError(f"Filename collision between real and synthetic rows: {sorted(overlap)[:10]}")

    test_files = set(pd.read_csv(TEST_CSV, dtype=str)["image_file"])
    leak = test_files & set(df["image_file"])
    if leak:
        raise ValueError(f"train_labels.csv rows overlap with held-out test/: {sorted(leak)[:10]}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_indices = list(skf.split(real_df, real_df["group"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_folds = []
    seen_val_files: set[str] = set()

    for i, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        val_rows = real_df.iloc[val_idx].reset_index(drop=True)
        train_real_rows = real_df.iloc[train_idx].reset_index(drop=True)
        train_rows = pd.concat([train_real_rows, synthetic_df], ignore_index=True)

        assert (val_rows["source"] == "real").all()
        assert set(val_rows["image_file"]).isdisjoint(set(train_rows["image_file"]))
        assert set(val_rows["image_file"]).isdisjoint(set(synthetic_df["image_file"]))
        assert set(val_rows["image_file"]).isdisjoint(test_files)
        seen_val_files.update(val_rows["image_file"])

        fold_dir = OUT_DIR / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_df in [("train", train_rows), ("val", val_rows)]:
            payload = {
                "fold": i,
                "split": split_name,
                "num_rows": len(split_df),
                "image_root": str(TRAIN_IMAGES_DIR),
                "columns": COLUMNS,
                "data": split_df[COLUMNS].to_dict(orient="records"),
            }
            (fold_dir / f"{split_name}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )

        meta_folds.append(
            {
                "fold": i,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "train_real_rows": len(train_real_rows),
                "train_synthetic_rows": len(synthetic_df),
                "train_label_distribution": {
                    str(k): int(v) for k, v in train_rows["group"].value_counts().sort_index().items()
                },
                "val_label_distribution": {
                    str(k): int(v) for k, v in val_rows["group"].value_counts().sort_index().items()
                },
                "train_json": f"cf_dataset_preprocessed_gan_augmented_5fold/fold_{i}/train.json",
                "val_json": f"cf_dataset_preprocessed_gan_augmented_5fold/fold_{i}/val.json",
            }
        )
        print(
            f"fold {i}: train={len(train_rows)} (real={len(train_real_rows)}, "
            f"synthetic={len(synthetic_df)}), val={len(val_rows)} (real-only)"
        )

    assert len(seen_val_files) == len(real_df), (
        f"expected {len(real_df)} unique real images across val folds, got {len(seen_val_files)}"
    )

    metadata = {
        "source_dataset_dir": str(SRC_DIR),
        "source_labels_csv": str(TRAIN_CSV),
        "source_images_dir": str(TRAIN_IMAGES_DIR),
        "columns": COLUMNS,
        "stratify_column": "group",
        "n_splits": N_SPLITS,
        "seed": SEED,
        "num_rows_total": len(df),
        "num_real_rows": len(real_df),
        "num_synthetic_rows": len(synthetic_df),
        "val_source_policy": "real_only",
        "folds": meta_folds,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nWrote metadata + {N_SPLITS} folds to {OUT_DIR}")


if __name__ == "__main__":
    main()
