#!/usr/bin/env python3
"""Backfill Cohen's kappa and MCC into the MoE CNN-fusion metrics files.

The training run (`moe_cnn_fusion.py`) saved per-image predictions but never
computed kappa or MCC. Both are pure functions of the confusion matrix, and
`moe_cnn_fusion.py` builds its prediction CSV from the *same* `all_labels` /
`all_predictions` arrays that produce the stored `acc` and the classification
report (see moe_cnn_fusion.py:1130-1177). Recomputing from those CSVs therefore
reproduces exactly the values the script would have written inline - no
retraining, no approximation.

The edit is strictly additive:
  * two new columns, `kappa_val` and `mcc`, placed immediately after `acc`
    (they are ensemble-level metrics, so they belong with `acc`, not with the
    per-expert accuracies);
  * four new columns on the summary file;
  * every pre-existing column keeps its position, bytes and value.

CSVs are edited as text rather than round-tripped through pandas so that
untouched columns cannot be reformatted. The two JSON files are regenerated the
way `moe_cnn_fusion.py` generates them (`to_json(orient="records", indent=2)`)
and then checked field-by-field against the originals.

Run with `--check` to verify without writing anything.

Naming follows the rest of the project (`Final Experiments 5 Folds`,
`Pretrained Naive Replication 5 Folds`, ...), which use `kappa_val` and `mcc`.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef

HERE = Path(__file__).resolve().parent
MODEL = "MixtureOfExpertsCNNFusion"
RUNS = {
    "DenseNet121": "moe_cnn_fusion_densenet121_5fold_outputs",
    "MobileNetV2": "moe_cnn_fusion_mobilenetv2_5fold_outputs",
    "VGG11": "moe_cnn_fusion_vgg11_5fold_outputs",
}
FOLDS = [1, 2, 3, 4, 5]
LABELS = [0, 1, 2, 3]
NEW_COLS = ["kappa_val", "mcc"]
ANCHOR_COL = "acc"  # new columns are inserted immediately after this one


def fold_values(run_dir: Path) -> pd.DataFrame:
    """kappa + MCC per fold, recomputed from the saved per-image predictions."""
    rows = []
    for fold in FOLDS:
        path = run_dir / "predictions" / MODEL / f"{MODEL}_fold_{fold}_predictions.csv"
        df = pd.read_csv(path)
        y_true = df["true_idx"].to_numpy(dtype=int)
        y_pred = df["pred_idx"].to_numpy(dtype=int)
        rows.append(
            {
                "fold": fold,
                "n": len(df),
                "kappa_val": cohen_kappa_score(y_true, y_pred, labels=LABELS),
                "mcc": matthews_corrcoef(y_true, y_pred),
                "acc_recomputed": accuracy_score(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


def assert_predictions_match_stored_acc(run_dir: Path, vals: pd.DataFrame) -> None:
    """Guard: the predictions must be the ones behind the stored `acc`."""
    stored = pd.read_csv(run_dir / "metrics" / f"{MODEL}_fold_metrics.csv")
    stored = stored[stored["fold"].astype(str).str.isdigit()]
    for _, row in vals.iterrows():
        want = float(stored.loc[stored["fold"].astype(int) == row["fold"], "acc"].iloc[0])
        got = float(row["acc_recomputed"])
        if abs(want - got) > 1e-12:
            raise SystemExit(
                f"REFUSING TO WRITE - {run_dir.name} fold {int(row['fold'])}: "
                f"accuracy from predictions ({got!r}) != stored acc ({want!r}). "
                "The prediction CSV does not correspond to the saved metrics."
            )


def append_csv_columns(
    path: Path, values_by_fold: Dict[str, List[float]], write: bool
) -> tuple[str, str]:
    """Append NEW_COLS after ANCHOR_COL, leaving every existing byte untouched.

    `values_by_fold` maps the row's `fold` cell (e.g. "1", "mean", "std") to the
    list of new values for that row.
    """
    original_text = path.read_text(encoding="utf-8")
    lines = original_text.splitlines()
    header = lines[0].split(",")
    if set(NEW_COLS) & set(header):
        return original_text, f"  [skip] {path.name}: already has {NEW_COLS}"

    fold_at = header.index("fold")
    insert_at = header.index(ANCHOR_COL) + 1
    out = [",".join(header[:insert_at] + NEW_COLS + header[insert_at:])]
    for line in lines[1:]:
        if not line.strip():
            continue
        cells = line.split(",")
        key = cells[fold_at]
        if key not in values_by_fold:
            raise SystemExit(f"REFUSING TO WRITE - {path.name}: no values for fold {key!r}")
        new = [repr(float(v)) for v in values_by_fold[key]]
        out.append(",".join(cells[:insert_at] + new + cells[insert_at:]))

    text = "\n".join(out) + "\n"
    if write:
        path.write_text(text, encoding="utf-8")
    return text, f"  {'wrote' if write else 'would write'} {path.name}  (+{', '.join(NEW_COLS)})"


def regenerate_json(csv_text: str, json_path: Path, write: bool) -> str:
    """Rebuild the JSON from the updated CSV exactly as moe_cnn_fusion.py does.

    Verifies that every field already present in the original JSON is unchanged
    before overwriting, so the only difference is the two added keys.
    """
    df = pd.read_csv(io.StringIO(csv_text))
    # moe_cnn_fusion.py writes fold as int for fold rows and as "mean"/"std" for
    # the summary rows; a CSV round-trip would stringify the ints.
    df["fold"] = [int(v) if str(v).isdigit() else v for v in df["fold"]]
    rebuilt = json.loads(df.to_json(orient="records", indent=2))

    original = json.loads(json_path.read_text(encoding="utf-8"))
    if len(original) != len(rebuilt):
        raise SystemExit(f"REFUSING TO WRITE - {json_path.name}: record count changed")
    for before, after in zip(original, rebuilt):
        for key, value in before.items():
            if after.get(key) != value:
                raise SystemExit(
                    f"REFUSING TO WRITE - {json_path.name}: existing field {key!r} "
                    f"changed from {value!r} to {after.get(key)!r}"
                )
        extra = set(after) - set(before)
        if extra != set(NEW_COLS):
            raise SystemExit(f"REFUSING TO WRITE - {json_path.name}: unexpected new keys {extra}")

    if write:
        json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    return f"  {'wrote' if write else 'would write'} {json_path.name}  (+{', '.join(NEW_COLS)})"


def append_summary_columns(path: Path, vals: pd.DataFrame, write: bool) -> str:
    """Add kappa_val_mean/std and mcc_mean/std, matching the script's ddof=1."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    new_cols = ["kappa_val_mean", "kappa_val_std", "mcc_mean", "mcc_std"]
    if set(new_cols) & set(header):
        return f"  [skip] {path.name}: already has the summary columns"
    new_vals = [
        vals["kappa_val"].mean(),
        vals["kappa_val"].std(ddof=1),
        vals["mcc"].mean(),
        vals["mcc"].std(ddof=1),
    ]
    out = [",".join(header + new_cols)]
    for line in lines[1:]:
        if not line.strip():
            continue
        out.append(",".join(line.split(",") + [repr(float(v)) for v in new_vals]))
    text = "\n".join(out) + "\n"
    if write:
        path.write_text(text, encoding="utf-8")
    return f"  {'wrote' if write else 'would write'} {path.name}  (+{', '.join(new_cols)})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Verify only; write nothing.")
    args = parser.parse_args()
    write = not args.check

    for backbone, run_name in RUNS.items():
        run_dir = HERE / run_name
        metrics_dir = run_dir / "metrics"
        print(f"\n{backbone}  ({run_name})")

        vals = fold_values(run_dir)
        assert_predictions_match_stored_acc(run_dir, vals)
        print(f"  verified: predictions reproduce stored acc on all {len(vals)} folds")

        by_fold: Dict[str, List[float]] = {
            str(int(r["fold"])): [r["kappa_val"], r["mcc"]] for _, r in vals.iterrows()
        }
        by_fold["mean"] = [vals["kappa_val"].mean(), vals["mcc"].mean()]
        by_fold["std"] = [vals["kappa_val"].std(ddof=1), vals["mcc"].std(ddof=1)]

        text, msg = append_csv_columns(metrics_dir / "all_models_all_folds_metrics.csv", by_fold, write)
        print(msg)
        print(regenerate_json(text, metrics_dir / "all_models_all_folds_metrics.json", write))

        text, msg = append_csv_columns(metrics_dir / f"{MODEL}_fold_metrics.csv", by_fold, write)
        print(msg)
        print(regenerate_json(text, metrics_dir / f"{MODEL}_fold_metrics_with_mean_std.json", write))

        print(append_summary_columns(metrics_dir / "all_models_summary_mean_std.csv", vals, write))

        print(f"  kappa = {vals['kappa_val'].mean():.4f} +/- {vals['kappa_val'].std(ddof=1):.4f}   "
              f"MCC = {vals['mcc'].mean():.4f} +/- {vals['mcc'].std(ddof=1):.4f}")

    print("\nDone." if write else "\nCheck only - nothing written.")


if __name__ == "__main__":
    main()
