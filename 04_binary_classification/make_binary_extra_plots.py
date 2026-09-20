#!/usr/bin/env python3
"""Binary-specific analysis figures for the pretrained + GAN-augmented 5-fold run.

These complement (they do not replace) the two figure sets inherited from the
reference experiments:

  * the per-fold figures written by the training script itself, and
  * the mean-across-folds publication figures from
    ``make_pretrained_gan_binary_5fold_mean_plots.py``.

Everything here is derived from the saved prediction/metric CSVs only - no model
is re-run - so it can be regenerated at any time without a GPU.

Figures produced (png + svg):
  1. precision_recall_curves      - PR curves, the honest view under 62/38 skew
  2. calibration_curves           - reliability diagrams + Brier score
  3. threshold_sweep              - sensitivity/specificity/F1/Youden vs cut-off
  4. operating_points             - sensitivity vs specificity per model/fold
  5. subgroup_error_attribution   - which original diagnosis each error came from
  6. fold_stability_heatmap       - metric x fold stability per model
  7. pooled_confusion_matrices    - all folds pooled, counts + row-normalised
  8. experiment_comparison        - this run vs the from-scratch binary baseline
  9. native_vs_collapsed_4class   - dedicated binary head vs collapsing the 4-class
                                    parent's predictions (identical folds -> paired)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_ORDER = ["VGG11", "DenseNet121", "ResNet18", "MobileNetV2", "EfficientNetB0"]
MODEL_COLORS = {
    "VGG11": "#1F77B4",
    "DenseNet121": "#FF7F0E",
    "ResNet18": "#2CA02C",
    "MobileNetV2": "#D62728",
    "EfficientNetB0": "#9467BD",
}
CLASS_NAMES = ["Non-Tumoral", "Tumoral"]
POS_COL = "prob_Tumoral"
GROUP_ORDER = [1, 3, 4, 2]
GROUP_NAMES = {1: "Diffuse goiter", 2: "Tumoral", 3: "Thyroiditis", 4: "Normal"}

# t(0.975, df=4) - the 5-fold CI half-width factor used by the mean-plot script.
T_CRIT_95_N5 = 2.7764451051977987


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=SCRIPT_DIR / "classification_pretrained_replication_5fold_gan_augmented_binary_outputs",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "figures" / "extra")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=PROJECT_ROOT / "Final Experiments 5 Folds Binary" / "outputs",
        help="From-scratch binary experiment used as the comparison baseline.",
    )
    parser.add_argument(
        "--parent-dir",
        type=Path,
        default=PROJECT_ROOT / "Pretrained Naive Replication 5 Folds GAN Augmented"
        / "classification_pretrained_replication_5fold_gan_augmented_outputs",
        help="The 4-class pretrained+GAN parent run, for the paired collapsed comparison.",
    )
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def ci95(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    n = int(np.sum(~np.isnan(arr)))
    if n < 2:
        return float("nan")
    return float(T_CRIT_95_N5 * np.nanstd(arr, ddof=1) / np.sqrt(n))


def save(fig: plt.Figure, output_dir: Path, name: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(output_dir / f"{name}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {name}.png / .svg")


def load_predictions(input_dir: Path, models: List[str]) -> pd.DataFrame:
    frames = []
    for model_name in models:
        model_dir = input_dir / "predictions" / model_name
        for path in sorted(model_dir.glob(f"{model_name}_fold_*_predictions.csv")):
            fold = int(path.stem.split("_fold_")[1].split("_")[0])
            frame = pd.read_csv(path)
            frame["model"] = model_name
            frame["fold"] = fold
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No prediction CSVs under {input_dir / 'predictions'}")
    return pd.concat(frames, ignore_index=True)


def load_fold_metrics(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "metrics" / "all_models_all_folds_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


# --------------------------------------------------------------------------- 1
def plot_precision_recall(preds: pd.DataFrame, models: List[str], output_dir: Path, dpi: int) -> pd.DataFrame:
    """Interpolated mean PR curve per model, plus per-fold AP."""
    recall_grid = np.linspace(0.0, 1.0, 201)
    rows = []
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for model_name in models:
        model_df = preds[preds["model"] == model_name]
        curves = []
        aps = []
        for fold, fold_df in model_df.groupby("fold"):
            y_true = (fold_df["true_idx"] == 1).astype(int).to_numpy()
            y_score = fold_df[POS_COL].to_numpy(dtype=float)
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            # precision_recall_curve returns recall descending; flip for interp.
            curves.append(np.interp(recall_grid, recall[::-1], precision[::-1]))
            ap = average_precision_score(y_true, y_score)
            aps.append(ap)
            rows.append({"model": model_name, "fold": int(fold), "average_precision": ap})
        mean_curve = np.mean(curves, axis=0)
        band = np.array([ci95(np.array([c[i] for c in curves])) for i in range(len(recall_grid))])
        color = MODEL_COLORS[model_name]
        ax.plot(
            recall_grid,
            mean_curve,
            color=color,
            linewidth=2.0,
            label=f"{model_name} (AP={np.mean(aps):.3f} +/- {ci95(np.array(aps)):.3f})",
        )
        ax.fill_between(recall_grid, mean_curve - band, mean_curve + band, color=color, alpha=0.15, linewidth=0)

    prevalence = float((preds[preds["model"] == models[0]]["true_idx"] == 1).mean())
    ax.axhline(prevalence, color="#444444", linestyle="--", linewidth=1.2)
    ax.text(0.02, prevalence + 0.015, f"chance = prevalence ({prevalence:.3f})", fontsize=9, color="#444444")
    ax.set_xlabel("Recall (Sensitivity for Tumoral)")
    ax.set_ylabel("Precision (PPV for Tumoral)")
    ax.set_title("Precision-Recall, Tumoral as positive class\nMean +/- 95% CI across 5 folds")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, output_dir, "precision_recall_curves", dpi)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2
def plot_calibration(preds: pd.DataFrame, models: List[str], output_dir: Path, dpi: int) -> pd.DataFrame:
    """Reliability diagram: does a predicted 0.8 really mean 80% tumoral?"""
    rows = []
    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model_name in zip(axes, models):
        model_df = preds[preds["model"] == model_name]
        for fold, fold_df in model_df.groupby("fold"):
            y_true = (fold_df["true_idx"] == 1).astype(int).to_numpy()
            y_score = fold_df[POS_COL].to_numpy(dtype=float)
            rows.append(
                {
                    "model": model_name,
                    "fold": int(fold),
                    "brier": brier_score_loss(y_true, y_score),
                    "mean_predicted": float(y_score.mean()),
                    "observed_prevalence": float(y_true.mean()),
                }
            )
            try:
                frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
            except ValueError:
                continue
            ax.plot(mean_pred, frac_pos, marker="o", markersize=3.5, linewidth=1.2, alpha=0.75,
                    color=MODEL_COLORS[model_name], label=f"Fold {fold}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        model_brier = np.mean([r["brier"] for r in rows if r["model"] == model_name])
        ax.set_title(f"{model_name}\nBrier = {model_brier:.3f}", fontsize=11)
        ax.set_xlabel("Mean predicted P(Tumoral)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Observed fraction Tumoral")
    fig.suptitle("Calibration (reliability) of the Tumoral probability, per fold", y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, output_dir, "calibration_curves", dpi)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 3
def plot_threshold_sweep(preds: pd.DataFrame, models: List[str], output_dir: Path, dpi: int) -> pd.DataFrame:
    """Every metric in the report assumes a 0.5 cut-off; this shows the cost of that."""
    grid = np.linspace(0.01, 0.99, 99)
    rows = []
    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model_name in zip(axes, models):
        model_df = preds[preds["model"] == model_name]
        sens_all, spec_all, f1_all, j_all = [], [], [], []
        for _, fold_df in model_df.groupby("fold"):
            y_true = (fold_df["true_idx"] == 1).astype(int).to_numpy()
            y_score = fold_df[POS_COL].to_numpy(dtype=float)
            sens, spec, f1s = [], [], []
            for t in grid:
                pred = (y_score >= t).astype(int)
                tp = float(((pred == 1) & (y_true == 1)).sum())
                fp = float(((pred == 1) & (y_true == 0)).sum())
                fn = float(((pred == 0) & (y_true == 1)).sum())
                tn = float(((pred == 0) & (y_true == 0)).sum())
                se = tp / (tp + fn) if (tp + fn) else np.nan
                sp = tn / (tn + fp) if (tn + fp) else np.nan
                pr = tp / (tp + fp) if (tp + fp) else 0.0
                sens.append(se)
                spec.append(sp)
                f1s.append(2 * pr * se / (pr + se) if (pr + se) else 0.0)
            sens_all.append(sens)
            spec_all.append(spec)
            f1_all.append(f1s)
            j_all.append(np.asarray(sens) + np.asarray(spec) - 1.0)

        sens_m = np.nanmean(sens_all, axis=0)
        spec_m = np.nanmean(spec_all, axis=0)
        f1_m = np.nanmean(f1_all, axis=0)
        j_m = np.nanmean(j_all, axis=0)
        best_idx = int(np.nanargmax(j_m))
        rows.append(
            {
                "model": model_name,
                "youden_optimal_threshold": float(grid[best_idx]),
                "sensitivity_at_optimal": float(sens_m[best_idx]),
                "specificity_at_optimal": float(spec_m[best_idx]),
                "youden_j_at_optimal": float(j_m[best_idx]),
                "sensitivity_at_0.5": float(np.interp(0.5, grid, sens_m)),
                "specificity_at_0.5": float(np.interp(0.5, grid, spec_m)),
                "youden_j_at_0.5": float(np.interp(0.5, grid, j_m)),
            }
        )

        ax.plot(grid, sens_m, color="#C44E52", linewidth=2, label="Sensitivity")
        ax.plot(grid, spec_m, color="#2F6B9A", linewidth=2, label="Specificity")
        ax.plot(grid, f1_m, color="#2F7D5E", linewidth=2, label="F1 (Tumoral)")
        ax.plot(grid, j_m, color="#8A5A9E", linewidth=1.6, linestyle="--", label="Youden J")
        ax.axvline(0.5, color="#888888", linestyle=":", linewidth=1.4)
        ax.axvline(grid[best_idx], color="#8A5A9E", linestyle="-.", linewidth=1.4)
        ax.set_title(f"{model_name}\nJ-optimal cut-off = {grid[best_idx]:.2f}", fontsize=11)
        ax.set_xlabel("Decision threshold on P(Tumoral)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Score (mean across folds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=10)
    fig.suptitle("Operating-threshold sweep: the 0.5 cut-off (dotted) vs the Youden-optimal one (dash-dot)",
                 y=1.03, fontsize=14)
    fig.tight_layout()
    save(fig, output_dir, "threshold_sweep", dpi)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 4
def plot_operating_points(fold_metrics: pd.DataFrame, models: List[str], output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    for model_name in models:
        df = fold_metrics[fold_metrics["model"] == model_name]
        ax.scatter(1 - df["specificity"], df["sensitivity"], s=46, alpha=0.55,
                   color=MODEL_COLORS[model_name], edgecolor="white", linewidth=0.6)
        ax.scatter(1 - df["specificity"].mean(), df["sensitivity"].mean(), s=230, marker="*",
                   color=MODEL_COLORS[model_name], edgecolor="#222222", linewidth=0.8,
                   label=f"{model_name} (mean)")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
    for level in (0.2, 0.4, 0.6):
        ax.plot([0, 1], [level, 1 + level], color="#BBBBBB", linewidth=0.8, linestyle=":")
    ax.set_xlabel("1 - Specificity (false-positive rate)")
    ax.set_ylabel("Sensitivity (true-positive rate)")
    ax.set_title("Achieved operating points at the 0.5 cut-off\nsmall = individual folds, star = model mean")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, output_dir, "operating_points", dpi)


# --------------------------------------------------------------------------- 5
def plot_subgroup_errors(preds: pd.DataFrame, models: List[str], output_dir: Path, dpi: int) -> pd.DataFrame:
    """Binary errors attributed back to the original 4-class diagnosis.

    "Non-Tumoral" is three merged diagnoses; this asks whether the false positives
    come from all three equally or from one in particular.
    """
    rows = []
    for model_name in models:
        model_df = preds[preds["model"] == model_name]
        for group in GROUP_ORDER:
            group_df = model_df[model_df["original_group"] == group]
            if group_df.empty:
                continue
            per_fold = []
            for _, fold_df in group_df.groupby("fold"):
                per_fold.append(float((fold_df["pred_idx"] != fold_df["true_idx"]).mean()))
            rows.append(
                {
                    "model": model_name,
                    "original_group": group,
                    "original_group_name": GROUP_NAMES[group],
                    "binary_truth": CLASS_NAMES[1] if group == 2 else CLASS_NAMES[0],
                    "n_rows": len(group_df),
                    "error_rate_mean": float(np.mean(per_fold)),
                    "error_rate_ci95": ci95(np.asarray(per_fold)),
                }
            )
    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    x = np.arange(len(GROUP_ORDER))
    width = 0.8 / len(models)
    for idx, model_name in enumerate(models):
        df = summary[summary["model"] == model_name].set_index("original_group").reindex(GROUP_ORDER)
        offset = width * (idx - (len(models) - 1) / 2)
        ax.bar(x + offset, df["error_rate_mean"], width, color=MODEL_COLORS[model_name], alpha=0.85)
        ax.errorbar(x + offset, df["error_rate_mean"], yerr=df["error_rate_ci95"], fmt="none",
                    ecolor="#222222", capsize=3, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{GROUP_NAMES[g]}\n({'Tumoral' if g == 2 else 'Non-Tumoral'})" for g in GROUP_ORDER])
    ax.set_ylabel("Binary error rate (mean +/- 95% CI)")
    ax.set_xlabel("Original 4-class diagnosis")
    ax.set_title("Where the binary errors come from")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=[Patch(facecolor=MODEL_COLORS[m], alpha=0.85, label=m) for m in models],
              frameon=False, fontsize=9, ncol=2)

    ax = axes[1]
    pivot = summary.pivot(index="original_group_name", columns="model", values="error_rate_mean")
    pivot = pivot.reindex([GROUP_NAMES[g] for g in GROUP_ORDER])[models]
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Reds", vmin=0, linewidths=0.6,
                linecolor="white", cbar_kws={"label": "Error rate"}, ax=ax)
    ax.set_title("Same numbers as a grid")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Binary error attribution to the merged sub-diagnoses", y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, output_dir, "subgroup_error_attribution", dpi)
    return summary


# --------------------------------------------------------------------------- 6
def plot_fold_stability(fold_metrics: pd.DataFrame, models: List[str], output_dir: Path, dpi: int) -> None:
    metrics = [
        ("acc", "Accuracy"),
        ("balanced_acc", "Balanced acc."),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("mcc", "MCC"),
        ("auc_binary", "AUC"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        pivot = fold_metrics.pivot(index="model", columns="fold", values=col).reindex(models)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", linewidths=0.6,
                    linecolor="white", ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title(title)
        ax.set_xlabel("Fold")
        ax.set_ylabel("")
    fig.suptitle("Per-fold stability: every model x fold value behind the reported means", y=1.01, fontsize=15)
    fig.tight_layout()
    save(fig, output_dir, "fold_stability_heatmap", dpi)


# --------------------------------------------------------------------------- 7
def plot_pooled_confusions(preds: pd.DataFrame, models: List[str], output_dir: Path, dpi: int) -> pd.DataFrame:
    """All 5 folds pooled = every real image scored exactly once."""
    rows = []
    fig, axes = plt.subplots(2, len(models), figsize=(3.6 * len(models), 7.4))
    axes = np.atleast_2d(axes)
    for col, model_name in enumerate(models):
        model_df = preds[preds["model"] == model_name]
        cm = confusion_matrix(model_df["true_idx"], model_df["pred_idx"], labels=[0, 1])
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        tn, fp, fn, tp = cm.ravel()
        rows.append(
            {
                "model": model_name,
                "n": int(cm.sum()),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "accuracy": float((tp + tn) / cm.sum()),
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
            }
        )
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=0.6, linecolor="white",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0, col])
        axes[0, col].set_title(f"{model_name}\npooled counts", fontsize=11)
        sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Oranges", vmin=0, vmax=1, cbar=False,
                    linewidths=0.6, linecolor="white",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1, col])
        axes[1, col].set_title("row-normalised", fontsize=11)
        for row in (0, 1):
            axes[row, col].set_xlabel("Predicted")
            axes[row, col].set_ylabel("True" if col == 0 else "")
    fig.suptitle("Pooled confusion matrices: all 5 folds combined (each real image scored once)",
                 y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, output_dir, "pooled_confusion_matrices", dpi)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 8
def plot_experiment_comparison(
    fold_metrics: pd.DataFrame,
    baseline_dir: Path,
    models: List[str],
    output_dir: Path,
    dpi: int,
) -> Optional[pd.DataFrame]:
    baseline_path = baseline_dir / "metrics" / "all_models_all_folds_metrics.csv"
    if not baseline_path.exists():
        print(f"  skipping experiment_comparison: no baseline at {baseline_path}")
        return None
    baseline = pd.read_csv(baseline_path)

    metrics = [("acc", "Accuracy"), ("f1_score", "Macro F1"), ("mcc", "MCC"), ("auc", "Macro AUC")]
    rows = []
    for label, frame in (("From-scratch binary", baseline), ("Pretrained + GAN binary", fold_metrics)):
        for model_name in models:
            df = frame[frame["model"] == model_name]
            for col, _ in metrics:
                values = df[col].to_numpy(dtype=float)
                rows.append(
                    {
                        "experiment": label,
                        "model": model_name,
                        "metric": col,
                        "mean": float(np.nanmean(values)),
                        "ci95": ci95(values),
                    }
                )
    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 5.4), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(models))
    width = 0.38
    exp_colors = {"From-scratch binary": "#9BB7D4", "Pretrained + GAN binary": "#0B3D91"}
    for ax, (col, title) in zip(axes, metrics):
        for idx, label in enumerate(exp_colors):
            df = summary[(summary["experiment"] == label) & (summary["metric"] == col)]
            df = df.set_index("model").reindex(models)
            offset = width * (idx - 0.5)
            ax.bar(x + offset, df["mean"], width, color=exp_colors[label], alpha=0.92)
            ax.errorbar(x + offset, df["mean"], yerr=df["ci95"], fmt="none", ecolor="#222222",
                        capsize=3, linewidth=1)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Score (mean +/- 95% CI over 5 folds)")
    fig.legend(handles=[Patch(facecolor=c, label=l) for l, c in exp_colors.items()],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=11)
    fig.suptitle("Effect of ImageNet pretraining + GAN augmentation on the same binary task", y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, output_dir, "experiment_comparison", dpi)

    wide = summary.pivot_table(index=["model", "metric"], columns="experiment", values="mean").reset_index()
    wide["delta"] = wide["Pretrained + GAN binary"] - wide["From-scratch binary"]
    return wide



# --------------------------------------------------------------------------- 9
def plot_native_vs_collapsed(
    preds: pd.DataFrame,
    parent_dir: Path,
    models: List[str],
    output_dir: Path,
    dpi: int,
) -> Optional[pd.DataFrame]:
    """Is a dedicated binary head worth it, or would collapsing the 4-class model do?

    The 4-class parent experiment was trained on the *same* fold split, so its
    validation rows are identical fold-for-fold and the comparison is genuinely
    paired. Its predictions are collapsed with the same rule used for the labels:
    group 2 -> Tumoral, everything else -> Non-Tumoral, and P(Tumoral) = prob_group_2.
    """
    if not (parent_dir / "predictions").exists():
        print(f"  skipping native_vs_collapsed_4class: no parent run at {parent_dir}")
        return None

    def fold_scores(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (float(v) for v in cm.ravel())
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        return {
            "accuracy": (tp + tn) / cm.sum(),
            "sensitivity": sens,
            "specificity": spec,
            "balanced_acc": float(np.nanmean([sens, spec])),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_score),
        }

    rows = []
    for model_name in models:
        native = preds[preds["model"] == model_name]
        for fold, fold_df in native.groupby("fold"):
            parent_path = parent_dir / "predictions" / model_name / f"{model_name}_fold_{fold}_predictions.csv"
            if not parent_path.exists():
                continue
            parent = pd.read_csv(parent_path)
            # align both frames on image_file so the pairing is exact
            merged = fold_df.merge(parent, on="image_file", suffixes=("_bin", "_4c"))
            if len(merged) != len(fold_df):
                print(f"  warning: {model_name} fold {fold} aligned {len(merged)}/{len(fold_df)} rows")
            y_true = merged["true_idx_bin"].to_numpy(dtype=int)
            for variant, y_pred, y_score in (
                ("Dedicated binary head", merged["pred_idx_bin"].to_numpy(dtype=int),
                 merged[POS_COL].to_numpy(dtype=float)),
                ("Collapsed 4-class", (merged["pred_group_label_4c"] == 2).astype(int).to_numpy(),
                 merged["prob_group_2"].to_numpy(dtype=float)),
            ):
                rows.append({"model": model_name, "fold": int(fold), "variant": variant,
                             **fold_scores(y_true, y_pred, y_score)})
    if not rows:
        print("  skipping native_vs_collapsed_4class: no aligned folds")
        return None
    per_fold = pd.DataFrame(rows)

    metrics = [("accuracy", "Accuracy"), ("sensitivity", "Sensitivity"),
               ("specificity", "Specificity"), ("mcc", "MCC"), ("auc", "AUC")]
    variants = ["Collapsed 4-class", "Dedicated binary head"]
    variant_colors = {"Collapsed 4-class": "#C7A76C", "Dedicated binary head": "#0B3D91"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 5.4), sharey=True)
    x = np.arange(len(models))
    width = 0.38
    for ax, (col, title) in zip(axes, metrics):
        for idx, variant in enumerate(variants):
            df = (per_fold[per_fold["variant"] == variant]
                  .groupby("model")[col].agg(["mean", ci95]).reindex(models))
            ax.bar(x + width * (idx - 0.5), df["mean"], width, color=variant_colors[variant], alpha=0.92)
            ax.errorbar(x + width * (idx - 0.5), df["mean"], yerr=df["ci95"], fmt="none",
                        ecolor="#222222", capsize=3, linewidth=1)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Score (mean +/- 95% CI over 5 paired folds)")
    fig.legend(handles=[Patch(facecolor=variant_colors[v], label=v) for v in variants],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=11)
    fig.suptitle("Training a binary head vs collapsing the 4-class model's predictions\n"
                 "(identical fold splits and validation images, so folds are paired)",
                 y=1.05, fontsize=15)
    fig.tight_layout()
    save(fig, output_dir, "native_vs_collapsed_4class", dpi)

    wide = per_fold.pivot_table(index=["model", "fold"], columns="variant",
                                values=[c for c, _ in metrics]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns.to_flat_index()]
    return wide


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = load_predictions(input_dir, args.models)
    fold_metrics = load_fold_metrics(input_dir)
    print(f"Loaded {len(preds)} prediction rows and {len(fold_metrics)} fold-metric rows from {input_dir}")

    summaries: Dict[str, pd.DataFrame] = {}
    summaries["average_precision_per_fold"] = plot_precision_recall(preds, args.models, output_dir, args.dpi)
    summaries["calibration_summary"] = plot_calibration(preds, args.models, output_dir, args.dpi)
    summaries["threshold_sweep_summary"] = plot_threshold_sweep(preds, args.models, output_dir, args.dpi)
    plot_operating_points(fold_metrics, args.models, output_dir, args.dpi)
    summaries["subgroup_error_attribution"] = plot_subgroup_errors(preds, args.models, output_dir, args.dpi)
    plot_fold_stability(fold_metrics, args.models, output_dir, args.dpi)
    summaries["pooled_confusion_summary"] = plot_pooled_confusions(preds, args.models, output_dir, args.dpi)
    comparison = plot_experiment_comparison(fold_metrics, args.baseline_dir.resolve(), args.models,
                                            output_dir, args.dpi)
    if comparison is not None:
        summaries["experiment_comparison"] = comparison
    collapsed = plot_native_vs_collapsed(preds, args.parent_dir.resolve(), args.models,
                                         output_dir, args.dpi)
    if collapsed is not None:
        summaries["native_vs_collapsed_4class"] = collapsed

    for name, frame in summaries.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
        print(f"  wrote {name}.csv")

    (output_dir / "extra_plots_manifest.json").write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "baseline_dir": str(args.baseline_dir.resolve()),
                "models": args.models,
                "dpi": args.dpi,
                "figures": sorted(p.name for p in output_dir.glob("*.png")),
                "summary_csvs": sorted(summaries),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done. Extra binary figures saved to {output_dir}")


if __name__ == "__main__":
    main()
