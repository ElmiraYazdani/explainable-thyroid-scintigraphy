#!/usr/bin/env python3
"""Create publication-ready mean plots from the pretrained + GAN-augmented binary 5-fold outputs.

Original experiment CSV files are read only. Derived figures and summary CSVs
are written under a separate mean_5fold output directory.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


T_CRIT_95_N5 = 2.7764451051977987
STYLE_KEYS = (
    "dpi",
    "font_family",
    "base_font_size",
    "title_font_size",
    "suptitle_font_family",
    "suptitle_font_size",
    "suptitle_font_weight",
    "suptitle_y",
    "label_font_size",
    "tick_font_size",
    "legend_font_size",
    "line_width",
    "ci_alpha",
    "grid_alpha",
    "tight_pad",
    "save_transparent",
    "include_individual_folds",
    "fold_line_alpha",
    "palette",
)


@dataclass
class PlotConfig:
    input_dir: str = "Pretrained Naive Replication 5 Folds GAN Augmented Binary/classification_pretrained_replication_5fold_gan_augmented_binary_outputs"
    output_dir: str = "Pretrained Naive Replication 5 Folds GAN Augmented Binary/figures/mean_5fold"
    artifact_prefix: str = "pretrained_gan_binary_5fold"
    model_names: List[str] = field(
        default_factory=lambda: [
            "VGG11",
            "DenseNet121",
            "ResNet18",
            "MobileNetV2",
            "EfficientNetB0",
        ]
    )
    folds: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    class_labels: List[str] = field(default_factory=lambda: ["Non-Tumoral", "Tumoral"])
    class_probability_columns: List[str] = field(
        default_factory=lambda: ["prob_Non-Tumoral", "prob_Tumoral"]
    )
    report_class_prefix: str = ""
    metric_summary_columns: List[str] = field(
        default_factory=lambda: ["acc", "precision", "recall", "f1_score"]
    )
    metric_summary_xticklabels: List[str] = field(
        default_factory=lambda: ["Accuracy", "Precision", "Recall", "F1-score"]
    )
    scale_accuracy_to_percent: bool = True
    dpi: int = 900
    font_family: str = "DejaVu Sans"
    base_font_size: int = 11
    title_font_size: int = 13
    suptitle_font_family: str = "DejaVu Sans"
    suptitle_font_size: int = 15
    suptitle_font_weight: str = "normal"
    suptitle_y: float = 1.02
    label_font_size: int = 11
    tick_font_size: int = 9
    legend_font_size: int = 9
    line_width: float = 2.0
    ci_alpha: float = 0.18
    grid_alpha: float = 0.28
    tight_pad: float = 1.1
    save_transparent: bool = False
    include_individual_folds: bool = False
    fold_line_alpha: float = 0.2
    create_interactive_html: bool = True
    palette: Dict[str, str] = field(
        default_factory=lambda: {
            "VGG11": "#1F77B4",
            "DenseNet121": "#FF7F0E",
            "ResNet18": "#2CA02C",
            "MobileNetV2": "#D62728",
            "EfficientNetB0": "#9467BD",
            "train_loss": "#2F6B9A",
            "val_loss": "#C44E52",
            "train_acc": "#317A43",
            "val_acc": "#8A5A9E",
            "lr": "#555555",
            "bar": "#4C78A8",
            "ci": "#222222",
        }
    )
    plots: Dict[str, Dict[str, object]] = field(
        default_factory=lambda: {
            "learning_curves": {
                "figsize": [15.5, 9.0],
                "suptitle": "Binary Classification Models: Mean Learning Curves Across 5 Folds",
                "train_alpha": 0.78,
                "val_alpha": 0.98,
                "gap_alpha": 0.9,
                "best_epoch_marker": True,
            },
            "confusion_matrices": {
                "figsize": [14, 22],
                "suptitle": "Binary Confusion Matrices: Mean +/- 95% CI Across 5 Folds",
                "annotation_font_size": 11,
                "row_height": 4.2,
                "subplot_top": 0.96,
                "subplot_bottom": 0.05,
                "subplot_left": 0.08,
                "subplot_right": 0.94,
                "subplot_hspace": 0.42,
                "subplot_wspace": 0.32,
            },
            "roc_curves": {
                "figsize": [22, 4.5],
                "suptitle": "Binary ROC Curves: Mean Across 5 Folds",
                "roc_grid_points": 101,
                "class_colors": ["#0B3D91", "#1F77B4"],
            },
            "per_class_metrics": {
                "figsize": [18, 6],
                "suptitle": "Binary Per-Class Metrics: Mean +/- 95% CI Across 5 Folds",
                "bar_alpha": 0.85,
                "legend_ncol": 5,
                "legend_bottom_margin": 0.14,
            },
            "prediction_confidence": {
                "figsize": [22, 4.5],
                "suptitle": "Binary Prediction Confidence Across 5 Folds",
                "bins": 20,
                "correct_color": "#2F7D5E",
                "incorrect_color": "#C44E52",
            },
            "model_metric_summary": {
                "figsize": [12, 5.5],
                "suptitle": "Binary Model Metrics: Mean +/- 95% CI Across 5 Folds",
                "metric_bar_colors": ["#0B3D91", "#1F77B4", "#4C9ED9", "#8CC6F0"],
            },
            "interactive_learning_curves": {
                "width": 1500,
                "height": 850,
                "title": "Binary Classification Models: Mean Learning Curves Across 5 Folds",
            },
        }
    )


def load_config(path: Path | None) -> PlotConfig:
    config = PlotConfig()
    if path is None:
        return config
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key, value in data.items():
        if key == "plots":
            merged_plots = config.plots
            for plot_name, plot_values in value.items():
                merged = dict(merged_plots.get(plot_name, {}))
                merged.update(plot_values)
                merged_plots[plot_name] = merged
            config.plots = merged_plots
            continue
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def plot_settings(config: PlotConfig, plot_name: str) -> Dict[str, object]:
    settings = {key: getattr(config, key) for key in STYLE_KEYS}
    settings["palette"] = dict(config.palette)
    overrides = dict(config.plots.get(plot_name, {}))
    if "palette" in overrides:
        merged_palette = dict(settings["palette"])
        merged_palette.update(overrides.pop("palette"))
        settings["palette"] = merged_palette
    settings.update(overrides)
    return settings


def expanded_config_dict(config: PlotConfig) -> Dict[str, object]:
    data = asdict(config)
    plot_names = list(PlotConfig().plots.keys())
    for plot_name in config.plots:
        if plot_name not in plot_names:
            plot_names.append(plot_name)
    data["plots"] = {plot_name: plot_settings(config, plot_name) for plot_name in plot_names}
    return data


def save_config(path: Path, config: PlotConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expanded_config_dict(config), indent=2), encoding="utf-8")


def rc_params(settings: Dict[str, object]) -> Dict[str, object]:
    return {
        "font.family": settings["font_family"],
        "font.size": settings["base_font_size"],
        "axes.titlesize": settings["title_font_size"],
        "axes.labelsize": settings["label_font_size"],
        "xtick.labelsize": settings["tick_font_size"],
        "ytick.labelsize": settings["tick_font_size"],
        "legend.fontsize": settings["legend_font_size"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }


def as_figsize(settings: Dict[str, object], default: Sequence[float]) -> tuple[float, float]:
    value = settings.get("figsize", default)
    return (float(value[0]), float(value[1]))


def confusion_matrix_figsize(n_models: int, settings: Dict[str, object]) -> tuple[float, float]:
    """Ensure enough vertical space per model row for readable 2x2 heatmaps."""
    default = [14.0, max(14.0, float(settings.get("row_height", 4.2)) * n_models + 1.5)]
    width, height = as_figsize(settings, default)
    min_height = float(settings.get("row_height", 4.2)) * n_models + 1.5
    return width, max(height, min_height)


def palette(settings: Dict[str, object]) -> Dict[str, str]:
    return settings["palette"]  # type: ignore[return-value]


def apply_figure_legend(
    fig: plt.Figure,
    handles: Sequence[object],
    labels: Sequence[str],
    settings: Dict[str, object],
    ncol: int | None = None,
) -> None:
    if not handles:
        return
    ncol = ncol or int(settings.get("legend_ncol", min(len(labels), 5)))
    fig.legend(
        handles,
        labels,
        loc=str(settings.get("legend_loc", "upper center")),
        bbox_to_anchor=(
            float(settings.get("legend_bbox_x", 0.5)),
            float(settings.get("legend_bbox_y", -0.02)),
        ),
        ncol=ncol,
        frameon=bool(settings.get("legend_frameon", False)),
        fontsize=float(settings["legend_font_size"]),
    )


def layout_with_bottom_legend(fig: plt.Figure, settings: Dict[str, object]) -> None:
    fig.subplots_adjust(
        bottom=float(settings.get("legend_bottom_margin", 0.12)),
        top=float(settings.get("legend_top_margin", 0.90)),
    )


def model_color_handles(config: PlotConfig, settings: Dict[str, object]) -> tuple[List[object], List[str]]:
    colors = palette(settings)
    alpha = float(settings.get("bar_alpha", 0.85))
    handles = []
    labels = []
    for model_name in config.model_names:
        handles.append(Patch(facecolor=colors.get(model_name, colors["bar"]), alpha=alpha, edgecolor="none"))
        labels.append(model_name)
    return handles, labels


def add_suptitle(fig: plt.Figure, settings: Dict[str, object]) -> None:
    if not settings.get("suptitle"):
        return
    fig.suptitle(
        str(settings["suptitle"]),
        y=float(settings["suptitle_y"]),
        fontsize=float(settings["suptitle_font_size"]),
        fontfamily=str(settings["suptitle_font_family"]),
        fontweight=str(settings["suptitle_font_weight"]),
    )


def save_figure(fig: plt.Figure, output_base: Path, settings: Dict[str, object]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "jpg", "jpeg"):
        fig.savefig(
            output_base.with_suffix(f".{ext}"),
            dpi=int(settings["dpi"]),
            bbox_inches="tight",
            pad_inches=0.04,
            transparent=bool(settings["save_transparent"]),
        )
    plt.close(fig)


def ci95(values: Sequence[float], axis: int = 0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n = np.sum(~np.isnan(arr), axis=axis)
    std = np.nanstd(arr, axis=axis, ddof=1)
    return T_CRIT_95_N5 * std / np.sqrt(n)


def mean_ci_frame(frames: Sequence[pd.DataFrame], columns: Sequence[str]) -> pd.DataFrame:
    long = []
    for frame in frames:
        long.append(frame[["epoch", *columns]].copy())
    data = pd.concat(long, ignore_index=True)
    rows = []
    for epoch, group in data.groupby("epoch", sort=True):
        row = {"epoch": int(epoch), "n_folds": len(group)}
        for column in columns:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.nanmean(values))
            row[f"{column}_std"] = float(np.nanstd(values, ddof=1))
            row[f"{column}_ci95"] = float(ci95(values))
        rows.append(row)
    return pd.DataFrame(rows)


def setup_matplotlib(config: PlotConfig) -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(rc_params(plot_settings(config, "global")))


def maybe_scale_history_accuracy(frame: pd.DataFrame, config: PlotConfig) -> pd.DataFrame:
    if not config.scale_accuracy_to_percent:
        return frame
    scaled = frame.copy()
    if scaled["val_acc"].max() <= 1.5:
        for column in ("train_acc", "val_acc"):
            scaled[column] = scaled[column] * 100.0
    return scaled


def read_history(config: PlotConfig, input_dir: Path, model_name: str, fold: int) -> pd.DataFrame:
    path = input_dir / "histories" / model_name / f"{model_name}_fold_{fold}_history.csv"
    return maybe_scale_history_accuracy(pd.read_csv(path), config)


def read_predictions(config: PlotConfig, input_dir: Path) -> Dict[str, pd.DataFrame]:
    result = {}
    for model_name in config.model_names:
        frames = []
        for fold in config.folds:
            path = input_dir / "predictions" / model_name / f"{model_name}_fold_{fold}_predictions.csv"
            frame = pd.read_csv(path)
            frame["fold"] = fold
            frame["model"] = model_name
            frames.append(frame)
        result[model_name] = pd.concat(frames, ignore_index=True)
    return result


def read_reports(config: PlotConfig, input_dir: Path) -> pd.DataFrame:
    frames = []
    for model_name in config.model_names:
        for fold in config.folds:
            path = input_dir / "reports" / model_name / f"{model_name}_fold_{fold}_classification_report.csv"
            frame = pd.read_csv(path, index_col=0).reset_index(names="label")
            frame["model"] = model_name
            frame["fold"] = fold
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def read_all_fold_metrics(config: PlotConfig, input_dir: Path) -> pd.DataFrame:
    combined_path = input_dir / "metrics" / "all_models_all_folds_metrics.csv"
    if combined_path.exists():
        return pd.read_csv(combined_path)

    frames = []
    for model_name in config.model_names:
        path = input_dir / "metrics" / f"{model_name}_fold_metrics.csv"
        frame = pd.read_csv(path)
        frame = frame[frame["fold"].astype(str).str.isdigit()].copy()
        frame["model"] = model_name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            "No metrics found. Expected metrics/all_models_all_folds_metrics.csv "
            "or per-model metrics/{model}_fold_metrics.csv files."
        )
    return pd.concat(frames, ignore_index=True)


def artifact_stem(config: PlotConfig, name: str) -> str:
    return f"{config.artifact_prefix}_{name}"


def plot_label(config: PlotConfig, report_label: str) -> str:
    label = str(report_label)
    if label in config.class_labels:
        return label
    prefix = config.report_class_prefix
    if prefix and label.startswith(prefix):
        try:
            group_num = int(label.split("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            return label
        if 1 <= group_num <= len(config.class_labels):
            return config.class_labels[group_num - 1]
    if label.startswith("Class "):
        try:
            idx = int(label.split()[1])
        except (IndexError, ValueError):
            return label
        if 0 <= idx < len(config.class_labels):
            return config.class_labels[idx]
    return label


def binarize_true_labels(y_true: np.ndarray, num_classes: int) -> np.ndarray:
    labels_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if num_classes == 2 and labels_bin.shape[1] == 1:
        labels_bin = np.column_stack([1 - labels_bin[:, 0], labels_bin[:, 0]])
    return labels_bin


def report_class_labels(config: PlotConfig, reports: pd.DataFrame) -> List[str]:
    exclude = {"accuracy", "macro avg", "weighted avg"}
    prefix = config.report_class_prefix
    labels = [
        label
        for label in reports["label"].drop_duplicates().tolist()
        if str(label) not in exclude
        and (
            str(label) in config.class_labels
            or str(label).startswith("Class ")
            or (prefix and str(label).startswith(prefix))
        )
    ]

    def sort_key(label: str) -> tuple[int, str]:
        text = str(label)
        if text in config.class_labels:
            return (0, config.class_labels.index(text))
        if prefix and text.startswith(prefix):
            try:
                return (0, int(text.split("_", maxsplit=1)[1]))
            except (IndexError, ValueError):
                return (1, text)
        if text.startswith("Class "):
            try:
                return (0, int(text.split()[1]))
            except (IndexError, ValueError):
                return (1, text)
        return (1, text)

    return sorted(labels, key=sort_key)


def discover_available_folds(config: PlotConfig, input_dir: Path) -> List[int]:
    available_sets = []
    for model_name in config.model_names:
        model_folds = []
        for fold in config.folds:
            path = input_dir / "predictions" / model_name / f"{model_name}_fold_{fold}_predictions.csv"
            if path.exists():
                model_folds.append(fold)
        if model_folds:
            available_sets.append(set(model_folds))
    if not available_sets:
        return list(config.folds)
    common = set.intersection(*available_sets)
    return sorted(common) if common else sorted(set.union(*available_sets))


def plot_learning_curves(config: PlotConfig, input_dir: Path, output_dir: Path) -> None:
    settings = plot_settings(config, "learning_curves")
    colors = palette(settings)
    summary_frames = []
    model_summaries = {}
    for model_name in config.model_names:
        histories = [read_history(config, input_dir, model_name, fold) for fold in config.folds]
        summary = mean_ci_frame(histories, ["train_loss", "val_loss", "train_acc", "val_acc", "lr"])
        summary["model"] = model_name
        summary_frames.append(summary)
        model_summaries[model_name] = (summary, histories)
    pd.concat(summary_frames, ignore_index=True).to_csv(
        output_dir / f"{artifact_stem(config, 'mean_learning_history_summary')}.csv", index=False
    )

    with plt.rc_context(rc_params(settings)):
        fig, axes = plt.subplots(2, 2, figsize=as_figsize(settings, [15.5, 9.0]))
        for model_name, (summary, histories) in model_summaries.items():
            model_color = colors.get(model_name, colors["bar"])
            epochs = summary["epoch"].to_numpy(dtype=float)
            for column, ax, label, linestyle, alpha in (
                ("train_loss", axes[0, 0], f"{model_name} train", "-.", float(settings["train_alpha"])),
                ("val_loss", axes[0, 0], f"{model_name} validation", "-", float(settings["val_alpha"])),
                ("train_acc", axes[0, 1], f"{model_name} train", "-.", float(settings["train_alpha"])),
                ("val_acc", axes[0, 1], f"{model_name} validation", "-", float(settings["val_alpha"])),
            ):
                mean = summary[f"{column}_mean"].to_numpy(dtype=float)
                ci = summary[f"{column}_ci95"].to_numpy(dtype=float)
                ax.plot(
                    epochs,
                    mean,
                    color=model_color,
                    linestyle=linestyle,
                    linewidth=float(settings["line_width"]),
                    alpha=alpha,
                    label=label,
                )
                ax.fill_between(epochs, mean - ci, mean + ci, color=model_color, alpha=float(settings["ci_alpha"]), linewidth=0)
            diff_mean = summary["val_loss_mean"] - summary["train_loss_mean"]
            diff_ci = np.sqrt(summary["val_loss_ci95"] ** 2 + summary["train_loss_ci95"] ** 2)
            axes[1, 0].plot(
                epochs,
                diff_mean,
                color=model_color,
                linewidth=float(settings["line_width"]),
                alpha=float(settings["gap_alpha"]),
                label=model_name,
            )
            axes[1, 0].fill_between(
                epochs,
                diff_mean - diff_ci,
                diff_mean + diff_ci,
                color=model_color,
                alpha=float(settings["ci_alpha"]),
                linewidth=0,
            )
            val_acc = summary["val_acc_mean"].to_numpy(dtype=float)
            val_ci = summary["val_acc_ci95"].to_numpy(dtype=float)
            axes[1, 1].plot(
                epochs,
                val_acc,
                color=model_color,
                linewidth=float(settings["line_width"]),
                label=model_name,
            )
            axes[1, 1].fill_between(epochs, val_acc - val_ci, val_acc + val_ci, color=model_color, alpha=float(settings["ci_alpha"]), linewidth=0)
            if bool(settings["best_epoch_marker"]):
                best_idx = int(np.nanargmax(val_acc))
                axes[1, 1].scatter(
                    [epochs[best_idx]],
                    [val_acc[best_idx]],
                    color=model_color,
                    edgecolor="white",
                    linewidth=0.8,
                    s=46,
                    zorder=4,
                )
            if bool(settings["include_individual_folds"]):
                for history in histories:
                    axes[1, 0].plot(
                        history["epoch"],
                        history["val_loss"] - history["train_loss"],
                        color=model_color,
                        alpha=float(settings["fold_line_alpha"]),
                        linewidth=0.8,
                    )

        titles = [
            "Training and Validation Loss",
            "Training and Validation Accuracy",
            "Generalization Gap",
            "Validation Accuracy Trajectory",
        ]
        ylabels = ["Loss", "Accuracy (%)", "Validation loss - training loss", "Validation accuracy (%)"]
        for ax, title, ylabel in zip(axes.ravel(), titles, ylabels):
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=float(settings["grid_alpha"]))
        axes[1, 0].axhline(0, color="#222222", linestyle="--", linewidth=1, alpha=0.6)
        axes[0, 1].set_ylim(0, 105)
        axes[1, 1].set_ylim(0, 105)
        add_suptitle(fig, settings)
        model_handles, model_labels = model_color_handles(config, settings)
        style_handles = [
            Line2D([0], [0], color="#333333", linestyle="-.", linewidth=2, label="Train"),
            Line2D([0], [0], color="#333333", linestyle="-", linewidth=2, label="Validation"),
        ]
        apply_figure_legend(
            fig,
            [*model_handles, *style_handles],
            [*model_labels, "Train", "Validation"],
            settings,
            ncol=int(settings.get("legend_ncol", len(config.model_names) + 2)),
        )
        layout_with_bottom_legend(fig, settings)
        fig.tight_layout(pad=float(settings["tight_pad"]))
        save_figure(fig, output_dir / artifact_stem(config, "mean_learning_curves"), settings)


def plot_confusion_matrices(config: PlotConfig, predictions: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    settings = plot_settings(config, "confusion_matrices")
    labels = list(range(len(config.class_labels)))
    n_models = len(config.model_names)
    with plt.rc_context(rc_params(settings)):
        fig, axes = plt.subplots(
            n_models,
            2,
            figsize=confusion_matrix_figsize(n_models, settings),
            squeeze=False,
        )
        for row_idx, model_name in enumerate(config.model_names):
            cms = []
            cm_norms = []
            for fold in config.folds:
                fold_df = predictions[model_name][predictions[model_name]["fold"] == fold]
                cm = confusion_matrix(fold_df["true_idx"], fold_df["pred_idx"], labels=labels)
                cms.append(cm)
                row_sum = cm.sum(axis=1, keepdims=True)
                cm_norms.append(np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0))
            cm_mean = np.mean(cms, axis=0)
            cm_ci = ci95(cms, axis=0)
            norm_mean = np.mean(cm_norms, axis=0)
            norm_ci = ci95(cm_norms, axis=0)
            pd.DataFrame(cm_mean, index=config.class_labels, columns=config.class_labels).to_csv(
                output_dir / f"{config.artifact_prefix}_{model_name}_mean_confusion_counts.csv"
            )
            pd.DataFrame(norm_mean, index=config.class_labels, columns=config.class_labels).to_csv(
                output_dir / f"{config.artifact_prefix}_{model_name}_mean_confusion_normalized.csv"
            )
            ann = np.empty_like(cm_mean, dtype=object)
            ann_norm = np.empty_like(norm_mean, dtype=object)
            for i in range(cm_mean.shape[0]):
                for j in range(cm_mean.shape[1]):
                    ann[i, j] = f"{cm_mean[i, j]:.1f}\n+/- {cm_ci[i, j]:.1f}"
                    ann_norm[i, j] = f"{norm_mean[i, j]:.2f}\n+/- {norm_ci[i, j]:.2f}"
            for ax, matrix, annot, cmap, title, vmin, vmax in (
                (axes[row_idx, 0], cm_mean, ann, "Blues", f"{model_name} Counts", None, None),
                (axes[row_idx, 1], norm_mean, ann_norm, "Oranges", f"{model_name} Normalized", 0.0, 1.0),
            ):
                sns.heatmap(
                    matrix,
                    annot=annot,
                    annot_kws={"fontsize": int(settings["annotation_font_size"])},
                    fmt="",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    xticklabels=config.class_labels,
                    yticklabels=config.class_labels,
                    linewidths=0.5,
                    linecolor="white",
                    cbar_kws={"shrink": 0.75},
                    ax=ax,
                )
                ax.set_title(title, pad=10)
                ax.set_xlabel("Predicted", labelpad=8)
                ax.set_ylabel("True", labelpad=8)
                ax.tick_params(axis="both", labelsize=int(settings.get("tick_font_size", settings["label_font_size"])))
        add_suptitle(fig, settings)
        fig.subplots_adjust(
            left=float(settings.get("subplot_left", 0.08)),
            right=float(settings.get("subplot_right", 0.94)),
            top=float(settings.get("subplot_top", 0.96)),
            bottom=float(settings.get("subplot_bottom", 0.05)),
            hspace=float(settings.get("subplot_hspace", 0.42)),
            wspace=float(settings.get("subplot_wspace", 0.32)),
        )
        save_figure(fig, output_dir / artifact_stem(config, "mean_confusion_matrices"), settings)


def plot_roc_curves(config: PlotConfig, predictions: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    settings = plot_settings(config, "roc_curves")
    class_colors = settings["class_colors"]
    grid = np.linspace(0, 1, int(settings["roc_grid_points"]))
    rows = []
    with plt.rc_context(rc_params(settings)):
        fig, axes = plt.subplots(1, len(config.model_names), figsize=as_figsize(settings, [22, 4.5]), squeeze=False)
        for ax, model_name in zip(axes.ravel(), config.model_names):
            model_df = predictions[model_name]
            for class_idx, class_label in enumerate(config.class_labels):
                tprs = []
                aucs = []
                for fold in config.folds:
                    fold_df = model_df[model_df["fold"] == fold]
                    y_bin = binarize_true_labels(
                        fold_df["true_idx"].to_numpy(dtype=int),
                        len(config.class_labels),
                    )
                    try:
                        fpr, tpr, _ = roc_curve(
                            y_bin[:, class_idx],
                            fold_df[config.class_probability_columns[class_idx]],
                        )
                        fold_auc = auc(fpr, tpr)
                        interp = np.interp(grid, fpr, tpr)
                        interp[0] = 0.0
                    except ValueError:
                        fold_auc = np.nan
                        interp = np.full_like(grid, np.nan)
                    tprs.append(interp)
                    aucs.append(fold_auc)
                mean_tpr = np.nanmean(tprs, axis=0)
                ci_tpr = ci95(tprs, axis=0)
                mean_auc = float(np.nanmean(aucs))
                ci_auc = float(ci95(aucs))
                rows.append({"model": model_name, "class": class_label, "auc_mean": mean_auc, "auc_ci95": ci_auc})
                color = class_colors[class_idx]
                ax.plot(
                    grid,
                    mean_tpr,
                    color=color,
                    linewidth=float(settings["line_width"]),
                    label=f"{class_label} AUC={mean_auc:.3f}",
                )
                ax.fill_between(grid, mean_tpr - ci_tpr, mean_tpr + ci_tpr, color=color, alpha=float(settings["ci_alpha"]), linewidth=0)
            ax.plot([0, 1], [0, 1], color="#222222", linestyle="--", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.set_title(model_name)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.grid(True, alpha=float(settings["grid_alpha"]))
            ax.legend(
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, float(settings.get("roc_legend_bbox_y", -0.28))),
                fontsize=float(settings["legend_font_size"]),
                ncol=1,
            )
        pd.DataFrame(rows).to_csv(output_dir / f"{artifact_stem(config, 'mean_roc_auc_summary')}.csv", index=False)
        add_suptitle(fig, settings)
        fig.subplots_adjust(bottom=float(settings.get("roc_bottom_margin", 0.22)), wspace=0.28)
        fig.tight_layout(pad=float(settings["tight_pad"]))
        save_figure(fig, output_dir / artifact_stem(config, "mean_roc_curves"), settings)


def plot_per_class_metrics(config: PlotConfig, reports: pd.DataFrame, output_dir: Path) -> None:
    settings = plot_settings(config, "per_class_metrics")
    class_metrics = ["precision", "recall", "f1-score"]
    plot_metrics = ["accuracy", *class_metrics]
    labels = report_class_labels(config, reports)
    rows = []
    for model_name in config.model_names:
        accuracy_df = reports[(reports["model"] == model_name) & (reports["label"] == "accuracy")]
        accuracy_values = accuracy_df["precision"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model_name,
                "class": "Overall",
                "metric": "accuracy",
                "mean": np.nanmean(accuracy_values),
                "std": np.nanstd(accuracy_values, ddof=1),
                "ci95": ci95(accuracy_values),
            }
        )
        for label in labels:
            label_df = reports[(reports["model"] == model_name) & (reports["label"] == label)]
            for metric in class_metrics:
                values = label_df[metric].to_numpy(dtype=float)
                rows.append(
                    {
                        "model": model_name,
                        "class": plot_label(config, label),
                        "metric": metric,
                        "mean": np.nanmean(values),
                        "std": np.nanstd(values, ddof=1),
                        "ci95": ci95(values),
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / f"{artifact_stem(config, 'per_class_metrics_mean_ci')}.csv", index=False)

    with plt.rc_context(rc_params(settings)):
        fig, axes = plt.subplots(1, 4, figsize=as_figsize(settings, [18, 6]), sharey=True)
        class_x = np.arange(len(config.class_labels))
        width = 0.8 / max(len(config.model_names), 1)
        colors = palette(settings)
        bar_alpha = float(settings.get("bar_alpha", 0.85))
        for metric_idx, (ax, metric) in enumerate(zip(axes, plot_metrics)):
            for model_idx, model_name in enumerate(config.model_names):
                df = summary[(summary["model"] == model_name) & (summary["metric"] == metric)]
                offset = width * (model_idx - (len(config.model_names) - 1) / 2)
                bar_color = colors.get(model_name, colors["bar"])
                if metric == "accuracy":
                    x = np.arange(1)
                    bar_x = x + offset
                    xticklabels = ["Overall"]
                else:
                    x = class_x
                    bar_x = x + offset
                    xticklabels = config.class_labels
                ax.bar(
                    bar_x,
                    df["mean"],
                    width,
                    color=bar_color,
                    alpha=bar_alpha,
                )
                ax.errorbar(bar_x, df["mean"], yerr=df["ci95"], fmt="none", ecolor="#222222", capsize=3, linewidth=1)
            ax.set_title("Accuracy" if metric == "accuracy" else metric.capitalize())
            ax.set_xlabel("Class")
            if metric_idx == 0:
                ax.set_ylabel("Score")
            ax.set_xticks(np.arange(1) if metric == "accuracy" else class_x)
            ax.set_xticklabels(xticklabels)
            ax.set_ylim(0, 1.05)
            ax.grid(True, axis="y", alpha=float(settings["grid_alpha"]))
        add_suptitle(fig, settings)
        legend_handles, legend_labels = model_color_handles(config, settings)
        apply_figure_legend(fig, legend_handles, legend_labels, settings)
        layout_with_bottom_legend(fig, settings)
        fig.tight_layout(pad=float(settings["tight_pad"]))
        save_figure(fig, output_dir / artifact_stem(config, "mean_per_class_metrics"), settings)


def plot_prediction_confidence(config: PlotConfig, predictions: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    settings = plot_settings(config, "prediction_confidence")
    rows = []
    with plt.rc_context(rc_params(settings)):
        fig, axes = plt.subplots(1, len(config.model_names), figsize=as_figsize(settings, [22, 4.5]), squeeze=False)
        for ax, model_name in zip(axes.ravel(), config.model_names):
            df = predictions[model_name].copy()
            df["correct"] = df["true_idx"] == df["pred_idx"]
            for correct_value, label, color in (
                (True, "Correct", settings["correct_color"]),
                (False, "Incorrect", settings["incorrect_color"]),
            ):
                values = df.loc[df["correct"] == correct_value, "max_prob"].to_numpy(dtype=float)
                ax.hist(values, bins=int(settings["bins"]), alpha=0.72, label=label, color=color)
                fold_means = df[df["correct"] == correct_value].groupby("fold")["max_prob"].mean().to_numpy(dtype=float)
                rows.append(
                    {
                        "model": model_name,
                        "prediction_group": label,
                        "mean_confidence": np.nanmean(fold_means),
                        "std_confidence": np.nanstd(fold_means, ddof=1),
                        "ci95_confidence": ci95(fold_means),
                    }
                )
            ax.set_title(model_name)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=float(settings["grid_alpha"]))
        confidence_handles = [
            Patch(facecolor=str(settings["correct_color"]), alpha=0.72, label="Correct"),
            Patch(facecolor=str(settings["incorrect_color"]), alpha=0.72, label="Incorrect"),
        ]
        apply_figure_legend(fig, confidence_handles, ["Correct", "Incorrect"], settings, ncol=2)
        pd.DataFrame(rows).to_csv(
            output_dir / f"{artifact_stem(config, 'prediction_confidence_summary')}.csv", index=False
        )
        add_suptitle(fig, settings)
        layout_with_bottom_legend(fig, settings)
        fig.tight_layout(pad=float(settings["tight_pad"]))
        save_figure(fig, output_dir / artifact_stem(config, "prediction_confidence"), settings)


def plot_model_metric_summary(config: PlotConfig, input_dir: Path, output_dir: Path) -> None:
    settings = plot_settings(config, "model_metric_summary")
    metrics_df = read_all_fold_metrics(config, input_dir)
    metrics = list(config.metric_summary_columns)
    rows = []
    for model_name in config.model_names:
        df = metrics_df[metrics_df["model"] == model_name]
        for metric in metrics:
            if metric not in df.columns:
                raise KeyError(f"Metric column '{metric}' not found for model {model_name}")
            values = df[metric].to_numpy(dtype=float)
            rows.append({"model": model_name, "metric": metric, "mean": np.nanmean(values), "ci95": ci95(values)})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / f"{artifact_stem(config, 'model_metric_summary_mean_ci')}.csv", index=False)
    with plt.rc_context(rc_params(settings)):
        fig, ax = plt.subplots(figsize=as_figsize(settings, [12, 5.5]))
        x = np.arange(len(metrics))
        width = 0.8 / max(len(config.model_names), 1)
        model_colors = palette(settings)
        bar_alpha = float(settings.get("bar_alpha", 0.85))
        for model_idx, model_name in enumerate(config.model_names):
            df = summary[summary["model"] == model_name]
            offset = width * (model_idx - (len(config.model_names) - 1) / 2)
            ax.bar(
                x + offset,
                df["mean"],
                width,
                label=model_name,
                color=model_colors.get(model_name, model_colors["bar"]),
                alpha=bar_alpha,
            )
            ax.errorbar(x + offset, df["mean"], yerr=df["ci95"], fmt="none", ecolor="#222222", capsize=3, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(list(config.metric_summary_xticklabels))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.grid(True, axis="y", alpha=float(settings["grid_alpha"]))
        add_suptitle(fig, settings)
        legend_handles, legend_labels = model_color_handles(config, settings)
        apply_figure_legend(fig, legend_handles, legend_labels, settings)
        layout_with_bottom_legend(fig, settings)
        fig.tight_layout(pad=float(settings["tight_pad"]))
        save_figure(fig, output_dir / artifact_stem(config, "model_metric_summary"), settings)


def create_interactive_html(config: PlotConfig, input_dir: Path, output_dir: Path) -> None:
    if not config.create_interactive_html:
        return
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return
    settings = plot_settings(config, "interactive_learning_curves")
    colors = palette(settings)
    fig = make_subplots(rows=2, cols=2, subplot_titles=["Train Loss", "Val Loss", "Train Accuracy", "Val Accuracy"])
    for model_name in config.model_names:
        histories = [read_history(config, input_dir, model_name, fold) for fold in config.folds]
        summary = mean_ci_frame(histories, ["train_loss", "val_loss", "train_acc", "val_acc"])
        color = colors.get(model_name, colors["bar"])
        for idx, column in enumerate(["train_loss", "val_loss", "train_acc", "val_acc"]):
            row = 1 if idx < 2 else 2
            col = 1 if idx % 2 == 0 else 2
            fig.add_trace(
                go.Scatter(
                    x=summary["epoch"],
                    y=summary[f"{column}_mean"],
                    mode="lines",
                    name=f"{model_name} {column}",
                    line={"color": color, "width": settings["line_width"]},
                ),
                row=row,
                col=col,
            )
    fig.update_layout(
        title=str(settings["title"]),
        template="plotly_white",
        width=int(settings["width"]),
        height=int(settings["height"]),
    )
    fig.write_html(output_dir / f"{artifact_stem(config, 'mean_learning_curves_interactive')}.html")


def resolve_input_dir(config: PlotConfig) -> Path:
    script_dir = Path(__file__).resolve().parent
    configured = Path(config.input_dir)
    if configured.is_absolute():
        return configured
    candidates = [
        configured.resolve(),
        (Path.cwd() / configured).resolve(),
        script_dir / "outputs",
        script_dir,
        (script_dir.parent / configured).resolve(),
    ]
    for path in candidates:
        if (path / "histories").exists():
            return path
    return (Path.cwd() / configured).resolve()


def resolve_output_dir(config: PlotConfig, input_dir: Path) -> Path:
    configured = Path(config.output_dir)
    if configured.is_absolute():
        return configured
    if configured.parts[-2:] == ("figures", "mean_5fold") or configured.name == "mean_5fold":
        script_dir = Path(__file__).resolve().parent
        return script_dir / "figures" / "mean_5fold"
    return (Path.cwd() / configured).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional JSON config file.")
    parser.add_argument("--write-config-template", type=Path, help="Write default JSON config and exit.")
    parser.add_argument("--input-dir", type=Path, help="Override input directory.")
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    parser.add_argument("--dpi", type=int, help="Override global PNG DPI.")
    parser.add_argument("--no-html", action="store_true", help="Skip optional interactive HTML output.")
    parser.add_argument("--include-individual-folds", action="store_true", help="Draw faint individual fold curves where supported.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_config_template:
        save_config(args.write_config_template, PlotConfig())
        print(f"Wrote config template to {args.write_config_template}")
        return
    config = load_config(args.config)
    if args.input_dir:
        config.input_dir = str(args.input_dir)
    if args.output_dir:
        config.output_dir = str(args.output_dir)
    if args.dpi:
        config.dpi = args.dpi
    if args.no_html:
        config.create_interactive_html = False
    if args.include_individual_folds:
        config.include_individual_folds = True

    input_dir = resolve_input_dir(config)
    output_dir = resolve_output_dir(config, input_dir)
    config.folds = discover_available_folds(config, input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib(config)
    save_config(output_dir / "pretrained_gan_binary_mean_plot_config_used.json", config)

    predictions = read_predictions(config, input_dir)
    reports = read_reports(config, input_dir)
    plot_learning_curves(config, input_dir, output_dir)
    plot_confusion_matrices(config, predictions, output_dir)
    plot_roc_curves(config, predictions, output_dir)
    plot_per_class_metrics(config, reports, output_dir)
    plot_prediction_confidence(config, predictions, output_dir)
    plot_model_metric_summary(config, input_dir, output_dir)
    create_interactive_html(config, input_dir, output_dir)
    print(f"Saved pretrained + GAN-augmented binary mean 5-fold plots and summaries to {output_dir}")


if __name__ == "__main__":
    main()
