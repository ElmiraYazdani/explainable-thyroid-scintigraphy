#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    ResNet18_Weights,
    VGG11_Weights,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 5-fold BINARY (Non-Tumoral vs Tumoral) training/evaluation for "
            "VGG/DenseNet/ResNet/MobileNet/EfficientNet using ImageNet-pretrained "
            "weights on naively replicated (3x channel copy) grayscale thyroid "
            "scintigraphy images, trained on the GAN-augmented 5-fold split "
            "(real-only validation, all synthetic rows in every training fold). "
            "Merges the pretrained + GAN-augmented harness with the binary "
            "diagnostic task."
        )
    )
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path("../cf_dataset_preprocessed_gan_augmented_5fold"),
        help="Directory containing fold_*/train.json and val.json.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("../cf_dataset_preprocessed_gan_augmented/train"),
        help="Main dataset directory that contains images or an images/ subdirectory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("classification_pretrained_replication_5fold_gan_augmented_binary_outputs"),
        help="Root directory where all artifacts will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--vgg11-learning-rate",
        type=float,
        default=1e-4,
        help=(
            "Separate learning rate for VGG11 only. VGG11 has no batch "
            "normalization and reproducibly collapses to a one-class solution "
            "under AdamW at 1e-3 (observed in every prior 5-fold experiment); a "
            "lower rate is the conventional fine-tuning choice for it. The four "
            "batch-normalised architectures keep --learning-rate unchanged so "
            "they stay directly comparable to the prior experiments."
        ),
    )
    parser.add_argument(
        "--vgg11-grad-clip",
        type=float,
        default=2.0,
        help=(
            "Max global grad-norm for VGG11 only (0 disables). Second safeguard "
            "against the early-training divergence; not applied to other models."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["VGG11", "DenseNet121", "ResNet18", "MobileNetV2", "EfficientNetB0"],
        help="Subset of models to run.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="If >0, print fold progress every N epochs. Default 0 is quiet.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable extra progress logs.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Force epochs=2 and write to a distinct '_smoketest'-suffixed output dir.",
    )
    args = parser.parse_args()
    if args.smoke_test:
        args.epochs = 2
        args.output_dir = args.output_dir.parent / (args.output_dir.name + "_smoketest")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log(msg: str, enabled: bool = True) -> None:
    if enabled:
        print(msg)


LABEL_GROUP_MAP = {
    1: 1,
    3: 1,
    2: 2,
    5: 2,
    6: 2,
    4: 3,
    7: 4,
    8: 4,
}

# Map the original 4-class groups to the binary diagnostic task:
# 0 = Non-Tumoral (diffuse goiter, thyroiditis, normal), 1 = Tumoral.
BINARY_GROUP_MAP = {
    1: 0,  # Diffuse goiter
    2: 1,  # Tumoral
    3: 0,  # Thyroiditis
    4: 0,  # Normal
}

BINARY_CLASS_NAMES = {
    0: "Non-Tumoral",
    1: "Tumoral",
}

GROUP_NAMES = {
    1: "Diffuse goiter",
    2: "Tumoral",
    3: "Thyroiditis",
    4: "Normal",
}

POSITIVE_CLASS_IDX = 1  # "Tumoral"


class ThyroidDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.image_dir / str(row["image_file"])
        image_l = Image.open(image_path).convert("L")
        image = Image.merge("RGB", (image_l, image_l, image_l))
        label = int(row["label_idx"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(row["image_file"])


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "VGG11":
        model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model

    if model_name == "DenseNet121":
        model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if model_name == "ResNet18":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def load_fold_split_df(
    fold_root: Path,
    fold_id: int,
    split_name: str,
    label_to_idx: Dict[int, int],
) -> pd.DataFrame:
    split_name = split_name.lower()
    json_path = fold_root / f"fold_{fold_id}" / f"{split_name}.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = payload.get("data", [])
    if not rows:
        raise ValueError(f"No rows in {json_path}")

    df = pd.DataFrame(rows)
    # Defensive sanity check: the GAN-augmented fold split already guarantees
    # real-only validation (val_source_policy == "real_only"), but assert it here
    # so any future accidental change to the fold JSONs is caught at load time.
    if split_name == "val" and "source" in df.columns:
        bad = df.loc[df["source"] != "real", "image_file"].tolist()
        if bad:
            raise AssertionError(
                f"Fold {fold_id}/val contains non-real (synthetic) rows: {bad[:10]}"
            )
    df["label_raw"] = df["label"].astype(float).astype(int)
    unknown = sorted(set(df["label_raw"].unique()) - set(LABEL_GROUP_MAP.keys()))
    if unknown:
        raise ValueError(f"Fold {fold_id}/{split_name} has unknown labels: {unknown}")
    df["label_group"] = df["label_raw"].map(LABEL_GROUP_MAP).astype(int)
    df["binary_group"] = df["label_group"].map(BINARY_GROUP_MAP).astype(int)
    df["label_idx"] = df["binary_group"].astype(int)
    return df


def as_count_dict(series: pd.Series, sort: bool = True) -> Dict[object, int]:
    """value_counts() as plain Python ints so the summary CSV stays readable."""
    counts = series.value_counts()
    if sort:
        counts = counts.sort_index()
    return {
        (int(k) if isinstance(k, (int, np.integer)) else str(k)): int(v)
        for k, v in counts.items()
    }


def resolve_image_dir(fold_root: Path, image_root: Path) -> Path:
    fold_1_train = fold_root / "fold_1" / "train.json"
    payload = json.loads(fold_1_train.read_text(encoding="utf-8"))
    sample_name = payload["data"][0]["image_file"]

    candidates = [
        image_root,
        image_root / "images",
    ]
    metadata_path = fold_root / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_images = metadata.get("source_images_dir")
        if source_images:
            candidates.append(Path(source_images))

    for c in candidates:
        if (c / sample_name).exists():
            return c.resolve()

    raise FileNotFoundError(
        f"Could not locate images for sample '{sample_name}' under: {candidates}"
    )


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_dir: Path,
    batch_size: int,
    num_workers: int,
    img_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_ds = ThyroidDataset(train_df, image_dir, transform=train_transform)
    val_ds = ThyroidDataset(val_df, image_dir, transform=test_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, total_correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def infer_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    probs_list, preds_list, targets_list, files = [], [], [], []
    for images, labels, file_names in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        probs_list.append(probs)
        preds_list.append(preds)
        targets_list.append(labels.numpy())
        files.extend(list(file_names))
    return (
        np.concatenate(probs_list),
        np.concatenate(preds_list),
        np.concatenate(targets_list),
        files,
    )


def npv_ppv_from_cm(cm: np.ndarray) -> Tuple[List[float], List[float]]:
    npv_vals, ppv_vals = [], []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv_vals.append(npv)
        ppv_vals.append(ppv)
    return npv_vals, ppv_vals


def binarize_labels(y_true: np.ndarray, num_classes: int) -> np.ndarray:
    """Return one-hot labels with shape (n_samples, num_classes).

    sklearn.label_binarize returns shape (n_samples, 1) for binary inputs; expand
    that explicitly so the per-class ROC/AUC logic below works for 2 classes.
    """
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    if num_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
    return y_true_bin


def positive_class_metrics(
    cm: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """Clinically oriented metrics for the Tumoral (positive) class.

    In a 2-class problem the macro NPV/PPV reported alongside collapse to a single
    shared number, which hides the diagnostic asymmetry; these keep the positive
    class explicit so sensitivity/specificity can be read directly.
    """
    tp = float(cm[POSITIVE_CLASS_IDX, POSITIVE_CLASS_IDX])
    fn = float(cm[POSITIVE_CLASS_IDX, :].sum() - tp)
    fp = float(cm[:, POSITIVE_CLASS_IDX].sum() - tp)
    tn = float(cm.sum() - tp - fn - fp)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv_pos = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv_pos = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    y_pos = (y_true == POSITIVE_CLASS_IDX).astype(int)
    score_pos = y_prob[:, POSITIVE_CLASS_IDX]
    try:
        auc_binary = roc_auc_score(y_pos, score_pos)
    except ValueError:
        auc_binary = np.nan
    try:
        ap_binary = average_precision_score(y_pos, score_pos)
    except ValueError:
        ap_binary = np.nan

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv_tumoral": ppv_pos,
        "npv_tumoral": npv_pos,
        "f1_tumoral": f1_score(
            y_true, y_pred, pos_label=POSITIVE_CLASS_IDX, average="binary", zero_division=0
        ),
        "balanced_acc": float(np.nanmean([sensitivity, specificity])),
        "auc_binary": auc_binary,
        "average_precision": ap_binary,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    idx_to_label: Dict[int, int],
) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    y_true_bin = binarize_labels(y_true, num_classes)

    fpr_by_class, tpr_by_class, auc_by_class = {}, {}, {}
    for i in range(num_classes):
        try:
            fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            auc_i = auc(fpr_i, tpr_i)
        except ValueError:
            fpr_i, tpr_i, auc_i = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.nan
        fpr_by_class[i] = fpr_i
        tpr_by_class[i] = tpr_i
        auc_by_class[i] = auc_i

    try:
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        auc_micro = roc_auc_score(y_true_bin.ravel(), y_prob.ravel())
    except ValueError:
        fpr_micro, tpr_micro, auc_micro = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.nan

    try:
        auc_macro = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        auc_weighted = roc_auc_score(
            y_true_bin, y_prob, average="weighted", multi_class="ovr"
        )
    except ValueError:
        auc_macro, auc_weighted = np.nan, np.nan

    acc = accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa_val = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    npv_vals, ppv_vals = npv_ppv_from_cm(cm)
    npv_macro = float(np.nanmean(npv_vals))
    ppv_macro = float(np.nanmean(ppv_vals))

    report = classification_report(
        y_true,
        y_pred,
        target_names=[BINARY_CLASS_NAMES[i] for i in range(num_classes)],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T

    scalar_metrics = {
        "kappa_val": kappa_val,
        "micro_avg": f1_micro,
        "macro_avg": f1_macro,
        "npv": npv_macro,
        "ppv": ppv_macro,
        "mcc": mcc,
        "f1_score": f1_macro,
        "recall": recall_macro,
        "precision": precision_macro,
        "acc": acc,
        "auc": auc_macro,
        "auc_micro": auc_micro,
        "auc_weighted": auc_weighted,
        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }
    scalar_metrics.update(positive_class_metrics(cm, y_true, y_pred, y_prob))

    return {
        "cm": cm,
        "cm_norm": cm_norm,
        "fpr_by_class": fpr_by_class,
        "tpr_by_class": tpr_by_class,
        "auc_by_class": auc_by_class,
        "fpr_micro": fpr_micro,
        "tpr_micro": tpr_micro,
        "report_df": report_df,
        "scalar_metrics": scalar_metrics,
    }


def save_learning_curves(
    all_histories: Dict[str, Dict[int, pd.DataFrame]],
    model_names: List[str],
    fold_ids: List[int],
    figures_dir: Path,
) -> Dict[str, pd.DataFrame]:
    model_mean_histories: Dict[str, pd.DataFrame] = {}

    for model_name in model_names:
        hist_dict = all_histories[model_name]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for fold_id in fold_ids:
            h = hist_dict[fold_id]
            axes[0, 0].plot(h["epoch"], h["train_loss"], label=f"Fold {fold_id}")
            axes[0, 1].plot(h["epoch"], h["val_loss"], label=f"Fold {fold_id}")
            axes[1, 0].plot(h["epoch"], h["train_acc"], label=f"Fold {fold_id}")
            axes[1, 1].plot(h["epoch"], h["val_acc"], label=f"Fold {fold_id}")

        axes[0, 0].set_title(f"{model_name} Train Loss (all folds)")
        axes[0, 1].set_title(f"{model_name} Val Loss (all folds)")
        axes[1, 0].set_title(f"{model_name} Train Acc (all folds)")
        axes[1, 1].set_title(f"{model_name} Val Acc (all folds)")
        for ax in axes.ravel():
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        axes[0, 0].set_ylabel("Loss")
        axes[0, 1].set_ylabel("Loss")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 1].set_ylabel("Accuracy")
        plt.tight_layout()

        model_fig_dir = figures_dir / model_name
        model_fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(model_fig_dir / f"{model_name}_fold_learning_curves.png", dpi=180)
        plt.close(fig)

        all_h = []
        for fold_id in fold_ids:
            h = hist_dict[fold_id].copy()
            h["fold"] = fold_id
            all_h.append(h)
        all_h_df = pd.concat(all_h, ignore_index=True)
        model_mean_histories[model_name] = all_h_df.groupby("epoch", as_index=False).mean(
            numeric_only=True
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for model_name in model_names:
        h = model_mean_histories[model_name]
        axes[0, 0].plot(h["epoch"], h["train_loss"], label=model_name)
        axes[0, 1].plot(h["epoch"], h["val_loss"], label=model_name)
        axes[1, 0].plot(h["epoch"], h["train_acc"], label=model_name)
        axes[1, 1].plot(h["epoch"], h["val_acc"], label=model_name)

    axes[0, 0].set_title("Train Loss Comparison (mean across folds)")
    axes[0, 1].set_title("Val Loss Comparison (mean across folds)")
    axes[1, 0].set_title("Train Accuracy Comparison (mean across folds)")
    axes[1, 1].set_title("Val Accuracy Comparison (mean across folds)")
    for ax in axes.ravel():
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 1].set_ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(figures_dir / "all_models_mean_learning_curves.png", dpi=180)
    plt.close(fig)

    return model_mean_histories


def save_confusion_and_roc_plots(
    all_fold_results: Dict[str, Dict[int, Dict[str, object]]],
    model_names: List[str],
    fold_ids: List[int],
    idx_to_label: Dict[int, int],
    num_classes: int,
    figures_dir: Path,
) -> None:
    for model_name in model_names:
        model_fig_dir = figures_dir / model_name
        model_fig_dir.mkdir(parents=True, exist_ok=True)
        for fold_id in fold_ids:
            metrics_pack = all_fold_results[model_name][fold_id]["metrics_pack"]
            cm = metrics_pack["cm"]
            cm_norm = metrics_pack["cm_norm"]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=[BINARY_CLASS_NAMES[i] for i in range(num_classes)],
                yticklabels=[BINARY_CLASS_NAMES[i] for i in range(num_classes)],
                ax=axes[0],
            )
            axes[0].set_title(f"{model_name} Fold {fold_id} - CM (Counts)")
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("True")
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".2f",
                cmap="Oranges",
                xticklabels=[BINARY_CLASS_NAMES[i] for i in range(num_classes)],
                yticklabels=[BINARY_CLASS_NAMES[i] for i in range(num_classes)],
                ax=axes[1],
            )
            axes[1].set_title(f"{model_name} Fold {fold_id} - CM (Normalized)")
            axes[1].set_xlabel("Predicted")
            axes[1].set_ylabel("True")
            plt.tight_layout()
            plt.savefig(
                model_fig_dir / f"{model_name}_fold_{fold_id}_confusion_matrix.png", dpi=180
            )
            plt.close(fig)

            fig = plt.figure(figsize=(8, 6))
            for i in range(num_classes):
                plt.plot(
                    metrics_pack["fpr_by_class"][i],
                    metrics_pack["tpr_by_class"][i],
                    linewidth=1.6,
                    label=f"{BINARY_CLASS_NAMES[i]} (AUC={metrics_pack['auc_by_class'][i]:.3f})",
                )
            plt.plot(
                metrics_pack["fpr_micro"],
                metrics_pack["tpr_micro"],
                linestyle="--",
                linewidth=2.0,
                label="Micro ROC",
            )
            plt.plot([0, 1], [0, 1], "k--", linewidth=1)
            auc_macro = metrics_pack["scalar_metrics"]["auc"]
            auc_micro = metrics_pack["scalar_metrics"]["auc_micro"]
            auc_weighted = metrics_pack["scalar_metrics"]["auc_weighted"]
            plt.title(
                f"{model_name} Fold {fold_id} ROC | macro={auc_macro:.3f}, "
                f"micro={auc_micro:.3f}, weighted={auc_weighted:.3f}"
            )
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right", fontsize=9)
            plt.tight_layout()
            plt.savefig(model_fig_dir / f"{model_name}_fold_{fold_id}_roc.png", dpi=180)
            plt.close(fig)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._save_activations)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach().clone()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx]
        score.backward(retain_graph=True)
        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    def close(self):
        self._fwd.remove()
        self._bwd.remove()


def get_last_conv_layer(model: nn.Module) -> nn.Module:
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found for Grad-CAM target.")
    return last_conv


def overlay_cam(gray_img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = np.stack([gray_img, gray_img, gray_img], axis=-1)
    heatmap = plt.cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heatmap, 0, 1)


def save_gradcam_plots(
    best_models_df: pd.DataFrame,
    all_fold_results: Dict[str, Dict[int, Dict[str, object]]],
    image_dir: Path,
    model_names: List[str],
    num_classes: int,
    idx_to_label: Dict[int, int],
    img_size: int,
    figures_dir: Path,
    device: torch.device,
) -> None:
    test_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    for _, row in best_models_df.iterrows():
        model_name = str(row["model"])
        if model_name not in model_names:
            continue
        best_fold = int(row["best_fold"])
        best_ckpt = Path(row["best_model_path"])

        model = build_model(model_name, num_classes).to(device)
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        cam = GradCAM(model, get_last_conv_layer(model))

        fold_result = all_fold_results[model_name][best_fold]
        preds = fold_result["preds"]
        targets = fold_result["targets"]
        files = fold_result["files"]

        correct_idx = np.where(preds == targets)[0]
        wrong_idx = np.where(preds != targets)[0]
        sample_ids = list(correct_idx[:2]) + list(wrong_idx[:2])
        if len(sample_ids) == 0:
            sample_ids = list(range(min(4, len(files))))

        fig = plt.figure(figsize=(14, 3 * len(sample_ids)))
        for r, sample_i in enumerate(sample_ids):
            file_name = files[sample_i]
            true_idx = int(targets[sample_i])
            pred_idx = int(preds[sample_i])
            pil_img_l = Image.open(image_dir / file_name).convert("L")
            pil_img = Image.merge("RGB", (pil_img_l, pil_img_l, pil_img_l))
            input_tensor = test_transform(pil_img).unsqueeze(0).to(device)
            cam_map, cam_class = cam.generate(input_tensor, class_idx=pred_idx)
            img_np = np.array(pil_img_l.resize((img_size, img_size))).astype(np.float32) / 255.0
            overlay = overlay_cam(img_np, cam_map)

            plt.subplot(len(sample_ids), 3, 3 * r + 1)
            plt.imshow(img_np, cmap="gray")
            plt.title(f"Original\n{file_name}")
            plt.axis("off")
            plt.subplot(len(sample_ids), 3, 3 * r + 2)
            plt.imshow(cam_map, cmap="jet")
            plt.title(f"CAM\nPred={BINARY_CLASS_NAMES[cam_class]}")
            plt.axis("off")
            plt.subplot(len(sample_ids), 3, 3 * r + 3)
            plt.imshow(overlay)
            plt.title(
                f"Overlay\nTrue={BINARY_CLASS_NAMES[true_idx]} | "
                f"Pred={BINARY_CLASS_NAMES[pred_idx]}"
            )
            plt.axis("off")

        plt.tight_layout()
        model_fig_dir = figures_dir / model_name
        model_fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(model_fig_dir / f"{model_name}_best_fold_{best_fold}_gradcam.png", dpi=180)
        plt.close(fig)
        cam.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    sns.set_style("whitegrid")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_root = args.fold_root.resolve()
    if not fold_root.exists():
        raise FileNotFoundError(f"Fold root not found: {fold_root}")
    fold_dirs = sorted(
        [p for p in fold_root.glob("fold_*") if p.is_dir()],
        key=lambda x: int(x.name.split("_")[1]),
    )
    fold_ids = [int(p.name.split("_")[1]) for p in fold_dirs]
    if not fold_ids:
        raise ValueError(f"No fold_* directories found in {fold_root}")

    image_dir = resolve_image_dir(fold_root, args.image_root.resolve())

    classes = sorted(set(BINARY_GROUP_MAP.values()))
    label_to_idx = {label: label for label in classes}
    idx_to_label = {idx: idx for idx in classes}
    num_classes = len(classes)

    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    history_dir = output_dir / "histories"
    figures_dir = output_dir / "figures"
    predictions_dir = output_dir / "predictions"
    reports_dir = output_dir / "reports"
    for d in [output_dir, checkpoint_dir, metrics_dir, history_dir, figures_dir, predictions_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    run_config = {
        "device": str(device),
        "fold_root": str(fold_root),
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "fold_ids": fold_ids,
        "models": args.models,
        "epochs": args.epochs,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "per_model_learning_rate": {
            m: (args.vgg11_learning_rate if m == "VGG11" else args.learning_rate)
            for m in args.models
        },
        "vgg11_grad_clip_max_norm": args.vgg11_grad_clip,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "label_group_map": LABEL_GROUP_MAP,
        "binary_group_map": BINARY_GROUP_MAP,
        "binary_class_names": BINARY_CLASS_NAMES,
        "task": "binary_non_tumoral_vs_tumoral",
        "positive_class": BINARY_CLASS_NAMES[POSITIVE_CLASS_IDX],
        "smoke_test": args.smoke_test,
        "pretrained": True,
        "pretrained_weights": "IMAGENET1K_V1",
        "channel_construction": "naive_replication",
        "dataset_variant": "gan_augmented",
        "class_weighting": None,
        "val_source_policy": "real_only",
        "fold_data_composition": [
            {
                "fold": f["fold"],
                "train_real_rows": f["train_real_rows"],
                "train_synthetic_rows": f["train_synthetic_rows"],
                "val_rows": f["val_rows"],
            }
            for f in json.loads(
                (fold_root / "metadata.json").read_text(encoding="utf-8")
            )["folds"]
        ],
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "torchvision_version": __import__("torchvision").__version__,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    log(f"Device: {device}", enabled=True)
    log(f"Fold root: {fold_root}", enabled=True)
    log(f"Image dir: {image_dir}", enabled=True)
    log(f"Output dir: {output_dir}", enabled=True)

    fold_summary_rows = []
    for fold_id in fold_ids:
        train_df_tmp = load_fold_split_df(fold_root, fold_id, "train", label_to_idx)
        val_df_tmp = load_fold_split_df(fold_root, fold_id, "val", label_to_idx)
        fold_summary_rows.append(
            {
                "fold": fold_id,
                "train_rows": len(train_df_tmp),
                "val_rows": len(val_df_tmp),
                "train_class_dist": as_count_dict(train_df_tmp["label_idx"]),
                "val_class_dist": as_count_dict(val_df_tmp["label_idx"]),
                "train_group_dist": as_count_dict(train_df_tmp["label_group"]),
                "val_group_dist": as_count_dict(val_df_tmp["label_group"]),
                "train_source_dist": (
                    as_count_dict(train_df_tmp["source"], sort=False)
                    if "source" in train_df_tmp.columns
                    else {}
                ),
            }
        )
    pd.DataFrame(fold_summary_rows).to_csv(output_dir / "fold_data_summary.csv", index=False)

    all_histories: Dict[str, Dict[int, pd.DataFrame]] = {m: {} for m in args.models}
    all_fold_results: Dict[str, Dict[int, Dict[str, object]]] = {m: {} for m in args.models}
    model_metrics_raw: Dict[str, pd.DataFrame] = {}
    model_metrics_with_stats: Dict[str, pd.DataFrame] = {}
    best_model_summary_rows: List[Dict[str, object]] = []

    for model_name in args.models:
        log(f"Running model: {model_name}", enabled=True)
        model_ckpt_dir = checkpoint_dir / model_name
        model_hist_dir = history_dir / model_name
        model_pred_dir = predictions_dir / model_name
        model_report_dir = reports_dir / model_name
        for d in [model_ckpt_dir, model_hist_dir, model_pred_dir, model_report_dir, figures_dir / model_name]:
            d.mkdir(parents=True, exist_ok=True)

        fold_metric_rows = []
        best_overall_acc = -1.0
        best_overall_fold = None
        best_overall_ckpt = model_ckpt_dir / f"{model_name}_best_overall.pth"

        for fold_id in fold_ids:
            log(f"  Fold {fold_id}...", enabled=True)

            train_df = load_fold_split_df(fold_root, fold_id, "train", label_to_idx)
            val_df = load_fold_split_df(fold_root, fold_id, "val", label_to_idx)
            train_loader, val_loader = make_loaders(
                train_df=train_df,
                val_df=val_df,
                image_dir=image_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                img_size=args.img_size,
            )

            model = build_model(model_name, num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            model_lr = (
                args.vgg11_learning_rate
                if model_name == "VGG11"
                else args.learning_rate
            )
            model_grad_clip = args.vgg11_grad_clip if model_name == "VGG11" else 0.0
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=model_lr, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-6
            )

            history = defaultdict(list)
            best_fold_val_acc = -1.0
            best_fold_ckpt = model_ckpt_dir / f"{model_name}_fold_{fold_id}_best.pth"

            for epoch in range(1, args.epochs + 1):
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device,
                    grad_clip=model_grad_clip,
                )
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                current_lr = optimizer.param_groups[0]["lr"]

                history["epoch"].append(epoch)
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["lr"].append(current_lr)

                if val_acc > best_fold_val_acc:
                    best_fold_val_acc = val_acc
                    torch.save(model.state_dict(), best_fold_ckpt)

                scheduler.step()

                if args.log_every > 0 and (epoch % args.log_every == 0):
                    log(
                        f"    epoch {epoch:03d}/{args.epochs} "
                        f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
                    )

            history_df = pd.DataFrame(history)
            all_histories[model_name][fold_id] = history_df
            history_df.to_csv(
                model_hist_dir / f"{model_name}_fold_{fold_id}_history.csv", index=False
            )

            model.load_state_dict(torch.load(best_fold_ckpt, map_location=device))
            probs, preds, targets, files = infer_model(model, val_loader, device)
            metrics_pack = compute_fold_metrics(targets, preds, probs, num_classes, idx_to_label)
            scalar_metrics = metrics_pack["scalar_metrics"]

            metrics_pack["report_df"].to_csv(
                model_report_dir / f"{model_name}_fold_{fold_id}_classification_report.csv"
            )

            # Keep the original 4-class group on every prediction row so the
            # binary errors can be attributed back to the sub-diagnosis they came
            # from (see make_binary_extra_plots.py).
            group_by_file = dict(
                zip(val_df["image_file"].astype(str), val_df["label_group"].astype(int))
            )
            raw_by_file = dict(
                zip(val_df["image_file"].astype(str), val_df["label_raw"].astype(int))
            )
            pred_df = pd.DataFrame(
                {
                    "image_file": files,
                    "true_idx": targets,
                    "pred_idx": preds,
                    "true_group_label": [idx_to_label[int(i)] for i in targets],
                    "pred_group_label": [idx_to_label[int(i)] for i in preds],
                    "true_class_name": [BINARY_CLASS_NAMES[int(i)] for i in targets],
                    "pred_class_name": [BINARY_CLASS_NAMES[int(i)] for i in preds],
                    "original_group": [group_by_file.get(str(f), -1) for f in files],
                    "original_group_name": [
                        GROUP_NAMES.get(group_by_file.get(str(f), -1), "unknown")
                        for f in files
                    ],
                    "original_label_raw": [raw_by_file.get(str(f), -1) for f in files],
                    "max_prob": probs.max(axis=1),
                }
            )
            for c in range(num_classes):
                pred_df[f"prob_{BINARY_CLASS_NAMES[c]}"] = probs[:, c]
            pred_df.to_csv(
                model_pred_dir / f"{model_name}_fold_{fold_id}_predictions.csv", index=False
            )

            row = {"fold": fold_id, "best_epoch_val_acc": best_fold_val_acc}
            row.update(scalar_metrics)
            fold_metric_rows.append(row)

            all_fold_results[model_name][fold_id] = {
                "metrics_pack": metrics_pack,
                "preds": preds,
                "targets": targets,
                "files": files,
                "best_fold_ckpt": str(best_fold_ckpt),
            }

            if scalar_metrics["acc"] > best_overall_acc:
                best_overall_acc = scalar_metrics["acc"]
                best_overall_fold = fold_id
                shutil.copy2(best_fold_ckpt, best_overall_ckpt)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            log(
                f"    done fold {fold_id}: best_val_acc={best_fold_val_acc:.4f}, "
                f"test_acc={scalar_metrics['acc']:.4f}",
                enabled=args.verbose,
            )

        fold_df = pd.DataFrame(fold_metric_rows).sort_values("fold").reset_index(drop=True)
        model_metrics_raw[model_name] = fold_df

        numeric_cols = fold_df.select_dtypes(include=[np.number]).columns.tolist()
        mean_row = {col: fold_df[col].mean() for col in numeric_cols}
        mean_row["fold"] = "mean"
        std_row = {col: fold_df[col].std(ddof=1) for col in numeric_cols}
        std_row["fold"] = "std"
        fold_df_with_stats = pd.concat([fold_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
        model_metrics_with_stats[model_name] = fold_df_with_stats

        fold_metrics_csv = metrics_dir / f"{model_name}_fold_metrics.csv"
        fold_df_with_stats.to_csv(fold_metrics_csv, index=False)
        (metrics_dir / f"{model_name}_fold_metrics_with_mean_std.json").write_text(
            fold_df_with_stats.to_json(orient="records", indent=2), encoding="utf-8"
        )

        best_model_summary_rows.append(
            {
                "model": model_name,
                "best_fold": best_overall_fold,
                "best_fold_test_acc": best_overall_acc,
                "best_model_path": str(best_overall_ckpt),
                "metrics_csv": str(fold_metrics_csv),
            }
        )
        log(
            f"  {model_name} complete. best_fold={best_overall_fold}, best_acc={best_overall_acc:.4f}",
            enabled=True,
        )

    best_models_df = pd.DataFrame(best_model_summary_rows)
    best_models_df.to_csv(metrics_dir / "best_models_summary.csv", index=False)

    save_learning_curves(
        all_histories=all_histories,
        model_names=args.models,
        fold_ids=fold_ids,
        figures_dir=figures_dir,
    )
    save_confusion_and_roc_plots(
        all_fold_results=all_fold_results,
        model_names=args.models,
        fold_ids=fold_ids,
        idx_to_label=idx_to_label,
        num_classes=num_classes,
        figures_dir=figures_dir,
    )

    summary_rows = []
    for model_name in args.models:
        fold_df = model_metrics_raw[model_name]
        summary_rows.append(
            {
                "model": model_name,
                "kappa_val_mean": fold_df["kappa_val"].mean(),
                "kappa_val_std": fold_df["kappa_val"].std(ddof=1),
                "micro_avg_mean": fold_df["micro_avg"].mean(),
                "macro_avg_mean": fold_df["macro_avg"].mean(),
                "npv_mean": fold_df["npv"].mean(),
                "ppv_mean": fold_df["ppv"].mean(),
                "mcc_mean": fold_df["mcc"].mean(),
                "f1_score_mean": fold_df["f1_score"].mean(),
                "recall_mean": fold_df["recall"].mean(),
                "precision_mean": fold_df["precision"].mean(),
                "acc_mean": fold_df["acc"].mean(),
                "auc_mean": fold_df["auc"].mean(),
                "auc_micro_mean": fold_df["auc_micro"].mean(),
                "auc_weighted_mean": fold_df["auc_weighted"].mean(),
                "sensitivity_mean": fold_df["sensitivity"].mean(),
                "sensitivity_std": fold_df["sensitivity"].std(ddof=1),
                "specificity_mean": fold_df["specificity"].mean(),
                "specificity_std": fold_df["specificity"].std(ddof=1),
                "balanced_acc_mean": fold_df["balanced_acc"].mean(),
                "balanced_acc_std": fold_df["balanced_acc"].std(ddof=1),
                "ppv_tumoral_mean": fold_df["ppv_tumoral"].mean(),
                "npv_tumoral_mean": fold_df["npv_tumoral"].mean(),
                "f1_tumoral_mean": fold_df["f1_tumoral"].mean(),
                "auc_binary_mean": fold_df["auc_binary"].mean(),
                "auc_binary_std": fold_df["auc_binary"].std(ddof=1),
                "average_precision_mean": fold_df["average_precision"].mean(),
                "acc_std": fold_df["acc"].std(ddof=1),
                "f1_score_std": fold_df["f1_score"].std(ddof=1),
                "mcc_std": fold_df["mcc"].std(ddof=1),
            }
        )
    model_summary_df = pd.DataFrame(summary_rows).sort_values("acc_mean", ascending=False)
    model_summary_df.to_csv(metrics_dir / "all_models_summary_mean_std.csv", index=False)

    metric_cols = [
        "acc_mean",
        "balanced_acc_mean",
        "sensitivity_mean",
        "specificity_mean",
        "precision_mean",
        "recall_mean",
        "f1_score_mean",
        "f1_tumoral_mean",
        "kappa_val_mean",
        "mcc_mean",
        "auc_binary_mean",
        "average_precision_mean",
    ]
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    for idx, metric_col in enumerate(metric_cols):
        ax = axes[idx // 3, idx % 3]
        sns.barplot(data=model_summary_df, x="model", y=metric_col, ax=ax)
        ax.set_title(metric_col)
        ax.tick_params(axis="x", rotation=25)
        if metric_col not in ["kappa_val_mean", "mcc_mean"]:
            ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(figures_dir / "all_models_metric_comparison.png", dpi=180)
    plt.close(fig)

    save_gradcam_plots(
        best_models_df=best_models_df,
        all_fold_results=all_fold_results,
        image_dir=image_dir,
        model_names=args.models,
        num_classes=num_classes,
        idx_to_label=idx_to_label,
        img_size=args.img_size,
        figures_dir=figures_dir,
        device=device,
    )

    all_fold_metrics_rows = []
    for model_name in args.models:
        df = model_metrics_raw[model_name].copy()
        df.insert(0, "model", model_name)
        all_fold_metrics_rows.append(df)
    all_fold_metrics_df = pd.concat(all_fold_metrics_rows, ignore_index=True)
    all_fold_metrics_df.to_csv(metrics_dir / "all_models_all_folds_metrics.csv", index=False)
    (metrics_dir / "all_models_all_folds_metrics.json").write_text(
        all_fold_metrics_df.to_json(orient="records", indent=2), encoding="utf-8"
    )

    log("Done. Artifacts saved in: " + str(output_dir), enabled=True)


if __name__ == "__main__":
    main()
