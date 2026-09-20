import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import (
    DenseNet121_Weights,
    MobileNet_V2_Weights,
    VGG11_Weights,
)
from collections import Counter
import pandas as pd
import numpy as np
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fixed ImageNet normalization is mandatory for pretrained transfer learning here
# (same reasoning as `Pretrained Naive Replication 5 Folds GAN Augmented`).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BACKBONE_CHOICES = ["DenseNet121", "MobileNetV2", "VGG11"]

# Per-backbone stabilization. DenseNet121 / MobileNetV2 keep the original script's
# hardcoded rates (expert lr 1e-4, fusion lr 1e-3, grad clip 1.0). VGG11 mirrors the
# stabilization used in Pretrained Naive Replication 5 Folds GAN Augmented's run_config.json
# (per_model_learning_rate.VGG11 = 1e-4, vgg11_grad_clip_max_norm = 2.0) for BOTH stages.
BACKBONE_TRAIN_CFG = {
    "DenseNet121": {"expert_lr": 1e-4, "fusion_lr": 1e-3, "grad_clip_norm": 1.0},
    "MobileNetV2": {"expert_lr": 1e-4, "fusion_lr": 1e-3, "grad_clip_norm": 1.0},
    "VGG11": {"expert_lr": 1e-4, "fusion_lr": 1e-4, "grad_clip_norm": 2.0},
}


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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Mixture of Experts with CNN fusion (image + expert logits) on the "
            "GAN-augmented 5-fold thyroid dataset, with every expert and the fusion "
            "network backed by an ImageNet-pretrained backbone."
        )
    )
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=BACKBONE_CHOICES,
        help="Pretrained backbone used for all 4 experts AND the fusion feature extractor.",
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
        required=True,
        help="Root directory where all artifacts will be saved (pass explicitly per backbone).",
    )
    parser.add_argument("--expert-epochs", type=int, default=100)
    parser.add_argument("--fusion-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Optional cap on number of folds (useful for smoke tests).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Force expert_epochs=2, fusion_epochs=2, max_folds=1 and append '_smoketest' "
            "to the output directory name so smoke artifacts can't collide with real runs."
        ),
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_project_path(path):
    path = Path(path)
    if path.exists() or path.is_absolute():
        return path.resolve()

    project_candidate = PROJECT_ROOT / path
    if project_candidate.exists():
        return project_candidate.resolve()

    project_named_candidate = PROJECT_ROOT / path.name
    if project_named_candidate.exists():
        return project_named_candidate.resolve()

    return path.resolve()


def load_fold_split_df(fold_root, fold_id, split_name, label_to_idx):
    json_path = Path(fold_root) / f"fold_{fold_id}" / f"{split_name.lower()}.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = payload.get("data", [])
    if not rows:
        raise ValueError(f"No rows in {json_path}")

    df = pd.DataFrame(rows)
    df["label_raw"] = df["label"].astype(float).astype(int)
    unknown = sorted(set(df["label_raw"].unique()) - set(LABEL_GROUP_MAP.keys()))
    if unknown:
        raise ValueError(f"Fold {fold_id}/{split_name} has unknown labels: {unknown}")
    df["label_group"] = df["label_raw"].map(LABEL_GROUP_MAP).astype(int)
    df["grouped_label"] = df["label_group"].map(label_to_idx).astype(int)

    # Defensive: validation must be real-only (val_source_policy == "real_only").
    if split_name.lower() == "val":
        if "source" not in df.columns:
            raise ValueError(
                f"Fold {fold_id}/val JSON has no 'source' column; cannot verify real_only policy"
            )
        non_real = df.loc[df["source"] != "real", "image_file"].tolist()
        if non_real:
            raise ValueError(
                f"Fold {fold_id}/val contains {len(non_real)} non-real rows, violating "
                f"val_source_policy=real_only. First offenders: {non_real[:10]}"
            )
        print(f"[assert] Fold {fold_id}/val: all {len(df)} rows have source == 'real'.")
    return df


def resolve_image_dir(fold_root, image_root):
    fold_1_train = Path(fold_root) / "fold_1" / "train.json"
    payload = json.loads(fold_1_train.read_text(encoding="utf-8"))
    sample_name = payload["data"][0]["image_file"]

    candidates = [
        Path(image_root),
        Path(image_root) / "images",
    ]
    metadata_path = Path(fold_root) / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_images = metadata.get("source_images_dir")
        if source_images:
            candidates.append(Path(source_images))

    for candidate in candidates:
        if (candidate / sample_name).exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not locate images for sample '{sample_name}' under: {candidates}"
    )


# Custom preprocessing function for thyroid scintigraphy images
class ThyroidPreprocessor:
    def __init__(self, target_size=128):
        self.target_size = target_size

    def __call__(self, image):
        # Convert to grayscale first to preserve medical image information
        if image.mode != 'L':
            image = image.convert('L')

        # Convert grayscale to RGB for model compatibility (3-channel replication)
        image = image.convert('RGB')

        width, height = image.size

        if width == self.target_size and height == self.target_size:
            return image

        shortest_side = min(width, height)
        left = (width - shortest_side) // 2
        top = (height - shortest_side) // 2
        right = left + shortest_side
        bottom = top + shortest_side
        image = image.crop((left, top, right, bottom))

        if shortest_side != self.target_size:
            image = image.resize((self.target_size, self.target_size))

        return image


def compute_mean_std(dataset, batch_size=64):
    """Diagnostic only: dataset's own raw pixel mean/std (NOT used for training transforms)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in tqdm(loader, desc="Computing mean/std (diagnostic)"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean.tolist(), std.tolist()


# Binary Expert Dataset - converts multi-class to binary for a specific class
class BinaryExpertDataset(Dataset):
    def __init__(self, df, image_dir, mean, std, target_class, is_train=True):
        self.df = df
        self.image_dir = image_dir
        self.target_class = target_class
        self.is_train = is_train

        self.original_samples = []
        for idx, row in df.iterrows():
            binary_label = 1 if row['grouped_label'] == target_class else 0
            self.original_samples.append({
                'image_path': os.path.join(image_dir, row['image_file']),
                'label': binary_label,
                'original_label': row['grouped_label']
            })

        self.class_counts = Counter([sample['label'] for sample in self.original_samples])

        if is_train:
            # Per-expert one-vs-rest oversampling. This addresses a DIFFERENT imbalance
            # axis than the GAN 4-way balancing and must stay active for every backbone.
            self.samples = self._create_balanced_samples()
            balanced_counts = Counter([sample['label'] for sample in self.samples])
            print(f"Expert {target_class} - Original: {dict(self.class_counts)}, Balanced: {dict(balanced_counts)}")
        else:
            self.samples = []
            for sample in self.original_samples:
                new_sample = sample.copy()
                new_sample['augment'] = False
                self.samples.append(new_sample)

        self.base_transform = transforms.Compose([
            ThyroidPreprocessor(target_size=128),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.aug_transform = transforms.Compose([
            ThyroidPreprocessor(target_size=128),
            transforms.RandomRotation(degrees=(-45, 45)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def _create_balanced_samples(self):
        """Balance positive and negative samples via oversampling with augmentation."""
        balanced_samples = []

        samples_by_class = {0: [], 1: []}
        for sample in self.original_samples:
            samples_by_class[sample['label']].append(sample)

        max_count = max(len(samples_by_class[0]), len(samples_by_class[1]))

        for class_label, class_samples in samples_by_class.items():
            original_count = len(class_samples)

            for sample in class_samples:
                new_sample = sample.copy()
                new_sample['augment'] = False
                balanced_samples.append(new_sample)

            if original_count < max_count:
                needed = max_count - original_count
                for i in range(needed):
                    base_sample = class_samples[i % original_count].copy()
                    base_sample['augment'] = True
                    balanced_samples.append(base_sample)

        return balanced_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample['image_path'])

        if sample.get('augment', False):
            image = self.aug_transform(image)
        else:
            image = self.base_transform(image)

        label = sample['label']
        original_label = sample['original_label']

        return image, label, original_label


# Validation Dataset (for multi-class evaluation / fusion training)
class ThyroidValidationDataset(Dataset):
    def __init__(self, df, image_dir, mean, std):
        self.df = df
        self.image_dir = image_dir

        self.samples = []
        for idx, row in df.iterrows():
            self.samples.append({
                'image_path': os.path.join(image_dir, row['image_file']),
                'label': row['grouped_label']
            })

        self.transform = transforms.Compose([
            ThyroidPreprocessor(target_size=128),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample['image_path'])
        image = self.transform(image)

        label = sample['label']
        return image, label


device = "cuda" if torch.cuda.is_available() else "cpu"


def build_pretrained_binary_expert(backbone_name, num_classes=2):
    """ImageNet-pretrained backbone with its final classification layer replaced by a
    2-way head. First conv is left untouched (3-channel-replicated input already matches)."""
    if backbone_name == "VGG11":
        model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif backbone_name == "DenseNet121":
        model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif backbone_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    return model


class _DenseFeat(nn.Module):
    """DenseNet121 embedding path: features -> relu -> global avg pool -> flatten (1024-d).
    Mirrors torchvision's DenseNet.forward up to, but not including, classifier."""

    def __init__(self, backbone):
        super().__init__()
        self.features = backbone.features

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)


class _MobileFeat(nn.Module):
    """MobileNetV2 embedding path: features -> global avg pool -> flatten (1280-d)."""

    def __init__(self, backbone):
        super().__init__()
        self.features = backbone.features

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)


class _VGGFeat(nn.Module):
    """VGG11 embedding path: features -> avgpool -> flatten -> classifier minus its final
    Linear (keeps the two hidden Linear+ReLU+Dropout blocks) -> 4096-d."""

    def __init__(self, backbone):
        super().__init__()
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(*list(backbone.classifier.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_pretrained_feature_extractor(backbone_name):
    """Returns (feature_extractor_module, embedding_dim). Its own classification head is
    discarded entirely - the fusion MLP does the final classification."""
    if backbone_name == "VGG11":
        model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        return _VGGFeat(model), 4096
    elif backbone_name == "DenseNet121":
        model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        return _DenseFeat(model), 1024
    elif backbone_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        return _MobileFeat(model), 1280
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


class PretrainedBinaryExpert(nn.Module):
    """Binary one-vs-rest expert backed by an ImageNet-pretrained backbone."""

    def __init__(self, backbone_name, dropout_rate=0.3):
        super().__init__()
        self.backbone_name = backbone_name
        self.net = build_pretrained_binary_expert(backbone_name, num_classes=2)

    def forward(self, x):
        return self.net(x)

    def get_positive_logit(self, x):
        """Logit for the positive class (class 1) - consumed by the fusion network."""
        return self.forward(x)[:, 1]


class PretrainedFusionNet(nn.Module):
    """Fusion network: pretrained backbone feature extractor (embedding only) + expert
    positive logits -> MLP -> num_classes. Same MLP shape as the original FusionCNN,
    only feature_dim changes (from 32768 to the backbone embedding dim)."""

    def __init__(self, backbone_name, num_experts=4, num_classes=4, dropout_rate=0.3):
        super().__init__()
        self.num_experts = num_experts
        self.feature_extractor, feature_dim = build_pretrained_feature_extractor(backbone_name)
        self.feature_dim = feature_dim

        mlp_in_dim = feature_dim + num_experts
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, images, expert_logits):
        if expert_logits.dim() != 2 or expert_logits.size(1) != self.num_experts:
            raise ValueError(
                f"expert_logits must be [B, {self.num_experts}], got {tuple(expert_logits.shape)}"
            )

        feats = self.feature_extractor(images)
        fused = torch.cat([feats, expert_logits], dim=1)
        return self.mlp(fused)


class MixtureOfExperts:
    """Mixture of Experts ensemble for 4-class classification with pretrained-backbone
    experts and a pretrained-backbone fusion feature extractor."""

    def __init__(self, backbone_name, num_classes=4, dropout_rate=0.3):
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        cfg = BACKBONE_TRAIN_CFG[backbone_name]
        self.expert_lr = cfg["expert_lr"]
        self.fusion_lr = cfg["fusion_lr"]
        self.grad_clip_norm = cfg["grad_clip_norm"]

        self.experts = [
            PretrainedBinaryExpert(backbone_name, dropout_rate=dropout_rate)
            for _ in range(num_classes)
        ]
        self.fusion_model = PretrainedFusionNet(
            backbone_name,
            num_experts=num_classes,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
        self.training_histories = {
            i: {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
            for i in range(num_classes)
        }
        self.fusion_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def to(self, device):
        for expert in self.experts:
            expert.to(device)
        self.fusion_model.to(device)
        return self

    def _collect_expert_logits(self, images):
        """Collect positive-class logits from all frozen experts: [B, num_experts]."""
        expert_logits = []
        for expert in self.experts:
            positive_logit = expert.get_positive_logit(images)
            expert_logits.append(positive_logit)
        return torch.stack(expert_logits, dim=1)

    def train_expert(self, expert_idx, train_dataloader, val_dataloader, num_epochs=100, save_path=None):
        """Train a single expert."""
        expert = self.experts[expert_idx]
        expert.to(device)

        optimizer = optim.AdamW(expert.parameters(), lr=self.expert_lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_acc = -1.0
        best_model_state = None

        history = self.training_histories[expert_idx]

        for epoch in range(num_epochs):
            expert.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for images, labels, _ in tqdm(train_dataloader, desc=f"Expert {expert_idx} Epoch {epoch+1}/{num_epochs}", leave=False):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = expert(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(expert.parameters(), max_norm=self.grad_clip_norm)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            avg_train_loss = train_loss / len(train_dataloader)
            train_acc = 100 * train_correct / train_total

            expert.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels, _ in val_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = expert(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_dataloader)
            val_acc = 100 * val_correct / val_total

            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)

            print(f"Expert {expert_idx} Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, "
                  f"Train Acc {train_acc:.2f}%, Val Loss {avg_val_loss:.4f}, Val Acc {val_acc:.2f}%")

            scheduler.step(avg_val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.clone() for k, v in expert.state_dict().items()}

        expert.load_state_dict(best_model_state)
        if save_path is None:
            save_path = f'moe_expert_{expert_idx}_cnn_best.pth'
        torch.save(best_model_state, save_path)
        print(f"Expert {expert_idx} - Best Val Accuracy: {best_val_acc:.2f}%")
        return best_val_acc

    def train_fusion(self, train_dataloader, val_dataloader, num_epochs=50, save_path=None):
        """Train the fusion network using frozen expert logits + input images."""
        for expert in self.experts:
            expert.eval()
            for param in expert.parameters():
                param.requires_grad = False

        self.fusion_model.to(device)
        self.fusion_model.train()

        optimizer = optim.AdamW(self.fusion_model.parameters(), lr=self.fusion_lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        best_val_acc = -1.0
        best_model_state = None

        for epoch in range(num_epochs):
            self.fusion_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for images, labels in tqdm(
                train_dataloader, desc=f"Fusion Epoch {epoch+1}/{num_epochs}", leave=False
            ):
                images, labels = images.to(device), labels.to(device)

                with torch.no_grad():
                    stacked_logits = self._collect_expert_logits(images)

                optimizer.zero_grad()
                outputs = self.fusion_model(images, stacked_logits)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fusion_model.parameters(), max_norm=self.grad_clip_norm)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            avg_train_loss = train_loss / len(train_dataloader)
            train_acc = 100 * train_correct / train_total

            self.fusion_model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    stacked_logits = self._collect_expert_logits(images)
                    outputs = self.fusion_model(images, stacked_logits)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_dataloader)
            val_acc = 100 * val_correct / val_total

            self.fusion_history["train_loss"].append(avg_train_loss)
            self.fusion_history["train_acc"].append(train_acc)
            self.fusion_history["val_loss"].append(avg_val_loss)
            self.fusion_history["val_acc"].append(val_acc)

            print(
                f"Fusion Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, "
                f"Train Acc {train_acc:.2f}%, Val Loss {avg_val_loss:.4f}, Val Acc {val_acc:.2f}%"
            )

            scheduler.step(avg_val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.clone() for k, v in self.fusion_model.state_dict().items()}

        self.fusion_model.load_state_dict(best_model_state)
        if save_path is None:
            save_path = "moe_fusion_cnn_best.pth"
        torch.save(best_model_state, save_path)
        print(f"Fusion CNN - Best Val Accuracy: {best_val_acc:.2f}%")
        return best_val_acc

    def predict(self, images, use_fusion=True):
        """Make predictions using all experts and the fusion network."""
        all_logits = []

        for expert in self.experts:
            expert.eval()
            with torch.no_grad():
                positive_logits = expert.get_positive_logit(images)
                all_logits.append(positive_logits)

        stacked_logits = torch.stack(all_logits, dim=1)

        if use_fusion:
            self.fusion_model.eval()
            with torch.no_grad():
                fusion_output = self.fusion_model(images, stacked_logits)
                predictions = torch.argmax(fusion_output, dim=1)
        else:
            predictions = torch.argmax(stacked_logits, dim=1)

        return predictions, stacked_logits

    def evaluate(self, dataloader):
        """Evaluate the ensemble on a dataset."""
        all_predictions = []
        all_labels = []
        all_logits = []

        for images, labels in tqdm(dataloader, desc="Evaluating ensemble"):
            images = images.to(device)
            predictions, logits = self.predict(images)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_logits.append(logits.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_logits = np.vstack(all_logits)

        accuracy = 100 * np.mean(all_predictions == all_labels)

        return accuracy, all_predictions, all_labels, all_logits


def plot_training_curves(moe, save_path='moe_training_curves.png'):
    """Plot training curves for all experts and the fusion network."""
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    class_names = ['Class 0 (Goiter/Thyroiditis)', 'Class 1 (MNG/Nodules)',
                   'Class 2 (Cold Nodule)', 'Class 3 (Normal/Warm)']

    for i in range(4):
        history = moe.training_histories[i]
        epochs = range(1, len(history['train_loss']) + 1)

        axes[0, i].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        axes[0, i].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        axes[0, i].set_title(f'Expert {i}: {class_names[i]}\nLoss')
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].legend()
        axes[0, i].grid(True)

        axes[1, i].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
        axes[1, i].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
        axes[1, i].set_title(f'Expert {i}: {class_names[i]}\nAccuracy')
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Accuracy (%)')
        axes[1, i].legend()
        axes[1, i].grid(True)

    fusion_history = moe.fusion_history
    if len(fusion_history['train_loss']) > 0:
        epochs = range(1, len(fusion_history['train_loss']) + 1)

        axes[0, 4].plot(epochs, fusion_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 4].plot(epochs, fusion_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 4].set_title('Fusion Net\nLoss')
        axes[0, 4].set_xlabel('Epoch')
        axes[0, 4].set_ylabel('Loss')
        axes[0, 4].legend()
        axes[0, 4].grid(True)

        axes[1, 4].plot(epochs, fusion_history['train_acc'], 'b-', label='Train Acc')
        axes[1, 4].plot(epochs, fusion_history['val_acc'], 'r-', label='Val Acc')
        axes[1, 4].set_title('Fusion Net\nAccuracy')
        axes[1, 4].set_xlabel('Epoch')
        axes[1, 4].set_ylabel('Accuracy (%)')
        axes[1, 4].legend()
        axes[1, 4].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_expert_decisions(all_logits, all_labels, all_predictions, save_path='moe_expert_decisions.png'):
    """Plot expert decisions on the validation set."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

    for i, ax in enumerate(axes.flat):
        class_mask = all_labels == i
        class_logits = all_logits[class_mask]
        class_preds = all_predictions[class_mask]

        if len(class_logits) > 0:
            bp = ax.boxplot([class_logits[:, j] for j in range(4)],
                           tick_labels=[f'Expert {j}' for j in range(4)],
                           patch_artist=True)

            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            bp['boxes'][i].set_facecolor('#2ECC71')
            bp['boxes'][i].set_alpha(1.0)

            correct_count = np.sum(class_preds == i)
            total_count = len(class_preds)

            ax.set_title(f'{class_names[i]} Samples (n={total_count})\n'
                        f'Correctly classified: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)')
            ax.set_ylabel('Positive Logit')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Expert Logit Distributions by True Class\n(Green = Correct Expert)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Expert decisions plot saved to {save_path}")


def plot_confusion_matrix(all_labels, all_predictions, save_path='moe_confusion_matrix.png'):
    """Plot confusion matrix for the ensemble."""
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2, 3])

    plt.figure(figsize=(10, 8))

    class_names = ['Class 0\n(Goiter/Thyroiditis)', 'Class 1\n(MNG/Nodules)',
                   'Class 2\n(Cold Nodule)', 'Class 3\n(Normal/Warm)']

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title('Mixture of Experts CNN Fusion - Confusion Matrix\n(Validation Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_logit_heatmap(all_logits, all_labels, save_path='moe_logit_heatmap.png'):
    """Plot average logits for each true class."""
    avg_logits = np.zeros((4, 4))

    for true_class in range(4):
        mask = all_labels == true_class
        if np.sum(mask) > 0:
            avg_logits[true_class] = np.mean(all_logits[mask], axis=0)

    plt.figure(figsize=(10, 8))

    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

    sns.heatmap(avg_logits, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                xticklabels=[f'Expert {i}' for i in range(4)],
                yticklabels=class_names)

    plt.title('Average Expert Logits by True Class\n(Higher = More Confident)')
    plt.xlabel('Expert')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Logit heatmap saved to {save_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    sns.set_style("whitegrid")

    if args.smoke_test:
        args.expert_epochs = 2
        args.fusion_epochs = 2
        args.max_folds = 1

    print(f"Using device: {device}")
    print(f"Backbone: {args.backbone}")

    fold_root = resolve_project_path(args.fold_root)
    if not fold_root.exists():
        raise FileNotFoundError(f"Fold root not found: {fold_root}")
    fold_dirs = sorted(
        [p for p in fold_root.glob("fold_*") if p.is_dir()],
        key=lambda x: int(x.name.split("_")[1]),
    )
    fold_ids = [int(p.name.split("_")[1]) for p in fold_dirs]
    if not fold_ids:
        raise ValueError(f"No fold_* directories found in {fold_root}")
    if args.max_folds is not None:
        fold_ids = fold_ids[: max(1, args.max_folds)]

    image_dir = resolve_image_dir(fold_root, resolve_project_path(args.image_root))
    classes = sorted(set(LABEL_GROUP_MAP.values()))
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    class_names = [f"Class {idx} (Group {idx_to_label[idx]})" for idx in range(len(classes))]

    output_dir = args.output_dir.resolve()
    if args.smoke_test:
        output_dir = output_dir.parent / f"{output_dir.name}_smoketest"
    checkpoint_dir = output_dir / "checkpoints" / "MixtureOfExpertsCNNFusion"
    metrics_dir = output_dir / "metrics"
    history_dir = output_dir / "histories" / "MixtureOfExpertsCNNFusion"
    figures_dir = output_dir / "figures" / "MixtureOfExpertsCNNFusion"
    predictions_dir = output_dir / "predictions" / "MixtureOfExpertsCNNFusion"
    reports_dir = output_dir / "reports" / "MixtureOfExpertsCNNFusion"
    for d in [output_dir, checkpoint_dir, metrics_dir, history_dir, figures_dir, predictions_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    dropout_rate = 0.3
    train_cfg = BACKBONE_TRAIN_CFG[args.backbone]

    print(f"Fold root: {fold_root}")
    print(f"Image dir: {image_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Folds to run: {fold_ids}")

    # Per-fold real/synthetic train composition (+ fold_data_summary.csv).
    fold_summary_rows = []
    fold_data_composition = []
    for fold_id in fold_ids:
        fold_train_df = load_fold_split_df(fold_root, fold_id, "train", label_to_idx)
        fold_val_df = load_fold_split_df(fold_root, fold_id, "val", label_to_idx)
        train_src = Counter(fold_train_df["source"]) if "source" in fold_train_df.columns else Counter()
        fold_summary_rows.append(
            {
                "fold": fold_id,
                "train_rows": len(fold_train_df),
                "val_rows": len(fold_val_df),
                "train_class_dist": dict(fold_train_df["label_group"].value_counts().sort_index()),
                "val_class_dist": dict(fold_val_df["label_group"].value_counts().sort_index()),
            }
        )
        fold_data_composition.append(
            {
                "fold": fold_id,
                "train_real_rows": int(train_src.get("real", 0)),
                "train_synthetic_rows": int(train_src.get("synthetic", 0)),
                "val_rows": len(fold_val_df),
            }
        )
    pd.DataFrame(fold_summary_rows).to_csv(output_dir / "fold_data_summary.csv", index=False)

    run_config = {
        "device": str(device),
        "backbone": args.backbone,
        "pretrained": True,
        "pretrained_weights": "IMAGENET1K_V1",
        "dataset_variant": "gan_augmented",
        "expert_architecture": "pretrained backbone, binary head",
        "fusion_architecture": "pretrained backbone feature extractor + expert-logit MLP",
        "normalization": "imagenet_fixed",
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "val_source_policy": "real_only",
        "fold_root": str(fold_root),
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "fold_ids": fold_ids,
        "models": ["MixtureOfExpertsCNNFusion"],
        "expert_epochs": args.expert_epochs,
        "fusion_epochs": args.fusion_epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "max_folds": args.max_folds,
        "smoke_test": args.smoke_test,
        "dropout_rate": dropout_rate,
        "expert_learning_rate": train_cfg["expert_lr"],
        "fusion_learning_rate": train_cfg["fusion_lr"],
        "grad_clip_max_norm": train_cfg["grad_clip_norm"],
        "vgg11_stabilization_source": (
            "Pretrained Naive Replication 5 Folds GAN Augmented/run_config.json "
            "(per_model_learning_rate.VGG11=1e-4, vgg11_grad_clip_max_norm=2.0)"
        ),
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau(mode=min, factor=0.5, patience=5)",
        "weight_decay": 1e-4,
        "fusion": "pretrained backbone embedding + concat expert logits into MLP",
        "fold_data_composition": fold_data_composition,
        "label_group_map": LABEL_GROUP_MAP,
        "torchvision_version": __import__("torchvision").__version__,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    all_fold_metrics = []
    best_overall_acc = -1.0
    best_overall_fold = None
    best_overall_paths = {}
    results_json = {
        "model": "pretrained-backbone binary experts + pretrained-backbone fusion (image embedding + expert logits into MLP)",
        "backbone": args.backbone,
        "dropout_rate": dropout_rate,
        "folds": {},
    }

    for fold_id in fold_ids:
        print(f"\n{'='*50}")
        print(f"Fold {fold_id}")
        print(f"{'='*50}")

        fold_train_df = load_fold_split_df(fold_root, fold_id, "train", label_to_idx)
        fold_val_df = load_fold_split_df(fold_root, fold_id, "val", label_to_idx)
        print(f"Fold train size: {len(fold_train_df)}, Fold val size: {len(fold_val_df)}")

        # Diagnostic only - raw pixel stats. Training/val transforms use fixed ImageNet stats.
        temp_dataset = ThyroidValidationDataset(
            fold_train_df, image_dir, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
        )
        mean, std = compute_mean_std(temp_dataset, batch_size=max(args.batch_size, 1))
        print(f"Diagnostic raw mean: {mean}, std: {std} (NOT used for transforms)")

        moe = MixtureOfExperts(args.backbone, num_classes=4, dropout_rate=dropout_rate)
        expert_val_accs = []
        fold_ckpt_dir = checkpoint_dir / f"fold_{fold_id}"
        fold_hist_dir = history_dir / f"fold_{fold_id}"
        fold_fig_dir = figures_dir / f"fold_{fold_id}"
        for d in [fold_ckpt_dir, fold_hist_dir, fold_fig_dir]:
            d.mkdir(parents=True, exist_ok=True)

        for expert_idx in range(4):
            print(f"\n{'='*50}")
            print(f"Training Expert {expert_idx} (Binary classifier for class {expert_idx})")
            print(f"{'='*50}")

            train_dataset = BinaryExpertDataset(
                fold_train_df, image_dir, IMAGENET_MEAN, IMAGENET_STD,
                target_class=expert_idx, is_train=True,
            )
            val_dataset = BinaryExpertDataset(
                fold_val_df, image_dir, IMAGENET_MEAN, IMAGENET_STD,
                target_class=expert_idx, is_train=False,
            )

            train_dataloader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            )

            expert_ckpt = fold_ckpt_dir / f"moe_fold_{fold_id}_expert_{expert_idx}_cnn_best.pth"
            best_val_acc = moe.train_expert(
                expert_idx, train_dataloader, val_dataloader,
                num_epochs=args.expert_epochs, save_path=expert_ckpt,
            )
            expert_val_accs.append(best_val_acc)
            print(f"Expert {expert_idx} best validation accuracy: {best_val_acc:.2f}%")

            expert_history_df = pd.DataFrame(moe.training_histories[expert_idx])
            expert_history_df.insert(0, "epoch", range(1, len(expert_history_df) + 1))
            expert_history_df.to_csv(
                fold_hist_dir / f"moe_fold_{fold_id}_expert_{expert_idx}_history.csv",
                index=False,
            )

        print(f"\n{'='*50}")
        print("Training Fusion Net (image embedding + expert logits)")
        print(f"{'='*50}")

        fusion_train_dataset = ThyroidValidationDataset(fold_train_df, image_dir, IMAGENET_MEAN, IMAGENET_STD)
        fusion_val_dataset = ThyroidValidationDataset(fold_val_df, image_dir, IMAGENET_MEAN, IMAGENET_STD)

        fusion_train_dataloader = DataLoader(
            fusion_train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        )
        fusion_val_dataloader = DataLoader(
            fusion_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        )

        fusion_ckpt = fold_ckpt_dir / f"moe_fold_{fold_id}_fusion_cnn_best.pth"
        fusion_val_acc = moe.train_fusion(
            fusion_train_dataloader, fusion_val_dataloader,
            num_epochs=args.fusion_epochs, save_path=fusion_ckpt,
        )
        fusion_history_df = pd.DataFrame(moe.fusion_history)
        fusion_history_df.insert(0, "epoch", range(1, len(fusion_history_df) + 1))
        fusion_history_df.to_csv(fold_hist_dir / f"moe_fold_{fold_id}_fusion_history.csv", index=False)

        plot_training_curves(moe, save_path=fold_fig_dir / "moe_training_curves.png")

        print(f"\n{'='*50}")
        print("Evaluating Mixture of Experts Ensemble")
        print(f"{'='*50}")

        val_dataset = ThyroidValidationDataset(fold_val_df, image_dir, IMAGENET_MEAN, IMAGENET_STD)
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        )

        val_acc, all_predictions, all_labels, all_logits = moe.evaluate(val_dataloader)
        print(f"\nEnsemble Fold {fold_id} Accuracy: {val_acc:.2f}%")

        print("\nClassification Report:")
        report_text = classification_report(
            all_labels, all_predictions, labels=list(range(len(classes))),
            target_names=class_names, zero_division=0,
        )
        print(report_text)

        report_df = pd.DataFrame(
            classification_report(
                all_labels, all_predictions, labels=list(range(len(classes))),
                target_names=class_names, output_dict=True, zero_division=0,
            )
        ).transpose()
        report_df.to_csv(reports_dir / f"MixtureOfExpertsCNNFusion_fold_{fold_id}_classification_report.csv")

        pred_df = pd.DataFrame(
            {
                "image_file": fold_val_df["image_file"].tolist(),
                "true_idx": all_labels,
                "pred_idx": all_predictions,
                "true_group_label": [idx_to_label[int(i)] for i in all_labels],
                "pred_group_label": [idx_to_label[int(i)] for i in all_predictions],
            }
        )
        for expert_idx in range(4):
            pred_df[f"expert_{expert_idx}_positive_logit"] = all_logits[:, expert_idx]
        pred_df.to_csv(
            predictions_dir / f"MixtureOfExpertsCNNFusion_fold_{fold_id}_predictions.csv",
            index=False,
        )

        plot_confusion_matrix(all_labels, all_predictions, save_path=fold_fig_dir / "moe_confusion_matrix.png")
        plot_expert_decisions(all_logits, all_labels, all_predictions, save_path=fold_fig_dir / "moe_expert_decisions.png")
        plot_logit_heatmap(all_logits, all_labels, save_path=fold_fig_dir / "moe_logit_heatmap.png")

        fold_metric_row = {
            "model": "MixtureOfExpertsCNNFusion",
            "fold": fold_id,
            "expert_0_val_acc": float(expert_val_accs[0] / 100.0),
            "expert_1_val_acc": float(expert_val_accs[1] / 100.0),
            "expert_2_val_acc": float(expert_val_accs[2] / 100.0),
            "expert_3_val_acc": float(expert_val_accs[3] / 100.0),
            "fusion_val_acc": float(fusion_val_acc / 100.0),
            "acc": float(val_acc / 100.0),
        }
        all_fold_metrics.append(fold_metric_row)
        results_json["folds"][str(fold_id)] = {
            "mean": mean,
            "std": std,
            "expert_val_accs": [float(acc) for acc in expert_val_accs],
            "fusion_val_acc": float(fusion_val_acc),
            "ensemble_val_acc": float(val_acc),
            "checkpoints": {
                "experts": [
                    str(fold_ckpt_dir / f"moe_fold_{fold_id}_expert_{expert_idx}_cnn_best.pth")
                    for expert_idx in range(4)
                ],
                "fusion": str(fusion_ckpt),
            },
        }

        if val_acc > best_overall_acc:
            best_overall_acc = float(val_acc)
            best_overall_fold = fold_id
            best_overall_dir = checkpoint_dir / "best_overall"
            best_overall_dir.mkdir(parents=True, exist_ok=True)
            copied_experts = []
            for expert_idx in range(4):
                src = fold_ckpt_dir / f"moe_fold_{fold_id}_expert_{expert_idx}_cnn_best.pth"
                dst = best_overall_dir / f"moe_best_overall_expert_{expert_idx}_cnn.pth"
                shutil.copy2(src, dst)
                copied_experts.append(str(dst))
            fusion_dst = best_overall_dir / "moe_best_overall_fusion_cnn.pth"
            shutil.copy2(fusion_ckpt, fusion_dst)
            best_overall_paths = {"experts": copied_experts, "fusion": str(fusion_dst)}

        del moe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_fold_metrics_df = pd.DataFrame(all_fold_metrics)
    all_fold_metrics_df.to_csv(metrics_dir / "all_models_all_folds_metrics.csv", index=False)
    (metrics_dir / "all_models_all_folds_metrics.json").write_text(
        all_fold_metrics_df.to_json(orient="records", indent=2), encoding="utf-8"
    )

    numeric_cols = all_fold_metrics_df.select_dtypes(include=[np.number]).columns.tolist()
    mean_row = {col: all_fold_metrics_df[col].mean() for col in numeric_cols}
    mean_row["model"] = "MixtureOfExpertsCNNFusion"
    mean_row["fold"] = "mean"
    std_row = {col: all_fold_metrics_df[col].std(ddof=1) for col in numeric_cols}
    std_row["model"] = "MixtureOfExpertsCNNFusion"
    std_row["fold"] = "std"
    fold_metrics_with_stats = pd.concat(
        [all_fold_metrics_df, pd.DataFrame([mean_row, std_row])],
        ignore_index=True,
    )
    fold_metrics_csv = metrics_dir / "MixtureOfExpertsCNNFusion_fold_metrics.csv"
    fold_metrics_with_stats.to_csv(fold_metrics_csv, index=False)
    (metrics_dir / "MixtureOfExpertsCNNFusion_fold_metrics_with_mean_std.json").write_text(
        fold_metrics_with_stats.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    best_models_df = pd.DataFrame(
        [
            {
                "model": "MixtureOfExpertsCNNFusion",
                "best_fold": best_overall_fold,
                "best_fold_test_acc": float(best_overall_acc / 100.0),
                "best_model_path": json.dumps(best_overall_paths),
                "metrics_csv": str(fold_metrics_csv),
            }
        ]
    )
    best_models_df.to_csv(metrics_dir / "best_models_summary.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {
                "model": "MixtureOfExpertsCNNFusion",
                "acc_mean": all_fold_metrics_df["acc"].mean(),
                "acc_std": all_fold_metrics_df["acc"].std(ddof=1),
                "fusion_val_acc_mean": all_fold_metrics_df["fusion_val_acc"].mean(),
            }
        ]
    )
    summary_df.to_csv(metrics_dir / "all_models_summary_mean_std.csv", index=False)

    results_json["best_fold"] = best_overall_fold
    results_json["best_fold_acc"] = best_overall_acc
    results_json["best_fold_checkpoints"] = best_overall_paths
    results_json["summary"] = summary_df.to_dict(orient="records")[0]
    (output_dir / "moe_results.json").write_text(json.dumps(results_json, indent=4), encoding="utf-8")

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Backbone: {args.backbone} (pretrained IMAGENET1K_V1)")
    print(f"Best Fold: {best_overall_fold}")
    print(f"Best Fold Accuracy: {best_overall_acc:.2f}%")
    print(f"Artifacts saved in: {output_dir}")
    print("\nMean fold results:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
