# Stage 3 — Baseline CNN framework (four classes)

Reference configuration for the study. Five established CNN architectures are fine-tuned from
ImageNet-pretrained weights on the GAN-augmented five-fold split and evaluated on real-only
validation folds. Every alternative framework in the manuscript is judged against these
numbers, so the protocol here is deliberately conventional.

Architectures: **VGG11, DenseNet121, ResNet18, MobileNetV2, EfficientNetB0**.

## Input construction

Each preprocessed 128 × 128 grayscale image is replicated across three identical channels and
normalised with the fixed ImageNet mean and standard deviation. The replication matches the
input dimensionality the pretrained backbones expect without introducing any new information;
the ImageNet statistics are used unchanged because the backbones were pretrained under them.
Only the final classification layer of each architecture is replaced, with a four-way linear
layer.

Training-time augmentation is random horizontal flipping (p = 0.5) and random rotation within
± 10°, applied to the training partition only. Validation images are resized and normalised,
nothing more.

## Training configuration

| Setting | Value |
|---|---|
| Epochs per fold | 100 |
| Batch size | 32 |
| Optimiser | AdamW, weight decay 1 × 10⁻⁴ |
| Learning rate | 1 × 10⁻³ (VGG11: 1 × 10⁻⁴) |
| Scheduler | CosineAnnealingLR, T_max = 100, η_min = 1 × 10⁻⁶ |
| Loss | cross-entropy, no class weighting |
| Gradient clipping | none (VGG11: max-norm 2.0) |
| Data-loader workers | 4 |
| Seed | 42 |
| Checkpoint | highest validation accuracy within the fold |

VGG11 has no batch normalisation and reproducibly collapses to a single-class solution under
AdamW at 1 × 10⁻³ on this dataset. It is therefore trained at 1 × 10⁻⁴ with gradient-norm
clipping, applied to VGG11 alone (`--vgg11-learning-rate`, `--vgg11-grad-clip`); the four
batch-normalised architectures keep the common rate so they remain directly comparable.

## Files

| File | Purpose |
|---|---|
| `classification_pretrained_replication_5fold_gan_augmented.py` | Training, per-fold evaluation, per-fold figures, Grad-CAM |
| `make_final_5fold_mean_plots.py` | Publication mean-across-folds figures |
| `final_5fold_mean_plot_config.json` | Paths, class labels, palette and typography for the above |

## Running it

Run from this directory; the default `--fold-root` and `--image-root` are relative to it and
assume the dataset directories sit at the repository root.

```bash
# Verify the data paths resolve (2 epochs, writes to a *_smoketest directory)
python classification_pretrained_replication_5fold_gan_augmented.py --smoke-test --verbose

# Full run: 5 architectures x 5 folds x 100 epochs
python classification_pretrained_replication_5fold_gan_augmented.py --log-every 10 --verbose

# Mean-across-folds figures (CPU only)
python make_final_5fold_mean_plots.py --config final_5fold_mean_plot_config.json
```

Useful arguments: `--models DenseNet121 ResNet18` to run a subset, `--epochs`, `--batch-size`,
`--output-dir`, and `--fold-root` / `--image-root` if the dataset lives elsewhere.

## Metrics

Per fold: accuracy, precision, recall and F1 (macro and micro), Cohen's κ, MCC, macro NPV and
PPV, and one-vs-rest ROC-AUC (macro, micro, weighted). Fold values are aggregated into mean
and standard deviation, written both per model and in a cross-model summary sorted by mean
accuracy.

Before loading a validation fold the script asserts that every row has `source == "real"` and
aborts otherwise, so a synthetic image can never reach a validation set through an accidental
change to the fold JSONs.

## Interpretability

Grad-CAM maps are generated for the best-performing fold of each architecture, hooked onto
the last convolutional layer of the model. Two correctly classified and two misclassified
validation images are shown per architecture, each with the original image, the class
activation map, and the overlay, so that attention on the thyroid region can be compared
between successes and failures.

## Output

```
classification_pretrained_replication_5fold_gan_augmented_outputs/
  run_config.json          resolved arguments, device, torchvision version, fold sizes
  fold_data_summary.csv    per-fold row counts and class distribution
  checkpoints/<model>/     <model>_fold_<k>_best.pth, <model>_best_overall.pth
  histories/<model>/       per-epoch train/val loss, accuracy, learning rate
  predictions/<model>/     per-image true/predicted label and class probabilities
  reports/<model>/         per-fold classification reports
  metrics/                 per-fold metrics with mean/std, best_models_summary.csv,
                           all_models_summary_mean_std.csv, all_models_all_folds_metrics.csv
  figures/<model>/         learning curves, confusion matrices, ROC curves, Grad-CAM
  figures/                 all_models_metric_comparison.png
  figures/mean_5fold/      publication figures from make_final_5fold_mean_plots.py
```
