# Stage 5 — Mixture-of-Experts framework

The proposed framework. Instead of asking one feature extractor and one classifier to model
all four diagnostic categories simultaneously, the task is decomposed into four class-specific
one-versus-rest experts whose responses are combined by a trainable fusion network.

Unlike sparse-routing MoE architectures, no expert is selected and the others discarded: all
four expert responses contribute to every decision. That is also what makes the framework
interpretable at the decision level — the four positive-class logits can be inspected directly
to see which experts responded to a given image, and whether the experts developed the
specialisation they were designed for.

## Architecture

**Experts.** Four independent networks, one per diagnostic group. Each is an
ImageNet-pretrained backbone with its final classification layer replaced by a two-way head.
For expert *c*, an image is a positive sample if its four-class label is *c*, and a negative
sample otherwise. Only the positive-class logit *s<sub>c</sub>(X)* is passed downstream.

**Fusion network.** The same pretrained backbone, used as a feature extractor with its own
classification head discarded, produces an image embedding. The four expert logits are
concatenated to that embedding and passed through an MLP:

```
[ backbone embedding (1024 / 1280 / 4096) ‖ 4 expert logits ]  →  256  →  64  →  4
                                                    ReLU + dropout(0.3) between layers
```

The final prediction is the argmax over the four fusion outputs. The experts are frozen while
the fusion network is trained, so fusion optimisation cannot alter expert specialisation.

**Backbones.** One complete run per backbone: `DenseNet121` (1024-d embedding),
`MobileNetV2` (1280-d), `VGG11` (4096-d, classifier truncated before its last linear layer).

## Input construction

`ThyroidPreprocessor` converts each image to grayscale, replicates it across three channels,
centre-crops to the shortest side and resizes to 128 × 128. Normalisation uses the fixed
ImageNet mean and standard deviation, as in stages 3 and 4. The per-fold raw pixel mean and
standard deviation are also computed and recorded, but only as a diagnostic — they are not
used by any transform.

## Class imbalance within each expert

Each expert faces a different imbalance from the four-class problem: its positive class is one
group against the pooled remaining three. This axis is *not* addressed by the GAN balancing of
stage 2, so it is handled separately. Within the training partition only, the minority class
is randomly oversampled with replacement to match the majority class, and the duplicated
samples receive a bounded random rotation of up to ± 45°. Horizontal flipping is not used
here. Validation partitions are neither oversampled nor augmented, so they retain their
original class distribution.

The fusion stage uses no oversampling and no rotation, since the four-class distribution of
the training partition is left unchanged there.

## Training configuration

Two stages per fold: experts first, then fusion with the experts frozen.

| Setting | Experts | Fusion |
|---|---|---|
| Epochs | 100 | 50 |
| Batch size | 16 | 16 |
| Optimiser | AdamW, weight decay 1 × 10⁻⁴ | AdamW, weight decay 1 × 10⁻⁴ |
| Learning rate | 1 × 10⁻⁴ | 1 × 10⁻³ (VGG11: 1 × 10⁻⁴) |
| Scheduler | ReduceLROnPlateau (min, factor 0.5, patience 5) | same |
| Gradient clipping | max-norm 1.0 (VGG11: 2.0) | max-norm 1.0 (VGG11: 2.0) |
| Loss | two-class cross-entropy | four-class cross-entropy |
| Dropout | 0.3 | 0.3 |
| Checkpoint | highest validation accuracy | highest validation accuracy |

Per-backbone values live in `BACKBONE_TRAIN_CFG` at the top of `moe_cnn_fusion.py`. The VGG11
entry mirrors the stabilisation used in stage 3 for the same reason: VGG11 has no batch
normalisation and is unstable at the higher rate on this dataset.

Seed 42, stratified five folds, real-only validation — identical to the other frameworks. As
in stage 3, the script aborts if a validation fold contains a non-real row.

## Files

| File | Purpose |
|---|---|
| `moe_cnn_fusion.py` | Training and evaluation for one backbone across all five folds |
| `add_mcc_kappa_to_metrics.py` | Backfills Cohen's κ and MCC into the saved metrics files |
| `build_backbone_comparison.py` | Aggregates the three completed backbone runs into one comparison table |

## Running it

`--backbone` and `--output-dir` are both required, so that runs for different backbones cannot
overwrite one another.

```bash
# Verify the data paths resolve (2 epochs, 1 fold, writes to a *_smoketest directory)
python moe_cnn_fusion.py --backbone DenseNet121 \
    --output-dir moe_cnn_fusion_densenet121_5fold_outputs --smoke-test

# Full runs, one per backbone
python moe_cnn_fusion.py --backbone DenseNet121 --output-dir moe_cnn_fusion_densenet121_5fold_outputs
python moe_cnn_fusion.py --backbone MobileNetV2 --output-dir moe_cnn_fusion_mobilenetv2_5fold_outputs
python moe_cnn_fusion.py --backbone VGG11       --output-dir moe_cnn_fusion_vgg11_5fold_outputs

# Post-processing, after all three runs complete
python add_mcc_kappa_to_metrics.py --check    # dry run, writes nothing
python add_mcc_kappa_to_metrics.py
python build_backbone_comparison.py
```

This is the most expensive stage of the study: each fold trains four experts for 100 epochs
plus a fusion network for 50, and the whole thing is repeated for three backbones.
`--max-folds` caps the number of folds for partial runs.

`add_mcc_kappa_to_metrics.py` exists because the training script saves per-image predictions
but does not compute κ or MCC inline. Both are pure functions of the confusion matrix, so they
are recomputed from the stored prediction CSVs — no retraining and no approximation — and
added as two columns immediately after `acc`. Every pre-existing column keeps its position and
value; run with `--check` first to confirm that before writing.

## Metrics reported

Per fold, alongside the standard metric set: each of the four experts' validation accuracies,
the fusion network's validation accuracy, and the accuracy of the plain argmax-over-expert-logits
ensemble without the fusion network. The last of these isolates how much the *learned*
aggregation contributes over a fixed maximum-confidence rule.

Expert accuracies are also what make expert specialisation visible: an expert whose validation
accuracy is high while the fusion accuracy is not indicates a group that is separable in
isolation but confusable against the others.

## Output

```
<output-dir>/
  run_config.json           backbone, learning rates, fold composition (real vs. synthetic rows)
  fold_data_summary.csv
  checkpoints/MixtureOfExpertsCNNFusion/fold_<k>/   4 expert + 1 fusion checkpoint per fold
  histories/MixtureOfExpertsCNNFusion/fold_<k>/     per-epoch curves for each expert and the fusion net
  predictions/MixtureOfExpertsCNNFusion/            per-image predictions and expert logits
  reports/MixtureOfExpertsCNNFusion/                per-fold classification reports
  metrics/                                          per-fold metrics with mean/std
  figures/MixtureOfExpertsCNNFusion/fold_<k>/       training curves, confusion matrix,
                                                    expert-decision plots, expert-logit heatmaps
backbone_comparison_summary.csv                     written by build_backbone_comparison.py
```

The expert-logit heatmaps and expert-decision plots are the interpretability output specific to
this framework: they show, per true class, how strongly each of the four experts responded.
