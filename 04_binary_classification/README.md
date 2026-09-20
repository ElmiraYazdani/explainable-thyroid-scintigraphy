# Stage 4 — Classification-granularity analysis (tumoral vs. non-tumoral)

The four-class task is reformulated as a two-class clinical question — tumoral versus
non-tumoral — to test how much of the difficulty in the multiclass setting comes from
diagnostic granularity rather than from image quality or dataset size.

Everything else is held fixed relative to stage 3: the same architectures, the same
ImageNet-pretrained backbones, the same fold split, optimiser, schedule, augmentation, seed,
epoch count, image size and batch size. Only the label mapping and the metric handling change,
so the two experiments are directly comparable.

## Label mapping

Raw label (1–8) → four-class group (`LABEL_GROUP_MAP`) → binary (`BINARY_GROUP_MAP`):

| Group | Diagnosis | Binary label |
|---|---|---|
| 1 | Diffuse goiter | 0 — Non-Tumoral |
| 2 | Tumoral disease | **1 — Tumoral** |
| 3 | Thyroiditis | 0 — Non-Tumoral |
| 4 | Normal scintigraphy | 0 — Non-Tumoral |

## A property of the data composition, not a defect

The GAN augmentation in stage 2 was designed to balance the **four-class** problem. Collapsing
those balanced groups into two makes the training pool 23.0 % tumoral, while the real-only
validation folds are 38.5 % tumoral. This prior shift is an inherent consequence of reusing
the four-class split for a binary task, and it is reported rather than corrected, so that the
binary and multiclass results remain based on identical partitions. `make_binary_extra_plots.py`
quantifies its effect; the threshold-sweep figure shows the operating point that compensates
for it.

## Files

| File | Purpose |
|---|---|
| `classification_pretrained_replication_5fold_gan_augmented_binary.py` | Training, per-fold evaluation, per-fold figures, Grad-CAM |
| `make_pretrained_gan_binary_5fold_mean_plots.py` | Publication mean-across-folds figures |
| `pretrained_gan_binary_mean_plot_config.json` | Paths, class names, palette and typography for the above |
| `make_binary_extra_plots.py` | Binary-specific analysis figures (see below) |

## Running it

```bash
# Verify the data paths resolve (2 epochs)
python classification_pretrained_replication_5fold_gan_augmented_binary.py --smoke-test --verbose

# Full run: 5 architectures x 5 folds x 100 epochs
python classification_pretrained_replication_5fold_gan_augmented_binary.py --log-every 10 --verbose

# Mean-across-folds figures (CPU only)
python make_pretrained_gan_binary_5fold_mean_plots.py --config pretrained_gan_binary_mean_plot_config.json
```

The mean-plot script resolves `input_dir` from the configuration file relative to either the
current directory or the repository root, so it runs correctly from either location.

## Additional binary analyses

`make_binary_extra_plots.py` produces figures that only make sense for a two-class problem:
precision–recall curves, reliability/calibration curves, a decision-threshold sweep, operating
points per fold, error attribution by subgroup, fold-to-fold stability, and pooled confusion
matrices.

Two of its figures compare against other runs:

- `--parent-dir` — the four-class run from stage 3, for the paired native-binary versus
  collapsed-multiclass comparison. Point it at
  `../03_baseline_cnn/classification_pretrained_replication_5fold_gan_augmented_outputs`.
- `--baseline-dir` — a from-scratch (non-pretrained) binary reference run that is **not part
  of this release**. Without it, the cross-experiment comparison figure is skipped; all other
  figures are produced normally.

```bash
python make_binary_extra_plots.py \
    --parent-dir ../03_baseline_cnn/classification_pretrained_replication_5fold_gan_augmented_outputs
```

## Metrics

The same metric set as stage 3, with binary-safe ROC and one-hot handling: accuracy,
precision, recall, F1, Cohen's κ, MCC, NPV, PPV and ROC-AUC, reported per fold and aggregated
as mean ± standard deviation. Given the prior shift described above, accuracy alone is a poor
summary here; κ, MCC and the threshold sweep are the informative quantities.

## Output

```
classification_pretrained_replication_5fold_gan_augmented_binary_outputs/
  run_config.json, fold_data_summary.csv
  checkpoints/, histories/, predictions/, reports/, metrics/, figures/
figures/mean_5fold/    from make_pretrained_gan_binary_5fold_mean_plots.py
figures/extra/         from make_binary_extra_plots.py (+ extra_plots_manifest.json)
```
