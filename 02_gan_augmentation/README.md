# Stage 2 — GAN-based class balancing and fold construction

After preprocessing, the four diagnostic groups remain imbalanced, with between 308 and 822
training images per group. Duplication-based oversampling raises class frequency but adds no
new image content. This stage instead trains a class-conditional GAN on the training split
and synthesises additional images for the under-represented groups, bringing every group up
to the size of the largest one (822 images).

The generator sees the **training split only**. The held-out test split is never read here,
and synthetic images are added to training folds only — validation folds stay entirely real.

## Architecture and training

A single generator/discriminator pair is trained jointly across all four groups, conditioned
on the group label, rather than four independent per-group GANs. With as few as 308 real
images in the smallest group, an independent generator for that group alone would have too
little data; pooling lets the shared convolutional layers learn general scintigraphy texture
and uptake statistics from the full training set while the class embedding steers which
diagnostic pattern is produced.

| Component | Design |
|---|---|
| Generator | 128-d latent vector concatenated with a 50-d label embedding, five transposed-convolution blocks (BatchNorm + ReLU) upsampling to 128 × 128, final `tanh` |
| Discriminator | image concatenated with the label embedding projected across the spatial grid, five spectrally normalised convolutions (LeakyReLU, slope 0.2), scalar realism score |
| Loss | hinge adversarial loss, plus mode-seeking regularisation on the generator (λ = 1.0) |
| Stabilisation | spectral normalisation; DiffAugment (colour, translation, cutout) applied to real *and* generated images before the discriminator |
| Optimiser | Adam (β₁ = 0.5, β₂ = 0.999), lr 2 × 10⁻⁴ for both networks |
| Schedule | 800 epochs, batch size 32, seed 42 |

Two choices deserve emphasis, because they address failure modes that are easy to miss at
this data scale:

- **DiffAugment.** Without it, the discriminator memorises the small real set within a few
  epochs and stops providing a useful gradient to the generator. The augmentation is
  differentiable, so gradients still reach the generator through it, and the underlying image
  distribution being learned is unchanged.
- **Mode-seeking loss.** An earlier run of this pipeline converged to near-identical images
  per group regardless of the latent code — mode collapse. Visual inspection of single images
  does not reveal this, since each output can still look like a plausible scintigraphy
  pattern; only comparing several outputs from the same group does. The mode-seeking term
  penalises the generator whenever two distant latent codes map to similar images for the same
  label, and diversity is additionally monitored quantitatively (see below).

## Diversity monitoring and checkpoint selection

Every 25 epochs, `diversity.py` measures the mean pairwise L1 pixel distance among freshly
generated images of each group and normalises it by the same quantity measured on the real
images of that group. A ratio near or above 1.0 indicates variability comparable to real
data; the collapsed run never exceeded ≈0.05.

The checkpoint used for generation is chosen on this criterion, not simply taken as the last
epoch. `generate_augmented_dataset.py` re-runs the check before generating anything and
**refuses to proceed** if any group falls below 15 % of its real-data diversity
(`--collapse-warning-ratio`; `--force` overrides, not recommended).

## Files

| File | Purpose |
|---|---|
| `gan_models.py` | Conditional DCGAN generator and discriminator |
| `diffaug.py` | Differentiable augmentation (colour, translation, cutout) |
| `diversity.py` | Pairwise-L1 diversity measurement, generated vs. real |
| `train_gan.py` | Training loop, periodic sampling, versioned checkpoints, diversity logging |
| `generate_augmented_dataset.py` | Builds the augmented dataset directory from a checkpoint |
| `diagnose_diversity.py` | Checks any single checkpoint for collapse, standalone |
| `../make_gan_augmented_5fold.py` | Builds the stratified 5-fold split (kept at the repository root; its paths are resolved relative to it) |

## Running it

```bash
cd 02_gan_augmentation

# 1. Train the conditional GAN on the training split only
python train_gan.py \
    --data-root ../cf_dataset_preprocessed_split/train \
    --output-dir ./gan_checkpoints \
    --epochs 800

# 2. Inspect a checkpoint before using it
python diagnose_diversity.py --checkpoint ./gan_checkpoints/checkpoints/checkpoint_epoch_0800.pt

# 3. Build the augmented dataset
python generate_augmented_dataset.py \
    --checkpoint ./gan_checkpoints/checkpoints/checkpoint_epoch_0800.pt \
    --data-root ../cf_dataset_preprocessed_split \
    --output-dir ../cf_dataset_preprocessed_gan_augmented

# 4. Build the 5-fold split used by every experiment
cd ..
python make_gan_augmented_5fold.py
```

Step 1 writes a sample grid and a versioned checkpoint every `--sample-every` epochs (default
25), together with the per-group diversity report, into `training_history.json`. Inspect both
the numbers and the grids before moving on: the eight images in a group's row should visibly
differ from one another, not merely look plausible individually.

Step 3 refuses to run if `--output-dir` already exists, so a previous augmentation run can
never be silently overwritten.

## Output

`generate_augmented_dataset.py` writes a new, self-contained directory that mirrors the layout
of `cf_dataset_preprocessed_split`, so downstream scripts need no changes:

```
cf_dataset_preprocessed_gan_augmented/
  train/images/          real images (copied unmodified) + GAN_g{group}_*.png
  train/train_labels.csv real + synthetic rows, with added 'group' and 'source' columns
  test/                  unchanged copy of the real test split
  augmentation_summary.json
```

Each synthetic image is assigned one of the original raw diagnostic labels subsumed by its
group, sampled in the same proportion observed among that group's real images, so the label
composition of a group mirrors the real data rather than treating the group as internally
homogeneous.

`make_gan_augmented_5fold.py` then produces
`cf_dataset_preprocessed_gan_augmented_5fold/fold_{1..5}/{train,val}.json` plus
`metadata.json`. It stratifies on the four-class group label with seed 42, places all
synthetic rows in every training fold, and asserts at build time that validation folds are
real-only, disjoint from training, and disjoint from the held-out test split.
