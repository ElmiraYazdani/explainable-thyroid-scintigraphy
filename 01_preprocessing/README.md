# Stage 1 — Image enhancement

Batch preprocessing of the quality-screened scintigraphy images, applied in a fixed order:

```
SwinIR denoising  →  NAFNet deblurring  →  CLAHE contrast enhancement
```

Thyroid scintigraphy is acquired at low spatial resolution and is affected by counting noise,
motion blur and variable contrast. Each step targets one of those degradations while leaving
the radiotracer uptake pattern intact. The order is part of the method and must not be
rearranged: deblurring a noisy image amplifies the noise, and CLAHE applied before
restoration amplifies both.

| Step | Model | Setting |
|---|---|---|
| Denoising | SwinIR, grayscale denoising (`004_grayDN_DFWB_s128w8_SwinIR-M`) | noise level 25 |
| Deblurring | NAFNet (`NAFNet-REDS-width64`) | no task-specific tuning |
| Contrast | CLAHE (OpenCV) | clip limit 2.0, tile grid 8 × 8 |

The noise level and the CLAHE parameters were selected empirically, by visual assessment of
the resulting images across centres; they are not the result of a formal parameter search.

## File

`preprocessor_exp.py` — exported from the Google Colab notebook in which the preprocessing
was run. It performs the whole stage end to end: cloning the SwinIR and NAFNet repositories,
downloading their pretrained weights, processing every image in an input directory, and
writing a four-column visualisation (original / denoised / deblurred / final) for a random
sample.

Because it is a notebook export, three parts are Colab-specific and have no effect outside
Colab:

- `drive.mount('/content/drive')` near the top,
- `files.download(...)` after the zip step (guarded by `try/except`),
- the `/content/...` default paths.

One positional change was made relative to the original export: `from __future__ import
annotations` was moved to the top of the file, which Python requires and which cell-by-cell
notebook execution does not. No other line was altered.

## Running it

In Colab, open the original notebook and run the cells in order. To run it locally, remove
the `drive.mount` call, then set the three paths near the middle of the file before executing:

```python
INPUT_DIR  = Path('.../cf_dataset/images')     # quality-screened source images
OUTPUT_DIR = Path('.../preprocessed_output')   # one enhanced PNG per input image
ZIP_PATH   = Path('.../preprocessed_output.zip')
```

Processing parameters are declared alongside them (`NOISE_LEVEL = 25`,
`CLAHE_CLIP_LIMIT = 2.0`, `CLAHE_GRID_SIZE = 8`).

## Dependencies

This stage installs its own dependencies at run time, so it is not covered by the
repository-level `requirements.txt`. It requires `opencv-python`, `torch`, `timm`, `gdown`
and `matplotlib`, and clones:

- SwinIR — <https://github.com/JingyunLiang/SwinIR>
- NAFNet — <https://github.com/megvii-research/NAFNet>

A CUDA GPU is strongly recommended; SwinIR on CPU is slow for a dataset of this size.

## Output

One enhanced PNG per input image (`<stem>_preprocessed.png`), plus a zip archive of the
output directory. These enhanced images, resized to 128 × 128 and split into train/test, form
the `cf_dataset_preprocessed_split/` directory consumed by stage 2.
