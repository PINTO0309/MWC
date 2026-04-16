# MWC
Mask wearing classifier.

## Setup

```bash
git clone https://github.com/PINTO0309/MWC.git && cd MWC
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

## Archive extraction

Extract images from the source archive into numbered folders under `data/`,
storing up to 2,000 images per folder:

```bash
python 00_extract_tar.py \
--archive /path/to/train_aug_120x120_part_masked_clean.tar.gz \
--output-dir data \
--images-per-dir 2000
```

## Dataset parquet

Generate a parquet dataset with embedded resized image bytes:

```bash
SIZE=48x48 # HxW
python 01_build_mask_parquet.py \
--root data \
--output data/dataset_${SIZE}.parquet \
--image-size ${SIZE}
```

Labels are derived from filenames:

- `*_mask_*` -> `masked` / `1`
- otherwise -> `no_masked` / `0`

## Data sample

|1|2|3|4|5|
|:-:|:-:|:-:|:-:|:-:|
|<img width="48" height="48" alt="_HELENFlip_HELEN_166033328_1_1_1_mask_1" src="https://github.com/user-attachments/assets/4d192a75-dda0-4d4d-a042-51703d7077cb" />|<img width="48" height="48" alt="_HELEN_HELEN_263567973_1_0_6" src="https://github.com/user-attachments/assets/a9fb2976-2c86-4745-8eb0-f5e1a9488537" />|<img width="48" height="48" alt="_HELENFlip_HELEN_2652699508_1_6_3_mask_1" src="https://github.com/user-attachments/assets/0b07847e-3bcc-442b-8bd5-47aedf258325" />|<img width="48" height="48" alt="_HELEN_HELEN_30427236_1_1_5" src="https://github.com/user-attachments/assets/9c4d422d-0542-4dbc-8be6-37ef28efe97a" />|<img width="48" height="48" alt="_HELENFlip_HELEN_3052865023_5_0_1" src="https://github.com/user-attachments/assets/73b702c1-447d-4d42-a2e1-bc66ebcd88d9" />|

## Training Pipeline

- Use the images located under `dataset/output/002_xxxx_front_yyyyyy` together with their annotations in `dataset/output/002_xxxx_front.csv`.
- Every augmented image that originates from the same `still_image` stays in the same split to prevent leakage.
- The training loop relies on `BCEWithLogitsLoss` plus class-balanced `pos_weight` to stabilise optimisation under class imbalance; inference produces sigmoid probabilities. Use `--train_resampling weighted` to switch on the previous `WeightedRandomSampler` behaviour, or `--train_resampling balanced` to physically duplicate minority classes before shuffling.
- Training history, validation metrics, optional test predictions, checkpoints, configuration JSON, and ONNX exports are produced automatically.
- Per-epoch checkpoints named like `mwc_epoch_0001.pt` are retained (latest 10), as well as the best checkpoints named `mwc_best_epoch0004_f1_0.9321.pt` (also latest 10).
- The backbone can be switched with `--arch_variant`. Supported combinations with `--head_variant` are:

  | `--arch_variant` | Default (`--head_variant auto`) | Explicitly selectable heads | Remarks |
  |------------------|-----------------------------|---------------------------|------|
  | `baseline`       | `avg`                       | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, you need to adjust the height and width of the feature map so that they are divisible by `--token_mixer_grid` (if left as is, an exception will occur during ONNX conversion or inference). |
  | `inverted_se`    | `avgmax_mlp`                | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, it is necessary to adjust `--token_mixer_grid` as above. |
  | `convnext`       | `transformer`               | `avg`, `avgmax_mlp`, `transformer`, `mlp_mixer` | For both heads, the grid must be divisible by the feature map (default `3x2` fits with 30x48 input). |
- The classification head is selected with `--head_variant` (`avg`, `avgmax_mlp`, `transformer`, `mlp_mixer`, or `auto` which derives a sensible default from the backbone).
- Pass `--rgb_to_yuv_to_y` to convert RGB crops to YUV, keep only the Y (luma) channel inside the network, and train a single-channel stem without modifying the dataloader.
- Alternatively, use `--rgb_to_lab` or `--rgb_to_luv` to convert inputs to CIE Lab/Luv (3-channel) before the stem; these options are mutually exclusive with each other and with `--rgb_to_yuv_to_y`.
- Mixed precision can be enabled with `--use_amp` when CUDA is available.
- Resume training with `--resume path/to/mwc_epoch_XXXX.pt`; all optimiser/scheduler/AMP states and history are restored.
- Loss/accuracy/F1 metrics are logged to TensorBoard under `output_dir`, and `tqdm` progress bars expose per-epoch progress for train/val/test loops.

Baseline depthwise-separable CNN:

```bash
SIZE=48x48
uv run python -m mwc train \
--data_root data/dataset.parquet \
--output_dir runs/mwc_${SIZE} \
--epochs 40 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant baseline \
--seed 42 \
--device auto \
--use_amp
```

Inverted residual + SE variant (recommended for higher capacity):

```bash
SIZE=48x48
VAR=s
uv run python -m mwc train \
--data_root data/dataset.parquet \
--output_dir runs/mwc_is_${VAR}_${SIZE} \
--epochs 40 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--seed 42 \
--device auto \
--use_amp
```

ConvNeXt-style backbone with transformer head over pooled tokens:

```bash
SIZE=48x48
uv run python -m mwc train \
--data_root data/dataset.parquet \
--output_dir runs/mwc_convnext_${SIZE} \
--epochs 40 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant convnext \
--head_variant transformer \
--token_mixer_grid 3x3 \
--seed 42 \
--device auto \
--use_amp
```

- Outputs include the latest 10 `mwc_epoch_*.pt`, the latest 10 `mwc_best_epochXXXX_f1_YYYY.pt` (highest validation F1, or training F1 when no validation split), `history.json`, `summary.json`, optional `test_predictions.csv`, and `train.log`.
- After every epoch a confusion matrix and ROC curve are saved under `runs/mwc/diagnostics/<split>/confusion_<split>_epochXXXX.png` and `roc_<split>_epochXXXX.png`.
- `--image_size` accepts either a single integer for square crops (e.g. `--image_size 48`) or `HEIGHTxWIDTH` to resize non-square frames (e.g. `--image_size 64x48`).
- Add `--resume <checkpoint>` to continue from an earlier epoch. Remember that `--epochs` indicates the desired total epoch count (e.g. resuming `--epochs 40` after training to epoch 30 will run 10 additional epochs).
- Launch TensorBoard with:
  ```bash
  tensorboard --logdir runs/mwc
  ```

### ONNX Export

```bash
uv run python -m mwc exportonnx \
--checkpoint runs/mwc_is_s_48x48/mwc_best_epoch0049_f1_0.9939.pt \
--output mwc_s_48x48.onnx \
--opset 17
```
