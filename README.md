# GLHL: Group-Level Hierarchical Learning Pipeline

A complete, single-entry-point pipeline for training and evaluating the GLHL
framework for imbalanced skin lesion classification, including all baselines
and ablation experiments needed for the IEEE Access paper.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Organize your dataset in ImageFolder format (one subdirectory per class):

```
data/
  train/
    Melanoma/
      img001.jpg
      img002.jpg
    Nevus/
      ...
    Basal_Cell_Carcinoma/
      ...
  test/
    Melanoma/
      ...
    Nevus/
      ...
```

For ISIC 2019, download from https://challenge.isic-archive.com/data/#2019
and organize into the above structure using the provided CSV labels.

---

## Usage

### 1. Train GLHL only (quickest run)

```bash
python main.py \
  --train_dir data/train \
  --test_dir  data/test \
  --output_dir outputs/glhl_run1
```

### 2. Train GLHL with 5-fold cross-validation

```bash
python main.py \
  --train_dir data/train \
  --test_dir  data/test \
  --output_dir outputs/glhl_cv \
  --cv_folds 5
```

### 3. Full paper run (GLHL + all baselines + ablation)

```bash
python main.py \
  --train_dir   data/train \
  --test_dir    data/test \
  --output_dir  outputs/full_run \
  --cv_folds    5 \
  --run_baselines \
  --run_ablation
```

### 4. Load saved checkpoints and evaluate only (no retraining)

```bash
python main.py \
  --train_dir     data/train \
  --test_dir      data/test \
  --output_dir    outputs/eval_only \
  --skip_train \
  --checkpoint_dir outputs/glhl_cv/checkpoints
```

### 5. Use a different backbone (e.g. EfficientNet-B4)

```bash
python main.py \
  --train_dir  data/train \
  --test_dir   data/test \
  --output_dir outputs/effnet_run \
  --backbone   efficientnet_b4 \
  --image_size 380
```

---

## Key Arguments

| Argument              | Default         | Description                                          |
|-----------------------|-----------------|------------------------------------------------------|
| `--train_dir`         | required        | Path to training data root                           |
| `--test_dir`          | required        | Path to test data root                               |
| `--output_dir`        | `./outputs`     | Where results, plots, and checkpoints are saved      |
| `--backbone`          | `mobilenet_v2`  | Backbone: mobilenet_v2 / resnet50 / efficientnet_b0/b4 |
| `--epochs`            | 10              | Training epochs per group model                      |
| `--freeze_epochs`     | 3               | Warm-up epochs with backbone frozen                  |
| `--batch_size`        | 32              | Batch size                                           |
| `--lr`                | 1e-3            | Initial learning rate                                |
| `--cv_folds`          | 5               | Number of stratified CV folds (1 = single split)     |
| `--grouping_strategy` | `quantile`      | quantile / fixed / logarithmic                       |
| `--group_thresholds`  | `500 3000`      | [fixed only] Two class-count thresholds              |
| `--num_groups`        | 3               | Number of groups (2, 3, or 4)                        |
| `--ensemble_strategy` | `max_confidence`| max_confidence / temperature_scaled                  |
| `--run_baselines`     | False           | Also run all comparison baselines                    |
| `--run_ablation`      | False           | Also run the ablation study                          |
| `--seed`              | 42              | Random seed for reproducibility                      |
| `--label_smoothing`   | 0.1             | Label smoothing epsilon                              |
| `--patience`          | 7               | Early stopping patience (epochs)                     |

---

## Outputs

All outputs are saved to `--output_dir`:

```
outputs/
  glhl_run.log                 ← Full training and evaluation log
  run_config.json              ← Complete hyperparameter record
  checkpoints/
    group0_fold0_best.pt       ← Best model checkpoint per group per fold
    group1_fold0_best.pt
    group2_fold0_best.pt
  glhl_results.json            ← Test set metrics (accuracy, F1, per-class)
  glhl_confusion_matrix.png    ← Confusion matrix (raw + normalized)
  glhl_training_curves.png     ← Loss and F1 curves per group
  vanilla_mobilenetv2_results.json      ← Baseline results
  resnet50_weighted_ce_results.json
  mobilenetv2_smote_results.json
  mobilenetv2_mixup_results.json
  comparison_table.json        ← Side-by-side comparison of all methods
  ablation/
    ablation_summary.json      ← Consolidated ablation results table
    ablation_2groups/          ← Per-variant plots and results
    ablation_no_label_smooth/
    ...
```

---

## Reproducing Paper Results

```bash
python main.py \
  --train_dir   /path/to/ISIC2019/train \
  --test_dir    /path/to/ISIC2019/test \
  --output_dir  outputs/paper_run \
  --backbone    mobilenet_v2 \
  --epochs      10 \
  --freeze_epochs 3 \
  --batch_size  32 \
  --lr          1e-3 \
  --weight_decay 1e-2 \
  --dropout     0.5 \
  --label_smoothing 0.1 \
  --cv_folds    5 \
  --grouping_strategy quantile \
  --num_groups  3 \
  --ensemble_strategy max_confidence \
  --seed        42 \
  --run_baselines \
  --run_ablation
```

---

## Citation

If you use this pipeline, please cite:

```
A. Umar, "GLHL: Group-Level Hierarchical Learning for Imbalanced
Skin Lesion Classification," IEEE Access, 2026.
```
