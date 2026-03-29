"""
glhl/config.py
==============
Central configuration object. Constructed once from parsed CLI args
and passed through the entire pipeline so every module shares one
source of truth.
"""

import os
import torch


class GLHLConfig:
    """
    Holds all hyperparameters, paths, and derived settings.
    Constructed from argparse Namespace in main.py.
    """

    def __init__(self, args):
        # ── Paths ─────────────────────────────────────────────────────────
        self.train_dir      = args.train_dir
        self.test_dir       = args.test_dir
        self.output_dir     = args.output_dir
        self.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # ── Training hyperparameters ──────────────────────────────────────
        self.epochs           = args.epochs
        self.freeze_epochs    = args.freeze_epochs
        self.batch_size       = args.batch_size
        self.lr               = args.lr
        self.weight_decay     = args.weight_decay
        self.dropout          = args.dropout
        self.label_smoothing  = args.label_smoothing
        self.patience         = args.patience
        self.seed             = args.seed
        self.num_workers      = args.num_workers

        # ── Grouping ──────────────────────────────────────────────────────
        self.grouping_strategy  = args.grouping_strategy
        self.group_thresholds   = args.group_thresholds   # [low, high] for fixed
        self.num_groups         = args.num_groups

        # ── Backbone & image ──────────────────────────────────────────────
        self.backbone    = args.backbone
        self.image_size  = args.image_size

        # ── Ensemble ──────────────────────────────────────────────────────
        self.ensemble_strategy = args.ensemble_strategy
        self.temperature       = args.temperature

        # ── Cross-validation ──────────────────────────────────────────────
        self.cv_folds   = args.cv_folds
        self.val_split  = args.val_split

        # ── Device ────────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Derived (filled in after dataset scan) ────────────────────────
        self.group_assignments = None   # dict: class_idx -> group_id
        self.class_names       = None   # list of class name strings
        self.class_counts      = None   # dict: class_name -> count
        self.num_classes       = None   # total number of classes

        # ImageNet normalisation stats (used by all models and baselines)
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std  = [0.229, 0.224, 0.225]

    def __repr__(self):
        lines = ["GLHLConfig("]
        for k, v in vars(self).items():
            lines.append(f"  {k} = {v}")
        lines.append(")")
        return "\n".join(lines)
