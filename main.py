"""
GLHL: Group-Level Hierarchical Learning for Imbalanced Skin Lesion Classification
==================================================================================
Main entry point. Run from terminal as:

    python main.py --train_dir /path/to/train --test_dir /path/to/test [OPTIONS]

Directory structure expected:
    train_dir/
        class_name_1/
            img1.jpg
            img2.jpg
        class_name_2/
            ...
    test_dir/
        class_name_1/
            ...

Author: Ahsan Umar
"""

import argparse
import os
import sys
import torch

from glhl.config      import GLHLConfig
from glhl.data        import build_dataloaders
from glhl.grouping    import assign_groups
from glhl.trainer     import GLHLTrainer
from glhl.evaluate    import GLHLEvaluator
from glhl.baselines   import run_all_baselines
from glhl.ablation    import run_ablation_study
from glhl.utils       import set_seed, setup_logger, print_banner, save_run_config


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="GLHL: Group-Level Hierarchical Learning Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Required paths ──────────────────────────────────────────────────────
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Path to training data root (class subdirectories expected).")
    parser.add_argument("--test_dir",  type=str, required=True,
                        help="Path to test data root (class subdirectories expected).")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory where all results, checkpoints, and plots are saved.")

    # ── Training control ────────────────────────────────────────────────────
    parser.add_argument("--epochs",      type=int,   default=10,
                        help="Total training epochs per group model.")
    parser.add_argument("--freeze_epochs", type=int, default=3,
                        help="Epochs to keep backbone frozen before full fine-tuning.")
    parser.add_argument("--batch_size",  type=int,   default=32,
                        help="Batch size for all dataloaders.")
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Initial learning rate for AdamW optimizer.")
    parser.add_argument("--weight_decay",type=float, default=1e-2,
                        help="AdamW weight decay (L2 regularization coefficient).")
    parser.add_argument("--dropout",     type=float, default=0.5,
                        help="Dropout probability applied before the classification head.")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing epsilon for cross-entropy loss.")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed for full reproducibility.")
    parser.add_argument("--num_workers", type=int,   default=4,
                        help="Number of DataLoader worker processes.")

    # ── Grouping strategy ───────────────────────────────────────────────────
    parser.add_argument("--grouping_strategy", type=str,
                        default="quantile",
                        choices=["quantile", "fixed", "logarithmic"],
                        help="Strategy for assigning classes to groups.\n"
                             "  quantile    – splits at 33rd and 66th percentile of class counts.\n"
                             "  fixed       – uses --group_thresholds values directly.\n"
                             "  logarithmic – logarithmic binning of class counts.")
    parser.add_argument("--group_thresholds", type=int, nargs=2,
                        default=[500, 3000],
                        metavar=("LOW", "HIGH"),
                        help="[fixed strategy only] Two thresholds that define three groups.\n"
                             "  Classes with count < LOW   → small group.\n"
                             "  Classes with LOW ≤ count < HIGH → medium group.\n"
                             "  Classes with count ≥ HIGH  → large group.")
    parser.add_argument("--num_groups", type=int, default=3,
                        choices=[2, 3, 4],
                        help="Number of groups to partition classes into.")

    # ── Backbone ────────────────────────────────────────────────────────────
    parser.add_argument("--backbone", type=str, default="mobilenet_v2",
                        choices=["mobilenet_v2", "resnet50", "efficientnet_b0",
                                 "efficientnet_b4"],
                        help="Backbone architecture for all group models.")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image resolution (square). All images resized to this.")

    # ── Ensemble inference ──────────────────────────────────────────────────
    parser.add_argument("--ensemble_strategy", type=str,
                        default="max_confidence",
                        choices=["max_confidence", "temperature_scaled"],
                        help="Strategy for combining group model predictions at inference.\n"
                             "  max_confidence     – argmax of highest softmax across groups.\n"
                             "  temperature_scaled – calibrate with temperature scaling first.")
    parser.add_argument("--temperature", type=float, default=1.5,
                        help="[temperature_scaled only] Temperature T for softmax calibration.")

    # ── Cross-validation ────────────────────────────────────────────────────
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of stratified k-folds for cross-validation.\n"
                             "Set to 1 to skip CV and use a single train/val split.")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="[cv_folds=1 only] Fraction of training data held out for validation.")

    # ── Run modes ───────────────────────────────────────────────────────────
    parser.add_argument("--run_baselines", action="store_true",
                        help="Also train and evaluate all comparison baselines.")
    parser.add_argument("--run_ablation",  action="store_true",
                        help="Also run the full ablation study.")
    parser.add_argument("--skip_train",    action="store_true",
                        help="Skip training; load saved checkpoints and evaluate only.")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="[skip_train only] Path to directory containing saved checkpoints.")

    # ── Early stopping ──────────────────────────────────────────────────────
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (epochs without validation improvement).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)
    print_banner(logger)

    logger.info(f"Train directory : {args.train_dir}")
    logger.info(f"Test directory  : {args.test_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device          : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"Seed            : {args.seed}")

    # ── Build config object ──────────────────────────────────────────────────
    config = GLHLConfig(args)
    save_run_config(config, args.output_dir)

    # ── Build full dataset and assign groups ─────────────────────────────────
    logger.info("Scanning dataset and assigning class groups...")
    group_assignments, class_counts, class_names = assign_groups(
        train_dir          = args.train_dir,
        num_groups         = args.num_groups,
        strategy           = args.grouping_strategy,
        fixed_thresholds   = args.group_thresholds,
        logger             = logger
    )
    config.group_assignments = group_assignments
    config.class_names       = class_names
    config.class_counts      = class_counts
    config.num_classes       = len(class_names)

    # ── Build dataloaders ────────────────────────────────────────────────────
    logger.info("Building dataloaders...")
    dataloaders = build_dataloaders(
        train_dir    = args.train_dir,
        test_dir     = args.test_dir,
        config       = config,
        logger       = logger
    )

    # ── Train GLHL ───────────────────────────────────────────────────────────
    trainer = GLHLTrainer(config=config, logger=logger)

    if not args.skip_train:
        logger.info("=" * 60)
        logger.info("PHASE 1: Training GLHL group models")
        logger.info("=" * 60)
        trainer.train(dataloaders)
    else:
        logger.info(f"Skipping training. Loading checkpoints from: {args.checkpoint_dir}")
        trainer.load_checkpoints(args.checkpoint_dir)

    # ── Evaluate GLHL ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2: Evaluating GLHL on test set")
    logger.info("=" * 60)
    evaluator = GLHLEvaluator(
        trainer      = trainer,
        config       = config,
        output_dir   = args.output_dir,
        logger       = logger
    )
    glhl_results = evaluator.evaluate(dataloaders["test"])
    evaluator.save_results(glhl_results, label="glhl")
    evaluator.plot_confusion_matrix(glhl_results, label="glhl")
    evaluator.plot_training_curves(trainer.history, label="glhl")

    # ── Baselines ────────────────────────────────────────────────────────────
    if args.run_baselines:
        logger.info("=" * 60)
        logger.info("PHASE 3: Training and evaluating baselines")
        logger.info("=" * 60)
        run_all_baselines(
            dataloaders = dataloaders,
            config      = config,
            output_dir  = args.output_dir,
            logger      = logger
        )

    # ── Ablation study ───────────────────────────────────────────────────────
    if args.run_ablation:
        logger.info("=" * 60)
        logger.info("PHASE 4: Running ablation study")
        logger.info("=" * 60)
        run_ablation_study(
            train_dir  = args.train_dir,
            test_dir   = args.test_dir,
            config     = config,
            output_dir = args.output_dir,
            logger     = logger
        )

    logger.info("=" * 60)
    logger.info("Pipeline complete. All results saved to: %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
