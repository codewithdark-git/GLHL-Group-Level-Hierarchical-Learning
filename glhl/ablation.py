"""
glhl/ablation.py
================
Runs a systematic ablation study to justify GLHL's design choices.

Ablation variants
-----------------
1. num_groups = 2     — Is 3 groups better than 2?
2. num_groups = 4     — Does more granularity help?
3. No label smoothing — Contribution of label smoothing (epsilon=0).
4. Plain Adam (no AdamW, no weight decay) — Contribution of AdamW.
5. No augmentation    — Contribution of data augmentation.
6. Threshold shift +20% — Sensitivity to grouping threshold boundaries.
7. Threshold shift -20% — Sensitivity to grouping threshold boundaries.
8. Ensemble: temperature_scaled vs max_confidence — Justified ensemble choice.

Each variant trains the full GLHL pipeline with only the specified change
and reports the same metric set for a clean controlled comparison.
"""

import os
import copy
import json
import numpy as np

from glhl.config   import GLHLConfig
from glhl.grouping import assign_groups
from glhl.data     import build_dataloaders
from glhl.trainer  import GLHLTrainer
from glhl.evaluate import GLHLEvaluator


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_ablation_study(train_dir, test_dir, config, output_dir, logger):
    """
    Run all ablation variants and save a consolidated results table.
    """

    ablation_dir = os.path.join(output_dir, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    results_all = {}

    variants = _define_variants(config)

    for variant_name, config_overrides in variants.items():
        logger.info(f"\n{'='*55}")
        logger.info(f"ABLATION: {variant_name}")
        logger.info(f"{'='*55}")

        try:
            var_config = _make_variant_config(config, config_overrides)
            var_output = os.path.join(ablation_dir, variant_name)
            os.makedirs(var_output, exist_ok=True)

            # Re-assign groups if num_groups or thresholds changed
            if "num_groups" in config_overrides or "group_thresholds" in config_overrides:
                group_assignments, class_counts, class_names = assign_groups(
                    train_dir         = train_dir,
                    num_groups        = var_config.num_groups,
                    strategy          = var_config.grouping_strategy,
                    fixed_thresholds  = var_config.group_thresholds,
                    logger            = logger
                )
                var_config.group_assignments = group_assignments
                var_config.class_counts      = class_counts
                var_config.class_names       = class_names
                var_config.num_classes       = len(class_names)

            dataloaders = build_dataloaders(
                train_dir = train_dir,
                test_dir  = test_dir,
                config    = var_config,
                logger    = logger
            )

            trainer   = GLHLTrainer(config=var_config, logger=logger)
            trainer.train(dataloaders)

            evaluator = GLHLEvaluator(
                trainer    = trainer,
                config     = var_config,
                output_dir = var_output,
                logger     = logger
            )
            results = evaluator.evaluate(dataloaders["test"])
            evaluator.save_results(results, label=variant_name)
            evaluator.plot_confusion_matrix(results, label=variant_name)

            results_all[variant_name] = {
                "accuracy"       : results["accuracy"],
                "macro_f1"       : results["macro_f1"],
                "macro_precision": results["macro_precision"],
                "macro_recall"   : results["macro_recall"],
                "overrides"      : config_overrides,
            }

        except Exception as e:
            logger.error(f"Ablation {variant_name} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # ── Print ablation table ──────────────────────────────────────────────────
    _print_ablation_table(results_all, logger)

    # ── Save ablation table ───────────────────────────────────────────────────
    path = os.path.join(ablation_dir, "ablation_summary.json")
    with open(path, "w") as f:
        json.dump(results_all, f, indent=2)
    logger.info(f"\nAblation summary saved: {path}")

    return results_all


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

def _define_variants(base_config):
    """
    Returns an ordered dict of {variant_name: {override_key: override_value}}.
    All unlisted keys inherit from base_config.
    """
    # Compute ±20% threshold shifts
    lo, hi = base_config.group_thresholds
    lo_down, hi_down = int(lo * 0.8), int(hi * 0.8)
    lo_up,   hi_up   = int(lo * 1.2), int(hi * 1.2)

    variants = {
        # ── Baseline (GLHL as reported) — for reference row ──────────────
        "glhl_full_3groups"        : {},

        # ── Number of groups ─────────────────────────────────────────────
        "ablation_2groups"         : {"num_groups": 2,
                                      "grouping_strategy": "quantile"},
        "ablation_4groups"         : {"num_groups": 4,
                                      "grouping_strategy": "quantile"},

        # ── Loss function components ──────────────────────────────────────
        "ablation_no_label_smooth" : {"label_smoothing": 0.0},
        "ablation_plain_adam"      : {"weight_decay": 0.0},   # AdamW → Adam

        # ── Augmentation ──────────────────────────────────────────────────
        # Handled via a flag that data.py checks
        "ablation_no_augmentation" : {"no_augmentation": True},

        # ── Threshold sensitivity ─────────────────────────────────────────
        "ablation_threshold_down20": {"group_thresholds": [lo_down, hi_down],
                                      "grouping_strategy": "fixed"},
        "ablation_threshold_up20"  : {"group_thresholds": [lo_up,   hi_up],
                                      "grouping_strategy": "fixed"},

        # ── Ensemble strategy ─────────────────────────────────────────────
        "ablation_temperature_ens" : {"ensemble_strategy": "temperature_scaled"},
    }

    return variants


def _make_variant_config(base_config, overrides):
    """Deep-copy base_config and apply overrides."""
    var_config = copy.deepcopy(base_config)
    for key, val in overrides.items():
        if hasattr(var_config, key):
            setattr(var_config, key, val)
        else:
            # Store as extra attribute (e.g., no_augmentation flag)
            setattr(var_config, key, val)
    return var_config


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_ablation_table(results_all, logger):
    logger.info("\n" + "="*75)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("="*75)
    header = f"  {'Variant':<40} {'Acc':>8} {'Macro F1':>10} {'Macro P':>9} {'Macro R':>9}"
    logger.info(header)
    logger.info(f"  {'-'*70}")

    # Print GLHL full first as reference
    if "glhl_full_3groups" in results_all:
        res = results_all["glhl_full_3groups"]
        logger.info(
            f"  {'glhl_full_3groups (reference)':<40} "
            f"{res['accuracy']:.4f}   "
            f"{res['macro_f1']:.4f}     "
            f"{res['macro_precision']:.4f}    "
            f"{res['macro_recall']:.4f}"
        )
        logger.info(f"  {'-'*70}")

    for name, res in results_all.items():
        if name == "glhl_full_3groups":
            continue
        logger.info(
            f"  {name:<40} "
            f"{res['accuracy']:.4f}   "
            f"{res['macro_f1']:.4f}     "
            f"{res['macro_precision']:.4f}    "
            f"{res['macro_recall']:.4f}"
        )
