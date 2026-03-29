"""
glhl/evaluate.py
================
GLHLEvaluator — runs ensemble inference and computes all metrics.

Ensemble strategies
-------------------
max_confidence
    For each test image, run through all group models.
    The final prediction is the class with the highest raw softmax
    probability across all groups.

temperature_scaled
    Same as max_confidence, but each group model's logits are divided
    by a temperature T before softmax, improving calibration.

Outputs
-------
Per-run:
  - Classification report (precision, recall, F1 per class + macro/weighted avg)
  - Confusion matrix plot
  - Training curves (loss and macro F1 vs epoch) per group
  - JSON results file with all scalar metrics
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)


class GLHLEvaluator:

    def __init__(self, trainer, config, output_dir, logger):
        self.trainer    = trainer
        self.config     = config
        self.output_dir = output_dir
        self.logger     = logger
        self.device     = config.device

        os.makedirs(output_dir, exist_ok=True)

    # ── Ensemble inference ────────────────────────────────────────────────────

    def evaluate(self, test_loader):
        """
        Run ensemble inference on the test set and return a results dict.
        """
        self.logger.info("Running ensemble inference on test set...")

        # Gather group models and their class-index mappings
        group_models   = self.trainer.models
        group_dls      = {}    # not needed here; we use group_assignments from config

        # Build mapping: group_id -> list of global class indices
        group_class_map = {}
        for cls_idx, grp_id in self.config.group_assignments.items():
            group_class_map.setdefault(grp_id, [])
            group_class_map[grp_id].append(cls_idx)

        # Set all models to eval
        for m in group_models.values():
            m.eval()

        all_preds  = []
        all_labels = []
        num_classes = self.config.num_classes

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs   = imgs.to(self.device)
                batch_size = imgs.size(0)

                # Accumulate probability scores in global class space
                global_probs = torch.zeros(batch_size, num_classes,
                                           device=self.device)

                for grp_id, model in group_models.items():
                    cls_indices = sorted(group_class_map[grp_id])

                    logits = model(imgs)   # (B, num_local_classes)

                    if self.config.ensemble_strategy == "temperature_scaled":
                        logits = logits / self.config.temperature

                    probs = F.softmax(logits, dim=1)   # (B, num_local_classes)

                    # Map local probs back to global class positions
                    for local_i, global_i in enumerate(cls_indices):
                        global_probs[:, global_i] = probs[:, local_i]

                preds = global_probs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)

        # ── Compute metrics ────────────────────────────────────────────────
        class_names = self.config.class_names
        report_dict = classification_report(
            all_labels, all_preds,
            target_names = class_names,
            output_dict  = True,
            zero_division= 0
        )

        acc       = accuracy_score(all_labels, all_preds)
        macro_f1  = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
        micro_f1  = f1_score(all_labels, all_preds, average="micro",    zero_division=0)
        w_f1      = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        macro_p   = precision_score(all_labels, all_preds, average="macro",    zero_division=0)
        macro_r   = recall_score(   all_labels, all_preds, average="macro",    zero_division=0)

        results = {
            "accuracy"          : float(acc),
            "macro_f1"          : float(macro_f1),
            "micro_f1"          : float(micro_f1),
            "weighted_f1"       : float(w_f1),
            "macro_precision"   : float(macro_p),
            "macro_recall"      : float(macro_r),
            "per_class_report"  : report_dict,
            "predictions"       : all_preds.tolist(),
            "ground_truth"      : all_labels.tolist(),
        }

        self._log_results(results)
        return results

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_results(self, results):
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST SET RESULTS")
        self.logger.info("="*60)
        self.logger.info(f"  Accuracy        : {results['accuracy']:.4f}")
        self.logger.info(f"  Macro F1        : {results['macro_f1']:.4f}")
        self.logger.info(f"  Macro Precision : {results['macro_precision']:.4f}")
        self.logger.info(f"  Macro Recall    : {results['macro_recall']:.4f}")
        self.logger.info(f"  Weighted F1     : {results['weighted_f1']:.4f}")
        self.logger.info("\n  Per-class breakdown:")
        report = results["per_class_report"]
        for cls_name in self.config.class_names:
            if cls_name in report:
                r = report[cls_name]
                self.logger.info(
                    f"    {cls_name:<35} "
                    f"P={r['precision']:.3f}  "
                    f"R={r['recall']:.3f}  "
                    f"F1={r['f1-score']:.3f}  "
                    f"N={int(r['support'])}"
                )

    # ── Save results to JSON ──────────────────────────────────────────────────

    def save_results(self, results, label="glhl"):
        path = os.path.join(self.output_dir, f"{label}_results.json")
        save_data = {k: v for k, v in results.items()
                     if k not in ("predictions", "ground_truth")}
        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)
        self.logger.info(f"Results saved to: {path}")

    # ── Confusion matrix ──────────────────────────────────────────────────────

    def plot_confusion_matrix(self, results, label="glhl"):
        preds  = np.array(results["predictions"])
        labels = np.array(results["ground_truth"])
        cm     = confusion_matrix(labels, preds)
        names  = self.config.class_names

        # Normalize
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        for ax, data, title, fmt in [
            (axes[0], cm,      "Confusion Matrix (counts)",       "d"),
            (axes[1], cm_norm, "Confusion Matrix (normalized)",   ".2f"),
        ]:
            sns.heatmap(
                data, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=names, yticklabels=names,
                ax=ax, cbar=True
            )
            ax.set_xlabel("Predicted", fontsize=12)
            ax.set_ylabel("True",      fontsize=12)
            ax.set_title(title,        fontsize=13)
            ax.tick_params(axis="x", rotation=45)
            ax.tick_params(axis="y", rotation=0)

        plt.suptitle(f"GLHL Ensemble — Test Set Evaluation ({label})", fontsize=14)
        plt.tight_layout()

        path = os.path.join(self.output_dir, f"{label}_confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Confusion matrix saved to: {path}")

    # ── Training curves ───────────────────────────────────────────────────────

    def plot_training_curves(self, history, label="glhl"):
        """
        Plot loss and macro F1 curves for each group model.
        history: {group_id: {train_loss, val_loss, val_acc, val_f1}}
        """
        num_groups = len(history)
        if num_groups == 0:
            return

        fig, axes = plt.subplots(num_groups, 2, figsize=(14, 4 * num_groups))
        if num_groups == 1:
            axes = [axes]

        for row, (grp_id, hist) in enumerate(sorted(history.items())):
            if "train_loss" not in hist:
                continue

            epochs = range(1, len(hist["train_loss"]) + 1)
            ax_loss, ax_f1 = axes[row]

            ax_loss.plot(epochs, hist["train_loss"], label="Train Loss", color="steelblue")
            ax_loss.plot(epochs, hist["val_loss"],   label="Val Loss",   color="coral")
            ax_loss.set_title(f"Group {grp_id} — Loss")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            ax_loss.grid(alpha=0.3)

            ax_f1.plot(epochs, hist["val_f1"],  label="Val Macro F1", color="seagreen")
            ax_f1.plot(epochs, hist["val_acc"], label="Val Accuracy",  color="mediumpurple",
                       linestyle="--")
            ax_f1.set_title(f"Group {grp_id} — Validation Metrics")
            ax_f1.set_xlabel("Epoch")
            ax_f1.set_ylabel("Score")
            ax_f1.set_ylim(0, 1.05)
            ax_f1.legend()
            ax_f1.grid(alpha=0.3)

        plt.suptitle(f"Training Curves — {label}", fontsize=13)
        plt.tight_layout()

        path = os.path.join(self.output_dir, f"{label}_training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Training curves saved to: {path}")

    # ── CV summary plot ───────────────────────────────────────────────────────

    def plot_cv_summary(self, cv_metrics, label="glhl"):
        """
        Bar chart of per-group mean ± std macro F1 across folds.
        cv_metrics: {group_id: [f1_fold1, f1_fold2, ...]}
        """
        groups = sorted(cv_metrics.keys())
        means  = [np.mean(cv_metrics[g]) for g in groups]
        stds   = [np.std(cv_metrics[g])  for g in groups]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            [f"Group {g}" for g in groups], means,
            yerr=stds, capsize=6,
            color=["steelblue", "coral", "seagreen"][:len(groups)],
            alpha=0.85
        )
        ax.set_ylabel("Macro F1 (mean ± std)")
        ax.set_title(f"Cross-Validation Performance by Group — {label}")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    mean + std + 0.01,
                    f"{mean:.3f}±{std:.3f}",
                    ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{label}_cv_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        self.logger.info(f"CV summary plot saved to: {path}")
