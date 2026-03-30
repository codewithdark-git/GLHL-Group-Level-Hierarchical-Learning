"""
glhl/trainer.py
===============
GLHLTrainer — orchestrates training of all group models.

Supports:
  - Single train/val split (cv_folds = 1)
  - Stratified K-fold cross-validation (cv_folds > 1)
  - Progressive fine-tuning (freeze backbone for first N epochs)
  - Early stopping with patience
  - Per-fold and per-epoch metric logging
  - Checkpoint saving for the best model per group per fold
"""

import os
import copy
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from glhl.models import GroupModel, build_loss


class EarlyStopping:
    """Stops training when validation macro-F1 does not improve for `patience` epochs."""

    def __init__(self, patience=7, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.stop       = False
        self.best_state = None

    def step(self, score, model):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# GLHLTrainer
# ---------------------------------------------------------------------------

class GLHLTrainer:

    def __init__(self, config, logger):
        self.config  = config
        self.logger  = logger
        self.device  = config.device
        self.models  = {}    # {group_id: GroupModel}  — final trained models
        self.history = {}    # {group_id: {train_loss, val_loss, val_acc, val_f1}}

    # ── Public: train all groups ─────────────────────────────────────────────

    def train(self, dataloaders):
        """Train all group models. Uses CV if cv_folds > 1."""

        if self.config.cv_folds > 1:
            self._train_with_cv(dataloaders)
        else:
            self._train_single_split(dataloaders)

    # ── Single split training ────────────────────────────────────────────────

    def _train_single_split(self, dataloaders):
        group_dls = dataloaders["groups"]

        for grp_id, grp_data in group_dls.items():
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Training Group {grp_id} model  "
                             f"({grp_data['subset'].num_local_classes} classes)")
            self.logger.info(f"{'='*50}")

            cls_indices  = grp_data["cls_indices"]
            group_subset = grp_data["subset"]

            # Class counts in local label order
            grp_class_counts = [
                self.config.class_counts[self.config.class_names[g_idx]]
                for g_idx in sorted(cls_indices)
            ]

            model = self._build_group_model(group_subset.num_local_classes)
            model, hist = self._run_training_loop(
                model            = model,
                train_loader     = grp_data["train"],
                val_loader       = grp_data["val"],
                grp_class_counts = grp_class_counts,
                group_id         = grp_id,
                fold_id          = 0,
            )

            self.models[grp_id]  = model
            self.history[grp_id] = hist

            # Save checkpoint
            self._save_checkpoint(model, grp_id, fold=0, tag="best")

    # ── K-fold CV training ───────────────────────────────────────────────────

    def _train_with_cv(self, dataloaders):
        cv_folds    = dataloaders["cv_folds"]
        num_groups  = self.config.num_groups

        # Store per-fold metrics for reporting
        cv_metrics = {g: [] for g in range(num_groups)}

        for fold_data in cv_folds:
            fold_idx = fold_data["fold"]
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"FOLD {fold_idx + 1} / {self.config.cv_folds}")
            self.logger.info(f"{'='*60}")

            fold_models = {}

            for grp_id, grp_dl in fold_data["groups"].items():
                self.logger.info(f"\n  -- Group {grp_id} --")

                cls_indices = grp_dl["cls_indices"]
                grp_class_counts = [
                    self.config.class_counts[self.config.class_names[g_idx]]
                    for g_idx in sorted(cls_indices)
                ]
                num_local_cls = len(cls_indices)

                model = self._build_group_model(num_local_cls)
                model, hist = self._run_training_loop(
                    model            = model,
                    train_loader     = grp_dl["train"],
                    val_loader       = grp_dl["val"],
                    grp_class_counts = grp_class_counts,
                    group_id         = grp_id,
                    fold_id          = fold_idx,
                )

                fold_models[grp_id] = model
                cv_metrics[grp_id].append(hist["best_val_f1"])
                self._save_checkpoint(model, grp_id, fold=fold_idx, tag="best")

            # Keep last fold's models as the final models (standard practice)
            self.models  = fold_models
            self.history = {g: {"fold_f1_scores": cv_metrics[g]} for g in cv_metrics}

        # Log CV summary
        self.logger.info("\n" + "="*60)
        self.logger.info("CROSS-VALIDATION SUMMARY")
        self.logger.info("="*60)
        import numpy as np
        for grp_id, scores in cv_metrics.items():
            arr = np.array(scores)
            self.logger.info(
                f"  Group {grp_id}: macro F1 = {arr.mean():.4f} ± {arr.std():.4f}"
            )

    # ── Core training loop ────────────────────────────────────────────────────

    def _run_training_loop(self, model, train_loader, val_loader,
                           grp_class_counts, group_id, fold_id):
        model      = model.to(self.device)
        criterion  = build_loss(grp_class_counts, self.config.label_smoothing, self.device)
        optimizer  = optim.AdamW(
            model.parameters(),
            lr           = self.config.lr,
            weight_decay = self.config.weight_decay
        )
        scheduler  = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        early_stop = EarlyStopping(patience=self.config.patience)

        history = {
            "train_loss": [], "val_loss": [],
            "val_acc": [],    "val_f1": [],
            "best_val_f1": 0.0
        }

        # Freeze backbone for warm-up
        model.freeze_backbone()
        self.logger.info(f"    Backbone frozen for first {self.config.freeze_epochs} epochs.")

        for epoch in range(1, self.config.epochs + 1):

            # Unfreeze after warm-up period
            if epoch == self.config.freeze_epochs + 1 and model._frozen:
                model.unfreeze_all()
                self.logger.info(f"    Epoch {epoch}: backbone unfrozen — full fine-tuning.")

            # ── Train ─────────────────────────────────────────────────────
            train_loss = self._train_one_epoch(model, train_loader, criterion, optimizer)

            # ── Validate ──────────────────────────────────────────────────
            val_loss, val_acc, val_f1 = self._validate(model, val_loader, criterion)

            scheduler.step(val_f1)
            early_stop.step(val_f1, model)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            if val_f1 > history["best_val_f1"]:
                history["best_val_f1"] = val_f1

            self.logger.info(
                f"    Epoch {epoch:02d}/{self.config.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"val_macro_f1={val_f1:.4f}"
            )

            if early_stop.stop:
                self.logger.info(f"    Early stopping at epoch {epoch}.")
                break

        # Restore best weights
        early_stop.restore_best(model)
        self.logger.info(
            f"    Best val macro F1 for Group {group_id}: {history['best_val_f1']:.4f}"
        )

        return model, history

    # ── Epoch-level helpers ──────────────────────────────────────────────────

    def _train_one_epoch(self, model, loader, criterion, optimizer):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _validate(self, model, loader, criterion):
        from sklearn.metrics import f1_score, accuracy_score
        import numpy as np

        model.eval()
        total_loss = 0.0
        all_preds  = []
        all_labels = []
        n_batches  = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits = model(imgs)
                loss   = criterion(logits, labels)
                preds  = logits.argmax(dim=1)
                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                n_batches  += 1

        val_loss = total_loss / max(n_batches, 1)
        val_acc  = accuracy_score(all_labels, all_preds)
        val_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return val_loss, val_acc, val_f1

    # ── Model builder ─────────────────────────────────────────────────────────

    def _build_group_model(self, num_classes):
        return GroupModel(
            backbone_name = self.config.backbone,
            num_classes   = num_classes,
            dropout       = self.config.dropout,
            freeze_layers = 50
        )

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _save_checkpoint(self, model, group_id, fold, tag="best"):
        path = os.path.join(
            self.config.checkpoint_dir,
            f"group{group_id}_fold{fold}_{tag}.pt"
        )
        torch.save(model.state_dict(), path)
        self.logger.info(f"    Checkpoint saved: {path}")

    def load_checkpoints(self, checkpoint_dir):
        """Load best checkpoints (fold 0) for all groups for inference."""
        import glob
        ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*_fold0_best.pt")))
        for ckpt_path in ckpt_files:
            grp_id = int(os.path.basename(ckpt_path).split("group")[1].split("_")[0])
            # We need to know num_classes — infer from saved state dict
            state  = torch.load(ckpt_path, map_location=self.device)
            n_cls  = state["classifier.weight"].shape[0]
            model  = GroupModel(
                backbone_name = self.config.backbone,
                num_classes   = n_cls,
                dropout       = self.config.dropout
            )
            model.load_state_dict(state)
            model = model.to(self.device)
            model.eval()
            self.models[grp_id] = model
            self.logger.info(f"Loaded checkpoint for group {grp_id}: {ckpt_path}")
