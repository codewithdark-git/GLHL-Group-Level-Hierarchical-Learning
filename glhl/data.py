"""
glhl/data.py
============
All dataset and dataloader logic for the GLHL pipeline.

Provides
--------
build_dataloaders()
    Returns a dict with keys:
      - "full_train"          : full training dataset (ImageFolder)
      - "test"                : test dataloader
      - "groups"              : dict {group_id -> {"train": DL, "val": DL}}
      - "cv_folds"            : list of K fold dicts {"train": DL, "val": DL}
      - "full_train_loader"   : full training dataloader (for baselines)

GroupSubset
    A torch Dataset that wraps an ImageFolder and exposes only the
    samples belonging to a specific group, with remapped local labels.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold, train_test_split


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transforms(image_size, norm_mean, norm_std):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])


def get_val_transforms(image_size, norm_mean, norm_std):
    return transforms.Compose([
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])


# ---------------------------------------------------------------------------
# GroupSubset  – exposes only samples from one group, with remapped labels
# ---------------------------------------------------------------------------

class GroupSubset(Dataset):
    """
    Wraps a torchvision ImageFolder and returns only the samples whose
    global class label belongs to `group_class_indices`.

    Labels are remapped from global [0 .. C-1] to local [0 .. |group|-1].
    The bijection and its inverse are stored as attributes for evaluation.

    Parameters
    ----------
    base_dataset        : torchvision ImageFolder (already loaded)
    group_class_indices : list[int]  global class indices in this group
    transform           : callable   applied to PIL images
    """

    def __init__(self, base_dataset, group_class_indices, transform=None):
        self.base_dataset         = base_dataset
        self.group_class_indices  = sorted(group_class_indices)
        self.transform            = transform

        # Build label bijections
        self.global_to_local = {g: l for l, g in
                                 enumerate(self.group_class_indices)}
        self.local_to_global = {l: g for g, l in
                                 self.global_to_local.items()}
        self.num_local_classes = len(self.group_class_indices)

        # Filter samples
        self.indices = [
            i for i, (_, lbl) in enumerate(base_dataset.samples)
            if lbl in self.global_to_local
        ]

        # Build parallel targets array for StratifiedKFold
        self.local_targets = np.array([
            self.global_to_local[base_dataset.samples[i][1]]
            for i in self.indices
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx  = self.indices[idx]
        img, glbl = self.base_dataset[real_idx]

        # Apply group-specific transform if provided
        if self.transform is not None:
            # base_dataset already applied its own transform;
            # to allow separate group transforms we re-open the image.
            path = self.base_dataset.samples[real_idx][0]
            from PIL import Image
            img = Image.open(path).convert("RGB")
            img = self.transform(img)

        local_lbl = self.global_to_local[glbl]
        return img, local_lbl


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_dataloaders(train_dir, test_dir, config, logger):
    """
    Build and return all dataloaders needed by the pipeline.

    Returns
    -------
    dict with keys:
      "full_train"        – ImageFolder of all training data
      "test"              – DataLoader for test set
      "groups"            – {group_id: {"train": DL, "val": DL, "subset": GroupSubset}}
      "cv_folds"          – list of fold dicts (only populated when cv_folds > 1)
      "full_train_loader" – single DataLoader over all training data (for baselines)
    """

    train_tf = get_train_transforms(config.image_size, config.norm_mean, config.norm_std)
    val_tf   = get_val_transforms(config.image_size, config.norm_mean, config.norm_std)

    # ── Full datasets ────────────────────────────────────────────────────────
    # Load without transform first so GroupSubset can re-open images.
    # We pass transform=None here and let GroupSubset handle transforms.
    full_train_raw = datasets.ImageFolder(root=train_dir, transform=None)
    full_test_ds   = datasets.ImageFolder(root=test_dir,  transform=val_tf)

    logger.info(f"Training samples : {len(full_train_raw)}")
    logger.info(f"Test samples     : {len(full_test_ds)}")
    logger.info(f"Classes          : {full_train_raw.classes}")

    # Verify class alignment between train and test
    if set(full_train_raw.classes) != set(full_test_ds.classes):
        missing_in_test  = set(full_train_raw.classes) - set(full_test_ds.classes)
        missing_in_train = set(full_test_ds.classes) - set(full_train_raw.classes)
        if missing_in_test:
            logger.warning(f"Classes in train but not test: {missing_in_test}")
        if missing_in_train:
            logger.warning(f"Classes in test but not train: {missing_in_train}")

    test_loader = DataLoader(
        full_test_ds,
        batch_size  = config.batch_size,
        shuffle     = False,
        num_workers = config.num_workers,
        pin_memory  = config.device.type == "cuda"
    )

    # ── Full training loader (used by baselines) ─────────────────────────────
    full_train_with_tf = datasets.ImageFolder(root=train_dir, transform=train_tf)
    full_train_loader  = DataLoader(
        full_train_with_tf,
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = config.num_workers,
        pin_memory  = config.device.type == "cuda"
    )

    # ── Group-level dataloaders ───────────────────────────────────────────────
    # group_assignments: {global_class_idx -> group_id}
    num_groups     = config.num_groups
    group_class_map = {g: [] for g in range(num_groups)}
    for cls_idx, grp in config.group_assignments.items():
        group_class_map[grp].append(cls_idx)

    group_dataloaders = {}

    for grp_id, cls_indices in group_class_map.items():
        logger.info(f"Building dataloaders for Group {grp_id} "
                    f"(classes: {[config.class_names[i] for i in cls_indices]})")

        group_subset = GroupSubset(
            base_dataset       = full_train_raw,
            group_class_indices= cls_indices,
            transform          = None   # transform applied inside __getitem__
        )

        if config.cv_folds <= 1:
            # Simple train/val split
            n      = len(group_subset)
            all_i  = np.arange(n)
            lbls   = group_subset.local_targets

            train_i, val_i = train_test_split(
                all_i,
                test_size    = config.val_split,
                stratify     = lbls,
                random_state = config.seed
            )

            train_ds = _SubsetWithTransform(group_subset, train_i, train_tf)
            val_ds   = _SubsetWithTransform(group_subset, val_i,   val_tf)

            group_dataloaders[grp_id] = {
                "train"  : _make_loader(train_ds, config, shuffle=True),
                "val"    : _make_loader(val_ds,   config, shuffle=False),
                "subset" : group_subset,
                "cls_indices": cls_indices,
            }
        else:
            # Store subset; folds built separately below
            group_dataloaders[grp_id] = {
                "subset"    : group_subset,
                "cls_indices": cls_indices,
            }

    # ── Cross-validation folds ───────────────────────────────────────────────
    cv_folds_data = []

    if config.cv_folds > 1:
        skf = StratifiedKFold(
            n_splits = config.cv_folds,
            shuffle  = True,
            random_state = config.seed
        )

        # Collect all training samples and their global labels for stratification
        all_targets = np.array([
            full_train_raw.samples[i][1] for i in range(len(full_train_raw))
        ])
        all_indices = np.arange(len(full_train_raw))

        for fold_idx, (fold_train_i, fold_val_i) in enumerate(
            skf.split(all_indices, all_targets)
        ):
            fold_data = {"fold": fold_idx, "groups": {}}

            for grp_id, cls_indices in group_class_map.items():
                grp_subset = group_dataloaders[grp_id]["subset"]

                # Map fold indices back to group subset local indices
                grp_global_set = set(grp_subset.indices)

                fold_train_grp = _intersect_indices(
                    grp_subset, fold_train_i, full_train_raw
                )
                fold_val_grp   = _intersect_indices(
                    grp_subset, fold_val_i, full_train_raw
                )

                if len(fold_train_grp) == 0 or len(fold_val_grp) == 0:
                    logger.warning(
                        f"Fold {fold_idx}, Group {grp_id}: "
                        f"empty train or val split — skipping."
                    )
                    continue

                train_ds = _SubsetWithTransform(grp_subset, fold_train_grp, train_tf)
                val_ds   = _SubsetWithTransform(grp_subset, fold_val_grp,   val_tf)

                fold_data["groups"][grp_id] = {
                    "train"      : _make_loader(train_ds, config, shuffle=True),
                    "val"        : _make_loader(val_ds,   config, shuffle=False),
                    "cls_indices": cls_indices,
                }

            cv_folds_data.append(fold_data)

        logger.info(f"Built {config.cv_folds}-fold stratified CV splits.")

    return {
        "full_train"        : full_train_raw,
        "test"              : test_loader,
        "groups"            : group_dataloaders,
        "cv_folds"          : cv_folds_data,
        "full_train_loader" : full_train_loader,
        "class_names"       : full_train_raw.classes,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _SubsetWithTransform(Dataset):
    """Wraps a GroupSubset and applies a given transform to a subset of its indices."""

    def __init__(self, group_subset, local_indices, transform):
        self.group_subset   = group_subset
        self.local_indices  = local_indices
        self.transform      = transform

    def __len__(self):
        return len(self.local_indices)

    def __getitem__(self, idx):
        real_local_idx = self.local_indices[idx]
        real_global_idx = self.group_subset.indices[real_local_idx]
        path = self.group_subset.base_dataset.samples[real_global_idx][0]
        lbl  = self.group_subset.local_targets[real_local_idx]

        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(lbl)


def _make_loader(dataset, config, shuffle):
    return DataLoader(
        dataset,
        batch_size  = config.batch_size,
        shuffle     = shuffle,
        num_workers = config.num_workers,
        pin_memory  = config.device.type == "cuda",
        drop_last   = shuffle   # drop last incomplete batch only during training
    )


def _intersect_indices(group_subset, fold_global_indices, base_dataset):
    """
    Given fold_global_indices (indices into base_dataset), return the
    corresponding local indices within group_subset.
    """
    fold_global_set   = set(fold_global_indices.tolist())
    local_indices = [
        local_i
        for local_i, global_i in enumerate(group_subset.indices)
        if global_i in fold_global_set
    ]
    return np.array(local_indices)
