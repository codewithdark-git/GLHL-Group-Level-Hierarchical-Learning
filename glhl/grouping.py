"""
glhl/grouping.py
================
Assigns each class to one of K groups based on training sample counts.

Three strategies are supported:
  - quantile    : splits at equal quantiles of the class-count distribution.
  - fixed       : uses two explicit threshold values supplied by the user.
  - logarithmic : bins classes using logarithmic intervals.

Returns
-------
group_assignments : dict {class_index (int) -> group_id (int)}
class_counts      : dict {class_name (str) -> count (int)}
class_names       : list of class names sorted alphabetically
"""

import os
import math
import numpy as np


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def assign_groups(train_dir, num_groups, strategy, fixed_thresholds, logger):
    """
    Scan train_dir, count samples per class, and assign classes to groups.

    Parameters
    ----------
    train_dir         : str   – root of training data (one subdir per class).
    num_groups        : int   – number of groups (2, 3, or 4).
    strategy          : str   – 'quantile', 'fixed', or 'logarithmic'.
    fixed_thresholds  : list  – [low, high] thresholds (used when strategy='fixed').
    logger            : logging.Logger

    Returns
    -------
    group_assignments : dict {int -> int}
    class_counts      : dict {str -> int}
    class_names       : list[str]
    """

    # ── Count samples per class ──────────────────────────────────────────────
    class_names  = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    class_counts = {}
    for cls in class_names:
        cls_dir = os.path.join(train_dir, cls)
        n = len([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
        ])
        class_counts[cls] = n

    counts_array = np.array([class_counts[c] for c in class_names])

    logger.info("Class distribution in training set:")
    logger.info(f"  {'Class':<35} {'Count':>8}  {'Group':>6}")
    logger.info(f"  {'-'*55}")

    # ── Compute thresholds ───────────────────────────────────────────────────
    if strategy == "quantile":
        thresholds = _quantile_thresholds(counts_array, num_groups)
    elif strategy == "fixed":
        thresholds = _fixed_thresholds(fixed_thresholds, num_groups)
    elif strategy == "logarithmic":
        thresholds = _log_thresholds(counts_array, num_groups)
    else:
        raise ValueError(f"Unknown grouping strategy: {strategy}")

    logger.info(f"  Grouping strategy : {strategy}")
    logger.info(f"  Thresholds used   : {thresholds}")

    # ── Assign classes to groups ─────────────────────────────────────────────
    group_assignments = {}
    group_sizes       = {g: [] for g in range(num_groups)}

    for idx, cls in enumerate(class_names):
        n      = class_counts[cls]
        grp_id = _assign_to_group(n, thresholds)
        group_assignments[idx] = grp_id
        group_sizes[grp_id].append(cls)

    # ── Log summary ──────────────────────────────────────────────────────────
    for idx, cls in enumerate(class_names):
        grp = group_assignments[idx]
        logger.info(f"  {cls:<35} {class_counts[cls]:>8}  Group {grp}")

    logger.info("")
    logger.info("Group summary:")
    for g in range(num_groups):
        members = group_sizes[g]
        cnts    = [class_counts[c] for c in members]
        ratio   = _imbalance_ratio(cnts)
        logger.info(
            f"  Group {g} | {len(members)} classes | "
            f"total={sum(cnts)} | imbalance ratio={ratio:.2f}:1"
        )
        for m in members:
            logger.info(f"    - {m}  ({class_counts[m]} samples)")

    # ── Validate: every group must have at least 1 class ────────────────────
    for g in range(num_groups):
        if len(group_sizes[g]) == 0:
            raise RuntimeError(
                f"Group {g} has 0 classes assigned. "
                f"Try a different grouping strategy or reduce --num_groups."
            )

    return group_assignments, class_counts, class_names


# ---------------------------------------------------------------------------
# Threshold computation helpers
# ---------------------------------------------------------------------------

def _quantile_thresholds(counts, num_groups):
    """Split at equal quantile boundaries of the count distribution."""
    quantiles = np.linspace(0, 100, num_groups + 1)[1:-1]
    return [int(np.percentile(counts, q)) for q in quantiles]


def _fixed_thresholds(user_thresholds, num_groups):
    """Use user-supplied thresholds. Extend or trim to match num_groups-1."""
    needed = num_groups - 1
    if len(user_thresholds) < needed:
        # Extend by evenly spacing beyond the last value
        step = user_thresholds[-1]
        while len(user_thresholds) < needed:
            user_thresholds = list(user_thresholds) + [user_thresholds[-1] + step]
    return sorted(user_thresholds[:needed])


def _log_thresholds(counts, num_groups):
    """Bin using logarithmic intervals between min and max counts."""
    c_min = max(counts.min(), 1)
    c_max = counts.max()
    log_min = math.log10(c_min)
    log_max = math.log10(c_max)
    splits  = np.linspace(log_min, log_max, num_groups + 1)[1:-1]
    return [int(10 ** s) for s in splits]


def _assign_to_group(count, thresholds):
    """Return group index for a given count given sorted thresholds list."""
    for g, t in enumerate(thresholds):
        if count < t:
            return g
    return len(thresholds)   # last group


def _imbalance_ratio(counts):
    """Return max/min ratio of a list of counts. Returns 1.0 for single-class groups."""
    if len(counts) < 2:
        return 1.0
    return max(counts) / max(min(counts), 1)
