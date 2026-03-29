"""
glhl/utils.py
=============
Shared utility functions used across the pipeline.
"""

import os
import json
import random
import logging
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(output_dir: str, name: str = "glhl") -> logging.Logger:
    """
    Create a logger that writes to both stdout and a log file.

    Parameters
    ----------
    output_dir : str   – directory where glhl_run.log will be saved.
    name       : str   – logger name.

    Returns
    -------
    logging.Logger
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "glhl_run.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt   = "[%(asctime)s] %(levelname)s — %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Log file: {log_path}")
    return logger


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def print_banner(logger: logging.Logger):
    banner = """
╔══════════════════════════════════════════════════════════════╗
║   GLHL: Group-Level Hierarchical Learning                    ║
║   Imbalanced Skin Lesion Classification Pipeline             ║
║   Author: Ahsan Umar                                         ║
╚══════════════════════════════════════════════════════════════╝
"""
    for line in banner.strip().split("\n"):
        logger.info(line)


# ---------------------------------------------------------------------------
# Config serialization
# ---------------------------------------------------------------------------

def save_run_config(config, output_dir: str):
    """Serialize the GLHLConfig to a JSON file for reproducibility records."""
    path = os.path.join(output_dir, "run_config.json")

    # Only serialize scalar-serializable attributes
    cfg_dict = {}
    for k, v in vars(config).items():
        if isinstance(v, (int, float, str, bool, list, type(None))):
            cfg_dict[k] = v
        elif isinstance(v, torch.device):
            cfg_dict[k] = str(v)
        elif isinstance(v, dict):
            try:
                json.dumps(v)   # test serializability
                cfg_dict[k] = v
            except TypeError:
                cfg_dict[k] = str(v)

    with open(path, "w") as f:
        json.dump(cfg_dict, f, indent=2)


# ---------------------------------------------------------------------------
# Metrics utilities
# ---------------------------------------------------------------------------

def compute_cv_summary(fold_metrics: list) -> dict:
    """
    Given a list of per-fold metric dicts, compute mean ± std for each metric.

    Parameters
    ----------
    fold_metrics : list of dicts, each with keys like 'accuracy', 'macro_f1', etc.

    Returns
    -------
    dict with keys '{metric}_mean' and '{metric}_std'
    """
    if not fold_metrics:
        return {}

    keys   = [k for k in fold_metrics[0].keys() if isinstance(fold_metrics[0][k], float)]
    summary = {}

    for k in keys:
        vals = np.array([m[k] for m in fold_metrics if k in m])
        summary[f"{k}_mean"] = float(vals.mean())
        summary[f"{k}_std"]  = float(vals.std())

    return summary


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Format a mean ± std pair as a string. E.g. '0.8814 ± 0.0123'."""
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} ± {fmt.format(std)}"
