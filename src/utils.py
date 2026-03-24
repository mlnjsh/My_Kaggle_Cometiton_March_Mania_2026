"""
Utility Functions
=================
Shared helpers for the competition pipeline.
"""

import numpy as np
import pandas as pd
from configs.config import SEED


def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def is_mens_team(team_id):
    """Check if team ID belongs to men's division."""
    return 1000 <= team_id <= 1999


def is_womens_team(team_id):
    """Check if team ID belongs to women's division."""
    return 3000 <= team_id <= 3999


def historical_seed_win_probability(seed_a, seed_b):
    """
    Return historical P(seed_a beats seed_b) based on NCAA data.
    Uses logistic approximation: P = 1 / (1 + 10^(seed_diff * 0.15))
    """
    seed_diff = seed_a - seed_b  # Negative means seed_a is better (lower seed)
    return 1.0 / (1.0 + 10.0 ** (seed_diff * 0.15))


def calibration_analysis(y_true, y_pred, n_bins=10):
    """Compute calibration statistics for reliability diagram."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    observed_freq = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() > 0:
            bin_centers.append(y_pred[mask].mean())
            observed_freq.append(y_true[mask].mean())
            bin_counts.append(mask.sum())

    return {
        "predicted": np.array(bin_centers),
        "observed": np.array(observed_freq),
        "counts": np.array(bin_counts),
    }
