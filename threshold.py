"""
src/evaluation/threshold.py
Adaptive threshold fitting on val-set (normal only) score distribution.
No ground-truth labels needed — purely unsupervised.
"""

from typing import Tuple

import numpy as np
from scipy.stats import norm as scipy_norm


def fit_normal_distribution(scores: np.ndarray) -> Tuple[float, float]:
    """
    Fit Gaussian to a score distribution.
    Returns (mu, sigma).
    """
    mu    = float(scores.mean())
    sigma = float(scores.std(ddof=1))
    return mu, sigma


def threshold_from_quantile(scores: np.ndarray, q: float = 0.95) -> float:
    """Set threshold at the q-th percentile of normal scores."""
    return float(np.percentile(scores, q * 100))


def threshold_from_sigma(scores: np.ndarray, k: float = 2.0) -> float:
    """μ + k·σ threshold."""
    mu, sigma = fit_normal_distribution(scores)
    return mu + k * sigma


def threshold_from_gaussian_fpr(scores: np.ndarray,
                                  target_fpr: float = 0.05) -> float:
    """
    Set threshold so that approx. target_fpr of normal scores exceed it,
    assuming Gaussian distribution.
    """
    mu, sigma = fit_normal_distribution(scores)
    # z-score such that P(Z > z) = target_fpr
    z = scipy_norm.ppf(1.0 - target_fpr)
    return float(mu + z * sigma)


def select_threshold(val_scores: np.ndarray,
                      method: str = "sigma",
                      k: float = 2.0,
                      quantile: float = 0.95,
                      target_fpr: float = 0.05) -> float:
    """
    Convenience wrapper.

    method:
        "sigma"    → μ + k·σ  (default)
        "quantile" → q-th percentile
        "gaussian_fpr" → Gaussian FPR control
    """
    if method == "quantile":
        return threshold_from_quantile(val_scores, quantile)
    if method == "gaussian_fpr":
        return threshold_from_gaussian_fpr(val_scores, target_fpr)
    return threshold_from_sigma(val_scores, k)
