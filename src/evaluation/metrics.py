from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score,
    confusion_matrix, classification_report,
)


# ──────────────────────────────────────────────
# ROC / PR
# ──────────────────────────────────────────────

def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))


def compute_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(labels, scores))


# ──────────────────────────────────────────────
# Threshold selection
# ──────────────────────────────────────────────

def youden_threshold(labels: np.ndarray,
                      scores: np.ndarray) -> Tuple[float, float, float]:
    """
    Maximise Youden's J = TPR - FPR.
    Returns (threshold, fpr_at_thr, tpr_at_thr).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j   = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx]), float(fpr[idx]), float(tpr[idx])


def sigma_threshold(scores: np.ndarray, k: float = 2.0) -> float:
    """μ + k·σ threshold on a score distribution."""
    return float(scores.mean() + k * scores.std())


def threshold_sweep(labels: np.ndarray,
                     scores: np.ndarray,
                     k_values: Optional[List[float]] = None) -> Dict:
    """
    Sweep μ + k·σ thresholds; return F1 at each k.
    Returns dict  {k: {"threshold": float, "f1": float}}
    """
    if k_values is None:
        k_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    results = {}
    for k in k_values:
        thr   = sigma_threshold(scores, k)
        preds = (scores >= thr).astype(int)
        f1    = float(f1_score(labels, preds, zero_division=0))
        results[k] = {"threshold": thr, "f1": f1}
    return results


# ──────────────────────────────────────────────
# Full evaluation
# ──────────────────────────────────────────────

def full_evaluation(labels: np.ndarray,
                     scores: np.ndarray,
                     threshold: Optional[float] = None) -> Dict:
    """
    Compute all metrics.
    If threshold is None, uses Youden's J.
    Returns dict with auroc, auprc, f1, confusion_matrix, report.
    """
    if threshold is None:
        threshold, _, _ = youden_threshold(labels, scores)

    preds = (scores >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(labels, scores)

    return {
        "auroc":      round(auc(fpr, tpr), 4),
        "auprc":      round(average_precision_score(labels, scores), 4),
        "f1":         round(float(f1_score(labels, preds, zero_division=0)), 4),
        "threshold":  round(threshold, 6),
        "cm":         confusion_matrix(labels, preds).tolist(),
        "report":     classification_report(
                          labels, preds,
                          target_names=["Normal", "Anomalous"],
                          zero_division=0
                      ),
        "fpr":        fpr.tolist(),
        "tpr":        tpr.tolist(),
    }


# ──────────────────────────────────────────────
# Printing
# ──────────────────────────────────────────────

def print_metrics(labels: np.ndarray,
                  scores: np.ndarray,
                  threshold: Optional[float] = None) -> Dict:
    """
    Print AUROC, AUPRC, F1, threshold, the μ + k·σ sweep,
    confusion matrix, and classification report.
    Operates on YOUR real labels/scores. Returns the results dict.
    """
    results = full_evaluation(labels, scores, threshold)

    print("=" * 45)
    print("EVALUATION METRICS")
    print("=" * 45)
    print(f"AUROC     : {results['auroc']}")
    print(f"AUPRC     : {results['auprc']}")
    print(f"F1        : {results['f1']}")
    print(f"threshold : {results['threshold']}"
          f"{'  (Youden J)' if threshold is None else ''}")

    print("\nμ + kσ threshold sweep:")
    print(f"{'k':>5} {'threshold':>12} {'F1':>10}")
    for k, r in threshold_sweep(labels, scores).items():
        print(f"{k:>5} {r['threshold']:>12.4f} {r['f1']:>10.4f}")

    print("\nConfusion matrix [ [TN FP] [FN TP] ]:")
    print(np.array(results["cm"]))

    print("\nClassification report:")
    print(results["report"])

    return results