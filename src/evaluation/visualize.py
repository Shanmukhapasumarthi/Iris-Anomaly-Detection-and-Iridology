
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter


# ──────────────────────────────────────────────
# ROC / PR curves
# ──────────────────────────────────────────────

def plot_roc_curve(fpr, tpr, auroc: float,
                   youden_pt: Optional[Tuple] = None,
                   save_path: Optional[Path] = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="#4C72B0",
            label=f"ROC  AUC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    if youden_pt is not None:
        ax.scatter(youden_pt[0], youden_pt[1],
                   color="crimson", zorder=5, s=60,
                   label=f"Youden J  ({youden_pt[0]:.3f}, {youden_pt[1]:.3f})")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    _save_or_show(fig, save_path)


def plot_pr_curve(precision, recall, ap: float,
                  save_path: Optional[Path] = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, color="#55A868",
            label=f"PR  AP = {ap:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# Score distribution
# ──────────────────────────────────────────────

def plot_score_distribution(scores: np.ndarray,
                              labels: Optional[np.ndarray] = None,
                              threshold: Optional[float] = None,
                              save_path: Optional[Path] = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))

    if labels is not None and labels.sum() > 0:
        ax.hist(scores[labels == 0], bins=50, alpha=0.65,
                color="#4C72B0", edgecolor="white",
                linewidth=0.3, label="Normal")
        ax.hist(scores[labels == 1], bins=50, alpha=0.65,
                color="#C44E52", edgecolor="white",
                linewidth=0.3, label="Anomalous")
        ax.legend(fontsize=9)
    else:
        ax.hist(scores, bins=60, color="#4C72B0",
                edgecolor="white", linewidth=0.4, label="All scores")

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--",
                   linewidth=1.5, label=f"Threshold = {threshold:.5f}")
        ax.legend(fontsize=9)

    ax.set_xlabel("Anomaly score"); ax.set_ylabel("Count")
    ax.set_title("Anomaly score distribution")
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# Confusion matrix
# ──────────────────────────────────────────────

def plot_confusion_matrix(cm_arr: np.ndarray,
                           save_path: Optional[Path] = None) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomalous"])
    ax.set_yticklabels(["Normal", "Anomalous"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# Training curves
# ──────────────────────────────────────────────

def plot_training_curves(history: Dict,
                          title: str = "Training curves",
                          save_path: Optional[Path] = None) -> None:
    keys   = list(history.keys())
    n_axes = max(1, len(keys) // 2)
    fig, axes = plt.subplots(1, n_axes, figsize=(7 * n_axes, 4))
    if n_axes == 1:
        axes = [axes]

    pairs = [(keys[i], keys[i + 1]) for i in range(0, len(keys) - 1, 2)]
    for ax, (train_k, val_k) in zip(axes, pairs):
        ax.plot(history[train_k], label="Train")
        ax.plot(history[val_k],   label="Val")
        ax.set_title(train_k.replace("train_", "").capitalize())
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# Sigma sweep
# ──────────────────────────────────────────────

def plot_sigma_sweep(sweep: Dict,
                      save_path: Optional[Path] = None) -> None:
    ks   = list(sweep.keys())
    f1s  = [sweep[k]["f1"]        for k in ks]
    thrs = [sweep[k]["threshold"] for k in ks]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.plot(ks, f1s,  marker="o", color="#4C72B0", label="F1")
    ax2.plot(ks, thrs, marker="s", color="#C44E52",
             linestyle="--", label="Threshold")
    ax1.set_xlabel("k  (μ + k·σ)")
    ax1.set_ylabel("F1 score",       color="#4C72B0")
    ax2.set_ylabel("Threshold value", color="#C44E52")
    ax1.set_title("Threshold sensitivity sweep")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# Heatmap helpers
# ──────────────────────────────────────────────

def error_map_to_heatmap(err_map: np.ndarray,
                          sigma: float = 2.0) -> np.ndarray:
    """Smooth error map → uint8 (H, W, 3) jet heatmap."""
    smoothed = gaussian_filter(err_map, sigma=sigma)
    mn, mx   = smoothed.min(), smoothed.max()
    norm     = (smoothed - mn) / (mx - mn + 1e-8)
    return (cm.jet(norm)[:, :, :3] * 255).astype(np.uint8)


def backproject_heatmap(orig_gray:     np.ndarray,
                         heatmap_strip: np.ndarray,
                         pupil:         Tuple,
                         iris:          Tuple,
                         strip_rows:    int = 64,
                         strip_cols:    int = 512,
                         save_path:     Optional[Path] = None) -> None:
    """
    Overlay polar heatmap back onto the original circular iris image.
    Falls back to side-by-side if pupil/iris not provided.
    """
    h, w  = orig_gray.shape
    color = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)

    if pupil is not None and iris is not None:
        px, py, pr = pupil
        ix, iy, ir = iris

        thetas  = np.linspace(0, 2 * np.pi, strip_cols, endpoint=False)
        r_norms = np.linspace(0, 1.0, strip_rows)
        cos_t   = np.cos(thetas); sin_t = np.sin(thetas)

        overlay = color.astype(np.float32)
        for row_i, r in enumerate(r_norms):
            for col_i in range(strip_cols):
                cx = int((1-r)*(px + pr*cos_t[col_i]) + r*(ix + ir*cos_t[col_i]))
                cy = int((1-r)*(py + pr*sin_t[col_i]) + r*(iy + ir*sin_t[col_i]))
                if 0 <= cx < w and 0 <= cy < h:
                    hp = heatmap_strip[row_i, col_i].astype(np.float32)
                    overlay[cy, cx] = overlay[cy, cx] * 0.5 + hp * 0.5
        blended = np.clip(overlay, 0, 255).astype(np.uint8)
    else:
        blended = color

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(orig_gray, cmap="gray"); axes[0].set_title("Original")
    axes[1].imshow(heatmap_strip);          axes[1].set_title("Error heatmap")
    axes[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Backprojected overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# Private helper
# ──────────────────────────────────────────────

def _save_or_show(fig, path: Optional[Path]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path}")
    else:
        plt.show()
