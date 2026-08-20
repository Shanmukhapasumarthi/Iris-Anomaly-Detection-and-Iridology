"""
scripts/evaluate.py
Load best checkpoint → score all images → compute metrics → save plots.

Usage:
    python scripts/evaluate.py                        # default: VAE
    python scripts/evaluate.py --model ae
    python scripts/evaluate.py --model patchcore
    python scripts/evaluate.py --threshold 0.012      # manual threshold
    python scripts/evaluate.py --gt data/gt_labels.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if (ROOT / "segmentation.py").is_file():
    pass
else:
    ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOT))

from src.Models.autoencoder    import ConvAutoencoder
from src.Models.vae            import ConvVAE
from src.Models.patch_ae       import PatchCoreDetector
from src.utils.dataset        import build_dataloaders, subject_disjoint_split
from src.utils.config         import get_device
from src.evaluation.metrics        import full_evaluation, threshold_sweep, youden_threshold
from src.evaluation.threshold      import select_threshold
from src.evaluation.visualize      import (
    plot_roc_curve, plot_pr_curve, plot_score_distribution,
    plot_confusion_matrix, plot_sigma_sweep,
    error_map_to_heatmap, backproject_heatmap,
)
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
NORM_REC_FILE  = ROOT / "data/normalized/normalization_records.json"
SEG_REC_FILE   = ROOT / "data/processed/segmentation_records.json"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR    = ROOT / "results"
EVAL_DIR       = RESULTS_DIR / "evaluation"


# ──────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────

def load_model(model_type: str, latent_dim: int, device: torch.device):
    if model_type == "ae":
        model = ConvAutoencoder(latent_dim)
        ckpt  = torch.load(CHECKPOINT_DIR / "best_ae.pth",
                           map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state"])
        model.to(device).eval()
        print(f"  Loaded AE checkpoint (epoch {ckpt.get('epoch','?')}, "
              f"val_loss={ckpt.get('val_loss',0):.5f})")
        return model, "ae"

    elif model_type == "vae":
        model = ConvVAE(latent_dim)
        ckpt  = torch.load(CHECKPOINT_DIR / "best_vae.pth",
                           map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state"])
        model.to(device).eval()
        print(f"  Loaded VAE checkpoint (epoch {ckpt.get('epoch','?')}, "
              f"val_loss={ckpt.get('val_loss',0):.5f})")
        return model, "vae"

    else:   # patchcore
        detector = PatchCoreDetector(device=device)
        detector.load(str(CHECKPOINT_DIR / "patchcore.npz"))
        return detector, "patchcore"


# ──────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────

@torch.no_grad()
def score_loader(model, loader, model_type: str,
                 device: torch.device) -> np.ndarray:
    """Return anomaly scores for every sample in loader."""
    scores = []
    for batch in loader:
        x = batch.to(device)
        if model_type == "patchcore":
            s = model.score_batch(x)
        else:
            s = model.anomaly_score(x).cpu().numpy()
        scores.append(s)
    return np.concatenate(scores)


# ──────────────────────────────────────────────
# Ground-truth labels
# ──────────────────────────────────────────────

def load_gt_labels(gt_file: Optional[Path],
                    records,
                    scores: np.ndarray) -> Optional[np.ndarray]:
    if gt_file and gt_file.exists():
        with open(gt_file) as f:
            gt = json.load(f)   # {path: 0|1}
        labels = np.array([gt.get(r.get("original", ""), 0)
                            for r in records], dtype=int)
        print(f"  GT labels loaded — anomalous: {labels.sum()} / {len(labels)}")
        return labels

    print("  No GT labels provided: skipping AUROC/AUPRC and pseudo-label evaluation.")
    print("  Results below are unsupervised threshold-only metrics.")
    return None


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Iris Anomaly — Evaluate")
    parser.add_argument("--model",      default="vae",
                        choices=["ae", "vae", "patchcore"])
    parser.add_argument("--latent-dim", type=int,   default=64,
                        dest="latent_dim")
    parser.add_argument("--batch-size", type=int,   default=16,
                        dest="batch_size")
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Manual threshold override")
    parser.add_argument("--gt",         type=Path,  default=None,
                        help="JSON file mapping image path → label (0/1)")
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"  Device : {device}")

    # ── Load data ──
    _, val_loader, test_loader = build_dataloaders(
        records_file=NORM_REC_FILE,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    with open(NORM_REC_FILE) as f:
        all_records = json.load(f)

    # ── Load model ──
    model, model_type = load_model(args.model, args.latent_dim, device)

    # ── Score val set (to fit threshold) ──
    print("\n  Scoring val set …")
    val_scores = score_loader(model, val_loader, model_type, device)

    # ── Score test set ──
    print("  Scoring test set …")
    test_scores = score_loader(model, test_loader, model_type, device)

    _, _, test_records = subject_disjoint_split(
        all_records,
        train_ratio=0.80,
        val_ratio=0.10,
        seed=42,
    )
    n_test = len(test_scores)

    # ── Threshold ──
    if args.threshold:
        threshold = args.threshold
        print(f"  Using manual threshold: {threshold}")
    else:
        threshold = select_threshold(val_scores, method="sigma", k=2.0)
        print(f"  Auto threshold (μ+2σ on val): {threshold:.5f}")

    # ── Labels ──
    labels = load_gt_labels(args.gt, test_records, test_scores)

    if labels is None:
        flagged = int((test_scores >= threshold).sum())
        print(f"\n{'─'*45}")
        print(f"  Threshold only evaluation")
        print(f"  Flagged samples: {flagged} / {n_test}")
        print(f"  Threshold      : {threshold:.5f}")
        print(f"{'─'*45}")

        summary = {
            "model": model_type,
            "threshold": float(threshold),
            "n_test": int(n_test),
            "flagged_samples": int(flagged),
            "gt_available": False,
            "note": "No ground-truth labels were provided; AUROC/AUPRC were not computed."
        }
        with open(EVAL_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  All evaluation outputs → {EVAL_DIR}/")
        print("✓ Threshold-only evaluation complete.")
        return

    # ── Metrics ──
    results = full_evaluation(labels, test_scores, threshold)
    sweep   = threshold_sweep(labels, test_scores)
    best_k  = max(sweep, key=lambda k: sweep[k]["f1"])

    fpr, tpr, _ = roc_curve(labels, test_scores)
    prec, rec, _= precision_recall_curve(labels, test_scores)
    y_thr, y_fpr, y_tpr = youden_threshold(labels, test_scores)

    print(f"\n{'─'*45}")
    print(f"  AUROC       : {results['auroc']:.4f}")
    print(f"  AUPRC       : {results['auprc']:.4f}")
    print(f"  F1 @ thr    : {results['f1']:.4f}")
    print(f"  Threshold   : {threshold:.5f}")
    print(f"  Best σ sweep: k={best_k}  F1={sweep[best_k]['f1']:.4f}")
    print(f"{'─'*45}")
    print(results["report"])

    # ── Plots ──
    plot_roc_curve(fpr, tpr, results["auroc"],
                   youden_pt=(y_fpr, y_tpr),
                   save_path=EVAL_DIR / "roc_curve.png")
    plot_pr_curve(prec, rec, results["auprc"],
                  save_path=EVAL_DIR / "pr_curve.png")
    plot_score_distribution(test_scores, labels, threshold,
                             save_path=EVAL_DIR / "score_dist.png")
    plot_confusion_matrix(np.array(results["cm"]),
                          save_path=EVAL_DIR / "confusion_matrix.png")
    plot_sigma_sweep(sweep, save_path=EVAL_DIR / "sigma_sweep.png")

    # ── Save JSON summary ──
    summary = {
        "model":          model_type,
        "auroc":          results["auroc"],
        "auprc":          results["auprc"],
        "f1":             results["f1"],
        "threshold":      float(threshold),
        "youden_thr":     float(y_thr),
        "best_sigma_k":   best_k,
        "n_test":         int(n_test),
        "n_normal":       int((labels == 0).sum()),
        "n_anomalous":    int((labels == 1).sum()),
        "confusion_matrix": results["cm"],
        "gt_available":   True,
    }
    with open(EVAL_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  All evaluation outputs → {EVAL_DIR}/")
    print("✓ Evaluation complete.")


if __name__ == "__main__":
    main()
