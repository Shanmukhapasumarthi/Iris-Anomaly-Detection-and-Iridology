import json
import time
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys_path = [
    str((ROOT / 'src').resolve()),
    str((ROOT / 'src' / 'Models').resolve()),
    str((ROOT / 'src' / 'training').resolve()),
    str((ROOT / 'src' / 'utils').resolve()),
    str((ROOT / 'src' / 'evaluation').resolve()),
    str((ROOT / 'src' / 'preprocessing').resolve()),
    str(ROOT),
]
for p in sys_path:
    if p not in sys.path:
        sys.path.insert(0, p)


from src.evaluation.metrics import full_evaluation
from src.Models.autoencoder import ConvAutoencoder
from src.Models.vae import ConvVAE
from src.Models.patch_ae import PatchCoreDetector
from src.utils.dataset import subject_disjoint_split
from scripts.train import ViTMAE

CHECKPOINT_DIR = ROOT / 'checkpoints'
NORM_REC_FILE = ROOT / 'data' / 'normalized' / 'normalization_records.json'

with open(NORM_REC_FILE, 'r', encoding='utf-8') as f:
    original_records = json.load(f)
records = []
for rec in original_records:
    strip_path = Path(rec['strip'])
    if not strip_path.exists():
        strip_path = ROOT / 'data' / 'normalized' / strip_path.name
    if not strip_path.exists():
        continue
    records.append({**rec, 'strip': str(strip_path)})

class FixedDataset:
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        path = Path(self.records[idx]['strip'])
        strip = np.load(str(path)).astype(np.float32)
        if strip.shape != (64, 512):
            import cv2
            strip = cv2.resize(strip, (512, 64), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(strip).unsqueeze(0)

from torch.utils.data import DataLoader

def build_fixed_loader(records, batch_size=16):
    return DataLoader(FixedDataset(records), batch_size=batch_size, shuffle=False, num_workers=0)

# Honest subject-disjoint split to avoid leakage
train_records, val_records, test_records = subject_disjoint_split(
    records,
    train_ratio=0.80,
    val_ratio=0.10,
    seed=42,
)
train_loader = build_fixed_loader(train_records)
val_loader = build_fixed_loader(val_records)
test_loader = build_fixed_loader(test_records)


def infer_latent_dim(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt.get('state', ckpt)
    key = next((k for k in state if 'fc_enc.weight' in k or 'fc_mu.weight' in k), None)
    if key is not None:
        return int(state[key].shape[0])
    return 256


def honest_threshold(scores):
    return float(scores.mean() + 2.0 * scores.std())


def load_gt_labels(gt_file=None):
    if gt_file is None:
        gt_file = ROOT / 'data' / 'gt_labels.json'
    if not gt_file.exists():
        return None
    try:
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt = json.load(f)
        labels = []
        for rec in test_records:
            origin = rec.get('original', '')
            labels.append(int(gt.get(origin, 0)))
        return np.asarray(labels, dtype=int)
    except Exception:
        return None


def summarize(name, scores, threshold, val_scores, labels=None):
    flagged = int((scores >= threshold).sum())
    flagged_pct = float(flagged / len(scores))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    if labels is not None:
        preds = (scores >= threshold).astype(int)
        accuracy = float(np.mean(preds == labels))
        accuracy_note = 'ground-truth accuracy'
    else:
        accuracy = float(np.mean(scores < threshold))
        accuracy_note = 'accuracy proxy (assumes most samples are normal)'

    return {
        'model': name,
        'threshold': float(threshold),
        'min_score': min_score,
        'max_score': max_score,
        'mean_score': mean_score,
        'std_score': std_score,
        'flagged_count': flagged,
        'flagged_pct': flagged_pct,
        'accuracy': accuracy,
        'accuracy_note': accuracy_note,
    }


def print_model_box(result):
    width = 60
    header = f"{'=' * width}"
    print(header)
    print(f"{result['model'].upper():>12} : RESULTS")
    print(header)
    print(f"Threshold      : {result['threshold']:.6f}")
    print(f"Anomaly Score  : min={result['min_score']:.6f}, max={result['max_score']:.6f}")
    print(f"Mean ± Std     : {result['mean_score']:.6f} ± {result['std_score']:.6f}")
    print(f"Flagged        : {result['flagged_count']}/{len(test_records)} ({result['flagged_pct']*100:.1f}%)")
    print(f"Accuracy       : {result['accuracy']:.4f}  ({result['accuracy_note']})")
    print(header)
    print()


results = []
models = []
for name, cls, path in [
    ('ae', ConvAutoencoder, CHECKPOINT_DIR / 'best_ae.pth'),
    ('vae', ConvVAE, CHECKPOINT_DIR / 'best_vae.pth'),
    ('patchcore', PatchCoreDetector, CHECKPOINT_DIR / 'patchcore.npz'),
    ('mae', ViTMAE, CHECKPOINT_DIR / 'best_mae.pth'),
]:
    if path.exists():
        models.append((name, cls, path))

gt_labels = None
if (ROOT / 'data' / 'gt_labels.json').exists():
    with open(ROOT / 'data' / 'gt_labels.json', 'r', encoding='utf-8') as f:
        gt_map = json.load(f)
    gt_labels = np.array([
        int(gt_map.get(rec.get('original', ''), 0))
        for rec in test_records
    ], dtype=int)

for name, cls, path in models:
    if name == 'patchcore':
        detector = cls(coreset_size=2000)
        detector.load(str(path))
        detector.device = torch.device('cpu')
        detector.extractor.to(detector.device)
        detector.extractor.eval()
        val_scores, test_scores = [], []
        start = time.time()
        with torch.no_grad():
            for batch in val_loader:
                val_scores.append(detector.score_batch(batch))
            for batch in test_loader:
                test_scores.append(detector.score_batch(batch))
        val_scores = np.concatenate(val_scores)
        test_scores = np.concatenate(test_scores)
    else:
        if name == 'mae':
            model = cls()
        else:
            latent_dim = infer_latent_dim(path)
            model = cls(latent_dim=latent_dim)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['state'])
        model.eval()
        val_scores, test_scores = [], []
        start = time.time()
        with torch.no_grad():
            for batch in val_loader:
                val_scores.append(model.anomaly_score(batch).cpu().numpy())
            for batch in test_loader:
                test_scores.append(model.anomaly_score(batch).cpu().numpy())
        val_scores = np.concatenate(val_scores)
        test_scores = np.concatenate(test_scores)
    threshold = honest_threshold(val_scores)
    result = summarize(name, test_scores, threshold, val_scores, gt_labels)
    results.append(result)
    print_model_box(result)

print('\n' + '='*60)
best = min(results, key=lambda x: x['threshold'])
print(f"BEST MODEL: {best['model'].upper()} (threshold={best['threshold']:.6f})")
print('='*60)
