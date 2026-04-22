# Iris Texture Anomaly Detection
### Polar Normalization + Deep Feature Learning

Unsupervised anomaly detection on iris images using Daugman's rubber-sheet
normalization and autoencoder-based reconstruction error scoring.

---

## Project Structure

```
iris_anomaly/
├── data/
│   ├── raw/              ← place your iris images here
│   ├── processed/        ← segmentation masks & records
│   └── normalized/       ← 64×512 polar strips (.npy)
├── src/
│   ├── preprocessing/
│   │   ├── segmentation.py    circular Hough / eyelid masking
│   │   ├── normalization.py   rubber-sheet model
│   │   └── augmentation.py    albumentations pipeline
│   ├── models/
│   │   ├── autoencoder.py     Conv AE baseline
│   │   ├── vae.py             Variational AE (main model)
│   │   └── patch_ae.py        PatchCore-style (advanced)
│   ├── training/
│   │   ├── trainer.py         unified train loops
│   │   ├── losses.py          MSE + SSIM + KL
│   │   └── scheduler.py       warmup + cosine LR
│   ├── evaluation/
│   │   ├── metrics.py         AUROC, AUPRC, F1
│   │   ├── threshold.py       adaptive thresholding
│   │   └── visualize.py       heatmaps, score dist.
│   └── utils/
│       ├── dataset.py         PyTorch Dataset + DataLoaders
│       └── config.py          YAML config loader
├── configs/
│   ├── model/vae.yaml
│   └── training/default.yaml
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_analysis.ipynb
├── scripts/
│   ├── prepare_data.py    stages 1–3 (EDA → seg → norm)
│   ├── train.py           model training
│   └── evaluate.py        metrics + plots
└── api/
    └── app.py             FastAPI inference server
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get dataset (choose one)

**Option A — HuggingFace (no login)**
```bash
python -c "
from huggingface_hub import hf_hub_download
import zipfile
z = hf_hub_download('chitradrishti/CASIA-IRIS',
    'CASIA-Iris-Interval.zip', repo_type='dataset', local_dir='data/raw')
zipfile.ZipFile(z).extractall('data/raw')
"
```

**Option B — Kaggle**
```bash
kaggle datasets download naureenmohammad/mmu-iris-dataset
unzip mmu-iris-dataset.zip -d data/raw/
```

### 3. Prepare data (stages 1–3)
```bash
python scripts/prepare_data.py
```

### 4. Train
```bash
python scripts/train.py --model vae --epochs 150
```

### 5. Evaluate
```bash
python scripts/evaluate.py --model vae
```

### 6. (Optional) Start API server
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
# Test:
curl -X POST http://localhost:8000/predict -F "file=@iris.jpg"
```

---

## Pipeline Overview

| Stage | Script | Output |
|-------|--------|--------|
| 1 EDA | `stage1_eda.py` | `reports/eda/` — plots, quality flags |
| 2 Segmentation | `stage2_segmentation.py` | `data/processed/` — masks, records |
| 3 Normalization | `stage3_normalization.py` | `data/normalized/` — 64×512 strips |
| 4 Dataset | `utils/dataset.py` | PyTorch DataLoaders |
| 5 Training | `scripts/train.py` | `checkpoints/best_vae.pth` |
| 6 Scoring | `scripts/evaluate.py` | `results/` — scores, heatmaps |
| 7 Evaluation | `scripts/evaluate.py` | `results/evaluation/` — ROC, PR, F1 |

---

## Model Architectures

| Model | Description | Anomaly Score |
|-------|-------------|---------------|
| `ConvAutoencoder` | Baseline Conv AE | Mean pixel MSE |
| `ConvVAE` | Variational AE | Recon MSE + β·KL |
| `PatchCoreDetector` | EfficientNet memory bank | Max patch NN distance |

---

## Expected Results

| Model | AUROC | Notes |
|-------|-------|-------|
| ConvAE | ≥ 0.82 | Baseline |
| ConvVAE | ≥ 0.87 | Main model |
| PatchCore | ≥ 0.91 | Best, no training |

---

## Configuration

Edit `configs/training/default.yaml` and `configs/model/vae.yaml`
to change hyperparameters without touching source code.

Key parameters:
- `latent_dim` — bottleneck size (default: 256)
- `gamma_kl` — KL weight for VAE (default: 1.0)
- `alpha_mse` / `beta_ssim` — reconstruction loss weights
- `epochs`, `lr`, `batch_size`
