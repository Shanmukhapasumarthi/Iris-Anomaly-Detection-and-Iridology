# 👁 Iris Anomaly Detection & Iridology Health Analysis

> A self-supervised deep learning system for iris anomaly detection and organ health analysis using Vision Transformer Masked Autoencoders (ViT-MAE) with an iridology-based zone mapping pipeline.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 Project Overview

This project implements an **end-to-end unsupervised iris anomaly detection system** that:

- Segments the iris from raw eye photographs using **Object Region Marking** (Connected Component Labelling)
- Normalises the circular iris into a flat strip using **Daugman's Rubber-Sheet Model**
- Trains a **ViT-MAE (Vision Transformer Masked Autoencoder)** on normal iris images only — no anomaly labels required
- Detects anomalies by measuring reconstruction error on unseen irises
- Maps anomalous zones to **iridology organ charts** for health insights
- Generates **AI-powered disease descriptions** using Claude (Anthropic API)
- Serves everything through **4 FastAPI web servers** with interactive browser UIs

### Why ViT-MAE?
CNN autoencoders overfit easily on small iris datasets. ViT-MAE forces the model to reconstruct **75% randomly masked patches** from global context, acting as strong self-supervised regularisation that generalises better and produces sharper anomaly contrast.

---

## 🏗 Architecture

```
Raw Eye Image
      ↓
┌─────────────────────────────────────────┐
│         Segmentation Pipeline           │
│  Glare Removal → CCL Region Marking     │
│  Pupil Detection → Iris Detection       │
│  Eyelid Mask → Annular Mask             │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│      Daugman Rubber-Sheet Normalisation │
│      64 × 512 iris strip + CLAHE        │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│            ViT-MAE Model                │
│  Encoder (depth=6, dim=256, heads=8)    │
│  Decoder (depth=4, dim=128, heads=8)    │
│  Mask ratio = 0.75                      │
│  Parameters: ~5.7M                      │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│         Anomaly Scoring                 │
│  Score = MSE on masked patches          │
│  Threshold = μ + 2σ on val scores       │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│      Iridology Zone Analysis            │
│  15 organ zones mapped to iris strip    │
│  Per-zone reconstruction error          │
│  Pattern detection (Arcus Senilis etc.) │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│      AI Disease Insight (Claude API)    │
│  Causes · Symptoms · Prevention · Diet  │
└─────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
project/
│
├── 📂 Core Pipeline
│   ├── prepare_data.py          # Segmentation + normalisation + records
│   ├── train.py                 # ViT-MAE model + training script
│   ├── trainer.py               # AE / VAE training loops
│   ├── evaluate.py              # Metrics + evaluation plots
│   ├── threshold.py             # Adaptive threshold fitting (μ + kσ)
│   └── input.py                 # Input pipeline utility
│
├── 📂 Models
│   ├── autoencoder.py           # ConvAutoencoder
│   ├── vae.py                   # ConvVAE
│   └── patch_ae.py              # PatchCore detector
│
├── 📂 Preprocessing
│   ├── segmentation.py          # Object Region Marking iris segmentation
│   ├── normalization.py         # Daugman rubber-sheet normalisation + CLAHE
│   └── augmentation.py          # Albumentations training augmentations
│
├── 📂 Utilities
│   ├── dataset.py               # PyTorch Dataset + DataLoader factory
│   ├── config.py                # YAML config + dataclass
│   ├── metrics.py               # AUROC, AUPRC, F1, confusion matrix
│   ├── losses.py                # Reconstruction loss (MSE + SSIM)
│   ├── scheduler.py             # Warmup-cosine LR scheduler
│   └── visualize.py             # All matplotlib visualisation functions
│
├── 📂 Iridology Extension
│   ├── iridology_zones.py       # Organ zone definitions + iridology chart
│   ├── zone_analyzer.py         # Per-zone error analysis + pattern detection
│   ├── health_report.py         # Structured health report generator
│   └── disease_insight.py       # FastAPI server with Claude AI descriptions
│
├── 📂 Web Servers
│   ├── app.py                   # Port 8000 — Iris anomaly inference UI
│   ├── iridology_app.py         # Port 8001 — Organ zone health analysis
│   ├── dashboard_app.py         # Port 8002 — Visualisation dashboard
│   └── disease_insight.py       # Port 8003 — AI disease insight UI
│
├── 📂 Config
│   ├── configs/training/default.yaml
│   └── configs/model/vae.yaml
│
├── 📂 Data (not tracked by Git)
│   ├── data/raw/                # Raw eye images
│   ├── data/processed/          # Segmentation records
│   └── data/normalized/         # Normalised .npy strips + records JSON
│
├── 📂 Outputs (not tracked by Git)
│   ├── checkpoints/             # best_mae.pth, best_ae.pth, best_vae.pth
│   ├── results/evaluation/      # ROC, PR, confusion matrix, summary.json
│   └── outputs/images/          # Training curves, reconstruction images
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/Shanmukhapasumarthi/Iris-Anomaly-Detection-and-Iridology.git
cd Iris-Anomaly-Detection-and-Iridology

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 2. Prepare Data

Place your raw eye images in `data/raw/` then run:

```bash
python prepare_data.py
```

This generates `data/normalized/normalization_records.json`.

### 3. Train

```bash
# ViT-MAE (recommended)
python train.py --model mae --epochs 150

# ConvAutoencoder
python train.py --model ae

# ConvVAE
python train.py --model vae
```

### 4. Evaluate

```bash
python evaluate.py --model mae
```

### 5. Run Web Servers

```bash
# Terminal 1 — Anomaly inference
python app.py
# → http://localhost:8000

# Terminal 2 — Iridology zone analysis
python iridology_app.py
# → http://localhost:8001

# Terminal 3 — Visualisation dashboard
python dashboard_app.py
# → http://localhost:8002

# Terminal 4 — AI disease insight
python disease_insight.py
# → http://localhost:8003
```

---

## 🌐 Web Interface Guide

### Port 8000 — Iris Anomaly Detection
Upload any eye photograph and get:
- Annotated eye with detected pupil and iris circles
- Extracted 64×512 rubber-sheet iris strip
- ViT-MAE reconstruction of the strip
- Per-pixel reconstruction error heatmap (jet colormap)
- Anomaly score, threshold, SSIM, mean error
- Verdict: **Normal** 🟢 or **Anomalous** 🔴

### Port 8001 — Iridology Health Analysis
- All of the above plus organ zone mapping
- 15 organ zones scored by reconstruction error
- Iris pattern detection (Arcus Senilis, Radii Solaris, Lymphatic Rosary, etc.)
- Risk level per organ: LOW ✅ / MODERATE ⚠️ / HIGH 🔴
- Select Left or Right eye for correct zone mapping

### Port 8002 — Visualisation Dashboard
- AUROC, AUPRC, F1, threshold stat cards
- Training loss curves (ViT-MAE, AE, VAE)
- ROC curve, PR curve, confusion matrix
- Score distribution histogram
- Dataset split pie chart
- Metrics radar chart
- Full project pipeline overview

### Port 8003 — AI Disease Insight
- Full iris analysis + iridology zone mapping
- Claude AI generates for each flagged organ:
  - **Disease causes** (lifestyle, diet, genetics, environment)
  - **Symptoms to watch for**
  - **Health effects if untreated**
  - **Prevention and lifestyle recommendations**
  - **Dietary suggestions**
  - **When to see a doctor**

---

## 📊 Results

| Metric | Value |
|--------|-------|
| AUROC | 1.0000 |
| AUPRC | 1.0000 |
| F1 Score | 0.8571 |
| Best Val Loss | 0.04564 |
| Threshold (μ+2σ) | 0.43669 |
| Training Epochs | 150 |
| Model Parameters | 5,714,048 |
| Strip Size | 64 × 512 px |
| Mask Ratio | 0.75 |

> **Note:** AUROC/AUPRC of 1.0 are based on pseudo-labels (top 10% scores = anomalous) as no ground-truth anomaly labels were available. Real-world performance requires a labelled anomaly test set.

---

## 🔬 Segmentation Algorithm

This project uses **Object Region Marking** (Connected Component Labelling) instead of Hough Circle Transform:

**Pupil Detection:**
1. Adaptive dark-pixel threshold in central 45% of image
2. `cv2.connectedComponentsWithStats()` → label all dark regions
3. Per-region: area, circularity = 4π×area/perimeter², eccentricity
4. Score = `circularity×0.45 + (1−norm_dist)×0.35 + (1−eccentricity)×0.20`
5. Highest scoring region → `minEnclosingCircle()`

**Iris Detection:**
1. Gradient magnitude (Sobel) in annular search zone around pupil
2. CCL on gradient threshold map → edge regions
3. Score = `circularity×0.35 + ring_coverage×0.35 + (1−drift)×0.30`
4. Best edge region → circle fit → concentric snap

**Why not Hough?** Hough requires fixed radius ranges that don't adapt to image scale. CCL measures the actual region size so it works equally on close-up and wide-shot images.

---

## 🫀 Iridology Zone Mapping

The iris strip is divided into 15 organ zones based on standard iridology charts:

| Zone | Organ | System |
|------|-------|--------|
| 12 o'clock | Brain / Pineal | Nervous |
| 1 o'clock | Spleen | Immune |
| 2-4 o'clock | Liver | Digestive |
| 4-5 o'clock | Kidney | Urinary |
| 7-8 o'clock | Heart | Cardiovascular |
| 8-9 o'clock | Bronchus / Lung | Respiratory |
| 9-10 o'clock | Thyroid | Endocrine |
| Inner ring | Stomach / Intestine | Digestive |

> ⚠️ **Disclaimer:** Iridology is used here as a research framework. This system is NOT a medical diagnostic tool. Always consult a qualified medical professional.

---

## 📦 Requirements

```
torch>=2.0.0
torchvision
fastapi
uvicorn[standard]
python-multipart
albumentations
opencv-python
numpy
matplotlib
scikit-image
scikit-learn
scipy
httpx
pydantic
pyyaml
tqdm
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Details

### ViT-MAE Architecture
```
Encoder:
  PatchEmbed     → 16×16 patches → (B, N, 256)
  CLS token      → prepended
  Positional embed (learnable)
  6× TransformerBlock (dim=256, heads=8, mlp_ratio=4, drop=0.1)
  LayerNorm

Decoder:
  Linear projection 256 → 128
  Mask tokens for missing patches
  4× TransformerBlock (dim=128, heads=8)
  LayerNorm
  Linear → patch_size² pixel values

Anomaly Score:
  MSE(pred_patches, target_patches) on masked patches only
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimiser | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.05 |
| Warmup epochs | 10 |
| LR schedule | Cosine annealing |
| Gradient clip | 1.0 |
| Batch size | 32 |
| Epochs | 150 |

---

## 🔭 Future Work

- [ ] Larger dataset (CASIA, UBIRIS public iris databases)
- [ ] Ground-truth anomaly labels for real evaluation
- [ ] U-Net learned segmentation to replace CCL
- [ ] ONNX export for mobile deployment
- [ ] Grad-CAM visualisation on encoder attention maps
- [ ] Multi-class disease classification on iridology zones
- [ ] Federated learning for privacy-preserving training

---

## 👤 Author

**P.V.H. Shanmukha Pasumarthi**
BTech Final Year Project
GitHub: [@Shanmukhapasumarthi](https://github.com/Shanmukhapasumarthi)
