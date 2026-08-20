from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ──────────────────────────────────────────────
# Dataclass config
# ──────────────────────────────────────────────

@dataclass
class DataConfig:
    raw_dir:          Path = Path("data/raw")
    processed_dir:    Path = Path("data/processed")
    normalized_dir:   Path = Path("data/normalized")
    strip_h:          int  = 64
    strip_w:          int  = 512
    train_ratio:      float = 0.80
    val_ratio:        float = 0.10
    batch_size:       int   = 16
    num_workers:      int   = 4
    use_clahe:        bool  = True


@dataclass
class ModelConfig:
    model_type:  str = "vae"      # "ae" | "vae" | "patchcore"
    latent_dim:  int = 64
    coreset_size: int = 2000       # PatchCore only


@dataclass
class TrainingConfig:
    epochs:        int   = 100
    lr:            float = 1e-4
    weight_decay:  float = 1e-4
    warmup_epochs: int   = 10
    grad_clip:     float = 1.0
    alpha_mse:     float = 0.5
    beta_ssim:     float = 0.5
    gamma_kl:      float = 1.0
    log_every:     int   = 10
    seed:          int   = 42


@dataclass
class PathsConfig:
    checkpoint_dir: Path = Path("checkpoints")
    log_dir:        Path = Path("logs")
    report_dir:     Path = Path("reports")
    results_dir:    Path = Path("results")


@dataclass
class Config:
    data:     DataConfig     = field(default_factory=DataConfig)
    model:    ModelConfig    = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths:    PathsConfig    = field(default_factory=PathsConfig)


# ──────────────────────────────────────────────
# YAML loader
# ──────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    if not _YAML_AVAILABLE:
        return {}
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(
    training_yaml: Path = Path("configs/training/default.yaml"),
    model_yaml:    Path = Path("configs/model/vae.yaml"),
) -> Config:
    """Load config from YAML files; fall back to defaults."""
    t_cfg = _load_yaml(training_yaml)
    m_cfg = _load_yaml(model_yaml)

    cfg = Config()

    # Override training
    for k, v in t_cfg.get("training", {}).items():
        if hasattr(cfg.training, k):
            setattr(cfg.training, k, v)

    for k, v in t_cfg.get("data", {}).items():
        if hasattr(cfg.data, k):
            setattr(cfg.data, k, Path(v) if "dir" in k else v)

    # Override model
    for k, v in m_cfg.get("model", {}).items():
        if hasattr(cfg.model, k):
            setattr(cfg.model, k, v)

    return cfg


# ──────────────────────────────────────────────
# Quick access
# ──────────────────────────────────────────────

def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
