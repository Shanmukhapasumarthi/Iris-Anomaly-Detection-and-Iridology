from typing import Tuple

import cv2
import numpy as np

# Default output dimensions
NORM_ROWS = 64    # radial (pupil → iris boundary)
NORM_COLS = 512   # angular (0 → 2π)

Circle = Tuple[int, int, int]   # (cx, cy, r)


def rubber_sheet_normalize(
    gray:  np.ndarray,
    pupil: Circle,
    iris:  Circle,
    rows:  int = NORM_ROWS,
    cols:  int = NORM_COLS,
) -> np.ndarray:
    """
    Daugman's rubber-sheet model.

    For each (r_norm, θ):
        x = (1 - r_norm)*x_pupil(θ) + r_norm*x_iris(θ)
        y = (1 - r_norm)*y_pupil(θ) + r_norm*y_iris(θ)

    r_norm ∈ [0,1]: 0 = pupil edge, 1 = iris edge
    θ      ∈ [0, 2π)

    Returns float32 array of shape (rows, cols) in [0, 1].
    """
    h, w = gray.shape
    px, py, pr = pupil
    ix, iy, ir = iris

    thetas = np.linspace(0.0, 2.0 * np.pi, cols, endpoint=False)
    cos_t  = np.cos(thetas)
    sin_t  = np.sin(thetas)

    x_pup = px + pr * cos_t   # (cols,)
    y_pup = py + pr * sin_t
    x_iri = ix + ir * cos_t
    y_iri = iy + ir * sin_t

    r_norms = np.linspace(0.0, 1.0, rows)[:, np.newaxis]   # (rows, 1)

    map_x = ((1 - r_norms) * x_pup + r_norms * x_iri).astype(np.float32)
    map_y = ((1 - r_norms) * y_pup + r_norms * y_iri).astype(np.float32)

    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)

    strip = cv2.remap(
        gray.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return (strip / 255.0).astype(np.float32)


def apply_clahe(strip: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE contrast enhancement on float32 strip [0,1] → [0,1]."""
    uint8    = (strip * 255).clip(0, 255).astype(np.uint8)
    clahe    = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(uint8)
    return enhanced.astype(np.float32) / 255.0


def strip_quality_ok(strip: np.ndarray, min_contrast: float = 0.03) -> bool:
    """Reject strips with near-zero contrast (fully occluded iris)."""
    return float(strip.std()) >= min_contrast
