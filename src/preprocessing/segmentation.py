from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ──────────────────────────────────────────────
# Type alias
# ──────────────────────────────────────────────
Circle = Tuple[int, int, int]   # (cx, cy, r)


# ──────────────────────────────────────────────
# Pupil detection
# ──────────────────────────────────────────────

def detect_pupil(gray: np.ndarray,
                 dp: float = 1.2,
                 param1: int = 80,
                 param2: int = 28) -> Optional[Circle]:
    """
    Detect pupil (dark inner circle) via Hough Circle Transform.
    Returns (cx, cy, r) or None.
    """
    h, w = gray.shape
    blurred  = cv2.GaussianBlur(gray, (7, 7), 1.5)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    min_r = max(5, int(min(h, w) * 0.04))
    max_r = min(h // 2, int(min(h, w) * 0.30))

    circles = cv2.HoughCircles(
        enhanced, cv2.HOUGH_GRADIENT,
        dp=dp, minDist=int(min(h, w) * 0.15),
        param1=param1, param2=param2,
        minRadius=min_r, maxRadius=max_r
    )
    if circles is None:
        return None

    # Pick circle whose centre region is darkest (= most pupil-like)
    best, best_score = None, np.inf
    for c in circles[0]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        cx = int(np.clip(cx, r, w - r))
        cy = int(np.clip(cy, r, h - r))
        patch = gray[max(0, cy - r // 2): cy + r // 2,
                     max(0, cx - r // 2): cx + r // 2]
        if patch.size == 0:
            continue
        score = float(patch.mean())
        if score < best_score:
            best_score = score
            best = (cx, cy, r)
    return best


# ──────────────────────────────────────────────
# Iris detection
# ──────────────────────────────────────────────

def detect_iris(gray: np.ndarray,
                pupil: Circle,
                dp: float = 1.2,
                param1: int = 60,
                param2: int = 22) -> Optional[Circle]:
    """
    Detect outer iris boundary, constrained to be concentric with pupil.
    Returns (cx, cy, r) or None.
    """
    h, w  = gray.shape
    px, py, pr = pupil
    blurred = cv2.GaussianBlur(gray, (9, 9), 2.0)

    min_r = pr + 6
    max_r = int(min(h, w) * 0.52)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=dp, minDist=int(min(h, w) * 0.1),
        param1=param1, param2=param2,
        minRadius=min_r, maxRadius=max_r
    )
    if circles is None:
        return None

    best, best_dist = None, np.inf
    for c in circles[0]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        if r <= pr * 1.3:
            continue
        dist = float(np.hypot(cx - px, cy - py))
        if dist < best_dist:
            best_dist = dist
            best = (cx, cy, r)
    return best


# ──────────────────────────────────────────────
# Eyelid mask
# ──────────────────────────────────────────────

def build_eyelid_mask(gray: np.ndarray, iris: Circle) -> np.ndarray:
    """
    Mask upper/lower eyelid occlusions using edge detection +
    parabolic fitting. Returns binary mask (255 = usable iris tissue).
    """
    h, w = gray.shape
    ix, iy, ir = iris

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (ix, iy), ir, 255, -1)

    band_top    = max(0, iy - ir)
    band_bottom = min(h, iy + ir)
    band        = gray[band_top:band_bottom, :]

    edges = cv2.Canny(cv2.GaussianBlur(band, (5, 5), 1.5), 30, 80)

    upper_y = np.full(w, band_top,    dtype=np.int32)
    lower_y = np.full(w, band_bottom, dtype=np.int32)
    mid     = band.shape[0] // 2

    for x in range(w):
        ys_up = np.where(edges[:mid, x] > 0)[0]
        if len(ys_up):
            upper_y[x] = band_top + ys_up[0]
        ys_dn = np.where(edges[mid:, x] > 0)[0]
        if len(ys_dn):
            lower_y[x] = band_top + mid + ys_dn[-1]

    xs = np.arange(w)
    try:
        yu = np.polyval(np.polyfit(xs, upper_y, 2), xs).astype(int)
        yl = np.polyval(np.polyfit(xs, lower_y, 2), xs).astype(int)
        for x in range(w):
            if yu[x] > 0:
                mask[:max(0, yu[x]), x] = 0
            if yl[x] < h:
                mask[min(h, yl[x]):, x] = 0
    except np.linalg.LinAlgError:
        pass   # keep circular mask if fitting fails

    return mask


# ──────────────────────────────────────────────
# Annular mask
# ──────────────────────────────────────────────

def build_annular_mask(shape: Tuple[int, int],
                        pupil: Circle,
                        iris: Circle,
                        eyelid_mask: np.ndarray) -> np.ndarray:
    """iris disk − pupil disk, intersected with eyelid mask."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (iris[0],  iris[1]),  iris[2],  255, -1)
    cv2.circle(mask, (pupil[0], pupil[1]), pupil[2],   0, -1)
    return cv2.bitwise_and(mask, eyelid_mask)


# ──────────────────────────────────────────────
# High-level wrapper
# ──────────────────────────────────────────────

def segment_iris(image_path: Path) -> Optional[dict]:
    """
    Full segmentation of one image.
    Returns dict with pupil, iris circles, mask path; or None on failure.
    """
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None

    pupil = detect_pupil(gray)
    if pupil is None:
        return None

    iris = detect_iris(gray, pupil)
    if iris is None:
        return None

    eyelid_mask  = build_eyelid_mask(gray, iris)
    annular_mask = build_annular_mask(gray.shape, pupil, iris, eyelid_mask)

    return {
        "gray":         gray,
        "pupil":        pupil,
        "iris":         iris,
        "annular_mask": annular_mask,
    }
