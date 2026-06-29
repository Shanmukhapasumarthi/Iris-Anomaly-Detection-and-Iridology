#!/usr/bin/env python
# coding: utf-8

# # Notebook 02 — Advanced Iris Segmentation, Polar Normalization & Anomaly Detection
# Fine-grained iris segmentation with 30° sector-wise anomaly scoring and reconstruction analysis.

# ─────────────────────────────────────────────
# CELL 1 — Imports & Setup
# ─────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import ndimage
from skimage import filters, morphology, feature, restoration
from skimage.draw import disk

try:
    from IPython import get_ipython
    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "inline")
    else:
        matplotlib.use("Agg")
except Exception:
    matplotlib.use("Agg")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.family"] = "monospace"


def _show_plot():
    if matplotlib.get_backend().lower() == "agg":
        plt.close("all")
    else:
        plt.show()


# ─────────────────────────────────────────────
# CELL 2 — Load Image
# ─────────────────────────────────────────────

DATA_ROOT = Path('raw')
images = list(DATA_ROOT.rglob('*.jpg')) + list(DATA_ROOT.rglob('*.png'))
print(f'Found {len(images)} images')

img_path = images[0]
gray_raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
print(f'Image: {img_path.name}  shape={gray_raw.shape}')

plt.imshow(gray_raw, cmap="gray")
plt.title("Original")
plt.axis("off")
_show_plot()


# ─────────────────────────────────────────────
# CELL 3 — Fine-Tuned Pre-processing
# Uses multi-stage denoising + adaptive contrast for better circle detection
# ─────────────────────────────────────────────

def fine_tune_preprocess(gray: np.ndarray) -> dict:
    """
    Multi-stage preprocessing pipeline for robust iris segmentation.

    Stages
    ------
    1. Non-local means denoising         — removes sensor noise
    2. Bilateral filter                  — edge-preserving smoothing
    3. CLAHE                             — local contrast enhancement
    4. Unsharp masking                   — sharpens limbus/pupil edges
    5. Specular-highlight inpainting     — removes bright corneal reflections
    """
    # 1. NLM denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)

    # 2. Bilateral smoothing
    bilateral = cv2.bilateralFilter(denoised, d=9, sigmaColor=40, sigmaSpace=40)

    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)

    # 4. Unsharp masking  (amount=1.5, radius=2)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharp = cv2.addWeighted(enhanced, 2.5, blurred, -1.5, 0)

    # 5. Specular-highlight inpainting  (bright blobs > 240)
    _, spec_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    spec_mask = cv2.dilate(spec_mask, np.ones((5, 5), np.uint8), iterations=2)
    inpainted = cv2.inpaint(sharp, spec_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return {
        "denoised":  denoised,
        "bilateral": bilateral,
        "enhanced":  enhanced,
        "sharp":     sharp,
        "inpainted": inpainted,
        "spec_mask": spec_mask,
    }


pp = fine_tune_preprocess(gray_raw)
gray = pp["inpainted"]   # ← use this for all downstream steps

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
titles = ["Original", "NLM Denoised", "Bilateral", "CLAHE Enhanced", "Unsharp Masked", "Specular Inpainted"]
imgs   = [gray_raw, pp["denoised"], pp["bilateral"], pp["enhanced"], pp["sharp"], pp["inpainted"]]
for ax, title, im in zip(axes.flatten(), titles, imgs):
    ax.imshow(im, cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
plt.suptitle("Fine-Tuned Pre-processing Pipeline", fontsize=12, fontweight="bold")
plt.tight_layout()
_show_plot()


# ─────────────────────────────────────────────
# CELL 4 — Refined Pupil Detection
# Combines Hough + intensity-based validation
# ─────────────────────────────────────────────

def detect_pupil_refined(gray: np.ndarray) -> tuple[int, int, int]:
    """
    Robust pupil detection:
    1. Dark-region masking (pupils are darkest region)
    2. Canny edge detection on blurred image
    3. Multi-scale Hough circle search
    4. Candidate validation by mean intensity inside circle
    """
    h, w = gray.shape

    # Dark region mask  (bottom 30% intensity)
    thresh = np.percentile(gray, 30)
    dark_mask = (gray < thresh).astype(np.uint8) * 255
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE,
                                  np.ones((7, 7), np.uint8), iterations=2)

    # Blur & edge map
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    edges   = cv2.Canny(blurred, 20, 70)
    edges   = cv2.bitwise_and(edges, dark_mask)

    # Hough — small radius range for pupil
    rmin = int(min(h, w) * 0.04)
    rmax = int(min(h, w) * 0.20)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1.5,
        minDist=min(h, w) // 4,
        param1=60,
        param2=0.55,
        minRadius=rmin,
        maxRadius=rmax,
    )

    best = None
    best_score = np.inf

    if circles is not None:
        for cx, cy, r in circles[0]:
            cx, cy, r = int(cx), int(cy), int(r)
            # Mask pixels inside candidate circle
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            mean_int = cv2.mean(gray, mask=mask)[0]
            # Good pupils are dark; score = mean intensity
            if mean_int < best_score:
                best_score = mean_int
                best = (cx, cy, r)

    if best is None:
        # Fallback: centroid of largest dark blob
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                area = cv2.contourArea(c)
                r    = int(np.sqrt(area / np.pi))
                best = (cx, cy, r)
        else:
            best = (w // 2, h // 2, rmin)

    return best


# ─────────────────────────────────────────────
# CELL 5 — Refined Iris Detection
# Uses limbus-aware Hough with specular suppression
# ─────────────────────────────────────────────

def detect_iris_refined(gray: np.ndarray, pupil: tuple) -> tuple[int, int, int]:
    """
    Iris outer boundary (limbus) detection:
    1. Suppress pupil region to prevent false inner circles
    2. Top-hat transform to reveal limbus ring
    3. Multi-param Hough search in physiological radius range
    4. Validate: iris center ≈ pupil center, iris_r >> pupil_r
    """
    h, w = gray.shape
    pcx, pcy, pr = pupil

    # Suppress pupil + specular highlights
    masked = gray.copy()
    cv2.circle(masked, (pcx, pcy), int(pr * 1.1), int(np.median(gray)), -1)

    # Top-hat reveals bright limbus ring on dark background
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    tophat  = cv2.morphologyEx(masked, cv2.MORPH_TOPHAT, kernel)
    blurred = cv2.GaussianBlur(tophat, (5, 5), 2)

    # Hough — iris is larger than pupil
    rmin = int(pr * 1.8)
    rmax = int(min(h, w) * 0.55)
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(masked, (9, 9), 3),
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=min(h, w) // 3,
        param1=40,
        param2=25,
        minRadius=rmin,
        maxRadius=rmax,
    )

    best = None
    best_dist = np.inf

    if circles is not None:
        for cx, cy, r in circles[0]:
            cx, cy, r = int(cx), int(cy), int(r)
            dist = np.hypot(cx - pcx, cy - pcy)
            # Prefer circles centered near pupil
            if dist < r * 0.4 and dist < best_dist:
                best_dist = dist
                best = (cx, cy, r)

    if best is None:
        # Fallback: concentric with pupil, r = 2.8× pupil
        best = (pcx, pcy, int(pr * 2.8))

    return best


# ─────────────────────────────────────────────
# CELL 6 — Eyelid & Noise Masks (fine-tuned)
# ─────────────────────────────────────────────

def build_eyelid_mask_refined(gray: np.ndarray, iris: tuple) -> np.ndarray:
    """
    Parabolic eyelid exclusion mask:
    - Fit upper & lower parabolas to detected eyelid edges
    - Mark pixels between the parabolas as valid (=255)
    Fallback: exclude top-15% and bottom-15% of iris disk.
    """
    icx, icy, ir = iris
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # ROI around iris
    x1 = max(0, icx - ir)
    y1 = max(0, icy - ir)
    x2 = min(w, icx + ir)
    y2 = min(h, icy + ir)
    roi = gray[y1:y2, x1:x2]

    # Horizontal Sobel to detect horizontal eyelid edges
    sobel_h = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=5)
    sobel_h = np.abs(sobel_h).astype(np.uint8)
    _, edge_bin = cv2.threshold(sobel_h, 30, 255, cv2.THRESH_BINARY)
    edge_bin = cv2.dilate(edge_bin, np.ones((3, 3), np.uint8))

    # Default parabola exclusion if edge fitting fails
    # Upper lid: exclude top ir*0.15 of iris; Lower lid: bottom ir*0.15
    cv2.ellipse(mask, (icx, icy), (ir, ir), 0, 0, 360, 255, -1)
    # Upper eyelid band
    pts_up = np.array([
        [icx - ir, icy - int(ir * 0.20)],
        [icx,      icy - int(ir * 0.85)],
        [icx + ir, icy - int(ir * 0.20)],
    ], dtype=np.int32)
    hull_up = cv2.convexHull(pts_up)
    exclude_top = np.zeros_like(mask)
    cv2.fillConvexPoly(exclude_top, pts_up, 255)
    # Lower eyelid band
    pts_dn = np.array([
        [icx - ir, icy + int(ir * 0.20)],
        [icx,      icy + int(ir * 0.80)],
        [icx + ir, icy + int(ir * 0.20)],
    ], dtype=np.int32)
    exclude_bot = np.zeros_like(mask)
    cv2.fillConvexPoly(exclude_bot, pts_dn, 255)

    mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude_top))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude_bot))

    # Morphological cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             np.ones((5, 5), np.uint8), iterations=1)
    return mask


def build_annular_mask(shape, pupil, iris, eyelid_mask):
    """Annular mask: iris disk minus pupil disk, intersected with eyelid mask."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (iris[0], iris[1]),   iris[2],   255, -1)
    cv2.circle(mask, (pupil[0], pupil[1]), pupil[2],  0,   -1)
    return cv2.bitwise_and(mask, eyelid_mask)


# ─────────────────────────────────────────────
# CELL 7 — Run Segmentation
# ─────────────────────────────────────────────

pupil  = detect_pupil_refined(gray)
iris_c = detect_iris_refined(gray, pupil)
print(f'Pupil : cx={pupil[0]}, cy={pupil[1]}, r={pupil[2]}')
print(f'Iris  : cx={iris_c[0]}, cy={iris_c[1]}, r={iris_c[2]}')

eyelid_mask  = build_eyelid_mask_refined(gray, iris_c)
annular_mask = build_annular_mask(gray.shape, pupil, iris_c, eyelid_mask)

vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
cv2.circle(vis, (pupil[0],  pupil[1]),  pupil[2],   (80, 80, 255), 2)
cv2.circle(vis, (iris_c[0], iris_c[1]), iris_c[2],  (80, 220, 80), 2)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
axes[0].set_title('Detected Boundaries'); axes[0].axis('off')
axes[1].imshow(eyelid_mask, cmap='gray')
axes[1].set_title('Eyelid Mask'); axes[1].axis('off')
axes[2].imshow(annular_mask, cmap='gray')
axes[2].set_title('Annular Iris Mask'); axes[2].axis('off')
plt.tight_layout()
_show_plot()


# ─────────────────────────────────────────────
# CELL 8 — Rubber-Sheet Normalization (high resolution)
# ─────────────────────────────────────────────

def rubber_sheet_normalize(
    gray: np.ndarray,
    pupil: tuple,
    iris:  tuple,
    radial_res: int = 64,
    angular_res: int = 512,
) -> np.ndarray:
    """
    Daugman rubber-sheet model.
    Maps (r, θ) → Cartesian  with bilinear interpolation.
    Returns float32 strip [0,1] of shape (radial_res, angular_res).
    """
    pcx, pcy, pr = pupil
    icx, icy, ir = iris
    theta = np.linspace(0, 2 * np.pi, angular_res, endpoint=False)
    r_norm = np.linspace(0, 1, radial_res)

    strip = np.zeros((radial_res, angular_res), dtype=np.float32)
    for ri, rn in enumerate(r_norm):
        # Pupil boundary point
        px = pcx + pr * np.cos(theta)
        py = pcy + pr * np.sin(theta)
        # Iris boundary point
        ix = icx + ir * np.cos(theta)
        iy = icy + ir * np.sin(theta)
        # Interpolated point
        x = (1 - rn) * px + rn * ix
        y = (1 - rn) * py + rn * iy
        # Bilinear sample
        x0 = np.clip(np.floor(x).astype(int), 0, gray.shape[1] - 1)
        x1 = np.clip(x0 + 1,                  0, gray.shape[1] - 1)
        y0 = np.clip(np.floor(y).astype(int), 0, gray.shape[0] - 1)
        y1 = np.clip(y0 + 1,                  0, gray.shape[0] - 1)
        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)
        vals = (wa * gray[y0, x0] + wb * gray[y0, x1] +
                wc * gray[y1, x0] + wd * gray[y1, x1])
        strip[ri, :] = vals / 255.0

    return strip


def apply_clahe_strip(strip: np.ndarray) -> np.ndarray:
    u8 = (strip * 255).clip(0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 4))
    return clahe.apply(u8).astype(np.float32) / 255.0


RADIAL_RES  = 64
ANGULAR_RES = 512   # 512 columns → each 30° sector = 512/12 ≈ 42 cols

strip       = rubber_sheet_normalize(gray, pupil, iris_c,
                                     radial_res=RADIAL_RES,
                                     angular_res=ANGULAR_RES)
strip_clahe = apply_clahe_strip(strip)

print(f'Strip shape : {strip.shape}')
print(f'Strip range : [{strip.min():.3f}, {strip.max():.3f}]')
print(f'Strip std   : {strip.std():.4f}')

fig, axes = plt.subplots(2, 1, figsize=(15, 6))
axes[0].imshow(strip,       cmap='gray', aspect='auto')
axes[0].set_title('Normalised Iris Strip (raw)')
axes[0].set_xlabel('Angular θ (0–360°)'); axes[0].set_ylabel('Radial r')
axes[1].imshow(strip_clahe, cmap='gray', aspect='auto')
axes[1].set_title('After CLAHE Enhancement')
axes[1].set_xlabel('Angular θ (0–360°)'); axes[1].set_ylabel('Radial r')

# 30° sector grid lines
for deg in range(0, 360, 30):
    col = int(deg / 360 * ANGULAR_RES)
    axes[0].axvline(col, color='cyan', lw=0.5, alpha=0.6)
    axes[1].axvline(col, color='cyan', lw=0.5, alpha=0.6)
    axes[1].text(col + 2, 5, f'{deg}°', color='cyan', fontsize=6, va='top')

plt.tight_layout()
_show_plot()


# ─────────────────────────────────────────────
# CELL 9 — Sector-Wise Anomaly Scoring (30° sectors)
# Method: per-sector z-score → reconstruction error proxy
# ─────────────────────────────────────────────

NUM_SECTORS  = 12          # 360° / 30° = 12 sectors
SECTOR_DEG   = 30

def compute_sector_stats(strip: np.ndarray, num_sectors: int = 12):
    """
    Divides the angular strip into `num_sectors` equal sectors.
    For each sector:
      - mean, std, min, max intensity
      - entropy (texture complexity)
      - reconstruction_score: how well the sector matches the global mean strip
      - anomaly_score: normalised z-score of sector mean vs all-sector means
    Returns a list of dicts (one per sector).
    """
    _, ncols = strip.shape
    cols_per_sector = ncols // num_sectors
    global_mean_col = strip.mean(axis=1)   # radial profile

    sector_means = []
    for s in range(num_sectors):
        c0 = s * cols_per_sector
        c1 = c0 + cols_per_sector
        sector_means.append(strip[:, c0:c1].mean())

    mu_all = np.mean(sector_means)
    sd_all = np.std(sector_means) + 1e-8

    results = []
    for s in range(num_sectors):
        angle_start = s * SECTOR_DEG
        angle_end   = angle_start + SECTOR_DEG
        c0 = s * cols_per_sector
        c1 = c0 + cols_per_sector
        patch = strip[:, c0:c1]

        # Reconstruction score: MSE of sector radial profile vs global
        sector_profile  = patch.mean(axis=1)
        recon_error     = float(np.mean((sector_profile - global_mean_col) ** 2))

        # Entropy
        hist, _ = np.histogram(patch, bins=32, range=(0, 1), density=True)
        hist    = hist + 1e-9
        entropy = float(-np.sum(hist * np.log2(hist)))

        # Anomaly z-score (unsigned)
        z_score = float(abs(sector_means[s] - mu_all) / sd_all)

        results.append({
            "sector":          s,
            "angle_start":     angle_start,
            "angle_end":       angle_end,
            "angle_label":     f"{angle_start}°–{angle_end}°",
            "mean_intensity":  float(sector_means[s]),
            "std_intensity":   float(patch.std()),
            "entropy":         entropy,
            "recon_error":     recon_error,
            "anomaly_score":   z_score,
        })

    return results


sector_stats = compute_sector_stats(strip_clahe, NUM_SECTORS)

# Pretty-print table
print(f"\n{'Sector':>6} {'Angle':>10} {'Mean':>7} {'Std':>7} "
      f"{'Entropy':>8} {'ReconErr':>10} {'AnomalyZ':>10}")
print("─" * 68)
for s in sector_stats:
    print(f"{s['sector']:>6} {s['angle_label']:>10} "
          f"{s['mean_intensity']:>7.4f} {s['std_intensity']:>7.4f} "
          f"{s['entropy']:>8.4f} {s['recon_error']:>10.6f} "
          f"{s['anomaly_score']:>10.4f}")


# ─────────────────────────────────────────────
# CELL 10 — Full Iris Visualisation with Anomaly Overlay
# ─────────────────────────────────────────────

def draw_sector_anomaly_overlay(
    gray:         np.ndarray,
    pupil:        tuple,
    iris_c:       tuple,
    sector_stats: list,
    alpha:        float = 0.45,
) -> np.ndarray:
    """
    Draws 30° sector wedges on the iris, color-coded by anomaly_score.
    Green = normal, Yellow = mild anomaly, Red = strong anomaly.
    """
    h, w = gray.shape
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    layer   = np.zeros((h, w, 3), dtype=np.float32)

    pcx, pcy, pr = pupil
    icx, icy, ir = iris_c

    # Normalise anomaly scores to [0, 1]
    scores = [s["anomaly_score"] for s in sector_stats]
    max_z  = max(scores) + 1e-8

    cmap = plt.cm.RdYlGn_r   # green=normal, red=anomaly

    for s in sector_stats:
        norm_score = s["anomaly_score"] / max_z
        r, g, b, _ = cmap(norm_score)
        color_bgr   = (b * 255, g * 255, r * 255)   # OpenCV BGR

        a_start = s["angle_start"]
        a_end   = s["angle_end"]

        # Filled wedge on the outer iris disk
        cv2.ellipse(layer, (icx, icy), (ir, ir),
                    0, a_start, a_end, color_bgr, -1)
        # Subtract pupil region
        cv2.ellipse(layer, (pcx, pcy), (pr, pr),
                    0, a_start, a_end, (0, 0, 0), -1)

    result = cv2.addWeighted(overlay, 1.0, layer, alpha, 0).astype(np.uint8)

    # Draw sector lines & labels
    for s in sector_stats:
        ang_mid_rad = np.radians(s["angle_start"] + SECTOR_DEG / 2)
        mid_r = (pr + ir) / 2
        lx = int(icx + mid_r * np.cos(ang_mid_rad))
        ly = int(icy + mid_r * np.sin(ang_mid_rad))
        txt = f"Z={s['anomaly_score']:.2f}"
        cv2.putText(result, txt, (lx - 18, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

        # Radial divider lines
        ang_rad = np.radians(s["angle_start"])
        x0 = int(icx + pr * np.cos(ang_rad))
        y0 = int(icy + pr * np.sin(ang_rad))
        x1 = int(icx + ir * np.cos(ang_rad))
        y1 = int(icy + ir * np.sin(ang_rad))
        cv2.line(result, (x0, y0), (x1, y1), (200, 200, 200), 1)

    cv2.circle(result, (icx, icy), ir,  (200, 200, 200), 1)
    cv2.circle(result, (pcx, pcy), pr,  (200, 200, 200), 1)

    return result


overlay_img = draw_sector_anomaly_overlay(gray, pupil, iris_c, sector_stats)

# ─── Build comprehensive figure ───
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#0a0a0f')

# ① Full iris overlay (large)
ax_main = fig.add_axes([0.01, 0.40, 0.45, 0.58])
ax_main.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
ax_main.set_title('Full Iris — Sector Anomaly Map (30° sectors)',
                  color='white', fontsize=11, pad=8)
ax_main.axis('off')

# Colorbar legend
sm = ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=Normalize(0, 1))
sm.set_array([])
cbar_ax = fig.add_axes([0.47, 0.46, 0.012, 0.44])
cb = plt.colorbar(sm, cax=cbar_ax)
cb.set_label('Normalised Anomaly Score', color='white', fontsize=8)
cb.ax.yaxis.set_tick_params(color='white')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='white', fontsize=7)

# ② Normalised strip
ax_strip = fig.add_axes([0.50, 0.72, 0.49, 0.24])
ax_strip.imshow(strip_clahe, cmap='gray', aspect='auto')
ax_strip.set_facecolor('#0a0a0f')
ax_strip.set_title('Normalised Iris Strip (CLAHE)', color='white', fontsize=9)
ax_strip.set_xlabel('Angular θ  (0° → 360°)', color='white', fontsize=7)
ax_strip.set_ylabel('Radial r', color='white', fontsize=7)
ax_strip.tick_params(colors='white', labelsize=6)
for deg in range(0, 360, 30):
    col = int(deg / 360 * ANGULAR_RES)
    ax_strip.axvline(col, color='cyan', lw=0.6, alpha=0.7)
    ax_strip.text(col + 2, 3, f'{deg}°', color='cyan', fontsize=5, va='top')

# ③ Anomaly score bar chart
ax_anom = fig.add_axes([0.50, 0.40, 0.49, 0.28])
angles  = [s["angle_label"] for s in sector_stats]
z_vals  = [s["anomaly_score"] for s in sector_stats]
colors  = [plt.cm.RdYlGn_r(v / (max(z_vals) + 1e-8)) for v in z_vals]
bars    = ax_anom.bar(range(NUM_SECTORS), z_vals, color=colors, edgecolor='#333', linewidth=0.5)
ax_anom.set_xticks(range(NUM_SECTORS))
ax_anom.set_xticklabels([f"{s['angle_start']}°" for s in sector_stats], fontsize=6, color='white')
ax_anom.set_ylabel('Anomaly Z-Score', color='white', fontsize=8)
ax_anom.set_title('Per-Sector Anomaly Scores', color='white', fontsize=9)
ax_anom.set_facecolor('#12121a')
ax_anom.tick_params(colors='white', labelsize=6)
ax_anom.spines[:].set_color('#333')
ax_anom.axhline(1.0, color='yellow', lw=0.8, ls='--', alpha=0.7, label='z=1 threshold')
ax_anom.axhline(2.0, color='red',    lw=0.8, ls='--', alpha=0.7, label='z=2 threshold')
ax_anom.legend(fontsize=6, labelcolor='white', facecolor='#1a1a2e')

# ④ Reconstruction error
ax_recon = fig.add_axes([0.01, 0.02, 0.48, 0.35])
recon_vals = [s["recon_error"] for s in sector_stats]
recon_colors = [plt.cm.plasma(v / (max(recon_vals) + 1e-8)) for v in recon_vals]
ax_recon.bar(range(NUM_SECTORS), recon_vals, color=recon_colors, edgecolor='#333', linewidth=0.5)
ax_recon.set_xticks(range(NUM_SECTORS))
ax_recon.set_xticklabels([f"{s['angle_start']}°\n{s['angle_label'].split('–')[1]}"
                           for s in sector_stats], fontsize=5.5, color='white')
ax_recon.set_ylabel('Reconstruction MSE', color='white', fontsize=8)
ax_recon.set_title('Per-Sector Reconstruction Error\n(deviation from global radial profile)',
                   color='white', fontsize=9)
ax_recon.set_facecolor('#12121a')
ax_recon.tick_params(colors='white', labelsize=6)
ax_recon.spines[:].set_color('#333')

# ⑤ Sector summary table
ax_tbl = fig.add_axes([0.50, 0.02, 0.49, 0.35])
ax_tbl.set_facecolor('#0a0a0f')
ax_tbl.axis('off')
col_labels = ["Sector", "Angle", "Mean", "Std", "Entropy", "ReconErr", "AnomalyZ"]
rows = [[
    str(s["sector"]),
    s["angle_label"],
    f"{s['mean_intensity']:.4f}",
    f"{s['std_intensity']:.4f}",
    f"{s['entropy']:.3f}",
    f"{s['recon_error']:.5f}",
    f"{s['anomaly_score']:.3f}",
] for s in sector_stats]
tbl = ax_tbl.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#333')
    if r == 0:
        cell.set_facecolor('#1a1a4a')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        z = sector_stats[r - 1]["anomaly_score"] / (max(z_vals) + 1e-8)
        cell.set_facecolor((*plt.cm.RdYlGn_r(z)[:3], 0.25))
        cell.set_text_props(color='white')

ax_tbl.set_title('Sector-by-Sector Analysis Table',
                 color='white', fontsize=9, pad=4)

plt.suptitle("Iris Anomaly Analysis — 30° Sector Decomposition",
             color='white', fontsize=14, fontweight='bold', y=0.99)
_show_plot()


# ─────────────────────────────────────────────
# CELL 11 — Final Summary
# ─────────────────────────────────────────────

max_anom = max(sector_stats, key=lambda s: s["anomaly_score"])
max_recon = max(sector_stats, key=lambda s: s["recon_error"])
print("\n" + "═" * 60)
print("  IRIS ANOMALY ANALYSIS SUMMARY")
print("═" * 60)
print(f"  Image              : {img_path.name}")
print(f"  Pupil              : center=({pupil[0]},{pupil[1]})  r={pupil[2]}px")
print(f"  Iris               : center=({iris_c[0]},{iris_c[1]})  r={iris_c[2]}px")
print(f"  Strip shape        : {strip_clahe.shape}  (radial × angular)")
print(f"  Sectors            : {NUM_SECTORS}  ({SECTOR_DEG}° each)")
print("─" * 60)
print(f"  Highest anomaly    : Sector {max_anom['sector']}  "
      f"({max_anom['angle_label']})  Z={max_anom['anomaly_score']:.4f}")
print(f"  Highest recon err  : Sector {max_recon['sector']}  "
      f"({max_recon['angle_label']})  MSE={max_recon['recon_error']:.6f}")
print("─" * 60)
print(f"  Global strip mean  : {strip_clahe.mean():.4f}")
print(f"  Global strip std   : {strip_clahe.std():.4f}")
print("═" * 60)