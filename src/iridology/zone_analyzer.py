import numpy as np
from typing import Dict, Tuple

from src.iridology.iridology_zones import (
    RADIAL_ZONES, ORGAN_INFO, IRIS_PATTERNS,
    get_zones, get_risk_level
)


# ──────────────────────────────────────────────
# Core analysis
# ──────────────────────────────────────────────

def compute_error_map(strip: np.ndarray,
                      recon: np.ndarray) -> np.ndarray:
    """
    Per-pixel absolute reconstruction error.
    Both inputs: float32 (64, 512) in [0, 1].
    Returns: float32 (64, 512) error map.
    """
    return np.abs(strip.astype(np.float32) - recon.astype(np.float32))


def analyze_zones(strip: np.ndarray,
                  recon: np.ndarray,
                  eye_side: str = "left") -> Dict:
    """
    Compute mean reconstruction error per iridology organ zone.

    Returns dict of organ_key → {score, risk, name, icon, system, col_range}
    Sorted highest error first.
    """
    err_map   = compute_error_map(strip, recon)
    zones     = get_zones(eye_side)
    results   = {}

    for organ_key, (col_start, col_end) in zones.items():
        zone_err = err_map[:, col_start:col_end]
        score    = float(zone_err.mean())
        risk     = get_risk_level(score)
        info     = ORGAN_INFO.get(organ_key, {})

        results[organ_key] = {
            "score":     round(score, 6),
            "risk":      risk["level"],
            "emoji":     risk["emoji"],
            "color":     risk["color"],
            "name":      info.get("name",   organ_key),
            "icon":      info.get("icon",   "🔬"),
            "system":    info.get("system", "Unknown"),
            "col_range": [col_start, col_end],
        }

    # Sort by score descending (most anomalous first)
    return dict(sorted(results.items(),
                        key=lambda x: x[1]["score"], reverse=True))


def analyze_radial_zones(strip: np.ndarray,
                          recon: np.ndarray) -> Dict:
    """
    Compute mean error per radial depth zone.
    Tells you HOW DEEP the anomaly is in the iris layers.
    """
    err_map = compute_error_map(strip, recon)
    results = {}

    for zone_name, (row_start, row_end) in RADIAL_ZONES.items():
        zone_err = err_map[row_start:row_end, :]
        score    = float(zone_err.mean())
        risk     = get_risk_level(score)
        results[zone_name] = {
            "score":     round(score, 6),
            "risk":      risk["level"],
            "emoji":     risk["emoji"],
            "row_range": [row_start, row_end],
        }

    return results


def detect_pattern(strip: np.ndarray,
                   recon: np.ndarray,
                   zone_scores: Dict) -> str:
    """
    Heuristic pattern detection based on error map characteristics.
    Returns one of the IRIS_PATTERNS keys.
    """
    err_map = compute_error_map(strip, recon)
    h, w    = err_map.shape

    # ── Radii Solaris: high error in radial spokes from centre ──
    # Check if error is high in stomach/intestine zone AND spreads outward
    inner_err = err_map[:h//4, :].mean()
    outer_err = err_map[3*h//4:, :].mean()
    radial_spread = inner_err > 0.06 and outer_err > 0.04

    # ── Arcus Senilis: high error in outermost rows ──
    arcus_err = err_map[int(h*0.80):, :].mean()
    has_arcus = arcus_err > 0.08

    # ── Lymphatic Rosary: periodic high error in outer ring ──
    outer_ring = err_map[int(h*0.70):, :]
    # Check for periodic peaks (rosary beads)
    col_means  = outer_ring.mean(axis=0)
    peaks      = (col_means > col_means.mean() + col_means.std()).sum()
    has_rosary = peaks > 40  # many periodic peaks = rosary pattern

    # ── Stomach ring patterns: error in innermost rows ──
    stomach_err = err_map[:h//5, int(w*0.3):int(w*0.7)].mean()
    has_stomach_anomaly = stomach_err > 0.07

    # ── Determine most likely pattern ──
    top_organs = list(zone_scores.keys())[:3]

    if has_arcus:
        return "arcus_senilis"
    elif has_rosary:
        return "lymphatic_rosary"
    elif radial_spread and any(o in top_organs
                               for o in ["liver", "stomach", "rectum_sigmoid"]):
        return "radii_solaris"
    elif has_stomach_anomaly:
        stomach_score = zone_scores.get("stomach", {}).get("score", 0)
        inner_mean    = err_map[:h//5, :].mean()
        if inner_mean < 0.04:
            return "underactive_stomach"
        else:
            return "overactive_stomach"
    else:
        # Check overall anomaly level
        global_err = err_map.mean()
        if global_err < 0.035:
            return "normal"
        return "healing_lines"


def top_concerns(zone_scores: Dict, n: int = 3) -> list:
    """Return top N organs with highest anomaly scores."""
    high_risk   = [(k, v) for k, v in zone_scores.items()
                   if v["risk"] == "high"]
    medium_risk = [(k, v) for k, v in zone_scores.items()
                   if v["risk"] == "moderate"]
    combined    = (high_risk + medium_risk)[:n]
    return [{"organ_key": k, **v} for k, v in combined]


def full_zone_analysis(strip: np.ndarray,
                        recon: np.ndarray,
                        eye_side: str = "left") -> Dict:
    """
    Complete iridology analysis combining all zone checks.
    Returns a structured result dict ready for the report generator.
    """
    zone_scores   = analyze_zones(strip, recon, eye_side)
    radial_scores = analyze_radial_zones(strip, recon)
    pattern_key   = detect_pattern(strip, recon, zone_scores)
    concerns      = top_concerns(zone_scores)

    # Global stats
    err_map       = compute_error_map(strip, recon)
    global_score  = float(err_map.mean())
    global_risk   = get_risk_level(global_score)

    return {
        "eye_side":      eye_side,
        "global_score":  round(global_score, 6),
        "global_risk":   global_risk,
        "pattern":       pattern_key,
        "zone_scores":   zone_scores,
        "radial_scores": radial_scores,
        "top_concerns":  concerns,
    }
