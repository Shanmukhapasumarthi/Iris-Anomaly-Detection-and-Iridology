"""
health_report.py
Generates a structured iridology health report from zone analysis results.
Produces both a dict (for API) and a formatted text summary.
"""

from typing import Dict
from iridology_zones import IRIS_PATTERNS


DISCLAIMER = (
    "⚠️  IMPORTANT DISCLAIMER: This analysis is based on iridology principles "
    "for research purposes only. It is NOT a medical diagnosis. "
    "Always consult a qualified medical professional for health concerns."
)


def generate_report(analysis: Dict,
                    anomaly_score: float,
                    threshold: float,
                    mean_error: float,
                    ssim_score: float,
                    filename: str = "iris") -> Dict:
    """
    Build a complete health report dict from zone analysis + model scores.

    Parameters
    ----------
    analysis      : output of full_zone_analysis()
    anomaly_score : ViT-MAE anomaly score
    threshold     : current threshold
    mean_error    : mean reconstruction error
    ssim_score    : SSIM between strip and reconstruction
    filename      : source image filename

    Returns
    -------
    Structured report dict suitable for JSON response.
    """
    pattern_key  = analysis["pattern"]
    pattern_info = IRIS_PATTERNS.get(pattern_key, IRIS_PATTERNS["normal"])
    global_risk  = analysis["global_risk"]
    concerns     = analysis["top_concerns"]

    # Build organ summary list
    organ_summary = []
    for key, data in analysis["zone_scores"].items():
        organ_summary.append({
            "organ_key":  key,
            "organ_name": data["name"],
            "icon":       data["icon"],
            "system":     data["system"],
            "score":      data["score"],
            "risk":       data["risk"],
            "risk_emoji": data["emoji"],
        })

    # Build radial summary
    radial_summary = []
    radial_labels  = {
        "stomach_intestine": "Stomach / Intestine Ring",
        "organ_core":        "Core Organ Zone",
        "musculoskeletal":   "Musculoskeletal Zone",
        "lymphatic_skin":    "Lymphatic / Skin Zone",
    }
    for key, data in analysis["radial_scores"].items():
        radial_summary.append({
            "zone":  radial_labels.get(key, key),
            "score": data["score"],
            "risk":  data["risk"],
            "emoji": data["emoji"],
        })

    # Primary concern organs
    primary_concerns = []
    for c in concerns:
        primary_concerns.append({
            "organ":      c["name"],
            "icon":       c["icon"],
            "system":     c["system"],
            "risk_score": c["score"],
            "risk_level": c["risk"],
        })

    report = {
        "filename":     filename,
        "eye_side":     analysis["eye_side"],
        "disclaimer":   DISCLAIMER,

        # Overall verdict
        "overall": {
            "verdict":       "anomalous" if anomaly_score >= threshold else "normal",
            "verdict_emoji": "🔴" if anomaly_score >= threshold else "🟢",
            "anomaly_score": round(anomaly_score, 6),
            "threshold":     round(threshold, 6),
            "global_error":  round(analysis["global_score"], 6),
            "risk_level":    global_risk["level"],
            "risk_emoji":    global_risk["emoji"],
        },

        # Reconstruction quality
        "reconstruction_quality": {
            "mean_error":   round(mean_error, 6),
            "ssim":         round(ssim_score, 6),
            "interpretation": _interpret_reconstruction(mean_error, ssim_score),
        },

        # Detected iris pattern
        "iris_pattern": {
            "key":         pattern_key,
            "name":        pattern_info["name"],
            "description": pattern_info["description"],
            "indication":  pattern_info["indication"],
            "organs":      pattern_info["organs"],
        },

        # Top concerns
        "primary_concerns": primary_concerns,

        # Full organ zone breakdown
        "organ_zones":  organ_summary,

        # Radial depth zones
        "radial_zones": radial_summary,

        # Summary text
        "summary_text": _build_summary(analysis, pattern_info,
                                        primary_concerns, anomaly_score,
                                        threshold),
    }

    return report


def _interpret_reconstruction(mean_error: float, ssim: float) -> str:
    """Human-readable reconstruction quality interpretation."""
    if mean_error < 0.03 and ssim > 0.85:
        return "Excellent — iris pattern closely matches trained normal patterns"
    elif mean_error < 0.06 and ssim > 0.70:
        return "Good — minor deviations from normal iris patterns detected"
    elif mean_error < 0.10 and ssim > 0.55:
        return "Moderate — notable iris pattern irregularities present"
    else:
        return "Poor — significant iris pattern anomalies detected"


def _build_summary(analysis: Dict,
                    pattern_info: Dict,
                    concerns: list,
                    anomaly_score: float,
                    threshold: float) -> str:
    """Generate a plain-text summary of the health report."""
    lines = []
    verdict = "ANOMALOUS" if anomaly_score >= threshold else "NORMAL"
    lines.append(f"IRIS HEALTH ANALYSIS — {analysis['eye_side'].upper()} EYE")
    lines.append(f"Overall Status   : {verdict}")
    lines.append(f"Anomaly Score    : {anomaly_score:.5f} (threshold: {threshold:.5f})")
    lines.append(f"Pattern Detected : {pattern_info['name']}")
    lines.append(f"Indication       : {pattern_info['indication']}")
    lines.append("")

    if concerns:
        lines.append("Areas of Concern:")
        for c in concerns:
            lines.append(f"  {c['icon']} {c['organ']} ({c['system']}) "
                         f"— risk score: {c['risk_score']:.4f} [{c['risk_level'].upper()}]")
    else:
        lines.append("No significant organ zone anomalies detected.")

    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)