# ──────────────────────────────────────────────
# Radial zones (rows 0-63)
# Inner zones = closer to pupil = digestive core
# Outer zones = closer to sclera = systemic/lymphatic
# ──────────────────────────────────────────────

RADIAL_ZONES = {
    "stomach_intestine": (0,  14),   # innermost ring
    "organ_core":        (14, 32),   # heart, liver, kidney etc.
    "musculoskeletal":   (32, 48),   # bones, muscles, spine
    "lymphatic_skin":    (48, 64),   # outermost — lymph, skin
}

# ──────────────────────────────────────────────
# Angular zones (cols 0-511) — RIGHT eye
# Based on iridology clock (12 o'clock = top = col ~256)
# ──────────────────────────────────────────────
# Clock positions mapped to strip columns:
#   12 o'clock → col 256  (top of iris)
#    3 o'clock → col 0    (right side)
#    6 o'clock → col 128  (bottom)
#    9 o'clock → col 384  (left side)

ANGULAR_ZONES_RIGHT = {
    "brain_pineal":      (224, 288),   # 12 o'clock
    "hypothalamus":      (200, 224),
    "eye_sight":         (176, 200),   # 11 o'clock area
    "ear":               (152, 176),
    "nose_sinus":        (128, 152),   # 10 o'clock area
    "thyroid_trachea":   (96,  128),   # 9-10 o'clock
    "bronchus_lung":     (64,  96),    # 8-9 o'clock
    "heart":             (32,  64),    # 7-8 o'clock
    "spleen":            (288, 320),   # 1 o'clock
    "pancreas":          (320, 352),   # 2 o'clock
    "liver":             (352, 416),   # 2-4 o'clock
    "kidney_right":      (416, 448),   # 4-5 o'clock
    "adrenal":           (448, 480),   # 5 o'clock
    "rectum_sigmoid":    (480, 512),   # 6 o'clock right
    "stomach":           (0,   32),    # 6 o'clock left
}

# LEFT eye — horizontally mirrored
ANGULAR_ZONES_LEFT = {
    "brain_pineal":      (224, 288),
    "hypothalamus":      (288, 312),
    "eye_sight":         (312, 336),
    "ear":               (336, 360),
    "nose_sinus":        (360, 384),
    "thyroid_trachea":   (384, 416),
    "bronchus_lung":     (416, 448),
    "heart":             (448, 512),
    "spleen":            (160, 224),
    "pancreas":          (128, 160),
    "liver":             (64,  128),
    "kidney_left":       (32,  64),
    "adrenal":           (0,   32),
    "rectum_sigmoid":    (480, 512),
    "stomach":           (200, 256),
}

# ──────────────────────────────────────────────
# Organ display names and icons
# ──────────────────────────────────────────────

ORGAN_INFO = {
    "brain_pineal":    {"name": "Brain / Pineal",     "icon": "🧠", "system": "Nervous"},
    "hypothalamus":    {"name": "Hypothalamus",        "icon": "🧠", "system": "Nervous"},
    "eye_sight":       {"name": "Eye & Sight",         "icon": "👁",  "system": "Sensory"},
    "ear":             {"name": "Ear",                 "icon": "👂", "system": "Sensory"},
    "nose_sinus":      {"name": "Nose / Sinus",        "icon": "👃", "system": "Respiratory"},
    "thyroid_trachea": {"name": "Thyroid / Trachea",   "icon": "🦋", "system": "Endocrine"},
    "bronchus_lung":   {"name": "Bronchus / Lung",     "icon": "🫁", "system": "Respiratory"},
    "heart":           {"name": "Heart",               "icon": "❤️",  "system": "Cardiovascular"},
    "spleen":          {"name": "Spleen",              "icon": "🫀", "system": "Immune"},
    "pancreas":        {"name": "Pancreas",            "icon": "🫀", "system": "Digestive"},
    "liver":           {"name": "Liver",               "icon": "🫀", "system": "Digestive"},
    "kidney_right":    {"name": "Kidney (Right)",      "icon": "🫘", "system": "Urinary"},
    "kidney_left":     {"name": "Kidney (Left)",       "icon": "🫘", "system": "Urinary"},
    "adrenal":         {"name": "Adrenal Gland",       "icon": "⚡",  "system": "Endocrine"},
    "rectum_sigmoid":  {"name": "Rectum / Sigmoid",    "icon": "🔄", "system": "Digestive"},
    "stomach":         {"name": "Stomach",             "icon": "🫃", "system": "Digestive"},
}

# ──────────────────────────────────────────────
# Iris patterns (from iridology literature)
# ──────────────────────────────────────────────

IRIS_PATTERNS = {
    "radii_solaris": {
        "name":        "Radii Solaris",
        "description": "Dark spoke-like lines radiating from pupil outward",
        "indication":  "Toxin accumulation, digestive stress, nervous system strain",
        "organs":      ["stomach", "intestine", "liver"],
    },
    "arcus_senilis": {
        "name":        "Arcus Senilis",
        "description": "White/grey arc near outer iris edge",
        "indication":  "Poor circulation, cardiovascular risk, high cholesterol",
        "organs":      ["heart", "brain_pineal"],
    },
    "lymphatic_rosary": {
        "name":        "Lymphatic Rosary",
        "description": "Small white/cream beads around outer iris ring",
        "indication":  "Lymphatic congestion, immune system stress",
        "organs":      ["spleen"],
    },
    "healing_lines": {
        "name":        "Healing Lines",
        "description": "White lines crossing dark areas in iris",
        "indication":  "Recovering inflammation, past trauma or infection",
        "organs":      [],
    },
    "overactive_stomach": {
        "name":        "Overactive Stomach Ring",
        "description": "Contracted, tight stomach ring near pupil",
        "indication":  "Hyperacidity, gastric tension, anxiety",
        "organs":      ["stomach", "pancreas"],
    },
    "underactive_stomach": {
        "name":        "Underactive Stomach Ring",
        "description": "Expanded, loose stomach ring near pupil",
        "indication":  "Low stomach acid, poor digestion, bloating",
        "organs":      ["stomach"],
    },
    "normal": {
        "name":        "Normal",
        "description": "No significant pattern anomaly detected",
        "indication":  "Iris texture within expected normal range",
        "organs":      [],
    },
}

# ──────────────────────────────────────────────
# Risk thresholds for zone error scores
# ──────────────────────────────────────────────

RISK_LEVELS = {
    "low":      (0.000, 0.040),   # green  ✅
    "moderate": (0.040, 0.080),   # yellow ⚠️
    "high":     (0.080, 1.000),   # red    🔴
}


def get_risk_level(score: float) -> dict:
    """Return risk level dict for a given zone error score."""
    if score < RISK_LEVELS["moderate"][0]:
        return {"level": "low",      "emoji": "✅", "color": "#00e676"}
    elif score < RISK_LEVELS["high"][0]:
        return {"level": "moderate", "emoji": "⚠️",  "color": "#ffd740"}
    else:
        return {"level": "high",     "emoji": "🔴", "color": "#ff3d5a"}


def get_zones(eye_side: str = "left") -> dict:
    """Return angular zone map for the given eye side."""
    return ANGULAR_ZONES_LEFT if eye_side.lower() == "left" else ANGULAR_ZONES_RIGHT
