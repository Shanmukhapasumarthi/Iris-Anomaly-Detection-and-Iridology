"""
scripts/prepare_data.py
End-to-end data preparation pipeline:
  Stage 1  — EDA & quality filtering
  Stage 2  — Iris segmentation
  Stage 3  — Polar normalization

Run from the project root:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --stages 2 3   # skip EDA
    python scripts/prepare_data.py --workers 8
"""

import argparse
import sys
import time
from pathlib import Path

# Allow imports from project root (script may live at repo root or in scripts/)
ROOT = Path(__file__).resolve().parent
if (ROOT / "segmentation.py").is_file():
    pass
else:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────
# Stage runners
# ──────────────────────────────────────────────

def run_eda():
    print("\n" + "═" * 55)
    print("  STAGE 1 — Data Ingestion & EDA")
    print("═" * 55)
    import json

    import cv2
    from tqdm import tqdm

    # Same sharpness rule as notebooks/01_eda.ipynb (Laplacian variance)
    SHARP_THRESH = 50.0
    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def collect_paths(root: Path):
        paths = []
        for ext in SUPPORTED_EXTS:
            paths.extend(root.rglob(f"*{ext}"))
        return sorted({p.resolve() for p in paths})

    def laplacian_sharpness(img) -> float:
        return float(cv2.Laplacian(img, cv2.CV_64F).var())

    raw_dir = ROOT / "raw"
    if not raw_dir.is_dir():
        raw_dir = ROOT / "data" / "raw"
    if not raw_dir.is_dir():
        print(f"  ERROR: Put iris images under {ROOT / 'raw'} or {ROOT / 'data' / 'raw'}")
        return

    paths = collect_paths(raw_dir)
    if not paths:
        print(f"  ERROR: No images found under {raw_dir}")
        return

    out_dir = ROOT / "data" / "processed"
    report_dir = ROOT / "reports" / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    clean_file = out_dir / "clean_image_paths.txt"

    records = []
    for p in tqdm(paths, desc="  EDA", ncols=65):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        records.append(
            {"path": str(p), "sharpness": laplacian_sharpness(gray)}
        )

    if not records:
        print("  ERROR: No readable images.")
        return

    clean = [r for r in records if r["sharpness"] >= SHARP_THRESH]
    if not clean:
        print(
            f"  WARNING: No images with sharpness >= {SHARP_THRESH}; "
            "keeping all readable images."
        )
        clean = records

    with open(clean_file, "w", encoding="utf-8", newline="\n") as f:
        for r in clean:
            f.write(r["path"] + "\n")

    summary = {
        "n_scanned": len(records),
        "n_kept": len(clean),
        "sharpness_threshold": SHARP_THRESH,
        "raw_dir": str(raw_dir),
    }
    with open(report_dir / "stage1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  OK: {len(clean)} / {len(records)} images → {clean_file}")
    print(f"  Summary → {report_dir / 'stage1_summary.json'}")


def run_segmentation(workers: int = 4):
    print("\n" + "═" * 55)
    print("  STAGE 2 — Iris Segmentation")
    print("═" * 55)
    import json
    import cv2
    from pathlib import Path
    from tqdm import tqdm
    from segmentation import segment_iris, build_annular_mask
    from segmentation import build_eyelid_mask

    CLEAN_FILE  = ROOT / "data/processed/clean_image_paths.txt"
    SEG_OUT_DIR = ROOT / "data/processed/segmented"
    SEG_REC_FILE= ROOT / "data/processed/segmentation_records.json"
    SEG_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CLEAN_FILE.exists():
        print(f"  ERROR: {CLEAN_FILE} not found — run Stage 1 first.")
        return

    with open(CLEAN_FILE) as f:
        paths = [Path(l.strip()) for l in f if l.strip()]

    print(f"  Segmenting {len(paths)} images …")
    records, failed = [], []

    for p in tqdm(paths, desc="  Segmenting", ncols=65):
        result = segment_iris(p)
        if result is None:
            failed.append(str(p))
            records.append({"path": str(p), "valid": False})
            continue

        pupil = result["pupil"]; iris = result["iris"]
        mask  = result["annular_mask"]
        mask_path = SEG_OUT_DIR / (p.stem + "_mask.png")
        cv2.imwrite(str(mask_path), mask)

        records.append({
            "path": str(p), "valid": True,
            "x_p": pupil[0], "y_p": pupil[1], "r_p": pupil[2],
            "x_i": iris[0],  "y_i": iris[1],  "r_i": iris[2],
            "mask": str(mask_path),
        })

    with open(SEG_REC_FILE, "w") as f:
        json.dump(records, f, indent=2)

    print(f"  OK: {sum(r['valid'] for r in records)} | Failed: {len(failed)}")
    print(f"  Records → {SEG_REC_FILE}")


def run_normalization():
    print("\n" + "═" * 55)
    print("  STAGE 3 — Polar Normalization")
    print("═" * 55)
    import json
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm
    import cv2
    from normalization import (
        rubber_sheet_normalize, strip_quality_ok
    )

    SEG_REC_FILE  = ROOT / "data/processed/segmentation_records.json"
    NORM_OUT_DIR  = ROOT / "data/normalized"
    NORM_REC_FILE = ROOT / "data/normalized/normalization_records.json"
    NORM_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SEG_REC_FILE.exists():
        print(f"  ERROR: {SEG_REC_FILE} not found — run Stage 2 first.")
        return

    with open(SEG_REC_FILE) as f:
        seg_records = [r for r in json.load(f) if r.get("valid")]

    print(f"  Normalising {len(seg_records)} segmented images …")
    norm_records, failed = [], []

    for rec in tqdm(seg_records, desc="  Normalising", ncols=65):
        p    = Path(rec["path"])
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            failed.append(str(p)); continue

        pupil = (rec["x_p"], rec["y_p"], rec["r_p"])
        iris  = (rec["x_i"], rec["y_i"], rec["r_i"])
        strip = rubber_sheet_normalize(gray, pupil, iris)

        if not strip_quality_ok(strip):
            failed.append(str(p)); continue

        out_path = NORM_OUT_DIR / (p.stem + "_norm.npy")
        np.save(str(out_path), strip)
        norm_records.append({
            "original": str(p),
            "strip":    str(out_path),
            "mean":     round(float(strip.mean()), 4),
            "std":      round(float(strip.std()),  4),
        })

    with open(NORM_REC_FILE, "w") as f:
        json.dump(norm_records, f, indent=2)

    print(f"  OK: {len(norm_records)} | Failed: {len(failed)}")
    print(f"  Records → {NORM_REC_FILE}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Iris — Data Preparation")
    parser.add_argument("--stages",  nargs="+", type=int,
                        default=[1, 2, 3])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    runners = {1: run_eda, 2: run_segmentation, 3: run_normalization}
    for s in sorted(set(args.stages)):
        t0 = time.time()
        if s == 2:
            runners[s](args.workers)
        else:
            runners[s]()
        print(f"  [stage {s} done in {time.time()-t0:.1f}s]")

    print("\n✓ Data preparation complete.")


if __name__ == "__main__":
    main()
