"""
Run:
    python iridology_app.py
Open: http://localhost:8001
"""

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config        import get_device
from src.preprocessing.normalization import apply_clahe, rubber_sheet_normalize
from src.evaluation.threshold     import select_threshold
from src.iridology.iridology_zones import get_zones, ORGAN_INFO, IRIS_PATTERNS, get_risk_level
from src.iridology.zone_analyzer   import full_zone_analysis
from src.iridology.healthreport   import generate_report
#from llm_analysis    import generate_disease_analysis, format_analysis_for_display

CHECKPOINT_DIR   = ROOT / "checkpoints"
NORM_REC_FILE    = ROOT / "data/normalized/normalization_records.json"
STRIP_H, STRIP_W = 64, 512

app = FastAPI(title="Iris Iridology Health Analysis v2", version="2.0.0")

_state: dict = dict(model=None, model_type=None, device=None,
                    threshold=None, loaded_at=None)

# Cache last analysis so /analyze/llm can use it without re-running inference
_last_report: dict = {}


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(model_type, device):
    if model_type == "mae":
        from scripts.train import ViTMAE
        p    = CHECKPOINT_DIR / "best_mae.pth"
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        ckpt  = torch.load(p, map_location=device, weights_only=False)
        model = ViTMAE().to(device)
        model.load_state_dict(ckpt["state"])
        model.eval()
        print(f"  [iridology] ViT-MAE loaded epoch={ckpt.get('epoch','?')}")
        return model
    elif model_type == "ae":
        from src.Models.autoencoder import ConvAutoencoder
        ckpt  = torch.load(CHECKPOINT_DIR/"best_ae.pth",
                           map_location=device, weights_only=False)
        model = ConvAutoencoder(256).to(device)
        model.load_state_dict(ckpt["state"]); model.eval()
        return model
    else:
        from src.Models.vae import ConvVAE
        ckpt  = torch.load(CHECKPOINT_DIR/"best_vae.pth",
                           map_location=device, weights_only=False)
        model = ConvVAE(256).to(device)
        model.load_state_dict(ckpt["state"]); model.eval()
        return model


def _fit_threshold(model, device):
    if not NORM_REC_FILE.exists():
        return 0.05
    from src.utils.dataset import build_dataloaders
    _, val_loader, _ = build_dataloaders(
        records_file=NORM_REC_FILE, batch_size=32, num_workers=0)
    scores = []
    with torch.no_grad():
        for batch in val_loader:
            scores.append(model.anomaly_score(batch.to(device)).cpu().numpy())
    thr = select_threshold(np.concatenate(scores), method="sigma", k=2.0)
    print(f"  [iridology] Threshold: {thr:.5f}")
    return thr


@app.on_event("startup")
async def on_startup():
    device     = get_device()
    model_type = os.environ.get("IRIS_MODEL", "mae")
    try:
        model = _load_model(model_type, device)
        _state.update(model=model, model_type=model_type, device=device,
                      threshold=_fit_threshold(model, device),
                      loaded_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        print("  [iridology] Ready.")
    except Exception as e:
        print(f"  [iridology] ERROR: {e}")


# ── Image processing ──────────────────────────────────────────────────────────

def _to_gray(img):
    if img.ndim == 2: return img
    if img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    b,g,r = img[:,:,0].astype(np.float32), img[:,:,1].astype(np.float32), img[:,:,2].astype(np.float32)
    return (0.55*r+0.30*g+0.15*b).clip(0,255).astype(np.uint8)

def _decode(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Cannot decode image.")
    return img

def _extract_strip(img):
    from src.preprocessing.segmentation import detect_pupil, detect_iris
    gray  = _to_gray(img)
    pupil = detect_pupil(gray)
    if pupil is None: raise ValueError("Pupil not detected.")
    iris  = detect_iris(gray, pupil)
    if iris is None:  raise ValueError("Iris boundary not detected.")
    strip = apply_clahe(rubber_sheet_normalize(gray, pupil, iris, STRIP_H, STRIP_W))
    return strip, pupil, iris, gray

@torch.no_grad()
def _infer(strip):
    model  = _state["model"]
    device = _state["device"]
    tensor = torch.from_numpy(strip).unsqueeze(0).unsqueeze(0).to(device)
    score  = model.anomaly_score(tensor).item()
    if _state["model_type"] == "mae":
        _, pred, _ = model(tensor)
        recon = model.unpatchify(pred, STRIP_H, STRIP_W)[0,0].cpu().numpy()
    else:
        recon = model(tensor)[0][0,0].cpu().numpy()
    return score, recon.clip(0,1)


# ── Visualisation ─────────────────────────────────────────────────────────────

def _to_b64(arr, cmap="gray"):
    fig, ax = plt.subplots(figsize=(8,1), dpi=100)
    ax.imshow(arr, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.axis("off"); plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _annotated_b64(img, pupil, iris):
    c = img.copy() if img.ndim==3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(c, pupil[:2], pupil[2], (0,255,255), 2)
    cv2.circle(c, iris[:2],  iris[2],  (0,200,255), 2)
    _, buf = cv2.imencode(".png", c)
    return base64.b64encode(buf.tobytes()).decode()

def _zone_heatmap_b64(strip, recon, zone_scores, eye_side):
    err      = np.abs(strip-recon)
    err_norm = (err-err.min())/(err.max()-err.min()+1e-8)
    fig, axes = plt.subplots(3,1, figsize=(10,4),
                              gridspec_kw={"height_ratios":[2,1,1]})
    fig.patch.set_facecolor("#0d1117")
    ax = axes[0]
    ax.imshow(err_norm, cmap="jet", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Error Map + Organ Zone Overlay", color="#00e5ff", fontsize=8, pad=4)
    ax.set_yticks([]); ax.tick_params(colors="white")
    zones = get_zones(eye_side)
    for organ_key,(c0,c1) in zones.items():
        data  = zone_scores.get(organ_key, {})
        color = data.get("color","#ffffff")
        rect  = mpatches.FancyBboxPatch((c0,0),c1-c0,STRIP_H,
                  boxstyle="square,pad=0",linewidth=1.5,
                  edgecolor=color,facecolor="none",alpha=0.7)
        ax.add_patch(rect)
        name = data.get("name",organ_key).split("/")[0].strip()[:8]
        ax.text((c0+c1)/2,STRIP_H/2,name,color=color,fontsize=4.5,
                ha="center",va="center",fontweight="bold",
                bbox=dict(facecolor="#0d1117",alpha=0.5,pad=1,linewidth=0))
    axes[1].imshow(strip, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("Original Strip", color="#c9d1d9", fontsize=7, pad=2)
    axes[1].axis("off")
    axes[2].imshow(recon, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[2].set_title("Reconstruction", color="#c9d1d9", fontsize=7, pad=2)
    axes[2].axis("off")
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _organ_bar_b64(zone_scores):
    items  = list(zone_scores.items())[:12]
    names  = [v["icon"]+" "+v["name"] for _,v in items]
    scores = [v["score"] for _,v in items]
    colors = [v["color"] for _,v in items]
    fig, ax = plt.subplots(figsize=(7, max(3,len(names)*0.45)))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    bars = ax.barh(names, scores, color=colors, edgecolor="#1e2d3d", height=0.6)
    ax.set_xlabel("Reconstruction Error Score", color="#c9d1d9", fontsize=8)
    ax.set_title("Organ Zone Anomaly Scores", color="#00e5ff", fontsize=9, pad=8)
    ax.tick_params(colors="#c9d1d9", labelsize=7)
    ax.spines[:].set_color("#1e2d3d")
    ax.axvline(0.04, color="#00e676", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(0.08, color="#ff3d5a", linestyle="--", linewidth=1, alpha=0.7)
    for bar,score in zip(bars,scores):
        ax.text(score+0.001, bar.get_y()+bar.get_height()/2,
                f"{score:.4f}", va="center", fontsize=6, color="#c9d1d9")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui(): return HTMLResponse(_HTML)

@app.get("/health", tags=["Health"])
def health():
    if _state["model"] is None: return {"status":"starting"}
    return {"status":"ok","model":_state["model_type"],
            "threshold":round(_state["threshold"],6)}


@app.post("/analyze", tags=["Iridology"])
async def analyze(
    file:     UploadFile = File(...),
    eye_side: str        = Query("left", enum=["left","right"]),
):
    """Upload eye image → full iridology zone analysis + health report."""
    if _state["model"] is None:
        raise HTTPException(503, "Model not loaded.")
    data = await file.read()
    if not data: raise HTTPException(400, "Empty file.")

    try: img = _decode(data)
    except ValueError as e: raise HTTPException(422, str(e))

    try: strip, pupil, iris, gray = _extract_strip(img)
    except ValueError as e: raise HTTPException(422, str(e))

    t0 = time.perf_counter()
    try: score, recon = _infer(strip)
    except Exception as e: raise HTTPException(500, f"Inference error: {e}")
    ms = (time.perf_counter()-t0)*1000

    mean_error = float(np.abs(strip-recon).mean())
    ssim_score = float(ssim(strip, recon, data_range=1.0))
    analysis   = full_zone_analysis(strip, recon, eye_side)
    report     = generate_report(analysis, score, _state["threshold"],
                                  mean_error, ssim_score,
                                  file.filename or "upload")

    err      = np.abs(strip-recon)
    err_norm = (err-err.min())/(err.max()-err.min()+1e-8)

    report["images"] = {
        "eye_annotated":  _annotated_b64(img, pupil, iris),
        "strip":          _to_b64(strip,   "gray"),
        "reconstruction": _to_b64(recon,   "gray"),
        "error_map":      _to_b64(err_norm,"jet"),
        "zone_heatmap":   _zone_heatmap_b64(strip, recon,
                                             analysis["zone_scores"], eye_side),
        "organ_chart":    _organ_bar_b64(analysis["zone_scores"]),
    }
    report["inference_ms"] = round(ms, 2)

    # Cache for LLM endpoint
    global _last_report
    _last_report = report

    return JSONResponse(content=report)


@app.post("/analyze/llm", tags=["AI Analysis"])
async def llm_analysis(
    file:     UploadFile = File(None),
    eye_side: str        = Query("left", enum=["left","right"]),
):
    """
    Generate detailed AI disease analysis using Claude.
    If a file is uploaded it runs /analyze first then calls Claude.
    If no file is uploaded it uses the cached last report.
    """
    global _last_report

    # Run iris analysis if new file uploaded
    if file and file.filename:
        data = await file.read()
        if data:
            try: img = _decode(data)
            except ValueError as e: raise HTTPException(422, str(e))
            try: strip, pupil, iris, gray = _extract_strip(img)
            except ValueError as e: raise HTTPException(422, str(e))
            try: score, recon = _infer(strip)
            except Exception as e: raise HTTPException(500, str(e))
            mean_error = float(np.abs(strip-recon).mean())
            ssim_score = float(ssim(strip, recon, data_range=1.0))
            analysis   = full_zone_analysis(strip, recon, eye_side)
            _last_report = generate_report(analysis, score, _state["threshold"],
                                           mean_error, ssim_score,
                                           file.filename or "upload")

    if not _last_report:
        raise HTTPException(400,
            "No analysis available. Run /analyze first or upload a file.")

    try:
        raw      = await generate_disease_analysis(_last_report)
        analysis = format_analysis_for_display(raw)
    except Exception as e:
        raise HTTPException(500, f"LLM error: {e}")

    return JSONResponse(content=analysis)


# ── HTML UI ───────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Iridology Health Analysis</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
:root{--bg:#080c10;--sur:#0d1117;--bdr:#1e2d3d;--acc:#00e5ff;--red:#ff3d5a;--grn:#00e676;--yel:#ffd740;--pur:#b388ff;--txt:#c9d1d9;--mut:#4a5568;--mono:'Share Tech Mono',monospace;--sans:'Rajdhani',sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:var(--sans);min-height:100vh;padding:2rem}
body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,229,255,.015) 2px,rgba(0,229,255,.015) 4px);pointer-events:none;z-index:0}
.wrap{max-width:1200px;margin:0 auto;position:relative;z-index:1}

/* Header */
header{display:flex;align-items:center;gap:1.2rem;margin-bottom:2rem;border-bottom:1px solid var(--bdr);padding-bottom:1.2rem}
.logo{width:48px;height:48px;border:2px solid var(--acc);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.5rem;animation:pulse 3s ease-in-out infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 18px rgba(0,229,255,.35)}50%{box-shadow:0 0 32px rgba(0,229,255,.7)}}
h1{font-size:1.8rem;font-weight:700;color:#fff;text-transform:uppercase;letter-spacing:.08em}
h1 span{color:var(--acc)}
.sub{font-family:var(--mono);font-size:.7rem;color:var(--mut)}

/* Controls */
.controls{display:grid;grid-template-columns:1fr auto;gap:1rem;margin-bottom:1.5rem;align-items:end}
.dropzone{border:2px dashed var(--bdr);border-radius:6px;padding:2rem;text-align:center;cursor:pointer;background:var(--sur);transition:border-color .2s}
.dropzone:hover,.dropzone.drag{border-color:var(--acc)}
.dropzone input{display:none}
.di{font-size:2rem;margin-bottom:.4rem}.dl{font-size:1rem;font-weight:600}
.dh{font-family:var(--mono);font-size:.7rem;color:var(--mut);margin-top:.2rem}
#pw{display:none;margin-top:.8rem;border:1px solid var(--bdr);border-radius:4px;overflow:hidden}
#prev{width:100%;max-height:180px;object-fit:contain;background:#000;display:block}
.side-panel{background:var(--sur);border:1px solid var(--bdr);border-radius:6px;padding:1rem;min-width:160px}
.side-label{font-family:var(--mono);font-size:.7rem;color:var(--acc);display:block;margin-bottom:.5rem;text-transform:uppercase}
.eye-btns{display:flex;gap:.5rem}
.eye-btn{flex:1;padding:.6rem;background:var(--bdr);border:1px solid var(--bdr);border-radius:4px;color:var(--txt);font-family:var(--sans);font-size:.9rem;font-weight:600;cursor:pointer;transition:all .15s;text-align:center}
.eye-btn.active{background:rgba(0,229,255,.15);border-color:var(--acc);color:var(--acc)}

/* Buttons */
.btn-row{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.5rem}
.btn{padding:.85rem;font-family:var(--sans);font-size:.95rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;border:none;border-radius:4px;cursor:pointer;transition:opacity .15s,transform .1s}
.btn:hover{opacity:.88}.btn:active{transform:scale(.98)}.btn:disabled{opacity:.35;cursor:not-allowed}
.btn-primary{background:var(--acc);color:#000}
.btn-llm{background:linear-gradient(135deg,#7c3aed,#b388ff);color:#fff;position:relative;overflow:hidden}
.btn-llm::before{content:'✨ ';font-size:1rem}

/* Spinner */
.spin{display:none;margin:1.5rem auto;text-align:center;font-family:var(--mono);color:var(--acc);font-size:.85rem}
.spin::before{content:'';display:block;width:36px;height:36px;border:3px solid var(--bdr);border-top-color:var(--acc);border-radius:50%;margin:0 auto .8rem;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.spin-llm{color:var(--pur)}.spin-llm::before{border-top-color:var(--pur)}

/* Results */
#results{display:none;margin-top:1.5rem}
.vbanner{padding:1rem 1.4rem;border-radius:6px;margin-bottom:1.5rem;display:flex;align-items:center;gap:1rem;border:1px solid}
.vbanner.normal{border-color:var(--grn);background:rgba(0,230,118,.07)}
.vbanner.anomalous{border-color:var(--red);background:rgba(255,61,90,.07)}
.vemoji{font-size:2rem}.vlabel{font-size:1.4rem;font-weight:700;text-transform:uppercase}
.vsub{font-family:var(--mono);font-size:.72rem;color:var(--mut);margin-top:.2rem}

.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:.8rem;margin-bottom:1.2rem}
.mc{background:var(--sur);border:1px solid var(--bdr);border-radius:6px;padding:.9rem 1rem}
.ml{font-family:var(--mono);font-size:.62rem;color:var(--mut);letter-spacing:.08em;text-transform:uppercase;margin-bottom:.3rem}
.mv{font-size:1.2rem;font-weight:700;color:#fff}.mv.ac{color:var(--acc)}

.sec-title{font-family:var(--mono);font-size:.72rem;color:var(--acc);text-transform:uppercase;letter-spacing:.12em;margin:1.5rem 0 .8rem;border-bottom:1px solid var(--bdr);padding-bottom:.4rem}

.pattern-card{background:var(--sur);border:1px solid var(--acc);border-radius:6px;padding:1.2rem;margin-bottom:1.2rem}
.pattern-name{font-size:1.2rem;font-weight:700;color:var(--acc);margin-bottom:.4rem}
.pattern-desc{font-size:.9rem;margin-bottom:.4rem}
.pattern-ind{font-family:var(--mono);font-size:.75rem;color:var(--yel)}

.concerns{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin-bottom:1.2rem}
.cc{background:var(--sur);border:1px solid var(--bdr);border-radius:6px;padding:.9rem}
.cc.high{border-color:var(--red)}.cc.moderate{border-color:var(--yel)}
.cc-icon{font-size:1.4rem;margin-bottom:.3rem}.cc-name{font-weight:700;font-size:1rem}
.cc-sys{font-family:var(--mono);font-size:.65rem;color:var(--mut);margin-bottom:.4rem}
.cc-score{font-family:var(--mono);font-size:.8rem}

.img-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.2rem}
.img-panel{background:var(--sur);border:1px solid var(--bdr);border-radius:6px;overflow:hidden}
.img-panel.wide{grid-column:1/-1}
.pt{padding:.45rem .8rem;font-family:var(--mono);font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--acc);border-bottom:1px solid var(--bdr);background:rgba(0,229,255,.04)}
.img-panel img{width:100%;display:block;background:#000}

.organ-table{width:100%;border-collapse:collapse;font-size:.85rem;margin-bottom:1.2rem}
.organ-table th{font-family:var(--mono);font-size:.63rem;color:var(--acc);text-transform:uppercase;padding:.6rem 1rem;border-bottom:1px solid var(--bdr);text-align:left;background:rgba(0,229,255,.04)}
.organ-table td{padding:.55rem 1rem;border-bottom:1px solid rgba(30,45,61,.5)}
.organ-table tr:hover td{background:rgba(0,229,255,.03)}
.badge{padding:.15rem .5rem;border-radius:3px;font-family:var(--mono);font-size:.63rem;font-weight:700}
.badge.high{background:rgba(255,61,90,.2);color:var(--red)}
.badge.moderate{background:rgba(255,215,64,.2);color:var(--yel)}
.badge.low{background:rgba(0,230,118,.2);color:var(--grn)}

/* ── LLM Analysis section ── */
#llm-results{display:none;margin-top:2rem}
.llm-header{display:flex;align-items:center;gap:.8rem;margin-bottom:1.2rem;padding:1rem 1.2rem;background:linear-gradient(135deg,rgba(124,58,237,.15),rgba(179,136,255,.08));border:1px solid rgba(179,136,255,.3);border-radius:8px}
.llm-icon{font-size:1.8rem}
.llm-title{font-size:1.2rem;font-weight:700;color:var(--pur);text-transform:uppercase;letter-spacing:.08em}
.llm-sub{font-family:var(--mono);font-size:.68rem;color:var(--mut)}

.summary-box{background:var(--sur);border:1px solid rgba(179,136,255,.3);border-radius:6px;padding:1.2rem;margin-bottom:1.2rem;font-size:.95rem;line-height:1.7;color:var(--txt)}
.summary-box .s-label{font-family:var(--mono);font-size:.65rem;color:var(--pur);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem}

.pattern-explain{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.5rem}
.pe-card{background:var(--sur);border:1px solid var(--bdr);border-radius:6px;padding:1rem}
.pe-title{font-family:var(--mono);font-size:.65rem;color:var(--pur);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem}
.pe-body{font-size:.88rem;line-height:1.6;color:var(--txt)}

.organ-analysis-cards{display:flex;flex-direction:column;gap:1.2rem;margin-bottom:1.5rem}
.oa-card{background:var(--sur);border:1px solid var(--bdr);border-radius:8px;overflow:hidden}
.oa-header{padding:.8rem 1.2rem;background:linear-gradient(90deg,rgba(124,58,237,.2),transparent);display:flex;align-items:center;gap:.8rem;border-bottom:1px solid var(--bdr)}
.oa-icon{font-size:1.4rem}
.oa-organ{font-size:1.1rem;font-weight:700;color:#fff}
.oa-system{font-family:var(--mono);font-size:.65rem;color:var(--mut);margin-top:.1rem}
.oa-body{padding:1rem 1.2rem;display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem}
.oa-section{padding:.8rem;background:rgba(0,0,0,.2);border-radius:6px}
.oa-sec-title{font-family:var(--mono);font-size:.63rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem}
.oa-sec-title.causes{color:#ff8a65}
.oa-sec-title.effects{color:#ef5350}
.oa-sec-title.prevention{color:#66bb6a}
.oa-list{list-style:none;font-size:.83rem;line-height:1.7;color:var(--txt)}
.oa-list li::before{content:"→ ";color:var(--mut)}
.oa-conditions{padding:.8rem 1.2rem;border-top:1px solid var(--bdr)}
.oa-cond-title{font-family:var(--mono);font-size:.63rem;color:var(--yel);text-transform:uppercase;margin-bottom:.5rem}
.cond-pills{display:flex;flex-wrap:wrap;gap:.4rem}
.cond-pill{background:rgba(255,215,64,.1);border:1px solid rgba(255,215,64,.3);color:var(--yel);padding:.2rem .7rem;border-radius:20px;font-size:.75rem;font-family:var(--mono)}
.oa-doctor{padding:.7rem 1.2rem;border-top:1px solid var(--bdr);font-family:var(--mono);font-size:.72rem;color:#ff8a65}
.oa-doctor::before{content:"🏥 "}

.lifestyle-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:.8rem;margin-bottom:1.5rem}
.ls-card{background:var(--sur);border:1px solid var(--bdr);border-radius:6px;padding:.9rem 1rem;display:flex;align-items:flex-start;gap:.7rem;font-size:.88rem;line-height:1.5}
.ls-num{font-family:var(--mono);font-size:.8rem;color:var(--grn);font-weight:700;min-width:24px}

.disclaimer-box{background:rgba(255,215,64,.07);border:1px solid rgba(255,215,64,.3);border-radius:6px;padding:1rem 1.2rem;font-family:var(--mono);font-size:.72rem;color:var(--yel);line-height:1.6}

.err-box{background:rgba(255,61,90,.1);border:1px solid var(--red);border-radius:6px;padding:1rem;font-family:var(--mono);font-size:.8rem;color:var(--red);margin-top:1rem;display:none}

@media(max-width:700px){
  .controls{grid-template-columns:1fr}
  .metrics{grid-template-columns:repeat(2,1fr)}
  .img-grid,.concerns,.pattern-explain,.oa-body,.lifestyle-grid{grid-template-columns:1fr}
}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">🔬</div>
    <div>
      <h1>Iridology <span>Health</span> Analysis</h1>
      <div class="sub">ViT-MAE · Zone Error Mapping · AI Disease Analysis</div>
    </div>
  </header>

  <div class="controls">
    <div class="dropzone" id="dz">
      <input type="file" id="fi" accept="image/*">
      <div class="di">👁</div>
      <div class="dl">Drop eye image here or click to upload</div>
      <div class="dh">PNG · JPG · BMP — clear frontal eye photograph</div>
      <div id="pw"><img id="prev" alt="Preview"></div>
    </div>
    <div class="side-panel">
      <span class="side-label">Eye Side</span>
      <div class="eye-btns">
        <div class="eye-btn active" id="bl" onclick="setEye('left')">👁 Left</div>
        <div class="eye-btn"        id="br" onclick="setEye('right')">👁 Right</div>
      </div>
      <div style="margin-top:1rem;font-family:var(--mono);font-size:.62rem;color:var(--mut);line-height:1.6">
        Left eye → left-side organs<br>Right eye → right-side organs
      </div>
    </div>
  </div>

  <div class="btn-row">
    <button class="btn btn-primary" id="btn-analyze" disabled>🔬 ANALYSE IRIS</button>
    <button class="btn btn-llm"     id="btn-llm"     disabled>AI DISEASE ANALYSIS</button>
  </div>

  <div class="spin"     id="spin">Analysing iris zones…</div>
  <div class="spin spin-llm" id="spin-llm">Generating AI disease analysis…</div>
  <div class="err-box"  id="err"></div>

  <!-- ── Iris Analysis Results ── -->
  <div id="results">
    <div class="vbanner" id="vb">
      <div class="vemoji" id="ve"></div>
      <div><div class="vlabel" id="vl"></div><div class="vsub" id="vs"></div></div>
    </div>

    <div class="metrics">
      <div class="mc"><div class="ml">Anomaly Score</div><div class="mv ac" id="m-score"></div></div>
      <div class="mc"><div class="ml">Global Risk</div><div class="mv" id="m-risk"></div></div>
      <div class="mc"><div class="ml">Mean Error</div><div class="mv" id="m-err"></div></div>
      <div class="mc"><div class="ml">SSIM</div><div class="mv" id="m-ssim"></div></div>
      <div class="mc"><div class="ml">Inference</div><div class="mv" id="m-ms"></div></div>
    </div>

    <div class="sec-title">🌀 Detected Iris Pattern</div>
    <div class="pattern-card">
      <div class="pattern-name" id="pat-name"></div>
      <div class="pattern-desc" id="pat-desc"></div>
      <div class="pattern-ind"  id="pat-ind"></div>
    </div>

    <div class="sec-title">⚠️ Primary Areas of Concern</div>
    <div class="concerns" id="concerns"></div>

    <div class="sec-title">🖼 Visual Analysis</div>
    <div class="img-grid">
      <div class="img-panel"><div class="pt">📷 Annotated Eye</div><img id="i-eye" alt="Eye"></div>
      <div class="img-panel"><div class="pt">🌀 Iris Strip</div><img id="i-strip" alt="Strip"></div>
      <div class="img-panel wide"><div class="pt">🗺 Zone Heatmap</div><img id="i-zone" alt="Zones"></div>
      <div class="img-panel wide"><div class="pt">📊 Organ Zone Chart</div><img id="i-chart" alt="Chart"></div>
    </div>

    <div class="sec-title">📋 Full Organ Zone Report</div>
    <table class="organ-table">
      <thead><tr><th>Organ</th><th>System</th><th>Error Score</th><th>Risk</th></tr></thead>
      <tbody id="organ-tbody"></tbody>
    </table>

    <div class="disclaimer-box" id="disclaimer"></div>
  </div>

  <!-- ── LLM Disease Analysis ── -->
  <div id="llm-results">
    <div class="llm-header">
      <div class="llm-icon">🤖</div>
      <div>
        <div class="llm-title">AI Disease Analysis</div>
        <div class="llm-sub">Generated by Claude · Based on iridology zone findings</div>
      </div>
    </div>

    <div class="summary-box">
      <div class="s-label">📋 Overall Summary</div>
      <div id="llm-summary"></div>
    </div>

    <div class="sec-title">🌀 Iris Pattern Explanation</div>
    <div class="pattern-explain">
      <div class="pe-card"><div class="pe-title">What It Is</div><div class="pe-body" id="pe-what"></div></div>
      <div class="pe-card"><div class="pe-title">How It Forms</div><div class="pe-body" id="pe-how"></div></div>
      <div class="pe-card"><div class="pe-title">What It Indicates</div><div class="pe-body" id="pe-indicates"></div></div>
    </div>

    <div class="sec-title">🏥 Organ-by-Organ Analysis</div>
    <div class="organ-analysis-cards" id="organ-cards"></div>

    <div class="sec-title">💚 Lifestyle Recommendations</div>
    <div class="lifestyle-grid" id="lifestyle-grid"></div>

    <div class="disclaimer-box" id="llm-disclaimer"></div>
  </div>

</div>

<script>
let file=null, eyeSide='left';
const dz=document.getElementById('dz'),fi=document.getElementById('fi'),
  btnA=document.getElementById('btn-analyze'),btnL=document.getElementById('btn-llm'),
  spin=document.getElementById('spin'),spinL=document.getElementById('spin-llm'),
  res=document.getElementById('results'),llmRes=document.getElementById('llm-results'),
  errBox=document.getElementById('err'),prev=document.getElementById('prev'),
  pw=document.getElementById('pw');

dz.addEventListener('click',()=>fi.click());
fi.addEventListener('change',e=>setFile(e.target.files[0]));
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('drag')});
dz.addEventListener('dragleave',()=>dz.classList.remove('drag'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('drag');if(e.dataTransfer.files[0])setFile(e.dataTransfer.files[0])});

function setFile(f){file=f;btnA.disabled=false;prev.src=URL.createObjectURL(f);pw.style.display='block';res.style.display='none';llmRes.style.display='none';errBox.style.display='none'}
function setEye(s){eyeSide=s;document.getElementById('bl').className='eye-btn'+(s==='left'?' active':'');document.getElementById('br').className='eye-btn'+(s==='right'?' active':'')}

// ── Iris analysis ──
btnA.addEventListener('click',async()=>{
  if(!file)return;
  btnA.disabled=true;spin.style.display='block';
  res.style.display='none';llmRes.style.display='none';errBox.style.display='none';
  const fd=new FormData();fd.append('file',file);
  try{
    const r=await fetch('/analyze?eye_side='+eyeSide,{method:'POST',body:fd});
    const d=await r.json();
    if(!r.ok)throw new Error(d.detail||r.statusText);
    renderAnalysis(d);
    btnL.disabled=false;
  }catch(e){showErr(e.message);}
  finally{spin.style.display='none';btnA.disabled=false;}
});

// ── LLM analysis ──
btnL.addEventListener('click',async()=>{
  btnL.disabled=true;spinL.style.display='block';
  llmRes.style.display='none';errBox.style.display='none';
  try{
    // Call LLM with no file — uses cached last report
    const r=await fetch('/analyze/llm?eye_side='+eyeSide,{method:'POST',body:new FormData()});
    const d=await r.json();
    if(!r.ok)throw new Error(d.detail||r.statusText);
    renderLLM(d);
  }catch(e){showErr('LLM error: '+e.message);}
  finally{spinL.style.display='none';btnL.disabled=false;}
});

function showErr(msg){errBox.textContent='⚠  '+msg;errBox.style.display='block';}

function renderAnalysis(d){
  const ov=d.overall,p=d.iris_pattern,rq=d.reconstruction_quality;
  document.getElementById('vb').className='vbanner '+ov.verdict;
  document.getElementById('ve').textContent=ov.verdict_emoji;
  document.getElementById('vl').textContent=ov.verdict.toUpperCase()+' — '+ov.risk_level.toUpperCase()+' RISK';
  document.getElementById('vs').textContent=`Score: ${ov.anomaly_score.toFixed(5)}  ·  Threshold: ${ov.threshold.toFixed(5)}  ·  Eye: ${d.eye_side.toUpperCase()}`;
  document.getElementById('m-score').textContent=ov.anomaly_score.toFixed(5);
  document.getElementById('m-risk').textContent=ov.risk_emoji+' '+ov.risk_level.toUpperCase();
  document.getElementById('m-err').textContent=rq.mean_error.toFixed(5);
  document.getElementById('m-ssim').textContent=rq.ssim.toFixed(5);
  document.getElementById('m-ms').textContent=d.inference_ms.toFixed(1)+' ms';
  document.getElementById('pat-name').textContent=p.name;
  document.getElementById('pat-desc').textContent=p.description;
  document.getElementById('pat-ind').textContent='💡 '+p.indication;

  const cDiv=document.getElementById('concerns');cDiv.innerHTML='';
  if(!d.primary_concerns.length){
    cDiv.innerHTML='<div style="color:var(--grn);font-family:var(--mono);font-size:.85rem">✅ No significant concerns detected.</div>';
  }else{
    d.primary_concerns.forEach(c=>{
      cDiv.innerHTML+=`<div class="cc ${c.risk_level}">
        <div class="cc-icon">${c.icon}</div>
        <div class="cc-name">${c.organ}</div>
        <div class="cc-sys">${c.system}</div>
        <div class="cc-score" style="color:${c.risk_level==='high'?'var(--red)':'var(--yel)'}">
          ${c.risk_score.toFixed(4)} · ${c.risk_level.toUpperCase()}</div>
      </div>`;
    });
  }

  document.getElementById('i-eye').src='data:image/png;base64,'+d.images.eye_annotated;
  document.getElementById('i-strip').src='data:image/png;base64,'+d.images.strip;
  document.getElementById('i-zone').src='data:image/png;base64,'+d.images.zone_heatmap;
  document.getElementById('i-chart').src='data:image/png;base64,'+d.images.organ_chart;

  const tb=document.getElementById('organ-tbody');tb.innerHTML='';
  d.organ_zones.forEach(o=>{
    tb.innerHTML+=`<tr><td>${o.icon} ${o.organ_name}</td>
      <td style="color:var(--mut);font-size:.8rem">${o.system}</td>
      <td style="font-family:var(--mono)">${o.score.toFixed(6)}</td>
      <td><span class="badge ${o.risk}">${o.risk_emoji} ${o.risk.toUpperCase()}</span></td></tr>`;
  });

  document.getElementById('disclaimer').textContent=d.disclaimer;
  res.style.display='block';
  res.scrollIntoView({behavior:'smooth'});
}

function renderLLM(d){
  document.getElementById('llm-summary').textContent=d.summary;
  document.getElementById('pe-what').textContent=d.pattern_explanation.what_it_is;
  document.getElementById('pe-how').textContent=d.pattern_explanation.how_it_forms;
  document.getElementById('pe-indicates').textContent=d.pattern_explanation.what_it_indicates;

  // Organ analysis cards
  const cards=document.getElementById('organ-cards');cards.innerHTML='';
  d.organ_analyses.forEach(o=>{
    const conds=o.possible_conditions.map(c=>`<span class="cond-pill">${c}</span>`).join('');
    const causes=o.causes.map(c=>`<li>${c}</li>`).join('');
    const effects=o.effects.map(e=>`<li>${e}</li>`).join('');
    const prev=o.prevention.map(p=>`<li>${p}</li>`).join('');
    cards.innerHTML+=`<div class="oa-card">
      <div class="oa-header">
        <div class="oa-icon">🏥</div>
        <div><div class="oa-organ">${o.organ}</div><div class="oa-system">${o.system}</div></div>
      </div>
      <div class="oa-conditions">
        <div class="oa-cond-title">Possible Conditions</div>
        <div class="cond-pills">${conds}</div>
      </div>
      <div class="oa-body">
        <div class="oa-section">
          <div class="oa-sec-title causes">⚡ Causes</div>
          <ul class="oa-list">${causes}</ul>
        </div>
        <div class="oa-section">
          <div class="oa-sec-title effects">🔴 Effects</div>
          <ul class="oa-list">${effects}</ul>
        </div>
        <div class="oa-section">
          <div class="oa-sec-title prevention">✅ Prevention</div>
          <ul class="oa-list">${prev}</ul>
        </div>
      </div>
      <div class="oa-doctor">${o.when_to_see_doctor}</div>
    </div>`;
  });

  // Lifestyle
  const lg=document.getElementById('lifestyle-grid');lg.innerHTML='';
  d.lifestyle_recommendations.forEach((r,i)=>{
    lg.innerHTML+=`<div class="ls-card"><span class="ls-num">${String(i+1).padStart(2,'0')}</span>${r}</div>`;
  });

  document.getElementById('llm-disclaimer').textContent=d.disclaimer;
  llmRes.style.display='block';
  llmRes.scrollIntoView({behavior:'smooth'});
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    uvicorn.run("iridology_app:app", host="0.0.0.0", port=8001,
                reload=False, workers=1)
