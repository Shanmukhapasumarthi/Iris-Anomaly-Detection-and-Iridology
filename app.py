"""
app.py  —  FastAPI iris anomaly detection server v2
GET  /          → browser UI
GET  /health    → model status
POST /predict   → strip + reconstruction + error map + anomaly score
GET  /results   → evaluate.py summary
POST /threshold/update → refit threshold

Run:
    pip install fastapi uvicorn python-multipart
    python app.py
Then open: http://localhost:8000
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
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config        import get_device
from normalization import apply_clahe, rubber_sheet_normalize
from segmentation  import detect_pupil, detect_iris
from threshold     import select_threshold

CHECKPOINT_DIR   = ROOT / "checkpoints"
RESULTS_DIR      = ROOT / "results" / "evaluation"
NORM_REC_FILE    = ROOT / "data/normalized/normalization_records.json"
STRIP_H, STRIP_W = 64, 512

app = FastAPI(title="Iris Anomaly Detection", version="2.0.0")

_state: dict = dict(model=None, model_type=None, device=None,
                    threshold=None, loaded_at=None)


# ── model loading ────────────────────────────────────────────────────────────

def _load_mae(device):
    from train import ViTMAE
    p = CHECKPOINT_DIR / "best_mae.pth"
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    ckpt  = torch.load(p, map_location=device, weights_only=False)
    model = ViTMAE().to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"  [app] ViT-MAE  epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss',0):.5f}")
    return model


def _load_ae(device):
    from autoencoder import ConvAutoencoder
    ckpt  = torch.load(CHECKPOINT_DIR/"best_ae.pth",
                       map_location=device, weights_only=False)
    model = ConvAutoencoder(256).to(device)
    model.load_state_dict(ckpt["state"]); model.eval()
    return model


def _load_vae(device):
    from vae import ConvVAE
    ckpt  = torch.load(CHECKPOINT_DIR/"best_vae.pth",
                       map_location=device, weights_only=False)
    model = ConvVAE(256).to(device)
    model.load_state_dict(ckpt["state"]); model.eval()
    return model


def _fit_threshold(model, device) -> float:
    if not NORM_REC_FILE.exists():
        print("  [app] records not found — fallback threshold 0.05")
        return 0.05
    from dataset import build_dataloaders
    _, val_loader, _ = build_dataloaders(
        records_file=NORM_REC_FILE, batch_size=32, num_workers=0)
    scores = []
    with torch.no_grad():
        for batch in val_loader:
            scores.append(model.anomaly_score(batch.to(device)).cpu().numpy())
    thr = select_threshold(np.concatenate(scores), method="sigma", k=2.0)
    print(f"  [app] Threshold fitted: {thr:.5f}")
    return thr


def startup_model(model_type: str = "mae"):
    device = get_device()
    print(f"  [app] Device: {device}  Model: {model_type}")
    loaders = dict(mae=_load_mae, ae=_load_ae, vae=_load_vae)
    if model_type not in loaders:
        raise ValueError(f"Unknown model: {model_type}")
    model = loaders[model_type](device)
    _state.update(model=model, model_type=model_type, device=device,
                  threshold=_fit_threshold(model, device),
                  loaded_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    print("  [app] Ready.")


@app.on_event("startup")
async def on_startup():
    try:
        startup_model(os.environ.get("IRIS_MODEL", "mae"))
    except Exception as e:
        print(f"  [app] ERROR: {e}")


# ── inference pipeline ───────────────────────────────────────────────────────

def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Cannot decode image — use PNG/JPG/BMP.")
    return img


def _extract(gray: np.ndarray):
    pupil = detect_pupil(gray)
    if pupil is None:
        raise ValueError("Pupil not detected. Use a clear frontal eye image.")
    iris  = detect_iris(gray, pupil)
    if iris is None:
        raise ValueError("Iris boundary not detected.")
    strip = apply_clahe(rubber_sheet_normalize(gray, pupil, iris, STRIP_H, STRIP_W))
    return strip, pupil, iris


@torch.no_grad()
def _infer(strip: np.ndarray):
    model  = _state["model"]
    device = _state["device"]
    tensor = torch.from_numpy(strip).unsqueeze(0).unsqueeze(0).to(device)
    score  = model.anomaly_score(tensor).item()

    if _state["model_type"] == "mae":
        _, pred, _ = model(tensor)
        recon = model.unpatchify(pred, STRIP_H, STRIP_W)[0,0].cpu().numpy()
    else:
        recon = model(tensor)[0][0,0].cpu().numpy()

    return score, recon.clip(0, 1)


def _to_b64(arr: np.ndarray, cmap="gray") -> str:
    fig, ax = plt.subplots(figsize=(8, 1), dpi=100)
    ax.imshow(arr, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.axis("off"); plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _annotated_b64(gray, pupil, iris) -> str:
    c = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(c, pupil[:2], pupil[2], (0,255,255), 2)
    cv2.circle(c, iris[:2],  iris[2],  (0,200,255), 2)
    _, buf = cv2.imencode(".png", c)
    return base64.b64encode(buf.tobytes()).decode()


# ── schemas ──────────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    filename:          str
    anomaly_score:     float
    threshold:         float
    verdict:           str
    verdict_emoji:     str
    confidence_pct:    float
    inference_ms:      float
    pupil:             dict
    iris:              dict
    eye_annotated_b64: str
    strip_b64:         str
    recon_b64:         str
    error_map_b64:     str


# ── endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return HTMLResponse(_HTML)


@app.get("/health", tags=["Health"])
def health():
    if _state["model"] is None:
        return {"status": "starting"}
    return {"status": "ok", "model": _state["model_type"],
            "threshold": round(_state["threshold"], 6),
            "loaded_at": _state["loaded_at"]}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    if _state["model"] is None:
        raise HTTPException(503, "Model not loaded.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")

    try:
        gray = _decode(data)
    except ValueError as e:
        raise HTTPException(422, str(e))

    try:
        strip, pupil, iris = _extract(gray)
    except ValueError as e:
        raise HTTPException(422, str(e))

    t0 = time.perf_counter()
    try:
        score, recon = _infer(strip)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")
    ms = (time.perf_counter() - t0) * 1000

    err   = np.abs(strip - recon)
    err   = (err - err.min()) / (err.max() - err.min() + 1e-8)
    thr   = _state["threshold"]
    anom  = score >= thr
    gap   = abs(score - thr)
    conf  = round(min(gap / (thr + 1e-8) * 100, 100.0), 2)

    return PredictResponse(
        filename=file.filename or "upload",
        anomaly_score=round(score,6), threshold=round(thr,6),
        verdict="anomalous" if anom else "normal",
        verdict_emoji="🔴" if anom else "🟢",
        confidence_pct=conf, inference_ms=round(ms,2),
        pupil=dict(cx=pupil[0], cy=pupil[1], r=pupil[2]),
        iris=dict(cx=iris[0],   cy=iris[1],  r=iris[2]),
        eye_annotated_b64=_annotated_b64(gray, pupil, iris),
        strip_b64=_to_b64(strip,  "gray"),
        recon_b64=_to_b64(recon,  "gray"),
        error_map_b64=_to_b64(err,"jet"),
    )


@app.get("/results", tags=["Evaluation"])
def get_results():
    p = RESULTS_DIR / "summary.json"
    if not p.exists():
        raise HTTPException(404, "Run python evaluate.py first.")
    return JSONResponse(json.load(open(p)))


@app.post("/threshold/update", tags=["Configuration"])
def update_threshold(k: float = Query(2.0, ge=0.5, le=5.0)):
    if _state["model"] is None:
        raise HTTPException(503, "Model not loaded.")
    if not NORM_REC_FILE.exists():
        raise HTTPException(404, "normalization_records.json not found.")
    from dataset import build_dataloaders
    _, val_loader, _ = build_dataloaders(
        records_file=NORM_REC_FILE, batch_size=32, num_workers=0)
    scores = []
    with torch.no_grad():
        for b in val_loader:
            scores.append(_state["model"].anomaly_score(
                b.to(_state["device"])).cpu().numpy())
    old = _state["threshold"]
    _state["threshold"] = select_threshold(np.concatenate(scores),
                                           method="sigma", k=k)
    return {"old": round(old,6), "new": round(_state["threshold"],6), "k": k}


# ── HTML UI ──────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Iris Anomaly Detection</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
:root{--bg:#080c10;--sur:#0d1117;--bdr:#1e2d3d;--acc:#00e5ff;--red:#ff3d5a;--grn:#00e676;--txt:#c9d1d9;--mut:#4a5568;--mono:'Share Tech Mono',monospace;--sans:'Rajdhani',sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:var(--sans);min-height:100vh;padding:2rem}
body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,229,255,.015) 2px,rgba(0,229,255,.015) 4px);pointer-events:none;z-index:0}
.wrap{max-width:1100px;margin:0 auto;position:relative;z-index:1}
header{display:flex;align-items:center;gap:1.2rem;margin-bottom:2.5rem;border-bottom:1px solid var(--bdr);padding-bottom:1.2rem}
.logo{width:44px;height:44px;border:2px solid var(--acc);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.4rem;animation:pulse 3s ease-in-out infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 18px rgba(0,229,255,.35)}50%{box-shadow:0 0 32px rgba(0,229,255,.7)}}
h1{font-size:1.7rem;font-weight:700;letter-spacing:.08em;color:#fff;text-transform:uppercase}
h1 span{color:var(--acc)}
.sub{font-family:var(--mono);font-size:.72rem;color:var(--mut);letter-spacing:.12em}
.dropzone{border:2px dashed var(--bdr);border-radius:6px;padding:2.5rem;text-align:center;cursor:pointer;transition:border-color .2s,background .2s;background:var(--sur)}
.dropzone:hover,.dropzone.drag{border-color:var(--acc);background:rgba(0,229,255,.04)}
.dropzone input{display:none}
.di{font-size:2.4rem;margin-bottom:.6rem}
.dl{font-size:1rem;font-weight:600}
.dh{font-family:var(--mono);font-size:.72rem;color:var(--mut);margin-top:.3rem}
#pw{display:none;margin-top:1rem;border:1px solid var(--bdr);border-radius:4px;overflow:hidden}
#prev{width:100%;max-height:240px;object-fit:contain;background:#000;display:block}
.btn{display:block;width:100%;margin-top:1.2rem;padding:.85rem;background:var(--acc);color:#000;font-family:var(--sans);font-size:1rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;border:none;border-radius:4px;cursor:pointer;transition:opacity .15s,transform .1s}
.btn:hover{opacity:.88}.btn:active{transform:scale(.98)}.btn:disabled{opacity:.4;cursor:not-allowed}
.spin{display:none;margin:2rem auto;text-align:center;font-family:var(--mono);color:var(--acc);font-size:.85rem}
.spin::before{content:'';display:block;width:36px;height:36px;border:3px solid var(--bdr);border-top-color:var(--acc);border-radius:50%;margin:0 auto .8rem;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
#results{display:none;margin-top:2rem}
.vbanner{padding:1.1rem 1.5rem;border-radius:6px;margin-bottom:1.5rem;display:flex;align-items:center;gap:1rem;border:1px solid}
.vbanner.normal{border-color:var(--grn);background:rgba(0,230,118,.07)}
.vbanner.anomalous{border-color:var(--red);background:rgba(255,61,90,.07)}
.vemoji{font-size:2rem}
.vlabel{font-size:1.5rem;font-weight:700;text-transform:uppercase}
.vsub{font-family:var(--mono);font-size:.75rem;color:var(--mut);margin-top:.2rem}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.5rem}
.mc{background:var(--sur);border:1px solid var(--bdr);border-radius:6px;padding:1rem 1.2rem}
.ml{font-family:var(--mono);font-size:.68rem;color:var(--mut);letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem}
.mv{font-size:1.5rem;font-weight:700;color:#fff}
.mv.ac{color:var(--acc)}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.panel{background:var(--sur);border:1px solid var(--bdr);border-radius:6px;overflow:hidden}
.pt{padding:.5rem .9rem;font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--acc);border-bottom:1px solid var(--bdr);background:rgba(0,229,255,.04)}
.panel img{width:100%;display:block;background:#000}
.crow{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem}
.cc{background:var(--sur);border:1px solid var(--bdr);border-radius:6px;padding:.8rem 1rem;font-family:var(--mono);font-size:.78rem}
.cc .cl{color:var(--acc);margin-bottom:.4rem;font-size:.68rem;text-transform:uppercase}
.err{background:rgba(255,61,90,.1);border:1px solid var(--red);border-radius:6px;padding:1rem 1.2rem;font-family:var(--mono);font-size:.82rem;color:var(--red);margin-top:1.2rem;display:none}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">👁</div>
    <div>
      <h1>Iris <span>Anomaly</span> Detection</h1>
      <div class="sub">ViT-MAE · Rubber-Sheet Normalization · Reconstruction Analysis</div>
    </div>
  </header>

  <div class="dropzone" id="dz">
    <input type="file" id="fi" accept="image/*">
    <div class="di">🔬</div>
    <div class="dl">Drop iris eye image here or click to upload</div>
    <div class="dh">PNG · JPG · BMP — full eye photograph</div>
    <div id="pw"><img id="prev" alt="Preview"></div>
  </div>

  <button class="btn" id="btn" disabled>▶ ANALYSE IRIS</button>
  <div class="spin" id="spin">Running anomaly detection…</div>
  <div class="err" id="err"></div>

  <div id="results">
    <div class="vbanner" id="vb">
      <div class="vemoji" id="ve"></div>
      <div>
        <div class="vlabel" id="vl"></div>
        <div class="vsub"   id="vs"></div>
      </div>
    </div>

    <div class="metrics">
      <div class="mc"><div class="ml">Anomaly Score</div><div class="mv ac" id="ms"></div></div>
      <div class="mc"><div class="ml">Threshold</div><div class="mv" id="mt"></div></div>
      <div class="mc"><div class="ml">Inference Time</div><div class="mv" id="mm"></div></div>
    </div>

    <div class="panels">
      <div class="panel"><div class="pt">📷 Annotated Eye</div><img id="ie" alt="Eye"></div>
      <div class="panel"><div class="pt">🌀 Iris Strip  64×512</div><img id="is" alt="Strip"></div>
      <div class="panel"><div class="pt">🔁 Reconstruction</div><img id="ir" alt="Recon"></div>
      <div class="panel"><div class="pt">🌡 Error Map (Jet)</div><img id="im" alt="Error"></div>
    </div>

    <div class="crow">
      <div class="cc"><div class="cl">🔵 Pupil</div><div id="pi">—</div></div>
      <div class="cc"><div class="cl">🟡 Iris Boundary</div><div id="ii">—</div></div>
    </div>
  </div>
</div>

<script>
const dz=document.getElementById('dz'),fi=document.getElementById('fi'),
      btn=document.getElementById('btn'),spin=document.getElementById('spin'),
      res=document.getElementById('results'),err=document.getElementById('err'),
      prev=document.getElementById('prev'),pw=document.getElementById('pw');
let file=null;

dz.addEventListener('click',()=>fi.click());
fi.addEventListener('change',e=>set(e.target.files[0]));
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('drag')});
dz.addEventListener('dragleave',()=>dz.classList.remove('drag'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('drag');if(e.dataTransfer.files[0])set(e.dataTransfer.files[0])});

function set(f){file=f;btn.disabled=false;prev.src=URL.createObjectURL(f);pw.style.display='block';res.style.display='none';err.style.display='none'}

btn.addEventListener('click',async()=>{
  if(!file)return;
  btn.disabled=true;spin.style.display='block';res.style.display='none';err.style.display='none';
  const fd=new FormData();fd.append('file',file);
  try{
    const r=await fetch('/predict',{method:'POST',body:fd});
    const d=await r.json();
    if(!r.ok)throw new Error(d.detail||r.statusText);

    const vb=document.getElementById('vb');
    vb.className='vbanner '+d.verdict;
    document.getElementById('ve').textContent=d.verdict_emoji;
    document.getElementById('vl').textContent=d.verdict.toUpperCase();
    document.getElementById('vs').textContent=
      `Score ${d.anomaly_score.toFixed(6)}  ·  Threshold ${d.threshold.toFixed(6)}  ·  Confidence ${d.confidence_pct}%`;
    document.getElementById('ms').textContent=d.anomaly_score.toFixed(6);
    document.getElementById('mt').textContent=d.threshold.toFixed(6);
    document.getElementById('mm').textContent=d.inference_ms.toFixed(1)+' ms';
    document.getElementById('ie').src='data:image/png;base64,'+d.eye_annotated_b64;
    document.getElementById('is').src='data:image/png;base64,'+d.strip_b64;
    document.getElementById('ir').src='data:image/png;base64,'+d.recon_b64;
    document.getElementById('im').src='data:image/png;base64,'+d.error_map_b64;
    document.getElementById('pi').textContent=`cx=${d.pupil.cx}  cy=${d.pupil.cy}  r=${d.pupil.r}`;
    document.getElementById('ii').textContent=`cx=${d.iris.cx}  cy=${d.iris.cy}  r=${d.iris.r}`;
    res.style.display='block';
  }catch(e){err.textContent='⚠  '+e.message;err.style.display='block';}
  finally{spin.style.display='none';btn.disabled=false;}
});
</script>
</body>
</html>"""

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, workers=1)