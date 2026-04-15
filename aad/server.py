"""API HTTP pour l'analyse vidéo."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aad.detector import analyze_video

APP_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = APP_ROOT / "static"

app = FastAPI(title="Détection d'actions anormales", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_scores(raw: list[float]) -> list[float]:
    if not raw:
        return []
    a = min(raw)
    b = max(raw)
    if b - a < 1e-12:
        return [0.5 for _ in raw]
    return [(x - a) / (b - a) for x in raw]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    window: int = Form(16),
    stride: int = Form(8),
):
    if not file.filename:
        raise HTTPException(400, "Fichier manquant")
    suffix = Path(file.filename).suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        raise HTTPException(400, "Format vidéo non supporté (utilisez mp4, avi, mov, mkv, webm).")

    data = await file.read()
    if len(data) > 500 * 1024 * 1024:
        raise HTTPException(413, "Fichier trop volumineux (max 500 Mo).")

    tmp = os.path.join(
        tempfile.gettempdir(),
        f"aad_{uuid.uuid4().hex}{suffix}",
    )
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        scores, meta = analyze_video(tmp, window=window, stride=stride)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"Analyse échouée: {e}") from e
    finally:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    raw_scores = [s.anomaly_score for s in scores]
    norm = _normalize_scores(raw_scores)
    payload = {
        "meta": meta,
        "windows": [
            {
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "start_sec": round(s.start_sec, 4),
                "end_sec": round(s.end_sec, 4),
                "anomaly_score": round(s.anomaly_score, 6),
                "anomaly_score_norm": round(norm[i], 6),
                "feature_delta_l2": round(s.feature_delta_l2, 6),
            }
            for i, s in enumerate(scores)
        ],
    }
    return JSONResponse(payload)


if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        return JSONResponse(
            {
                "message": "Interface non installée. Placez static/index.html ou appelez /api/analyze.",
            }
        )
    return FileResponse(index_path)
