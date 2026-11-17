"""VisionGuard AI REST API.

Run:  uvicorn api.main:app --reload --port 8002     → docs at http://localhost:8002/docs
"""
from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from visionguard import __version__  # noqa: E402
from visionguard.common.config import get_settings  # noqa: E402
from visionguard.common.utils import to_jsonable  # noqa: E402
from visionguard.explainability.maps import overlay  # noqa: E402
from visionguard.inference.predictor import HUMAN_REVIEW, Predictor  # noqa: E402
from visionguard.storage.database import Database, utcnow  # noqa: E402

MAX_BYTES = 8 * 1024 * 1024
app = FastAPI(title="VisionGuard AI API", version=__version__,
              description="Surface-defect classification with occlusion heatmaps. " + HUMAN_REVIEW)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])
settings = get_settings()
db = Database(settings.db_path)
_predictors: dict[str, Predictor] = {}


def predictor(name: str) -> Predictor:
    if name not in _predictors:
        try:
            _predictors[name] = Predictor(name)
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc
    return _predictors[name]


class PredictionOut(BaseModel):
    label: str
    confidence: float
    defect_score: float
    probabilities: dict[str, float]
    needs_human_review: bool
    model: str
    model_version: str
    inference_ms: float
    heatmap_png_base64: str | None = None
    disclaimer: str = HUMAN_REVIEW


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "time": utcnow(),
            "models": [p.stem for p in settings.checkpoints_dir.glob("*.joblib")]}


@app.post("/predict", response_model=PredictionOut, tags=["inference"])
async def predict(file: UploadFile = File(..., description="PNG/JPG surface image"),
                  model: Literal["improved", "robust", "baseline"] = Query("improved"),
                  explain: bool = Query(False, description="Return an occlusion heatmap overlay (slower)")) -> PredictionOut:
    data = await file.read()
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(413 if data else 422, "Image must be between 1 byte and 8 MB.")
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception as exc:
        raise HTTPException(422, "File is not a readable image.") from exc
    p = predictor(model)
    res = p.predict(data, explain=explain)
    heat_b64 = None
    if res.heatmap is not None:
        buf = io.BytesIO()
        Image.fromarray(overlay(p.preprocess(data), res.heatmap)).save(buf, format="PNG")
        heat_b64 = base64.b64encode(buf.getvalue()).decode()
    db.log_inference("api", model, res.label, res.confidence, res.defect_score, res.latency_ms)
    return PredictionOut(label=res.label, confidence=round(res.confidence, 4), defect_score=round(res.defect_score, 4),
                         probabilities={k: round(v, 4) for k, v in res.probabilities.items()},
                         needs_human_review=res.needs_review, model=model, model_version=f"vg-{__version__}-{model}",
                         inference_ms=round(res.latency_ms, 2), heatmap_png_base64=heat_b64)


@app.get("/metrics", tags=["research"])
def metrics() -> dict[str, Any]:
    s = db.get_meta("summary")
    if not s:
        raise HTTPException(404, "Run the pipeline first.")
    return to_jsonable({k: s[k] for k in ("models", "explanations", "audit", "task")})


@app.get("/robustness", tags=["research"])
def robustness(model: Literal["baseline", "improved", "robust"] | None = None) -> list[dict[str, Any]]:
    sql, params = "SELECT * FROM robustness", []
    if model:
        sql += " WHERE model=?"
        params.append(model)
    return to_jsonable(db.query_df(sql, params).to_dict(orient="records"))


@app.get("/inference-log", tags=["system"])
def inference_log(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    return to_jsonable(db.query_df("SELECT * FROM inference_log ORDER BY id DESC LIMIT ?", (limit,)).to_dict(orient="records"))
