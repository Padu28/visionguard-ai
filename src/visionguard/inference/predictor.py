"""Inference wrapper used by the API and dashboard."""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from visionguard.common.config import get_settings
from visionguard.explainability.maps import occlusion_map

HUMAN_REVIEW = ("Decision support only — route every flagged or low-confidence item to a qualified human "
                "inspector. Do not use as the sole basis for accept/reject decisions.")


@dataclass
class Prediction:
    label: str
    confidence: float
    defect_score: float
    probabilities: dict[str, float]
    latency_ms: float
    model: str
    needs_review: bool
    heatmap: np.ndarray | None = None


class Predictor:
    def __init__(self, model_name: str = "improved", path: Path | None = None) -> None:
        s = get_settings()
        self.settings = s
        self.size = s.data["synthetic"]["image_size"]
        self.path = path or s.checkpoints_dir / f"{model_name}.joblib"
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} missing — run `python -m visionguard.cli pipeline`.")
        self.model = joblib.load(self.path)
        self.name = model_name

    def preprocess(self, data: bytes | np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(data, np.ndarray):
            img = Image.fromarray((np.clip(data, 0, 1) * 255).astype(np.uint8)) if data.dtype != np.uint8 else Image.fromarray(data)
        elif isinstance(data, Image.Image):
            img = data
        else:
            img = Image.open(io.BytesIO(data))
        img = img.convert("L")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def predict(self, data, explain: bool = False, review_below: float = 0.7) -> Prediction:
        img = self.preprocess(data)
        t0 = time.perf_counter()
        proba = self.model.predict_proba(img[None])[0]
        ms = (time.perf_counter() - t0) * 1000
        k = int(proba.argmax())
        heat = None
        if explain:
            cfg = self.settings.training["explainability"]
            heat, _, _ = occlusion_map(self.model.predict_proba, img, k, cfg["occlusion_patch"], cfg["occlusion_stride"])
        label = self.settings.classes[k]
        return Prediction(label=label, confidence=float(proba[k]), defect_score=float(1 - proba[0]),
                          probabilities={c: float(p) for c, p in zip(self.settings.classes, proba)}, latency_ms=ms,
                          model=self.name, needs_review=bool(label != "normal" or proba[k] < review_below), heatmap=heat)
