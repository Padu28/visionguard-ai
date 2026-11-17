"""Cached access to VisionGuard results, models and images."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from visionguard.common.config import get_settings  # noqa: E402
from visionguard.datasets.dataset import load_image  # noqa: E402
from visionguard.storage.database import Database  # noqa: E402

settings = get_settings()


def db() -> Database:
    return Database(settings.db_path)


def ensure_pipeline() -> None:
    if db().get_meta("summary") is None or not (settings.checkpoints_dir / "improved.joblib").exists():
        with st.status("First launch — generating images and training models (≈4 min)…", expanded=True) as s:
            from visionguard.cli import export_samples
            from visionguard.training.pipeline import run_pipeline

            st.write("Generating 1 500 synthetic surfaces with defect masks · training 3 models · robustness study · "
                     "explanation verification…")
            run_pipeline()
            export_samples()
            s.update(label="Pipeline complete", state="complete")
        st.cache_data.clear()


@st.cache_data(ttl=300)
def summary() -> dict:
    return db().get_meta("summary", {})


@st.cache_data(ttl=300)
def metadata() -> pd.DataFrame:
    return pd.read_csv(settings.image_dir / "metadata.csv")


@st.cache_data(ttl=300)
def models_table() -> pd.DataFrame:
    m = db().query_df("SELECT * FROM models")
    m["metrics"] = m.metrics_json.apply(json.loads)
    return m


@st.cache_data(ttl=300)
def predictions(model: str) -> pd.DataFrame:
    p = db().query_df("SELECT * FROM predictions WHERE model=?", (model,))
    p["probs"] = p.probs_json.apply(json.loads)
    return p.merge(metadata()[["id", "path", "class_name", "instance", "defect_area"]], left_on="image_id", right_on="id")


@st.cache_data(ttl=300)
def robustness() -> pd.DataFrame:
    return db().query_df("SELECT * FROM robustness")


@st.cache_data(ttl=300)
def explanations() -> pd.DataFrame:
    return db().query_df("SELECT * FROM explanations")


@st.cache_data(ttl=60)
def inference_log() -> pd.DataFrame:
    return db().query_df("SELECT * FROM inference_log ORDER BY id DESC LIMIT 200")


@st.cache_data(max_entries=512)
def image(path: str) -> np.ndarray:
    return load_image(settings.image_dir / path)


@st.cache_data(max_entries=512)
def mask(path: str) -> np.ndarray:
    return (load_image(settings.image_dir / "masks" / path) > 0.5).astype(np.uint8)


@st.cache_resource
def predictor(name: str):
    from visionguard.inference.predictor import Predictor

    return Predictor(name)
