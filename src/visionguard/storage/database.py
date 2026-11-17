"""SQLite store for experiment results and the inference log."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value_json TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS models (
    name TEXT PRIMARY KEY, version TEXT, kind TEXT, path TEXT, n_features INTEGER, size_kb REAL,
    latency_ms_p50 REAL, latency_ms_p95 REAL, metrics_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS predictions (
    image_id INTEGER, model TEXT, split TEXT, label INTEGER, pred INTEGER, confidence REAL,
    defect_score REAL, probs_json TEXT, PRIMARY KEY (image_id, model));
CREATE TABLE IF NOT EXISTS robustness (
    model TEXT, corruption TEXT, level REAL, macro_f1 REAL, defect_recall REAL, false_alarm_rate REAL,
    accuracy REAL);
CREATE TABLE IF NOT EXISTS explanations (
    image_id INTEGER, model TEXT, target INTEGER, pointing_hit INTEGER, energy_in_mask REAL, mask_area REAL);
CREATE TABLE IF NOT EXISTS inference_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, source TEXT, model TEXT, pred TEXT, confidence REAL,
    defect_score REAL, latency_ms REAL);
"""


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def reset(self) -> None:
        with self.connect() as c:
            for t in ("meta", "models", "predictions", "robustness", "explanations"):
                c.execute(f"DELETE FROM {t}")

    def put_meta(self, key: str, value: Any) -> None:
        with self.connect() as c:
            c.execute("INSERT OR REPLACE INTO meta VALUES (?, ?, ?)", (key, json.dumps(value, default=str), utcnow()))

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self.connect() as c:
            r = c.execute("SELECT value_json FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def write_df(self, table: str, df: pd.DataFrame) -> None:
        with self.connect() as c:
            df.to_sql(table, c, if_exists="append", index=False)

    def query_df(self, sql: str, params: tuple | list = ()) -> pd.DataFrame:
        with self.connect() as c:
            return pd.read_sql_query(sql, c, params=params)

    def log_inference(self, source: str, model: str, pred: str, confidence: float, defect_score: float,
                      latency_ms: float) -> None:
        with self.connect() as c:
            c.execute("INSERT INTO inference_log (ts, source, model, pred, confidence, defect_score, latency_ms) "
                      "VALUES (?,?,?,?,?,?,?)", (utcnow(), source, model, pred, confidence, defect_score, latency_ms))
