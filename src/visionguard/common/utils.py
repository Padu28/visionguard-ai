"""Small shared utilities: seeding, timing, JSON helpers."""
from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Usage: ``with timer() as t: ...; t['ms']``"""
    out: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield out
    finally:
        out["ms"] = (time.perf_counter() - start) * 1000.0


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def to_jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=_default))
