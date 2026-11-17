"""Configuration loading and project paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.getenv("VISIONGUARD_ROOT", Path(__file__).resolve().parents[3]))
CONFIG_DIR = PROJECT_ROOT / "configs"


def _load(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Settings:
    data: dict[str, Any]
    augmentation: dict[str, Any]
    training: dict[str, Any]

    @property
    def classes(self) -> list[str]:
        return list(self.data["classes"])

    @property
    def image_dir(self) -> Path:
        return PROJECT_ROOT / self.data["processed_dir"]

    @property
    def db_path(self) -> Path:
        o = os.getenv("VISIONGUARD_DB")
        return Path(o) if o else PROJECT_ROOT / self.training["storage"]["database"]

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / self.training["storage"]["reports_dir"]

    @property
    def checkpoints_dir(self) -> Path:
        return PROJECT_ROOT / self.training["storage"]["checkpoints_dir"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(_load("data.yaml"), _load("augmentation.yaml"), _load("training.yaml"))
