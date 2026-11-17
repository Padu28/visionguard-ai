import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def small():
    from visionguard.datasets.synthetic import generate_dataset

    mix = {"normal": 0.4, "scratch": 0.15, "inclusion": 0.15, "crack": 0.15, "patch": 0.15}
    return generate_dataset(n_instances=60, crops_per_instance=5, size=96, class_mix=mix, seed=3)
