"""API tests — skipped without FastAPI/httpx or trained checkpoints."""
import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import api.main as main

    if not (main.settings.checkpoints_dir / "improved.joblib").exists():
        pytest.skip("Run `python -m visionguard.cli pipeline` first")
    return TestClient(main.app)


def _png() -> bytes:
    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray((np.random.default_rng(0).random((96, 96)) * 255).astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict(client):
    r = client.post("/predict?explain=true", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in body["probabilities"] and body["heatmap_png_base64"]


def test_rejects_non_image(client):
    assert client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 422
