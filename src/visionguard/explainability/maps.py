"""Visual explanations and their verification against ground-truth masks.

* ``occlusion_map`` – model-agnostic: slide a grey patch, record the drop in the
  predicted-class probability. Works for any ``predict_proba(images)`` model.
* ``gradcam``       – for PyTorch CNNs (optional dependency).
* ``pointing_game`` / ``energy_in_mask`` – quantify whether the heatmap actually
  lands on the defect. Heatmaps are diagnostic aids, not proof of causal reasoning.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import ndimage


def occlusion_map(predict_proba: Callable[[np.ndarray], np.ndarray], img: np.ndarray, target: int | None = None,
                  patch: int = 16, stride: int = 8) -> tuple[np.ndarray, int, float]:
    base = predict_proba(img[None])[0]
    target = int(base.argmax()) if target is None else target
    h, w = img.shape
    fill = float(np.median(img))
    coords = [(y, x) for y in range(0, h - patch + 1, stride) for x in range(0, w - patch + 1, stride)]
    batch = np.repeat(img[None], len(coords), axis=0)
    for k, (y, x) in enumerate(coords):
        batch[k, y:y + patch, x:x + patch] = fill
    probs = predict_proba(batch)[:, target]
    heat = np.zeros_like(img, dtype=np.float32)
    count = np.zeros_like(img, dtype=np.float32)
    for (y, x), p in zip(coords, probs):
        heat[y:y + patch, x:x + patch] += base[target] - p
        count[y:y + patch, x:x + patch] += 1
    heat = np.maximum(heat / np.maximum(count, 1), 0)
    heat = ndimage.gaussian_filter(heat, stride / 2)
    if heat.max() > 0:
        heat /= heat.max()
    return heat, target, float(base[target])


def pointing_game(heat: np.ndarray, mask: np.ndarray, tolerance: int = 4) -> bool:
    """Hit if the heatmap maximum falls inside the (slightly dilated) defect mask."""
    if mask.sum() == 0:
        return False
    m = ndimage.binary_dilation(mask.astype(bool), iterations=tolerance)
    y, x = np.unravel_index(np.argmax(heat), heat.shape)
    return bool(m[y, x])


def energy_in_mask(heat: np.ndarray, mask: np.ndarray, tolerance: int = 4) -> float:
    """Share of heatmap energy inside the dilated mask (random baseline ≈ mask area)."""
    if heat.sum() == 0 or mask.sum() == 0:
        return 0.0
    m = ndimage.binary_dilation(mask.astype(bool), iterations=tolerance)
    return float(heat[m].sum() / heat.sum())


def gradcam(model, layer, x, target: int | None = None) -> np.ndarray:  # pragma: no cover - requires torch
    """Grad-CAM for a PyTorch model. ``x``: (1, 1, H, W) tensor."""
    import torch

    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.eval()
    out = model(x)
    target = int(out.argmax(1)) if target is None else target
    model.zero_grad()
    out[0, target].backward()
    h1.remove()
    h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return cam / (cam.max() + 1e-9)


def overlay(img: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """RGB uint8 overlay (heat in a magma-like ramp) for display."""
    g = np.stack([img] * 3, -1)
    ramp = np.stack([np.clip(1.6 * heat, 0, 1), np.clip(1.6 * heat - 0.6, 0, 1), np.clip(0.6 - heat, 0, 0.6) * heat * 2], -1)
    out = (1 - alpha * heat[..., None]) * g + alpha * heat[..., None] * ramp
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)
