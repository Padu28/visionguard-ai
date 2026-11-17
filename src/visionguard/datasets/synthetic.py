"""Synthetic steel-surface images with pixel-accurate defect masks.

Each *instance* is a steel sheet with its own texture (grain direction, roughness,
illumination gradient). Several crops are taken per sheet, so the train/val/test
split is done BY INSTANCE to prevent near-duplicate leakage.

Defect classes (one per image):
  scratch   – thin straight bright/dark line
  inclusion – a few dark elliptical spots
  crack     – jagged dark poly-line with an optional branch
  patch     – large, low-contrast irregular stain

SYNTHETIC data: suitable for method development and tests, not as evidence of
performance on real production imagery.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

CLASSES = ("normal", "scratch", "inclusion", "crack", "patch")


@dataclass
class SheetTexture:
    angle: float
    roughness: float
    grain: float
    base: float
    gradient: tuple[float, float]


def _texture(rng: np.random.Generator, size: int, t: SheetTexture) -> np.ndarray:
    noise = ndimage.gaussian_filter(rng.normal(0, 1, (size * 2, size * 2)), sigma=t.roughness)
    stripes = ndimage.gaussian_filter1d(rng.normal(0, 1, (size * 2, size * 2)), sigma=6, axis=1)
    tex = noise / (noise.std() + 1e-9) * 0.5 + stripes / (stripes.std() + 1e-9) * t.grain
    tex = ndimage.rotate(tex, t.angle, reshape=False, mode="reflect")
    c = size // 2
    tex = tex[c:c + size, c:c + size]
    yy, xx = np.mgrid[0:size, 0:size] / size
    illum = t.gradient[0] * (xx - 0.5) + t.gradient[1] * (yy - 0.5)
    img = t.base + 0.06 * tex + illum
    return img


def _line_mask(size: int, p0, p1, width: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy + 1e-9
    tpar = np.clip(((xx - x0) * dx + (yy - y0) * dy) / L2, 0, 1)
    d = np.hypot(xx - (x0 + tpar * dx), yy - (y0 + tpar * dy))
    return np.clip(1 - d / width, 0, 1)


def _defect(rng: np.random.Generator, img: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    size = img.shape[0]
    mask = np.zeros_like(img)
    if kind == "scratch":
        a = rng.uniform(0, np.pi)
        L = rng.uniform(0.45, 0.9) * size
        cx, cy = rng.uniform(0.3, 0.7, 2) * size
        p0 = (cx - np.cos(a) * L / 2, cy - np.sin(a) * L / 2)
        p1 = (cx + np.cos(a) * L / 2, cy + np.sin(a) * L / 2)
        m = _line_mask(size, p0, p1, rng.uniform(0.9, 1.6))
        img = img + rng.choice([-1, 1]) * rng.uniform(0.10, 0.18) * m
        mask = m
    elif kind == "inclusion":
        yy, xx = np.mgrid[0:size, 0:size]
        for _ in range(rng.integers(2, 6)):
            cx, cy = rng.uniform(0.15, 0.85, 2) * size
            rx, ry = rng.uniform(1.5, 4.5, 2)
            m = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
            img = img - rng.uniform(0.12, 0.22) * m
            mask = np.maximum(mask, m)
    elif kind == "crack":
        pts = [rng.uniform(0.1, 0.3, 2) * size]
        direction = rng.uniform(0, 2 * np.pi)
        for _ in range(rng.integers(6, 10)):
            direction += rng.normal(0, 0.6)
            step = rng.uniform(5, 10)
            pts.append(np.clip(pts[-1] + step * np.array([np.cos(direction), np.sin(direction)]), 2, size - 3))
        m = np.zeros_like(img)
        for p0, p1 in zip(pts[:-1], pts[1:]):
            m = np.maximum(m, _line_mask(size, p0, p1, rng.uniform(0.8, 1.3)))
        if rng.random() < 0.6:
            k = rng.integers(2, len(pts) - 1)
            b = pts[k] + rng.uniform(-14, 14, 2)
            m = np.maximum(m, _line_mask(size, pts[k], np.clip(b, 2, size - 3), 0.9))
        img = img - rng.uniform(0.14, 0.24) * m
        mask = m
    elif kind == "patch":
        blob = ndimage.gaussian_filter(rng.normal(0, 1, img.shape), sigma=rng.uniform(7, 11))
        thr = np.quantile(blob, rng.uniform(0.80, 0.90))
        m = ndimage.gaussian_filter((blob > thr).astype(float), 1.5)
        img = img + rng.choice([-1, 1]) * rng.uniform(0.07, 0.12) * m
        mask = m
    return img, (mask > 0.25).astype(np.uint8)


def generate_dataset(n_instances: int, crops_per_instance: int, size: int, class_mix: dict[str, float],
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return images (N,H,W float32 in [0,1]), labels (N,), masks (N,H,W uint8), instance ids (N,)."""
    rng = np.random.default_rng(seed)
    names = list(class_mix)
    probs = np.array([class_mix[k] for k in names], dtype=float)
    probs /= probs.sum()
    imgs, labels, masks, inst = [], [], [], []
    for i in range(n_instances):
        tex = SheetTexture(angle=rng.uniform(-8, 8) + rng.choice([0, 90]), roughness=rng.uniform(0.8, 2.0),
                           grain=rng.uniform(0.2, 0.6), base=rng.uniform(0.42, 0.62),
                           gradient=(rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08)))
        for _ in range(crops_per_instance):
            kind = names[rng.choice(len(names), p=probs)]
            img = _texture(rng, size, tex)
            m = np.zeros((size, size), np.uint8)
            if kind != "normal":
                img, m = _defect(rng, img, kind)
            img = img + rng.normal(0, 0.008, img.shape)
            imgs.append(np.clip(img, 0, 1).astype(np.float32))
            labels.append(CLASSES.index(kind))
            masks.append(m)
            inst.append(i)
    return np.stack(imgs), np.array(labels), np.stack(masks), np.array(inst)
