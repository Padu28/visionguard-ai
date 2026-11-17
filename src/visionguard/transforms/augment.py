"""Meaning-preserving augmentations and controlled corruptions for robustness tests."""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


# ------------------------------------------------------------- training augmentation
def augment(img: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    out = img
    if rng.random() < cfg.get("hflip", 0):
        out = out[:, ::-1]
    if rng.random() < cfg.get("vflip", 0):
        out = out[::-1, :]
    if cfg.get("rot90"):
        out = np.rot90(out, rng.integers(0, 4))
    b = cfg.get("brightness", 0)
    c = cfg.get("contrast", 0)
    out = (out - out.mean()) * (1 + rng.uniform(-c, c)) + out.mean() + rng.uniform(-b, b)
    out = out + rng.normal(0, cfg.get("gaussian_noise_std", 0), out.shape)
    return np.clip(out, 0, 1).astype(np.float32)


def augment_batch(imgs: np.ndarray, labels: np.ndarray, cfg: dict, copies: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)
    aug = [augment(im, rng, cfg) for _ in range(copies) for im in imgs]
    return np.concatenate([imgs, np.stack(aug)]), np.concatenate([labels] + [labels] * copies)


# ------------------------------------------------------------- corruptions
def brightness(img: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(img + delta, 0, 1)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return img if sigma <= 0 else ndimage.gaussian_filter(img, sigma)


def noise(img: np.ndarray, std: float, seed: int = 0) -> np.ndarray:
    if std <= 0:
        return img
    return np.clip(img + np.random.default_rng(seed).normal(0, std, img.shape), 0, 1)


def resolution(img: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1:
        return img
    h, w = img.shape
    small = Image.fromarray((img * 255).astype(np.uint8)).resize((max(4, int(w * scale)), max(4, int(h * scale))), Image.BILINEAR)
    return np.asarray(small.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0


def corruption_augment(img: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Robustness-aware augmentation: apply one random, mild corruption (or none)."""
    kind = rng.choice(["none", "blur", "noise", "resolution", "brightness"])
    if kind == "blur":
        return blur(img, rng.uniform(0, cfg["max_blur_sigma"])).astype(np.float32)
    if kind == "noise":
        return noise(img, rng.uniform(0, cfg["max_noise_std"]), seed=int(rng.integers(1 << 31))).astype(np.float32)
    if kind == "resolution":
        return resolution(img, rng.uniform(cfg["min_resolution_scale"], 1.0)).astype(np.float32)
    if kind == "brightness":
        return brightness(img, rng.uniform(-cfg["max_brightness"], cfg["max_brightness"])).astype(np.float32)
    return img


def robust_augment_batch(imgs: np.ndarray, labels: np.ndarray, aug_cfg: dict, corr_cfg: dict, copies: int = 2,
                         seed: int = 0):
    rng = np.random.default_rng(seed)
    out = [augment(corruption_augment(im, rng, corr_cfg), rng, aug_cfg) for _ in range(copies) for im in imgs]
    return np.concatenate([imgs, np.stack(out)]), np.concatenate([labels] + [labels] * copies)


CORRUPTIONS = {
    "brightness": brightness,
    "blur_sigma": blur,
    "noise_std": noise,
    "resolution_scale": resolution,
}


def corrupt_batch(imgs: np.ndarray, kind: str, level: float) -> np.ndarray:
    fn = CORRUPTIONS[kind]
    if kind == "noise_std":
        return np.stack([fn(im, level, seed=i) for i, im in enumerate(imgs)]).astype(np.float32)
    return np.stack([fn(im, level) for im in imgs]).astype(np.float32)
