"""Handcrafted image descriptors (no deep-learning dependency).

* intensity – global statistics + histogram (illumination-normalised)
* gradient  – Sobel magnitude statistics and orientation-weighted histogram
* hog       – cell-based histogram of oriented gradients (HOG-lite)
* lbp       – rotation-invariant uniform local binary patterns (P=8, R=1)
* fft       – radial and angular spectral energy bands (captures lines and periodic grain)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _norm(img: np.ndarray) -> np.ndarray:
    return (img - img.mean()) / (img.std() + 1e-6)


def intensity(img: np.ndarray) -> np.ndarray:
    z = _norm(img)
    hist = np.histogram(np.clip(z, -4, 4), bins=16, range=(-4, 4))[0] / z.size
    q = np.quantile(z, [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    return np.r_[hist, q, ((z - z.mean()) ** 3).mean(), ((z - z.mean()) ** 4).mean()]


def _sobel(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = _norm(ndimage.gaussian_filter(img, 0.7))
    gx, gy = ndimage.sobel(z, 1), ndimage.sobel(z, 0)
    return np.hypot(gx, gy), np.mod(np.arctan2(gy, gx), np.pi)


def gradient(img: np.ndarray) -> np.ndarray:
    mag, ang = _sobel(img)
    q = np.quantile(mag, [0.5, 0.9, 0.99, 0.999])
    oh = np.histogram(ang, bins=8, range=(0, np.pi), weights=mag)[0]
    oh = oh / (oh.sum() + 1e-9)
    return np.r_[q, mag.mean(), mag.std(), oh, oh.max() - oh.min()]


def hog(img: np.ndarray, cells: int = 4, bins: int = 9) -> np.ndarray:
    mag, ang = _sobel(img)
    h, w = img.shape
    ch, cw = h // cells, w // cells
    feats = []
    for i in range(cells):
        for j in range(cells):
            m = mag[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
            a = ang[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
            hh = np.histogram(a, bins=bins, range=(0, np.pi), weights=m)[0]
            feats.append(hh / (np.linalg.norm(hh) + 1e-6))
    f = np.stack(feats)
    # summarise spatially so features are translation-tolerant
    return np.r_[f.mean(0), f.max(0), f.std(0)]


def lbp(img: np.ndarray) -> np.ndarray:
    z = ndimage.gaussian_filter(img, 0.5)
    c = z[1:-1, 1:-1]
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    bits = np.stack([(z[1 + dy:z.shape[0] - 1 + dy, 1 + dx:z.shape[1] - 1 + dx] >= c) for dy, dx in offs]).astype(np.uint8)
    transitions = (bits != np.roll(bits, 1, axis=0)).sum(0)
    ones = bits.sum(0)
    code = np.where(transitions <= 2, ones, 9)   # uniform patterns 0..8, non-uniform = 9
    hist = np.bincount(code.ravel(), minlength=10) / code.size
    # local contrast of uniform-pattern map captures localised texture breaks
    blocks = code.reshape(code.shape[0] // 2, 2, code.shape[1] // 2, 2).mean((1, 3)) if code.shape[0] % 2 == 0 else code
    return np.r_[hist, blocks.std()]


def fft(img: np.ndarray) -> np.ndarray:
    z = _norm(img)
    F = np.abs(np.fft.fftshift(np.fft.fft2(z))) ** 2
    h, w = F.shape
    yy, xx = np.mgrid[-h // 2:h // 2, -w // 2:w // 2]
    r = np.hypot(yy, xx)
    th = np.mod(np.arctan2(yy, xx), np.pi)
    total = F.sum() + 1e-9
    radial = [F[(r >= a) & (r < b)].sum() / total for a, b in [(1, 3), (3, 6), (6, 12), (12, 24), (24, 48)]]
    ang = np.histogram(th[r > 3], bins=8, range=(0, np.pi), weights=F[r > 3])[0]
    ang = ang / (ang.sum() + 1e-9)
    return np.r_[radial, ang, ang.max() / (ang.mean() + 1e-9)]


def local_extremes(img: np.ndarray) -> np.ndarray:
    """Blob/line evidence: extreme responses of a Difference-of-Gaussians filter."""
    z = _norm(img)
    dog = ndimage.gaussian_filter(z, 1.0) - ndimage.gaussian_filter(z, 4.0)
    return np.r_[np.quantile(dog, [0.001, 0.01, 0.99, 0.999]), (np.abs(dog) > 1.0).mean()]


EXTRACTORS = {"intensity": intensity, "gradient": gradient, "hog": hog, "lbp": lbp, "fft": fft, "dog": local_extremes}


def extract(images: np.ndarray, names: list[str]) -> np.ndarray:
    fns = [EXTRACTORS[n] for n in names]
    return np.stack([np.concatenate([f(im) for f in fns]) for im in images]).astype(np.float32)
