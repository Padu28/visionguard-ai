"""Dataset materialisation, loading and the deterministic instance-level split."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from visionguard.common.config import PROJECT_ROOT, get_settings
from visionguard.datasets.synthetic import CLASSES, generate_dataset


@dataclass
class ImageSet:
    images: np.ndarray       # (N, H, W) float32 [0, 1]
    labels: np.ndarray       # (N,)
    masks: np.ndarray        # (N, H, W) uint8 (zeros when unavailable)
    meta: pd.DataFrame       # id, path, label, class_name, instance, split

    def subset(self, split: str) -> ImageSet:
        idx = np.flatnonzero(self.meta["split"].to_numpy() == split)
        return ImageSet(self.images[idx], self.labels[idx], self.masks[idx], self.meta.iloc[idx].reset_index(drop=True))

    def __len__(self) -> int:
        return len(self.labels)


def instance_split(instances: np.ndarray, fractions: dict[str, float], seed: int) -> np.ndarray:
    """Assign whole instances to splits so related crops never cross split boundaries."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(instances)
    rng.shuffle(uniq)
    n_tr = int(len(uniq) * fractions["train"])
    n_va = int(len(uniq) * fractions["validation"])
    assign = {u: "train" for u in uniq[:n_tr]}
    assign.update({u: "validation" for u in uniq[n_tr:n_tr + n_va]})
    assign.update({u: "test" for u in uniq[n_tr + n_va:]})
    return np.array([assign[i] for i in instances])


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def materialise(out_dir: Path | None = None) -> ImageSet:
    """Generate the synthetic dataset and write PNGs + masks + metadata.csv."""
    s = get_settings()
    cfg = s.data["synthetic"]
    imgs, labels, masks, inst = generate_dataset(cfg["n_instances"], cfg["crops_per_instance"], cfg["image_size"],
                                                 cfg["class_mix"], cfg["seed"])
    split = instance_split(inst, s.data["split"], cfg["seed"])
    out = Path(out_dir or s.image_dir)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (im, lb, mk, ins, sp) in enumerate(zip(imgs, labels, masks, inst, split)):
        name = f"img_{i:05d}_{CLASSES[lb]}.png"
        Image.fromarray(to_uint8(im)).save(out / name)
        Image.fromarray(mk * 255).save(out / "masks" / name)
        rows.append({"id": i, "path": name, "label": int(lb), "class_name": CLASSES[lb], "instance": int(ins),
                     "split": sp, "defect_area": float(mk.mean())})
    meta = pd.DataFrame(rows)
    meta.to_csv(out / "metadata.csv", index=False)
    return ImageSet(imgs, labels, masks, meta)


def load_image(path: str | Path, size: int | None = None) -> np.ndarray:
    im = Image.open(path).convert("L")
    if size and im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def load_dataset() -> ImageSet:
    """Load from data/processed (generate first if missing) or from a class-folder dataset."""
    s = get_settings()
    size = s.data["synthetic"]["image_size"]
    if s.data["source"] == "folder":
        root = PROJECT_ROOT / s.data["raw_dir"]
        rows, imgs = [], []
        for ci, cls in enumerate(s.classes):
            for p in sorted((root / cls).glob("*.png")) + sorted((root / cls).glob("*.jpg")):
                imgs.append(load_image(p, size))
                rows.append({"id": len(rows), "path": str(p), "label": ci, "class_name": cls, "instance": len(rows)})
        meta = pd.DataFrame(rows)
        meta["split"] = instance_split(meta.instance.to_numpy(), s.data["split"], 42)
        arr = np.stack(imgs)
        return ImageSet(arr, meta.label.to_numpy(), np.zeros_like(arr, dtype=np.uint8), meta)

    meta_path = s.image_dir / "metadata.csv"
    if not meta_path.exists():
        return materialise()
    meta = pd.read_csv(meta_path)
    imgs = np.stack([load_image(s.image_dir / p) for p in meta.path])
    masks = np.stack([(load_image(s.image_dir / "masks" / p) > 0.5).astype(np.uint8) for p in meta.path])
    return ImageSet(imgs, meta.label.to_numpy(), masks, meta)


def audit(ds: ImageSet) -> dict:
    m = ds.meta
    by_split = m.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    leak = set(m[m.split == "train"].instance) & set(m[m.split == "test"].instance)
    return {
        "n_images": len(m), "image_shape": list(ds.images.shape[1:]),
        "class_counts": m.class_name.value_counts().to_dict(),
        "by_split": {k: v for k, v in by_split.to_dict(orient="index").items()},
        "instances": int(m.instance.nunique()), "instance_leakage": len(leak),
        "pixel_mean": float(ds.images.mean()), "pixel_std": float(ds.images.std()),
        "mean_defect_area": m[m.label > 0].defect_area.mean() if "defect_area" in m else None,
    }
