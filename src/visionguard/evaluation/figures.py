"""Report figures: sample gallery, robustness curves, confusion matrix, error gallery with heatmaps."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from visionguard.common.config import get_settings  # noqa: E402
from visionguard.explainability.maps import occlusion_map  # noqa: E402

COL = {"baseline": "#94a3b8", "improved": "#10b981", "robust": "#f59e0b"}


def make_figures(ds, models: dict, rob: pd.DataFrame, out: Path) -> None:
    s = get_settings()
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    te = ds.subset("test")

    # 1) sample gallery
    fig, ax = plt.subplots(2, 5, figsize=(12, 5))
    for c, name in enumerate(s.classes):
        idx = np.flatnonzero(ds.labels == c)[:1]
        ax[0, c].imshow(ds.images[idx[0]], cmap="gray", vmin=0, vmax=1)
        ax[0, c].set_title(name)
        ax[1, c].imshow(ds.masks[idx[0]], cmap="magma")
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle("Synthetic surfaces (top) and ground-truth defect masks (bottom)")
    fig.tight_layout()
    fig.savefig(out / "sample_gallery.png")
    plt.close(fig)

    # 2) robustness curves
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for a, (kind, g) in zip(axes, rob.groupby("corruption")):
        for m, gg in g.groupby("model"):
            a.plot(gg.level, gg.macro_f1, "o-", color=COL[m], label=m, lw=2)
        a.set(title=kind, xlabel="severity", ylabel="macro-F1", ylim=(0, 1))
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "robustness_curves.png")
    plt.close(fig)

    # 3) confusion matrix (improved)
    m = models["improved"]
    pred = m.predict(te.images)
    cm = pd.crosstab(pd.Series(te.labels, name="true"), pd.Series(pred, name="pred")).reindex(
        index=range(5), columns=range(5), fill_value=0)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm.values, cmap="Greens")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm.values[i, j], ha="center", va="center", color="black" if cm.values[i, j] < cm.values.max() / 2 else "white")
    ax.set_xticks(range(5), s.classes, rotation=30)
    ax.set_yticks(range(5), s.classes)
    ax.set(title="Confusion matrix — improved (test)", xlabel="predicted", ylabel="true")
    fig.tight_layout()
    fig.savefig(out / "confusion_improved.png")
    plt.close(fig)

    # 4) error gallery: correct, false negative, false positive with occlusion heatmaps
    cases = []
    ok = np.flatnonzero((pred == te.labels) & (te.labels > 0))[:2]
    fn = np.flatnonzero((pred == 0) & (te.labels > 0))[:2]
    fp = np.flatnonzero((pred > 0) & (te.labels == 0))[:2]
    for tag, idx in (("correct", ok), ("missed defect (FN)", fn), ("false alarm (FP)", fp)):
        cases += [(tag, i) for i in idx]
    if cases:
        fig, ax = plt.subplots(2, len(cases), figsize=(3 * len(cases), 6.2), squeeze=False)
        for k, (tag, i) in enumerate(cases):
            heat, tgt, p = occlusion_map(m.predict_proba, te.images[i])
            ax[0, k].imshow(te.images[i], cmap="gray", vmin=0, vmax=1)
            ax[0, k].contour(te.masks[i], levels=[0.5], colors="#22d3ee", linewidths=0.8)
            ax[0, k].set_title(f"{tag}\ntrue {s.classes[te.labels[i]]}", fontsize=9)
            ax[1, k].imshow(te.images[i], cmap="gray", vmin=0, vmax=1)
            ax[1, k].imshow(heat, cmap="magma", alpha=0.5)
            ax[1, k].set_title(f"pred {s.classes[tgt]} ({p:.2f})", fontsize=9)
        for a in ax.ravel():
            a.axis("off")
        fig.tight_layout()
        fig.savefig(out / "error_gallery.png")
        plt.close(fig)
