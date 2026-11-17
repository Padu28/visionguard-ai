"""VisionGuard CLI.

    python -m visionguard.cli pipeline        # dataset → models → robustness → explanations → reports
    python -m visionguard.cli train-cnn       # optional PyTorch transfer learning (requires torch)
    python -m visionguard.cli export-samples  # copy a few PNGs per class to data/sample/
"""
from __future__ import annotations

import argparse
import shutil

from visionguard.common.config import PROJECT_ROOT, get_settings
from visionguard.common.logging import get_logger

log = get_logger("visionguard.cli")


def export_samples(per_class: int = 4) -> None:
    import pandas as pd

    s = get_settings()
    meta = pd.read_csv(s.image_dir / "metadata.csv")
    out = PROJECT_ROOT / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    for _c, g in meta[meta.split == "test"].groupby("class_name"):
        for p in g.path.head(per_class):
            shutil.copy(s.image_dir / p, out / p)
    log.info("Exported samples to %s", out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="visionguard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pipeline")
    sub.add_parser("train-cnn")
    sub.add_parser("export-samples")
    a = ap.parse_args(argv)
    if a.cmd == "pipeline":
        from visionguard.training.pipeline import run_pipeline

        run_pipeline()
        export_samples()
    elif a.cmd == "train-cnn":
        from visionguard.training.train_cnn import train_cnn

        log.info("CNN result: %s", {k: v for k, v in train_cnn().items() if k != "history"})
    else:
        export_samples()


if __name__ == "__main__":
    main()
