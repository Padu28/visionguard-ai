"""Transfer-learning training loop (optional — requires PyTorch).

Freeze backbone → train head → unfreeze → fine-tune, with class-weighted loss,
augmentation, early stopping on validation macro-F1 and a saved best checkpoint.

    python -m visionguard.cli train-cnn
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import f1_score

from visionguard.common.config import get_settings
from visionguard.common.logging import get_logger
from visionguard.datasets.dataset import load_dataset
from visionguard.transforms.augment import augment

log = get_logger("visionguard.cnn")


def train_cnn() -> dict:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    from visionguard.models.cnn import build, set_backbone_trainable

    s = get_settings()
    cfg = s.training["cnn"]
    torch.manual_seed(s.training["random_state"])
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ds = load_dataset()
    tr, va = ds.subset("train"), ds.subset("validation")
    mean, std = float(tr.images.mean()), float(tr.images.std())

    class Images(Dataset):
        def __init__(self, x, y, train: bool) -> None:
            self.x, self.y, self.train = x, y, train
            self.rng = np.random.default_rng(0)

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, i):
            im = augment(self.x[i], self.rng, s.augmentation["train"]) if self.train else self.x[i]
            return torch.from_numpy(((im - mean) / std)[None].copy()).float(), int(self.y[i])

    dl_tr = DataLoader(Images(tr.images, tr.labels, True), batch_size=cfg["batch_size"], shuffle=True)
    dl_va = DataLoader(Images(va.images, va.labels, False), batch_size=128)
    model = build(cfg["architecture"], len(s.classes), cfg["pretrained"]).to(device)
    counts = np.bincount(tr.labels, minlength=len(s.classes))
    weights = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights if cfg["class_weighted_loss"] else None)

    def evaluate() -> float:
        model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in dl_va:
                preds.append(model(x.to(device)).argmax(1).cpu().numpy())
        return float(f1_score(va.labels, np.concatenate(preds), average="macro"))

    best, best_epoch, history = -1.0, 0, []
    ckpt = s.checkpoints_dir / f"{cfg['architecture']}_best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    set_backbone_trainable(model, False)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"] * 3,
                            weight_decay=cfg["weight_decay"])
    for epoch in range(cfg["epochs"]):
        if epoch == cfg["freeze_backbone_epochs"]:
            set_backbone_trainable(model, True)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        model.train()
        t0, total = time.time(), 0.0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        f1 = evaluate()
        history.append({"epoch": epoch, "loss": total / len(tr), "val_macro_f1": f1, "seconds": time.time() - t0})
        log.info("epoch %02d loss %.4f val macro-F1 %.4f", epoch, total / len(tr), f1)
        if f1 > best:
            best, best_epoch = f1, epoch
            torch.save({"state_dict": model.state_dict(), "arch": cfg["architecture"], "mean": mean, "std": std,
                        "classes": s.classes}, ckpt)
        elif epoch - best_epoch >= cfg["early_stopping_patience"]:
            log.info("Early stopping at epoch %d", epoch)
            break
    return {"best_val_macro_f1": best, "best_epoch": best_epoch, "history": history, "checkpoint": str(ckpt)}
