"""Classification metrics with per-class detail and a binary defect view."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate(y: np.ndarray, proba: np.ndarray, classes: list[str]) -> dict:
    pred = proba.argmax(1)
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=range(len(classes)), zero_division=0)
    defect_true = (y != 0).astype(int)
    defect_score = 1 - proba[:, 0]               # P(any defect) = 1 − P(normal)
    defect_pred = (pred != 0).astype(int)
    fn_defects = int(((defect_true == 1) & (defect_pred == 0)).sum())
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(n[i])}
                      for i, c in enumerate(classes)},
        "confusion": confusion_matrix(y, pred, labels=range(len(classes))).tolist(),
        "defect_pr_auc": float(average_precision_score(defect_true, defect_score)),
        "defect_roc_auc": float(roc_auc_score(defect_true, defect_score)),
        "defect_recall": float(defect_pred[defect_true == 1].mean()),
        "false_alarm_rate": float(defect_pred[defect_true == 0].mean()),
        "missed_defects": fn_defects,
    }


def bootstrap_macro_f1(y: np.ndarray, pred: np.ndarray, n: int = 300, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = [f1_score(y[i], pred[i], average="macro") for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
