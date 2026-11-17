"""End-to-end experiment: dataset → audit → baseline vs improved → test metrics →
robustness study → explanation verification → latency/size → persist & report."""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from visionguard import __version__
from visionguard.common.config import get_settings
from visionguard.common.logging import get_logger
from visionguard.common.utils import dump_json, to_jsonable
from visionguard.datasets.dataset import audit, materialise
from visionguard.evaluation.metrics import bootstrap_macro_f1, evaluate
from visionguard.explainability.maps import energy_in_mask, occlusion_map, pointing_game
from visionguard.models.classical import FeatureClassifier
from visionguard.storage.database import Database, utcnow
from visionguard.transforms.augment import augment_batch, corrupt_batch, robust_augment_batch

log = get_logger("visionguard.pipeline")


def _latency(model: FeatureClassifier, imgs: np.ndarray, n: int = 40) -> tuple[float, float]:
    t = []
    for im in imgs[:n]:
        t0 = time.perf_counter()
        model.predict_proba(im[None])
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(t, 50)), float(np.percentile(t, 95))


def run_pipeline(n_explain: int = 40) -> dict:
    s = get_settings()
    tc = s.training
    db = Database(s.db_path)
    db.reset()
    s.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ds = materialise()
    aud = audit(ds)
    log.info("Dataset: %d images · %d instances · leakage %d", aud["n_images"], aud["instances"], aud["instance_leakage"])
    tr, va, te = ds.subset("train"), ds.subset("validation"), ds.subset("test")
    x_tr, y_tr = augment_batch(tr.images, tr.labels, s.augmentation["train"], copies=1, seed=tc["random_state"])
    ct = s.augmentation["corruption_training"]
    x_rb, y_rb = robust_augment_batch(tr.images, tr.labels, s.augmentation["train"], ct, ct["copies"], tc["random_state"])

    models: dict[str, FeatureClassifier] = {}
    summary_models: dict[str, dict] = {}
    for name in ("baseline", "improved", "robust"):
        cfg = tc[name]
        t0 = time.perf_counter()
        if name == "improved":   # small, validation-only grid search over C
            best = None
            X_tr = None
            for C in (2.0, 8.0, 32.0):
                m = FeatureClassifier(name, cfg["features"] + ["dog"], cfg["model"], {**cfg["params"], "C": C}, s.classes,
                                      tc["random_state"])
                X_tr = m.featurize(x_tr) if X_tr is None else X_tr
                m.fit(x_tr, y_tr, X=X_tr)
                f1 = f1_score(va.labels, m.predict(va.images), average="macro")
                log.info("improved C=%-5s val macro-F1 %.4f", C, f1)
                if best is None or f1 > best[0]:
                    best = (f1, C, m)
            model, chosen = best[2], {"C": best[1], "val_macro_f1": best[0]}
        elif name == "robust":
            model = FeatureClassifier(name, cfg["features"] + ["dog"], cfg["model"],
                                      {**cfg["params"], "C": summary_models["improved"]["C"]}, s.classes, tc["random_state"])
            model.fit(x_rb, y_rb)
            chosen = {"val_macro_f1": float(f1_score(va.labels, model.predict(va.images), average="macro")),
                      "C": summary_models["improved"]["C"]}
        else:
            model = FeatureClassifier(name, cfg["features"], cfg["model"], cfg["params"], s.classes, tc["random_state"])
            model.fit(x_tr, y_tr)
            chosen = {"val_macro_f1": float(f1_score(va.labels, model.predict(va.images), average="macro"))}
        train_s = time.perf_counter() - t0
        proba = model.predict_proba(te.images)
        met = evaluate(te.labels, proba, s.classes)
        met["macro_f1_ci95"] = bootstrap_macro_f1(te.labels, proba.argmax(1))
        met.update(chosen)
        path = s.checkpoints_dir / f"{name}.joblib"
        joblib.dump(model, path)
        p50, p95 = _latency(model, te.images)
        size_kb = path.stat().st_size / 1024
        db.write_df("models", pd.DataFrame([{
            "name": name, "version": f"vg-{__version__}-{name}", "kind": f"{cfg['model']} on {'+'.join(model.features)}",
            "path": str(path.name), "n_features": model.n_features, "size_kb": size_kb, "latency_ms_p50": p50,
            "latency_ms_p95": p95, "metrics_json": json.dumps(to_jsonable(met)), "created_at": utcnow()}]))
        db.write_df("predictions", pd.DataFrame({
            "image_id": te.meta.id, "model": name, "split": "test", "label": te.labels, "pred": proba.argmax(1),
            "confidence": proba.max(1), "defect_score": 1 - proba[:, 0],
            "probs_json": [json.dumps(np.round(p, 4).tolist()) for p in proba]}))
        models[name] = model
        summary_models[name] = {**met, "latency_ms_p50": p50, "latency_ms_p95": p95, "size_kb": size_kb,
                                "train_seconds": train_s, "n_features": model.n_features}
        log.info("%-9s test macro-F1 %.3f · defect recall %.3f · false alarms %.3f · p95 %.1f ms",
                 name, met["macro_f1"], met["defect_recall"], met["false_alarm_rate"], p95)

    # ---------------------------------------------------------------- robustness
    rob_rows = []
    for kind, levels in s.augmentation["robustness"].items():
        for lvl in levels:
            xc = corrupt_batch(te.images, kind, float(lvl))
            for name, m in models.items():
                met = evaluate(te.labels, m.predict_proba(xc), s.classes)
                rob_rows.append({"model": name, "corruption": kind, "level": float(lvl), "macro_f1": met["macro_f1"],
                                 "defect_recall": met["defect_recall"], "false_alarm_rate": met["false_alarm_rate"],
                                 "accuracy": met["accuracy"]})
        log.info("robustness: %s done", kind)
    rob = pd.DataFrame(rob_rows)
    db.write_df("robustness", rob)

    # ---------------------------------------------------------------- explanation verification
    ec = tc["explainability"]
    m = models["improved"]
    defect_idx = np.flatnonzero(te.labels > 0)
    rng = np.random.default_rng(0)
    pick = rng.choice(defect_idx, size=min(n_explain, len(defect_idx)), replace=False)
    ex_rows = []
    for i in pick:
        heat, tgt, _ = occlusion_map(m.predict_proba, te.images[i], None, ec["occlusion_patch"], ec["occlusion_stride"])
        ex_rows.append({"image_id": int(te.meta.id.iloc[i]), "model": "improved", "target": tgt,
                        "pointing_hit": int(pointing_game(heat, te.masks[i])),
                        "energy_in_mask": energy_in_mask(heat, te.masks[i]), "mask_area": float(te.masks[i].mean())})
    ex = pd.DataFrame(ex_rows)
    db.write_df("explanations", ex)
    expl_summary = {"n": len(ex), "pointing_game": float(ex.pointing_hit.mean()), "energy_in_mask": float(ex.energy_in_mask.mean()),
                    "random_baseline_energy": float(np.mean([min(1.0, a * 2.2) for a in ex.mask_area]))}
    log.info("explanations: pointing game %.2f · energy-in-mask %.2f", expl_summary["pointing_game"], expl_summary["energy_in_mask"])

    summary = {"version": __version__, "created_at": utcnow(), "audit": to_jsonable(aud), "models": to_jsonable(summary_models),
               "explanations": expl_summary, "classes": s.classes, "task": "image classification (normal + 4 defect classes)",
               "data_source": s.data["source"]}
    db.put_meta("summary", summary)
    dump_json(summary, s.reports_dir / "evaluation_summary.json")
    write_report(summary, rob, s.reports_dir / "evaluation_report.md")
    try:
        from visionguard.evaluation.figures import make_figures
        make_figures(ds, models, rob, s.reports_dir / "figures")
    except Exception as exc:
        log.warning("Figures skipped: %s", exc)
    return summary


def write_report(summary: dict, rob: pd.DataFrame, path) -> None:
    b, i, r = summary["models"]["baseline"], summary["models"]["improved"], summary["models"]["robust"]
    cls = summary["classes"]
    lines = [
        "# VisionGuard AI — Evaluation Report", "",
        f"*Generated {summary['created_at']} · task: {summary['task']} · data: {summary['data_source']} (synthetic)*", "",
        "> Synthetic steel-surface images with pixel masks. Results show method behaviour, not production performance.", "",
        "## Held-out test set (split by sheet instance — no leakage)", "",
        "| Metric | Baseline (intensity+gradient → LR) | Improved (HOG+LBP+FFT+DoG → RBF-SVM) | Robust (improved + corruption aug.) |",
        "|---|---|---|---|",
        f"| Macro-F1 (95% CI) | {_ci(b)} | {_ci(i)} | {_ci(r)} |",
        f"| Accuracy | {b['accuracy']:.3f} | {i['accuracy']:.3f} | {r['accuracy']:.3f} |",
        f"| Defect recall (any class) | {b['defect_recall']:.3f} | {i['defect_recall']:.3f} | {r['defect_recall']:.3f} |",
        f"| False-alarm rate on normal | {b['false_alarm_rate']:.3f} | {i['false_alarm_rate']:.3f} | {r['false_alarm_rate']:.3f} |",
        f"| Missed defects (FN) | {b['missed_defects']} | {i['missed_defects']} | {r['missed_defects']} |",
        f"| Defect PR-AUC | {b['defect_pr_auc']:.3f} | {i['defect_pr_auc']:.3f} | {r['defect_pr_auc']:.3f} |",
        f"| Latency p95 (ms / image, CPU) | {b['latency_ms_p95']:.1f} | {i['latency_ms_p95']:.1f} | {r['latency_ms_p95']:.1f} |",
        f"| Model size (KB) | {b['size_kb']:.0f} | {i['size_kb']:.0f} | {r['size_kb']:.0f} |", "",
        "## Per-class recall", "", "| Class | Baseline | Improved | Robust | Support |", "|---|---|---|---|---|",
    ]
    for c in cls:
        lines.append(f"| {c} | {b['per_class'][c]['recall']:.3f} | {i['per_class'][c]['recall']:.3f} | {r['per_class'][c]['recall']:.3f} | {i['per_class'][c]['support']} |")
    lines += ["", "## Robustness (macro-F1: improved → robust)", ""]
    for kind, g in rob.groupby("corruption"):
        gi, gr = g[g.model == "improved"], g[g.model == "robust"]
        lines.append(f"- **{kind}**: " + " · ".join(f"{lv:g}: {a:.2f}→{c:.2f}" for lv, a, c in zip(gi.level, gi.macro_f1, gr.macro_f1)))
    e = summary["explanations"]
    lines += ["", "## Explanation verification (occlusion maps vs. true masks)", "",
              f"- Pointing game (heatmap peak inside defect): **{e['pointing_game']:.0%}** of {e['n']} defect images",
              f"- Heatmap energy inside defect region: **{e['energy_in_mask']:.0%}** (area-matched random ≈ {e['random_baseline_energy']:.0%})",
              "", "## Limitations",
              "- Synthetic textures; validate on NEU-DET / MVTec AD before claims.",
              "- The robust model is trained with milder corruptions than the strongest test levels — the tail measures extrapolation.",
              "- CNN transfer learning is implemented (`train-cnn`) but its results are not reported here unless run.",
              "- Heatmaps are diagnostic aids, not proof of the model's reasoning.",
              "- Human inspection remains mandatory for every flagged or low-confidence item."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ci(m: dict) -> str:
    lo, hi = m["macro_f1_ci95"]
    return f"{m['macro_f1']:.3f} ({lo:.3f}–{hi:.3f})"
