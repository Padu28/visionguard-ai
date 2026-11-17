# VisionGuard AI — Evaluation Report

*Generated 2026-11-17T13:37:39+00:00 · task: image classification (normal + 4 defect classes) · data: synthetic (synthetic)*

> Synthetic steel-surface images with pixel masks. Results show method behaviour, not production performance.

## Held-out test set (split by sheet instance — no leakage)

| Metric | Baseline (intensity+gradient → LR) | Improved (HOG+LBP+FFT+DoG → RBF-SVM) | Robust (improved + corruption aug.) |
|---|---|---|---|
| Macro-F1 (95% CI) | 0.651 (0.584–0.705) | 0.850 (0.802–0.894) | 0.825 (0.775–0.871) |
| Accuracy | 0.713 | 0.873 | 0.853 |
| Defect recall (any class) | 0.929 | 0.909 | 0.864 |
| False-alarm rate on normal | 0.205 | 0.075 | 0.062 |
| Missed defects (FN) | 11 | 14 | 21 |
| Defect PR-AUC | 0.974 | 0.973 | 0.960 |
| Latency p95 (ms / image, CPU) | 3.6 | 8.1 | 12.4 |
| Model size (KB) | 3 | 1005 | 1548 |

## Per-class recall

| Class | Baseline | Improved | Robust | Support |
|---|---|---|---|---|
| normal | 0.795 | 0.925 | 0.938 | 146 |
| scratch | 0.545 | 0.636 | 0.568 | 44 |
| inclusion | 0.512 | 0.878 | 0.854 | 41 |
| crack | 0.725 | 0.925 | 0.875 | 40 |
| patch | 0.828 | 0.897 | 0.828 | 29 |

## Robustness (macro-F1: improved → robust)

- **blur_sigma**: 0: 0.85→0.82 · 0.5: 0.70→0.74 · 1: 0.07→0.48 · 1.5: 0.05→0.10 · 2: 0.05→0.05 · 3: 0.05→0.05
- **brightness**: -0.3: 0.78→0.77 · -0.2: 0.85→0.82 · -0.1: 0.85→0.82 · 0: 0.85→0.82 · 0.1: 0.85→0.82 · 0.2: 0.85→0.82 · 0.3: 0.79→0.79
- **noise_std**: 0: 0.85→0.82 · 0.02: 0.78→0.83 · 0.04: 0.21→0.64 · 0.06: 0.05→0.53 · 0.08: 0.05→0.39 · 0.12: 0.05→0.22
- **resolution_scale**: 1: 0.85→0.82 · 0.75: 0.20→0.57 · 0.5: 0.05→0.17 · 0.375: 0.05→0.08 · 0.25: 0.05→0.05

## Explanation verification (occlusion maps vs. true masks)

- Pointing game (heatmap peak inside defect): **57%** of 40 defect images
- Heatmap energy inside defect region: **31%** (area-matched random ≈ 13%)

## Limitations
- Synthetic textures; validate on NEU-DET / MVTec AD before claims.
- The robust model is trained with milder corruptions than the strongest test levels — the tail measures extrapolation.
- CNN transfer learning is implemented (`train-cnn`) but its results are not reported here unless run.
- Heatmaps are diagnostic aids, not proof of the model's reasoning.
- Human inspection remains mandatory for every flagged or low-confidence item.
