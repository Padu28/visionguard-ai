# Contributing & engineering standards

Thanks for your interest! These conventions keep the project reproducible.

## Workflow
1. Branch from `main`: `feat/<short-name>` or `fix/<short-name>`.
2. Keep commits small and meaningful (`add ADWIN detector tests`).
3. Run `ruff check .` and `pytest -q` before pushing; CI runs both plus an end-to-end pipeline.
4. Open a PR describing *what changed, why, and the measured effect* (metrics before/after).

## Research hygiene
* Fixed seeds for reproducibility, **multiple** seeds or bootstrap CIs for conclusions.
* Thresholds and model choices are made on validation data and frozen before test evaluation.
* Exploration lives in `notebooks/`; reusable code lives in `src/`.
* Every dataset gets a data card (source, licence, schema, preprocessing, limitations).
* Report negative and inconvenient results — they are part of the finding.

## Never commit
API keys, passwords, `.env`, personal or patient data, private datasets, large raw data, trained artefacts.
Use `.env.example` with placeholder names only.

## Code style
Python 3.11+, type hints on public functions, `ruff` (line length 120), docstrings that explain *why*.
