"""Feature-based classifiers (baseline and improved) with a uniform image-in API."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from visionguard.models.features import extract

ESTIMATORS = {"logistic_regression": LogisticRegression, "svm_rbf": SVC}


class FeatureClassifier:
    """Image → handcrafted features → scaler → estimator."""

    def __init__(self, name: str, features: list[str], model: str, params: dict[str, Any], classes: list[str],
                 random_state: int = 42) -> None:
        self.name, self.features, self.classes = name, list(features), list(classes)
        p = dict(params)
        if "random_state" in ESTIMATORS[model]().get_params():
            p["random_state"] = random_state
        est = ESTIMATORS[model](**p)
        if model == "svm_rbf":  # Platt-calibrated probabilities (replaces deprecated SVC(probability=True))
            est = CalibratedClassifierCV(est, method="sigmoid", cv=3, ensemble=False)
        self.pipe = Pipeline([("scale", StandardScaler()), ("clf", est)])

    def featurize(self, images: np.ndarray) -> np.ndarray:
        return extract(images, self.features)

    def fit(self, images: np.ndarray, labels: np.ndarray, X: np.ndarray | None = None) -> FeatureClassifier:
        self.pipe.fit(self.featurize(images) if X is None else X, labels)
        return self

    def predict_proba(self, images: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        return self.pipe.predict_proba(self.featurize(images) if X is None else X)

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self.predict_proba(images).argmax(1)

    @property
    def n_features(self) -> int:
        return int(self.pipe.named_steps["scale"].n_features_in_)
