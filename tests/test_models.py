import numpy as np
import pytest
from sklearn.metrics import f1_score

from visionguard.datasets.dataset import instance_split
from visionguard.datasets.synthetic import CLASSES
from visionguard.evaluation.metrics import evaluate
from visionguard.explainability.maps import energy_in_mask, occlusion_map, overlay, pointing_game
from visionguard.models.classical import FeatureClassifier
from visionguard.models.features import EXTRACTORS, extract


@pytest.fixture(scope="module")
def trained(small):
    imgs, labels, masks, inst = small
    split = instance_split(inst, {"train": 0.7, "validation": 0.0, "test": 0.3}, 0)
    tr, te = split == "train", split == "test"
    m = FeatureClassifier("t", ["intensity", "gradient", "hog", "lbp", "fft", "dog"], "svm_rbf",
                          {"C": 4.0, "gamma": "scale", "class_weight": "balanced"}, list(CLASSES))
    m.fit(imgs[tr], labels[tr])
    return m, imgs[te], labels[te], masks[te]


def test_feature_vectors_are_finite(small):
    X = extract(small[0][:8], list(EXTRACTORS))
    assert X.shape[0] == 8 and np.isfinite(X).all()


def test_features_are_illumination_invariant(small):
    img = small[0][0]
    a = extract(img[None], ["hog", "lbp", "fft"])
    b = extract(np.clip(img + 0.05, 0, 1)[None], ["hog", "lbp", "fft"])
    assert np.allclose(a, b, atol=1e-3)


def test_model_beats_chance(trained):
    m, x, y, _ = trained
    proba = m.predict_proba(x)
    assert proba.shape == (len(y), len(CLASSES))
    assert np.allclose(proba.sum(1), 1, atol=1e-6)
    assert f1_score(y, proba.argmax(1), average="macro") > 0.5


def test_evaluate_report(trained):
    m, x, y, _ = trained
    rep = evaluate(y, m.predict_proba(x), list(CLASSES))
    assert set(rep["per_class"]) == set(CLASSES)
    assert 0 <= rep["false_alarm_rate"] <= 1
    assert np.array(rep["confusion"]).sum() == len(y)


def test_occlusion_map_and_verification(trained):
    m, x, y, masks = trained
    i = int(np.flatnonzero(y == CLASSES.index("inclusion"))[0])
    heat, tgt, p = occlusion_map(m.predict_proba, x[i])
    assert heat.shape == x[i].shape and 0 <= heat.min() and heat.max() <= 1
    assert isinstance(pointing_game(heat, masks[i]), bool)
    assert 0 <= energy_in_mask(heat, masks[i]) <= 1
    assert overlay(x[i], heat).shape == (*x[i].shape, 3)


def test_pointing_game_logic():
    heat = np.zeros((20, 20))
    heat[5, 5] = 1
    mask = np.zeros((20, 20), np.uint8)
    mask[4:7, 4:7] = 1
    assert pointing_game(heat, mask, tolerance=0)
    mask2 = np.zeros((20, 20), np.uint8)
    mask2[15:18, 15:18] = 1
    assert not pointing_game(heat, mask2, tolerance=1)
