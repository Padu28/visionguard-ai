import numpy as np

from visionguard.datasets.dataset import instance_split
from visionguard.datasets.synthetic import CLASSES, generate_dataset
from visionguard.transforms.augment import CORRUPTIONS, augment, corrupt_batch


def test_shapes_and_ranges(small):
    imgs, labels, masks, inst = small
    assert imgs.shape == masks.shape == (300, 96, 96)
    assert imgs.min() >= 0 and imgs.max() <= 1
    assert set(np.unique(labels)) == set(range(len(CLASSES)))


def test_masks_match_labels(small):
    _, labels, masks, _ = small
    areas = masks.reshape(len(masks), -1).mean(1)
    assert (areas[labels == 0] == 0).all()
    assert (areas[labels > 0] > 0).all()


def test_deterministic():
    a = generate_dataset(5, 2, 64, {"normal": 0.5, "crack": 0.5}, seed=1)[0]
    b = generate_dataset(5, 2, 64, {"normal": 0.5, "crack": 0.5}, seed=1)[0]
    assert np.array_equal(a, b)


def test_instance_split_has_no_leakage(small):
    inst = small[3]
    split = instance_split(inst, {"train": 0.6, "validation": 0.2, "test": 0.2}, 0)
    tr, te = set(inst[split == "train"]), set(inst[split == "test"])
    assert tr.isdisjoint(te)
    assert {"train", "validation", "test"} == set(split)


def test_augment_preserves_shape_and_range(small):
    img = small[0][0]
    out = augment(img, np.random.default_rng(0), {"hflip": 1, "vflip": 1, "rot90": True, "brightness": 0.1,
                                                  "contrast": 0.1, "gaussian_noise_std": 0.01})
    assert out.shape == img.shape and 0 <= out.min() and out.max() <= 1


def test_corruptions_identity_at_zero(small):
    img = small[0][:3]
    assert np.allclose(corrupt_batch(img, "blur_sigma", 0.0), img)
    assert np.allclose(corrupt_batch(img, "noise_std", 0.0), img)
    assert np.allclose(corrupt_batch(img, "brightness", 0.0), img)
    assert set(CORRUPTIONS) == {"brightness", "blur_sigma", "noise_std", "resolution_scale"}
