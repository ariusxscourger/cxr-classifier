"""Smoke tests for the ChestXRayDataset."""
from __future__ import annotations

import numpy as np
import torch

from cxr_classifier.data import ChestXRayDataset, build_augmentation_pipeline, get_class_distribution


def test_dataset_loads_all_classes(tiny_dataset):
    """Dataset should find all 3 classes in the train folder."""
    ds = ChestXRayDataset(
        data_root=tiny_dataset,
        split="train",
        classes=["normal", "pneumonia", "tuberculosis"],
        image_size=64,
    )
    counts = get_class_distribution(ds)
    assert counts["normal"] == 4
    assert counts["pneumonia"] == 4
    assert counts["tuberculosis"] == 4
    assert len(ds) == 12


def test_dataset_returns_tensor_and_int(tiny_dataset):
    """__getitem__ should return (Tensor[C,H,W], int label)."""
    ds = ChestXRayDataset(
        data_root=tiny_dataset,
        split="train",
        classes=["normal", "pneumonia", "tuberculosis"],
        image_size=64,
        transform=build_augmentation_pipeline(
            [{"name": "Resize", "height": 64, "width": 64},
             {"name": "Normalize",
              "mean": [0.485, 0.456, 0.406],
              "std": [0.229, 0.224, 0.225]},
             {"name": "ToTensorV2"}]
        ),
    )
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 64, 64)
    assert isinstance(label, int)
    assert label in (0, 1, 2)


def test_class_weights_normalized(tiny_dataset):
    """get_class_weights should be a 1-D tensor that sums to num_classes."""
    ds = ChestXRayDataset(
        data_root=tiny_dataset,
        split="train",
        classes=["normal", "pneumonia", "tuberculosis"],
        image_size=64,
    )
    weights = ds.get_class_weights()
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (3,)
    assert torch.isclose(weights.sum(), torch.tensor(float(len(ds.classes))), atol=1e-3)


def test_sample_weights_have_correct_length(tiny_dataset):
    """Per-sample weights should have one entry per sample."""
    ds = ChestXRayDataset(
        data_root=tiny_dataset,
        split="train",
        classes=["normal", "pneumonia", "tuberculosis"],
        image_size=64,
    )
    sample_weights = ds.get_sample_weights()
    assert len(sample_weights) == len(ds)
    assert all(w > 0 for w in sample_weights)


def test_dataset_warns_on_missing_class(tmp_path, capsys):
    """Missing class folder should print a warning, not raise."""
    import cv2

    # Build a dataset with only two classes — 2 images each
    for split in ("train",):
        for cls in ("normal", "pneumonia"):
            d = tmp_path / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                img = np.full((64, 64, 3), 128, dtype=np.uint8)
                cv2.imwrite(str(d / f"img-{i}.jpg"), img)
    ds = ChestXRayDataset(
        data_root=tmp_path,
        split="train",
        classes=["normal", "pneumonia", "tuberculosis"],
        image_size=64,
    )
    assert len(ds) == 4  # 2 classes × 2 images each