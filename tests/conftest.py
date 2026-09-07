"""Shared test utilities and fixtures for the cxr-classifier test suite.

The fixture ``tiny_dataset`` builds an in-memory synthetic dataset of 12
images (4 per class) so the test suite can run without the real Kaggle
download. This makes CI hermetic — no 5 GB of JPEGs required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cxr_classifier.config import Config


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a small synthetic CXR dataset on disk and return its root."""
    import cv2

    root = tmp_path_factory.mktemp("cxr_tiny")
    classes = ["normal", "pneumonia", "tuberculosis"]
    n_per_class = 4
    rng = np.random.default_rng(42)
    for split in ("train", "val", "test"):
        for cls in classes:
            class_dir = root / split / cls
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                # Synthetic grayscale gradient + a circle for "lung field"
                img = np.full((256, 256, 3), 200, dtype=np.uint8)
                cv2.circle(img, (128, 128), 60, (40, 40, 40), thickness=2)
                noise = rng.integers(0, 30, img.shape, dtype=np.uint8)
                img = np.clip(img.astype(int) - noise, 0, 255).astype(np.uint8)
                cv2.imwrite(str(class_dir / f"{cls}-{i:03d}.jpg"), img)
    return root


@pytest.fixture(scope="session")
def small_config(tiny_dataset: Path, tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Build a Config that points at the tiny dataset and uses tiny values."""
    cfg_dict = {
        "dataset": {
            "data_root": str(tiny_dataset),
            "train_dir": "train",
            "val_dir": "val",
            "test_dir": "test",
            "classes": ["normal", "pneumonia", "tuberculosis"],
            "num_classes": 3,
            "image_size": 64,  # tiny so tests are fast
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "model": {
            "name": "resnet18",  # smallest model available offline
            "pretrained": False,
            "num_classes": 3,
            "drop_path_rate": 0.0,
            "drop_rate": 0.0,
        },
        "training": {
            "epochs": 1,
            "optimizer": {
                "name": "adam",
                "lr": 1e-3,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
            "scheduler": {
                "name": "cosine_annealing",
                "t_0": 1,
                "t_mult": 1,
                "eta_min": 0.0,
                "warmup_epochs": 0,
                "warmup_lr": 1e-6,
            },
            "loss": {"name": "cross_entropy", "label_smoothing": 0.0},
            "gradient_clip": 0.0,
            "mixed_precision": False,
            "early_stopping": {"patience": 100, "min_delta": 0.0, "mode": "max"},
            "save_best_only": True,
            "save_top_k": 1,
        },
        "augmentation": {
            "train": [
                {"name": "Resize", "height": 64, "width": 64},
                {"name": "Normalize",
                 "mean": [0.485, 0.456, 0.406],
                 "std": [0.229, 0.224, 0.225]},
                {"name": "ToTensorV2"},
            ],
            "val": [
                {"name": "Resize", "height": 64, "width": 64},
                {"name": "Normalize",
                 "mean": [0.485, 0.456, 0.406],
                 "std": [0.229, 0.224, 0.225]},
                {"name": "ToTensorV2"},
            ],
            "test": [
                {"name": "Resize", "height": 64, "width": 64},
                {"name": "Normalize",
                 "mean": [0.485, 0.456, 0.406],
                 "std": [0.229, 0.224, 0.225]},
                {"name": "ToTensorV2"},
            ],
        },
        "logging": {
            "use_wandb": False,
            "wandb_project": "test",
            "wandb_entity": None,
            "use_tensorboard": False,
            "log_dir": str(tmp_path_factory.mktemp("logs")),
            "log_interval": 1,
            "save_dir": str(tmp_path_factory.mktemp("outputs")),
            "save_interval": 1,
        },
        "evaluation": {
            "metrics": ["accuracy", "f1", "auc"],
            "class_names": ["Normal", "Pneumonia", "Tuberculosis"],
        },
        "hardware": {
            "device": "cpu",
            "mixed_precision": False,
            "compile_model": False,
        },
    }
    # Validate by round-tripping through the Config dataclass
    return Config.from_dict(cfg_dict)


def get_device() -> torch.device:
    return torch.device("cpu")