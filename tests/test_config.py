"""Smoke tests for the Config dataclass + YAML roundtrip."""
from __future__ import annotations

import yaml

from cxr_classifier.config import (
    Config,
    DatasetConfig,
    HardwareConfig,
    LoggingConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
)


def test_default_config_is_valid():
    """Default Config should have all nested dataclasses populated."""
    cfg = Config()
    assert isinstance(cfg.dataset, DatasetConfig)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.logging, LoggingConfig)
    assert isinstance(cfg.hardware, HardwareConfig)


def test_yaml_roundtrip(tmp_path):
    """Saving and reloading a Config via YAML should preserve values."""
    cfg = Config()
    cfg.model.name = "resnet18"
    cfg.dataset.image_size = 128
    out = tmp_path / "cfg.yaml"
    cfg.save(out)

    loaded = load_config(out)
    assert loaded.model.name == "resnet18"
    assert loaded.dataset.image_size == 128
    assert loaded.model.num_classes == 3


def test_load_real_config_file():
    """The repo's own configs/config.yaml should load without errors."""
    cfg = load_config("configs/config.yaml")
    assert cfg.model.num_classes == 3
    assert "normal" in cfg.dataset.classes
    # CLAHE should be in the val pipeline after the recent fix
    val_names = [t["name"] for t in cfg.augmentation.val]
    assert "CLAHE" in val_names
    assert "Resize" in val_names
    assert "CenterCrop" in val_names


def test_to_dict_is_yaml_safe():
    """to_dict() output should round-trip cleanly through yaml.safe_load."""
    cfg = Config()
    d = cfg.to_dict()
    yaml_str = yaml.safe_dump(d)
    d2 = yaml.safe_load(yaml_str)
    assert d2["model"]["name"] == cfg.model.name
    assert d2["training"]["epochs"] == cfg.training.epochs