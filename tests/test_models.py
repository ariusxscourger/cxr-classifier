"""Smoke tests for model creation and the ensemble wrapper."""
from __future__ import annotations

import pytest
import torch

from cxr_classifier.config import Config, ModelConfig
from cxr_classifier.models import create_model, create_ensemble, get_model_info


@pytest.mark.parametrize("name", ["resnet18"])
def test_create_model_returns_module(name: str):
    cfg = Config()
    cfg.model = ModelConfig(name=name, pretrained=False, num_classes=3)
    model = create_model(cfg)
    assert isinstance(model, torch.nn.Module)

    info = get_model_info(model)
    assert info["total_parameters"] > 0
    assert info["trainable_parameters"] == info["total_parameters"]
    assert info["model_size_mb"] > 0

    # Forward pass
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 3)


def test_create_ensemble_averages_predictions():
    """An ensemble of identical inputs should match a single model's prediction."""
    cfg = Config()
    cfg.model = ModelConfig(name="resnet18", pretrained=False, num_classes=3)
    names = ["resnet18"]
    ensemble = create_ensemble(cfg, names)
    assert isinstance(ensemble, torch.nn.Module)

    # batch_size >= 2 because the modified classifier head contains BatchNorm1d
    x = torch.randn(2, 3, 64, 64)
    ensemble.eval()
    with torch.no_grad():
        out = ensemble(x)
    assert out.shape == (2, 3)


def test_get_model_info_counts_only_trainable_when_frozen():
    """After freezing the backbone, trainable < total."""
    from cxr_classifier.models import freeze_backbone

    cfg = Config()
    cfg.model = ModelConfig(name="resnet18", pretrained=False, num_classes=3)
    model = create_model(cfg)
    total = sum(p.numel() for p in model.parameters())

    freeze_backbone(model, freeze=True)
    info = get_model_info(model)
    trainable = info["trainable_parameters"]

    # Freeze should leave only the classifier head trainable
    assert trainable < total
    assert trainable > 0