"""
Model architectures for Chest X-Ray Classification.
Supports modern CNN and Vision Transformer models via timm.
"""

import torch
import torch.nn as nn
from typing import Optional, List
import timm

from cxr_classifier.config import Config


def create_model(config: Config) -> nn.Module:
    """
    Create model based on configuration.

    Args:
        config: Configuration object

    Returns:
        PyTorch model
    """
    model_name = config.model.name
    num_classes = config.model.num_classes
    pretrained = config.model.pretrained
    drop_path_rate = config.model.drop_path_rate
    drop_rate = config.model.drop_rate

    # Create model using timm
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
    )

    # Modify classifier head for better performance
    model = _modify_classifier(model, num_classes, drop_rate)

    return model


def _modify_classifier(model: nn.Module, num_classes: int, drop_rate: float) -> nn.Module:
    """Modify the classifier head for better regularization."""
    # Find the classifier layer
    if hasattr(model, "head"):
        # ConvNeXt, EfficientNet, etc.
        # Check if it's a NormMlpClassifierHead (ConvNeXt style with built-in pooling)
        if hasattr(model.head, 'fc') and hasattr(model.head, 'global_pool'):
            # ConvNeXt style: keep the pooling, replace only the final fc layer
            in_features = model.head.fc.in_features
            model.head.fc = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate),
                nn.Linear(512, num_classes),
            )
        else:
            # Other models with simple head
            in_features = model.head.in_features
            model.head = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate),
                nn.Linear(512, num_classes),
            )
    elif hasattr(model, "fc"):
        # ResNet
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes),
        )
    elif hasattr(model, "classifier"):
        # ViT, DeiT
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate),
                nn.Linear(512, num_classes),
            )
        else:
            # Already a Sequential
            pass
    elif hasattr(model, "head.fc"):
        # Some models have head.fc
        in_features = model.head.fc.in_features
        model.head.fc = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes),
        )

    return model


def get_model_info(model: nn.Module) -> dict:
    """Get model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 ** 2),  # Assuming float32
    }


def freeze_backbone(model: nn.Module, freeze: bool = True) -> nn.Module:
    """Freeze or unfreeze backbone parameters."""
    # Find backbone parameters (all except classifier head)
    for name, param in model.named_parameters():
        if "head" not in name and "fc" not in name and "classifier" not in name:
            param.requires_grad = not freeze
    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
    return model


# Model ensemble for improved performance
class ModelEnsemble(nn.Module):
    """Ensemble of multiple models."""

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0 / len(models)] * len(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for model, weight in zip(self.models, self.weights):
            outputs.append(model(x) * weight)
        return torch.stack(outputs).sum(dim=0)


def create_ensemble(config: Config, model_names: List[str]) -> ModelEnsemble:
    """Create an ensemble of models."""
    models = []
    for name in model_names:
        config.model.name = name
        model = create_model(config)
        models.append(model)
    return ModelEnsemble(models)