"""
CXR-Classifier Package.

This package provides a complete pipeline for multi-class chest X-ray classification
(Normal, Pneumonia, Tuberculosis) using state-of-the-art CNN and Vision Transformer architectures.
"""

__version__ = "0.1.0"
__author__ = "Muhammad Saqib"
__email__ = "your.email@example.com"
__description__ = "CXR-Classifier: Chest X-Ray Multi-class Classification"

from cxr_classifier.config import Config, load_config
from cxr_classifier.data import ChestXRayDataset, get_dataloaders
from cxr_classifier.evaluation import Evaluator
from cxr_classifier.models import create_model
from cxr_classifier.training import Trainer

__all__ = [
    "Config",
    "load_config",
    "ChestXRayDataset",
    "get_dataloaders",
    "create_model",
    "Trainer",
    "Evaluator",
]
