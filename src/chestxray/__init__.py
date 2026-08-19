"""
Chest X-Ray Classification Package for CSC Scholarship Application.

This package provides a complete pipeline for multi-class chest X-ray classification
(Normal, Pneumonia, Tuberculosis) using state-of-the-art CNN and Vision Transformer architectures.
"""

__version__ = "0.1.0"
__author__ = "Muhammad Saqib"
__email__ = "your.email@example.com"
__description__ = "Chest X-Ray Multi-class Classification for CSC Scholarship"

from chestxray.config import Config, load_config
from chestxray.data import ChestXRayDataset, get_dataloaders
from chestxray.models import create_model
from chestxray.training import Trainer
from chestxray.evaluation import Evaluator

__all__ = [
    "Config",
    "load_config",
    "ChestXRayDataset",
    "get_dataloaders",
    "create_model",
    "Trainer",
    "Evaluator",
]