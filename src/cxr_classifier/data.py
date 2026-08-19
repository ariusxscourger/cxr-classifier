"""
Data module for Chest X-Ray Classification.
Handles dataset loading, augmentation, and dataloader creation.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from cxr_classifier.config import Config


class ChestXRayDataset(Dataset):
    """Chest X-Ray Dataset for multi-class classification."""

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str,
        classes: List[str],
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        """
        Initialize dataset.

        Args:
            data_root: Root directory containing train/val/test subdirectories
            split: One of 'train', 'val', 'test'
            classes: List of class names
            transform: Albumentations transform pipeline
            image_size: Target image size
        """
        self.data_root = Path(data_root)
        self.split = split
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.transform = transform
        self.image_size = image_size

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all sample paths and labels."""
        samples = []
        split_dir = self.data_root / self.split

        for class_name in self.classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Directory {class_dir} does not exist")
                continue

            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                for img_path in class_dir.glob(ext):
                    samples.append((img_path, self.class_to_idx[class_name]))

        print(f"Loaded {len(samples)} samples for {self.split} split")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            # Fallback to PIL
            image = np.array(Image.open(img_path).convert("RGB"))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Default transform
            image = self._default_transform(image)

        return image, label

    def _default_transform(self, image: np.ndarray) -> torch.Tensor:
        """Default transform if none provided."""
        transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        return transform(image=image)["image"]

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced sampling."""
        class_counts = np.zeros(len(self.classes))
        for _, label in self.samples:
            class_counts[label] += 1

        # Inverse frequency weighting
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(self.classes)
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> List[float]:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return [class_weights[label].item() for _, label in self.samples]


def build_augmentation_pipeline(cfg: Dict[str, Any]) -> A.Compose:
    """Build Albumentations pipeline from config."""
    transforms_list = []

    for transform_cfg in cfg:
        name = transform_cfg.pop("name")
        params = transform_cfg

        if name == "Resize":
            transforms_list.append(A.Resize(**params))
        elif name == "RandomCrop":
            transforms_list.append(A.RandomCrop(**params))
        elif name == "CenterCrop":
            transforms_list.append(A.CenterCrop(**params))
        elif name == "HorizontalFlip":
            transforms_list.append(A.HorizontalFlip(**params))
        elif name == "VerticalFlip":
            transforms_list.append(A.VerticalFlip(**params))
        elif name == "Rotate":
            transforms_list.append(A.Rotate(**params))
        elif name == "ShiftScaleRotate":
            transforms_list.append(A.ShiftScaleRotate(**params))
        elif name == "RandomBrightnessContrast":
            transforms_list.append(A.RandomBrightnessContrast(**params))
        elif name == "HueSaturationValue":
            transforms_list.append(A.HueSaturationValue(**params))
        elif name == "GaussNoise":
            transforms_list.append(A.GaussNoise(**params))
        elif name == "GaussianBlur":
            transforms_list.append(A.GaussianBlur(**params))
        elif name == "MotionBlur":
            transforms_list.append(A.MotionBlur(**params))
        elif name == "CoarseDropout":
            transforms_list.append(A.CoarseDropout(**params))
        elif name == "Cutout":
            transforms_list.append(A.Cutout(**params))
        elif name == "Normalize":
            transforms_list.append(A.Normalize(**params))
        elif name == "ToTensorV2":
            transforms_list.append(ToTensorV2(**params))
        else:
            print(f"Warning: Unknown transform {name}")

    return A.Compose(transforms_list)


def get_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Build transforms
    train_transform = build_augmentation_pipeline(config.augmentation.train)
    val_transform = build_augmentation_pipeline(config.augmentation.val)
    test_transform = build_augmentation_pipeline(config.augmentation.test)

    # Create datasets
    train_dataset = ChestXRayDataset(
        data_root=config.dataset.data_root,
        split=config.dataset.train_dir,
        classes=config.dataset.classes,
        transform=train_transform,
        image_size=config.dataset.image_size,
    )

    val_dataset = ChestXRayDataset(
        data_root=config.dataset.data_root,
        split=config.dataset.val_dir,
        classes=config.dataset.classes,
        transform=val_transform,
        image_size=config.dataset.image_size,
    )

    test_dataset = ChestXRayDataset(
        data_root=config.dataset.data_root,
        split=config.dataset.test_dir,
        classes=config.dataset.classes,
        transform=test_transform,
        image_size=config.dataset.image_size,
    )

    # Create weighted sampler for training (handle class imbalance)
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        sampler=sampler,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        persistent_workers=config.dataset.persistent_workers and config.dataset.num_workers > 0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        persistent_workers=config.dataset.persistent_workers and config.dataset.num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        persistent_workers=config.dataset.persistent_workers and config.dataset.num_workers > 0,
    )

    return train_loader, val_loader, test_loader


def get_class_distribution(dataset: ChestXRayDataset) -> Dict[str, int]:
    """Get class distribution for a dataset."""
    distribution = {cls: 0 for cls in dataset.classes}
    for _, label in dataset.samples:
        distribution[dataset.classes[label]] += 1
    return distribution