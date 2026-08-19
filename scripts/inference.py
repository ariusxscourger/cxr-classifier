#!/usr/bin/env python3
"""
Inference script for CXR-Classifier.
Make predictions on new images.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Disable HF Hub and wandb network calls for fully offline operation
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "offline"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cxr_classifier.config import load_config
from cxr_classifier.models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CXR-Classifier Inference")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to show",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Determine the device to use."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_arg)


def load_image(image_path: str, transform: A.Compose) -> torch.Tensor:
    """Load and preprocess image."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        # Fallback to PIL
        image = np.array(Image.open(image_path).convert("RGB"))
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply transform
    transformed = transform(image=image)
    return transformed["image"].unsqueeze(0)  # Add batch dimension


def main() -> None:
    args = parse_args()

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    config.hardware.device = str(device)

    # Build test transform
    test_transform = A.Compose([
        A.Resize(config.dataset.image_size, config.dataset.image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Create model
    print("\nCreating model...")
    model = create_model(config)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Load and preprocess image
    print(f"\nProcessing image: {args.image}")
    image_tensor = load_image(args.image, test_transform).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

    # Get top-k predictions
    probs, indices = torch.topk(probabilities, k=min(args.top_k, config.model.num_classes), dim=1)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]

    # Print results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    for i, (idx, prob) in enumerate(zip(indices, probs)):
        class_name = config.evaluation.class_names[idx]
        print(f"  {i+1}. {class_name}: {prob*100:.2f}%")

    # All class probabilities
    all_probs = probabilities.cpu().numpy()[0]
    print("\nAll class probabilities:")
    for i, class_name in enumerate(config.evaluation.class_names):
        print(f"  {class_name}: {all_probs[i]*100:.2f}%")

    print("=" * 50)


if __name__ == "__main__":
    main()