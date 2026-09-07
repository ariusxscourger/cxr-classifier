#!/usr/bin/env python3
"""
Evaluation script for CXR-Classifier.
Evaluates a trained model on test set and generates comprehensive reports.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Disable HF Hub and wandb network calls for fully offline operation
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "offline"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cxr_classifier.config import load_config
from cxr_classifier.data import get_dataloaders
from cxr_classifier.device import detect_best_device, get_torch_device
from cxr_classifier.evaluation import evaluate_model
from cxr_classifier.models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CXR-Classifier")
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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps", "directml", "gpu"],
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Resolve the requested device (CUDA -> DirectML -> MPS -> CPU)."""
    return get_torch_device(device_arg)


def main() -> None:
    args = parse_args()

    # Get device
    device = get_device(args.device)
    dev_name = detect_best_device()[1] if args.device in ("auto", "gpu") else str(device).split(":")[0]
    print(f"Using device: {device} ({dev_name})")

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    config.hardware.device = str(device)

    # Create dataloaders (we only need test loader, but get_dataloaders returns all three)
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Select appropriate loader
    if args.split == "train":
        eval_loader = train_loader
    elif args.split == "val":
        eval_loader = val_loader
    else:
        eval_loader = test_loader

    print(f"Evaluating on {args.split} set: {len(eval_loader.dataset)} samples")

    # Create model
    print("\nCreating model...")
    model = create_model(config)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    # Load on CPU first then move to device (torch.load on DirectML fails
    # with TypeError because torch_directml.device() can't compare a
    # torch.device object to int).
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in checkpoint:
        checkpoint["model_state_dict"] = {
            k: v.to(device) for k, v in checkpoint["model_state_dict"].items()
        }

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("Model loaded successfully")

    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    results = evaluate_model(
        model=model,
        dataloader=eval_loader,
        device=device,
        class_names=config.evaluation.class_names,
        save_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1 Macro: {results['metrics']['f1_macro']:.4f}")
    print(f"AUC Macro: {results['metrics']['auc_macro']:.4f}")


if __name__ == "__main__":
    main()
