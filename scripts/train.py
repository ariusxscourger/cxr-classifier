#!/usr/bin/env python3
"""
Main training script for CXR-Classifier.
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
from cxr_classifier.evaluation import evaluate_model
from cxr_classifier.models import create_model, get_model_info
from cxr_classifier.training import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CXR-Classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (fewer epochs, smaller dataset)",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def main() -> None:
    args = parse_args()

    # Set seed
    set_seed(args.seed)

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

    print("Configuration loaded:")
    print(f"  Model: {config.model.name}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.dataset.batch_size}")
    print(f"  Image size: {config.dataset.image_size}")
    print(f"  Classes: {config.dataset.classes}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    print("\nCreating model...")
    model = create_model(config)
    model_info = get_model_info(model)
    print(f"Model: {config.model.name}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if trainer.scheduler and checkpoint.get("scheduler_state_dict"):
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Train
    print("\nStarting training...")
    try:
        results = trainer.train()
        print("\nTraining completed successfully!")
        print(f"Best validation metric: {results['best_metric']:.4f}")

        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_results = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=config.evaluation.class_names,
            save_dir=str(Path(config.logging.save_dir) / "test_results"),
        )
        print(f"Test Accuracy: {test_results['metrics']['accuracy']:.4f}")
        print(f"Test F1 Macro: {test_results['metrics']['f1_macro']:.4f}")
        print(f"Test AUC Macro: {test_results['metrics']['auc_macro']:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save emergency checkpoint
        emergency_path = Path(config.logging.save_dir) / "emergency_checkpoint.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "config": config.to_dict(),
            },
            emergency_path,
        )
        print(f"Emergency checkpoint saved to {emergency_path}")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
