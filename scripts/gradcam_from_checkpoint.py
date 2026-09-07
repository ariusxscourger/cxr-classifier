"""Generate Grad-CAM heatmap grid from a trained model checkpoint.

Usage:
    python scripts/gradcam_from_checkpoint.py \
        --checkpoint outputs/best_model.pth \
        --config configs/fast_gpu.yaml \
        --data-root dataset_test \
        --num-samples 2 \
        --output-dir outputs/gradcam_fast
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cxr_classifier.config import load_config
from cxr_classifier.data import get_dataloaders
from cxr_classifier.device import get_torch_device
from cxr_classifier.evaluation import generate_gradcam_grid
from cxr_classifier.models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap grid from a trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/fast_gpu.yaml", help="Config file")
    parser.add_argument("--data-root", type=str, default="dataset_test", help="Dataset root (must contain test/ subdirs)")
    parser.add_argument("--num_samples", type=int, default=2, help="Samples per class")
    parser.add_argument("--output-dir", type=str, default="outputs/gradcam_fast", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps, directml, gpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    config.dataset.data_root = args.data_root

    print(f"Loading checkpoint from {args.checkpoint}")
    device = get_torch_device(args.device)
    print(f"Using device: {device}")

    _, _, test_loader = get_dataloaders(config)
    print(f"Test set: {len(test_loader.dataset)} samples")

    print(f"Creating model: {config.model.name}")
    model = create_model(config).to(device)
    # Load checkpoint on CPU then move to device — torch.load on DirectML
    # fails with "TypeError: '>=' not supported between instances of
    # 'torch.device' and 'int'" when given a DirectML device.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("Model loaded and set to eval mode")

    out_path = generate_gradcam_grid(
        model=model,
        dataset=test_loader.dataset,
        device=device,
        class_names=config.evaluation.class_names,
        save_dir=args.output_dir,
        num_samples=args.num_samples,
    )
    print(f"\nGrad-CAM grid saved to: {out_path}")


if __name__ == "__main__":
    main()