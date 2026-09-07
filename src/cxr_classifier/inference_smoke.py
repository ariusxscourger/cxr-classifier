"""Single-image inference helper for the smoke test.

Mirrors the logic in scripts/inference.py but reuses an in-memory model
instead of reloading from a checkpoint. Returns predictions for printing.
"""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2


def predict_single_image(
    model: torch.nn.Module,
    config,
    image_path: str,
    device: torch.device,
) -> None:
    """Load + preprocess an image and print top-k predictions."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Could not read {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = A.Compose(
        [
            A.Resize(config.dataset.image_size, config.dataset.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )
    x = transform(image=img)["image"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx = int(np.argmax(probs))
    print(f"  predicted = {config.evaluation.class_names[top_idx]} ({probs[top_idx]*100:.1f}%)")
    for i, name in enumerate(config.evaluation.class_names):
        print(f"    {name}: {probs[i]*100:5.1f}%")