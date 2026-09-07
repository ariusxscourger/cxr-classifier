"""End-to-end smoke test against dataset_test/.

Runs:
  1. Config load + dataloader build (verifies class counts)
  2. 1 epoch of training from scratch
  3. Evaluation on the test split (with all plots)
  4. Grad-CAM grid on test images
  5. Single-image inference via the inference path

Run with: python scripts/smoke_run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cxr_classifier.config import load_config
from cxr_classifier.data import get_dataloaders, get_class_distribution
from cxr_classifier.models import create_model, get_model_info
from cxr_classifier.training import Trainer
from cxr_classifier.evaluation import evaluate_model, generate_gradcam_grid
from cxr_classifier.inference_smoke import predict_single_image

import torch


def main() -> None:
    print("=" * 70)
    print("STEP 1 — Load config + build dataloaders")
    print("=" * 70)
    config = load_config("configs/smoke.yaml")
    print(f"data_root = {config.dataset.data_root}")
    print(f"classes   = {config.dataset.classes}")
    print(f"image_size = {config.dataset.image_size}")
    print(f"batch_size = {config.dataset.batch_size}")

    train_loader, val_loader, test_loader = get_dataloaders(config)
    train_dist = get_class_distribution(train_loader.dataset)
    val_dist = get_class_distribution(val_loader.dataset)
    test_dist = get_class_distribution(test_loader.dataset)
    print(f"\ntrain distribution: {train_dist} (total={len(train_loader.dataset)})")
    print(f"val   distribution: {val_dist} (total={len(val_loader.dataset)})")
    print(f"test  distribution: {test_dist} (total={len(test_loader.dataset)})")

    print("\n" + "=" * 70)
    print("STEP 2 — Build model")
    print("=" * 70)
    model = create_model(config)
    info = get_model_info(model)
    print(f"model: {config.model.name}")
    print(f"parameters: {info['total_parameters']:,} ({info['model_size_mb']:.1f} MB)")

    print("\n" + "=" * 70)
    print("STEP 3 — Train 1 epoch from scratch (CPU)")
    print("=" * 70)
    device = torch.device("cpu")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    results = trainer.train()
    print(f"\nTraining finished. best_metric (f1_macro) = {results['best_metric']:.4f}")

    print("\n" + "=" * 70)
    print("STEP 4 — Evaluate on test split")
    print("=" * 70)
    test_results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=config.evaluation.class_names,
        save_dir="outputs/smoke/test_results",
    )
    metrics = test_results["metrics"]
    print(f"\nFinal test metrics:")
    print(f"  accuracy     = {metrics['accuracy']:.4f}")
    print(f"  f1_macro     = {metrics['f1_macro']:.4f}")
    print(f"  auc_macro    = {metrics['auc_macro']:.4f}")

    print("\n" + "=" * 70)
    print("STEP 5 — Grad-CAM grid on test set")
    print("=" * 70)
    out = generate_gradcam_grid(
        model=model,
        dataset=test_loader.dataset,
        device=device,
        class_names=config.evaluation.class_names,
        save_dir="outputs/smoke/gradcam",
        num_samples=2,  # tiny dataset → only 2 per class
    )
    print(f"Saved {out}")

    print("\n" + "=" * 70)
    print("STEP 6 — Single-image inference (one image per class)")
    print("=" * 70)
    for cls in config.dataset.classes:
        cls_dir = Path("dataset_test/test") / cls
        if not cls_dir.exists():
            continue
        sample_paths = sorted(cls_dir.iterdir())
        if cls == "pneumonia":
            # skip the misplaced normal-1.jpg
            sample_paths = [p for p in sample_paths if not p.name.startswith("normal-")]
        if sample_paths:
            img_path = str(sample_paths[0])
            print(f"\n→ {cls}: {img_path}")
            predict_single_image(model, config, img_path, device)


if __name__ == "__main__":
    main()