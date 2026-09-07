#!/usr/bin/env python3
"""
End-to-end CXR-Classifier pipeline.

Runs the full lifecycle in one command:
  1. Train (or resume from the existing best checkpoint)
  2. Evaluate on the test set (full metrics, confusion matrix, ROC, PR)
  3. Grad-CAM visualization (per-class grid + per-class single-class montages)
  4. A polished one-page summary figure that combines the key results

Usage (from the project root, with venv activated):

    # Full run (train + evaluate + gradcam + summary)
    .venv\\Scripts\\python.exe scripts\\run_pipeline.py --config configs\\fast_gpu.yaml

    # Just evaluate and visualize, skipping training
    .venv\\Scripts\\python.exe scripts\\run_pipeline.py --config configs\\fast_gpu.yaml --skip-train

    # Custom device / output dir
    .venv\\Scripts\\python.exe scripts\\run_pipeline.py --config configs\\fast_gpu.yaml --device cpu --output-dir outputs/run1

This is a beginner-friendly entry point — it does everything you need in one command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make `cxr_classifier` importable when running from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader

from cxr_classifier.config import load_config
from cxr_classifier.data import ChestXRayDataset, get_dataloaders
from cxr_classifier.device import get_torch_device
from cxr_classifier.evaluation import (
    evaluate_model as run_evaluation,
    generate_gradcam_grid,
)
from cxr_classifier.models import create_model
from cxr_classifier.training import Trainer


# ---------------------------------------------------------------------------
# CLI + helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full CXR-Classifier pipeline (train/eval/gradcam/summary)."
    )
    parser.add_argument(
        "--config", type=str, default="configs/fast_gpu.yaml",
        help="Path to the YAML config (default: configs/fast_gpu.yaml).",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto | cpu | cuda | directml | mps (default: auto).",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override the dataset root from the config (default: use config).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/pipeline",
        help="Top-level output directory (default: outputs/pipeline).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint to evaluate/visualize. If omitted, uses outputs/best_model.pth.",
    )
    parser.add_argument(
        "--num-gradcam-samples", type=int, default=4,
        help="Number of correctly-classified samples per class for Grad-CAM (default: 4).",
    )
    parser.add_argument("--skip-train", action="store_true", help="Skip training.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation.")
    parser.add_argument("--skip-gradcam", action="store_true", help="Skip Grad-CAM.")
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Skip train/eval/gradcam — only rebuild the summary figure from "
             "existing artifacts in --output-dir.",
    )
    return parser.parse_args()


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _to_python(obj):
    """Recursively coerce numpy scalars to plain Python so json.dump works."""
    import numpy as _np
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return _to_python(obj.tolist())
    return obj


def load_checkpoint(checkpoint_path: str, model, device) -> None:
    """Load a checkpoint into ``model`` (DirectML-safe: load on CPU first)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    state = {k: v.to(device) for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()


def _loaders(config, data_root_override=None):
    """Return (train_loader, val_loader, test_loader) using the project's pipeline."""
    if data_root_override:
        config.dataset.data_root = data_root_override
    return get_dataloaders(config)


def build_loader(config, split: str, data_root_override=None, shuffle=False) -> DataLoader:
    """Build a single-split DataLoader by reusing the project's ``get_dataloaders``."""
    train_loader, val_loader, test_loader = _loaders(config, data_root_override)
    if split == "train":
        return train_loader
    if split == "val":
        return val_loader
    return test_loader


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_train(config, device, output_dir: Path) -> str:
    banner("[1/4] TRAINING")
    trainer = Trainer(
        config=config,
        train_loader=build_loader(config, "train", shuffle=True),
        val_loader=build_loader(config, "val"),
        device=device,
        output_dir=str(output_dir),
    )
    trainer.train()
    best_path = output_dir / "best_model.pth"
    print(f"\n[1/4] Training complete. Best checkpoint: {best_path}")
    return str(best_path)


def step_evaluate(config, device, checkpoint_path: str, eval_dir: Path) -> dict:
    banner("[2/4] EVALUATION")
    model = create_model(config)
    load_checkpoint(checkpoint_path, model, device)

    test_loader = build_loader(config, "test")
    metrics = run_evaluation(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=config.dataset.classes,
        save_dir=str(eval_dir),
    )
    print(
        f"[2/4] Evaluation complete. "
        f"Acc={metrics.get('accuracy', 0):.4f}, "
        f"F1={metrics.get('f1_macro', 0):.4f}, "
        f"AUC={metrics.get('auc_macro', 0):.4f}"
    )
    return metrics


def step_gradcam(
    config, device, checkpoint_path: str, gc_dir: Path, num_samples: int
) -> str:
    banner("[3/4] GRAD-CAM VISUALIZATION")
    # DirectML can crash during backward on some ResNet layers ("GPU device
    # instance has been suspended"). Use CPU for Grad-CAM as a safe fallback
    # — it's only inference + a single backward per image, so it's fast enough.
    import torch as _torch
    gc_device = _torch.device("cpu") if str(device).startswith("privateuseone") else device
    print(f"  Grad-CAM device: {gc_device} (DirectML unstable on backward)")

    model = create_model(config)
    load_checkpoint(checkpoint_path, model, gc_device)

    # Use a simple PIL transform so autograd works (CLAHE/ALBUMENTATIONS uses
    # cv2 ops that are not differentiable, which would break Grad-CAM's
    # backward pass). Just resize + normalize.
    pil_transform = T.Compose([
        T.Resize((config.dataset.image_size, config.dataset.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Wrap the PIL transform so ChestXRayDataset (which uses albumentations
    # np.array -> transform) still works: it accepts np.ndarray, applies
    # transform, returns dict. To use a torchvision transform we need a tiny
    # adapter: convert np.array -> PIL -> tensor.
    from PIL import Image

    class _Adapter:
        def __call__(self, image, **kwargs):
            # albumentations-style call: transform(image=image) -> dict.
            # Return {"image": tensor} so the dataset's transformed["image"]
            # indexing still works.
            pil = Image.fromarray(image)
            return {"image": pil_transform(pil)}

    dataset = ChestXRayDataset(
        data_root=config.dataset.data_root,
        split="test",
        transform=_Adapter(),
        classes=config.dataset.classes,
    )

    grid_path = generate_gradcam_grid(
        model=model,
        dataset=dataset,
        device=gc_device,
        class_names=config.dataset.classes,
        save_dir=str(gc_dir),
        num_samples=num_samples,
    )
    print(f"[3/4] Saved Grad-CAM grid: {grid_path}")
    return str(grid_path)


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def build_summary_figure(
    metrics: dict,
    gradcam_grid_path: str | None,
    config,
    output_path: Path,
) -> None:
    """
    Compose a one-page summary figure:
      - Top-left: per-class bar chart (Precision / Recall / F1)
      - Top-right: headline metrics table
      - Middle row: ROC and PR curves
      - Bottom: Grad-CAM grid

    Accepts both flat and nested metrics dicts. ``run_evaluation`` returns
    ``{"metrics": {...}, "targets": [...], ...}`` (nested); callers using
    the raw ``compute_metrics`` output pass a flat dict. We normalise.
    """
    banner("[4/4] SUMMARY FIGURE")

    # Unwrap nested {"metrics": {...}} if present.
    if isinstance(metrics, dict) and "metrics" in metrics and isinstance(metrics["metrics"], dict):
        m = metrics["metrics"]
    else:
        m = metrics

    classes = config.dataset.classes
    n = len(classes)
    x = np.arange(n)
    width = 0.27

    # Per-class values from the metrics dict (keys end in _per_class).
    precision = m.get("precision_per_class", [0.0] * n)
    recall = m.get("recall_per_class", [0.0] * n)
    f1 = m.get("f1_per_class", [0.0] * n)

    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.4], hspace=0.35, wspace=0.25)

    # --- Top left: per-class bar chart ---
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_bar.bar(x - width, precision, width, label="Precision", color="#3b82f6")
    ax_bar.bar(x, recall, width, label="Recall", color="#10b981")
    ax_bar.bar(x + width, f1, width, label="F1", color="#f59e0b")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([c.capitalize() for c in classes])
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Per-Class Metrics", fontsize=14, fontweight="bold")
    ax_bar.legend(loc="lower right")
    ax_bar.grid(axis="y", alpha=0.3)

    # --- Top right: headline metrics table ---
    ax_table = fig.add_subplot(gs[0, 1])
    ax_table.axis("off")
    rows = [
        ["Accuracy", f"{m.get('accuracy', 0):.4f}"],
        ["Precision (macro)", f"{m.get('precision_macro', 0):.4f}"],
        ["Recall (macro)", f"{m.get('recall_macro', 0):.4f}"],
        ["F1 (macro)", f"{m.get('f1_macro', 0):.4f}"],
        ["AUC (macro)", f"{m.get('auc_macro', 0):.4f}"],
        ["AP (macro)", f"{m.get('ap_macro', 0):.4f}"],
    ]
    tbl = ax_table.table(
        cellText=rows, colLabels=["Metric", "Value"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.2)
    for (row, _), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
    ax_table.set_title("Headline Metrics", fontsize=14, fontweight="bold", pad=14)

    # --- Middle: ROC + PR curves ---
    ax_roc = fig.add_subplot(gs[1, 0])
    ax_pr = fig.add_subplot(gs[1, 1])
    eval_dir = output_path.parent.parent / "eval"
    for ax, name, title in (
        (ax_roc, "roc_curves.png", "ROC Curves"),
        (ax_pr, "pr_curves.png", "Precision-Recall Curves"),
    ):
        img_path = eval_dir / name
        if img_path.exists():
            ax.imshow(plt.imread(str(img_path)))
        else:
            ax.text(0.5, 0.5, f"{title}\n(file not found)", ha="center", va="center")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    # --- Bottom: Grad-CAM grid ---
    ax_gc = fig.add_subplot(gs[2, :])
    if gradcam_grid_path and Path(gradcam_grid_path).exists():
        ax_gc.imshow(plt.imread(gradcam_grid_path))
    else:
        ax_gc.text(0.5, 0.5, "Grad-CAM grid not generated", ha="center", va="center")
    ax_gc.set_title(
        "Grad-CAM: class activation maps (rows = true class)",
        fontsize=14, fontweight="bold",
    )
    ax_gc.axis("off")

    fig.suptitle(
        "CXR-Classifier: Pipeline Summary",
        fontsize=18, fontweight="bold", y=0.995,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[4/4] Summary figure: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    start = time.time()

    # Resolve config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = load_config(str(config_path))
    if args.data_root:
        config.dataset.data_root = args.data_root

    # Device
    device = get_torch_device(args.device)
    print(f"Using device: {device}")

    # Output dirs
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    gc_dir = output_dir / "gradcam"
    gc_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "resolved_config.yaml", "w") as f:
        f.write(str(config))

    # ----- 1) Train (optional) -----
    if args.summary_only:
        banner("[1-3/4] SKIPPED (--summary-only)")
        ckpt_path = args.checkpoint or str(PROJECT_ROOT / "outputs" / "best_model.pth")
        metrics_path = eval_dir / "metrics.json"
        metrics = json.load(open(metrics_path)) if metrics_path.exists() else {}
        # In summary-only mode, look for the Grad-CAM from the smoke run or
        # the most recent pipeline gradcam directory.
        candidates = [
            gc_dir / "gradcam_grid.png",
            PROJECT_ROOT / "outputs" / "smoke" / "gradcam" / "gradcam_grid.png",
            PROJECT_ROOT / "outputs" / "gradcam_trained" / "gradcam_grid.png",
        ]
        gradcam_grid_path = str(
            next((p for p in candidates if p.exists()), candidates[0])
        )
    else:
        if args.skip_train:
            banner("[1/4] SKIPPED (using existing checkpoint)")
            ckpt_path = args.checkpoint or str(PROJECT_ROOT / "outputs" / "best_model.pth")
        else:
            ckpt_path = step_train(config, device, output_dir)
            if args.checkpoint:
                ckpt_path = args.checkpoint  # user override

        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # ----- 2) Evaluate (optional) -----
        if args.skip_eval:
            banner("[2/4] SKIPPED")
            metrics_path = eval_dir / "metrics.json"
            metrics = json.load(open(metrics_path)) if metrics_path.exists() else {}
        else:
            metrics = step_evaluate(config, device, ckpt_path, eval_dir)
            with open(eval_dir / "metrics.json", "w") as f:
                json.dump(_to_python(metrics), f, indent=2)

        # ----- 3) Grad-CAM (optional) -----
        if args.skip_gradcam:
            banner("[3/4] SKIPPED")
            gradcam_grid_path = str(gc_dir / "gradcam_grid.png")
        else:
            gradcam_grid_path = step_gradcam(
                config, device, ckpt_path, gc_dir, args.num_gradcam_samples
            )

    # ----- 4) Summary -----
    build_summary_figure(
        metrics=metrics,
        gradcam_grid_path=gradcam_grid_path,
        config=config,
        output_path=summary_dir / "summary.png",
    )
    # Save metrics next to summary for the convenience of CI / artifact
    # consumers. Some metrics are numpy types; coerce to plain Python.
    if metrics:
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(_to_python(metrics), f, indent=2)

    elapsed = time.time() - start
    banner(f"DONE in {elapsed/60:.1f} minutes")
    print(f"All outputs saved under: {output_dir}")
    print(f"Summary figure: {summary_dir / 'summary.png'}")


if __name__ == "__main__":
    main()
