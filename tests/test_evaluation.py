"""Smoke tests for the Evaluator (metrics, plots)."""
from __future__ import annotations

import numpy as np
import torch

from cxr_classifier.evaluation import Evaluator, GradCAM


def _make_synthetic_data(n: int = 30, num_classes: int = 3, seed: int = 0):
    """Build balanced synthetic targets/predictions/probabilities."""
    rng = np.random.default_rng(seed)
    targets = np.array([i % num_classes for i in range(n)])
    # Pretend the model gets ~70% correct
    preds = targets.copy()
    flips = rng.choice(n, size=n // 3, replace=False)
    for i in flips:
        preds[i] = (preds[i] + 1) % num_classes
    # Probabilities: high confidence for the prediction
    probs = np.zeros((n, num_classes), dtype=np.float32)
    for i in range(n):
        probs[i, preds[i]] = 0.7
        others = [c for c in range(num_classes) if c != preds[i]]
        rem = 0.3 / len(others)
        for c in others:
            probs[i, c] = rem
    return targets.tolist(), preds.tolist(), probs.tolist()


def test_compute_metrics_keys_present():
    """All expected metric keys should appear in the output."""
    evaluator = Evaluator(["A", "B", "C"], torch.device("cpu"))
    targets, preds, probs = _make_synthetic_data()
    metrics = evaluator.compute_metrics(targets, preds, probs)

    for key in [
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
        "auc_macro", "auc_weighted",
        "ap_macro", "ap_weighted",
        "precision_per_class", "recall_per_class", "f1_per_class",
        "auc_per_class", "ap_per_class",
    ]:
        assert key in metrics, f"missing metric: {key}"

    assert metrics["accuracy"] >= 0.6  # ~70% by construction
    assert metrics["f1_macro"] >= 0.5
    assert len(metrics["precision_per_class"]) == 3


def test_compute_metrics_handles_perfect_classifier():
    """A perfect classifier should score 1.0 across the board."""
    evaluator = Evaluator(["A", "B", "C"], torch.device("cpu"))
    targets = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    preds = targets.copy()
    probs = np.eye(3)[targets].astype(np.float32).tolist()
    metrics = evaluator.compute_metrics(targets, preds, probs)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["auc_macro"] == 1.0


def test_get_confusion_matrix_normalization():
    """Normalized confusion matrix should have rows that sum to ~1."""
    evaluator = Evaluator(["A", "B", "C"], torch.device("cpu"))
    targets, preds, _ = _make_synthetic_data()
    cm = evaluator.get_confusion_matrix(targets, preds, normalize="true")
    row_sums = cm.sum(axis=1)
    # Allow NaN rows (for classes with zero support), which we replace with 0
    assert np.allclose(row_sums[~np.isnan(row_sums)], 1.0, atol=1e-3)


def test_gradcam_runs_on_synthetic_input():
    """GradCAM should produce a heatmap of the right shape on a synthetic batch."""
    from cxr_classifier.models import create_model
    from cxr_classifier.config import Config, ModelConfig

    cfg = Config()
    cfg.model = ModelConfig(name="resnet18", pretrained=False, num_classes=3)
    model = create_model(cfg)
    model.eval()

    gradcam = GradCAM(model)
    try:
        x = torch.randn(1, 3, 64, 64)
        heatmap = gradcam(x, class_idx=1)
        assert heatmap.shape == (64, 64)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0 + 1e-5
    finally:
        gradcam.remove_hooks()