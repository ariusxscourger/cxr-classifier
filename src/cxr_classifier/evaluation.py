"""
Evaluation module for Chest X-Ray Classification.
Computes comprehensive metrics and generates visualizations.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import seaborn as sns


class Evaluator:
    """Comprehensive evaluation for multi-class classification."""

    def __init__(self, class_names: List[str], device: torch.device):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device = device

    def compute_metrics(
        self,
        targets: List[int],
        predictions: List[int],
        probabilities: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            targets: Ground truth labels
            predictions: Predicted labels
            probabilities: Predicted probabilities for each class

        Returns:
            Dictionary of metrics
        """
        targets = np.array(targets)
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)

        metrics = {}

        # Overall accuracy
        metrics["accuracy"] = accuracy_score(targets, predictions)

        # Per-class metrics
        metrics["precision_per_class"] = precision_score(targets, predictions, average=None, zero_division=0).tolist()
        metrics["recall_per_class"] = recall_score(targets, predictions, average=None, zero_division=0).tolist()
        metrics["f1_per_class"] = f1_score(targets, predictions, average=None, zero_division=0).tolist()

        # Macro averages (equally weight each class)
        metrics["precision_macro"] = precision_score(targets, predictions, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(targets, predictions, average="macro", zero_division=0)
        metrics["f1_macro"] = f1_score(targets, predictions, average="macro", zero_division=0)

        # Weighted averages (weight by support)
        metrics["precision_weighted"] = precision_score(targets, predictions, average="weighted", zero_division=0)
        metrics["recall_weighted"] = recall_score(targets, predictions, average="weighted", zero_division=0)
        metrics["f1_weighted"] = f1_score(targets, predictions, average="weighted", zero_division=0)

        # AUC metrics (one-vs-rest)
        try:
            # Macro AUC
            metrics["auc_macro"] = roc_auc_score(
                targets, probabilities, multi_class="ovr", average="macro"
            )
            # Weighted AUC
            metrics["auc_weighted"] = roc_auc_score(
                targets, probabilities, multi_class="ovr", average="weighted"
            )
            # Per-class AUC
            metrics["auc_per_class"] = roc_auc_score(
                targets, probabilities, multi_class="ovr", average=None
            ).tolist()
        except ValueError:
            # Handle case where a class has no samples
            metrics["auc_macro"] = 0.0
            metrics["auc_weighted"] = 0.0
            metrics["auc_per_class"] = [0.0] * self.num_classes

        # Average Precision (AP)
        try:
            metrics["ap_macro"] = average_precision_score(
                self._to_onehot(targets), probabilities, average="macro"
            )
            metrics["ap_weighted"] = average_precision_score(
                self._to_onehot(targets), probabilities, average="weighted"
            )
            metrics["ap_per_class"] = average_precision_score(
                self._to_onehot(targets), probabilities, average=None
            ).tolist()
        except ValueError:
            metrics["ap_macro"] = 0.0
            metrics["ap_weighted"] = 0.0
            metrics["ap_per_class"] = [0.0] * self.num_classes

        return metrics

    def _to_onehot(self, targets: np.ndarray) -> np.ndarray:
        """Convert integer targets to one-hot encoding."""
        onehot = np.zeros((len(targets), self.num_classes))
        onehot[np.arange(len(targets)), targets] = 1
        return onehot

    def get_confusion_matrix(
        self,
        targets: List[int],
        predictions: List[int],
        normalize: str = "true",
    ) -> np.ndarray:
        """Compute confusion matrix."""
        cm = confusion_matrix(targets, predictions, labels=range(self.num_classes))
        if normalize == "true":
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
        elif normalize == "pred":
            cm = cm.astype("float") / cm.sum(axis=0, keepdims=True)
            cm = np.nan_to_num(cm)
        elif normalize == "all":
            cm = cm.astype("float") / cm.sum()
        return cm

    def plot_confusion_matrix(
        self,
        targets: List[int],
        predictions: List[int],
        save_path: Optional[str] = None,
        normalize: str = "true",
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot confusion matrix."""
        cm = self.get_confusion_matrix(targets, predictions, normalize=normalize)

        normalized = normalize is not None and normalize != "none"
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalized else "d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            cbar_kws={"label": "Normalized Frequency" if normalized else "Count"},
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        normalize_label = normalize.capitalize() if normalize else "Raw"
        ax.set_title(f"Confusion Matrix ({normalize_label})")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_roc_curves(
        self,
        targets: List[int],
        probabilities: List[np.ndarray],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot ROC curves for each class."""
        targets = np.array(targets)
        probabilities = np.array(probabilities)
        onehot_targets = self._to_onehot(targets)

        fig, ax = plt.subplots(figsize=figsize)

        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(onehot_targets[:, i], probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{self.class_names[i]} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (One-vs-Rest)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_precision_recall_curves(
        self,
        targets: List[int],
        probabilities: List[np.ndarray],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot Precision-Recall curves for each class."""
        targets = np.array(targets)
        probabilities = np.array(probabilities)
        onehot_targets = self._to_onehot(targets)

        fig, ax = plt.subplots(figsize=figsize)

        for i in range(self.num_classes):
            precision, recall, _ = precision_recall_curve(onehot_targets[:, i], probabilities[:, i])
            ap = average_precision_score(onehot_targets[:, i], probabilities[:, i])
            ax.plot(recall, precision, label=f"{self.class_names[i]} (AP = {ap:.3f})")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        val_metrics: List[Dict[str, float]],
        learning_rates: List[float],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10),
    ) -> plt.Figure:
        """Plot training curves."""
        epochs = range(1, len(train_losses) + 1)

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        # Loss curves
        axes[0].plot(epochs, train_losses, label="Train Loss", color="blue")
        axes[0].plot(epochs, val_losses, label="Val Loss", color="orange")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        val_acc = [m.get("accuracy", 0) for m in val_metrics]
        axes[1].plot(epochs, val_acc, label="Val Accuracy", color="green")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # F1 Macro
        val_f1 = [m.get("f1_macro", 0) for m in val_metrics]
        axes[2].plot(epochs, val_f1, label="Val F1 Macro", color="red")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("F1 Score")
        axes[2].set_title("Validation F1 Macro")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # AUC Macro
        val_auc = [m.get("auc_macro", 0) for m in val_metrics]
        axes[3].plot(epochs, val_auc, label="Val AUC Macro", color="purple")
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("AUC")
        axes[3].set_title("Validation AUC Macro")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        # Learning Rate
        axes[4].plot(epochs, learning_rates, label="Learning Rate", color="brown")
        axes[4].set_xlabel("Epoch")
        axes[4].set_ylabel("Learning Rate")
        axes[4].set_title("Learning Rate Schedule")
        axes[4].set_yscale("log")
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

        # Per-class F1
        for i, class_name in enumerate(self.class_names):
            class_f1 = [m.get("f1_per_class", [0]*self.num_classes)[i] for m in val_metrics]
            axes[5].plot(epochs, class_f1, label=class_name)
        axes[5].set_xlabel("Epoch")
        axes[5].set_ylabel("F1 Score")
        axes[5].set_title("Per-Class F1 Score")
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def generate_classification_report(
        self,
        targets: List[int],
        predictions: List[int],
        save_path: Optional[str] = None,
    ) -> str:
        """Generate detailed classification report."""
        report = classification_report(
            targets,
            predictions,
            target_names=self.class_names,
            digits=4,
            zero_division=0,
        )

        if save_path:
            with open(save_path, "w") as f:
                f.write(report)

        return report

    def save_metrics(
        self,
        metrics: Dict[str, float],
        save_path: str,
    ) -> None:
        """Save metrics to JSON file."""
        import json

        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        metrics = convert(metrics)

        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """Print metrics in a formatted way."""
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (Macro):    {metrics['recall_macro']:.4f}")
        print(f"F1 (Macro):        {metrics['f1_macro']:.4f}")
        print(f"AUC (Macro):       {metrics['auc_macro']:.4f}")
        print(f"AP (Macro):        {metrics['ap_macro']:.4f}")
        print("-" * 60)
        print("Per-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision_per_class'][i]:.4f}")
            print(f"    Recall:    {metrics['recall_per_class'][i]:.4f}")
            print(f"    F1:        {metrics['f1_per_class'][i]:.4f}")
            print(f"    AUC:       {metrics['auc_per_class'][i]:.4f}")
        print("=" * 60)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete model evaluation pipeline.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run evaluation on
        class_names: List of class names
        save_dir: Directory to save results

    Returns:
        Dictionary containing all metrics and predictions
    """
    model.eval()
    evaluator = Evaluator(class_names, device)

    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Compute metrics
    metrics = evaluator.compute_metrics(all_targets, all_predictions, all_probabilities)

    # Print metrics
    evaluator.print_metrics(metrics)

    # Generate classification report
    report = evaluator.generate_classification_report(all_targets, all_predictions)

    # Save results if directory provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        evaluator.save_metrics(metrics, save_dir / "metrics.json")

        with open(save_dir / "classification_report.txt", "w") as f:
            f.write(report)

        # Save predictions
        np.save(save_dir / "targets.npy", np.array(all_targets))
        np.save(save_dir / "predictions.npy", np.array(all_predictions))
        np.save(save_dir / "probabilities.npy", np.array(all_probabilities))

        # Generate plots
        evaluator.plot_confusion_matrix(
            all_targets, all_predictions, save_dir / "confusion_matrix.png"
        )
        evaluator.plot_confusion_matrix(
            all_targets, all_predictions, save_dir / "confusion_matrix_raw.png", normalize="none"
        )
        evaluator.plot_roc_curves(all_targets, all_probabilities, save_dir / "roc_curves.png")
        evaluator.plot_precision_recall_curves(
            all_targets, all_probabilities, save_dir / "pr_curves.png"
        )

    return {
        "metrics": metrics,
        "targets": all_targets,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
        "classification_report": report,
    }


# Add tqdm import if not already imported
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable