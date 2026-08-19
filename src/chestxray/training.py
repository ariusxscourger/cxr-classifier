"""
Training module for Chest X-Ray Classification.
Handles training loop, validation, checkpointing, and logging.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from chestxray.config import Config
from chestxray.evaluation import Evaluator


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Trainer:
    """Main training class."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Config,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss function
        self.criterion = self._create_loss(config.training.loss)

        # Optimizer
        self.optimizer = self._create_optimizer(config.training.optimizer)

        # Scheduler
        self.scheduler = self._create_scheduler(config.training.scheduler)

        # Mixed precision
        self.scaler = GradScaler(enabled=config.training.mixed_precision and device.type == "cuda")

        # Gradient clipping
        self.gradient_clip = config.training.gradient_clip

        # Early stopping
        self.early_stopping_patience = config.training.early_stopping.patience
        self.early_stopping_min_delta = config.training.early_stopping.min_delta
        self.early_stopping_mode = config.training.early_stopping.mode
        self.best_metric = float("-inf") if self.early_stopping_mode == "max" else float("inf")
        self.early_stopping_counter = 0

        # Logging
        self.log_dir = Path(config.logging.log_dir)
        self.save_dir = Path(config.logging.save_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.log_interval = config.logging.log_interval
        self.save_interval = config.logging.save_interval
        self.save_best_only = config.training.save_best_only
        self.save_top_k = config.training.save_top_k

        # Metrics tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_metrics: List[Dict[str, float]] = []
        self.learning_rates: List[float] = []

        # WandB and TensorBoard
        self.use_wandb = config.logging.use_wandb
        self.use_tensorboard = config.logging.use_tensorboard
        self._init_loggers(config)

        # Evaluator
        self.evaluator = Evaluator(config.evaluation.class_names, device)

        # Compile model if requested (PyTorch 2.0+)
        if config.hardware.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

    def _create_loss(self, loss_config) -> nn.Module:
        """Create loss function."""
        if loss_config.name == "label_smoothing_cross_entropy":
            return LabelSmoothingCrossEntropy(smoothing=loss_config.label_smoothing)
        elif loss_config.name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_config.name == "focal":
            return FocalLoss(alpha=1, gamma=2)
        else:
            raise ValueError(f"Unknown loss: {loss_config.name}")

    def _create_optimizer(self, opt_config) -> Optimizer:
        """Create optimizer."""
        if opt_config.name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas,
                eps=opt_config.eps,
            )
        elif opt_config.name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas,
                eps=opt_config.eps,
            )
        elif opt_config.name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.name}")

    def _create_scheduler(self, sched_config) -> _LRScheduler:
        """Create learning rate scheduler."""
        if sched_config.name == "cosine_annealing_warm_restarts":
            base_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_config.t_0,
                T_mult=sched_config.t_mult,
                eta_min=sched_config.eta_min,
            )
            return WarmupScheduler(
                self.optimizer,
                base_scheduler,
                warmup_epochs=sched_config.warmup_epochs,
                warmup_lr=sched_config.warmup_lr,
            )
        elif sched_config.name == "cosine_annealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=sched_config.eta_min,
            )
        elif sched_config.name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=10,
                min_lr=sched_config.eta_min,
            )
        elif sched_config.name == "one_cycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.training.optimizer.lr,
                epochs=self.config.training.epochs,
                steps_per_epoch=len(self.train_loader),
            )
        else:
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)

    def _init_loggers(self, config: Config) -> None:
        """Initialize logging backends."""
        self.wandb_run = None
        self.tb_writer = None

        if self.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.logging.wandb_project,
                    entity=config.logging.wandb_entity,
                    config=config.to_dict(),
                    dir=str(self.log_dir),
                )
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                self.use_wandb = False

        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
            except Exception as e:
                print(f"Warning: Failed to initialize tensorboard: {e}")
                self.use_tensorboard = False

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed precision forward
            with autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{total_loss / (batch_idx + 1):.4f}",
                "acc": f"{100.0 * correct / total:.2f}%",
            })

            # Log batch metrics
            if batch_idx % self.log_interval == 0:
                self._log_batch(epoch, batch_idx, loss.item(), correct / total if total > 0 else 0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with autocast(enabled=self.scaler.is_enabled()):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                probs = F.softmax(outputs, dim=1)
                _, preds = outputs.max(1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)

        # Compute metrics
        metrics = self.evaluator.compute_metrics(
            all_targets, all_preds, all_probs
        )
        metrics["loss"] = avg_loss

        return metrics

    def _log_batch(self, epoch: int, batch_idx: int, loss: float, accuracy: float) -> None:
        """Log batch metrics."""
        step = epoch * len(self.train_loader) + batch_idx

        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log({
                "train/batch_loss": loss,
                "train/batch_accuracy": accuracy,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
            }, step=step)

        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_scalar("train/batch_loss", loss, step)
            self.tb_writer.add_scalar("train/batch_accuracy", accuracy, step)
            self.tb_writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], step)

    def _log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict) -> None:
        """Log epoch metrics."""
        if self.use_wandb and self.wandb_run:
            import wandb
            log_dict = {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics.get("accuracy", 0),
                "val/precision_macro": val_metrics.get("precision_macro", 0),
                "val/recall_macro": val_metrics.get("recall_macro", 0),
                "val/f1_macro": val_metrics.get("f1_macro", 0),
                "val/auc_macro": val_metrics.get("auc_macro", 0),
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            wandb.log(log_dict, step=epoch)

        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            self.tb_writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
            self.tb_writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            self.tb_writer.add_scalar("val/accuracy", val_metrics.get("accuracy", 0), epoch)
            self.tb_writer.add_scalar("val/precision_macro", val_metrics.get("precision_macro", 0), epoch)
            self.tb_writer.add_scalar("val/recall_macro", val_metrics.get("recall_macro", 0), epoch)
            self.tb_writer.add_scalar("val/f1_macro", val_metrics.get("f1_macro", 0), epoch)
            self.tb_writer.add_scalar("val/auc_macro", val_metrics.get("auc_macro", 0), epoch)
            self.tb_writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

    def _check_early_stopping(self, metric: float) -> bool:
        """Check early stopping condition."""
        if self.early_stopping_mode == "max":
            improved = metric > self.best_metric + self.early_stopping_min_delta
        else:
            improved = metric < self.best_metric - self.early_stopping_min_delta

        if improved:
            self.best_metric = metric
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= self.early_stopping_patience

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "metrics": metrics,
            "config": self.config.to_dict(),
        }

        # Save latest
        if not self.save_best_only:
            latest_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model at epoch {epoch}")

        # Save top-k
        if self.save_top_k > 0:
            self._manage_top_k_checkpoints(epoch, metrics)

    def _manage_top_k_checkpoints(self, epoch: int, metrics: Dict) -> None:
        """Manage top-k checkpoints."""
        # Simple implementation: keep track of best k epochs
        pass  # Implement based on needs

    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model: {self.config.model.name}")
        print(f"Epochs: {self.config.training.epochs}")

        start_time = time.time()

        for epoch in range(1, self.config.training.epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Step scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("f1_macro", val_metrics.get("accuracy", 0)))
                else:
                    self.scheduler.step()

            # Record metrics
            self.train_losses.append(train_metrics["loss"])
            self.val_losses.append(val_metrics["loss"])
            self.val_metrics.append(val_metrics)
            self.learning_rates.append(self.optimizer.param_groups[0]["lr"])

            # Log epoch
            self._log_epoch(epoch, train_metrics, val_metrics)

            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch}/{self.config.training.epochs} - "
                  f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}% | "
                  f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics.get('accuracy', 0):.2f}% | "
                  f"Val F1: {val_metrics.get('f1_macro', 0):.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.1f}s")

            # Check early stopping
            monitor_metric = val_metrics.get("f1_macro", val_metrics.get("accuracy", 0))
            if self._check_early_stopping(monitor_metric):
                print(f"Early stopping triggered at epoch {epoch}")
                break

            # Save checkpoint
            is_best = monitor_metric == self.best_metric
            if epoch % self.save_interval == 0 or is_best:
                self._save_checkpoint(epoch, val_metrics, is_best)

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f}s")

        # Save final model
        final_path = self.save_dir / "final_model.pth"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
            "final_metrics": val_metrics,
        }, final_path)

        # Close loggers
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.close()

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_metrics": self.val_metrics,
            "learning_rates": self.learning_rates,
            "best_metric": self.best_metric,
        }


class WarmupScheduler:
    """Warmup wrapper for learning rate schedulers."""

    def __init__(
        self,
        optimizer: Optimizer,
        base_scheduler: _LRScheduler,
        warmup_epochs: int,
        warmup_lr: float,
    ):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self) -> None:
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            progress = self.current_epoch / self.warmup_epochs
            for i, group in enumerate(self.optimizer.param_groups):
                group["lr"] = self.warmup_lr + (self.base_lrs[i] - self.warmup_lr) * progress
        else:
            self.base_scheduler.step()
        self.current_epoch += 1

    def state_dict(self) -> Dict:
        return {
            "base_scheduler": self.base_scheduler.state_dict(),
            "current_epoch": self.current_epoch,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.base_scheduler.load_state_dict(state_dict["base_scheduler"])
        self.current_epoch = state_dict["current_epoch"]


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""

    def __init__(self, alpha: float = 1, gamma: float = 2, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss