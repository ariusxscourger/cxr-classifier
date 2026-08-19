"""
Configuration management for Chest X-Ray Classification.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    data_root: str
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    classes: list[str] = field(default_factory=lambda: ["normal", "pneumonia", "tuberculosis"])
    num_classes: int = 3
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    def __post_init__(self) -> None:
        self.num_classes = int(self.num_classes)
        self.image_size = int(self.image_size)
        self.batch_size = int(self.batch_size)
        self.num_workers = int(self.num_workers)
        self.pin_memory = bool(self.pin_memory)
        self.persistent_workers = bool(self.persistent_workers)


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = "convnext_tiny"
    pretrained: bool = True
    num_classes: int = 3
    drop_path_rate: float = 0.1
    drop_rate: float = 0.2

    def __post_init__(self) -> None:
        self.pretrained = bool(self.pretrained)
        self.num_classes = int(self.num_classes)
        self.drop_path_rate = float(self.drop_path_rate)
        self.drop_rate = float(self.drop_rate)


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.05
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.lr = float(self.lr)
        self.weight_decay = float(self.weight_decay)
        self.eps = float(self.eps)
        if self.betas:
            self.betas = [float(b) for b in self.betas]


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    name: str = "cosine_annealing_warm_restarts"
    t_0: int = 10
    t_mult: int = 2
    eta_min: float = 1e-6
    warmup_epochs: int = 5
    warmup_lr: float = 1e-6

    def __post_init__(self) -> None:
        self.t_0 = int(self.t_0)
        self.t_mult = int(self.t_mult)
        self.eta_min = float(self.eta_min)
        self.warmup_epochs = int(self.warmup_epochs)
        self.warmup_lr = float(self.warmup_lr)


@dataclass
class LossConfig:
    """Loss function configuration."""

    name: str = "label_smoothing_cross_entropy"
    label_smoothing: float = 0.1

    def __post_init__(self) -> None:
        self.label_smoothing = float(self.label_smoothing)


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""

    patience: int = 15
    min_delta: float = 1e-4
    mode: str = "max"

    def __post_init__(self) -> None:
        self.patience = int(self.patience)
        self.min_delta = float(self.min_delta)


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int = 100
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    save_best_only: bool = True
    save_top_k: int = 3

    def __post_init__(self) -> None:
        self.epochs = int(self.epochs)
        self.gradient_clip = float(self.gradient_clip)
        self.mixed_precision = bool(self.mixed_precision)
        self.save_best_only = bool(self.save_best_only)
        self.save_top_k = int(self.save_top_k)


@dataclass
class AugmentationConfig:
    """Augmentation configuration."""

    train: list[dict[str, Any]] = field(default_factory=list)
    val: list[dict[str, Any]] = field(default_factory=list)
    test: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LoggingConfig:
    """Logging configuration."""

    use_wandb: bool = True
    wandb_project: str = "cxr-classifier"
    wandb_entity: str | None = None
    use_tensorboard: bool = True
    log_dir: str = "logs"
    log_interval: int = 10
    save_dir: str = "outputs"
    save_interval: int = 1

    def __post_init__(self) -> None:
        self.use_wandb = bool(self.use_wandb)
        self.use_tensorboard = bool(self.use_tensorboard)
        self.log_interval = int(self.log_interval)
        self.save_interval = int(self.save_interval)


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    metrics: list[str] = field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1", "auc", "confusion_matrix"]
    )
    class_names: list[str] = field(default_factory=lambda: ["Normal", "Pneumonia", "Tuberculosis"])


@dataclass
class HardwareConfig:
    """Hardware configuration."""

    device: str = "cpu"
    mixed_precision: bool = True
    compile_model: bool = False

    def __post_init__(self) -> None:
        self.mixed_precision = bool(self.mixed_precision)
        self.compile_model = bool(self.compile_model)


@dataclass
class Config:
    """Main configuration class."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            cfg_dict = yaml.safe_load(f)
        return cls.from_dict(cfg_dict)

    @classmethod
    def from_dict(cls, cfg_dict: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(
            dataset=DatasetConfig(**cfg_dict.get("dataset", {})),
            model=ModelConfig(**cfg_dict.get("model", {})),
            training=TrainingConfig(
                **{
                    k: v
                    for k, v in cfg_dict.get("training", {}).items()
                    if k not in ["optimizer", "scheduler", "loss", "early_stopping"]
                },
                optimizer=OptimizerConfig(**cfg_dict.get("training", {}).get("optimizer", {})),
                scheduler=SchedulerConfig(**cfg_dict.get("training", {}).get("scheduler", {})),
                loss=LossConfig(**cfg_dict.get("training", {}).get("loss", {})),
                early_stopping=EarlyStoppingConfig(
                    **cfg_dict.get("training", {}).get("early_stopping", {})
                ),
            ),
            augmentation=AugmentationConfig(**cfg_dict.get("augmentation", {})),
            logging=LoggingConfig(**cfg_dict.get("logging", {})),
            evaluation=EvaluationConfig(**cfg_dict.get("evaluation", {})),
            hardware=HardwareConfig(**cfg_dict.get("hardware", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return OmegaConf.to_container(OmegaConf.structured(self), resolve=True)

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(path: str | Path) -> Config:
    """Load configuration from YAML file."""
    return Config.from_yaml(path)
