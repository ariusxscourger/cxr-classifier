"""
Configuration management for Chest X-Ray Classification.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from omegaconf import OmegaConf, DictConfig


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    data_root: str
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    classes: List[str] = field(default_factory=lambda: ["normal", "pneumonia", "tuberculosis"])
    num_classes: int = 3
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "convnext_tiny"
    pretrained: bool = True
    num_classes: int = 3
    drop_path_rate: float = 0.1
    drop_rate: float = 0.2


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.05
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    name: str = "cosine_annealing_warm_restarts"
    t_0: int = 10
    t_mult: int = 2
    eta_min: float = 1e-6
    warmup_epochs: int = 5
    warmup_lr: float = 1e-6


@dataclass
class LossConfig:
    """Loss function configuration."""
    name: str = "label_smoothing_cross_entropy"
    label_smoothing: float = 0.1


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    patience: int = 15
    min_delta: float = 1e-4
    mode: str = "max"


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


@dataclass
class AugmentationConfig:
    """Augmentation configuration."""
    train: List[Dict[str, Any]] = field(default_factory=list)
    val: List[Dict[str, Any]] = field(default_factory=list)
    test: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    use_wandb: bool = True
    wandb_project: str = "cxr-classifier"
    wandb_entity: Optional[str] = None
    use_tensorboard: bool = True
    log_dir: str = "logs"
    log_interval: int = 10
    save_dir: str = "outputs"
    save_interval: int = 1


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1", "auc", "confusion_matrix"])
    class_names: List[str] = field(default_factory=lambda: ["Normal", "Pneumonia", "Tuberculosis"])


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    device: str = "cpu"
    mixed_precision: bool = True
    compile_model: bool = False


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
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        return cls.from_dict(cfg_dict)

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(
            dataset=DatasetConfig(**cfg_dict.get("dataset", {})),
            model=ModelConfig(**cfg_dict.get("model", {})),
            training=TrainingConfig(
                **{k: v for k, v in cfg_dict.get("training", {}).items() 
                   if k not in ["optimizer", "scheduler", "loss", "early_stopping"]},
                optimizer=OptimizerConfig(**cfg_dict.get("training", {}).get("optimizer", {})),
                scheduler=SchedulerConfig(**cfg_dict.get("training", {}).get("scheduler", {})),
                loss=LossConfig(**cfg_dict.get("training", {}).get("loss", {})),
                early_stopping=EarlyStoppingConfig(**cfg_dict.get("training", {}).get("early_stopping", {})),
            ),
            augmentation=AugmentationConfig(**cfg_dict.get("augmentation", {})),
            logging=LoggingConfig(**cfg_dict.get("logging", {})),
            evaluation=EvaluationConfig(**cfg_dict.get("evaluation", {})),
            hardware=HardwareConfig(**cfg_dict.get("hardware", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return OmegaConf.to_container(OmegaConf.structured(self), resolve=True)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(path: Union[str, Path]) -> Config:
    """Load configuration from YAML file."""
    return Config.from_yaml(path)