# Chest X-Ray Multi-Class Classification

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Deep Learning Project for Chinese Government Scholarship (CSC) Application**
>
> Multi-class classification of chest X-rays: **Normal**, **Pneumonia**, and **Tuberculosis**

## 📋 Project Overview

This project implements a state-of-the-art deep learning pipeline for automated chest X-ray diagnosis, classifying images into three categories: **Normal**, **Pneumonia**, and **Tuberculosis**. The implementation follows best practices for medical AI research and is designed to meet the rigorous standards expected for **CSC (Chinese Government Scholarship)** applications and academic evaluation by professors.

### Key Highlights

- **Dataset**: 23,553 chest X-ray images (20,450 train / 2,534 val / 2,569 test)
- **Classes**: Normal (9,088), Pneumonia (5,824), Tuberculosis (10,641)
- **Architecture**: Modern CNNs (ConvNeXt, EfficientNet) and Vision Transformers (ViT, Swin)
- **Framework**: PyTorch 2.4+ with timm, Albumentations, mixed precision
- **Tracking**: Weights & Biases + TensorBoard experiment tracking
- **Evaluation**: Comprehensive metrics (AUC, F1, Precision, Recall, ROC/PR curves, Confusion Matrix)

---

## 🏗️ Project Structure

```
ChestXRay-CSC-Project/
├── configs/
│   └── config.yaml                 # Main configuration (YAML)
├── notebooks/
│   └── exploration.ipynb           # Interactive exploration & demo
├── scripts/
│   ├── train.py                    # Training entry point
│   ├── evaluate.py                 # Evaluation entry point
│   └── inference.py                # Single image inference
├── src/
│   └── chestxray/
│       ├── __init__.py
│       ├── config.py               # Configuration dataclasses
│       ├── data.py                 # Dataset & DataLoader
│       ├── models.py               # Model creation & modification
│       ├── training.py             # Training loop & utilities
│       └── evaluation.py           # Metrics & visualization
├── outputs/                        # Model checkpoints & results
├── logs/                           # Training logs (TensorBoard)
├── pyproject.toml                  # Package metadata & dependencies
├── requirements.txt                # Pip requirements
├── .gitignore
└── README.md                       # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **uv** (recommended) or pip
- **8GB+ RAM** (CPU training)
- **Optional**: NVIDIA GPU with 8GB+ VRAM for faster training

### Installation

```bash
# Clone and navigate
cd ~/Documents/Code/ChestXRay-CSC-Project

# Create virtual environment with uv (recommended)
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev,notebook]"

# Or with pip
# pip install -e ".[dev,notebook]"
```

### Dataset

The dataset is expected at `/run/media/saqib/067E00563129A4C4/Chest X-Ray/` with structure:

```
Chest X-Ray/
├── data.yaml
├── train/
│   ├── normal/
│   ├── pneumonia/
│   └── tuberculosis/
├── val/
│   ├── normal/
│   ├── pneumonia/
│   └── tuberculosis/
└── test/
    ├── normal/
    ├── pneumonia/
    └── tuberculosis/
```

**Class Distribution:**
| Split | Normal | Pneumonia | Tuberculosis | Total |
|-------|--------|-----------|--------------|-------|
| Train | 7,263 | 4,674 | 8,513 | 20,450 |
| Val | 900 | 570 | 1,064 | 2,534 |
| Test | 925 | 580 | 1,064 | 2,569 |
| **Total** | **9,088** | **5,824** | **10,641** | **25,553** |

---

## ⚙️ Configuration

All hyperparameters are managed via `configs/config.yaml`:

```yaml
# Model
model:
  name: "convnext_tiny"        # convnext_tiny, convnext_small, efficientnet_b0, etc.
  pretrained: true
  num_classes: 3
  drop_path_rate: 0.1
  drop_rate: 0.2

# Training
training:
  epochs: 100
  optimizer:
    name: "adamw"
    lr: 1e-4
    weight_decay: 0.05
  scheduler:
    name: "cosine_annealing_warm_restarts"
    warmup_epochs: 5
  loss:
    name: "label_smoothing_cross_entropy"
    label_smoothing: 0.1
  early_stopping:
    patience: 15
    mode: "max"

# Augmentation (Albumentations)
augmentation:
  train: [...]
  val: [...]
  test: [...]
```

---

## 🏃 Training

### Basic Training

```bash
# Train with default config
python scripts/train.py --config configs/config.yaml

# Train on specific device
python scripts/train.py --device cpu  # or cuda, mps

# Resume from checkpoint
python scripts/train.py --resume outputs/checkpoint_epoch_50.pth

# Debug mode (1 epoch, smaller dataset)
python scripts/train.py --debug
```

### Monitoring

**TensorBoard:**
```bash
tensorboard --logdir logs
# Open http://localhost:6006
```

**Weights & Biases:**
```bash
# Set your API key
wandb login
# Dashboard at https://wandb.ai/<your-entity>/chestxray-csc
```

### Expected Outputs

```
outputs/
├── best_model.pth              # Best validation model
├── final_model.pth             # Final epoch model
├── checkpoint_epoch_XX.pth     # Periodic checkpoints
└── test_results/               # Test evaluation results
    ├── metrics.json
    ├── classification_report.txt
    ├── confusion_matrix.png
    ├── confusion_matrix_raw.png
    ├── roc_curves.png
    └── pr_curves.png
```

---

## 📊 Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --checkpoint outputs/best_model.pth \
    --config configs/config.yaml \
    --output-dir outputs/evaluation

# Evaluate on validation set
python scripts/evaluate.py --split val --checkpoint outputs/best_model.pth
```

### Key Metrics Reported

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification accuracy |
| **Precision (Macro/Weighted)** | Per-class & averaged precision |
| **Recall (Macro/Weighted)** | Per-class & averaged recall (sensitivity) |
| **F1 (Macro/Weighted)** | Per-class & averaged F1-score |
| **AUC (Macro/Weighted)** | One-vs-Rest ROC AUC |
| **AP (Macro/Weighted)** | Average Precision (PR AUC) |
| **Confusion Matrix** | Normalized & raw counts |
| **ROC Curves** | Per-class ROC visualization |
| **PR Curves** | Per-class Precision-Recall visualization |

---

## 🔮 Inference

```bash
# Single image prediction
python scripts/inference.py \
    --checkpoint outputs/best_model.pth \
    --image path/to/chest_xray.jpg \
    --top-k 3
```

**Output:**
```
==================================================
PREDICTION RESULTS
==================================================
  1. Pneumonia: 87.32%
  2. Tuberculosis: 10.45%
  3. Normal: 2.23%

All class probabilities:
  Normal: 2.23%
  Pneumonia: 87.32%
  Tuberculosis: 10.45%
==================================================
```

---

## 📈 Model Zoo

| Model | Parameters | Size (MB) | Description |
|-------|------------|-----------|-------------|
| `convnext_tiny` | 28.6M | 109 MB | Modern CNN, excellent accuracy/speed |
| `convnext_small` | 50.2M | 192 MB | Larger ConvNeXt variant |
| `efficientnet_b0` | 5.3M | 20 MB | Efficient compound scaling |
| `efficientnet_b3` | 12.2M | 47 MB | Larger EfficientNet |
| `resnet50` | 25.6M | 98 MB | Classic ResNet baseline |
| `vit_tiny_patch16_224` | 5.7M | 22 MB | Vision Transformer tiny |
| `swin_tiny_patch4_window7_224` | 28.3M | 108 MB | Hierarchical Vision Transformer |

**Recommendation for CSC**: Start with `convnext_tiny` (best accuracy/efficiency balance), then try `convnext_small` or `swin_tiny` for higher performance.

---

## 🧪 Reproducibility

```bash
# Set seed for reproducibility
python scripts/train.py --seed 42

# All random seeds controlled:
# - Python random
# - NumPy
# - PyTorch (CPU & CUDA)
# - cuDNN deterministic mode
```

---

## 📝 CSC Scholarship - What Professors Expect

This project demonstrates:

### 1. **Technical Rigor**
- ✅ Clean, modular codebase with type hints
- ✅ Configuration-driven experiments (YAML + Hydra-ready)
- ✅ Proper train/val/test splits with no data leakage
- ✅ Class imbalance handling (WeightedRandomSampler)
- ✅ Modern architectures (ConvNeXt, ViT, Swin)
- ✅ Advanced augmentations (Albumentations)
- ✅ Mixed precision training (AMP)
- ✅ Learning rate scheduling with warmup
- ✅ Label smoothing & focal loss
- ✅ Early stopping & model checkpointing
- ✅ Gradient clipping for stability

### 2. **Evaluation Thoroughness**
- ✅ Per-class metrics (critical for medical imbalance)
- ✅ Macro & weighted averages
- ✅ ROC curves (one-vs-rest)
- ✅ Precision-Recall curves
- ✅ Confusion matrices (normalized & raw)
- ✅ Classification reports
- ✅ Training curve visualization

### 3. **Experiment Tracking**
- ✅ Weights & Biases integration
- ✅ TensorBoard integration
- ✅ Hyperparameter logging
- ✅ Metric logging per batch & epoch
- ✅ Model artifact versioning

### 4. **Software Engineering**
- ✅ Package structure (`src/chestxray/`)
- ✅ Configuration management (dataclasses + YAML)
- ✅ Entry point scripts (train/eval/inference)
- ✅ Jupyter notebook for exploration
- ✅ Type hints throughout
- ✅ Linting (ruff, black, isort, mypy)
- ✅ Testing ready (pytest structure)

### 5. **Documentation**
- ✅ Comprehensive README (this file)
- ✅ Inline docstrings (Google style)
- ✅ Configuration documentation
- ✅ Usage examples
- ✅ Architecture decision rationale

---

## 🔬 Methodology Details

### Data Augmentation Strategy

**Training:**
- Resize 256→224 random crop
- Horizontal flip (p=0.5)
- Rotation ±15° (p=0.5)
- Brightness/Contrast ±20% (p=0.5)
- Gaussian noise (p=0.3)
- CoarseDropout (8 holes, 32×32, p=0.3)
- ImageNet normalization

**Validation/Test:**
- Resize 224→224 center crop
- ImageNet normalization

### Loss Function

**Primary**: Label Smoothing Cross Entropy (ε=0.1)
- Prevents overconfident predictions
- Improves calibration
- Better generalization

**Alternative**: Focal Loss (γ=2, α=1)
- Focuses on hard examples
- Better for severe class imbalance

### Optimization

- **Optimizer**: AdamW (lr=1e-4, wd=0.05)
- **Scheduler**: Cosine Annealing Warm Restarts (T₀=10, T_mult=2)
- **Warmup**: 5 epochs linear warmup (1e-6 → 1e-4)
- **Gradient Clipping**: 1.0
- **Mixed Precision**: FP16 (CUDA) / BF16 (CPU with AMP)

### Class Imbalance Handling

- **WeightedRandomSampler**: Inverse frequency weighting
- **Per-class metrics**: Macro averaging (equally weight classes)
- **Loss weighting**: Optional class-weighted loss

---

## 📚 References

1. **ConvNeXt**: Liu et al., "A ConvNet for the 2020s", CVPR 2022
2. **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling", ICML 2019
3. **Vision Transformer**: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
4. **Swin Transformer**: Liu et al., "Swin Transformer: Hierarchical Vision Transformer", ICCV 2021
5. **Label Smoothing**: Szegedy et al., "Rethinking the Inception Architecture", CVPR 2016
6. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
7. **Albumentations**: Buslaev et al., "Albumentations: Fast and Flexible Image Augmentations", Information 2020

---

## 🎓 Academic Use

This project is designed for:
- **CSC Scholarship Application** (Chinese Government Scholarship)
- **Master's/PhD Research Portfolio**
- **Medical AI Course Projects**
- **Conference Paper Implementation** (MICCAI, MIDL, CVPR workshops)

### Citation

If you use this codebase for academic work, please cite:

```bibtex
@software{chestxray_csc_2025,
  author = {Muhammad Saqib},
  title = {Chest X-Ray Multi-Class Classification for CSC Scholarship},
  year = {2025},
  url = {https://github.com/yourusername/ChestXRay-CSC-Project}
}
```

---

## 🤝 Contributing

This is an academic project for scholarship application. However, improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Muhammad Saqib**
- BS Software Engineering, SZABIST (CGPA 3.13)
- JPMorgan FinTech Backend (2023-2025)
- Solutec Power BI/ML Engineer (2025-present)
- Target: Fully-funded Master's AI/ML (Fall 2027)
- Specializations: MLOps, Urdu NLP/LLMs, CV Medical/Industrial, FinTech AI, Data Engineering

---

## 🙏 Acknowledgments

- **timm** library by Ross Wightman for model zoo
- **Albumentations** team for augmentation pipeline
- **PyTorch** team for the framework
- Chest X-Ray dataset contributors
- CSC scholarship program for motivation

---

**Built with ❤️ for CSC Scholarship Application 2025-2026**