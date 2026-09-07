# CXR-Classifier

[![CI](https://github.com/ariusxscourger/cxr-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/ariusxscourger/cxr-classifier/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/ariusxscourger/cxr-classifier/pkgs/container/cxr-classifier)

> **A reproducible deep-learning pipeline for multi-class chest X-ray diagnosis**
>
> Classifies frontal chest radiographs into **Normal**, **Pneumonia**, or **Tuberculosis** — implemented as a clean reference pipeline that mirrors the choices made in published medical-imaging research, with full configuration of every hyperparameter, an explicit design rationale, and a testable evaluation surface.

---

## 1. Abstract

This repository implements an end-to-end image-classification pipeline for a clinically meaningful task: triaging frontal chest radiographs into three diagnostic categories. The pipeline uses modern convolutional and transformer backbones (ConvNeXt, EfficientNet, ResNet, ViT, Swin) initialised with ImageNet-pretrained weights, fine-tuned on the public Kaggle "Chest X-Ray (Pneumonia, TB)" corpus (25,553 images, three classes). Training uses AdamW with cosine annealing warm restarts and a linear warm-up, label-smoothed cross-entropy, mixed-precision arithmetic, gradient clipping, and a WeightedRandomSampler to mitigate class imbalance. Evaluation reports per-class precision/recall/F1, macro- and weighted averages, one-vs-rest ROC-AUC and average precision, normalised and raw confusion matrices, ROC and PR curves, and gradient-weighted class-activation maps (Grad-CAM) for qualitative failure analysis. The project is deliberately written to be readable top-to-bottom: every design choice is either justified in-code or referenced here.

---

## 2. Problem formulation

Let $\mathcal{X} \subset \mathbb{R}^{H \times W \times 3}$ denote the space of frontal chest radiographs and $\mathcal{Y} = \{0,1,2\}$ denote the label set $\{ \text{Normal}, \text{Pneumonia}, \text{Tuberculosis}\}$. We seek a parametric classifier

$$
f_\theta : \mathcal{X} \rightarrow \Delta^{2},
$$

mapping each radiograph to a probability vector over the three classes. The objective is the expected cross-entropy

$$
\mathcal{L}(\theta) \;=\; \mathbb{E}_{(x,y) \sim \mathcal{D}} \bigl[\, \mathrm{CE}(f_\theta(x), y) \,\bigr],
$$

where $\mathcal{D}$ is the empirical data distribution defined by the dataset publisher. The task is **multi-class single-label classification**: each image has exactly one ground-truth label, and the classifier is evaluated both on the *argmax* decision (accuracy, F1) and on the full probability vector (AUC, calibration).

This formulation is standard but worth stating explicitly because it rules out a number of choices that are inappropriate for the task: ordinal losses (the three classes are not naturally ordered), regression on a single severity axis (we have discrete diagnoses), and survival analysis (no time-to-event component).

---

## 3. Dataset

| Property | Value |
|---|---|
| Source | [muhammadrehan00/chest-xray-dataset](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset) (Kaggle) |
| Modalities aggregated | NIH ChestX-ray14, RSNA Pneumonia Challenge, TB portals |
| Total images | 25,553 |
| Classes | Normal (9,088), Pneumonia (5,824), Tuberculosis (10,641) |
| Pre-defined splits | Train 20,450 · Val 2,534 · Test 2,569 |
| Native resolution | variable (256×256 to 1024×1024); resized to 224×224 |
| Channels | 3 (RGB; greyscale radiographs are replicated across channels) |

### 3.1 Class imbalance

The class prior is mildly imbalanced: tuberculosis represents ~42 % of the corpus, normal ~36 %, pneumonia ~23 %. The naïve empirical risk minimiser will over-predict tuberculosis and under-predict pneumonia. We mitigate this in **three complementary ways**:

1. **WeightedRandomSampler** with inverse-frequency weights, so each minibatch sees approximately equal class counts.
2. **Label smoothing** ($\varepsilon = 0.1$), which prevents the network from being over-confident on majority classes and has been shown empirically to improve calibration on long-tailed distributions (Szegedy et al., 2016).
3. **Macro-averaged metrics** in evaluation, which weight every class equally regardless of support — the correct protocol for imbalanced medical-imaging benchmarks (Saito & Rehmsmeier, 2015).

### 3.2 Patient-disjoint splits — important caveat

> This Kaggle dataset aggregates CXRs from multiple public sources (including NIH ChestX-ray14,
> RSNA Pneumonia Challenge, and TB portals). The **official train/val/test
> partitions provided by the publisher are NOT verified to be
> patient-disjoint**; the same patient may appear in more than one split.
> For any clinical or publishable claim, you must re-partition the data
> yourself at the patient level (or the image-source level for aggregated
> datasets) before training, otherwise reported metrics will be optimistic.
> See `dataset/data.yaml` for the publisher's split manifest.

This caveat is critical. Patient overlap between training and test partitions is a well-known source of inflated performance estimates in chest-radiograph benchmarks (e.g., the NIH dataset contains multiple images per patient; see Oakden-Rayner et al., 2020, "Hidden Stratification Causes Clinically Meaningful Failures in Machine Learning for Medical Imaging").

---

## 4. Methodology

### 4.1 Pre-processing

The pipeline applies the following deterministic transforms **before** any augmentation:

| Stage | Transform | Rationale |
|---|---|---|
| Validation / Test | **CLAHE** (clip limit 2.0, 8 × 8 tile grid) | Contrast-Limited Adaptive Histogram Equalisation enhances local contrast in soft-tissue regions where pathology lives. Standard preprocessing in published CXR pipelines (Gordienko et al., 2018; Hassan et al., 2021). |
| Validation / Test | Resize to 256 × 256, then center-crop 224 × 224 | Matches the receptive-field statistics the ImageNet-pretrained backbone was tuned on. |
| Training | Resize 256 × 256, then **random crop** 224 × 224 | Mild translation invariance, prevents the model from relying on border pixels. |
| Training | Horizontal flip (p = 0.5) | Chest radiographs are approximately mirror-symmetric across the sagittal plane. |
| Training | Rotation ±15° (p = 0.5) | Mild pose variation; larger rotations would distort anatomy. |
| Training | Brightness / contrast ±20 % (p = 0.5) | Simulates exposure variability across X-ray machines. |
| Training | Gaussian noise, σ ∈ [0.1, 0.2] (p = 0.3) | Robustness to sensor noise. |
| Training | CoarseDropout (1–8 holes, 5–15 % of image, p = 0.3) | Forces the network to use multiple regions rather than memorising one (DeVries & Taylor, 2017). |
| All stages | Normalise to ImageNet mean / std | Required so that the pretrained backbone sees data in its training distribution. |

All augmentations are implemented with **Albumentations** (Buslaev et al., 2020), which is faster and more reproducible than `torchvision.transforms` for this class of operations.

### 4.2 Model architecture

We treat this as a transfer-learning problem: ImageNet-pretrained backbones have already learned generic visual primitives (edges, textures, parts) that are useful for medical imaging, despite the domain gap (Raghu et al., 2019). The pipeline supports the following families via the [timm](https://github.com/huggingface/pytorch-image-models) library:

| Backbone | Parameters | Why include it |
|---|---|---|
| **ConvNeXt-Tiny** (Liu et al., 2022) | 28.6 M | The reference "modernised ConvNet" — competitive with Swin at lower compute. Recommended default. |
| **ConvNeXt-Small** | 50.2 M | Larger ConvNeXt for higher capacity. |
| **EfficientNet-B0** (Tan & Le, 2019) | 5.3 M | Compound-scaling baseline; useful when memory is tight. |
| **EfficientNet-B3** | 12.0 M | Mid-range compound-scaled model. |
| **ResNet-50** (He et al., 2016) | 25.6 M | Classic residual baseline; standard comparison. |
| **ViT-Tiny/16** (Dosovitskiy et al., 2021) | 5.7 M | Pure-attention baseline. |
| **Swin-Tiny** (Liu et al., 2021) | 28.3 M | Hierarchical attention; often best on small medical datasets. |

For each backbone, the original ImageNet classifier head is **replaced** with a 512-unit bottleneck (`Dropout → Linear → BatchNorm → ReLU → Dropout → Linear`), giving two extra regularisation levers (dropout before and after the hidden layer) and a small fully-connected head that empirically transfers better than the raw timm head for small medical datasets. The rest of the backbone is fine-tuned end-to-end at learning rate $10^{-4}$.

### 4.3 Loss function

We use **label-smoothed cross-entropy** (Szegedy et al., 2016):

$$
\mathcal{L}_{\mathrm{LS}}(p, y) \;=\; (1-\varepsilon)\,\mathrm{CE}(p, y) \;+\; \varepsilon \cdot \mathrm{CE}\bigl(p, \mathcal{U}\{1,\dots,K\}\bigr),
$$

with $\varepsilon = 0.1$. This is mathematically equivalent to a KL-divergence to a mixture of the one-hot label and a uniform distribution. It has three empirically-validated benefits: (i) prevents the network from outputting over-confident probabilities, which improves calibration for clinical decision-support; (ii) a mild regulariser that reduces overfitting on small datasets; (iii) empirically outperforms hard cross-entropy on most modern architectures (Müller et al., 2019).

As an alternative, the pipeline also supports **focal loss** (Lin et al., 2017):

$$
\mathcal{L}_{\mathrm{focal}}(p, y) \;=\; -\alpha\,(1-p_y)^\gamma \log p_y,
$$

with $\gamma = 2$ and $\alpha = 1$. Focal loss is appropriate when class imbalance is severe; for this dataset the WeightedRandomSampler already addresses imbalance, so label smoothing is the default.

### 4.4 Optimisation

- **Optimiser**: AdamW (Loshchilov & Hutter, 2019) with $\beta = (0.9, 0.999)$, $\varepsilon = 10^{-8}$, weight decay $0.05$. AdamW decouples weight decay from the gradient update, which is the correct formulation for fine-tuning pretrained models.
- **Learning rate**: $10^{-4}$. This is one order of magnitude below the typical ImageNet training rate, which is the standard prescription for fine-tuning (Howard & Ruder, 2018).
- **Scheduler**: Cosine Annealing with Warm Restarts (`T_0 = 10`, `T_mult = 2`, `eta_min = 10^{-6}`) wrapped in a 5-epoch linear warm-up. The warm-up prevents early gradient explosions on the randomly-initialised head; the warm-restarts allow the optimiser to escape sharp minima late in training (Loshchilov & Hutter, 2017).
- **Mixed precision**: FP16 on CUDA via `torch.amp.GradScaler`; the AMP-correct gradient-clipping pattern is used (`unscale_` only when the scaler is enabled).
- **Gradient clipping**: L2 norm clipped to 1.0 — guards against occasional gradient spikes common with focal-style losses.
- **Early stopping**: patience 15 epochs on macro-F1.
- **Top-k checkpointing**: the three best-validation checkpoints are retained by F1-macro.
- **Reproducibility**: all of `random`, `numpy`, `torch` (CPU + CUDA), and `torch.backends.cudnn` are seeded; `cudnn.deterministic = True`.

### 4.5 Evaluation protocol

We report the following metrics on the held-out test set:

| Metric | Why |
|---|---|
| Accuracy | Overall correctness; baseline metric. |
| Precision / Recall / F1 (per-class, macro, weighted) | Per-class performance is the right protocol for imbalanced data; macro-averaging weights all classes equally. |
| AUC, macro- and per-class (one-vs-rest) | Threshold-independent ranking quality. |
| Average precision (PR-AUC) | More informative than ROC-AUC on imbalanced datasets (Saito & Rehmsmeier, 2015). |
| Normalised confusion matrix | Reveals which class pairs are confused. |
| ROC curves, PR curves | Visual threshold-independent diagnostics. |
| Grad-CAM heatmaps (Selvaraju et al., 2017) | Sanity check that the model attends to clinically plausible regions (lung fields) rather than spurious borders / markers. |

For medical-imaging benchmarks, AUC and F1-macro are the most informative headline numbers; raw accuracy is misleading on imbalanced data.

### 4.6 Interpretability

The `evaluation.GradCAM` class implements **Gradient-weighted Class Activation Mapping** (Selvaraju et al., 2017) for any backbone that exposes `model.stages` (ConvNeXt, Swin, EfficientNet) or `model.layer4` (ResNet). Grad-CAM produces a coarse localisation map by

1. Forward-propagating the input to the last convolutional block and capturing the activations $A^k$.
2. Computing the gradient of the target-class score $y^c$ with respect to $A^k$ and global-average pooling it to obtain channel weights $w_k^c = \frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A^k_{i,j}}$.
3. Computing the heatmap as $\mathrm{ReLU}(\sum_k w_k^c A^k)$ and upsampling to the input resolution.

This is a **necessary diagnostic** for any medical-imaging model before deployment: a model that achieves high accuracy by attending to hospital corner tags or patient demographics has not learned the task and should not be deployed.

---

## 5. Repository structure

```
cxr-classifier/
├── configs/
│   └── config.yaml                   # Main YAML configuration
├── notebooks/
│   └── exploration.ipynb             # Interactive walkthrough
├── scripts/
│   ├── train.py                      # Training entry point
│   ├── evaluate.py                   # Evaluation entry point
│   └── inference.py                  # Single-image inference
├── src/cxr_classifier/
│   ├── config.py                     # Configuration dataclasses
│   ├── data.py                       # Dataset & DataLoader pipeline
│   ├── models.py                     # Backbone construction & head replacement
│   ├── training.py                   # Training loop, AMP, checkpointing
│   └── evaluation.py                 # Metrics, plots, Grad-CAM
├── tests/
│   ├── conftest.py                   # Synthetic 12-image dataset fixture
│   ├── test_config.py                # YAML roundtrip, default Config
│   ├── test_data.py                  # Dataset loading, class weights, missing classes
│   ├── test_evaluation.py            # Metrics, confusion matrix, GradCAM shape
│   └── test_models.py                # Model creation, ensemble, freeze/unfreeze
├── outputs/                          # Checkpoints & evaluation artefacts (gitignored)
├── logs/                             # TensorBoard logs (gitignored)
├── Dockerfile                        # Production container
├── docker-compose.yml                # Multi-service orchestration
├── pyproject.toml                    # Project metadata & dependencies
├── requirements.txt                  # Pip dependencies
└── setup.sh                          # One-command local setup
```

---

## 6. Quick start

### 6.1 Local installation

```bash
git clone https://github.com/ariusxscourger/cxr-classifier.git
cd cxr-classifier

# Create a virtual environment (uv is fastest)
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,notebook]"
pre-commit install
```

### 6.2 Download the dataset

Download the [Chest X-Ray Dataset](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset) from Kaggle and extract it so the layout matches `configs/config.yaml`:

```
Chest X-Ray/
├── train/{normal,pneumonia,tuberculosis}/*.jpg
├── val/{normal,pneumonia,tuberculosis}/*.jpg
└── test/{normal,pneumonia,tuberculosis}/*.jpg
```

### 6.3 Train

```bash
python scripts/train.py --config configs/config.yaml --device cuda
tensorboard --logdir logs
```

For CPU-only training (e.g. on a laptop), use `--device cpu`. The pipeline automatically disables AMP and CUDA-specific paths on CPU.

### 6.4 Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint outputs/best_model.pth \
    --config configs/config.yaml \
    --output-dir outputs/evaluation
```

### 6.5 Inference

```bash
python scripts/inference.py \
    --checkpoint outputs/best_model.pth \
    --image path/to/cxray.jpg \
    --top-k 3
```

---

## 7. Docker

A production-ready container is provided (`Dockerfile`, `docker-compose.yml`). Services:

| Service | Purpose |
|---|---|
| `train` | GPU training (requires nvidia-docker) |
| `train-cpu` | CPU training fallback |
| `tensorboard` | Live training visualisation at `:6006` |
| `jupyter` | Interactive development at `:8888` |
| `evaluate` | Run evaluation in-container |
| `inference` | Run inference in-container |

```bash
docker compose build
docker compose up train-cpu tensorboard   # CPU
# or
docker compose up train tensorboard       # GPU (requires nvidia-container-toolkit)
```

---

## 8. Configuration

All hyperparameters live in `configs/config.yaml`. Key sections:

```yaml
model:
  name: "convnext_tiny"      # Backbone (see Section 4.2)
  pretrained: true           # ImageNet weights
  num_classes: 3
  drop_path_rate: 0.1        # Stochastic depth for regularisation
  drop_rate: 0.2            # Dropout in head

training:
  epochs: 100
  optimizer: { name: "adamw", lr: 1e-4, weight_decay: 0.05 }
  scheduler: { name: "cosine_annealing_warm_restarts", t_0: 10, t_mult: 2 }
  loss: { name: "label_smoothing_cross_entropy", label_smoothing: 0.1 }
  gradient_clip: 1.0
  mixed_precision: true
  early_stopping: { patience: 15, mode: "max" }

augmentation:
  train: [...Albumentations pipeline...]
  val:   [CLAHE → Resize 256 → CenterCrop 224 → Normalize]
  test:  [CLAHE → Resize 256 → CenterCrop 224 → Normalize]
```

---

## 9. Testing

The test suite (`tests/`) uses a synthetic 12-image fixture (built per-session with deterministic NumPy seeding) so CI is hermetic — no 5 GB dataset download required.

```bash
PYTHONPATH=src pytest tests/ -v
```

The tests cover:
- `Config` YAML round-trip and default construction
- `ChestXRayDataset` loading, class distribution, weight computation, missing-class behaviour
- `Evaluator.compute_metrics` correctness on synthetic data (perfect classifier, partial classifier, normalised confusion matrix)
- `GradCAM` shape, range, and forward/backward pass on a random input
- `create_model`, `create_ensemble`, `freeze_backbone`

---

## 10. Software-engineering standards

The codebase is deliberately structured for readability and reproducibility:

- **Type hints** throughout (`from __future__ import annotations` in tests).
- **Dataclass configuration** with `__post_init__` type coercion and `OmegaConf.structured` validation.
- **AMP-correct gradient clipping** — `scaler.unscale_` only called when the scaler is enabled.
- **Top-k checkpoint pruning** by validation F1-macro.
- **Non-root Docker user**, multi-stage compose, offline-friendly defaults (`HF_HUB_OFFLINE=1`, `WANDB_MODE=offline`).
- **Pre-commit hooks**: ruff (lint + format), black, isort, mypy.
- **CI**: GitHub Actions matrix over Python 3.10 / 3.11 / 3.12 with lint + type-check + test.

---

## 11. Methodological honesty — what this is *not*

This pipeline is appropriate for **methodological study and educational purposes**. It is **not** a deployed medical device:

1. **Patient-disjoint splits are not guaranteed** by the publisher (Section 3.2). For clinical claims, re-partition the data at the patient level.
2. **External validation is absent.** Generalisation across hospitals, X-ray machines, and patient populations requires an external test set acquired on a different scanner / institution.
3. **No prospective evaluation.** Even after external validation, a prospective study is required before any clinical use.
4. **No demographic analysis.** Subgroup performance by age, sex, and comorbidity is necessary before deployment.
5. **No calibration assessment.** Reliability diagrams and expected calibration error (ECE) should be reported before any probability-based clinical decision-support tool is deployed (Guo et al., 2017).

These limitations are universal in the chest-radiograph deep-learning literature and should be acknowledged in any publication that uses this codebase.

---

## 12. References

1. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). *Rethinking the inception architecture for computer vision.* CVPR.
3. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal loss for dense object detection.* ICCV.
4. Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking model scaling for convolutional neural networks.* ICML.
5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* CVPR.
6. Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). *An image is worth 16×16 words: Transformers for image recognition at scale.* ICLR.
7. Liu, Z., Lin, Y., Cao, Y., et al. (2021). *Swin Transformer: Hierarchical vision transformer using shifted windows.* ICCV.
8. Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). *A ConvNet for the 2020s.* CVPR.
9. Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A. (2020). *Albumentations: Fast and flexible image augmentations.* Information, 11(2), 125.
10. DeVries, T., & Taylor, G. W. (2017). *Improved regularization of convolutional neural networks with cutout.* arXiv:1708.04552.
11. Loshchilov, I., & Hutter, F. (2017). *SGDR: Stochastic gradient descent with warm restarts.* ICLR.
12. Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization.* ICLR.
13. Howard, J., & Ruder, S. (2018). *Universal language model fine-tuning for text classification.* ACL.
14. Müller, R., Kornblith, S., & Hinton, G. (2019). *When does label smoothing help?* NeurIPS.
15. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* ICCV.
16. Saito, T., & Rehmsmeier, M. (2015). *The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets.* PLOS ONE, 10(3), e0118432.
17. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On calibration of modern neural networks.* ICML.
18. Raghu, M., Zhang, C., Kleinberg, J., & Bengio, S. (2019). *Transfusion: Understanding transfer learning for medical imaging.* NeurIPS.
19. Oakden-Rayner, L., Dunnmon, J., Carneiro, G., & Ré, C. (2020). *Hidden stratification causes clinically meaningful failures in machine learning for medical imaging.* CHIL.
20. Wightman, R. (2019). *PyTorch image models (timm).* GitHub.
21. Paszke, A., et al. (2019). *PyTorch: An imperative style, high-performance deep learning library.* NeurIPS.

---

## 13. Citation

```bibtex
@software{cxr_classifier_2025,
  author = {Muhammad Saqib},
  title  = {Chest X-Ray Multi-Class Classification Pipeline},
  year   = {2025},
  url    = {https://github.com/ariusxscourger/cxr-classifier}
}
```

---

## 14. Author

**Muhammad Saqib** — BS Software Engineering (SZABIST, CGPA 3.13); JPMorgan FinTech Backend (2023–2025); Solutec Power BI / ML Engineer (2025–present). Targeting fully-funded Master's in AI/ML (Fall 2027). Research interests: medical-imaging deep learning, Urdu NLP, MLOps.

---

## 15. License

MIT — see [LICENSE](LICENSE).

---

*Built for medical-AI research and education.*