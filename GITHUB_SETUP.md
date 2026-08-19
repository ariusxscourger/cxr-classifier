# Chest X-Ray Classification - CI/CD Pipeline

## GitHub Actions Workflows

### 1. CI Pipeline (`.github/workflows/ci.yml`)

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run Ruff (linting)
        run: ruff check src/ scripts/

      - name: Run Black (formatting check)
        run: black --check src/ scripts/

      - name: Run isort (import sorting check)
        run: isort --check-only src/ scripts/

      - name: Run MyPy (type checking)
        run: mypy src/

      - name: Run Tests
        run: pytest tests/ -v --cov=src/chestxray --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
```

### 2. Release Pipeline (`.github/workflows/release.yml`)

```yaml
name: Release

on:
  push:
    tags:
      - "v*"

jobs:
  build-and-release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip build twine

      - name: Build package
        run: python -m build

      - name: Check package
        run: twine check dist/*

      - name: Publish to PyPI
        if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

### 3. Docker Build (`.github/workflows/docker.yml`)

```yaml
name: Docker

on:
  push:
    branches: [main]
    tags: ["v*"]
  pull_request:
    branches: [main]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GHCR
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## Repository Setup Checklist

### Required GitHub Settings

1. **Branch Protection** (Settings → Branches → Add rule):
   - Branch: `main`
   - Require PR reviews: ✅
   - Require status checks: ✅ (lint-and-test)
   - Require branches up to date: ✅
   - Include administrators: ✅

2. **Secrets** (Settings → Secrets → Actions):
   - `PYPI_API_TOKEN` - For PyPI publishing
   - `CODECOV_TOKEN` - For coverage reporting (optional)

3. **Environments** (Settings → Environments):
   - Create `release` environment with protection rules

### Badges for README

Add to top of README.md:

```markdown
[![CI](https://github.com/YOUR_USERNAME/ChestXRay-CSC-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/ChestXRay-CSC-Project/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/chestxray-csc-project.svg)](https://pypi.org/project/chestxray-csc-project/)
[![Python](https://img.shields.io/pypi/pyversions/chestxray-csc-project.svg)](https://pypi.org/project/chestxray-csc-project/)
[![License](https://img.shields.io/github/license/YOUR_USERNAME/ChestXRay-CSC-Project.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/YOUR_USERNAME/ChestXRay-CSC-Project/pkgs/container/chestxray-csc-project)
```

---

## Docker Support

### Dockerfile (for inclusion in repo)

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install package
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Default command
ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--config", "configs/config.yaml"]
```

### docker-compose.yml (for local development)

```yaml
version: "3.8"

services:
  train:
    build: .
    volumes:
      - ./configs:/app/configs
      - ./outputs:/app/outputs
      - ./logs:/app/logs
      - /path/to/your/dataset:/data:ro
    environment:
      - HF_HUB_OFFLINE=1
      - WANDB_MODE=offline
    command: python scripts/train.py --config configs/config.yaml --device cpu

  tensorboard:
    image: tensorflow/tensorflow:latest-py3
    ports:
      - "6006:6006"
    volumes:
      - ./logs:/logs
    command: tensorboard --logdir /logs --host 0.0.0.0
```

---

## Issue Templates

### Bug Report (`.github/ISSUE_TEMPLATE/bug_report.yml`)

```yaml
name: Bug Report
description: Report a bug
title: "[BUG]: "
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: "Thanks for reporting a bug! Please fill out the details below."
  - type: textarea
    id: description
    attributes:
      label: Description
      description: Clear description of the bug
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Steps to Reproduce
      description: Minimal steps to reproduce the issue
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
    validations:
      required: true
  - type: textarea
    id: actual
    attributes:
      label: Actual Behavior
    validations:
      required: true
  - type: input
    id: environment
    attributes:
      label: Environment
      description: OS, Python version, GPU (if any)
    validations:
      required: true
```

### Feature Request (`.github/ISSUE_TEMPLATE/feature_request.yml`)

```yaml
name: Feature Request
description: Suggest a new feature
title: "[FEAT]: "
labels: ["enhancement"]
body:
  - type: markdown
    attributes:
      value: "Thanks for suggesting a feature! Please describe your idea."
  - type: textarea
    id: problem
    attributes:
      label: Problem Statement
      description: What problem does this solve?
    validations:
      required: true
  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: How would you like to solve it?
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
      description: Other approaches you've considered
  - type: dropdown
    id: priority
    attributes:
      label: Priority
      options: [Low, Medium, High, Critical]
```

---

## Security Policy (SECURITY.md)

```markdown
# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a Vulnerability

Please report security vulnerabilities privately via:
- GitHub Security Advisories (private)
- Email: security@yourdomain.com

Do NOT open public issues for security vulnerabilities.

## Response Timeline

- Acknowledgment: Within 48 hours
- Initial assessment: Within 1 week
- Fix timeline: Depends on severity
</markdown>
```

---

## GitHub Repository Description

For the repo settings:

```
Chest X-Ray Multi-Class Classification (Normal, Pneumonia, Tuberculosis). Production-ready PyTorch pipeline with ConvNeXt/EfficientNet/ViT/Swin, Albumentations, TensorBoard, and comprehensive evaluation.
```

### Topics/Tags

```
deep-learning, pytorch, computer-vision, medical-imaging, chest-xray, classification, cnn, vision-transformer, convnext, efficientnet, vit, swin, scholarship, csc, medical-ai
```

---

## First Release Checklist

- [ ] All tests passing
- [ ] README complete with badges
- [ ] CONTRIBUTING.md added
- [ ] LICENSE (MIT) added
- [ ] SECURITY.md added
- [ ] GitHub Actions CI configured
- [ ] Issue templates added
- [ ] Branch protection enabled
- [ ] Docker image builds
- [ ] PyPI publishing configured
- [ ] Tag v0.1.0 and create release
- [ ] Update README with installation from PyPI