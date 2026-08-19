# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Build args
ARG UID=1000
ARG GID=1000

# Create non-root user
RUN groupadd -g ${GID} appuser && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser

WORKDIR /app

# System dependencies for OpenCV, matplotlib, and visualization
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Install package in production mode
RUN pip install --no-cache-dir -e .

# Create directories for outputs and data
RUN mkdir -p /app/logs /app/outputs /app/data /app/dataset && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables for offline operation
ENV HF_HUB_OFFLINE=1
ENV WANDB_MODE=offline
ENV PYTHONUNBUFFERED=1

# Default command
ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--config", "configs/config.yaml", "--device", "cpu"]
