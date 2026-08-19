#!/usr/bin/env bash
# Setup script for Chest X-Ray CSC Project

set -e

echo "🚀 Setting up Chest X-Ray Classification Project for CSC Scholarship"
echo "=================================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "✅ uv found: $(uv --version)"
    USE_UV=true
else
    echo "⚠️  uv not found, using pip"
    USE_UV=false
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    if [ "$USE_UV" = true ]; then
        uv venv --python 3.11
    else
        python3 -m venv .venv
    fi
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
if [ "$USE_UV" = true ]; then
    uv pip install --upgrade pip
else
    pip install --upgrade pip
fi

# Install package in development mode
echo "📥 Installing package dependencies..."
if [ "$USE_UV" = true ]; then
    uv pip install -e ".[dev,notebook]"
else
    pip install -e ".[dev,notebook]"
fi

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
if [ "$USE_UV" = true ]; then
    uv pip install pre-commit
else
    pip install pre-commit
fi
pre-commit install

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs outputs models data

# Check dataset
DATASET_PATH="/run/media/saqib/067E00563129A4C4/Chest X-Ray"
if [ -d "$DATASET_PATH" ]; then
    echo "✅ Dataset found at: $DATASET_PATH"
    echo "   Train: $(find "$DATASET_PATH/train" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l) images"
    echo "   Val:   $(find "$DATASET_PATH/val" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l) images"
    echo "   Test:  $(find "$DATASET_PATH/test" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l) images"
else
    echo "⚠️  Dataset not found at: $DATASET_PATH"
    echo "   Please ensure the dataset is mounted at this location"
fi

# Check device
echo "🖥️  Checking compute device..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'   CUDA: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')
elif torch.backends.mps.is_available():
    print('   MPS: Available (Apple Silicon)')
else:
    print('   CPU: Will use CPU training (slower but works)')
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Run training: python scripts/train.py --config configs/config.yaml"
echo "  3. Monitor: tensorboard --logdir logs"
echo "  4. Or open notebook: jupyter lab notebooks/exploration.ipynb"
echo ""
echo "For CSC application, see README.md for what professors expect!"