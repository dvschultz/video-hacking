#!/bin/bash
# Install ImageBind from GitHub

echo "Installing ImageBind..."
echo ""

# Detect Python command (prefer python over python3 if in conda env)
if command -v python &> /dev/null && python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
    PYTHON_CMD=python
    PIP_CMD=pip
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PIP_CMD=pip3
else
    echo "Error: Python 3.8+ not found"
    exit 1
fi

echo "Using: $PYTHON_CMD ($(command -v $PYTHON_CMD))"

# Check if already installed
if $PYTHON_CMD -c "from imagebind.models import imagebind_model" 2>/dev/null; then
    echo "✓ ImageBind already installed"
    exit 0
fi

# Install from GitHub
echo "Installing from GitHub repository..."
$PIP_CMD install --force-reinstall --no-cache-dir git+https://github.com/facebookresearch/ImageBind.git

# Verify installation
echo ""
echo "Verifying installation..."
if $PYTHON_CMD -c "from imagebind.models import imagebind_model" 2>&1; then
    echo ""
    echo "✓ ImageBind successfully installed!"
else
    echo ""
    echo "✗ ImageBind import failed (see error above)"
    echo ""
    echo "Troubleshooting:"
    echo "1. Make sure you're in your conda environment: conda activate vh"
    echo "2. Try manually: $PIP_CMD install git+https://github.com/facebookresearch/ImageBind.git"
    echo "3. Verify PyTorch is installed: $PYTHON_CMD -c 'import torch; print(torch.__version__)'"
    exit 1
fi
