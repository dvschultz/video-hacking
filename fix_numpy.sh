#!/bin/bash
# Fix NumPy version conflicts

echo "Fixing NumPy version..."
echo ""

# Detect pip command
if command -v pip &> /dev/null; then
    PIP_CMD=pip
elif command -v pip3 &> /dev/null; then
    PIP_CMD=pip3
else
    echo "Error: pip not found"
    exit 1
fi

echo "Current versions:"
$PIP_CMD list | grep -E "numpy|opencv"

echo ""
echo "Downgrading NumPy to 1.26.4..."
$PIP_CMD uninstall -y numpy
$PIP_CMD install "numpy<2.0"

echo ""
echo "Reinstalling OpenCV..."
$PIP_CMD uninstall -y opencv-python
$PIP_CMD install opencv-python==4.8.1.78

echo ""
echo "✓ Fixed! Current versions:"
$PIP_CMD list | grep -E "numpy|opencv"
