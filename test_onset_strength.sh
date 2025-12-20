#!/bin/bash
# Test script for onset strength analysis
# Usage: ./test_onset_strength.sh [fps] [power] [window_size] [tolerance]

FPS=${1:-24}
POWER=${2:-0.6}
WINDOW_SIZE=${3:-1}
TOLERANCE=${4:-0.3}

# Detect Python command
if command -v python &> /dev/null && python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "Error: Python 3.8+ not found"
    exit 1
fi

echo "Testing onset strength analysis..."
echo "Parameters: fps=$FPS, power=$POWER, window_size=$WINDOW_SIZE, tolerance=$TOLERANCE"

# Find the most recent separated stems
SEPARATED_DIR=$(find data/separated/htdemucs -mindepth 1 -maxdepth 1 -type d | head -n 1)

if [ -z "$SEPARATED_DIR" ]; then
    echo "Error: No separated stems found. Run ./test_audio_analysis.sh first"
    exit 1
fi

echo "Using stems from: $SEPARATED_DIR"
echo ""

# Run onset strength analysis
$PYTHON_CMD src/onset_strength_analysis.py \
    --audio "$SEPARATED_DIR/other.wav" \
    --output data/output/onset_strength.json \
    --fps "$FPS" \
    --power "$POWER" \
    --window-size "$WINDOW_SIZE" \
    --tolerance "$TOLERANCE" \
    --visualize \
    --viz-output data/output/onset_strength.png \
    --threshold 0.2

echo ""
echo "Generating interactive visualizer..."
$PYTHON_CMD src/interactive_strength_visualizer.py \
    --audio "$SEPARATED_DIR/other.wav" \
    --strength data/output/onset_strength.json \
    --output data/output/onset_strength_visualizer.html \
    --threshold 0.2

echo ""
echo "============================================"
echo "✓ Onset Strength Analysis Complete!"
echo "============================================"
echo ""
echo "Output files:"
echo "  - Onset strength data: data/output/onset_strength.json"
echo "  - Static visualization: data/output/onset_strength.png"
echo "  - Interactive visualizer: data/output/onset_strength_visualizer.html"
echo ""
echo "Next steps:"
echo "  1. Open data/output/onset_strength_visualizer.html in your browser"
echo "  2. Adjust the threshold slider to see which frames trigger cuts"
echo "  3. Play audio and watch the onset strength curve in real-time"
echo "  4. Tune parameters by re-running:"
echo "     ./test_onset_strength.sh [fps] [power] [window_size] [tolerance]"
echo ""
echo "Parameter guide:"
echo "  fps: 24 (default), 30, 60 - frames per second"
echo "  power: 0.5-1.0 (default 0.6) - lower = more compression"
echo "  window_size: 0-5 (default 1) - smoothing window"
echo "  tolerance: 0.0-1.0 (default 0.3) - noise removal threshold"
echo ""
