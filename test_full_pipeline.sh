#!/bin/bash
# Complete pipeline test for short audio/video
# Usage: ./test_full_pipeline.sh <audio_file> <video_file>

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./test_full_pipeline.sh <audio_file> <video_file>"
    echo ""
    echo "Example:"
    echo "  ./test_full_pipeline.sh data/input/test_song.mp3 data/input/test_video.mp4"
    exit 1
fi

AUDIO_FILE="$1"
VIDEO_FILE="$2"
AUDIO_BASENAME=$(basename "$AUDIO_FILE" | sed 's/\.[^.]*$//')

echo "============================================"
echo "Full Pipeline Test"
echo "============================================"
echo "Audio: $AUDIO_FILE"
echo "Video: $VIDEO_FILE"
echo ""

# Clean previous test outputs
echo "Cleaning previous test data..."
rm -rf data/segments
mkdir -p data/segments/audio data/segments/video

# Step 1: Separate audio (if not already done)
echo ""
echo "[1/5] Checking for separated audio stems..."
SEPARATED_DIR="data/separated/htdemucs/$AUDIO_BASENAME"

if [ ! -d "$SEPARATED_DIR" ]; then
    echo "  Separating audio with Demucs..."
    demucs -n htdemucs "$AUDIO_FILE" -o data/separated
else
    echo "  ✓ Using existing stems from: $SEPARATED_DIR"
fi

# Detect Python command
if command -v python &> /dev/null && python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "Error: Python 3.8+ not found"
    exit 1
fi

# Step 2: Analyze onset strength
echo ""
echo "[2/5] Analyzing onset strength..."
$PYTHON_CMD src/onset_strength_analysis.py \
    --audio "$SEPARATED_DIR/other.wav" \
    --output data/output/onset_strength.json \
    --fps 24 \
    --power 0.6 \
    --window-size 1 \
    --tolerance 0.3 \
    --visualize \
    --viz-output data/output/onset_strength.png \
    --threshold 0.2

# Generate interactive visualizer
$PYTHON_CMD src/interactive_strength_visualizer.py \
    --audio "$SEPARATED_DIR/other.wav" \
    --strength data/output/onset_strength.json \
    --output data/output/onset_strength_visualizer.html \
    --threshold 0.2

# Step 3: Segment audio
echo ""
echo "[3/5] Segmenting audio..."
$PYTHON_CMD src/audio_segmenter.py \
    --audio "$SEPARATED_DIR/other.wav" \
    --onset-strength data/output/onset_strength.json \
    --output-dir data/segments/audio \
    --metadata-output data/segments/audio_segments.json \
    --threshold 0.2 \
    --prefix audio_seg

# Step 4: Extract ImageBind audio embeddings
echo ""
echo "[4/5] Extracting ImageBind audio embeddings..."
echo "This may take a few minutes depending on number of segments..."

# Check if ImageBind is installed
if ! $PYTHON_CMD -c "from imagebind.models import imagebind_model" 2>/dev/null; then
    echo "  ImageBind not installed. Installing..."
    ./install_imagebind.sh
fi

$PYTHON_CMD src/imagebind_audio_embedder.py \
    --segments-metadata data/segments/audio_segments.json \
    --segments-dir data/segments/audio \
    --output data/segments/audio_embeddings.json \
    --batch-size 8 \
    --device auto

echo ""
echo "✓ Phase 2 complete: Audio embeddings extracted"

# Step 5: Extract ImageBind video embeddings
echo ""
echo "[5/5] Extracting ImageBind video embeddings..."
echo "This uses a sliding window over the video frames..."

$PYTHON_CMD src/imagebind_video_embedder.py \
    --video "$VIDEO_FILE" \
    --output data/segments/video_embeddings.json \
    --fps 24 \
    --window-size 5 \
    --stride 6 \
    --batch-size 4 \
    --chunk-size 300 \
    --device auto

echo ""
echo "============================================"
echo "✓ Phase 3 Complete: Video Embeddings"
echo "============================================"
echo ""
echo "Output files:"
echo "  - Onset strength: data/output/onset_strength.json"
echo "  - Visualizer: data/output/onset_strength_visualizer.html"
echo "  - Audio segments: data/segments/audio/"
echo "  - Audio segment metadata: data/segments/audio_segments.json"
echo "  - Audio embeddings: data/segments/audio_embeddings.json"
echo "  - Video embeddings: data/segments/video_embeddings.json"
echo ""
echo "Next steps:"
echo "  1. Verify embeddings:"
echo "     cat data/segments/audio_embeddings.json | head -20"
echo "     cat data/segments/video_embeddings.json | head -20"
echo "  2. When ready, proceed to Phase 4 (Semantic matching)"
echo ""
