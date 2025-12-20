#!/bin/bash
# Test semantic matching
# Usage: ./test_semantic_matching.sh [reuse_policy]

REUSE_POLICY=${1:-allow}

# Detect Python command
if command -v python &> /dev/null && python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "Error: Python 3.8+ not found"
    exit 1
fi

echo "============================================"
echo "Testing Semantic Matching"
echo "============================================"
echo "Reuse policy: $REUSE_POLICY"
echo ""

# Check if embeddings exist
if [ ! -f "data/segments/audio_embeddings.json" ]; then
    echo "Error: Audio embeddings not found"
    echo "Run ./test_full_pipeline.sh first"
    exit 1
fi

if [ ! -f "data/segments/video_embeddings.json" ]; then
    echo "Error: Video embeddings not found"
    echo "Run ./test_full_pipeline.sh first"
    exit 1
fi

# Run semantic matching
$PYTHON_CMD src/semantic_matcher.py \
    --audio-embeddings data/segments/audio_embeddings.json \
    --video-embeddings data/segments/video_embeddings.json \
    --audio-segments data/segments/audio_segments.json \
    --output data/segments/matches.json \
    --reuse-policy "$REUSE_POLICY" \
    --min-gap 5 \
    --max-reuse-count 3 \
    --max-reuse-percentage 0.3

echo ""
echo "============================================"
echo "✓ Semantic Matching Complete!"
echo "============================================"
echo ""
echo "Output:"
echo "  - Matches: data/segments/matches.json"
echo ""
echo "Try different reuse policies:"
echo "  ./test_semantic_matching.sh none       # No reuse"
echo "  ./test_semantic_matching.sh allow      # Unlimited reuse (default)"
echo "  ./test_semantic_matching.sh min_gap    # Minimum gap between reuses"
echo "  ./test_semantic_matching.sh limited    # Limited reuse count"
echo "  ./test_semantic_matching.sh percentage # Percentage-based reuse"
echo ""
