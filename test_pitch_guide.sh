#!/bin/bash
# Test script for pitch guide analyzer
# Analyzes a guide video to extract pitch sequence

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 <guide_video.mp4> [options]"
    echo ""
    echo "Options:"
    echo "  --output PATH        Output JSON path (default: data/segments/guide_sequence.json)"
    echo "  --fps N              Video frame rate (default: 24)"
    echo "  --threshold N        Pitch change threshold in cents (default: 50)"
    echo "                       Splits when pitch changes by N cents OR at silence"
    echo "                       Each note holds until next pitch change or silence"
    echo "                       Lower = more segments, Higher = smoother playback"
    echo "  --min-duration N     Minimum segment duration (default: 0.1s)"
    echo "                       Filters very short notes/syllables"
    echo "  --silence-threshold N  Silence detection threshold in dB (default: -50)"
    echo "                       Lower = more permissive (detects quieter sounds)"
    echo "                       Higher = more strict (treats quiet sounds as silence)"
    echo "                       Try -60 if too much silence, -45 if too little"
    echo "  --pitch-smoothing N  Median filter window size (default: 0=off)"
    echo "                       Reduces vibrato/pitch waver. Try 5-7 for wavy vocals"
    echo "                       Higher values = more smoothing, but may miss quick notes"
    echo "  --pitch-method METHOD  Pitch detection algorithm (default: crepe)"
    echo "                       Options: crepe (accurate), basic-pitch (multipitch),"
    echo "                                swift-f0 (fast CPU), hybrid (crepe+swift-f0),"
    echo "                                pyin (fallback)"
    echo "  --min-rest-duration N  Minimum gap to create rest segment (default: 0.1s)"
    echo "  --no-rest-segments   Disable rest detection (pitched segments only)"
    echo "  --no-verify-rest-rms Skip RMS verification for rests (faster)"
    echo "  --device DEVICE      Device for neural networks: auto, cuda, cpu (default: auto)"
    echo "                       Use --device cuda to force GPU acceleration"
    echo "  --use-pyin           (Deprecated) Use pYIN - prefer --pitch-method pyin"
    echo ""
    echo "Examples:"
    echo "  $0 data/input/singing_guide.mp4"
    echo "  $0 data/input/singing_guide.mp4 --output data/segments/my_guide.json"
    echo "  $0 data/input/singing_guide.mp4 --min-duration 0.15  # Filter short notes"
    echo "  $0 data/input/singing_guide.mp4 --pitch-smoothing 5  # Smooth out vibrato"
    echo "  $0 data/input/singing_guide.mp4 --pitch-method hybrid  # Best of both CREPE+SwiftF0"
    echo "  $0 data/input/singing_guide.mp4 --pitch-method swift-f0  # Use fast SwiftF0"
    echo "  $0 data/input/singing_guide.mp4 --pitch-method basic-pitch  # Use Spotify's Basic Pitch"
    echo "  $0 data/input/singing_guide.mp4 --silence-threshold -60  # More permissive silence detection"
    echo "  $0 data/input/singing_guide.mp4 --pitch-method rmvpe --device cuda  # Use GPU"
    exit 1
fi

GUIDE_VIDEO="$1"
shift  # Remove first argument, rest are options

# Check if file exists
if [ ! -f "$GUIDE_VIDEO" ]; then
    echo -e "${YELLOW}Error: Guide video not found: $GUIDE_VIDEO${NC}"
    exit 1
fi

# Setup paths
OUTPUT_DIR="data/segments"
TEMP_DIR="data/temp"
OUTPUT_JSON="$OUTPUT_DIR/guide_sequence.json"

# Parse --output from extra args if provided, and build filtered args
ARGS=("$@")
FILTERED_ARGS=()
i=0
while [ $i -lt ${#ARGS[@]} ]; do
    if [[ "${ARGS[$i]}" == "--output" ]] && [ $((i + 1)) -lt ${#ARGS[@]} ]; then
        OUTPUT_JSON="${ARGS[$((i + 1))]}"
        i=$((i + 2))  # Skip --output and its value
    else
        FILTERED_ARGS+=("${ARGS[$i]}")
        i=$((i + 1))
    fi
done

# Create directories
mkdir -p "$(dirname "$OUTPUT_JSON")"
mkdir -p "$TEMP_DIR"

# Detect Python command
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo -e "${YELLOW}Error: Python not found${NC}"
    exit 1
fi

$PYTHON_CMD src/pitch_guide_analyzer.py \
    --video "$GUIDE_VIDEO" \
    --output "$OUTPUT_JSON" \
    --temp-dir "$TEMP_DIR" \
    "${FILTERED_ARGS[@]}"

# Create MIDI preview video
PREVIEW_DIR=$(dirname "$OUTPUT_JSON")
PREVIEW_VIDEO="$PREVIEW_DIR/guide_midi_preview.mp4"
mkdir -p "$PREVIEW_DIR"

echo ""
if [ -f "$OUTPUT_JSON" ]; then
    echo "Creating MIDI preview video..."
    $PYTHON_CMD src/pitch_video_preview.py \
        --video "$GUIDE_VIDEO" \
        --pitch-json "$OUTPUT_JSON" \
        --output "$PREVIEW_VIDEO"
else
    echo -e "${YELLOW}Warning: Output JSON not found, skipping preview video${NC}"
fi

echo ""
echo "Next steps:"
echo "  1. Watch $PREVIEW_VIDEO to verify pitch detection"
echo "     (MIDI tones should match the singing!)"
echo "  2. Adjust --threshold if too many/few notes detected"
echo "  3. Run source video analysis with same settings"
echo ""
