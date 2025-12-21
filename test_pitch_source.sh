#!/bin/bash
# Test script for pitch source database builder
# Analyzes a source video to build comprehensive pitch database

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Pitch Source Database Builder ===${NC}\n"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <source_video.mp4> [options]"
    echo ""
    echo "Options:"
    echo "  --fps N              Video frame rate (default: 24)"
    echo "  --threshold N        Onset threshold (default: 0.12, lower = more clips)"
    echo "  --min-confidence N   Min pitch confidence (default: 0.5)"
    echo "  --use-pyin           Use pYIN instead of CREPE"
    echo "  --max-clips N        Limit number of clips (for testing)"
    echo ""
    echo "Example:"
    echo "  $0 data/input/singing_source.mp4"
    echo "  $0 data/input/singing_source.mp4 --threshold 0.1 --fps 30"
    echo "  $0 data/input/singing_source.mp4 --max-clips 500  # Test mode"
    exit 1
fi

SOURCE_VIDEO="$1"
shift  # Remove first argument, rest are options

# Check if file exists
if [ ! -f "$SOURCE_VIDEO" ]; then
    echo -e "${YELLOW}Error: Source video not found: $SOURCE_VIDEO${NC}"
    exit 1
fi

# Setup paths
OUTPUT_DIR="data/segments"
TEMP_DIR="data/temp"
OUTPUT_JSON="$OUTPUT_DIR/source_database.json"

# Create directories
mkdir -p "$OUTPUT_DIR"
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

echo -e "${GREEN}Step 1: Building pitch database from source video${NC}"
echo "Input: $SOURCE_VIDEO"
echo "Output: $OUTPUT_JSON"
echo ""
echo "Note: This may take a while for long videos..."
echo "      Use --max-clips 500 for quick testing"
echo ""

$PYTHON_CMD src/pitch_source_analyzer.py \
    --video "$SOURCE_VIDEO" \
    --output "$OUTPUT_JSON" \
    --temp-dir "$TEMP_DIR" \
    "$@"

echo ""
echo -e "${GREEN}Step 2: Creating MIDI preview video (optional)${NC}"
echo ""

if [ -f "$OUTPUT_JSON" ]; then
    read -p "Generate preview video? This helps verify pitch detection. (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        PREVIEW_VIDEO="$OUTPUT_DIR/source_midi_preview.mp4"
        $PYTHON_CMD src/pitch_video_preview.py \
            --video "$SOURCE_VIDEO" \
            --pitch-json "$OUTPUT_JSON" \
            --output "$PREVIEW_VIDEO"
        echo "Preview video saved: $PREVIEW_VIDEO"
    else
        echo "Skipping preview video generation."
    fi
fi

echo ""
echo -e "${GREEN}=== Database Building Complete ===${NC}"
echo ""
echo "Results saved to:"
echo "  - Pitch database: $OUTPUT_JSON"
if [ -f "$OUTPUT_DIR/source_midi_preview.mp4" ]; then
    echo "  - Preview video: $OUTPUT_DIR/source_midi_preview.mp4"
fi
echo ""
echo "Database info:"
if [ -f "$OUTPUT_JSON" ]; then
    # Extract key stats from JSON
    NUM_CLIPS=$(grep -o '"num_clips": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*')
    NUM_PITCHES=$(grep -o '"num_unique_pitches": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*')

    echo "  - Total clips: $NUM_CLIPS"
    echo "  - Unique pitches: $NUM_PITCHES"
fi
echo ""
echo "Next steps:"
echo "  1. Review database statistics above"
echo "  2. Ensure you have clips for needed pitch range"
echo "  3. Run pitch matcher to match guide sequence to source clips"
echo "  4. If missing pitches, adjust --threshold to capture more clips"
echo ""
