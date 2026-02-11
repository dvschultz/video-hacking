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
    echo "  --output PATH        Output JSON path (default: data/segments/source_database.json)"
    echo "  --append             Append to existing database instead of overwriting"
    echo "                       Use this to combine multiple source videos into one database"
    echo "  --fps N              Video frame rate (default: auto-detect from video)"
    echo "  --threshold N        Pitch change threshold in cents (default: 50)"
    echo "                       Lower = more segments, Higher = fewer segments"
    echo "  --min-duration N     Minimum segment duration (default: 0.1s)"
    echo "  --silence-threshold N  Silence detection threshold in dB (default: -50)"
    echo "                       Lower = more permissive, detects quieter sounds"
    echo "  --pitch-smoothing N  Median filter window size (default: 0=off)"
    echo "                       Try 5-7 to reduce vibrato/waver"
    echo "  --pitch-method METHOD  Pitch detection algorithm (default: crepe)"
    echo "                       Options: crepe, swift-f0, basic-pitch, hybrid, pyin"
    echo "  --normalize          Normalize audio loudness before analysis (recommended)"
    echo "  --target-lufs N      Target loudness in LUFS when normalizing (default: -16)"
    echo "  --no-rms-dip-split   Disable splitting segments on RMS dips (volume drops)"
    echo "  --rms-dip-threshold N  RMS dip threshold in dB below local mean (default: -6)"
    echo "  --min-dip-duration N   Minimum dip duration in seconds (default: 0.02)"
    echo "  --rms-window-ms N    Rolling window size in ms for dip detection (default: 50)"
    echo "  --use-pyin           (Deprecated) Use pYIN - prefer --pitch-method pyin"
    echo ""
    echo "Examples:"
    echo "  # Single video with normalization:"
    echo "  $0 data/input/singing_source.mp4 --normalize"
    echo ""
    echo "  # Multiple videos combined into one database:"
    echo "  $0 video1.mp4                    # Create new database"
    echo "  $0 video2.mp4 --append           # Add to database"
    echo "  $0 video3.mp4 --append           # Add more clips"
    echo ""
    echo "  # Other examples:"
    echo "  $0 data/input/singing_source.mp4 --output data/segments/source1.json"
    echo "  $0 data/input/singing_source.mp4 --threshold 75  # Fewer segments"
    echo "  $0 data/input/singing_source.mp4 --pitch-method hybrid  # Best accuracy"
    echo "  $0 data/input/singing_source.mp4 --pitch-smoothing 5  # Reduce waver"
    exit 1
fi

SOURCE_VIDEO="$1"
shift  # Remove first argument, rest are options

# Check if file exists
if [ ! -f "$SOURCE_VIDEO" ]; then
    echo -e "${YELLOW}Error: Source video not found: $SOURCE_VIDEO${NC}"
    exit 1
fi

# Parse --output from extra args if provided, and build filtered args
TEMP_DIR="data/temp"
OUTPUT_JSON="data/segments/source_database.json"
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

echo -e "${GREEN}Building pitch database from source video${NC}"
echo "Input: $SOURCE_VIDEO"
echo "Output: $OUTPUT_JSON"
echo ""

$PYTHON_CMD src/pitch_source_analyzer.py \
    --video "$SOURCE_VIDEO" \
    --output "$OUTPUT_JSON" \
    --temp-dir "$TEMP_DIR" \
    "${FILTERED_ARGS[@]}"

echo ""
echo -e "${GREEN}=== Database Building Complete ===${NC}"
echo ""
echo "Results saved to:"
echo "  - Pitch database: $OUTPUT_JSON"
echo ""
echo "Database info:"
if [ -f "$OUTPUT_JSON" ]; then
    # Extract key stats from JSON
    NUM_SEGMENTS=$(grep -o '"num_segments": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*')
    NUM_PITCHES=$(grep -o '"num_unique_pitches": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*')
    NUM_SILENCES=$(grep -o '"num_silence_gaps": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*')

    echo "  - Total segments: $NUM_SEGMENTS"
    echo "  - Unique pitches: $NUM_PITCHES"
    echo "  - Silent gaps: $NUM_SILENCES"
fi
echo ""
echo "Next steps:"
echo "  1. Review database statistics above"
echo "  2. Ensure you have segments for needed pitch range"
echo "  3. Run pitch matcher to match guide sequence to source segments"
echo ""
