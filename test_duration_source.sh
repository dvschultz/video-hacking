#!/bin/bash
# Test script for duration source database builder
# Scans a folder of video clips and builds a duration database

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder_path> [options]"
    echo ""
    echo "Options:"
    echo "  --output PATH        Output JSON path (default: data/segments/duration_database.json)"
    echo "  --extensions EXTS    Video extensions (default: mp4,mov,avi,mkv,webm,m4v)"
    echo "  --recursive          Search subdirectories"
    echo "  --append             Append to existing database"
    echo "  --min-duration N     Skip clips shorter than N seconds (default: 0.1)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/clips"
    echo "  $0 /path/to/clips --output my_database.json"
    echo "  $0 /path/to/clips --extensions mp4,mov"
    echo "  $0 /path/to/more_clips --append"
    echo "  $0 /path/to/clips --recursive --min-duration 0.5"
    exit 1
fi

FOLDER_PATH="$1"
shift  # Remove first argument, rest are options

# Check if folder exists
if [ ! -d "$FOLDER_PATH" ]; then
    echo -e "${YELLOW}Error: Folder not found: $FOLDER_PATH${NC}"
    exit 1
fi

# Setup paths
OUTPUT_DIR="data/segments"
OUTPUT_JSON="$OUTPUT_DIR/duration_database.json"

# Parse --output argument if provided and build filtered args
ARGS=("$@")
FILTERED_ARGS=()
i=0
while [ $i -lt ${#ARGS[@]} ]; do
    if [[ "${ARGS[$i]}" == "--output" ]] && [ $((i + 1)) -lt ${#ARGS[@]} ]; then
        OUTPUT_JSON="${ARGS[$((i + 1))]}"
        OUTPUT_DIR=$(dirname "$OUTPUT_JSON")
        i=$((i + 2))  # Skip --output and its value
    else
        FILTERED_ARGS+=("${ARGS[$i]}")
        i=$((i + 1))
    fi
done

# Create directories
mkdir -p "$OUTPUT_DIR"

# Detect Python command
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo -e "${YELLOW}Error: Python not found${NC}"
    exit 1
fi

$PYTHON_CMD src/duration_source_analyzer.py \
    --folder "$FOLDER_PATH" \
    --output "$OUTPUT_JSON" \
    "${FILTERED_ARGS[@]}"

echo ""
echo "Next steps:"
echo "  1. Create a guide sequence (MIDI or pitch analysis)"
echo "  2. Run duration matcher to match guide to clips"
echo "  3. Assemble final video"
echo ""
