#!/bin/bash
# Test script for duration video assembler
# Assembles final video from duration match plan

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Duration Video Assembler ===${NC}\n"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <duration_match_plan.json> [options]"
    echo ""
    echo "Options:"
    echo "  --output PATH       Output video path (default: data/output/duration_matched_video.mp4)"
    echo "  --resolution WxH    Output resolution (e.g., 1920x1080)"
    echo "  --auto-resolution   Use smallest source resolution"
    echo "  --fps N             Output frame rate"
    echo "  --auto-fps          Use smallest source fps"
    echo "  --parallel N        Parallel workers (default: auto)"
    echo "  --no-audio          Output video only"
    echo "  --normalize-audio   Apply EBU R128 loudness normalization to each clip"
    echo "  --target-lufs N     Target loudness in LUFS for normalization (default: -16.0)"
    echo "  --no-cleanup        Keep temp files"
    echo "  --edl               Generate EDL file alongside video"
    echo "  --edl-only          Generate EDL file only (skip video assembly)"
    echo "  --edl-output PATH   Custom EDL output path"
    echo ""
    echo "Examples:"
    echo "  $0 data/segments/duration_match_plan.json"
    echo "  $0 data/segments/duration_match_plan.json --auto-resolution --auto-fps"
    echo "  $0 data/segments/duration_match_plan.json --resolution 1280x720 --fps 30"
    echo "  $0 data/segments/duration_match_plan.json --output custom_output.mp4"
    echo "  $0 data/segments/duration_match_plan.json --edl"
    echo "  $0 data/segments/duration_match_plan.json --edl-only --fps 24"
    exit 1
fi

MATCH_PLAN="$1"
shift  # Remove first argument, rest are options

# Check if file exists
if [ ! -f "$MATCH_PLAN" ]; then
    echo -e "${YELLOW}Error: Match plan not found: $MATCH_PLAN${NC}"
    exit 1
fi

# Parse --output from extra args if provided, and build filtered args
TEMP_DIR="data/temp"
OUTPUT_VIDEO="data/output/duration_matched_video.mp4"
ARGS=("$@")
FILTERED_ARGS=()
i=0
while [ $i -lt ${#ARGS[@]} ]; do
    if [[ "${ARGS[$i]}" == "--output" ]] && [ $((i + 1)) -lt ${#ARGS[@]} ]; then
        OUTPUT_VIDEO="${ARGS[$((i + 1))]}"
        i=$((i + 2))  # Skip --output and its value
    else
        FILTERED_ARGS+=("${ARGS[$i]}")
        i=$((i + 1))
    fi
done

# Create directories
mkdir -p "$(dirname "$OUTPUT_VIDEO")"
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

echo -e "${GREEN}Assembling video from duration match plan${NC}"
echo "Match plan: $MATCH_PLAN"
echo "Output: $OUTPUT_VIDEO"
echo ""

$PYTHON_CMD src/duration_video_assembler.py \
    --match-plan "$MATCH_PLAN" \
    --output "$OUTPUT_VIDEO" \
    --temp-dir "$TEMP_DIR" \
    "${FILTERED_ARGS[@]}"

echo ""
echo -e "${GREEN}=== Video Assembly Complete ===${NC}"
echo ""

if [ -f "$OUTPUT_VIDEO" ]; then
    echo "Output saved to: $OUTPUT_VIDEO"

    # Get video info
    if command -v ffprobe &> /dev/null; then
        DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT_VIDEO" 2>/dev/null)
        SIZE=$(du -h "$OUTPUT_VIDEO" | cut -f1)
        echo "  Duration: ${DURATION}s"
        echo "  Size: $SIZE"
    fi
fi
echo ""
