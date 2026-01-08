#!/bin/bash
# Test script for pitch video assembler
# Assembles final video from match plan

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Pitch Video Assembler ===${NC}\n"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <match_plan.json> [options]"
    echo ""
    echo "Options:"
    echo "  --output PATH       Output video path (default: data/output/pitch_matched_video.mp4)"
    echo "  --temp-dir PATH     Temporary directory (default: data/temp)"
    echo "  --resolution WxH    Output resolution (e.g., 1920x1080). Will prompt if not specified."
    echo "  --auto-resolution   Use smallest source resolution automatically (no prompt)"
    echo "  --fps N             Output frame rate (e.g., 24, 29.97). Will prompt if not specified."
    echo "  --auto-fps          Use smallest source frame rate automatically (no prompt)"
    echo "  --parallel N        Number of parallel workers for normalization (default: auto, max 8)"
    echo "  --true-silence      Use black frames + muted audio for rests (instead of source silence clips)"
    echo "  --no-cleanup        Keep temporary files for debugging"
    echo ""
    echo "Examples:"
    echo "  # Basic usage (will prompt for resolution and fps):"
    echo "  $0 data/segments/match_plan.json"
    echo ""
    echo "  # Auto-select smallest resolution and fps:"
    echo "  $0 data/segments/match_plan.json --auto-resolution --auto-fps"
    echo ""
    echo "  # Specify exact resolution and fps:"
    echo "  $0 data/segments/match_plan.json --resolution 1280x720 --fps 30"
    echo ""
    echo "  # Custom output location:"
    echo "  $0 data/segments/match_plan.json --output videos/final.mp4"
    exit 1
fi

MATCH_PLAN="$1"
shift  # Remove first argument, rest are options

# Check if file exists
if [ ! -f "$MATCH_PLAN" ]; then
    echo -e "${YELLOW}Error: Match plan not found: $MATCH_PLAN${NC}"
    exit 1
fi

# Setup paths
OUTPUT_DIR="data/output"
OUTPUT_VIDEO="$OUTPUT_DIR/pitch_matched_video.mp4"
TEMP_DIR="data/temp"

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

echo -e "${GREEN}Assembling video from match plan${NC}"
echo "Match plan: $MATCH_PLAN"
echo "Output: $OUTPUT_VIDEO"
echo ""

# Run assembler
$PYTHON_CMD src/pitch_video_assembler.py \
    --match-plan "$MATCH_PLAN" \
    --output "$OUTPUT_VIDEO" \
    --temp-dir "$TEMP_DIR" \
    "$@"

echo ""
echo -e "${GREEN}=== Video Assembly Complete ===${NC}"
echo ""

# Check if output file exists and show info
if [ -f "$OUTPUT_VIDEO" ]; then
    echo "Output saved to:"
    echo "  - $OUTPUT_VIDEO"
    echo ""

    # Get video duration and size using ffprobe
    if command -v ffprobe &> /dev/null; then
        DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT_VIDEO" 2>/dev/null || echo "unknown")
        SIZE=$(du -h "$OUTPUT_VIDEO" | cut -f1)

        if [ "$DURATION" != "unknown" ]; then
            DURATION_FORMATTED=$(printf "%.1f" "$DURATION")
            echo "Video info:"
            echo "  - Duration: ${DURATION_FORMATTED}s"
            echo "  - File size: $SIZE"
            echo ""
        fi
    fi

    echo "Next steps:"
    echo "  1. Review the output video: open $OUTPUT_VIDEO"
    echo "  2. Check if pitch matching sounds natural"
    echo "  3. Adjust match plan parameters if needed (reuse policy, transposition, etc.)"
else
    echo -e "${YELLOW}Warning: Output file not created${NC}"
fi

echo ""
