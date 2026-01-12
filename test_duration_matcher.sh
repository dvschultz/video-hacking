#!/bin/bash
# Test script for duration matcher
# Matches guide segments to source clips by duration

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Duration Matcher ===${NC}\n"

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <guide_sequence.json> <duration_database.json> [options]"
    echo ""
    echo "Options:"
    echo "  --output PATH           Output match plan JSON (default: data/segments/duration_match_plan.json)"
    echo "  --crop-mode MODE        start, middle, end (default: middle)"
    echo "  --reuse-policy POLICY   none, allow, min_gap, limited, percentage (default: min_gap)"
    echo "  --min-reuse-gap N       Minimum segments between reuses (default: 5)"
    echo "  --max-reuses N          Maximum reuses per clip (default: 3)"
    echo "  --reuse-percentage N    Max reuse percentage (default: 0.3)"
    echo "  --no-prefer-closest     Use shortest valid clip instead of closest duration"
    echo "  --match-rests           Match rest segments with video clips instead of black frames"
    echo "  --edl                   Generate EDL file alongside match plan"
    echo "  --edl-output PATH       Custom EDL output path"
    echo "  --fps N                 Frame rate for EDL timecode (default: 24)"
    echo ""
    echo "Crop Modes:"
    echo "  start   - Use first N seconds of clip"
    echo "  middle  - Trim equally from start/end (centered)"
    echo "  end     - Use last N seconds of clip"
    echo ""
    echo "Reuse Policies:"
    echo "  none       - Each clip used at most once"
    echo "  allow      - Unlimited reuse"
    echo "  min_gap    - Can reuse after N segments gap"
    echo "  limited    - Maximum K reuses per clip"
    echo "  percentage - At most P% of output can be reuses"
    echo ""
    echo "Examples:"
    echo "  $0 data/segments/guide_sequence.json data/segments/duration_database.json"
    echo "  $0 guide.json source.json --crop-mode start"
    echo "  $0 guide.json source.json --reuse-policy allow"
    echo "  $0 guide.json source.json --crop-mode end --reuse-policy none"
    echo "  $0 guide.json source.json --match-rests"
    exit 1
fi

GUIDE_JSON="$1"
SOURCE_JSON="$2"
shift 2  # Remove first two arguments, rest are options

# Check if files exist
if [ ! -f "$GUIDE_JSON" ]; then
    echo -e "${YELLOW}Error: Guide file not found: $GUIDE_JSON${NC}"
    exit 1
fi

if [ ! -f "$SOURCE_JSON" ]; then
    echo -e "${YELLOW}Error: Source database not found: $SOURCE_JSON${NC}"
    exit 1
fi

# Setup paths
OUTPUT_DIR="data/segments"
OUTPUT_JSON="$OUTPUT_DIR/duration_match_plan.json"

# Check if user provided custom output path
for arg in "$@"; do
    if [[ "$arg" == "--output" ]]; then
        CUSTOM_OUTPUT=true
        break
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

echo -e "${GREEN}Matching guide to source by duration${NC}"
echo "Guide: $GUIDE_JSON"
echo "Source: $SOURCE_JSON"

# Build command with conditional output flag
if [ -z "$CUSTOM_OUTPUT" ]; then
    echo "Output: $OUTPUT_JSON"
    echo ""
    $PYTHON_CMD src/duration_matcher.py \
        --guide "$GUIDE_JSON" \
        --source "$SOURCE_JSON" \
        --output "$OUTPUT_JSON" \
        "$@"
else
    echo "Output: (custom path specified)"
    echo ""
    $PYTHON_CMD src/duration_matcher.py \
        --guide "$GUIDE_JSON" \
        --source "$SOURCE_JSON" \
        "$@"
fi

echo ""
echo -e "${GREEN}=== Matching Complete ===${NC}"
echo ""

if [ -f "$OUTPUT_JSON" ]; then
    echo "Match statistics:"
    MATCHED=$(grep -o '"matched_segments": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*')
    MATCHED_RESTS=$(grep -o '"matched_rest_segments": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*')
    UNMATCHED=$(grep -o '"unmatched_segments": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*')
    REST=$(grep -o '"rest_segments_black_frames": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*')
    UNIQUE=$(grep -o '"unique_clips_used": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*')

    echo "  - Matched: $MATCHED"
    if [ -n "$MATCHED_RESTS" ] && [ "$MATCHED_RESTS" -gt 0 ]; then
        echo "  - Rests matched with clips: $MATCHED_RESTS"
    fi
    echo "  - Unmatched: $UNMATCHED"
    echo "  - Rest segments (black frames): $REST"
    echo "  - Unique clips used: $UNIQUE"
fi

echo ""
echo "Next step:"
echo "  Run ./test_duration_assembly.sh $OUTPUT_JSON"
echo ""
