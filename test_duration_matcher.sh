#!/bin/bash
# Test script for duration matcher
# Matches guide segments to source clips by duration

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# Parse --output from extra args if provided, and build filtered args
OUTPUT_JSON="data/segments/duration_match_plan.json"
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

# Detect Python command
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo -e "${YELLOW}Error: Python not found${NC}"
    exit 1
fi

$PYTHON_CMD src/duration_matcher.py \
    --guide "$GUIDE_JSON" \
    --source "$SOURCE_JSON" \
    --output "$OUTPUT_JSON" \
    "${FILTERED_ARGS[@]}"

echo ""
echo "Next step:"
echo "  Run ./test_duration_assembly.sh $OUTPUT_JSON"
echo ""
