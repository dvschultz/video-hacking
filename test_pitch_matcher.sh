#!/bin/bash
# Test script for pitch matcher
# Matches guide sequence to source database

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <guide_sequence.json> <source_database.json> [options]"
    echo ""
    echo "Options:"
    echo "  --output PATH           Output match plan JSON (default: data/segments/match_plan.json)"
    echo "  --reuse-policy POLICY   How to handle clip reuse (default: min_gap)"
    echo "                          Options: none, allow, min_gap, limited, percentage"
    echo "  --min-reuse-gap N       Minimum segments between reuses (default: 5)"
    echo "  --max-reuses N          Maximum reuses per segment (default: 3)"
    echo "  --reuse-percentage N    Max reuse percentage (default: 0.3)"
    echo "  --duration-weight N     Weight for duration matching (default: 0.3)"
    echo "  --confidence-weight N   Weight for pitch confidence matching (default: 0.4)"
    echo "  --consistency-weight N  Weight for loopability/consistency matching (default: 0.3)"
    echo "  --min-volume-db N       Minimum volume in dB to use segments (e.g., -40)"
    echo "                          Segments quieter than this threshold are excluded"
    echo "  --no-transposition      Disable pitch transposition"
    echo "  --max-transpose N       Maximum semitones to transpose (default: 12)"
    echo "  --no-combine-clips      Disable combining clips for duration"
    echo "  --one-video-per-note    Lock each unique guide note to one source video"
    echo "                          for visual variety (different notes = different visuals)"
    echo "  --edl                   Generate EDL file alongside match plan"
    echo "  --edl-output PATH       Custom EDL output path"
    echo "  --fps N                 Frame rate for EDL timecode (default: 24)"
    echo ""
    echo "Reuse Policies:"
    echo "  none        - Each source clip used only once (maximum variety)"
    echo "  allow       - Unlimited reuse (best matches)"
    echo "  min_gap     - Minimum 5 segments between reuses (default)"
    echo "  limited     - Each clip reused max 3 times"
    echo "  percentage  - Max 30% of segments can be reuses"
    echo ""
    echo "Examples:"
    echo "  # Basic usage with default settings:"
    echo "  $0 data/segments/guide_sequence.json data/segments/source_database.json"
    echo ""
    echo "  # Allow unlimited reuse for best matches:"
    echo "  $0 guide.json source.json --reuse-policy allow"
    echo ""
    echo "  # No reuse, maximum variety:"
    echo "  $0 guide.json source.json --reuse-policy none"
    echo ""
    echo "  # Exact matches only, no transposition:"
    echo "  $0 guide.json source.json --no-transposition"
    echo ""
    echo "  # Adjust scoring weights:"
    echo "  $0 guide.json source.json --duration-weight 0.4 --confidence-weight 0.4 --consistency-weight 0.2"
    echo ""
    echo "  # Generate EDL for NLE import:"
    echo "  $0 guide.json source.json --edl --fps 24"
    exit 1
fi

GUIDE_JSON="$1"
SOURCE_JSON="$2"
shift 2  # Remove first two arguments, rest are options

# Check if files exist
if [ ! -f "$GUIDE_JSON" ]; then
    echo -e "${YELLOW}Error: Guide sequence not found: $GUIDE_JSON${NC}"
    exit 1
fi

if [ ! -f "$SOURCE_JSON" ]; then
    echo -e "${YELLOW}Error: Source database not found: $SOURCE_JSON${NC}"
    exit 1
fi

# Parse --output from extra args if provided, and build filtered args
OUTPUT_JSON="data/segments/match_plan.json"
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

$PYTHON_CMD src/pitch_matcher.py \
    --guide "$GUIDE_JSON" \
    --source "$SOURCE_JSON" \
    --output "$OUTPUT_JSON" \
    "${FILTERED_ARGS[@]}"

echo ""
echo "Next steps:"
echo "  1. Review match plan statistics above"
echo "  2. Check missing_pitches in JSON if any segments couldn't be matched"
echo "  3. Run video assembler to create final video from match plan"
echo ""
