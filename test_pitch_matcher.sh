#!/bin/bash
# Test script for pitch matcher
# Matches guide sequence to source database

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Pitch Matcher ===${NC}\n"

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
    echo "  --no-transposition      Disable pitch transposition"
    echo "  --max-transpose N       Maximum semitones to transpose (default: 12)"
    echo "  --no-combine-clips      Disable combining clips for duration"
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

# Setup paths
OUTPUT_DIR="data/segments"
OUTPUT_JSON="$OUTPUT_DIR/match_plan.json"

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

echo -e "${GREEN}Matching guide sequence to source database${NC}"
echo "Guide: $GUIDE_JSON"
echo "Source: $SOURCE_JSON"
echo "Output: $OUTPUT_JSON"
echo ""

$PYTHON_CMD src/pitch_matcher.py \
    --guide "$GUIDE_JSON" \
    --source "$SOURCE_JSON" \
    --output "$OUTPUT_JSON" \
    "$@"

echo ""
echo -e "${GREEN}=== Matching Complete ===${NC}"
echo ""
echo "Results saved to:"
echo "  - Match plan: $OUTPUT_JSON"
echo ""

# Extract and display stats
if [ -f "$OUTPUT_JSON" ]; then
    echo "Match statistics:"

    TOTAL=$(grep -o '"total_guide_segments": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*')
    EXACT=$(grep -o '"exact_matches": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*')
    TRANSPOSED=$(grep -o '"transposed_matches": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*')
    MISSING=$(grep -o '"missing_matches": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*')
    UNIQUE=$(grep -o '"unique_source_segments_used": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*')
    REUSED=$(grep -o '"segments_reused": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*')

    echo "  - Total guide segments: $TOTAL"
    echo "  - Exact matches: $EXACT"
    echo "  - Transposed matches: $TRANSPOSED"
    echo "  - Missing matches: $MISSING"
    echo "  - Unique source segments used: $UNIQUE"
    echo "  - Segments reused: $REUSED"
fi

echo ""
echo "Next steps:"
echo "  1. Review match plan statistics above"
echo "  2. Check missing_pitches in JSON if any segments couldn't be matched"
echo "  3. Run video assembler to create final video from match plan"
echo ""
