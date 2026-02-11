#!/bin/bash
# Merge two pitch source database JSON files into one
# Usage: ./merge_source_databases.sh <db1.json> <db2.json> [--output <merged.json>]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Merge Source Databases ===${NC}"

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <database1.json> <database2.json> [options]"
    echo ""
    echo "Merges two pre-computed pitch source databases into a single file."
    echo ""
    echo "Options:"
    echo "  --output PATH   Output path for merged database"
    echo "                  (default: data/segments/source_database_merged.json)"
    echo ""
    echo "Examples:"
    echo "  # Merge two databases:"
    echo "  $0 data/segments/source_db_a.json data/segments/source_db_b.json"
    echo ""
    echo "  # Merge with custom output path:"
    echo "  $0 db1.json db2.json --output data/segments/combined.json"
    echo ""
    echo "  # Use merged database in pitch matching:"
    echo "  ./test_pitch_matcher.sh data/segments/guide_sequence.json data/segments/source_database_merged.json"
    exit 1
fi

DB1="$1"
DB2="$2"
shift 2

# Check if input files exist
if [ ! -f "$DB1" ]; then
    echo -e "${YELLOW}Error: Database file not found: $DB1${NC}"
    exit 1
fi

if [ ! -f "$DB2" ]; then
    echo -e "${YELLOW}Error: Database file not found: $DB2${NC}"
    exit 1
fi

# Parse remaining arguments
OUTPUT_JSON="data/segments/source_database_merged.json"
while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_JSON="$2"
            shift 2
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create output directory
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

$PYTHON_CMD src/merge_source_databases.py "$DB1" "$DB2" --output "$OUTPUT_JSON"

echo ""
echo "Next steps:"
echo "  1. Run pitch matcher: ./test_pitch_matcher.sh data/segments/guide_sequence.json $OUTPUT_JSON"
echo ""
