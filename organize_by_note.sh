#!/bin/bash
# Organize pitch database clips by musical note
# Usage: ./organize_by_note.sh <database.json> <output_dir> <mode> [options]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Organize by Note ===${NC}\n"

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <database.json> <output_dir> <mode> [options]"
    echo ""
    echo "Arguments:"
    echo "  database.json    Path to source_database.json"
    echo "  output_dir       Output directory path"
    echo "  mode             Output mode: 'clips', 'edl', or 'clips-edl'"
    echo ""
    echo "Modes:"
    echo "  clips       Extract video clips only"
    echo "  edl         Generate EDL files referencing original sources"
    echo "  clips-edl   Extract clips AND generate EDLs referencing extracted clips"
    echo ""
    echo "Options (for clips/clips-edl modes):"
    echo "  --resolution WxH    Output resolution (e.g., 1920x1080)"
    echo "  --auto-resolution   Use smallest source resolution (no prompt)"
    echo "  --fps N             Frame rate (e.g., 30)"
    echo "  --auto-fps          Use smallest source fps (no prompt)"
    echo "  --format FORMAT     Video format (default: mp4)"
    echo ""
    echo "Examples:"
    echo "  # Extract clips with interactive prompts:"
    echo "  $0 data/segments/source_database.json output/notes clips"
    echo ""
    echo "  # Extract clips with specific resolution:"
    echo "  $0 data/segments/source_database.json output/notes clips --resolution 1920x1080 --fps 30"
    echo ""
    echo "  # Extract clips AND generate EDLs referencing them:"
    echo "  $0 data/segments/source_database.json output/notes clips-edl"
    echo ""
    echo "  # Generate EDL files (original sources):"
    echo "  $0 data/segments/source_database.json output/notes edl"
    exit 1
fi

DATABASE="$1"
OUTPUT_DIR="$2"
MODE="$3"
shift 3  # Remove first three arguments

# Check if database exists
if [ ! -f "$DATABASE" ]; then
    echo -e "${YELLOW}Error: Database not found: $DATABASE${NC}"
    exit 1
fi

# Validate mode
if [ "$MODE" != "clips" ] && [ "$MODE" != "edl" ] && [ "$MODE" != "clips-edl" ]; then
    echo -e "${YELLOW}Error: Mode must be 'clips', 'edl', or 'clips-edl', got: $MODE${NC}"
    exit 1
fi

# Detect Python command
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo -e "${YELLOW}Error: Python not found${NC}"
    exit 1
fi

echo -e "${GREEN}Organizing clips by note${NC}"
echo "Database: $DATABASE"
echo "Output: $OUTPUT_DIR"
echo "Mode: $MODE"
echo ""

$PYTHON_CMD src/organize_by_note.py \
    --database "$DATABASE" \
    --output "$OUTPUT_DIR" \
    --mode "$MODE" \
    "$@"

echo ""
echo -e "${GREEN}=== Complete ===${NC}"
echo ""
echo "Output saved to: $OUTPUT_DIR"
