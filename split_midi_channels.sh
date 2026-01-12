#!/bin/bash
# Split MIDI file into separate WAV files per channel
# Useful for identifying which channel contains the vocal/melody track

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== MIDI Channel Splitter ===${NC}\n"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <midi_file> [output_dir]"
    echo ""
    echo "Exports each MIDI channel as a separate WAV file."
    echo ""
    echo "Arguments:"
    echo "  midi_file    Path to MIDI file"
    echo "  output_dir   Output directory (default: data/temp/midi_channels)"
    echo ""
    echo "Options (pass after arguments):"
    echo "  --list-only      List channels without exporting"
    echo "  --sample-rate N  Audio sample rate (default: 22050)"
    echo ""
    echo "Examples:"
    echo "  $0 song.mid"
    echo "  $0 song.mid data/output/channels"
    echo "  $0 song.mid --list-only"
    echo ""
    echo "After running, listen to each WAV file to identify the melody,"
    echo "then use that channel with: ./test_midi_guide.sh song.mid <channel>"
    exit 1
fi

MIDI_FILE="$1"
shift

# Check if file exists
if [ ! -f "$MIDI_FILE" ]; then
    echo -e "${YELLOW}Error: MIDI file not found: $MIDI_FILE${NC}"
    exit 1
fi

# Default output directory
OUTPUT_DIR="data/temp/midi_channels"

# Check if second argument is a directory path (not an option)
if [ $# -ge 1 ] && [[ ! "$1" == --* ]]; then
    OUTPUT_DIR="$1"
    shift
fi

# Create output directory
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

echo -e "${GREEN}Splitting MIDI channels${NC}"
echo "Input: $MIDI_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

$PYTHON_CMD src/midi_channel_splitter.py \
    --midi "$MIDI_FILE" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
