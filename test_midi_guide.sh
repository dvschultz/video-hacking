#!/bin/bash
# Convert MIDI file to guide sequence for pitch matching pipeline
# Usage: ./test_midi_guide.sh <midi_file.mid> <channel> [options]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MIDI Guide Converter ===${NC}\n"

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <midi_file.mid> <channel> [options]"
    echo ""
    echo "Arguments:"
    echo "  midi_file.mid    Path to MIDI file"
    echo "  channel          MIDI channel to extract (0-15)"
    echo ""
    echo "Options:"
    echo "  --output PATH        Output JSON path (default: data/segments/guide_sequence.json)"
    echo "  --min-rest N         Minimum rest duration to preserve (default: 0.1s)"
    echo "  --sample-rate N      Sample rate for audio preview (default: 22050)"
    echo "  --no-audio           Skip audio preview generation"
    echo "  --list-channels      List available channels and exit"
    echo ""
    echo "Examples:"
    echo "  # List channels in a MIDI file:"
    echo "  $0 data/input/song.mid 0 --list-channels"
    echo ""
    echo "  # Convert melody from channel 1:"
    echo "  $0 data/input/song.mid 1"
    echo ""
    echo "  # Convert with smaller minimum rest:"
    echo "  $0 data/input/song.mid 1 --min-rest 0.05"
    exit 1
fi

MIDI_FILE="$1"
CHANNEL="$2"
shift 2  # Remove first two arguments

# Check if file exists
if [ ! -f "$MIDI_FILE" ]; then
    echo -e "${YELLOW}Error: MIDI file not found: $MIDI_FILE${NC}"
    exit 1
fi

# Setup paths
OUTPUT_DIR="data/segments"
TEMP_DIR="data/temp"
OUTPUT_JSON="$OUTPUT_DIR/guide_sequence.json"

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

echo -e "${GREEN}Converting MIDI to guide sequence${NC}"
echo "Input: $MIDI_FILE"
echo "Channel: $CHANNEL"
echo "Output: $OUTPUT_JSON"
echo ""

$PYTHON_CMD src/midi_guide_converter.py \
    --midi "$MIDI_FILE" \
    --channel "$CHANNEL" \
    --output "$OUTPUT_JSON" \
    --temp-dir "$TEMP_DIR" \
    "$@"

echo ""
echo -e "${GREEN}=== Conversion Complete ===${NC}"
echo ""
echo "Results saved to:"
echo "  - Guide sequence: $OUTPUT_JSON"

# Get the basename without extension for audio preview path
MIDI_BASENAME=$(basename "$MIDI_FILE" | sed 's/\.[^.]*$//')
echo "  - Audio preview: $TEMP_DIR/${MIDI_BASENAME}_midi_preview.wav"
echo ""
echo "Next steps:"
echo "  1. Listen to the audio preview to verify notes are correct"
echo "  2. Run pitch matcher to match against source database:"
echo "     ./test_pitch_matcher.sh $OUTPUT_JSON data/segments/source_database.json"
echo ""
