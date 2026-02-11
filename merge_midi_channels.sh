#!/bin/bash
# Merge multiple MIDI channels into a single channel
# Usage: ./merge_midi_channels.sh <input.mid> --channels 0,1 [--output merged.mid]

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== Merge MIDI Channels ===${NC}"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.mid> [options]"
    echo ""
    echo "Merges note events from multiple MIDI channels into a single channel."
    echo ""
    echo "Options:"
    echo "  --channels N,N       Comma-separated channels to merge (e.g., 0,1)"
    echo "  --target-channel N   Target channel for merged notes (default: first in --channels)"
    echo "  --output PATH        Output MIDI file (default: input_merged.mid)"
    echo "  --list-channels      List channels in MIDI file and exit"
    echo ""
    echo "Examples:"
    echo "  # List channels in a MIDI file:"
    echo "  $0 song.mid --list-channels"
    echo ""
    echo "  # Merge channels 0 and 1:"
    echo "  $0 song.mid --channels 0,1"
    echo ""
    echo "  # Merge channels 0,1,2 onto channel 0 with custom output:"
    echo "  $0 song.mid --channels 0,1,2 --output merged_song.mid"
    echo ""
    echo "  # Then use merged file as guide:"
    echo "  ./test_midi_guide.sh merged_song.mid 0"
    exit 1
fi

MIDI_FILE="$1"
shift

# Check if file exists
if [ ! -f "$MIDI_FILE" ]; then
    echo -e "${YELLOW}Error: MIDI file not found: $MIDI_FILE${NC}"
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

$PYTHON_CMD src/merge_midi_channels.py --midi "$MIDI_FILE" "$@"
