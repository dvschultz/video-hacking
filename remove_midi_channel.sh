#!/bin/bash
# Remove a specific MIDI channel from a MIDI file
# Usage: ./remove_midi_channel.sh <input.mid> <channel> <output.mid> [options]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MIDI Channel Remover ===${NC}\n"

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input.mid> <channel> [output.mid] [options]"
    echo ""
    echo "Arguments:"
    echo "  input.mid     Path to input MIDI file"
    echo "  channel       MIDI channel to remove (0-15)"
    echo "  output.mid    Path to output MIDI file (required unless --list-channels)"
    echo ""
    echo "Options:"
    echo "  --list-channels    List available channels and exit"
    echo ""
    echo "Examples:"
    echo "  # List channels in a MIDI file:"
    echo "  $0 data/input/song.mid 0 --list-channels"
    echo ""
    echo "  # Remove channel 5 from a MIDI file:"
    echo "  $0 data/input/song.mid 5 data/output/song_no_ch5.mid"
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

# Check for --list-channels flag
LIST_CHANNELS=false
OUTPUT_FILE=""
EXTRA_ARGS=""

for arg in "$@"; do
    if [ "$arg" = "--list-channels" ]; then
        LIST_CHANNELS=true
    elif [ -z "$OUTPUT_FILE" ]; then
        OUTPUT_FILE="$arg"
    else
        EXTRA_ARGS="$EXTRA_ARGS $arg"
    fi
done

# Detect Python command
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo -e "${YELLOW}Error: Python not found${NC}"
    exit 1
fi

if [ "$LIST_CHANNELS" = true ]; then
    echo -e "${GREEN}Listing channels in MIDI file${NC}"
    echo "Input: $MIDI_FILE"
    echo ""

    $PYTHON_CMD src/midi_channel_remover.py \
        --midi "$MIDI_FILE" \
        --channel "$CHANNEL" \
        --list-channels
else
    # Require output file
    if [ -z "$OUTPUT_FILE" ]; then
        echo -e "${YELLOW}Error: Output file required when not using --list-channels${NC}"
        echo "Usage: $0 <input.mid> <channel> <output.mid>"
        exit 1
    fi

    # Create output directory if needed
    OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
    if [ "$OUTPUT_DIR" != "." ]; then
        mkdir -p "$OUTPUT_DIR"
    fi

    echo -e "${GREEN}Removing channel $CHANNEL from MIDI file${NC}"
    echo "Input: $MIDI_FILE"
    echo "Output: $OUTPUT_FILE"
    echo ""

    $PYTHON_CMD src/midi_channel_remover.py \
        --midi "$MIDI_FILE" \
        --channel "$CHANNEL" \
        --output "$OUTPUT_FILE" \
        $EXTRA_ARGS

    echo ""
    echo -e "${GREEN}=== Complete ===${NC}"
    echo ""
    echo "Output saved to: $OUTPUT_FILE"
fi
