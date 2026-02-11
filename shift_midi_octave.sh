#!/bin/bash
# Shift MIDI channel by octaves
# Usage: ./shift_midi_octave.sh <input.mid> <channel> <octave_shift> [output.mid] [options]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MIDI Octave Shifter ===${NC}\n"

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input.mid> <channel> <octave_shift> [output.mid] [options]"
    echo ""
    echo "Arguments:"
    echo "  input.mid       Path to input MIDI file"
    echo "  channel         MIDI channel to shift (0-15)"
    echo "  octave_shift    Octaves to shift (-N=down, +N=up)"
    echo "  output.mid      Path to output MIDI file (optional, auto-generated)"
    echo ""
    echo "Options:"
    echo "  --list-channels    List available channels and exit"
    echo "  --no-audio         Skip WAV preview generation"
    echo "  --audio-output     Custom WAV output path"
    echo ""
    echo "Examples:"
    echo "  # List channels in a MIDI file:"
    echo "  $0 data/input/song.mid 0 --list-channels"
    echo ""
    echo "  # Shift channel 0 down one octave:"
    echo "  $0 data/input/song.mid 0 -1"
    echo ""
    echo "  # Shift channel 2 up two octaves with custom output:"
    echo "  $0 data/input/song.mid 2 +2 data/output/song_high.mid"
    echo ""
    echo "  # Shift without generating audio preview:"
    echo "  $0 data/input/song.mid 0 -1 --no-audio"
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

# Parse remaining arguments
LIST_CHANNELS=false
OCTAVE_SHIFT=""
OUTPUT_FILE=""
NO_AUDIO=false
AUDIO_OUTPUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --list-channels)
            LIST_CHANNELS=true
            shift
            ;;
        --no-audio)
            NO_AUDIO=true
            shift
            ;;
        --audio-output)
            AUDIO_OUTPUT="$2"
            shift 2
            ;;
        -*)
            # Negative number (octave shift)
            if [ -z "$OCTAVE_SHIFT" ]; then
                OCTAVE_SHIFT="$1"
            else
                echo -e "${YELLOW}Error: Unknown option: $1${NC}"
                exit 1
            fi
            shift
            ;;
        +*)
            # Positive number with explicit +
            if [ -z "$OCTAVE_SHIFT" ]; then
                OCTAVE_SHIFT="$1"
            else
                echo -e "${YELLOW}Error: Unknown option: $1${NC}"
                exit 1
            fi
            shift
            ;;
        *)
            # Positional argument
            if [ -z "$OCTAVE_SHIFT" ]; then
                OCTAVE_SHIFT="$1"
            elif [ -z "$OUTPUT_FILE" ]; then
                OUTPUT_FILE="$1"
            else
                echo -e "${YELLOW}Error: Unexpected argument: $1${NC}"
                exit 1
            fi
            shift
            ;;
    esac
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

    $PYTHON_CMD src/midi_octave_shifter.py \
        --midi "$MIDI_FILE" \
        --channel "$CHANNEL" \
        --list-channels
else
    # Require octave shift for non-list mode
    if [ -z "$OCTAVE_SHIFT" ]; then
        echo -e "${YELLOW}Error: Octave shift required when not using --list-channels${NC}"
        echo "Usage: $0 <input.mid> <channel> <octave_shift> [output.mid]"
        exit 1
    fi

    echo -e "${GREEN}Shifting channel $CHANNEL by $OCTAVE_SHIFT octave(s)${NC}"
    echo "Input: $MIDI_FILE"

    # Build command
    CMD="$PYTHON_CMD src/midi_octave_shifter.py --midi \"$MIDI_FILE\" --channel $CHANNEL --octaves $OCTAVE_SHIFT"

    if [ -n "$OUTPUT_FILE" ]; then
        echo "Output: $OUTPUT_FILE"
        # Create output directory if needed
        OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
        if [ "$OUTPUT_DIR" != "." ]; then
            mkdir -p "$OUTPUT_DIR"
        fi
        CMD="$CMD --output \"$OUTPUT_FILE\""
    fi

    if [ "$NO_AUDIO" = true ]; then
        CMD="$CMD --no-audio"
    fi

    if [ -n "$AUDIO_OUTPUT" ]; then
        CMD="$CMD --audio-output \"$AUDIO_OUTPUT\""
    fi

    echo ""

    # Execute command
    eval $CMD

    echo ""
    echo -e "${GREEN}=== Complete ===${NC}"
fi
