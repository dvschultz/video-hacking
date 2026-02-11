#!/bin/bash
# Render MIDI file to WAV (all channels merged)
#
# Supports two synthesis modes:
# - FluidSynth (default): Uses SoundFont for realistic instrument sounds
# - Simple synthesis (--no-fluidsynth): Basic sine wave synthesis

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== MIDI to WAV Renderer ===${NC}\n"

# Show help if no arguments or --help/-h flag
if [ $# -lt 1 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 <midi_file> [output.wav] [options]"
    echo ""
    echo "Renders a MIDI file to WAV with all channels merged."
    echo ""
    echo "Arguments:"
    echo "  midi_file    Path to MIDI file"
    echo "  output.wav   Output WAV path (default: <midi_name>.wav)"
    echo ""
    echo "Options (pass after arguments):"
    echo "  --soundfont PATH   Path to SoundFont file (.sf2)"
    echo "  --no-fluidsynth    Use simple synthesis instead of FluidSynth"
    echo "  --sample-rate N    Audio sample rate for simple synthesis (default: 22050)"
    echo ""
    echo "FluidSynth Mode (default):"
    echo "  Uses SoundFonts for realistic instrument sounds. Requires:"
    echo "  - FluidSynth library: brew install fluidsynth (macOS)"
    echo "  - pyfluidsynth: pip install pyfluidsynth"
    echo "  - SoundFont file: Download to data/soundfonts/"
    echo ""
    echo "  Recommended SoundFont (free, ~30MB):"
    echo "  https://schristiancollins.com/generaluser.php"
    echo "  Save to: data/soundfonts/GeneralUser_GS.sf2"
    echo ""
    echo "Examples:"
    echo "  $0 song.mid"
    echo "  $0 song.mid output.wav"
    echo "  $0 song.mid --soundfont data/soundfonts/GeneralUser_GS.sf2"
    echo "  $0 song.mid output.wav --soundfont data/soundfonts/GeneralUser_GS.sf2"
    echo "  $0 song.mid --no-fluidsynth"
    exit 1
fi

MIDI_FILE="$1"
shift

# Check if file exists
if [ ! -f "$MIDI_FILE" ]; then
    echo -e "${YELLOW}Error: MIDI file not found: $MIDI_FILE${NC}"
    exit 1
fi

# Check if second argument is an output path (not an option)
OUTPUT_ARG=""
if [ $# -ge 1 ] && [[ ! "$1" == --* ]]; then
    OUTPUT_ARG="--output $1"
    shift
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

echo -e "${GREEN}Rendering MIDI to WAV${NC}"
echo "Input: $MIDI_FILE"
echo ""

$PYTHON_CMD src/midi_renderer.py \
    --midi "$MIDI_FILE" \
    $OUTPUT_ARG \
    "$@"
