#!/bin/bash
# Split polyphonic MIDI channel into monophonic voice guide sequences
# Usage: ./split_midi_voices.sh <midi_file> <channel> [output_dir] [options]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MIDI Voice Splitter ===${NC}\n"

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <midi_file> <channel> [output_dir] [options]"
    echo ""
    echo "Arguments:"
    echo "  midi_file      Path to MIDI file"
    echo "  channel        MIDI channel to split (0-15)"
    echo "  output_dir     Output directory (default: data/segments)"
    echo ""
    echo "Options:"
    echo "  --min-rest N       Minimum rest duration to preserve (default: 0.1s)"
    echo "  --sample-rate N    Sample rate for audio previews (default: 22050)"
    echo "  --no-audio         Skip audio preview generation"
    echo "  --strategy S       Voice assignment strategy: pitch-ordered, balanced (default: pitch-ordered)"
    echo "  --list-channels    List channels with polyphony info and exit"
    echo ""
    echo "Examples:"
    echo "  # List channels with polyphony info:"
    echo "  $0 data/input/song.mid 0 --list-channels"
    echo ""
    echo "  # Split channel 5 into voice files:"
    echo "  $0 data/input/song.mid 5"
    echo ""
    echo "  # Split with balanced voice assignment:"
    echo "  $0 data/input/song.mid 5 --strategy balanced"
    echo ""
    echo "  # Split with custom output directory:"
    echo "  $0 data/input/song.mid 5 data/segments/voices"
    exit 1
fi

MIDI_FILE="$1"
CHANNEL="$2"
shift 2  # Remove first two arguments

# Check if third arg is output dir (not starting with --)
OUTPUT_DIR="data/segments"
if [ $# -gt 0 ] && [[ ! "$1" == --* ]]; then
    OUTPUT_DIR="$1"
    shift
fi

# Check if file exists
if [ ! -f "$MIDI_FILE" ]; then
    echo -e "${YELLOW}Error: MIDI file not found: $MIDI_FILE${NC}"
    exit 1
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "data/temp"

# Detect Python command
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo -e "${YELLOW}Error: Python not found${NC}"
    exit 1
fi

echo -e "${GREEN}Splitting MIDI voices${NC}"
echo "Input: $MIDI_FILE"
echo "Channel: $CHANNEL"
echo "Output dir: $OUTPUT_DIR"
echo ""

$PYTHON_CMD src/midi_voice_splitter.py \
    --midi "$MIDI_FILE" \
    --channel "$CHANNEL" \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo ""
echo -e "${GREEN}=== Split Complete ===${NC}"
echo ""
echo "Next steps (repeat for each voice):"
echo ""
echo "  1. Listen to audio previews to verify the split:"
echo "     open data/temp/guide_voice1_preview.wav"
echo ""
echo "  2. Run pitch matcher per voice:"
echo "     ./test_pitch_matcher.sh $OUTPUT_DIR/guide_sequence_voice1.json \\"
echo "         data/segments/source_database.json \\"
echo "         --output data/segments/match_plan_voice1.json"
echo ""
echo "  3. Assemble video per voice:"
echo "     ./test_pitch_video_assembly.sh data/segments/match_plan_voice1.json"
echo ""
echo "  4. Composite all voice videos in your editor"
echo ""
