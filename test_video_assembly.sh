#!/bin/bash
# Test video assembly
# Usage: ./test_video_assembly.sh <source_video> <guidance_audio>

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./test_video_assembly.sh <source_video> <guidance_audio> [options]"
    echo ""
    echo "Options:"
    echo "  --fps N             Frame rate for EDL timecode (default: 24)"
    echo "  --edl               Generate EDL file alongside video"
    echo "  --edl-only          Generate EDL file only (skip video assembly)"
    echo "  --edl-output PATH   Custom EDL output path"
    echo ""
    echo "Examples:"
    echo "  ./test_video_assembly.sh video.mp4 audio.wav"
    echo "  ./test_video_assembly.sh video.mp4 audio.wav --edl"
    echo "  ./test_video_assembly.sh video.mp4 audio.wav --edl-only --fps 24"
    exit 1
fi

SOURCE_VIDEO="$1"
GUIDANCE_AUDIO="$2"
shift 2  # Remove first two arguments, rest are options

# Detect Python command
if command -v python &> /dev/null && python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "Error: Python 3.8+ not found"
    exit 1
fi

echo "============================================"
echo "Video Assembly Test"
echo "============================================"
echo "Source video: $SOURCE_VIDEO"
echo "Guidance audio: $GUIDANCE_AUDIO"
echo ""

# Check if matches exist
if [ ! -f "data/segments/matches.json" ]; then
    echo "Error: Matches not found"
    echo "Run ./test_semantic_matching.sh first"
    exit 1
fi

# Create output directory
mkdir -p data/output

# Run assembly
$PYTHON_CMD src/video_assembler.py \
    --video "$SOURCE_VIDEO" \
    --audio "$GUIDANCE_AUDIO" \
    --matches data/segments/matches.json \
    --output data/output/final_video.mp4 \
    "$@"

echo ""
echo "============================================"
echo "✓ Assembly Complete!"
echo "============================================"
echo ""
echo "H.264 versions (smaller file size):"
echo "  1. Original audio:  data/output/final_video_original_audio.mp4"
echo "  2. Guidance audio:  data/output/final_video.mp4"
echo ""
echo "ProRes 422 version (high quality for editing):"
echo "  - Original audio:   data/output/final_video_original_audio_prores.mov"
echo ""
echo "Play H.264 versions:"
echo "  open data/output/final_video_original_audio.mp4  # macOS"
echo "  open data/output/final_video.mp4                 # macOS"
echo ""
echo "Use ProRes version for further editing in professional video software!"
echo ""
