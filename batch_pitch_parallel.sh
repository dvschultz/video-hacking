#!/bin/bash
# Parallel batch processing for pitch source database
# Processes multiple videos simultaneously for faster database building

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== Parallel Batch Pitch Analyzer ===${NC}\n"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder_path> [options]"
    echo ""
    echo "Processes all video files in a folder using parallel workers."
    echo "Much faster than sequential processing for large video collections."
    echo ""
    echo "Options:"
    echo "  --output PATH        Output JSON path (default: data/segments/source_database.json)"
    echo "  --extensions EXTS    Video extensions (default: mp4,mov,avi,mkv,webm)"
    echo "  --parallel N         Number of parallel workers (default: auto)"
    echo "  --fps N              Video frame rate (default: 24)"
    echo "  --threshold N        Pitch change threshold in cents (default: 50)"
    echo "  --min-duration N     Minimum segment duration (default: 0.1)"
    echo "  --silence-threshold N  Silence threshold in dB (default: -40)"
    echo "  --pitch-smoothing N  Median filter window (default: 0=off)"
    echo "  --pitch-method METHOD  Pitch algorithm: crepe, swift-f0, hybrid, pyin"
    echo "  --normalize          Normalize audio loudness (recommended)"
    echo "  --target-lufs N      Target loudness in LUFS (default: -16)"
    echo "  --dry-run            List files without processing"
    echo ""
    echo "Examples:"
    echo "  # Process with 4 parallel workers:"
    echo "  $0 /path/to/videos --parallel 4 --normalize"
    echo ""
    echo "  # Fast processing with swift-f0:"
    echo "  $0 /path/to/videos --pitch-method swift-f0 --parallel 4"
    echo ""
    echo "  # Preview files:"
    echo "  $0 /path/to/videos --dry-run"
    echo ""
    echo "Performance tips:"
    echo "  - Use --pitch-method swift-f0 for ~5x faster CPU processing"
    echo "  - Use --parallel N where N = CPU cores / 2 (default)"
    echo "  - Ensure GPU drivers installed for CREPE acceleration"
    exit 1
fi

FOLDER_PATH="$1"
shift

# Check folder exists
if [ ! -d "$FOLDER_PATH" ]; then
    echo -e "${YELLOW}Error: Folder not found: $FOLDER_PATH${NC}"
    exit 1
fi

# Detect Python
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo -e "${YELLOW}Error: Python not found${NC}"
    exit 1
fi

# Run parallel batch analyzer
$PYTHON_CMD src/batch_pitch_analyzer.py "$FOLDER_PATH" "$@"
