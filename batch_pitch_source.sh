#!/bin/bash
# Batch process a folder of videos into a single source database
# Usage: ./batch_pitch_source.sh <folder_path> [options]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Batch Pitch Source Database Builder ===${NC}"
echo -e "${YELLOW}TIP: For faster processing, use ./batch_pitch_parallel.sh instead${NC}\n"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder_path> [options]"
    echo ""
    echo "Processes all video files in a folder into a single source database."
    echo ""
    echo "Options:"
    echo "  --output PATH        Output JSON path (default: data/segments/source_database.json)"
    echo "  --extensions EXTS    Video extensions to process (default: mp4,mov,avi,mkv,webm)"
    echo "  --fps N              Video frame rate (default: 24)"
    echo "  --threshold N        Pitch change threshold in cents (default: 50)"
    echo "  --min-duration N     Minimum segment duration (default: 0.1s)"
    echo "  --silence-threshold N  Silence detection threshold in dB (default: -50)"
    echo "  --pitch-smoothing N  Median filter window size (default: 0=off)"
    echo "  --pitch-method METHOD  Pitch detection algorithm (default: crepe)"
    echo "                       Options: crepe, swift-f0, basic-pitch, hybrid, pyin"
    echo "  --normalize          Normalize audio loudness before analysis (recommended)"
    echo "  --target-lufs N      Target loudness in LUFS when normalizing (default: -16)"
    echo "  --dry-run            List files that would be processed without processing"
    echo ""
    echo "Examples:"
    echo "  # Process all videos in a folder with normalization:"
    echo "  $0 /path/to/videos --normalize"
    echo ""
    echo "  # Process only mp4 and mov files:"
    echo "  $0 /path/to/videos --extensions mp4,mov"
    echo ""
    echo "  # Custom output and settings:"
    echo "  $0 /path/to/videos --output my_database.json --pitch-method hybrid --normalize"
    echo ""
    echo "  # Preview what would be processed:"
    echo "  $0 /path/to/videos --dry-run"
    exit 1
fi

FOLDER_PATH="$1"
shift  # Remove first argument

# Check if folder exists
if [ ! -d "$FOLDER_PATH" ]; then
    echo -e "${RED}Error: Folder not found: $FOLDER_PATH${NC}"
    exit 1
fi

# Parse arguments
OUTPUT_JSON="data/segments/source_database.json"
EXTENSIONS="mp4,mov,avi,mkv,webm"
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_JSON="$2"
            shift 2
            ;;
        --extensions)
            EXTENSIONS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Setup paths
OUTPUT_DIR=$(dirname "$OUTPUT_JSON")
TEMP_DIR="data/temp"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# Build find pattern for extensions
IFS=',' read -ra EXT_ARRAY <<< "$EXTENSIONS"
FIND_PATTERN=""
for ext in "${EXT_ARRAY[@]}"; do
    if [ -n "$FIND_PATTERN" ]; then
        FIND_PATTERN="$FIND_PATTERN -o"
    fi
    FIND_PATTERN="$FIND_PATTERN -iname *.$ext"
done

# Find all video files
echo -e "${GREEN}Scanning folder: $FOLDER_PATH${NC}"
echo "Looking for extensions: $EXTENSIONS"
echo ""

# Use find to get video files, sorted alphabetically
# Skip hidden files (starting with .)
VIDEO_FILES=()
while IFS= read -r -d '' file; do
    # Skip files starting with .
    basename=$(basename "$file")
    if [[ "$basename" != .* ]]; then
        VIDEO_FILES+=("$file")
    fi
done < <(find "$FOLDER_PATH" -type f \( $FIND_PATTERN \) -print0 | sort -z)

TOTAL_FILES=${#VIDEO_FILES[@]}

if [ $TOTAL_FILES -eq 0 ]; then
    echo -e "${YELLOW}No video files found with extensions: $EXTENSIONS${NC}"
    exit 1
fi

echo -e "${GREEN}Found $TOTAL_FILES video files${NC}"
echo ""

# List files
echo "Files to process:"
for i in "${!VIDEO_FILES[@]}"; do
    echo "  $((i+1)). $(basename "${VIDEO_FILES[$i]}")"
done
echo ""

# Dry run - just list files and exit
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}Dry run - no files processed${NC}"
    exit 0
fi

# Detect Python command
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Process each video
PROCESSED=0
FAILED=0
START_TIME=$(date +%s)

for i in "${!VIDEO_FILES[@]}"; do
    VIDEO="${VIDEO_FILES[$i]}"
    VIDEO_NAME=$(basename "$VIDEO")
    CURRENT=$((i+1))

    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}[$CURRENT/$TOTAL_FILES] Processing: $VIDEO_NAME${NC}"
    echo -e "${BLUE}========================================${NC}"

    # First video creates database, subsequent videos append
    if [ $i -eq 0 ]; then
        APPEND_FLAG=""
        echo "Creating new database..."
    else
        APPEND_FLAG="--append"
        echo "Appending to database..."
    fi

    # Run the pitch source analyzer
    if $PYTHON_CMD src/pitch_source_analyzer.py \
        --video "$VIDEO" \
        --output "$OUTPUT_JSON" \
        --temp-dir "$TEMP_DIR" \
        $APPEND_FLAG \
        "${EXTRA_ARGS[@]}"; then
        PROCESSED=$((PROCESSED+1))
        echo -e "${GREEN}Successfully processed: $VIDEO_NAME${NC}"
    else
        FAILED=$((FAILED+1))
        echo -e "${RED}Failed to process: $VIDEO_NAME${NC}"
    fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}=== Batch Processing Complete ===${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Summary:"
echo "  - Total files found: $TOTAL_FILES"
echo "  - Successfully processed: $PROCESSED"
echo "  - Failed: $FAILED"
echo "  - Time elapsed: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Output saved to: $OUTPUT_JSON"
echo ""

# Show database stats
if [ -f "$OUTPUT_JSON" ]; then
    echo "Database statistics:"
    NUM_VIDEOS=$(grep -o '"num_videos": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*' || echo "?")
    NUM_SEGMENTS=$(grep -o '"num_segments": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*' || echo "?")
    NUM_PITCHES=$(grep -o '"num_unique_pitches": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*' || echo "?")
    NUM_SILENCES=$(grep -o '"num_silence_gaps": [0-9]*' "$OUTPUT_JSON" | grep -o '[0-9]*' || echo "?")

    echo "  - Videos in database: $NUM_VIDEOS"
    echo "  - Total segments: $NUM_SEGMENTS"
    echo "  - Unique pitches: $NUM_PITCHES"
    echo "  - Silence gaps: $NUM_SILENCES"
fi
echo ""
echo "Next steps:"
echo "  1. Create guide sequence from MIDI or video"
echo "  2. Run pitch matcher: ./test_pitch_matcher.sh data/segments/guide_sequence.json $OUTPUT_JSON"
echo ""
