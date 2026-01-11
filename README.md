# Video Hacking - Audio-Driven Video Art

A complete Python toolkit for creating audio-reactive video art by cutting and reassembling video footage based on semantic audio-visual matching using ImageBind embeddings.

## Overview

This project uses **ImageBind multimodal embeddings** to semantically match video clips to audio segments. It analyzes audio using onset strength detection, extracts embeddings from both audio and video, and reassembles the video based on semantic similarity.

## Features

### Audio-Driven Video Art
- **Onset strength analysis**: Frame-accurate continuous onset strength values for precise cut points
- **Source separation**: Separate audio into stems using Demucs (drums, bass, vocals, other)
- **ImageBind embeddings**: Unified audio-visual embeddings for semantic matching
- **Semantic video matching**: Intelligently match video segments to audio based on content
- **Flexible reuse policies**: Control how video segments can be reused (none, allow, min_gap, limited, percentage)
- **High-quality output**: H.264 (CRF 18) and ProRes 422 outputs for editing
- **Interactive visualization**: Real-time threshold adjustment with playback

### Pitch-Matching Video Recutting
- **Multiple pitch detection methods**: CREPE, SwiftF0, Basic Pitch, or hybrid mixture-of-experts
- **Intelligent pitch tracking**: Handles vibrato, pitch drift, and silence detection
- **Pitch smoothing**: Median filtering to reduce false segmentation from natural vocal variations
- **MIDI preview videos**: Visual verification of pitch detection accuracy
- **Configurable parameters**: Silence threshold, pitch change sensitivity, segment duration filters

### Duration-Based Video Matching
- **Folder-based source clips**: Match guide segments to a folder of video clips
- **Duration matching**: Automatically selects shortest clip >= target duration
- **Three crop modes**: Crop from start, middle (centered), or end of clips
- **Same reuse policies**: Control clip repetition (none, allow, min_gap, limited, percentage)
- **Video metadata tracking**: Stores resolution, fps, codec for each clip

## Installation

### Requirements

- Python 3.10 or 3.11 (recommended)
- FFmpeg (for video processing)
- macOS, Linux, or Windows
- GPU recommended (but not required)

### Setup

```bash
# Create conda environment (recommended)
conda create -n vh python=3.11
conda activate vh

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (if not already installed)
# macOS:
brew install ffmpeg

# Linux:
sudo apt-get install ffmpeg
```

**Important:**
- Always activate your conda/virtual environment before running scripts
- This installs PyTorch 2.1.0 and NumPy 1.x for compatibility
- ImageBind will be installed automatically on first use

## Quick Start

### Complete Pipeline (All 5 Phases)

```bash
# Run the complete pipeline
./test_full_pipeline.sh path/to/guidance_audio.wav path/to/source_video.mp4

# This will:
# 1. Separate audio into stems (Demucs)
# 2. Analyze onset strength
# 3. Segment audio based on onset points
# 4. Extract ImageBind audio embeddings
# 5. Extract ImageBind video embeddings (sliding window)
```

Then run semantic matching and assembly:

```bash
# Create semantic matches between audio and video
./test_semantic_matching.sh allow

# Assemble final videos
./test_video_assembly.sh path/to/source_video.mp4 path/to/guidance_audio.wav
```

## Pipeline Phases

### Phase 1: Audio Analysis & Segmentation

**Onset Strength Analysis:**
```bash
python src/onset_strength_analysis.py \
  --audio data/separated/htdemucs/song/other.wav \
  --output data/output/onset_strength.json \
  --fps 24 \
  --power 0.6 \
  --threshold 0.2
```

**Parameters:**
- `--fps`: Frame rate (24, 30, or 60)
- `--power`: Power-law compression (0.5-1.0, lower = more sensitive)
- `--window-size`: Smoothing window (0-5 frames)
- `--tolerance`: Noise removal (0.0-1.0)
- `--threshold`: Cut point threshold (0.0-1.0)

**Audio Segmentation:**
```bash
python src/audio_segmenter.py \
  --audio data/separated/htdemucs/song/other.wav \
  --onset-strength data/output/onset_strength.json \
  --output-dir data/segments/audio \
  --threshold 0.2
```

### Phase 2: Audio Embeddings (ImageBind)

```bash
python src/imagebind_audio_embedder.py \
  --segments-metadata data/segments/audio_segments.json \
  --segments-dir data/segments/audio \
  --output data/segments/audio_embeddings.json \
  --batch-size 8 \
  --device auto
```

Extracts 1024-dimensional ImageBind embeddings for each audio segment.

### Phase 3: Video Embeddings (ImageBind)

```bash
python src/imagebind_video_embedder.py \
  --video path/to/video.mp4 \
  --output data/segments/video_embeddings.json \
  --fps 24 \
  --window-size 5 \
  --stride 6 \
  --chunk-size 500
```

**Parameters:**
- `--window-size`: Frames per window (default 5 = ~0.2s at 24fps)
- `--stride`: Frame step for sliding window (default 6 = 0.25s at 24fps)
- `--chunk-size`: Max frames loaded in memory at once (default 500, reduce if out of memory)

### Phase 4: Semantic Matching

```bash
python src/semantic_matcher.py \
  --audio-embeddings data/segments/audio_embeddings.json \
  --video-embeddings data/segments/video_embeddings.json \
  --audio-segments data/segments/audio_segments.json \
  --output data/segments/matches.json \
  --reuse-policy allow
```

**Reuse Policies:**
- `none`: Each video segment used only once (maximum variety)
- `allow`: Unlimited reuse (best semantic matches)
- `min_gap`: Minimum 5 segments between reuses
- `limited`: Each video segment reused max 3 times
- `percentage`: Max 30% of segments can be reuses

### Phase 5: Video Assembly

```bash
python src/video_assembler.py \
  --video path/to/source_video.mp4 \
  --audio path/to/guidance_audio.wav \
  --matches data/segments/matches.json \
  --output data/output/final_video.mp4
```

**Outputs:**
1. `final_video_original_audio.mp4` - H.264 with original video audio
2. `final_video.mp4` - H.264 with guidance audio
3. `final_video_original_audio_prores.mov` - ProRes 422 (for editing)

**Quality Settings:**
- H.264: CRF 18 (visually lossless), slow preset, 320kbps AAC audio
- ProRes 422: 10-bit 4:2:2 color, uncompressed PCM audio

## Pitch-Matching Video Recutting

Create videos that match the pitch sequence of a guide vocal by recutting source singing footage.

### Quick Start

**Step 1: Create guide sequence** from either a video or MIDI file:

*Option A: From video* (extracts pitch from singing):
```bash
./test_pitch_guide.sh data/input/guide_video.mp4

# This creates:
# - data/segments/guide_sequence.json (pitch data)
# - data/segments/guide_midi_preview.mp4 (verification video)
```

*Option B: From MIDI file* (uses exact note data):
```bash
# First, list available channels in the MIDI file
./test_midi_guide.sh data/input/melody.mid 0 --list-channels

# Then convert the desired channel (e.g., channel 1)
./test_midi_guide.sh data/input/melody.mid 1

# This creates:
# - data/segments/guide_sequence.json (pitch data)
# - data/temp/melody_midi_preview.wav (audio preview)
```

**Step 2: Build source video database** with pitch-indexed segments:
```bash
# Single video:
./test_pitch_source.sh data/input/source_video.mp4

# Batch process entire folder:
./batch_pitch_source.sh /path/to/videos/folder

# This creates:
# - data/segments/source_database.json (searchable pitch database)
# Contains pitch, timestamps, volume, and silence gaps
```

**Step 3: Match guide sequence to source database:**
```bash
./test_pitch_matcher.sh \
  data/segments/guide_sequence.json \
  data/segments/source_database.json

# This creates:
# - data/segments/match_plan.json (matching instructions for video assembly)
```

**Step 4: Assemble final video:**
```bash
./test_pitch_video_assembly.sh data/segments/match_plan.json

# This creates:
# - data/output/pitch_matched_video.mp4 (final assembled video)
```

### Pitch Detection Methods

Choose from four pitch detection algorithms:

**CREPE** (default, most accurate):
```bash
./test_pitch_guide.sh video.mp4 --pitch-method crepe
```
- Deep learning-based pitch detection
- Most accurate for monophonic singing
- Requires TensorFlow, slower but reliable

**SwiftF0** (fastest):
```bash
./test_pitch_guide.sh video.mp4 --pitch-method swift-f0
```
- CPU-optimized, very fast (132ms for 5s audio)
- Good accuracy, no GPU required
- May add spurious low bass notes

**Hybrid** (best of both):
```bash
./test_pitch_guide.sh video.mp4 --pitch-method hybrid
```
- Mixture-of-experts combining CREPE + SwiftF0
- Uses CREPE as primary, fills gaps with SwiftF0
- Filters SwiftF0 outliers (bass notes, pitch jumps)
- **Recommended for best results**

**Basic Pitch** (multipitch):
```bash
./test_pitch_guide.sh video.mp4 --pitch-method basic-pitch
```
- Spotify's multipitch detection
- Can detect harmonies
- 3x sub-semitone resolution

### Tuning Parameters

**Pitch Change Threshold** (cents):
```bash
# More sensitive (more segments)
./test_pitch_guide.sh video.mp4 --threshold 30

# Less sensitive (smoother, fewer segments)
./test_pitch_guide.sh video.mp4 --threshold 100
```
- Default: 50 cents
- Lower = splits on smaller pitch changes
- Higher = ignores vibrato/drift

**Silence Detection**:
```bash
# More permissive (catches quiet singing)
./test_pitch_guide.sh video.mp4 --silence-threshold -60

# More strict (treats quiet sounds as silence)
./test_pitch_guide.sh video.mp4 --silence-threshold -45
```
- Default: -50 dB
- Lower (more negative) = more permissive
- Helps with quiet consonants, soft singing

**Pitch Smoothing**:
```bash
# Reduce vibrato/waver
./test_pitch_guide.sh video.mp4 --pitch-smoothing 5

# Aggressive smoothing
./test_pitch_guide.sh video.mp4 --pitch-smoothing 7
```
- Default: 0 (off)
- Median filter window size: 5-7 recommended
- Smooths pitch curve before segmentation
- Higher values may miss quick note changes

**Minimum Duration**:
```bash
# Filter out very short notes
./test_pitch_guide.sh video.mp4 --min-duration 0.15
```
- Default: 0.1 seconds
- Filters brief pitch fluctuations
- Useful for cleaning up noisy detections

### Complete Example

```bash
# Best settings for wavy vocals
./test_pitch_guide.sh guide.mp4 \
  --pitch-method hybrid \
  --pitch-smoothing 5 \
  --silence-threshold -60 \
  --threshold 75 \
  --min-duration 0.12

# Review MIDI preview
open data/segments/guide_midi_preview.mp4
```

### How It Works

1. **Extract audio** from video
2. **Detect continuous pitch** using selected method
3. **Apply smoothing** (optional) to reduce vibrato
4. **Segment on changes**: Split when pitch changes >threshold OR silence detected
5. **Filter segments**: Remove very short segments
6. **Generate MIDI preview**: Create video with synthesized tones for verification

### Output Files

- `guide_sequence.json` - Pitch sequence data (time, Hz, MIDI note, confidence)
- `guide_midi_preview.mp4` - Video with MIDI playback for verification

### Using MIDI Files as Guide

Instead of extracting pitch from a video, you can use a MIDI file directly:

```bash
./test_midi_guide.sh data/input/melody.mid <channel> [options]
```

**Options:**
- `--list-channels` - Show available channels in the MIDI file
- `--min-rest N` - Minimum rest duration to preserve (default: 0.1s)
- `--sample-rate N` - Audio preview sample rate (default: 22050)
- `--no-audio` - Skip audio preview generation

**Examples:**
```bash
# List what's in the MIDI file
./test_midi_guide.sh song.mid 0 --list-channels

# Convert channel 1 with default settings
./test_midi_guide.sh song.mid 1

# Convert with smaller rest threshold (keeps more gaps)
./test_midi_guide.sh song.mid 1 --min-rest 0.05
```

**Advantages of MIDI input:**
- Exact pitch values (no detection errors)
- Perfect timing information
- Works with any melody you can create or export as MIDI

### Tips for Best Results

1. **Use hybrid mode** for most vocals - best accuracy with gap filling
2. **Watch the MIDI preview** - tones should match singing closely
3. **Adjust silence threshold** if too many/few gaps detected
4. **Use pitch smoothing (5-7)** for vibrato-heavy vocals
5. **Increase threshold** if you get too many micro-segments

### Source Video Database

After analyzing the guide video, build a searchable pitch database from source singing footage:

```bash
./test_pitch_source.sh data/input/source_video.mp4
```

**Combine multiple source videos:**
```bash
# Build database from first video
./test_pitch_source.sh video1.mp4

# Add more videos to the same database
./test_pitch_source.sh video2.mp4 --append
./test_pitch_source.sh video3.mp4 --append

# Now source_database.json contains clips from all three videos!
```

**All the same parameters work:**
```bash
./test_pitch_source.sh source.mp4 \
  --pitch-method hybrid \
  --pitch-smoothing 5 \
  --threshold 50 \
  --silence-threshold -50
```

**What it does:**
1. Extracts audio from source video
2. Detects all pitch segments using continuous tracking
3. Builds MIDI note index for fast lookup
4. Tracks silence gaps between segments
5. Calculates RMS volume for each segment
6. Saves comprehensive JSON database

**Database contains:**
- **pitch\_database**: Array of pitch segments with timing, Hz, MIDI note, confidence, volume, video frame numbers
- **silence\_segments**: Gaps between pitch segments (>50ms)
- **pitch\_index**: Fast lookup mapping MIDI note → segment IDs
- **Metadata**: Total segments, unique pitches, duration statistics

**Example database entry:**
```json
{
  "segment_id": 42,
  "start_time": 3.25,
  "end_time": 3.68,
  "duration": 0.43,
  "pitch_hz": 440.0,
  "pitch_midi": 69,
  "pitch_note": "A4",
  "pitch_confidence": 0.95,
  "rms_db": -18.5,
  "video_start_frame": 78,
  "video_end_frame": 88,
  "video_path": "/path/to/source_video.mp4"
}
```

**Usage tips:**
- **Combine multiple videos** for more variety and better pitch coverage
- Use same parameters as guide video for consistency
- Check statistics: ensure coverage of needed pitch range
- Each segment tracks which source video it came from

### Pitch Matching

Match the guide video's pitch sequence to your source database to create assembly instructions:

```bash
./test_pitch_matcher.sh \
  data/segments/guide_sequence.json \
  data/segments/source_database.json
```

**Matching Strategy:**
1. **Exact pitch match** (preferred) - finds source clips with same MIDI note
2. **Transposed match** - transposes source clips if no exact match available
3. **Missing pitches** - tracks pitches not found in source (to inform future videos)
4. **Duration handling** - trims, loops, or combines clips to match guide timing
5. **Smart scoring** - 60% duration match + 40% pitch confidence

**Reuse Policies:**

Control how source clips can be reused:

```bash
# Minimum gap between reuses (default, natural variety)
./test_pitch_matcher.sh guide.json source.json --reuse-policy min_gap

# No reuse (maximum variety, needs large source database)
./test_pitch_matcher.sh guide.json source.json --reuse-policy none

# Unlimited reuse (best matches, may be repetitive)
./test_pitch_matcher.sh guide.json source.json --reuse-policy allow

# Limited reuses (each clip max 3 times)
./test_pitch_matcher.sh guide.json source.json --reuse-policy limited --max-reuses 3

# Percentage limit (max 30% reuses)
./test_pitch_matcher.sh guide.json source.json --reuse-policy percentage --reuse-percentage 0.3
```

**Advanced Options:**

```bash
# Exact matches only, no transposition
./test_pitch_matcher.sh guide.json source.json --no-transposition

# Adjust scoring weights (favor duration or confidence)
./test_pitch_matcher.sh guide.json source.json \
  --duration-weight 0.7 \
  --confidence-weight 0.3

# Disable combining clips (only loop single clips)
./test_pitch_matcher.sh guide.json source.json --no-combine-clips

# Limit transposition range
./test_pitch_matcher.sh guide.json source.json --max-transpose 3
```

**Output (match\_plan.json):**
- Array of matches linking each guide segment to source clips
- Transposition amounts (if needed)
- Duration handling instructions (trim/loop/combine)
- Statistics: exact vs transposed matches, reuse counts
- Missing pitches list for expanding source database

**Match quality indicators:**
- `exact`: Perfect pitch match from source database
- `transposed`: Source clip transposed to target pitch
- `missing`: No suitable clip found (needs more source videos)

### Video Assembly

Assemble the final video from the match plan:

```bash
./test_pitch_video_assembly.sh data/segments/match_plan.json
```

**What it does:**
1. **Extracts video clips** from source videos based on frame numbers
2. **Extracts and transposes audio** using librosa pitch shifting
3. **Handles duration**:
  - Trims longer clips to match guide timing
  - Loops shorter clips to fill guide duration
  - Combines multiple clips when specified
4. **Concatenates clips** into final seamless video
5. **Outputs high-quality video** (H.264 CRF 18, AAC 320kbps)

**Options:**

```bash
# Custom output location
./test_pitch_video_assembly.sh match_plan.json --output videos/my_video.mp4

# Keep temporary files for debugging
./test_pitch_video_assembly.sh match_plan.json --no-cleanup

# Custom temporary directory
./test_pitch_video_assembly.sh match_plan.json --temp-dir /tmp/video_temp
```

**Output quality:**
- Video: H.264, CRF 18 (visually lossless), slow preset
- Audio: AAC 320kbps (high quality)
- Resolution: Matches source videos

**Process flow:**
```
For each match in match_plan:
  1. Extract video clip (start_frame → end_frame)
  2. Extract audio clip
  3. Transpose audio by N semitones (if needed)
  4. Combine video + transposed audio
  5. Trim/loop/combine as needed

Concatenate all clips → Final video
```

**Tips:**
- First run may take time (extracting and processing many clips)
- Use `--no-cleanup` to inspect individual clips if issues occur
- Check match plan statistics before assembly (missing matches will be skipped)
- Higher quality source videos = higher quality output

## Duration-Based Video Matching

Match guide segments to a folder of pre-existing video clips based on duration. Unlike pitch matching (which analyzes singing), this workflow uses individual video clip files and matches them by length.

### Quick Start

**Step 1: Build duration database** from a folder of video clips:
```bash
./test_duration_source.sh /path/to/video/clips

# Options:
#   --output PATH           Output JSON path (default: data/segments/duration_database.json)
#   --extensions mp4,mov    Video file extensions to include
#   --recursive             Search subdirectories
#   --append                Add to existing database
#   --min-duration 0.5      Skip clips shorter than N seconds
```

**Step 2: Create guide sequence** (use existing MIDI or pitch guide tools):
```bash
# From MIDI file:
./test_midi_guide.sh melody.mid 1

# OR from video:
./test_pitch_guide.sh guide_video.mp4
```

**Step 3: Match guide to clips by duration:**
```bash
./test_duration_matcher.sh \
  data/segments/guide_sequence.json \
  data/segments/duration_database.json \
  --crop-mode middle \
  --reuse-policy min_gap
```

**Step 4: Assemble final video:**
```bash
./test_duration_assembly.sh data/segments/duration_match_plan.json \
  --auto-resolution --auto-fps
```

### Crop Modes

Control how clips are trimmed to match guide segment durations:

- **start**: Use the first N seconds of each clip
- **middle**: Trim equally from start and end (centered crop)
- **end**: Use the last N seconds of each clip

```bash
# Examples:
./test_duration_matcher.sh guide.json source.json --crop-mode start
./test_duration_matcher.sh guide.json source.json --crop-mode middle  # default
./test_duration_matcher.sh guide.json source.json --crop-mode end
```

### Reuse Policies

Same policies as pitch matching:

```bash
# No reuse (each clip used once)
./test_duration_matcher.sh guide.json source.json --reuse-policy none

# Unlimited reuse
./test_duration_matcher.sh guide.json source.json --reuse-policy allow

# Minimum gap between reuses (default)
./test_duration_matcher.sh guide.json source.json --reuse-policy min_gap --min-reuse-gap 5

# Limited reuses per clip
./test_duration_matcher.sh guide.json source.json --reuse-policy limited --max-reuses 3

# Percentage limit
./test_duration_matcher.sh guide.json source.json --reuse-policy percentage --reuse-percentage 0.3
```

### How It Works

1. **Scan folder**: Extracts duration and metadata (resolution, fps, codec) from each clip
2. **Build sorted index**: Clips indexed by duration for efficient lookup
3. **Match by duration**: For each guide segment, finds shortest clip >= target duration
4. **Calculate crop frames**: Determines start/end frames based on crop mode
5. **Assemble video**: Extracts cropped clips and concatenates with normalization

### Output Files

- `data/segments/duration_database.json` - Clip metadata with durations
- `data/segments/duration_match_plan.json` - Matching instructions with crop frames
- `data/output/duration_matched_video.mp4` - Final assembled video

## Interactive Tools

### Onset Strength Visualizer

```bash
python src/interactive_strength_visualizer.py \
  --audio data/separated/htdemucs/song/other.wav \
  --strength data/output/onset_strength.json \
  --output data/output/visualizer.html \
  --threshold 0.2
```

**Features:**
- Audio playback synchronized with onset strength curve
- Adjustable threshold slider
- Real-time segment statistics
- Click timeline to seek

## Source Separation

### Recommended: gaudio (Highest Quality)

[gaudio](https://gaudiolab.com/) provides superior quality stems:
1. Upload audio to gaudiolab.com
2. Download separated stems
3. Place in `data/separated/gaudiolab/song/`

### Alternative: Demucs (Open Source)

```bash
demucs -n htdemucs data/input/song.mp3 -o data/separated
```

**For best results:** Use the `other` stem for irregular, artistic video cuts.

## Project Structure

```
video-hacking/
├── src/
│   ├── onset_strength_analysis.py      # Audio onset analysis
│   ├── audio_segmenter.py               # Cut audio into segments
│   ├── imagebind_audio_embedder.py      # Extract audio embeddings
│   ├── imagebind_video_embedder.py      # Extract video embeddings
│   ├── semantic_matcher.py              # Match audio to video
│   ├── video_assembler.py               # Assemble final videos
│   ├── pitch_guide_analyzer.py          # Analyze guide video pitch
│   ├── pitch_source_analyzer.py         # Build source pitch database
│   ├── pitch_matcher.py                 # Match guide to source
│   ├── pitch_video_assembler.py         # Assemble pitch-matched video
│   ├── duration_source_analyzer.py      # Build duration database from clips
│   ├── duration_matcher.py              # Match guide to clips by duration
│   ├── duration_video_assembler.py      # Assemble duration-matched video
│   └── interactive_strength_visualizer.py  # HTML visualizer
├── data/
│   ├── input/                           # Source files
│   ├── separated/                       # Audio stems
│   ├── segments/                        # Audio/video segments + embeddings
│   └── output/                          # Final videos + analysis
├── test_full_pipeline.sh                # Complete pipeline (Phases 1-3)
├── test_semantic_matching.sh            # Phase 4 testing
├── test_video_assembly.sh               # Phase 5 testing
├── test_onset_strength.sh               # Audio analysis testing
├── test_pitch_guide.sh                  # Analyze guide video pitch
├── test_pitch_source.sh                 # Build source pitch database
├── test_pitch_matcher.sh                # Match guide to source
├── test_pitch_video_assembly.sh         # Assemble pitch-matched video
├── test_duration_source.sh              # Build duration database from clips
├── test_duration_matcher.sh             # Match guide to clips by duration
├── test_duration_assembly.sh            # Assemble duration-matched video
├── install_imagebind.sh                 # ImageBind installer
├── fix_numpy.sh                         # NumPy version fixer
└── requirements.txt
```

## Complete Workflow Examples

### Audio-Driven Video Art

```bash
# Step 1: Activate environment
conda activate vh

# Step 2: Run complete pipeline (Phases 1-3)
./test_full_pipeline.sh \
  data/input/guidance_audio.wav \
  data/input/source_video.mp4

# Step 3: Review onset strength visualizer
open data/output/onset_strength_visualizer.html

# Step 4: Create semantic matches (try different policies)
./test_semantic_matching.sh none       # No reuse (max variety)
./test_semantic_matching.sh allow      # Unlimited reuse (best matches)
./test_semantic_matching.sh limited    # Limited reuse

# Step 5: Assemble final videos
./test_video_assembly.sh \
  data/input/source_video.mp4 \
  data/input/guidance_audio.wav

# Step 6: View results
open data/output/final_video_original_audio.mp4  # Original audio
open data/output/final_video.mp4                 # Guidance audio
```

### Pitch-Matching Video Recutting

```bash
# Step 1: Activate environment
conda activate vh

# Step 2: Analyze guide video (the target pitch sequence)
./test_pitch_guide.sh data/input/guide_video.mp4 \
  --pitch-method hybrid \
  --pitch-smoothing 5 \
  --silence-threshold -60

# Step 3: Review MIDI preview to verify pitch detection
open data/segments/guide_midi_preview.mp4

# Step 4: Build source database from singing footage
# Add multiple videos for more variety
./test_pitch_source.sh data/input/source1.mp4 \
  --pitch-method hybrid \
  --pitch-smoothing 5

./test_pitch_source.sh data/input/source2.mp4 --append
./test_pitch_source.sh data/input/source3.mp4 --append

# Step 5: Match guide to source database
./test_pitch_matcher.sh \
  data/segments/guide_sequence.json \
  data/segments/source_database.json \
  --reuse-policy min_gap

# Step 6: Assemble final video
./test_pitch_video_assembly.sh data/segments/match_plan.json

# Step 7: View result
open data/output/pitch_matched_video.mp4
```

## Output Files

### H.264 Versions (Smaller File Size)
- `final_video_original_audio.mp4` - Cut video with original audio
- `final_video.mp4` - Cut video with guidance audio

### ProRes 422 Version (High Quality for Editing)
- `final_video_original_audio_prores.mov` - 10-bit 4:2:2, PCM audio

## Tips for Best Results

1. **Use the "other" stem** for irregular, artistic cuts (not drums)
2. **Video length:** Use source video 10-20x longer than audio for variety
3. **Threshold tuning:** Use visualizer to find optimal cut density
4. **Reuse policy:**
  - Use `none` for maximum visual variety (requires long source video)
  - Use `allow` for best semantic matches (may repeat clips)
  - Use `limited` or `percentage` for balance
5. **Quality:** ProRes output is perfect for further color grading/editing

## How It Works

### Semantic Matching Process

1. **Audio Analysis:** Onset strength analysis identifies musical changes
2. **Segmentation:** Audio cut into segments at onset points
3. **Audio Embeddings:** Each audio segment → 1024-dim ImageBind embedding
4. **Video Analysis:** Sliding window extracts frames from entire video
5. **Video Embeddings:** Each video window → 1024-dim ImageBind embedding
6. **Matching:** Cosine similarity finds best video for each audio segment
7. **Assembly:** Video segments concatenated and synced with audio

### Why ImageBind?

ImageBind creates a unified embedding space for multiple modalities (audio, video, text, etc.). This means:
- Audio and video embeddings are directly comparable
- Semantically similar content has similar embeddings
- No need for bridging between CLAP (audio) and CLIP (video)

## Troubleshooting

### Out of Memory (Killed: 9)

If video embeddings extraction crashes with "Killed: 9", reduce chunk size:

```bash
# In test_full_pipeline.sh, add --chunk-size parameter (line ~111):
$PYTHON_CMD src/imagebind_video_embedder.py \
    --video "$VIDEO_FILE" \
    --output data/segments/video_embeddings.json \
    --fps 24 \
    --window-size 5 \
    --stride 6 \
    --batch-size 4 \
    --chunk-size 200 \  # Reduce from default 500
    --device auto

# Or run manually with smaller chunk:
python src/imagebind_video_embedder.py \
    --video path/to/video.mp4 \
    --output data/segments/video_embeddings.json \
    --chunk-size 200
```

**Chunk size guidelines:**
- Default 500: Works for most videos (8-16GB RAM)
- 200-300: For long videos or limited RAM (4-8GB)
- 100-150: For very long videos or 4GB RAM

### NumPy Version Errors

```bash
./fix_numpy.sh
# or manually:
pip uninstall -y numpy && pip install "numpy<2.0"
```

### ImageBind Installation

```bash
./install_imagebind.sh
# or manually:
pip install git+https://github.com/facebookresearch/ImageBind.git
```

### OpenCV Errors

```bash
pip uninstall -y opencv-python
pip install opencv-python==4.8.1.78
```

### GPU vs CPU

- **Onset analysis:** CPU is fine (fast enough)
- **Demucs separation:** GPU recommended (10-20x faster)
- **ImageBind embeddings:** GPU recommended (5-10x faster)
- **Video assembly:** CPU only (ffmpeg)

All scripts auto-detect GPU with `--device auto`

## Performance Notes

- **Audio embeddings:** ~0.5s per segment on CPU, ~0.1s on GPU
- **Video embeddings:** ~2-5s per 100 windows on GPU
- **Video assembly:** Depends on segment count and video codec
- **Full pipeline (30s audio, 10min video):** ~5-10 minutes total

## Dependencies

### Core
- **torch/torchaudio** (2.1.0): PyTorch for ImageBind
- **numpy** (<2.0): Numerical computing
- **opencv-python** (4.8.1.78): Video frame extraction
- **ffmpeg**: Video processing (external)

### Audio Analysis
- **librosa**: Onset detection
- **soundfile**: Audio I/O
- **demucs**: Source separation

### Pitch Detection
- **crepe**: Deep learning pitch detection (requires TensorFlow)
- **tensorflow** (<2.16.0): Required by CREPE
- **swift-f0**: Fast CPU-based pitch detection
- **basic-pitch**: Spotify's multipitch detection
- **scipy**: Signal processing (median filtering)

### Embeddings
- **imagebind**: Multimodal embeddings (auto-installed)
- **transformers** (<4.36.0): Required by ImageBind and Basic Pitch
- **Pillow**: Image processing

## Next Steps

### Audio-Driven Video Art (Complete)
- [x] Onset strength analysis
- [x] Audio segmentation
- [x] ImageBind audio embeddings
- [x] ImageBind video embeddings (sliding window)
- [x] Semantic matching with reuse policies
- [x] Video assembly with high quality output
- [ ] Real-time preview mode
- [ ] Batch processing for multiple videos
- [ ] Additional reuse strategies
- [ ] Color grading integration

### Pitch-Matching Video Recutting (Complete)
- [x] Multiple pitch detection methods (CREPE, SwiftF0, Basic Pitch)
- [x] Hybrid mixture-of-experts (CREPE + SwiftF0)
- [x] Pitch smoothing and silence detection
- [x] MIDI preview video generation
- [x] Configurable parameters (threshold, smoothing, silence)
- [x] Source video pitch analysis with searchable database
- [x] Volume tracking (RMS amplitude) per pitch segment
- [x] Silence gap detection and tracking
- [x] Pitch matching between guide and source
- [x] Smart scoring (duration + confidence weighting)
- [x] Pitch transposition for missing notes
- [x] Duration handling (trim, loop, combine clips)
- [x] Reuse policies (none, allow, min_gap, limited, percentage)
- [x] Final video assembly based on pitch matches
- [x] Audio pitch shifting with librosa
- [x] Clip extraction, trimming, looping, and concatenation

### Duration-Based Video Matching (Complete)
- [x] Folder scanning with video metadata extraction
- [x] Duration-sorted index for efficient matching
- [x] Three crop modes (start, middle, end)
- [x] Same reuse policies as other matchers
- [x] Parallel clip normalization and concatenation
- [x] Auto-detection of resolution and frame rate

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **ImageBind** by Meta AI Research
- **Demucs** by Alexandre Défossez
- **librosa** by Brian McFee
- **CREPE** by Jong Wook Kim et al.
- **SwiftF0** by lars76
- **Basic Pitch** by Spotify Research
