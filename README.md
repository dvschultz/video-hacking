# Video Hacking - Audio-Driven Video Art

A complete Python toolkit for creating audio-reactive video art by cutting and reassembling video footage based on semantic audio-visual matching using ImageBind embeddings.

## Overview

This project uses **ImageBind multimodal embeddings** to semantically match video clips to audio segments. It analyzes audio using onset strength detection, extracts embeddings from both audio and video, and reassembles the video based on semantic similarity.

## Features

- **Onset strength analysis**: Frame-accurate continuous onset strength values for precise cut points
- **Source separation**: Separate audio into stems using Demucs (drums, bass, vocals, other)
- **ImageBind embeddings**: Unified audio-visual embeddings for semantic matching
- **Semantic video matching**: Intelligently match video segments to audio based on content
- **Flexible reuse policies**: Control how video segments can be reused (none, allow, min_gap, limited, percentage)
- **High-quality output**: H.264 (CRF 18) and ProRes 422 outputs for editing
- **Interactive visualization**: Real-time threshold adjustment with playback

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
├── install_imagebind.sh                 # ImageBind installer
├── fix_numpy.sh                         # NumPy version fixer
└── requirements.txt
```

## Complete Workflow Example

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

### Embeddings
- **imagebind**: Multimodal embeddings (auto-installed)
- **transformers**: Required by ImageBind dependencies
- **Pillow**: Image processing

## Next Steps

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

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **ImageBind** by Meta AI Research
- **Demucs** by Alexandre Défossez
- **librosa** by Brian McFee
