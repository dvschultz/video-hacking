# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Audio-driven video art toolkit that uses **ImageBind multimodal embeddings** for semantic audio-visual matching and **pitch detection** for musical video recutting. Three main workflows:

1. **Semantic Matching**: Cuts video based on audio onset strength, matches clips semantically using ImageBind
2. **Pitch Matching**: Recuts singing footage by matching pitch sequences between guide and source videos
3. **Duration Matching**: Matches guide segments to a folder of video clips based on duration

## Common Commands

### Environment Setup
```bash
conda activate vh                    # Activate the project environment
pip install -r requirements.txt      # Install dependencies
./install_imagebind.sh              # Install ImageBind (auto-installs on first use)
./fix_numpy.sh                      # Fix NumPy version if errors occur
```

### Audio-Driven Video Art Pipeline
```bash
# Complete pipeline (Phases 1-3): separation, onset analysis, embeddings
./test_full_pipeline.sh <audio.wav> <video.mp4>

# Phase 4: Semantic matching (reuse policy: none|allow|min_gap|limited|percentage)
./test_semantic_matching.sh allow

# Phase 5: Video assembly
./test_video_assembly.sh <video.mp4> <audio.wav>
```

### Pitch Matching Pipeline
```bash
# Step 1a: Analyze guide video pitch sequence
# --pitch-method: crepe, rmvpe (fast), rmvpe-crepe, hybrid, swift-f0, basic-pitch, pyin
./test_pitch_guide.sh <guide_video.mp4> [--pitch-method rmvpe] [--pitch-smoothing 5]

# Step 1b: OR use MIDI file as guide (alternative to video)
./test_midi_guide.sh <melody.mid> <channel> [--min-rest 0.1] [--list-channels]

# Step 2: Build source video pitch database
./test_pitch_source.sh <source_video.mp4> [--append]  # --append adds to existing DB
./batch_pitch_source.sh <folder_path> [--extensions mp4,mov] [--dry-run]  # Batch process folder

# Step 3: Match guide to source
./test_pitch_matcher.sh data/segments/guide_sequence.json data/segments/source_database.json

# Step 4: Assemble final video
./test_pitch_video_assembly.sh data/segments/match_plan.json
```

### Duration Matching Pipeline
```bash
# Step 1: Build duration database from folder of video clips
./test_duration_source.sh <clips_folder> [--extensions mp4,mov] [--append]

# Step 2: Create guide sequence (use existing MIDI or pitch guide tools)
./test_midi_guide.sh <melody.mid> <channel>
# OR
./test_pitch_guide.sh <guide_video.mp4>

# Step 3: Match guide to clips by duration
# --crop-mode: start, middle, end (default: middle)
# --match-rests: match rest segments with clips instead of black frames
./test_duration_matcher.sh data/segments/guide_sequence.json data/segments/duration_database.json [--crop-mode middle] [--match-rests]

# Step 4: Assemble final video
./test_duration_assembly.sh data/segments/duration_match_plan.json [--auto-resolution] [--auto-fps]
```

### Running Individual Components
```bash
# Onset strength analysis
python src/onset_strength_analysis.py --audio <audio.wav> --output <output.json> --fps 24 --threshold 0.2

# Audio segmentation
python src/audio_segmenter.py --audio <audio.wav> --onset-strength <onset.json> --output-dir data/segments/audio

# ImageBind audio embeddings
python src/imagebind_audio_embedder.py --segments-metadata <segments.json> --segments-dir data/segments/audio --output <embeddings.json>

# ImageBind video embeddings (reduce --chunk-size if OOM)
python src/imagebind_video_embedder.py --video <video.mp4> --output <embeddings.json> --chunk-size 500

# Semantic matching
python src/semantic_matcher.py --audio-embeddings <audio.json> --video-embeddings <video.json> --audio-segments <segments.json> --output <matches.json>
```

## Architecture

### Core Modules

**Audio Analysis** (`src/onset_strength_analysis.py`, `src/audio_segmenter.py`):
- `OnsetStrengthAnalyzer` class generates frame-accurate onset strength values using librosa
- Key params: `--fps` (frame rate), `--power` (compression), `--threshold` (cut points)

**Pitch Detection** (`src/pitch_guide_analyzer.py`, `src/pitch_source_analyzer.py`, `src/pitch_change_detector.py`):
- Multiple pitch detection methods available via `--pitch-method`:
  - `crepe` (default): Deep learning, most accurate, requires TensorFlow
  - `rmvpe`: Fast vocal pitch estimation, auto-downloads model on first use (~180MB)
  - `rmvpe-crepe`: Hybrid combining RMVPE (primary) + CREPE (validation/gap-filling)
  - `hybrid`: CREPE + SwiftF0 mixture-of-experts
  - `swift-f0`: Fast CPU-optimized detection
  - `basic-pitch`: Spotify's multipitch detection
  - `pyin`: Librosa's probabilistic YIN (fallback)
- `PitchChangeDetector` segments audio on pitch changes + silence
- Output: JSON with pitch sequences indexed by MIDI note number

**MIDI Guide** (`src/midi_guide_converter.py`):
- Alternative to video-based pitch extraction - uses MIDI files directly
- Parses MIDI with tempo handling, extracts notes from specified channel
- Generates audio preview for verification
- Output: Same JSON format as pitch_guide_analyzer.py

**Embeddings** (`src/imagebind_audio_embedder.py`, `src/imagebind_video_embedder.py`):
- Extract 1024-dim ImageBind embeddings for audio segments and video windows
- Video uses sliding window with configurable stride and chunk-size for memory management

**Matching** (`src/semantic_matcher.py`, `src/pitch_matcher.py`, `src/duration_matcher.py`):
- Semantic: cosine similarity between audio/video embeddings
- Pitch: exact MIDI match, nearest pitch fallback, duration-weighted scoring
- Duration: matches guide segments to clips by finding shortest clip >= target duration
- Reuse policies control clip repetition: `none`, `allow`, `min_gap`, `limited`, `percentage`

**Duration Matching** (`src/duration_source_analyzer.py`, `src/duration_matcher.py`, `src/duration_video_assembler.py`):
- Scans folder of video clips, catalogs duration and video metadata
- Matches guide segments to nearest longer clip, crops to exact length
- Three crop modes: `start` (use first N seconds), `middle` (centered), `end` (use last N seconds)

**Assembly** (`src/video_assembler.py`, `src/pitch_video_assembler.py`):
- FFmpeg-based concatenation with H.264 (CRF 18) and ProRes 422 outputs
- Pitch assembler handles audio transposition via librosa pitch_shift

### Data Flow

```
Audio → Demucs separation → Onset analysis → Audio segments → ImageBind embeddings
                                                                    ↓
Video → Frame extraction → Sliding window → ImageBind embeddings → Matching → Assembly
```

For pitch matching:
```
Guide video → Pitch detection → Pitch sequence (target)
Source video → Pitch detection → Pitch database (searchable by MIDI note)
                                        ↓
Pitch matching → Match plan → Video assembly with transposition
```

For duration matching:
```
Folder of clips → Duration analyzer → Duration database (sorted by duration)
Guide sequence → Duration matcher → Match plan with crop instructions
                                         ↓
                              Video assembly with cropping
```

### Key Data Files

- `data/segments/audio_segments.json` - Audio segment metadata
- `data/segments/audio_embeddings.json` - ImageBind audio embeddings
- `data/segments/video_embeddings.json` - ImageBind video embeddings
- `data/segments/matches.json` - Semantic match results
- `data/segments/guide_sequence.json` - Guide video pitch sequence
- `data/segments/source_database.json` - Source pitch database with MIDI index
- `data/segments/match_plan.json` - Pitch matching instructions
- `data/segments/duration_database.json` - Duration database with clip metadata
- `data/segments/duration_match_plan.json` - Duration matching instructions with crop frames

## Dependencies

- PyTorch 2.1.0 with NumPy <2.0 (compatibility requirement)
- TensorFlow <2.16.0 for CREPE pitch detection
- ImageBind installed from GitHub (auto-installs on first use)
- FFmpeg for video processing
- GPU recommended for embeddings (auto-detected with `--device auto`)

## Memory Management

For long videos, reduce `--chunk-size` in video embedder (default 500, try 200 for limited RAM).
