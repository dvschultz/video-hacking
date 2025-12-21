# Pitch-Based Video Recutting Tool - Implementation Plan v2

## Project Goal

Create a tool that takes **two singing videos** and recreates the pitch sequence of video 1 by recutting clips from video 2 based on pitch matching.

**Inputs:**
- **Video 1 (Guide)**: Person A singing song X - defines target pitch sequence
- **Video 2 (Source)**: Person B singing song Y - provides clips to recut

**Output:**
- New video of Person B appearing to sing the melody from song X (by reordering their clips from song Y)

---

## High-Level Workflow

```
Video 1 (Guide) → [Onset Detection + Pitch Analysis] → Pitch Sequence (e.g., C4, D4, E4...)
                                                              ↓
Video 2 (Source) → [Onset Detection + Pitch Analysis] → Pitch Database (all pitches + timestamps)
                                                              ↓
                                          [Pitch Matcher: Find matching pitches in source]
                                                              ↓
                                          [Video Assembler: Recut source to match guide]
                                                              ↓
                                          Output: Source video recut to guide's melody
```

---

## Phase 1: Guide Video Analysis

**Goal:** Extract the pitch sequence from video 1 that we want to recreate

### 1.1 Audio Extraction
- Extract audio from video 1 using ffmpeg
- Sample rate: 22050 Hz (standard for pitch detection)

### 1.2 Onset Detection
- **Reuse existing code:** `onset_strength_analysis.py`
- Detect onsets (musical note beginnings) at frame-accurate precision
- Output: Array of onset times and strengths

### 1.3 Pitch Detection per Onset
- For each onset-to-onset segment, extract pitch information:
  - **Library options:**
    - **CREPE** (recommended): Deep learning, very accurate for singing
    - **librosa.pyin()**: Built-in, decent accuracy
  - Extract frame-by-frame pitch values for the segment
  - Calculate **median pitch** for the segment (Hz)
  - Convert to **MIDI note number** (0-127)
  - Calculate **confidence score** (how stable is the pitch?)
  - Store segment duration

### 1.4 Output Format

```json
{
  "video_path": "guide_video.mp4",
  "fps": 24,
  "audio_sample_rate": 22050,
  "guide_sequence": [
    {
      "index": 0,
      "start_time": 0.5,
      "end_time": 0.85,
      "duration": 0.35,
      "pitch_hz": 261.6,
      "pitch_midi": 60,
      "pitch_note": "C4",
      "pitch_confidence": 0.95,
      "onset_strength": 0.78
    },
    {
      "index": 1,
      "start_time": 0.85,
      "end_time": 1.2,
      "duration": 0.35,
      "pitch_hz": 293.7,
      "pitch_midi": 62,
      "pitch_note": "D4",
      "pitch_confidence": 0.92,
      "onset_strength": 0.65
    }
    // ... more segments
  ]
}
```

### Implementation File
`src/pitch_guide_analyzer.py`

---

## Phase 2: Source Video Analysis

**Goal:** Build a searchable database of ALL pitches available in video 2

### 2.1 Audio Extraction
- Extract audio from video 2 (same settings as guide)

### 2.2 Comprehensive Onset Detection
- Detect ALL onsets in source video (not just matching guide)
- More granular detection to capture every possible note/syllable
- Lower threshold to catch more potential clips

### 2.3 Pitch Database Construction
- For each detected onset segment:
  - Extract median pitch (Hz + MIDI)
  - Store start/end times in both audio and video
  - Store frame numbers for precise video cutting
  - Calculate duration and confidence
  - Index by MIDI note for fast lookup

### 2.4 Output Format

```json
{
  "video_path": "source_video.mp4",
  "fps": 24,
  "audio_sample_rate": 22050,
  "pitch_database": [
    {
      "clip_id": 0,
      "start_time": 1.25,
      "end_time": 1.58,
      "duration": 0.33,
      "pitch_hz": 261.8,
      "pitch_midi": 60,
      "pitch_note": "C4",
      "pitch_confidence": 0.89,
      "video_start_frame": 30,
      "video_end_frame": 38,
      "onset_strength": 0.71
    },
    {
      "clip_id": 1,
      "start_time": 1.58,
      "end_time": 1.95,
      "duration": 0.37,
      "pitch_hz": 329.6,
      "pitch_midi": 64,
      "pitch_note": "E4",
      "pitch_confidence": 0.94,
      "video_start_frame": 38,
      "video_end_frame": 47,
      "onset_strength": 0.82
    }
    // ... all detected segments
  ],
  "pitch_index": {
    "60": [0, 45, 89, 123],  // Clip IDs with MIDI note 60 (C4)
    "62": [12, 56, 78],       // D4
    "64": [1, 34, 67, 99]     // E4
    // ... all MIDI notes found
  }
}
```

### Implementation File
`src/pitch_source_analyzer.py`

---

## Phase 3: Pitch Matching

**Goal:** For each pitch in the guide sequence, find the best matching clip from source database

### 3.1 Matching Algorithm

**Strategy 1: Exact MIDI Match (Priority)**
1. For guide segment with MIDI note N:
2. Look up all source clips with MIDI note N
3. Rank by:
   - Pitch confidence (higher is better)
   - Duration similarity to guide segment
   - Reuse count (prefer unused clips)

**Strategy 2: Nearest Pitch Match (Fallback)**
- If no exact MIDI match found:
  - Search MIDI notes N±1 (adjacent semitones)
  - Search MIDI notes N±2 (if still no match)
  - Calculate pitch distance in cents (100 cents = 1 semitone)
  - Prefer matches within ±50 cents

**Strategy 3: Duration Weighting**
- Combined score: `pitch_match * duration_similarity`
- Duration similarity: `1 - abs(source_duration - guide_duration) / guide_duration`
- Prefer clips that are close in length to avoid stretching

### 3.2 Reuse Policies

Control how often source clips can be reused:

- **`none`**: Each source clip used max once (requires very long source video)
- **`allow`**: Unlimited reuse (best pitch matches, may be repetitive)
- **`min_gap`**: Minimum 5 seconds between reuses of same clip
- **`limited`**: Each clip reused max 3 times total
- **`percentage`**: Max 30% of final video can be reused clips

### 3.3 Output Format

```json
{
  "guide_video": "guide_video.mp4",
  "source_video": "source_video.mp4",
  "reuse_policy": "min_gap",
  "matches": [
    {
      "guide_index": 0,
      "guide_pitch_midi": 60,
      "guide_duration": 0.35,
      "source_clip_id": 45,
      "source_pitch_midi": 60,
      "pitch_distance_cents": 8.2,
      "duration_ratio": 0.94,
      "match_score": 0.91,
      "source_start_time": 12.5,
      "source_end_time": 12.83,
      "source_start_frame": 300,
      "source_end_frame": 308,
      "reuse_count": 0
    }
    // ... one match per guide segment
  ],
  "stats": {
    "total_segments": 145,
    "exact_matches": 132,
    "semitone_matches": 11,
    "no_match_found": 2,
    "avg_pitch_distance": 12.3,
    "clips_reused": 23
  }
}
```

### Implementation File
`src/pitch_matcher.py`

---

## Phase 4: Video Assembly

**Goal:** Cut and reassemble source video based on matched pitch sequence

### 4.1 Clip Extraction

For each match:
1. Extract video segment from source using ffmpeg
2. Handle duration mismatches:
   - **Option A (Simple)**: Trim to exact guide duration
   - **Option B (Advanced)**: Time-stretch audio/video to match (±20% max)
3. Save temporary clip files

### 4.2 Concatenation

Use ffmpeg concat filter to join all clips:
```bash
ffmpeg -f concat -safe 0 -i filelist.txt -c copy output.mp4
```

### 4.3 Audio Replacement Options

**Option 1: Source Audio (Reconstructed Melody)**
- Use the audio from matched source clips
- Result: Source person singing guide melody (pitch-matched reconstruction)

**Option 2: Guide Audio (Original)**
- Replace entire audio track with guide video's audio
- Result: Source person's video synced to guide's original singing

**Option 3: Both Versions**
- Generate both outputs for comparison

### 4.4 Output Files

1. `pitch_matched_source_audio.mp4` - Source clips with their own audio (reconstructed melody)
2. `pitch_matched_guide_audio.mp4` - Source clips with guide audio overlaid
3. `pitch_matched_source_audio_prores.mov` - ProRes 422 (editing-friendly)

**Quality Settings:**
- H.264: CRF 18, slow preset
- Audio: 320kbps AAC or uncompressed PCM
- ProRes: 422 10-bit for professional editing

### Implementation File
`src/pitch_video_assembler.py`

---

## Phase 5: Visualization & Debugging

### 5.1 Pitch Contour Visualizer

Interactive HTML tool showing:
- Guide pitch sequence (line graph)
- Matched source pitches overlaid
- Pitch distance errors highlighted
- Audio playback synchronized with visualization
- Click to jump to problematic segments

### 5.2 Match Quality Report

Generate statistics:
- Percentage of exact pitch matches
- Average pitch error (cents)
- Duration match accuracy
- Reuse statistics
- Segments with low confidence
- Suggested threshold adjustments

### Implementation File
`src/pitch_visualizer.py`

---

## Code Reuse from Existing Repository

### Files to Reuse Directly
1. **`onset_strength_analysis.py`** - Core onset detection (lines 1-150)
   - Reuse `OnsetStrengthAnalyzer` class
   - Already handles frame-accurate onset detection

2. **`audio_segmenter.py`** - Audio segmentation logic
   - Reuse segment cutting at onset times
   - Modify to include pitch extraction per segment

3. **`video_assembler.py`** - Video concatenation logic
   - Reuse ffmpeg concat approach
   - Reuse quality settings (CRF 18, ProRes options)

### Files to Create New
1. **`pitch_guide_analyzer.py`** - Guide video pitch analysis
2. **`pitch_source_analyzer.py`** - Source database builder
3. **`pitch_matcher.py`** - Pitch matching algorithm
4. **`pitch_video_assembler.py`** - Modified assembler for pitch-based cuts
5. **`pitch_visualizer.py`** - Pitch contour visualization

### Shared Utilities
Create `src/pitch_utils.py` for:
- Hz to MIDI conversion
- MIDI to note name conversion
- Pitch distance calculation (cents)
- Pitch confidence scoring

---

## Dependencies

### Required New Libraries

```txt
# Pitch detection
crepe>=0.0.12                 # ML-based pitch tracking (recommended)

# OR use librosa's built-in pyin
# (librosa>=0.10.0 already installed)

# Optional: Time stretching
pyrubberband>=0.3.0           # For duration adjustment
```

### Already Installed (from existing project)
- librosa (onset detection, pitch detection)
- numpy (array processing)
- soundfile (audio I/O)
- ffmpeg (video/audio processing)
- opencv-python (if needed for video frames)

---

## Project Structure

```
video-hacking/
├── src/
│   ├── onset_strength_analysis.py      # [REUSE] Onset detection
│   ├── audio_segmenter.py               # [REUSE] Audio cutting
│   ├── video_assembler.py               # [REUSE] Video concatenation
│   ├── pitch_guide_analyzer.py          # [NEW] Phase 1
│   ├── pitch_source_analyzer.py         # [NEW] Phase 2
│   ├── pitch_matcher.py                 # [NEW] Phase 3
│   ├── pitch_video_assembler.py         # [NEW] Phase 4
│   ├── pitch_visualizer.py              # [NEW] Phase 5
│   └── pitch_utils.py                   # [NEW] Shared utilities
├── scripts/
│   ├── test_pitch_pipeline.sh           # Complete pipeline
│   ├── test_pitch_analysis.sh           # Test pitch detection
│   └── test_pitch_matching.sh           # Test matching only
└── PITCH_TOOL_README.md                 # User documentation
```

---

## Implementation Order

### Sprint 1: Core Pitch Detection (Days 1-2)
- [ ] Install CREPE library
- [ ] Create `pitch_utils.py` with conversion functions
- [ ] Implement `pitch_guide_analyzer.py`
  - Integrate onset detection from existing code
  - Add CREPE pitch detection per segment
  - Export guide sequence JSON
- [ ] Test with simple singing video
- [ ] Verify pitch accuracy (compare to ground truth)

### Sprint 2: Source Database (Days 3-4)
- [ ] Implement `pitch_source_analyzer.py`
  - Reuse onset detection logic
  - Build comprehensive pitch database
  - Create MIDI note index for fast lookup
- [ ] Test with longer source video (5-10 minutes)
- [ ] Verify database completeness (all pitches detected)

### Sprint 3: Matching Algorithm (Days 5-6)
- [ ] Implement `pitch_matcher.py`
  - Exact MIDI matching
  - Nearest pitch fallback
  - Duration weighting
  - Reuse policy enforcement
- [ ] Test different reuse policies
- [ ] Generate match statistics
- [ ] Tune matching thresholds

### Sprint 4: Video Assembly (Days 7-8)
- [ ] Implement `pitch_video_assembler.py`
  - Modify `video_assembler.py` for pitch-based cuts
  - Clip extraction and trimming
  - Concatenation with both audio options
- [ ] Generate first complete pitch-matched video
- [ ] Compare source audio vs guide audio versions
- [ ] Test with different video durations

### Sprint 5: Visualization & Polish (Days 9-10)
- [ ] Implement `pitch_visualizer.py`
  - Pitch contour graph (guide vs matched)
  - Interactive playback
  - Error highlighting
- [ ] Create shell scripts for pipeline automation
- [ ] Write comprehensive README
- [ ] Add example videos and test cases
- [ ] Performance optimization

---

## Key Technical Decisions

### 1. Pitch Detection Library
**Decision: CREPE**
- Rationale: Most accurate for singing (deep learning-based)
- Fallback: librosa.pyin() if CREPE installation fails
- Performance: ~0.3s per second of audio (acceptable)

### 2. Onset Detection Parameters
**Decision: Reuse existing onset_strength_analysis.py settings**
- FPS: 24 (standard video)
- Power: 0.6 (balanced sensitivity)
- Threshold: 0.2 (adjustable per video)

### 3. Matching Strategy Priority
**Decision: Exact MIDI > Nearest Pitch > Duration Weighted**
1. Try exact MIDI match first (preferred)
2. Fallback to ±1 semitone if needed
3. Weight by duration for final ranking

### 4. Duration Handling
**Decision: Phase 1 = Simple trim, Phase 2 = Time-stretch**
- v1: Trim clips to exact duration (simpler, no artifacts)
- v2: Add time-stretching for ±20% duration mismatches
- Avoid pitch-shifting (would alter the detected pitch)

### 5. Default Reuse Policy
**Decision: `min_gap` (5 seconds)**
- Balances pitch accuracy with visual variety
- Prevents jarring back-to-back repetitions
- User can override for `allow` or `none`

---

## Testing Strategy

### Test Videos Needed
1. **Guide video**: 30-60 seconds, simple melody (e.g., "Twinkle Twinkle")
2. **Source video**: 5-10 minutes, different song, same singer if possible
3. Format: MP4, 24fps, clear audio (no background noise)

### Success Metrics
- **Pitch accuracy**: >85% exact MIDI matches, >95% within ±1 semitone
- **Temporal alignment**: Onset timing within ±100ms
- **Visual quality**: No visible encoding artifacts
- **Audio quality**: No clicks, pops, or volume jumps

### Debugging Workflow
1. Visualize guide pitch sequence (are onsets detected correctly?)
2. Check source database coverage (does it have all needed pitches?)
3. Review match quality report (where are the errors?)
4. Watch output video (does it look/sound natural?)
5. Iterate on threshold/parameters

---

## Open Questions & Design Choices

### Q1: What if source database doesn't have a needed pitch?
**Options:**
- A) Use nearest available pitch (±1-2 semitones)
- B) Insert silence/black frame
- C) Pitch-shift nearest clip (adds artifacts)
- **Recommendation: A, with warning logged**

### Q2: How to handle very short notes (<0.1s)?
**Options:**
- A) Extend to minimum duration (0.15s)
- B) Merge with adjacent note
- C) Skip entirely
- **Recommendation: A, with crossfade**

### Q3: Should we preserve vibrato/pitch bends?
**Options:**
- A) Use median pitch only (simpler)
- B) Match full pitch contour (complex)
- **Recommendation: A for v1, B for v2**

### Q4: Audio output preference?
**Options:**
- A) Source audio (reconstructed melody)
- B) Guide audio (original performance)
- C) Both versions
- **Recommendation: C (generate both)**

### Q5: Transition style between clips?
**Options:**
- A) Hard cuts (current approach)
- B) Short crossfades (50-100ms)
- **Recommendation: A for v1, add B as option**

---

## Risk Mitigation

### Risk 1: Pitch detection errors
- **Mitigation**: Use CREPE (most accurate), filter low-confidence segments
- **Fallback**: Manual pitch correction JSON editing

### Risk 2: Insufficient source coverage
- **Mitigation**: Require source video 10-20x longer than guide
- **Fallback**: Allow semitone mismatches, log warnings

### Risk 3: Timing/sync issues
- **Mitigation**: Frame-accurate onset detection, manual threshold tuning
- **Fallback**: Visualizer tool for debugging

### Risk 4: Visual discontinuity
- **Mitigation**: Prefer clips with similar duration, add crossfade option
- **Fallback**: User can manually exclude jarring clips

---

## Future Enhancements (Post-MVP)

1. **Facial expression matching** (ImageBind embeddings)
   - Combine pitch match + visual coherence score
   - Prefer clips where mouth shape roughly matches

2. **Pitch correction/auto-tune**
   - Fine-tune source clips to exact guide pitch
   - Use rubberband or librosa pitch_shift

3. **Vibrato preservation**
   - Match full pitch contour, not just median
   - Use DTW (Dynamic Time Warping) for alignment

4. **Real-time preview**
   - Web interface for adjusting thresholds
   - Instant re-matching and preview

5. **Batch processing**
   - Process multiple guide/source pairs
   - Generate comparison videos automatically

---

## Documentation Deliverables

1. **PITCH_TOOL_README.md** - User guide with examples
2. **Code comments** - Docstrings for all functions
3. **Example scripts** - Shell scripts for common workflows
4. **Test cases** - Sample videos and expected outputs
5. **Troubleshooting guide** - Common errors and fixes

---

## Success Criteria

**Minimum Viable Product (MVP):**
- ✅ Accurately detect pitches in guide video
- ✅ Build searchable database from source video
- ✅ Match >85% of guide pitches exactly
- ✅ Generate playable output video with both audio options
- ✅ Complete pipeline script (one command to run all phases)

**Stretch Goals:**
- ✅ Interactive visualizer for pitch contours
- ✅ Time-stretching for duration matching
- ✅ Multiple reuse policies
- ✅ Comprehensive match quality report

---

## Next Steps

1. **Review this plan** - Confirm approach and priorities
2. **Prepare test videos** - Find/create guide and source singing videos
3. **Install dependencies** - Add CREPE to requirements.txt
4. **Start Sprint 1** - Implement pitch_guide_analyzer.py
5. **Iterate and test** - Validate pitch detection before moving to matching

---

## Estimated Effort

- **Core implementation (Sprints 1-4)**: 8-10 days
- **Visualization & polish (Sprint 5)**: 2-3 days
- **Testing & refinement**: 2-3 days
- **Documentation**: 1-2 days

**Total: ~2 weeks for complete, polished tool**

---

**End of Plan**
