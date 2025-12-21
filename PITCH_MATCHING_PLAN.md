# Pitch-Based Video Matching - Implementation Plan

## Project Overview

A tool that analyzes the pitch contour of a "guide" singing video and reconstructs it by cutting and reassembling clips from a "source" singing video, matching pitches note-by-note.

**Input:**
- Guide video: Person singing song A (defines the target pitch sequence)
- Source video: Person singing song B (provides the video clips to reassemble)

**Output:**
- New video: Source person appears to sing the guide song's melody

---

## Phase 1: Pitch Analysis & Onset Detection

### 1.1 Guide Video Analysis

**Goal:** Extract pitch contour aligned with onset events

**Implementation:**
```python
# src/pitch_onset_analyzer.py
```

**Steps:**
1. Extract audio from guide video (ffmpeg)
2. Detect onsets using librosa (similar to current onset_strength_analysis.py)
3. Extract pitch for each frame using one of:
   - **CREPE** (recommended): Deep learning pitch tracker, very accurate
   - **pYIN**: Probabilistic YIN, good for singing
   - **librosa.pyin()**: Built-in, decent quality
4. Segment audio into "notes" based on onsets
5. For each note segment:
   - Calculate median/mean pitch
   - Store pitch in Hz and MIDI note number
   - Store onset time, duration, and offset time
   - Detect pitch stability/vibrato characteristics

**Output Format:**
```json
{
  "video_path": "guide.mp4",
  "fps": 24,
  "segments": [
    {
      "index": 0,
      "start_time": 0.5,
      "end_time": 0.8,
      "duration": 0.3,
      "pitch_hz": 440.0,
      "pitch_midi": 69,
      "pitch_note": "A4",
      "pitch_confidence": 0.95,
      "pitch_stability": 0.88,
      "onset_strength": 0.7
    },
    ...
  ]
}
```

**Dependencies:**
- `crepe` or `librosa` for pitch detection
- `librosa` for onset detection
- Existing onset_strength_analysis.py as reference

---

### 1.2 Source Video Analysis

**Goal:** Build a searchable database of all pitches in source video

**Implementation:**
```python
# src/pitch_database_builder.py
```

**Steps:**
1. Extract audio from source video
2. Detect ALL onsets (regardless of guide video)
3. Extract pitch for every frame
4. Create sliding window analysis:
   - For each onset, create a segment
   - Calculate median pitch for segment
   - Store start/end times in video
5. Build pitch database with all available clips

**Output Format:**
```json
{
  "video_path": "source.mp4",
  "fps": 24,
  "database": [
    {
      "clip_id": 0,
      "start_time": 1.2,
      "end_time": 1.5,
      "duration": 0.3,
      "pitch_hz": 392.0,
      "pitch_midi": 67,
      "pitch_note": "G4",
      "pitch_confidence": 0.92,
      "video_start_frame": 28,
      "video_end_frame": 36
    },
    ...
  ]
}
```

---

## Phase 2: Pitch Matching

### 2.1 Pitch Matching Algorithm

**Goal:** Find best source clip for each guide note

**Implementation:**
```python
# src/pitch_matcher.py
```

**Matching Strategies:**

1. **Exact Pitch Match** (strict)
   - Match MIDI note number exactly
   - Fallback: ±1 semitone if no exact match

2. **Closest Pitch Match** (flexible)
   - Find clip with minimum pitch distance in Hz
   - Prefer clips within ±50 cents (half semitone)

3. **Duration-Weighted Match**
   - Match pitch AND similar duration
   - Score = pitch_similarity * duration_similarity

4. **Pitch Contour Match** (advanced)
   - Match pitch trajectory over the note
   - Good for vibrato/pitch bends

**Reuse Policies:**
- `none`: Each source clip used once (requires long source video)
- `allow`: Unlimited reuse (best pitch matches)
- `min_gap`: Minimum N seconds between reuses
- `limited`: Each clip reused max M times

**Output Format:**
```json
{
  "guide_video": "guide.mp4",
  "source_video": "source.mp4",
  "matches": [
    {
      "guide_segment_idx": 0,
      "source_clip_id": 45,
      "pitch_distance_cents": 5.2,
      "duration_ratio": 0.95,
      "confidence_score": 0.88,
      "source_start_time": 12.5,
      "source_end_time": 12.8,
      "guide_target_duration": 0.3
    },
    ...
  ]
}
```

---

## Phase 3: Video Assembly

### 3.1 Time-Stretching & Pitch-Correction

**Challenge:** Source clips may not match guide duration exactly

**Options:**

1. **Simple Cut** (no stretching)
   - Just trim source clip to match guide duration
   - Pros: Simple, maintains quality
   - Cons: May cut off notes awkwardly

2. **Time-Stretch without Pitch Change**
   - Use ffmpeg atempo or rubberband
   - Stretch/compress clip to match duration
   - Keep original pitch
   - Pros: Better timing match
   - Cons: Slight artifacts if stretch >20%

3. **Pitch-Shift to Exact Match** (optional enhancement)
   - Use rubberband or ffmpeg to pitch-shift
   - Ensure source matches guide pitch exactly
   - Pros: Perfect pitch matching
   - Cons: More artifacts, may sound unnatural

**Recommended:** Start with option 1 (simple cut), add option 2 for Phase 2

---

### 3.2 Video Reassembly

**Implementation:**
```python
# src/pitch_video_assembler.py
```

**Steps:**
1. For each match:
   - Extract video clip from source video
   - Trim to target duration (or time-stretch)
   - Save to temp file
2. Concatenate all clips using ffmpeg
3. Replace audio with:
   - **Option A:** Guide audio (they sing guide melody)
   - **Option B:** Source clips' audio (pitch-matched reconstruction)
   - **Option C:** Both versions

**Quality Settings:**
- Same as current system: CRF 18, slow preset
- ProRes 422 option for editing

---

## Phase 4: Enhancements (Optional)

### 4.1 Transition Smoothing
- Add crossfades between clips (50-100ms)
- Audio: Crossfade to avoid clicks
- Video: Crossfade or cut to beat

### 4.2 Pitch Correction
- Use Auto-Tune style pitch correction
- Shift source clips to exact guide pitch
- Library: `pyrubberband` or `soundstretch`

### 4.3 Vibrato/Expression Matching
- Match not just pitch but pitch contour
- Preserve vibrato characteristics
- Use DTW (Dynamic Time Warping) for matching

### 4.4 Visual Coherence
- Facial expression matching (ImageBind embeddings)
- Ensure mouth movements roughly match pitch
- Combined score: pitch_match * visual_match

---

## Implementation Order

### Sprint 1: Core Pitch Analysis
1. ✅ Install dependencies (CREPE, librosa pitch)
2. ✅ Implement pitch_onset_analyzer.py for guide video
3. ✅ Implement pitch_database_builder.py for source video
4. ✅ Test with simple singing clips
5. ✅ Verify pitch detection accuracy

### Sprint 2: Matching & Assembly
1. ✅ Implement pitch_matcher.py
2. ✅ Test different matching strategies
3. ✅ Implement pitch_video_assembler.py (simple cut)
4. ✅ Create test script for full pipeline
5. ✅ Generate first pitch-matched video

### Sprint 3: Quality Improvements
1. ✅ Add time-stretching support
2. ✅ Implement reuse policies
3. ✅ Add visualizer for pitch contours
4. ✅ Tune matching parameters
5. ✅ Documentation and examples

### Sprint 4: Advanced Features (Optional)
1. ⬜ Pitch correction/shifting
2. ⬜ Transition smoothing
3. ⬜ Expression matching
4. ⬜ Real-time preview

---

## Technical Stack

### New Dependencies
```txt
# Pitch detection
crepe>=0.0.12              # Deep learning pitch tracker (recommended)
# OR
librosa>=0.10.0            # Has pyin built-in

# Time stretching (optional, for Phase 3.1)
pyrubberband>=0.3.0        # Rubberband time/pitch manipulation
```

### Existing Dependencies (Reuse)
- librosa: Onset detection
- ffmpeg: Video/audio extraction and assembly
- numpy: Numerical processing
- opencv-python: Video frame extraction (if needed)

---

## File Structure

```
video-hacking/
├── src/
│   ├── pitch_onset_analyzer.py      # Phase 1.1: Analyze guide video
│   ├── pitch_database_builder.py    # Phase 1.2: Build source pitch DB
│   ├── pitch_matcher.py             # Phase 2: Match pitches
│   ├── pitch_video_assembler.py     # Phase 3: Assemble final video
│   └── pitch_visualizer.py          # (Optional) Visualize pitch contours
├── test_pitch_pipeline.sh           # Complete pitch matching pipeline
├── test_pitch_analysis.sh           # Test pitch detection only
└── PITCH_MATCHING_README.md         # Documentation for pitch tool
```

---

## Key Decisions to Make

### 1. Pitch Detection Library
- **CREPE**: Most accurate, ML-based, slower (0.3s per second of audio)
- **pYIN**: Good accuracy, faster, built into librosa
- **Recommendation:** Start with CREPE, add pYIN as fast option

### 2. Matching Strategy
- Start with exact MIDI note matching
- Add duration weighting in v2
- Advanced: pitch contour matching in v3

### 3. Duration Handling
- v1: Simple trimming (may cut notes short)
- v2: Time-stretching up to ±20%
- v3: Smart splitting/merging for large duration mismatches

### 4. Reuse Policy
- Default: `allow` (best matches, some repetition)
- For variety: `min_gap` (5-10 seconds between reuses)
- For long source: `none` (no reuse)

---

## Testing Strategy

### Test Data Needed
1. **Guide video:** Person singing simple melody (e.g., "Happy Birthday")
2. **Source video:** Same person singing different song (e.g., national anthem)
3. Duration: 30-60 seconds each
4. Clean audio, clear singing

### Success Metrics
1. **Pitch accuracy:** >90% of notes within ±25 cents
2. **Temporal alignment:** Notes start within ±50ms of target
3. **Visual coherence:** Cuts feel natural, not jarring
4. **Audio quality:** No clicks, pops, or artifacts

### Debugging Tools
1. Pitch contour visualizer (plot guide vs matched)
2. Side-by-side video comparison
3. Pitch detection confidence scores
4. Match quality statistics

---

## Differences from Semantic Matching Tool

| Aspect | Semantic Tool | Pitch Tool |
|--------|--------------|------------|
| **Matching** | ImageBind embeddings (semantic) | Pitch frequency (musical) |
| **Analysis** | Visual + audio semantics | Audio pitch only |
| **Use case** | Artistic video cuts | Musical performance reconstruction |
| **Precision** | Approximate mood/content | Exact musical notes |
| **Reuse** | Visual variety important | Pitch accuracy critical |

---

## Open Questions

1. **Should we pitch-shift source clips to match guide exactly?**
   - Pro: Perfect pitch matching
   - Con: Artifacts, may sound unnatural
   - Recommendation: Make it optional

2. **How to handle sustained notes?**
   - Option A: Use one long clip from source
   - Option B: Loop/repeat short clips
   - Recommendation: Find longest matching clip, loop if needed

3. **What about rests/silence?**
   - Insert black frames or source video b-roll?
   - Keep source video playing with silence?
   - Recommendation: Keep video playing, mute audio

4. **Transition style?**
   - Hard cuts (current semantic tool)
   - Crossfades (50-100ms)
   - Recommendation: Start with hard cuts, add crossfade option

---

## Next Steps

1. Review this plan with user
2. Discuss key decisions (pitch library, matching strategy, duration handling)
3. Create project structure
4. Implement Phase 1.1 (guide video pitch analysis)
5. Test pitch detection accuracy
6. Proceed to subsequent phases

---

## Estimated Timeline

- **Phase 1:** 2-3 days (pitch analysis + database)
- **Phase 2:** 1-2 days (matching algorithm)
- **Phase 3:** 1-2 days (video assembly)
- **Testing & refinement:** 2-3 days
- **Total:** ~1 week for MVP, 2 weeks with enhancements
