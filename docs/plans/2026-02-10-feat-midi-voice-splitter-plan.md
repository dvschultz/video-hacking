---
title: "feat: Add MIDI polyphonic voice splitter"
type: feat
date: 2026-02-10
brainstorm: docs/brainstorms/2026-02-10-midi-voice-splitter-brainstorm.md
---

# feat: Add MIDI polyphonic voice splitter

## Overview

New tool (`src/midi_voice_splitter.py` + `split_midi_voices.sh`) that splits a polyphonic MIDI channel into N monophonic `guide_sequence` JSON files using pitch-ordered voice assignment. Each voice file feeds directly into the existing pitch matching pipeline (`test_pitch_matcher.sh` → `test_pitch_video_assembly.sh`), producing separate videos that can be composited externally.

## Problem Statement

The existing `midi_guide_converter.py` is strictly monophonic — it silently drops overlapping notes. For channels with chords (e.g., Bohemian Rhapsody channel 5: 117 notes, max 4-voice polyphony, 62% polyphonic), most musical content is lost. Users need a way to preserve all voices as separate guide sequences.

## Proposed Solution

A standalone tool that:
1. Parses a polyphonic MIDI channel, extracting all notes with timing
2. Assigns each note to a voice using pitch-ordered assignment at note start time
3. Outputs N `guide_sequence_voiceN.json` files, each with the full timeline (rests fill gaps)
4. Generates optional audio previews per voice for verification

## Technical Approach

### Voice Assignment Algorithm

```
1. Parse MIDI, collect tempo changes from all tracks
2. Extract ALL notes from target channel (preserving overlaps, unlike midi_guide_converter)
3. Convert ticks to seconds using tempo map
4. Sort notes by start_time (secondary sort: pitch ascending for simultaneous starts)
5. Walk through notes chronologically:
   a. At each note's start_tick, process any note_off events at the same tick FIRST
   b. Batch all note_on events at the same tick
   c. Get all currently active notes (sustained + new batch)
   d. Sort active notes by pitch ascending (tiebreak: note_on order)
   e. Assign each NEW note to voice = (its position in sorted active list + 1)
   f. Sustained notes keep their existing voice assignment (no mid-note reassignment)
6. Track max voices encountered = N (number of output files)
7. For each voice, build segments with rests filling gaps
```

**Key design decisions:**

- **Assignment happens once per note** at its start time, based on pitch position among all active notes. A sustained note does NOT get reassigned when the chord around it changes.
- **For clean chord boundaries** (all notes start/end together, as in the Bohemian Rhapsody example), this produces perfectly pitch-ordered voices.
- **For staggered entries** (a note sustains while others change), a voice may briefly not be the Nth-lowest pitch. This is acceptable — voice identity stability is more useful than strict moment-by-moment pitch ordering.
- **Simultaneous events**: Process `note_off` before `note_on` at the same tick. Batch all `note_on` events at the same tick and assign together.
- **Unisons** (same pitch in chord): tiebreak by note_on processing order (first encountered = lower voice number).

### Output Format

Each voice file uses the same JSON structure as `midi_guide_converter.py`, compatible with `pitch_matcher.py`:

```json
{
    "video_path": null,
    "audio_path": "data/temp/bohemian_ch5_voice1_preview.wav",
    "midi_path": "/path/to/source.mid",
    "sample_rate": 22050,
    "pitch_detection_method": "MIDI_VOICE_SPLIT",
    "midi_channel": 5,
    "voice_number": 1,
    "total_voices": 4,
    "num_segments": 42,
    "total_duration": 120.5,
    "pitch_segments": [
        {
            "index": 0,
            "start_time": 0.0,
            "end_time": 2.5,
            "duration": 2.5,
            "pitch_hz": 0.0,
            "pitch_midi": -1,
            "pitch_note": "REST",
            "pitch_confidence": 1.0,
            "is_rest": true
        },
        {
            "index": 1,
            "start_time": 2.5,
            "end_time": 3.5,
            "duration": 1.0,
            "pitch_hz": 392.0,
            "pitch_midi": 67,
            "pitch_note": "G4",
            "pitch_confidence": 1.0,
            "is_rest": false
        }
    ]
}
```

**Added metadata fields**: `voice_number` (1-indexed), `total_voices`, `pitch_detection_method: "MIDI_VOICE_SPLIT"`. The matcher reads only `pitch_segments`, so extra top-level keys are safe.

### Reusable Components

From `midi_guide_converter.py`:
- `_collect_tempo_changes()` — tempo map construction (copy into new module)
- `_ticks_to_seconds()` — tick-to-seconds conversion (copy into new module)
- `merge_small_rests()` — apply per-voice after splitting
- Segment building logic from `convert_to_segments()` — adapted for per-voice notes
- Audio preview generation via `MIDIPlayer`

From `pitch_utils.py`:
- `midi_to_hz()`, `midi_to_note_name()` — imported directly

From `midi_channel_splitter.py`:
- `list_channels()` display pattern — adapted to include polyphony info

### CLI Interface

```bash
# List channels with polyphony info
python src/midi_voice_splitter.py --midi song.mid --list-channels

# Split channel 5 into voice files
python src/midi_voice_splitter.py --midi song.mid --channel 5 --output-dir data/segments/

# With options
python src/midi_voice_splitter.py --midi song.mid --channel 5 \
    --output-dir data/segments/ \
    --min-rest 0.1 \
    --no-audio

# Shell wrapper
./split_midi_voices.sh <midi_file> <channel> [output_dir]
```

**Arguments:**
- `--midi` (required): Path to MIDI file
- `--channel` (required unless `--list-channels`): MIDI channel (0-15)
- `--output-dir` (default: `data/segments/`): Directory for output JSON files
- `--min-rest` (default: 0.1): Minimum rest duration in seconds (per-voice)
- `--list-channels` (flag): Show channels with polyphony info and exit
- `--no-audio` (flag): Skip audio preview generation
- `--sample-rate` (default: 22050): Audio preview sample rate

### `--list-channels` Output

```
Channels found in lr0060_Bohemian-Rhapsody.mid:
------------------------------------------------------------
  Channel  0:   234 notes,  45 unique, range: C3-C6, max polyphony: 1, Piano
  Channel  4:   117 notes,  18 unique, range: A3-Eb5, max polyphony: 4, Strings
  Channel  9:    89 notes,  12 unique, range: C2-B4, max polyphony: 2, Drums *
------------------------------------------------------------
* = percussion channel (voice splitting may not produce meaningful results)
```

### Split Output

```
Channel 5: 117 notes, max polyphony: 4
Splitting into 4 voice files...

  Voice 1 (lowest):  45 notes, 60.2s sounding (50.1% of song)
    → data/segments/guide_sequence_voice1.json
  Voice 2:           38 notes, 45.8s sounding (38.1% of song)
    → data/segments/guide_sequence_voice2.json
  Voice 3:           27 notes, 30.1s sounding (25.0% of song)
    → data/segments/guide_sequence_voice3.json
  Voice 4 (highest):  7 notes,  3.2s sounding (2.7% of song)
    → data/segments/guide_sequence_voice4.json

Audio previews:
  → data/temp/guide_voice1_preview.wav
  → data/temp/guide_voice2_preview.wav
  → data/temp/guide_voice3_preview.wav
  → data/temp/guide_voice4_preview.wav
```

## Acceptance Criteria

- [x] `--list-channels` displays note count, unique notes, range, max polyphony, and instrument per channel
- [x] Polyphonic channel splits into N files where N = max simultaneous notes
- [x] Each voice file is valid `guide_sequence` JSON consumable by `pitch_matcher.py`
- [x] Each voice file spans the full song duration with rest segments filling gaps
- [x] Voice 1 gets the lowest note at each chord (pitch-ordered assignment at note start)
- [x] Sustained notes keep their voice assignment (no mid-note reassignment)
- [x] Monophonic channel outputs a single file with a "no splitting needed" message
- [x] Channel with 0 notes produces a clear error
- [x] Channel 9 (drums) produces a warning
- [x] `--min-rest` merging applies per-voice
- [x] Audio previews generate per-voice (skippable with `--no-audio`)
- [x] Shell wrapper `split_midi_voices.sh` follows existing project patterns

## Implementation Steps

### Step 1: `src/midi_voice_splitter.py` — Core Module

**New file.** Class `MIDIVoiceSplitter` with these methods:

| Method | Source | Notes |
|--------|--------|-------|
| `__init__()` | New | Store midi_path, channel, output_dir, min_rest, etc. |
| `load_midi()` | Adapt from `midi_guide_converter.py` | Load file, store ticks_per_beat |
| `_collect_tempo_changes()` | Copy from `midi_guide_converter.py:86-103` | Identical logic |
| `_ticks_to_seconds()` | Copy from `midi_guide_converter.py:105-135` | Identical logic |
| `list_channels()` | Adapt from `midi_channel_splitter.py:175-225` | Add polyphony calculation |
| `extract_all_notes()` | **New** — adapted from `extract_notes()` | Collect ALL overlapping notes (not monophonic) |
| `calculate_max_polyphony()` | **New** | Walk note events, track max simultaneous count |
| `assign_voices()` | **New** — core algorithm | Pitch-ordered assignment with persistence |
| `build_voice_segments()` | Adapt from `convert_to_segments()` | Per-voice segment building with rests |
| `merge_small_rests()` | Copy from `midi_guide_converter.py:209-242` | Apply per voice |
| `save_voice_results()` | Adapt from `save_results()` | One JSON file per voice, with voice metadata |
| `generate_audio_previews()` | Adapt from `generate_audio_preview()` | One WAV per voice |
| `split()` | **New** | Orchestrator: load → extract → assign → build → save |

**Key difference from `midi_guide_converter.py`**: `extract_all_notes()` does NOT enforce monophonic behavior. It collects all note_on/note_off pairs on the channel, preserving overlaps. The resulting list may have notes with overlapping time ranges.

**`assign_voices()` pseudocode:**

```python
def assign_voices(self):
    # notes is list of {pitch_midi, start_time, end_time, duration, velocity}
    # sorted by start_time, then pitch ascending

    voice_assignments = {}  # note_id -> voice_number
    active_notes = []       # [(note_id, pitch_midi, end_time, assigned_voice)]
    max_voices = 0

    # Group notes by start_time
    time_groups = group_notes_by_start_time(self.notes)

    for start_time, new_notes in time_groups:
        # Remove expired notes (end_time <= start_time)
        active_notes = [(nid, p, end, v) for nid, p, end, v in active_notes
                        if end > start_time]

        # Sort all active notes (sustained + new) by pitch
        all_active = active_notes + [(n.id, n.pitch_midi, n.end_time, None)
                                      for n in new_notes]
        all_active.sort(key=lambda x: x[1])  # sort by pitch

        # Assign new notes to their position in sorted order
        for position, (nid, pitch, end, existing_voice) in enumerate(all_active):
            voice = position + 1  # 1-indexed
            if existing_voice is None:  # new note
                voice_assignments[nid] = voice
            # sustained notes keep existing_voice (ignore new position)

        # Update active_notes with assignments
        active_notes = []
        for nid, pitch, end, existing_voice in all_active:
            assigned = existing_voice if existing_voice else voice_assignments[nid]
            active_notes.append((nid, pitch, end, assigned))

        max_voices = max(max_voices, len(active_notes))

    self.voice_assignments = voice_assignments
    self.num_voices = max_voices
```

### Step 2: `split_midi_voices.sh` — Shell Wrapper

**New file.** Follow `test_midi_guide.sh` pattern:
- Positional args: `<midi_file> <channel> [output_dir]`
- Python detection, file existence checks, colored output
- Pass-through of remaining args via `"$@"`
- "Next steps" guidance showing per-voice pitch matcher commands

### Step 3: `tests/unit/test_midi_voice_splitter.py` — Unit Tests

**New file.** Follow `test_midi_guide_converter.py` patterns:

| Test Class | Tests |
|------------|-------|
| `TestMIDIVoiceSplitter` | Initialization, load_midi |
| `TestVoiceAssignment` | 2-note chord, 3-note chord, 4-note chord, monophonic input |
| `TestVoiceAssignmentEdgeCases` | Simultaneous note_on, note_off before note_on at same tick, sustained note across chord change, unisons |
| `TestSegmentBuilding` | Rest insertion, leading/trailing rests, min-rest merging |
| `TestOutputFormat` | JSON structure matches pitch_matcher expectations, voice metadata present |
| `TestListChannels` | Polyphony info in channel listing |
| `TestErrorHandling` | Missing file, empty channel, channel 9 warning |

Use mock MIDI messages (same pattern as `test_midi_channel_splitter.py`):
```python
MagicMock(type='note_on', channel=4, note=67, velocity=100, time=0)
```

## Workflow (End-to-End)

```
1. Split:
   ./split_midi_voices.sh bohemian.mid 5 data/segments/

2. Listen to previews to verify split:
   open data/temp/guide_voice1_preview.wav
   open data/temp/guide_voice2_preview.wav
   ...

3. Per voice (repeat for each):
   ./test_pitch_matcher.sh data/segments/guide_sequence_voice1.json \
       data/segments/source_database.json \
       --output data/segments/match_plan_voice1.json

   ./test_pitch_video_assembly.sh data/segments/match_plan_voice1.json

   mv output/final_video.mov output/voice1_video.mov

4. Composite in video editor:
   Import all voice videos, align from frame 0, layer/arrange as desired.
```

## Dependencies & Risks

- **mido**: Already a project dependency, used by all MIDI tools
- **pitch_utils**: Already exists, import directly
- **MIDIPlayer**: Already exists in `src/midi_player.py`, used for audio preview
- **No new dependencies needed**

**Risks:**
- Voice assignment algorithm edge cases with sustained notes during chord changes. Mitigated by unit tests covering these cases.
- User may need to run the matcher N times with different output paths. Documented in workflow section and shell wrapper "Next steps" output.

## References

- Brainstorm: `docs/brainstorms/2026-02-10-midi-voice-splitter-brainstorm.md`
- Pattern source: `src/midi_guide_converter.py` (reuse tempo/timing/segment logic)
- Pattern source: `src/midi_channel_splitter.py` (reuse channel listing UI)
- Output consumer: `src/pitch_matcher.py:104-116` (`load_guide_sequence()`)
- Test pattern: `tests/unit/test_midi_guide_converter.py`
- Example MIDI: `/Volumes/X902/love-song-sing-along/midi-files/886431derrick@bustbright.com/lr0060_Bohemian-Rhapsody.mid` (channel 5, max 4-voice polyphony)
