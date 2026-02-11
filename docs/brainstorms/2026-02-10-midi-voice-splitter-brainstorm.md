# MIDI Voice Splitter for Polyphonic Channels

**Date**: 2026-02-10
**Status**: Brainstorm

## What We're Building

A new tool (`src/midi_voice_splitter.py` + `split_midi_voices.sh`) that takes a polyphonic MIDI channel and splits it into N monophonic guide_sequence JSON files using pitch-ordered voice assignment.

**Problem**: The existing `midi_guide_converter.py` is strictly monophonic — it silently drops overlapping notes. For a channel like channel 5 in Bohemian Rhapsody (117 notes, max 4-voice polyphony, 62% of note-on events are polyphonic), this loses most of the musical content.

**Goal**: Produce N separate `guide_sequence_voiceN.json` files (one per voice), each preserving the full song timeline with rests in the gaps. Each file feeds directly into `test_pitch_matcher.sh` to produce a separate video. The user composites the videos externally (Premiere, DaVinci, etc.).

## Why This Approach

- **Pitch-ordered layers**: At each chord, voice 1 gets the lowest note, voice 2 the next, etc. Simple, predictable, no ambiguity.
- **Direct JSON output**: Outputs guide_sequence JSON files that feed directly into `test_pitch_matcher.sh`, skipping the intermediate MIDI-to-guide conversion step. Fewer steps in the workflow.
- **Full timeline with rests**: Each voice file spans the entire song duration. Gaps where a voice isn't playing become rest segments. All resulting videos stay in sync for external compositing.
- **No compositing built in**: The user handles video layering in their editor. Keeps scope minimal.

## Key Decisions

1. **Voice assignment**: Pitch-ordered (lowest = voice 1). A note is assigned to a voice at its start time based on its pitch position among all currently active notes. No mid-note reassignment.

2. **Number of voices**: Automatically determined by maximum simultaneous polyphony on the channel. For Bohemian Rhapsody channel 5, this is 4.

3. **Output format**: `guide_sequence_voiceN.json` files in the same format as `midi_guide_converter.py` output (`pitch_segments` array). Each file is immediately usable by `pitch_matcher.py`.

4. **Timeline preservation**: Each voice JSON spans the full song duration. Gaps where a voice isn't playing become rest segments (`is_rest: true`, `pitch_midi: -1`).

5. **Scope**: Splitting + JSON conversion only. No batch automation. User runs the matching/assembly pipeline per voice manually.

## Algorithm

```
1. Parse MIDI file, collect tempo changes from all tracks
2. Extract all notes from target channel as (start_tick, end_tick, pitch, velocity) tuples
3. Convert ticks to seconds using tempo map (reuse logic from midi_guide_converter)
4. Determine max polyphony (N) by walking through note events
5. For each note, at its start time:
   a. Find all notes active at that time (including the new note)
   b. Sort active notes by pitch (ascending)
   c. Assign the new note to voice = (its position in sorted list + 1)
6. For each voice (1..N), build a guide_sequence JSON:
   a. Collect that voice's notes in chronological order
   b. Insert rest segments to fill gaps between notes and at start/end
   c. Apply min_rest_duration merging (same as midi_guide_converter)
   d. Write as guide_sequence_voiceN.json with pitch_segments array
```

## Interface

```bash
# List channels and their polyphony
python src/midi_voice_splitter.py --midi song.mid --list-channels

# Split channel 5 into monophonic voice JSON files
python src/midi_voice_splitter.py --midi song.mid --channel 5 --output-dir data/segments/

# With min-rest merging (same as midi_guide_converter)
python src/midi_voice_splitter.py --midi song.mid --channel 5 --output-dir data/segments/ --min-rest 0.1

# Shell wrapper
./split_midi_voices.sh song.mid 5 [output_dir]
```

**Output**:
```
Channel 5: 117 notes, max polyphony: 4
Splitting into 4 voice files...
  Voice 1 (lowest):  45 notes → guide_sequence_voice1.json
  Voice 2:           38 notes → guide_sequence_voice2.json
  Voice 3:           27 notes → guide_sequence_voice3.json
  Voice 4 (highest):  7 notes → guide_sequence_voice4.json
```

## Workflow (End-to-End)

```
1. Split:    ./split_midi_voices.sh bohemian.mid 5 data/segments/
2. Per voice (repeat for each voice file):
   a. Match:    ./test_pitch_matcher.sh data/segments/guide_sequence_voice1.json data/segments/source_database.json
   b. Assemble: ./test_pitch_video_assembly.sh data/segments/match_plan.json
   c. Rename output video (e.g., voice1_output.mov)
3. Composite in video editor: layer all voice videos, aligned from start
```

## Open Questions

- **Should `--list-channels` show polyphony info alongside note counts?** Probably yes — helps identify which channels need splitting vs. which are already monophonic.
- **Should the tool warn if a channel is already monophonic?** Yes, output a single guide_sequence.json (same as midi_guide_converter) and note that no splitting was needed.

## What This Does NOT Include

- No batch pipeline automation (run matching/assembly per voice manually)
- No video compositing/layering
- No voice-leading algorithm (pitch-ordered only)
- No intermediate MIDI file output (JSON only)
- No per-voice instrument assignment
