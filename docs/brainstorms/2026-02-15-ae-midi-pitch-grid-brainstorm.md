# AE MIDI Pitch Grid Visualizer

**Date:** 2026-02-15
**Status:** Brainstorm
**Related:** `adobe-hut/ae/midi_generative_visuals.jsx` (existing script, different approach)

## What We're Building

An After Effects ExtendScript (.jsx) that creates a 12-column x 7-row grid of rectangles representing the piano pitch space for channel 2 (Grand Piano) of Bohemian Rhapsody. Each rectangle maps to a specific MIDI pitch. When that note is playing, the rectangle snaps to full opacity; when it's not, it's invisible. The MIDI data (1,763 notes from channel 2) is embedded directly in the script.

**Input:** Nothing at runtime. All data hardcoded.
**Output:** An AE composition with 61 shape layers (one per used pitch), each with opacity keyframes driven by note-on/note-off events.

## Why This Approach

- **Embedded data** removes file I/O complexity and makes it a single self-contained script
- **Binary on/off opacity** is clean, graphic, and easy to enhance later in AE (add colors, effects, transitions)
- **Piano-style 12x7 grid** is musically intuitive: columns = pitch classes (C through B), rows = octaves (1 through 7)
- **Only used pitches** get layers (61 of 84 possible cells), reducing layer count while all 12 pitch classes and all 7 octaves are represented

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Grid layout | 12 cols (pitch class) x 7 rows (octave) | Piano-intuitive, compact |
| Visibility trigger | Binary opacity 0/100 | Clean, graphic, AE-enhanceable |
| Data source | Embedded in script | Zero-setup, single-purpose |
| Rectangle color | White on black | Minimal; user colorizes in AE |
| Comp settings | 1920x1080 @ 24fps | Cinematic standard |
| Empty pitches | Skip (only 61 layers) | All pitch classes and octaves still represented |
| Duration | Full song (~365s) | Complete visualization |

## Design Details

### Grid Layout

```
         C   C#   D   D#   E    F   F#   G   G#   A   A#   B
Oct 7   [ ] [ ]  [ ] [ ]  [ ]  [x] [ ]  [ ] [ ]  [ ] [ ]  [ ]
Oct 6   [x] [x]  [x] [x]  [x]  [x] [x]  [x] [x]  [x] [x]  [x]
Oct 5   [x] [x]  [x] [x]  [x]  [x] [x]  [x] [x]  [x] [x]  [x]
Oct 4   [x] [x]  [x] [x]  [x]  [x] [x]  [x] [x]  [x] [x]  [x]
Oct 3   [x] [x]  [x] [x]  [x]  [x] [x]  [x] [x]  [x] [x]  [x]
Oct 2   [x] [x]  [x] [x]  [x]  [x] [x]  [x] [x]  [x] [x]  [x]
Oct 1   [ ] [ ]  [ ] [ ]  [ ]  [x] [x]  [x] [x]  [x] [x]  [x]
```

- [x] = layer created (note appears in data), [ ] = skipped
- High octaves at top, low at bottom
- Grid centered in 1920x1080 comp with margin

### Layer Structure

Each used pitch gets one shape layer containing:
- Rectangle shape (sized to fill its grid cell with small gap)
- White fill
- Opacity keyframes: 0% default, 100% during note-on to note-off

Layer naming: `"D#4"`, `"A#2"`, etc. (pitch name + octave)

### Keyframe Strategy

For each note event in the embedded data:
1. Set opacity to 100 at `start_sec`
2. Set opacity to 0 at `end_sec`
3. Use HOLD interpolation (step function, no easing)

Notes may overlap on the same pitch (re-trigger). The script handles this by ensuring opacity stays at 100 through overlapping regions.

### Data Size

- 61 layers x ~29 avg keyframes/layer = ~1,770 opacity keyframe pairs
- Well within AE's capabilities

## Scope

**In scope:**
- Grid creation with proper spacing and centering
- Opacity keyframing from embedded MIDI data
- Hold interpolation for binary on/off
- Layer naming by pitch
- Undo group wrapping

**Out of scope:**
- Color/styling beyond white rectangles
- Audio import or sync
- Runtime MIDI file selection
- Labels or annotations on the grid
- Any effects or animation beyond opacity toggle

## Open Questions

None - all design decisions resolved.
