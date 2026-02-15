---
title: "feat: AE MIDI Pitch Grid Visualizer"
type: feat
status: active
date: 2026-02-15
brainstorm: docs/brainstorms/2026-02-15-ae-midi-pitch-grid-brainstorm.md
---

# AE MIDI Pitch Grid Visualizer

## Overview

A Python generator script that reads a MIDI file, extracts a single channel's note data, merges overlapping intervals, and writes a self-contained After Effects ExtendScript (.jsx) file. The .jsx creates a 12-column x 7-row grid of rectangles where each cell maps to a MIDI pitch. Rectangles snap visible (opacity 100%) when their note is playing and invisible (0%) when it's not.

**Target data:** Channel 2 (Grand Piano) of `lr0060_Bohemian-Rhapsody.mid` — 1,763 notes, 61 unique pitches, F1–F7, ~365 seconds.

## Problem Statement / Motivation

The existing `midi_generative_visuals.jsx` maps MIDI to position keyframes on a single shape — useful but limited. A pitch grid visualization shows the harmonic structure of a performance over time: which notes are active, how polyphony builds and resolves, and the registral spread of the music. This is a single-purpose tool for Bohemian Rhapsody channel 2, with the data baked in.

## Proposed Solution

Two deliverables:

1. **Python generator** (`tools/midi_to_ae_pitch_grid.py`) — Reads the MIDI file, extracts channel notes with tempo-aware timing, merges overlapping intervals per pitch, and writes a complete `.jsx` file with embedded data.

2. **Generated .jsx file** (`output/bohemian_rhapsody_ch2_pitch_grid.jsx`) — Self-contained ExtendScript that creates the AE composition with 61 shape layers + 1 background solid, all keyframed.

### Why a Python generator instead of a standalone .jsx?

- Python has `mido` for robust MIDI parsing with tempo handling (already a project dependency)
- Overlap-merging is complex logic better debugged in Python than ExtendScript (ES3, no debugger)
- The generator pattern is reusable for other MIDI files/channels in the future
- ExtendScript stays dead simple: just read embedded arrays and create layers

## Technical Approach

### Architecture

```
lr0060_Bohemian-Rhapsody.mid
         │
         ▼
┌─────────────────────────────┐
│ midi_to_ae_pitch_grid.py    │
│                             │
│ 1. Parse MIDI (mido)        │
│ 2. Extract channel notes    │
│ 3. Tick-to-seconds (tempo)  │
│ 4. Merge overlapping notes  │
│ 5. Write .jsx with data     │
└─────────────────────────────┘
         │
         ▼
bohemian_rhapsody_ch2_pitch_grid.jsx
         │
         ▼  (run in After Effects)
┌─────────────────────────────┐
│ AE Composition              │
│ 1920x1080, 24fps, ~367s     │
│                             │
│ 61 shape layers (grid)      │
│ + 1 black solid background  │
│ Opacity keyframes (HOLD)    │
└─────────────────────────────┘
```

### Phase 1: Python Generator (`tools/midi_to_ae_pitch_grid.py`)

**CLI interface:**
```bash
python tools/midi_to_ae_pitch_grid.py \
  --midi <path_to_midi> \
  --channel 2 \
  --output <output.jsx>
```

**Core logic:**

1. **Parse MIDI** — Use `mido` to read the file. Build a tempo map from `set_tempo` events. Convert ticks to seconds using tempo-aware arithmetic (reuse pattern from existing `midi_guide_converter.py`).

2. **Extract channel notes** — Collect note-on/note-off pairs for the specified channel. Track active notes per pitch to handle overlapping note-ons. Output: list of `(pitch, start_sec, end_sec)` tuples.

3. **Merge overlapping intervals per pitch** — For each unique MIDI pitch, sort intervals by start time, then merge overlapping or adjacent intervals. This prevents the keyframe conflict where a note-off kills visibility while another note-on is still active.

   ```python
   # Merge algorithm (per pitch):
   sorted_intervals = sorted(intervals, key=lambda x: x[0])
   merged = [sorted_intervals[0]]
   for start, end in sorted_intervals[1:]:
       if start <= merged[-1][1]:  # overlapping or touching
           merged[-1] = (merged[-1][0], max(merged[-1][1], end))
       else:
           merged.append((start, end))
   ```

4. **Enforce minimum 1-frame visibility** — At 24fps, 1 frame = 41.67ms. If any merged interval is shorter than this, extend the end time to `start + 1/24`. This ensures every note produces at least one visible frame.

5. **Compute grid geometry** — Calculate cell positions for a 12x7 grid centered in 1920x1080:
   ```
   Margin: 60px each side
   Usable area: 1800 x 960
   Cell width: 150px (1800 / 12)
   Cell height: 137px (floor(960 / 7))
   Gap: 4px (rect is 146 x 133 within each cell)
   Grid offset: center any remaining pixels
   ```

6. **Write .jsx file** — Generate a complete ExtendScript file using string templating. The embedded data format is a per-pitch dictionary of interval arrays:
   ```javascript
   var DATA = {
       29: [[12.5, 13.8], [45.2, 46.0]],  // F1
       34: [[0.8, 1.5], [2.3, 3.1]],       // A#1
       // ... 61 entries
   };
   ```

### Phase 2: Generated ExtendScript Template

The .jsx file follows established `adobe-hut/ae/` conventions:

```javascript
#target aftereffects

(function() {
    // ========== EMBEDDED DATA ==========
    var DATA = { ... };  // Generated by Python
    var GRID = { ... };  // Grid geometry constants

    // ========== UTILITIES ==========
    // Note name lookup, grid position calculator

    // ========== CORE FUNCTIONS ==========
    // createComp() - new 1920x1080 24fps comp
    // createBackground() - black solid at bottom
    // createPitchLayer() - shape layer with rect + white fill
    // keyframeOpacity() - set HOLD keyframes from intervals

    // ========== MAIN ==========
    function main() {
        var comp = createComp();
        createBackground(comp);
        // For each pitch in DATA (descending order):
        //   Create layer, position in grid, keyframe opacity
        alert("Done! Created " + layerCount + " layers with " + kfCount + " keyframes.");
    }

    // ========== EXECUTE ==========
    app.beginUndoGroup("MIDI Pitch Grid");
    try { main(); } catch (e) { alert("Error: " + e.message); }
    app.endUndoGroup();
})();
```

**Key implementation details:**

- **Comp creation**: `app.project.items.addComp("Bohemian Rhapsody Ch2 Grid", 1920, 1080, 1, duration, 24)`
- **Duration**: `Math.ceil(maxEndTime) + 2` computed from embedded data
- **Background**: `comp.layers.addSolid([0,0,0], "Background", 1920, 1080, 1)`
- **Shape layers**: One per pitch, named `"D#4"`, `"A#2"`, etc.
- **Rectangle size**: From GRID constants (146x133 with 4px gap)
- **Position**: Layer anchor at [0,0], position set to grid cell top-left
- **Initial state**: Opacity keyframe at t=0 set to 0% for every layer
- **Note-on/off**: `setValueAtTime(startSec, 100)` and `setValueAtTime(endSec, 0)`
- **Interpolation**: `KeyframeInterpolationType.HOLD` on every keyframe
- **Layer order**: Created in descending pitch order so high octaves appear at top of Timeline panel
- **Progress**: `$.writeln()` after each layer for console feedback

## Acceptance Criteria

- [x] Python script reads the specified MIDI file and channel
- [x] Overlapping notes on the same pitch are correctly merged into continuous intervals
- [x] Sub-frame notes (< 41.67ms) are extended to minimum 1 frame
- [ ] Generated .jsx runs in AE CC 2020+ without errors
- [x] Creates a 1920x1080 24fps composition with correct duration
- [x] 61 shape layers created (one per used pitch in channel 2)
- [x] Each layer has a white rectangle positioned correctly in the 12x7 grid
- [x] Opacity keyframes use HOLD interpolation (binary on/off, no easing)
- [x] Rectangles are invisible (0%) by default, visible (100%) only during note events
- [x] Layer names match pitch notation (e.g., "D#4", "A#1")
- [x] Black solid background layer at bottom of layer stack
- [x] Script wrapped in undo group

## File Inventory

| File | Type | Purpose |
|------|------|---------|
| `tools/midi_to_ae_pitch_grid.py` | New | Python generator script |
| `output/bohemian_rhapsody_ch2_pitch_grid.jsx` | Generated | AE ExtendScript (output) |

## Dependencies & Risks

**Dependencies:**
- `mido` Python package (already in project environment)
- After Effects CC 2020+ for running the generated script

**Risks:**
- **Execution time in AE**: ~3,500 keyframes across 61 layers. May take 30-120 seconds. Mitigated by undo group wrapping and console progress logging.
- **Large .jsx file**: Embedded data for 61 pitches with merged intervals. Estimated ~15-30KB. Well within ExtendScript limits.

## References & Research

### Internal References
- Existing MIDI script: `adobe-hut/ae/midi_generative_visuals.jsx`
- Grid creation pattern: `adobe-hut/ae/wallpaper_pattern_loop.jsx:414-447`
- HOLD interpolation pattern: `adobe-hut/ae/overlap_tiling_masks.jsx:398-402`
- Shape layer creation: `adobe-hut/ae/midi_generative_visuals.jsx:160-184`
- MIDI parsing with tempo: `src/midi_guide_converter.py`
- Frame boundary lesson: `adobe-hut/docs/solutions/logic-errors/overlap-tiling-masks-missed-overlap-layer-20260212.md`

### Institutional Learnings Applied
- Use frame-index arithmetic for temporal boundaries (from overlap tiling masks solution)
- Set `file.encoding = "UTF-8"` for any file I/O in ExtendScript
- ExtendScript is ES3: no `let`, `const`, arrow functions, template literals
- Keyframe and layer indices are 1-based in AE
- Colors are `[R,G,B]` in 0-1 range, not 0-255
- Use ADBE match names for property access (e.g., `"ADBE Vector Shape - Rect"`)

### Brainstorm
- `docs/brainstorms/2026-02-15-ae-midi-pitch-grid-brainstorm.md`
