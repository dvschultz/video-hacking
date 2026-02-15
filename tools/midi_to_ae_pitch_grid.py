#!/usr/bin/env python3
"""
MIDI to After Effects Pitch Grid Generator

Reads a MIDI file, extracts notes from a specified channel, merges overlapping
intervals per pitch, and writes a self-contained After Effects ExtendScript (.jsx)
that creates a grid of rectangles with opacity keyframes.

Usage:
    python tools/midi_to_ae_pitch_grid.py \
        --midi <path_to_midi> \
        --channel 2 \
        --output <output.jsx>
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import mido


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Grid constants
COMP_WIDTH = 1440
COMP_HEIGHT = 1080
FPS = 24
MARGIN = 0
GRID_COLS = 7   # octaves (columns)
GRID_ROWS = 12  # pitch classes (rows)
GAP = 0
MIN_FRAME_DURATION = 1.0 / FPS  # ~41.67ms


def note_name(midi_note):
    """Convert MIDI note number to name like 'C4', 'D#3'."""
    return f"{NOTE_NAMES[midi_note % 12]}{midi_note // 12 - 1}"


def build_tempo_map(track):
    """Build list of (abs_tick, microseconds_per_beat) from a track."""
    tempo_map = []
    abs_tick = 0
    for msg in track:
        abs_tick += msg.time
        if msg.type == 'set_tempo':
            tempo_map.append((abs_tick, msg.tempo))
    if not tempo_map or tempo_map[0][0] != 0:
        tempo_map.insert(0, (0, 500000))  # default 120 BPM
    return tempo_map


def tick_to_seconds(tick, tempo_map, ticks_per_beat):
    """Convert absolute tick to seconds using tempo map."""
    seconds = 0.0
    prev_tick = 0
    prev_tempo = tempo_map[0][1]

    for map_tick, map_tempo in tempo_map:
        if map_tick >= tick:
            break
        if map_tick > prev_tick:
            seconds += (map_tick - prev_tick) * prev_tempo / (ticks_per_beat * 1_000_000)
            prev_tick = map_tick
        prev_tempo = map_tempo

    seconds += (tick - prev_tick) * prev_tempo / (ticks_per_beat * 1_000_000)
    return seconds


def extract_channel_notes(midi_file, channel):
    """Extract note-on/note-off pairs from a MIDI file for a given channel.

    Returns list of (pitch, start_sec, end_sec) tuples.
    """
    mid = mido.MidiFile(midi_file)
    ticks_per_beat = mid.ticks_per_beat

    # For Type 0 files, all events are in track 0.
    # For Type 1 files, events may be spread across tracks.
    # We process all tracks and accumulate events for the target channel.
    all_events = []

    for track in mid.tracks:
        tempo_map = build_tempo_map(track)
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if hasattr(msg, 'channel') and msg.channel == channel:
                if msg.type in ('note_on', 'note_off'):
                    time_sec = tick_to_seconds(abs_tick, tempo_map, ticks_per_beat)
                    all_events.append((time_sec, msg.type, msg.note, msg.velocity))

    # For Type 0 files, tempo events are in the same track as notes.
    # For Type 1, we need a global tempo map. Let's rebuild from all tracks.
    if mid.type == 1:
        # Rebuild with global tempo map from track 0 (convention)
        global_tempo_map = build_tempo_map(mid.tracks[0])
        all_events = []
        for track in mid.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if hasattr(msg, 'channel') and msg.channel == channel:
                    if msg.type in ('note_on', 'note_off'):
                        time_sec = tick_to_seconds(abs_tick, global_tempo_map, ticks_per_beat)
                        all_events.append((time_sec, msg.type, msg.note, msg.velocity))

    # Sort by time for proper note-on/note-off pairing
    all_events.sort(key=lambda e: e[0])

    # Pair note-on with note-off
    notes_on = defaultdict(list)  # pitch -> [(start_sec, velocity)]
    completed = []

    for time_sec, msg_type, pitch, velocity in all_events:
        if msg_type == 'note_on' and velocity > 0:
            notes_on[pitch].append(time_sec)
        elif msg_type == 'note_off' or (msg_type == 'note_on' and velocity == 0):
            if notes_on[pitch]:
                start_sec = notes_on[pitch].pop(0)
                completed.append((pitch, start_sec, time_sec))

    return completed


def merge_intervals(intervals):
    """Merge overlapping or touching intervals.

    Args:
        intervals: list of (start, end) tuples, sorted by start.

    Returns:
        list of merged (start, end) tuples.
    """
    if not intervals:
        return []

    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_ivs[0])]

    for start, end in sorted_ivs[1:]:
        if start <= merged[-1][1]:  # overlapping or touching
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(s, e) for s, e in merged]


def enforce_min_duration(intervals, min_dur):
    """Extend any interval shorter than min_dur."""
    result = []
    for start, end in intervals:
        if end - start < min_dur:
            end = start + min_dur
        result.append((start, end))
    return result


def compute_grid_geometry():
    """Compute grid cell positions for a 12-row x 7-col grid in 1440x1080.

    Layout: 12 rows = pitch classes (C at top, B at bottom)
            7 cols = octaves (octave 1 at left, octave 7 at right)
    No margin — grid fills the full frame.

    Returns dict with cell_width, cell_height, rect_width, rect_height,
    and a positions dict mapping (pitch_class, octave) -> (x, y).
    """
    cell_w = COMP_WIDTH / GRID_COLS   # 1440 / 7 = ~205.71
    cell_h = COMP_HEIGHT / GRID_ROWS  # 1080 / 12 = 90.0

    rect_w = cell_w - GAP
    rect_h = cell_h - GAP

    offset_x = GAP / 2
    offset_y = GAP / 2

    positions = {}
    for octave in range(1, 8):  # octaves 1-7
        col = octave - 1  # octave 1 at col 0 (left), octave 7 at col 6 (right)
        for pc in range(12):  # pitch classes 0-11
            row = pc  # C at row 0 (top), B at row 11 (bottom)
            x = offset_x + col * cell_w
            y = offset_y + row * cell_h
            positions[(pc, octave)] = (round(x, 2), round(y, 2))

    return {
        'cell_w': cell_w,
        'cell_h': cell_h,
        'rect_w': round(rect_w, 2),
        'rect_h': round(rect_h, 2),
        'positions': positions,
    }


def generate_jsx(pitch_intervals, grid, output_path):
    """Generate a self-contained After Effects ExtendScript .jsx file.

    Args:
        pitch_intervals: dict mapping MIDI pitch -> list of (start_sec, end_sec)
        grid: grid geometry from compute_grid_geometry()
        output_path: path to write the .jsx file
    """
    # Compute duration from data
    max_end = 0
    total_keyframes = 0
    for intervals in pitch_intervals.values():
        for _, end in intervals:
            if end > max_end:
                max_end = end
        total_keyframes += len(intervals) * 2  # on + off per interval
    total_keyframes += len(pitch_intervals)  # initial 0% keyframe per pitch

    comp_duration = int(max_end) + 3  # round up + buffer

    # Build DATA object as JS literal
    data_lines = []
    for pitch in sorted(pitch_intervals.keys(), reverse=True):
        intervals = pitch_intervals[pitch]
        intervals_str = ', '.join(
            f'[{s:.4f}, {e:.4f}]' for s, e in intervals
        )
        data_lines.append(f'        {pitch}: [{intervals_str}]')

    data_js = '{\n' + ',\n'.join(data_lines) + '\n    }'

    # Build GRID positions for used pitches only
    grid_lines = []
    for pitch in sorted(pitch_intervals.keys(), reverse=True):
        pc = pitch % 12
        octave = pitch // 12 - 1
        pos = grid['positions'].get((pc, octave))
        if pos:
            grid_lines.append(
                f'        {pitch}: [{pos[0]}, {pos[1]}]'
            )

    grid_js = '{\n' + ',\n'.join(grid_lines) + '\n    }'

    # Build pitch-to-name mapping
    name_lines = []
    for pitch in sorted(pitch_intervals.keys(), reverse=True):
        name_lines.append(f'        {pitch}: "{note_name(pitch)}"')
    names_js = '{\n' + ',\n'.join(name_lines) + '\n    }'

    jsx = f'''#target aftereffects
/*
 * Bohemian Rhapsody — Channel 2 (Grand Piano) Pitch Grid
 *
 * Auto-generated by midi_to_ae_pitch_grid.py
 * Source: lr0060_Bohemian-Rhapsody.mid, channel 2
 *
 * Creates a 12-row x 7-col grid of rectangles (pitch classes down, octaves across).
 * Each rectangle becomes visible (opacity 100%) when its MIDI pitch is playing.
 *
 * {len(pitch_intervals)} pitches, ~{total_keyframes} keyframes, {comp_duration}s duration
 *
 * Usage: File > Scripts > Run Script File...
 */

(function() {{

    // ========== EMBEDDED DATA ==========

    // Per-pitch merged intervals: pitch -> [[startSec, endSec], ...]
    var DATA = {data_js};

    // Grid positions: pitch -> [x, y] (top-left of cell)
    var GRID_POS = {grid_js};

    // Pitch names: pitch -> "C4", "D#3", etc.
    var PITCH_NAMES = {names_js};

    // Grid geometry
    var RECT_W = {grid['rect_w']};
    var RECT_H = {grid['rect_h']};
    var COMP_W = {COMP_WIDTH};
    var COMP_H = {COMP_HEIGHT};
    var COMP_DUR = {comp_duration};
    var COMP_FPS = {FPS};

    // ========== UTILITIES ==========

    function getKeys(obj) {{
        var keys = [];
        for (var k in obj) {{
            if (obj.hasOwnProperty(k)) keys.push(parseInt(k, 10));
        }}
        return keys;
    }}

    // ========== CORE FUNCTIONS ==========

    function createComp() {{
        return app.project.items.addComp(
            "Bohemian Rhapsody Ch2 Grid",
            COMP_W, COMP_H, 1, COMP_DUR, COMP_FPS
        );
    }}

    function createBackground(comp) {{
        var bg = comp.layers.addSolid([0, 0, 0], "Background", COMP_W, COMP_H, 1);
        bg.locked = true;
        return bg;
    }}

    function createPitchLayer(comp, pitch) {{
        var name = PITCH_NAMES[pitch];
        var pos = GRID_POS[pitch];

        var layer = comp.layers.addShape();
        layer.name = name;

        // Add rectangle shape
        var contents = layer.property("Contents");
        var shapeGroup = contents.addProperty("ADBE Vector Group");
        shapeGroup.name = "Rect";

        var shapesInGroup = shapeGroup.property("Contents");
        var rect = shapesInGroup.addProperty("ADBE Vector Shape - Rect");
        rect.property("Size").setValue([RECT_W, RECT_H]);

        // Add white fill
        var fill = shapesInGroup.addProperty("ADBE Vector Graphic - Fill");
        fill.property("Color").setValue([1, 1, 1]);

        // Position: anchor at center of shape, move to cell center
        var centerX = pos[0] + RECT_W / 2;
        var centerY = pos[1] + RECT_H / 2;
        layer.property("Transform").property("Position").setValue([centerX, centerY]);

        return layer;
    }}

    function keyframeOpacity(layer, intervals) {{
        var opacity = layer.property("Transform").property("Opacity");
        var kfCount = 0;

        // Initial state: invisible at t=0
        opacity.setValueAtTime(0, 0);
        kfCount++;

        // For each merged interval: on at start, off at end
        for (var i = 0; i < intervals.length; i++) {{
            var startSec = intervals[i][0];
            var endSec = intervals[i][1];

            opacity.setValueAtTime(startSec, 100);
            opacity.setValueAtTime(endSec, 0);
            kfCount += 2;
        }}

        // Set ALL keyframes to HOLD interpolation
        for (var k = 1; k <= opacity.numKeys; k++) {{
            opacity.setInterpolationTypeAtKey(k, KeyframeInterpolationType.HOLD);
        }}

        return kfCount;
    }}

    // ========== MAIN ==========

    function main() {{
        var pitches = getKeys(DATA);
        // Sort descending so high octaves are created first (top of Timeline)
        pitches.sort(function(a, b) {{ return b - a; }});

        var comp = createComp();

        // Background created first — each subsequent addShape() goes on top of it
        createBackground(comp);

        var layerCount = 0;
        var totalKf = 0;

        // Create pitch layers (each new layer is added above the background)
        for (var i = 0; i < pitches.length; i++) {{
            var pitch = pitches[i];
            var intervals = DATA[pitch];

            var layer = createPitchLayer(comp, pitch);
            var kf = keyframeOpacity(layer, intervals);

            totalKf += kf;
            layerCount++;
            $.writeln("Created " + PITCH_NAMES[pitch] + " (" + layerCount + "/" + pitches.length + ") - " + intervals.length + " intervals, " + kf + " keyframes");
        }}

        alert("Done!\\n\\nCreated " + layerCount + " pitch layers + background\\n" + totalKf + " total keyframes\\nDuration: " + COMP_DUR + "s");
    }}

    // ========== EXECUTE ==========

    app.beginUndoGroup("MIDI Pitch Grid");
    try {{
        main();
    }} catch (e) {{
        alert("Script error: " + e.message);
    }}
    app.endUndoGroup();

}})();
'''

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(jsx)

    print(f"Generated: {output_path}")
    print(f"  Pitches: {len(pitch_intervals)}")
    print(f"  Keyframes: ~{total_keyframes}")
    print(f"  Duration: {comp_duration}s")
    print(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description='Generate After Effects pitch grid script from MIDI file'
    )
    parser.add_argument('--midi', required=True, help='Path to MIDI file')
    parser.add_argument('--channel', type=int, required=True,
                        help='MIDI channel number (1-indexed as displayed)')
    parser.add_argument('--output', required=True, help='Output .jsx file path')
    parser.add_argument('--fps', type=int, default=FPS,
                        help=f'Frame rate for minimum duration enforcement (default: {FPS})')
    args = parser.parse_args()

    midi_path = Path(args.midi)
    if not midi_path.exists():
        print(f"Error: MIDI file not found: {midi_path}", file=sys.stderr)
        sys.exit(1)

    # MIDI channels are 0-indexed in mido but typically displayed 1-indexed
    # The user specifies the displayed channel number
    channel_idx = args.channel  # Keep as-is since this file uses 0-indexed internally

    print(f"Reading: {midi_path}")
    print(f"Channel: {args.channel} (0-indexed: {channel_idx})")

    # Extract notes
    notes = extract_channel_notes(str(midi_path), channel_idx)
    if not notes:
        print(f"Error: No notes found on channel {args.channel}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracted {len(notes)} note events")

    # Group by pitch
    pitch_notes = defaultdict(list)
    for pitch, start, end in notes:
        pitch_notes[pitch].append((start, end))

    print(f"Unique pitches: {len(pitch_notes)}")

    # Merge overlapping intervals per pitch
    min_dur = 1.0 / args.fps
    pitch_intervals = {}
    total_intervals = 0
    for pitch in sorted(pitch_notes.keys()):
        merged = merge_intervals(pitch_notes[pitch])
        merged = enforce_min_duration(merged, min_dur)
        pitch_intervals[pitch] = merged
        total_intervals += len(merged)

    print(f"After merging: {total_intervals} intervals (from {len(notes)} notes)")

    # Show overlap reduction stats
    overlap_reduced = len(notes) - total_intervals
    if overlap_reduced > 0:
        print(f"Merged {overlap_reduced} overlapping note events")

    # Compute grid
    grid = compute_grid_geometry()

    # Verify all pitches fit in grid
    for pitch in pitch_intervals:
        pc = pitch % 12
        octave = pitch // 12 - 1
        if (pc, octave) not in grid['positions']:
            print(f"Warning: pitch {note_name(pitch)} (MIDI {pitch}) "
                  f"octave {octave} outside grid range 1-7", file=sys.stderr)

    # Generate .jsx
    generate_jsx(pitch_intervals, grid, args.output)


if __name__ == '__main__':
    main()
