#!/usr/bin/env python3
"""Generate an interactive HTML visualization of MIDI channel polyphony.

Parses a MIDI file, tracks note on/off events on a specified channel,
and produces a self-contained HTML file with:
  - Piano roll with bars colored by polyphony count
  - Polyphony count timeline graph below the piano roll
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import mido


def parse_midi_channel(midi_path: str, channel: int):
    """Parse MIDI file and extract note events for the given channel.

    Returns:
        notes: list of dicts {note, start, end, velocity}
        tempo_changes: list of (tick, tempo_us)
        ticks_per_beat: int
    """
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat

    # Collect tempo changes across all tracks
    tempo_changes = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempo_changes.append((abs_tick, msg.tempo))
    tempo_changes.sort(key=lambda x: x[0])
    if not tempo_changes:
        tempo_changes = [(0, 500000)]  # default 120 BPM

    # Build tick-to-seconds converter
    def tick_to_seconds(target_tick):
        elapsed = 0.0
        prev_tick = 0
        current_tempo = 500000  # default
        for tc_tick, tc_tempo in tempo_changes:
            if tc_tick >= target_tick:
                break
            elapsed += (tc_tick - prev_tick) * (current_tempo / 1_000_000) / ticks_per_beat
            prev_tick = tc_tick
            current_tempo = tc_tempo
        elapsed += (target_tick - prev_tick) * (current_tempo / 1_000_000) / ticks_per_beat
        return elapsed

    # Collect note on/off events for the target channel
    active_notes = {}  # (note) -> (start_tick, velocity)
    notes = []

    for track in mid.tracks:
        abs_tick = 0
        track_active = {}
        for msg in track:
            abs_tick += msg.time
            if not hasattr(msg, "channel") or msg.channel != channel:
                continue
            if msg.type == "note_on" and msg.velocity > 0:
                track_active[msg.note] = (abs_tick, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in track_active:
                    start_tick, vel = track_active.pop(msg.note)
                    notes.append({
                        "note": msg.note,
                        "start_tick": start_tick,
                        "end_tick": abs_tick,
                        "start": tick_to_seconds(start_tick),
                        "end": tick_to_seconds(abs_tick),
                        "velocity": vel,
                    })

    notes.sort(key=lambda n: (n["start"], n["note"]))
    return notes, tick_to_seconds


def compute_polyphony_timeline(notes, resolution=0.01):
    """Compute polyphony count over time at given resolution (seconds)."""
    if not notes:
        return [], 0

    max_time = max(n["end"] for n in notes)
    num_samples = int(max_time / resolution) + 1
    counts = [0] * num_samples

    for n in notes:
        start_idx = int(n["start"] / resolution)
        end_idx = int(n["end"] / resolution)
        for i in range(start_idx, min(end_idx, num_samples)):
            counts[i] += 1

    max_poly = max(counts) if counts else 0
    # Downsample for the timeline - keep every 10th sample for performance
    step = max(1, num_samples // 5000)
    timeline = []
    for i in range(0, num_samples, step):
        timeline.append({"time": round(i * resolution, 3), "count": counts[i]})

    return timeline, max_poly


def compute_note_polyphony(notes):
    """For each note, compute how many notes overlap at its midpoint."""
    for i, n in enumerate(notes):
        mid_t = (n["start"] + n["end"]) / 2
        count = 0
        for other in notes:
            if other["start"] <= mid_t < other["end"]:
                count += 1
        n["polyphony"] = count
    return notes


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_note_name(note_num):
    octave = (note_num // 12) - 1
    name = NOTE_NAMES[note_num % 12]
    return f"{name}{octave}"


def generate_html(notes, timeline, max_poly, channel, midi_filename):
    """Generate a self-contained HTML visualization."""

    if not notes:
        return "<html><body><h1>No notes found on this channel</h1></body></html>"

    min_note = min(n["note"] for n in notes)
    max_note = max(n["note"] for n in notes)
    max_time = max(n["end"] for n in notes)

    # Prepare note data for JS
    notes_json = json.dumps([{
        "note": n["note"],
        "name": midi_note_name(n["note"]),
        "start": round(n["start"], 4),
        "end": round(n["end"], 4),
        "velocity": n["velocity"],
        "polyphony": n["polyphony"],
    } for n in notes])

    timeline_json = json.dumps(timeline)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MIDI Polyphony — Channel {channel} — {midi_filename}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #1a1a2e;
    color: #e0e0e0;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    overflow-x: hidden;
}}
.header {{
    padding: 20px 30px;
    background: #16213e;
    border-bottom: 2px solid #0f3460;
}}
.header h1 {{
    font-size: 18px;
    color: #e94560;
    margin-bottom: 4px;
}}
.header .subtitle {{
    font-size: 13px;
    color: #888;
}}
.stats {{
    display: flex;
    gap: 24px;
    padding: 12px 30px;
    background: #16213e;
    border-bottom: 1px solid #0f3460;
    font-size: 13px;
}}
.stats .stat {{
    display: flex;
    gap: 6px;
}}
.stats .stat .label {{ color: #888; }}
.stats .stat .value {{ color: #e94560; font-weight: bold; }}
.controls {{
    display: flex;
    gap: 16px;
    align-items: center;
    padding: 10px 30px;
    background: #1a1a2e;
    border-bottom: 1px solid #0f3460;
    font-size: 12px;
}}
.controls label {{ color: #888; }}
.controls input[type="range"] {{ width: 120px; }}
.controls .zoom-val {{ color: #e94560; min-width: 30px; }}
.canvas-container {{
    position: relative;
    overflow-x: auto;
    overflow-y: hidden;
}}
canvas {{
    display: block;
}}
.tooltip {{
    position: fixed;
    background: #16213e;
    border: 1px solid #0f3460;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
    pointer-events: none;
    z-index: 100;
    display: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}}
.tooltip .tt-note {{ color: #e94560; font-weight: bold; }}
.tooltip .tt-time {{ color: #aaa; }}
.tooltip .tt-poly {{ font-weight: bold; }}
.legend {{
    display: flex;
    gap: 16px;
    padding: 8px 30px;
    font-size: 11px;
    background: #16213e;
    border-top: 1px solid #0f3460;
}}
.legend .swatch {{
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 3px;
    vertical-align: middle;
    margin-right: 4px;
}}
</style>
</head>
<body>

<div class="header">
    <h1>MIDI Polyphony Visualization</h1>
    <div class="subtitle">Channel {channel} — {midi_filename}</div>
</div>

<div class="stats">
    <div class="stat"><span class="label">Notes:</span><span class="value">{len(notes)}</span></div>
    <div class="stat"><span class="label">Duration:</span><span class="value">{max_time:.1f}s ({int(max_time//60)}:{int(max_time%60):02d})</span></div>
    <div class="stat"><span class="label">Range:</span><span class="value">{midi_note_name(min_note)} — {midi_note_name(max_note)}</span></div>
    <div class="stat"><span class="label">Max polyphony:</span><span class="value" style="color:#ff4444">{max_poly}</span></div>
</div>

<div class="controls">
    <label>Zoom:</label>
    <input type="range" id="zoomSlider" min="1" max="20" value="4" step="0.5">
    <span class="zoom-val" id="zoomVal">4x</span>
    <label style="margin-left:16px">Scroll to time:</label>
    <input type="range" id="scrollSlider" min="0" max="100" value="0" step="0.1">
    <span class="zoom-val" id="scrollVal">0:00</span>
</div>

<div class="canvas-container" id="canvasContainer">
    <canvas id="pianoRoll"></canvas>
</div>
<div class="canvas-container" id="timelineContainer">
    <canvas id="timelineCanvas"></canvas>
</div>

<div class="legend">
    <span><span class="swatch" style="background:#4ade80"></span>1-2 voices</span>
    <span><span class="swatch" style="background:#facc15"></span>3-4 voices</span>
    <span><span class="swatch" style="background:#fb923c"></span>5-6 voices</span>
    <span><span class="swatch" style="background:#ef4444"></span>7+ voices</span>
</div>

<div class="tooltip" id="tooltip">
    <div class="tt-note"></div>
    <div class="tt-time"></div>
    <div class="tt-poly"></div>
</div>

<script>
const NOTES = {notes_json};
const TIMELINE = {timeline_json};
const MAX_POLY = {max_poly};
const MIN_NOTE = {min_note};
const MAX_NOTE = {max_note};
const MAX_TIME = {max_time:.4f};

const NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
function noteName(n) {{ return NOTE_NAMES[n % 12] + (Math.floor(n/12)-1); }}

function polyColor(p) {{
    if (p <= 2) return '#4ade80';
    if (p <= 4) return '#facc15';
    if (p <= 6) return '#fb923c';
    return '#ef4444';
}}

function polyColorAlpha(p, a) {{
    if (p <= 2) return `rgba(74,222,128,${{a}})`;
    if (p <= 4) return `rgba(250,204,21,${{a}})`;
    if (p <= 6) return `rgba(251,146,60,${{a}})`;
    return `rgba(239,68,68,${{a}})`;
}}

const pianoCanvas = document.getElementById('pianoRoll');
const pianoCtx = pianoCanvas.getContext('2d');
const tlCanvas = document.getElementById('timelineCanvas');
const tlCtx = tlCanvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const zoomSlider = document.getElementById('zoomSlider');
const zoomVal = document.getElementById('zoomVal');
const scrollSlider = document.getElementById('scrollSlider');
const scrollVal = document.getElementById('scrollVal');
const container = document.getElementById('canvasContainer');
const tlContainer = document.getElementById('timelineContainer');

const LABEL_W = 60;
const NOTE_H = 14;
const TL_H = 120;
const NOTE_RANGE = MAX_NOTE - MIN_NOTE + 1;
const PIANO_H = NOTE_RANGE * NOTE_H + 40;

let zoom = 4;
let pxPerSec = 0;
let canvasW = 0;
let scrollX = 0;

function formatTime(s) {{
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return m + ':' + (sec < 10 ? '0' : '') + sec;
}}

function recalc() {{
    zoom = parseFloat(zoomSlider.value);
    zoomVal.textContent = zoom + 'x';
    const viewW = window.innerWidth;
    pxPerSec = (viewW - LABEL_W) * zoom / MAX_TIME;
    canvasW = Math.ceil(LABEL_W + MAX_TIME * pxPerSec + 20);

    const dpr = window.devicePixelRatio || 1;
    pianoCanvas.width = canvasW * dpr;
    pianoCanvas.height = PIANO_H * dpr;
    pianoCanvas.style.width = canvasW + 'px';
    pianoCanvas.style.height = PIANO_H + 'px';
    pianoCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    tlCanvas.width = canvasW * dpr;
    tlCanvas.height = TL_H * dpr;
    tlCanvas.style.width = canvasW + 'px';
    tlCanvas.style.height = TL_H + 'px';
    tlCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}}

function drawPianoRoll() {{
    pianoCtx.clearRect(0, 0, canvasW, PIANO_H);

    // Background
    pianoCtx.fillStyle = '#1a1a2e';
    pianoCtx.fillRect(0, 0, canvasW, PIANO_H);

    // Note lane backgrounds
    for (let n = MIN_NOTE; n <= MAX_NOTE; n++) {{
        const y = (MAX_NOTE - n) * NOTE_H;
        const isBlack = [1,3,6,8,10].includes(n % 12);
        pianoCtx.fillStyle = isBlack ? '#151528' : '#1e1e38';
        pianoCtx.fillRect(LABEL_W, y, canvasW - LABEL_W, NOTE_H);

        // Gridline
        pianoCtx.strokeStyle = '#252545';
        pianoCtx.lineWidth = 0.5;
        pianoCtx.beginPath();
        pianoCtx.moveTo(LABEL_W, y + NOTE_H);
        pianoCtx.lineTo(canvasW, y + NOTE_H);
        pianoCtx.stroke();
    }}

    // Note labels
    pianoCtx.font = '10px SF Mono, Consolas, monospace';
    pianoCtx.textAlign = 'right';
    pianoCtx.textBaseline = 'middle';
    for (let n = MIN_NOTE; n <= MAX_NOTE; n++) {{
        const y = (MAX_NOTE - n) * NOTE_H + NOTE_H / 2;
        const name = noteName(n);
        pianoCtx.fillStyle = name.includes('#') ? '#666' : '#999';
        pianoCtx.fillText(name, LABEL_W - 6, y);
    }}

    // Time gridlines
    const interval = zoom > 8 ? 1 : zoom > 3 ? 5 : 10;
    pianoCtx.font = '10px SF Mono, Consolas, monospace';
    pianoCtx.textAlign = 'center';
    pianoCtx.textBaseline = 'bottom';
    for (let t = 0; t <= MAX_TIME; t += interval) {{
        const x = LABEL_W + t * pxPerSec;
        pianoCtx.strokeStyle = '#252545';
        pianoCtx.lineWidth = 0.5;
        pianoCtx.beginPath();
        pianoCtx.moveTo(x, 0);
        pianoCtx.lineTo(x, NOTE_RANGE * NOTE_H);
        pianoCtx.stroke();
    }}

    // Draw notes
    for (const n of NOTES) {{
        const x = LABEL_W + n.start * pxPerSec;
        const w = Math.max(2, (n.end - n.start) * pxPerSec);
        const y = (MAX_NOTE - n.note) * NOTE_H + 1;
        const h = NOTE_H - 2;

        pianoCtx.fillStyle = polyColor(n.polyphony);
        pianoCtx.globalAlpha = 0.3 + (n.velocity / 127) * 0.7;
        pianoCtx.fillRect(x, y, w, h);
        pianoCtx.globalAlpha = 1;

        // Border
        pianoCtx.strokeStyle = polyColor(n.polyphony);
        pianoCtx.lineWidth = 1;
        pianoCtx.strokeRect(x, y, w, h);
    }}

    // Time labels at bottom
    const labelY = NOTE_RANGE * NOTE_H + 14;
    pianoCtx.fillStyle = '#666';
    pianoCtx.font = '10px SF Mono, Consolas, monospace';
    pianoCtx.textAlign = 'center';
    for (let t = 0; t <= MAX_TIME; t += interval) {{
        const x = LABEL_W + t * pxPerSec;
        pianoCtx.fillText(formatTime(t), x, labelY);
    }}
}}

function drawTimeline() {{
    tlCtx.clearRect(0, 0, canvasW, TL_H);
    tlCtx.fillStyle = '#16213e';
    tlCtx.fillRect(0, 0, canvasW, TL_H);

    if (TIMELINE.length < 2) return;

    const graphH = TL_H - 30;
    const graphY = 10;

    // Horizontal gridlines
    for (let p = 1; p <= MAX_POLY; p++) {{
        const y = graphY + graphH - (p / MAX_POLY) * graphH;
        tlCtx.strokeStyle = '#252545';
        tlCtx.lineWidth = 0.5;
        tlCtx.beginPath();
        tlCtx.moveTo(LABEL_W, y);
        tlCtx.lineTo(canvasW, y);
        tlCtx.stroke();

        tlCtx.fillStyle = '#555';
        tlCtx.font = '9px SF Mono, Consolas, monospace';
        tlCtx.textAlign = 'right';
        tlCtx.textBaseline = 'middle';
        tlCtx.fillText(p.toString(), LABEL_W - 6, y);
    }}

    // Filled area
    tlCtx.beginPath();
    tlCtx.moveTo(LABEL_W + TIMELINE[0].time * pxPerSec, graphY + graphH);
    for (const pt of TIMELINE) {{
        const x = LABEL_W + pt.time * pxPerSec;
        const y = graphY + graphH - (pt.count / MAX_POLY) * graphH;
        tlCtx.lineTo(x, y);
    }}
    tlCtx.lineTo(LABEL_W + TIMELINE[TIMELINE.length-1].time * pxPerSec, graphY + graphH);
    tlCtx.closePath();

    // Gradient fill
    const grad = tlCtx.createLinearGradient(0, graphY, 0, graphY + graphH);
    grad.addColorStop(0, 'rgba(239,68,68,0.4)');
    grad.addColorStop(0.3, 'rgba(251,146,60,0.3)');
    grad.addColorStop(0.6, 'rgba(250,204,21,0.2)');
    grad.addColorStop(1, 'rgba(74,222,128,0.1)');
    tlCtx.fillStyle = grad;
    tlCtx.fill();

    // Line on top — color-coded segments
    tlCtx.lineWidth = 1.5;
    for (let i = 1; i < TIMELINE.length; i++) {{
        const prev = TIMELINE[i-1];
        const cur = TIMELINE[i];
        const x1 = LABEL_W + prev.time * pxPerSec;
        const y1 = graphY + graphH - (prev.count / MAX_POLY) * graphH;
        const x2 = LABEL_W + cur.time * pxPerSec;
        const y2 = graphY + graphH - (cur.count / MAX_POLY) * graphH;
        tlCtx.strokeStyle = polyColor(Math.max(prev.count, cur.count));
        tlCtx.beginPath();
        tlCtx.moveTo(x1, y1);
        tlCtx.lineTo(x2, y2);
        tlCtx.stroke();
    }}

    // Label
    tlCtx.fillStyle = '#888';
    tlCtx.font = '10px SF Mono, Consolas, monospace';
    tlCtx.textAlign = 'left';
    tlCtx.fillText('Polyphony', LABEL_W + 4, graphY + graphH + 16);
}}

function draw() {{
    recalc();
    drawPianoRoll();
    drawTimeline();
}}

// Sync scroll positions
container.addEventListener('scroll', () => {{
    tlContainer.scrollLeft = container.scrollLeft;
    const maxScroll = container.scrollWidth - container.clientWidth;
    if (maxScroll > 0) {{
        const pct = container.scrollLeft / maxScroll * 100;
        scrollSlider.value = pct;
        const t = (container.scrollLeft / (canvasW - window.innerWidth)) * MAX_TIME;
        scrollVal.textContent = formatTime(Math.max(0, t));
    }}
}});
tlContainer.addEventListener('scroll', () => {{
    container.scrollLeft = tlContainer.scrollLeft;
}});

zoomSlider.addEventListener('input', () => {{
    // Save current center time before redraw
    const viewCenter = container.scrollLeft + container.clientWidth / 2;
    const centerTime = (viewCenter - LABEL_W) / pxPerSec;

    draw();

    // Restore scroll to keep same time centered
    const newX = LABEL_W + centerTime * pxPerSec - container.clientWidth / 2;
    container.scrollLeft = Math.max(0, newX);
    tlContainer.scrollLeft = container.scrollLeft;
}});

scrollSlider.addEventListener('input', () => {{
    const pct = parseFloat(scrollSlider.value) / 100;
    const maxScroll = container.scrollWidth - container.clientWidth;
    container.scrollLeft = pct * maxScroll;
    tlContainer.scrollLeft = container.scrollLeft;
    const t = pct * MAX_TIME;
    scrollVal.textContent = formatTime(t);
}});

// Tooltip
pianoCanvas.addEventListener('mousemove', (e) => {{
    const rect = pianoCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left + container.scrollLeft;
    const my = e.clientY - rect.top;
    const time = (mx - LABEL_W) / pxPerSec;

    let hit = null;
    for (const n of NOTES) {{
        const x = LABEL_W + n.start * pxPerSec;
        const w = Math.max(2, (n.end - n.start) * pxPerSec);
        const y = (MAX_NOTE - n.note) * NOTE_H + 1;
        const h = NOTE_H - 2;
        if (mx >= x && mx <= x + w && my >= y && my <= y + h) {{
            hit = n;
            break;
        }}
    }}

    if (hit) {{
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 14) + 'px';
        tooltip.style.top = (e.clientY - 10) + 'px';
        tooltip.querySelector('.tt-note').textContent = hit.name + ' (MIDI ' + hit.note + ')';
        tooltip.querySelector('.tt-time').textContent = formatTime(hit.start) + ' — ' + formatTime(hit.end) + ' (' + ((hit.end - hit.start)*1000).toFixed(0) + 'ms)';
        const polyEl = tooltip.querySelector('.tt-poly');
        polyEl.textContent = hit.polyphony + ' voices';
        polyEl.style.color = polyColor(hit.polyphony);
    }} else {{
        tooltip.style.display = 'none';
    }}
}});
pianoCanvas.addEventListener('mouseleave', () => {{
    tooltip.style.display = 'none';
}});

window.addEventListener('resize', draw);
draw();
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="MIDI polyphony HTML visualizer")
    parser.add_argument("midi_file", help="Path to MIDI file")
    parser.add_argument("--channel", type=int, default=5, help="MIDI channel (0-indexed, default: 5)")
    parser.add_argument("--output", "-o", help="Output HTML file path")
    args = parser.parse_args()

    midi_path = Path(args.midi_file)
    if not midi_path.exists():
        print(f"Error: MIDI file not found: {midi_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(midi_path.parent / f"channel{args.channel}_polyphony.html")

    print(f"Parsing {midi_path.name}, channel {args.channel}...")
    notes, tick_to_sec = parse_midi_channel(str(midi_path), args.channel)
    print(f"  Found {len(notes)} notes")

    if not notes:
        print("No notes found on this channel!", file=sys.stderr)
        sys.exit(1)

    min_note = min(n["note"] for n in notes)
    max_note = max(n["note"] for n in notes)
    max_time = max(n["end"] for n in notes)
    print(f"  Note range: {midi_note_name(min_note)} — {midi_note_name(max_note)}")
    print(f"  Duration: {max_time:.1f}s ({int(max_time//60)}:{int(max_time%60):02d})")

    print("Computing polyphony...")
    notes = compute_note_polyphony(notes)
    timeline, max_poly = compute_polyphony_timeline(notes)
    print(f"  Max polyphony: {max_poly}")

    print(f"Generating HTML → {output_path}")
    html = generate_html(notes, timeline, max_poly, args.channel, midi_path.name)
    Path(output_path).write_text(html)
    print(f"  Written {len(html):,} bytes")
    print("Done!")


if __name__ == "__main__":
    main()
