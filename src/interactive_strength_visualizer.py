#!/usr/bin/env python3
"""
Interactive HTML Visualizer for Onset Strength Analysis

Generates an interactive HTML page with audio playback and continuous
onset strength curve visualization.
"""

import argparse
import json
import base64
import numpy as np
from pathlib import Path


def generate_html_visualizer(
    audio_path: str,
    strength_json_path: str,
    output_html_path: str,
    threshold: float = 0.1
):
    """
    Generate interactive HTML visualizer for onset strength.

    Args:
        audio_path: Path to audio file
        strength_json_path: Path to JSON file with onset strength values
        output_html_path: Path to output HTML file
        threshold: Threshold for highlighting cut points
    """
    # Load onset strength data
    with open(strength_json_path, 'r') as f:
        data = json.load(f)

    values = data['onset_strength_values']
    times = data['times']
    duration = data['duration']
    fps = data['analysis_rate']

    # Calculate segment statistics based on threshold
    def calculate_segment_stats(values, times, threshold):
        """Calculate segment durations between frames above threshold."""
        cut_indices = [i for i, v in enumerate(values) if v > threshold]

        if len(cut_indices) == 0:
            return {
                'num_cuts': 0,
                'num_segments': 0,
                'avg_duration': 0,
                'min_duration': 0,
                'max_duration': 0,
                'durations': []
            }

        # Calculate durations between consecutive cuts
        durations = []
        for i in range(len(cut_indices) - 1):
            duration = times[cut_indices[i + 1]] - times[cut_indices[i]]
            durations.append(duration)

        # Add final segment (from last cut to end)
        if cut_indices:
            final_duration = times[-1] - times[cut_indices[-1]]
            durations.append(final_duration)

        return {
            'num_cuts': len(cut_indices),
            'num_segments': len(durations),
            'avg_duration': np.mean(durations) if durations else 0,
            'min_duration': np.min(durations) if durations else 0,
            'max_duration': np.max(durations) if durations else 0,
            'durations': durations
        }

    initial_stats = calculate_segment_stats(values, times, threshold)

    # Encode audio as base64
    audio_path = Path(audio_path)
    with open(audio_path, 'rb') as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')

    # Determine MIME type
    ext = audio_path.suffix.lower()
    mime_types = {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.ogg': 'audio/ogg',
        '.m4a': 'audio/mp4',
        '.flac': 'audio/flac'
    }
    mime_type = mime_types.get(ext, 'audio/mpeg')

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Onset Strength Visualizer - {audio_path.name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        h1 {{
            margin-bottom: 10px;
            color: #fff;
        }}

        .info {{
            margin-bottom: 20px;
            color: #999;
            font-size: 14px;
        }}

        .visualizer {{
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}

        canvas {{
            width: 100%;
            display: block;
            background: #1e1e1e;
            border-radius: 4px;
            cursor: crosshair;
        }}

        .controls {{
            display: flex;
            gap: 15px;
            align-items: center;
            margin-top: 15px;
            padding: 15px;
            background: #333;
            border-radius: 4px;
            flex-wrap: wrap;
        }}

        button {{
            padding: 10px 20px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }}

        button:hover {{
            background: #45a049;
        }}

        .time-display {{
            font-family: 'Courier New', monospace;
            font-size: 18px;
            color: #4CAF50;
            min-width: 150px;
        }}

        .threshold-control {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .threshold-control input {{
            width: 150px;
        }}

        .threshold-control label {{
            font-size: 14px;
            color: #999;
        }}

        .speed-control {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .speed-control input {{
            width: 100px;
        }}

        .speed-control label {{
            font-size: 14px;
            color: #999;
        }}

        .stats {{
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}

        .stat-item {{
            background: #333;
            padding: 15px;
            border-radius: 4px;
        }}

        .stat-label {{
            color: #999;
            font-size: 12px;
            margin-bottom: 5px;
        }}

        .stat-value {{
            color: #fff;
            font-size: 24px;
            font-weight: bold;
        }}

        .legend {{
            display: flex;
            gap: 20px;
            margin-top: 15px;
            font-size: 14px;
            flex-wrap: wrap;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .legend-color {{
            width: 20px;
            height: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Onset Strength Visualizer</h1>
        <div class="info">
            <strong>{audio_path.name}</strong> &bull;
            Analysis rate: {fps} FPS &bull;
            Duration: {duration:.2f}s &bull;
            {data['num_frames']} frames
        </div>

        <div class="visualizer">
            <canvas id="strengthCanvas" width="1200" height="200"></canvas>

            <div class="controls">
                <button id="playPauseBtn">Play</button>
                <button id="stopBtn">Stop</button>
                <div class="time-display" id="timeDisplay">0:00.000 / {int(duration//60)}:{int(duration%60):02d}.000</div>

                <div class="threshold-control">
                    <label for="thresholdControl">Threshold:</label>
                    <input type="range" id="thresholdControl" min="0" max="1" step="0.01" value="{threshold}">
                    <span id="thresholdValue">{threshold:.2f}</span>
                </div>

                <div class="speed-control">
                    <label for="speedControl">Speed:</label>
                    <input type="range" id="speedControl" min="0.25" max="2" step="0.25" value="1">
                    <span id="speedValue">1.0x</span>
                </div>
            </div>

            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #4CAF50;"></div>
                    <span>Playhead</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff4444;"></div>
                    <span>Onset Strength Curve</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #00ff00;"></div>
                    <span>Threshold Line</span>
                </div>
                <div class="legend-item">
                    <span>Click to seek</span>
                </div>
            </div>
        </div>

        <div class="stats">
            <div class="stat-item">
                <div class="stat-label">Total Cuts</div>
                <div class="stat-value" id="totalCuts">{initial_stats['num_cuts']}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Total Segments</div>
                <div class="stat-value" id="totalSegments">{initial_stats['num_segments']}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Avg Segment Duration</div>
                <div class="stat-value" id="avgDuration">{initial_stats['avg_duration']:.3f}s</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Min Segment</div>
                <div class="stat-value" id="minDuration">{initial_stats['min_duration']:.3f}s</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Max Segment</div>
                <div class="stat-value" id="maxDuration">{initial_stats['max_duration']:.3f}s</div>
            </div>
        </div>
    </div>

    <audio id="audioPlayer" preload="auto">
        <source src="data:{mime_type};base64,{audio_data}" type="{mime_type}">
    </audio>

    <script>
        // Data
        const onsetValues = {json.dumps(values)};
        const times = {json.dumps(times)};
        const duration = {duration};
        const fps = {fps};

        // Elements
        const audioPlayer = document.getElementById('audioPlayer');
        const playPauseBtn = document.getElementById('playPauseBtn');
        const stopBtn = document.getElementById('stopBtn');
        const timeDisplay = document.getElementById('timeDisplay');
        const canvas = document.getElementById('strengthCanvas');
        const ctx = canvas.getContext('2d');
        const thresholdControl = document.getElementById('thresholdControl');
        const thresholdValue = document.getElementById('thresholdValue');
        const speedControl = document.getElementById('speedControl');
        const speedValue = document.getElementById('speedValue');
        const totalCutsEl = document.getElementById('totalCuts');
        const totalSegmentsEl = document.getElementById('totalSegments');
        const avgDurationEl = document.getElementById('avgDuration');
        const minDurationEl = document.getElementById('minDuration');
        const maxDurationEl = document.getElementById('maxDuration');

        let currentThreshold = {threshold};
        let animationFrameId = null;

        // Calculate segment statistics
        function calculateSegmentStats(threshold) {{
            const cutIndices = [];
            for (let i = 0; i < onsetValues.length; i++) {{
                if (onsetValues[i] > threshold) {{
                    cutIndices.push(i);
                }}
            }}

            if (cutIndices.length === 0) {{
                return {{
                    numCuts: 0,
                    numSegments: 0,
                    avgDuration: 0,
                    minDuration: 0,
                    maxDuration: 0
                }};
            }}

            // Calculate durations between consecutive cuts
            const durations = [];
            for (let i = 0; i < cutIndices.length - 1; i++) {{
                const duration = times[cutIndices[i + 1]] - times[cutIndices[i]];
                durations.push(duration);
            }}

            // Add final segment
            if (cutIndices.length > 0) {{
                const finalDuration = times[times.length - 1] - times[cutIndices[cutIndices.length - 1]];
                durations.push(finalDuration);
            }}

            const avgDuration = durations.reduce((a, b) => a + b, 0) / durations.length;
            const minDuration = Math.min(...durations);
            const maxDuration = Math.max(...durations);

            return {{
                numCuts: cutIndices.length,
                numSegments: durations.length,
                avgDuration: avgDuration,
                minDuration: minDuration,
                maxDuration: maxDuration
            }};
        }}

        // Draw onset strength curve
        function drawCurve() {{
            const width = canvas.width;
            const height = canvas.height;
            const padding = 10;
            const plotHeight = height - 2 * padding;

            // Clear
            ctx.fillStyle = '#1e1e1e';
            ctx.fillRect(0, 0, width, height);

            // Draw grid
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 10; i++) {{
                const y = padding + (plotHeight * i / 10);
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }}

            // Draw threshold line
            const thresholdY = padding + plotHeight * (1 - currentThreshold);
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(0, thresholdY);
            ctx.lineTo(width, thresholdY);
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw onset strength curve
            ctx.strokeStyle = '#ff4444';
            ctx.lineWidth = 2;
            ctx.beginPath();

            for (let i = 0; i < onsetValues.length; i++) {{
                const x = (times[i] / duration) * width;
                const y = padding + plotHeight * (1 - onsetValues[i]);

                if (i === 0) {{
                    ctx.moveTo(x, y);
                }} else {{
                    ctx.lineTo(x, y);
                }}
            }}

            ctx.stroke();

            // Draw filled area above threshold
            ctx.fillStyle = 'rgba(255, 68, 68, 0.3)';
            ctx.beginPath();

            for (let i = 0; i < onsetValues.length; i++) {{
                if (onsetValues[i] > currentThreshold) {{
                    const x = (times[i] / duration) * width;
                    const y = padding + plotHeight * (1 - onsetValues[i]);
                    const baseY = thresholdY;

                    ctx.fillRect(x - 1, y, 2, baseY - y);
                }}
            }}
        }}

        // Draw playhead
        function drawPlayhead() {{
            const currentTime = audioPlayer.currentTime;
            const x = (currentTime / duration) * canvas.width;

            // Redraw curve first
            drawCurve();

            // Draw playhead
            ctx.strokeStyle = '#4CAF50';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }}

        // Update time display
        function updateTimeDisplay() {{
            const currentTime = audioPlayer.currentTime;
            const minutes = Math.floor(currentTime / 60);
            const seconds = Math.floor(currentTime % 60);
            const ms = Math.floor((currentTime % 1) * 1000);

            const totalMinutes = Math.floor(duration / 60);
            const totalSeconds = Math.floor(duration % 60);

            timeDisplay.textContent = `${{minutes}}:${{seconds.toString().padStart(2, '0')}}.${{ms.toString().padStart(3, '0')}} / ${{totalMinutes}}:${{totalSeconds.toString().padStart(2, '0')}}.000`;
        }}

        // Update segment statistics display
        function updateSegmentStats() {{
            const stats = calculateSegmentStats(currentThreshold);
            totalCutsEl.textContent = stats.numCuts;
            totalSegmentsEl.textContent = stats.numSegments;
            avgDurationEl.textContent = stats.avgDuration.toFixed(3) + 's';
            minDurationEl.textContent = stats.minDuration.toFixed(3) + 's';
            maxDurationEl.textContent = stats.maxDuration.toFixed(3) + 's';
        }}

        // Animation loop
        function animate() {{
            drawPlayhead();
            updateTimeDisplay();
            if (!audioPlayer.paused) {{
                animationFrameId = requestAnimationFrame(animate);
            }}
        }}

        // Play/Pause
        playPauseBtn.addEventListener('click', () => {{
            if (audioPlayer.paused) {{
                audioPlayer.play();
                playPauseBtn.textContent = 'Pause';
                animate();
            }} else {{
                audioPlayer.pause();
                playPauseBtn.textContent = 'Play';
            }}
        }});

        // Stop
        stopBtn.addEventListener('click', () => {{
            audioPlayer.pause();
            audioPlayer.currentTime = 0;
            playPauseBtn.textContent = 'Play';
            drawPlayhead();
            updateTimeDisplay();
        }});

        // Threshold control
        thresholdControl.addEventListener('input', (e) => {{
            currentThreshold = parseFloat(e.target.value);
            thresholdValue.textContent = currentThreshold.toFixed(2);
            drawPlayhead();
            updateSegmentStats();
        }});

        // Speed control
        speedControl.addEventListener('input', (e) => {{
            const speed = parseFloat(e.target.value);
            audioPlayer.playbackRate = speed;
            speedValue.textContent = speed.toFixed(1) + 'x';
        }});

        // Click to seek
        canvas.addEventListener('click', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const seekTime = (x / canvas.width) * duration;
            audioPlayer.currentTime = seekTime;
            drawPlayhead();
            updateTimeDisplay();
        }});

        // Audio ended
        audioPlayer.addEventListener('ended', () => {{
            playPauseBtn.textContent = 'Play';
        }});

        // Initial draw
        drawCurve();
        updateTimeDisplay();
        updateSegmentStats();
    </script>
</body>
</html>"""

    # Write HTML
    with open(output_html_path, 'w') as f:
        f.write(html_content)

    print(f"✓ Interactive visualizer created: {output_html_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive HTML visualizer for onset strength'
    )
    parser.add_argument('--audio', required=True, help='Path to audio file')
    parser.add_argument('--strength', required=True, help='Path to onset strength JSON')
    parser.add_argument('--output', required=True, help='Path to output HTML file')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Initial threshold value (default: 0.1)')

    args = parser.parse_args()

    generate_html_visualizer(
        audio_path=args.audio,
        strength_json_path=args.strength,
        output_html_path=args.output,
        threshold=args.threshold
    )


if __name__ == '__main__':
    main()
