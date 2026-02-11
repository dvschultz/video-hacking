#!/usr/bin/env python3
"""
Print the duration of a MIDI file.

Usage:
    python midi_duration.py <file.mid> [file2.mid ...]
"""

import sys
from pathlib import Path

try:
    import mido
except ImportError:
    print("Error: mido not installed. Install with: pip install mido")
    sys.exit(1)


def format_duration(seconds):
    """Format seconds as M:SS.ms"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.mid> [file2.mid ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        if not Path(path).exists():
            print(f"Error: {path} not found", file=sys.stderr)
            continue

        mid = mido.MidiFile(path)
        duration = mid.length
        print(f"{path}: {format_duration(duration)} ({duration:.2f}s)")


if __name__ == '__main__':
    main()
