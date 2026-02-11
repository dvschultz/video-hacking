#!/usr/bin/env python3
"""
Merge two pitch source database JSON files into one.

Standalone tool that combines pre-computed databases without re-analyzing video.
Uses the same merge logic as PitchSourceAnalyzer.save_database() with --append.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from pitch_utils import midi_to_note_name


def build_pitch_index(database):
    """Build pitch index from a database of segments."""
    pitch_index = defaultdict(list)
    for segment in database:
        pitch_index[segment['pitch_midi']].append(segment['segment_id'])
    for midi in pitch_index:
        pitch_index[midi].sort()
    return pitch_index


def merge_databases(db1_path, db2_path, output_path):
    """Merge two source database JSON files."""
    # Load both databases
    with open(db1_path, 'r') as f:
        db1 = json.load(f)
    with open(db2_path, 'r') as f:
        db2 = json.load(f)

    print(f"Database 1: {db1_path}")
    print(f"  Videos: {db1.get('num_videos', 0)}, Segments: {db1.get('num_segments', 0)}")
    print(f"Database 2: {db2_path}")
    print(f"  Videos: {db2.get('num_videos', 0)}, Segments: {db2.get('num_segments', 0)}")

    # Offset segment IDs in db2 to avoid conflicts
    id_offset = len(db1['pitch_database'])
    print(f"\nRenumbering database 2 segments starting from ID {id_offset}")
    for seg in db2['pitch_database']:
        seg['segment_id'] += id_offset
    for seg in db2.get('silence_segments', []):
        if 'segment_id' in seg:
            seg['segment_id'] += id_offset

    # Concatenate lists
    merged_pitch_db = db1['pitch_database'] + db2['pitch_database']
    merged_silence = db1.get('silence_segments', []) + db2.get('silence_segments', [])
    merged_videos = db1.get('source_videos', []) + db2.get('source_videos', [])

    # Rebuild pitch index from scratch
    pitch_index = build_pitch_index(merged_pitch_db)
    pitch_index_dict = {str(midi): segment_ids for midi, segment_ids in pitch_index.items()}

    # Recalculate summary fields
    total_musical = db1.get('total_musical_duration', 0) + db2.get('total_musical_duration', 0)
    total_silence = db1.get('total_silence_duration', 0) + db2.get('total_silence_duration', 0)

    merged = {
        'source_videos': merged_videos,
        'num_videos': len(merged_videos),
        'num_segments': len(merged_pitch_db),
        'num_unique_pitches': len(pitch_index),
        'num_silence_gaps': len(merged_silence),
        'total_musical_duration': total_musical,
        'total_silence_duration': total_silence,
        'pitch_database': merged_pitch_db,
        'silence_segments': merged_silence,
        'pitch_index': pitch_index_dict
    }

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)

    # Print summary
    print(f"\nMerged database saved to: {output_path}")
    print(f"  Videos: {merged['num_videos']}")
    print(f"  Segments: {merged['num_segments']}")
    print(f"  Unique pitches: {merged['num_unique_pitches']}")
    print(f"  Silence gaps: {merged['num_silence_gaps']}")
    print(f"  Musical duration: {total_musical:.2f}s")
    print(f"  Silence duration: {total_silence:.2f}s")

    if merged_pitch_db:
        midi_values = [seg['pitch_midi'] for seg in merged_pitch_db]
        min_midi = min(midi_values)
        max_midi = max(midi_values)
        print(f"  Pitch range: {midi_to_note_name(min_midi)} (MIDI {min_midi}) to "
              f"{midi_to_note_name(max_midi)} (MIDI {max_midi})")


def main():
    parser = argparse.ArgumentParser(
        description='Merge two pitch source database JSON files into one.'
    )
    parser.add_argument('database1', help='First source database JSON file')
    parser.add_argument('database2', help='Second source database JSON file')
    parser.add_argument('--output', required=True, help='Output path for merged database')

    args = parser.parse_args()

    # Validate inputs exist
    for path in [args.database1, args.database2]:
        if not Path(path).exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)

    merge_databases(args.database1, args.database2, args.output)


if __name__ == '__main__':
    main()
