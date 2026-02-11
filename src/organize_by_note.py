#!/usr/bin/env python3
"""
Organize by Note

Reads a pitch source database and organizes clips by musical note.
Can extract video clips or generate EDL files for each note.

Usage:
    python organize_by_note.py --database source_database.json --output ./notes --mode clips
    python organize_by_note.py --database source_database.json --output ./notes --mode edl
"""

import argparse
import json
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple

from edl_generator import EDLGenerator
import video_utils


def load_database(database_path: str) -> dict:
    """Load and validate source database."""
    with open(database_path, 'r') as f:
        data = json.load(f)

    # Validate required fields
    if 'pitch_database' not in data:
        raise ValueError("Database missing 'pitch_database' field")
    if 'pitch_index' not in data:
        raise ValueError("Database missing 'pitch_index' field")

    return data


def get_segments_by_note(data: dict) -> dict:
    """
    Group segments by note name.

    Returns:
        Dict mapping note name (e.g., "C4") to list of segments
    """
    pitch_database = data['pitch_database']
    pitch_index = data['pitch_index']

    # Group by note name instead of MIDI number for folder names
    segments_by_note = defaultdict(list)

    for midi_str, segment_ids in pitch_index.items():
        for seg_id in segment_ids:
            segment = pitch_database[seg_id]
            note_name = segment.get('pitch_note', f"MIDI_{midi_str}")
            segments_by_note[note_name].append(segment)

    # Sort segments within each note by duration (longest first)
    for note in segments_by_note:
        segments_by_note[note].sort(key=lambda s: s.get('duration', 0), reverse=True)

    return dict(segments_by_note)


def analyze_source_videos(segments_by_note: dict) -> Dict:
    """
    Analyze all unique source videos to find resolutions and frame rates.

    Returns:
        Analysis dict with min/max resolution, fps, unique values
    """
    # Collect unique video paths
    unique_videos = set()
    for segments in segments_by_note.values():
        for segment in segments:
            video_path = segment.get('video_path', '')
            if video_path and Path(video_path).exists():
                unique_videos.add(video_path)

    print(f"\nAnalyzing {len(unique_videos)} source videos...")

    resolutions = []
    fps_values = []

    for video_path in sorted(unique_videos):
        width, height = video_utils.get_video_resolution(video_path)
        fps = video_utils.get_video_fps(video_path)
        resolutions.append((width, height))
        fps_values.append(fps)
        print(f"  {Path(video_path).name}: {width}x{height} @ {fps:.2f}fps")

    if not resolutions:
        return {
            'min_width': 1920, 'min_height': 1080,
            'max_width': 1920, 'max_height': 1080,
            'min_fps': 24.0, 'max_fps': 24.0,
            'unique_resolutions': [(1920, 1080)],
            'unique_fps': [24.0]
        }

    return {
        'min_width': min(r[0] for r in resolutions),
        'min_height': min(r[1] for r in resolutions),
        'max_width': max(r[0] for r in resolutions),
        'max_height': max(r[1] for r in resolutions),
        'min_fps': min(fps_values),
        'max_fps': max(fps_values),
        'unique_resolutions': list(set(resolutions)),
        'unique_fps': list(set(fps_values))
    }


def extract_clips(segments_by_note: dict, output_dir: Path,
                  width: int, height: int, fps: float,
                  video_format: str = "mp4") -> Dict[str, list]:
    """
    Extract video clips using ffmpeg with scale and crop.

    Args:
        segments_by_note: Dict mapping note name to segments
        output_dir: Output directory path
        width: Target width
        height: Target height
        fps: Target frame rate
        video_format: Output video format

    Returns:
        Dict mapping note name to list of extracted clip info dicts
    """
    extracted_clips = defaultdict(list)

    # Scale and crop filter: scale to fill, then crop to exact dimensions
    # This ensures the video fills the frame without letterboxing
    scale_crop_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
    )

    for note_name, segments in segments_by_note.items():
        # Create folder for this note
        note_dir = output_dir / note_name
        note_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{note_name}: {len(segments)} segments")

        for i, segment in enumerate(segments):
            video_path = segment.get('video_path', '')
            start_time = segment.get('start_time', 0)
            duration = segment.get('duration', 0)
            segment_id = segment.get('segment_id', i)

            if not video_path or not Path(video_path).exists():
                print(f"  Warning: Video not found: {video_path}")
                continue

            # Output filename: segment_XXXX_0.XXs.mp4
            output_name = f"segment_{segment_id:04d}_{duration:.2f}s.{video_format}"
            output_path = note_dir / output_name

            # ffmpeg command with scale+crop
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(duration),
                '-vf', scale_crop_filter,
                '-r', str(fps),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-c:a', 'aac',
                '-ar', '44100',
                '-b:a', '192k',
                str(output_path)
            ]

            try:
                subprocess.run(cmd, capture_output=True, check=True, stdin=subprocess.DEVNULL)
                extracted_clips[note_name].append({
                    'clip_path': str(output_path),
                    'duration': duration,
                    'segment_id': segment_id
                })
                print(f"  Extracted: {output_name}")
            except subprocess.CalledProcessError as e:
                print(f"  Error extracting {output_name}: {e.stderr.decode()[:100]}")

    return dict(extracted_clips)


def generate_edls(segments_by_note: dict, output_dir: Path, frame_rate: float = 24.0) -> int:
    """
    Generate EDL files for each note (referencing original source videos).

    Returns:
        Number of EDL files created
    """
    total_edls = 0

    for note_name, segments in segments_by_note.items():
        # Create folder for this note
        note_dir = output_dir / note_name
        note_dir.mkdir(parents=True, exist_ok=True)

        # Create EDL for this note
        edl = EDLGenerator(f"{note_name} Segments", frame_rate=frame_rate)

        for segment in segments:
            video_path = segment.get('video_path', '')
            start_time = segment.get('start_time', 0)
            end_time = segment.get('end_time', start_time)
            duration = segment.get('duration', end_time - start_time)
            segment_id = segment.get('segment_id', 0)

            if not video_path:
                continue

            edl.add_event(
                source_path=video_path,
                source_in=start_time,
                source_out=end_time,
                comment=f"Segment {segment_id}, {duration:.3f}s"
            )

        # Write EDL file
        edl_path = note_dir / f"{note_name}_segments.edl"
        edl.write(str(edl_path))
        total_edls += 1

        print(f"{note_name}: {len(segments)} segments -> {edl_path.name}")

    return total_edls


def generate_edls_from_clips(extracted_clips: Dict[str, list], output_dir: Path,
                              frame_rate: float = 24.0) -> int:
    """
    Generate EDL files for each note, referencing extracted clips.

    Args:
        extracted_clips: Dict mapping note name to list of clip info dicts
        output_dir: Output directory path
        frame_rate: Frame rate for EDL timecodes

    Returns:
        Number of EDL files created
    """
    total_edls = 0

    for note_name, clips in extracted_clips.items():
        if not clips:
            continue

        # Create folder for this note (should already exist from extraction)
        note_dir = output_dir / note_name
        note_dir.mkdir(parents=True, exist_ok=True)

        # Create EDL for this note
        edl = EDLGenerator(f"{note_name} Clips", frame_rate=frame_rate)

        for clip_info in clips:
            clip_path = clip_info['clip_path']
            duration = clip_info['duration']
            segment_id = clip_info['segment_id']

            # For extracted clips, source in/out is 0 to duration
            # (the clip is already trimmed)
            edl.add_event(
                source_path=clip_path,
                source_in=0.0,
                source_out=duration,
                comment=f"Segment {segment_id}, {duration:.3f}s"
            )

        # Write EDL file
        edl_path = note_dir / f"{note_name}_clips.edl"
        edl.write(str(edl_path))
        total_edls += 1

        print(f"{note_name}: {len(clips)} clips -> {edl_path.name}")

    return total_edls


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Organize pitch database clips by musical note",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract clips with interactive prompts:
  %(prog)s --database source_database.json --output ./notes --mode clips

  # Extract clips with specific resolution and fps:
  %(prog)s --database source_database.json --output ./notes --mode clips --resolution 1920x1080 --fps 30

  # Auto-select smallest resolution/fps (no prompts):
  %(prog)s --database source_database.json --output ./notes --mode clips --auto-resolution --auto-fps

  # Generate EDL files (referencing original sources):
  %(prog)s --database source_database.json --output ./notes --mode edl

  # Extract clips AND generate EDLs referencing those clips:
  %(prog)s --database source_database.json --output ./notes --mode clips-edl
"""
    )
    parser.add_argument(
        '--database',
        type=str,
        required=True,
        help='Path to source_database.json'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory path'
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['clips', 'edl', 'clips-edl'],
        help='Output mode: clips (extract videos), edl (generate EDL files), or clips-edl (both)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='mp4',
        help='Video format for clips mode (default: mp4)'
    )
    parser.add_argument(
        '--resolution',
        type=str,
        help='Output resolution WxH for clips mode (e.g., 1920x1080)'
    )
    parser.add_argument(
        '--auto-resolution',
        action='store_true',
        help='Use smallest source resolution (no prompt)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        help='Frame rate for clips mode or EDL timecodes'
    )
    parser.add_argument(
        '--auto-fps',
        action='store_true',
        help='Use smallest source fps (no prompt)'
    )

    args = parser.parse_args()

    print("=== Organize by Note ===\n")

    # Check database exists
    if not Path(args.database).exists():
        print(f"Error: Database not found: {args.database}")
        return 1

    # Load database
    print(f"Loading database: {args.database}")
    data = load_database(args.database)

    num_segments = data.get('num_segments', len(data['pitch_database']))
    num_unique = data.get('num_unique_pitches', len(data['pitch_index']))
    print(f"  Total segments: {num_segments}")
    print(f"  Unique notes: {num_unique}")

    # Group segments by note
    segments_by_note = get_segments_by_note(data)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process based on mode
    if args.mode in ('clips', 'clips-edl'):
        # Parse resolution if provided
        target_width = None
        target_height = None
        if args.resolution:
            try:
                target_width, target_height = map(int, args.resolution.lower().split('x'))
            except ValueError:
                print(f"Error: Invalid resolution format: {args.resolution}")
                return 1

        # Analyze source videos
        analysis = analyze_source_videos(segments_by_note)

        # Get resolution
        if target_width is None or target_height is None:
            if args.auto_resolution:
                target_width = analysis['min_width']
                target_height = analysis['min_height']
                print(f"\nUsing resolution: {target_width}x{target_height} (smallest)")
            else:
                target_width, target_height = video_utils.prompt_for_resolution(analysis)

        # Get fps
        if args.fps:
            target_fps = args.fps
        elif args.auto_fps:
            target_fps = analysis['min_fps']
            print(f"Using frame rate: {target_fps:.2f} fps (smallest)")
        else:
            target_fps = video_utils.prompt_for_fps(analysis)

        print(f"\nExtracting clips to: {output_dir}")
        print(f"Output: {target_width}x{target_height} @ {target_fps:.2f}fps (scale+crop)")

        extracted_clips = extract_clips(
            segments_by_note, output_dir,
            target_width, target_height, target_fps,
            args.format
        )
        total_clips = sum(len(clips) for clips in extracted_clips.values())
        print(f"\n=== Extracted {total_clips} clips ===")

        # For clips-edl mode, also generate EDLs referencing the extracted clips
        if args.mode == 'clips-edl':
            print(f"\nGenerating EDL files referencing extracted clips...")
            total_edls = generate_edls_from_clips(extracted_clips, output_dir, target_fps)
            print(f"\n=== Done: Extracted {total_clips} clips, created {total_edls} EDL files ===")
        else:
            print(f"\n=== Done ===")
    else:
        # EDL mode (references original source videos)
        fps = args.fps or 24.0
        print(f"\nGenerating EDL files to: {output_dir}")
        total = generate_edls(segments_by_note, output_dir, fps)
        print(f"\n=== Done: Created {total} EDL files ===")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
