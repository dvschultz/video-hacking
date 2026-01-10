#!/usr/bin/env python3
"""
Duration Source Analyzer

Scans a folder of video clips and builds a duration database.
Each clip is cataloged with its duration and video metadata (resolution, fps, codec).
The database is sorted by duration for efficient matching.

Usage:
    python duration_source_analyzer.py --folder /path/to/clips --output database.json
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict


class DurationSourceAnalyzer:
    """Analyzes a folder of video clips and builds a duration database."""

    def __init__(self, folder_path: str, min_duration: float = 0.1):
        """
        Initialize the analyzer.

        Args:
            folder_path: Path to folder containing video clips
            min_duration: Minimum clip duration to include (seconds)
        """
        self.folder_path = Path(folder_path)
        self.min_duration = min_duration
        self.clips = []
        self.duration_index = []

    def scan_folder(self, extensions: List[str] = None, recursive: bool = False) -> List[Path]:
        """
        Find all video files in the folder.

        Args:
            extensions: List of video file extensions to include
            recursive: Whether to search subdirectories

        Returns:
            List of video file paths
        """
        if extensions is None:
            extensions = ['mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v']

        # Normalize extensions (remove leading dots, lowercase)
        extensions = [ext.lower().lstrip('.') for ext in extensions]

        video_files = []

        if recursive:
            for ext in extensions:
                video_files.extend(self.folder_path.rglob(f'*.{ext}'))
        else:
            for ext in extensions:
                video_files.extend(self.folder_path.glob(f'*.{ext}'))

        # Sort by filename for consistent ordering
        video_files = sorted(video_files, key=lambda p: p.name.lower())

        print(f"Found {len(video_files)} video files in {self.folder_path}")
        return video_files

    def get_video_metadata(self, video_path: Path) -> Optional[Dict]:
        """
        Extract metadata from a video file using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video metadata, or None if extraction fails
        """
        # Get duration and format info
        cmd_format = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration,size',
            '-of', 'json',
            str(video_path)
        ]

        # Get stream info (resolution, fps, codec)
        cmd_stream = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,codec_name,nb_frames',
            '-of', 'json',
            str(video_path)
        ]

        try:
            # Get format info
            result_format = subprocess.run(
                cmd_format, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL
            )
            format_data = json.loads(result_format.stdout)

            # Get stream info
            result_stream = subprocess.run(
                cmd_stream, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL
            )
            stream_data = json.loads(result_stream.stdout)

            # Extract values
            format_info = format_data.get('format', {})
            stream_info = stream_data.get('streams', [{}])[0] if stream_data.get('streams') else {}

            duration = float(format_info.get('duration', 0))
            file_size = int(format_info.get('size', 0))

            width = stream_info.get('width', 0)
            height = stream_info.get('height', 0)
            codec = stream_info.get('codec_name', 'unknown')

            # Parse frame rate
            fps_str = stream_info.get('r_frame_rate', '24/1')
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den) if float(den) > 0 else 24.0
            else:
                fps = float(fps_str) if fps_str else 24.0

            # Get frame count (may be missing for some formats)
            nb_frames = stream_info.get('nb_frames')
            if nb_frames:
                total_frames = int(nb_frames)
            else:
                # Estimate from duration and fps
                total_frames = int(duration * fps)

            return {
                'duration': duration,
                'width': width,
                'height': height,
                'fps': round(fps, 3),
                'codec': codec,
                'total_frames': total_frames,
                'file_size_bytes': file_size
            }

        except subprocess.CalledProcessError as e:
            print(f"  Warning: ffprobe failed for {video_path.name}: {e}")
            return None
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"  Warning: Could not parse metadata for {video_path.name}: {e}")
            return None

    def analyze_clips(self, video_files: List[Path]) -> List[Dict]:
        """
        Analyze all video clips and build metadata list.

        Args:
            video_files: List of video file paths

        Returns:
            List of clip metadata dictionaries
        """
        print(f"\nAnalyzing {len(video_files)} clips...")
        clips = []
        skipped = 0

        for i, video_path in enumerate(video_files):
            if (i + 1) % 10 == 0 or i == len(video_files) - 1:
                print(f"  Processing clip {i + 1}/{len(video_files)}: {video_path.name}")

            metadata = self.get_video_metadata(video_path)

            if metadata is None:
                print(f"  Skipping {video_path.name} (failed to read metadata)")
                skipped += 1
                continue

            if metadata['duration'] < self.min_duration:
                print(f"  Skipping {video_path.name} (duration {metadata['duration']:.3f}s < min {self.min_duration}s)")
                skipped += 1
                continue

            clip = {
                'clip_id': len(clips),
                'video_path': str(video_path.absolute()),
                'filename': video_path.name,
                **metadata
            }
            clips.append(clip)

        print(f"\nAnalyzed {len(clips)} valid clips (skipped {skipped})")
        return clips

    def build_duration_index(self):
        """Sort clips by duration (ascending) for efficient lookup."""
        # Create sorted list of (duration, clip_id) tuples
        duration_clip_pairs = [(clip['duration'], clip['clip_id']) for clip in self.clips]
        duration_clip_pairs.sort(key=lambda x: x[0])

        # Extract just the clip IDs in sorted order
        self.duration_index = [clip_id for _, clip_id in duration_clip_pairs]

    def save_database(self, output_path: str, append: bool = False):
        """
        Save database to JSON file.

        Args:
            output_path: Path for output JSON file
            append: If True, append to existing database
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build duration index
        self.build_duration_index()

        # Calculate statistics
        if self.clips:
            durations = [clip['duration'] for clip in self.clips]
            duration_stats = {
                'min': round(min(durations), 3),
                'max': round(max(durations), 3),
                'mean': round(sum(durations) / len(durations), 3),
                'total': round(sum(durations), 3)
            }
        else:
            duration_stats = {'min': 0, 'max': 0, 'mean': 0, 'total': 0}

        # Build new database entry
        new_data = {
            'source_folders': [{
                'folder_path': str(self.folder_path.absolute()),
                'num_clips': len(self.clips),
                'total_duration': duration_stats['total']
            }],
            'num_folders': 1,
            'num_clips': len(self.clips),
            'duration_range': duration_stats,
            'clips': self.clips,
            'duration_index': {
                'sorted_by_duration': self.duration_index
            }
        }

        # Handle append mode
        if append and output_path.exists():
            print(f"\nAppending to existing database: {output_path}")
            with open(output_path, 'r') as f:
                existing_data = json.load(f)

            # Calculate ID offset for new clips
            id_offset = existing_data['num_clips']

            # Update clip IDs
            for clip in self.clips:
                clip['clip_id'] += id_offset

            # Merge data
            existing_data['source_folders'].extend(new_data['source_folders'])
            existing_data['num_folders'] = len(existing_data['source_folders'])
            existing_data['clips'].extend(self.clips)
            existing_data['num_clips'] = len(existing_data['clips'])

            # Recalculate duration statistics
            all_durations = [clip['duration'] for clip in existing_data['clips']]
            existing_data['duration_range'] = {
                'min': round(min(all_durations), 3),
                'max': round(max(all_durations), 3),
                'mean': round(sum(all_durations) / len(all_durations), 3),
                'total': round(sum(all_durations), 3)
            }

            # Rebuild sorted index
            duration_clip_pairs = [(clip['duration'], clip['clip_id']) for clip in existing_data['clips']]
            duration_clip_pairs.sort(key=lambda x: x[0])
            existing_data['duration_index'] = {
                'sorted_by_duration': [clip_id for _, clip_id in duration_clip_pairs]
            }

            data_to_save = existing_data
        else:
            data_to_save = new_data

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)

        print(f"\nSaved database to: {output_path}")
        self.print_summary(data_to_save)

    def print_summary(self, data: Dict = None):
        """Print database statistics."""
        if data is None:
            data = {
                'num_clips': len(self.clips),
                'duration_range': {
                    'min': min(c['duration'] for c in self.clips) if self.clips else 0,
                    'max': max(c['duration'] for c in self.clips) if self.clips else 0,
                    'total': sum(c['duration'] for c in self.clips) if self.clips else 0
                }
            }

        print("\n" + "=" * 50)
        print("Duration Database Summary")
        print("=" * 50)
        print(f"  Total clips: {data['num_clips']}")
        print(f"  Duration range: {data['duration_range']['min']:.3f}s - {data['duration_range']['max']:.3f}s")
        print(f"  Average duration: {data['duration_range'].get('mean', 0):.3f}s")
        print(f"  Total duration: {data['duration_range']['total']:.1f}s ({data['duration_range']['total']/60:.1f} min)")

        if 'source_folders' in data:
            print(f"  Source folders: {len(data['source_folders'])}")
            for folder in data['source_folders']:
                print(f"    - {folder['folder_path']}: {folder['num_clips']} clips")


def main():
    parser = argparse.ArgumentParser(
        description='Build a duration database from a folder of video clips.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --folder /path/to/clips
  %(prog)s --folder /path/to/clips --extensions mp4,mov
  %(prog)s --folder /path/to/more_clips --append
  %(prog)s --folder /path/to/clips --recursive --min-duration 0.5
        """
    )

    parser.add_argument('--folder', required=True,
                        help='Folder containing video clips')
    parser.add_argument('--output', default='data/segments/duration_database.json',
                        help='Output JSON path (default: data/segments/duration_database.json)')
    parser.add_argument('--extensions', default='mp4,mov,avi,mkv,webm,m4v',
                        help='Comma-separated video extensions (default: mp4,mov,avi,mkv,webm,m4v)')
    parser.add_argument('--recursive', action='store_true',
                        help='Search subdirectories')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing database')
    parser.add_argument('--min-duration', type=float, default=0.1,
                        help='Minimum clip duration in seconds (default: 0.1)')

    args = parser.parse_args()

    # Validate folder
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"Error: Folder not found: {args.folder}")
        return 1
    if not folder_path.is_dir():
        print(f"Error: Not a directory: {args.folder}")
        return 1

    # Parse extensions
    extensions = [ext.strip() for ext in args.extensions.split(',')]

    # Create analyzer and run
    analyzer = DurationSourceAnalyzer(args.folder, min_duration=args.min_duration)
    video_files = analyzer.scan_folder(extensions=extensions, recursive=args.recursive)

    if not video_files:
        print(f"No video files found with extensions: {extensions}")
        return 1

    analyzer.clips = analyzer.analyze_clips(video_files)

    if not analyzer.clips:
        print("No valid clips to save")
        return 1

    analyzer.save_database(args.output, append=args.append)
    return 0


if __name__ == '__main__':
    exit(main())
