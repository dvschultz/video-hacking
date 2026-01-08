#!/usr/bin/env python3
"""
Batch Pitch Analyzer with Parallel Processing

Processes multiple source videos in parallel to build a combined pitch database.
Each video is analyzed independently, then results are merged at the end.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


def analyze_single_video(args: Tuple) -> Tuple[str, Optional[Dict], Optional[str]]:
    """
    Analyze a single video file (worker function).

    Args:
        args: Tuple of (video_path, temp_dir, options_dict)

    Returns:
        Tuple of (video_path, result_dict or None, error_message or None)
    """
    video_path, temp_dir, options = args

    try:
        # Import here to avoid issues with multiprocessing
        from pitch_source_analyzer import PitchSourceAnalyzer

        video_name = Path(video_path).name

        # Create analyzer
        analyzer = PitchSourceAnalyzer(video_path, pitch_method=options['pitch_method'])

        # Extract audio
        analyzer.extract_audio(
            output_dir=temp_dir,
            normalize=options.get('normalize', False),
            target_lufs=options.get('target_lufs', -16.0)
        )

        # Analyze pitch
        pitch_segments = analyzer.analyze_continuous_pitch(
            fps=options['fps'],
            pitch_change_threshold=options['threshold'],
            min_segment_duration=options['min_duration'],
            min_rms_db=options['silence_threshold'],
            pitch_smoothing=options.get('pitch_smoothing', 0)
        )

        # Build database
        analyzer.build_database(pitch_segments)
        analyzer.build_pitch_index()

        # Return the data (not saved to file yet)
        result = {
            'video_path': str(video_path),
            'pitch_database': analyzer.pitch_database,
            'silence_segments': analyzer.silence_segments,
            'pitch_index': {str(k): v for k, v in analyzer.pitch_index.items()},
            'num_segments': len(analyzer.pitch_database),
            'num_silences': len(analyzer.silence_segments),
        }

        return (video_path, result, None)

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return (video_path, None, error_msg)


def merge_databases(results: List[Dict], output_path: str):
    """
    Merge multiple video analysis results into a single database.

    Args:
        results: List of result dictionaries from analyze_single_video
        output_path: Path to save merged database
    """
    print("\nMerging databases...")

    combined_database = []
    combined_silences = []
    combined_pitch_index = defaultdict(list)
    source_videos = []

    segment_id_offset = 0

    for result in results:
        video_path = result['video_path']
        print(f"  Adding: {Path(video_path).name} ({result['num_segments']} segments)")

        # Add video info
        source_videos.append({
            'video_path': video_path,
            'num_segments': result['num_segments'],
            'num_silences': result['num_silences']
        })

        # Renumber segment IDs and add to combined database
        for seg in result['pitch_database']:
            seg['segment_id'] += segment_id_offset
            seg['video_path'] = video_path
            combined_database.append(seg)

        # Add silence segments
        for silence in result['silence_segments']:
            silence['video_path'] = video_path
            combined_silences.append(silence)

        # Update pitch index with new segment IDs
        for midi_str, seg_ids in result['pitch_index'].items():
            midi = int(midi_str)
            new_ids = [sid + segment_id_offset for sid in seg_ids]
            combined_pitch_index[midi].extend(new_ids)

        segment_id_offset += result['num_segments']

    # Sort pitch index entries
    for midi in combined_pitch_index:
        combined_pitch_index[midi].sort()

    # Calculate statistics
    total_duration = sum(seg['duration'] for seg in combined_database)
    silence_duration = sum(seg['duration'] for seg in combined_silences)
    unique_pitches = len(combined_pitch_index)

    # Build final output
    output_data = {
        'num_videos': len(results),
        'num_segments': len(combined_database),
        'num_unique_pitches': unique_pitches,
        'num_silence_gaps': len(combined_silences),
        'total_musical_duration': total_duration,
        'total_silence_duration': silence_duration,
        'source_videos': source_videos,
        'pitch_database': combined_database,
        'pitch_index': {str(k): v for k, v in combined_pitch_index.items()},
        'silence_segments': combined_silences
    }

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDatabase saved to: {output_path}")
    print(f"  Videos: {len(results)}")
    print(f"  Total segments: {len(combined_database)}")
    print(f"  Unique pitches: {unique_pitches}")
    print(f"  Silence gaps: {len(combined_silences)}")


def find_video_files(folder: str, extensions: List[str]) -> List[str]:
    """Find all video files in folder with given extensions."""
    folder = Path(folder)
    video_files = []

    for ext in extensions:
        # Case-insensitive matching
        video_files.extend(folder.glob(f"*.{ext}"))
        video_files.extend(folder.glob(f"*.{ext.upper()}"))

    # Filter out hidden files and sort
    video_files = [
        str(f) for f in video_files
        if not f.name.startswith('.')
    ]
    video_files.sort()

    return video_files


def main():
    parser = argparse.ArgumentParser(
        description="Batch process videos for pitch analysis with parallel processing"
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Folder containing video files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/segments/source_database.json',
        help='Output database path'
    )
    parser.add_argument(
        '--extensions',
        type=str,
        default='mp4,mov,avi,mkv,webm',
        help='Video extensions to process (comma-separated)'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count / 2)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=24,
        help='Video frame rate'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=50.0,
        help='Pitch change threshold in cents'
    )
    parser.add_argument(
        '--min-duration',
        type=float,
        default=0.1,
        help='Minimum segment duration'
    )
    parser.add_argument(
        '--silence-threshold',
        type=float,
        default=-40.0,
        help='Silence detection threshold in dB'
    )
    parser.add_argument(
        '--pitch-smoothing',
        type=int,
        default=0,
        help='Pitch smoothing window size'
    )
    parser.add_argument(
        '--pitch-method',
        type=str,
        default='crepe',
        choices=['crepe', 'swift-f0', 'basic-pitch', 'hybrid', 'pyin'],
        help='Pitch detection method'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize audio loudness before analysis'
    )
    parser.add_argument(
        '--target-lufs',
        type=float,
        default=-16.0,
        help='Target loudness in LUFS when normalizing'
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default='data/temp',
        help='Temporary directory'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List files without processing'
    )

    args = parser.parse_args()

    print("=== Parallel Batch Pitch Analyzer ===\n")

    # Find video files
    extensions = [ext.strip() for ext in args.extensions.split(',')]
    video_files = find_video_files(args.folder, extensions)

    if not video_files:
        print(f"No video files found in {args.folder}")
        print(f"Extensions searched: {extensions}")
        sys.exit(1)

    print(f"Found {len(video_files)} video files:")
    for i, vf in enumerate(video_files):
        print(f"  {i+1}. {Path(vf).name}")
    print()

    if args.dry_run:
        print("Dry run - no files processed")
        sys.exit(0)

    # Determine worker count
    if args.parallel is None:
        # Default: half of CPU count, minimum 1, maximum 4
        # (pitch detection is CPU/GPU intensive, don't want to overload)
        workers = max(1, min((os.cpu_count() or 2) // 2, 4))
    else:
        workers = args.parallel

    print(f"Processing with {workers} parallel workers")
    print(f"Pitch method: {args.pitch_method}")
    if args.normalize:
        print(f"Audio normalization: ON (target {args.target_lufs} LUFS)")
    print()

    # Prepare options for workers
    options = {
        'fps': args.fps,
        'threshold': args.threshold,
        'min_duration': args.min_duration,
        'silence_threshold': args.silence_threshold,
        'pitch_smoothing': args.pitch_smoothing,
        'pitch_method': args.pitch_method,
        'normalize': args.normalize,
        'target_lufs': args.target_lufs,
    }

    # Create temp directory
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Prepare work items
    work_items = [
        (video_path, str(temp_dir), options)
        for video_path in video_files
    ]

    # Process in parallel
    results = []
    failed = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(analyze_single_video, item): item[0]
            for item in work_items
        }

        completed = 0
        for future in as_completed(futures):
            video_path = futures[future]
            video_name = Path(video_path).name
            completed += 1

            try:
                path, result, error = future.result()

                if result:
                    results.append(result)
                    print(f"[{completed}/{len(video_files)}] ✓ {video_name} "
                          f"({result['num_segments']} segments)")
                else:
                    failed.append((video_name, error))
                    print(f"[{completed}/{len(video_files)}] ✗ {video_name} - FAILED")
                    if error:
                        # Print first line of error
                        print(f"    {error.split(chr(10))[0][:80]}")

            except Exception as e:
                failed.append((video_name, str(e)))
                print(f"[{completed}/{len(video_files)}] ✗ {video_name} - FAILED: {e}")

    elapsed = time.time() - start_time

    print(f"\n{'='*50}")
    print(f"Processing complete in {elapsed:.1f}s")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print("\nFailed videos:")
        for name, error in failed:
            print(f"  - {name}")

    if results:
        # Merge and save
        merge_databases(results, args.output)
    else:
        print("\nNo videos were successfully processed!")
        sys.exit(1)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
