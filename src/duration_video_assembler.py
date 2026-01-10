#!/usr/bin/env python3
"""
Duration Video Assembler

Assembles final video from duration match plan.
Extracts clips from source videos using pre-calculated crop frames,
normalizes resolution/fps, and concatenates into final video.

Process:
1. Load match plan
2. For each match, extract video clip with cropping already applied
3. Generate black frames for rest segments
4. Normalize and concatenate all clips

Usage:
    python duration_video_assembler.py --match-plan plan.json --output video.mp4
"""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class DurationVideoAssembler:
    """Assembles video from duration match plan."""

    def __init__(self, match_plan_path: str, output_path: str, temp_dir: str = "data/temp",
                 target_width: int = None, target_height: int = None, target_fps: float = None,
                 parallel_workers: int = None, keep_audio: bool = True):
        """
        Initialize the assembler.

        Args:
            match_plan_path: Path to match plan JSON
            output_path: Path for output video
            temp_dir: Temporary directory for intermediate files
            target_width: Target video width (None = auto-detect)
            target_height: Target video height (None = auto-detect)
            target_fps: Target frame rate (None = auto-detect)
            parallel_workers: Number of parallel workers for normalization (None = auto)
            keep_audio: If True, preserve original audio from clips
        """
        self.match_plan_path = Path(match_plan_path)
        self.output_path = Path(output_path)
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Resolution settings
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps

        # Audio handling
        self.keep_audio = keep_audio

        # Parallel processing (default to CPU count, max 8 to avoid I/O saturation)
        if parallel_workers is None:
            parallel_workers = min(os.cpu_count() or 4, 8)
        self.parallel_workers = parallel_workers

        # Data
        self.match_plan = None
        self.matches = []
        self.clip_paths = []

        # Cached video metadata
        self.video_resolutions = {}
        self.video_fps_cache = {}

    def load_match_plan(self):
        """Load match plan from JSON."""
        print(f"Loading match plan from: {self.match_plan_path}")
        with open(self.match_plan_path, 'r') as f:
            self.match_plan = json.load(f)

        self.matches = self.match_plan['matches']
        print(f"  Loaded {len(self.matches)} matches")

        stats = self.match_plan['statistics']
        print(f"  Duration matches: {stats['matched_segments']}")
        print(f"  Unmatched: {stats['unmatched_segments']}")
        print(f"  Rest segments: {stats['rest_segments']}")
        print(f"  Total output duration: {stats['total_output_duration']:.1f}s")

    def get_video_resolution(self, video_path: str) -> Tuple[int, int]:
        """Get video resolution using ffprobe (cached)."""
        if video_path in self.video_resolutions:
            return self.video_resolutions[video_path]

        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
            width, height = map(int, result.stdout.strip().split(','))
            self.video_resolutions[video_path] = (width, height)
            return (width, height)
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"Warning: Could not get resolution for {video_path}: {e}")
            return (1920, 1080)

    def get_video_fps(self, video_path: str) -> float:
        """Get video frame rate using ffprobe (cached)."""
        if video_path in self.video_fps_cache:
            return self.video_fps_cache[video_path]

        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'csv=p=0',
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
            fps_str = result.stdout.strip()
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            self.video_fps_cache[video_path] = fps
            return fps
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"Warning: Could not get FPS for {video_path}: {e}")
            self.video_fps_cache[video_path] = 24.0
            return 24.0

    def analyze_source_videos(self) -> Dict:
        """Analyze all unique source videos to find resolutions and frame rates."""
        print("\nAnalyzing source videos...")

        video_paths = set()
        for match in self.matches:
            for clip in match.get('source_clips', []):
                if clip.get('video_path'):
                    video_paths.add(clip['video_path'])

        resolutions = []
        fps_values = []

        for video_path in video_paths:
            width, height = self.get_video_resolution(video_path)
            fps = self.get_video_fps(video_path)
            resolutions.append((width, height))
            fps_values.append(fps)
            print(f"  {Path(video_path).name}: {width}x{height} @ {fps:.2f}fps")

        if not resolutions:
            return {
                'min_width': 1920, 'min_height': 1080,
                'max_width': 1920, 'max_height': 1080,
                'min_fps': 24.0, 'max_fps': 24.0
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

    def extract_clip(self, video_path: str, start_frame: int, end_frame: int,
                     fps: float, output_path: str):
        """
        Extract video clip using ffmpeg with accurate seeking.

        Args:
            video_path: Source video path
            start_frame: Start frame number
            end_frame: End frame number (exclusive)
            fps: Video frame rate
            output_path: Output clip path
        """
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps

        if duration <= 0:
            duration = 0.1  # Minimum duration

        # Build ffmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
        ]

        if self.keep_audio:
            cmd.extend(['-c:a', 'aac', '-ar', '44100', '-b:a', '320k'])
        else:
            cmd.append('-an')

        cmd.append(str(output_path))

        try:
            subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting clip: {e}")
            if e.stderr:
                print(f"stderr: {e.stderr.decode()[:200]}")
            raise

    def generate_rest_clip(self, duration: float, output_path: str,
                           width: int = 1920, height: int = 1080, fps: float = 24):
        """Generate a black video clip with silence for rest segments."""
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', f'color=c=black:s={width}x{height}:r={fps}:d={duration}',
            '-f', 'lavfi',
            '-i', f'anullsrc=r=44100:cl=stereo:d={duration}',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-c:a', 'aac',
            '-ar', '44100',
            '-b:a', '320k',
            '-t', str(duration),
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error generating rest clip: {e}")
            if e.stderr:
                print(f"stderr: {e.stderr.decode()[:200]}")
            raise

    def process_match(self, match: Dict, match_idx: int) -> List[str]:
        """
        Process a single match and create clip file(s).

        Args:
            match: Match dictionary from match plan
            match_idx: Index of this match

        Returns:
            List of created clip file paths
        """
        clips = []
        match_type = match.get('match_type', 'unknown')

        if match_type == 'rest':
            # Generate black frame with silence
            duration = match['guide_duration']
            output_path = self.temp_dir / f"match_{match_idx:04d}_rest.mp4"

            width = self.target_width or 1920
            height = self.target_height or 1080
            fps = self.target_fps or 24

            self.generate_rest_clip(duration, str(output_path), width, height, fps)
            clips.append(str(output_path))

        elif match_type == 'duration':
            # Extract cropped clip
            for clip_idx, source_clip in enumerate(match.get('source_clips', [])):
                video_path = source_clip['video_path']
                start_frame = source_clip['video_start_frame']
                end_frame = source_clip['video_end_frame']

                fps = self.get_video_fps(video_path)
                output_path = self.temp_dir / f"match_{match_idx:04d}_clip_{clip_idx:02d}.mp4"

                self.extract_clip(video_path, start_frame, end_frame, fps, str(output_path))
                clips.append(str(output_path))

        elif match_type == 'unmatched':
            # Generate black frame for unmatched segment
            duration = match['guide_duration']
            output_path = self.temp_dir / f"match_{match_idx:04d}_unmatched.mp4"

            width = self.target_width or 1920
            height = self.target_height or 1080
            fps = self.target_fps or 24

            self.generate_rest_clip(duration, str(output_path), width, height, fps)
            clips.append(str(output_path))

        return clips

    def is_valid_clip(self, clip_path: str) -> bool:
        """Check if a clip file is valid and has non-zero duration."""
        if not Path(clip_path).exists():
            return False

        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                str(clip_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
            duration = float(result.stdout.strip())
            return duration > 0.01
        except (subprocess.CalledProcessError, ValueError):
            return False

    def _normalize_single_clip(self, args: Tuple) -> Tuple[int, Optional[str], Optional[str]]:
        """Normalize a single clip (for parallel processing)."""
        idx, clip_file, width, height, fps, temp_dir = args

        normalized_path = Path(temp_dir) / f"normalized_{idx:04d}.mp4"

        cmd = [
            'ffmpeg', '-y',
            '-i', str(clip_file),
            '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps}',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-c:a', 'aac',
            '-ar', '44100',
            '-ac', '2',
            '-b:a', '320k',
            str(normalized_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL)
            return (idx, str(normalized_path), None)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode()[:200] if e.stderr else "Unknown error"
            return (idx, None, error_msg)

    def concatenate_clips(self, clip_files: List[str]):
        """Concatenate all clips into final video with normalization."""
        target_width = self.target_width or 1920
        target_height = self.target_height or 1080
        target_fps = self.target_fps or 24

        print(f"\nConcatenating {len(clip_files)} clips...")
        print(f"Normalizing to {target_width}x{target_height} @ {target_fps}fps")
        print(f"Using {self.parallel_workers} parallel workers")

        # Validate clips
        valid_clips = []
        skipped_count = 0
        for i, clip_file in enumerate(clip_files):
            if self.is_valid_clip(clip_file):
                valid_clips.append((i, clip_file))
            else:
                print(f"  Warning: Skipping invalid clip {i}: {Path(clip_file).name}")
                skipped_count += 1

        if skipped_count > 0:
            print(f"  Skipped {skipped_count} invalid clips, {len(valid_clips)} remaining")

        if not valid_clips:
            print("ERROR: No valid clips to concatenate")
            return

        # Prepare arguments for parallel normalization
        normalize_args = [
            (idx, clip_file, target_width, target_height, target_fps, self.temp_dir)
            for idx, clip_file in valid_clips
        ]

        # Normalize clips in parallel
        results = {}
        failed_count = 0

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = {
                executor.submit(self._normalize_single_clip, args): args[0]
                for args in normalize_args
            }

            for future in as_completed(futures):
                idx, normalized_path, error = future.result()
                if normalized_path:
                    results[idx] = normalized_path
                else:
                    print(f"  Warning: Error normalizing clip {idx}, skipping")
                    if error:
                        print(f"    {error[:100]}")
                    failed_count += 1

        # Sort results by original index
        normalized_clips = [results[idx] for idx in sorted(results.keys())]

        print(f"Normalized {len(normalized_clips)} clips ({failed_count} failed)")

        if not normalized_clips:
            print("ERROR: No clips were successfully normalized")
            return

        # Create concat file
        concat_file = self.temp_dir / "concat_list.txt"
        with open(concat_file, 'w') as f:
            for clip_file in normalized_clips:
                f.write(f"file '{Path(clip_file).absolute()}'\n")

        # Concatenate
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            str(self.output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL)
            print(f"Final video saved to: {self.output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error concatenating clips: {e}")
            if e.stderr:
                print(f"stderr: {e.stderr.decode()[:200]}")
            raise
        finally:
            # Clean up
            if concat_file.exists():
                concat_file.unlink()
            for clip in normalized_clips:
                Path(clip).unlink(missing_ok=True)

    def assemble_video(self, skip_resolution_prompt: bool = False, skip_fps_prompt: bool = False):
        """
        Main assembly process.

        Args:
            skip_resolution_prompt: If True, auto-select smallest resolution
            skip_fps_prompt: If True, auto-select smallest fps
        """
        print("\n=== Assembling Video ===\n")

        # Analyze source videos
        need_resolution = self.target_width is None or self.target_height is None
        need_fps = self.target_fps is None

        if need_resolution or need_fps:
            analysis = self.analyze_source_videos()

            if need_resolution:
                self.target_width = analysis['min_width']
                self.target_height = analysis['min_height']
                print(f"\nUsing resolution: {self.target_width}x{self.target_height}")

            if need_fps:
                self.target_fps = analysis['min_fps']
                print(f"Using frame rate: {self.target_fps:.2f} fps")

        # Process all matches
        all_clips = []
        total_matches = len(self.matches)

        for i, match in enumerate(self.matches):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Processing match {i+1}/{total_matches}...")

            clips = self.process_match(match, i)
            all_clips.extend(clips)

        print(f"\nCreated {len(all_clips)} total clips")

        if not all_clips:
            print("ERROR: No clips were created. Cannot assemble video.")
            return

        # Concatenate
        self.concatenate_clips(all_clips)

        print("\n=== Video Assembly Complete ===")

    def cleanup(self):
        """Clean up temporary files."""
        print("\nCleaning up temporary files...")

        for pattern in ['match_*.mp4', 'normalized_*.mp4']:
            for temp_file in self.temp_dir.glob(pattern):
                try:
                    temp_file.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete {temp_file}: {e}")

        print("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description='Assemble final video from duration match plan.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --match-plan plan.json
  %(prog)s --match-plan plan.json --output video.mp4
  %(prog)s --match-plan plan.json --resolution 1920x1080 --fps 30
  %(prog)s --match-plan plan.json --auto-resolution --auto-fps
        """
    )

    parser.add_argument('--match-plan', required=True,
                        help='Match plan JSON path')
    parser.add_argument('--output', default='data/output/duration_matched_video.mp4',
                        help='Output video path (default: data/output/duration_matched_video.mp4)')
    parser.add_argument('--temp-dir', default='data/temp',
                        help='Temporary directory (default: data/temp)')
    parser.add_argument('--resolution', type=str,
                        help='Output resolution WxH (e.g., 1920x1080)')
    parser.add_argument('--auto-resolution', action='store_true',
                        help='Use smallest source resolution')
    parser.add_argument('--fps', type=float,
                        help='Output frame rate')
    parser.add_argument('--auto-fps', action='store_true',
                        help='Use smallest source fps')
    parser.add_argument('--parallel', type=int,
                        help='Number of parallel workers (default: auto)')
    parser.add_argument('--no-audio', action='store_true',
                        help='Output video only, no audio')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Keep temp files for debugging')

    args = parser.parse_args()

    # Validate match plan
    if not Path(args.match_plan).exists():
        print(f"Error: Match plan not found: {args.match_plan}")
        return 1

    # Parse resolution
    target_width = None
    target_height = None
    if args.resolution:
        try:
            target_width, target_height = map(int, args.resolution.lower().split('x'))
        except ValueError:
            print(f"Error: Invalid resolution format: {args.resolution}")
            print("Expected format: WxH (e.g., 1920x1080)")
            return 1

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create assembler
    assembler = DurationVideoAssembler(
        args.match_plan,
        args.output,
        temp_dir=args.temp_dir,
        target_width=target_width,
        target_height=target_height,
        target_fps=args.fps,
        parallel_workers=args.parallel,
        keep_audio=not args.no_audio
    )

    assembler.load_match_plan()
    assembler.assemble_video(
        skip_resolution_prompt=args.auto_resolution or target_width is not None,
        skip_fps_prompt=args.auto_fps or args.fps is not None
    )

    if not args.no_cleanup:
        assembler.cleanup()

    return 0


if __name__ == '__main__':
    exit(main())
