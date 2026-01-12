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

from edl_generator import EDLGenerator
import video_utils


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
        if stats.get('matched_rest_segments', 0) > 0:
            print(f"  Rests matched with clips: {stats['matched_rest_segments']}")
        print(f"  Unmatched: {stats['unmatched_segments']}")
        # Handle both old and new field names for backwards compatibility
        rest_count = stats.get('rest_segments_black_frames', stats.get('rest_segments', 0))
        print(f"  Rest segments (black frames): {rest_count}")
        print(f"  Total output duration: {stats['total_output_duration']:.1f}s")

    def get_video_resolution(self, video_path: str) -> Tuple[int, int]:
        """Get video resolution using ffprobe (cached)."""
        return video_utils.get_video_resolution(video_path, self.video_resolutions)

    def get_video_fps(self, video_path: str) -> float:
        """Get video frame rate using ffprobe (cached)."""
        return video_utils.get_video_fps(video_path, self.video_fps_cache)

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

    def prompt_for_resolution(self, analysis: Dict) -> Tuple[int, int]:
        """
        Prompt user to select output resolution.

        Args:
            analysis: Resolution analysis from analyze_source_videos()

        Returns:
            Tuple of (width, height) selected by user
        """
        return video_utils.prompt_for_resolution(analysis)

    def prompt_for_fps(self, analysis: Dict) -> float:
        """
        Prompt user to select output frame rate.

        Args:
            analysis: Analysis from analyze_source_videos()

        Returns:
            Frame rate selected by user
        """
        return video_utils.prompt_for_fps(analysis)

    def extract_clip(self, video_path: str, start_frame: int, fps: float,
                     target_duration: float, output_path: str):
        """
        Extract video clip using ffmpeg with accurate seeking.

        Args:
            video_path: Source video path
            start_frame: Start frame number
            fps: Video frame rate
            target_duration: Exact duration to extract (from guide)
            output_path: Output clip path
        """
        start_time = start_frame / fps

        if target_duration <= 0:
            target_duration = 0.1  # Minimum duration

        # Build ffmpeg command with output-seeking for accuracy
        # Use -ss after -i for frame-accurate seeking (slower but precise)
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-ss', f'{start_time:.6f}',  # Seek after input for accuracy
            '-t', f'{target_duration:.6f}',  # Use high precision duration
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
        video_utils.generate_black_clip(duration, output_path, width, height, fps)

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
                # Use the exact target duration from guide, not calculated from frames
                target_duration = source_clip.get('duration', match['guide_duration'])

                fps = self.get_video_fps(video_path)
                output_path = self.temp_dir / f"match_{match_idx:04d}_clip_{clip_idx:02d}.mp4"

                self.extract_clip(video_path, start_frame, fps, target_duration, str(output_path))
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
        return video_utils.is_valid_clip(clip_path)

    def _normalize_single_clip(self, args: Tuple) -> Tuple[int, Optional[str], Optional[str]]:
        """Normalize a single clip (for parallel processing)."""
        idx, clip_file, width, height, fps, temp_dir = args

        normalized_path = Path(temp_dir) / f"normalized_{idx:04d}.mp4"

        success, error_msg = video_utils.normalize_clip(
            clip_file, str(normalized_path), width, height
        )

        if success:
            return (idx, str(normalized_path), None)
        else:
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

        # Calculate expected total duration from match plan
        expected_duration = sum(m['guide_duration'] for m in self.matches)
        print(f"Expected output duration: {expected_duration:.4f}s")

        # Concatenate with re-encode
        # Use -r for exact frame rate and -t for exact duration
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-r', str(target_fps),  # Force exact frame rate
            '-t', f'{expected_duration:.6f}',  # Force exact duration
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-c:a', 'aac',
            '-ar', '44100',
            '-b:a', '320k',
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
                if skip_resolution_prompt:
                    self.target_width = analysis['min_width']
                    self.target_height = analysis['min_height']
                    print(f"\nUsing resolution: {self.target_width}x{self.target_height}")
                else:
                    self.target_width, self.target_height = self.prompt_for_resolution(analysis)

            if need_fps:
                if skip_fps_prompt:
                    self.target_fps = analysis['min_fps']
                    print(f"Using frame rate: {self.target_fps:.2f} fps")
                else:
                    self.target_fps = self.prompt_for_fps(analysis)

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

    def generate_edl(self, output_path: str, frame_rate: float = None) -> str:
        """
        Generate EDL file from match plan.

        Args:
            output_path: Path for output EDL file
            frame_rate: Frame rate for EDL timecode display (uses target_fps if None)

        Returns:
            Path to generated EDL file
        """
        if frame_rate is None:
            frame_rate = self.target_fps or 24.0

        print(f"\n=== Generating EDL ===")
        print(f"EDL frame rate: {frame_rate} fps")

        # Determine title from output path
        title = Path(output_path).stem

        edl = EDLGenerator(title, frame_rate=frame_rate)

        # Build a mapping of video paths to their fps for accurate source timecode
        video_fps_map = {}

        # Track timeline position using guide durations
        timeline_position = 0.0

        for match in self.matches:
            match_type = match.get('match_type', 'unknown')
            guide_duration = match.get('guide_duration', 0)
            source_clips = match.get('source_clips', [])

            # Check if this is a rest/unmatched with no clips assigned
            if (match_type == 'rest' or match_type == 'unmatched') and not source_clips:
                # Black/rest segment
                comment = f"Guide segment {match.get('guide_segment_id', '?')}"
                if match.get('is_rest'):
                    comment += " (REST)"
                edl.add_black(guide_duration, record_in=timeline_position, comment=comment)
            elif source_clips:
                # Video clip (including matched rests with --match-rests)
                clip = source_clips[0]
                video_path = clip.get('video_path', '')

                # Get source video fps (cached)
                if video_path not in video_fps_map:
                    video_fps_map[video_path] = self.get_video_fps(video_path)
                source_fps = video_fps_map[video_path]

                # Calculate source timecode from frames using source video's fps
                start_frame = clip.get('video_start_frame', 0)
                clip_duration = clip.get('duration', guide_duration)
                source_in = start_frame / source_fps
                source_out = source_in + clip_duration

                # Build comment
                comment_parts = [f"Guide segment {match.get('guide_segment_id', '?')}"]
                if match_type == 'rest':
                    comment_parts.append("(matched rest)")
                crop_mode = clip.get('crop_mode', match.get('crop_mode', ''))
                if crop_mode:
                    comment_parts.append(f"Crop: {crop_mode}")

                # Use guide_duration for timeline positioning (record IN/OUT)
                edl.add_event(
                    source_path=video_path,
                    source_in=source_in,
                    source_out=source_out,
                    record_in=timeline_position,
                    record_out=timeline_position + guide_duration,
                    comment=", ".join(comment_parts)
                )

            # Advance timeline by guide duration
            timeline_position += guide_duration

        # Write EDL
        edl_path = edl.write(output_path)
        print(f"EDL saved to: {edl_path}")
        print(f"  Events: {edl.event_count}")
        print(f"  Total duration: {edl.total_duration:.2f}s")

        return edl_path

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
    parser.add_argument('--edl', action='store_true',
                        help='Generate EDL file alongside video')
    parser.add_argument('--edl-only', action='store_true',
                        help='Generate EDL file only, skip video assembly')
    parser.add_argument('--edl-output', type=str,
                        help='Custom EDL output path (default: same as video with .edl extension)')

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

    # Determine EDL output path
    if args.edl_output:
        edl_output_path = args.edl_output
    else:
        edl_output_path = str(output_path.with_suffix('.edl'))

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

    # Handle EDL-only mode
    if args.edl_only:
        # For EDL-only, we need fps but don't need to process video
        fps = args.fps or 24.0
        assembler.generate_edl(edl_output_path, frame_rate=fps)
        return 0

    # Normal assembly
    assembler.assemble_video(
        skip_resolution_prompt=args.auto_resolution or target_width is not None,
        skip_fps_prompt=args.auto_fps or args.fps is not None
    )

    # Generate EDL if requested
    if args.edl:
        assembler.generate_edl(edl_output_path)

    if not args.no_cleanup:
        assembler.cleanup()

    return 0


if __name__ == '__main__':
    exit(main())
