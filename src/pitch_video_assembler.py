#!/usr/bin/env python3
"""
Pitch Video Assembler

Assembles final video from pitch match plan.
Extracts clips from source videos, applies pitch transposition,
handles duration (trim/loop/combine), and concatenates into final video.

Process:
1. Load match plan
2. For each match, extract video clip from source
3. Extract and transpose audio if needed
4. Handle duration (trim, loop, or combine clips)
5. Concatenate all clips into final video
"""

import argparse
import json
import os
import numpy as np
import librosa
import soundfile as sf
import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class PitchVideoAssembler:
    """Assembles video from pitch match plan."""

    def __init__(self, match_plan_path: str, output_path: str, temp_dir: str = "data/temp",
                 target_width: int = None, target_height: int = None, target_fps: int = 24,
                 parallel_workers: int = None, use_true_silence: bool = False):
        """
        Initialize the assembler.

        Args:
            match_plan_path: Path to match plan JSON
            output_path: Path for output video
            temp_dir: Temporary directory for intermediate files
            target_width: Target video width (None = auto-detect)
            target_height: Target video height (None = auto-detect)
            target_fps: Target frame rate
            parallel_workers: Number of parallel workers for normalization (None = auto)
            use_true_silence: If True, use black frames with muted audio for rests
                              instead of source "silence" clips (default: False)
        """
        self.match_plan_path = Path(match_plan_path)
        self.output_path = Path(output_path)
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Resolution settings
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps

        # Silence handling
        self.use_true_silence = use_true_silence

        # Parallel processing (default to CPU count, max 8 to avoid I/O saturation)
        if parallel_workers is None:
            parallel_workers = min(os.cpu_count() or 4, 8)
        self.parallel_workers = parallel_workers

        # Data
        self.match_plan = None
        self.matches = []
        self.clip_paths = []  # List of temporary clip files

        # Cached video metadata (populated once, reused everywhere)
        self.video_resolutions = {}  # video_path -> (width, height)
        self.video_fps_cache = {}    # video_path -> fps

    def load_match_plan(self):
        """Load match plan from JSON."""
        print(f"Loading match plan from: {self.match_plan_path}")
        with open(self.match_plan_path, 'r') as f:
            self.match_plan = json.load(f)

        self.matches = self.match_plan['matches']
        print(f"  Loaded {len(self.matches)} matches")

        stats = self.match_plan['statistics']
        print(f"  Exact matches: {stats['exact_matches']}")
        print(f"  Transposed matches: {stats['transposed_matches']}")
        print(f"  Missing matches: {stats['missing_matches']}")
        if 'rest_segments' in stats:
            print(f"  Rest segments: {stats['rest_segments']}")

    def get_video_resolution(self, video_path: str) -> Tuple[int, int]:
        """
        Get video resolution using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (width, height)
        """
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
            return (1920, 1080)  # Default fallback

    def get_video_fps(self, video_path: str) -> float:
        """
        Get video frame rate using ffprobe (cached).

        Args:
            video_path: Path to video file

        Returns:
            Frame rate as float
        """
        # Check cache first
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
            return 24.0  # Default fallback

    def analyze_source_videos(self) -> Dict:
        """
        Analyze all unique source videos to find their resolutions and frame rates.

        Returns:
            Dictionary with resolution and fps analysis
        """
        print("\nAnalyzing source videos...")

        # Collect unique video paths from matches
        video_paths = set()
        for match in self.matches:
            for clip in match.get('source_clips', []):
                if 'video_path' in clip:
                    video_paths.add(clip['video_path'])

        print(f"  Found {len(video_paths)} unique source videos")

        # Get resolution and fps for each video
        resolutions = {}
        framerates = {}
        for video_path in video_paths:
            width, height = self.get_video_resolution(video_path)
            fps = self.get_video_fps(video_path)
            resolutions[video_path] = (width, height)
            framerates[video_path] = fps

        # Analyze resolutions
        widths = [r[0] for r in resolutions.values()]
        heights = [r[1] for r in resolutions.values()]
        unique_resolutions = set(resolutions.values())

        # Analyze frame rates (round to common values)
        fps_values = list(framerates.values())
        unique_fps = sorted(set(round(fps, 2) for fps in fps_values))

        analysis = {
            'resolutions': resolutions,
            'framerates': framerates,
            'unique_resolutions': sorted(unique_resolutions),
            'unique_fps': unique_fps,
            'min_width': min(widths) if widths else 1920,
            'max_width': max(widths) if widths else 1920,
            'min_height': min(heights) if heights else 1080,
            'max_height': max(heights) if heights else 1080,
            'min_fps': min(fps_values) if fps_values else 24,
            'max_fps': max(fps_values) if fps_values else 24,
        }

        print(f"\n  Resolutions found:")
        print(f"    Range: {analysis['min_width']}x{analysis['min_height']} to {analysis['max_width']}x{analysis['max_height']}")
        for res in sorted(unique_resolutions):
            count = sum(1 for r in resolutions.values() if r == res)
            print(f"    {res[0]}x{res[1]}: {count} video(s)")

        print(f"\n  Frame rates found:")
        for fps in unique_fps:
            count = sum(1 for f in fps_values if round(f, 2) == fps)
            print(f"    {fps:.2f} fps: {count} video(s)")

        return analysis

    def prompt_for_resolution(self, analysis: Dict) -> Tuple[int, int]:
        """
        Prompt user to select output resolution.

        Args:
            analysis: Resolution analysis from analyze_source_resolutions()

        Returns:
            Tuple of (width, height) selected by user
        """
        # Reset terminal settings in case subprocess messed them up
        os.system('stty sane 2>/dev/null')

        print("\n=== Select Output Resolution ===")
        print("")

        # Build options
        options = []

        # Option 1: Smallest dimensions (recommended for no upscaling)
        min_w, min_h = analysis['min_width'], analysis['min_height']
        options.append((min_w, min_h, "smallest (no upscaling)"))

        # Add unique resolutions as options
        for res in sorted(analysis['unique_resolutions'], reverse=True):
            if res != (min_w, min_h):
                options.append((res[0], res[1], "from sources"))

        # Common presets if not already in list
        presets = [(1920, 1080, "1080p"), (1280, 720, "720p"), (3840, 2160, "4K")]
        for w, h, name in presets:
            if not any(o[0] == w and o[1] == h for o in options):
                options.append((w, h, name))

        # Display options
        for i, (w, h, desc) in enumerate(options):
            marker = " (recommended)" if i == 0 else ""
            print(f"  {i + 1}. {w}x{h} - {desc}{marker}")
        print(f"  {len(options) + 1}. Custom resolution")

        # Get user input
        while True:
            try:
                choice = input(f"\nSelect option [1-{len(options) + 1}] (default: 1): ").strip()
                if choice == "":
                    choice = 1
                else:
                    choice = int(choice)

                if 1 <= choice <= len(options):
                    selected = options[choice - 1]
                    print(f"\nUsing resolution: {selected[0]}x{selected[1]}")
                    return (selected[0], selected[1])
                elif choice == len(options) + 1:
                    custom = input("Enter resolution (WxH, e.g., 1280x720): ").strip()
                    w, h = map(int, custom.lower().split('x'))
                    print(f"\nUsing custom resolution: {w}x{h}")
                    return (w, h)
                else:
                    print(f"Invalid choice. Please enter 1-{len(options) + 1}")
            except (ValueError, KeyboardInterrupt):
                print("\nUsing default (smallest resolution)")
                return (min_w, min_h)

    def prompt_for_fps(self, analysis: Dict) -> float:
        """
        Prompt user to select output frame rate.

        Args:
            analysis: Analysis from analyze_source_videos()

        Returns:
            Frame rate selected by user
        """
        # Reset terminal settings in case subprocess messed them up
        os.system('stty sane 2>/dev/null')

        print("\n=== Select Output Frame Rate ===")
        print("")

        # Build options
        options = []

        # Common frame rates
        common_fps = [23.976, 24, 25, 29.97, 30, 50, 59.94, 60]

        # Add source frame rates first
        for fps in sorted(analysis['unique_fps']):
            label = f"{fps:.2f} fps (from sources)"
            if fps == analysis['min_fps']:
                label = f"{fps:.2f} fps (smallest from sources)"
            options.append((fps, label))

        # Add common presets not already in list
        for fps in common_fps:
            # Check if close to any existing option
            if not any(abs(o[0] - fps) < 0.1 for o in options):
                options.append((fps, f"{fps:.2f} fps"))

        # Sort by fps
        options = sorted(options, key=lambda x: x[0])

        # Move smallest source fps to top as recommended
        min_fps = analysis['min_fps']
        for i, (fps, label) in enumerate(options):
            if abs(fps - min_fps) < 0.1:
                options.insert(0, options.pop(i))
                break

        # Display options
        for i, (fps, desc) in enumerate(options):
            marker = " (recommended)" if i == 0 else ""
            print(f"  {i + 1}. {desc}{marker}")
        print(f"  {len(options) + 1}. Custom frame rate")

        # Get user input
        while True:
            try:
                choice = input(f"\nSelect option [1-{len(options) + 1}] (default: 1): ").strip()
                if choice == "":
                    choice = 1
                else:
                    choice = int(choice)

                if 1 <= choice <= len(options):
                    selected = options[choice - 1]
                    print(f"\nUsing frame rate: {selected[0]:.2f} fps")
                    return selected[0]
                elif choice == len(options) + 1:
                    custom = input("Enter frame rate (e.g., 30): ").strip()
                    fps = float(custom)
                    print(f"\nUsing custom frame rate: {fps:.2f} fps")
                    return fps
                else:
                    print(f"Invalid choice. Please enter 1-{len(options) + 1}")
            except (ValueError, KeyboardInterrupt):
                print("\nUsing default (smallest source fps)")
                return min_fps

    def extract_video_clip(self, video_path: str, start_frame: int, end_frame: int,
                          fps: int, output_path: str):
        """
        Extract video clip using ffmpeg.

        Args:
            video_path: Source video path
            start_frame: Start frame number
            end_frame: End frame number (exclusive)
            fps: Video frame rate
            output_path: Output clip path
        """
        # Convert frames to time
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps

        # Use ffmpeg to extract clip
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', str(video_path),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-an',  # No audio for now (we'll add transposed audio separately)
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting video clip: {e}")
            print(f"stderr: {e.stderr.decode()}")
            raise

    def extract_clip_with_audio(self, video_path: str, start_frame: int, end_frame: int,
                               fps: int, output_path: str):
        """
        Extract video clip with original audio (for silence segments).

        Args:
            video_path: Source video path
            start_frame: Start frame number
            end_frame: End frame number
            fps: Video frame rate
            output_path: Output clip path
        """
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps

        if duration <= 0:
            duration = 0.1  # Minimum duration

        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', str(video_path),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-c:a', 'aac',
            '-b:a', '320k',
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting clip with audio: {e}")
            print(f"stderr: {e.stderr.decode()}")
            raise

    def generate_rest_clip(self, duration: float, output_path: str,
                          width: int = 1920, height: int = 1080, fps: int = 24):
        """
        Generate a black video clip with silence for rest segments.

        Args:
            duration: Duration in seconds
            output_path: Output clip path
            width: Video width (default 1920)
            height: Video height (default 1080)
            fps: Frame rate (default 24)
        """
        # Generate black video with silent audio
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
            '-b:a', '320k',
            '-shortest',
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error generating rest clip: {e}")
            print(f"stderr: {e.stderr.decode()}")
            raise

    def extract_and_transpose_audio(self, video_path: str, start_frame: int, end_frame: int,
                                    fps: int, transpose_semitones: int,
                                    output_path: str, sample_rate: int = 22050):
        """
        Extract audio clip and transpose pitch if needed.

        Args:
            video_path: Source video path
            start_frame: Start frame number
            end_frame: End frame number
            fps: Video frame rate
            transpose_semitones: Semitones to transpose (0 = no transpose)
            output_path: Output audio path
            sample_rate: Audio sample rate
        """
        # Convert frames to time
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps

        # Extract audio to temporary file
        temp_audio = self.temp_dir / f"temp_audio_{Path(output_path).stem}.wav"

        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', str(video_path),
            '-t', str(duration),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '1',  # Mono
            str(temp_audio)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio: {e}")
            print(f"stderr: {e.stderr.decode()}")
            raise

        # Load audio
        audio, sr = librosa.load(str(temp_audio), sr=sample_rate)

        # Transpose if needed
        if transpose_semitones != 0:
            print(f"    Transposing by {transpose_semitones:+d} semitones")
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=transpose_semitones)

        # Save transposed audio
        sf.write(str(output_path), audio, sr)

        # Clean up temp file
        temp_audio.unlink()

    def combine_video_and_audio(self, video_path: str, audio_path: str, output_path: str):
        """
        Combine video and audio files.

        Args:
            video_path: Video file path (no audio)
            audio_path: Audio file path
            output_path: Output combined video path
        """
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-i', str(audio_path),
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error combining video and audio: {e}")
            print(f"stderr: {e.stderr.decode()}")
            raise

    def _merge_short_clips(self, source_clips: List[Dict], min_duration: float = 0.04) -> List[Dict]:
        """
        Merge clips shorter than min_duration with adjacent clips.

        Short clips are extended by borrowing frames from the previous clip's end
        or the next clip's start. If a clip is still too short after merging,
        it's combined with its neighbor.

        Args:
            source_clips: List of clip dictionaries with video_start_frame, video_end_frame
            min_duration: Minimum clip duration in seconds (default: 0.04s = 1 frame at 24fps)

        Returns:
            List of merged clip dictionaries
        """
        if not source_clips:
            return source_clips

        # Group clips by video path (can only merge clips from same video)
        # For now, process all clips together since they're usually from same video
        merged = []
        pending_short = None  # Accumulate short clips to merge with next valid clip

        for clip in source_clips:
            video_path = clip['video_path']
            start_frame = int(clip['video_start_frame'])
            end_frame = int(clip['video_end_frame'])

            # Get fps for duration calculation
            fps = self.get_video_fps(video_path)
            clip_duration = (end_frame - start_frame) / fps

            if clip_duration < min_duration:
                # This clip is too short
                if merged:
                    # Extend previous clip's end to include this short clip
                    prev = merged[-1]
                    if prev['video_path'] == video_path:
                        # Use max to avoid shrinking previous clip if short clip is from looped segment
                        prev['video_end_frame'] = max(int(prev['video_end_frame']), end_frame)
                        prev['duration'] = (prev['video_end_frame'] - prev['video_start_frame']) / fps
                        print(f"    Merged short clip ({clip_duration:.3f}s) into previous clip")
                        continue

                # No previous clip to merge with - save for merging with next
                if pending_short is None:
                    pending_short = clip.copy()
                else:
                    # Extend pending short clip
                    pending_short['video_end_frame'] = end_frame
                continue

            # This clip is long enough
            if pending_short is not None:
                # Merge pending short clip into this one by extending start backwards
                if pending_short['video_path'] == video_path:
                    start_frame = int(pending_short['video_start_frame'])
                    print(f"    Merged pending short clip into current clip")
                pending_short = None

            # Add this clip (possibly with extended start)
            new_clip = clip.copy()
            new_clip['video_start_frame'] = start_frame
            new_clip['video_end_frame'] = end_frame
            new_clip['duration'] = (end_frame - start_frame) / fps
            merged.append(new_clip)

        # Handle any remaining pending short clip
        if pending_short is not None:
            if merged:
                # Extend last clip to include pending
                last = merged[-1]
                if last['video_path'] == pending_short['video_path']:
                    # Use max to avoid shrinking if pending is from looped segment
                    last['video_end_frame'] = max(int(last['video_end_frame']), int(pending_short['video_end_frame']))
                    fps = self.get_video_fps(last['video_path'])
                    last['duration'] = (last['video_end_frame'] - last['video_start_frame']) / fps
                    print(f"    Merged trailing short clip into previous clip")
                else:
                    # Different video, have to include as-is
                    merged.append(pending_short)
            else:
                # Only short clips, include as-is
                merged.append(pending_short)

        return merged

    def process_match(self, match: Dict, match_idx: int) -> List[str]:
        """
        Process a single match and create clip(s).

        Args:
            match: Match dictionary from match plan
            match_idx: Index of this match

        Returns:
            List of clip file paths (may be multiple if looped/combined)
        """
        if match['match_type'] == 'missing':
            print(f"  Match {match_idx}: MISSING - {match['guide_pitch_note']} - skipping")
            return []

        # Handle rest segments
        if match['match_type'] == 'rest':
            duration = match['guide_duration']
            source_clips = match.get('source_clips', [])

            # Use true silence (black frames + muted audio) if requested
            if self.use_true_silence:
                rest_clip = self.temp_dir / f"match_{match_idx:04d}_rest.mp4"
                print(f"  Match {match_idx}: REST ({duration:.3f}s) [true silence]")

                width = self.target_width or 1920
                height = self.target_height or 1080
                fps = self.target_fps or 24

                self.generate_rest_clip(duration, str(rest_clip), width=width, height=height, fps=fps)
                return [str(rest_clip)]

            # Otherwise use verified silence clips from source database
            if source_clips:
                # Merge short silence clips with neighbors
                source_clips = self._merge_short_clips(source_clips)

                print(f"  Match {match_idx}: REST ({duration:.3f}s, {len(source_clips)} silence clip(s))")
                clip_files = []

                for clip_idx, clip in enumerate(source_clips):
                    video_path = clip['video_path']
                    start_frame = int(clip['video_start_frame'])
                    end_frame = int(clip['video_end_frame'])

                    # Get actual fps from source video
                    fps = self.get_video_fps(video_path)

                    # Validate clip duration
                    clip_duration = (end_frame - start_frame) / fps
                    if clip_duration <= 0:
                        print(f"    WARNING: Silence clip {clip_idx} has invalid duration: {clip_duration:.3f}s")
                        continue

                    # Extract silence clip (video with original silent audio)
                    temp_clip = self.temp_dir / f"match_{match_idx:04d}_rest_{clip_idx:02d}.mp4"
                    self.extract_clip_with_audio(video_path, start_frame, end_frame, fps, str(temp_clip))
                    clip_files.append(str(temp_clip))

                if clip_files:
                    return clip_files

            # Fallback to black frames if no silence clips available
            rest_clip = self.temp_dir / f"match_{match_idx:04d}_rest.mp4"
            print(f"  Match {match_idx}: REST ({duration:.3f}s) [black frames - no silence clips]")

            width = self.target_width or 1920
            height = self.target_height or 1080
            fps = self.target_fps or 24

            self.generate_rest_clip(duration, str(rest_clip), width=width, height=height, fps=fps)
            return [str(rest_clip)]

        source_clips = match['source_clips']
        if not source_clips:
            print(f"  Match {match_idx}: No source clips - skipping")
            return []

        # Merge short clips with neighbors to avoid dropping frames
        source_clips = self._merge_short_clips(source_clips)
        if not source_clips:
            print(f"  Match {match_idx}: No valid clips after merging - skipping")
            return []

        clip_files = []
        transpose_semitones = match.get('transpose_semitones', 0)

        print(f"  Match {match_idx}: {match['guide_pitch_note']} "
              f"({match['match_type']}, {len(source_clips)} clip(s))", end='')
        if transpose_semitones != 0:
            print(f" [transpose {transpose_semitones:+d}]", end='')
        print()

        # Debug: show clip details if multiple clips
        if len(source_clips) > 1:
            for ci, sc in enumerate(source_clips):
                sf = sc.get('video_start_frame', '?')
                ef = sc.get('video_end_frame', '?')
                dur = sc.get('duration', '?')
                print(f"    clip[{ci}]: frames {sf}-{ef}, dur={dur}")

        for clip_idx, clip in enumerate(source_clips):
            video_path = clip['video_path']
            start_frame = int(clip['video_start_frame'])
            end_frame = int(clip['video_end_frame'])

            # Get actual fps from source video (uses class-level cache)
            fps = self.get_video_fps(video_path)

            # Calculate duration and validate
            clip_duration = (end_frame - start_frame) / fps
            if clip_duration <= 0:
                print(f"    WARNING: Clip {clip_idx} has invalid duration: {clip_duration:.3f}s "
                      f"(frames {start_frame}-{end_frame})")
                continue

            # Create temporary video clip (no audio)
            temp_video = self.temp_dir / f"match_{match_idx:04d}_clip_{clip_idx:02d}_video.mp4"
            self.extract_video_clip(video_path, start_frame, end_frame, fps, temp_video)

            # Create temporary audio clip (with transposition if needed)
            temp_audio = self.temp_dir / f"match_{match_idx:04d}_clip_{clip_idx:02d}_audio.wav"
            self.extract_and_transpose_audio(video_path, start_frame, end_frame, fps,
                                            transpose_semitones, temp_audio)

            # Combine video and audio
            final_clip = self.temp_dir / f"match_{match_idx:04d}_clip_{clip_idx:02d}.mp4"
            self.combine_video_and_audio(temp_video, temp_audio, final_clip)

            clip_files.append(str(final_clip))

            # Clean up intermediate files
            temp_video.unlink()
            temp_audio.unlink()

        return clip_files

    def is_valid_clip(self, clip_path: str) -> bool:
        """
        Check if a clip file is valid (has video stream and non-zero duration).

        Args:
            clip_path: Path to clip file

        Returns:
            True if clip is valid, False otherwise
        """
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=duration',
            '-of', 'csv=p=0',
            str(clip_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration_str = result.stdout.strip()

            # Check if we got a valid duration
            if not duration_str or duration_str == 'N/A':
                # Try format duration instead
                cmd2 = [
                    'ffprobe', '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'csv=p=0',
                    str(clip_path)
                ]
                result2 = subprocess.run(cmd2, capture_output=True, text=True, check=True)
                duration_str = result2.stdout.strip()

                if not duration_str or duration_str == 'N/A':
                    return False

            duration = float(duration_str)
            return duration > 0.01  # At least 10ms

        except (subprocess.CalledProcessError, ValueError):
            return False

    def _normalize_single_clip(self, args: Tuple[int, str, int, int, float, Path]) -> Tuple[int, Optional[str], Optional[str]]:
        """
        Normalize a single clip (worker function for parallel processing).

        Args:
            args: Tuple of (index, clip_file, width, height, fps, temp_dir)

        Returns:
            Tuple of (index, normalized_path or None, error_message or None)
        """
        idx, clip_file, width, height, fps, temp_dir = args
        normalized_path = temp_dir / f"normalized_{idx:05d}.mp4"

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
            subprocess.run(cmd, check=True, capture_output=True)
            return (idx, str(normalized_path), None)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode()[:200] if e.stderr else "Unknown error"
            return (idx, None, error_msg)

    def concatenate_clips(self, clip_files: List[str], output_path: str):
        """
        Concatenate all clips into final video.
        Normalizes all clips to same resolution/fps to handle mixed sources.
        Uses parallel processing for normalization.

        Args:
            clip_files: List of clip file paths
            output_path: Output video path
        """
        target_width = self.target_width or 1920
        target_height = self.target_height or 1080
        target_fps = self.target_fps or 24

        print(f"\nConcatenating {len(clip_files)} clips...")
        print(f"Normalizing to {target_width}x{target_height} @ {target_fps}fps")
        print(f"Using {self.parallel_workers} parallel workers")

        # First, validate and filter clips
        valid_clips = []
        skipped_count = 0
        for i, clip_file in enumerate(clip_files):
            if self.is_valid_clip(clip_file):
                valid_clips.append((i, clip_file))
            else:
                print(f"  Warning: Skipping invalid/empty clip {i}: {Path(clip_file).name}")
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
                    clip_name = Path(valid_clips[list(dict(valid_clips).keys()).index(idx)][1]).name if valid_clips else f"clip_{idx}"
                    print(f"  Warning: Error normalizing clip {idx}, skipping")
                    if error:
                        print(f"    {error[:100]}")
                    failed_count += 1

        # Sort results by original index to maintain order
        normalized_clips = [results[idx] for idx in sorted(results.keys())]

        print(f"Normalized {len(normalized_clips)} clips ({failed_count} failed)")

        if not normalized_clips:
            print("ERROR: No clips were successfully normalized")
            return

        # Create concat file for ffmpeg
        concat_file = self.temp_dir / "concat_list.txt"
        with open(concat_file, 'w') as f:
            for clip_file in normalized_clips:
                f.write(f"file '{Path(clip_file).absolute()}'\n")

        # Concatenate using ffmpeg (now all clips have same format)
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',  # Can use copy since all normalized
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Final video saved to: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error concatenating clips: {e}")
            print(f"stderr: {e.stderr.decode()}")
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
            skip_fps_prompt: If True, auto-select smallest fps from sources
        """
        print("\n=== Assembling Video ===\n")

        # Analyze source videos for resolution and fps
        need_resolution = self.target_width is None or self.target_height is None
        need_fps = self.target_fps is None

        if need_resolution or need_fps:
            analysis = self.analyze_source_videos()

            # Determine output resolution
            if need_resolution:
                if skip_resolution_prompt:
                    # Use smallest resolution automatically
                    self.target_width = analysis['min_width']
                    self.target_height = analysis['min_height']
                    print(f"\nUsing smallest resolution: {self.target_width}x{self.target_height}")
                else:
                    # Prompt user for resolution
                    self.target_width, self.target_height = self.prompt_for_resolution(analysis)

            # Determine output frame rate
            if need_fps:
                if skip_fps_prompt:
                    # Use smallest fps automatically
                    self.target_fps = analysis['min_fps']
                    print(f"\nUsing smallest frame rate: {self.target_fps:.2f} fps")
                else:
                    # Prompt user for fps
                    self.target_fps = self.prompt_for_fps(analysis)

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

        # Concatenate all clips
        self.concatenate_clips(all_clips, self.output_path)

        print("\n=== Video Assembly Complete ===")

    def cleanup(self):
        """Clean up temporary files."""
        print("\nCleaning up temporary files...")

        # Remove all temporary clip files
        for pattern in ['match_*.mp4', 'match_*.wav', 'temp_*.wav']:
            for temp_file in self.temp_dir.glob(pattern):
                try:
                    temp_file.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete {temp_file}: {e}")

        print("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Assemble final video from pitch match plan"
    )
    parser.add_argument(
        '--match-plan',
        type=str,
        required=True,
        help='Path to match plan JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/output/pitch_matched_video.mp4',
        help='Output video path'
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default='data/temp',
        help='Temporary directory for intermediate files'
    )
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Keep temporary files (for debugging)'
    )
    parser.add_argument(
        '--resolution',
        type=str,
        default=None,
        help='Output resolution (WxH, e.g., 1920x1080). If not specified, will prompt.'
    )
    parser.add_argument(
        '--auto-resolution',
        action='store_true',
        help='Automatically use smallest source resolution (no prompt)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=None,
        help='Output frame rate (e.g., 24, 29.97). If not specified, will prompt.'
    )
    parser.add_argument(
        '--auto-fps',
        action='store_true',
        help='Automatically use smallest source frame rate (no prompt)'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=None,
        help='Number of parallel workers for clip normalization (default: auto, max 8)'
    )
    parser.add_argument(
        '--true-silence',
        action='store_true',
        help='Use black frames with muted audio for rest segments instead of source silence clips'
    )

    args = parser.parse_args()

    print("=== Pitch Video Assembler ===\n")
    print(f"Match plan: {args.match_plan}")
    print(f"Output: {args.output}")
    print(f"Temp dir: {args.temp_dir}")

    # Parse resolution if provided
    target_width = None
    target_height = None
    if args.resolution:
        try:
            target_width, target_height = map(int, args.resolution.lower().split('x'))
            print(f"Resolution: {target_width}x{target_height}")
        except ValueError:
            print(f"Warning: Invalid resolution format '{args.resolution}', will prompt")

    # Parse fps if provided
    target_fps = args.fps
    if target_fps:
        print(f"Frame rate: {target_fps} fps")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize assembler
    assembler = PitchVideoAssembler(
        match_plan_path=args.match_plan,
        output_path=args.output,
        temp_dir=args.temp_dir,
        target_width=target_width,
        target_height=target_height,
        target_fps=target_fps,
        parallel_workers=args.parallel,
        use_true_silence=args.true_silence
    )

    # Load match plan
    assembler.load_match_plan()

    # Assemble video
    assembler.assemble_video(
        skip_resolution_prompt=args.auto_resolution,
        skip_fps_prompt=args.auto_fps
    )

    # Cleanup temporary files
    if not args.no_cleanup:
        assembler.cleanup()
    else:
        print("\nSkipping cleanup (--no-cleanup specified)")

    print("\n=== Assembly Complete ===")


if __name__ == "__main__":
    main()
