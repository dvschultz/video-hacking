#!/usr/bin/env python3
"""
Shared video utility functions for video assemblers.

This module provides common video operations used by both
duration_video_assembler.py and pitch_video_assembler.py.
"""

import os
import platform
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple


def reset_terminal() -> None:
    """Reset terminal state after interruption (Unix only)."""
    if platform.system() != 'Windows':
        os.system('stty sane 2>/dev/null')


@lru_cache(maxsize=256)
def get_video_resolution(video_path: str) -> Tuple[int, int]:
    """
    Get video resolution using ffprobe.

    Results are cached using lru_cache to avoid repeated ffprobe calls.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height)
    """
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
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid resolution: {width}x{height}")
        return (width, height)
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Warning: Could not get resolution for {video_path}: {e}")
        return (1920, 1080)  # Default fallback


@lru_cache(maxsize=256)
def get_video_fps(video_path: str) -> float:
    """
    Get video frame rate using ffprobe.

    Results are cached using lru_cache to avoid repeated ffprobe calls.

    Args:
        video_path: Path to video file

    Returns:
        Frame rate as float

    Raises:
        ValueError: If FPS is zero or negative
    """
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
            if float(den) == 0:
                raise ValueError("Division by zero in frame rate")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        if fps <= 0:
            raise ValueError(f"Invalid FPS ({fps}) returned for {video_path}")

        return fps
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Warning: Could not get FPS for {video_path}: {e}")
        return 24.0  # Default fallback


def is_valid_clip(clip_path: str) -> bool:
    """
    Check if a clip file is valid (exists and has non-zero duration).

    Args:
        clip_path: Path to clip file

    Returns:
        True if clip is valid, False otherwise
    """
    if not Path(clip_path).exists():
        return False

    # First try stream duration
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=duration',
        '-of', 'csv=p=0',
        str(clip_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
        duration_str = result.stdout.strip()

        # Check if we got a valid duration
        if not duration_str or duration_str == 'N/A':
            # Try format duration as fallback
            cmd2 = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                str(clip_path)
            ]
            result2 = subprocess.run(cmd2, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
            duration_str = result2.stdout.strip()

            if not duration_str or duration_str == 'N/A':
                return False

        duration = float(duration_str)
        return duration > 0.01  # At least 10ms

    except (subprocess.CalledProcessError, ValueError):
        return False


def prompt_for_resolution(analysis: Dict) -> Tuple[int, int]:
    """
    Prompt user to select output resolution.

    Args:
        analysis: Resolution analysis dictionary containing:
            - min_width, min_height: Smallest source dimensions
            - unique_resolutions: List of (width, height) tuples from sources

    Returns:
        Tuple of (width, height) selected by user
    """
    # Reset terminal settings in case subprocess messed them up
    reset_terminal()

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


def prompt_for_fps(analysis: Dict) -> float:
    """
    Prompt user to select output frame rate.

    Args:
        analysis: Analysis dictionary containing:
            - min_fps: Smallest source frame rate
            - unique_fps: List of frame rates from sources

    Returns:
        Frame rate selected by user
    """
    # Reset terminal settings in case subprocess messed them up
    reset_terminal()

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


def normalize_clip(
    clip_file: str,
    output_path: str,
    width: int,
    height: int
) -> Tuple[bool, Optional[str]]:
    """
    Normalize a single video clip to target resolution.

    Args:
        clip_file: Input clip path
        output_path: Output normalized clip path
        width: Target width
        height: Target height

    Returns:
        Tuple of (success, error_message)
    """
    # Only scale resolution - frame rate conversion happens in final concatenation
    cmd = [
        'ffmpeg', '-y',
        '-i', str(clip_file),
        '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-c:a', 'aac',
        '-ar', '44100',
        '-ac', '2',
        '-b:a', '320k',
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL)
        return (True, None)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode()[:200] if e.stderr else "Unknown error"
        return (False, error_msg)


def generate_black_clip(
    duration: float,
    output_path: str,
    width: int = 1920,
    height: int = 1080,
    fps: float = 24.0
) -> None:
    """
    Generate a black video clip with silence.

    Args:
        duration: Duration in seconds
        output_path: Output clip path
        width: Video width
        height: Video height
        fps: Frame rate

    Raises:
        ValueError: If any parameter is invalid (non-positive)
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid dimensions: {width}x{height}")
    if duration <= 0:
        raise ValueError(f"Invalid duration: {duration}")
    if fps <= 0:
        raise ValueError(f"Invalid fps: {fps}")

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
        print(f"Error generating black clip: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr.decode()[:200]}")
        raise


def clear_caches() -> None:
    """Clear all video metadata caches."""
    get_video_resolution.cache_clear()
    get_video_fps.cache_clear()
