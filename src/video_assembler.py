#!/usr/bin/env python3
"""
Video Assembler

Cuts and reassembles video based on semantic matches from audio-driven analysis.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict
import tempfile
import shutil

from edl_generator import EDLGenerator


class VideoAssembler:
    """Cut and reassemble video based on audio-video matches."""

    def __init__(
        self,
        video_path: str,
        audio_path: str,
        matches_path: str,
        output_path: str
    ):
        """
        Initialize video assembler.

        Args:
            video_path: Path to source video file
            audio_path: Path to guidance audio file
            matches_path: Path to matches JSON
            output_path: Path for output video
        """
        self.video_path = Path(video_path)
        self.audio_path = Path(audio_path)
        self.output_path = Path(output_path)

        # Load matches
        with open(matches_path, 'r') as f:
            data = json.load(f)
        self.matches = data['matches']

        print(f"Loaded {len(self.matches)} matches")
        print(f"Source video: {self.video_path}")
        print(f"Guidance audio: {self.audio_path}")
        print(f"Output: {self.output_path}")

    def cut_video_segments(self, temp_dir: Path, preserve_audio: bool = False) -> List[Path]:
        """
        Cut video into segments based on matches.

        Args:
            temp_dir: Temporary directory for segment files
            preserve_audio: If True, preserve original video audio

        Returns:
            List of paths to video segment files
        """
        print(f"\nCutting {len(self.matches)} video segments...")

        segment_paths = []

        for i, match in enumerate(self.matches):
            # Get video timing
            video_start = match['video_start_time']
            video_end = match['video_end_time']
            video_duration = video_end - video_start

            # Get audio duration (this is how long the segment should be)
            audio_duration = match['audio_duration']

            # Output path for this segment
            segment_path = temp_dir / f"segment_{i:04d}.mp4"

            # Use ffmpeg to cut the video segment with high quality
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-ss', str(video_start),  # Start time
                '-i', str(self.video_path),  # Input video
                '-t', str(audio_duration),  # Duration (audio segment length)
                '-c:v', 'libx264',  # Video codec
                '-crf', '18',  # High quality (lower = better, 18 is visually lossless)
                '-preset', 'slow',  # Slower but better compression
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            ]

            # Optionally preserve audio or remove it
            if preserve_audio:
                cmd.extend(['-c:a', 'aac', '-b:a', '320k'])  # High quality audio
            else:
                cmd.append('-an')  # No audio

            cmd.append(str(segment_path))


            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                segment_paths.append(segment_path)

                if (i + 1) % 20 == 0 or (i + 1) == len(self.matches):
                    print(f"  Cut {i + 1}/{len(self.matches)} segments")

            except subprocess.CalledProcessError as e:
                print(f"Error cutting segment {i}: {e}")
                print(f"stderr: {e.stderr}")
                # Create a black frame as fallback
                self._create_black_segment(segment_path, audio_duration)
                segment_paths.append(segment_path)

        print(f"✓ Cut {len(segment_paths)} video segments")
        return segment_paths

    def _create_black_segment(self, output_path: Path, duration: float):
        """Create a black video segment as fallback."""
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'lavfi',
            '-i', f'color=c=black:s=1920x1080:d={duration}',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def concatenate_segments(
        self,
        segment_paths: List[Path],
        temp_dir: Path
    ) -> Path:
        """
        Concatenate video segments.

        Args:
            segment_paths: List of segment file paths
            temp_dir: Temporary directory

        Returns:
            Path to concatenated video
        """
        print(f"\nConcatenating {len(segment_paths)} segments...")

        # Create concat file for ffmpeg
        concat_file = temp_dir / "concat_list.txt"
        with open(concat_file, 'w') as f:
            for seg_path in segment_paths:
                f.write(f"file '{seg_path.absolute()}'\n")

        # Output path for concatenated video (no audio yet)
        concat_output = temp_dir / "concatenated.mp4"

        # Concatenate with ffmpeg
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            str(concat_output)
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✓ Concatenated video created")
            return concat_output

        except subprocess.CalledProcessError as e:
            print(f"Error concatenating: {e}")
            print(f"stderr: {e.stderr}")
            raise

    def add_audio(self, video_path: Path) -> Path:
        """
        Add guidance audio to video (H.264).

        Args:
            video_path: Path to video file

        Returns:
            Path to final output
        """
        print(f"\nAdding guidance audio (H.264)...")

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add audio with ffmpeg
        cmd = [
            'ffmpeg',
            '-y',
            '-i', str(video_path),
            '-i', str(self.audio_path),
            '-c:v', 'copy',  # Copy video stream
            '-c:a', 'aac',   # Encode audio to AAC
            '-b:a', '320k',  # High quality audio
            '-shortest',     # Match shortest stream
            str(self.output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✓ Added audio to video")
            return self.output_path

        except subprocess.CalledProcessError as e:
            print(f"Error adding audio: {e}")
            print(f"stderr: {e.stderr}")
            raise

    def create_prores_version(self, h264_video: Path) -> Path:
        """
        Create ProRes 422 version of the final video.

        Args:
            h264_video: Path to H.264 video file

        Returns:
            Path to ProRes output
        """
        print(f"\nCreating ProRes 422 version...")

        # ProRes output path
        prores_output = self.output_path.parent / (
            self.output_path.stem + "_prores.mov"
        )

        # Convert to ProRes 422
        cmd = [
            'ffmpeg',
            '-y',
            '-i', str(h264_video),
            '-c:v', 'prores_ks',  # ProRes encoder
            '-profile:v', '2',     # ProRes 422 (0=Proxy, 1=LT, 2=Standard, 3=HQ)
            '-vendor', 'apl0',     # Apple vendor code
            '-pix_fmt', 'yuv422p10le',  # 10-bit 4:2:2
            '-c:a', 'pcm_s16le',   # Uncompressed audio
            str(prores_output)
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✓ ProRes version created: {prores_output}")
            return prores_output

        except subprocess.CalledProcessError as e:
            print(f"Warning: ProRes creation failed: {e}")
            print(f"stderr: {e.stderr}")
            print("ProRes 422 requires ffmpeg compiled with prores_ks support")
            return None

    def assemble(self):
        """Complete assembly pipeline - creates both versions."""
        print("\n" + "=" * 50)
        print("Starting Video Assembly")
        print("=" * 50)

        outputs = {}

        # Create temporary directories for both versions
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            print(f"\nTemp directory: {temp_dir}")

            # VERSION 1: Original video audio
            print("\n" + "=" * 50)
            print("VERSION 1: Original Video Audio")
            print("=" * 50)

            temp_dir_original = temp_dir / "original_audio"
            temp_dir_original.mkdir()

            # Cut segments WITH audio
            segment_paths_original = self.cut_video_segments(
                temp_dir_original,
                preserve_audio=True
            )

            # Concatenate segments
            concat_video_original = self.concatenate_segments(
                segment_paths_original,
                temp_dir_original
            )

            # Save output with original audio
            output_original = self.output_path.parent / (
                self.output_path.stem + "_original_audio" + self.output_path.suffix
            )
            shutil.copy(concat_video_original, output_original)
            outputs['original_audio'] = output_original
            print(f"✓ Version 1 saved: {output_original}")

            # VERSION 2: Guidance audio
            print("\n" + "=" * 50)
            print("VERSION 2: Guidance Audio")
            print("=" * 50)

            temp_dir_guidance = temp_dir / "guidance_audio"
            temp_dir_guidance.mkdir()

            # Cut segments WITHOUT audio
            segment_paths_guidance = self.cut_video_segments(
                temp_dir_guidance,
                preserve_audio=False
            )

            # Concatenate segments
            concat_video_guidance = self.concatenate_segments(
                segment_paths_guidance,
                temp_dir_guidance
            )

            # Add guidance audio
            final_output = self.add_audio(concat_video_guidance)
            outputs['guidance_audio'] = final_output
            print(f"✓ Version 2 saved: {final_output}")

        # Create ProRes version (with original audio only)
        print("\n" + "=" * 50)
        print("Creating ProRes 422 Version (Original Audio)")
        print("=" * 50)

        prores_original = self.create_prores_version(outputs['original_audio'])
        if prores_original:
            outputs['original_audio_prores'] = prores_original

        print("\n" + "=" * 50)
        print("✓ Video Assembly Complete!")
        print("=" * 50)
        print(f"\nH.264 versions:")
        print(f"  1. Original audio: {outputs['original_audio']}")
        print(f"  2. Guidance audio: {outputs['guidance_audio']}")

        if 'original_audio_prores' in outputs:
            print(f"\nProRes 422 version (original audio):")
            print(f"  - {outputs['original_audio_prores']}")

        print(f"\nTotal segments: {len(self.matches)}")

    def generate_edl(self, output_path: str, frame_rate: float = 24.0) -> str:
        """
        Generate EDL file from matches.

        Args:
            output_path: Path for output EDL file
            frame_rate: Frame rate for timecode conversion

        Returns:
            Path to generated EDL file
        """
        print(f"\n=== Generating EDL ===")
        print(f"Frame rate: {frame_rate} fps")

        # Determine title from output path
        title = Path(output_path).stem

        edl = EDLGenerator(title, frame_rate=frame_rate)

        for i, match in enumerate(self.matches):
            # Get video timing
            video_start = match['video_start_time']
            audio_duration = match['audio_duration']

            # Build comment
            comment_parts = [f"Audio segment {match.get('audio_segment_index', i)}"]
            similarity = match.get('similarity_score', 0)
            if similarity:
                comment_parts.append(f"Similarity: {similarity:.3f}")
            if match.get('is_reused'):
                comment_parts.append(f"Reuse #{match.get('usage_count', 1)}")

            edl.add_event(
                source_path=str(self.video_path),
                source_in=video_start,
                source_out=video_start + audio_duration,
                comment=", ".join(comment_parts)
            )

        # Write EDL
        edl_path = edl.write(output_path)
        print(f"EDL saved to: {edl_path}")
        print(f"  Events: {edl.event_count}")
        print(f"  Total duration: {edl.total_duration:.2f}s")

        return edl_path


def main():
    parser = argparse.ArgumentParser(
        description='Assemble final video from semantic matches'
    )
    parser.add_argument('--video', required=True,
                       help='Path to source video file')
    parser.add_argument('--audio', required=True,
                       help='Path to guidance audio file')
    parser.add_argument('--matches', required=True,
                       help='Path to matches JSON file')
    parser.add_argument('--output', required=True,
                       help='Path for output video file')
    parser.add_argument('--fps', type=float, default=24.0,
                       help='Frame rate for EDL timecode (default: 24.0)')
    parser.add_argument('--edl', action='store_true',
                       help='Generate EDL file alongside video')
    parser.add_argument('--edl-only', action='store_true',
                       help='Generate EDL file only, skip video assembly')
    parser.add_argument('--edl-output', type=str,
                       help='Custom EDL output path (default: same as output with .edl extension)')

    args = parser.parse_args()

    # Check if ffmpeg is available
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            check=True,
            capture_output=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg not found. Please install ffmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Linux: sudo apt-get install ffmpeg")
        sys.exit(1)

    # Determine EDL output path
    if args.edl_output:
        edl_output_path = args.edl_output
    else:
        edl_output_path = str(Path(args.output).with_suffix('.edl'))

    # Initialize assembler
    assembler = VideoAssembler(
        video_path=args.video,
        audio_path=args.audio,
        matches_path=args.matches,
        output_path=args.output
    )

    # Handle EDL-only mode
    if args.edl_only:
        assembler.generate_edl(edl_output_path, frame_rate=args.fps)
        return

    # Run assembly
    assembler.assemble()

    # Generate EDL if requested
    if args.edl:
        assembler.generate_edl(edl_output_path, frame_rate=args.fps)


if __name__ == '__main__':
    main()
