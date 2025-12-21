#!/usr/bin/env python3
"""
Pitch Video Preview Generator

Creates a preview video with MIDI playback overlaid on the original video.
This allows visual verification of pitch detection accuracy - you can see
the person singing while hearing the detected notes as MIDI tones.

Useful for debugging pitch detection issues.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict

from midi_player import MIDIPlayer


class PitchVideoPreview:
    """Generates preview videos with MIDI playback."""

    def __init__(self, video_path: str, pitch_json: str):
        """
        Initialize the preview generator.

        Args:
            video_path: Path to original video file
            pitch_json: Path to pitch analysis JSON (guide or source)
        """
        self.video_path = Path(video_path)
        self.pitch_json = Path(pitch_json)

        # Load pitch data
        with open(pitch_json, 'r') as f:
            self.pitch_data = json.load(f)

        self.notes = self._extract_notes()

    def _extract_notes(self) -> List[Dict]:
        """Extract note sequence from pitch JSON."""
        notes = []

        # Handle different JSON formats
        if 'guide_sequence' in self.pitch_data:
            # Guide video format
            for segment in self.pitch_data['guide_sequence']:
                notes.append({
                    'pitch_midi': segment.get('pitch_midi'),
                    'pitch_hz': segment.get('pitch_hz'),
                    'duration': segment.get('duration'),
                    'note_name': segment.get('pitch_note'),
                    'confidence': segment.get('pitch_confidence', 1.0),
                    'start_time': segment.get('start_time'),
                    'end_time': segment.get('end_time')
                })

        elif 'pitch_database' in self.pitch_data:
            # Source video format
            for clip in self.pitch_data['pitch_database']:
                notes.append({
                    'pitch_midi': clip.get('pitch_midi'),
                    'pitch_hz': clip.get('pitch_hz'),
                    'duration': clip.get('duration'),
                    'note_name': clip.get('pitch_note'),
                    'confidence': clip.get('pitch_confidence', 1.0),
                    'start_time': clip.get('start_time'),
                    'end_time': clip.get('end_time')
                })

        elif 'segments' in self.pitch_data:
            # Generic segments format
            for segment in self.pitch_data['segments']:
                notes.append({
                    'pitch_midi': segment.get('pitch_midi'),
                    'pitch_hz': segment.get('pitch_hz'),
                    'duration': segment.get('duration'),
                    'note_name': segment.get('pitch_note'),
                    'confidence': segment.get('pitch_confidence', 1.0),
                    'start_time': segment.get('start_time'),
                    'end_time': segment.get('end_time')
                })

        return notes

    def generate_midi_audio(self, output_path: str) -> Path:
        """
        Generate MIDI playback audio file with precise timing.

        Args:
            output_path: Path to save audio file

        Returns:
            Path to generated audio file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Generating MIDI playback audio...")
        print(f"Total notes: {len(self.notes)}")

        import numpy as np

        # Generate audio using MIDI player
        player = MIDIPlayer(sample_rate=44100)

        # Calculate total duration from notes
        if len(self.notes) > 0:
            total_duration = max(note.get('end_time', note.get('start_time', 0) + note.get('duration', 0))
                               for note in self.notes)
        else:
            total_duration = 0.0

        # Create empty audio buffer for entire duration
        total_samples = int(total_duration * player.sample_rate)
        audio_buffer = np.zeros(total_samples, dtype=np.float32)

        print(f"Total audio duration: {total_duration:.2f}s")

        # Generate and place each note at its exact timestamp
        for i, note in enumerate(self.notes):
            start_time = note.get('start_time', 0.0)
            end_time = note.get('end_time', start_time + note.get('duration', 0.5))
            duration = end_time - start_time
            midi = note.get('pitch_midi')

            if midi is not None and midi > 0:
                # Get frequency
                freq = note.get('pitch_hz')
                if freq is None:
                    from pitch_utils import midi_to_hz
                    freq = midi_to_hz(midi)

                # Generate tone for this note
                tone = player.generate_tone(freq, duration)

                # Calculate sample positions
                start_sample = int(start_time * player.sample_rate)
                end_sample = start_sample + len(tone)

                # Make sure we don't exceed buffer
                if end_sample > len(audio_buffer):
                    end_sample = len(audio_buffer)
                    tone = tone[:end_sample - start_sample]

                # Place tone in buffer at exact position
                audio_buffer[start_sample:end_sample] = tone

            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{len(self.notes)} notes...")

        # Convert mono to stereo for compatibility
        audio_stereo = np.stack([audio_buffer, audio_buffer], axis=1)

        # Save audio
        import soundfile as sf
        sf.write(str(output_path), audio_stereo, player.sample_rate)
        print(f"Saved MIDI audio: {output_path}")
        print(f"Audio length: {len(audio_buffer) / player.sample_rate:.2f}s")

        return output_path

    def get_video_duration(self) -> float:
        """Get duration of the video file in seconds."""
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(self.video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            return duration
        except Exception as e:
            print(f"Warning: Could not get video duration: {e}")
            # Fall back to calculating from notes
            if len(self.notes) > 0:
                return max(note.get('end_time', 0) for note in self.notes)
            return 0.0

    def create_preview_video(self, output_path: str,
                           temp_audio_path: str = None) -> Path:
        """
        Create preview video with MIDI audio replacing original audio.

        Args:
            output_path: Path to save output video
            temp_audio_path: Optional path for temp audio file

        Returns:
            Path to generated video
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get video duration to ensure audio matches
        video_duration = self.get_video_duration()
        print(f"Video duration: {video_duration:.2f}s")

        # Generate MIDI audio
        if temp_audio_path is None:
            temp_audio_path = output_path.parent / f"{output_path.stem}_midi_temp.wav"

        midi_audio = self.generate_midi_audio(str(temp_audio_path))

        print(f"\nCreating preview video...")
        print(f"Input video: {self.video_path}")
        print(f"MIDI audio: {midi_audio}")
        print(f"Output: {output_path}")

        # Use ffmpeg to replace audio track
        # Re-encode video to ensure compatibility
        cmd = [
            'ffmpeg', '-y',
            '-i', str(self.video_path),  # Video input
            '-i', str(midi_audio),        # Audio input
            '-map', '0:v:0',              # Use video from first input
            '-map', '1:a:0',              # Use audio from second input
            '-c:v', 'libx264',            # Encode video as H.264
            '-preset', 'medium',          # Encoding speed
            '-crf', '18',                 # High quality
            '-c:a', 'aac',                # Encode audio as AAC
            '-b:a', '192k',               # Audio bitrate
            '-ac', '2',                   # Stereo audio
            '-ar', '44100',               # Sample rate
            '-pix_fmt', 'yuv420p',        # Pixel format for compatibility
            '-movflags', '+faststart',    # Enable streaming
            '-shortest',                  # Match shortest stream
            str(output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"\nPreview video created successfully!")
            print(f"Output: {output_path}")

            # Optionally remove temp audio file
            # temp_audio_path.unlink()

        except subprocess.CalledProcessError as e:
            print(f"Error creating preview video: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate preview video with MIDI playback audio"
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to original video file'
    )
    parser.add_argument(
        '--pitch-json',
        type=str,
        required=True,
        help='Path to pitch analysis JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output video path (default: input_name_midi_preview.mp4)'
    )
    parser.add_argument(
        '--temp-audio',
        type=str,
        help='Path for temporary MIDI audio file'
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        video_path = Path(args.video)
        output_path = video_path.parent / f"{video_path.stem}_midi_preview.mp4"

    print("=== Pitch Video Preview Generator ===\n")
    print(f"Video: {args.video}")
    print(f"Pitch JSON: {args.pitch_json}")
    print(f"Output: {output_path}\n")

    # Generate preview
    generator = PitchVideoPreview(args.video, args.pitch_json)
    generator.create_preview_video(
        str(output_path),
        temp_audio_path=args.temp_audio
    )

    print("\n=== Preview Complete ===")
    print(f"\nWatch the preview video to verify pitch detection:")
    print(f"  {output_path}")
    print(f"\nThe MIDI tones should match the singing in the video.")
    print(f"If they don't match, adjust pitch detection parameters.\n")


if __name__ == "__main__":
    main()
