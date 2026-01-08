#!/usr/bin/env python3
"""
MIDI Guide Converter

Converts MIDI files to guide sequence JSON format for the pitch matching pipeline.
Alternative to pitch_guide_analyzer.py when you have a MIDI melody to match.

Usage:
    python midi_guide_converter.py --midi song.mid --channel 1
    python midi_guide_converter.py --midi song.mid --channel 0 --list-channels
"""

import argparse
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

try:
    import mido
except ImportError:
    print("Error: mido not installed. Install with: pip install mido")
    import sys
    sys.exit(1)

from pitch_utils import midi_to_hz, midi_to_note_name
from midi_player import MIDIPlayer


class MIDIGuideConverter:
    """Converts MIDI files to pitch matching guide sequence format."""

    def __init__(self, midi_path: str, channel: int):
        """
        Initialize the converter.

        Args:
            midi_path: Path to MIDI file
            channel: MIDI channel to extract (0-15)
        """
        self.midi_path = Path(midi_path)
        self.channel = channel

        self.midi_file = None
        self.ticks_per_beat = 480  # Default, will be updated from file

        self.notes = []           # Raw note events
        self.pitch_segments = []  # Output segments

    def load_midi(self) -> None:
        """Load and parse MIDI file."""
        print(f"Loading MIDI file: {self.midi_path}")
        self.midi_file = mido.MidiFile(str(self.midi_path))
        self.ticks_per_beat = self.midi_file.ticks_per_beat
        print(f"  Ticks per beat: {self.ticks_per_beat}")
        print(f"  Number of tracks: {len(self.midi_file.tracks)}")

    def list_channels(self) -> Dict[int, Dict]:
        """
        List all channels found in the MIDI file with note counts.

        Returns:
            Dictionary mapping channel number to info dict
        """
        if self.midi_file is None:
            self.load_midi()

        channel_info = defaultdict(lambda: {'note_count': 0, 'notes': set()})

        for track in self.midi_file.tracks:
            for msg in track:
                if hasattr(msg, 'channel'):
                    if msg.type == 'note_on' and msg.velocity > 0:
                        channel_info[msg.channel]['note_count'] += 1
                        channel_info[msg.channel]['notes'].add(msg.note)

        # Convert sets to counts for display
        for ch in channel_info:
            channel_info[ch]['unique_notes'] = len(channel_info[ch]['notes'])
            del channel_info[ch]['notes']

        return dict(channel_info)

    def _collect_tempo_changes(self) -> List[Tuple[int, int]]:
        """
        Collect all tempo changes from the MIDI file.

        Returns:
            List of (tick, tempo_microseconds) tuples, sorted by tick
        """
        tempo_changes = [(0, 500000)]  # Default: 120 BPM

        for track in self.midi_file.tracks:
            tick = 0
            for msg in track:
                tick += msg.time
                if msg.type == 'set_tempo':
                    tempo_changes.append((tick, msg.tempo))

        tempo_changes.sort(key=lambda x: x[0])
        return tempo_changes

    def _ticks_to_seconds(self, ticks: int, tempo_changes: List[Tuple[int, int]]) -> float:
        """
        Convert MIDI ticks to seconds, accounting for tempo changes.

        Args:
            ticks: Target tick position
            tempo_changes: List of (tick, tempo_microseconds) tuples

        Returns:
            Time in seconds
        """
        seconds = 0.0
        last_tick = 0
        current_tempo = 500000  # Default 120 BPM

        for change_tick, tempo in tempo_changes:
            if change_tick >= ticks:
                break

            # Add time from last_tick to change_tick at current_tempo
            tick_delta = change_tick - last_tick
            seconds += (tick_delta / self.ticks_per_beat) * (current_tempo / 1_000_000)

            last_tick = change_tick
            current_tempo = tempo

        # Add remaining ticks at final tempo
        remaining_ticks = ticks - last_tick
        seconds += (remaining_ticks / self.ticks_per_beat) * (current_tempo / 1_000_000)

        return seconds

    def extract_notes(self) -> List[Dict]:
        """
        Extract note events from the specified channel.

        Handles:
        - Tempo changes throughout the file
        - Note on/off pairing
        - Conversion from ticks to absolute time

        Returns:
            List of note dictionaries with start_time, end_time, pitch_midi
        """
        if self.midi_file is None:
            self.load_midi()

        tempo_changes = self._collect_tempo_changes()
        notes = []
        active_notes = {}  # pitch -> start_tick

        # Process all tracks
        for track in self.midi_file.tracks:
            current_tick = 0

            for msg in track:
                current_tick += msg.time

                # Skip messages not on our channel
                if hasattr(msg, 'channel') and msg.channel != self.channel:
                    continue

                if msg.type == 'note_on' and msg.velocity > 0:
                    # Note start - if there's already an active note with same pitch,
                    # end it first
                    if msg.note in active_notes:
                        start_tick = active_notes.pop(msg.note)
                        start_time = self._ticks_to_seconds(start_tick, tempo_changes)
                        end_time = self._ticks_to_seconds(current_tick, tempo_changes)

                        if end_time > start_time:
                            notes.append({
                                'pitch_midi': msg.note,
                                'start_time': start_time,
                                'end_time': end_time,
                                'duration': end_time - start_time
                            })

                    active_notes[msg.note] = current_tick

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    # Note end
                    if msg.note in active_notes:
                        start_tick = active_notes.pop(msg.note)

                        # Convert ticks to seconds
                        start_time = self._ticks_to_seconds(start_tick, tempo_changes)
                        end_time = self._ticks_to_seconds(current_tick, tempo_changes)

                        if end_time > start_time:
                            notes.append({
                                'pitch_midi': msg.note,
                                'start_time': start_time,
                                'end_time': end_time,
                                'duration': end_time - start_time
                            })

        # Sort by start time
        notes.sort(key=lambda x: x['start_time'])
        self.notes = notes

        print(f"Extracted {len(notes)} notes from channel {self.channel}")
        return notes

    def merge_small_rests(self, min_rest_duration: float = 0.1) -> List[Dict]:
        """
        Merge consecutive notes with small gaps between them.

        Small gaps are absorbed into the preceding note by extending its end time.

        Args:
            min_rest_duration: Minimum rest duration to preserve (seconds)

        Returns:
            Notes with small gaps removed (previous note extended to fill gap)
        """
        if not self.notes:
            return []

        merged_notes = [self.notes[0].copy()]

        for i in range(1, len(self.notes)):
            current_note = self.notes[i]
            prev_note = merged_notes[-1]

            gap = current_note['start_time'] - prev_note['end_time']

            if gap < min_rest_duration and gap > 0:
                # Small gap - extend previous note to fill it
                prev_note['end_time'] = current_note['start_time']
                prev_note['duration'] = prev_note['end_time'] - prev_note['start_time']

            # Add current note
            merged_notes.append(current_note.copy())

        self.notes = merged_notes
        print(f"After merging small rests (<{min_rest_duration}s): {len(merged_notes)} notes")
        return merged_notes

    def convert_to_segments(self, min_rest_duration: float = 0.1) -> List[Dict]:
        """
        Convert notes to pitch segment format matching guide_sequence.json structure.
        Includes rest segments for gaps between notes.

        Args:
            min_rest_duration: Minimum rest duration to include as segment (seconds)

        Returns:
            List of segment dictionaries with all required fields
        """
        segments = []
        current_time = 0.0
        segment_index = 0

        for note in self.notes:
            # Check for rest before this note
            gap = note['start_time'] - current_time
            if gap >= min_rest_duration:
                # Add rest segment
                rest_segment = {
                    'index': segment_index,
                    'start_time': current_time,
                    'end_time': note['start_time'],
                    'duration': gap,
                    'pitch_hz': 0.0,
                    'pitch_midi': -1,  # Special marker for rest
                    'pitch_note': 'REST',
                    'pitch_confidence': 1.0,
                    'is_rest': True
                }
                segments.append(rest_segment)
                segment_index += 1

            # Add note segment
            pitch_midi = note['pitch_midi']
            pitch_hz = midi_to_hz(pitch_midi)
            pitch_note = midi_to_note_name(pitch_midi)

            segment = {
                'index': segment_index,
                'start_time': note['start_time'],
                'end_time': note['end_time'],
                'duration': note['duration'],
                'pitch_hz': float(pitch_hz),
                'pitch_midi': pitch_midi,
                'pitch_note': pitch_note,
                'pitch_confidence': 1.0,  # MIDI notes are exact
                'is_rest': False
            }
            segments.append(segment)
            segment_index += 1
            current_time = note['end_time']

        self.pitch_segments = segments
        note_count = sum(1 for s in segments if not s.get('is_rest', False))
        rest_count = sum(1 for s in segments if s.get('is_rest', False))
        print(f"Converted to {len(segments)} segments ({note_count} notes, {rest_count} rests)")
        return segments

    def generate_audio_preview(self, output_path: str, sample_rate: int = 22050) -> Path:
        """
        Generate synthesized audio preview from MIDI notes.

        Args:
            output_path: Path to save audio file
            sample_rate: Audio sample rate

        Returns:
            Path to generated audio file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        player = MIDIPlayer(sample_rate=sample_rate)

        # Calculate total duration
        if self.pitch_segments:
            total_duration = max(seg['end_time'] for seg in self.pitch_segments)
        else:
            total_duration = 0.0

        # Create empty buffer with some padding
        total_samples = int((total_duration + 1.0) * sample_rate)
        audio_buffer = np.zeros(total_samples, dtype=np.float32)

        print(f"Generating audio preview ({total_duration:.2f}s)...")

        # Place each note at its exact timestamp
        for seg in self.pitch_segments:
            start_time = seg['start_time']
            duration = seg['duration']
            freq = seg['pitch_hz']

            if freq > 0 and duration > 0:
                # Generate tone
                tone = player.generate_tone(freq, duration)

                # Calculate sample positions
                start_sample = int(start_time * sample_rate)
                end_sample = start_sample + len(tone)

                # Bounds check
                if end_sample > len(audio_buffer):
                    end_sample = len(audio_buffer)
                    tone = tone[:end_sample - start_sample]

                # Mix into buffer
                if start_sample < len(audio_buffer):
                    audio_buffer[start_sample:end_sample] += tone

        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio_buffer))
        if max_val > 0:
            audio_buffer = audio_buffer / max_val * 0.8

        # Trim trailing silence
        audio_buffer = audio_buffer[:int((total_duration + 0.5) * sample_rate)]

        # Save audio
        sf.write(str(output_path), audio_buffer, sample_rate)

        print(f"Audio preview saved: {output_path}")
        return output_path

    def save_results(self, output_path: str, audio_path: Optional[str] = None) -> None:
        """
        Save pitch segments to JSON file.

        Args:
            output_path: Output JSON file path
            audio_path: Path to generated audio preview (optional)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate total duration
        if self.pitch_segments:
            total_duration = max(seg['end_time'] for seg in self.pitch_segments)
        else:
            total_duration = 0.0

        data = {
            'video_path': None,
            'audio_path': str(audio_path) if audio_path else None,
            'midi_path': str(self.midi_path),
            'sample_rate': 22050,
            'pitch_detection_method': 'MIDI',
            'midi_channel': self.channel,
            'num_segments': len(self.pitch_segments),
            'total_duration': total_duration,
            'pitch_segments': self.pitch_segments
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Guide sequence saved: {output_path}")
        print(f"Total segments: {len(self.pitch_segments)}")
        print(f"Total duration: {total_duration:.2f}s")

    def print_summary(self, limit: int = 20) -> None:
        """Print summary of extracted note sequence."""
        print(f"\n=== Note Sequence Summary ===")
        print(f"Total notes: {len(self.pitch_segments)}")

        if self.pitch_segments:
            # Pitch range
            pitches = [seg['pitch_midi'] for seg in self.pitch_segments]
            print(f"Pitch range: {midi_to_note_name(min(pitches))} to {midi_to_note_name(max(pitches))}")
            print(f"MIDI range: {min(pitches)} to {max(pitches)}")

            # Duration stats
            durations = [seg['duration'] for seg in self.pitch_segments]
            print(f"Duration range: {min(durations):.3f}s to {max(durations):.3f}s")
            print(f"Average duration: {np.mean(durations):.3f}s")

            # Show first N notes
            print(f"\nFirst {min(limit, len(self.pitch_segments))} notes:")
            for seg in self.pitch_segments[:limit]:
                print(f"  {seg['pitch_note']:4s} ({seg['pitch_midi']:3d}): "
                      f"{seg['start_time']:.3f}s - {seg['end_time']:.3f}s "
                      f"(dur: {seg['duration']:.3f}s)")

            if len(self.pitch_segments) > limit:
                print(f"  ... and {len(self.pitch_segments) - limit} more notes")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert MIDI file to guide sequence for pitch matching pipeline"
    )
    parser.add_argument(
        '--midi',
        type=str,
        required=True,
        help='Path to MIDI file'
    )
    parser.add_argument(
        '--channel',
        type=int,
        required=True,
        help='MIDI channel to extract (0-15)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/segments/guide_sequence.json',
        help='Output JSON file path (default: data/segments/guide_sequence.json)'
    )
    parser.add_argument(
        '--min-rest',
        type=float,
        default=0.1,
        help='Minimum rest duration to preserve in seconds (default: 0.1)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=22050,
        help='Sample rate for audio preview (default: 22050)'
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default='data/temp',
        help='Directory for generated audio preview (default: data/temp)'
    )
    parser.add_argument(
        '--list-channels',
        action='store_true',
        help='List available channels in MIDI file and exit'
    )
    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Skip audio preview generation'
    )

    args = parser.parse_args()

    print("=== MIDI Guide Converter ===\n")

    # Validate channel
    if not 0 <= args.channel <= 15:
        print(f"Error: Channel must be 0-15, got {args.channel}")
        return 1

    # Initialize converter
    converter = MIDIGuideConverter(args.midi, args.channel)

    # List channels mode
    if args.list_channels:
        converter.load_midi()
        channels = converter.list_channels()

        print("\nChannels found in MIDI file:")
        print("-" * 40)

        if not channels:
            print("  No note events found in any channel")
        else:
            for ch in sorted(channels.keys()):
                info = channels[ch]
                print(f"  Channel {ch:2d}: {info['note_count']:4d} notes, "
                      f"{info['unique_notes']:3d} unique pitches")

        print("-" * 40)
        return 0

    # Full conversion
    converter.load_midi()
    converter.extract_notes()

    if not converter.notes:
        print(f"\nWarning: No notes found on channel {args.channel}")
        print("Use --list-channels to see available channels")
        return 1

    converter.merge_small_rests(args.min_rest)
    converter.convert_to_segments(args.min_rest)

    # Generate audio preview
    audio_path = None
    if not args.no_audio:
        midi_basename = Path(args.midi).stem
        audio_path = Path(args.temp_dir) / f"{midi_basename}_midi_preview.wav"
        converter.generate_audio_preview(str(audio_path), args.sample_rate)

    # Save results
    converter.save_results(args.output, audio_path)

    # Print summary
    converter.print_summary()

    print("\n=== Conversion Complete ===")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
