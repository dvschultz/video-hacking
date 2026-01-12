#!/usr/bin/env python3
"""
MIDI Channel Splitter

Exports each MIDI channel as a separate WAV file for identifying
which channel contains the main vocal/melody track.

Usage:
    python midi_channel_splitter.py --midi song.mid --output-dir data/temp/midi_channels
    python midi_channel_splitter.py --midi song.mid --list-only
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

try:
    import mido
except ImportError:
    print("Error: mido not installed. Install with: pip install mido")
    import sys
    sys.exit(1)

from pitch_utils import midi_to_hz, midi_to_note_name
from midi_player import MIDIPlayer


class MIDIChannelSplitter:
    """Splits MIDI file into separate WAV files per channel."""

    def __init__(self, midi_path: str):
        """
        Initialize the splitter.

        Args:
            midi_path: Path to MIDI file
        """
        self.midi_path = Path(midi_path)
        self.midi_file = None
        self.ticks_per_beat = 480

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

        channel_info = defaultdict(lambda: {
            'note_count': 0,
            'notes': set(),
            'min_note': 127,
            'max_note': 0
        })

        for track in self.midi_file.tracks:
            for msg in track:
                if hasattr(msg, 'channel'):
                    if msg.type == 'note_on' and msg.velocity > 0:
                        ch = msg.channel
                        channel_info[ch]['note_count'] += 1
                        channel_info[ch]['notes'].add(msg.note)
                        channel_info[ch]['min_note'] = min(channel_info[ch]['min_note'], msg.note)
                        channel_info[ch]['max_note'] = max(channel_info[ch]['max_note'], msg.note)

        # Convert sets to counts and add note names
        for ch in channel_info:
            info = channel_info[ch]
            info['unique_notes'] = len(info['notes'])
            if info['note_count'] > 0:
                info['note_range'] = f"{midi_to_note_name(info['min_note'])} - {midi_to_note_name(info['max_note'])}"
            else:
                info['note_range'] = "N/A"
            del info['notes']

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

            tick_delta = change_tick - last_tick
            seconds += (tick_delta / self.ticks_per_beat) * (current_tempo / 1_000_000)

            last_tick = change_tick
            current_tempo = tempo

        remaining_ticks = ticks - last_tick
        seconds += (remaining_ticks / self.ticks_per_beat) * (current_tempo / 1_000_000)

        return seconds

    def extract_notes_for_channel(self, channel: int) -> List[Dict]:
        """
        Extract note events from a specific channel.

        Args:
            channel: MIDI channel (0-15)

        Returns:
            List of note dictionaries with start_time, end_time, pitch_midi
        """
        if self.midi_file is None:
            self.load_midi()

        tempo_changes = self._collect_tempo_changes()
        notes = []
        active_notes = {}  # pitch -> start_tick

        for track in self.midi_file.tracks:
            current_tick = 0

            for msg in track:
                current_tick += msg.time

                if not hasattr(msg, 'channel') or msg.channel != channel:
                    continue

                if msg.type == 'note_on' and msg.velocity > 0:
                    # End any existing note with same pitch
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

        notes.sort(key=lambda x: x['start_time'])
        return notes

    def synthesize_channel(self, notes: List[Dict], sample_rate: int = 22050,
                           total_duration: float = None) -> np.ndarray:
        """
        Synthesize audio for a list of notes.

        Args:
            notes: List of note dictionaries
            sample_rate: Audio sample rate
            total_duration: Total duration in seconds (uses MIDI file length if None)

        Returns:
            Audio waveform as numpy array
        """
        if not notes:
            return np.array([])

        player = MIDIPlayer(sample_rate=sample_rate)

        # Use provided duration or fall back to last note's end time
        if total_duration is None:
            total_duration = max(note['end_time'] for note in notes)

        total_samples = int((total_duration + 1.0) * sample_rate)
        audio_buffer = np.zeros(total_samples, dtype=np.float32)

        # Place each note at its exact timestamp
        for note in notes:
            freq = midi_to_hz(note['pitch_midi'])
            duration = note['duration']
            start_time = note['start_time']

            if freq > 0 and duration > 0:
                tone = player.generate_tone(freq, duration)

                start_sample = int(start_time * sample_rate)
                end_sample = start_sample + len(tone)

                if end_sample > len(audio_buffer):
                    end_sample = len(audio_buffer)
                    tone = tone[:end_sample - start_sample]

                if start_sample < len(audio_buffer):
                    audio_buffer[start_sample:end_sample] += tone

        # Normalize
        max_val = np.max(np.abs(audio_buffer))
        if max_val > 0:
            audio_buffer = audio_buffer / max_val * 0.8

        # Trim to exact total duration (plus small buffer)
        audio_buffer = audio_buffer[:int((total_duration + 0.1) * sample_rate)]

        return audio_buffer

    def export_all_channels(self, output_dir: str, sample_rate: int = 22050) -> List[Path]:
        """
        Export all channels as separate WAV files.

        Args:
            output_dir: Output directory for WAV files
            sample_rate: Audio sample rate

        Returns:
            List of created file paths
        """
        if self.midi_file is None:
            self.load_midi()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        channels = self.list_channels()
        created_files = []
        midi_basename = self.midi_path.stem

        # Get total MIDI file duration to include trailing silence
        midi_total_duration = self.midi_file.length
        print(f"\nMIDI file total duration: {midi_total_duration:.2f}s")
        print(f"Exporting {len(channels)} channels to: {output_dir}")

        for channel in sorted(channels.keys()):
            info = channels[channel]
            print(f"\n  Channel {channel:2d}: {info['note_count']} notes, range {info['note_range']}")

            notes = self.extract_notes_for_channel(channel)
            if not notes:
                print(f"    Skipping (no notes)")
                continue

            # Use full MIDI duration to include any trailing silence
            audio = self.synthesize_channel(notes, sample_rate, total_duration=midi_total_duration)
            if len(audio) == 0:
                print(f"    Skipping (empty audio)")
                continue

            # Create output filename
            output_path = output_dir / f"{midi_basename}_channel_{channel:02d}.wav"
            sf.write(str(output_path), audio, sample_rate)

            duration = len(audio) / sample_rate
            print(f"    Saved: {output_path.name} ({duration:.1f}s)")
            created_files.append(output_path)

        return created_files


def main():
    parser = argparse.ArgumentParser(
        description='Split MIDI file into separate WAV files per channel.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --midi song.mid --output-dir data/temp/midi_channels
  %(prog)s --midi song.mid --list-only
  %(prog)s --midi song.mid --sample-rate 44100
        """
    )

    parser.add_argument('--midi', required=True,
                        help='Path to MIDI file')
    parser.add_argument('--output-dir', default='data/temp/midi_channels',
                        help='Output directory for WAV files (default: data/temp/midi_channels)')
    parser.add_argument('--sample-rate', type=int, default=22050,
                        help='Audio sample rate (default: 22050)')
    parser.add_argument('--list-only', action='store_true',
                        help='List channels only, do not export')

    args = parser.parse_args()

    # Validate MIDI file
    midi_path = Path(args.midi)
    if not midi_path.exists():
        print(f"Error: MIDI file not found: {args.midi}")
        return 1

    print("=== MIDI Channel Splitter ===\n")

    splitter = MIDIChannelSplitter(args.midi)
    splitter.load_midi()

    channels = splitter.list_channels()

    print("\nChannels found:")
    print("-" * 60)
    if not channels:
        print("  No note events found in any channel")
    else:
        for ch in sorted(channels.keys()):
            info = channels[ch]
            print(f"  Channel {ch:2d}: {info['note_count']:5d} notes, "
                  f"{info['unique_notes']:3d} unique, range: {info['note_range']}")
    print("-" * 60)

    if args.list_only:
        print("\n(--list-only mode, not exporting)")
        return 0

    # Export all channels
    created_files = splitter.export_all_channels(args.output_dir, args.sample_rate)

    print(f"\n=== Export Complete ===")
    print(f"Created {len(created_files)} WAV files in: {args.output_dir}")
    print("\nListen to each file to identify the vocal/melody channel,")
    print("then use that channel number with: ./test_midi_guide.sh <midi> <channel>")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main() or 0)
