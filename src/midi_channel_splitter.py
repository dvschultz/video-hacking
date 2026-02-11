#!/usr/bin/env python3
"""
MIDI Channel Splitter

Exports each MIDI channel as a separate WAV file for identifying
which channel contains the main vocal/melody track.

Supports two synthesis modes:
- FluidSynth (default): Uses SoundFont for realistic instrument sounds
- Simple synthesis: Uses basic sine wave additive synthesis (fallback)

Usage:
    python midi_channel_splitter.py --midi song.mid --output-dir data/temp/midi_channels
    python midi_channel_splitter.py --midi song.mid --soundfont data/soundfonts/GeneralUser_GS.sf2
    python midi_channel_splitter.py --midi song.mid --list-only
    python midi_channel_splitter.py --midi song.mid --no-fluidsynth
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

try:
    import mido
except ImportError:
    print("Error: mido not installed. Install with: pip install mido")
    import sys
    sys.exit(1)

from pitch_utils import midi_to_hz, midi_to_note_name
from midi_player import MIDIPlayer

# Import FluidSynth player (optional)
try:
    from fluidsynth_player import (
        FluidSynthPlayer, is_fluidsynth_available, find_soundfont
    )
    FLUIDSYNTH_IMPORT_OK = True
except ImportError:
    FLUIDSYNTH_IMPORT_OK = False

    def is_fluidsynth_available():
        return False

    def find_soundfont():
        return None


# General MIDI instrument names (program 0-127)
GM_INSTRUMENTS = [
    # Piano (0-7)
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano",
    "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
    # Chromatic Percussion (8-15)
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba", "Xylophone",
    "Tubular Bells", "Dulcimer",
    # Organ (16-23)
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ", "Reed Organ",
    "Accordion", "Harmonica", "Tango Accordion",
    # Guitar (24-31)
    "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)",
    "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar",
    "Distortion Guitar", "Guitar Harmonics",
    # Bass (32-39)
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
    "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
    # Strings (40-47)
    "Violin", "Viola", "Cello", "Contrabass", "Tremolo Strings", "Pizzicato Strings",
    "Orchestral Harp", "Timpani",
    # Ensemble (48-55)
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2",
    "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit",
    # Brass (56-63)
    "Trumpet", "Trombone", "Tuba", "Muted Trumpet", "French Horn", "Brass Section",
    "Synth Brass 1", "Synth Brass 2",
    # Reed (64-71)
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe", "English Horn",
    "Bassoon", "Clarinet",
    # Pipe (72-79)
    "Piccolo", "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi",
    "Whistle", "Ocarina",
    # Synth Lead (80-87)
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)",
    "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)",
    # Synth Pad (88-95)
    "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)",
    "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
    # Synth Effects (96-103)
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
    # Ethnic (104-111)
    "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bagpipe", "Fiddle", "Shanai",
    # Percussive (112-119)
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko Drum", "Melodic Tom",
    "Synth Drum", "Reverse Cymbal",
    # Sound Effects (120-127)
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet", "Telephone Ring",
    "Helicopter", "Applause", "Gunshot"
]


def get_instrument_name(program: int, channel: int = 0) -> str:
    """
    Get instrument name for a program number.

    Args:
        program: MIDI program number (0-127)
        channel: MIDI channel (channel 9 is drums)

    Returns:
        Instrument name string
    """
    if channel == 9:
        return "Drums"
    if 0 <= program < len(GM_INSTRUMENTS):
        return GM_INSTRUMENTS[program]
    return f"Program {program}"


class MIDIChannelSplitter:
    """Splits MIDI file into separate WAV files per channel."""

    def __init__(self, midi_path: str, soundfont_path: Optional[str] = None,
                 use_fluidsynth: bool = True):
        """
        Initialize the splitter.

        Args:
            midi_path: Path to MIDI file
            soundfont_path: Path to SoundFont file (.sf2) for FluidSynth
            use_fluidsynth: Whether to use FluidSynth (True) or simple synthesis (False)
        """
        self.midi_path = Path(midi_path)
        self.midi_file = None
        self.ticks_per_beat = 480
        self.fluidsynth_player = None
        self.use_fluidsynth = use_fluidsynth

        # Initialize FluidSynth if requested and available
        if use_fluidsynth and FLUIDSYNTH_IMPORT_OK and is_fluidsynth_available():
            sf_path = soundfont_path
            if sf_path is None:
                sf_path = find_soundfont()

            if sf_path is not None:
                try:
                    self.fluidsynth_player = FluidSynthPlayer(
                        str(sf_path), sample_rate=44100
                    )
                    print(f"Using FluidSynth with SoundFont: {sf_path}")
                except Exception as e:
                    print(f"Warning: Failed to initialize FluidSynth: {e}")
                    print("Falling back to simple synthesis")
            else:
                print("Warning: No SoundFont found, using simple synthesis")
                print("Download a SoundFont to data/soundfonts/ for realistic sounds")
        elif use_fluidsynth and not FLUIDSYNTH_IMPORT_OK:
            print("Warning: pyfluidsynth not installed, using simple synthesis")
            print("Install with: pip install pyfluidsynth")
        elif use_fluidsynth and not is_fluidsynth_available():
            print("Warning: FluidSynth library not available, using simple synthesis")
            print("Install FluidSynth: brew install fluidsynth (macOS)")

    def load_midi(self) -> None:
        """Load and parse MIDI file."""
        print(f"Loading MIDI file: {self.midi_path}")
        self.midi_file = mido.MidiFile(str(self.midi_path))
        self.ticks_per_beat = self.midi_file.ticks_per_beat
        print(f"  Ticks per beat: {self.ticks_per_beat}")
        print(f"  Number of tracks: {len(self.midi_file.tracks)}")

    def list_channels(self) -> Dict[int, Dict]:
        """
        List all channels found in the MIDI file with note counts and instruments.

        Returns:
            Dictionary mapping channel number to info dict
        """
        if self.midi_file is None:
            self.load_midi()

        channel_info = defaultdict(lambda: {
            'note_count': 0,
            'notes': set(),
            'min_note': 127,
            'max_note': 0,
            'program': None
        })

        for track in self.midi_file.tracks:
            for msg in track:
                if msg.type == 'program_change':
                    # Capture program change (instrument assignment)
                    channel_info[msg.channel]['program'] = msg.program
                elif hasattr(msg, 'channel'):
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

            # Resolve instrument name
            program = info['program']
            if ch == 9:
                info['instrument'] = "Drums"
            elif program is not None:
                info['instrument'] = get_instrument_name(program, ch)
            else:
                info['instrument'] = None

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
                           total_duration: float = None, program: int = 0,
                           channel: int = 0) -> np.ndarray:
        """
        Synthesize audio for a list of notes.

        Args:
            notes: List of note dictionaries
            sample_rate: Audio sample rate
            total_duration: Total duration in seconds (uses MIDI file length if None)
            program: GM instrument program number (0-127)
            channel: MIDI channel (9 = drums)

        Returns:
            Audio waveform as numpy array
        """
        if not notes:
            return np.array([])

        # Use FluidSynth if available
        if self.fluidsynth_player is not None:
            return self._synthesize_fluidsynth(
                notes, total_duration, program, channel
            )

        # Fall back to simple synthesis
        return self._synthesize_simple(notes, sample_rate, total_duration)

    def _synthesize_fluidsynth(self, notes: List[Dict], total_duration: float,
                                program: int, channel: int) -> np.ndarray:
        """
        Synthesize using FluidSynth with real instrument sounds.

        Args:
            notes: List of note dictionaries
            total_duration: Total duration in seconds
            program: GM instrument program (0-127)
            channel: MIDI channel (9 = drums)

        Returns:
            Audio waveform as numpy array
        """
        # FluidSynth uses 44100 Hz sample rate
        audio = self.fluidsynth_player.render_notes(
            notes, program=program, channel=channel
        )

        # Trim or pad to exact duration
        target_samples = int((total_duration + 0.1) * 44100)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))

        return audio

    def _synthesize_simple(self, notes: List[Dict], sample_rate: int,
                           total_duration: float) -> np.ndarray:
        """
        Synthesize using simple additive synthesis (fallback).

        Args:
            notes: List of note dictionaries
            sample_rate: Audio sample rate
            total_duration: Total duration in seconds

        Returns:
            Audio waveform as numpy array
        """
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
            sample_rate: Audio sample rate (only used for simple synthesis;
                        FluidSynth always uses 44100)

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

        # Determine actual sample rate (FluidSynth uses 44100)
        actual_sample_rate = 44100 if self.fluidsynth_player else sample_rate

        for channel in sorted(channels.keys()):
            info = channels[channel]
            instrument_str = info.get('instrument', '')
            program = info.get('program', 0) or 0

            print(f"\n  Channel {channel:2d}: {info['note_count']} notes, "
                  f"range {info['note_range']}")
            if instrument_str:
                print(f"    Instrument: {instrument_str} (program {program})")

            notes = self.extract_notes_for_channel(channel)
            if not notes:
                print(f"    Skipping (no notes)")
                continue

            # Use full MIDI duration to include any trailing silence
            # Pass program and channel for FluidSynth instrument selection
            audio = self.synthesize_channel(
                notes,
                sample_rate,
                total_duration=midi_total_duration,
                program=program,
                channel=channel
            )
            if len(audio) == 0:
                print(f"    Skipping (empty audio)")
                continue

            # Create output filename
            output_path = output_dir / f"{midi_basename}_channel_{channel:02d}.wav"
            sf.write(str(output_path), audio, actual_sample_rate)

            duration = len(audio) / actual_sample_rate
            print(f"    Saved: {output_path.name} ({duration:.1f}s)")
            created_files.append(output_path)

        return created_files

    def cleanup(self):
        """Release resources (FluidSynth synth, etc.)."""
        if self.fluidsynth_player is not None:
            self.fluidsynth_player.cleanup()
            self.fluidsynth_player = None


def main():
    parser = argparse.ArgumentParser(
        description='Split MIDI file into separate WAV files per channel.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --midi song.mid --output-dir data/temp/midi_channels
  %(prog)s --midi song.mid --soundfont data/soundfonts/GeneralUser_GS.sf2
  %(prog)s --midi song.mid --list-only
  %(prog)s --midi song.mid --no-fluidsynth

FluidSynth Mode (default):
  Uses SoundFonts for realistic instrument sounds. Requires:
  - FluidSynth library: brew install fluidsynth (macOS)
  - pyfluidsynth: pip install pyfluidsynth
  - SoundFont file: Download to data/soundfonts/

Simple Mode (--no-fluidsynth):
  Uses basic additive synthesis. All instruments sound similar.
        """
    )

    parser.add_argument('--midi', required=True,
                        help='Path to MIDI file')
    parser.add_argument('--output-dir', default='data/temp/midi_channels',
                        help='Output directory for WAV files (default: data/temp/midi_channels)')
    parser.add_argument('--sample-rate', type=int, default=22050,
                        help='Audio sample rate for simple synthesis (default: 22050)')
    parser.add_argument('--soundfont', type=str,
                        help='Path to SoundFont file (.sf2) for FluidSynth')
    parser.add_argument('--no-fluidsynth', action='store_true',
                        help='Use simple synthesis instead of FluidSynth')
    parser.add_argument('--list-only', action='store_true',
                        help='List channels only, do not export')

    args = parser.parse_args()

    # Validate MIDI file
    midi_path = Path(args.midi)
    if not midi_path.exists():
        print(f"Error: MIDI file not found: {args.midi}")
        return 1

    # Validate SoundFont if specified
    if args.soundfont and not Path(args.soundfont).exists():
        print(f"Error: SoundFont not found: {args.soundfont}")
        return 1

    print("=== MIDI Channel Splitter ===\n")

    use_fluidsynth = not args.no_fluidsynth
    splitter = MIDIChannelSplitter(
        args.midi,
        soundfont_path=args.soundfont,
        use_fluidsynth=use_fluidsynth
    )

    try:
        splitter.load_midi()

        channels = splitter.list_channels()

        print("\nChannels found:")
        print("-" * 80)
        if not channels:
            print("  No note events found in any channel")
        else:
            for ch in sorted(channels.keys()):
                info = channels[ch]
                instrument_str = ""
                if info['instrument']:
                    instrument_str = f", {info['instrument']}"
                elif info['program'] is not None:
                    instrument_str = f", Program {info['program']}"

                print(f"  Channel {ch:2d}: {info['note_count']:5d} notes, "
                      f"{info['unique_notes']:3d} unique, range: {info['note_range']}{instrument_str}")
        print("-" * 80)

        if args.list_only:
            print("\n(--list-only mode, not exporting)")
            return 0

        # Export all channels
        created_files = splitter.export_all_channels(args.output_dir, args.sample_rate)

        print(f"\n=== Export Complete ===")
        synth_mode = "FluidSynth" if splitter.fluidsynth_player else "simple synthesis"
        print(f"Created {len(created_files)} WAV files using {synth_mode}")
        print(f"Output directory: {args.output_dir}")
        print("\nListen to each file to identify the vocal/melody channel,")
        print("then use that channel number with: ./test_midi_guide.sh <midi> <channel>")

        return 0

    finally:
        splitter.cleanup()


if __name__ == '__main__':
    import sys
    sys.exit(main() or 0)
