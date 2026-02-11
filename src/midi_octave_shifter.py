#!/usr/bin/env python3
"""
MIDI Octave Shifter

Shifts all notes on a specific MIDI channel by a given number of octaves.

Usage:
    python midi_octave_shifter.py --midi input.mid --channel 0 --octaves -1 --output output.mid
    python midi_octave_shifter.py --midi input.mid --channel 0 --octaves 1 --list-channels
"""

import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

try:
    import mido
except ImportError:
    print("Error: mido not installed. Install with: pip install mido")
    import sys
    sys.exit(1)


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


class MIDIOctaveShifter:
    """Shifts notes on a MIDI channel by octaves."""

    def __init__(self, midi_path: str, channel: int, octave_shift: int):
        """
        Initialize the octave shifter.

        Args:
            midi_path: Path to input MIDI file
            channel: MIDI channel to shift (0-15)
            octave_shift: Number of octaves to shift (+/- integer)
        """
        self.midi_path = Path(midi_path)
        self.channel = channel
        self.octave_shift = octave_shift
        self.midi_file = None

    def load(self) -> bool:
        """Load the MIDI file."""
        if not self.midi_path.exists():
            print(f"Error: MIDI file not found: {self.midi_path}")
            return False

        self.midi_file = mido.MidiFile(str(self.midi_path))
        return True

    def shift_note(self, note: int) -> int:
        """
        Shift note by octaves, clamping to valid MIDI range.

        Args:
            note: Original MIDI note number (0-127)

        Returns:
            Shifted note number clamped to 0-127
        """
        shifted = note + (self.octave_shift * 12)
        return max(0, min(127, shifted))

    def shift(self) -> mido.MidiFile:
        """
        Shift all notes on the target channel by the specified octaves.

        Returns:
            New MidiFile with shifted notes
        """
        if self.midi_file is None:
            raise ValueError("MIDI file not loaded. Call load() first.")

        # Create new MIDI file with same properties
        new_midi = mido.MidiFile(ticks_per_beat=self.midi_file.ticks_per_beat)

        shifted_count = 0
        clamped_count = 0

        for track in self.midi_file.tracks:
            new_track = mido.MidiTrack()

            for msg in track:
                if hasattr(msg, 'channel') and msg.channel == self.channel:
                    if msg.type in ('note_on', 'note_off'):
                        original_note = msg.note
                        shifted_note = self.shift_note(original_note)

                        # Track clamped notes
                        expected_note = original_note + (self.octave_shift * 12)
                        if shifted_note != expected_note:
                            clamped_count += 1

                        # Create modified message
                        msg = msg.copy(note=shifted_note)
                        shifted_count += 1

                new_track.append(msg)

            new_midi.tracks.append(new_track)

        print(f"Shifted {shifted_count} note events by {self.octave_shift:+d} octave(s)")
        if clamped_count > 0:
            print(f"  Warning: {clamped_count} notes were clamped to MIDI range 0-127")

        return new_midi

    def get_notes_for_preview(self) -> list:
        """
        Extract notes from the shifted channel for audio preview.

        Returns:
            List of note dictionaries compatible with MIDIPlayer
        """
        if self.midi_file is None:
            raise ValueError("MIDI file not loaded. Call load() first.")

        # Get tempo from MIDI file
        tempo = 500000  # Default: 120 BPM
        for track in self.midi_file.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break

        ticks_per_beat = self.midi_file.ticks_per_beat

        # Collect note events with absolute times
        note_events = []
        for track in self.midi_file.tracks:
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if hasattr(msg, 'channel') and msg.channel == self.channel:
                    if msg.type in ('note_on', 'note_off'):
                        note_events.append({
                            'type': msg.type,
                            'note': self.shift_note(msg.note),
                            'velocity': msg.velocity if msg.type == 'note_on' else 0,
                            'time': abs_time
                        })

        # Sort by time
        note_events.sort(key=lambda x: x['time'])

        # Convert to note durations
        notes = []
        active_notes = {}  # note -> start_time

        for event in note_events:
            note = event['note']
            is_note_on = event['type'] == 'note_on' and event['velocity'] > 0

            if is_note_on:
                active_notes[note] = event['time']
            else:
                if note in active_notes:
                    start_time = active_notes[note]
                    duration_ticks = event['time'] - start_time
                    # Convert ticks to seconds
                    duration_sec = mido.tick2second(duration_ticks, ticks_per_beat, tempo)
                    start_sec = mido.tick2second(start_time, ticks_per_beat, tempo)

                    notes.append({
                        'pitch_midi': note,
                        'duration': max(0.05, duration_sec),  # Minimum 50ms
                        'start_time': start_sec
                    })
                    del active_notes[note]

        # Sort by start time and simplify to sequential playback
        notes.sort(key=lambda x: x['start_time'])

        # Remove start_time for MIDIPlayer compatibility
        for note in notes:
            del note['start_time']

        return notes


def list_channels(midi_file: mido.MidiFile) -> dict:
    """
    List all channels found in the MIDI file with note counts and instruments.

    Args:
        midi_file: Loaded mido MidiFile object

    Returns:
        Dictionary mapping channel number to info dict
    """
    channel_info = defaultdict(lambda: {
        'note_count': 0,
        'notes': set(),
        'program': None,
        'instrument_name': None
    })

    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'program_change':
                # Capture program change (instrument assignment)
                channel_info[msg.channel]['program'] = msg.program
            elif hasattr(msg, 'channel'):
                if msg.type == 'note_on' and msg.velocity > 0:
                    channel_info[msg.channel]['note_count'] += 1
                    channel_info[msg.channel]['notes'].add(msg.note)

    # Convert sets to counts and resolve instrument names
    for ch in channel_info:
        notes = channel_info[ch]['notes']
        channel_info[ch]['unique_notes'] = len(notes)
        if notes:
            channel_info[ch]['pitch_range'] = f"{min(notes)}-{max(notes)}"
        else:
            channel_info[ch]['pitch_range'] = "N/A"
        del channel_info[ch]['notes']

        # Resolve instrument name
        program = channel_info[ch]['program']
        if ch == 9:
            channel_info[ch]['instrument'] = "Drums"
        elif program is not None:
            channel_info[ch]['instrument'] = get_instrument_name(program, ch)
        else:
            channel_info[ch]['instrument'] = None

    return dict(channel_info)


def generate_output_path(input_path: Path, octave_shift: int, suffix: str = '') -> Path:
    """
    Generate output path with octave shift in filename.

    Args:
        input_path: Input file path
        octave_shift: Octave shift value
        suffix: Additional suffix before extension

    Returns:
        Generated output path
    """
    stem = input_path.stem
    ext = input_path.suffix
    sign = '+' if octave_shift >= 0 else ''
    return input_path.parent / f"{stem}_octave{sign}{octave_shift}{suffix}{ext}"


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Shift MIDI channel by octaves"
    )
    parser.add_argument(
        '--midi',
        type=str,
        required=True,
        help='Path to input MIDI file'
    )
    parser.add_argument(
        '--channel',
        type=int,
        required=True,
        help='MIDI channel to shift (0-15)'
    )
    parser.add_argument(
        '--octaves',
        type=int,
        required=False,
        help='Octaves to shift (negative=down, positive=up)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=False,
        help='Output MIDI path (default: auto-generated)'
    )
    parser.add_argument(
        '--audio-output',
        type=str,
        required=False,
        help='Output WAV path (default: auto-generated)'
    )
    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Skip audio preview generation'
    )
    parser.add_argument(
        '--list-channels',
        action='store_true',
        help='List channels in MIDI file and exit'
    )

    args = parser.parse_args()

    print("=== MIDI Octave Shifter ===\n")

    # Validate channel
    if not 0 <= args.channel <= 15:
        print(f"Error: Channel must be 0-15, got {args.channel}")
        return 1

    # Check input file exists
    midi_path = Path(args.midi)
    if not midi_path.exists():
        print(f"Error: MIDI file not found: {args.midi}")
        return 1

    # Load MIDI file
    print(f"Loading MIDI file: {args.midi}")
    midi_file = mido.MidiFile(str(midi_path))
    print(f"  Ticks per beat: {midi_file.ticks_per_beat}")
    print(f"  Number of tracks: {len(midi_file.tracks)}")

    # List channels mode
    if args.list_channels:
        channels = list_channels(midi_file)

        print("\nChannels found in MIDI file:")
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

                print(f"  Channel {ch:2d}: {info['note_count']:4d} notes, "
                      f"{info['unique_notes']:3d} unique, "
                      f"range: {info['pitch_range']:>7s}{instrument_str}")

        print("-" * 80)
        return 0

    # Require octaves for non-list mode
    if args.octaves is None:
        print("Error: --octaves is required when not using --list-channels")
        return 1

    # Check if channel exists
    channels = list_channels(midi_file)
    if args.channel not in channels:
        print(f"\nWarning: Channel {args.channel} has no note events")
        print("Available channels:", sorted(channels.keys()))
        return 1

    # Show original channel info
    ch_info = channels[args.channel]
    print(f"\nChannel {args.channel}: {ch_info['note_count']} notes, "
          f"range: {ch_info['pitch_range']}")
    print(f"Shifting by {args.octaves:+d} octave(s) ({args.octaves * 12:+d} semitones)")

    # Create shifter and process
    shifter = MIDIOctaveShifter(args.midi, args.channel, args.octaves)
    shifter.load()
    new_midi = shifter.shift()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = generate_output_path(midi_path, args.octaves)

    # Save MIDI output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_midi.save(str(output_path))
    print(f"\nSaved shifted MIDI to: {output_path}")

    # Show new pitch range
    new_channels = list_channels(new_midi)
    if args.channel in new_channels:
        new_info = new_channels[args.channel]
        print(f"  New range: {new_info['pitch_range']}")

    # Generate audio preview unless skipped
    if not args.no_audio:
        print("\nGenerating audio preview...")
        try:
            # Import MIDIPlayer from same directory
            import sys
            src_dir = Path(__file__).parent
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))

            from midi_player import MIDIPlayer

            # Get notes from shifted channel
            notes = shifter.get_notes_for_preview()

            if notes:
                player = MIDIPlayer()
                audio = player.play_note_sequence(notes, gap_duration=0.02, verbose=False)

                # Determine audio output path
                if args.audio_output:
                    audio_path = Path(args.audio_output)
                else:
                    audio_path = generate_output_path(midi_path, args.octaves, '_preview')
                    audio_path = audio_path.with_suffix('.wav')

                player.save_audio(audio, str(audio_path))
                print(f"  Generated {len(notes)} notes, {len(audio) / player.sample_rate:.2f}s duration")
            else:
                print("  Warning: No notes found for preview")

        except ImportError as e:
            print(f"  Warning: Could not import MIDIPlayer: {e}")
            print("  Audio preview skipped")
        except Exception as e:
            print(f"  Warning: Audio preview failed: {e}")

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
