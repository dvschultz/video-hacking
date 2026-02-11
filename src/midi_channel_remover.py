#!/usr/bin/env python3
"""
MIDI Channel Remover

Removes a specific MIDI channel from a MIDI file and saves the result.

Usage:
    python midi_channel_remover.py --midi input.mid --channel 5 --output output.mid
    python midi_channel_remover.py --midi input.mid --channel 0 --list-channels
"""

import argparse
from pathlib import Path
from collections import defaultdict

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


def remove_channel(midi_file: mido.MidiFile, channel: int) -> mido.MidiFile:
    """
    Remove all messages from a specific channel.

    Args:
        midi_file: Loaded mido MidiFile object
        channel: Channel to remove (0-15)

    Returns:
        New MidiFile with the channel removed
    """
    # Create new MIDI file with same properties
    new_midi = mido.MidiFile(ticks_per_beat=midi_file.ticks_per_beat)

    removed_count = 0

    for track in midi_file.tracks:
        new_track = mido.MidiTrack()
        accumulated_time = 0

        for msg in track:
            # Check if this message should be removed
            if hasattr(msg, 'channel') and msg.channel == channel:
                # Accumulate the time for the next message we keep
                accumulated_time += msg.time
                removed_count += 1
            else:
                # Keep this message, adding any accumulated time
                new_msg = msg.copy(time=msg.time + accumulated_time)
                new_track.append(new_msg)
                accumulated_time = 0

        new_midi.tracks.append(new_track)

    print(f"Removed {removed_count} messages from channel {channel}")
    return new_midi


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Remove a specific MIDI channel from a MIDI file"
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
        help='MIDI channel to remove (0-15)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=False,
        help='Path to output MIDI file (required unless using --list-channels)'
    )
    parser.add_argument(
        '--list-channels',
        action='store_true',
        help='List available channels in MIDI file and exit'
    )

    args = parser.parse_args()

    print("=== MIDI Channel Remover ===\n")

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

    # Require output for non-list mode
    if not args.output:
        print("Error: --output is required when not using --list-channels")
        return 1

    # Check if channel exists
    channels = list_channels(midi_file)
    if args.channel not in channels:
        print(f"\nWarning: Channel {args.channel} has no note events")
        print("Available channels:", sorted(channels.keys()))
        print("Proceeding anyway (will remove any non-note messages on this channel)...")

    # Remove the channel
    print(f"\nRemoving channel {args.channel}...")
    new_midi = remove_channel(midi_file, args.channel)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_midi.save(str(output_path))
    print(f"\nSaved to: {output_path}")

    # Show remaining channels
    remaining = list_channels(new_midi)
    print("\nRemaining channels:")
    if not remaining:
        print("  No channels with notes remaining")
    else:
        for ch in sorted(remaining.keys()):
            info = remaining[ch]
            print(f"  Channel {ch:2d}: {info['note_count']} notes")

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
