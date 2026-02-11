#!/usr/bin/env python3
"""
Merge multiple MIDI channels into a single channel and export a new MIDI file.

Useful when an instrument's notes are split across multiple channels
(e.g., channel 0 and 1 both playing piano). Merges all note events
onto a single target channel.

Usage:
    python merge_midi_channels.py --midi song.mid --channels 0,1 --output merged.mid
    python merge_midi_channels.py --midi song.mid --list-channels
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

try:
    import mido
except ImportError:
    print("Error: mido not installed. Install with: pip install mido")
    sys.exit(1)


def list_channels(midi_path):
    """List all channels in a MIDI file with note counts and pitch ranges."""
    mid = mido.MidiFile(midi_path)

    channel_info = defaultdict(lambda: {'note_count': 0, 'notes': set()})

    for track_idx, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                if hasattr(msg, 'channel'):
                    channel_info[msg.channel]['note_count'] += 1
                    channel_info[msg.channel]['notes'].add(msg.note)

    print(f"\nMIDI file: {midi_path}")
    print(f"Type: {mid.type}, Tracks: {len(mid.tracks)}, Ticks/beat: {mid.ticks_per_beat}")
    print(f"\nChannels found:")

    if not channel_info:
        print("  No note events found in any channel")
        return

    for ch in sorted(channel_info.keys()):
        info = channel_info[ch]
        notes = sorted(info['notes'])
        note_range = f"MIDI {notes[0]}-{notes[-1]}" if notes else "none"
        print(f"  Channel {ch:2d}: {info['note_count']:4d} notes, "
              f"{len(notes):3d} unique pitches, range: {note_range}")


def merge_channels(midi_path, channels, target_channel, output_path):
    """
    Merge note events from multiple channels onto a single target channel.

    Non-note messages (tempo, control change, etc.) are preserved.
    Note events from the specified channels are remapped to the target channel.
    Note events from other channels are dropped.
    """
    mid = mido.MidiFile(midi_path)
    new_mid = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)

    channels_set = set(channels)
    total_remapped = 0
    total_dropped = 0
    total_kept = 0

    for track_idx, track in enumerate(mid.tracks):
        new_track = mido.MidiTrack()
        pending_delta = 0  # Accumulate delta time from dropped messages

        for msg in track:
            if hasattr(msg, 'channel'):
                if msg.channel in channels_set:
                    # Remap to target channel, absorbing any pending delta
                    new_msg = msg.copy(channel=target_channel, time=msg.time + pending_delta)
                    pending_delta = 0
                    new_track.append(new_msg)
                    if msg.type in ('note_on', 'note_off'):
                        total_remapped += 1
                    else:
                        total_kept += 1
                else:
                    # Drop note events from other channels, keep non-note messages
                    if msg.type in ('note_on', 'note_off'):
                        pending_delta += msg.time
                        total_dropped += 1
                    else:
                        new_msg = msg.copy(time=msg.time + pending_delta)
                        pending_delta = 0
                        new_track.append(new_msg)
                        total_kept += 1
            else:
                # Non-channel messages (tempo, time signature, etc.) — always keep
                new_msg = msg.copy(time=msg.time + pending_delta)
                pending_delta = 0
                new_track.append(new_msg)
                total_kept += 1

        new_mid.tracks.append(new_track)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_mid.save(str(output_path))

    channels_str = ', '.join(str(c) for c in channels)
    print(f"\nMerged channels [{channels_str}] → channel {target_channel}")
    print(f"  Note events remapped: {total_remapped}")
    print(f"  Note events dropped (other channels): {total_dropped}")
    print(f"  Non-note messages kept: {total_kept}")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple MIDI channels into one and export a new MIDI file.'
    )
    parser.add_argument('--midi', required=True, help='Input MIDI file')
    parser.add_argument('--channels', type=str,
                        help='Comma-separated channels to merge (e.g., 0,1)')
    parser.add_argument('--target-channel', type=int, default=None,
                        help='Target channel for merged notes (default: first in --channels)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output MIDI file path (default: input_merged.mid)')
    parser.add_argument('--list-channels', action='store_true',
                        help='List channels in MIDI file and exit')

    args = parser.parse_args()

    midi_path = Path(args.midi)
    if not midi_path.exists():
        print(f"Error: MIDI file not found: {midi_path}", file=sys.stderr)
        sys.exit(1)

    if args.list_channels:
        list_channels(str(midi_path))
        sys.exit(0)

    if not args.channels:
        print("Error: --channels is required (e.g., --channels 0,1)", file=sys.stderr)
        print("Use --list-channels to see available channels")
        sys.exit(1)

    # Parse channel list
    try:
        channels = [int(c.strip()) for c in args.channels.split(',')]
    except ValueError:
        print(f"Error: Invalid channel list: {args.channels}", file=sys.stderr)
        sys.exit(1)

    for ch in channels:
        if not 0 <= ch <= 15:
            print(f"Error: Channel must be 0-15, got {ch}", file=sys.stderr)
            sys.exit(1)

    if len(channels) < 2:
        print("Warning: Only one channel specified, nothing to merge", file=sys.stderr)

    target_channel = args.target_channel if args.target_channel is not None else channels[0]

    # Default output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(midi_path.with_stem(midi_path.stem + '_merged'))

    print(f"Input: {midi_path}")
    print(f"Merging channels: {channels}")
    print(f"Target channel: {target_channel}")

    # Show before state
    list_channels(str(midi_path))

    merge_channels(str(midi_path), channels, target_channel, output_path)

    # Show after state
    print("\n--- Merged file ---")
    list_channels(output_path)


if __name__ == '__main__':
    main()
