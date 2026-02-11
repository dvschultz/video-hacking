#!/usr/bin/env python3
"""
MIDI Renderer

Renders a MIDI file to a single WAV file with all channels merged.

Unlike midi_channel_splitter.py which exports each channel separately,
this script combines all channels into one stereo/mono audio file.

Supports two synthesis modes:
- FluidSynth (default): Uses SoundFont for realistic instrument sounds
- Simple synthesis: Uses basic sine wave additive synthesis (fallback)

Usage:
    python midi_renderer.py --midi song.mid --output song.wav
    python midi_renderer.py --midi song.mid --soundfont data/soundfonts/GeneralUser_GS.sf2
    python midi_renderer.py --midi song.mid --no-fluidsynth
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional

try:
    import mido
except ImportError:
    print("Error: mido not installed. Install with: pip install mido")
    import sys
    sys.exit(1)

from midi_channel_splitter import MIDIChannelSplitter


def render_midi_to_wav(
    midi_path: str,
    output_path: str,
    soundfont_path: Optional[str] = None,
    use_fluidsynth: bool = True,
    sample_rate: int = 22050
) -> Path:
    """
    Render a MIDI file to a single WAV file with all channels merged.

    Args:
        midi_path: Path to MIDI file
        output_path: Path for output WAV file
        soundfont_path: Path to SoundFont file (.sf2) for FluidSynth
        use_fluidsynth: Whether to use FluidSynth (True) or simple synthesis (False)
        sample_rate: Audio sample rate for simple synthesis (FluidSynth uses 44100)

    Returns:
        Path to created WAV file
    """
    splitter = MIDIChannelSplitter(
        midi_path,
        soundfont_path=soundfont_path,
        use_fluidsynth=use_fluidsynth
    )

    try:
        splitter.load_midi()

        channels = splitter.list_channels()
        if not channels:
            raise ValueError(f"No channels with notes found in MIDI file: {midi_path}")

        midi_duration = splitter.midi_file.length
        print(f"\nMIDI file duration: {midi_duration:.2f}s")
        print(f"Channels to render: {sorted(channels.keys())}")

        # Determine actual sample rate (FluidSynth uses 44100)
        actual_sample_rate = 44100 if splitter.fluidsynth_player else sample_rate

        # Render each channel
        channel_audio = []

        for channel in sorted(channels.keys()):
            info = channels[channel]
            program = info.get('program', 0) or 0
            instrument = info.get('instrument', f'Program {program}')

            print(f"\n  Rendering channel {channel}: {info['note_count']} notes ({instrument})")

            notes = splitter.extract_notes_for_channel(channel)
            if not notes:
                print(f"    Skipping (no notes)")
                continue

            audio = splitter.synthesize_channel(
                notes,
                sample_rate=sample_rate,
                total_duration=midi_duration,
                program=program,
                channel=channel
            )

            if len(audio) == 0:
                print(f"    Skipping (empty audio)")
                continue

            channel_audio.append(audio)
            print(f"    Rendered {len(audio)} samples")

        if not channel_audio:
            raise ValueError("No audio rendered from any channel")

        # Mix all channels together
        print(f"\nMixing {len(channel_audio)} channels...")

        # Find maximum length and pad all to same length
        max_len = max(len(a) for a in channel_audio)
        mixed = np.zeros(max_len, dtype=np.float64)

        for audio in channel_audio:
            mixed[:len(audio)] += audio

        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed = mixed / max_val * 0.9  # Leave some headroom

        # Convert to float32 for output
        mixed = mixed.astype(np.float32)

        # Write output file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(output_path), mixed, actual_sample_rate)

        duration = len(mixed) / actual_sample_rate
        print(f"\nSaved: {output_path} ({duration:.1f}s, {actual_sample_rate} Hz)")

        return output_path

    finally:
        splitter.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description='Render MIDI file to WAV with all channels merged.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --midi song.mid --output song.wav
  %(prog)s --midi song.mid  # Output defaults to song.wav
  %(prog)s --midi song.mid --soundfont data/soundfonts/GeneralUser_GS.sf2
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
    parser.add_argument('--output', '-o',
                        help='Output WAV path (default: <midi_name>.wav)')
    parser.add_argument('--sample-rate', type=int, default=22050,
                        help='Audio sample rate for simple synthesis (default: 22050)')
    parser.add_argument('--soundfont', type=str,
                        help='Path to SoundFont file (.sf2) for FluidSynth')
    parser.add_argument('--no-fluidsynth', action='store_true',
                        help='Use simple synthesis instead of FluidSynth')

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

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = midi_path.with_suffix('.wav')

    print("=== MIDI Renderer ===\n")
    print(f"Input:  {midi_path}")
    print(f"Output: {output_path}")

    use_fluidsynth = not args.no_fluidsynth

    try:
        render_midi_to_wav(
            str(midi_path),
            str(output_path),
            soundfont_path=args.soundfont,
            use_fluidsynth=use_fluidsynth,
            sample_rate=args.sample_rate
        )
        print("\n=== Render Complete ===")
        return 0

    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main() or 0)
