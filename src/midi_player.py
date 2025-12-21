#!/usr/bin/env python3
"""
MIDI Note Player

Plays back detected pitch sequences as MIDI notes for testing and verification.
Can play from pitch analysis JSON files and allows manual testing of note sequences.

Supports multiple playback modes:
- Audio synthesis (using sounddevice + numpy)
- MIDI file export
- Real-time playback with visualization
"""

import argparse
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not available. Install with: pip install sounddevice")

from pitch_utils import midi_to_hz, midi_to_note_name


class MIDIPlayer:
    """Plays pitch sequences as synthesized audio."""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the MIDI player.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate

    def generate_tone(self, frequency: float, duration: float,
                     attack: float = 0.005, decay: float = 0.005,
                     sustain_level: float = 1.0, release: float = 0.01) -> np.ndarray:
        """
        Generate a synthesized tone with ADSR envelope.

        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            attack: Attack time (seconds)
            decay: Decay time (seconds)
            sustain_level: Sustain amplitude (0-1)
            release: Release time (seconds)

        Returns:
            Audio waveform as numpy array
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)

        # Generate sine wave
        waveform = np.sin(2 * np.pi * frequency * t)

        # Add harmonics for richer sound (simple additive synthesis)
        waveform += 0.5 * np.sin(4 * np.pi * frequency * t)  # 2nd harmonic
        waveform += 0.25 * np.sin(6 * np.pi * frequency * t)  # 3rd harmonic

        # Apply ADSR envelope
        envelope = self._generate_adsr_envelope(
            num_samples, attack, decay, sustain_level, release
        )
        waveform *= envelope

        # Normalize
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform)) * 0.5

        return waveform

    def _generate_adsr_envelope(self, num_samples: int, attack: float,
                               decay: float, sustain_level: float,
                               release: float) -> np.ndarray:
        """Generate ADSR envelope."""
        envelope = np.ones(num_samples)

        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)

        # Attack phase
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

        # Decay phase
        if decay_samples > 0:
            start = attack_samples
            end = start + decay_samples
            if end <= num_samples:
                envelope[start:end] = np.linspace(1, sustain_level, decay_samples)

        # Sustain phase (already set to sustain_level or 1)
        sustain_start = attack_samples + decay_samples
        sustain_end = max(sustain_start, num_samples - release_samples)
        if sustain_start < sustain_end:
            envelope[sustain_start:sustain_end] = sustain_level

        # Release phase
        if release_samples > 0 and num_samples - release_samples >= 0:
            start = num_samples - release_samples
            envelope[start:] = np.linspace(sustain_level, 0, release_samples)

        return envelope

    def play_note_sequence(self, notes: List[Dict], gap_duration: float = 0.05,
                          verbose: bool = True) -> np.ndarray:
        """
        Play a sequence of notes.

        Args:
            notes: List of note dictionaries with keys:
                   - pitch_midi: MIDI note number
                   - duration: Duration in seconds
                   - (optional) pitch_hz: Frequency in Hz
            gap_duration: Gap between notes in seconds
            verbose: Print playback progress

        Returns:
            Complete audio waveform
        """
        audio_segments = []

        for i, note in enumerate(notes):
            midi = note.get('pitch_midi')
            duration = note.get('duration', 0.5)

            if midi is None or midi <= 0:
                # Silent note
                silence = np.zeros(int(duration * self.sample_rate))
                audio_segments.append(silence)
            else:
                # Get frequency
                if 'pitch_hz' in note:
                    freq = note['pitch_hz']
                else:
                    freq = midi_to_hz(midi)

                # Generate tone
                tone = self.generate_tone(freq, duration)
                audio_segments.append(tone)

                if verbose:
                    note_name = midi_to_note_name(int(midi))
                    print(f"  Note {i+1}/{len(notes)}: {note_name} ({freq:.1f} Hz) - {duration:.2f}s")

            # Add gap between notes (except after last note)
            if i < len(notes) - 1 and gap_duration > 0:
                gap = np.zeros(int(gap_duration * self.sample_rate))
                audio_segments.append(gap)

        # Concatenate all segments
        complete_audio = np.concatenate(audio_segments)
        return complete_audio

    def play_audio(self, audio: np.ndarray, blocking: bool = True):
        """
        Play audio through speakers.

        Args:
            audio: Audio waveform
            blocking: Wait for playback to complete
        """
        if not SOUNDDEVICE_AVAILABLE:
            print("Error: sounddevice not installed. Cannot play audio.")
            print("Install with: pip install sounddevice")
            return

        sd.play(audio, self.sample_rate, blocking=blocking)

    def save_audio(self, audio: np.ndarray, output_path: str):
        """
        Save audio to file.

        Args:
            audio: Audio waveform
            output_path: Output file path (.wav)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(output_path), audio, self.sample_rate)
        print(f"Saved audio to: {output_path}")


def load_pitch_sequence_from_json(json_path: str) -> List[Dict]:
    """
    Load pitch sequence from guide/source analysis JSON.

    Args:
        json_path: Path to pitch analysis JSON file

    Returns:
        List of note dictionaries
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    notes = []

    # Handle different JSON formats
    if 'guide_sequence' in data:
        # Guide video format
        for segment in data['guide_sequence']:
            notes.append({
                'pitch_midi': segment.get('pitch_midi'),
                'pitch_hz': segment.get('pitch_hz'),
                'duration': segment.get('duration'),
                'note_name': segment.get('pitch_note'),
                'confidence': segment.get('pitch_confidence', 1.0)
            })

    elif 'pitch_database' in data:
        # Source video format
        for clip in data['pitch_database']:
            notes.append({
                'pitch_midi': clip.get('pitch_midi'),
                'pitch_hz': clip.get('pitch_hz'),
                'duration': clip.get('duration'),
                'note_name': clip.get('pitch_note'),
                'confidence': clip.get('pitch_confidence', 1.0)
            })

    elif 'segments' in data:
        # Generic segments format
        for segment in data['segments']:
            notes.append({
                'pitch_midi': segment.get('pitch_midi'),
                'pitch_hz': segment.get('pitch_hz'),
                'duration': segment.get('duration'),
                'note_name': segment.get('pitch_note'),
                'confidence': segment.get('pitch_confidence', 1.0)
            })

    else:
        print(f"Warning: Unknown JSON format in {json_path}")

    return notes


def play_melody_from_notes(note_names: List[str], duration_per_note: float = 0.5,
                           output_path: Optional[str] = None) -> np.ndarray:
    """
    Quick helper to play a melody from note names.

    Args:
        note_names: List of note names (e.g., ['C4', 'D4', 'E4'])
        duration_per_note: Duration for each note in seconds
        output_path: Optional path to save audio file

    Returns:
        Audio waveform

    Examples:
        >>> play_melody_from_notes(['C4', 'E4', 'G4', 'C5'])
    """
    from pitch_utils import note_name_to_midi

    notes = []
    for name in note_names:
        midi = note_name_to_midi(name)
        if midi is not None:
            notes.append({
                'pitch_midi': midi,
                'duration': duration_per_note
            })

    player = MIDIPlayer()
    audio = player.play_note_sequence(notes, verbose=True)

    if output_path:
        player.save_audio(audio, output_path)

    if SOUNDDEVICE_AVAILABLE:
        print("\nPlaying melody...")
        player.play_audio(audio, blocking=True)

    return audio


def main():
    parser = argparse.ArgumentParser(
        description="Play MIDI note sequences from pitch analysis files"
    )
    parser.add_argument(
        '--json',
        type=str,
        help='Path to pitch analysis JSON file'
    )
    parser.add_argument(
        '--melody',
        type=str,
        help='Comma-separated note names (e.g., "C4,E4,G4,C5")'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=0.5,
        help='Duration per note in seconds (default: 0.5)'
    )
    parser.add_argument(
        '--gap',
        type=float,
        default=0.05,
        help='Gap between notes in seconds (default: 0.05)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output audio file path (.wav)'
    )
    parser.add_argument(
        '--no-play',
        action='store_true',
        help='Do not play audio (only save to file)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of notes to play'
    )

    args = parser.parse_args()

    player = MIDIPlayer()

    # Load notes
    notes = []

    if args.json:
        print(f"Loading pitch sequence from: {args.json}")
        notes = load_pitch_sequence_from_json(args.json)
        print(f"Loaded {len(notes)} notes")

    elif args.melody:
        from pitch_utils import note_name_to_midi
        note_names = args.melody.split(',')
        print(f"Playing melody: {' '.join(note_names)}")

        for name in note_names:
            name = name.strip()
            midi = note_name_to_midi(name)
            if midi is not None:
                notes.append({
                    'pitch_midi': midi,
                    'duration': args.duration
                })
            else:
                print(f"Warning: Invalid note name '{name}', skipping")

    else:
        # Default: Play a test scale (C major)
        print("No input provided. Playing C major scale as test...")
        test_notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
        from pitch_utils import note_name_to_midi

        for name in test_notes:
            notes.append({
                'pitch_midi': note_name_to_midi(name),
                'duration': args.duration
            })

    if not notes:
        print("Error: No notes to play")
        return

    # Limit notes if requested
    if args.limit:
        notes = notes[:args.limit]
        print(f"Limited to first {args.limit} notes")

    # Generate audio
    print(f"\nGenerating {len(notes)} notes...")
    audio = player.play_note_sequence(notes, gap_duration=args.gap, verbose=True)

    total_duration = len(audio) / player.sample_rate
    print(f"\nTotal duration: {total_duration:.2f}s")

    # Save to file if requested
    if args.output:
        player.save_audio(audio, args.output)

    # Play audio
    if not args.no_play and SOUNDDEVICE_AVAILABLE:
        print("\nPlaying audio...")
        player.play_audio(audio, blocking=True)
        print("Playback complete!")
    elif not args.no_play and not SOUNDDEVICE_AVAILABLE:
        print("\nCannot play audio: sounddevice not installed")
        print("Install with: pip install sounddevice")


if __name__ == "__main__":
    main()
