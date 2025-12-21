#!/usr/bin/env python3
"""
Pitch Utilities Module

Provides conversion functions and utilities for pitch manipulation:
- Hz ↔ MIDI note number conversions
- MIDI ↔ note name conversions
- Pitch distance calculations (cents)
- Pitch statistics and analysis
"""

import numpy as np
from typing import Union, List, Tuple, Optional


# MIDI note names (C0 = MIDI 12, A4 = MIDI 69 = 440 Hz)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def hz_to_midi(frequency_hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert frequency in Hz to MIDI note number.

    Uses the standard formula: MIDI = 69 + 12 * log2(f / 440)
    where A4 = 440 Hz = MIDI note 69

    Args:
        frequency_hz: Frequency in Hz (can be scalar or array)

    Returns:
        MIDI note number(s) as float (can have decimals for pitch bends)

    Examples:
        >>> hz_to_midi(440.0)  # A4
        69.0
        >>> hz_to_midi(261.63)  # C4
        60.0
    """
    if isinstance(frequency_hz, np.ndarray):
        # Handle arrays, filtering out zeros/negatives
        valid_mask = frequency_hz > 0
        result = np.zeros_like(frequency_hz)
        result[valid_mask] = 69 + 12 * np.log2(frequency_hz[valid_mask] / 440.0)
        result[~valid_mask] = np.nan
        return result
    else:
        if frequency_hz <= 0:
            return np.nan
        return 69 + 12 * np.log2(frequency_hz / 440.0)


def midi_to_hz(midi_note: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert MIDI note number to frequency in Hz.

    Uses the formula: f = 440 * 2^((MIDI - 69) / 12)

    Args:
        midi_note: MIDI note number (0-127, can be float for pitch bends)

    Returns:
        Frequency in Hz

    Examples:
        >>> midi_to_hz(69)  # A4
        440.0
        >>> midi_to_hz(60)  # C4
        261.63
    """
    return 440.0 * np.power(2.0, (midi_note - 69) / 12.0)


def midi_to_note_name(midi_note: int, include_octave: bool = True) -> str:
    """
    Convert MIDI note number to note name (e.g., 'C4', 'A#5').

    Args:
        midi_note: MIDI note number (0-127)
        include_octave: Include octave number in output

    Returns:
        Note name string (e.g., 'C4', 'D#5')

    Examples:
        >>> midi_to_note_name(60)
        'C4'
        >>> midi_to_note_name(69)
        'A4'
        >>> midi_to_note_name(61)
        'C#4'
    """
    if not (0 <= midi_note <= 127):
        return "Unknown"

    note_name = NOTE_NAMES[midi_note % 12]

    if include_octave:
        # MIDI octave: C4 (middle C) = MIDI 60
        octave = (midi_note // 12) - 1
        return f"{note_name}{octave}"
    else:
        return note_name


def note_name_to_midi(note_name: str) -> Optional[int]:
    """
    Convert note name to MIDI note number.

    Args:
        note_name: Note name (e.g., 'C4', 'A#5', 'Db3')

    Returns:
        MIDI note number, or None if invalid

    Examples:
        >>> note_name_to_midi('C4')
        60
        >>> note_name_to_midi('A4')
        69
    """
    # Normalize: convert 'b' (flat) to '#' equivalent
    note_name = note_name.strip().upper()

    # Extract note and octave
    if len(note_name) < 2:
        return None

    # Handle flats by converting to sharps
    note_name = note_name.replace('DB', 'C#').replace('EB', 'D#').replace('GB', 'F#') \
                         .replace('AB', 'G#').replace('BB', 'A#')

    # Parse note and octave
    if note_name[1] == '#':
        note = note_name[:2]
        octave_str = note_name[2:]
    else:
        note = note_name[0]
        octave_str = note_name[1:]

    try:
        octave = int(octave_str)
    except ValueError:
        return None

    if note not in NOTE_NAMES:
        return None

    note_index = NOTE_NAMES.index(note)
    midi_note = (octave + 1) * 12 + note_index

    if 0 <= midi_note <= 127:
        return midi_note
    else:
        return None


def pitch_distance_cents(freq1_hz: float, freq2_hz: float) -> float:
    """
    Calculate pitch distance in cents (1/100th of a semitone).

    100 cents = 1 semitone
    1200 cents = 1 octave

    Args:
        freq1_hz: First frequency in Hz
        freq2_hz: Second frequency in Hz

    Returns:
        Distance in cents (positive or negative)

    Examples:
        >>> pitch_distance_cents(440, 440)  # Same pitch
        0.0
        >>> pitch_distance_cents(440, 466.16)  # A4 to A#4 (1 semitone)
        100.0
    """
    if freq1_hz <= 0 or freq2_hz <= 0:
        return np.nan

    return 1200 * np.log2(freq2_hz / freq1_hz)


def pitch_distance_semitones(freq1_hz: float, freq2_hz: float) -> float:
    """
    Calculate pitch distance in semitones.

    Args:
        freq1_hz: First frequency in Hz
        freq2_hz: Second frequency in Hz

    Returns:
        Distance in semitones (positive or negative)
    """
    return pitch_distance_cents(freq1_hz, freq2_hz) / 100.0


def round_to_nearest_midi(frequency_hz: float) -> int:
    """
    Round a frequency to the nearest MIDI note number.

    Args:
        frequency_hz: Frequency in Hz

    Returns:
        Nearest integer MIDI note number (0-127)
    """
    midi_float = hz_to_midi(frequency_hz)
    if np.isnan(midi_float):
        return 0
    return int(np.round(np.clip(midi_float, 0, 127)))


def calculate_pitch_statistics(pitch_values_hz: np.ndarray,
                               confidence_values: Optional[np.ndarray] = None,
                               confidence_threshold: float = 0.5) -> dict:
    """
    Calculate statistics for a sequence of pitch values.

    Args:
        pitch_values_hz: Array of pitch values in Hz
        confidence_values: Optional array of confidence scores (0-1)
        confidence_threshold: Minimum confidence to include in stats

    Returns:
        Dictionary with pitch statistics:
        - median_hz: Median pitch
        - mean_hz: Mean pitch
        - std_hz: Standard deviation
        - median_midi: Median as MIDI note
        - mean_midi: Mean as MIDI note
        - note_name: Nearest note name
        - stability: Pitch stability score (0-1, higher = more stable)
        - mean_confidence: Average confidence
    """
    # Filter by confidence if provided
    if confidence_values is not None:
        valid_mask = (pitch_values_hz > 0) & (confidence_values >= confidence_threshold)
        valid_pitches = pitch_values_hz[valid_mask]
        valid_confidences = confidence_values[valid_mask]
    else:
        valid_pitches = pitch_values_hz[pitch_values_hz > 0]
        valid_confidences = None

    if len(valid_pitches) == 0:
        return {
            'median_hz': 0.0,
            'mean_hz': 0.0,
            'std_hz': 0.0,
            'median_midi': 0,
            'mean_midi': 0.0,
            'note_name': 'Unknown',
            'stability': 0.0,
            'mean_confidence': 0.0,
            'num_valid_frames': 0
        }

    # Calculate pitch statistics
    median_hz = float(np.median(valid_pitches))
    mean_hz = float(np.mean(valid_pitches))
    std_hz = float(np.std(valid_pitches))

    # Convert to MIDI
    median_midi = round_to_nearest_midi(median_hz)
    mean_midi = float(hz_to_midi(mean_hz))

    # Calculate stability (inverse of coefficient of variation)
    # Higher = more stable pitch
    if mean_hz > 0:
        cv = std_hz / mean_hz
        stability = float(1.0 / (1.0 + cv * 10))  # Scale to 0-1
    else:
        stability = 0.0

    # Mean confidence
    mean_confidence = float(np.mean(valid_confidences)) if valid_confidences is not None else 1.0

    return {
        'median_hz': median_hz,
        'mean_hz': mean_hz,
        'std_hz': std_hz,
        'median_midi': median_midi,
        'mean_midi': mean_midi,
        'note_name': midi_to_note_name(median_midi),
        'stability': stability,
        'mean_confidence': mean_confidence,
        'num_valid_frames': len(valid_pitches)
    }


def find_semitone_matches(target_midi: int, available_midi_notes: List[int],
                         max_distance: int = 2) -> List[Tuple[int, int]]:
    """
    Find MIDI notes within N semitones of target.

    Args:
        target_midi: Target MIDI note number
        available_midi_notes: List of available MIDI notes
        max_distance: Maximum semitone distance (default: 2)

    Returns:
        List of (midi_note, distance) tuples, sorted by distance

    Examples:
        >>> find_semitone_matches(60, [59, 60, 61, 63], max_distance=2)
        [(60, 0), (59, 1), (61, 1)]
    """
    matches = []
    for midi in available_midi_notes:
        distance = abs(midi - target_midi)
        if distance <= max_distance:
            matches.append((midi, distance))

    # Sort by distance (closest first)
    matches.sort(key=lambda x: x[1])
    return matches


def is_pitch_in_range(frequency_hz: float,
                      min_note: str = "C2",
                      max_note: str = "C6") -> bool:
    """
    Check if a frequency is within a specified note range.

    Useful for filtering out unrealistic pitches (too low/high for singing).

    Args:
        frequency_hz: Frequency to check
        min_note: Minimum note name (e.g., 'C2')
        max_note: Maximum note name (e.g., 'C6')

    Returns:
        True if frequency is in range
    """
    if frequency_hz <= 0:
        return False

    midi = hz_to_midi(frequency_hz)
    if np.isnan(midi):
        return False

    min_midi = note_name_to_midi(min_note)
    max_midi = note_name_to_midi(max_note)

    if min_midi is None or max_midi is None:
        return True  # Can't validate range

    return min_midi <= midi <= max_midi


def cents_to_ratio(cents: float) -> float:
    """
    Convert cents (pitch distance) to frequency ratio.

    Args:
        cents: Pitch distance in cents

    Returns:
        Frequency ratio

    Examples:
        >>> cents_to_ratio(100)  # 1 semitone
        1.0594...
    """
    return 2.0 ** (cents / 1200.0)


if __name__ == "__main__":
    # Test the utilities
    print("=== Pitch Utilities Test ===\n")

    # Test Hz <-> MIDI conversions
    print("Hz to MIDI conversions:")
    test_freqs = [440.0, 261.63, 329.63, 392.0]
    test_names = ['A4', 'C4', 'E4', 'G4']
    for freq, name in zip(test_freqs, test_names):
        midi = hz_to_midi(freq)
        note = midi_to_note_name(round_to_nearest_midi(freq))
        print(f"  {freq:.2f} Hz -> MIDI {midi:.2f} -> {note} (expected: {name})")

    print("\nMIDI to Hz conversions:")
    test_midi = [60, 69, 72, 48]
    for midi in test_midi:
        freq = midi_to_hz(midi)
        note = midi_to_note_name(midi)
        print(f"  MIDI {midi} -> {freq:.2f} Hz ({note})")

    print("\nPitch distance calculations:")
    print(f"  440 Hz to 466.16 Hz: {pitch_distance_cents(440, 466.16):.1f} cents (expected: ~100)")
    print(f"  440 Hz to 880 Hz: {pitch_distance_cents(440, 880):.1f} cents (expected: 1200)")

    print("\nPitch statistics (simulated vibrato on A4):")
    vibrato = 440 + 5 * np.sin(np.linspace(0, 4*np.pi, 100))  # ±5 Hz vibrato
    stats = calculate_pitch_statistics(vibrato)
    print(f"  Median: {stats['median_hz']:.2f} Hz ({stats['note_name']})")
    print(f"  Stability: {stats['stability']:.3f}")

    print("\n=== Tests Complete ===")
