"""
Unit tests for pitch_utils.py - Pure pitch conversion and utility functions.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pitch_utils import (
    hz_to_midi, midi_to_hz, midi_to_note_name, note_name_to_midi,
    pitch_distance_cents, pitch_distance_semitones, round_to_nearest_midi,
    calculate_pitch_statistics, find_semitone_matches, is_pitch_in_range,
    cents_to_ratio
)


class TestHzMidiConversions:
    """Test Hz <-> MIDI conversions."""

    def test_hz_to_midi_a4(self):
        """A4 = 440 Hz should be MIDI 69."""
        assert hz_to_midi(440.0) == pytest.approx(69.0)

    def test_hz_to_midi_c4(self):
        """C4 = 261.63 Hz should be MIDI 60."""
        assert hz_to_midi(261.63) == pytest.approx(60.0, rel=0.01)

    def test_hz_to_midi_c5(self):
        """C5 = 523.25 Hz should be MIDI 72."""
        assert hz_to_midi(523.25) == pytest.approx(72.0, rel=0.01)

    def test_hz_to_midi_zero(self):
        """Zero Hz should return NaN."""
        assert np.isnan(hz_to_midi(0.0))

    def test_hz_to_midi_negative(self):
        """Negative Hz should return NaN."""
        assert np.isnan(hz_to_midi(-100.0))

    def test_hz_to_midi_array(self):
        """Test with numpy array input."""
        freqs = np.array([440.0, 261.63, 0.0, 880.0])
        result = hz_to_midi(freqs)
        assert result[0] == pytest.approx(69.0)
        assert result[3] == pytest.approx(81.0)
        assert np.isnan(result[2])

    def test_hz_to_midi_array_with_negatives(self):
        """Test array with negative values."""
        freqs = np.array([440.0, -100.0, 220.0])
        result = hz_to_midi(freqs)
        assert result[0] == pytest.approx(69.0)
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(57.0)

    def test_midi_to_hz_a4(self):
        """MIDI 69 should be 440 Hz."""
        assert midi_to_hz(69) == pytest.approx(440.0)

    def test_midi_to_hz_c4(self):
        """MIDI 60 should be ~261.63 Hz."""
        assert midi_to_hz(60) == pytest.approx(261.63, rel=0.01)

    def test_midi_to_hz_octave_relationship(self):
        """Octave should double the frequency."""
        assert midi_to_hz(69) == pytest.approx(midi_to_hz(57) * 2)

    def test_roundtrip_hz_midi_hz(self):
        """Converting Hz -> MIDI -> Hz should be identity."""
        original = 440.0
        midi = hz_to_midi(original)
        recovered = midi_to_hz(midi)
        assert recovered == pytest.approx(original)

    def test_roundtrip_midi_hz_midi(self):
        """Converting MIDI -> Hz -> MIDI should be identity."""
        original = 60
        hz = midi_to_hz(original)
        recovered = hz_to_midi(hz)
        assert recovered == pytest.approx(float(original))


class TestNoteNameConversions:
    """Test note name <-> MIDI conversions."""

    @pytest.mark.parametrize("midi,expected", [
        (60, "C4"),
        (69, "A4"),
        (61, "C#4"),
        (72, "C5"),
        (48, "C3"),
        (71, "B4"),
        (59, "B3"),
    ])
    def test_midi_to_note_name(self, midi, expected):
        assert midi_to_note_name(midi) == expected

    def test_midi_to_note_name_without_octave(self):
        assert midi_to_note_name(60, include_octave=False) == "C"
        assert midi_to_note_name(61, include_octave=False) == "C#"
        assert midi_to_note_name(69, include_octave=False) == "A"

    def test_midi_to_note_name_out_of_range_high(self):
        assert midi_to_note_name(128) == "Unknown"

    def test_midi_to_note_name_out_of_range_low(self):
        assert midi_to_note_name(-1) == "Unknown"

    def test_midi_to_note_name_edge_cases(self):
        assert midi_to_note_name(0) == "C-1"
        assert midi_to_note_name(127) == "G9"

    @pytest.mark.parametrize("name,expected", [
        ("C4", 60),
        ("A4", 69),
        ("C#4", 61),
        ("D4", 62),
        ("C5", 72),
    ])
    def test_note_name_to_midi(self, name, expected):
        assert note_name_to_midi(name) == expected

    def test_note_name_to_midi_flats(self):
        """Test flat notation converts correctly."""
        assert note_name_to_midi("Db4") == 61  # Same as C#4
        assert note_name_to_midi("Eb4") == 63  # Same as D#4
        assert note_name_to_midi("Gb4") == 66  # Same as F#4
        assert note_name_to_midi("Ab4") == 68  # Same as G#4
        assert note_name_to_midi("Bb4") == 70  # Same as A#4

    def test_note_name_to_midi_case_insensitive(self):
        """Test case insensitivity."""
        assert note_name_to_midi("c4") == 60
        assert note_name_to_midi("a4") == 69
        assert note_name_to_midi("C#4") == note_name_to_midi("c#4")

    def test_note_name_to_midi_invalid(self):
        assert note_name_to_midi("X4") is None
        assert note_name_to_midi("C") is None  # No octave
        assert note_name_to_midi("") is None
        assert note_name_to_midi("4") is None


class TestPitchDistance:
    """Test pitch distance calculations."""

    def test_pitch_distance_cents_same_pitch(self):
        assert pitch_distance_cents(440, 440) == pytest.approx(0.0)

    def test_pitch_distance_cents_semitone(self):
        """A4 to A#4 should be ~100 cents."""
        assert pitch_distance_cents(440, 466.16) == pytest.approx(100, rel=0.01)

    def test_pitch_distance_cents_octave(self):
        """A4 to A5 should be 1200 cents."""
        assert pitch_distance_cents(440, 880) == pytest.approx(1200)

    def test_pitch_distance_cents_negative(self):
        """Lower pitch should give negative cents."""
        assert pitch_distance_cents(440, 220) == pytest.approx(-1200)

    def test_pitch_distance_cents_zero_input(self):
        assert np.isnan(pitch_distance_cents(0, 440))
        assert np.isnan(pitch_distance_cents(440, 0))
        assert np.isnan(pitch_distance_cents(0, 0))

    def test_pitch_distance_semitones(self):
        assert pitch_distance_semitones(440, 466.16) == pytest.approx(1.0, rel=0.01)

    def test_pitch_distance_semitones_octave(self):
        assert pitch_distance_semitones(440, 880) == pytest.approx(12.0)


class TestRoundToNearestMidi:
    """Test rounding frequencies to nearest MIDI note."""

    def test_round_exact_a4(self):
        assert round_to_nearest_midi(440.0) == 69

    def test_round_exact_c4(self):
        assert round_to_nearest_midi(261.63) == 60

    def test_round_slightly_sharp(self):
        assert round_to_nearest_midi(445.0) == 69  # Still closest to A4

    def test_round_slightly_flat(self):
        assert round_to_nearest_midi(435.0) == 69  # Still closest to A4

    def test_round_halfway(self):
        """Test halfway between notes."""
        # Halfway between A4 (440) and A#4 (466.16)
        halfway = (440.0 + 466.16) / 2
        result = round_to_nearest_midi(halfway)
        assert result in [69, 70]  # Could round either way

    def test_round_invalid_zero(self):
        assert round_to_nearest_midi(0.0) == 0

    def test_round_invalid_negative(self):
        assert round_to_nearest_midi(-100.0) == 0


class TestCalculatePitchStatistics:
    """Test pitch statistics calculation."""

    def test_statistics_basic(self):
        pitches = np.array([440.0, 440.0, 440.0, 440.0])
        stats = calculate_pitch_statistics(pitches)

        assert stats['median_hz'] == pytest.approx(440.0)
        assert stats['mean_hz'] == pytest.approx(440.0)
        assert stats['median_midi'] == 69
        assert stats['note_name'] == 'A4'
        assert stats['stability'] > 0.9  # Very stable (no variation)
        assert stats['num_valid_frames'] == 4

    def test_statistics_with_variation(self):
        # Small vibrato around A4
        pitches = np.array([435.0, 440.0, 445.0, 440.0])
        stats = calculate_pitch_statistics(pitches)

        assert stats['median_hz'] == pytest.approx(440.0)
        assert stats['std_hz'] > 0
        assert stats['stability'] < 1.0

    def test_statistics_with_confidence(self):
        pitches = np.array([440.0, 440.0, 100.0, 440.0])
        confidence = np.array([0.9, 0.9, 0.1, 0.9])  # Low confidence on outlier

        stats = calculate_pitch_statistics(pitches, confidence, confidence_threshold=0.5)

        assert stats['num_valid_frames'] == 3  # Excludes low confidence
        assert stats['median_hz'] == pytest.approx(440.0)

    def test_statistics_all_zeros(self):
        pitches = np.array([0.0, 0.0, 0.0])
        stats = calculate_pitch_statistics(pitches)

        assert stats['median_hz'] == 0.0
        assert stats['num_valid_frames'] == 0
        assert stats['note_name'] == 'Unknown'

    def test_statistics_mixed_zeros(self):
        pitches = np.array([0.0, 440.0, 0.0, 440.0])
        stats = calculate_pitch_statistics(pitches)

        assert stats['num_valid_frames'] == 2
        assert stats['median_hz'] == pytest.approx(440.0)


class TestFindSemitoneMatches:
    """Test finding nearby pitch matches."""

    def test_exact_match(self):
        matches = find_semitone_matches(60, [58, 60, 62], max_distance=2)
        assert matches[0] == (60, 0)  # Exact match first

    def test_nearby_matches_sorted(self):
        matches = find_semitone_matches(60, [58, 59, 61, 62], max_distance=2)
        assert len(matches) == 4
        # Should be sorted by distance
        distances = [m[1] for m in matches]
        assert distances == sorted(distances)

    def test_nearby_matches_filtered(self):
        matches = find_semitone_matches(60, [58, 59, 61, 62, 65], max_distance=2)
        assert len(matches) == 4  # 65 is too far (5 semitones)

    def test_no_matches(self):
        matches = find_semitone_matches(60, [50, 70], max_distance=2)
        assert len(matches) == 0

    def test_empty_available(self):
        matches = find_semitone_matches(60, [], max_distance=2)
        assert len(matches) == 0

    def test_max_distance_zero(self):
        matches = find_semitone_matches(60, [59, 60, 61], max_distance=0)
        assert len(matches) == 1
        assert matches[0] == (60, 0)


class TestIsPitchInRange:
    """Test pitch range validation."""

    def test_in_range(self):
        # A4 (440 Hz) should be in C2-C6 range
        assert is_pitch_in_range(440.0, "C2", "C6") == True

    def test_below_range(self):
        # Very low frequency
        assert is_pitch_in_range(30.0, "C2", "C6") == False

    def test_above_range(self):
        # Very high frequency
        assert is_pitch_in_range(5000.0, "C2", "C6") == False

    def test_at_boundaries(self):
        c2_hz = midi_to_hz(36)  # C2
        c6_hz = midi_to_hz(84)  # C6
        assert is_pitch_in_range(c2_hz, "C2", "C6") == True
        assert is_pitch_in_range(c6_hz, "C2", "C6") == True

    def test_zero_frequency(self):
        assert is_pitch_in_range(0.0, "C2", "C6") == False

    def test_negative_frequency(self):
        assert is_pitch_in_range(-440.0, "C2", "C6") == False


class TestCentsToRatio:
    """Test cents to frequency ratio conversion."""

    def test_zero_cents(self):
        assert cents_to_ratio(0) == pytest.approx(1.0)

    def test_semitone(self):
        # 100 cents = 1 semitone
        expected = 2 ** (1/12)
        assert cents_to_ratio(100) == pytest.approx(expected, rel=0.001)

    def test_octave(self):
        # 1200 cents = 1 octave = double frequency
        assert cents_to_ratio(1200) == pytest.approx(2.0)

    def test_negative_cents(self):
        # -1200 cents = -1 octave = half frequency
        assert cents_to_ratio(-1200) == pytest.approx(0.5)

    def test_roundtrip_with_distance(self):
        """cents_to_ratio should be inverse of distance calculation."""
        f1, f2 = 440.0, 880.0
        cents = pitch_distance_cents(f1, f2)
        ratio = cents_to_ratio(cents)
        assert f1 * ratio == pytest.approx(f2)
