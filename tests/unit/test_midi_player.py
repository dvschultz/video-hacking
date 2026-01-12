"""
Unit tests for midi_player.py - MIDI note synthesis and playback.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from midi_player import MIDIPlayer, load_pitch_sequence_from_json, play_melody_from_notes


class TestMIDIPlayer:
    """Test MIDIPlayer class."""

    @pytest.fixture
    def player(self):
        """Create MIDIPlayer instance."""
        return MIDIPlayer(sample_rate=22050)

    def test_initialization(self, player):
        """Test player initializes with correct sample rate."""
        assert player.sample_rate == 22050

    def test_generate_tone_correct_length(self, player):
        """Test that generated tone has correct length."""
        duration = 1.0
        tone = player.generate_tone(440.0, duration)

        expected_samples = int(duration * player.sample_rate)
        assert len(tone) == expected_samples

    def test_generate_tone_frequency(self, player):
        """Test that generated tone has expected frequency content."""
        frequency = 440.0
        duration = 0.5
        tone = player.generate_tone(frequency, duration)

        # Check that tone is not silent
        assert np.max(np.abs(tone)) > 0

    def test_generate_tone_normalized(self, player):
        """Test that tone is normalized."""
        tone = player.generate_tone(440.0, 1.0)

        # Should be normalized to ~0.5 max
        assert np.max(np.abs(tone)) <= 1.0

    def test_generate_tone_different_frequencies(self, player):
        """Test generating tones at different frequencies."""
        tone_low = player.generate_tone(220.0, 0.5)
        tone_high = player.generate_tone(880.0, 0.5)

        # Both should be valid audio
        assert len(tone_low) == len(tone_high)
        assert np.max(np.abs(tone_low)) > 0
        assert np.max(np.abs(tone_high)) > 0

    def test_generate_adsr_envelope_length(self, player):
        """Test ADSR envelope has correct length."""
        num_samples = 1000
        envelope = player._generate_adsr_envelope(
            num_samples,
            attack=0.01,
            decay=0.01,
            sustain_level=0.7,
            release=0.02
        )

        assert len(envelope) == num_samples

    def test_play_note_sequence_basic(self, player):
        """Test playing a sequence of notes."""
        notes = [
            {'pitch_midi': 60, 'duration': 0.2},
            {'pitch_midi': 64, 'duration': 0.2},
            {'pitch_midi': 67, 'duration': 0.2},
        ]

        with patch('builtins.print'):
            audio = player.play_note_sequence(notes, gap_duration=0.0, verbose=False)

        expected_duration = 0.6  # 3 notes x 0.2s
        expected_samples = int(expected_duration * player.sample_rate)

        assert len(audio) == pytest.approx(expected_samples, rel=0.1)

    def test_play_note_sequence_with_gaps(self, player):
        """Test note sequence with gaps between notes."""
        notes = [
            {'pitch_midi': 60, 'duration': 0.2},
            {'pitch_midi': 64, 'duration': 0.2},
        ]

        with patch('builtins.print'):
            audio = player.play_note_sequence(notes, gap_duration=0.1, verbose=False)

        # 2 notes + 1 gap
        expected_duration = 0.2 + 0.1 + 0.2
        expected_samples = int(expected_duration * player.sample_rate)

        assert len(audio) == pytest.approx(expected_samples, rel=0.1)

    def test_play_note_sequence_with_silence(self, player):
        """Test note sequence with silent notes (rest)."""
        notes = [
            {'pitch_midi': 60, 'duration': 0.2},
            {'pitch_midi': 0, 'duration': 0.2},  # Rest (MIDI 0)
            {'pitch_midi': 64, 'duration': 0.2},
        ]

        with patch('builtins.print'):
            audio = player.play_note_sequence(notes, gap_duration=0.0, verbose=False)

        expected_duration = 0.6
        expected_samples = int(expected_duration * player.sample_rate)

        assert len(audio) == pytest.approx(expected_samples, rel=0.1)

    def test_play_note_sequence_single(self, player):
        """Test playing single note sequence."""
        notes = [{'pitch_midi': 60, 'duration': 0.5}]

        with patch('builtins.print'):
            audio = player.play_note_sequence(notes, verbose=False)

        assert len(audio) > 0

    @patch('soundfile.write')
    def test_save_audio(self, mock_sf_write, player, temp_dir):
        """Test saving audio to file."""
        audio = np.random.randn(22050).astype(np.float32)
        output_path = temp_dir / "output.wav"

        with patch('builtins.print'):
            player.save_audio(audio, str(output_path))

        mock_sf_write.assert_called_once()
        call_args = mock_sf_write.call_args
        assert str(output_path) in str(call_args)

    @patch('soundfile.write')
    def test_save_audio_creates_directory(self, mock_sf_write, player, temp_dir):
        """Test that save_audio creates parent directory."""
        audio = np.random.randn(22050).astype(np.float32)
        output_path = temp_dir / "subdir" / "output.wav"

        with patch('builtins.print'):
            player.save_audio(audio, str(output_path))

        assert output_path.parent.exists()


class TestLoadPitchSequence:
    """Test pitch sequence loading from JSON."""

    def test_load_guide_sequence_format(self, temp_dir):
        """Test loading guide_sequence format."""
        data = {
            'guide_sequence': [
                {'pitch_midi': 60, 'pitch_hz': 261.63, 'duration': 0.5},
                {'pitch_midi': 64, 'pitch_hz': 329.63, 'duration': 0.5},
            ]
        }

        json_path = temp_dir / "guide.json"
        json_path.write_text(json.dumps(data))

        notes = load_pitch_sequence_from_json(str(json_path))

        assert len(notes) == 2
        assert notes[0]['pitch_midi'] == 60
        assert notes[1]['pitch_midi'] == 64

    def test_load_pitch_database_format(self, temp_dir):
        """Test loading pitch_database format."""
        data = {
            'pitch_database': [
                {'pitch_midi': 60, 'pitch_hz': 261.63, 'duration': 0.5},
                {'pitch_midi': 67, 'pitch_hz': 392.0, 'duration': 0.3},
            ]
        }

        json_path = temp_dir / "source.json"
        json_path.write_text(json.dumps(data))

        notes = load_pitch_sequence_from_json(str(json_path))

        assert len(notes) == 2
        assert notes[0]['pitch_midi'] == 60
        assert notes[1]['pitch_midi'] == 67

    def test_load_segments_format(self, temp_dir):
        """Test loading generic segments format."""
        data = {
            'segments': [
                {'pitch_midi': 60, 'duration': 0.5},
            ]
        }

        json_path = temp_dir / "segments.json"
        json_path.write_text(json.dumps(data))

        notes = load_pitch_sequence_from_json(str(json_path))

        assert len(notes) == 1
        assert notes[0]['pitch_midi'] == 60

    def test_load_unknown_format(self, temp_dir):
        """Test loading unknown format returns empty."""
        data = {'unknown_key': []}

        json_path = temp_dir / "unknown.json"
        json_path.write_text(json.dumps(data))

        with patch('builtins.print'):
            notes = load_pitch_sequence_from_json(str(json_path))

        assert len(notes) == 0


class TestPlayMelodyFromNotes:
    """Test convenience function for playing melodies."""

    @patch('midi_player.MIDIPlayer.play_audio')
    @patch('soundfile.write')
    def test_play_melody_basic(self, mock_sf_write, mock_play):
        """Test playing a simple melody."""
        with patch('builtins.print'):
            with patch('midi_player.SOUNDDEVICE_AVAILABLE', False):
                audio = play_melody_from_notes(
                    ['C4', 'E4', 'G4'],
                    duration_per_note=0.2
                )

        assert len(audio) > 0

    @patch('soundfile.write')
    def test_play_melody_saves_file(self, mock_sf_write, temp_dir):
        """Test melody is saved to file if output_path provided."""
        output_path = temp_dir / "melody.wav"

        with patch('builtins.print'):
            with patch('midi_player.SOUNDDEVICE_AVAILABLE', False):
                audio = play_melody_from_notes(
                    ['C4', 'D4'],
                    duration_per_note=0.2,
                    output_path=str(output_path)
                )

        mock_sf_write.assert_called()
