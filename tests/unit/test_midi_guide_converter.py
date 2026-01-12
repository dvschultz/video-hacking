"""
Unit tests for midi_guide_converter.py - MIDI to guide sequence conversion.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestMIDIGuideConverter:
    """Test MIDIGuideConverter class."""

    @pytest.fixture
    def converter(self, temp_dir):
        """Create MIDIGuideConverter with a simple setup."""
        with patch('builtins.print'):
            from midi_guide_converter import MIDIGuideConverter
            converter = MIDIGuideConverter.__new__(MIDIGuideConverter)
            converter.midi_path = temp_dir / "test.mid"
            converter.channel = 0
            converter.midi_file = None
            converter.ticks_per_beat = 480
            converter.notes = []
            converter.pitch_segments = []

        return converter

    def test_initialization(self, converter):
        """Test converter has correct initial state."""
        assert converter.channel == 0
        assert converter.ticks_per_beat == 480
        assert converter.notes == []

    def test_merge_small_rests(self, converter):
        """Test merging small rests between notes."""
        converter.notes = [
            {'pitch_midi': 60, 'start_time': 0.0, 'end_time': 0.5, 'duration': 0.5},
            {'pitch_midi': 64, 'start_time': 0.52, 'end_time': 1.0, 'duration': 0.48},
        ]

        with patch('builtins.print'):
            merged = converter.merge_small_rests(min_rest_duration=0.1)

        assert merged[0]['end_time'] == pytest.approx(0.52)

    def test_merge_small_rests_preserves_large_gaps(self, converter):
        """Test that large gaps are preserved."""
        converter.notes = [
            {'pitch_midi': 60, 'start_time': 0.0, 'end_time': 0.5, 'duration': 0.5},
            {'pitch_midi': 64, 'start_time': 1.0, 'end_time': 1.5, 'duration': 0.5},
        ]

        with patch('builtins.print'):
            merged = converter.merge_small_rests(min_rest_duration=0.1)

        assert merged[0]['end_time'] == 0.5

    def test_convert_to_segments(self, converter):
        """Test converting notes to segment format."""
        converter.notes = [
            {'pitch_midi': 60, 'start_time': 0.0, 'end_time': 0.5, 'duration': 0.5},
            {'pitch_midi': 64, 'start_time': 1.0, 'end_time': 1.5, 'duration': 0.5},
        ]

        with patch('builtins.print'):
            segments = converter.convert_to_segments(min_rest_duration=0.1)

        assert len(segments) >= 2

        for seg in segments:
            assert 'index' in seg
            assert 'start_time' in seg
            assert 'end_time' in seg
            assert 'duration' in seg
            assert 'pitch_midi' in seg
            assert 'pitch_note' in seg
            assert 'is_rest' in seg

    def test_convert_to_segments_includes_rests(self, converter):
        """Test that rest segments are included for gaps."""
        converter.notes = [
            {'pitch_midi': 60, 'start_time': 0.0, 'end_time': 0.5, 'duration': 0.5},
            {'pitch_midi': 64, 'start_time': 1.0, 'end_time': 1.5, 'duration': 0.5},
        ]

        with patch('builtins.print'):
            segments = converter.convert_to_segments(min_rest_duration=0.1)

        rest_segments = [s for s in segments if s.get('is_rest', False)]
        assert len(rest_segments) >= 1

        for rest in rest_segments:
            assert rest['pitch_midi'] == -1
            assert rest['pitch_note'] == 'REST'

    def test_convert_to_segments_handles_overlaps(self, converter):
        """Test handling of overlapping notes (monophonic mode)."""
        converter.notes = [
            {'pitch_midi': 60, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            {'pitch_midi': 64, 'start_time': 0.5, 'end_time': 1.5, 'duration': 1.0},
        ]

        with patch('builtins.print'):
            segments = converter.convert_to_segments(min_rest_duration=0.1)

        for i in range(len(segments) - 1):
            assert segments[i]['end_time'] <= segments[i + 1]['start_time'] + 0.01

    def test_save_results(self, converter, temp_dir):
        """Test saving results to JSON."""
        converter.pitch_segments = [
            {
                'index': 0,
                'start_time': 0.0,
                'end_time': 0.5,
                'duration': 0.5,
                'pitch_hz': 261.63,
                'pitch_midi': 60,
                'pitch_note': 'C4',
                'pitch_confidence': 1.0,
                'is_rest': False
            }
        ]

        output_path = temp_dir / "guide_sequence.json"

        with patch('builtins.print'):
            converter.save_results(str(output_path))

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert 'pitch_segments' in data
        assert data['pitch_detection_method'] == 'MIDI'
        assert data['midi_channel'] == 0

    @patch('soundfile.write')
    def test_generate_audio_preview(self, mock_sf_write, converter, temp_dir):
        """Test audio preview generation."""
        converter.pitch_segments = [
            {
                'start_time': 0.0,
                'end_time': 0.5,
                'duration': 0.5,
                'pitch_hz': 261.63,
                'pitch_midi': 60,
                'is_rest': False
            },
            {
                'start_time': 0.5,
                'end_time': 1.0,
                'duration': 0.5,
                'pitch_hz': 329.63,
                'pitch_midi': 64,
                'is_rest': False
            }
        ]

        output_path = temp_dir / "preview.wav"

        with patch('builtins.print'):
            result = converter.generate_audio_preview(str(output_path))

        assert mock_sf_write.called


class TestMIDIGuideConverterTicksToSeconds:
    """Test tempo and timing calculations."""

    @pytest.fixture
    def converter(self, temp_dir):
        """Create converter for tempo tests."""
        with patch('builtins.print'):
            from midi_guide_converter import MIDIGuideConverter
            converter = MIDIGuideConverter.__new__(MIDIGuideConverter)
            converter.ticks_per_beat = 480
            converter.midi_path = Path("test.mid")
            converter.channel = 0

        return converter

    def test_ticks_to_seconds_default_tempo(self, converter):
        """Test conversion with default tempo (120 BPM)."""
        tempo_changes = [(0, 500000)]

        result = converter._ticks_to_seconds(480, tempo_changes)
        assert result == pytest.approx(0.5)

    def test_ticks_to_seconds_tempo_change(self, converter):
        """Test conversion with tempo change."""
        tempo_changes = [
            (0, 500000),
            (480, 250000),
        ]

        result = converter._ticks_to_seconds(960, tempo_changes)
        assert result == pytest.approx(0.75)

    def test_ticks_to_seconds_zero(self, converter):
        """Test conversion of zero ticks."""
        tempo_changes = [(0, 500000)]

        result = converter._ticks_to_seconds(0, tempo_changes)
        assert result == 0.0
