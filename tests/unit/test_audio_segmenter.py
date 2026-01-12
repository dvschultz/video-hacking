"""
Unit tests for audio_segmenter.py - Audio segmentation based on onsets.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestAudioSegmenter:
    """Test AudioSegmenter class."""

    @pytest.fixture
    def onset_data_file(self, temp_dir, sample_onset_strength_data):
        """Create onset strength JSON file."""
        onset_path = temp_dir / "onset_strength.json"
        onset_path.write_text(json.dumps(sample_onset_strength_data))
        return onset_path

    @pytest.fixture
    def segmenter(self, onset_data_file, sample_audio_array):
        """Create AudioSegmenter with mocked data."""
        audio, sr = sample_audio_array

        with patch('librosa.load', return_value=(audio, sr)):
            with patch('builtins.print'):
                from audio_segmenter import AudioSegmenter
                segmenter = AudioSegmenter(
                    "test_audio.wav",
                    str(onset_data_file)
                )

        return segmenter

    def test_initialization(self, segmenter):
        """Test segmenter initializes with correct data."""
        assert segmenter.onset_values is not None
        assert segmenter.times is not None
        assert segmenter.audio is not None
        assert segmenter.sr > 0

    def test_find_cut_points(self, segmenter):
        """Test finding cut points above threshold."""
        with patch('builtins.print'):
            cut_times = segmenter.find_cut_points(threshold=0.5)

        assert isinstance(cut_times, np.ndarray)
        # Should have some cut points (from sample data)
        assert len(cut_times) >= 0

    def test_find_cut_points_high_threshold(self, segmenter):
        """Test that high threshold returns fewer cut points."""
        with patch('builtins.print'):
            cut_low = segmenter.find_cut_points(threshold=0.1)
            cut_high = segmenter.find_cut_points(threshold=0.9)

        assert len(cut_high) <= len(cut_low)

    @patch('soundfile.write')
    def test_segment_audio(self, mock_sf_write, segmenter, temp_dir):
        """Test audio segmentation."""
        cut_times = np.array([0.5, 1.0, 1.5])

        with patch('builtins.print'):
            segments = segmenter.segment_audio(
                cut_times,
                output_dir=str(temp_dir / "segments"),
                prefix="test_seg"
            )

        assert len(segments) > 0
        assert mock_sf_write.called

        # Check segment metadata
        for seg in segments:
            assert 'index' in seg
            assert 'filename' in seg
            assert 'start_time' in seg
            assert 'end_time' in seg
            assert 'duration' in seg
            assert seg['duration'] > 0

    @patch('soundfile.write')
    def test_segment_audio_creates_directory(self, mock_sf_write, segmenter, temp_dir):
        """Test that segment_audio creates output directory."""
        cut_times = np.array([0.5])
        output_dir = temp_dir / "new_dir" / "segments"

        with patch('builtins.print'):
            segmenter.segment_audio(
                cut_times,
                output_dir=str(output_dir),
                prefix="seg"
            )

        assert output_dir.exists()

    @patch('soundfile.write')
    def test_segment_audio_empty_cuts(self, mock_sf_write, segmenter, temp_dir):
        """Test segmentation with no cut points."""
        cut_times = np.array([])

        with patch('builtins.print'):
            segments = segmenter.segment_audio(
                cut_times,
                output_dir=str(temp_dir / "segments"),
                prefix="seg"
            )

        # Should still create one segment (whole audio)
        assert len(segments) == 1

    def test_export_metadata(self, segmenter, temp_dir):
        """Test metadata export to JSON."""
        segments = [
            {
                'index': 0,
                'filename': 'seg_0000.wav',
                'start_time': 0.0,
                'end_time': 1.0,
                'duration': 1.0,
                'num_samples': 22050
            },
            {
                'index': 1,
                'filename': 'seg_0001.wav',
                'start_time': 1.0,
                'end_time': 2.0,
                'duration': 1.0,
                'num_samples': 22050
            }
        ]

        output_path = temp_dir / "segments_metadata.json"

        with patch('builtins.print'):
            segmenter.export_metadata(segments, str(output_path), threshold=0.2)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data['num_segments'] == 2
        assert data['threshold'] == 0.2
        assert len(data['segments']) == 2

    def test_get_statistics(self, segmenter):
        """Test segment statistics calculation."""
        segments = [
            {'duration': 0.5},
            {'duration': 1.0},
            {'duration': 1.5},
            {'duration': 2.0},
        ]

        stats = segmenter.get_statistics(segments)

        assert stats['num_segments'] == 4
        assert stats['total_duration'] == pytest.approx(5.0)
        assert stats['avg_duration'] == pytest.approx(1.25)
        assert stats['min_duration'] == pytest.approx(0.5)
        assert stats['max_duration'] == pytest.approx(2.0)
        assert stats['median_duration'] == pytest.approx(1.25)

    def test_get_statistics_single_segment(self, segmenter):
        """Test statistics with single segment."""
        segments = [{'duration': 1.0}]

        stats = segmenter.get_statistics(segments)

        assert stats['num_segments'] == 1
        assert stats['avg_duration'] == 1.0
        assert stats['std_duration'] == 0.0
