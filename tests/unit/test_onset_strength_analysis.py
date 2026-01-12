"""
Unit tests for onset_strength_analysis.py - Audio onset strength analysis.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestOnsetStrengthAnalyzer:
    """Test OnsetStrengthAnalyzer class."""

    @pytest.fixture
    def mock_audio_data(self, sample_audio_array):
        """Mock audio loading."""
        audio, sr = sample_audio_array
        return audio, sr

    @pytest.fixture
    def analyzer(self, mock_audio_data):
        """Create analyzer with mocked audio loading."""
        audio, sr = mock_audio_data

        with patch('librosa.load', return_value=(audio, sr)):
            with patch('librosa.get_duration', return_value=1.0):
                with patch('builtins.print'):
                    from onset_strength_analysis import OnsetStrengthAnalyzer
                    analyzer = OnsetStrengthAnalyzer("test_audio.wav", sr=sr)

        return analyzer

    def test_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer.y is not None
        assert analyzer.sr > 0
        assert analyzer.onset_strength_values is None  # Not analyzed yet

    @patch('librosa.onset.onset_strength')
    @patch('librosa.times_like')
    def test_analyze_returns_array(self, mock_times, mock_onset, analyzer):
        """Test analyze() returns onset strength array."""
        num_frames = 24
        mock_onset.return_value = np.random.rand(num_frames) * 0.5
        mock_times.return_value = np.linspace(0, 1, num_frames)

        with patch('builtins.print'):
            result = analyzer.analyze(analysis_rate=24)

        assert isinstance(result, np.ndarray)
        assert len(result) == num_frames
        assert np.all(result >= 0)

    @patch('librosa.onset.onset_strength')
    @patch('librosa.times_like')
    def test_analyze_normalizes_values(self, mock_times, mock_onset, analyzer):
        """Test that analyze normalizes values to [0, 1] range."""
        num_frames = 24
        mock_onset.return_value = np.array([0.0] * 12 + [0.5, 1.0, 0.3, 0.8] + [0.0] * 8)
        mock_times.return_value = np.linspace(0, 1, num_frames)

        with patch('builtins.print'):
            result = analyzer.analyze(analysis_rate=24, ignore_start_duration=0.0)

        assert np.max(result) <= 1.0
        assert np.min(result) >= 0.0

    @patch('librosa.onset.onset_strength')
    @patch('librosa.times_like')
    def test_analyze_ignores_start(self, mock_times, mock_onset, analyzer):
        """Test that analyze zeros out start section."""
        num_frames = 48
        mock_onset.return_value = np.ones(num_frames)  # All ones
        mock_times.return_value = np.linspace(0, 2, num_frames)

        with patch('builtins.print'):
            result = analyzer.analyze(
                analysis_rate=24,
                ignore_start_duration=0.5  # Zero out first 0.5s = 12 frames
            )

        # First 12 frames should be zeroed
        assert np.all(result[:12] == 0)

    @patch('librosa.onset.onset_strength')
    @patch('librosa.times_like')
    def test_analyze_applies_power_scaling(self, mock_times, mock_onset, analyzer):
        """Test that power scaling is applied."""
        num_frames = 24
        mock_onset.return_value = np.array([0.0] * 12 + [0.5] * 12)
        mock_times.return_value = np.linspace(0, 1, num_frames)

        with patch('builtins.print'):
            result_low_power = analyzer.analyze(analysis_rate=24, power=0.5, ignore_start_duration=0.0)

        # Reset for second analysis
        analyzer.onset_strength_values = None
        mock_onset.return_value = np.array([0.0] * 12 + [0.5] * 12)

        with patch('builtins.print'):
            result_high_power = analyzer.analyze(analysis_rate=24, power=1.0, ignore_start_duration=0.0)

        # Low power should compress values (push mid values higher)
        # 0.5^0.5 = 0.707, 0.5^1.0 = 0.5
        # After normalization, the ratios change

    def test_get_cut_frames_raises_without_analysis(self, analyzer):
        """Test get_cut_frames raises error if analyze() not called."""
        with pytest.raises(ValueError, match="Must run analyze"):
            analyzer.get_cut_frames(threshold=0.1)

    @patch('librosa.onset.onset_strength')
    @patch('librosa.times_like')
    def test_get_cut_frames_filters_by_threshold(self, mock_times, mock_onset, analyzer):
        """Test get_cut_frames returns only frames above threshold."""
        num_frames = 24
        mock_onset.return_value = np.array([0.0] * 12 + [0.1, 0.3, 0.6, 0.2, 0.8] + [0.0] * 7)
        mock_times.return_value = np.linspace(0, 1, num_frames)

        with patch('builtins.print'):
            analyzer.analyze(analysis_rate=24, ignore_start_duration=0.0)
            cut_frames = analyzer.get_cut_frames(threshold=0.5)

        # Check all returned frames have values above threshold
        for frame in cut_frames:
            assert analyzer.onset_strength_values[frame] > 0.5

    @patch('librosa.onset.onset_strength')
    @patch('librosa.times_like')
    def test_get_cut_times(self, mock_times, mock_onset, analyzer):
        """Test get_cut_times returns timestamps."""
        num_frames = 24
        mock_onset.return_value = np.array([0.0] * 12 + [0.8] + [0.0] * 11)
        mock_times.return_value = np.linspace(0, 1, num_frames)

        with patch('builtins.print'):
            analyzer.analyze(analysis_rate=24, ignore_start_duration=0.0)
            cut_times = analyzer.get_cut_times(threshold=0.5)

        assert len(cut_times) > 0
        assert isinstance(cut_times[0], (float, np.floating))

    @patch('librosa.onset.onset_strength')
    @patch('librosa.times_like')
    @patch('librosa.get_duration')
    def test_export_json(self, mock_duration, mock_times, mock_onset, analyzer, temp_dir):
        """Test exporting analysis to JSON."""
        num_frames = 24
        mock_onset.return_value = np.array([0.0] * 12 + [0.5] * 12)
        mock_times.return_value = np.linspace(0, 1, num_frames)
        mock_duration.return_value = 1.0

        with patch('builtins.print'):
            analyzer.analyze(analysis_rate=24)
            output_path = temp_dir / "onset_output.json"
            analyzer.export(str(output_path), format='json')

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert 'onset_strength_values' in data
        assert 'times' in data
        assert data['analysis_rate'] == 24
        assert len(data['onset_strength_values']) == num_frames

    @patch('librosa.onset.onset_strength')
    @patch('librosa.times_like')
    @patch('librosa.get_duration')
    def test_export_txt(self, mock_duration, mock_times, mock_onset, analyzer, temp_dir):
        """Test exporting to text format."""
        num_frames = 10
        mock_onset.return_value = np.array([0.1] * num_frames)
        mock_times.return_value = np.linspace(0, 1, num_frames)
        mock_duration.return_value = 1.0

        with patch('builtins.print'):
            analyzer.analyze(analysis_rate=24)
            output_path = temp_dir / "onset_output.txt"
            analyzer.export(str(output_path), format='txt')

        assert output_path.exists()

        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == num_frames

    @patch('librosa.onset.onset_strength')
    @patch('librosa.times_like')
    @patch('librosa.get_duration')
    def test_get_statistics(self, mock_duration, mock_times, mock_onset, analyzer):
        """Test statistics calculation."""
        num_frames = 10
        mock_onset.return_value = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        mock_times.return_value = np.linspace(0, 1, num_frames)
        mock_duration.return_value = 1.0

        with patch('builtins.print'):
            analyzer.analyze(analysis_rate=24, ignore_start_duration=0.0)
            stats = analyzer.get_statistics()

        assert 'num_frames' in stats
        assert 'mean_value' in stats
        assert 'non_zero_frames' in stats
        assert stats['num_frames'] == num_frames

    def test_get_statistics_raises_without_analysis(self, analyzer):
        """Test get_statistics raises error without analysis."""
        with pytest.raises(ValueError, match="Must run analyze"):
            analyzer.get_statistics()
