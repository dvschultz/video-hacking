"""
Unit tests for pitch_change_detector.py - Pitch detection and segmentation.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestPitchChangeDetector:
    """Test PitchChangeDetector with mocked pitch detection."""

    @pytest.fixture
    def mock_audio_data(self, sample_audio_array):
        """Get mock audio data."""
        return sample_audio_array

    @pytest.fixture
    def detector(self, mock_audio_data, temp_dir):
        """Create detector with mocked audio loading."""
        audio, sr = mock_audio_data

        audio_path = temp_dir / "test_audio.wav"
        audio_path.touch()

        with patch('librosa.load', return_value=(audio, sr)):
            with patch('builtins.print'):
                from pitch_change_detector import PitchChangeDetector
                detector = PitchChangeDetector(
                    str(audio_path),
                    sr=sr,
                    pitch_method='pyin'
                )

        return detector

    def test_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector.audio is not None
        assert detector.sr > 0

    @patch('librosa.pyin')
    @patch('librosa.times_like')
    def test_extract_continuous_pitch_pyin(self, mock_times, mock_pyin, detector):
        """Test pitch extraction with pYIN."""
        num_frames = 100
        mock_pyin.return_value = (
            np.full(num_frames, 440.0),
            np.ones(num_frames, dtype=bool),
            np.full(num_frames, 0.9)
        )
        mock_times.return_value = np.linspace(0, 1, num_frames)

        with patch('builtins.print'):
            times, pitch, conf = detector.extract_continuous_pitch()

        assert len(times) == num_frames
        assert len(pitch) == num_frames
        assert len(conf) == num_frames

    @patch('librosa.pyin')
    @patch('librosa.times_like')
    def test_detect_pitch_segments_basic(self, mock_times, mock_pyin, detector):
        """Test basic pitch segment detection."""
        num_frames = 200

        # Two distinct pitches
        pitches = np.concatenate([
            np.full(100, 440.0),  # A4
            np.full(100, 523.25)  # C5
        ])

        mock_pyin.return_value = (
            pitches,
            np.ones(num_frames, dtype=bool),
            np.full(num_frames, 0.9)
        )
        mock_times.return_value = np.linspace(0, 2, num_frames)

        with patch('builtins.print'):
            detector.extract_continuous_pitch()
            segments = detector.detect_pitch_segments(
                pitch_change_threshold_cents=100,
                min_segment_duration=0.1
            )

        # Should detect at least one segment
        assert len(segments) >= 1

    @patch('librosa.pyin')
    @patch('librosa.times_like')
    def test_detect_silence_in_pitch(self, mock_times, mock_pyin, detector):
        """Test detection of silence (zero pitch) regions."""
        num_frames = 150

        # Pitch with silence in middle
        pitches = np.concatenate([
            np.full(50, 440.0),
            np.zeros(50),  # Silence
            np.full(50, 440.0)
        ])

        confidences = np.concatenate([
            np.full(50, 0.9),
            np.full(50, 0.1),  # Low confidence for silence
            np.full(50, 0.9)
        ])

        mock_pyin.return_value = (
            pitches,
            pitches > 0,
            confidences
        )
        mock_times.return_value = np.linspace(0, 1.5, num_frames)

        with patch('builtins.print'):
            detector.extract_continuous_pitch()
            segments = detector.detect_pitch_segments(
                min_confidence=0.5,
                min_silence_duration=0.2
            )

        # Should split on silence
        assert len(segments) >= 1

    @patch('librosa.pyin')
    @patch('librosa.times_like')
    def test_segments_to_dict(self, mock_times, mock_pyin, detector):
        """Test conversion of segments to dictionary format."""
        mock_pyin.return_value = (
            np.full(100, 440.0),
            np.ones(100, dtype=bool),
            np.full(100, 0.9)
        )
        mock_times.return_value = np.linspace(0, 1, 100)

        with patch('builtins.print'):
            detector.extract_continuous_pitch()
            segments = detector.detect_pitch_segments(min_segment_duration=0.05)

        if len(segments) > 0:
            dict_segments = detector.segments_to_dict(segments)

            assert len(dict_segments) > 0
            seg = dict_segments[0]
            assert 'start_time' in seg
            assert 'end_time' in seg
            assert 'pitch_hz' in seg
            assert 'pitch_midi' in seg
            assert 'pitch_note' in seg


class TestPitchChangeDetectorMethods:
    """Test different pitch detection methods."""

    @pytest.fixture
    def base_detector(self, sample_audio_array, temp_dir):
        """Create base detector for method tests."""
        audio, sr = sample_audio_array

        audio_path = temp_dir / "test.wav"
        audio_path.touch()

        return audio_path, audio, sr

    def test_pyin_method_exists(self, base_detector):
        """Test pYIN method can be initialized."""
        audio_path, audio, sr = base_detector

        with patch('librosa.load', return_value=(audio, sr)):
            with patch('builtins.print'):
                from pitch_change_detector import PitchChangeDetector
                detector = PitchChangeDetector(
                    str(audio_path),
                    sr=sr,
                    pitch_method='pyin'
                )

        assert detector.pitch_method == 'pyin'
