"""
Unit tests for pitch_source_analyzer.py - Source video pitch database building.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestPitchSourceAnalyzer:
    """Test PitchSourceAnalyzer class."""

    @pytest.fixture
    def analyzer(self, temp_dir):
        """Create analyzer with mocked dependencies."""
        video_path = temp_dir / "source_video.mp4"
        video_path.touch()

        with patch('subprocess.run') as mock_run:
            # Mock ffprobe for FPS detection
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=b'24/1\n'
            )

            with patch('builtins.print'):
                from pitch_source_analyzer import PitchSourceAnalyzer
                analyzer = PitchSourceAnalyzer(str(video_path))

        return analyzer

    def test_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer.video_path is not None

    @patch('subprocess.run')
    def test_detect_video_fps(self, mock_run, temp_dir):
        """Test FPS detection."""
        video_path = temp_dir / "test.mp4"
        video_path.touch()

        # Mock returns text output (text=True in actual call)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='30/1\n'
        )

        with patch('builtins.print'):
            from pitch_source_analyzer import PitchSourceAnalyzer
            analyzer = PitchSourceAnalyzer(str(video_path))
            fps = analyzer.detect_video_fps()

        # FPS should be detected from mock
        assert fps == 30.0

    @patch('subprocess.run')
    def test_detect_video_fps_fractional(self, mock_run, temp_dir):
        """Test FPS detection with fractional rate."""
        video_path = temp_dir / "test.mp4"
        video_path.touch()

        # Mock returns text output (text=True in actual call)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='30000/1001\n'  # 29.97 fps
        )

        with patch('builtins.print'):
            from pitch_source_analyzer import PitchSourceAnalyzer
            analyzer = PitchSourceAnalyzer(str(video_path))
            fps = analyzer.detect_video_fps()

        assert fps == pytest.approx(29.97, rel=0.01)

    def test_save_database(self, analyzer, temp_dir):
        """Test saving database to JSON."""
        analyzer.pitch_database = [
            {
                'segment_id': 0,
                'pitch_midi': 60,
                'pitch_hz': 261.63,
                'pitch_note': 'C4',
                'duration': 0.5,
                'pitch_confidence': 0.9,
                'video_path': 'test.mp4',
                'video_start_frame': 0,
                'video_end_frame': 12
            }
        ]
        analyzer.silence_segments = []
        analyzer.pitch_index = {60: [0]}

        output_path = temp_dir / "source_database.json"

        with patch('builtins.print'):
            analyzer.save_database(str(output_path))

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert 'pitch_database' in data
        assert 'pitch_index' in data
        assert len(data['pitch_database']) == 1


class TestPitchSourceAnalyzerAppend:
    """Test append mode for database building."""

    def test_save_database_append(self, temp_dir):
        """Test appending to existing database."""
        # Create existing database with correct structure
        existing_db = {
            'pitch_database': [
                {
                    'segment_id': 0,
                    'pitch_midi': 60,
                    'video_path': 'video1.mp4',
                    'duration': 0.5,
                    'pitch_hz': 261.63,
                    'pitch_note': 'C4',
                    'pitch_confidence': 0.9,
                    'video_start_frame': 0,
                    'video_end_frame': 12
                }
            ],
            'pitch_index': {'60': [0]},
            'silence_segments': [],
            'source_videos': [{'video_path': 'video1.mp4', 'fps': 24}],
            'num_videos': 1,
            'num_segments': 1,  # Required by save_database append mode
            'num_unique_pitches': 1,
            'num_silence_gaps': 0,
            'total_musical_duration': 0.5,
            'total_silence_duration': 0.0
        }

        db_path = temp_dir / "database.json"
        db_path.write_text(json.dumps(existing_db))

        # Create new analyzer
        video_path = temp_dir / "video2.mp4"
        video_path.touch()

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b'24/1\n')

            with patch('builtins.print'):
                from pitch_source_analyzer import PitchSourceAnalyzer
                analyzer = PitchSourceAnalyzer(str(video_path))

        # Set required attributes for save_database
        analyzer.fps = 24.0
        analyzer.sr = 22050
        analyzer.audio = None
        analyzer.audio_path = video_path.with_suffix('.wav')
        analyzer.pitch_method = 'pyin'

        analyzer.pitch_database = [
            {
                'segment_id': 0,
                'pitch_midi': 64,
                'video_path': str(video_path),
                'duration': 0.3,
                'pitch_hz': 329.63,
                'pitch_note': 'E4',
                'pitch_confidence': 0.85,
                'video_start_frame': 0,
                'video_end_frame': 7
            }
        ]
        analyzer.silence_segments = []
        analyzer.pitch_index = {64: [0]}

        with patch('builtins.print'):
            analyzer.save_database(str(db_path), append=True)

        with open(db_path) as f:
            data = json.load(f)

        # Should have both segments with unique IDs
        assert len(data['pitch_database']) == 2
        ids = [seg['segment_id'] for seg in data['pitch_database']]
        assert len(ids) == len(set(ids))
