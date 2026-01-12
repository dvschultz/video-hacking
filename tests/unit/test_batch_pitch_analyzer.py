"""
Unit tests for batch_pitch_analyzer.py - Parallel batch processing.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestBatchPitchAnalyzer:
    """Test batch processing functions."""

    def test_find_video_files(self, temp_dir):
        """Test finding video files in directory."""
        # Create test video files
        (temp_dir / "video1.mp4").touch()
        (temp_dir / "video2.mov").touch()
        (temp_dir / "audio.wav").touch()  # Not a video
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "video3.mp4").touch()

        from batch_pitch_analyzer import find_video_files

        videos = find_video_files(str(temp_dir), extensions=['mp4', 'mov'])

        assert len(videos) >= 2  # At least video1.mp4 and video2.mov

    def test_find_video_files_specific_extensions(self, temp_dir):
        """Test filtering by specific extensions."""
        (temp_dir / "video1.mp4").touch()
        (temp_dir / "video2.avi").touch()

        from batch_pitch_analyzer import find_video_files

        videos = find_video_files(str(temp_dir), extensions=['mp4'])

        # Should only find mp4 files
        assert all(str(v).endswith('.mp4') for v in videos)

    def test_merge_databases_structure(self, temp_dir):
        """Test that merge_databases writes correctly structured output."""
        from batch_pitch_analyzer import merge_databases

        results = [
            {
                'video_path': 'video1.mp4',
                'pitch_database': [
                    {'segment_id': 0, 'pitch_midi': 60, 'duration': 0.5}
                ],
                'pitch_index': {'60': [0]},
                'silence_segments': [],
                'num_segments': 1,
                'num_silences': 0
            },
            {
                'video_path': 'video2.mp4',
                'pitch_database': [
                    {'segment_id': 0, 'pitch_midi': 64, 'duration': 0.3}
                ],
                'pitch_index': {'64': [0]},
                'silence_segments': [],
                'num_segments': 1,
                'num_silences': 0
            }
        ]

        output_path = temp_dir / "merged.json"

        with patch('builtins.print'):
            merge_databases(results, str(output_path))

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert 'pitch_database' in data
        assert 'pitch_index' in data
        assert len(data['pitch_database']) == 2
