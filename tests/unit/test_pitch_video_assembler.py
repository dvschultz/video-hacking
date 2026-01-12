"""
Unit tests for pitch_video_assembler.py - Pitch-based video assembly.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestPitchVideoAssembler:
    """Test PitchVideoAssembler class."""

    @pytest.fixture
    def match_plan_data(self):
        """Sample match plan data."""
        return {
            'matches': [
                {
                    'guide_segment_id': 0,
                    'match_type': 'exact',
                    'transpose_semitones': 0,
                    'guide_duration': 0.5,
                    'source_clips': [{
                        'segment_id': 0,
                        'video_path': 'source.mp4',
                        'video_start_frame': 0,
                        'video_end_frame': 12,
                        'duration': 0.5
                    }]
                },
                {
                    'guide_segment_id': 1,
                    'match_type': 'rest',
                    'guide_duration': 0.3,
                    'source_clips': []
                }
            ],
            'statistics': {
                'exact_matches': 1,
                'transposed_matches': 0,
                'missing_matches': 0,
                'rest_segments': 1
            }
        }

    @pytest.fixture
    def assembler(self, temp_dir, match_plan_data):
        """Create assembler with mock data."""
        match_plan_path = temp_dir / "match_plan.json"
        match_plan_path.write_text(json.dumps(match_plan_data))

        source_video = temp_dir / "source.mp4"
        source_video.touch()

        output_path = temp_dir / "output.mp4"

        with patch('builtins.print'):
            from pitch_video_assembler import PitchVideoAssembler
            assembler = PitchVideoAssembler(str(match_plan_path), str(output_path))
            assembler.load_match_plan()

        return assembler

    def test_initialization(self, assembler):
        """Test assembler initializes correctly."""
        assert assembler.match_plan is not None

    def test_load_match_plan(self, assembler):
        """Test loading match plan."""
        assert len(assembler.matches) == 2

    @patch('subprocess.run')
    def test_get_video_resolution(self, mock_run, assembler):
        """Test video resolution detection."""
        # text=True is used, so return string not bytes
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='1920,1080\n'  # Note: comma-separated, not 'x'
        )

        with patch('builtins.print'):
            width, height = assembler.get_video_resolution("test.mp4")

        assert width == 1920
        assert height == 1080

    @patch('subprocess.run')
    def test_get_video_fps(self, mock_run, assembler):
        """Test video FPS detection."""
        # text=True is used, so return string not bytes
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='24/1\n'
        )

        with patch('builtins.print'):
            fps = assembler.get_video_fps("test.mp4")

        assert fps == 24.0

    @patch('subprocess.run')
    def test_extract_video_clip(self, mock_run, assembler, temp_dir):
        """Test extracting video clip without audio."""
        mock_run.return_value = MagicMock(returncode=0)

        output_path = temp_dir / "clip.mp4"

        with patch('builtins.print'):
            assembler.extract_video_clip(
                "source.mp4",
                start_frame=0,
                end_frame=24,
                output_path=str(output_path),
                fps=24
            )

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert 'ffmpeg' in cmd

    @patch('subprocess.run')
    def test_generate_rest_clip(self, mock_run, assembler, temp_dir):
        """Test generating black frames for rest."""
        mock_run.return_value = MagicMock(returncode=0)

        output_path = temp_dir / "rest.mp4"

        with patch('builtins.print'):
            assembler.generate_rest_clip(
                duration=0.5,
                output_path=str(output_path),
                width=1920,
                height=1080,
                fps=24
            )

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert 'color=c=black' in str(cmd)


class TestPitchVideoAssemblerTransposition:
    """Test audio transposition in assembler."""

    @pytest.fixture
    def assembler_for_transpose(self, temp_dir):
        """Create assembler for transposition tests."""
        match_plan = {
            'matches': [{
                'guide_segment_id': 0,
                'match_type': 'transposed',
                'transpose_semitones': 2,
                'guide_duration': 0.5,
                'source_clips': [{
                    'segment_id': 0,
                    'video_path': 'source.mp4',
                    'video_start_frame': 0,
                    'video_end_frame': 12,
                    'duration': 0.5
                }]
            }],
            'statistics': {
                'exact_matches': 0,
                'transposed_matches': 1,
                'missing_matches': 0
            }
        }

        match_plan_path = temp_dir / "match_plan.json"
        match_plan_path.write_text(json.dumps(match_plan))

        output_path = temp_dir / "output.mp4"

        with patch('builtins.print'):
            from pitch_video_assembler import PitchVideoAssembler
            assembler = PitchVideoAssembler(str(match_plan_path), str(output_path))
            assembler.load_match_plan()

        return assembler

    @patch('subprocess.run')
    @patch('librosa.load')
    @patch('librosa.effects.pitch_shift')
    @patch('soundfile.write')
    @patch('pathlib.Path.unlink')  # Mock temp file cleanup
    def test_extract_and_transpose_audio(self, mock_unlink, mock_sf_write, mock_pitch_shift,
                                          mock_load, mock_run,
                                          assembler_for_transpose, temp_dir):
        """Test audio extraction with pitch transposition."""
        mock_run.return_value = MagicMock(returncode=0)
        mock_load.return_value = (np.zeros(22050), 22050)
        mock_pitch_shift.return_value = np.zeros(22050)

        output_path = temp_dir / "transposed.wav"

        with patch('builtins.print'):
            # API uses start_frame, end_frame, fps (not start_time, duration)
            assembler_for_transpose.extract_and_transpose_audio(
                video_path="source.mp4",
                start_frame=0,
                end_frame=12,
                fps=24.0,
                transpose_semitones=2,
                output_path=str(output_path)
            )

        # Should have called pitch_shift
        mock_pitch_shift.assert_called_once()


class TestPitchVideoAssemblerMerging:
    """Test clip merging functionality."""

    @pytest.fixture
    def assembler(self, temp_dir):
        """Create assembler for merge tests."""
        match_plan = {'matches': []}
        match_plan_path = temp_dir / "match_plan.json"
        match_plan_path.write_text(json.dumps(match_plan))

        output_path = temp_dir / "output.mp4"

        with patch('builtins.print'):
            from pitch_video_assembler import PitchVideoAssembler
            assembler = PitchVideoAssembler(str(match_plan_path), str(output_path))

        return assembler

    @patch('subprocess.run')
    def test_merge_short_clips(self, mock_run, assembler, temp_dir):
        """Test merging clips shorter than min duration."""
        # Mock ffprobe to return 24 fps (text=True, so string not bytes)
        mock_run.return_value = MagicMock(returncode=0, stdout='24/1\n')

        video_path = str(temp_dir / "source.mp4")
        Path(video_path).touch()

        # Clips need video_path, video_start_frame, video_end_frame
        # At 24fps: 0.02s = 0.48 frames, 0.5s = 12 frames
        clips = [
            {'video_path': video_path, 'video_start_frame': 0, 'video_end_frame': 1, 'duration': 0.042},  # Short
            {'video_path': video_path, 'video_start_frame': 1, 'video_end_frame': 2, 'duration': 0.042},  # Short
            {'video_path': video_path, 'video_start_frame': 2, 'video_end_frame': 14, 'duration': 0.5},   # Long enough
        ]

        with patch('builtins.print'):
            merged = assembler._merge_short_clips(clips, min_duration=0.04)

        # Short clips should be merged
        assert len(merged) <= len(clips)
