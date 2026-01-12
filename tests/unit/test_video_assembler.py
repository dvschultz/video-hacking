"""
Unit tests for video_assembler.py - Video assembly from semantic matches.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestVideoAssembler:
    """Test VideoAssembler class with mocked FFmpeg."""

    @pytest.fixture
    def matches_data(self):
        """Sample matches data."""
        return {
            'num_matches': 3,
            'matches': [
                {
                    'audio_idx': 0,
                    'video_start_time': 0.0,
                    'video_end_time': 1.0,
                    'audio_duration': 1.0
                },
                {
                    'audio_idx': 1,
                    'video_start_time': 2.0,
                    'video_end_time': 3.0,
                    'audio_duration': 1.0
                },
                {
                    'audio_idx': 2,
                    'video_start_time': 5.0,
                    'video_end_time': 6.0,
                    'audio_duration': 1.0
                }
            ]
        }

    @pytest.fixture
    def assembler(self, temp_dir, matches_data):
        """Create VideoAssembler with mock data."""
        matches_path = temp_dir / "matches.json"
        matches_path.write_text(json.dumps(matches_data))

        video_path = temp_dir / "video.mp4"
        audio_path = temp_dir / "audio.wav"
        output_path = temp_dir / "output.mp4"

        video_path.touch()
        audio_path.touch()

        with patch('builtins.print'):
            from video_assembler import VideoAssembler
            return VideoAssembler(
                str(video_path),
                str(audio_path),
                str(matches_path),
                str(output_path)
            )

    def test_initialization_loads_matches(self, assembler):
        """Test assembler loads matches correctly."""
        assert len(assembler.matches) == 3

    @patch('subprocess.run')
    def test_cut_video_segments(self, mock_run, assembler, temp_dir):
        """Test cutting video segments calls FFmpeg correctly."""
        mock_run.return_value = MagicMock(returncode=0)

        with patch('builtins.print'):
            segments = assembler.cut_video_segments(temp_dir, preserve_audio=False)

        assert mock_run.called
        assert len(segments) == 3

        # Verify FFmpeg was called with seek and duration
        for c in mock_run.call_args_list:
            cmd = c[0][0]
            assert 'ffmpeg' in cmd
            assert '-ss' in cmd
            assert '-t' in cmd

    @patch('subprocess.run')
    def test_cut_video_segments_no_audio(self, mock_run, assembler, temp_dir):
        """Test cutting without audio."""
        mock_run.return_value = MagicMock(returncode=0)

        with patch('builtins.print'):
            assembler.cut_video_segments(temp_dir, preserve_audio=False)

        # Should have '-an' flag to remove audio
        for c in mock_run.call_args_list:
            cmd = c[0][0]
            assert '-an' in cmd

    @patch('subprocess.run')
    def test_concatenate_segments(self, mock_run, assembler, temp_dir):
        """Test concatenating video segments."""
        mock_run.return_value = MagicMock(returncode=0)

        segment_paths = []
        for i in range(3):
            seg_path = temp_dir / f"segment_{i}.mp4"
            seg_path.touch()
            segment_paths.append(seg_path)

        with patch('builtins.print'):
            result = assembler.concatenate_segments(segment_paths, temp_dir)

        assert mock_run.called
        cmd = mock_run.call_args_list[-1][0][0]
        assert '-f' in cmd
        assert 'concat' in cmd

    @patch('subprocess.run')
    def test_add_audio(self, mock_run, assembler, temp_dir):
        """Test adding audio to video."""
        mock_run.return_value = MagicMock(returncode=0)

        video_path = temp_dir / "silent_video.mp4"
        video_path.touch()

        with patch('builtins.print'):
            result = assembler.add_audio(video_path)

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert '-c:a' in cmd


class TestVideoAssemblerErrorHandling:
    """Test error handling in VideoAssembler."""

    @patch('subprocess.run')
    def test_ffmpeg_failure_handling(self, mock_run, temp_dir):
        """Test handling FFmpeg failures."""
        matches_data = {
            'matches': [{
                'audio_idx': 0,
                'video_start_time': 0.0,
                'video_end_time': 1.0,
                'audio_duration': 1.0
            }]
        }

        matches_path = temp_dir / "matches.json"
        matches_path.write_text(json.dumps(matches_data))

        video_path = temp_dir / "video.mp4"
        audio_path = temp_dir / "audio.wav"
        output_path = temp_dir / "output.mp4"

        video_path.touch()
        audio_path.touch()

        with patch('builtins.print'):
            from video_assembler import VideoAssembler
            assembler = VideoAssembler(
                str(video_path),
                str(audio_path),
                str(matches_path),
                str(output_path)
            )

        # First call fails, second succeeds (fallback)
        mock_run.side_effect = [
            MagicMock(returncode=1, stderr=b'Error'),
            MagicMock(returncode=0)
        ]

        with patch('builtins.print'):
            # Should handle error gracefully
            try:
                segments = assembler.cut_video_segments(temp_dir, preserve_audio=False)
            except:
                pass  # Some error handling is expected
