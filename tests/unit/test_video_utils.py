"""Unit tests for video_utils.py."""

import os
import platform
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src import video_utils


class TestGetVideoResolution:
    """Tests for get_video_resolution function."""

    def setup_method(self):
        """Clear caches before each test."""
        video_utils.clear_caches()

    def test_get_video_resolution_success(self):
        """Test successful resolution extraction."""
        mock_result = MagicMock()
        mock_result.stdout = "1920,1080\n"

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            width, height = video_utils.get_video_resolution("/path/to/video.mp4")

            assert width == 1920
            assert height == 1080
            mock_run.assert_called_once()

    def test_get_video_resolution_cached(self):
        """Test that results are cached via lru_cache."""
        mock_result = MagicMock()
        mock_result.stdout = "1280,720\n"

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            # First call
            video_utils.get_video_resolution("/path/to/cached_video.mp4")
            # Second call should use cache
            video_utils.get_video_resolution("/path/to/cached_video.mp4")

            # Should only call subprocess once
            assert mock_run.call_count == 1

    def test_get_video_resolution_fallback_on_error(self):
        """Test fallback to default resolution on ffprobe error."""
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'ffprobe')):
            width, height = video_utils.get_video_resolution("/path/to/invalid.mp4")

            # Should return default 1920x1080
            assert width == 1920
            assert height == 1080

    def test_get_video_resolution_invalid_output(self):
        """Test fallback on invalid ffprobe output."""
        mock_result = MagicMock()
        mock_result.stdout = "invalid"

        with patch('subprocess.run', return_value=mock_result):
            width, height = video_utils.get_video_resolution("/path/to/bad.mp4")

            # Should return default
            assert width == 1920
            assert height == 1080


class TestGetVideoFps:
    """Tests for get_video_fps function."""

    def setup_method(self):
        """Clear caches before each test."""
        video_utils.clear_caches()

    def test_get_video_fps_success(self):
        """Test successful FPS extraction."""
        mock_result = MagicMock()
        mock_result.stdout = "24000/1001\n"

        with patch('subprocess.run', return_value=mock_result):
            fps = video_utils.get_video_fps("/path/to/video.mp4")

            # 24000/1001 ≈ 23.976
            assert abs(fps - 23.976) < 0.01

    def test_get_video_fps_integer(self):
        """Test FPS extraction with integer value."""
        mock_result = MagicMock()
        mock_result.stdout = "30\n"

        with patch('subprocess.run', return_value=mock_result):
            fps = video_utils.get_video_fps("/path/to/int_fps.mp4")

            assert fps == 30.0

    def test_get_video_fps_cached(self):
        """Test that FPS results are cached."""
        mock_result = MagicMock()
        mock_result.stdout = "25\n"

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            video_utils.get_video_fps("/path/to/fps_cached.mp4")
            video_utils.get_video_fps("/path/to/fps_cached.mp4")

            assert mock_run.call_count == 1

    def test_get_video_fps_division_by_zero(self):
        """Test handling of division by zero in frame rate."""
        mock_result = MagicMock()
        mock_result.stdout = "24/0\n"

        with patch('subprocess.run', return_value=mock_result):
            fps = video_utils.get_video_fps("/path/to/zero_den.mp4")

            # Should fallback to 24.0
            assert fps == 24.0

    def test_get_video_fps_negative_value(self):
        """Test handling of negative FPS value."""
        mock_result = MagicMock()
        mock_result.stdout = "-30\n"

        with patch('subprocess.run', return_value=mock_result):
            fps = video_utils.get_video_fps("/path/to/negative.mp4")

            # Should fallback to 24.0
            assert fps == 24.0

    def test_get_video_fps_fallback_on_error(self):
        """Test fallback on ffprobe error."""
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'ffprobe')):
            fps = video_utils.get_video_fps("/path/to/error.mp4")

            assert fps == 24.0


class TestIsValidClip:
    """Tests for is_valid_clip function."""

    def test_is_valid_clip_nonexistent_file(self, tmp_path):
        """Test that nonexistent files are invalid."""
        assert video_utils.is_valid_clip(str(tmp_path / "nonexistent.mp4")) is False

    def test_is_valid_clip_valid_duration(self, tmp_path):
        """Test clip with valid duration."""
        # Create a dummy file
        test_file = tmp_path / "valid.mp4"
        test_file.touch()

        mock_result = MagicMock()
        mock_result.stdout = "5.0\n"

        with patch('subprocess.run', return_value=mock_result):
            assert video_utils.is_valid_clip(str(test_file)) is True

    def test_is_valid_clip_zero_duration(self, tmp_path):
        """Test clip with zero duration is invalid."""
        test_file = tmp_path / "zero.mp4"
        test_file.touch()

        mock_result = MagicMock()
        mock_result.stdout = "0.0\n"

        with patch('subprocess.run', return_value=mock_result):
            assert video_utils.is_valid_clip(str(test_file)) is False


class TestResetTerminal:
    """Tests for reset_terminal function."""

    @patch('platform.system')
    @patch('os.system')
    def test_reset_terminal_unix(self, mock_os_system, mock_platform):
        """Test terminal reset on Unix systems."""
        mock_platform.return_value = 'Linux'

        video_utils.reset_terminal()

        mock_os_system.assert_called_once_with('stty sane 2>/dev/null')

    @patch('platform.system')
    @patch('os.system')
    def test_reset_terminal_macos(self, mock_os_system, mock_platform):
        """Test terminal reset on macOS."""
        mock_platform.return_value = 'Darwin'

        video_utils.reset_terminal()

        mock_os_system.assert_called_once_with('stty sane 2>/dev/null')

    @patch('platform.system')
    @patch('os.system')
    def test_reset_terminal_windows(self, mock_os_system, mock_platform):
        """Test terminal reset skipped on Windows."""
        mock_platform.return_value = 'Windows'

        video_utils.reset_terminal()

        mock_os_system.assert_not_called()


class TestGenerateBlackClip:
    """Tests for generate_black_clip function."""

    def test_generate_black_clip_invalid_width(self, tmp_path):
        """Test validation of width parameter."""
        with pytest.raises(ValueError, match="Invalid dimensions"):
            video_utils.generate_black_clip(
                duration=1.0,
                output_path=str(tmp_path / "black.mp4"),
                width=0,
                height=1080
            )

    def test_generate_black_clip_invalid_height(self, tmp_path):
        """Test validation of height parameter."""
        with pytest.raises(ValueError, match="Invalid dimensions"):
            video_utils.generate_black_clip(
                duration=1.0,
                output_path=str(tmp_path / "black.mp4"),
                width=1920,
                height=-100
            )

    def test_generate_black_clip_invalid_duration(self, tmp_path):
        """Test validation of duration parameter."""
        with pytest.raises(ValueError, match="Invalid duration"):
            video_utils.generate_black_clip(
                duration=0,
                output_path=str(tmp_path / "black.mp4")
            )

    def test_generate_black_clip_invalid_fps(self, tmp_path):
        """Test validation of fps parameter."""
        with pytest.raises(ValueError, match="Invalid fps"):
            video_utils.generate_black_clip(
                duration=1.0,
                output_path=str(tmp_path / "black.mp4"),
                fps=-24
            )

    def test_generate_black_clip_success(self, tmp_path):
        """Test successful black clip generation."""
        with patch('subprocess.run') as mock_run:
            video_utils.generate_black_clip(
                duration=1.0,
                output_path=str(tmp_path / "black.mp4"),
                width=1920,
                height=1080,
                fps=24.0
            )

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            cmd = call_args[0][0]

            # Verify FFmpeg command structure
            assert cmd[0] == 'ffmpeg'
            assert '-y' in cmd
            assert 'color=c=black:s=1920x1080:r=24.0:d=1.0' in ' '.join(cmd)


class TestClearCaches:
    """Tests for clear_caches function."""

    def test_clear_caches(self):
        """Test that clear_caches clears lru_cache."""
        mock_result = MagicMock()
        mock_result.stdout = "1920,1080\n"

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            # First call
            video_utils.get_video_resolution("/path/to/clear_test.mp4")
            assert mock_run.call_count == 1

            # Clear caches
            video_utils.clear_caches()

            # Call again - should call subprocess again since cache was cleared
            video_utils.get_video_resolution("/path/to/clear_test.mp4")
            assert mock_run.call_count == 2


class TestNormalizeClip:
    """Tests for normalize_clip function."""

    def test_normalize_clip_success(self, tmp_path):
        """Test successful clip normalization."""
        with patch('subprocess.run') as mock_run:
            success, error = video_utils.normalize_clip(
                clip_file=str(tmp_path / "input.mp4"),
                output_path=str(tmp_path / "output.mp4"),
                width=1920,
                height=1080
            )

            assert success is True
            assert error is None
            mock_run.assert_called_once()

    def test_normalize_clip_failure(self, tmp_path):
        """Test clip normalization failure."""
        error_msg = b"FFmpeg error message"
        mock_error = subprocess.CalledProcessError(1, 'ffmpeg')
        mock_error.stderr = error_msg

        with patch('subprocess.run', side_effect=mock_error):
            success, error = video_utils.normalize_clip(
                clip_file=str(tmp_path / "input.mp4"),
                output_path=str(tmp_path / "output.mp4"),
                width=1920,
                height=1080
            )

            assert success is False
            assert "FFmpeg error" in error
