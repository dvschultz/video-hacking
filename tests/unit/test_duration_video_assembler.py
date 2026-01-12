"""
Unit tests for duration_video_assembler.py - Video assembly from duration match plans.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from duration_video_assembler import DurationVideoAssembler


class TestDurationVideoAssemblerInitialization:
    """Test DurationVideoAssembler initialization."""

    def test_init_default_values(self, temp_dir):
        """Test default initialization values."""
        match_plan = temp_dir / "plan.json"
        match_plan.write_text('{"matches": [], "statistics": {}}')

        assembler = DurationVideoAssembler(
            str(match_plan),
            str(temp_dir / "output.mp4")
        )

        assert assembler.target_width is None
        assert assembler.target_height is None
        assert assembler.target_fps is None
        assert assembler.keep_audio == True

    def test_init_custom_resolution(self, temp_dir):
        """Test initialization with custom resolution."""
        match_plan = temp_dir / "plan.json"
        match_plan.write_text('{"matches": [], "statistics": {}}')

        assembler = DurationVideoAssembler(
            str(match_plan),
            str(temp_dir / "output.mp4"),
            target_width=1920,
            target_height=1080
        )

        assert assembler.target_width == 1920
        assert assembler.target_height == 1080

    def test_init_creates_temp_dir(self, temp_dir):
        """Test that temp directory is created."""
        match_plan = temp_dir / "plan.json"
        match_plan.write_text('{"matches": [], "statistics": {}}')

        temp_subdir = temp_dir / "custom_temp"

        assembler = DurationVideoAssembler(
            str(match_plan),
            str(temp_dir / "output.mp4"),
            temp_dir=str(temp_subdir)
        )

        assert temp_subdir.exists()


class TestDurationVideoAssemblerLoadMatchPlan:
    """Test loading match plans."""

    def test_load_match_plan_basic(self, temp_dir):
        """Test loading a basic match plan."""
        match_plan_data = {
            'matches': [
                {
                    'guide_segment_id': 0,
                    'match_type': 'duration',
                    'guide_duration': 2.0,
                    'source_clips': [{
                        'video_path': '/path/to/clip.mp4',
                        'video_start_frame': 0,
                        'video_end_frame': 48,
                        'duration': 2.0
                    }]
                }
            ],
            'statistics': {
                'matched_segments': 1,
                'unmatched_segments': 0,
                'rest_segments_black_frames': 0,
                'total_output_duration': 2.0
            }
        }

        match_plan = temp_dir / "plan.json"
        match_plan.write_text(json.dumps(match_plan_data))

        assembler = DurationVideoAssembler(
            str(match_plan),
            str(temp_dir / "output.mp4")
        )

        with patch('builtins.print'):
            assembler.load_match_plan()

        assert len(assembler.matches) == 1
        assert assembler.matches[0]['guide_duration'] == 2.0


class TestDurationVideoAssemblerVideoMetadata:
    """Test video metadata retrieval."""

    def setup_method(self):
        """Clear lru_cache before each test to ensure test isolation."""
        from src import video_utils
        video_utils.clear_caches()

    def test_get_video_resolution(self, temp_dir):
        """Test getting video resolution via ffprobe."""
        match_plan = temp_dir / "plan.json"
        match_plan.write_text('{"matches": [], "statistics": {}}')

        assembler = DurationVideoAssembler(
            str(match_plan),
            str(temp_dir / "output.mp4")
        )

        mock_result = MagicMock()
        mock_result.stdout = "1920,1080\n"

        with patch('subprocess.run', return_value=mock_result):
            width, height = assembler.get_video_resolution("/path/to/resolution_test_video.mp4")

        assert width == 1920
        assert height == 1080

    def test_get_video_resolution_cached(self, temp_dir):
        """Test that resolution is cached via lru_cache."""
        match_plan = temp_dir / "plan.json"
        match_plan.write_text('{"matches": [], "statistics": {}}')

        assembler = DurationVideoAssembler(
            str(match_plan),
            str(temp_dir / "output.mp4")
        )

        mock_result = MagicMock()
        mock_result.stdout = "1920,1080\n"

        # Use a unique path for this test
        test_path = "/path/to/cache_test_video.mp4"

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            assembler.get_video_resolution(test_path)
            assembler.get_video_resolution(test_path)

            # Should only call subprocess once due to lru_cache
            assert mock_run.call_count == 1

    def test_get_video_fps(self, temp_dir):
        """Test getting video frame rate via ffprobe."""
        match_plan = temp_dir / "plan.json"
        match_plan.write_text('{"matches": [], "statistics": {}}')

        assembler = DurationVideoAssembler(
            str(match_plan),
            str(temp_dir / "output.mp4")
        )

        mock_result = MagicMock()
        mock_result.stdout = "24/1\n"

        with patch('subprocess.run', return_value=mock_result):
            fps = assembler.get_video_fps("/path/to/fps_test_video.mp4")

        assert fps == 24.0

    def test_get_video_fps_fractional(self, temp_dir):
        """Test getting fractional frame rate (e.g., 29.97)."""
        match_plan = temp_dir / "plan.json"
        match_plan.write_text('{"matches": [], "statistics": {}}')

        assembler = DurationVideoAssembler(
            str(match_plan),
            str(temp_dir / "output.mp4")
        )

        mock_result = MagicMock()
        mock_result.stdout = "30000/1001\n"

        with patch('subprocess.run', return_value=mock_result):
            fps = assembler.get_video_fps("/path/to/fps_fractional_video.mp4")

        assert abs(fps - 29.97) < 0.01

    def test_get_video_fps_fallback(self, temp_dir):
        """Test fallback to 24 fps on error."""
        import subprocess
        match_plan = temp_dir / "plan.json"
        match_plan.write_text('{"matches": [], "statistics": {}}')

        assembler = DurationVideoAssembler(
            str(match_plan),
            str(temp_dir / "output.mp4")
        )

        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'ffprobe')):
            with patch('builtins.print'):
                fps = assembler.get_video_fps("/path/to/video.mp4")

        assert fps == 24.0


class TestDurationVideoAssemblerProcessMatch:
    """Test match processing."""

    @pytest.fixture
    def assembler(self, temp_dir):
        """Create assembler instance."""
        match_plan = temp_dir / "plan.json"
        match_plan.write_text('{"matches": [], "statistics": {}}')

        return DurationVideoAssembler(
            str(match_plan),
            str(temp_dir / "output.mp4"),
            temp_dir=str(temp_dir / "temp")
        )

    def test_process_rest_match(self, assembler):
        """Test processing a rest/black match."""
        match = {
            'match_type': 'rest',
            'guide_duration': 2.0
        }

        assembler.target_width = 1920
        assembler.target_height = 1080
        assembler.target_fps = 24

        with patch.object(assembler, 'generate_rest_clip') as mock_gen:
            clips = assembler.process_match(match, 0)

        mock_gen.assert_called_once()
        assert len(clips) == 1

    def test_process_duration_match(self, assembler):
        """Test processing a duration match."""
        match = {
            'match_type': 'duration',
            'guide_duration': 2.0,
            'source_clips': [{
                'video_path': '/path/to/clip.mp4',
                'video_start_frame': 0,
                'duration': 2.0
            }]
        }

        with patch.object(assembler, 'get_video_fps', return_value=24.0):
            with patch.object(assembler, 'extract_clip') as mock_extract:
                clips = assembler.process_match(match, 0)

        mock_extract.assert_called_once()
        assert len(clips) == 1

    def test_process_unmatched(self, assembler):
        """Test processing an unmatched segment."""
        match = {
            'match_type': 'unmatched',
            'guide_duration': 1.5
        }

        assembler.target_width = 1920
        assembler.target_height = 1080
        assembler.target_fps = 24

        with patch.object(assembler, 'generate_rest_clip') as mock_gen:
            clips = assembler.process_match(match, 0)

        mock_gen.assert_called_once()


class TestDurationVideoAssemblerClipValidation:
    """Test clip validation."""

    def test_is_valid_clip_missing_file(self, temp_dir):
        """Test validation of missing file."""
        match_plan = temp_dir / "plan.json"
        match_plan.write_text('{"matches": [], "statistics": {}}')

        assembler = DurationVideoAssembler(
            str(match_plan),
            str(temp_dir / "output.mp4")
        )

        result = assembler.is_valid_clip("/nonexistent/clip.mp4")
        assert result == False

    def test_is_valid_clip_valid(self, temp_dir):
        """Test validation of valid clip."""
        match_plan = temp_dir / "plan.json"
        match_plan.write_text('{"matches": [], "statistics": {}}')

        # Create a dummy file
        clip_file = temp_dir / "clip.mp4"
        clip_file.write_text("dummy")

        assembler = DurationVideoAssembler(
            str(match_plan),
            str(temp_dir / "output.mp4")
        )

        mock_result = MagicMock()
        mock_result.stdout = "2.5\n"

        with patch('subprocess.run', return_value=mock_result):
            result = assembler.is_valid_clip(str(clip_file))

        assert result == True

    def test_is_valid_clip_zero_duration(self, temp_dir):
        """Test validation of zero-duration clip."""
        match_plan = temp_dir / "plan.json"
        match_plan.write_text('{"matches": [], "statistics": {}}')

        clip_file = temp_dir / "clip.mp4"
        clip_file.write_text("dummy")

        assembler = DurationVideoAssembler(
            str(match_plan),
            str(temp_dir / "output.mp4")
        )

        mock_result = MagicMock()
        mock_result.stdout = "0.0\n"

        with patch('subprocess.run', return_value=mock_result):
            result = assembler.is_valid_clip(str(clip_file))

        assert result == False


class TestDurationVideoAssemblerEDLGeneration:
    """Test EDL generation."""

    def test_generate_edl(self, temp_dir):
        """Test EDL generation from match plan."""
        match_plan_data = {
            'matches': [
                {
                    'guide_segment_id': 0,
                    'match_type': 'duration',
                    'guide_duration': 2.0,
                    'source_clips': [{
                        'video_path': '/path/to/clip.mp4',
                        'video_start_frame': 0,
                        'video_end_frame': 48,
                        'duration': 2.0
                    }]
                },
                {
                    'guide_segment_id': 1,
                    'match_type': 'rest',
                    'guide_duration': 1.0,
                    'is_rest': True,
                    'source_clips': []
                }
            ],
            'statistics': {
                'matched_segments': 1,
                'unmatched_segments': 0,
                'rest_segments_black_frames': 1,
                'total_output_duration': 3.0
            }
        }

        match_plan = temp_dir / "plan.json"
        match_plan.write_text(json.dumps(match_plan_data))

        assembler = DurationVideoAssembler(
            str(match_plan),
            str(temp_dir / "output.mp4")
        )

        with patch('builtins.print'):
            assembler.load_match_plan()

        assembler.target_fps = 24.0

        with patch.object(assembler, 'get_video_fps', return_value=24.0):
            edl_path = temp_dir / "output.edl"
            result = assembler.generate_edl(str(edl_path))

        assert Path(result).exists()
        content = Path(result).read_text()
        assert "TITLE:" in content
        assert "clip.mp4" in content


class TestDurationVideoAssemblerAnalysis:
    """Test source video analysis."""

    def test_analyze_source_videos(self, temp_dir):
        """Test analyzing source videos for resolution/fps."""
        match_plan_data = {
            'matches': [
                {
                    'guide_segment_id': 0,
                    'match_type': 'duration',
                    'guide_duration': 2.0,
                    'source_clips': [{
                        'video_path': '/path/to/clip1.mp4',
                        'video_start_frame': 0,
                        'duration': 2.0
                    }]
                },
                {
                    'guide_segment_id': 1,
                    'match_type': 'duration',
                    'guide_duration': 3.0,
                    'source_clips': [{
                        'video_path': '/path/to/clip2.mp4',
                        'video_start_frame': 0,
                        'duration': 3.0
                    }]
                }
            ],
            'statistics': {
                'matched_segments': 2,
                'unmatched_segments': 0,
                'rest_segments_black_frames': 0,
                'total_output_duration': 5.0
            }
        }

        match_plan = temp_dir / "plan.json"
        match_plan.write_text(json.dumps(match_plan_data))

        assembler = DurationVideoAssembler(
            str(match_plan),
            str(temp_dir / "output.mp4")
        )

        with patch('builtins.print'):
            assembler.load_match_plan()

        with patch.object(assembler, 'get_video_resolution', return_value=(1920, 1080)):
            with patch.object(assembler, 'get_video_fps', return_value=24.0):
                analysis = assembler.analyze_source_videos()

        assert analysis['min_width'] == 1920
        assert analysis['min_height'] == 1080
        assert analysis['min_fps'] == 24.0

    def test_analyze_source_videos_empty(self, temp_dir):
        """Test analyzing with no source videos."""
        match_plan_data = {
            'matches': [
                {
                    'guide_segment_id': 0,
                    'match_type': 'rest',
                    'guide_duration': 2.0,
                    'source_clips': []
                }
            ],
            'statistics': {
                'matched_segments': 0,
                'unmatched_segments': 0,
                'rest_segments_black_frames': 1,
                'total_output_duration': 2.0
            }
        }

        match_plan = temp_dir / "plan.json"
        match_plan.write_text(json.dumps(match_plan_data))

        assembler = DurationVideoAssembler(
            str(match_plan),
            str(temp_dir / "output.mp4")
        )

        with patch('builtins.print'):
            assembler.load_match_plan()
            analysis = assembler.analyze_source_videos()

        # Should return defaults when no videos found
        assert analysis['min_width'] == 1920
        assert analysis['min_height'] == 1080
