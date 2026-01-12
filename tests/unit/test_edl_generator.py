"""
Unit tests for edl_generator.py - EDL (Edit Decision List) generation.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from edl_generator import EDLGenerator, EDLEvent, generate_edl_from_match_plan


class TestEDLEventTimecode:
    """Test EDLEvent timecode conversion."""

    def test_seconds_to_timecode_basic(self):
        """Test basic seconds to timecode conversion."""
        event = EDLEvent(
            event_num=1,
            source_path="/test/video.mp4",
            source_in=0.0,
            source_out=5.0,
            record_in=0.0,
            record_out=5.0,
            frame_rate=24.0
        )

        # 0 seconds = 00:00:00:00
        assert event._seconds_to_timecode(0.0) == "00:00:00:00"

    def test_seconds_to_timecode_one_second(self):
        """Test one second conversion at 24fps."""
        event = EDLEvent(
            event_num=1,
            source_path="/test/video.mp4",
            source_in=0.0,
            source_out=1.0,
            record_in=0.0,
            record_out=1.0,
            frame_rate=24.0
        )

        # 1 second at 24fps = 00:00:01:00
        assert event._seconds_to_timecode(1.0) == "00:00:01:00"

    def test_seconds_to_timecode_with_frames(self):
        """Test conversion with fractional seconds."""
        event = EDLEvent(
            event_num=1,
            source_path="/test/video.mp4",
            source_in=0.0,
            source_out=1.5,
            record_in=0.0,
            record_out=1.5,
            frame_rate=24.0
        )

        # 1.5 seconds at 24fps = 36 frames = 00:00:01:12
        assert event._seconds_to_timecode(1.5) == "00:00:01:12"

    def test_seconds_to_timecode_minutes(self):
        """Test conversion with minutes."""
        event = EDLEvent(
            event_num=1,
            source_path="/test/video.mp4",
            source_in=0.0,
            source_out=65.0,
            record_in=0.0,
            record_out=65.0,
            frame_rate=24.0
        )

        # 65 seconds = 1 minute 5 seconds = 00:01:05:00
        assert event._seconds_to_timecode(65.0) == "00:01:05:00"

    def test_seconds_to_timecode_hours(self):
        """Test conversion with hours."""
        event = EDLEvent(
            event_num=1,
            source_path="/test/video.mp4",
            source_in=0.0,
            source_out=3661.0,
            record_in=0.0,
            record_out=3661.0,
            frame_rate=24.0
        )

        # 3661 seconds = 1 hour, 1 minute, 1 second = 01:01:01:00
        assert event._seconds_to_timecode(3661.0) == "01:01:01:00"

    def test_seconds_to_timecode_negative_clamped(self):
        """Test that negative seconds are clamped to zero."""
        event = EDLEvent(
            event_num=1,
            source_path="/test/video.mp4",
            source_in=0.0,
            source_out=1.0,
            record_in=0.0,
            record_out=1.0,
            frame_rate=24.0
        )

        assert event._seconds_to_timecode(-5.0) == "00:00:00:00"

    def test_seconds_to_timecode_drop_frame_separator(self):
        """Test drop-frame uses semicolon separator."""
        event = EDLEvent(
            event_num=1,
            source_path="/test/video.mp4",
            source_in=0.0,
            source_out=1.0,
            record_in=0.0,
            record_out=1.0,
            frame_rate=29.97
        )

        tc = event._seconds_to_timecode(1.0, drop_frame=True)
        assert ";" in tc


class TestEDLEventFormatting:
    """Test EDLEvent string formatting."""

    def test_to_string_video_event(self):
        """Test video event string formatting."""
        event = EDLEvent(
            event_num=1,
            source_path="/path/to/clip.mp4",
            source_in=0.0,
            source_out=5.0,
            record_in=0.0,
            record_out=5.0,
            frame_rate=24.0,
            reel_name="001"
        )

        output = event.to_string()

        assert "001" in output
        assert "V     C" in output
        assert "FROM CLIP NAME: clip.mp4" in output
        assert "SOURCE FILE: /path/to/clip.mp4" in output

    def test_to_string_black_event(self):
        """Test black/rest event formatting."""
        event = EDLEvent(
            event_num=2,
            source_path="",
            source_in=0.0,
            source_out=2.0,
            record_in=5.0,
            record_out=7.0,
            frame_rate=24.0,
            is_black=True
        )

        output = event.to_string()

        assert "BL" in output
        assert "BLACK/REST SEGMENT" in output
        assert "2.000 seconds" in output

    def test_to_string_with_comment(self):
        """Test event with custom comment."""
        event = EDLEvent(
            event_num=1,
            source_path="/path/to/clip.mp4",
            source_in=0.0,
            source_out=5.0,
            record_in=0.0,
            record_out=5.0,
            frame_rate=24.0,
            comment="Custom note"
        )

        output = event.to_string()
        assert "* Custom note" in output


class TestEDLGeneratorInitialization:
    """Test EDLGenerator initialization."""

    def test_init_default_values(self):
        """Test default initialization values."""
        edl = EDLGenerator("Test Project")

        assert edl.title == "Test Project"
        assert edl.frame_rate == 24.0
        assert edl.drop_frame == False
        assert edl.events == []

    def test_init_custom_frame_rate(self):
        """Test initialization with custom frame rate."""
        edl = EDLGenerator("Test", frame_rate=30.0)

        assert edl.frame_rate == 30.0

    def test_init_drop_frame(self):
        """Test initialization with drop frame."""
        edl = EDLGenerator("Test", frame_rate=29.97, drop_frame=True)

        assert edl.drop_frame == True

    def test_init_invalid_frame_rate_zero(self):
        """Test that zero frame rate raises error."""
        with pytest.raises(ValueError, match="positive number"):
            EDLGenerator("Test", frame_rate=0)

    def test_init_invalid_frame_rate_negative(self):
        """Test that negative frame rate raises error."""
        with pytest.raises(ValueError, match="positive number"):
            EDLGenerator("Test", frame_rate=-24.0)


class TestEDLGeneratorAddEvent:
    """Test EDLGenerator.add_event()."""

    def test_add_event_basic(self):
        """Test adding a basic event."""
        edl = EDLGenerator("Test")
        edl.add_event("/path/to/clip.mp4", 0.0, 5.0)

        assert len(edl.events) == 1
        assert edl.events[0].source_path == "/path/to/clip.mp4"
        assert edl.events[0].source_in == 0.0
        assert edl.events[0].source_out == 5.0

    def test_add_event_auto_record_times(self):
        """Test auto-calculated record times."""
        edl = EDLGenerator("Test")
        edl.add_event("/path/to/clip1.mp4", 0.0, 5.0)
        edl.add_event("/path/to/clip2.mp4", 0.0, 3.0)

        # First event: record 0-5
        assert edl.events[0].record_in == 0.0
        assert edl.events[0].record_out == 5.0

        # Second event: record 5-8 (auto-positioned after first)
        assert edl.events[1].record_in == 5.0
        assert edl.events[1].record_out == 8.0

    def test_add_event_explicit_record_times(self):
        """Test explicit record times."""
        edl = EDLGenerator("Test")
        edl.add_event("/path/to/clip.mp4", 0.0, 5.0, record_in=10.0, record_out=15.0)

        assert edl.events[0].record_in == 10.0
        assert edl.events[0].record_out == 15.0

    def test_add_event_chaining(self):
        """Test method chaining."""
        edl = EDLGenerator("Test")
        result = edl.add_event("/path/clip1.mp4", 0.0, 5.0) \
                    .add_event("/path/clip2.mp4", 0.0, 3.0)

        assert result is edl
        assert len(edl.events) == 2


class TestEDLGeneratorAddBlack:
    """Test EDLGenerator.add_black()."""

    def test_add_black_basic(self):
        """Test adding a black segment."""
        edl = EDLGenerator("Test")
        edl.add_black(2.0)

        assert len(edl.events) == 1
        assert edl.events[0].is_black == True
        assert edl.events[0].source_out == 2.0

    def test_add_black_auto_position(self):
        """Test black segment auto-positioned after video."""
        edl = EDLGenerator("Test")
        edl.add_event("/path/to/clip.mp4", 0.0, 5.0)
        edl.add_black(2.0)

        assert edl.events[1].record_in == 5.0
        assert edl.events[1].record_out == 7.0

    def test_add_black_explicit_position(self):
        """Test black segment at explicit position."""
        edl = EDLGenerator("Test")
        edl.add_black(2.0, record_in=10.0)

        assert edl.events[0].record_in == 10.0
        assert edl.events[0].record_out == 12.0

    def test_add_black_chaining(self):
        """Test method chaining with black segments."""
        edl = EDLGenerator("Test")
        result = edl.add_black(1.0).add_event("/path/clip.mp4", 0.0, 5.0).add_black(1.0)

        assert result is edl
        assert len(edl.events) == 3


class TestEDLGeneratorOutput:
    """Test EDLGenerator output methods."""

    def test_to_string_header(self):
        """Test EDL string header."""
        edl = EDLGenerator("My Project", frame_rate=24.0)
        output = edl.to_string()

        assert "TITLE: My Project" in output
        assert "FCM: NON-DROP FRAME" in output

    def test_to_string_drop_frame_header(self):
        """Test drop-frame header."""
        edl = EDLGenerator("My Project", frame_rate=29.97, drop_frame=True)
        output = edl.to_string()

        assert "FCM: DROP FRAME" in output

    def test_to_string_with_events(self):
        """Test EDL string with events."""
        edl = EDLGenerator("Test")
        edl.add_event("/path/clip1.mp4", 0.0, 5.0)
        edl.add_black(1.0)
        edl.add_event("/path/clip2.mp4", 0.0, 3.0)

        output = edl.to_string()

        assert "001" in output
        assert "002" in output
        assert "003" in output
        assert "clip1.mp4" in output
        assert "clip2.mp4" in output

    def test_write_file(self, temp_dir):
        """Test writing EDL to file."""
        edl = EDLGenerator("Test Project")
        edl.add_event("/path/to/clip.mp4", 0.0, 5.0)

        output_path = temp_dir / "test.edl"
        result = edl.write(str(output_path))

        assert output_path.exists()
        assert result == str(output_path)

        content = output_path.read_text()
        assert "TITLE: Test Project" in content

    def test_write_creates_directories(self, temp_dir):
        """Test that write creates parent directories."""
        edl = EDLGenerator("Test")
        edl.add_event("/path/to/clip.mp4", 0.0, 5.0)

        output_path = temp_dir / "subdir" / "nested" / "test.edl"
        edl.write(str(output_path))

        assert output_path.exists()


class TestEDLGeneratorProperties:
    """Test EDLGenerator properties."""

    def test_total_duration_empty(self):
        """Test total duration with no events."""
        edl = EDLGenerator("Test")
        assert edl.total_duration == 0.0

    def test_total_duration_with_events(self):
        """Test total duration calculation."""
        edl = EDLGenerator("Test")
        edl.add_event("/path/clip1.mp4", 0.0, 5.0)  # 0-5
        edl.add_black(2.0)  # 5-7
        edl.add_event("/path/clip2.mp4", 0.0, 3.0)  # 7-10

        assert edl.total_duration == 10.0

    def test_event_count_empty(self):
        """Test event count with no events."""
        edl = EDLGenerator("Test")
        assert edl.event_count == 0

    def test_event_count_with_events(self):
        """Test event count."""
        edl = EDLGenerator("Test")
        edl.add_event("/path/clip1.mp4", 0.0, 5.0)
        edl.add_black(2.0)
        edl.add_event("/path/clip2.mp4", 0.0, 3.0)

        assert edl.event_count == 3


class TestGenerateEDLFromMatchPlan:
    """Test generate_edl_from_match_plan helper function."""

    def test_generate_from_duration_match_plan(self, temp_dir):
        """Test EDL generation from duration match plan."""
        match_plan = {
            'matches': [
                {
                    'guide_segment_id': 0,
                    'match_type': 'duration',
                    'guide_duration': 2.0,
                    'source_clips': [{
                        'video_path': '/path/to/clip.mp4',
                        'video_start_frame': 0,
                        'duration': 2.0
                    }]
                },
                {
                    'guide_segment_id': 1,
                    'match_type': 'rest',
                    'guide_duration': 1.0,
                    'source_clips': []
                }
            ]
        }

        output_path = temp_dir / "test.edl"
        result = generate_edl_from_match_plan(
            match_plan,
            str(output_path),
            assembler_type="duration"
        )

        assert Path(result).exists()
        content = Path(result).read_text()
        assert "clip.mp4" in content

    def test_generate_from_pitch_match_plan(self, temp_dir):
        """Test EDL generation from pitch match plan."""
        match_plan = {
            'matches': [
                {
                    'guide_segment_id': 0,
                    'match_type': 'exact',
                    'guide_duration': 0.5,
                    'guide_pitch_note': 'C4',
                    'transpose_semitones': 0,
                    'source_clips': [{
                        'video_path': '/path/to/source.mp4',
                        'video_start_frame': 24,
                        'video_end_frame': 36
                    }]
                }
            ]
        }

        output_path = temp_dir / "test.edl"
        result = generate_edl_from_match_plan(
            match_plan,
            str(output_path),
            assembler_type="pitch"
        )

        assert Path(result).exists()

    def test_generate_uses_filename_as_title(self, temp_dir):
        """Test that filename is used as title when not specified."""
        match_plan = {'matches': []}

        output_path = temp_dir / "my_project.edl"
        generate_edl_from_match_plan(match_plan, str(output_path))

        content = output_path.read_text()
        assert "TITLE: my_project" in content
