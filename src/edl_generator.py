#!/usr/bin/env python3
"""
EDL Generator

Generates CMX 3600 EDL (Edit Decision List) files for video editing software.
Supports DaVinci Resolve, Premiere Pro, Final Cut Pro, Avid, etc.

CMX 3600 Format:
    TITLE: Project Name
    FCM: NON-DROP FRAME

    001  001      V     C        00:00:00:00 00:00:05:00 00:00:00:00 00:00:05:00
    * FROM CLIP NAME: clip1.mp4
    * SOURCE FILE: /full/path/to/clip1.mp4
"""

from pathlib import Path
from typing import List, Optional


class EDLEvent:
    """Represents a single event/edit in an EDL."""

    def __init__(self, event_num: int, source_path: str,
                 source_in: float, source_out: float,
                 record_in: float, record_out: float,
                 frame_rate: float, reel_name: str = None,
                 is_black: bool = False, comment: str = None,
                 pitch_shift_semitones: float = 0):
        self.event_num = event_num
        self.source_path = source_path
        self.source_in = source_in
        self.source_out = source_out
        self.record_in = record_in
        self.record_out = record_out
        self.frame_rate = frame_rate
        self.reel_name = reel_name or f"{event_num:03d}"
        self.is_black = is_black
        self.comment = comment
        self.pitch_shift_semitones = pitch_shift_semitones

    def _seconds_to_timecode(self, seconds: float, drop_frame: bool = False) -> str:
        """Convert seconds to SMPTE timecode HH:MM:SS:FF"""
        if seconds < 0:
            seconds = 0

        frames = round(seconds * self.frame_rate)
        frame_rate_int = int(round(self.frame_rate))

        ff = frames % frame_rate_int
        total_seconds = frames // frame_rate_int
        ss = total_seconds % 60
        mm = (total_seconds // 60) % 60
        hh = total_seconds // 3600

        sep = ';' if drop_frame else ':'
        return f"{hh:02d}:{mm:02d}:{ss:02d}{sep}{ff:02d}"

    def to_string(self, drop_frame: bool = False) -> str:
        """Generate EDL event string."""
        lines = []

        # Reel name - use "BL" for black, otherwise clip identifier
        reel = "BL" if self.is_black else self.reel_name[:8].ljust(8)

        # Timecodes
        src_in = self._seconds_to_timecode(self.source_in, drop_frame)
        src_out = self._seconds_to_timecode(self.source_out, drop_frame)
        rec_in = self._seconds_to_timecode(self.record_in, drop_frame)
        rec_out = self._seconds_to_timecode(self.record_out, drop_frame)

        # Main event line
        # Format: EVENT  REEL     TRACK EDIT     SRC_IN      SRC_OUT     REC_IN      REC_OUT
        event_line = f"{self.event_num:03d}  {reel} V     C        {src_in} {src_out} {rec_in} {rec_out}"
        lines.append(event_line)

        # Comments
        if self.is_black:
            duration = self.record_out - self.record_in
            lines.append(f"* BLACK/REST SEGMENT - {duration:.3f} seconds")
        else:
            clip_name = Path(self.source_path).name
            lines.append(f"* FROM CLIP NAME: {clip_name}")
            lines.append(f"* SOURCE FILE: {self.source_path}")
            lines.append(f"* PITCH SHIFT: {self.pitch_shift_semitones:+.1f} semitones")

        if self.comment:
            lines.append(f"* {self.comment}")

        return '\n'.join(lines)


class EDLGenerator:
    """
    Generates CMX 3600 EDL files.

    Usage:
        edl = EDLGenerator("My Project", frame_rate=24.0)
        edl.add_event("/path/to/clip1.mp4", 0.0, 5.0, 0.0, 5.0)
        edl.add_event("/path/to/clip2.mp4", 2.5, 5.5, 5.0, 8.0)
        edl.add_black(2.0, 8.0)  # 2 second black at timeline position 8.0
        edl.write("output.edl")
    """

    def __init__(self, title: str, frame_rate: float = 24.0, drop_frame: bool = False):
        """
        Initialize EDL generator.

        Args:
            title: Project title for EDL header
            frame_rate: Frame rate for timecode conversion (default: 24.0)
            drop_frame: Use drop-frame timecode (for 29.97/59.94 fps)

        Raises:
            ValueError: If frame_rate is not a positive number
        """
        if frame_rate <= 0:
            raise ValueError("Frame rate must be a positive number.")
        self.title = title
        self.frame_rate = frame_rate
        self.drop_frame = drop_frame
        self.events: List[EDLEvent] = []
        self._event_counter = 0
        self._current_record_position = 0.0

    def add_event(self, source_path: str, source_in: float, source_out: float,
                  record_in: float = None, record_out: float = None,
                  reel_name: str = None, comment: str = None,
                  pitch_shift_semitones: float = 0) -> 'EDLGenerator':
        """
        Add a video clip event to the EDL.

        Args:
            source_path: Full path to source video file
            source_in: Start time in source clip (seconds)
            source_out: End time in source clip (seconds)
            record_in: Start time on timeline (seconds), auto-calculated if None
            record_out: End time on timeline (seconds), auto-calculated if None
            reel_name: Optional reel identifier (max 8 chars)
            comment: Optional comment to add
            pitch_shift_semitones: Pitch shift to apply in semitones (0 = no shift)

        Returns:
            self for chaining
        """
        self._event_counter += 1

        duration = source_out - source_in

        if record_in is None:
            record_in = self._current_record_position
        if record_out is None:
            record_out = record_in + duration

        event = EDLEvent(
            event_num=self._event_counter,
            source_path=source_path,
            source_in=source_in,
            source_out=source_out,
            record_in=record_in,
            record_out=record_out,
            frame_rate=self.frame_rate,
            reel_name=reel_name,
            is_black=False,
            comment=comment,
            pitch_shift_semitones=pitch_shift_semitones
        )

        self.events.append(event)
        self._current_record_position = record_out

        return self

    def add_black(self, duration: float, record_in: float = None,
                  comment: str = None) -> 'EDLGenerator':
        """
        Add a black/rest segment to the EDL.

        Args:
            duration: Duration of black segment (seconds)
            record_in: Start time on timeline (seconds), auto-calculated if None
            comment: Optional comment

        Returns:
            self for chaining
        """
        self._event_counter += 1

        if record_in is None:
            record_in = self._current_record_position
        record_out = record_in + duration

        event = EDLEvent(
            event_num=self._event_counter,
            source_path="",
            source_in=0.0,
            source_out=duration,
            record_in=record_in,
            record_out=record_out,
            frame_rate=self.frame_rate,
            reel_name="BL",
            is_black=True,
            comment=comment
        )

        self.events.append(event)
        self._current_record_position = record_out

        return self

    def to_string(self) -> str:
        """Generate complete EDL as string."""
        lines = []

        # Header
        lines.append(f"TITLE: {self.title}")
        fcm = "DROP FRAME" if self.drop_frame else "NON-DROP FRAME"
        lines.append(f"FCM: {fcm}")
        lines.append("")

        # Events
        for event in self.events:
            lines.append(event.to_string(self.drop_frame))
            lines.append("")

        return '\n'.join(lines)

    def write(self, output_path: str) -> str:
        """
        Write EDL to file.

        Args:
            output_path: Output file path

        Returns:
            Path to written file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(self.to_string())

        return str(output_path)

    @property
    def total_duration(self) -> float:
        """Get total timeline duration."""
        return self._current_record_position

    @property
    def event_count(self) -> int:
        """Get number of events."""
        return len(self.events)


def generate_edl_from_matches(
    matches: List[dict],
    output_path: str,
    title: str = None,
    frame_rate: float = 24.0,
    video_fps_map: dict = None,
    verbose: bool = False,
    force_black_rests: bool = False
) -> str:
    """
    Generate EDL from a list of match dictionaries.

    This is the primary EDL generation function used by all matchers and assemblers.
    It handles duration, pitch, and semantic match types with proper FPS lookup.

    Args:
        matches: List of match dictionaries with match_type, guide_duration, source_clips, etc.
        output_path: Output EDL file path
        title: Optional title (defaults to output filename)
        frame_rate: Frame rate for EDL timecode display (default: 24.0)
        video_fps_map: Optional dict mapping video_path -> fps for accurate source timecodes
        verbose: If True, print progress messages
        force_black_rests: If True, treat all rest segments as black regardless of source clips

    Returns:
        Path to written EDL file
    """
    if title is None:
        title = Path(output_path).stem

    if video_fps_map is None:
        video_fps_map = {}

    edl = EDLGenerator(title, frame_rate=frame_rate)
    timeline_position = 0.0

    for match in matches:
        match_type = match.get('match_type', 'unknown')
        guide_duration = match.get('guide_duration', 0)
        source_clips = match.get('source_clips', [])

        # Check if this is a black segment
        is_black_segment = (
            match_type in ('rest', 'unmatched', 'missing') and not source_clips
        ) or (
            force_black_rests and match_type == 'rest'
        )

        if is_black_segment:
            # Black/rest segment
            comment_parts = [f"Guide segment {match.get('guide_segment_id', '?')}"]
            if match.get('is_rest'):
                comment_parts.append("(REST)")
            if match.get('guide_pitch_note'):
                comment_parts.append(f"Note: {match.get('guide_pitch_note')}")
            edl.add_black(guide_duration, record_in=timeline_position,
                          comment=", ".join(comment_parts))
        elif source_clips:
            # Video clip (including matched rests or silence clips)
            clip = source_clips[0]
            video_path = clip.get('video_path', '')

            # Get source video fps from mapping or fall back to EDL frame rate
            source_fps = video_fps_map.get(video_path, frame_rate)

            # Calculate source timecode from frames
            start_frame = clip.get('video_start_frame', 0)
            end_frame = clip.get('video_end_frame', start_frame)
            clip_duration = clip.get('duration', guide_duration)

            # Use frames if available, otherwise calculate from duration
            if end_frame > start_frame:
                source_in = start_frame / source_fps
                source_out = end_frame / source_fps
            else:
                source_in = start_frame / source_fps
                source_out = source_in + clip_duration

            # Build comment
            comment_parts = [f"Guide segment {match.get('guide_segment_id', '?')}"]
            if match.get('guide_pitch_note'):
                comment_parts.append(f"Note: {match.get('guide_pitch_note')}")
            if match_type == 'rest':
                comment_parts.append("(matched rest)" if match.get('is_rest') else "(silence clip)")
            crop_mode = clip.get('crop_mode', match.get('crop_mode', ''))
            if crop_mode:
                comment_parts.append(f"Crop: {crop_mode}")

            transpose = match.get('transpose_semitones', 0)

            edl.add_event(
                source_path=video_path,
                source_in=source_in,
                source_out=source_out,
                record_in=timeline_position,
                record_out=timeline_position + guide_duration,
                comment=", ".join(comment_parts),
                pitch_shift_semitones=transpose
            )

        # Advance timeline by guide duration
        timeline_position += guide_duration

    edl_path = edl.write(output_path)

    if verbose:
        print(f"EDL saved to: {edl_path}")
        print(f"  Events: {edl.event_count}")
        print(f"  Total duration: {edl.total_duration:.2f}s")

    return edl_path


def generate_edl_from_match_plan(match_plan: dict, output_path: str,
                                  title: str = None, frame_rate: float = 24.0,
                                  assembler_type: str = "duration") -> str:
    """
    Generate EDL from a match plan dictionary.

    DEPRECATED: Use generate_edl_from_matches() for new code.
    This wrapper exists for backwards compatibility.

    Args:
        match_plan: Match plan dictionary with 'matches' list
        output_path: Output EDL file path
        title: Optional title (defaults to output filename)
        frame_rate: Frame rate for timecode (default: 24.0)
        assembler_type: Type of assembler ("duration", "pitch", or "semantic")

    Returns:
        Path to written EDL file
    """
    matches = match_plan.get('matches', [])

    # Build video_fps_map from match_plan if available
    video_fps_map = match_plan.get('source_fps', {})

    return generate_edl_from_matches(
        matches=matches,
        output_path=output_path,
        title=title,
        frame_rate=frame_rate,
        video_fps_map=video_fps_map,
        verbose=False
    )
