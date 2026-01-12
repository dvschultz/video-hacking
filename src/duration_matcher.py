#!/usr/bin/env python3
"""
Duration Matcher

Matches guide video segments to source video clips based on duration.
Selects the shortest source clip that is >= the required duration,
then calculates crop frames based on the specified crop mode.

Matching Strategy:
1. Find shortest clip >= guide segment duration
2. Apply reuse policy to control clip repetition
3. Calculate crop frames (start, middle, end)
4. Handle rest segments with black frames

Usage:
    python duration_matcher.py --guide guide.json --source database.json --crop-mode middle
"""

import argparse
import bisect
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from edl_generator import EDLGenerator


class DurationMatcher:
    """Matches guide segments to source clips by duration."""

    def __init__(self,
                 guide_path: str,
                 duration_db_path: str,
                 reuse_policy: str = 'min_gap',
                 min_reuse_gap: int = 5,
                 max_reuses: int = 3,
                 reuse_percentage: float = 0.3,
                 crop_mode: str = 'middle',
                 prefer_closest_duration: bool = True,
                 match_rests: bool = False):
        """
        Initialize the duration matcher.

        Args:
            guide_path: Path to guide sequence JSON
            duration_db_path: Path to duration database JSON
            reuse_policy: How to handle reusing clips ('none', 'allow', 'min_gap', 'limited', 'percentage')
            min_reuse_gap: Minimum segments between reuses (for 'min_gap' policy)
            max_reuses: Maximum times each clip can be reused (for 'limited' policy)
            reuse_percentage: Maximum percentage of segments that can be reuses (for 'percentage' policy)
            crop_mode: How to crop clips ('start', 'middle', 'end')
            prefer_closest_duration: Prefer clips closest to target duration (vs just shortest valid)
            match_rests: Match rest segments with video clips (vs black frames)
        """
        self.guide_path = Path(guide_path)
        self.duration_db_path = Path(duration_db_path)

        # Reuse policy settings
        self.reuse_policy = reuse_policy
        self.min_reuse_gap = min_reuse_gap
        self.max_reuses = max_reuses
        self.reuse_percentage = reuse_percentage

        # Crop settings
        self.crop_mode = crop_mode
        self.prefer_closest_duration = prefer_closest_duration
        self.match_rests = match_rests

        # Data
        self.guide_sequence = None
        self.duration_database = None
        self.clips = []
        self.sorted_durations = []  # For binary search
        self.duration_to_clips = {}  # duration -> list of clip_ids

        # Match state
        self.matches = []
        self.clip_usage = defaultdict(int)  # clip_id -> usage count
        self.last_used_position = {}  # clip_id -> last position used
        self.unmatched_segments = []

    def load_guide_sequence(self):
        """Load guide sequence from JSON (uses duration, ignores pitch info)."""
        print(f"Loading guide sequence from: {self.guide_path}")
        with open(self.guide_path, 'r') as f:
            data = json.load(f)

        self.guide_sequence = data['pitch_segments']
        print(f"  Loaded {len(self.guide_sequence)} guide segments")

        # Calculate duration statistics
        if self.guide_sequence:
            durations = [seg['duration'] for seg in self.guide_sequence]
            print(f"  Duration range: {min(durations):.3f}s - {max(durations):.3f}s")
            print(f"  Total guide duration: {sum(durations):.1f}s")

            # Count rest segments
            rest_count = sum(1 for seg in self.guide_sequence if seg.get('is_rest', False))
            if rest_count > 0:
                print(f"  Rest segments: {rest_count}")

    def load_duration_database(self):
        """Load duration database from JSON."""
        print(f"\nLoading duration database from: {self.duration_db_path}")
        with open(self.duration_db_path, 'r') as f:
            data = json.load(f)

        self.duration_database = data
        self.clips = data['clips']
        print(f"  Loaded {len(self.clips)} source clips")
        print(f"  Duration range: {data['duration_range']['min']:.3f}s - {data['duration_range']['max']:.3f}s")

        # Build duration lookup structures
        self._build_duration_index()

    def _build_duration_index(self):
        """Build sorted duration list and duration-to-clips mapping."""
        # Get sorted clip IDs from database
        sorted_ids = self.duration_database['duration_index']['sorted_by_duration']

        # Build sorted duration list (for binary search)
        self.sorted_durations = []
        for clip_id in sorted_ids:
            clip = self.clips[clip_id]
            self.sorted_durations.append((clip['duration'], clip_id))

        # Build duration -> clips mapping (for finding all clips at a duration)
        self.duration_to_clips = defaultdict(list)
        for clip in self.clips:
            # Round duration to 3 decimal places for grouping
            key = round(clip['duration'], 3)
            self.duration_to_clips[key].append(clip['clip_id'])

    def can_use_clip(self, clip_id: int, current_position: int) -> bool:
        """
        Check if a clip can be used based on reuse policy.

        Args:
            clip_id: Source clip ID
            current_position: Current position in guide sequence

        Returns:
            True if clip can be used
        """
        usage_count = self.clip_usage[clip_id]

        if self.reuse_policy == 'allow':
            return True
        elif self.reuse_policy == 'none':
            return usage_count == 0
        elif self.reuse_policy == 'min_gap':
            if usage_count == 0:
                return True
            last_pos = self.last_used_position.get(clip_id, -999)
            return (current_position - last_pos) >= self.min_reuse_gap
        elif self.reuse_policy == 'limited':
            return usage_count < self.max_reuses
        elif self.reuse_policy == 'percentage':
            if not self.matches:
                return True
            reused_count = sum(1 for count in self.clip_usage.values() if count > 1)
            return (reused_count / len(self.matches)) < self.reuse_percentage
        else:
            return True

    def find_matching_clip(self, target_duration: float, position: int) -> Optional[Dict]:
        """
        Find the best source clip for a target duration.

        Args:
            target_duration: Desired duration in seconds
            position: Current position in guide sequence

        Returns:
            Best matching clip dict, or None if no valid clip found
        """
        # Binary search to find first clip >= target_duration
        durations_only = [d for d, _ in self.sorted_durations]
        insert_idx = bisect.bisect_left(durations_only, target_duration)

        # Check all clips from insert_idx onwards
        candidates = []
        for i in range(insert_idx, len(self.sorted_durations)):
            duration, clip_id = self.sorted_durations[i]

            # Verify duration is actually >= target (handle edge cases)
            if duration < target_duration:
                continue

            # Check reuse policy
            if not self.can_use_clip(clip_id, position):
                continue

            clip = self.clips[clip_id]
            excess = duration - target_duration

            candidates.append((excess, clip_id, clip))

            # If we found a clip and don't prefer closest, stop
            if not self.prefer_closest_duration and candidates:
                break

            # If prefer_closest, keep searching but limit to reasonable range
            # (clips more than 10x target duration are probably too long)
            if self.prefer_closest_duration and duration > target_duration * 10:
                break

        if not candidates:
            return None

        # Select best candidate
        if self.prefer_closest_duration:
            # Sort by excess duration (smallest first)
            candidates.sort(key=lambda x: x[0])

        # Return the best clip
        _, clip_id, clip = candidates[0]
        return clip

    def calculate_crop_frames(self, clip: Dict, target_duration: float) -> Tuple[int, int]:
        """
        Calculate start and end frames for cropping.

        Args:
            clip: Source clip metadata
            target_duration: Desired output duration

        Returns:
            Tuple of (start_frame, end_frame)
        """
        fps = clip['fps']
        total_frames = clip['total_frames']
        target_frames = int(round(target_duration * fps))

        # Ensure we don't exceed available frames
        target_frames = min(target_frames, total_frames)

        if self.crop_mode == 'start':
            # Use first N frames
            start_frame = 0
            end_frame = target_frames
        elif self.crop_mode == 'end':
            # Use last N frames
            start_frame = total_frames - target_frames
            end_frame = total_frames
        else:  # 'middle' (centered)
            # Trim equally from start and end
            trim_frames = total_frames - target_frames
            start_frame = trim_frames // 2
            end_frame = start_frame + target_frames

        return (start_frame, end_frame)

    def match_guide_to_source(self):
        """Match all guide segments to source clips."""
        print(f"\nMatching guide to source clips...")
        print(f"  Reuse policy: {self.reuse_policy}")
        print(f"  Crop mode: {self.crop_mode}")
        print(f"  Match rests: {self.match_rests}")

        self.matches = []
        self.unmatched_segments = []

        for i, guide_seg in enumerate(self.guide_sequence):
            guide_duration = guide_seg['duration']
            is_rest = guide_seg.get('is_rest', False) or guide_seg.get('pitch_midi', 0) == -1

            match = {
                'guide_segment_id': i,
                'guide_start_time': guide_seg['start_time'],
                'guide_end_time': guide_seg['end_time'],
                'guide_duration': guide_duration,
                'is_rest': is_rest,
            }

            if is_rest and not self.match_rests:
                # Rest segment - will be handled as black frames
                match['match_type'] = 'rest'
                match['source_clips'] = []
            else:
                # Find matching clip
                clip = self.find_matching_clip(guide_duration, i)

                if clip is None:
                    # No valid clip found
                    match['match_type'] = 'unmatched'
                    match['source_clips'] = []
                    self.unmatched_segments.append({
                        'guide_segment_id': i,
                        'required_duration': guide_duration,
                        'reason': 'no_clip_long_enough'
                    })
                else:
                    # Calculate crop frames
                    start_frame, end_frame = self.calculate_crop_frames(clip, guide_duration)

                    match['match_type'] = 'duration'
                    match['source_clip_id'] = clip['clip_id']
                    match['source_video_path'] = clip['video_path']
                    match['source_duration'] = clip['duration']
                    match['crop_mode'] = self.crop_mode
                    match['source_clips'] = [{
                        'video_path': clip['video_path'],
                        'video_start_frame': start_frame,
                        'video_end_frame': end_frame,
                        'duration': guide_duration,
                        'crop_mode': self.crop_mode
                    }]

                    # Update usage tracking
                    clip_id = clip['clip_id']
                    self.clip_usage[clip_id] += 1
                    self.last_used_position[clip_id] = i

            self.matches.append(match)

        # Print summary
        matched = sum(1 for m in self.matches if m['match_type'] == 'duration' and not m.get('is_rest', False))
        matched_rests = sum(1 for m in self.matches if m['match_type'] == 'duration' and m.get('is_rest', False))
        unmatched = sum(1 for m in self.matches if m['match_type'] == 'unmatched')
        rest_black = sum(1 for m in self.matches if m['match_type'] == 'rest')

        print(f"\nMatching complete:")
        print(f"  Duration matches: {matched}")
        if matched_rests > 0:
            print(f"  Rests matched with clips: {matched_rests}")
        print(f"  Unmatched: {unmatched}")
        print(f"  Rest segments (black frames): {rest_black}")

        if unmatched > 0:
            print(f"\nWarning: {unmatched} segments could not be matched")
            for seg in self.unmatched_segments[:5]:  # Show first 5
                print(f"  Segment {seg['guide_segment_id']}: needed {seg['required_duration']:.3f}s")

    def save_match_plan(self, output_path: str):
        """Save match plan to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate statistics
        matched = sum(1 for m in self.matches if m['match_type'] == 'duration' and not m.get('is_rest', False))
        matched_rests = sum(1 for m in self.matches if m['match_type'] == 'duration' and m.get('is_rest', False))
        unmatched = sum(1 for m in self.matches if m['match_type'] == 'unmatched')
        rest = sum(1 for m in self.matches if m['match_type'] == 'rest')

        unique_clips = len([c for c, count in self.clip_usage.items() if count > 0])
        reused_clips = sum(1 for count in self.clip_usage.values() if count > 1)
        max_reuse = max(self.clip_usage.values()) if self.clip_usage else 0

        total_duration = sum(m['guide_duration'] for m in self.matches)

        data = {
            'guide_sequence_path': str(self.guide_path.absolute()),
            'duration_database_path': str(self.duration_db_path.absolute()),
            'matching_config': {
                'reuse_policy': self.reuse_policy,
                'min_reuse_gap': self.min_reuse_gap,
                'max_reuses': self.max_reuses,
                'reuse_percentage': self.reuse_percentage,
                'crop_mode': self.crop_mode,
                'prefer_closest_duration': self.prefer_closest_duration,
                'match_rests': self.match_rests
            },
            'statistics': {
                'total_guide_segments': len(self.matches),
                'matched_segments': matched,
                'matched_rest_segments': matched_rests,
                'unmatched_segments': unmatched,
                'rest_segments_black_frames': rest,
                'unique_clips_used': unique_clips,
                'clips_reused': reused_clips,
                'max_reuse_count': max_reuse,
                'total_output_duration': round(total_duration, 3)
            },
            'unmatched_segments': self.unmatched_segments,
            'matches': self.matches
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved match plan to: {output_path}")

    def generate_edl(self, output_path: str, frame_rate: float = 24.0) -> str:
        """
        Generate EDL file from match plan.

        Args:
            output_path: Path for output EDL file
            frame_rate: Frame rate for timecode conversion

        Returns:
            Path to generated EDL file
        """
        print(f"\n=== Generating EDL ===")
        print(f"Frame rate: {frame_rate} fps")

        title = Path(output_path).stem
        edl = EDLGenerator(title, frame_rate=frame_rate)

        for match in self.matches:
            match_type = match.get('match_type', 'unknown')
            guide_duration = match.get('guide_duration', 0)

            if match_type == 'rest' or match_type == 'unmatched':
                # Black/rest segment
                comment = f"Guide segment {match.get('guide_segment_id', '?')}"
                if match.get('is_rest'):
                    comment += " (REST)"
                edl.add_black(guide_duration, comment=comment)
            else:
                # Video clip
                source_clips = match.get('source_clips', [])
                if source_clips:
                    clip = source_clips[0]
                    video_path = clip.get('video_path', '')

                    # Get source video fps from database or use frame_rate
                    source_fps = frame_rate
                    for db_clip in self.duration_db.get('clips', []):
                        if db_clip.get('path') == video_path:
                            source_fps = db_clip.get('fps', frame_rate)
                            break

                    # Calculate source timecode from frames
                    start_frame = clip.get('video_start_frame', 0)
                    clip_duration = clip.get('duration', guide_duration)
                    source_in = start_frame / source_fps
                    source_out = source_in + clip_duration

                    # Build comment
                    comment_parts = [f"Guide segment {match.get('guide_segment_id', '?')}"]
                    crop_mode = clip.get('crop_mode', match.get('crop_mode', ''))
                    if crop_mode:
                        comment_parts.append(f"Crop: {crop_mode}")

                    edl.add_event(
                        source_path=video_path,
                        source_in=source_in,
                        source_out=source_out,
                        comment=", ".join(comment_parts)
                    )

        edl_path = edl.write(output_path)
        print(f"EDL saved to: {edl_path}")
        print(f"  Events: {edl.event_count}")
        print(f"  Total duration: {edl.total_duration:.2f}s")

        return edl_path


def main():
    parser = argparse.ArgumentParser(
        description='Match guide segments to source clips by duration.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Crop Modes:
  start   - Use first N seconds of clip
  middle  - Trim equally from start/end (centered)
  end     - Use last N seconds of clip

Reuse Policies:
  none       - Each clip used at most once
  allow      - Unlimited reuse
  min_gap    - Can reuse after N segments gap
  limited    - Maximum K reuses per clip
  percentage - At most P%% of output can be reuses

Examples:
  %(prog)s --guide guide.json --source database.json
  %(prog)s --guide guide.json --source database.json --crop-mode start
  %(prog)s --guide guide.json --source database.json --reuse-policy allow
        """
    )

    parser.add_argument('--guide', required=True,
                        help='Guide sequence JSON path')
    parser.add_argument('--source', required=True,
                        help='Duration database JSON path')
    parser.add_argument('--output', default='data/segments/duration_match_plan.json',
                        help='Output match plan JSON (default: data/segments/duration_match_plan.json)')
    parser.add_argument('--crop-mode', choices=['start', 'middle', 'end'], default='middle',
                        help='How to crop clips (default: middle)')
    parser.add_argument('--reuse-policy', choices=['none', 'allow', 'min_gap', 'limited', 'percentage'],
                        default='min_gap', help='Clip reuse policy (default: min_gap)')
    parser.add_argument('--min-reuse-gap', type=int, default=5,
                        help='Minimum segments between reuses (default: 5)')
    parser.add_argument('--max-reuses', type=int, default=3,
                        help='Maximum reuses per clip (default: 3)')
    parser.add_argument('--reuse-percentage', type=float, default=0.3,
                        help='Maximum reuse percentage (default: 0.3)')
    parser.add_argument('--no-prefer-closest', action='store_true',
                        help='Use shortest valid clip instead of closest duration')
    parser.add_argument('--match-rests', action='store_true',
                        help='Match rest segments with video clips instead of black frames')
    parser.add_argument('--edl', action='store_true',
                        help='Generate EDL file alongside match plan')
    parser.add_argument('--edl-output', type=str, default=None,
                        help='Custom EDL output path (default: same as output with .edl extension)')
    parser.add_argument('--fps', type=float, default=24.0,
                        help='Frame rate for EDL timecode (default: 24.0)')

    args = parser.parse_args()

    # Validate files
    if not Path(args.guide).exists():
        print(f"Error: Guide file not found: {args.guide}")
        return 1
    if not Path(args.source).exists():
        print(f"Error: Source database not found: {args.source}")
        return 1

    # Create matcher and run
    matcher = DurationMatcher(
        args.guide,
        args.source,
        reuse_policy=args.reuse_policy,
        min_reuse_gap=args.min_reuse_gap,
        max_reuses=args.max_reuses,
        reuse_percentage=args.reuse_percentage,
        crop_mode=args.crop_mode,
        prefer_closest_duration=not args.no_prefer_closest,
        match_rests=args.match_rests
    )

    matcher.load_guide_sequence()
    matcher.load_duration_database()
    matcher.match_guide_to_source()
    matcher.save_match_plan(args.output)

    # Generate EDL if requested
    if args.edl:
        if args.edl_output:
            edl_path = args.edl_output
        else:
            edl_path = str(Path(args.output).with_suffix('.edl'))
        matcher.generate_edl(edl_path, frame_rate=args.fps)

    return 0


if __name__ == '__main__':
    exit(main())
