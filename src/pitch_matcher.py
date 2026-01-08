#!/usr/bin/env python3
"""
Pitch Matcher

Matches guide video pitch sequence to source video database.
Creates a match plan that can be used for video assembly.

Matching Strategy:
1. Exact MIDI note match (preferred)
2. Transposed match if no exact match available
3. Tracks missing pitches for database expansion
4. Handles duration via trimming, looping, or combining clips
5. Supports reuse policies to control clip repetition
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

from pitch_utils import midi_to_note_name


class PitchMatcher:
    """Matches guide pitch sequence to source database."""

    def __init__(self,
                 guide_path: str,
                 source_db_path: str,
                 reuse_policy: str = 'min_gap',
                 min_reuse_gap: int = 5,
                 max_reuses: int = 3,
                 reuse_percentage: float = 0.3,
                 duration_weight: float = 0.3,
                 confidence_weight: float = 0.4,
                 consistency_weight: float = 0.3,
                 allow_transposition: bool = True,
                 max_transposition_semitones: int = 12,
                 prefer_small_transposition: bool = True,
                 combine_clips_for_duration: bool = True):
        """
        Initialize the pitch matcher.

        Args:
            guide_path: Path to guide sequence JSON
            source_db_path: Path to source database JSON
            reuse_policy: How to handle reusing clips ('none', 'allow', 'min_gap', 'limited', 'percentage')
            min_reuse_gap: Minimum segments between reuses (for 'min_gap' policy)
            max_reuses: Maximum times each segment can be reused (for 'limited' policy)
            reuse_percentage: Maximum percentage of segments that can be reuses (for 'percentage' policy)
            duration_weight: Weight for duration matching in scoring (0-1)
            confidence_weight: Weight for confidence in scoring (0-1)
            consistency_weight: Weight for loopability/consistency in scoring (0-1)
            allow_transposition: Allow pitch transposition if no exact match
            max_transposition_semitones: Maximum semitones to transpose
            prefer_small_transposition: Prefer smaller transposition amounts
            combine_clips_for_duration: Combine multiple clips to meet duration (vs just looping one)
        """
        self.guide_path = Path(guide_path)
        self.source_db_path = Path(source_db_path)

        # Reuse policy settings
        self.reuse_policy = reuse_policy
        self.min_reuse_gap = min_reuse_gap
        self.max_reuses = max_reuses
        self.reuse_percentage = reuse_percentage

        # Scoring weights
        self.duration_weight = duration_weight
        self.confidence_weight = confidence_weight
        self.consistency_weight = consistency_weight

        # Transposition settings
        self.allow_transposition = allow_transposition
        self.max_transposition_semitones = max_transposition_semitones
        self.prefer_small_transposition = prefer_small_transposition

        # Duration handling
        self.combine_clips_for_duration = combine_clips_for_duration

        # Data
        self.guide_sequence = None
        self.source_database = None
        self.source_pitch_index = None

        # Match state
        self.matches = []
        self.segment_usage = defaultdict(int)  # segment_id -> usage count
        self.last_used_position = {}  # segment_id -> last position used
        self.missing_pitches = set()  # MIDI notes not in source database

    def load_guide_sequence(self):
        """Load guide sequence from JSON."""
        print(f"Loading guide sequence from: {self.guide_path}")
        with open(self.guide_path, 'r') as f:
            data = json.load(f)

        self.guide_sequence = data['pitch_segments']
        print(f"  Loaded {len(self.guide_sequence)} guide segments")

        # Calculate pitch range
        if len(self.guide_sequence) > 0:
            guide_midis = [seg['pitch_midi'] for seg in self.guide_sequence]
            print(f"  Guide pitch range: MIDI {min(guide_midis)} to {max(guide_midis)}")

    def load_source_database(self):
        """Load source database from JSON."""
        print(f"\nLoading source database from: {self.source_db_path}")
        with open(self.source_db_path, 'r') as f:
            data = json.load(f)

        self.source_database = data['pitch_database']

        # Convert pitch index keys from strings to ints
        self.source_pitch_index = {
            int(midi): segment_ids
            for midi, segment_ids in data['pitch_index'].items()
        }

        # Load silence segments and pre-sort by duration (longest first)
        # This avoids re-sorting on every find_silence_match call
        self.silence_segments = sorted(
            data.get('silence_segments', []),
            key=lambda x: x['duration'],
            reverse=True
        )

        print(f"  Loaded {len(self.source_database)} source segments")
        print(f"  Loaded {len(self.silence_segments)} silence segments")
        print(f"  Source videos: {data.get('num_videos', 1)}")
        print(f"  Unique pitches: {len(self.source_pitch_index)}")

        # Calculate pitch range
        if len(self.source_pitch_index) > 0:
            source_midis = list(self.source_pitch_index.keys())
            print(f"  Source pitch range: MIDI {min(source_midis)} to {max(source_midis)}")

    def score_segment(self, source_seg: Dict, target_duration: float) -> float:
        """
        Score a source segment for matching.

        Args:
            source_seg: Source segment dictionary
            target_duration: Desired duration from guide

        Returns:
            Score (0-1, higher is better)
        """
        # Duration score (1.0 if durations match, lower if different)
        duration_diff = abs(source_seg['duration'] - target_duration)
        max_duration = max(source_seg['duration'], target_duration)
        duration_score = 1.0 - (duration_diff / max_duration) if max_duration > 0 else 1.0

        # Confidence score (already 0-1)
        confidence_score = source_seg['pitch_confidence']

        # Consistency/loopability score (from database, fallback to confidence if not available)
        # Loopability combines RMS consistency and pitch consistency
        consistency_score = source_seg.get('loopability', confidence_score)

        # Combined score with three-way weighting
        score = (self.duration_weight * duration_score +
                 self.confidence_weight * confidence_score +
                 self.consistency_weight * consistency_score)

        return score

    def can_use_segment(self, segment_id: int, current_position: int) -> bool:
        """
        Check if a segment can be used based on reuse policy.

        Args:
            segment_id: Source segment ID
            current_position: Current position in guide sequence

        Returns:
            True if segment can be used
        """
        if self.reuse_policy == 'allow':
            return True

        if self.reuse_policy == 'none':
            return self.segment_usage[segment_id] == 0

        if self.reuse_policy == 'min_gap':
            if self.segment_usage[segment_id] == 0:
                return True
            last_pos = self.last_used_position.get(segment_id, -999)
            return (current_position - last_pos) >= self.min_reuse_gap

        if self.reuse_policy == 'limited':
            return self.segment_usage[segment_id] < self.max_reuses

        if self.reuse_policy == 'percentage':
            # Check if we're under the reuse percentage threshold
            total_matches = len(self.matches)
            reused_count = sum(1 for count in self.segment_usage.values() if count > 1)
            if total_matches == 0:
                return True
            return (reused_count / total_matches) < self.reuse_percentage

        return True

    def find_exact_match(self, guide_seg: Dict, position: int) -> Optional[Dict]:
        """
        Find exact pitch match from source database.

        Args:
            guide_seg: Guide segment dictionary
            position: Current position in guide sequence

        Returns:
            Best matching source segment or None
        """
        target_midi = guide_seg['pitch_midi']

        # Check if this pitch exists in source database
        if target_midi not in self.source_pitch_index:
            return None

        # Get all segments with this pitch
        candidate_ids = self.source_pitch_index[target_midi]

        # Filter by reuse policy
        available_ids = [
            seg_id for seg_id in candidate_ids
            if self.can_use_segment(seg_id, position)
        ]

        if not available_ids:
            return None

        # Find best scoring segment directly (O(n) instead of O(n log n) sort)
        target_duration = guide_seg['duration']
        candidates = [self.source_database[seg_id] for seg_id in available_ids]

        # Use max() with scoring function - more efficient than sorting all
        best_candidate = max(
            candidates,
            key=lambda seg: self.score_segment(seg, target_duration)
        )

        return best_candidate

    def find_transposed_match(self, guide_seg: Dict, position: int) -> Optional[Tuple[Dict, int]]:
        """
        Find pitch match via transposition.

        Args:
            guide_seg: Guide segment dictionary
            position: Current position in guide sequence

        Returns:
            Tuple of (best matching source segment, semitones to transpose) or None
        """
        if not self.allow_transposition:
            return None

        target_midi = guide_seg['pitch_midi']

        # Try transpositions in order of preference (smaller first if preferred)
        transposition_range = range(1, self.max_transposition_semitones + 1)

        if self.prefer_small_transposition:
            # Try ±1, ±2, ±3, etc.
            transpositions_to_try = []
            for offset in transposition_range:
                transpositions_to_try.extend([offset, -offset])
        else:
            # Try all positive, then all negative
            transpositions_to_try = list(transposition_range) + [-t for t in transposition_range]

        best_match = None
        best_score = -1
        best_transpose = 0

        target_duration = guide_seg['duration']

        for transpose in transpositions_to_try:
            source_midi = target_midi + transpose

            # Check if this transposed pitch exists
            if source_midi not in self.source_pitch_index:
                continue

            # Get available segments
            candidate_ids = self.source_pitch_index[source_midi]
            available_ids = [
                seg_id for seg_id in candidate_ids
                if self.can_use_segment(seg_id, position)
            ]

            if not available_ids:
                continue

            # Score candidates
            for seg_id in available_ids:
                seg = self.source_database[seg_id]
                score = self.score_segment(seg, target_duration)

                # Penalize larger transpositions slightly
                transposition_penalty = abs(transpose) * 0.01
                score = score * (1.0 - transposition_penalty)

                if score > best_score:
                    best_score = score
                    best_match = seg
                    best_transpose = transpose

        if best_match:
            return (best_match, best_transpose)
        return None

    def find_silence_match(self, target_duration: float) -> Optional[List[Dict]]:
        """
        Find silence segments from source database to fill a rest.

        Args:
            target_duration: Desired duration for the rest

        Returns:
            List of silence segments that together cover the target duration, or None
        """
        if not self.silence_segments:
            return None

        # silence_segments already pre-sorted by duration (longest first) during load
        selected = []
        remaining_duration = target_duration

        for silence in self.silence_segments:
            if remaining_duration <= 0:
                break

            # Use this silence segment
            use_duration = min(silence['duration'], remaining_duration)
            selected.append({
                'video_path': silence['video_path'],
                'video_start_frame': silence['video_start_frame'],
                'video_end_frame': silence['video_start_frame'] + int(
                    use_duration * (silence['video_end_frame'] - silence['video_start_frame']) / silence['duration']
                ) if silence['duration'] > 0 else silence['video_end_frame'],
                'duration': use_duration,
                'start_time': silence['start_time'],
                'end_time': silence['start_time'] + use_duration,
                'is_silence': True
            })
            remaining_duration -= use_duration

            # If we've filled the duration, we're done
            if remaining_duration <= 0.01:  # Small tolerance
                break

        # If we couldn't fill the duration, loop the longest segment
        if remaining_duration > 0.01 and selected:
            last_silence = self.silence_segments[0]  # Use longest (first in pre-sorted list)
            while remaining_duration > 0.01:
                use_duration = min(last_silence['duration'], remaining_duration)
                selected.append({
                    'video_path': last_silence['video_path'],
                    'video_start_frame': last_silence['video_start_frame'],
                    'video_end_frame': last_silence['video_start_frame'] + int(
                        use_duration * (last_silence['video_end_frame'] - last_silence['video_start_frame']) / last_silence['duration']
                    ) if last_silence['duration'] > 0 else last_silence['video_end_frame'],
                    'duration': use_duration,
                    'start_time': last_silence['start_time'],
                    'end_time': last_silence['start_time'] + use_duration,
                    'is_silence': True,
                    'looped': True
                })
                remaining_duration -= use_duration

        return selected if selected else None

    def handle_duration(self, source_segments: List[Dict], target_duration: float) -> List[Dict]:
        """
        Handle duration mismatch by trimming, looping, or combining clips.

        Args:
            source_segments: List of source segments (usually just one, or multiple for combining)
            target_duration: Target duration from guide

        Returns:
            List of clip instructions with timing info
        """
        clips = []

        if len(source_segments) == 0:
            return clips

        # If we have one segment
        if len(source_segments) == 1:
            seg = source_segments[0]

            if seg['duration'] >= target_duration:
                # Trim to fit
                num_frames = seg['video_end_frame'] - seg['video_start_frame']
                trimmed_frames = int(target_duration * num_frames / seg['duration'])
                clips.append({
                    'segment_id': seg['segment_id'],
                    'video_path': seg['video_path'],
                    'video_start_frame': seg['video_start_frame'],
                    'video_end_frame': seg['video_start_frame'] + trimmed_frames,
                    'duration': target_duration,
                    'trim': True,
                    'original_duration': seg['duration']
                })
            else:
                # Need to loop
                remaining_duration = target_duration
                loop_count = 0

                # Minimum clip duration (1 frame at 24fps = ~0.042s, use 0.04s)
                MIN_CLIP_DURATION = 0.04

                while remaining_duration > MIN_CLIP_DURATION:
                    clip_duration = min(seg['duration'], remaining_duration)
                    num_frames = seg['video_end_frame'] - seg['video_start_frame']
                    end_frame = seg['video_start_frame'] + int(clip_duration * num_frames / seg['duration'])

                    # Skip if this would create a zero-frame clip
                    if end_frame <= seg['video_start_frame']:
                        break

                    clips.append({
                        'segment_id': seg['segment_id'],
                        'video_path': seg['video_path'],
                        'video_start_frame': seg['video_start_frame'],
                        'video_end_frame': end_frame,
                        'duration': clip_duration,
                        'loop_iteration': loop_count,
                        'looped': True
                    })
                    remaining_duration -= clip_duration
                    loop_count += 1

        else:
            # Combine multiple segments
            remaining_duration = target_duration

            # Minimum clip duration (1 frame at 24fps = ~0.042s, use 0.04s)
            MIN_CLIP_DURATION = 0.04

            for seg in source_segments:
                if remaining_duration <= MIN_CLIP_DURATION:
                    break

                clip_duration = min(seg['duration'], remaining_duration)
                num_frames = seg['video_end_frame'] - seg['video_start_frame']
                end_frame = seg['video_start_frame'] + int(clip_duration * num_frames / seg['duration'])

                # Skip if this would create a zero-frame clip
                if end_frame <= seg['video_start_frame']:
                    continue

                clips.append({
                    'segment_id': seg['segment_id'],
                    'video_path': seg['video_path'],
                    'video_start_frame': seg['video_start_frame'],
                    'video_end_frame': end_frame,
                    'duration': clip_duration,
                    'combined': True
                })
                remaining_duration -= clip_duration

            # If still need more duration, loop the last segment
            if remaining_duration > MIN_CLIP_DURATION and len(source_segments) > 0:
                seg = source_segments[-1]
                loop_count = 1

                while remaining_duration > MIN_CLIP_DURATION:
                    clip_duration = min(seg['duration'], remaining_duration)
                    num_frames = seg['video_end_frame'] - seg['video_start_frame']
                    end_frame = seg['video_start_frame'] + int(clip_duration * num_frames / seg['duration'])

                    # Skip if this would create a zero-frame clip
                    if end_frame <= seg['video_start_frame']:
                        break

                    clips.append({
                        'segment_id': seg['segment_id'],
                        'video_path': seg['video_path'],
                        'video_start_frame': seg['video_start_frame'],
                        'video_end_frame': end_frame,
                        'duration': clip_duration,
                        'loop_iteration': loop_count,
                        'looped': True,
                        'combined': True
                    })
                    remaining_duration -= clip_duration
                    loop_count += 1

        return clips

    def match_guide_to_source(self):
        """Main matching logic - match each guide segment to source."""
        print("\n=== Matching Guide to Source ===\n")

        total_segments = len(self.guide_sequence)
        exact_matches = 0
        transposed_matches = 0
        missing_matches = 0

        rest_count = 0
        for i, guide_seg in enumerate(self.guide_sequence):
            if (i + 1) % 10 == 0:
                print(f"Processing segment {i+1}/{total_segments}...")

            # Check if this is a rest segment
            if guide_seg.get('is_rest', False) or guide_seg.get('pitch_midi', 0) == -1:
                # Find silence segments from source database
                silence_clips = self.find_silence_match(guide_seg['duration'])
                self.matches.append({
                    'guide_segment_id': i,
                    'guide_pitch_note': 'REST',
                    'guide_pitch_midi': -1,
                    'guide_start_time': guide_seg['start_time'],
                    'guide_end_time': guide_seg['end_time'],
                    'guide_duration': guide_seg['duration'],
                    'match_type': 'rest',
                    'source_clips': silence_clips if silence_clips else []
                })
                rest_count += 1
                continue

            # Try exact match first
            source_match = self.find_exact_match(guide_seg, i)

            if source_match:
                # Exact match found
                match_type = 'exact'
                transpose_semitones = 0
                exact_matches += 1
            else:
                # Try transposition
                transposed_result = self.find_transposed_match(guide_seg, i)

                if transposed_result:
                    source_match, transpose_semitones = transposed_result
                    match_type = 'transposed'
                    transposed_matches += 1
                else:
                    # No match found
                    self.missing_pitches.add(guide_seg['pitch_midi'])
                    match_type = 'missing'
                    transpose_semitones = 0
                    missing_matches += 1

                    # Create placeholder match
                    self.matches.append({
                        'guide_segment_id': i,
                        'guide_pitch_note': guide_seg['pitch_note'],
                        'guide_pitch_midi': guide_seg['pitch_midi'],
                        'guide_start_time': guide_seg['start_time'],
                        'guide_end_time': guide_seg['end_time'],
                        'guide_duration': guide_seg['duration'],
                        'match_type': 'missing',
                        'source_clips': []
                    })
                    continue

            # Handle duration
            clips = self.handle_duration([source_match], guide_seg['duration'])

            # Record match
            self.matches.append({
                'guide_segment_id': i,
                'guide_pitch_note': guide_seg['pitch_note'],
                'guide_pitch_midi': guide_seg['pitch_midi'],
                'guide_start_time': guide_seg['start_time'],
                'guide_end_time': guide_seg['end_time'],
                'guide_duration': guide_seg['duration'],
                'match_type': match_type,
                'transpose_semitones': transpose_semitones,
                'source_segment_id': source_match['segment_id'],
                'source_pitch_note': source_match['pitch_note'],
                'source_pitch_midi': source_match['pitch_midi'],
                'source_clips': clips
            })

            # Update usage tracking
            self.segment_usage[source_match['segment_id']] += 1
            self.last_used_position[source_match['segment_id']] = i

        print(f"\n=== Matching Complete ===")
        print(f"Total guide segments: {total_segments}")
        print(f"Exact matches: {exact_matches}")
        print(f"Transposed matches: {transposed_matches}")
        print(f"Missing matches: {missing_matches}")
        print(f"Rest segments: {rest_count}")

        if self.missing_pitches:
            print(f"\nMissing pitches (not in source database):")
            missing_notes = sorted(self.missing_pitches)
            for midi in missing_notes:
                print(f"  {midi_to_note_name(midi)} (MIDI {midi})")

        # Reuse statistics
        reused_segments = sum(1 for count in self.segment_usage.values() if count > 1)
        max_reuse = max(self.segment_usage.values()) if self.segment_usage else 0

        print(f"\nReuse statistics:")
        print(f"  Unique source segments used: {len(self.segment_usage)}")
        print(f"  Segments reused: {reused_segments}")
        print(f"  Maximum reuse count: {max_reuse}")

    def save_match_plan(self, output_path: str):
        """
        Save match plan to JSON file.

        Args:
            output_path: Output JSON file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert missing pitches set to list for JSON
        missing_pitches_list = [
            {'midi': midi, 'note': midi_to_note_name(midi)}
            for midi in sorted(self.missing_pitches)
        ]

        # Calculate statistics
        exact_count = sum(1 for m in self.matches if m['match_type'] == 'exact')
        transposed_count = sum(1 for m in self.matches if m['match_type'] == 'transposed')
        missing_count = sum(1 for m in self.matches if m['match_type'] == 'missing')
        rest_count = sum(1 for m in self.matches if m['match_type'] == 'rest')

        reused_segments = sum(1 for count in self.segment_usage.values() if count > 1)

        data = {
            'guide_sequence_path': str(self.guide_path),
            'source_database_path': str(self.source_db_path),
            'matching_config': {
                'reuse_policy': self.reuse_policy,
                'duration_weight': self.duration_weight,
                'confidence_weight': self.confidence_weight,
                'consistency_weight': self.consistency_weight,
                'allow_transposition': self.allow_transposition,
                'max_transposition_semitones': self.max_transposition_semitones,
                'combine_clips_for_duration': self.combine_clips_for_duration
            },
            'statistics': {
                'total_guide_segments': len(self.guide_sequence),
                'exact_matches': exact_count,
                'transposed_matches': transposed_count,
                'missing_matches': missing_count,
                'rest_segments': rest_count,
                'unique_source_segments_used': len(self.segment_usage),
                'segments_reused': reused_segments,
                'max_reuse_count': max(self.segment_usage.values()) if self.segment_usage else 0
            },
            'missing_pitches': missing_pitches_list,
            'matches': self.matches
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nMatch plan saved to: {output_path}")
        print(f"Total matches: {len(self.matches)}")


def main():
    parser = argparse.ArgumentParser(
        description="Match guide pitch sequence to source video database"
    )
    parser.add_argument(
        '--guide',
        type=str,
        required=True,
        help='Path to guide sequence JSON'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to source database JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/segments/match_plan.json',
        help='Output match plan JSON path'
    )
    parser.add_argument(
        '--reuse-policy',
        type=str,
        default='min_gap',
        choices=['none', 'allow', 'min_gap', 'limited', 'percentage'],
        help='Reuse policy for source segments (default: min_gap)'
    )
    parser.add_argument(
        '--min-reuse-gap',
        type=int,
        default=5,
        help='Minimum segments between reuses for min_gap policy (default: 5)'
    )
    parser.add_argument(
        '--max-reuses',
        type=int,
        default=3,
        help='Maximum reuses per segment for limited policy (default: 3)'
    )
    parser.add_argument(
        '--reuse-percentage',
        type=float,
        default=0.3,
        help='Maximum reuse percentage for percentage policy (default: 0.3)'
    )
    parser.add_argument(
        '--duration-weight',
        type=float,
        default=0.3,
        help='Weight for duration matching (default: 0.3)'
    )
    parser.add_argument(
        '--confidence-weight',
        type=float,
        default=0.4,
        help='Weight for pitch confidence matching (default: 0.4)'
    )
    parser.add_argument(
        '--consistency-weight',
        type=float,
        default=0.3,
        help='Weight for loopability/consistency matching (default: 0.3)'
    )
    parser.add_argument(
        '--no-transposition',
        action='store_true',
        help='Disable pitch transposition (only exact matches)'
    )
    parser.add_argument(
        '--max-transpose',
        type=int,
        default=12,
        help='Maximum semitones to transpose (default: 12)'
    )
    parser.add_argument(
        '--no-combine-clips',
        action='store_true',
        help='Disable combining multiple clips for duration (only loop single clips)'
    )

    args = parser.parse_args()

    print("=== Pitch Matcher ===\n")
    print(f"Guide sequence: {args.guide}")
    print(f"Source database: {args.source}")
    print(f"Output: {args.output}")
    print(f"Reuse policy: {args.reuse_policy}")

    # Initialize matcher
    matcher = PitchMatcher(
        guide_path=args.guide,
        source_db_path=args.source,
        reuse_policy=args.reuse_policy,
        min_reuse_gap=args.min_reuse_gap,
        max_reuses=args.max_reuses,
        reuse_percentage=args.reuse_percentage,
        duration_weight=args.duration_weight,
        confidence_weight=args.confidence_weight,
        consistency_weight=args.consistency_weight,
        allow_transposition=not args.no_transposition,
        max_transposition_semitones=args.max_transpose,
        prefer_small_transposition=True,
        combine_clips_for_duration=not args.no_combine_clips
    )

    # Load data
    matcher.load_guide_sequence()
    matcher.load_source_database()

    # Perform matching
    matcher.match_guide_to_source()

    # Save match plan
    matcher.save_match_plan(args.output)

    print("\n=== Matching Complete ===")


if __name__ == "__main__":
    main()
