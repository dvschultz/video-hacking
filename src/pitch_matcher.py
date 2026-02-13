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
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from pitch_utils import midi_to_note_name

# Minimum clip duration (1 frame at 24fps = ~0.042s)
MIN_CLIP_DURATION = 0.04


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
                 combine_clips_for_duration: bool = True,
                 min_volume_db: float = None,
                 one_video_per_note: bool = False):
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
            min_volume_db: Minimum RMS volume in dB (e.g., -40). Segments quieter than this are excluded.
            one_video_per_note: Lock each unique guide note to one source video file for visual variety.
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

        # Volume filtering
        self.min_volume_db = min_volume_db

        # One-video-per-note
        self.one_video_per_note = one_video_per_note
        self.note_video_assignments = {}    # midi -> video_path
        self.video_note_assignments = {}    # video_path -> midi
        self.unassigned_notes = set()       # notes that couldn't get a unique video

        # Data
        self.guide_sequence = None
        self.source_database = None
        self.source_pitch_index = None

        # Match state
        self.matches = []
        self.segment_usage = defaultdict(int)  # segment_id -> usage count
        self.last_used_position = {}  # segment_id -> last position used
        self.missing_pitches = set()  # MIDI notes not in source database
        self.volume_filtered_count = 0  # Segments excluded by volume filter

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

        # Build video path -> fps mapping for EDL generation
        self.video_fps_map = {}
        for video_info in data.get('source_videos', []):
            video_path = video_info.get('video_path', '')
            fps = video_info.get('fps', 24.0)
            if video_path:
                self.video_fps_map[video_path] = fps

        # Build video-level pitch index: video_path -> {midi -> [segment_ids]}
        self.video_pitch_index = defaultdict(lambda: defaultdict(list))
        for seg in self.source_database:
            vpath = seg.get('video_path', '')
            midi = seg['pitch_midi']
            self.video_pitch_index[vpath][midi].append(seg['segment_id'])

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
        usage_count = self.segment_usage[segment_id]

        if self.reuse_policy == 'allow':
            return True
        elif self.reuse_policy == 'none':
            return usage_count == 0
        elif self.reuse_policy == 'min_gap':
            if usage_count == 0:
                return True
            last_pos = self.last_used_position.get(segment_id, -999)
            return (current_position - last_pos) >= self.min_reuse_gap
        elif self.reuse_policy == 'limited':
            return usage_count < self.max_reuses
        elif self.reuse_policy == 'percentage':
            if not self.matches:
                return True
            reused_count = sum(1 for count in self.segment_usage.values() if count > 1)
            return (reused_count / len(self.matches)) < self.reuse_percentage
        else:
            return True

    def _get_available_segments(self, candidate_ids: List[int], position: int,
                               target_midi: int = None) -> List[int]:
        """
        Filter segment IDs by video assignment, volume, and reuse policy.

        Args:
            candidate_ids: List of candidate segment IDs
            position: Current position in guide sequence
            target_midi: Guide note MIDI number (for one-video-per-note filtering)

        Returns:
            List of available segment IDs after filtering
        """
        # Filter by one-video-per-note assignment
        if (self.one_video_per_note and target_midi is not None
                and target_midi in self.note_video_assignments
                and target_midi not in self.unassigned_notes):
            assigned_video = self.note_video_assignments[target_midi]
            candidate_ids = [
                seg_id for seg_id in candidate_ids
                if self.source_database[seg_id].get('video_path') == assigned_video
            ]

        initial_count = len(candidate_ids)

        # Filter by minimum volume if specified
        if self.min_volume_db is not None:
            candidate_ids = [
                seg_id for seg_id in candidate_ids
                if self.source_database[seg_id].get('rms_db', -float('inf')) >= self.min_volume_db
            ]
            self.volume_filtered_count += initial_count - len(candidate_ids)

        # Filter by reuse policy
        return [
            seg_id for seg_id in candidate_ids
            if self.can_use_segment(seg_id, position)
        ]

    def _score_video_for_note(self, video_path: str, target_midi: int,
                               total_guide_duration: float) -> float:
        """
        Score how well a video can serve a particular guide note.

        Args:
            video_path: Path to the source video
            target_midi: MIDI note number from the guide
            total_guide_duration: Total duration needed for this note across all guide segments

        Returns:
            Score (0-1, higher is better)
        """
        video_pitches = self.video_pitch_index.get(video_path, {})

        # Check exact pitch segments
        exact_seg_ids = video_pitches.get(target_midi, [])

        # Filter by volume if needed
        if self.min_volume_db is not None:
            exact_seg_ids = [
                sid for sid in exact_seg_ids
                if self.source_database[sid].get('rms_db', -float('inf')) >= self.min_volume_db
            ]

        # Check nearby pitches (transposable) within max_transposition_semitones
        nearby_seg_ids = []
        if not exact_seg_ids and self.allow_transposition:
            for offset in range(1, self.max_transposition_semitones + 1):
                for delta in [offset, -offset]:
                    nearby_midi = target_midi + delta
                    candidates = video_pitches.get(nearby_midi, [])
                    if self.min_volume_db is not None:
                        candidates = [
                            sid for sid in candidates
                            if self.source_database[sid].get('rms_db', -float('inf')) >= self.min_volume_db
                        ]
                    nearby_seg_ids.extend(candidates)

        usable_seg_ids = exact_seg_ids or nearby_seg_ids
        if not usable_seg_ids:
            return 0.0

        # Coverage score: total usable duration vs guide needs
        total_usable_duration = sum(
            self.source_database[sid]['duration'] for sid in usable_seg_ids
        )
        coverage = min(1.0, total_usable_duration / total_guide_duration) if total_guide_duration > 0 else 1.0

        # Quality score: average confidence and loopability
        confidences = [self.source_database[sid]['pitch_confidence'] for sid in usable_seg_ids]
        loopabilities = [self.source_database[sid].get('loopability', 0.5) for sid in usable_seg_ids]
        quality = (np.mean(confidences) + np.mean(loopabilities)) / 2.0

        # Quantity score: more segments = more variety
        quantity = min(1.0, len(usable_seg_ids) / 10.0)

        # Transposition closeness: 1.0 for exact, penalized for nearby
        if exact_seg_ids:
            transposition_closeness = 1.0
        else:
            # Find minimum transposition distance needed
            min_distance = float('inf')
            for sid in nearby_seg_ids:
                dist = abs(self.source_database[sid]['pitch_midi'] - target_midi)
                min_distance = min(min_distance, dist)
            transposition_closeness = max(0.0, 1.0 - min_distance * 0.05)

        score = (0.4 * coverage +
                 0.3 * quality +
                 0.2 * quantity +
                 0.1 * transposition_closeness)

        return score

    def _assign_videos_to_notes(self):
        """
        Pre-assign one video per unique guide note for visual variety.
        Called at the start of match_guide_to_source() when one_video_per_note is enabled.
        """
        print("\n=== One-Video-Per-Note Assignment ===\n")

        # Step 1: Collect unique non-REST guide notes with their duration needs and occurrence counts
        note_info = defaultdict(lambda: {'duration': 0.0, 'count': 0})
        for seg in self.guide_sequence:
            if seg.get('is_rest', False) or seg.get('pitch_midi', 0) == -1:
                continue
            midi = seg['pitch_midi']
            note_info[midi]['duration'] += seg['duration']
            note_info[midi]['count'] += 1

        unique_notes = list(note_info.keys())
        available_videos = list(self.video_pitch_index.keys())

        print(f"  Unique guide notes: {len(unique_notes)}")
        print(f"  Available source videos: {len(available_videos)}")

        if not unique_notes or not available_videos:
            print("  Skipping assignment: no notes or no videos")
            return

        # Step 2: Score all (note, video) pairs
        scores = {}  # (midi, video_path) -> score
        note_candidates = defaultdict(list)  # midi -> [(score, video_path)]

        for midi in unique_notes:
            total_dur = note_info[midi]['duration']
            for vpath in available_videos:
                score = self._score_video_for_note(vpath, midi, total_dur)
                if score > 0:
                    scores[(midi, vpath)] = score
                    note_candidates[midi].append((score, vpath))

        # Step 3: Sort notes by difficulty (fewest candidate videos first)
        notes_by_difficulty = sorted(unique_notes, key=lambda m: len(note_candidates.get(m, [])))

        # Step 4: Greedy assign - pick highest-scoring unassigned video for each note
        assigned_videos = set()
        self.note_video_assignments = {}
        self.video_note_assignments = {}
        assignment_scores = {}

        for midi in notes_by_difficulty:
            candidates = sorted(note_candidates.get(midi, []), key=lambda x: -x[0])
            assigned = False
            for score, vpath in candidates:
                if vpath not in assigned_videos:
                    self.note_video_assignments[midi] = vpath
                    self.video_note_assignments[vpath] = midi
                    assigned_videos.add(vpath)
                    assignment_scores[midi] = score
                    assigned = True
                    break
            if not assigned:
                self.unassigned_notes.add(midi)

        # Step 5: Second pass for unassigned notes (more notes than videos)
        if self.unassigned_notes:
            unassigned_by_rarity = sorted(
                self.unassigned_notes,
                key=lambda m: note_info[m]['count']
            )
            for midi in unassigned_by_rarity:
                candidates = sorted(note_candidates.get(midi, []), key=lambda x: -x[0])
                if candidates:
                    best_score, best_vpath = candidates[0]
                    self.note_video_assignments[midi] = best_vpath
                    assignment_scores[midi] = best_score
                    # Don't remove from unassigned_notes - keeps track that this is a shared assignment

        # Print assignment summary
        print(f"\n  {'Note':<8} {'MIDI':<6} {'Video':<60} {'Score':<8} {'Shared'}")
        print(f"  {'-'*8} {'-'*6} {'-'*60} {'-'*8} {'-'*6}")
        for midi in sorted(self.note_video_assignments.keys()):
            vpath = self.note_video_assignments[midi]
            score = assignment_scores.get(midi, 0.0)
            shared = "YES" if midi in self.unassigned_notes else ""
            note_name = midi_to_note_name(midi)
            video_name = Path(vpath).name if vpath else 'N/A'
            print(f"  {note_name:<8} {midi:<6} {video_name:<60} {score:<8.3f} {shared}")

        assigned_unique = len(self.note_video_assignments) - len(self.unassigned_notes)
        print(f"\n  Unique assignments: {assigned_unique}/{len(unique_notes)} notes")
        if self.unassigned_notes:
            print(f"  Shared assignments: {len(self.unassigned_notes)} notes (more notes than videos)")

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

        # Get all segments with this pitch and filter by video/volume/reuse policy
        candidate_ids = self.source_pitch_index[target_midi]
        available_ids = self._get_available_segments(
            candidate_ids, position, target_midi=guide_seg['pitch_midi']
        )

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

    def _get_transposition_order(self) -> List[int]:
        """Get transposition offsets in order of preference."""
        offsets = range(1, self.max_transposition_semitones + 1)

        if self.prefer_small_transposition:
            # Interleave positive and negative: ±1, ±2, ±3, etc.
            result = []
            for offset in offsets:
                result.extend([offset, -offset])
            return result

        # All positive, then all negative
        return list(offsets) + [-t for t in offsets]

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
        target_duration = guide_seg['duration']

        best_match = None
        best_score = -1
        best_transpose = 0

        for transpose in self._get_transposition_order():
            source_midi = target_midi + transpose

            if source_midi not in self.source_pitch_index:
                continue

            candidate_ids = self.source_pitch_index[source_midi]
            # Pass guide's target_midi so video filter applies to the guide note's assigned video
            available_ids = self._get_available_segments(
                candidate_ids, position, target_midi=target_midi
            )

            if not available_ids:
                continue

            # Penalize larger transpositions
            transposition_penalty = abs(transpose) * 0.01

            for seg_id in available_ids:
                seg = self.source_database[seg_id]
                score = self.score_segment(seg, target_duration) * (1.0 - transposition_penalty)

                if score > best_score:
                    best_score = score
                    best_match = seg
                    best_transpose = transpose

        if best_match:
            return (best_match, best_transpose)
        return None

    def _create_silence_clip(self, silence: Dict, use_duration: float, looped: bool = False) -> Dict:
        """Create a silence clip dictionary with optional trimming."""
        num_frames = silence['video_end_frame'] - silence['video_start_frame']
        if silence['duration'] > 0:
            end_frame = silence['video_start_frame'] + int(use_duration * num_frames / silence['duration'])
        else:
            end_frame = silence['video_end_frame']

        clip = {
            'video_path': silence['video_path'],
            'video_start_frame': silence['video_start_frame'],
            'video_end_frame': end_frame,
            'duration': use_duration,
            'start_time': silence['start_time'],
            'end_time': silence['start_time'] + use_duration,
            'is_silence': True
        }
        if looped:
            clip['looped'] = True
        return clip

    def find_silence_match(self, target_duration: float) -> Optional[List[Dict]]:
        """
        Find silence segments from source database to fill a rest.
        Uses random selection, preferring clips >= target duration.

        Args:
            target_duration: Desired duration for the rest

        Returns:
            List of silence segments that together cover the target duration, or None
        """
        if not self.silence_segments:
            return None

        selected = []
        remaining_duration = target_duration

        # Create a shuffled copy to randomize selection
        available = self.silence_segments.copy()
        random.shuffle(available)

        while remaining_duration > 0.01 and available:
            # Try to find clips >= remaining duration (can be trimmed to exact fit)
            long_enough = [s for s in available if s['duration'] >= remaining_duration]

            if long_enough:
                silence = random.choice(long_enough)
                available.remove(silence)
                selected.append(self._create_silence_clip(silence, remaining_duration))
                remaining_duration = 0
            else:
                silence = available.pop(0)
                selected.append(self._create_silence_clip(silence, silence['duration']))
                remaining_duration -= silence['duration']

        # If we still need more duration, loop a random silence segment
        if remaining_duration > 0.01 and selected:
            loop_silence = random.choice(self.silence_segments)
            while remaining_duration > 0.01:
                use_duration = min(loop_silence['duration'], remaining_duration)
                selected.append(self._create_silence_clip(loop_silence, use_duration, looped=True))
                remaining_duration -= use_duration

        return selected if selected else None

    def _create_clip(self, seg: Dict, clip_duration: float, **extra_fields) -> Optional[Dict]:
        """Create a clip dictionary from a segment with the given duration."""
        num_frames = seg['video_end_frame'] - seg['video_start_frame']
        end_frame = seg['video_start_frame'] + int(clip_duration * num_frames / seg['duration'])

        # Return None if this would create a zero-frame clip
        if end_frame <= seg['video_start_frame']:
            return None

        clip = {
            'segment_id': seg['segment_id'],
            'video_path': seg['video_path'],
            'video_start_frame': seg['video_start_frame'],
            'video_end_frame': end_frame,
            'duration': clip_duration,
            **extra_fields
        }
        return clip

    def _loop_segment(self, seg: Dict, remaining_duration: float, start_loop_count: int = 0,
                      combined: bool = False) -> Tuple[List[Dict], float]:
        """Loop a segment to fill remaining duration. Returns (clips, remaining_duration)."""
        clips = []
        loop_count = start_loop_count

        while remaining_duration > MIN_CLIP_DURATION:
            clip_duration = min(seg['duration'], remaining_duration)
            extra = {'loop_iteration': loop_count, 'looped': True}
            if combined:
                extra['combined'] = True

            clip = self._create_clip(seg, clip_duration, **extra)
            if clip is None:
                break

            clips.append(clip)
            remaining_duration -= clip_duration
            loop_count += 1

        return clips, remaining_duration

    def handle_duration(self, source_segments: List[Dict], target_duration: float) -> List[Dict]:
        """
        Handle duration mismatch by trimming, looping, or combining clips.

        Args:
            source_segments: List of source segments (usually just one, or multiple for combining)
            target_duration: Target duration from guide

        Returns:
            List of clip instructions with timing info
        """
        if not source_segments:
            return []

        clips = []

        # Single segment case
        if len(source_segments) == 1:
            seg = source_segments[0]

            if seg['duration'] >= target_duration:
                # Trim to fit
                clip = self._create_clip(seg, target_duration, trim=True, original_duration=seg['duration'])
                if clip:
                    clips.append(clip)
            else:
                # Need to loop
                loop_clips, _ = self._loop_segment(seg, target_duration)
                clips.extend(loop_clips)
        else:
            # Combine multiple segments
            remaining_duration = target_duration

            for seg in source_segments:
                if remaining_duration <= MIN_CLIP_DURATION:
                    break

                clip_duration = min(seg['duration'], remaining_duration)
                clip = self._create_clip(seg, clip_duration, combined=True)
                if clip:
                    clips.append(clip)
                    remaining_duration -= clip_duration

            # If still need more duration, loop the last segment
            if remaining_duration > MIN_CLIP_DURATION:
                loop_clips, _ = self._loop_segment(
                    source_segments[-1], remaining_duration, start_loop_count=1, combined=True
                )
                clips.extend(loop_clips)

        return clips

    def _build_guide_match(self, guide_seg: Dict, position: int, **extra_fields) -> Dict:
        """Build a match dictionary with common guide segment fields."""
        match = {
            'guide_segment_id': position,
            'guide_pitch_note': guide_seg.get('pitch_note', 'REST'),
            'guide_pitch_midi': guide_seg.get('pitch_midi', -1),
            'guide_start_time': guide_seg['start_time'],
            'guide_end_time': guide_seg['end_time'],
            'guide_duration': guide_seg['duration'],
            **extra_fields
        }
        return match

    def match_guide_to_source(self):
        """Main matching logic - match each guide segment to source."""
        # Run pre-assignment if one-video-per-note is enabled
        if self.one_video_per_note:
            self._assign_videos_to_notes()

        print("\n=== Matching Guide to Source ===\n")

        total_segments = len(self.guide_sequence)
        exact_matches = 0
        transposed_matches = 0
        missing_matches = 0
        rest_count = 0

        for i, guide_seg in enumerate(self.guide_sequence):
            if (i + 1) % 10 == 0:
                print(f"Processing segment {i+1}/{total_segments}...")

            # Handle rest segments
            if guide_seg.get('is_rest', False) or guide_seg.get('pitch_midi', 0) == -1:
                silence_clips = self.find_silence_match(guide_seg['duration'])
                self.matches.append(self._build_guide_match(
                    guide_seg, i,
                    guide_pitch_note='REST',
                    guide_pitch_midi=-1,
                    match_type='rest',
                    source_clips=silence_clips or []
                ))
                rest_count += 1
                continue

            # Try exact match first
            source_match = self.find_exact_match(guide_seg, i)
            transpose_semitones = 0

            if source_match:
                match_type = 'exact'
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
                    missing_matches += 1
                    self.matches.append(self._build_guide_match(
                        guide_seg, i,
                        match_type='missing',
                        source_clips=[]
                    ))
                    continue

            # Handle duration and record match
            clips = self.handle_duration([source_match], guide_seg['duration'])
            self.matches.append(self._build_guide_match(
                guide_seg, i,
                match_type=match_type,
                transpose_semitones=transpose_semitones,
                source_segment_id=source_match['segment_id'],
                source_pitch_note=source_match['pitch_note'],
                source_pitch_midi=source_match['pitch_midi'],
                source_rms_db=source_match.get('rms_db'),
                source_video_path=source_match.get('video_path'),
                source_clips=clips
            ))

            # Update usage tracking
            self.segment_usage[source_match['segment_id']] += 1
            self.last_used_position[source_match['segment_id']] = i

        # Print summary
        print(f"\n=== Matching Complete ===")
        print(f"Total guide segments: {total_segments}")
        print(f"Exact matches: {exact_matches}")
        print(f"Transposed matches: {transposed_matches}")
        print(f"Missing matches: {missing_matches}")
        print(f"Rest segments: {rest_count}")

        if self.missing_pitches:
            print(f"\nMissing pitches (not in source database):")
            for midi in sorted(self.missing_pitches):
                print(f"  {midi_to_note_name(midi)} (MIDI {midi})")

        # Reuse statistics
        reused_segments = sum(1 for count in self.segment_usage.values() if count > 1)
        max_reuse = max(self.segment_usage.values()) if self.segment_usage else 0

        print(f"\nReuse statistics:")
        print(f"  Unique source segments used: {len(self.segment_usage)}")
        print(f"  Segments reused: {reused_segments}")
        print(f"  Maximum reuse count: {max_reuse}")

        if self.min_volume_db is not None:
            print(f"\nVolume filtering:")
            print(f"  Minimum volume threshold: {self.min_volume_db} dB")
            print(f"  Segments filtered out: {self.volume_filtered_count}")

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
                'combine_clips_for_duration': self.combine_clips_for_duration,
                'min_volume_db': self.min_volume_db,
                'one_video_per_note': self.one_video_per_note
            },
            'statistics': {
                'total_guide_segments': len(self.guide_sequence),
                'exact_matches': exact_count,
                'transposed_matches': transposed_count,
                'missing_matches': missing_count,
                'rest_segments': rest_count,
                'unique_source_segments_used': len(self.segment_usage),
                'segments_reused': reused_segments,
                'max_reuse_count': max(self.segment_usage.values()) if self.segment_usage else 0,
                'segments_filtered_by_volume': self.volume_filtered_count
            },
            'missing_pitches': missing_pitches_list,
        }

        # Add video assignments when one-video-per-note is active
        if self.one_video_per_note and self.note_video_assignments:
            data['video_assignments'] = [
                {
                    'note': midi_to_note_name(midi),
                    'midi': midi,
                    'video_path': vpath,
                    'shared': midi in self.unassigned_notes
                }
                for midi, vpath in sorted(self.note_video_assignments.items())
            ]

        data['matches'] = self.matches

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nMatch plan saved to: {output_path}")
        print(f"Total matches: {len(self.matches)}")

    def generate_edl(self, output_path: str, frame_rate: float = 24.0) -> str:
        """
        Generate EDL file from match plan.

        Args:
            output_path: Path for output EDL file
            frame_rate: Frame rate for EDL timecode display

        Returns:
            Path to generated EDL file
        """
        from edl_generator import generate_edl_from_matches

        print(f"\n=== Generating EDL ===")
        print(f"EDL frame rate: {frame_rate} fps")

        return generate_edl_from_matches(
            matches=self.matches,
            output_path=output_path,
            title=Path(output_path).stem,
            frame_rate=frame_rate,
            video_fps_map=self.video_fps_map,
            verbose=True
        )


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
        '--min-volume-db',
        type=float,
        default=None,
        help='Minimum RMS volume in dB (e.g., -40). Segments quieter than this are excluded.'
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
    parser.add_argument(
        '--one-video-per-note',
        action='store_true',
        help='Lock each unique guide note to one source video for visual variety'
    )
    parser.add_argument(
        '--edl',
        action='store_true',
        help='Generate EDL file alongside match plan'
    )
    parser.add_argument(
        '--edl-output',
        type=str,
        default=None,
        help='Custom EDL output path (default: same as output with .edl extension)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=24.0,
        help='Frame rate for EDL timecode (default: 24.0)'
    )

    args = parser.parse_args()

    print("=== Pitch Matcher ===\n")
    print(f"Guide sequence: {args.guide}")
    print(f"Source database: {args.source}")
    print(f"Output: {args.output}")
    print(f"Reuse policy: {args.reuse_policy}")
    if args.min_volume_db is not None:
        print(f"Minimum volume: {args.min_volume_db} dB")
    if args.one_video_per_note:
        print(f"One-video-per-note: enabled")

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
        combine_clips_for_duration=not args.no_combine_clips,
        min_volume_db=args.min_volume_db,
        one_video_per_note=args.one_video_per_note
    )

    # Load data
    matcher.load_guide_sequence()
    matcher.load_source_database()

    # Perform matching
    matcher.match_guide_to_source()

    # Save match plan
    matcher.save_match_plan(args.output)

    # Generate EDL if requested
    if args.edl:
        if args.edl_output:
            edl_path = args.edl_output
        else:
            edl_path = str(Path(args.output).with_suffix('.edl'))
        matcher.generate_edl(edl_path, frame_rate=args.fps)



if __name__ == "__main__":
    main()
