"""
Unit tests for pitch_matcher.py - Pitch-based audio-video matching.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pitch_matcher import PitchMatcher, MIN_CLIP_DURATION


class TestPitchMatcherInitialization:
    """Test PitchMatcher initialization and data loading."""

    @pytest.fixture
    def matcher(self, guide_sequence_file, source_database_file):
        """Create PitchMatcher with test data files."""
        with patch('builtins.print'):
            matcher = PitchMatcher(
                str(guide_sequence_file),
                str(source_database_file),
                reuse_policy='allow'
            )
            matcher.load_guide_sequence()
            matcher.load_source_database()
        return matcher

    def test_load_guide_sequence(self, matcher):
        """Test loading guide sequence."""
        assert len(matcher.guide_sequence) == 10
        assert matcher.guide_sequence[0]['pitch_midi'] == 69

    def test_load_source_database(self, matcher):
        """Test loading source database."""
        assert len(matcher.source_database) == 20
        assert len(matcher.source_pitch_index) > 0

    def test_silence_segments_loaded(self, matcher):
        """Test that silence segments are loaded."""
        assert len(matcher.silence_segments) >= 1

    def test_pitch_index_correct(self, matcher):
        """Test pitch index maps MIDI to segment IDs correctly."""
        # Check that MIDI 60 (C4) exists and points to valid segments
        assert 60 in matcher.source_pitch_index
        for seg_id in matcher.source_pitch_index[60]:
            assert matcher.source_database[seg_id]['pitch_midi'] == 60


class TestPitchMatcherScoring:
    """Test segment scoring in PitchMatcher."""

    @pytest.fixture
    def matcher(self, guide_sequence_file, source_database_file):
        """Create PitchMatcher with test data."""
        with patch('builtins.print'):
            matcher = PitchMatcher(
                str(guide_sequence_file),
                str(source_database_file)
            )
            matcher.load_guide_sequence()
            matcher.load_source_database()
        return matcher

    def test_score_segment_basic(self, matcher):
        """Test basic segment scoring."""
        source_seg = {
            'duration': 0.5,
            'pitch_confidence': 0.9,
            'loopability': 0.8
        }
        score = matcher.score_segment(source_seg, target_duration=0.5)

        assert 0 <= score <= 1

    def test_score_segment_duration_match(self, matcher):
        """Test that duration match gives higher score."""
        source_seg = {
            'duration': 0.5,
            'pitch_confidence': 0.9,
            'loopability': 0.8
        }

        score_exact = matcher.score_segment(source_seg, target_duration=0.5)
        score_mismatch = matcher.score_segment(source_seg, target_duration=2.0)

        assert score_exact > score_mismatch

    def test_score_segment_confidence_impact(self, matcher):
        """Test that confidence affects score."""
        high_conf = {
            'duration': 0.5,
            'pitch_confidence': 0.95,
            'loopability': 0.8
        }
        low_conf = {
            'duration': 0.5,
            'pitch_confidence': 0.5,
            'loopability': 0.8
        }

        score_high = matcher.score_segment(high_conf, target_duration=0.5)
        score_low = matcher.score_segment(low_conf, target_duration=0.5)

        assert score_high > score_low

    def test_score_segment_loopability_impact(self, matcher):
        """Test that loopability affects score."""
        high_loop = {
            'duration': 0.5,
            'pitch_confidence': 0.9,
            'loopability': 0.95
        }
        low_loop = {
            'duration': 0.5,
            'pitch_confidence': 0.9,
            'loopability': 0.3
        }

        score_high = matcher.score_segment(high_loop, target_duration=0.5)
        score_low = matcher.score_segment(low_loop, target_duration=0.5)

        assert score_high > score_low


class TestPitchMatcherReusePolicies:
    """Test reuse policies in PitchMatcher."""

    @pytest.fixture
    def matcher(self, guide_sequence_file, source_database_file):
        """Create PitchMatcher with test data."""
        with patch('builtins.print'):
            matcher = PitchMatcher(
                str(guide_sequence_file),
                str(source_database_file)
            )
            matcher.load_guide_sequence()
            matcher.load_source_database()
        return matcher

    def test_can_use_segment_allow_policy(self, matcher):
        """Test 'allow' reuse policy."""
        matcher.reuse_policy = 'allow'
        matcher.segment_usage[0] = 10  # Used many times

        assert matcher.can_use_segment(0, 100) == True

    def test_can_use_segment_none_policy(self, matcher):
        """Test 'none' reuse policy."""
        matcher.reuse_policy = 'none'

        assert matcher.can_use_segment(0, 0) == True
        matcher.segment_usage[0] = 1
        assert matcher.can_use_segment(0, 1) == False

    def test_can_use_segment_min_gap_policy_allowed(self, matcher):
        """Test 'min_gap' policy when gap is sufficient."""
        matcher.reuse_policy = 'min_gap'
        matcher.min_reuse_gap = 5

        matcher.segment_usage[0] = 1
        matcher.last_used_position[0] = 0

        # Position 6 should be allowed (gap of 6 >= 5)
        assert matcher.can_use_segment(0, 6) == True

    def test_can_use_segment_min_gap_policy_denied(self, matcher):
        """Test 'min_gap' policy when gap is too small."""
        matcher.reuse_policy = 'min_gap'
        matcher.min_reuse_gap = 5

        matcher.segment_usage[0] = 1
        matcher.last_used_position[0] = 0

        # Position 3 should be denied (gap of 3 < 5)
        assert matcher.can_use_segment(0, 3) == False

    def test_can_use_segment_min_gap_first_use(self, matcher):
        """Test 'min_gap' policy for first use."""
        matcher.reuse_policy = 'min_gap'
        matcher.min_reuse_gap = 5

        # First use should always be allowed
        assert matcher.can_use_segment(0, 0) == True

    def test_can_use_segment_limited_policy(self, matcher):
        """Test 'limited' reuse policy."""
        matcher.reuse_policy = 'limited'
        matcher.max_reuses = 3

        assert matcher.can_use_segment(0, 0) == True
        matcher.segment_usage[0] = 2
        assert matcher.can_use_segment(0, 10) == True
        matcher.segment_usage[0] = 3
        assert matcher.can_use_segment(0, 20) == False

    def test_can_use_segment_percentage_policy(self, matcher):
        """Test 'percentage' reuse policy."""
        matcher.reuse_policy = 'percentage'
        matcher.reuse_percentage = 0.3

        # First use is always allowed
        assert matcher.can_use_segment(0, 0) == True


class TestPitchMatcherFindMatch:
    """Test finding matches in PitchMatcher."""

    @pytest.fixture
    def matcher(self, guide_sequence_file, source_database_file):
        """Create PitchMatcher with test data."""
        with patch('builtins.print'):
            matcher = PitchMatcher(
                str(guide_sequence_file),
                str(source_database_file),
                reuse_policy='allow'
            )
            matcher.load_guide_sequence()
            matcher.load_source_database()
        return matcher

    def test_find_exact_match_found(self, matcher):
        """Test finding exact pitch match."""
        guide_seg = {
            'pitch_midi': 60,  # C4 exists in fixture
            'duration': 0.3
        }

        match = matcher.find_exact_match(guide_seg, position=0)

        assert match is not None
        assert match['pitch_midi'] == 60

    def test_find_exact_match_not_found(self, matcher):
        """Test when no exact match exists."""
        guide_seg = {
            'pitch_midi': 100,  # Not in fixture range (60-71)
            'duration': 0.3
        }

        match = matcher.find_exact_match(guide_seg, position=0)
        assert match is None

    def test_find_transposed_match(self, matcher):
        """Test finding match via transposition."""
        # Request C5 (MIDI 72), which is not in database (only 60-71)
        guide_seg = {
            'pitch_midi': 72,
            'duration': 0.3
        }

        result = matcher.find_transposed_match(guide_seg, position=0)

        if result is not None:
            source_seg, transpose_semitones = result
            # Should find a transposition within max range
            assert abs(transpose_semitones) <= matcher.max_transposition_semitones
            # Transposed source should give target pitch
            assert source_seg['pitch_midi'] + transpose_semitones == 72 or \
                   source_seg['pitch_midi'] - transpose_semitones == 72

    def test_find_transposed_match_disabled(self, matcher):
        """Test transposition when disabled."""
        matcher.allow_transposition = False

        guide_seg = {
            'pitch_midi': 100,
            'duration': 0.3
        }

        result = matcher.find_transposed_match(guide_seg, position=0)
        assert result is None


class TestPitchMatcherDurationHandling:
    """Test duration handling in PitchMatcher."""

    @pytest.fixture
    def matcher(self, guide_sequence_file, source_database_file):
        """Create PitchMatcher with test data."""
        with patch('builtins.print'):
            matcher = PitchMatcher(
                str(guide_sequence_file),
                str(source_database_file)
            )
            matcher.load_guide_sequence()
            matcher.load_source_database()
        return matcher

    def test_handle_duration_trim(self, matcher):
        """Test duration handling when source is longer."""
        source_seg = {
            'segment_id': 0,
            'video_path': 'test.mp4',
            'video_start_frame': 0,
            'video_end_frame': 100,
            'duration': 1.0
        }

        clips = matcher.handle_duration([source_seg], target_duration=0.5)

        assert len(clips) == 1
        assert clips[0].get('trim') == True
        assert clips[0]['duration'] == 0.5

    def test_handle_duration_loop(self, matcher):
        """Test duration handling when source needs looping."""
        source_seg = {
            'segment_id': 0,
            'video_path': 'test.mp4',
            'video_start_frame': 0,
            'video_end_frame': 24,
            'duration': 0.2
        }

        clips = matcher.handle_duration([source_seg], target_duration=0.5)

        # Should create multiple clips to fill duration
        assert len(clips) >= 2
        total_duration = sum(c['duration'] for c in clips)
        assert total_duration >= 0.5 - MIN_CLIP_DURATION

    def test_handle_duration_exact_fit(self, matcher):
        """Test when source exactly matches target duration."""
        source_seg = {
            'segment_id': 0,
            'video_path': 'test.mp4',
            'video_start_frame': 0,
            'video_end_frame': 24,
            'duration': 0.5
        }

        clips = matcher.handle_duration([source_seg], target_duration=0.5)

        assert len(clips) == 1
        # Should be trimmed (technically) since duration matches
        assert clips[0]['duration'] == 0.5

    def test_handle_duration_empty_input(self, matcher):
        """Test with empty source segments."""
        clips = matcher.handle_duration([], target_duration=0.5)
        assert clips == []

    def test_handle_duration_combine_clips(self, matcher):
        """Test combining multiple segments."""
        source_segs = [
            {
                'segment_id': 0,
                'video_path': 'test.mp4',
                'video_start_frame': 0,
                'video_end_frame': 24,
                'duration': 0.3
            },
            {
                'segment_id': 1,
                'video_path': 'test.mp4',
                'video_start_frame': 24,
                'video_end_frame': 48,
                'duration': 0.3
            }
        ]

        clips = matcher.handle_duration(source_segs, target_duration=0.5)

        assert len(clips) >= 1
        total_duration = sum(c['duration'] for c in clips)
        assert total_duration >= 0.5 - MIN_CLIP_DURATION


class TestPitchMatcherSilenceHandling:
    """Test silence/rest segment handling."""

    @pytest.fixture
    def matcher(self, guide_sequence_file, source_database_file):
        """Create matcher with silence segments."""
        with patch('builtins.print'):
            matcher = PitchMatcher(
                str(guide_sequence_file),
                str(source_database_file)
            )
            matcher.load_guide_sequence()
            matcher.load_source_database()
        return matcher

    def test_find_silence_match_found(self, matcher):
        """Test finding silence segments for rest."""
        clips = matcher.find_silence_match(target_duration=0.5)

        assert clips is not None
        assert len(clips) >= 1
        assert clips[0].get('is_silence') == True

    def test_find_silence_match_total_duration(self, matcher):
        """Test silence clips cover target duration."""
        target = 0.5
        clips = matcher.find_silence_match(target_duration=target)

        if clips:
            total = sum(c['duration'] for c in clips)
            assert total >= target - 0.02  # Allow small tolerance


class TestPitchMatcherFullMatching:
    """Test full matching workflow."""

    @pytest.fixture
    def matcher(self, temp_dir, sample_guide_sequence_data, sample_source_database_data):
        """Create matcher with compatible guide and source data."""
        # Modify guide to use pitches that exist in source
        guide_data = sample_guide_sequence_data.copy()
        guide_data['pitch_segments'] = [
            {
                'index': i,
                'start_time': i * 0.5,
                'end_time': (i + 1) * 0.5,
                'duration': 0.5,
                'pitch_hz': 261.63 * (2 ** ((60 + i % 6) - 60) / 12),
                'pitch_midi': 60 + (i % 6),  # C4 to F4 (all exist in source)
                'pitch_note': ['C4', 'C#4', 'D4', 'D#4', 'E4', 'F4'][i % 6],
                'pitch_confidence': 0.9,
                'is_rest': False
            }
            for i in range(5)
        ]

        guide_path = temp_dir / "guide.json"
        source_path = temp_dir / "source.json"

        guide_path.write_text(json.dumps(guide_data))
        source_path.write_text(json.dumps(sample_source_database_data))

        with patch('builtins.print'):
            matcher = PitchMatcher(str(guide_path), str(source_path))
            matcher.load_guide_sequence()
            matcher.load_source_database()
        return matcher

    def test_match_guide_to_source(self, matcher):
        """Test full matching workflow."""
        with patch('builtins.print'):
            matcher.match_guide_to_source()

        assert len(matcher.matches) == 5

        # Check match structure
        for match in matcher.matches:
            assert 'guide_segment_id' in match
            assert 'match_type' in match
            assert 'source_clips' in match

    def test_save_match_plan(self, matcher, temp_dir):
        """Test saving match plan."""
        with patch('builtins.print'):
            matcher.match_guide_to_source()
            output_path = temp_dir / "match_plan.json"
            matcher.save_match_plan(str(output_path))

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert 'matches' in data
        assert 'statistics' in data
        assert 'matching_config' in data
        assert data['statistics']['total_guide_segments'] == 5


class TestPitchMatcherRestSegments:
    """Test rest segment handling in matching."""

    @pytest.fixture
    def matcher_with_rests(self, temp_dir, sample_source_database_data):
        """Create matcher with rest segments in guide."""
        guide_data = {
            'pitch_segments': [
                {
                    'index': 0,
                    'start_time': 0.0,
                    'end_time': 0.5,
                    'duration': 0.5,
                    'pitch_midi': 60,
                    'pitch_note': 'C4',
                    'pitch_confidence': 0.9,
                    'is_rest': False
                },
                {
                    'index': 1,
                    'start_time': 0.5,
                    'end_time': 1.0,
                    'duration': 0.5,
                    'pitch_midi': -1,
                    'pitch_note': 'REST',
                    'pitch_confidence': 1.0,
                    'is_rest': True
                }
            ]
        }

        guide_path = temp_dir / "guide.json"
        source_path = temp_dir / "source.json"

        guide_path.write_text(json.dumps(guide_data))
        source_path.write_text(json.dumps(sample_source_database_data))

        with patch('builtins.print'):
            matcher = PitchMatcher(str(guide_path), str(source_path))
            matcher.load_guide_sequence()
            matcher.load_source_database()
        return matcher

    def test_rest_segments_matched(self, matcher_with_rests):
        """Test that rest segments are matched correctly."""
        with patch('builtins.print'):
            matcher_with_rests.match_guide_to_source()

        assert len(matcher_with_rests.matches) == 2

        # Second match should be a rest
        rest_match = matcher_with_rests.matches[1]
        assert rest_match['match_type'] == 'rest'
        assert rest_match['guide_pitch_note'] == 'REST'
