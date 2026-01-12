"""
Unit tests for semantic_matcher.py - Semantic audio-video matching.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestSemanticMatcher:
    """Test SemanticMatcher class."""

    @pytest.fixture
    def matcher(self, audio_embeddings_file, video_embeddings_file, audio_segments_file):
        """Create SemanticMatcher with test data files."""
        # Suppress print statements during tests
        with patch('builtins.print'):
            from semantic_matcher import SemanticMatcher
            return SemanticMatcher(
                str(audio_embeddings_file),
                str(video_embeddings_file),
                str(audio_segments_file)
            )

    def test_initialization_loads_data(self, matcher):
        """Test matcher loads all data correctly."""
        assert len(matcher.audio_embeddings) == 5
        assert len(matcher.video_embeddings) == 40
        assert len(matcher.audio_segments['segments']) == 5

    def test_cosine_similarity_identical_vectors(self, matcher):
        """Identical vectors should have similarity 1.0."""
        vec = np.random.randn(1024)
        sim = matcher.cosine_similarity(vec, vec)
        assert sim == pytest.approx(1.0, abs=0.001)

    def test_cosine_similarity_orthogonal_vectors(self, matcher):
        """Orthogonal vectors should have similarity 0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        sim = matcher.cosine_similarity(vec1, vec2)
        assert sim == pytest.approx(0.0, abs=0.001)

    def test_cosine_similarity_opposite_vectors(self, matcher):
        """Opposite vectors should have similarity -1.0."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([-1.0, 0.0])
        sim = matcher.cosine_similarity(vec1, vec2)
        assert sim == pytest.approx(-1.0, abs=0.001)

    def test_cosine_similarity_normalized(self, matcher):
        """Similarity should be same regardless of vector magnitude."""
        vec1 = np.array([1.0, 0.0])
        vec2_small = np.array([0.5, 0.5])
        vec2_large = np.array([5.0, 5.0])

        sim_small = matcher.cosine_similarity(vec1, vec2_small)
        sim_large = matcher.cosine_similarity(vec1, vec2_large)

        assert sim_small == pytest.approx(sim_large, abs=0.001)

    def test_cosine_similarity_range(self, matcher):
        """Similarity should always be in [-1, 1] range."""
        for _ in range(10):
            vec1 = np.random.randn(1024)
            vec2 = np.random.randn(1024)
            sim = matcher.cosine_similarity(vec1, vec2)
            assert -1.0 <= sim <= 1.0

    def test_find_best_matches_allow_reuse(self, matcher):
        """Test matching with 'allow' reuse policy."""
        with patch('builtins.print'):
            matches = matcher.find_best_matches(reuse_policy='allow')

        assert len(matches) == 5  # One match per audio segment

        for match in matches:
            assert 'audio_idx' in match
            assert 'video_idx' in match
            assert 'similarity_score' in match
            assert 'audio_duration' in match
            assert 'video_start_time' in match
            assert -1.0 <= match['similarity_score'] <= 1.0

    def test_find_best_matches_no_reuse(self, matcher):
        """Test matching with 'none' reuse policy."""
        with patch('builtins.print'):
            matches = matcher.find_best_matches(reuse_policy='none')

        # All video indices should be unique
        video_indices = [m['video_idx'] for m in matches]
        assert len(video_indices) == len(set(video_indices))

    def test_find_best_matches_min_gap(self, matcher):
        """Test matching with 'min_gap' reuse policy."""
        with patch('builtins.print'):
            matches = matcher.find_best_matches(reuse_policy='min_gap', min_gap=3)

        # Verify min_gap constraint for any reused segments
        video_last_used = {}
        for i, match in enumerate(matches):
            vid_idx = match['video_idx']
            if vid_idx in video_last_used:
                gap = i - video_last_used[vid_idx]
                # If reused within gap, it should be marked as reused
                if gap < 3:
                    assert match.get('is_reused', False) == True
            video_last_used[vid_idx] = i

    def test_find_best_matches_limited(self, matcher):
        """Test matching with 'limited' reuse policy."""
        with patch('builtins.print'):
            matches = matcher.find_best_matches(reuse_policy='limited', max_reuse_count=2)

        # Count usage per video segment
        usage_count = {}
        for match in matches:
            vid_idx = match['video_idx']
            usage_count[vid_idx] = usage_count.get(vid_idx, 0) + 1

        # No segment should exceed max_reuse_count
        for count in usage_count.values():
            assert count <= 2

    def test_find_best_matches_percentage(self, matcher):
        """Test matching with 'percentage' reuse policy."""
        with patch('builtins.print'):
            matches = matcher.find_best_matches(
                reuse_policy='percentage',
                max_reuse_percentage=0.5
            )

        # Just verify it completes without error
        assert len(matches) == 5

    def test_find_best_matches_contains_timing_info(self, matcher):
        """Test that matches contain all required timing information."""
        with patch('builtins.print'):
            matches = matcher.find_best_matches(reuse_policy='allow')

        for match in matches:
            assert 'audio_start_time' in match
            assert 'audio_end_time' in match
            assert 'audio_duration' in match
            assert 'video_start_time' in match
            assert 'video_end_time' in match
            assert match['audio_duration'] > 0

    def test_save_matches(self, matcher, temp_dir):
        """Test saving matches to JSON."""
        with patch('builtins.print'):
            matches = matcher.find_best_matches()
            output_path = temp_dir / "matches_output.json"
            matcher.save_matches(matches, str(output_path))

        assert output_path.exists()

        with open(output_path) as f:
            saved = json.load(f)

        assert saved['num_matches'] == len(matches)
        assert 'statistics' in saved
        assert 'matches' in saved
        assert saved['statistics']['avg_similarity'] >= -1.0
        assert saved['statistics']['unique_video_segments'] > 0

    def test_save_matches_creates_directory(self, matcher, temp_dir):
        """Test that save_matches creates parent directories."""
        with patch('builtins.print'):
            matches = matcher.find_best_matches()
            output_path = temp_dir / "subdir" / "nested" / "matches.json"
            matcher.save_matches(matches, str(output_path))

        assert output_path.exists()


class TestSemanticMatcherEdgeCases:
    """Test edge cases in SemanticMatcher."""

    def test_empty_reuse_tracking(self, audio_embeddings_file, video_embeddings_file,
                                   audio_segments_file):
        """Test that reuse tracking starts empty."""
        with patch('builtins.print'):
            from semantic_matcher import SemanticMatcher
            matcher = SemanticMatcher(
                str(audio_embeddings_file),
                str(video_embeddings_file),
                str(audio_segments_file)
            )
            matches = matcher.find_best_matches(reuse_policy='none')

        # First audio segment should not be marked as reused
        assert matches[0].get('is_reused', False) == False

    def test_similarity_deterministic(self, audio_embeddings_file, video_embeddings_file,
                                       audio_segments_file):
        """Test that matching is deterministic with same data."""
        with patch('builtins.print'):
            from semantic_matcher import SemanticMatcher
            matcher1 = SemanticMatcher(
                str(audio_embeddings_file),
                str(video_embeddings_file),
                str(audio_segments_file)
            )
            matcher2 = SemanticMatcher(
                str(audio_embeddings_file),
                str(video_embeddings_file),
                str(audio_segments_file)
            )

            matches1 = matcher1.find_best_matches(reuse_policy='allow')
            matches2 = matcher2.find_best_matches(reuse_policy='allow')

        # Same video indices should be selected
        for m1, m2 in zip(matches1, matches2):
            assert m1['video_idx'] == m2['video_idx']
            assert m1['similarity_score'] == pytest.approx(m2['similarity_score'])
