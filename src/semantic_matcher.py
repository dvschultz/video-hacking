#!/usr/bin/env python3
"""
Semantic Matcher

Matches audio segments to video segments using ImageBind embeddings.
Supports multiple reuse policies for video segments.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


class SemanticMatcher:
    """Match audio segments to video segments using embeddings."""

    def __init__(
        self,
        audio_embeddings_path: str,
        video_embeddings_path: str,
        audio_segments_path: str
    ):
        """
        Initialize matcher with embedding data.

        Args:
            audio_embeddings_path: Path to audio embeddings JSON
            video_embeddings_path: Path to video embeddings JSON
            audio_segments_path: Path to audio segments metadata JSON
        """
        # Load audio embeddings (uses 'segments' key from audio_embedder.py)
        with open(audio_embeddings_path, 'r') as f:
            audio_data = json.load(f)
        self.audio_embeddings = audio_data['segments']  # Changed from 'embeddings'

        # Load video embeddings (uses 'embeddings' key from video_embedder.py)
        with open(video_embeddings_path, 'r') as f:
            video_data = json.load(f)
        self.video_embeddings = video_data['embeddings']

        # Load audio segment metadata
        with open(audio_segments_path, 'r') as f:
            self.audio_segments = json.load(f)

        print(f"Loaded {len(self.audio_embeddings)} audio embeddings")
        print(f"Loaded {len(self.video_embeddings)} video embeddings")
        print(f"Loaded {len(self.audio_segments['segments'])} audio segments")

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        # Embeddings should already be normalized, but ensure it
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.dot(emb1, emb2))

    def find_best_matches(
        self,
        reuse_policy: str = 'allow',
        min_gap: int = 5,
        max_reuse_count: int = 3,
        max_reuse_percentage: float = 0.3
    ) -> List[Dict]:
        """
        Find best video matches for each audio segment.

        Args:
            reuse_policy: How to handle video segment reuse:
                - 'none': No reuse allowed (each video segment used once)
                - 'allow': Allow unlimited reuse
                - 'min_gap': Require minimum gap between reuses
                - 'limited': Limit reuse count per video segment
                - 'percentage': Limit reuse to percentage of total
            min_gap: Minimum segments between reuses (for 'min_gap' policy)
            max_reuse_count: Max times a video segment can be reused (for 'limited')
            max_reuse_percentage: Max percentage of segments that can reuse (for 'percentage')

        Returns:
            List of matches with audio/video segment info and similarity scores
        """
        print(f"\nFinding matches with policy: {reuse_policy}")

        matches = []
        used_video_indices = set()  # For 'none' policy
        video_usage_count = {}  # For 'limited' policy
        last_usage = {}  # For 'min_gap' policy
        reuse_count = 0  # For 'percentage' policy

        for audio_idx, audio_emb_data in enumerate(self.audio_embeddings):
            audio_emb = np.array(audio_emb_data['embedding'])
            audio_seg = self.audio_segments['segments'][audio_idx]

            # Calculate similarities with all video embeddings
            similarities = []
            for video_idx, video_emb_data in enumerate(self.video_embeddings):
                video_emb = np.array(video_emb_data['embedding'])
                similarity = self.cosine_similarity(audio_emb, video_emb)

                # Apply reuse policy filters
                valid = True

                if reuse_policy == 'none':
                    # Can't reuse any video segment
                    if video_idx in used_video_indices:
                        valid = False

                elif reuse_policy == 'min_gap':
                    # Must have gap since last use
                    if video_idx in last_usage:
                        gap = audio_idx - last_usage[video_idx]
                        if gap < min_gap:
                            valid = False

                elif reuse_policy == 'limited':
                    # Can't exceed max reuse count
                    count = video_usage_count.get(video_idx, 0)
                    if count >= max_reuse_count:
                        valid = False

                elif reuse_policy == 'percentage':
                    # Can't exceed percentage of total segments
                    is_reused = video_idx in used_video_indices
                    if is_reused:
                        max_reuses = int(len(self.audio_embeddings) * max_reuse_percentage)
                        if reuse_count >= max_reuses:
                            valid = False

                if valid:
                    similarities.append({
                        'video_idx': video_idx,
                        'similarity': similarity,
                        'video_data': video_emb_data
                    })

            # Find best match
            if not similarities:
                print(f"Warning: No valid matches for audio segment {audio_idx}")
                # Fall back to best match ignoring policy
                similarities = []
                for video_idx, video_emb_data in enumerate(self.video_embeddings):
                    video_emb = np.array(video_emb_data['embedding'])
                    similarity = self.cosine_similarity(audio_emb, video_emb)
                    similarities.append({
                        'video_idx': video_idx,
                        'similarity': similarity,
                        'video_data': video_emb_data
                    })

            best_match = max(similarities, key=lambda x: x['similarity'])

            # Update tracking
            video_idx = best_match['video_idx']
            is_reused = video_idx in used_video_indices

            used_video_indices.add(video_idx)
            video_usage_count[video_idx] = video_usage_count.get(video_idx, 0) + 1
            last_usage[video_idx] = audio_idx

            if is_reused and reuse_policy == 'percentage':
                reuse_count += 1

            # Create match record
            match = {
                'audio_idx': audio_idx,
                'audio_segment_index': audio_seg['index'],
                'audio_filename': audio_seg['filename'],
                'audio_start_time': audio_seg['start_time'],
                'audio_end_time': audio_seg['end_time'],
                'audio_duration': audio_seg['duration'],
                'video_idx': video_idx,
                'video_window': best_match['video_data']['window_idx'],
                'video_start_time': best_match['video_data']['start_time'],
                'video_end_time': best_match['video_data']['end_time'],
                'video_center_time': best_match['video_data']['center_time'],
                'similarity_score': best_match['similarity'],
                'is_reused': is_reused,
                'usage_count': video_usage_count[video_idx]
            }

            matches.append(match)

        # Print statistics
        print(f"\n✓ Matched {len(matches)} audio segments")
        print(f"\nReuse Statistics:")
        print(f"  Unique video segments used: {len(used_video_indices)}/{len(self.video_embeddings)}")
        print(f"  Video segments reused: {sum(1 for c in video_usage_count.values() if c > 1)}")
        print(f"  Max usage count: {max(video_usage_count.values())}")
        print(f"  Avg similarity: {np.mean([m['similarity_score'] for m in matches]):.3f}")
        print(f"  Min similarity: {np.min([m['similarity_score'] for m in matches]):.3f}")
        print(f"  Max similarity: {np.max([m['similarity_score'] for m in matches]):.3f}")

        return matches

    def save_matches(self, matches: List[Dict], output_path: str):
        """
        Save matches to JSON file.

        Args:
            matches: List of match dictionaries
            output_path: Path to output JSON file
        """
        output_data = {
            'num_matches': len(matches),
            'matches': matches,
            'statistics': {
                'avg_similarity': float(np.mean([m['similarity_score'] for m in matches])),
                'min_similarity': float(np.min([m['similarity_score'] for m in matches])),
                'max_similarity': float(np.max([m['similarity_score'] for m in matches])),
                'unique_video_segments': len(set(m['video_idx'] for m in matches)),
                'total_video_segments': len(self.video_embeddings),
                'reused_segments': sum(1 for m in matches if m['is_reused'])
            }
        }

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Saved matches to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Match audio segments to video segments using ImageBind embeddings'
    )
    parser.add_argument('--audio-embeddings', required=True,
                       help='Path to audio embeddings JSON')
    parser.add_argument('--video-embeddings', required=True,
                       help='Path to video embeddings JSON')
    parser.add_argument('--audio-segments', required=True,
                       help='Path to audio segments metadata JSON')
    parser.add_argument('--output', required=True,
                       help='Path to output matches JSON')
    parser.add_argument('--reuse-policy', default='allow',
                       choices=['none', 'allow', 'min_gap', 'limited', 'percentage'],
                       help='Video segment reuse policy (default: allow)')
    parser.add_argument('--min-gap', type=int, default=5,
                       help='Min gap between reuses for min_gap policy (default: 5)')
    parser.add_argument('--max-reuse-count', type=int, default=3,
                       help='Max reuse count for limited policy (default: 3)')
    parser.add_argument('--max-reuse-percentage', type=float, default=0.3,
                       help='Max reuse percentage for percentage policy (default: 0.3)')

    args = parser.parse_args()

    # Initialize matcher
    matcher = SemanticMatcher(
        audio_embeddings_path=args.audio_embeddings,
        video_embeddings_path=args.video_embeddings,
        audio_segments_path=args.audio_segments
    )

    # Find matches
    matches = matcher.find_best_matches(
        reuse_policy=args.reuse_policy,
        min_gap=args.min_gap,
        max_reuse_count=args.max_reuse_count,
        max_reuse_percentage=args.max_reuse_percentage
    )

    # Save results
    matcher.save_matches(matches, args.output)

    print("\n✓ Semantic matching complete!")


if __name__ == '__main__':
    main()
