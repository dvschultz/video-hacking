#!/usr/bin/env python3
"""
Audio Segmenter

Cuts audio into segments based on onset strength analysis.
Each segment corresponds to the time between consecutive onset points.
"""

import argparse
import json
import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import List, Tuple


class AudioSegmenter:
    """Segments audio based on onset strength threshold."""

    def __init__(self, audio_path: str, onset_strength_json: str):
        """
        Initialize the segmenter.

        Args:
            audio_path: Path to audio file to segment
            onset_strength_json: Path to onset strength analysis JSON
        """
        self.audio_path = Path(audio_path)
        self.onset_json_path = Path(onset_strength_json)

        # Load onset strength data
        with open(onset_strength_json, 'r') as f:
            self.onset_data = json.load(f)

        self.onset_values = np.array(self.onset_data['onset_strength_values'])
        self.times = np.array(self.onset_data['times'])
        self.sample_rate = self.onset_data['sample_rate']
        self.fps = self.onset_data['analysis_rate']

        # Load audio
        self.audio, self.sr = librosa.load(str(audio_path), sr=None)
        print(f"Loaded audio: {len(self.audio)/self.sr:.2f}s @ {self.sr}Hz")

    def find_cut_points(self, threshold: float = 0.2) -> np.ndarray:
        """
        Find frame indices where onset strength exceeds threshold.

        Args:
            threshold: Onset strength threshold (0.0-1.0)

        Returns:
            Array of frame indices where cuts should occur
        """
        cut_indices = np.where(self.onset_values > threshold)[0]
        cut_times = self.times[cut_indices]

        print(f"Found {len(cut_indices)} cut points (threshold={threshold})")
        return cut_times

    def segment_audio(
        self,
        cut_times: np.ndarray,
        output_dir: str,
        prefix: str = "segment"
    ) -> List[dict]:
        """
        Cut audio at specified times and save segments.

        Args:
            cut_times: Array of times (in seconds) to cut at
            output_dir: Directory to save segments
            prefix: Prefix for segment filenames

        Returns:
            List of segment metadata dicts
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        segments = []

        # Add start and end boundaries
        all_times = np.concatenate([[0.0], cut_times, [len(self.audio) / self.sr]])
        all_times = np.unique(all_times)  # Remove duplicates, sort

        print(f"Creating {len(all_times) - 1} audio segments...")

        for i in range(len(all_times) - 1):
            start_time = all_times[i]
            end_time = all_times[i + 1]
            duration = end_time - start_time

            # Convert times to samples
            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)

            # Extract segment
            segment_audio = self.audio[start_sample:end_sample]

            # Save segment
            segment_filename = f"{prefix}_{i:04d}.wav"
            segment_path = output_dir / segment_filename

            sf.write(segment_path, segment_audio, self.sr)

            # Store metadata
            segment_info = {
                'index': i,
                'filename': segment_filename,
                'start_time': float(start_time),
                'end_time': float(end_time),
                'duration': float(duration),
                'num_samples': len(segment_audio)
            }
            segments.append(segment_info)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(all_times) - 1} segments")

        print(f"✓ Saved {len(segments)} segments to {output_dir}")

        return segments

    def export_metadata(self, segments: List[dict], output_path: str, threshold: float):
        """
        Export segment metadata to JSON.

        Args:
            segments: List of segment metadata dicts
            output_path: Path to output JSON file
            threshold: Threshold used for segmentation
        """
        metadata = {
            'source_audio': str(self.audio_path),
            'onset_strength_json': str(self.onset_json_path),
            'threshold': threshold,
            'sample_rate': self.sr,
            'num_segments': len(segments),
            'total_duration': float(len(self.audio) / self.sr),
            'segments': segments
        }

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Exported metadata to {output_path}")

    def get_statistics(self, segments: List[dict]) -> dict:
        """Get statistics about the segments."""
        durations = [s['duration'] for s in segments]

        return {
            'num_segments': len(segments),
            'total_duration': sum(durations),
            'avg_duration': np.mean(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'median_duration': np.median(durations),
            'std_duration': np.std(durations)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Segment audio based on onset strength analysis'
    )
    parser.add_argument('--audio', required=True,
                       help='Path to audio file to segment')
    parser.add_argument('--onset-strength', required=True,
                       help='Path to onset strength JSON file')
    parser.add_argument('--output-dir', default='data/segments/audio',
                       help='Directory to save audio segments')
    parser.add_argument('--metadata-output', default='data/segments/audio_segments.json',
                       help='Path to save segment metadata JSON')
    parser.add_argument('--threshold', type=float, default=0.2,
                       help='Onset strength threshold (0.0-1.0)')
    parser.add_argument('--prefix', default='audio_seg',
                       help='Prefix for segment filenames')

    args = parser.parse_args()

    # Initialize segmenter
    segmenter = AudioSegmenter(args.audio, args.onset_strength)

    # Find cut points
    cut_times = segmenter.find_cut_points(threshold=args.threshold)

    # Segment audio
    segments = segmenter.segment_audio(
        cut_times,
        output_dir=args.output_dir,
        prefix=args.prefix
    )

    # Export metadata
    segmenter.export_metadata(segments, args.metadata_output, args.threshold)

    # Print statistics
    stats = segmenter.get_statistics(segments)
    print("\n=== Segmentation Statistics ===")
    print(f"Total segments: {stats['num_segments']}")
    print(f"Total duration: {stats['total_duration']:.2f}s")
    print(f"Avg segment: {stats['avg_duration']:.3f}s")
    print(f"Min segment: {stats['min_duration']:.3f}s")
    print(f"Max segment: {stats['max_duration']:.3f}s")
    print(f"Median segment: {stats['median_duration']:.3f}s")

    print("\n✓ Audio segmentation complete!")


if __name__ == '__main__':
    main()
