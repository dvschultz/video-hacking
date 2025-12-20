#!/usr/bin/env python3
"""
Onset Strength Analysis Module

This module analyzes audio files to generate continuous onset strength values
per video frame, suitable for frame-accurate video cutting and effects.

Based on the track_sound_change() approach - generates a continuous curve
rather than discrete onset points.
"""

import argparse
import json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Tuple, Dict, Optional


class OnsetStrengthAnalyzer:
    """Analyzes audio to generate continuous onset strength values per frame."""

    def __init__(self, audio_path: str, sr: Optional[int] = None):
        """
        Initialize the analyzer.

        Args:
            audio_path: Path to audio file
            sr: Sample rate (None = use file's native rate)
        """
        self.audio_path = Path(audio_path)
        self.sr = sr
        self.y = None
        self.onset_strength_values = None
        self.times = None
        self.analysis_rate = None

        # Load audio file
        self._load_audio()

    def _load_audio(self):
        """Load audio file using librosa."""
        print(f"Loading audio file: {self.audio_path}")
        self.y, self.sr = librosa.load(self.audio_path, sr=self.sr)
        duration = librosa.get_duration(y=self.y, sr=self.sr)
        print(f"Loaded audio: {duration:.2f}s @ {self.sr}Hz")

    def analyze(
        self,
        analysis_rate: int = 24,
        ignore_start_duration: float = 0.5,
        power: float = 0.6,
        smoothing_window_size: int = 1,
        smoothing_tolerance: float = 0.3
    ) -> np.ndarray:
        """
        Analyze audio to generate continuous onset strength values.

        This generates one value per frame at the specified frame rate,
        suitable for video synchronization.

        Args:
            analysis_rate: Frame rate (values per second, e.g., 24 for 24fps)
            ignore_start_duration: Duration at start to zero out (seconds)
            power: Power-law scaling exponent (0.5-1.0, lower = more compression)
            smoothing_window_size: Window size for adaptive smoothing (frames on each side)
            smoothing_tolerance: Tolerance for noise removal (0.0-1.0)

        Returns:
            Array of onset strength values (one per frame)
        """
        self.analysis_rate = analysis_rate

        print(f"Analyzing onset strength at {analysis_rate} FPS...")

        # 1. Calculate onset strength envelope
        hop_length = int(self.sr / analysis_rate)
        onset_env = librosa.onset.onset_strength(
            y=self.y,
            sr=self.sr,
            hop_length=hop_length
        )

        # 2. Normalize, ignoring initial spike
        frames_to_ignore = int(ignore_start_duration * analysis_rate)
        if frames_to_ignore >= len(onset_env):
            frames_to_ignore = max(0, len(onset_env) - 1)

        max_value_slice = onset_env[frames_to_ignore:]
        if max_value_slice.size > 0 and np.max(max_value_slice) > 0:
            max_value = np.max(max_value_slice)
        else:
            max_value = np.max(onset_env) if onset_env.size > 0 else 1

        normalized = onset_env / max_value if max_value > 0 else np.zeros_like(onset_env)

        # 3. Apply power-law scaling (compression)
        scaled = np.power(normalized, power)

        # 4. Zero out initial section
        if frames_to_ignore > 0:
            scaled[:frames_to_ignore] = 0

        # 5. Apply adaptive smoothing filter
        if smoothing_window_size > 0 and len(scaled) > 0:
            original = np.copy(scaled)
            for i in range(len(scaled)):
                start = max(0, i - smoothing_window_size)
                end = min(len(scaled), i + smoothing_window_size + 1)
                window = original[start:end]

                # If change within window is small, it's noise
                if (np.max(window) - np.min(window)) < smoothing_tolerance:
                    scaled[i] = 0

        self.onset_strength_values = scaled
        self.times = librosa.times_like(
            scaled,
            sr=self.sr,
            hop_length=hop_length
        )

        print(f"Generated {len(scaled)} onset strength values")
        print(f"Value range: {np.min(scaled):.3f} to {np.max(scaled):.3f}")
        print(f"Non-zero values: {np.count_nonzero(scaled)} ({np.count_nonzero(scaled)/len(scaled)*100:.1f}%)")

        return scaled

    def get_cut_frames(self, threshold: float = 0.1) -> np.ndarray:
        """
        Get frame numbers where onset strength exceeds threshold.

        Args:
            threshold: Minimum onset strength to register as a cut point (0.0-1.0)

        Returns:
            Array of frame numbers where cuts should occur
        """
        if self.onset_strength_values is None:
            raise ValueError("Must run analyze() first")

        cut_frames = np.where(self.onset_strength_values > threshold)[0]
        print(f"Found {len(cut_frames)} frames above threshold {threshold}")

        return cut_frames

    def get_cut_times(self, threshold: float = 0.1) -> np.ndarray:
        """
        Get timestamps where onset strength exceeds threshold.

        Args:
            threshold: Minimum onset strength to register as a cut point (0.0-1.0)

        Returns:
            Array of timestamps (seconds) where cuts should occur
        """
        cut_frames = self.get_cut_frames(threshold)
        cut_times = self.times[cut_frames]

        return cut_times

    def visualize(
        self,
        output_path: Optional[str] = None,
        threshold: Optional[float] = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Visualize waveform and onset strength curve.

        Args:
            output_path: Path to save figure (None = display)
            threshold: Optional threshold line to display
            figsize: Figure size (width, height)
        """
        if self.onset_strength_values is None:
            raise ValueError("Must run analyze() first")

        fig, axes = plt.subplots(2, 1, figsize=figsize)

        # Plot waveform
        librosa.display.waveshow(
            self.y,
            sr=self.sr,
            ax=axes[0],
            color='royalblue',
            alpha=0.6
        )
        axes[0].set_title('Audio Waveform')
        axes[0].set_ylabel('Amplitude')

        # Plot onset strength curve
        axes[1].plot(
            self.times,
            self.onset_strength_values,
            label='Onset Strength',
            color='crimson',
            linewidth=2
        )

        if threshold is not None:
            axes[1].axhline(
                y=threshold,
                color='green',
                linestyle='--',
                linewidth=1.5,
                label=f'Threshold ({threshold})'
            )

        axes[1].set_title(f'Onset Strength Curve (Analysis Rate: {self.analysis_rate} FPS)')
        axes[1].set_ylabel('Onset Strength (Scaled)')
        axes[1].set_xlabel('Time (s)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([-0.05, 1.05])

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()

        plt.close()

    def export(
        self,
        output_path: str,
        format: str = 'json',
        include_values: bool = True
    ):
        """
        Export onset strength analysis results.

        Args:
            output_path: Path to output file
            format: Output format ('json' or 'txt')
            include_values: Whether to include full value array
        """
        if self.onset_strength_values is None:
            raise ValueError("Must run analyze() first")

        output_path = Path(output_path)

        if format == 'json':
            data = {
                'audio_file': str(self.audio_path),
                'sample_rate': self.sr,
                'analysis_rate': self.analysis_rate,
                'duration': float(librosa.get_duration(y=self.y, sr=self.sr)),
                'num_frames': len(self.onset_strength_values),
                'min_value': float(np.min(self.onset_strength_values)),
                'max_value': float(np.max(self.onset_strength_values)),
                'mean_value': float(np.mean(self.onset_strength_values)),
                'non_zero_frames': int(np.count_nonzero(self.onset_strength_values))
            }

            if include_values:
                data['onset_strength_values'] = self.onset_strength_values.tolist()
                data['times'] = self.times.tolist()

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

        elif format == 'txt':
            # Simple text format: one value per line
            with open(output_path, 'w') as f:
                for value in self.onset_strength_values:
                    f.write(f"{value}\n")

        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"Results exported to: {output_path}")

    def get_statistics(self) -> Dict:
        """Get statistics about onset strength values."""
        if self.onset_strength_values is None:
            raise ValueError("Must run analyze() first")

        return {
            'num_frames': len(self.onset_strength_values),
            'duration': float(librosa.get_duration(y=self.y, sr=self.sr)),
            'analysis_rate': self.analysis_rate,
            'min_value': float(np.min(self.onset_strength_values)),
            'max_value': float(np.max(self.onset_strength_values)),
            'mean_value': float(np.mean(self.onset_strength_values)),
            'std_value': float(np.std(self.onset_strength_values)),
            'non_zero_frames': int(np.count_nonzero(self.onset_strength_values)),
            'non_zero_percentage': float(np.count_nonzero(self.onset_strength_values) / len(self.onset_strength_values) * 100)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze audio onset strength for frame-accurate video cutting'
    )
    parser.add_argument('--audio', required=True, help='Path to audio file')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    parser.add_argument('--fps', type=int, default=24,
                       help='Frame rate for analysis (default: 24)')
    parser.add_argument('--power', type=float, default=0.6,
                       help='Power scaling (0.5-1.0, lower=more compression, default: 0.6)')
    parser.add_argument('--window-size', type=int, default=1,
                       help='Smoothing window size (frames on each side, default: 1)')
    parser.add_argument('--tolerance', type=float, default=0.3,
                       help='Smoothing tolerance (0.0-1.0, default: 0.3)')
    parser.add_argument('--ignore-start', type=float, default=0.5,
                       help='Ignore start duration in seconds (default: 0.5)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization')
    parser.add_argument('--viz-output', default=None,
                       help='Path to save visualization')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Threshold for visualization (0.0-1.0)')
    parser.add_argument('--format', default='json', choices=['json', 'txt'],
                       help='Output format')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = OnsetStrengthAnalyzer(args.audio)

    # Analyze
    analyzer.analyze(
        analysis_rate=args.fps,
        ignore_start_duration=args.ignore_start,
        power=args.power,
        smoothing_window_size=args.window_size,
        smoothing_tolerance=args.tolerance
    )

    # Print statistics
    stats = analyzer.get_statistics()
    print("\n=== Onset Strength Statistics ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    # Export
    analyzer.export(args.output, format=args.format)

    # Visualize if requested
    if args.visualize:
        analyzer.visualize(
            output_path=args.viz_output,
            threshold=args.threshold
        )

    print("\n✓ Onset strength analysis complete!")


if __name__ == '__main__':
    main()
