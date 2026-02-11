#!/usr/bin/env python3
"""
Pitch Guide Analyzer

Analyzes a "guide" video to extract the target pitch sequence.
Uses pitch change detection to find when the singer changes notes or syllables.

This generates the target sequence that we'll recreate by recutting the source video.
"""

import argparse
import json
import numpy as np
import librosa
import soundfile as sf
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Import pitch change detection
from pitch_change_detector import PitchChangeDetector
from pitch_utils import (
    hz_to_midi, midi_to_note_name, round_to_nearest_midi,
    calculate_pitch_statistics, is_pitch_in_range
)

try:
    import crepe
    CREPE_AVAILABLE = True
except ImportError:
    CREPE_AVAILABLE = False
    print("Warning: CREPE not available. Will use librosa.pyin instead.")
    print("For best results, install CREPE: pip install crepe")


class PitchGuideAnalyzer:
    """Analyzes guide video to extract target pitch sequence."""

    def __init__(self, video_path: str, pitch_method: str = 'crepe', device: str = 'auto'):
        """
        Initialize the analyzer.

        Args:
            video_path: Path to guide video file
            pitch_method: Pitch detection method ('crepe', 'basic-pitch', or 'pyin')
            device: Device for neural network models ('auto', 'cuda', 'cpu')
        """
        self.video_path = Path(video_path)
        self.audio_path = None
        self.pitch_method = pitch_method
        self.device = device

        self.audio = None
        self.sr = None
        self.fps = None

        self.onset_analyzer = None
        self.onset_times = None
        self.onset_strength = None

        self.pitch_sequence = []

    def extract_audio(self, output_dir: str = "data/temp",
                     sample_rate: int = 22050) -> Path:
        """
        Extract audio from video file.

        Args:
            output_dir: Directory to save extracted audio
            sample_rate: Audio sample rate (22050 Hz is good for pitch detection)

        Returns:
            Path to extracted audio file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_filename = self.video_path.stem + "_audio.wav"
        audio_path = output_dir / audio_filename

        print(f"Extracting audio from: {self.video_path}")
        print(f"Output: {audio_path}")

        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg', '-y',
            '-i', str(self.video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono
            str(audio_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Audio extracted successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio: {e}")
            print(f"stderr: {e.stderr.decode()}")
            raise

        self.audio_path = audio_path
        self.sr = sample_rate

        # Load audio for processing
        self.audio, _ = librosa.load(str(audio_path), sr=sample_rate)

        return audio_path

    def analyze_pitch_changes(self, fps: int = 24,
                              pitch_change_threshold: float = 50.0,
                              min_segment_duration: float = 0.1,
                              min_rms_db: float = -50.0,
                              pitch_smoothing: int = 0,
                              min_rest_duration: float = 0.1,
                              include_rests: bool = True,
                              verify_rest_rms: bool = True) -> List[Dict]:
        """
        Detect pitch changes (when singer changes notes or syllables).

        Args:
            fps: Video frame rate
            pitch_change_threshold: Cents difference to trigger new segment (default: 50)
            min_segment_duration: Minimum segment length in seconds
            min_rms_db: Minimum RMS amplitude to detect silence (default: -50dB)
            pitch_smoothing: Median filter window size for pitch smoothing (0=off, 5-7 recommended)
            min_rest_duration: Minimum gap duration to create rest segment (default: 0.1s)
            include_rests: Include explicit rest segments for gaps (default: True)
            verify_rest_rms: Verify gaps are silent via RMS check (default: True)

        Returns:
            List of pitch segment dictionaries (includes rest segments if include_rests=True)
        """
        if self.audio_path is None:
            raise ValueError("Must extract audio first")

        self.fps = fps

        print(f"\nDetecting pitch changes (threshold={pitch_change_threshold} cents)...")

        # Use pitch change detector
        detector = PitchChangeDetector(str(self.audio_path), sr=self.sr, pitch_method=self.pitch_method, device=self.device)

        # Extract continuous pitch with optional smoothing
        detector.extract_continuous_pitch(frame_time=0.01, pitch_smoothing_window=pitch_smoothing)

        # Detect pitch segments (splits on pitch changes OR silence)
        segments = detector.detect_pitch_segments(
            pitch_change_threshold_cents=pitch_change_threshold,
            min_segment_duration=min_segment_duration,
            min_confidence=0.5,
            min_rms_db=min_rms_db
        )

        # Convert to dictionaries
        self.pitch_sequence = detector.segments_to_dict(segments)

        print(f"Detected {len(self.pitch_sequence)} pitch segments")

        # Insert explicit rest segments for gaps between pitched segments
        if include_rests and len(self.pitch_sequence) > 0:
            self.pitch_sequence = self._insert_rest_segments(
                self.pitch_sequence,
                min_rest_duration=min_rest_duration,
                verify_rms=verify_rest_rms,
                silence_threshold_db=min_rms_db
            )
            num_rests = sum(1 for s in self.pitch_sequence if s.get('is_rest', False))
            num_pitched = len(self.pitch_sequence) - num_rests
            print(f"After adding rests: {num_pitched} pitched + {num_rests} rest = {len(self.pitch_sequence)} total segments")

        if len(self.pitch_sequence) > 0:
            print(f"First segment: {self.pitch_sequence[0]['start_time']:.3f}s")
            print(f"Last segment: {self.pitch_sequence[-1]['end_time']:.3f}s")

        return self.pitch_sequence

    def extract_pitch_crepe(self, audio_segment: np.ndarray,
                           sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch using CREPE (deep learning method).

        Args:
            audio_segment: Audio waveform segment
            sr: Sample rate

        Returns:
            (pitch_hz, confidence) arrays
        """
        if not CREPE_AVAILABLE:
            raise ImportError("CREPE not available")

        # CREPE expects at least 1024 samples
        if len(audio_segment) < 1024:
            return np.array([]), np.array([])

        # Run CREPE pitch detection
        time, frequency, confidence, activation = crepe.predict(
            audio_segment,
            sr,
            viterbi=True,  # Temporal smoothing
            model_capacity='full',  # Most accurate model
            step_size=10  # 10ms steps
        )

        return frequency, confidence

    def extract_pitch_pyin(self, audio_segment: np.ndarray,
                          sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch using librosa's pyin (fallback method).

        Args:
            audio_segment: Audio waveform segment
            sr: Sample rate

        Returns:
            (pitch_hz, voiced_flag) arrays
        """
        # pyin returns frequency and voiced flag
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_segment,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz (low male voice)
            fmax=librosa.note_to_hz('C6'),  # ~1046 Hz (high female voice)
            sr=sr,
            frame_length=2048
        )

        # Use voiced probabilities as confidence
        confidence = voiced_probs

        # Replace unvoiced frames with 0
        frequency = np.where(voiced_flag, f0, 0)

        return frequency, confidence

    def analyze_segment_pitch(self, start_time: float, end_time: float,
                             min_confidence: float = 0.5,
                             min_rms_db: float = -40.0) -> Optional[Dict]:
        """
        Analyze pitch for a single onset segment.

        Args:
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            min_confidence: Minimum confidence threshold
            min_rms_db: Minimum RMS amplitude in dB (default: -40dB, filters silence)

        Returns:
            Dictionary with pitch statistics, or None if segment is too short
        """
        duration = end_time - start_time

        # Skip very short segments (< 50ms)
        if duration < 0.05:
            return None

        # Extract audio segment
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)
        audio_segment = self.audio[start_sample:end_sample]

        if len(audio_segment) < 100:  # Too short
            return None

        # Check if segment is silence (RMS amplitude too low)
        rms = np.sqrt(np.mean(audio_segment**2))
        rms_db = 20 * np.log10(rms + 1e-10)  # Convert to dB (add epsilon to avoid log(0))

        if rms_db < min_rms_db:
            # Segment is too quiet, likely silence
            return None

        # Extract pitch
        try:
            if self.pitch_method == 'crepe':
                pitch_hz, confidence = self.extract_pitch_crepe(audio_segment, self.sr)
            else:
                pitch_hz, confidence = self.extract_pitch_pyin(audio_segment, self.sr)
        except Exception as e:
            print(f"Warning: Pitch extraction failed for segment {start_time:.3f}-{end_time:.3f}: {e}")
            return None

        if len(pitch_hz) == 0:
            return None

        # Calculate pitch statistics
        stats = calculate_pitch_statistics(
            pitch_hz,
            confidence,
            confidence_threshold=min_confidence
        )

        # Check if we have enough valid pitch data
        if stats['num_valid_frames'] < 3:
            return None

        # Check if pitch is in reasonable singing range
        if not is_pitch_in_range(stats['median_hz'], 'C2', 'C6'):
            return None

        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'pitch_hz': stats['median_hz'],
            'pitch_midi': stats['median_midi'],
            'pitch_note': stats['note_name'],
            'pitch_confidence': stats['mean_confidence'],
            'pitch_stability': stats['stability'],
            'num_pitch_frames': stats['num_valid_frames']
        }

    def analyze_full_sequence(self, min_confidence: float = 0.5,
                             merge_same_pitch: bool = True,
                             merge_threshold_cents: float = 50.0) -> List[Dict]:
        """
        Analyze pitch for entire onset sequence.

        Args:
            min_confidence: Minimum pitch confidence threshold
            merge_same_pitch: Merge consecutive segments with same pitch
            merge_threshold_cents: Max pitch difference for merging (cents)

        Returns:
            List of pitch segment dictionaries
        """
        if self.onset_times is None or len(self.onset_times) == 0:
            raise ValueError("Must detect onsets first")

        print(f"\nAnalyzing pitch for {len(self.onset_times)} onset segments...")

        segments = []

        # Analyze each onset-to-onset segment
        for i in range(len(self.onset_times)):
            start_time = self.onset_times[i]

            # End time is next onset, or end of audio
            if i < len(self.onset_times) - 1:
                end_time = self.onset_times[i + 1]
            else:
                end_time = len(self.audio) / self.sr

            # Analyze pitch for this segment
            segment = self.analyze_segment_pitch(start_time, end_time, min_confidence)

            if segment is not None:
                # Get onset strength for this segment
                onset_frame = int(start_time * self.fps)
                if onset_frame < len(self.onset_strength):
                    segment['onset_strength'] = float(self.onset_strength[onset_frame])
                else:
                    segment['onset_strength'] = 0.0

                segments.append(segment)

        print(f"Valid pitch segments: {len(segments)}")

        # Optionally merge consecutive same-pitch segments
        if merge_same_pitch and len(segments) > 1:
            segments = self._merge_same_pitch_segments(segments, merge_threshold_cents)
            print(f"After merging same pitches: {len(segments)} segments")

        # Add index to each segment
        for i, seg in enumerate(segments):
            seg['index'] = i

        self.pitch_sequence = segments
        return segments

    def _merge_same_pitch_segments(self, segments: List[Dict],
                                   threshold_cents: float = 50.0) -> List[Dict]:
        """
        Merge consecutive segments with same pitch.

        Args:
            segments: List of pitch segments
            threshold_cents: Max pitch difference for merging

        Returns:
            Merged segment list
        """
        from pitch_utils import pitch_distance_cents

        if len(segments) <= 1:
            return segments

        merged = [segments[0]]

        for seg in segments[1:]:
            prev = merged[-1]

            # Check if pitches are similar
            pitch_diff = abs(pitch_distance_cents(prev['pitch_hz'], seg['pitch_hz']))

            # Merge if same MIDI note and within threshold
            if (prev['pitch_midi'] == seg['pitch_midi'] and
                pitch_diff < threshold_cents):

                # Merge: extend the previous segment
                merged[-1] = {
                    'start_time': prev['start_time'],
                    'end_time': seg['end_time'],
                    'duration': seg['end_time'] - prev['start_time'],
                    'pitch_hz': (prev['pitch_hz'] + seg['pitch_hz']) / 2,  # Average
                    'pitch_midi': prev['pitch_midi'],
                    'pitch_note': prev['pitch_note'],
                    'pitch_confidence': max(prev['pitch_confidence'], seg['pitch_confidence']),
                    'pitch_stability': (prev['pitch_stability'] + seg['pitch_stability']) / 2,
                    'onset_strength': max(prev['onset_strength'], seg['onset_strength']),
                    'num_pitch_frames': prev['num_pitch_frames'] + seg['num_pitch_frames']
                }
            else:
                # Different pitch, keep as separate segment
                merged.append(seg)

        return merged

    def _get_rms_db(self, start_time: float, end_time: float) -> float:
        """
        Calculate RMS amplitude in dB for a time range.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            RMS amplitude in dB
        """
        if self.audio is None or self.sr is None:
            return 0.0

        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)

        # Clamp to valid range
        start_sample = max(0, start_sample)
        end_sample = min(len(self.audio), end_sample)

        if end_sample <= start_sample:
            return -100.0  # Treat as silent

        audio_segment = self.audio[start_sample:end_sample]
        rms = np.sqrt(np.mean(audio_segment ** 2))

        if rms < 1e-10:
            return -100.0  # Effectively silent

        return 20 * np.log10(rms)

    def _insert_rest_segments(self,
                              pitch_segments: List[Dict],
                              min_rest_duration: float = 0.1,
                              verify_rms: bool = True,
                              silence_threshold_db: float = -50.0) -> List[Dict]:
        """
        Insert rest segments for gaps between pitched segments.

        Args:
            pitch_segments: List of pitched segment dictionaries (already sorted by time)
            min_rest_duration: Minimum gap duration to create a rest (seconds)
            verify_rms: If True, verify the gap is actually silent via RMS check
            silence_threshold_db: RMS threshold below which audio is considered silent

        Returns:
            New list with rest segments interleaved
        """
        if not pitch_segments:
            return pitch_segments

        result = []
        current_time = 0.0  # Start from beginning of audio

        for seg in pitch_segments:
            gap_start = current_time
            gap_end = seg['start_time']
            gap_duration = gap_end - gap_start

            # Check if gap is long enough to be a rest
            if gap_duration >= min_rest_duration:
                # Optional: verify gap is actually silent
                is_silence = True
                if verify_rms and self.audio is not None:
                    rms_db = self._get_rms_db(gap_start, gap_end)
                    is_silence = rms_db < silence_threshold_db

                if is_silence:
                    rest_segment = {
                        'start_time': gap_start,
                        'end_time': gap_end,
                        'duration': gap_duration,
                        'pitch_hz': 0.0,
                        'pitch_midi': -1,
                        'pitch_note': 'REST',
                        'pitch_confidence': 1.0,
                        'is_rest': True
                    }
                    result.append(rest_segment)

            # Add the pitched segment (with is_rest: false for clarity)
            seg['is_rest'] = False
            result.append(seg)
            current_time = seg['end_time']

        # Check for trailing rest (gap after last segment to end of audio)
        if self.audio is not None:
            audio_end = len(self.audio) / self.sr
            trailing_gap = audio_end - current_time

            if trailing_gap >= min_rest_duration:
                is_silence = True
                if verify_rms:
                    rms_db = self._get_rms_db(current_time, audio_end)
                    is_silence = rms_db < silence_threshold_db

                if is_silence:
                    rest_segment = {
                        'start_time': current_time,
                        'end_time': audio_end,
                        'duration': trailing_gap,
                        'pitch_hz': 0.0,
                        'pitch_midi': -1,
                        'pitch_note': 'REST',
                        'pitch_confidence': 1.0,
                        'is_rest': True
                    }
                    result.append(rest_segment)

        # Re-index all segments
        for i, seg in enumerate(result):
            seg['index'] = i

        return result

    def save_results(self, output_path: str):
        """
        Save pitch sequence to JSON file.

        Args:
            output_path: Output JSON file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Count pitched vs rest segments
        num_rests = sum(1 for s in self.pitch_sequence if s.get('is_rest', False))
        num_pitched = len(self.pitch_sequence) - num_rests

        # Calculate durations
        pitched_segments = [s for s in self.pitch_sequence if not s.get('is_rest', False)]
        rest_segments = [s for s in self.pitch_sequence if s.get('is_rest', False)]
        musical_duration = sum(s['duration'] for s in pitched_segments)
        rest_duration = sum(s['duration'] for s in rest_segments)

        # Prepare output data
        data = {
            'video_path': str(self.video_path),
            'audio_path': str(self.audio_path),
            'fps': self.fps,
            'sample_rate': self.sr,
            'pitch_detection_method': self.pitch_method.upper(),
            'num_segments': len(self.pitch_sequence),
            'num_pitched_segments': num_pitched,
            'num_rest_segments': num_rests,
            'total_duration': float(self.audio.shape[0] / self.sr) if self.audio is not None else 0,
            'musical_duration': musical_duration,
            'rest_duration': rest_duration,
            'guide_sequence': self.pitch_sequence
        }

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved pitch sequence to: {output_path}")
        print(f"Total segments: {len(self.pitch_sequence)} ({num_pitched} pitched, {num_rests} rests)")

        # Print statistics
        if num_pitched > 0:
            pitches = [s['pitch_midi'] for s in pitched_segments]
            confidences = [s['pitch_confidence'] for s in pitched_segments]

            print(f"Pitch range: {midi_to_note_name(min(pitches))} to {midi_to_note_name(max(pitches))}")
            print(f"Average confidence: {np.mean(confidences):.3f}")
            print(f"Musical duration: {musical_duration:.2f}s")
            if num_rests > 0:
                print(f"Rest duration: {rest_duration:.2f}s")

    def print_sequence_summary(self, limit: int = 20):
        """Print summary of detected pitch sequence."""
        if len(self.pitch_sequence) == 0:
            print("No pitch sequence detected")
            return

        print(f"\n=== Pitch Sequence Summary ===")
        print(f"Total segments: {len(self.pitch_sequence)}")

        print(f"\nFirst {min(limit, len(self.pitch_sequence))} segments:")
        print(f"{'Idx':<5} {'Time':<12} {'Dur':<7} {'Note':<6} {'Hz':<8} {'Conf':<6}")
        print("-" * 55)

        for seg in self.pitch_sequence[:limit]:
            print(f"{seg['index']:<5} "
                  f"{seg['start_time']:>5.2f}-{seg['end_time']:<4.2f} "
                  f"{seg['duration']:<7.3f} "
                  f"{seg['pitch_note']:<6} "
                  f"{seg['pitch_hz']:<8.1f} "
                  f"{seg['pitch_confidence']:<6.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze guide video to extract target pitch sequence"
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to guide video file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/segments/guide_sequence.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=24,
        help='Video frame rate (default: 24)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=50.0,
        help='Pitch change threshold in cents (default: 50, lower = more segments)'
    )
    parser.add_argument(
        '--min-duration',
        type=float,
        default=0.1,
        help='Minimum segment duration in seconds (default: 0.1)'
    )
    parser.add_argument(
        '--silence-threshold',
        type=float,
        default=-50.0,
        help='Silence threshold in dB (default: -50, lower = more permissive)'
    )
    parser.add_argument(
        '--pitch-smoothing',
        type=int,
        default=0,
        help='Pitch smoothing window size (0=off, 5-7 recommended to reduce vibrato/waver)'
    )
    parser.add_argument(
        '--pitch-method',
        type=str,
        default='crepe',
        choices=['crepe', 'basic-pitch', 'swift-f0', 'hybrid', 'rmvpe', 'rmvpe-crepe', 'pyin'],
        help='Pitch detection method (default: crepe). Options: crepe (accurate), basic-pitch (multipitch), swift-f0 (fast CPU), hybrid (crepe+swift-f0), rmvpe (fast vocal), rmvpe-crepe (rmvpe+crepe hybrid), pyin (fallback)'
    )
    parser.add_argument(
        '--use-pyin',
        action='store_true',
        help='(Deprecated) Use pYIN instead of CREPE - use --pitch-method pyin instead'
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default='data/temp',
        help='Temporary directory for extracted audio'
    )
    parser.add_argument(
        '--min-rest-duration',
        type=float,
        default=0.1,
        help='Minimum gap duration to create rest segment (default: 0.1s)'
    )
    parser.add_argument(
        '--no-rest-segments',
        action='store_true',
        help='Disable rest segment detection (only output pitched segments)'
    )
    parser.add_argument(
        '--no-verify-rest-rms',
        action='store_true',
        help='Skip RMS verification for rest segments (faster, less accurate)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device for neural network models (default: auto). Use cuda for GPU acceleration'
    )

    args = parser.parse_args()

    print("=== Pitch Guide Analyzer ===\n")
    print(f"Guide video: {args.video}")
    print(f"Output: {args.output}")

    # Handle deprecated --use-pyin flag
    pitch_method = args.pitch_method
    if args.use_pyin:
        pitch_method = 'pyin'
        print("Note: --use-pyin is deprecated, use --pitch-method pyin instead")

    print(f"Pitch detection: {pitch_method.upper()}")
    print(f"Device: {args.device}\n")

    # Initialize analyzer
    analyzer = PitchGuideAnalyzer(args.video, pitch_method=pitch_method, device=args.device)

    # Step 1: Extract audio
    analyzer.extract_audio(output_dir=args.temp_dir)

    # Step 2: Analyze pitch changes
    analyzer.analyze_pitch_changes(
        fps=args.fps,
        pitch_change_threshold=args.threshold,
        min_segment_duration=args.min_duration,
        min_rms_db=args.silence_threshold,
        pitch_smoothing=args.pitch_smoothing,
        min_rest_duration=args.min_rest_duration,
        include_rests=not args.no_rest_segments,
        verify_rest_rms=not args.no_verify_rest_rms
    )

    # Step 3: Print summary
    analyzer.print_sequence_summary(limit=30)

    # Step 4: Save results
    analyzer.save_results(args.output)

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
