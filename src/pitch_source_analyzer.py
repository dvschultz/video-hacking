#!/usr/bin/env python3
"""
Pitch Source Analyzer

Analyzes a "source" video to build a comprehensive pitch database.
Detects ALL pitches available in the video and indexes them for fast lookup.

This creates the searchable database that we'll use to find matching clips
for each note in the guide sequence.
"""

import argparse
import json
import numpy as np
import librosa
import soundfile as sf
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Import existing onset detection
from onset_strength_analysis import OnsetStrengthAnalyzer
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


class PitchSourceAnalyzer:
    """Analyzes source video to build searchable pitch database."""

    def __init__(self, video_path: str, use_crepe: bool = True):
        """
        Initialize the analyzer.

        Args:
            video_path: Path to source video file
            use_crepe: Use CREPE for pitch detection (more accurate)
        """
        self.video_path = Path(video_path)
        self.audio_path = None
        self.use_crepe = use_crepe and CREPE_AVAILABLE

        self.audio = None
        self.sr = None
        self.fps = None

        self.onset_analyzer = None
        self.onset_times = None
        self.onset_strength = None

        self.pitch_database = []
        self.pitch_index = defaultdict(list)  # MIDI note -> list of clip IDs

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

        audio_filename = self.video_path.stem + "_source_audio.wav"
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

    def detect_onsets(self, fps: int = 24, threshold: float = 0.12,
                     power: float = 0.6) -> np.ndarray:
        """
        Detect ALL onset times in the source audio.

        Uses a lower threshold than guide analysis to capture every possible clip.

        Args:
            fps: Video frame rate
            threshold: Onset strength threshold (lower = more clips)
            power: Power-law compression (lower = more sensitive)

        Returns:
            Array of onset times in seconds
        """
        if self.audio_path is None:
            raise ValueError("Must extract audio first")

        self.fps = fps

        print(f"\nDetecting onsets (threshold={threshold}, power={power})...")
        print("Note: Using lower threshold to capture all possible clips")

        # Use existing onset detection code
        self.onset_analyzer = OnsetStrengthAnalyzer(str(self.audio_path), sr=self.sr)
        onset_strength = self.onset_analyzer.analyze(
            analysis_rate=fps,
            power=power,
            smoothing_window_size=1,
            smoothing_tolerance=0.2
        )

        self.onset_strength = onset_strength
        self.onset_times = self.onset_analyzer.get_cut_times(threshold=threshold)

        print(f"Detected {len(self.onset_times)} onsets in source video")

        if len(self.onset_times) > 0:
            print(f"First onset: {self.onset_times[0]:.3f}s")
            print(f"Last onset: {self.onset_times[-1]:.3f}s")
            avg_gap = np.mean(np.diff(self.onset_times)) if len(self.onset_times) > 1 else 0
            print(f"Average gap between onsets: {avg_gap:.3f}s")

        return self.onset_times

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

    def analyze_clip_pitch(self, clip_id: int, start_time: float, end_time: float,
                          min_confidence: float = 0.5,
                          min_rms_db: float = -40.0) -> Optional[Dict]:
        """
        Analyze pitch for a single clip in the source database.

        Args:
            clip_id: Unique clip identifier
            start_time: Clip start time in seconds
            end_time: Clip end time in seconds
            min_confidence: Minimum confidence threshold
            min_rms_db: Minimum RMS amplitude in dB (default: -40dB, filters silence)

        Returns:
            Dictionary with pitch statistics and video info, or None if invalid
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
            if self.use_crepe:
                pitch_hz, confidence = self.extract_pitch_crepe(audio_segment, self.sr)
            else:
                pitch_hz, confidence = self.extract_pitch_pyin(audio_segment, self.sr)
        except Exception as e:
            # Silently skip failed segments (too verbose otherwise)
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

        # Calculate video frame numbers
        video_start_frame = int(start_time * self.fps)
        video_end_frame = int(end_time * self.fps)

        # Get onset strength for this clip
        onset_frame = int(start_time * self.fps)
        if onset_frame < len(self.onset_strength):
            onset_strength_val = float(self.onset_strength[onset_frame])
        else:
            onset_strength_val = 0.0

        return {
            'clip_id': clip_id,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'pitch_hz': stats['median_hz'],
            'pitch_midi': stats['median_midi'],
            'pitch_note': stats['note_name'],
            'pitch_confidence': stats['mean_confidence'],
            'pitch_stability': stats['stability'],
            'video_start_frame': video_start_frame,
            'video_end_frame': video_end_frame,
            'onset_strength': onset_strength_val,
            'num_pitch_frames': stats['num_valid_frames']
        }

    def build_database(self, min_confidence: float = 0.5,
                      max_clips: Optional[int] = None) -> List[Dict]:
        """
        Build comprehensive pitch database from all onsets.

        Args:
            min_confidence: Minimum pitch confidence threshold
            max_clips: Optional limit on number of clips (for testing)

        Returns:
            List of clip dictionaries
        """
        if self.onset_times is None or len(self.onset_times) == 0:
            raise ValueError("Must detect onsets first")

        print(f"\nBuilding pitch database from {len(self.onset_times)} onsets...")

        clips = []
        clip_id = 0

        # Analyze each onset-to-onset segment
        num_to_process = len(self.onset_times)
        if max_clips is not None:
            num_to_process = min(num_to_process, max_clips)

        for i in range(num_to_process):
            start_time = self.onset_times[i]

            # End time is next onset, or end of audio
            if i < len(self.onset_times) - 1:
                end_time = self.onset_times[i + 1]
            else:
                end_time = len(self.audio) / self.sr

            # Analyze pitch for this clip
            clip = self.analyze_clip_pitch(clip_id, start_time, end_time, min_confidence)

            if clip is not None:
                clips.append(clip)
                clip_id += 1

            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{num_to_process} onsets, found {len(clips)} valid clips...")

        print(f"Database complete: {len(clips)} valid clips from {num_to_process} onsets")
        print(f"Success rate: {len(clips)/num_to_process*100:.1f}%")

        self.pitch_database = clips
        return clips

    def build_pitch_index(self):
        """
        Build MIDI note index for fast pitch lookup.

        Creates a mapping: MIDI note number -> list of clip IDs with that pitch
        """
        print("\nBuilding pitch index...")

        self.pitch_index = defaultdict(list)

        for clip in self.pitch_database:
            midi = clip['pitch_midi']
            clip_id = clip['clip_id']
            self.pitch_index[midi].append(clip_id)

        # Sort clip IDs for each MIDI note
        for midi in self.pitch_index:
            self.pitch_index[midi].sort()

        # Print statistics
        print(f"Indexed {len(self.pitch_index)} unique MIDI notes")

        # Find pitch range
        if len(self.pitch_index) > 0:
            min_midi = min(self.pitch_index.keys())
            max_midi = max(self.pitch_index.keys())
            print(f"Pitch range: {midi_to_note_name(min_midi)} (MIDI {min_midi}) to "
                  f"{midi_to_note_name(max_midi)} (MIDI {max_midi})")

            # Find most common pitches
            pitch_counts = [(midi, len(clips)) for midi, clips in self.pitch_index.items()]
            pitch_counts.sort(key=lambda x: x[1], reverse=True)

            print(f"\nMost common pitches:")
            for midi, count in pitch_counts[:10]:
                print(f"  {midi_to_note_name(midi)}: {count} clips")

    def save_database(self, output_path: str):
        """
        Save pitch database to JSON file.

        Args:
            output_path: Output JSON file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert defaultdict to regular dict for JSON serialization
        pitch_index_dict = {str(midi): clips for midi, clips in self.pitch_index.items()}

        # Prepare output data
        data = {
            'video_path': str(self.video_path),
            'audio_path': str(self.audio_path),
            'fps': self.fps,
            'sample_rate': self.sr,
            'pitch_detection_method': 'CREPE' if self.use_crepe else 'pYIN',
            'num_clips': len(self.pitch_database),
            'num_unique_pitches': len(self.pitch_index),
            'total_duration': float(self.audio.shape[0] / self.sr) if self.audio is not None else 0,
            'pitch_database': self.pitch_database,
            'pitch_index': pitch_index_dict
        }

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved pitch database to: {output_path}")
        print(f"Total clips: {len(self.pitch_database)}")
        print(f"Unique pitches: {len(self.pitch_index)}")

        # Print statistics
        if len(self.pitch_database) > 0:
            durations = [clip['duration'] for clip in self.pitch_database]
            confidences = [clip['pitch_confidence'] for clip in self.pitch_database]

            print(f"\nClip statistics:")
            print(f"  Average duration: {np.mean(durations):.3f}s")
            print(f"  Duration range: {np.min(durations):.3f}s to {np.max(durations):.3f}s")
            print(f"  Average confidence: {np.mean(confidences):.3f}")
            print(f"  Total musical duration: {sum(durations):.2f}s")

    def print_database_summary(self, samples_per_pitch: int = 3):
        """
        Print summary of pitch database.

        Args:
            samples_per_pitch: Number of sample clips to show per pitch
        """
        if len(self.pitch_database) == 0:
            print("No clips in database")
            return

        print(f"\n=== Pitch Database Summary ===")
        print(f"Total clips: {len(self.pitch_database)}")
        print(f"Unique pitches: {len(self.pitch_index)}")

        # Sample clips from each pitch
        print(f"\nSample clips (showing {samples_per_pitch} per pitch):")

        # Sort pitches by MIDI number
        sorted_pitches = sorted(self.pitch_index.keys())

        for midi in sorted_pitches[:15]:  # Show first 15 pitches
            note = midi_to_note_name(midi)
            clip_ids = self.pitch_index[midi]

            print(f"\n{note} (MIDI {midi}): {len(clip_ids)} clips")

            # Show sample clips
            for clip_id in clip_ids[:samples_per_pitch]:
                clip = self.pitch_database[clip_id]
                print(f"  Clip {clip_id}: {clip['start_time']:.2f}-{clip['end_time']:.2f}s, "
                      f"dur={clip['duration']:.3f}s, conf={clip['pitch_confidence']:.3f}")

        if len(sorted_pitches) > 15:
            print(f"\n... and {len(sorted_pitches) - 15} more pitches")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze source video to build searchable pitch database"
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to source video file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/segments/source_database.json',
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
        default=0.12,
        help='Onset detection threshold (default: 0.12, lower = more clips)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Minimum pitch confidence (default: 0.5)'
    )
    parser.add_argument(
        '--use-pyin',
        action='store_true',
        help='Use pYIN instead of CREPE for pitch detection'
    )
    parser.add_argument(
        '--max-clips',
        type=int,
        help='Maximum number of clips to process (for testing)'
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default='data/temp',
        help='Temporary directory for extracted audio'
    )

    args = parser.parse_args()

    print("=== Pitch Source Database Builder ===\n")
    print(f"Source video: {args.video}")
    print(f"Output: {args.output}")
    print(f"Pitch detection: {'pYIN' if args.use_pyin else 'CREPE'}\n")

    # Initialize analyzer
    analyzer = PitchSourceAnalyzer(args.video, use_crepe=not args.use_pyin)

    # Step 1: Extract audio
    analyzer.extract_audio(output_dir=args.temp_dir)

    # Step 2: Detect ALL onsets (lower threshold for more coverage)
    analyzer.detect_onsets(fps=args.fps, threshold=args.threshold)

    # Step 3: Build comprehensive pitch database
    analyzer.build_database(
        min_confidence=args.min_confidence,
        max_clips=args.max_clips
    )

    # Step 4: Build pitch index for fast lookup
    analyzer.build_pitch_index()

    # Step 5: Print summary
    analyzer.print_database_summary(samples_per_pitch=3)

    # Step 6: Save database
    analyzer.save_database(args.output)

    print("\n=== Database Building Complete ===")


if __name__ == "__main__":
    main()
