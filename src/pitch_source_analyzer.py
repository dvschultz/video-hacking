#!/usr/bin/env python3
"""
Pitch Source Analyzer

Analyzes source videos to build a comprehensive pitch database.
Uses continuous pitch tracking to detect ALL pitches and silent sections.

This creates the searchable database that we'll use to find matching clips
for each note in the guide sequence.
"""

import argparse
import json
import numpy as np
import librosa
import subprocess
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# Import pitch change detection
from pitch_change_detector import PitchChangeDetector
from pitch_utils import (
    hz_to_midi, midi_to_note_name, round_to_nearest_midi,
    calculate_pitch_statistics, is_pitch_in_range
)


class PitchSourceAnalyzer:
    """Analyzes source video to build searchable pitch database."""

    def __init__(self, video_path: str, pitch_method: str = 'crepe'):
        """
        Initialize the analyzer.

        Args:
            video_path: Path to source video file
            pitch_method: Pitch detection method ('crepe', 'basic-pitch', 'swift-f0', 'hybrid', or 'pyin')
        """
        self.video_path = Path(video_path)
        self.audio_path = None
        self.pitch_method = pitch_method

        self.audio = None
        self.sr = None
        self.fps = None

        self.pitch_database = []
        self.pitch_index = defaultdict(list)  # MIDI note -> list of segment IDs
        self.silence_segments = []  # Track silent sections

    def detect_video_fps(self) -> float:
        """
        Auto-detect video frame rate using ffprobe.

        Returns:
            Frame rate as float (e.g., 29.97, 24.0, 30.0)
        """
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'csv=p=0',
            str(self.video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
            fps_str = result.stdout.strip()
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            return fps
        except (subprocess.CalledProcessError, ValueError, ZeroDivisionError) as e:
            print(f"Warning: Could not detect video fps: {e}")
            print("Falling back to 24 fps")
            return 24.0

    def extract_audio(self, output_dir: str = "data/temp",
                     sample_rate: int = 22050,
                     normalize: bool = False,
                     target_lufs: float = -16.0) -> Path:
        """
        Extract audio from video file.

        Args:
            output_dir: Directory to save extracted audio
            sample_rate: Audio sample rate (22050 Hz is good for pitch detection)
            normalize: If True, normalize audio loudness using EBU R128 standard
            target_lufs: Target loudness in LUFS (default -16, broadcast standard)

        Returns:
            Path to extracted audio file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_filename = self.video_path.stem + "_source_audio.wav"
        audio_path = output_dir / audio_filename

        print(f"Extracting audio from: {self.video_path}")
        if normalize:
            print(f"  Normalizing to {target_lufs} LUFS")
        print(f"Output: {audio_path}")

        # Build ffmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-i', str(self.video_path),
            '-vn',  # No video
        ]

        # Add loudnorm filter for EBU R128 normalization if requested
        if normalize:
            af_filter = f'loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=summary'
            cmd.extend(['-af', af_filter])

        cmd.extend([
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono
            str(audio_path)
        ])

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

    def analyze_continuous_pitch(self, fps: float = None,
                                  pitch_change_threshold: float = 50.0,
                                  min_segment_duration: float = 0.1,
                                  min_rms_db: float = -50.0,
                                  pitch_smoothing: int = 0,
                                  split_on_rms_dips: bool = True,
                                  rms_dip_threshold_db: float = -6.0,
                                  min_dip_duration: float = 0.02,
                                  rms_window_ms: float = 50.0) -> List[Dict]:
        """
        Analyze continuous pitch throughout the source video.

        Args:
            fps: Video frame rate (None = auto-detect from video)
            pitch_change_threshold: Cents difference to trigger new segment
            min_segment_duration: Minimum segment length in seconds
            min_rms_db: Minimum RMS amplitude to detect silence
            pitch_smoothing: Median filter window size (0=off, 5-7 recommended)
            split_on_rms_dips: Split segments on RMS dips (volume drops)
            rms_dip_threshold_db: Dip threshold in dB below local mean
            min_dip_duration: Minimum dip duration in seconds
            rms_window_ms: Rolling window size in ms for local mean

        Returns:
            List of pitch segment dictionaries
        """
        if self.audio_path is None:
            raise ValueError("Must extract audio first")

        # Auto-detect fps if not provided
        if fps is None:
            fps = self.detect_video_fps()
            print(f"Auto-detected video fps: {fps:.3f}")

        self.fps = fps

        print(f"\nAnalyzing continuous pitch (threshold={pitch_change_threshold} cents)...")

        # Use pitch change detector
        detector = PitchChangeDetector(str(self.audio_path), sr=self.sr, pitch_method=self.pitch_method)

        # Extract continuous pitch with optional smoothing
        detector.extract_continuous_pitch(frame_time=0.01, pitch_smoothing_window=pitch_smoothing)

        # Detect pitch segments (splits on pitch changes, silence, OR RMS dips)
        segments = detector.detect_pitch_segments(
            pitch_change_threshold_cents=pitch_change_threshold,
            min_segment_duration=min_segment_duration,
            min_confidence=0.5,
            min_rms_db=min_rms_db,
            split_on_rms_dips=split_on_rms_dips,
            rms_dip_threshold_db=rms_dip_threshold_db,
            min_dip_duration=min_dip_duration,
            rms_window_ms=rms_window_ms
        )

        # Convert to dictionaries
        pitch_segments = detector.segments_to_dict(segments)

        # Track verified silence gaps between segments
        self._extract_silence_gaps(pitch_segments, silence_threshold_db=min_rms_db)

        print(f"Detected {len(pitch_segments)} pitch segments")
        print(f"Detected {len(self.silence_segments)} verified silent gaps")

        return pitch_segments

    def _get_rms_db(self, start_time: float, end_time: float) -> float:
        """
        Calculate RMS amplitude in dB for a time range.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            RMS amplitude in dB (negative values, -100 for silence)
        """
        if self.audio is None or self.sr is None:
            return 0.0

        start_sample = max(0, int(start_time * self.sr))
        end_sample = min(len(self.audio), int(end_time * self.sr))

        if end_sample <= start_sample:
            return -100.0

        audio_segment = self.audio[start_sample:end_sample]
        rms = np.sqrt(np.mean(audio_segment ** 2))

        if rms < 1e-10:
            return -100.0

        return 20 * np.log10(rms)

    def _extract_silence_gaps(self, pitch_segments: List[Dict], silence_threshold_db: float = -40.0):
        """
        Extract verified silent gaps between pitch segments.

        Only includes gaps where the audio RMS is below the silence threshold.

        Args:
            pitch_segments: List of pitch segment dictionaries
            silence_threshold_db: Maximum RMS in dB to consider as silence (default -40)
        """
        self.silence_segments = []

        if len(pitch_segments) < 2:
            return

        gaps_checked = 0

        for i in range(len(pitch_segments) - 1):
            gap_start = pitch_segments[i]['end_time']
            gap_end = pitch_segments[i + 1]['start_time']
            gap_duration = gap_end - gap_start

            if gap_duration <= 0.05:
                continue

            gaps_checked += 1
            rms_db = self._get_rms_db(gap_start, gap_end)

            if rms_db < silence_threshold_db:
                self.silence_segments.append({
                    'start_time': gap_start,
                    'end_time': gap_end,
                    'duration': gap_duration,
                    'video_start_frame': int(gap_start * self.fps),
                    'video_end_frame': int(gap_end * self.fps),
                    'rms_db': rms_db
                })

        rejected_count = gaps_checked - len(self.silence_segments)
        print(f"  Verified {len(self.silence_segments)} silent gaps (rejected {rejected_count} with audio)")

    def build_database(self, pitch_segments: List[Dict]) -> List[Dict]:
        """
        Build comprehensive pitch database from pitch segments.

        Args:
            pitch_segments: List of pitch segment dictionaries from continuous pitch analysis

        Returns:
            List of database entries with video frame info, volume, and consistency metrics
        """
        print(f"\nBuilding pitch database from {len(pitch_segments)} segments...")

        database = []

        for i, seg in enumerate(pitch_segments):
            # Add video frame information
            video_start_frame = int(seg['start_time'] * self.fps)
            video_end_frame = int(seg['end_time'] * self.fps)

            # Calculate median RMS volume for this segment
            start_sample = int(seg['start_time'] * self.sr)
            end_sample = int(seg['end_time'] * self.sr)
            audio_segment = self.audio[start_sample:end_sample]

            # Calculate overall RMS in dB
            rms = np.sqrt(np.mean(audio_segment**2))
            rms_db = 20 * np.log10(rms + 1e-10)

            # === Calculate RMS consistency (stability of volume) ===
            # Use frame-by-frame RMS variance
            frame_size = int(0.01 * self.sr)  # 10ms frames
            hop_size = frame_size // 2  # 5ms hop
            rms_values = []

            for j in range(0, len(audio_segment) - frame_size, hop_size):
                frame = audio_segment[j:j + frame_size]
                frame_rms = np.sqrt(np.mean(frame ** 2))
                if frame_rms > 0:
                    rms_values.append(frame_rms)

            if len(rms_values) > 1:
                rms_values = np.array(rms_values)
                rms_mean = np.mean(rms_values)
                rms_std = np.std(rms_values)
                cv_rms = rms_std / (rms_mean + 1e-10)  # Coefficient of variation
                # Score: 1.0 for perfectly stable, lower for more variable
                rms_consistency = float(1.0 / (1.0 + cv_rms * 5))
            else:
                rms_consistency = 1.0

            # === Calculate pitch consistency ===
            # Use pitch_confidence as a proxy (high confidence = stable pitch)
            # A more accurate approach would track per-frame pitch variance
            # but that would require storing detector state
            pitch_consistency = float(seg['pitch_confidence'])

            # === Combined loopability score ===
            # Equal weight to RMS and pitch consistency
            loopability = float(0.5 * rms_consistency + 0.5 * pitch_consistency)

            entry = {
                'segment_id': i,
                'start_time': seg['start_time'],
                'end_time': seg['end_time'],
                'duration': seg['duration'],
                'pitch_hz': seg['pitch_hz'],
                'pitch_midi': seg['pitch_midi'],
                'pitch_note': seg['pitch_note'],
                'pitch_confidence': seg['pitch_confidence'],
                'rms_db': float(rms_db),
                'rms_consistency': rms_consistency,
                'pitch_consistency': pitch_consistency,
                'loopability': loopability,
                'video_start_frame': video_start_frame,
                'video_end_frame': video_end_frame
            }

            database.append(entry)

        self.pitch_database = database
        print(f"Database complete: {len(database)} segments")

        # Print consistency statistics
        if database:
            avg_loopability = np.mean([s['loopability'] for s in database])
            avg_rms_cons = np.mean([s['rms_consistency'] for s in database])
            avg_pitch_cons = np.mean([s['pitch_consistency'] for s in database])
            print(f"  Avg loopability: {avg_loopability:.3f} (RMS: {avg_rms_cons:.3f}, pitch: {avg_pitch_cons:.3f})")

        return database

    def build_pitch_index(self):
        """
        Build MIDI note index for fast pitch lookup.

        Creates a mapping: MIDI note number -> list of segment IDs with that pitch
        """
        print("\nBuilding pitch index...")

        self.pitch_index = defaultdict(list)

        for segment in self.pitch_database:
            midi = segment['pitch_midi']
            segment_id = segment['segment_id']
            self.pitch_index[midi].append(segment_id)

        # Sort segment IDs for each MIDI note
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
            pitch_counts = [(midi, len(segments)) for midi, segments in self.pitch_index.items()]
            pitch_counts.sort(key=lambda x: x[1], reverse=True)

            print(f"\nMost common pitches:")
            for midi, count in pitch_counts[:10]:
                print(f"  {midi_to_note_name(midi)}: {count} segments")

    def _build_pitch_index_from_database(self, database: List[Dict]) -> defaultdict:
        """Build pitch index from a database of segments."""
        pitch_index = defaultdict(list)
        for segment in database:
            pitch_index[segment['pitch_midi']].append(segment['segment_id'])
        for midi in pitch_index:
            pitch_index[midi].sort()
        return pitch_index

    def save_database(self, output_path: str, append: bool = False):
        """
        Save pitch database to JSON file.

        Args:
            output_path: Output JSON file path
            append: If True, merge with existing database instead of overwriting
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate durations for this video
        total_duration = float(self.audio.shape[0] / self.sr) if self.audio is not None else 0
        musical_duration = sum(seg['duration'] for seg in self.pitch_database)
        silence_duration = sum(seg['duration'] for seg in self.silence_segments)

        # Video source info
        video_info = {
            'video_path': str(self.video_path),
            'audio_path': str(self.audio_path),
            'fps': self.fps,
            'sample_rate': self.sr,
            'pitch_detection_method': self.pitch_method.upper(),
            'total_duration': total_duration,
            'musical_duration': musical_duration,
            'silence_duration': silence_duration
        }

        # Add video_path to each segment for tracking
        video_path_str = str(self.video_path)
        for seg in self.pitch_database:
            seg['video_path'] = video_path_str
        for seg in self.silence_segments:
            seg['video_path'] = video_path_str

        # Load existing data and merge if appending
        existing_data = None
        if append and output_path.exists():
            print(f"\nLoading existing database from: {output_path}")
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
            print(f"  Found {existing_data['num_segments']} existing segments from {len(existing_data.get('source_videos', []))} videos")

            # Renumber segment IDs to avoid conflicts
            id_offset = existing_data['num_segments']
            print(f"  Renumbering new segments starting from ID {id_offset}")
            for seg in self.pitch_database:
                seg['segment_id'] += id_offset

            # Merge databases
            self.pitch_database = existing_data['pitch_database'] + self.pitch_database
            self.silence_segments = existing_data.get('silence_segments', []) + self.silence_segments
            self.pitch_index = self._build_pitch_index_from_database(self.pitch_database)

            # Merge source videos and durations
            source_videos = existing_data.get('source_videos', []) + [video_info]
            total_musical = existing_data.get('total_musical_duration', 0) + musical_duration
            total_silence = existing_data.get('total_silence_duration', 0) + silence_duration
        else:
            source_videos = [video_info]
            total_musical = musical_duration
            total_silence = silence_duration

        # Prepare output data
        pitch_index_dict = {str(midi): segment_ids for midi, segment_ids in self.pitch_index.items()}
        data = {
            'source_videos': source_videos,
            'num_videos': len(source_videos),
            'num_segments': len(self.pitch_database),
            'num_unique_pitches': len(self.pitch_index),
            'num_silence_gaps': len(self.silence_segments),
            'total_musical_duration': total_musical,
            'total_silence_duration': total_silence,
            'pitch_database': self.pitch_database,
            'silence_segments': self.silence_segments,
            'pitch_index': pitch_index_dict
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Print summary
        print(f"\nSaved pitch database to: {output_path}")
        if existing_data:
            print(f"Total videos in database: {data['num_videos']}")
        print(f"Total segments: {len(self.pitch_database)}")
        print(f"Unique pitches: {len(self.pitch_index)}")
        print(f"Silent gaps: {len(self.silence_segments)}")

        if self.pitch_database:
            durations = [seg['duration'] for seg in self.pitch_database]
            confidences = [seg['pitch_confidence'] for seg in self.pitch_database]
            print(f"\nCombined database statistics:")
            print(f"  Total segments: {len(self.pitch_database)}")
            print(f"  Average segment duration: {np.mean(durations):.3f}s")
            print(f"  Duration range: {np.min(durations):.3f}s to {np.max(durations):.3f}s")
            print(f"  Average confidence: {np.mean(confidences):.3f}")
            print(f"  Total musical duration: {data['total_musical_duration']:.2f}s")
            print(f"  Total silence duration: {data['total_silence_duration']:.2f}s")

    def print_database_summary(self, samples_per_pitch: int = 3):
        """
        Print summary of pitch database.

        Args:
            samples_per_pitch: Number of sample segments to show per pitch
        """
        if len(self.pitch_database) == 0:
            print("No segments in database")
            return

        print(f"\n=== Pitch Database Summary ===")
        print(f"Total segments: {len(self.pitch_database)}")
        print(f"Unique pitches: {len(self.pitch_index)}")
        print(f"Silent gaps: {len(self.silence_segments)}")

        # Sample segments from each pitch
        print(f"\nSample segments (showing {samples_per_pitch} per pitch):")

        # Sort pitches by MIDI number
        sorted_pitches = sorted(self.pitch_index.keys())

        for midi in sorted_pitches[:15]:  # Show first 15 pitches
            note = midi_to_note_name(midi)
            segment_ids = self.pitch_index[midi]

            print(f"\n{note} (MIDI {midi}): {len(segment_ids)} segments")

            # Show sample segments
            for segment_id in segment_ids[:samples_per_pitch]:
                segment = self.pitch_database[segment_id]
                print(f"  Segment {segment_id}: {segment['start_time']:.2f}-{segment['end_time']:.2f}s, "
                      f"dur={segment['duration']:.3f}s, conf={segment['pitch_confidence']:.3f}")

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
        type=str,
        default='auto',
        help='Video frame rate (default: auto-detect from video). Use "auto" or specify a number like 29.97'
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
        '--append',
        action='store_true',
        help='Append to existing database instead of overwriting (allows combining multiple source videos)'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize audio loudness before analysis (recommended for consistent silence detection)'
    )
    parser.add_argument(
        '--target-lufs',
        type=float,
        default=-16.0,
        help='Target loudness in LUFS when normalizing (default: -16, broadcast standard)'
    )
    parser.add_argument(
        '--no-rms-dip-split',
        action='store_true',
        help='Disable splitting segments on RMS dips (volume drops)'
    )
    parser.add_argument(
        '--rms-dip-threshold',
        type=float,
        default=-6.0,
        help='RMS dip threshold in dB below local mean (default: -6.0, negative = stricter)'
    )
    parser.add_argument(
        '--min-dip-duration',
        type=float,
        default=0.02,
        help='Minimum RMS dip duration in seconds (default: 0.02)'
    )
    parser.add_argument(
        '--rms-window-ms',
        type=float,
        default=50.0,
        help='RMS rolling window size in ms for dip detection (default: 50.0)'
    )

    args = parser.parse_args()

    print("=== Pitch Source Database Builder ===\n")
    print(f"Source video: {args.video}")
    print(f"Output: {args.output}")

    # Handle deprecated --use-pyin flag
    pitch_method = args.pitch_method
    if args.use_pyin:
        pitch_method = 'pyin'
        print("Note: --use-pyin is deprecated, use --pitch-method pyin instead")

    print(f"Pitch detection: {pitch_method.upper()}")
    if args.normalize:
        print(f"Audio normalization: ON (target {args.target_lufs} LUFS)")

    # Parse fps argument
    if args.fps.lower() == 'auto':
        fps = None  # Will auto-detect
        print("FPS: auto-detect")
    else:
        try:
            fps = float(args.fps)
            print(f"FPS: {fps}")
        except ValueError:
            print(f"Warning: Invalid fps value '{args.fps}', using auto-detect")
            fps = None
    print()

    # Initialize analyzer
    analyzer = PitchSourceAnalyzer(args.video, pitch_method=pitch_method)

    # Step 1: Extract audio (with optional normalization)
    analyzer.extract_audio(
        output_dir=args.temp_dir,
        normalize=args.normalize,
        target_lufs=args.target_lufs
    )

    # Step 2: Analyze continuous pitch
    pitch_segments = analyzer.analyze_continuous_pitch(
        fps=fps,
        pitch_change_threshold=args.threshold,
        min_segment_duration=args.min_duration,
        min_rms_db=args.silence_threshold,
        pitch_smoothing=args.pitch_smoothing,
        split_on_rms_dips=not args.no_rms_dip_split,
        rms_dip_threshold_db=args.rms_dip_threshold,
        min_dip_duration=args.min_dip_duration,
        rms_window_ms=args.rms_window_ms
    )

    # Step 3: Build comprehensive pitch database
    analyzer.build_database(pitch_segments)

    # Step 4: Build pitch index for fast lookup
    analyzer.build_pitch_index()

    # Step 5: Print summary
    analyzer.print_database_summary(samples_per_pitch=3)

    # Step 6: Save database
    analyzer.save_database(args.output, append=args.append)

    print("\n=== Database Building Complete ===")


if __name__ == "__main__":
    main()
