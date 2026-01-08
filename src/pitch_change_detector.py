#!/usr/bin/env python3
"""
Pitch Change Detector

Detects when a singer changes pitch or says a different word/syllable.
More appropriate for singing analysis than onset detection.

Instead of detecting percussive onsets, this:
1. Extracts continuous pitch throughout the audio
2. Detects when pitch changes significantly (new note)
3. Detects when pitch stops/starts (new syllable)
4. Segments based on stable pitch regions
"""

import numpy as np
import librosa
from typing import List, Tuple, Optional
from pathlib import Path
from scipy.signal import medfilt

try:
    import crepe
    CREPE_AVAILABLE = True
except ImportError:
    CREPE_AVAILABLE = False

try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False

try:
    from swift_f0 import SwiftF0
    SWIFT_F0_AVAILABLE = True
except ImportError:
    SWIFT_F0_AVAILABLE = False


class PitchChangeDetector:
    """Detects pitch changes and syllable boundaries in singing."""

    def __init__(self, audio_path: str, sr: int = 22050, pitch_method: str = 'crepe'):
        """
        Initialize detector.

        Args:
            audio_path: Path to audio file
            sr: Sample rate
            pitch_method: Pitch detection method ('crepe', 'basic-pitch', 'swift-f0', 'hybrid', or 'pyin')
        """
        self.audio_path = Path(audio_path)
        self.sr = sr
        self.pitch_method = pitch_method.lower()

        # Validate pitch method
        if self.pitch_method == 'crepe' and not CREPE_AVAILABLE:
            print("Warning: CREPE not available, falling back to pYIN")
            self.pitch_method = 'pyin'
        elif self.pitch_method == 'basic-pitch' and not BASIC_PITCH_AVAILABLE:
            print("Warning: Basic Pitch not available, falling back to pYIN")
            self.pitch_method = 'pyin'
        elif self.pitch_method == 'swift-f0' and not SWIFT_F0_AVAILABLE:
            print("Warning: SwiftF0 not available, falling back to pYIN")
            self.pitch_method = 'pyin'
        elif self.pitch_method == 'hybrid':
            if not CREPE_AVAILABLE and not SWIFT_F0_AVAILABLE:
                print("Warning: Hybrid mode requires CREPE or SwiftF0, falling back to pYIN")
                self.pitch_method = 'pyin'
            elif not CREPE_AVAILABLE:
                print("Warning: CREPE not available for hybrid, using SwiftF0 only")
            elif not SWIFT_F0_AVAILABLE:
                print("Warning: SwiftF0 not available for hybrid, using CREPE only")

        # Load audio
        self.audio, self.sr = librosa.load(str(audio_path), sr=sr)

        # Pitch tracking results
        self.pitch_hz = None
        self.confidence = None
        self.times = None

    def extract_continuous_pitch(self, frame_time: float = 0.01,
                                pitch_smoothing_window: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract continuous pitch track from entire audio.

        Args:
            frame_time: Time between pitch estimates (seconds)
            pitch_smoothing_window: Median filter window size (0 = no smoothing, 5-7 recommended)

        Returns:
            (times, pitch_hz, confidence) arrays
        """
        print(f"Extracting continuous pitch track...")

        if self.pitch_method == 'crepe':
            # CREPE: Deep learning pitch tracker
            print("Using CREPE (this may take a moment)...")
            step_size = int(frame_time * 1000)  # CREPE uses milliseconds

            times, pitch_hz, confidence, _ = crepe.predict(
                self.audio,
                self.sr,
                viterbi=True,  # Smooth pitch trajectory
                model_capacity='full',
                step_size=step_size
            )

        elif self.pitch_method == 'basic-pitch':
            # Basic Pitch: Spotify's multipitch detection
            print("Using Basic Pitch (this may take a moment)...")
            times, pitch_hz, confidence = self.extract_pitch_basic_pitch(frame_time)

        elif self.pitch_method == 'swift-f0':
            # SwiftF0: Fast CPU-based pitch detection
            print("Using SwiftF0 (fast, CPU-optimized)...")
            times, pitch_hz, confidence = self.extract_pitch_swift_f0(frame_time)

        elif self.pitch_method == 'hybrid':
            # Hybrid: CREPE + SwiftF0 mixture of experts
            print("Using Hybrid (CREPE + SwiftF0 mixture of experts)...")
            times, pitch_hz, confidence = self.extract_pitch_hybrid(frame_time)

        else:
            # pYIN: Probabilistic YIN
            print("Using pYIN...")
            hop_length = int(frame_time * self.sr)

            pitch_hz, voiced_flag, voiced_probs = librosa.pyin(
                self.audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C6'),
                sr=self.sr,
                hop_length=hop_length,
                frame_length=2048
            )

            times = librosa.times_like(pitch_hz, sr=self.sr, hop_length=hop_length)
            confidence = voiced_probs

            # Replace NaN with 0
            pitch_hz = np.nan_to_num(pitch_hz, nan=0.0)
            confidence = np.nan_to_num(confidence, nan=0.0)

        # Apply pitch smoothing if requested
        if pitch_smoothing_window > 0:
            # Only smooth non-zero (voiced) regions
            voiced_mask = pitch_hz > 0
            if np.any(voiced_mask):
                # Apply median filter to smooth out vibrato/waver
                # Must be odd number for median filter
                window_size = pitch_smoothing_window if pitch_smoothing_window % 2 == 1 else pitch_smoothing_window + 1
                smoothed = medfilt(pitch_hz, kernel_size=window_size)
                # Only replace voiced regions
                pitch_hz = np.where(voiced_mask, smoothed, pitch_hz)
                print(f"Applied pitch smoothing (window={window_size})")

        self.times = times
        self.pitch_hz = pitch_hz
        self.confidence = confidence

        # Statistics
        voiced_frames = np.sum(confidence > 0.5)
        print(f"Extracted {len(times)} pitch frames ({len(times) * frame_time:.2f}s)")
        print(f"Voiced frames: {voiced_frames} ({voiced_frames/len(times)*100:.1f}%)")

        return times, pitch_hz, confidence

    def extract_pitch_basic_pitch(self, frame_time: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract continuous pitch using Basic Pitch (Spotify's model).

        Args:
            frame_time: Target time between frames (will be quantized to Basic Pitch's frame rate)

        Returns:
            (times, pitch_hz, confidence) arrays
        """
        import tempfile
        import soundfile as sf

        # Basic Pitch needs a file path, so write audio to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, self.audio, self.sr)

        try:
            # Run Basic Pitch prediction
            model_output, midi_data, note_events = predict(tmp_path)

            # Extract contour data (shape: n_times x n_freq_bins)
            # contour has 264 bins (3 per semitone) covering 88 semitones
            contour = model_output['contour']  # Frame-by-frame pitch contour
            note_activations = model_output['note']  # Frame-by-frame note activity

            # Basic Pitch constants (from their constants.py)
            ANNOTATIONS_FPS = 22050 / 256  # ~86.1 frames per second
            ANNOTATIONS_BASE_FREQUENCY = 27.5  # A0 in Hz
            CONTOURS_BINS_PER_SEMITONE = 3
            N_FREQ_BINS_CONTOURS = 264

            # Create time array
            n_frames = contour.shape[0]
            times = np.arange(n_frames) / ANNOTATIONS_FPS

            # Extract dominant pitch at each frame
            pitch_hz = np.zeros(n_frames)
            confidence = np.zeros(n_frames)

            for i in range(n_frames):
                # Find peak in contour (weighted by note activation)
                frame_contour = contour[i, :]
                frame_notes = note_activations[i, :]

                # Combine contour and note activation for better pitch estimate
                # Average the 3 contour bins per semitone to align with note bins
                contour_by_semitone = frame_contour.reshape(88, 3).max(axis=1)
                combined = contour_by_semitone * frame_notes

                if combined.max() > 0.1:  # Threshold for voiced frame
                    # Find peak semitone
                    peak_semitone = np.argmax(combined)
                    confidence[i] = combined[peak_semitone]

                    # Convert semitone to Hz
                    # freq = base_freq * 2^(semitones/12)
                    pitch_hz[i] = ANNOTATIONS_BASE_FREQUENCY * (2 ** (peak_semitone / 12.0))

                    # Refine with contour sub-bins for more precise pitch
                    contour_start = peak_semitone * 3
                    contour_bins = frame_contour[contour_start:contour_start + 3]
                    if contour_bins.max() > 0:
                        sub_bin_offset = np.argmax(contour_bins)
                        # Add fractional semitone offset (each bin = 1/3 semitone)
                        semitone_offset = (sub_bin_offset - 1) / 3.0  # -1/3, 0, or +1/3 semitone
                        pitch_hz[i] = ANNOTATIONS_BASE_FREQUENCY * (2 ** ((peak_semitone + semitone_offset) / 12.0))

            # Normalize confidence to 0-1 range
            if confidence.max() > 0:
                confidence = confidence / confidence.max()

        finally:
            # Clean up temp file
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return times, pitch_hz, confidence

    def extract_pitch_swift_f0(self, frame_time: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract continuous pitch using SwiftF0 (fast, CPU-optimized).

        Args:
            frame_time: Target time between frames (note: SwiftF0 uses fixed 16kHz/256-sample hop)

        Returns:
            (times, pitch_hz, confidence) arrays
        """
        # Initialize SwiftF0 detector
        # fmin/fmax cover C2 (~65Hz) to C7 (~2093Hz) - full singing range
        detector = SwiftF0(
            fmin=46.875,      # G1 (SwiftF0 lower limit)
            fmax=2093.75,     # C7 (SwiftF0 upper limit)
            confidence_threshold=0.5  # Lower threshold, we'll filter later
        )

        # SwiftF0 can work directly with file or array
        # Using file path for efficiency
        result = detector.detect_from_file(str(self.audio_path))

        # Extract results
        times = result.timestamps      # Frame center times in seconds
        pitch_hz = result.pitch_hz     # F0 estimates in Hz
        confidence = result.confidence # Model confidence (0.0-1.0)

        # SwiftF0 already provides voicing decisions, but we use confidence
        # Note: SwiftF0 sets unvoiced frames to 0 Hz
        # Ensure unvoiced frames have 0 confidence
        pitch_hz = np.where(result.voicing, pitch_hz, 0.0)

        return times, pitch_hz, confidence

    def extract_pitch_hybrid(self, frame_time: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Hybrid pitch detection using CREPE + SwiftF0 mixture of experts.

        Strategy:
        1. Use CREPE as primary (more accurate)
        2. Fill CREPE gaps with SwiftF0 (catches missed notes)
        3. Filter SwiftF0 outliers using pitch continuity

        Args:
            frame_time: Target time between frames

        Returns:
            (times, pitch_hz, confidence) arrays
        """
        print("Running CREPE (primary detector)...")

        # Run CREPE
        if not CREPE_AVAILABLE:
            print("Warning: CREPE not available for hybrid mode, using SwiftF0 only")
            return self.extract_pitch_swift_f0(frame_time)

        step_size = int(frame_time * 1000)
        crepe_times, crepe_pitch, crepe_conf, _ = crepe.predict(
            self.audio,
            self.sr,
            viterbi=True,
            model_capacity='full',
            step_size=step_size
        )

        print("Running SwiftF0 (gap filler)...")

        # Run SwiftF0
        if not SWIFT_F0_AVAILABLE:
            print("Warning: SwiftF0 not available for hybrid mode, using CREPE only")
            return crepe_times, crepe_pitch, crepe_conf

        detector = SwiftF0(
            fmin=46.875,
            fmax=2093.75,
            confidence_threshold=0.3  # Lower threshold for gap filling
        )
        result = detector.detect_from_file(str(self.audio_path))

        # Interpolate SwiftF0 to match CREPE's time grid
        swift_pitch_interp = np.interp(
            crepe_times,
            result.timestamps,
            result.pitch_hz,
            left=0,
            right=0
        )
        swift_conf_interp = np.interp(
            crepe_times,
            result.timestamps,
            result.confidence,
            left=0,
            right=0
        )

        # Hybrid fusion logic
        hybrid_pitch = np.zeros_like(crepe_pitch)
        hybrid_conf = np.zeros_like(crepe_conf)

        # Thresholds
        CREPE_MIN_CONFIDENCE = 0.5  # Only trust CREPE above this
        SWIFT_MIN_CONFIDENCE = 0.5  # Only use SwiftF0 above this for gap filling
        MAX_PITCH_JUMP_SEMITONES = 5  # Max semitone jump to accept SwiftF0
        BASS_THRESHOLD_HZ = 80  # Flag very low notes as suspicious

        for i in range(len(crepe_times)):
            crepe_has_pitch = crepe_pitch[i] > 0 and crepe_conf[i] >= CREPE_MIN_CONFIDENCE
            swift_has_pitch = swift_pitch_interp[i] > 0 and swift_conf_interp[i] >= SWIFT_MIN_CONFIDENCE

            if crepe_has_pitch:
                # CREPE is confident - use it
                hybrid_pitch[i] = crepe_pitch[i]
                hybrid_conf[i] = crepe_conf[i]

            elif swift_has_pitch:
                # CREPE missed it, check if SwiftF0 makes sense
                swift_freq = swift_pitch_interp[i]

                # Filter out very low bass notes (likely octave errors)
                if swift_freq < BASS_THRESHOLD_HZ:
                    continue

                # Check pitch continuity with context
                accept_swift = False

                # Look for nearby CREPE pitches to validate
                context_start = max(0, i - 10)
                context_end = min(len(crepe_times), i + 10)

                nearby_crepe_pitches = []
                for j in range(context_start, context_end):
                    if crepe_pitch[j] > 0 and crepe_conf[j] >= CREPE_MIN_CONFIDENCE:
                        nearby_crepe_pitches.append(crepe_pitch[j])

                if len(nearby_crepe_pitches) > 0:
                    # Check if SwiftF0 pitch is within reasonable range of nearby CREPE pitches
                    median_nearby = np.median(nearby_crepe_pitches)
                    semitone_diff = abs(12 * np.log2(swift_freq / median_nearby))

                    if semitone_diff <= MAX_PITCH_JUMP_SEMITONES:
                        accept_swift = True
                else:
                    # No nearby CREPE context, accept SwiftF0 if not too low
                    accept_swift = True

                if accept_swift:
                    hybrid_pitch[i] = swift_freq
                    # Reduce confidence since it's gap-filled
                    hybrid_conf[i] = swift_conf_interp[i] * 0.8

        # Count contributions
        crepe_frames = np.sum((hybrid_pitch > 0) & (crepe_pitch > 0))
        swift_frames = np.sum((hybrid_pitch > 0) & (crepe_pitch == 0))

        print(f"Hybrid result: {crepe_frames} CREPE frames, {swift_frames} SwiftF0 gap-fills")

        return crepe_times, hybrid_pitch, hybrid_conf

    def detect_pitch_segments(self,
                             min_confidence: float = 0.5,
                             min_rms_db: float = -40.0,
                             pitch_change_threshold_cents: float = 50.0,
                             min_segment_duration: float = 0.1,
                             max_segment_duration: float = 3.0,
                             min_silence_duration: float = 0.1,
                             split_on_rms_dips: bool = True,
                             rms_dip_threshold_db: float = -6.0,
                             min_dip_duration: float = 0.02,
                             rms_window_ms: float = 50.0) -> List[Tuple[float, float, float, float]]:
        """
        Detect segments where pitch is stable (single note/syllable).

        Splits when EITHER:
        - Pitch changes by more than threshold (new note)
        - Sustained silence is detected (end of phrase)
        - RMS dip detected (syllable/word boundary) - if enabled
        - Max duration exceeded

        Each note holds through small pitch variations (vibrato, drift) until
        a significant pitch change or sustained silence occurs.

        Args:
            min_confidence: Minimum pitch confidence
            min_rms_db: Minimum RMS amplitude (dB) to detect silence
            pitch_change_threshold_cents: Pitch change to trigger new segment (cents)
            min_segment_duration: Minimum segment length (seconds)
            max_segment_duration: Maximum segment length (seconds)
            min_silence_duration: Minimum silence duration to split (seconds)
            split_on_rms_dips: Enable splitting on volume dips (syllable boundaries)
            rms_dip_threshold_db: Threshold below local mean to count as dip (e.g., -6dB)
            min_dip_duration: Minimum dip duration to trigger split (seconds)
            rms_window_ms: Rolling window for computing local RMS mean (ms)

        Returns:
            List of (start_time, end_time, median_pitch_hz, confidence)
        """
        if self.pitch_hz is None:
            raise ValueError("Must extract pitch first")

        print(f"\nDetecting pitch segments...")
        split_criteria = f"pitch changes >{pitch_change_threshold_cents} cents OR silence >{min_silence_duration}s"
        if split_on_rms_dips:
            split_criteria += f" OR RMS dips <{rms_dip_threshold_db}dB for >{min_dip_duration*1000:.0f}ms"
        print(f"Splits on: {split_criteria}")
        print(f"Notes hold through pitch drift until significant change or sustained silence")

        segments = []
        current_segment_start = None
        current_segment_frames = []

        # Track silence duration
        silence_start = None
        silence_frames = 0

        # Track last confident pitch value (carry forward when confidence drops)
        last_confident_pitch = None

        for i in range(len(self.times)):
            time = self.times[i]
            pitch = self.pitch_hz[i]
            conf = self.confidence[i]

            # PRIMARY CHECK: RMS amplitude (actual audio energy) - this is truth
            frame_start = int(time * self.sr)
            frame_end = int((time + 0.01) * self.sr)  # Assume 10ms frames

            has_sound = False
            if frame_end <= len(self.audio):
                audio_frame = self.audio[frame_start:frame_end]
                rms = np.sqrt(np.mean(audio_frame**2))
                rms_db = 20 * np.log10(rms + 1e-10)
                has_sound = rms_db >= min_rms_db

            # If no sound (RMS says silence), treat as silence regardless of pitch
            if not has_sound:
                is_valid = False
            else:
                # There IS sound - now check pitch confidence
                has_confident_pitch = pitch > 0 and conf >= min_confidence

                if has_confident_pitch:
                    # Good pitch detection - use it and remember it
                    last_confident_pitch = pitch
                    is_valid = True
                elif last_confident_pitch is not None:
                    # Low confidence but we have sound - carry forward last good pitch
                    # This handles consonants, transitions, etc.
                    pitch = last_confident_pitch  # Use carried-forward pitch
                    is_valid = True
                else:
                    # No confident pitch yet and current is unreliable
                    is_valid = False

            if is_valid:
                # Voiced/sound detected - reset silence counter
                silence_start = None
                silence_frames = 0

                should_start_new_segment = False

                if current_segment_frames:
                    # Calculate median pitch of current segment
                    segment_pitches = [self.pitch_hz[idx] for idx in current_segment_frames]
                    segment_median = np.median(segment_pitches)

                    # Check if pitch changed significantly from current segment
                    if segment_median > 0 and pitch > 0:
                        cents_diff = 1200 * np.log2(pitch / segment_median)
                        if abs(cents_diff) > pitch_change_threshold_cents:
                            should_start_new_segment = True

                    # Also start new segment if it's getting too long
                    segment_duration = time - self.times[current_segment_frames[0]]
                    if segment_duration > max_segment_duration:
                        should_start_new_segment = True

                if should_start_new_segment or current_segment_start is None:
                    # Save previous segment if long enough
                    if current_segment_frames:
                        self._save_segment(current_segment_frames, segments, min_segment_duration)

                    # Start new segment
                    current_segment_start = time
                    current_segment_frames = [i]
                else:
                    # Continue current segment (hold the note)
                    current_segment_frames.append(i)

                    # Check for RMS dips if segment is long enough
                    if split_on_rms_dips and len(current_segment_frames) > 20:  # At least 200ms
                        seg_start_time = self.times[current_segment_frames[0]]
                        seg_current_time = time
                        seg_duration = seg_current_time - seg_start_time

                        # Only check if segment is at least 150ms (to allow splitting into 2x75ms)
                        if seg_duration >= 0.15:
                            dip_times = self._detect_rms_dips(
                                seg_start_time,
                                seg_current_time,
                                dip_threshold_db=rms_dip_threshold_db,
                                min_dip_duration=min_dip_duration,
                                window_ms=rms_window_ms
                            )

                            # Split at first dip that leaves both parts long enough
                            for dip_time in dip_times:
                                first_part_duration = dip_time - seg_start_time
                                second_part_duration = seg_current_time - dip_time

                                if first_part_duration >= min_segment_duration and second_part_duration >= 0.05:
                                    # Find frame index closest to dip time
                                    split_frame_idx = None
                                    for idx, frame_idx in enumerate(current_segment_frames):
                                        if self.times[frame_idx] >= dip_time:
                                            split_frame_idx = idx
                                            break

                                    if split_frame_idx and split_frame_idx > 0:
                                        # Save first part
                                        first_part_frames = current_segment_frames[:split_frame_idx]
                                        self._save_segment(first_part_frames, segments, min_segment_duration)

                                        # Continue with second part
                                        current_segment_frames = current_segment_frames[split_frame_idx:]
                                        current_segment_start = self.times[current_segment_frames[0]] if current_segment_frames else None
                                        break  # Only split once per iteration

            else:
                # Unvoiced/silence frame detected
                if silence_start is None:
                    # Start tracking silence
                    silence_start = time
                    silence_frames = 1
                else:
                    # Continue tracking silence
                    silence_frames += 1

                # Check if silence has lasted long enough to split
                silence_duration = (silence_frames - 1) * 0.01  # Approximate frame duration

                if silence_duration >= min_silence_duration:
                    # Sustained silence - end current segment
                    if current_segment_frames:
                        self._save_segment(current_segment_frames, segments, min_segment_duration)
                        current_segment_frames = []
                        current_segment_start = None
                    # Reset silence tracking
                    silence_start = None
                    silence_frames = 0

        # Save final segment
        if current_segment_frames:
            self._save_segment(current_segment_frames, segments, min_segment_duration)

        print(f"Detected {len(segments)} pitch segments")

        return segments

    def _save_segment(self, frame_indices: List[int], segments: List, min_duration: float):
        """Save a segment if it meets duration requirements."""
        if not frame_indices:
            return

        start_time = self.times[frame_indices[0]]
        end_time = self.times[frame_indices[-1]]
        duration = end_time - start_time

        if duration >= min_duration:
            # Calculate median pitch and confidence
            pitches = [self.pitch_hz[i] for i in frame_indices]
            confidences = [self.confidence[i] for i in frame_indices]

            median_pitch = np.median(pitches)
            mean_confidence = np.mean(confidences)

            segments.append((start_time, end_time, median_pitch, mean_confidence))

    def _detect_rms_dips(self,
                         start_time: float,
                         end_time: float,
                         dip_threshold_db: float = -6.0,
                         min_dip_duration: float = 0.02,
                         window_ms: float = 50.0) -> List[float]:
        """
        Detect RMS dips within a time range that indicate syllable/word boundaries.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            dip_threshold_db: Threshold below rolling mean to count as dip (negative, e.g., -6dB)
            min_dip_duration: Minimum duration of dip in seconds (e.g., 0.02 = 20ms)
            window_ms: Rolling window size for computing local mean RMS

        Returns:
            List of dip center times (in seconds) suitable for segment splits
        """
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)

        if end_sample > len(self.audio):
            end_sample = len(self.audio)

        segment_audio = self.audio[start_sample:end_sample]

        if len(segment_audio) < int(0.05 * self.sr):  # Need at least 50ms
            return []

        # Compute frame-by-frame RMS (10ms frames, 50% overlap)
        frame_size = int(0.01 * self.sr)  # 10ms
        hop_size = frame_size // 2  # 5ms hop

        rms_frames = []
        frame_times = []

        for i in range(0, len(segment_audio) - frame_size, hop_size):
            frame = segment_audio[i:i + frame_size]
            rms = np.sqrt(np.mean(frame ** 2))
            rms_db = 20 * np.log10(rms + 1e-10)
            rms_frames.append(rms_db)
            # Time is relative to segment start, convert to absolute
            frame_times.append(start_time + (i + frame_size // 2) / self.sr)

        if len(rms_frames) < 5:
            return []

        rms_frames = np.array(rms_frames)
        frame_times = np.array(frame_times)

        # Compute rolling mean RMS
        window_frames = max(3, int(window_ms / 5))  # 5ms per hop

        # Use convolution for rolling mean
        kernel = np.ones(window_frames) / window_frames
        rolling_mean = np.convolve(rms_frames, kernel, mode='same')

        # Find frames that are significantly below local mean
        dip_mask = rms_frames < (rolling_mean + dip_threshold_db)

        # Find contiguous dip regions
        dips = []
        in_dip = False
        dip_start_idx = 0

        for i, is_dip in enumerate(dip_mask):
            if is_dip and not in_dip:
                in_dip = True
                dip_start_idx = i
            elif not is_dip and in_dip:
                in_dip = False
                dip_end_idx = i
                # Duration: each frame is ~5ms (hop size)
                dip_duration = (dip_end_idx - dip_start_idx) * (hop_size / self.sr)

                if dip_duration >= min_dip_duration:
                    # Return center of dip as split point
                    dip_center_idx = (dip_start_idx + dip_end_idx) // 2
                    if dip_center_idx < len(frame_times):
                        dips.append(frame_times[dip_center_idx])

        return dips

    def segments_to_dict(self, segments: List[Tuple[float, float, float, float]]) -> List[dict]:
        """
        Convert segment tuples to dictionaries with full pitch info.

        Args:
            segments: List of (start_time, end_time, pitch_hz, confidence)

        Returns:
            List of segment dictionaries
        """
        from pitch_utils import hz_to_midi, midi_to_note_name, round_to_nearest_midi

        result = []
        for i, (start_time, end_time, pitch_hz, confidence) in enumerate(segments):
            midi = round_to_nearest_midi(pitch_hz)
            note_name = midi_to_note_name(midi)

            result.append({
                'index': i,
                'start_time': float(start_time),
                'end_time': float(end_time),
                'duration': float(end_time - start_time),
                'pitch_hz': float(pitch_hz),
                'pitch_midi': int(midi),
                'pitch_note': note_name,
                'pitch_confidence': float(confidence)
            })

        return result


def main():
    """Test pitch change detection."""
    import argparse

    parser = argparse.ArgumentParser(description="Test pitch change detection")
    parser.add_argument('--audio', type=str, required=True, help='Audio file')
    parser.add_argument('--threshold', type=float, default=50.0,
                       help='Pitch change threshold (cents, default: 50)')
    parser.add_argument('--use-pyin', action='store_true', help='Use pYIN instead of CREPE')

    args = parser.parse_args()

    # Initialize detector
    detector = PitchChangeDetector(args.audio, use_crepe=not args.use_pyin)

    # Extract continuous pitch
    detector.extract_continuous_pitch(frame_time=0.01)

    # Detect segments
    segments = detector.detect_pitch_segments(
        pitch_change_threshold_cents=args.threshold,
        min_segment_duration=0.1
    )

    # Convert to dicts
    segment_dicts = detector.segments_to_dict(segments)

    # Print results
    print(f"\n=== Detected Segments ===")
    print(f"{'Idx':<5} {'Time':<12} {'Dur':<7} {'Note':<6} {'Hz':<8} {'Conf':<6}")
    print("-" * 50)

    for seg in segment_dicts[:30]:  # Show first 30
        print(f"{seg['index']:<5} "
              f"{seg['start_time']:>5.2f}-{seg['end_time']:<4.2f} "
              f"{seg['duration']:<7.3f} "
              f"{seg['pitch_note']:<6} "
              f"{seg['pitch_hz']:<8.1f} "
              f"{seg['pitch_confidence']:<6.3f}")

    if len(segment_dicts) > 30:
        print(f"\n... and {len(segment_dicts) - 30} more segments")


if __name__ == "__main__":
    main()
