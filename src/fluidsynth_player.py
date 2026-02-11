#!/usr/bin/env python3
"""
FluidSynth Player

Provides instrument-aware MIDI synthesis using FluidSynth and SoundFonts.
This module wraps the FluidSynth library to render MIDI notes with realistic
instrument sounds based on General MIDI program changes.

Usage:
    from fluidsynth_player import FluidSynthPlayer

    player = FluidSynthPlayer(soundfont_path="data/soundfonts/GeneralUser_GS.sf2")
    audio = player.render_notes(notes, program=0, channel=0)  # Piano
    player.cleanup()
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import warnings

# FluidSynth availability flag
FLUIDSYNTH_AVAILABLE = False
fluidsynth = None

try:
    import fluidsynth as _fluidsynth
    fluidsynth = _fluidsynth
    FLUIDSYNTH_AVAILABLE = True
except ImportError:
    pass


def is_fluidsynth_available() -> bool:
    """Check if FluidSynth is available."""
    return FLUIDSYNTH_AVAILABLE


class FluidSynthPlayer:
    """
    FluidSynth-based MIDI synthesizer with SoundFont support.

    Renders MIDI notes with realistic instrument sounds using the FluidSynth
    library and General MIDI SoundFonts.
    """

    def __init__(self, soundfont_path: str, sample_rate: int = 44100, gain: float = 0.5):
        """
        Initialize FluidSynth with a SoundFont.

        Args:
            soundfont_path: Path to .sf2 SoundFont file
            sample_rate: Audio sample rate (default: 44100)
            gain: Master gain/volume (0.0-1.0, default: 0.5)

        Raises:
            ImportError: If pyfluidsynth is not installed
            FileNotFoundError: If SoundFont file doesn't exist
            RuntimeError: If FluidSynth fails to initialize
        """
        if not FLUIDSYNTH_AVAILABLE:
            raise ImportError(
                "pyfluidsynth not installed. Install with: pip install pyfluidsynth\n"
                "Also ensure FluidSynth library is installed:\n"
                "  macOS: brew install fluidsynth\n"
                "  Ubuntu: sudo apt-get install fluidsynth libfluidsynth-dev\n"
                "  Windows: Download from https://www.fluidsynth.org/"
            )

        soundfont_path = Path(soundfont_path)
        if not soundfont_path.exists():
            raise FileNotFoundError(f"SoundFont not found: {soundfont_path}")

        self.soundfont_path = soundfont_path
        self.sample_rate = sample_rate
        self.gain = gain

        # Initialize FluidSynth
        self.fs = fluidsynth.Synth(samplerate=float(sample_rate), gain=gain)

        # Load SoundFont
        self.sfid = self.fs.sfload(str(soundfont_path))
        if self.sfid == -1:
            self.fs.delete()
            raise RuntimeError(f"Failed to load SoundFont: {soundfont_path}")

        # Track which channels have been configured
        self._configured_channels = set()

    def set_instrument(self, channel: int, program: int, bank: int = 0):
        """
        Set the instrument for a MIDI channel.

        Args:
            channel: MIDI channel (0-15, channel 9 is drums)
            program: GM instrument program number (0-127)
            bank: SoundFont bank number (0 for melodic, 128 for drums)
        """
        if channel == 9:
            # Channel 9 (10 in 1-indexed) is always drums
            self.fs.program_select(channel, self.sfid, 128, 0)
        else:
            self.fs.program_select(channel, self.sfid, bank, program)

        self._configured_channels.add(channel)

    def render_notes(self, notes: List[Dict], program: int = 0,
                     channel: int = 0, velocity: int = 100) -> np.ndarray:
        """
        Render a list of notes with the specified instrument.

        Args:
            notes: List of note dictionaries with keys:
                   - pitch_midi: MIDI note number (0-127)
                   - start_time: Start time in seconds
                   - duration: Note duration in seconds
                   - (optional) velocity: Note velocity (0-127)
            program: GM instrument program (0-127)
            channel: MIDI channel for rendering (0-15)
            velocity: Default velocity if not specified per-note

        Returns:
            Audio waveform as numpy float32 array (mono, normalized to [-1, 1])
        """
        if not notes:
            return np.array([], dtype=np.float32)

        # Set instrument for this channel
        self.set_instrument(channel, program)

        # Calculate total duration needed
        end_times = [n['start_time'] + n['duration'] for n in notes]
        total_duration = max(end_times) if end_times else 0

        # Add small buffer at end for release
        total_duration += 0.5

        # Pre-allocate output buffer
        total_samples = int(total_duration * self.sample_rate)
        audio_buffer = np.zeros(total_samples, dtype=np.float32)

        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda x: x['start_time'])

        # Build event list: (time, event_type, note, velocity)
        # event_type: 'on' or 'off'
        events = []
        for note in sorted_notes:
            midi_note = note.get('pitch_midi')
            if midi_note is None or midi_note <= 0:
                continue

            start = note['start_time']
            end = start + note['duration']
            vel = note.get('velocity', velocity)

            events.append((start, 'on', int(midi_note), vel))
            events.append((end, 'off', int(midi_note), 0))

        # Sort events by time, with 'off' events before 'on' events at same time
        events.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1))

        # Process events and render audio
        current_time = 0.0

        for event_time, event_type, midi_note, vel in events:
            # Render audio up to this event
            if event_time > current_time:
                samples_to_render = int((event_time - current_time) * self.sample_rate)
                if samples_to_render > 0:
                    rendered = self._get_samples(samples_to_render)
                    start_sample = int(current_time * self.sample_rate)
                    end_sample = start_sample + len(rendered)
                    if end_sample <= len(audio_buffer):
                        audio_buffer[start_sample:end_sample] = rendered
                current_time = event_time

            # Process the event
            if event_type == 'on':
                self.fs.noteon(channel, midi_note, vel)
            else:
                self.fs.noteoff(channel, midi_note)

        # Render remaining audio (including release tails)
        remaining_samples = total_samples - int(current_time * self.sample_rate)
        if remaining_samples > 0:
            rendered = self._get_samples(remaining_samples)
            start_sample = int(current_time * self.sample_rate)
            end_sample = start_sample + len(rendered)
            if end_sample <= len(audio_buffer):
                audio_buffer[start_sample:end_sample] = rendered

        # All notes off to reset state
        self.fs.all_notes_off(channel)

        # Normalize to [-1, 1] range
        max_val = np.max(np.abs(audio_buffer))
        if max_val > 0:
            audio_buffer = audio_buffer / max_val * 0.9

        return audio_buffer

    def _get_samples(self, num_samples: int) -> np.ndarray:
        """
        Get rendered audio samples from FluidSynth.

        Args:
            num_samples: Number of samples to render

        Returns:
            Audio samples as float32 array (mono)
        """
        # FluidSynth returns interleaved stereo int16
        samples = self.fs.get_samples(num_samples)

        # Convert to numpy array
        audio = np.frombuffer(samples, dtype=np.int16).astype(np.float32)

        # Convert from int16 range to float [-1, 1]
        audio = audio / 32768.0

        # Convert stereo to mono by averaging channels
        if len(audio) >= 2:
            left = audio[0::2]
            right = audio[1::2]
            audio = (left + right) / 2.0

        return audio

    def render_drum_pattern(self, hits: List[Dict], velocity: int = 100) -> np.ndarray:
        """
        Render a drum pattern using General MIDI drum sounds.

        Args:
            hits: List of drum hit dictionaries with keys:
                  - pitch_midi: GM drum note (35-81, e.g., 36=kick, 38=snare)
                  - start_time: Start time in seconds
                  - duration: Hit duration (typically short, ~0.1s)
                  - (optional) velocity: Hit velocity (0-127)
            velocity: Default velocity if not specified

        Returns:
            Audio waveform as numpy float32 array
        """
        # Force channel 9 for drums (channel 10 in 1-indexed MIDI)
        return self.render_notes(hits, program=0, channel=9, velocity=velocity)

    def cleanup(self):
        """Release FluidSynth resources."""
        if hasattr(self, 'fs') and self.fs is not None:
            self.fs.delete()
            self.fs = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        self.cleanup()
        return False

    def __del__(self):
        """Destructor - ensure cleanup."""
        self.cleanup()


# General MIDI drum map (channel 9/10)
GM_DRUM_MAP = {
    35: "Acoustic Bass Drum",
    36: "Bass Drum 1",
    37: "Side Stick",
    38: "Acoustic Snare",
    39: "Hand Clap",
    40: "Electric Snare",
    41: "Low Floor Tom",
    42: "Closed Hi-Hat",
    43: "High Floor Tom",
    44: "Pedal Hi-Hat",
    45: "Low Tom",
    46: "Open Hi-Hat",
    47: "Low-Mid Tom",
    48: "Hi-Mid Tom",
    49: "Crash Cymbal 1",
    50: "High Tom",
    51: "Ride Cymbal 1",
    52: "Chinese Cymbal",
    53: "Ride Bell",
    54: "Tambourine",
    55: "Splash Cymbal",
    56: "Cowbell",
    57: "Crash Cymbal 2",
    58: "Vibraslap",
    59: "Ride Cymbal 2",
    60: "Hi Bongo",
    61: "Low Bongo",
    62: "Mute Hi Conga",
    63: "Open Hi Conga",
    64: "Low Conga",
    65: "High Timbale",
    66: "Low Timbale",
    67: "High Agogo",
    68: "Low Agogo",
    69: "Cabasa",
    70: "Maracas",
    71: "Short Whistle",
    72: "Long Whistle",
    73: "Short Guiro",
    74: "Long Guiro",
    75: "Claves",
    76: "Hi Wood Block",
    77: "Low Wood Block",
    78: "Mute Cuica",
    79: "Open Cuica",
    80: "Mute Triangle",
    81: "Open Triangle",
}


def get_drum_name(midi_note: int) -> str:
    """Get the GM drum name for a MIDI note number."""
    return GM_DRUM_MAP.get(midi_note, f"Drum {midi_note}")


def find_soundfont() -> Optional[Path]:
    """
    Search for a SoundFont file in common locations.

    Returns:
        Path to SoundFont file if found, None otherwise
    """
    # Common locations to search
    search_paths = [
        # Project-local
        Path("data/soundfonts"),
        Path("soundfonts"),
        # User directories
        Path.home() / ".local/share/soundfonts",
        Path.home() / "soundfonts",
        # System directories (macOS)
        Path("/usr/local/share/soundfonts"),
        Path("/opt/homebrew/share/soundfonts"),
        # System directories (Linux)
        Path("/usr/share/sounds/sf2"),
        Path("/usr/share/soundfonts"),
    ]

    # Common SoundFont filenames
    soundfont_names = [
        "GeneralUser_GS.sf2",
        "GeneralUser GS.sf2",
        "FluidR3_GM.sf2",
        "FluidR3_GM2-2.sf2",
        "default.sf2",
        "TimGM6mb.sf2",
    ]

    for search_path in search_paths:
        if not search_path.exists():
            continue

        # Check for known filenames
        for name in soundfont_names:
            sf_path = search_path / name
            if sf_path.exists():
                return sf_path

        # Check for any .sf2 file
        sf2_files = list(search_path.glob("*.sf2"))
        if sf2_files:
            return sf2_files[0]

    return None


def main():
    """Test FluidSynth player with a simple melody."""
    import argparse

    parser = argparse.ArgumentParser(description="Test FluidSynth player")
    parser.add_argument("--soundfont", "-s", help="Path to SoundFont file (.sf2)")
    parser.add_argument("--program", "-p", type=int, default=0,
                        help="GM instrument program (0-127, default: 0=Piano)")
    parser.add_argument("--output", "-o", help="Output WAV file")
    args = parser.parse_args()

    # Find SoundFont
    if args.soundfont:
        sf_path = Path(args.soundfont)
    else:
        sf_path = find_soundfont()
        if sf_path is None:
            print("Error: No SoundFont found. Specify with --soundfont or download one:")
            print("  mkdir -p data/soundfonts")
            print("  # Download GeneralUser GS from:")
            print("  # https://schristiancollins.com/generaluser.php")
            return 1

    print(f"Using SoundFont: {sf_path}")

    if not FLUIDSYNTH_AVAILABLE:
        print("Error: pyfluidsynth not installed")
        print("Install with: pip install pyfluidsynth")
        print("Also install FluidSynth library:")
        print("  macOS: brew install fluidsynth")
        print("  Ubuntu: sudo apt-get install fluidsynth libfluidsynth-dev")
        return 1

    # Create test melody (C major scale)
    notes = []
    midi_notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
    time = 0.0

    for midi_note in midi_notes:
        notes.append({
            'pitch_midi': midi_note,
            'start_time': time,
            'duration': 0.4
        })
        time += 0.5

    # Render
    print(f"Rendering {len(notes)} notes with program {args.program}...")

    with FluidSynthPlayer(str(sf_path)) as player:
        audio = player.render_notes(notes, program=args.program)

    print(f"Generated {len(audio) / 44100:.2f}s of audio")

    # Save or play
    if args.output:
        import soundfile as sf
        sf.write(args.output, audio, 44100)
        print(f"Saved to: {args.output}")
    else:
        try:
            import sounddevice as sd
            print("Playing...")
            sd.play(audio, 44100)
            sd.wait()
        except ImportError:
            print("sounddevice not installed - cannot play audio")
            print("Save to file with: --output test.wav")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
