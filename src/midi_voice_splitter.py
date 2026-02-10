#!/usr/bin/env python3
"""
MIDI Voice Splitter

Splits a polyphonic MIDI channel into N monophonic guide_sequence JSON files
using pitch-ordered voice assignment. Each voice file feeds directly into the
existing pitch matching pipeline.

Usage:
    python midi_voice_splitter.py --midi song.mid --channel 5
    python midi_voice_splitter.py --midi song.mid --list-channels
    python midi_voice_splitter.py --midi song.mid --channel 5 --no-audio
"""

import argparse
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from itertools import groupby

try:
    import mido
except ImportError:
    print("Error: mido not installed. Install with: pip install mido")
    import sys
    sys.exit(1)

from pitch_utils import midi_to_hz, midi_to_note_name
from midi_player import MIDIPlayer

# General MIDI instrument names (program 0-127) — imported pattern from midi_channel_splitter
GM_INSTRUMENTS = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano",
    "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba", "Xylophone",
    "Tubular Bells", "Dulcimer",
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ", "Reed Organ",
    "Accordion", "Harmonica", "Tango Accordion",
    "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)",
    "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar",
    "Distortion Guitar", "Guitar Harmonics",
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
    "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
    "Violin", "Viola", "Cello", "Contrabass", "Tremolo Strings", "Pizzicato Strings",
    "Orchestral Harp", "Timpani",
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2",
    "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit",
    "Trumpet", "Trombone", "Tuba", "Muted Trumpet", "French Horn", "Brass Section",
    "Synth Brass 1", "Synth Brass 2",
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe", "English Horn",
    "Bassoon", "Clarinet",
    "Piccolo", "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi",
    "Whistle", "Ocarina",
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)",
    "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)",
    "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)",
    "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
    "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bagpipe", "Fiddle", "Shanai",
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko Drum", "Melodic Tom",
    "Synth Drum", "Reverse Cymbal",
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet", "Telephone Ring",
    "Helicopter", "Applause", "Gunshot"
]


def get_instrument_name(program: int, channel: int = 0) -> str:
    """Get instrument name for a program number."""
    if channel == 9:
        return "Drums"
    if 0 <= program < len(GM_INSTRUMENTS):
        return GM_INSTRUMENTS[program]
    return f"Program {program}"


class MIDIVoiceSplitter:
    """Splits a polyphonic MIDI channel into monophonic voice files."""

    STRATEGIES = ('pitch-ordered', 'balanced')

    def __init__(self, midi_path: str, channel: int, output_dir: str = 'data/segments',
                 min_rest: float = 0.1, sample_rate: int = 22050,
                 strategy: str = 'pitch-ordered'):
        """
        Initialize the voice splitter.

        Args:
            midi_path: Path to MIDI file
            channel: MIDI channel to split (0-15)
            output_dir: Directory for output JSON files
            min_rest: Minimum rest duration in seconds (per-voice)
            sample_rate: Audio preview sample rate
            strategy: Voice assignment strategy ('pitch-ordered' or 'balanced')
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}', must be one of {self.STRATEGIES}")
        self.midi_path = Path(midi_path)
        self.channel = channel
        self.output_dir = Path(output_dir)
        self.min_rest = min_rest
        self.sample_rate = sample_rate
        self.strategy = strategy

        self.midi_file = None
        self.ticks_per_beat = 480

        self.notes = []                # All extracted notes with IDs
        self.voice_assignments = {}    # note_id -> voice_number (1-indexed)
        self.num_voices = 0
        self.voice_segments = {}       # voice_number -> list of pitch segments

    def load_midi(self) -> None:
        """Load and parse MIDI file."""
        print(f"Loading MIDI file: {self.midi_path}")
        self.midi_file = mido.MidiFile(str(self.midi_path))
        self.ticks_per_beat = self.midi_file.ticks_per_beat
        print(f"  Ticks per beat: {self.ticks_per_beat}")
        print(f"  Number of tracks: {len(self.midi_file.tracks)}")

    def _collect_tempo_changes(self) -> List[Tuple[int, int]]:
        """
        Collect all tempo changes from the MIDI file.

        Returns:
            List of (tick, tempo_microseconds) tuples, sorted by tick
        """
        tempo_changes = [(0, 500000)]  # Default: 120 BPM

        for track in self.midi_file.tracks:
            tick = 0
            for msg in track:
                tick += msg.time
                if msg.type == 'set_tempo':
                    tempo_changes.append((tick, msg.tempo))

        tempo_changes.sort(key=lambda x: x[0])
        return tempo_changes

    def _ticks_to_seconds(self, ticks: int, tempo_changes: List[Tuple[int, int]]) -> float:
        """
        Convert MIDI ticks to seconds, accounting for tempo changes.

        Args:
            ticks: Target tick position
            tempo_changes: List of (tick, tempo_microseconds) tuples

        Returns:
            Time in seconds
        """
        seconds = 0.0
        last_tick = 0
        current_tempo = 500000  # Default 120 BPM

        for change_tick, tempo in tempo_changes:
            if change_tick >= ticks:
                break

            tick_delta = change_tick - last_tick
            seconds += (tick_delta / self.ticks_per_beat) * (current_tempo / 1_000_000)

            last_tick = change_tick
            current_tempo = tempo

        remaining_ticks = ticks - last_tick
        seconds += (remaining_ticks / self.ticks_per_beat) * (current_tempo / 1_000_000)

        return seconds

    def list_channels(self) -> Dict[int, Dict]:
        """
        List all channels with note counts, range, polyphony, and instrument info.

        Returns:
            Dictionary mapping channel number to info dict
        """
        if self.midi_file is None:
            self.load_midi()

        channel_info = defaultdict(lambda: {
            'note_count': 0,
            'notes': set(),
            'min_note': 127,
            'max_note': 0,
            'program': None
        })

        # First pass: collect notes and program changes
        for track in self.midi_file.tracks:
            for msg in track:
                if msg.type == 'program_change':
                    channel_info[msg.channel]['program'] = msg.program
                elif hasattr(msg, 'channel'):
                    if msg.type == 'note_on' and msg.velocity > 0:
                        ch = msg.channel
                        channel_info[ch]['note_count'] += 1
                        channel_info[ch]['notes'].add(msg.note)
                        channel_info[ch]['min_note'] = min(channel_info[ch]['min_note'], msg.note)
                        channel_info[ch]['max_note'] = max(channel_info[ch]['max_note'], msg.note)

        # Second pass: calculate max polyphony per channel
        for ch in list(channel_info.keys()):
            channel_info[ch]['max_polyphony'] = self._calculate_channel_polyphony(ch)

        # Convert sets to counts and add names
        for ch in channel_info:
            info = channel_info[ch]
            info['unique_notes'] = len(info['notes'])
            if info['note_count'] > 0:
                info['note_range'] = (
                    f"{midi_to_note_name(info['min_note'])}-"
                    f"{midi_to_note_name(info['max_note'])}"
                )
            else:
                info['note_range'] = "N/A"
            del info['notes']

            program = info['program']
            if ch == 9:
                info['instrument'] = "Drums"
            elif program is not None:
                info['instrument'] = get_instrument_name(program, ch)
            else:
                info['instrument'] = None

        return dict(channel_info)

    def _calculate_channel_polyphony(self, channel: int) -> int:
        """
        Calculate max simultaneous notes for a channel.

        Args:
            channel: MIDI channel number

        Returns:
            Maximum number of simultaneous notes
        """
        # Track active notes at each tick using note_on/note_off events
        active = 0
        max_active = 0

        for track in self.midi_file.tracks:
            # Process each track independently for active count is wrong —
            # we need to merge all tracks first. Use a simpler approach:
            # collect all events with absolute ticks, sort, then walk.
            pass

        # Collect all note events across tracks with absolute tick positions
        events = []  # (tick, type, note, channel)
        for track in self.midi_file.tracks:
            tick = 0
            for msg in track:
                tick += msg.time
                if not hasattr(msg, 'channel') or msg.channel != channel:
                    continue
                if msg.type == 'note_on' and msg.velocity > 0:
                    events.append((tick, 'on', msg.note))
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    events.append((tick, 'off', msg.note))

        # Sort: by tick, then 'off' before 'on' at same tick
        events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))

        active = 0
        max_active = 0
        for _, event_type, _ in events:
            if event_type == 'on':
                active += 1
                max_active = max(max_active, active)
            else:
                active = max(0, active - 1)

        return max_active

    def extract_all_notes(self) -> List[Dict]:
        """
        Extract ALL note events from the specified channel, preserving overlaps.

        Unlike midi_guide_converter's extract_notes(), this does NOT enforce
        monophonic behavior. Multiple notes can overlap in time.

        Returns:
            List of note dicts with id, pitch_midi, start_time, end_time, duration
        """
        if self.midi_file is None:
            self.load_midi()

        tempo_changes = self._collect_tempo_changes()
        notes = []
        note_id = 0

        # Collect events with absolute ticks across all tracks
        # Use a per-pitch active list to handle overlaps properly
        for track in self.midi_file.tracks:
            current_tick = 0
            active_notes = {}  # pitch -> list of (start_tick, note_id)

            for msg in track:
                current_tick += msg.time

                if hasattr(msg, 'channel') and msg.channel != self.channel:
                    continue

                if msg.type == 'note_on' and msg.velocity > 0:
                    # Start a new note — allow multiple notes on same pitch
                    if msg.note not in active_notes:
                        active_notes[msg.note] = []
                    active_notes[msg.note].append((current_tick, note_id))
                    note_id += 1

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes and active_notes[msg.note]:
                        # End the earliest active note with this pitch
                        start_tick, nid = active_notes[msg.note].pop(0)
                        if not active_notes[msg.note]:
                            del active_notes[msg.note]

                        start_time = self._ticks_to_seconds(start_tick, tempo_changes)
                        end_time = self._ticks_to_seconds(current_tick, tempo_changes)

                        if end_time > start_time:
                            notes.append({
                                'id': nid,
                                'pitch_midi': msg.note,
                                'start_time': start_time,
                                'end_time': end_time,
                                'duration': end_time - start_time
                            })

        # Sort by start_time, then pitch ascending for simultaneous starts
        notes.sort(key=lambda n: (n['start_time'], n['pitch_midi']))
        self.notes = notes

        print(f"Extracted {len(notes)} notes from channel {self.channel} (preserving overlaps)")
        return notes

    def assign_voices(self) -> Dict[int, int]:
        """
        Assign each note to a voice using the configured strategy.

        Dispatches to strategy-specific method based on self.strategy.

        Returns:
            Dictionary mapping note_id to voice_number (1-indexed)
        """
        if not self.notes:
            self.voice_assignments = {}
            self.num_voices = 0
            return {}

        if self.strategy == 'balanced':
            return self._assign_voices_balanced()
        return self._assign_voices_pitch_ordered()

    def _assign_voices_pitch_ordered(self) -> Dict[int, int]:
        """
        Assign each note to a voice using pitch-ordered assignment.

        At each note's start time:
        1. Remove expired notes (end_time <= start_time)
        2. Batch all note_on events at the same start time
        3. Sort all active notes (sustained + new) by pitch ascending
        4. Assign each NEW note to voice = its position in sorted list + 1
        5. Sustained notes keep their existing voice assignment

        Returns:
            Dictionary mapping note_id to voice_number (1-indexed)
        """
        voice_assignments = {}
        # active_notes: list of (note_id, pitch_midi, end_time, assigned_voice)
        active_notes = []
        max_voices = 0

        # Group notes by start_time
        # Notes are already sorted by (start_time, pitch_midi)
        time_groups = []
        for start_time, group in groupby(self.notes, key=lambda n: n['start_time']):
            time_groups.append((start_time, list(group)))

        for start_time, new_notes in time_groups:
            # Remove expired notes (end_time <= start_time)
            active_notes = [
                (nid, p, end, v) for nid, p, end, v in active_notes
                if end > start_time
            ]

            # Build combined list: sustained + new (new have voice=None)
            all_active = list(active_notes)
            for n in new_notes:
                all_active.append((n['id'], n['pitch_midi'], n['end_time'], None))

            # Sort by pitch ascending, tiebreak: notes with existing voice first
            # (sustained notes), then by note_on order (new notes keep their order)
            all_active.sort(key=lambda x: (x[1], 0 if x[3] is not None else 1))

            # Assign voice numbers: position in sorted list + 1
            for position, (nid, pitch, end, existing_voice) in enumerate(all_active):
                voice = position + 1
                if existing_voice is None:
                    # New note — assign based on current position
                    voice_assignments[nid] = voice

            # Rebuild active_notes with final assignments
            active_notes = []
            for nid, pitch, end, existing_voice in all_active:
                assigned = existing_voice if existing_voice is not None else voice_assignments[nid]
                active_notes.append((nid, pitch, end, assigned))

            max_voices = max(max_voices, len(active_notes))

        self.voice_assignments = voice_assignments
        self.num_voices = max_voices

        print(f"Assigned {len(voice_assignments)} notes to {max_voices} voices (pitch-ordered)")
        return voice_assignments

    def _assign_voices_balanced(self) -> Dict[int, int]:
        """
        Assign each note to a voice using balanced (least-time) assignment.

        At each note's start time:
        1. Remove expired notes (end_time <= start_time)
        2. Batch all note_on events at the same start time
        3. Sustained notes keep their voice (no mid-note reassignment)
        4. New notes are assigned to available voices (not occupied by sustained
           notes) sorted by least cumulative sounding time, ties broken by
           lowest voice number

        Returns:
            Dictionary mapping note_id to voice_number (1-indexed)
        """
        voice_assignments = {}
        # active_notes: list of (note_id, end_time, assigned_voice)
        active_notes = []
        # Cumulative sounding time per voice
        voice_sounding_time = defaultdict(float)
        max_voices = 0

        # First pass: determine max polyphony (needed to know how many voices exist)
        time_groups = []
        for start_time, group in groupby(self.notes, key=lambda n: n['start_time']):
            time_groups.append((start_time, list(group)))

        # Calculate max simultaneous notes to know voice count
        temp_active = []
        for start_time, new_notes in time_groups:
            temp_active = [end for end in temp_active if end > start_time]
            for n in new_notes:
                temp_active.append(n['end_time'])
            max_voices = max(max_voices, len(temp_active))

        for start_time, new_notes in time_groups:
            # Remove expired notes
            active_notes = [
                (nid, end, v) for nid, end, v in active_notes
                if end > start_time
            ]

            # Voices occupied by sustained notes
            occupied_voices = {v for _, _, v in active_notes}

            # Available voices sorted by (cumulative_time, voice_number)
            available = sorted(
                [v for v in range(1, max_voices + 1) if v not in occupied_voices],
                key=lambda v: (voice_sounding_time[v], v)
            )

            # Assign new notes to available voices in order
            for i, n in enumerate(new_notes):
                if i < len(available):
                    voice = available[i]
                else:
                    # More new notes than available voices — shouldn't happen if
                    # max_voices was calculated correctly, but handle gracefully
                    voice = max_voices + 1
                    max_voices = voice
                    available.append(voice)

                voice_assignments[n['id']] = voice
                voice_sounding_time[voice] += n['duration']
                active_notes.append((n['id'], n['end_time'], voice))

        self.voice_assignments = voice_assignments
        self.num_voices = max_voices

        print(f"Assigned {len(voice_assignments)} notes to {max_voices} voices (balanced)")
        return voice_assignments

    def build_voice_segments(self) -> Dict[int, List[Dict]]:
        """
        Build pitch segments for each voice, with rests filling gaps.

        Returns:
            Dictionary mapping voice_number to list of pitch segment dicts
        """
        if not self.voice_assignments:
            return {}

        # Group notes by voice
        voice_notes = defaultdict(list)
        for note in self.notes:
            voice = self.voice_assignments.get(note['id'])
            if voice is not None:
                voice_notes[voice].append(note)

        # Sort each voice's notes by start_time
        for voice in voice_notes:
            voice_notes[voice].sort(key=lambda n: n['start_time'])

        # Get total MIDI duration for trailing rests
        midi_total_duration = self.midi_file.length

        voice_segments = {}

        for voice_num in range(1, self.num_voices + 1):
            notes = voice_notes.get(voice_num, [])
            segments = self._build_segments_for_voice(notes, midi_total_duration)
            voice_segments[voice_num] = segments

        self.voice_segments = voice_segments
        return voice_segments

    def _build_segments_for_voice(self, notes: List[Dict],
                                   total_duration: float) -> List[Dict]:
        """
        Build segment list for a single voice with rest segments filling gaps.

        Args:
            notes: Sorted list of notes for this voice
            total_duration: Total MIDI file duration

        Returns:
            List of segment dicts compatible with pitch_matcher
        """
        segments = []
        current_time = 0.0
        segment_index = 0

        for note in notes:
            note_start = note['start_time']
            note_end = note['end_time']

            # Add rest before this note if there's a gap
            gap = note_start - current_time
            if gap >= self.min_rest:
                segments.append({
                    'index': segment_index,
                    'start_time': current_time,
                    'end_time': note_start,
                    'duration': gap,
                    'pitch_hz': 0.0,
                    'pitch_midi': -1,
                    'pitch_note': 'REST',
                    'pitch_confidence': 1.0,
                    'is_rest': True
                })
                segment_index += 1

            # Add note segment
            pitch_hz = float(midi_to_hz(note['pitch_midi']))
            pitch_note = midi_to_note_name(note['pitch_midi'])

            segments.append({
                'index': segment_index,
                'start_time': note_start,
                'end_time': note_end,
                'duration': note_end - note_start,
                'pitch_hz': pitch_hz,
                'pitch_midi': note['pitch_midi'],
                'pitch_note': pitch_note,
                'pitch_confidence': 1.0,
                'is_rest': False
            })
            segment_index += 1
            current_time = note_end

        # Trailing rest to fill out the song duration
        if current_time < total_duration:
            trailing_gap = total_duration - current_time
            if trailing_gap >= self.min_rest:
                segments.append({
                    'index': segment_index,
                    'start_time': current_time,
                    'end_time': total_duration,
                    'duration': trailing_gap,
                    'pitch_hz': 0.0,
                    'pitch_midi': -1,
                    'pitch_note': 'REST',
                    'pitch_confidence': 1.0,
                    'is_rest': True
                })

        return segments

    def merge_small_rests(self) -> None:
        """
        Merge consecutive notes with small gaps between them, per voice.

        Small gaps are absorbed into the preceding note by extending its end time.
        Applied before build_voice_segments to clean up micro-gaps.
        """
        for voice_num in range(1, self.num_voices + 1):
            notes_for_voice = []
            for note in self.notes:
                if self.voice_assignments.get(note['id']) == voice_num:
                    notes_for_voice.append(note)
            notes_for_voice.sort(key=lambda n: n['start_time'])

            if len(notes_for_voice) < 2:
                continue

            for i in range(1, len(notes_for_voice)):
                prev = notes_for_voice[i - 1]
                curr = notes_for_voice[i]
                gap = curr['start_time'] - prev['end_time']
                if 0 < gap < self.min_rest:
                    prev['end_time'] = curr['start_time']
                    prev['duration'] = prev['end_time'] - prev['start_time']

    def save_voice_results(self) -> List[Path]:
        """
        Save each voice's segments to a separate JSON file.

        Returns:
            List of paths to created JSON files
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        created_files = []

        for voice_num in range(1, self.num_voices + 1):
            segments = self.voice_segments.get(voice_num, [])

            if segments:
                total_duration = max(seg['end_time'] for seg in segments)
            else:
                total_duration = 0.0

            data = {
                'video_path': None,
                'audio_path': None,
                'midi_path': str(self.midi_path),
                'sample_rate': self.sample_rate,
                'pitch_detection_method': 'MIDI_VOICE_SPLIT',
                'midi_channel': self.channel,
                'voice_number': voice_num,
                'total_voices': self.num_voices,
                'num_segments': len(segments),
                'total_duration': total_duration,
                'pitch_segments': segments
            }

            output_path = self.output_dir / f"guide_sequence_voice{voice_num}.json"
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            created_files.append(output_path)

        return created_files

    def generate_audio_previews(self, temp_dir: str = 'data/temp') -> List[Path]:
        """
        Generate audio preview WAV files for each voice.

        Args:
            temp_dir: Directory for preview WAV files

        Returns:
            List of paths to created audio files
        """
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)

        player = MIDIPlayer(sample_rate=self.sample_rate)
        created_files = []

        for voice_num in range(1, self.num_voices + 1):
            segments = self.voice_segments.get(voice_num, [])
            if not segments:
                continue

            total_duration = max(seg['end_time'] for seg in segments)
            total_samples = int((total_duration + 1.0) * self.sample_rate)
            audio_buffer = np.zeros(total_samples, dtype=np.float32)

            for seg in segments:
                freq = seg['pitch_hz']
                if freq > 0 and seg['duration'] > 0:
                    tone = player.generate_tone(freq, seg['duration'])
                    start_sample = int(seg['start_time'] * self.sample_rate)
                    end_sample = start_sample + len(tone)

                    if end_sample > len(audio_buffer):
                        end_sample = len(audio_buffer)
                        tone = tone[:end_sample - start_sample]

                    if start_sample < len(audio_buffer):
                        audio_buffer[start_sample:end_sample] += tone

            # Normalize
            max_val = np.max(np.abs(audio_buffer))
            if max_val > 0:
                audio_buffer = audio_buffer / max_val * 0.8

            # Trim
            audio_buffer = audio_buffer[:int((total_duration + 0.5) * self.sample_rate)]

            output_path = temp_path / f"guide_voice{voice_num}_preview.wav"
            sf.write(str(output_path), audio_buffer, self.sample_rate)
            created_files.append(output_path)

        return created_files

    def split(self, generate_audio: bool = True,
              temp_dir: str = 'data/temp') -> Dict:
        """
        Orchestrate the full splitting pipeline.

        Args:
            generate_audio: Whether to generate audio previews
            temp_dir: Directory for audio previews

        Returns:
            Summary dict with results
        """
        self.load_midi()
        self.extract_all_notes()

        if not self.notes:
            print(f"\nError: No notes found on channel {self.channel}")
            print("Use --list-channels to see available channels")
            return {'error': 'no_notes', 'num_voices': 0}

        # Calculate polyphony info
        max_poly = self._calculate_channel_polyphony(self.channel)

        if max_poly <= 1:
            print(f"\nChannel {self.channel}: monophonic (max polyphony: 1)")
            print("No splitting needed — use midi_guide_converter.py instead.")
            # Still produce a single voice file for convenience
            self.assign_voices()
            self.merge_small_rests()
            self.build_voice_segments()
        else:
            print(f"\nChannel {self.channel}: {len(self.notes)} notes, max polyphony: {max_poly}")
            print(f"Splitting into {max_poly} voice files...")
            self.assign_voices()
            self.merge_small_rests()
            self.build_voice_segments()

        # Save JSON files
        json_files = self.save_voice_results()

        # Update audio_path in saved files if generating audio
        audio_files = []
        if generate_audio:
            audio_files = self.generate_audio_previews(temp_dir)
            # Update the audio_path in each JSON
            for voice_num, audio_path in enumerate(audio_files, 1):
                json_path = self.output_dir / f"guide_sequence_voice{voice_num}.json"
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    data['audio_path'] = str(audio_path)
                    with open(json_path, 'w') as f:
                        json.dump(data, f, indent=2)

        # Print summary
        print()
        midi_total_duration = self.midi_file.length
        for voice_num in range(1, self.num_voices + 1):
            segments = self.voice_segments.get(voice_num, [])
            note_segs = [s for s in segments if not s.get('is_rest', False)]
            sounding_time = sum(s['duration'] for s in note_segs)
            pct = (sounding_time / midi_total_duration * 100) if midi_total_duration > 0 else 0

            label = ""
            if voice_num == 1:
                label = " (lowest)"
            elif voice_num == self.num_voices:
                label = " (highest)"

            print(f"  Voice {voice_num}{label}: {len(note_segs):3d} notes, "
                  f"{sounding_time:.1f}s sounding ({pct:.1f}% of song)")
            print(f"    → {self.output_dir / f'guide_sequence_voice{voice_num}.json'}")

        if audio_files:
            print("\nAudio previews:")
            for path in audio_files:
                print(f"  → {path}")

        return {
            'num_voices': self.num_voices,
            'json_files': [str(p) for p in json_files],
            'audio_files': [str(p) for p in audio_files],
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Split polyphonic MIDI channel into monophonic voice guide sequences"
    )
    parser.add_argument(
        '--midi', type=str, required=True,
        help='Path to MIDI file'
    )
    parser.add_argument(
        '--channel', type=int, default=None,
        help='MIDI channel to split (0-15)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data/segments',
        help='Output directory for JSON files (default: data/segments)'
    )
    parser.add_argument(
        '--min-rest', type=float, default=0.1,
        help='Minimum rest duration to preserve in seconds (default: 0.1)'
    )
    parser.add_argument(
        '--sample-rate', type=int, default=22050,
        help='Sample rate for audio previews (default: 22050)'
    )
    parser.add_argument(
        '--no-audio', action='store_true',
        help='Skip audio preview generation'
    )
    parser.add_argument(
        '--strategy', type=str, default='pitch-ordered',
        choices=['pitch-ordered', 'balanced'],
        help='Voice assignment strategy (default: pitch-ordered)'
    )
    parser.add_argument(
        '--list-channels', action='store_true',
        help='List channels with polyphony info and exit'
    )

    args = parser.parse_args()

    print("=== MIDI Voice Splitter ===\n")

    # Validate MIDI file
    midi_path = Path(args.midi)
    if not midi_path.exists():
        print(f"Error: MIDI file not found: {args.midi}")
        return 1

    # List channels mode
    if args.list_channels:
        splitter = MIDIVoiceSplitter(args.midi, channel=0)
        splitter.load_midi()
        channels = splitter.list_channels()

        print(f"Channels found in {midi_path.name}:")
        print("-" * 60)

        if not channels:
            print("  No note events found in any channel")
        else:
            for ch in sorted(channels.keys()):
                info = channels[ch]
                instrument_str = ""
                if info['instrument']:
                    instrument_str = f", {info['instrument']}"

                drum_marker = " *" if ch == 9 else ""

                print(f"  Channel {ch:2d}: {info['note_count']:5d} notes, "
                      f"{info['unique_notes']:3d} unique, range: {info['note_range']}, "
                      f"max polyphony: {info['max_polyphony']}{instrument_str}{drum_marker}")

        print("-" * 60)

        # Check for percussion
        if 9 in channels:
            print("* = percussion channel (voice splitting may not produce meaningful results)")

        return 0

    # Validate channel
    if args.channel is None:
        print("Error: --channel is required (unless using --list-channels)")
        return 1

    if not 0 <= args.channel <= 15:
        print(f"Error: Channel must be 0-15, got {args.channel}")
        return 1

    # Warn about drums
    if args.channel == 9:
        print("Warning: Channel 9 is the percussion channel.")
        print("Voice splitting may not produce meaningful results.\n")

    # Run split
    splitter = MIDIVoiceSplitter(
        args.midi, args.channel,
        output_dir=args.output_dir,
        min_rest=args.min_rest,
        sample_rate=args.sample_rate,
        strategy=args.strategy
    )

    result = splitter.split(
        generate_audio=not args.no_audio,
        temp_dir='data/temp'
    )

    if result.get('error') == 'no_notes':
        return 1

    if result['num_voices'] <= 1:
        print("\n=== Split Complete (monophonic) ===")
    else:
        print(f"\n=== Split Complete ({result['num_voices']} voices) ===")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
