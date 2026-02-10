"""
Unit tests for midi_voice_splitter.py - MIDI polyphonic voice splitting.
"""

import pytest
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ============== Helper to build mock MIDI files ==============

def make_mock_midi(tracks_data, ticks_per_beat=480, length=10.0):
    """
    Build a mock mido.MidiFile from a list of track event lists.

    Each track is a list of mock messages with attributes:
        type, time, channel, note, velocity (for note events)
        type, time, tempo (for tempo events)

    Args:
        tracks_data: list of lists of dicts describing messages
        ticks_per_beat: MIDI resolution
        length: total file length in seconds (mocked)

    Returns:
        MagicMock mimicking mido.MidiFile
    """
    tracks = []
    for track_msgs in tracks_data:
        msgs = []
        for m in track_msgs:
            msg = MagicMock()
            msg.type = m['type']
            msg.time = m.get('time', 0)
            if 'channel' in m:
                msg.channel = m['channel']
            else:
                # Make hasattr(msg, 'channel') return False for meta messages
                del msg.channel
            if 'note' in m:
                msg.note = m['note']
            if 'velocity' in m:
                msg.velocity = m['velocity']
            if 'tempo' in m:
                msg.tempo = m['tempo']
            if 'program' in m:
                msg.program = m['program']
            msgs.append(msg)
        tracks.append(msgs)

    midi_file = MagicMock()
    midi_file.ticks_per_beat = ticks_per_beat
    midi_file.tracks = tracks
    type(midi_file).length = PropertyMock(return_value=length)
    return midi_file


def note_on(channel, note, velocity=100, time=0):
    return {'type': 'note_on', 'channel': channel, 'note': note,
            'velocity': velocity, 'time': time}


def note_off(channel, note, time=0):
    return {'type': 'note_off', 'channel': channel, 'note': note,
            'velocity': 0, 'time': time}


def tempo_change(tempo, time=0):
    return {'type': 'set_tempo', 'tempo': tempo, 'time': time}


def program_change(channel, program, time=0):
    return {'type': 'program_change', 'channel': channel, 'program': program, 'time': time}


# ============== Test Classes ==============

class TestMIDIVoiceSplitter:
    """Test basic initialization and loading."""

    @pytest.fixture
    def splitter(self, temp_dir):
        """Create a MIDIVoiceSplitter instance."""
        with patch('builtins.print'):
            from midi_voice_splitter import MIDIVoiceSplitter
            s = MIDIVoiceSplitter.__new__(MIDIVoiceSplitter)
            s.midi_path = temp_dir / "test.mid"
            s.channel = 4
            s.output_dir = temp_dir / "output"
            s.min_rest = 0.1
            s.sample_rate = 22050
            s.midi_file = None
            s.ticks_per_beat = 480
            s.notes = []
            s.voice_assignments = {}
            s.num_voices = 0
            s.voice_segments = {}
            s.strategy = 'pitch-ordered'
        return s

    def test_initialization(self, splitter):
        """Test splitter has correct initial state."""
        assert splitter.channel == 4
        assert splitter.ticks_per_beat == 480
        assert splitter.notes == []
        assert splitter.num_voices == 0

    def test_load_midi(self, splitter):
        """Test loading a MIDI file sets ticks_per_beat."""
        mock_midi = make_mock_midi([[]], ticks_per_beat=960)

        with patch('mido.MidiFile', return_value=mock_midi):
            with patch('builtins.print'):
                splitter.load_midi()

        assert splitter.ticks_per_beat == 960
        assert splitter.midi_file is not None


class TestVoiceAssignment:
    """Test voice assignment algorithm with various chord configurations."""

    @pytest.fixture
    def splitter(self, temp_dir):
        with patch('builtins.print'):
            from midi_voice_splitter import MIDIVoiceSplitter
            s = MIDIVoiceSplitter.__new__(MIDIVoiceSplitter)
            s.midi_path = temp_dir / "test.mid"
            s.channel = 0
            s.output_dir = temp_dir
            s.min_rest = 0.1
            s.sample_rate = 22050
            s.midi_file = None
            s.ticks_per_beat = 480
            s.notes = []
            s.voice_assignments = {}
            s.num_voices = 0
            s.voice_segments = {}
            s.strategy = 'pitch-ordered'
        return s

    def test_two_note_chord(self, splitter):
        """Two simultaneous notes: lower pitch = voice 1, higher = voice 2."""
        splitter.notes = [
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            {'id': 1, 'pitch_midi': 64, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
        ]
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        assert assignments[0] == 1  # C4 -> voice 1 (lowest)
        assert assignments[1] == 2  # E4 -> voice 2
        assert splitter.num_voices == 2

    def test_three_note_chord(self, splitter):
        """Three simultaneous notes: pitch-ordered voices."""
        splitter.notes = [
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            {'id': 1, 'pitch_midi': 64, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            {'id': 2, 'pitch_midi': 67, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
        ]
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        assert assignments[0] == 1  # C4 -> voice 1
        assert assignments[1] == 2  # E4 -> voice 2
        assert assignments[2] == 3  # G4 -> voice 3
        assert splitter.num_voices == 3

    def test_four_note_chord(self, splitter):
        """Four simultaneous notes: pitch-ordered voices."""
        splitter.notes = [
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            {'id': 1, 'pitch_midi': 64, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            {'id': 2, 'pitch_midi': 67, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            {'id': 3, 'pitch_midi': 72, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
        ]
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        assert assignments[0] == 1  # C4
        assert assignments[1] == 2  # E4
        assert assignments[2] == 3  # G4
        assert assignments[3] == 4  # C5
        assert splitter.num_voices == 4

    def test_monophonic_input(self, splitter):
        """Monophonic input: all notes assigned to voice 1."""
        splitter.notes = [
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 0.5, 'duration': 0.5},
            {'id': 1, 'pitch_midi': 64, 'start_time': 0.5, 'end_time': 1.0, 'duration': 0.5},
            {'id': 2, 'pitch_midi': 67, 'start_time': 1.0, 'end_time': 1.5, 'duration': 0.5},
        ]
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        assert assignments[0] == 1
        assert assignments[1] == 1
        assert assignments[2] == 1
        assert splitter.num_voices == 1

    def test_sequential_chords(self, splitter):
        """Two sequential chords: voices reassigned independently."""
        splitter.notes = [
            # Chord 1: C4, E4
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            {'id': 1, 'pitch_midi': 64, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            # Chord 2: G4, B4 (after chord 1 ends)
            {'id': 2, 'pitch_midi': 67, 'start_time': 1.0, 'end_time': 2.0, 'duration': 1.0},
            {'id': 3, 'pitch_midi': 71, 'start_time': 1.0, 'end_time': 2.0, 'duration': 1.0},
        ]
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        assert assignments[0] == 1  # C4 -> voice 1
        assert assignments[1] == 2  # E4 -> voice 2
        assert assignments[2] == 1  # G4 -> voice 1 (chord 1 expired)
        assert assignments[3] == 2  # B4 -> voice 2
        assert splitter.num_voices == 2

    def test_empty_notes(self, splitter):
        """Empty note list produces no assignments."""
        splitter.notes = []
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        assert assignments == {}
        assert splitter.num_voices == 0


class TestVoiceAssignmentEdgeCases:
    """Test edge cases in voice assignment."""

    @pytest.fixture
    def splitter(self, temp_dir):
        with patch('builtins.print'):
            from midi_voice_splitter import MIDIVoiceSplitter
            s = MIDIVoiceSplitter.__new__(MIDIVoiceSplitter)
            s.midi_path = temp_dir / "test.mid"
            s.channel = 0
            s.output_dir = temp_dir
            s.min_rest = 0.1
            s.sample_rate = 22050
            s.midi_file = None
            s.ticks_per_beat = 480
            s.notes = []
            s.voice_assignments = {}
            s.num_voices = 0
            s.voice_segments = {}
            s.strategy = 'pitch-ordered'
        return s

    def test_sustained_note_keeps_voice(self, splitter):
        """A sustained note keeps its voice even when chord changes around it."""
        splitter.notes = [
            # C4 sustains across both chords
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 2.0, 'duration': 2.0},
            # E4 in first chord only
            {'id': 1, 'pitch_midi': 64, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            # G4 enters in second chord
            {'id': 2, 'pitch_midi': 67, 'start_time': 1.0, 'end_time': 2.0, 'duration': 1.0},
        ]
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        # At t=0: C4 and E4 start → C4=voice1, E4=voice2
        assert assignments[0] == 1  # C4 -> voice 1
        assert assignments[1] == 2  # E4 -> voice 2
        # At t=1: E4 ends, C4 sustained (voice1), G4 enters
        # Active: C4(voice1, sustained) + G4(new)
        # Sorted by pitch: C4, G4 → positions 1, 2
        # C4 keeps voice 1 (sustained), G4 gets voice 2
        assert assignments[2] == 2  # G4 -> voice 2

    def test_note_off_before_note_on_at_same_tick(self, splitter):
        """When notes end and start at the same time, expired notes are removed first."""
        splitter.notes = [
            # First note ends at t=1.0
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            # Second note starts at t=1.0
            {'id': 1, 'pitch_midi': 64, 'start_time': 1.0, 'end_time': 2.0, 'duration': 1.0},
        ]
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        # Both should be voice 1 since they don't overlap
        assert assignments[0] == 1
        assert assignments[1] == 1
        assert splitter.num_voices == 1

    def test_unisons_same_pitch(self, splitter):
        """Unison notes (same pitch) get separate voices by note_on order."""
        splitter.notes = [
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            {'id': 1, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
        ]
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        # Both are pitch 60; first gets voice 1, second gets voice 2
        voices = {assignments[0], assignments[1]}
        assert voices == {1, 2}
        assert splitter.num_voices == 2

    def test_growing_polyphony(self, splitter):
        """Notes enter one at a time, building up polyphony."""
        splitter.notes = [
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 3.0, 'duration': 3.0},
            {'id': 1, 'pitch_midi': 64, 'start_time': 1.0, 'end_time': 3.0, 'duration': 2.0},
            {'id': 2, 'pitch_midi': 67, 'start_time': 2.0, 'end_time': 3.0, 'duration': 1.0},
        ]
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        # t=0: only C4 → voice 1
        assert assignments[0] == 1
        # t=1: C4(sustained, voice1) + E4(new) → sorted: C4, E4 → E4 gets voice 2
        assert assignments[1] == 2
        # t=2: C4(sustained, voice1) + E4(sustained, voice2) + G4(new)
        # sorted: C4, E4, G4 → G4 gets voice 3
        assert assignments[2] == 3
        assert splitter.num_voices == 3


class TestSegmentBuilding:
    """Test segment building for voices."""

    @pytest.fixture
    def splitter(self, temp_dir):
        with patch('builtins.print'):
            from midi_voice_splitter import MIDIVoiceSplitter
            s = MIDIVoiceSplitter.__new__(MIDIVoiceSplitter)
            s.midi_path = temp_dir / "test.mid"
            s.channel = 0
            s.output_dir = temp_dir
            s.min_rest = 0.1
            s.sample_rate = 22050
            s.midi_file = make_mock_midi([[]], length=5.0)
            s.ticks_per_beat = 480
            s.notes = []
            s.voice_assignments = {}
            s.num_voices = 0
            s.voice_segments = {}
            s.strategy = 'pitch-ordered'
        return s

    def test_rest_insertion(self, splitter):
        """Rest segments fill gaps between notes."""
        splitter.notes = [
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 0.5, 'duration': 0.5},
            {'id': 1, 'pitch_midi': 64, 'start_time': 1.0, 'end_time': 1.5, 'duration': 0.5},
        ]
        splitter.voice_assignments = {0: 1, 1: 1}
        splitter.num_voices = 1

        with patch('builtins.print'):
            segments = splitter.build_voice_segments()

        voice_1 = segments[1]
        rests = [s for s in voice_1 if s['is_rest']]
        notes = [s for s in voice_1 if not s['is_rest']]

        assert len(notes) == 2
        assert len(rests) >= 1  # At least the gap between notes

        # Check rest segment properties
        gap_rest = [r for r in rests if r['start_time'] == pytest.approx(0.5)]
        assert len(gap_rest) == 1
        assert gap_rest[0]['end_time'] == pytest.approx(1.0)
        assert gap_rest[0]['pitch_midi'] == -1
        assert gap_rest[0]['pitch_note'] == 'REST'

    def test_leading_rest(self, splitter):
        """Leading rest when first note doesn't start at 0."""
        splitter.notes = [
            {'id': 0, 'pitch_midi': 60, 'start_time': 1.0, 'end_time': 2.0, 'duration': 1.0},
        ]
        splitter.voice_assignments = {0: 1}
        splitter.num_voices = 1

        with patch('builtins.print'):
            segments = splitter.build_voice_segments()

        voice_1 = segments[1]
        assert voice_1[0]['is_rest'] is True
        assert voice_1[0]['start_time'] == 0.0
        assert voice_1[0]['end_time'] == 1.0

    def test_trailing_rest(self, splitter):
        """Trailing rest fills to end of MIDI file."""
        splitter.notes = [
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 2.0, 'duration': 2.0},
        ]
        splitter.voice_assignments = {0: 1}
        splitter.num_voices = 1

        with patch('builtins.print'):
            segments = splitter.build_voice_segments()

        voice_1 = segments[1]
        last_seg = voice_1[-1]
        assert last_seg['is_rest'] is True
        assert last_seg['end_time'] == pytest.approx(5.0)

    def test_voice_spans_full_duration(self, splitter):
        """Each voice's segments span the full song duration."""
        splitter.notes = [
            {'id': 0, 'pitch_midi': 60, 'start_time': 1.0, 'end_time': 3.0, 'duration': 2.0},
            {'id': 1, 'pitch_midi': 64, 'start_time': 1.0, 'end_time': 3.0, 'duration': 2.0},
        ]
        splitter.voice_assignments = {0: 1, 1: 2}
        splitter.num_voices = 2

        with patch('builtins.print'):
            segments = splitter.build_voice_segments()

        for voice_num in [1, 2]:
            segs = segments[voice_num]
            assert segs[0]['start_time'] == 0.0
            assert segs[-1]['end_time'] == pytest.approx(5.0)

    def test_multiple_voices_independent(self, splitter):
        """Each voice gets its own independent segments."""
        splitter.notes = [
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            {'id': 1, 'pitch_midi': 67, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
        ]
        splitter.voice_assignments = {0: 1, 1: 2}
        splitter.num_voices = 2

        with patch('builtins.print'):
            segments = splitter.build_voice_segments()

        voice_1_notes = [s for s in segments[1] if not s['is_rest']]
        voice_2_notes = [s for s in segments[2] if not s['is_rest']]

        assert len(voice_1_notes) == 1
        assert voice_1_notes[0]['pitch_midi'] == 60
        assert len(voice_2_notes) == 1
        assert voice_2_notes[0]['pitch_midi'] == 67


class TestOutputFormat:
    """Test JSON output format compatibility with pitch_matcher."""

    @pytest.fixture
    def splitter(self, temp_dir):
        with patch('builtins.print'):
            from midi_voice_splitter import MIDIVoiceSplitter
            s = MIDIVoiceSplitter.__new__(MIDIVoiceSplitter)
            s.midi_path = temp_dir / "test.mid"
            s.channel = 4
            s.output_dir = temp_dir / "output"
            s.min_rest = 0.1
            s.sample_rate = 22050
            s.midi_file = make_mock_midi([[]], length=5.0)
            s.ticks_per_beat = 480
            s.notes = [
                {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
                {'id': 1, 'pitch_midi': 64, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            ]
            s.voice_assignments = {0: 1, 1: 2}
            s.num_voices = 2
            s.voice_segments = {}
            s.strategy = 'pitch-ordered'
        return s

    def test_json_structure(self, splitter, temp_dir):
        """Output JSON has all required fields for pitch_matcher."""
        with patch('builtins.print'):
            splitter.build_voice_segments()
            files = splitter.save_voice_results()

        assert len(files) == 2

        for i, filepath in enumerate(files):
            assert filepath.exists()
            with open(filepath) as f:
                data = json.load(f)

            # Required top-level fields
            assert 'pitch_segments' in data
            assert 'num_segments' in data
            assert 'total_duration' in data
            assert 'midi_path' in data
            assert 'midi_channel' in data

            # Voice-specific metadata
            assert data['pitch_detection_method'] == 'MIDI_VOICE_SPLIT'
            assert data['voice_number'] == i + 1
            assert data['total_voices'] == 2

            # Segment format
            for seg in data['pitch_segments']:
                assert 'index' in seg
                assert 'start_time' in seg
                assert 'end_time' in seg
                assert 'duration' in seg
                assert 'pitch_hz' in seg
                assert 'pitch_midi' in seg
                assert 'pitch_note' in seg
                assert 'pitch_confidence' in seg
                assert 'is_rest' in seg

    def test_pitch_matcher_can_read(self, splitter, temp_dir):
        """Output JSON is readable by pitch_matcher's load_guide_sequence pattern."""
        with patch('builtins.print'):
            splitter.build_voice_segments()
            files = splitter.save_voice_results()

        # Simulate what pitch_matcher.load_guide_sequence does
        for filepath in files:
            with open(filepath) as f:
                data = json.load(f)

            guide_sequence = data['pitch_segments']
            assert len(guide_sequence) > 0

            # pitch_matcher accesses these fields
            for seg in guide_sequence:
                _ = seg['pitch_midi']
                _ = seg['start_time']
                _ = seg['end_time']
                _ = seg['duration']


class TestListChannels:
    """Test channel listing with polyphony info."""

    @pytest.fixture
    def splitter(self, temp_dir):
        with patch('builtins.print'):
            from midi_voice_splitter import MIDIVoiceSplitter
            s = MIDIVoiceSplitter.__new__(MIDIVoiceSplitter)
            s.midi_path = temp_dir / "test.mid"
            s.channel = 0
            s.output_dir = temp_dir
            s.min_rest = 0.1
            s.sample_rate = 22050
            s.ticks_per_beat = 480
            s.notes = []
            s.voice_assignments = {}
            s.num_voices = 0
            s.voice_segments = {}
            s.strategy = 'pitch-ordered'
        return s

    def test_list_channels_shows_polyphony(self, splitter):
        """Channel listing includes max_polyphony field."""
        # Channel 0: monophonic (notes sequential)
        # Channel 4: polyphonic (2 simultaneous notes at tick 0)
        track_data = [
            # Track with channel 0 (monophonic)
            note_on(0, 60, time=0),
            note_off(0, 60, time=480),
            note_on(0, 64, time=0),
            note_off(0, 64, time=480),
            # Track with channel 4 (polyphonic)
            note_on(4, 60, time=0),
            note_on(4, 64, time=0),
            note_off(4, 60, time=480),
            note_off(4, 64, time=0),
        ]

        splitter.midi_file = make_mock_midi([track_data])

        with patch('builtins.print'):
            channels = splitter.list_channels()

        assert 0 in channels
        assert 4 in channels

        assert channels[0]['max_polyphony'] == 1
        assert channels[4]['max_polyphony'] == 2

    def test_list_channels_shows_instrument(self, splitter):
        """Channel listing includes instrument name from program_change."""
        track_data = [
            program_change(0, 48),  # String Ensemble 1
            note_on(0, 60, time=0),
            note_off(0, 60, time=480),
        ]
        splitter.midi_file = make_mock_midi([track_data])

        with patch('builtins.print'):
            channels = splitter.list_channels()

        assert channels[0]['instrument'] == 'String Ensemble 1'

    def test_list_channels_drums(self, splitter):
        """Channel 9 shows as Drums regardless of program."""
        track_data = [
            note_on(9, 36, time=0),
            note_off(9, 36, time=480),
        ]
        splitter.midi_file = make_mock_midi([track_data])

        with patch('builtins.print'):
            channels = splitter.list_channels()

        assert channels[9]['instrument'] == 'Drums'

    def test_list_channels_note_range(self, splitter):
        """Channel listing includes note range."""
        track_data = [
            note_on(0, 48, time=0),
            note_off(0, 48, time=480),
            note_on(0, 72, time=0),
            note_off(0, 72, time=480),
        ]
        splitter.midi_file = make_mock_midi([track_data])

        with patch('builtins.print'):
            channels = splitter.list_channels()

        assert 'C3' in channels[0]['note_range']
        assert 'C5' in channels[0]['note_range']


class TestExtractAllNotes:
    """Test note extraction preserving overlaps."""

    @pytest.fixture
    def splitter(self, temp_dir):
        with patch('builtins.print'):
            from midi_voice_splitter import MIDIVoiceSplitter
            s = MIDIVoiceSplitter.__new__(MIDIVoiceSplitter)
            s.midi_path = temp_dir / "test.mid"
            s.channel = 0
            s.output_dir = temp_dir
            s.min_rest = 0.1
            s.sample_rate = 22050
            s.ticks_per_beat = 480
            s.midi_file = None
            s.notes = []
            s.voice_assignments = {}
            s.num_voices = 0
            s.voice_segments = {}
            s.strategy = 'pitch-ordered'
        return s

    def test_overlapping_notes_preserved(self, splitter):
        """Overlapping notes are all extracted (not monophonic)."""
        track_data = [
            note_on(0, 60, time=0),     # C4 starts at tick 0
            note_on(0, 64, time=0),     # E4 starts at tick 0
            note_off(0, 60, time=480),  # C4 ends at tick 480
            note_off(0, 64, time=0),    # E4 ends at tick 480
        ]
        splitter.midi_file = make_mock_midi([track_data])

        with patch('builtins.print'):
            notes = splitter.extract_all_notes()

        assert len(notes) == 2
        pitches = {n['pitch_midi'] for n in notes}
        assert pitches == {60, 64}

    def test_notes_have_ids(self, splitter):
        """Each extracted note has a unique ID."""
        track_data = [
            note_on(0, 60, time=0),
            note_on(0, 64, time=0),
            note_off(0, 60, time=480),
            note_off(0, 64, time=0),
        ]
        splitter.midi_file = make_mock_midi([track_data])

        with patch('builtins.print'):
            notes = splitter.extract_all_notes()

        ids = [n['id'] for n in notes]
        assert len(ids) == len(set(ids))  # All unique

    def test_filters_other_channels(self, splitter):
        """Only notes from the target channel are extracted."""
        track_data = [
            note_on(0, 60, time=0),
            note_on(1, 72, time=0),  # Different channel
            note_off(0, 60, time=480),
            note_off(1, 72, time=0),
        ]
        splitter.midi_file = make_mock_midi([track_data])

        with patch('builtins.print'):
            notes = splitter.extract_all_notes()

        assert len(notes) == 1
        assert notes[0]['pitch_midi'] == 60

    def test_sorted_by_start_time_then_pitch(self, splitter):
        """Notes are sorted by start_time, then pitch ascending."""
        track_data = [
            note_on(0, 67, time=0),     # G4 first in MIDI
            note_on(0, 60, time=0),     # C4 second in MIDI
            note_off(0, 67, time=480),
            note_off(0, 60, time=0),
        ]
        splitter.midi_file = make_mock_midi([track_data])

        with patch('builtins.print'):
            notes = splitter.extract_all_notes()

        assert notes[0]['pitch_midi'] == 60  # C4 first (lower pitch)
        assert notes[1]['pitch_midi'] == 67  # G4 second


class TestErrorHandling:
    """Test error conditions."""

    def test_missing_file(self, temp_dir):
        """Missing MIDI file produces error."""
        from midi_voice_splitter import MIDIVoiceSplitter
        s = MIDIVoiceSplitter(str(temp_dir / "nonexistent.mid"), channel=0)

        with pytest.raises(Exception):
            s.load_midi()

    def test_empty_channel(self, temp_dir):
        """Channel with no notes returns empty list."""
        from midi_voice_splitter import MIDIVoiceSplitter
        s = MIDIVoiceSplitter.__new__(MIDIVoiceSplitter)
        s.midi_path = temp_dir / "test.mid"
        s.channel = 5
        s.output_dir = temp_dir
        s.min_rest = 0.1
        s.sample_rate = 22050
        s.ticks_per_beat = 480
        s.notes = []
        s.voice_assignments = {}
        s.num_voices = 0
        s.voice_segments = {}
        s.strategy = 'pitch-ordered'

        # MIDI file with no notes on channel 5
        track_data = [
            note_on(0, 60, time=0),
            note_off(0, 60, time=480),
        ]
        s.midi_file = make_mock_midi([track_data])

        with patch('builtins.print'):
            notes = s.extract_all_notes()

        assert len(notes) == 0


class TestMergeSmallRests:
    """Test per-voice small rest merging."""

    @pytest.fixture
    def splitter(self, temp_dir):
        with patch('builtins.print'):
            from midi_voice_splitter import MIDIVoiceSplitter
            s = MIDIVoiceSplitter.__new__(MIDIVoiceSplitter)
            s.midi_path = temp_dir / "test.mid"
            s.channel = 0
            s.output_dir = temp_dir
            s.min_rest = 0.1
            s.sample_rate = 22050
            s.midi_file = make_mock_midi([[]], length=5.0)
            s.ticks_per_beat = 480
            s.notes = []
            s.voice_assignments = {}
            s.num_voices = 0
            s.voice_segments = {}
            s.strategy = 'pitch-ordered'
        return s

    def test_small_gaps_merged(self, splitter):
        """Small gaps between notes on same voice are absorbed."""
        splitter.notes = [
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 0.95,
             'duration': 0.95},
            {'id': 1, 'pitch_midi': 64, 'start_time': 1.0, 'end_time': 2.0,
             'duration': 1.0},
        ]
        splitter.voice_assignments = {0: 1, 1: 1}
        splitter.num_voices = 1

        with patch('builtins.print'):
            splitter.merge_small_rests()

        # The first note's end_time should be extended to 1.0
        assert splitter.notes[0]['end_time'] == pytest.approx(1.0)

    def test_large_gaps_preserved(self, splitter):
        """Gaps >= min_rest are preserved."""
        splitter.notes = [
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 0.5,
             'duration': 0.5},
            {'id': 1, 'pitch_midi': 64, 'start_time': 1.0, 'end_time': 2.0,
             'duration': 1.0},
        ]
        splitter.voice_assignments = {0: 1, 1: 1}
        splitter.num_voices = 1

        with patch('builtins.print'):
            splitter.merge_small_rests()

        # Gap is 0.5s (>= 0.1), so no change
        assert splitter.notes[0]['end_time'] == pytest.approx(0.5)


class TestBalancedAssignment:
    """Test balanced (least-time) voice assignment strategy."""

    @pytest.fixture
    def splitter(self, temp_dir):
        with patch('builtins.print'):
            from midi_voice_splitter import MIDIVoiceSplitter
            s = MIDIVoiceSplitter.__new__(MIDIVoiceSplitter)
            s.midi_path = temp_dir / "test.mid"
            s.channel = 0
            s.output_dir = temp_dir
            s.min_rest = 0.1
            s.sample_rate = 22050
            s.midi_file = None
            s.ticks_per_beat = 480
            s.notes = []
            s.voice_assignments = {}
            s.num_voices = 0
            s.voice_segments = {}
            s.strategy = 'balanced'
        return s

    def test_sequential_notes_round_robin(self, splitter):
        """Sequential notes are assigned to the voice with least accumulated time."""
        # With 2-voice polyphony established by a chord, sequential notes
        # should alternate between voices based on accumulated time
        splitter.notes = [
            # Chord establishes 2 voices
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            {'id': 1, 'pitch_midi': 64, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            # Sequential notes after chord — each goes to voice with least time
            {'id': 2, 'pitch_midi': 67, 'start_time': 1.0, 'end_time': 2.0, 'duration': 1.0},
            {'id': 3, 'pitch_midi': 72, 'start_time': 2.0, 'end_time': 3.0, 'duration': 1.0},
        ]
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        # Chord: both voices get 1.0s each (voices 1 and 2, order depends on available sort)
        # After chord: both voices have 1.0s, so next note goes to voice 1 (tiebreak lowest)
        assert assignments[2] in (1, 2)
        # Next note goes to whichever voice has less time
        assert assignments[3] in (1, 2)
        # The two sequential notes should go to different voices (round-robin effect)
        assert assignments[2] != assignments[3]

    def test_chord_uses_all_voices(self, splitter):
        """A chord with N notes uses N voices (no choice)."""
        splitter.notes = [
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            {'id': 1, 'pitch_midi': 64, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            {'id': 2, 'pitch_midi': 67, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
        ]
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        voices = {assignments[0], assignments[1], assignments[2]}
        assert voices == {1, 2, 3}
        assert splitter.num_voices == 3

    def test_sustained_notes_keep_voice(self, splitter):
        """A sustained note keeps its voice; new notes go to available voices."""
        splitter.notes = [
            # C4 sustains across both time groups
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 3.0, 'duration': 3.0},
            # E4 in first time group
            {'id': 1, 'pitch_midi': 64, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            # G4 enters after E4 ends
            {'id': 2, 'pitch_midi': 67, 'start_time': 1.0, 'end_time': 2.0, 'duration': 1.0},
        ]
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        # C4 and E4 get voices from the chord
        voice_c4 = assignments[0]
        voice_e4 = assignments[1]
        assert voice_c4 != voice_e4

        # At t=1.0: C4 still sustained (keeps its voice), E4 expired
        # G4 must go to the available voice (E4's old voice), not C4's
        voice_g4 = assignments[2]
        assert voice_g4 != voice_c4  # Can't use sustained voice
        assert voice_g4 == voice_e4  # Only available voice

    def test_balanced_distributes_evenly(self, splitter):
        """Monophonic section after a chord distributes notes more evenly than pitch-ordered."""
        splitter.notes = [
            # Opening chord establishes 3 voices
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            {'id': 1, 'pitch_midi': 64, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            {'id': 2, 'pitch_midi': 67, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            # 6 sequential notes — should distribute ~2 per voice
            {'id': 3, 'pitch_midi': 72, 'start_time': 1.0, 'end_time': 2.0, 'duration': 1.0},
            {'id': 4, 'pitch_midi': 74, 'start_time': 2.0, 'end_time': 3.0, 'duration': 1.0},
            {'id': 5, 'pitch_midi': 76, 'start_time': 3.0, 'end_time': 4.0, 'duration': 1.0},
            {'id': 6, 'pitch_midi': 77, 'start_time': 4.0, 'end_time': 5.0, 'duration': 1.0},
            {'id': 7, 'pitch_midi': 79, 'start_time': 5.0, 'end_time': 6.0, 'duration': 1.0},
            {'id': 8, 'pitch_midi': 81, 'start_time': 6.0, 'end_time': 7.0, 'duration': 1.0},
        ]
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        # Count notes per voice
        voice_counts = defaultdict(int)
        for voice in assignments.values():
            voice_counts[voice] += 1

        # Each voice should have 3 notes (1 from chord + 2 from sequence)
        assert splitter.num_voices == 3
        for v in range(1, 4):
            assert voice_counts[v] == 3, f"Voice {v} has {voice_counts[v]} notes, expected 3"

    def test_tiebreak_lowest_voice(self, splitter):
        """When voices have equal time, the lowest voice number wins."""
        # All sequential notes with equal duration — voice 1 should get first pick each round
        splitter.notes = [
            # Chord to establish 2 voices with equal time
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            {'id': 1, 'pitch_midi': 64, 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0},
            # First sequential note — both voices have 1.0s, voice 1 wins tiebreak
            {'id': 2, 'pitch_midi': 67, 'start_time': 1.0, 'end_time': 2.0, 'duration': 1.0},
        ]
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        # Voice 1 should win the tiebreak (lowest voice number)
        assert assignments[2] == 1

    def test_empty_notes_balanced(self, splitter):
        """Empty note list produces no assignments."""
        splitter.notes = []
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        assert assignments == {}
        assert splitter.num_voices == 0

    def test_monophonic_input_balanced(self, splitter):
        """Monophonic input: all notes assigned to voice 1."""
        splitter.notes = [
            {'id': 0, 'pitch_midi': 60, 'start_time': 0.0, 'end_time': 0.5, 'duration': 0.5},
            {'id': 1, 'pitch_midi': 64, 'start_time': 0.5, 'end_time': 1.0, 'duration': 0.5},
            {'id': 2, 'pitch_midi': 67, 'start_time': 1.0, 'end_time': 1.5, 'duration': 0.5},
        ]
        with patch('builtins.print'):
            assignments = splitter.assign_voices()

        assert assignments[0] == 1
        assert assignments[1] == 1
        assert assignments[2] == 1
        assert splitter.num_voices == 1


class TestTicksToSeconds:
    """Test tempo-aware tick to seconds conversion."""

    @pytest.fixture
    def splitter(self, temp_dir):
        with patch('builtins.print'):
            from midi_voice_splitter import MIDIVoiceSplitter
            s = MIDIVoiceSplitter.__new__(MIDIVoiceSplitter)
            s.ticks_per_beat = 480
            s.midi_path = temp_dir / "test.mid"
            s.channel = 0
        return s

    def test_default_tempo(self, splitter):
        """120 BPM: 480 ticks = 0.5 seconds."""
        tempo_changes = [(0, 500000)]  # 120 BPM
        result = splitter._ticks_to_seconds(480, tempo_changes)
        assert result == pytest.approx(0.5)

    def test_tempo_change(self, splitter):
        """Tempo change mid-file is handled correctly."""
        tempo_changes = [
            (0, 500000),    # 120 BPM
            (480, 250000),  # 240 BPM at beat 2
        ]
        result = splitter._ticks_to_seconds(960, tempo_changes)
        # First 480 ticks at 120BPM = 0.5s
        # Next 480 ticks at 240BPM = 0.25s
        assert result == pytest.approx(0.75)

    def test_zero_ticks(self, splitter):
        """Zero ticks = 0 seconds."""
        tempo_changes = [(0, 500000)]
        result = splitter._ticks_to_seconds(0, tempo_changes)
        assert result == 0.0
