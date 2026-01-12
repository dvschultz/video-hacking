"""
Unit tests for midi_channel_splitter.py - MIDI channel splitting and audio export.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestMIDIChannelSplitterInitialization:
    """Test MIDIChannelSplitter initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        with patch.dict('sys.modules', {'mido': MagicMock()}):
            from midi_channel_splitter import MIDIChannelSplitter

            splitter = MIDIChannelSplitter("/path/to/test.mid")

            assert splitter.midi_path == Path("/path/to/test.mid")
            assert splitter.midi_file is None
            assert splitter.ticks_per_beat == 480


class TestMIDIChannelSplitterChannelListing:
    """Test channel listing functionality."""

    @pytest.fixture
    def mock_mido(self):
        """Create mock mido module."""
        mock = MagicMock()

        # Create mock MIDI file with tracks
        mock_track = [
            MagicMock(type='note_on', channel=0, note=60, velocity=100, time=0),
            MagicMock(type='note_off', channel=0, note=60, velocity=0, time=480),
            MagicMock(type='note_on', channel=1, note=64, velocity=80, time=0),
            MagicMock(type='note_off', channel=1, note=64, velocity=0, time=480),
            MagicMock(type='note_on', channel=0, note=62, velocity=100, time=0),
            MagicMock(type='note_off', channel=0, note=62, velocity=0, time=480),
        ]

        # Set hasattr for channel detection
        for msg in mock_track:
            if msg.type in ['note_on', 'note_off']:
                msg.channel = msg.channel  # Already set above

        mock_midi_file = MagicMock()
        mock_midi_file.tracks = [mock_track]
        mock_midi_file.ticks_per_beat = 480

        mock.MidiFile.return_value = mock_midi_file

        return mock

    def test_list_channels_finds_channels(self, mock_mido):
        """Test that list_channels finds all used channels."""
        with patch.dict('sys.modules', {'mido': mock_mido}):
            # Re-import to get patched version
            import importlib
            import midi_channel_splitter
            importlib.reload(midi_channel_splitter)

            splitter = midi_channel_splitter.MIDIChannelSplitter("/test.mid")
            splitter.midi_file = mock_mido.MidiFile.return_value

            channels = splitter.list_channels()

            # Should find channels 0 and 1
            assert 0 in channels
            assert 1 in channels

    def test_list_channels_counts_notes(self, mock_mido):
        """Test that list_channels counts notes correctly."""
        with patch.dict('sys.modules', {'mido': mock_mido}):
            import importlib
            import midi_channel_splitter
            importlib.reload(midi_channel_splitter)

            splitter = midi_channel_splitter.MIDIChannelSplitter("/test.mid")
            splitter.midi_file = mock_mido.MidiFile.return_value

            channels = splitter.list_channels()

            # Channel 0 should have 2 notes (60, 62)
            assert channels[0]['note_count'] == 2
            # Channel 1 should have 1 note (64)
            assert channels[1]['note_count'] == 1


class TestMIDIChannelSplitterTempoConversion:
    """Test tempo and tick conversion."""

    def test_ticks_to_seconds_default_tempo(self):
        """Test tick to seconds with default tempo (120 BPM)."""
        with patch.dict('sys.modules', {'mido': MagicMock()}):
            from midi_channel_splitter import MIDIChannelSplitter

            splitter = MIDIChannelSplitter("/test.mid")
            splitter.ticks_per_beat = 480

            # Default tempo: 500000 microseconds/beat = 120 BPM
            # 1 beat = 0.5 seconds
            # 480 ticks = 1 beat = 0.5 seconds
            tempo_changes = [(0, 500000)]

            result = splitter._ticks_to_seconds(480, tempo_changes)

            assert abs(result - 0.5) < 0.001

    def test_ticks_to_seconds_custom_tempo(self):
        """Test tick to seconds with custom tempo."""
        with patch.dict('sys.modules', {'mido': MagicMock()}):
            from midi_channel_splitter import MIDIChannelSplitter

            splitter = MIDIChannelSplitter("/test.mid")
            splitter.ticks_per_beat = 480

            # 60 BPM = 1000000 microseconds/beat
            # 1 beat = 1 second
            # 480 ticks = 1 beat = 1 second
            tempo_changes = [(0, 1000000)]

            result = splitter._ticks_to_seconds(480, tempo_changes)

            assert abs(result - 1.0) < 0.001

    def test_ticks_to_seconds_tempo_change(self):
        """Test tick to seconds with tempo change."""
        with patch.dict('sys.modules', {'mido': MagicMock()}):
            from midi_channel_splitter import MIDIChannelSplitter

            splitter = MIDIChannelSplitter("/test.mid")
            splitter.ticks_per_beat = 480

            # Start at 120 BPM, change to 60 BPM at tick 480
            tempo_changes = [
                (0, 500000),    # 120 BPM
                (480, 1000000)  # 60 BPM at tick 480
            ]

            # First 480 ticks at 120 BPM = 0.5s
            # Next 480 ticks at 60 BPM = 1.0s
            # Total 960 ticks = 1.5s
            result = splitter._ticks_to_seconds(960, tempo_changes)

            assert abs(result - 1.5) < 0.001

    def test_ticks_to_seconds_zero(self):
        """Test tick to seconds with zero ticks."""
        with patch.dict('sys.modules', {'mido': MagicMock()}):
            from midi_channel_splitter import MIDIChannelSplitter

            splitter = MIDIChannelSplitter("/test.mid")
            splitter.ticks_per_beat = 480

            tempo_changes = [(0, 500000)]
            result = splitter._ticks_to_seconds(0, tempo_changes)

            assert result == 0.0


class TestMIDIChannelSplitterNoteExtraction:
    """Test note extraction from channels."""

    @pytest.fixture
    def mock_mido_with_notes(self):
        """Create mock mido with note events."""
        mock = MagicMock()

        # Create track with note_on and note_off pairs
        # Note: msg.time is delta time from previous event
        mock_msgs = [
            MagicMock(type='set_tempo', tempo=500000, time=0),
            MagicMock(type='note_on', channel=0, note=60, velocity=100, time=0),
            MagicMock(type='note_off', channel=0, note=60, velocity=0, time=480),
            MagicMock(type='note_on', channel=0, note=64, velocity=100, time=0),
            MagicMock(type='note_off', channel=0, note=64, velocity=0, time=480),
        ]

        # hasattr check
        for msg in mock_msgs:
            if hasattr(msg, 'channel'):
                pass  # Already has channel

        mock_midi_file = MagicMock()
        mock_midi_file.tracks = [mock_msgs]
        mock_midi_file.ticks_per_beat = 480

        mock.MidiFile.return_value = mock_midi_file

        return mock

    def test_extract_notes_basic(self, mock_mido_with_notes):
        """Test basic note extraction."""
        with patch.dict('sys.modules', {'mido': mock_mido_with_notes}):
            import importlib
            import midi_channel_splitter
            importlib.reload(midi_channel_splitter)

            splitter = midi_channel_splitter.MIDIChannelSplitter("/test.mid")
            splitter.midi_file = mock_mido_with_notes.MidiFile.return_value
            splitter.ticks_per_beat = 480

            notes = splitter.extract_notes_for_channel(0)

            # Should find 2 notes
            assert len(notes) == 2

    def test_extract_notes_has_required_fields(self, mock_mido_with_notes):
        """Test that extracted notes have required fields."""
        with patch.dict('sys.modules', {'mido': mock_mido_with_notes}):
            import importlib
            import midi_channel_splitter
            importlib.reload(midi_channel_splitter)

            splitter = midi_channel_splitter.MIDIChannelSplitter("/test.mid")
            splitter.midi_file = mock_mido_with_notes.MidiFile.return_value
            splitter.ticks_per_beat = 480

            notes = splitter.extract_notes_for_channel(0)

            if notes:
                note = notes[0]
                assert 'pitch_midi' in note
                assert 'start_time' in note
                assert 'end_time' in note
                assert 'duration' in note


class TestMIDIChannelSplitterSynthesis:
    """Test audio synthesis."""

    def test_synthesize_empty_notes(self):
        """Test synthesis with empty note list."""
        with patch.dict('sys.modules', {'mido': MagicMock()}):
            from midi_channel_splitter import MIDIChannelSplitter

            splitter = MIDIChannelSplitter("/test.mid")

            # Mock MIDIPlayer
            with patch('midi_channel_splitter.MIDIPlayer'):
                audio = splitter.synthesize_channel([], sample_rate=22050)

            assert len(audio) == 0

    def test_synthesize_normalizes_output(self):
        """Test that synthesis normalizes output."""
        with patch.dict('sys.modules', {'mido': MagicMock()}):
            from midi_channel_splitter import MIDIChannelSplitter

            splitter = MIDIChannelSplitter("/test.mid")

            notes = [
                {'pitch_midi': 60, 'start_time': 0.0, 'end_time': 0.5, 'duration': 0.5}
            ]

            # Mock MIDIPlayer to return a simple tone
            mock_player = MagicMock()
            mock_player.generate_tone.return_value = np.ones(11025) * 0.5

            with patch('midi_channel_splitter.MIDIPlayer', return_value=mock_player):
                audio = splitter.synthesize_channel(notes, sample_rate=22050, total_duration=1.0)

            # Output should be normalized (max around 0.8)
            if len(audio) > 0:
                max_val = np.max(np.abs(audio))
                assert max_val <= 1.0


class TestMIDIChannelSplitterExport:
    """Test WAV export functionality."""

    def test_export_creates_directory(self, temp_dir):
        """Test that export creates output directory."""
        with patch.dict('sys.modules', {'mido': MagicMock()}):
            from midi_channel_splitter import MIDIChannelSplitter

            splitter = MIDIChannelSplitter("/test.mid")
            splitter.midi_path = Path("/test.mid")

            # Mock everything
            splitter.midi_file = MagicMock()
            splitter.midi_file.ticks_per_beat = 480
            splitter.midi_file.length = 1.0
            splitter.midi_file.tracks = []

            with patch.object(splitter, 'list_channels', return_value={}):
                output_dir = temp_dir / "midi_output"
                files = splitter.export_all_channels(str(output_dir))

            assert output_dir.exists()
