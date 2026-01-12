"""
Shared fixtures and configuration for video-hacking tests.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import shutil
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============== Utility Fixtures ==============

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


# ============== Audio Fixtures ==============

@pytest.fixture
def sample_audio_array():
    """Generate a simple test audio array (440Hz sine wave)."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def sample_pitch_sequence_audio():
    """Generate audio with changing pitches (C4, E4, G4)."""
    sr = 22050
    duration_per_note = 0.5
    freqs = [261.63, 329.63, 392.0]  # C4, E4, G4

    audio = []
    for freq in freqs:
        t = np.linspace(0, duration_per_note, int(sr * duration_per_note))
        audio.append(0.5 * np.sin(2 * np.pi * freq * t))

    return np.concatenate(audio).astype(np.float32), sr


# ============== JSON Data Fixtures ==============

@pytest.fixture
def sample_onset_strength_data():
    """Sample onset strength analysis JSON structure."""
    return {
        'audio_file': 'test_audio.wav',
        'sample_rate': 22050,
        'analysis_rate': 24,
        'duration': 10.0,
        'num_frames': 240,
        'min_value': 0.0,
        'max_value': 1.0,
        'mean_value': 0.3,
        'non_zero_frames': 100,
        'onset_strength_values': [0.0] * 12 + [0.8, 0.1, 0.0] * 76,
        'times': [i / 24 for i in range(240)]
    }


@pytest.fixture
def sample_audio_segments_data():
    """Sample audio segments metadata JSON structure."""
    return {
        'source_audio': 'test_audio.wav',
        'onset_strength_json': 'onset_strength.json',
        'threshold': 0.2,
        'sample_rate': 22050,
        'num_segments': 5,
        'total_duration': 10.0,
        'segments': [
            {
                'index': i,
                'filename': f'segment_{i:04d}.wav',
                'start_time': i * 2.0,
                'end_time': (i + 1) * 2.0,
                'duration': 2.0,
                'num_samples': 44100
            }
            for i in range(5)
        ]
    }


@pytest.fixture
def sample_audio_embeddings_data():
    """Sample audio embeddings JSON structure."""
    np.random.seed(42)
    return {
        'model': 'imagebind_huge',
        'modality': 'audio',
        'num_segments': 5,
        'embedding_dim': 1024,
        'segments': [
            {
                'index': i,
                'filename': f'segment_{i:04d}.wav',
                'start_time': i * 2.0,
                'end_time': (i + 1) * 2.0,
                'duration': 2.0,
                'embedding': np.random.randn(1024).tolist(),
                'embedding_dim': 1024
            }
            for i in range(5)
        ]
    }


@pytest.fixture
def sample_video_embeddings_data():
    """Sample video embeddings JSON structure."""
    np.random.seed(42)
    return {
        'video_path': 'test_video.mp4',
        'video_fps': 24.0,
        'extraction_fps': 24,
        'total_frames': 240,
        'window_size': 5,
        'stride': 6,
        'num_windows': 40,
        'embeddings': [
            {
                'window_idx': i,
                'start_frame': i * 6,
                'end_frame': i * 6 + 5,
                'center_frame': i * 6 + 2,
                'start_time': i * 0.25,
                'end_time': i * 0.25 + 0.2083,
                'center_time': i * 0.25 + 0.1042,
                'embedding': np.random.randn(1024).tolist()
            }
            for i in range(40)
        ]
    }


@pytest.fixture
def sample_guide_sequence_data():
    """Sample guide sequence JSON structure."""
    return {
        'video_path': 'guide_video.mp4',
        'audio_path': 'guide_audio.wav',
        'fps': 24,
        'sample_rate': 22050,
        'pitch_detection_method': 'CREPE',
        'num_segments': 10,
        'total_duration': 5.0,
        'pitch_segments': [
            {
                'index': i,
                'start_time': i * 0.5,
                'end_time': (i + 1) * 0.5,
                'duration': 0.5,
                'pitch_hz': 440 * (2 ** (i / 12)),
                'pitch_midi': 69 + i,
                'pitch_note': ['A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5'][i],
                'pitch_confidence': 0.9,
                'is_rest': False
            }
            for i in range(10)
        ]
    }


@pytest.fixture
def sample_source_database_data():
    """Sample source database JSON structure."""
    pitch_database = []
    pitch_index = {}

    for i in range(20):
        midi = 60 + (i % 12)  # C4 to B4
        seg = {
            'segment_id': i,
            'start_time': i * 0.3,
            'end_time': (i + 1) * 0.3,
            'duration': 0.3,
            'pitch_hz': 261.63 * (2 ** ((midi - 60) / 12)),
            'pitch_midi': midi,
            'pitch_note': ['C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4'][midi - 60],
            'pitch_confidence': 0.85,
            'video_path': 'source_video.mp4',
            'video_start_frame': i * 7,
            'video_end_frame': (i + 1) * 7,
            'loopability': 0.8
        }
        pitch_database.append(seg)

        midi_str = str(midi)
        if midi_str not in pitch_index:
            pitch_index[midi_str] = []
        pitch_index[midi_str].append(i)

    return {
        'source_videos': [{'video_path': 'source_video.mp4', 'fps': 24}],
        'num_videos': 1,
        'num_segments': 20,
        'num_unique_pitches': 12,
        'pitch_database': pitch_database,
        'pitch_index': pitch_index,
        'silence_segments': [
            {
                'video_path': 'source_video.mp4',
                'video_start_frame': 200,
                'video_end_frame': 250,
                'duration': 2.0,
                'start_time': 8.33
            }
        ]
    }


@pytest.fixture
def sample_match_plan_data():
    """Sample match plan JSON structure."""
    return {
        'guide_sequence_path': 'guide_sequence.json',
        'source_database_path': 'source_database.json',
        'matching_config': {
            'reuse_policy': 'min_gap',
            'duration_weight': 0.3,
            'confidence_weight': 0.4,
            'allow_transposition': True,
            'max_transposition_semitones': 12
        },
        'statistics': {
            'total_guide_segments': 5,
            'exact_matches': 4,
            'transposed_matches': 1,
            'missing_matches': 0,
            'rest_segments': 0
        },
        'matches': [
            {
                'guide_segment_id': i,
                'guide_pitch_note': 'C4',
                'guide_pitch_midi': 60,
                'guide_start_time': i * 0.5,
                'guide_end_time': (i + 1) * 0.5,
                'guide_duration': 0.5,
                'match_type': 'exact',
                'transpose_semitones': 0,
                'source_segment_id': i,
                'source_clips': [{
                    'segment_id': i,
                    'video_path': 'source.mp4',
                    'video_start_frame': i * 12,
                    'video_end_frame': (i + 1) * 12,
                    'duration': 0.5
                }]
            }
            for i in range(5)
        ]
    }


# ============== Mock Fixtures ==============

@pytest.fixture
def mock_ffmpeg():
    """Mock FFmpeg subprocess calls."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b'',
            stderr=b''
        )
        yield mock_run


@pytest.fixture
def mock_ffprobe():
    """Mock FFprobe for video metadata."""
    with patch('subprocess.run') as mock_run:
        def side_effect(cmd, *args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = b'1920,1080\n'
            if '-show_entries' in cmd and 'r_frame_rate' in str(cmd):
                result.stdout = b'24/1\n'
            elif '-show_entries' in cmd and 'duration' in str(cmd):
                result.stdout = b'10.0\n'
            return result

        mock_run.side_effect = side_effect
        yield mock_run


@pytest.fixture
def mock_librosa_load():
    """Mock librosa.load to avoid reading actual audio files."""
    with patch('librosa.load') as mock_load:
        sr = 22050
        duration = 5.0
        audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
        mock_load.return_value = (audio, sr)
        yield mock_load


@pytest.fixture
def mock_soundfile_write():
    """Mock soundfile.write to avoid writing actual audio files."""
    with patch('soundfile.write') as mock_write:
        yield mock_write


@pytest.fixture
def mock_crepe_predict():
    """Mock CREPE pitch detection."""
    with patch('crepe.predict') as mock_predict:
        num_frames = 100
        times = np.linspace(0, 1, num_frames)
        frequencies = np.full(num_frames, 440.0)
        confidences = np.full(num_frames, 0.9)
        activations = np.zeros((num_frames, 360))

        mock_predict.return_value = (times, frequencies, confidences, activations)
        yield mock_predict


# ============== File-based Fixtures ==============

@pytest.fixture
def audio_embeddings_file(temp_dir, sample_audio_embeddings_data):
    """Create a temporary audio embeddings JSON file."""
    path = temp_dir / "audio_embeddings.json"
    path.write_text(json.dumps(sample_audio_embeddings_data))
    return path


@pytest.fixture
def video_embeddings_file(temp_dir, sample_video_embeddings_data):
    """Create a temporary video embeddings JSON file."""
    path = temp_dir / "video_embeddings.json"
    path.write_text(json.dumps(sample_video_embeddings_data))
    return path


@pytest.fixture
def audio_segments_file(temp_dir, sample_audio_segments_data):
    """Create a temporary audio segments JSON file."""
    path = temp_dir / "audio_segments.json"
    path.write_text(json.dumps(sample_audio_segments_data))
    return path


@pytest.fixture
def guide_sequence_file(temp_dir, sample_guide_sequence_data):
    """Create a temporary guide sequence JSON file."""
    path = temp_dir / "guide_sequence.json"
    path.write_text(json.dumps(sample_guide_sequence_data))
    return path


@pytest.fixture
def source_database_file(temp_dir, sample_source_database_data):
    """Create a temporary source database JSON file."""
    path = temp_dir / "source_database.json"
    path.write_text(json.dumps(sample_source_database_data))
    return path
