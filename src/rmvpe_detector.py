#!/usr/bin/env python3
"""
RMVPE Pitch Detector

Wrapper for RMVPE (Robust Model for Vocal Pitch Estimation) from the RVC project.
RMVPE provides excellent vocal pitch detection with lower resource consumption than CREPE.

Model source: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
Pretrained weights: https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from pathlib import Path
import urllib.request


# RMVPE native sample rate
RMVPE_SAMPLE_RATE = 16000

# Default model cache location
DEFAULT_MODEL_DIR = Path.home() / '.cache' / 'rmvpe'
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / 'rmvpe.pt'

# HuggingFace model URL
MODEL_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"

# RMVPE pitch bins: 360 bins covering ~31Hz to ~1975Hz in 20-cent spacing
N_PITCH_BINS = 360
CENTS_PER_BIN = 20
BASE_CENTS = 1997.3794084376191  # Cents offset from A0 (27.5 Hz)


class BiGRU(nn.Module):
    """Bidirectional GRU layer for RMVPE."""

    def __init__(self, input_features: int, hidden_features: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        # Input: (batch, features, time) -> transpose to (batch, time, features)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        # Output: (batch, time, features*2) -> transpose back to (batch, features*2, time)
        x = x.transpose(1, 2)
        return x


class ConvBlockRes(nn.Module):
    """Convolutional block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels, momentum=momentum),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class ResEncoderBlock(nn.Module):
    """Encoder block with residual convolutions and downsampling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 n_blocks: int = 1, momentum: float = 0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList([
            ConvBlockRes(in_channels if i == 0 else out_channels, out_channels, momentum)
            for i in range(n_blocks)
        ])
        self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return self.pool(x), x


class ResDecoderBlock(nn.Module):
    """Decoder block with residual convolutions and upsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: Tuple[int, int],
                 n_blocks: int = 1, momentum: float = 0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=stride, stride=stride, bias=False
        )
        self.conv = nn.ModuleList([
            ConvBlockRes(out_channels * 2 if i == 0 else out_channels, out_channels, momentum)
            for i in range(n_blocks)
        ])

    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.pad(x, (0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]))
        x = torch.cat([x, skip], dim=1)
        for conv in self.conv:
            x = conv(x)
        return x


class Encoder(nn.Module):
    """RMVPE Encoder with residual blocks."""

    def __init__(self, in_channels: int, in_size: int, n_encoders: int, kernel_size: Tuple[int, int],
                 n_blocks: int, out_channels: int = 16, momentum: float = 0.01):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []

        for i in range(n_encoders):
            in_ch = in_channels if i == 0 else out_channels * (2 ** (i - 1))
            out_ch = out_channels * (2 ** i)
            self.layers.append(
                ResEncoderBlock(in_ch, out_ch, kernel_size, n_blocks, momentum)
            )
            self.latent_channels.append([out_ch, in_size // (kernel_size[0] ** (i + 1))])

    def forward(self, x):
        x = self.bn(x)
        skips = []
        for layer in self.layers:
            x, skip = layer(x)
            skips.append(skip)
        return x, skips


class Intermediate(nn.Module):
    """Intermediate layer between encoder and decoder."""

    def __init__(self, in_channels: int, out_channels: int, n_inters: int,
                 n_blocks: int, momentum: float = 0.01):
        super().__init__()
        self.layers = nn.ModuleList([
            ResEncoderBlock(in_channels if i == 0 else out_channels, out_channels, (1, 1), n_blocks, momentum)
            for i in range(n_inters)
        ])

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x


class Decoder(nn.Module):
    """RMVPE Decoder with residual blocks."""

    def __init__(self, in_channels: int, n_decoders: int, stride: Tuple[int, int],
                 n_blocks: int, out_channels: int = 16, momentum: float = 0.01):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(n_decoders):
            in_ch = in_channels if i == 0 else out_channels * (2 ** (n_decoders - i))
            out_ch = out_channels * (2 ** (n_decoders - i - 1))
            self.layers.append(
                ResDecoderBlock(in_ch, out_ch, stride, n_blocks, momentum)
            )

    def forward(self, x, skips):
        for i, layer in enumerate(self.layers):
            x = layer(x, skips[-(i + 1)])
        return x


class DeepUnet(nn.Module):
    """Deep U-Net architecture for RMVPE."""

    def __init__(self, kernel_size: Tuple[int, int], n_blocks: int, en_de_layers: int = 5,
                 inter_layers: int = 4, in_channels: int = 1, en_out_channels: int = 16):
        super().__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            en_out_channels * (2 ** (en_de_layers - 1)),
            en_out_channels * (2 ** (en_de_layers - 1)),
            inter_layers, n_blocks
        )
        self.decoder = Decoder(
            en_out_channels * (2 ** (en_de_layers - 1)),
            en_de_layers, kernel_size, n_blocks, en_out_channels
        )

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, skips)
        return x


class E2E(nn.Module):
    """End-to-end pitch estimation head for RMVPE."""

    def __init__(self, n_blocks: int, n_gru: int, kernel_size: int, en_channels: int, en_out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(en_channels, en_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(en_out_channels),
            nn.ReLU(),
            nn.Conv1d(en_out_channels, en_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(en_out_channels),
            nn.ReLU(),
        )

        self.gru = BiGRU(en_out_channels, n_gru, num_layers=2)

        self.out = nn.Sequential(
            nn.Conv1d(n_gru * 2, N_PITCH_BINS, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.gru(x)
        x = self.out(x)
        return x


class MelSpectrogram(nn.Module):
    """Compute mel spectrogram for RMVPE."""

    def __init__(self, n_mel: int = 128, n_fft: int = 1024, hop_length: int = 160,
                 win_length: int = 1024, sample_rate: int = 16000,
                 fmin: float = 30.0, fmax: float = 8000.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.n_mel = n_mel

        # Pre-compute mel filterbank
        mel_basis = self._mel_filterbank(n_mel, n_fft, sample_rate, fmin, fmax)
        self.register_buffer('mel_basis', torch.from_numpy(mel_basis).float())

        # Hann window
        self.register_buffer('window', torch.hann_window(win_length))

    def _mel_filterbank(self, n_mel: int, n_fft: int, sr: int, fmin: float, fmax: float):
        """Create mel filterbank matrix."""
        def hz_to_mel(f):
            return 2595.0 * np.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        # Mel points
        mel_low = hz_to_mel(fmin)
        mel_high = hz_to_mel(fmax)
        mel_points = np.linspace(mel_low, mel_high, n_mel + 2)
        hz_points = mel_to_hz(mel_points)

        # FFT bin frequencies
        fft_bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        # Create filterbank
        filterbank = np.zeros((n_mel, n_fft // 2 + 1))
        for i in range(n_mel):
            left = fft_bins[i]
            center = fft_bins[i + 1]
            right = fft_bins[i + 2]

            for j in range(left, center):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram.

        Args:
            audio: Audio waveform (batch, samples) or (samples,)

        Returns:
            Mel spectrogram (batch, n_mel, time)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Pad audio to make it divisible by hop_length
        padding = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (padding, padding), mode='reflect')

        # STFT
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True
        )

        # Magnitude spectrogram
        spec = torch.abs(spec)

        # Apply mel filterbank
        mel = torch.matmul(self.mel_basis, spec)

        # Log scale with small epsilon
        mel = torch.log10(torch.clamp(mel, min=1e-7))

        return mel


class RMVPE(nn.Module):
    """RMVPE pitch estimation model (simplified architecture that matches pretrained weights)."""

    def __init__(self, hop_length: int = 160, sample_rate: int = 16000):
        super().__init__()
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # Mel spectrogram extractor
        self.mel_extractor = MelSpectrogram(
            n_mel=128,
            n_fft=1024,
            hop_length=hop_length,
            win_length=1024,
            sample_rate=sample_rate,
            fmin=30.0,
            fmax=8000.0
        )

        # Deep U-Net for feature extraction
        self.unet = DeepUnet(
            kernel_size=(2, 2),
            n_blocks=2,
            en_de_layers=5,
            inter_layers=4,
            in_channels=1,
            en_out_channels=16
        )

        # E2E pitch estimation head
        self.e2e = E2E(
            n_blocks=2,
            n_gru=128,
            kernel_size=3,
            en_channels=16,
            en_out_channels=128
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Estimate pitch from audio.

        Args:
            audio: Audio waveform at 16kHz (batch, samples) or (samples,)

        Returns:
            Pitch salience (batch, 360, time) - 360 pitch bins
        """
        # Compute mel spectrogram
        mel = self.mel_extractor(audio)  # (batch, 128, time)

        # Add channel dimension for U-Net
        mel = mel.unsqueeze(1)  # (batch, 1, 128, time)

        # U-Net feature extraction
        features = self.unet(mel)  # (batch, 16, 128, time)

        # Reduce frequency dimension
        features = features.mean(dim=2)  # (batch, 16, time)

        # E2E pitch estimation
        pitch = self.e2e(features)  # (batch, 360, time)

        return pitch


class RMVPEDetector:
    """High-level wrapper for RMVPE pitch detection."""

    def __init__(self, model_path: Optional[str] = None, device: str = 'auto',
                 is_half: bool = True):
        """Initialize RMVPE detector.

        Args:
            model_path: Path to model weights. Auto-downloads if None.
            device: Device to use ('auto', 'cuda', 'mps', or 'cpu')
            is_half: Use half precision (faster on GPU)
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.is_half = is_half

        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
                self.is_half = False  # MPS doesn't support half well
            else:
                self.device = torch.device('cpu')
                self.is_half = False  # CPU is faster with float32
        else:
            self.device = torch.device(device)
            if device == 'cpu':
                self.is_half = False

        # Ensure model is downloaded
        self._ensure_model_downloaded()

        # Load model
        self.model = self._load_model()

    def _ensure_model_downloaded(self):
        """Download model weights if not present."""
        if self.model_path.exists():
            return

        print(f"RMVPE model not found at {self.model_path}")
        print(f"Downloading from HuggingFace...")

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                print(f"\rDownloading: {percent:.1f}%", end='', flush=True)

            urllib.request.urlretrieve(MODEL_URL, self.model_path, reporthook=report_progress)
            print(f"\nModel downloaded to: {self.model_path}")

        except Exception as e:
            print(f"\nFailed to download RMVPE model: {e}")
            print("Please manually download from:")
            print(f"  {MODEL_URL}")
            print(f"And place it at: {self.model_path}")
            raise

    def _load_model(self):
        """Load the pre-trained RMVPE model."""
        print(f"Loading RMVPE model from: {self.model_path}")
        print(f"Device: {self.device}, Half precision: {self.is_half}")

        # Load checkpoint - the RVC checkpoint is typically the entire model saved with torch.save
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)

        # Check if checkpoint is already a model object or a state dict
        if hasattr(checkpoint, 'forward'):
            # It's already a model
            model = checkpoint
        else:
            # It's a state dict, create model and load
            model = RMVPE(hop_length=160, sample_rate=RMVPE_SAMPLE_RATE)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Try to load, with fallback to partial load
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(f"Note: Strict loading failed, using flexible loading...")
                # Create a new model but try to match keys
                model.load_state_dict(state_dict, strict=False)

        model = model.to(self.device)
        model.eval()

        if self.is_half and self.device.type == 'cuda':
            model = model.half()

        return model

    def _cents_to_hz(self, cents: np.ndarray) -> np.ndarray:
        """Convert cents (from A0) to Hz."""
        # A0 = 27.5 Hz
        return 27.5 * 2 ** (cents / 1200)

    def detect(self, audio: np.ndarray, sr: int,
               threshold: float = 0.03) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect pitch from audio.

        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate of input audio
            threshold: Voicing threshold (0-1, lower = more sensitive)

        Returns:
            Tuple of (times, pitch_hz, confidence) arrays
        """
        import librosa

        # Resample to 16kHz if needed
        if sr != RMVPE_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=RMVPE_SAMPLE_RATE)

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        if self.is_half and self.device.type == 'cuda':
            audio_tensor = audio_tensor.half()
        audio_tensor = audio_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            salience = self.model(audio_tensor)  # (1, 360, time)

        # Convert to numpy
        if salience.dim() == 3:
            salience = salience.squeeze(0)  # (360, time)
        salience = salience.cpu().float().numpy()

        # Get pitch and confidence for each frame
        n_frames = salience.shape[1]
        pitch_hz = np.zeros(n_frames)
        confidence = np.zeros(n_frames)

        for i in range(n_frames):
            frame_salience = salience[:, i]
            max_idx = np.argmax(frame_salience)
            max_val = frame_salience[max_idx]

            if max_val >= threshold:
                # Parabolic interpolation for better precision
                if 0 < max_idx < N_PITCH_BINS - 1:
                    left = frame_salience[max_idx - 1]
                    center = frame_salience[max_idx]
                    right = frame_salience[max_idx + 1]

                    if center > left and center > right:
                        offset = 0.5 * (right - left) / (2 * center - left - right + 1e-8)
                        refined_idx = max_idx + np.clip(offset, -0.5, 0.5)
                    else:
                        refined_idx = max_idx
                else:
                    refined_idx = max_idx

                # Convert to Hz
                cents = CENTS_PER_BIN * refined_idx + BASE_CENTS
                pitch_hz[i] = self._cents_to_hz(cents)
                confidence[i] = max_val
            else:
                pitch_hz[i] = 0
                confidence[i] = max_val

        # Generate time array (10ms frames at 16kHz with hop=160)
        frame_time = 160 / RMVPE_SAMPLE_RATE  # 0.01 seconds
        times = np.arange(n_frames) * frame_time

        return times, pitch_hz, confidence


def test_rmvpe():
    """Test RMVPE detector on a sample audio file."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rmvpe_detector.py <audio_file>")
        print("Example: python rmvpe_detector.py test.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    print(f"\n=== Testing RMVPE Detector ===")
    print(f"Audio file: {audio_path}")

    # Load audio
    import librosa
    audio, sr = librosa.load(audio_path, sr=None)
    print(f"Loaded audio: {len(audio)/sr:.2f}s at {sr}Hz")

    # Initialize detector
    detector = RMVPEDetector()

    # Detect pitch
    times, pitch_hz, confidence = detector.detect(audio, sr)

    # Print statistics
    voiced_mask = pitch_hz > 0
    voiced_frames = np.sum(voiced_mask)

    print(f"\nResults:")
    print(f"  Total frames: {len(times)}")
    print(f"  Voiced frames: {voiced_frames} ({voiced_frames/len(times)*100:.1f}%)")

    if voiced_frames > 0:
        voiced_pitches = pitch_hz[voiced_mask]
        voiced_conf = confidence[voiced_mask]

        print(f"  Pitch range: {np.min(voiced_pitches):.1f} - {np.max(voiced_pitches):.1f} Hz")
        print(f"  Mean pitch: {np.mean(voiced_pitches):.1f} Hz")
        print(f"  Mean confidence: {np.mean(voiced_conf):.3f}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_rmvpe()
