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
        return self.gru(x)[0]


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
        )
        # Shortcut connection when dimensions change
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=True)
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        x = self.conv(x)
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        return F.relu(x + residual)


class ResEncoderBlock(nn.Module):
    """Encoder block with residual convolutions and downsampling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 n_blocks: int = 1, momentum: float = 0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return self.pool(x), x


class ResDecoderBlock(nn.Module):
    """Decoder block with upsampling and residual convolutions."""

    def __init__(self, in_channels: int, out_channels: int, stride: Tuple[int, int],
                 n_blocks: int = 1, momentum: float = 0.01):
        super().__init__()
        self.n_blocks = n_blocks
        # Calculate output padding for transpose conv
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        # Residual conv blocks - first one takes concatenated skip connection
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, skip):
        x = self.conv1(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.pad(x, (0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]))
        x = torch.cat([x, skip], dim=1)
        for conv in self.conv2:
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
            self.latent_channels.append(out_ch)

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
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, (1, 1), n_blocks, momentum))
        for _ in range(n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, (1, 1), n_blocks, momentum))

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
        # Intermediate: 256 -> 512 channels (doubles)
        encoder_out_channels = en_out_channels * (2 ** (en_de_layers - 1))
        intermediate_channels = encoder_out_channels * 2
        self.intermediate = Intermediate(
            encoder_out_channels,
            intermediate_channels,
            inter_layers, n_blocks
        )
        # Decoder starts from intermediate output channels
        self.decoder = Decoder(
            intermediate_channels,
            en_de_layers, kernel_size, n_blocks, en_out_channels
        )

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, skips)
        return x


class E2E(nn.Module):
    """End-to-end RMVPE model (contains DeepUnet + CNN + FC)."""

    def __init__(self, n_blocks: int, n_gru: int, kernel_size: Tuple[int, int],
                 en_de_layers: int = 5, inter_layers: int = 4,
                 in_channels: int = 1, en_out_channels: int = 16):
        super().__init__()
        self.unet = DeepUnet(
            kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * 128, 360),
                nn.Dropout(0.25),
                nn.Sigmoid()
            )

    def forward(self, mel):
        """Forward pass for E2E model.

        Args:
            mel: (batch, 128, time) mel spectrogram
        """
        # Original: mel.transpose(-1, -2).unsqueeze(1)
        # (batch, 128, time) -> (batch, time, 128) -> (batch, 1, time, 128)
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class MelSpectrogram(nn.Module):
    """Mel spectrogram extractor for RMVPE."""

    def __init__(self, n_mel: int = 128, n_fft: int = 1024, hop_length: int = 160,
                 win_length: int = 1024, sample_rate: int = 16000,
                 fmin: float = 30.0, fmax: float = 8000.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.n_mel = n_mel

        # Register window as buffer
        self.register_buffer('window', torch.hann_window(win_length))

        # Create mel filterbank
        mel_basis = self._create_mel_filterbank(sample_rate, n_fft, n_mel, fmin, fmax)
        self.register_buffer('mel_basis', mel_basis)

    def _create_mel_filterbank(self, sr: int, n_fft: int, n_mels: int,
                                fmin: float, fmax: float) -> torch.Tensor:
        """Create mel filterbank matrix using librosa for proper normalization."""
        import librosa
        # Use librosa's properly normalized mel filterbank
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        return torch.from_numpy(mel_basis).float()

    def forward(self, audio: torch.Tensor, center: bool = True) -> torch.Tensor:
        """Compute mel spectrogram."""
        # Ensure audio is 2D (batch, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Compute STFT on CPU if needed (MPS doesn't support complex tensors)
        device = audio.device
        if device.type == 'mps':
            audio_cpu = audio.cpu()
            fft = torch.stft(
                audio_cpu,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window.cpu(),
                center=center,
                return_complex=True
            )
            # Compute magnitude as in original: sqrt(real^2 + imag^2)
            magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2)).to(device)
        else:
            fft = torch.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=center,
                return_complex=True
            )
            # Compute magnitude as in original: sqrt(real^2 + imag^2)
            magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))

        # Apply mel filterbank
        mel = torch.matmul(self.mel_basis, magnitude)

        # Log scale with small epsilon (natural log, matching original RMVPE)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        return mel


class RMVPE(nn.Module):
    """RMVPE pitch estimation model."""

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

        # E2E model (contains DeepUnet + CNN + FC)
        self.model = E2E(
            n_blocks=4,
            n_gru=1,
            kernel_size=(2, 2),
            en_de_layers=5,
            inter_layers=4,
            in_channels=1,
            en_out_channels=16
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Estimate pitch from audio."""
        # Compute mel spectrogram
        mel = self.mel_extractor(audio)  # (batch, 128, time)

        # Run through mel2hidden with proper padding
        pitch_salience = self.mel2hidden(mel)  # (batch, time, 360)

        return pitch_salience

    def mel2hidden(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to hidden states with proper padding.

        The model expects input aligned to 32-frame boundaries.
        """
        n_frames = mel.shape[-1]
        # Pad to nearest multiple of 32 frames using reflection padding
        pad_amount = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if pad_amount > 0:
            mel = F.pad(mel, (0, pad_amount), mode="reflect")

        # Run through E2E model
        hidden = self.model(mel)

        # Truncate back to original frame count
        return hidden[:, :n_frames]


class RMVPEDetector:
    """High-level wrapper for RMVPE pitch detection."""

    def __init__(self, model_path: Optional[str] = None, device: str = 'auto',
                 is_half: bool = True, use_onnx: bool = True):
        """Initialize RMVPE detector.

        Args:
            model_path: Path to model weights. Auto-downloads if None.
            device: Device to use ('auto', 'cuda', 'mps', or 'cpu')
            is_half: Use half precision (faster on GPU, ignored for ONNX)
            use_onnx: Use ONNX Runtime for inference (recommended, more reliable)
        """
        self.use_onnx = use_onnx
        self.is_half = is_half if not use_onnx else False

        # Set model paths
        if model_path:
            self.model_path = Path(model_path)
            self.onnx_path = self.model_path.with_suffix('.onnx')
        else:
            self.model_path = DEFAULT_MODEL_PATH
            self.onnx_path = DEFAULT_MODEL_DIR / 'rmvpe.onnx'

        # Determine device
        # Note: MPS (Apple Silicon) doesn't support complex tensors needed for STFT,
        # so we force CPU for RMVPE even on Apple Silicon
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                # Force CPU - MPS doesn't support STFT/complex tensors properly
                self.device = torch.device('cpu')
                self.is_half = False  # CPU is faster with float32
        elif device == 'mps':
            print("Warning: MPS doesn't support STFT operations, falling back to CPU")
            self.device = torch.device('cpu')
            self.is_half = False
        else:
            self.device = torch.device(device)
            if device == 'cpu':
                self.is_half = False

        # Ensure model is downloaded
        self._ensure_model_downloaded()

        # Load model
        if use_onnx:
            self.model = self._load_onnx_model()
            # Create mel extractor for preprocessing
            self.mel_extractor = MelSpectrogram()
        else:
            self.model = self._load_model()

        # Precompute cents mapping for decoding
        cents_mapping = 20 * np.arange(360) + BASE_CENTS
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # Pad for interpolation

    def _ensure_model_downloaded(self):
        """Download model weights if not present."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Download ONNX model if using ONNX and not present
        if self.use_onnx and not self.onnx_path.exists():
            onnx_url = MODEL_URL.replace('.pt', '.onnx')
            print(f"RMVPE ONNX model not found at {self.onnx_path}")
            print(f"Downloading from HuggingFace...")

            try:
                def report_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(100, downloaded * 100 / total_size)
                    print(f"\rDownloading: {percent:.1f}%", end='', flush=True)

                urllib.request.urlretrieve(onnx_url, self.onnx_path, reporthook=report_progress)
                print(f"\nONNX model downloaded to: {self.onnx_path}")

            except Exception as e:
                print(f"\nFailed to download RMVPE ONNX model: {e}")
                print("Falling back to PyTorch model...")
                self.use_onnx = False

        # Download PyTorch model if needed (for non-ONNX or as fallback)
        if not self.use_onnx and not self.model_path.exists():
            print(f"RMVPE model not found at {self.model_path}")
            print(f"Downloading from HuggingFace...")

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

    def _load_onnx_model(self):
        """Load ONNX model for inference."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX inference. Install with: pip install onnxruntime")

        print(f"Loading RMVPE ONNX model from: {self.onnx_path}")

        # Determine providers based on device
        if self.device.type == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        session = ort.InferenceSession(str(self.onnx_path), providers=providers)

        # Get input/output names
        self.onnx_input_name = session.get_inputs()[0].name
        self.onnx_output_name = session.get_outputs()[0].name

        print(f"ONNX model loaded (providers: {session.get_providers()})")
        return session

    def _load_model(self):
        """Load the pre-trained RMVPE model."""
        print(f"Loading RMVPE model from: {self.model_path}")
        print(f"Device: {self.device}, Half precision: {self.is_half}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)

        # Create model
        model = RMVPE(hop_length=160, sample_rate=RMVPE_SAMPLE_RATE)

        # Get state dict from checkpoint
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # The checkpoint uses 'cnn', 'fc', 'unet' directly, but our model
        # wraps them in 'model' (E2E). We need to remap the keys.
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('unet.') or key.startswith('cnn.') or key.startswith('fc.'):
                new_key = 'model.' + key
            else:
                new_key = key
            new_state_dict[new_key] = value

        # Try to load weights
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print("Model loaded successfully (strict)")
        except RuntimeError as e:
            print(f"Strict loading failed: {e}")
            print("Attempting flexible loading...")
            # Try non-strict loading and report missing/unexpected keys
            result = model.load_state_dict(new_state_dict, strict=False)
            if result.missing_keys:
                print(f"Missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}...")
            if result.unexpected_keys:
                print(f"Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}...")

        model = model.to(self.device)
        model.eval()

        if self.is_half and self.device.type == 'cuda':
            model = model.half()

        return model

    def _cents_to_hz(self, cents: np.ndarray) -> np.ndarray:
        """Convert cents to Hz using RMVPE's reference (10 Hz at 0 cents)."""
        return 10 * (2 ** (cents / 1200))

    def _to_local_average_cents(self, salience: np.ndarray, threshold: float = 0.03) -> np.ndarray:
        """Convert salience to cents using local weighted averaging.

        This performs sub-bin pitch refinement by taking a weighted average
        over a 9-bin window around the peak, providing better frequency resolution
        than just taking the argmax.

        Args:
            salience: (time, 360) salience values from model
            threshold: Confidence threshold for voiced detection

        Returns:
            cents: Pitch in cents for each frame (0 for unvoiced)
        """
        # Find peak indices for each frame
        peak_indices = np.argmax(salience, axis=1)
        peak_values = np.max(salience, axis=1)

        # Initialize output
        cents = np.zeros(len(peak_indices))

        # For each frame, compute weighted average around peak
        for i, (peak_idx, peak_val) in enumerate(zip(peak_indices, peak_values)):
            if peak_val < threshold:
                cents[i] = 0
                continue

            # Extract 9-bin window around peak (using padded cents_mapping)
            # cents_mapping is padded by 4 on each side
            start_idx = peak_idx  # Already offset by padding
            window = salience[i, max(0, peak_idx - 4):min(360, peak_idx + 5)]
            cents_window = self.cents_mapping[start_idx:start_idx + len(window)]

            # Weighted average
            if window.sum() > 0:
                cents[i] = np.sum(window * cents_window) / np.sum(window)
            else:
                cents[i] = 0

        return cents

    def detect(self, audio: np.ndarray, sr: int,
               threshold: float = 0.03) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect pitch from audio.

        Args:
            audio: Audio waveform (mono)
            sr: Sample rate
            threshold: Confidence threshold for voiced detection

        Returns:
            times: Time points in seconds
            pitch_hz: Pitch values in Hz (0 for unvoiced)
            confidence: Confidence values
        """
        import librosa

        # Resample to RMVPE sample rate if needed
        if sr != RMVPE_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=RMVPE_SAMPLE_RATE)
            sr = RMVPE_SAMPLE_RATE

        if self.use_onnx:
            salience = self._detect_onnx(audio)
        else:
            salience = self._detect_pytorch(audio)

        # Get confidence (max salience per frame)
        confidence = np.max(salience, axis=1)

        # Decode pitch using weighted averaging
        cents = self._to_local_average_cents(salience, threshold)

        # Convert cents to Hz
        pitch_hz = self._cents_to_hz(cents)
        # Zero out unvoiced (cents=0 maps to 10 Hz, set to 0)
        pitch_hz[cents == 0] = 0

        # Generate time axis
        hop_time = 160 / RMVPE_SAMPLE_RATE  # ~10ms
        times = np.arange(len(pitch_hz)) * hop_time

        return times, pitch_hz, confidence

    def _detect_onnx(self, audio: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        # Convert to tensor for mel extraction
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

        # Compute mel spectrogram
        mel = self.mel_extractor(audio_tensor, center=True)  # (1, 128, time)

        # Pad to 32-frame alignment
        n_frames = mel.shape[-1]
        pad_amount = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if pad_amount > 0:
            mel = F.pad(mel, (0, pad_amount), mode='reflect')

        # Run ONNX inference
        mel_np = mel.numpy()
        result = self.model.run([self.onnx_output_name], {self.onnx_input_name: mel_np})[0]

        # Truncate to original frame count
        salience = result[0, :n_frames, :]  # (time, 360)

        return salience

    def _detect_pytorch(self, audio: np.ndarray) -> np.ndarray:
        """Run PyTorch inference."""
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        if self.is_half:
            audio_tensor = audio_tensor.half()
        audio_tensor = audio_tensor.unsqueeze(0).to(self.device)  # (1, samples)

        # Run inference
        with torch.no_grad():
            salience = self.model(audio_tensor)  # (1, time, 360)

        salience = salience.squeeze(0).cpu().numpy()  # (time, 360)

        return salience


# For backwards compatibility and direct testing
def test_rmvpe():
    """Test RMVPE detector with ONNX backend."""
    import librosa

    print("Testing RMVPE detector (ONNX backend)...")

    # Create detector with ONNX
    detector = RMVPEDetector(device='cpu', use_onnx=True)

    # Generate test signal (440 Hz sine wave)
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Detect pitch
    times, pitch_hz, confidence = detector.detect(audio, sr)

    print(f"\nResults for 440 Hz sine wave:")
    print(f"  Detected {len(times)} frames")
    print(f"  Confidence range: [{np.min(confidence):.4f}, {np.max(confidence):.4f}]")
    print(f"  Mean confidence: {np.mean(confidence):.4f}")

    voiced_mask = pitch_hz > 0
    if voiced_mask.any():
        print(f"  Voiced frames: {np.sum(voiced_mask)} / {len(times)}")
        print(f"  Mean pitch (voiced): {np.mean(pitch_hz[voiced_mask]):.1f} Hz")
        print(f"  Pitch range (voiced): [{np.min(pitch_hz[voiced_mask]):.1f}, {np.max(pitch_hz[voiced_mask]):.1f}] Hz")
    else:
        print("  No voiced frames detected (expected for synthetic sine wave)")

    print(f"\n  Note: RMVPE is trained on vocal audio, synthetic tones may not be detected.")


def test_rmvpe_vocal():
    """Test RMVPE detector with actual vocal audio."""
    import librosa
    import os

    print("Testing RMVPE detector with vocal audio...")

    # Find a test audio file
    test_files = []
    if os.path.exists('data/temp'):
        for f in os.listdir('data/temp'):
            if f.endswith('.wav'):
                test_files.append(os.path.join('data/temp', f))

    if not test_files:
        print("No test audio files found in data/temp/")
        return

    # Create detector
    detector = RMVPEDetector(device='cpu', use_onnx=True)

    # Load audio (skip initial silence)
    audio_file = test_files[0]
    print(f"Loading: {audio_file}")
    audio, sr = librosa.load(audio_file, sr=16000, mono=True, offset=10.0, duration=5.0)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (0.95 / peak)

    print(f"Audio: {len(audio)} samples, range=[{audio.min():.3f}, {audio.max():.3f}]")

    # Detect pitch
    times, pitch_hz, confidence = detector.detect(audio, sr)

    print(f"\nResults:")
    print(f"  Total frames: {len(times)}")
    print(f"  Confidence range: [{np.min(confidence):.4f}, {np.max(confidence):.4f}]")
    print(f"  Mean confidence: {np.mean(confidence):.4f}")

    voiced_mask = pitch_hz > 0
    print(f"  Voiced frames: {np.sum(voiced_mask)} / {len(times)} ({100*np.mean(voiced_mask):.1f}%)")

    if voiced_mask.any():
        print(f"  Pitch range (voiced): [{np.min(pitch_hz[voiced_mask]):.1f}, {np.max(pitch_hz[voiced_mask]):.1f}] Hz")
        print(f"  Mean pitch (voiced): {np.mean(pitch_hz[voiced_mask]):.1f} Hz")
        # Convert to MIDI
        midi = 12 * np.log2(pitch_hz[voiced_mask] / 440) + 69
        print(f"  MIDI range: [{np.min(midi):.1f}, {np.max(midi):.1f}]")


if __name__ == "__main__":
    test_rmvpe()
    print("\n" + "="*60 + "\n")
    test_rmvpe_vocal()
