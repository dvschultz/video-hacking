#!/usr/bin/env python3
"""
ImageBind Audio Embedder

Extracts ImageBind embeddings from audio segments for semantic matching.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch

# Monkey-patch torchaudio.load to use soundfile backend instead of torchcodec
import torchaudio
import soundfile as sf

# Save original load function
_original_torchaudio_load = torchaudio.load

def _patched_torchaudio_load(filepath, *args, **kwargs):
    """Load audio using soundfile instead of torchcodec"""
    # Use soundfile to load audio
    waveform, sample_rate = sf.read(filepath, dtype='float32')
    # Convert to torch tensor and match torchaudio's format (channels, samples)
    waveform = torch.from_numpy(waveform)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension
    else:
        waveform = waveform.T  # Transpose to (channels, samples)
    return waveform, sample_rate

# Replace torchaudio.load with patched version
torchaudio.load = _patched_torchaudio_load

# Check if imagebind is installed
try:
    # ImageBind must be installed from GitHub
    # pip install git+https://github.com/facebookresearch/ImageBind.git
    from imagebind import data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
except ImportError:
    print("Error: ImageBind not installed")
    print("Install with: pip install git+https://github.com/facebookresearch/ImageBind.git")
    sys.exit(1)


class ImageBindAudioEmbedder:
    """Extract ImageBind embeddings from audio segments."""

    def __init__(self, device: str = 'auto'):
        """
        Initialize ImageBind model.

        Args:
            device: 'cuda', 'cpu', or 'auto'
        """
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading ImageBind model on {self.device}...")
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        print("✓ ImageBind model loaded")

    def extract_audio_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract embedding for a single audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Embedding vector as numpy array
        """
        # Load and transform audio
        audio_paths = [audio_path]
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(
                audio_paths,
                self.device
            )
        }

        # Extract embedding
        with torch.no_grad():
            embeddings = self.model(inputs)
            audio_emb = embeddings[ModalityType.AUDIO]

        # Convert to numpy and normalize
        embedding = audio_emb.cpu().numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)  # L2 normalize

        return embedding

    def extract_batch_embeddings(
        self,
        audio_paths: List[str],
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Extract embeddings for multiple audio files in batches.

        Args:
            audio_paths: List of audio file paths
            batch_size: Number of files to process at once

        Returns:
            Array of embeddings (num_files, embedding_dim)
        """
        embeddings = []

        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]

            # Process batch
            inputs = {
                ModalityType.AUDIO: data.load_and_transform_audio_data(
                    batch_paths,
                    self.device
                )
            }

            with torch.no_grad():
                batch_embeddings = self.model(inputs)
                audio_embs = batch_embeddings[ModalityType.AUDIO]

            # Convert to numpy and normalize
            batch_embs = audio_embs.cpu().numpy()
            batch_embs = batch_embs / np.linalg.norm(batch_embs, axis=1, keepdims=True)

            embeddings.append(batch_embs)

            if (i + batch_size) % 50 == 0:
                print(f"  Processed {min(i + batch_size, len(audio_paths))}/{len(audio_paths)} audio files")

        return np.concatenate(embeddings, axis=0)

    def process_audio_segments(
        self,
        segments_metadata_path: str,
        segments_dir: str,
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Process all audio segments and extract embeddings.

        Args:
            segments_metadata_path: Path to audio segments metadata JSON
            segments_dir: Directory containing audio segment files
            batch_size: Batch size for processing

        Returns:
            List of dicts with segment info and embeddings
        """
        # Load metadata
        with open(segments_metadata_path, 'r') as f:
            metadata = json.load(f)

        segments = metadata['segments']
        segments_dir = Path(segments_dir)

        print(f"Processing {len(segments)} audio segments...")

        # Build list of audio paths
        audio_paths = []
        valid_segments = []

        for segment in segments:
            audio_path = segments_dir / segment['filename']
            if audio_path.exists():
                audio_paths.append(str(audio_path))
                valid_segments.append(segment)
            else:
                print(f"Warning: Segment not found: {audio_path}")

        if not audio_paths:
            raise ValueError("No valid audio segments found")

        # Extract embeddings in batches
        embeddings = self.extract_batch_embeddings(audio_paths, batch_size)

        print(f"✓ Extracted {len(embeddings)} embeddings (dim={embeddings.shape[1]})")

        # Combine with metadata
        results = []
        for segment, embedding in zip(valid_segments, embeddings):
            result = {
                **segment,
                'embedding': embedding.tolist(),
                'embedding_dim': len(embedding)
            }
            results.append(result)

        return results

    def export_embeddings(
        self,
        embeddings_data: List[Dict],
        output_path: str,
        source_metadata: dict = None
    ):
        """
        Export embeddings to JSON.

        Args:
            embeddings_data: List of segment dicts with embeddings
            output_path: Path to output JSON file
            source_metadata: Optional source metadata to include
        """
        output = {
            'model': 'imagebind_huge',
            'modality': 'audio',
            'num_segments': len(embeddings_data),
            'embedding_dim': embeddings_data[0]['embedding_dim'] if embeddings_data else 0,
            'segments': embeddings_data
        }

        if source_metadata:
            output['source_metadata'] = source_metadata

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✓ Exported embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract ImageBind embeddings from audio segments'
    )
    parser.add_argument('--segments-metadata', required=True,
                       help='Path to audio segments metadata JSON')
    parser.add_argument('--segments-dir', required=True,
                       help='Directory containing audio segment files')
    parser.add_argument('--output', required=True,
                       help='Path to output embeddings JSON')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing (default: 8)')
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (default: auto)')

    args = parser.parse_args()

    # Initialize embedder
    embedder = ImageBindAudioEmbedder(device=args.device)

    # Process segments
    embeddings_data = embedder.process_audio_segments(
        segments_metadata_path=args.segments_metadata,
        segments_dir=args.segments_dir,
        batch_size=args.batch_size
    )

    # Load original metadata for reference
    with open(args.segments_metadata, 'r') as f:
        source_metadata = json.load(f)

    # Export
    embedder.export_embeddings(
        embeddings_data,
        args.output,
        source_metadata={
            'source_audio': source_metadata.get('source_audio'),
            'threshold': source_metadata.get('threshold'),
            'total_duration': source_metadata.get('total_duration')
        }
    )

    print("\n✓ Audio embedding extraction complete!")


if __name__ == '__main__':
    main()
