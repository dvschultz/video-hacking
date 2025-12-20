#!/usr/bin/env python3
"""
ImageBind Video Embedder

Extracts ImageBind embeddings from video using a sliding window approach.
"""

import sys

# Check NumPy version before importing anything else
import numpy as np
if int(np.__version__.split('.')[0]) >= 2:
    print(f"Error: NumPy {np.__version__} is too new. This requires NumPy 1.x")
    print("Please run: pip uninstall -y numpy && pip install 'numpy<2.0'")
    sys.exit(1)

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import cv2
from PIL import Image

# Check if imagebind is installed
try:
    from imagebind import data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
except ImportError:
    print("Error: ImageBind not installed")
    print("Install with: pip install git+https://github.com/facebookresearch/ImageBind.git")
    sys.exit(1)


class ImageBindVideoEmbedder:
    """Extract ImageBind embeddings from video frames using sliding window."""

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

    def extract_frames(
        self,
        video_path: str,
        fps: int = 24,
        max_frames: int = None
    ) -> Tuple[List[np.ndarray], float]:
        """
        Extract frames from video at specified FPS.

        Args:
            video_path: Path to video file
            fps: Target frames per second
            max_frames: Maximum number of frames to extract (None = all)

        Returns:
            Tuple of (frames list, video_fps)
        """
        print(f"Extracting frames from video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        print(f"Video: {duration:.2f}s @ {video_fps:.2f} FPS ({total_frames} frames)")
        print(f"Extracting at {fps} FPS...")

        # Calculate frame interval
        frame_interval = int(video_fps / fps)

        frames = []
        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract every Nth frame
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                extracted_count += 1

                if max_frames and extracted_count >= max_frames:
                    break

            frame_count += 1

        cap.release()
        print(f"Extracted {len(frames)} frames")

        return frames, video_fps

    def extract_embeddings_from_video(
        self,
        video_path: str,
        fps: int = 24,
        window_size: int = 5,
        stride: int = 6,
        batch_size: int = 4,
        temp_dir: str = "data/temp_frames",
        chunk_size: int = 500
    ) -> List[Dict]:
        """
        Extract embeddings using sliding window, processing video in chunks to save memory.

        Args:
            video_path: Path to video file
            fps: Target frames per second
            window_size: Number of frames per window (default 5 = ~0.2s at 24fps)
            stride: Number of frames to slide forward (default 6 = 0.25s at 24fps)
            batch_size: Number of windows to process at once
            temp_dir: Directory for temporary frame files
            chunk_size: Max frames to load into memory at once

        Returns:
            List of dicts with window info and embeddings
        """
        print(f"\nExtracting embeddings with sliding window:")
        print(f"  Window size: {window_size} frames")
        print(f"  Stride: {stride} frames (~{stride/24:.3f}s at 24fps)")
        print(f"  Batch size: {batch_size} windows")
        print(f"  Chunk size: {chunk_size} frames (to save memory)")

        # Create temp directory
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)

        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        cap.release()

        print(f"\nVideo: {duration:.2f}s @ {video_fps:.2f} FPS ({total_frames} frames)")
        print(f"Extracting at {fps} FPS...")

        # Calculate frame interval
        frame_interval = int(video_fps / fps)
        extracted_frames_count = total_frames // frame_interval
        num_windows = (extracted_frames_count - window_size) // stride + 1

        print(f"  Total extracted frames: {extracted_frames_count}")
        print(f"  Total windows: {num_windows}")

        embeddings_data = []

        # Process video in chunks
        chunk_start_frame = 0
        global_extracted_idx = 0

        while chunk_start_frame < total_frames:
            # Open video for this chunk
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start_frame)

            # Extract chunk of frames
            chunk_frames = []
            chunk_extracted_indices = []
            frame_count = chunk_start_frame

            while len(chunk_frames) < chunk_size and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract every Nth frame
                if frame_count % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    chunk_frames.append(frame_rgb)
                    chunk_extracted_indices.append(global_extracted_idx)
                    global_extracted_idx += 1

                frame_count += 1

            cap.release()

            if len(chunk_frames) == 0:
                break

            print(f"\nProcessing chunk: frames {chunk_start_frame}-{frame_count} ({len(chunk_frames)} extracted)")

            # Process windows in this chunk
            for i in range(len(embeddings_data), num_windows):
                window_start_idx = i * stride
                window_end_idx = window_start_idx + window_size

                # Check if this window is completely within current chunk
                chunk_start_idx = chunk_extracted_indices[0] if chunk_extracted_indices else 0
                chunk_end_idx = chunk_extracted_indices[-1] if chunk_extracted_indices else 0

                if window_start_idx < chunk_start_idx:
                    continue  # Window is before this chunk
                if window_start_idx >= chunk_end_idx:
                    break  # Window is after this chunk, process in next chunk
                if window_end_idx > chunk_end_idx:
                    break  # Window extends past chunk, get it in next iteration

                # Calculate local indices within chunk
                local_start = window_start_idx - chunk_start_idx
                local_center = local_start + window_size // 2

                if local_center >= len(chunk_frames):
                    break

                # Get center frame
                center_frame = chunk_frames[local_center]

                # Save frame temporarily
                temp_frame_path = temp_path / f"frame_{i:06d}.jpg"
                pil_image = Image.fromarray(center_frame)
                pil_image.save(temp_frame_path)

                # Process immediately to avoid memory buildup
                try:
                    inputs = {
                        ModalityType.VISION: data.load_and_transform_vision_data(
                            [str(temp_frame_path)],
                            self.device
                        )
                    }

                    with torch.no_grad():
                        batch_embeddings = self.model(inputs)
                        vision_embs = batch_embeddings[ModalityType.VISION]

                    embedding = vision_embs.cpu().numpy()[0]
                    embedding = embedding / np.linalg.norm(embedding)

                    embeddings_data.append({
                        'window_idx': i,
                        'start_frame': window_start_idx,
                        'end_frame': window_end_idx,
                        'center_frame': window_start_idx + window_size // 2,
                        'embedding': embedding.tolist()
                    })

                    if (len(embeddings_data)) % 20 == 0:
                        print(f"  Processed {len(embeddings_data)}/{num_windows} windows")

                except Exception as e:
                    print(f"Error processing window {i}: {e}")
                finally:
                    # Clean up temp file
                    try:
                        temp_frame_path.unlink()
                    except:
                        pass

            # Move to next chunk
            chunk_start_frame = frame_count

            # Clear chunk from memory
            del chunk_frames

        # Clean up temp directory
        try:
            temp_path.rmdir()
        except:
            pass

        print(f"\n✓ Extracted {len(embeddings_data)} video embeddings")
        return embeddings_data

    def process_video(
        self,
        video_path: str,
        output_path: str,
        fps: int = 24,
        window_size: int = 5,
        stride: int = 6,
        batch_size: int = 4,
        chunk_size: int = 500
    ):
        """
        Complete pipeline: extract frames and embeddings from video.
        Processes video in chunks to avoid memory issues with long videos.

        Args:
            video_path: Path to video file
            output_path: Path to output JSON file
            fps: Frame extraction rate
            window_size: Frames per embedding window
            stride: Frame stride for sliding window
            batch_size: Windows per batch (deprecated, kept for compatibility)
            chunk_size: Max frames to load into memory at once (default 500)
        """
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Extract embeddings using chunked approach
        embeddings_data = self.extract_embeddings_from_video(
            video_path=video_path,
            fps=fps,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
            chunk_size=chunk_size
        )

        # Calculate time information
        frame_duration = 1.0 / fps

        for item in embeddings_data:
            item['start_time'] = item['start_frame'] * frame_duration
            item['end_time'] = item['end_frame'] * frame_duration
            item['center_time'] = item['center_frame'] * frame_duration

        # Calculate extracted frames count
        frame_interval = int(video_fps / fps)
        extracted_frames = total_frames // frame_interval

        # Save results
        output = {
            'video_path': str(video_path),
            'video_fps': float(video_fps),
            'extraction_fps': fps,
            'total_frames': extracted_frames,
            'window_size': window_size,
            'stride': stride,
            'num_windows': len(embeddings_data),
            'embeddings': embeddings_data
        }

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ Saved video embeddings to: {output_path}")

        if len(embeddings_data) > 0:
            print(f"\nSummary:")
            print(f"  Total windows: {len(embeddings_data)}")
            print(f"  Time coverage: 0.00s - {embeddings_data[-1]['end_time']:.2f}s")
            print(f"  Embedding dimension: {len(embeddings_data[0]['embedding'])}")
        else:
            print("\nWarning: No embeddings were extracted!")
            print("Check the error messages above for details.")


def main():
    parser = argparse.ArgumentParser(
        description='Extract ImageBind embeddings from video using sliding window'
    )
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    parser.add_argument('--fps', type=int, default=24,
                       help='Frame extraction rate (default: 24)')
    parser.add_argument('--window-size', type=int, default=5,
                       help='Frames per window (default: 5)')
    parser.add_argument('--stride', type=int, default=6,
                       help='Frame stride for sliding window (default: 6)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Windows to process per batch (default: 4, deprecated)')
    parser.add_argument('--chunk-size', type=int, default=500,
                       help='Max frames to load into memory at once (default: 500)')
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (default: auto)')

    args = parser.parse_args()

    # Initialize embedder
    embedder = ImageBindVideoEmbedder(device=args.device)

    # Process video
    embedder.process_video(
        video_path=args.video,
        output_path=args.output,
        fps=args.fps,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size
    )


if __name__ == '__main__':
    main()
