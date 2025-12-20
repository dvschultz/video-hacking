# In a Colab notebook, run this cell first to install the required libraries
# !pip install librosa soundfile matplotlib

import os
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from typing import Union, Tuple

# In Colab, we can use this to play audio
from IPython.display import Audio, display

def track_sound_change(
    audio_source: Union[str, Tuple[np.ndarray, int]],
    analysis_rate: int = 8,
    ignore_start_duration: float = 0.5,
    power: float = 0.5,
    smoothing_window_size: int = 2,
    smoothing_tolerance: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Analyzes a sound file to track changes using an adaptive smoothing filter.

    This function calculates a novelty curve, normalizes it, applies power-law
    scaling, and finally applies a context-aware smoothing filter to remove noise.

    Args:
        audio_source (Union[str, Tuple[np.ndarray, int]]):
            The source of the audio.
        analysis_rate (int): The number of times per second to perform the analysis.
        ignore_start_duration (float): Duration at the start to ignore and zero out.
        power (float): Compression exponent for scaling (e.g., 0.5 for sqrt).
        smoothing_window_size (int): The number of points to look at on either side
                                     of a value to determine if it's noise.
        smoothing_tolerance (float): If the difference between the max and min value
                                     within the smoothing window is less than this
                                     tolerance, the point is considered noise and
                                     is set to 0.

    Returns:
        A tuple containing:
        - np.ndarray: The final, cleaned change values.
        - np.ndarray: The loaded audio waveform.
        - int: The sample rate of the audio.
    """
    try:
        # 1. Load the audio
        if isinstance(audio_source, str):
            if not os.path.exists(audio_source):
                print(f"Error: File not found at '{audio_source}'")
                return np.array([]), np.array([]), 0
            y, sr = librosa.load(audio_source, sr=None)
        elif isinstance(audio_source, tuple):
            y, sr = audio_source
        else:
            raise ValueError("audio_source must be a file path or a (waveform, sr) tuple.")

        # 2. Calculate onset strength
        hop_length = int(sr / analysis_rate)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        # 3. Normalize the result, ignoring the initial spike
        frames_to_ignore = int(ignore_start_duration * analysis_rate)
        if frames_to_ignore >= len(onset_env):
            frames_to_ignore = len(onset_env) - 1 if len(onset_env) > 1 else 0

        max_value_slice = onset_env[frames_to_ignore:]
        if max_value_slice.size > 0 and np.max(max_value_slice) > 0:
            max_value = np.max(max_value_slice)
        else:
            max_value = np.max(onset_env) if onset_env.size > 0 else 1

        normalized_change = onset_env / max_value if max_value > 0 else np.zeros_like(onset_env)

        # 4. Apply power-law scaling
        scaled_change = np.power(normalized_change, power)

        # 5. Zero out the initial ignored section
        if frames_to_ignore > 0:
            scaled_change[:frames_to_ignore] = 0

        # 6. --- NEW: Apply adaptive smoothing filter ---
        if smoothing_window_size > 0 and len(scaled_change) > 0:
            # Create a copy to read from while we modify the original
            original_values = np.copy(scaled_change)
            for i in range(len(scaled_change)):
                # Define the window boundaries, handling edges
                start = max(0, i - smoothing_window_size)
                end = min(len(scaled_change), i + smoothing_window_size + 1)

                window = original_values[start:end]

                # If the change within the window is too small, it's noise
                if (np.max(window) - np.min(window)) < smoothing_tolerance:
                    scaled_change[i] = 0
        # --- END NEW SECTION ---

        return scaled_change, y, sr

    except Exception as e:
        print(f"An error occurred: {e}")
        return np.array([]), np.array([]), 0

# --- Demonstration for Colab ---
if __name__ == '__main__':
    # --- 1. Set the file path and parameters ---
    audio_file_path = "/content/drive/MyDrive/SMS/_audio/Paul Woolford - Untitled (Call Out Your Name) (Original Mix)_gaudiolab_other.mp3"  # <--- CHANGE THIS TO YOUR FILE'S NAME

    # --- Parameters for the analysis ---
    power_scaling = 0.6      # Compresses range. 0.5 is square root.
    window_size = 1          # Looks at 2 points before and 2 points after.
    tolerance = 0.3        # If local change is less than this, it's noise.
    fps = 24

    # --- 2. Run the analysis ---
    print(f"\nAnalyzing the audio file: {audio_file_path}")
    change_values, waveform, sample_rate = track_sound_change(
        audio_source=audio_file_path,
        analysis_rate=fps,
        ignore_start_duration=0.5,
        power=power_scaling,
        smoothing_window_size=window_size,
        smoothing_tolerance=tolerance
    )

    # --- 3. Save the results and visualize ---
    if waveform is not None and waveform.any():
        base_name = os.path.splitext(audio_file_path)[0]
        output_filename = f"{base_name}_changes.txt"

        try:
            with open(output_filename, 'w') as f:
                for value in change_values:
                    f.write(f"{value}\n")
            print(f"\nSuccessfully saved change values to '{output_filename}'")
        except Exception as e:
            print(f"\nCould not save the file. Error: {e}")

        print("\nDisplaying audio for playback:")
        display(Audio(data=waveform, rate=sample_rate))

        print(f"\nAnalysis complete. Generated {len(change_values)} change values.")

        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12, 6))
        fig.suptitle('Sound Change Analysis', fontsize=16)

        librosa.display.waveshow(waveform, sr=sample_rate, ax=ax[0], color='royalblue')
        ax[0].set_title("Audio Waveform")
        ax[0].set_ylabel("Amplitude")

        times = librosa.times_like(change_values, sr=sample_rate, hop_length=int(sample_rate/8))
        ax[1].plot(times, change_values, label='Smoothed Change', color='crimson', linewidth=2)
        ax[1].set_title(f"Smoothed Sound Change (Window={window_size*2+1}, Tolerance={tolerance})")
        ax[1].set_ylabel("Change (Scaled)")
        ax[1].set_xlabel("Time (s)")
        ax[1].legend()
        ax[1].grid(True, alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()