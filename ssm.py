import librosa
import numpy as np
import io
import matplotlib.pyplot as plt

class SSM:
    def __init__(self):
        pass

    def compute_ssm(self, audio_data, hop_length_factor, n_chroma, bins_per_octave_multiplier, hop_length_multiplier, color_map, threshold):
        sr, audio = audio_data
        if audio.ndim > 1:  # Ensure audio is mono
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32) / np.max(np.abs(audio))

        # Ensure audio is at least 2048 in length by padding with zeros if necessary
        audio = np.pad(audio, (0, max(0, 2048 - len(audio))), mode='constant') if len(audio) < 2048 else audio

        # Compute HPCP (Harmonic Pitch Class Profile) features from an audio signal
        hop_length = int(0.0695 * sr * hop_length_factor * hop_length_multiplier)
        bins_per_octave = int(bins_per_octave_multiplier * n_chroma)
        X = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length, n_chroma=n_chroma, bins_per_octave=bins_per_octave).T

        # Compute the segment similarity matrix (SSM)
        S = librosa.segment.recurrence_matrix(X, mode='affinity', metric='cosine', sparse=False)
        
        # Upscale and apply threshold
        S_upscaled = np.kron(S, np.ones((20, 20)))
        S_upscaled[S_upscaled < threshold] = 0
        
        # Visualize using matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(S_upscaled, cmap=color_map)
        plt.axis('off')
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        buffer.seek(0)
        plt.close(fig)  # Close the plot to free memory
        
        # Return the image in the form of a byte array
        return buffer.getvalue()
