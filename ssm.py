from PIL import Image
import librosa
import numpy as np
import io
import matplotlib.pyplot as plt

class SSM:
    def __init__(self, hop_length_factor=1.0, n_chroma=24, bins_per_octave_multiplier=2.0, hop_length_multiplier=1.0, color_map='inferno', threshold=0.5):
        self.hop_length_factor = hop_length_factor
        self.n_chroma = n_chroma
        self.bins_per_octave_multiplier = bins_per_octave_multiplier
        self.hop_length_multiplier = hop_length_multiplier
        self.color_map = color_map
        self.threshold = threshold

    def compute_ssm(self, audio_input, hop_length_factor=None, n_chroma=None, bins_per_octave_multiplier=None, hop_length_multiplier=None, color_map=None, threshold=None):
        hop_length_factor = hop_length_factor if hop_length_factor is not None else self.hop_length_factor
        n_chroma = n_chroma if n_chroma is not None else self.n_chroma
        bins_per_octave_multiplier = bins_per_octave_multiplier if bins_per_octave_multiplier is not None else self.bins_per_octave_multiplier
        hop_length_multiplier = hop_length_multiplier if hop_length_multiplier is not None else self.hop_length_multiplier
        color_map = color_map if color_map is not None else self.color_map
        threshold = threshold if threshold is not None else self.threshold
        
        # audio loading and cleaning
        if isinstance(audio_input, str):
            sr, audio = librosa.load(audio_input, sr=None)  # Load with default sr to preserve the original
        elif isinstance(audio_input, tuple) and len(audio_input) == 2:
            sr, audio = audio_input
        else:
            raise ValueError("Invalid audio input. Must be either a file path or a tuple (sr, audio).")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32) / np.max(np.abs(audio))
        audio = np.pad(audio, (0, max(0, 2048 - len(audio))), mode='constant') if len(audio) < 2048 else audio

        # the actual SSM algorithm
        hop_length = int(0.0695 * sr * hop_length_factor * hop_length_multiplier)
        bins_per_octave = int(bins_per_octave_multiplier * n_chroma)
        X = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length, n_chroma=n_chroma, bins_per_octave=bins_per_octave).T
        S = librosa.segment.recurrence_matrix(X, mode='affinity', metric='cosine', sparse=False)

        S_upscaled = np.kron(S, np.ones((20, 20)))
        S_upscaled[S_upscaled < threshold] = 0

        # plot and return
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(S_upscaled, cmap=color_map)
        plt.axis('off')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        buffer.seek(0)
        plt.close(fig)

        return Image.open(buffer)
