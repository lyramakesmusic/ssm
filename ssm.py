import librosa
import numpy as np
import io
import matplotlib.pyplot as plt

class SSM:
    def __init__(self):
        pass

    def compute_ssm(self, audio_input, hop_length_factor, n_chroma, bins_per_octave_multiplier, hop_length_multiplier, color_map, threshold):
        
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

        return buffer.getvalue()
