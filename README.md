# ssm
python class to create self-similarity matrices (SSMs) from audio

## Installation

Clone and install:

```
git clone https://github.com/lyramakesmusic/ssm.git
cd ssm
pip install -e .
```

## Usage

```py
# Basic usage:

from ssm import SSM
ssm = SSM()

image = ssm.create_img("/path/to/audio/file")
image.save('generated_ssm.png')

# Pass audio directly:

import librosa
sr, audio = librosa.load('path/to/audio/file.wav', sr=None)
image = ssm.create_img(audio)
image.save('generated_ssm.png')

# Change params....

# Directly in the function call:
image = ssm.create_img(audio, n_chroma=36, hop_length_multiplier=2.0, bins_per_octave_multiplier=5.0)
image.save('generated_ssm.png')

# In the constructor:
ssm = SSM(hop_length_factor=1.0, n_chroma=24, bins_per_octave_multiplier=2.0, hop_length_multiplier=1.0, color_map='inferno', threshold=0.5)
image = ssm.create_img(audio)
image.save('generated_ssm.png')

# As a JSON object:
params = {
    'hop_length_factor': 1.0,
    'n_chroma': 24,
    'bins_per_octave_multiplier': 2.0,
    'hop_length_multiplier': 1.0,
    'color_map': 'inferno',
    'threshold': 0.5
}
image = ssm.create_img(audio, **params)
image.save('generated_ssm.png')

# Get the SSM data directly (not as an image):

S = ssm.compute_ssm(audio, **params)
print(S, S.shape)

```

Colormaps can be found here: https://matplotlib.org/stable/users/explain/colors/colormaps.html

Credit to https://github.com/RhizoNymph/ssm-gradio 
