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

image = ssm.compute_ssm("/path/to/audio/file")
image.save('generated_ssm.png')

# Pass audio directly:

import librosa
sr, audio = librosa.load('path/to/audio/file.wav', sr=None)
image = ssm.compute_ssm(audio)
image.save('generated_ssm.png')

# Change params two different ways:

image = ssm.compute_ssm(audio, n_chroma=36, hop_length_multiplier=2.0, bins_per_octave_multiplier=5.0)
image.save('generated_ssm.png')

ssm = SSM(hop_length_factor=1.0, n_chroma=24, bins_per_octave_multiplier=2.0, hop_length_multiplier=1.0, color_map='inferno', threshold=0.5)
image = ssm.compute_ssm(audio)
image.save('generated_ssm.png')
```

Credit to https://github.com/RhizoNymph/ssm-gradio 
