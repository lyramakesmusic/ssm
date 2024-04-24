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
from ssm import SSM
ssm = SSM()

# Basic usage:
image = ssm.compute_ssm("/path/to/audio/file")
with open('ssm.png', 'w') as f:
    f.write(image)

# Pass audio directly:
import librosa
sr, audio = librosa.load('path/to/audio/file.wav', sr=None)
image = ssm.compute_ssm(audio)

# Change params:
image = ssm.compute_ssm(audio, n_chroma=36, hop_length_multiplier=2.0, bins_per_octave_multiplier=5.0)

# OR
ssm = SSM(hop_length_factor=1.0, n_chroma=24, bins_per_octave_multiplier=2.0, hop_length_multiplier=1.0, color_map='inferno', threshold=0.5)
image = ssm.compute_ssm(audio)
```

Credit to https://github.com/RhizoNymph/ssm-gradio
