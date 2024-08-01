import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio


audio_file = "your_voice.wav"  # Replace with your audio file path

# Load audio file
waveform, sample_rate = librosa.load(audio_file, sr=None)

# Convert to PyTorch tensor
waveform = torch.tensor(waveform)
