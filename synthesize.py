# File: synthesize_voice.py
import torch
from tacotron2.model import Tacotron2
from waveglow.model import WaveGlow
import numpy as np
import soundfile as sf

# Load trained models
tacotron2 = Tacotron2()
tacotron2.load_state_dict(torch.load("tacotron2_epoch_100.pt"))
tacotron2.eval()

waveglow = WaveGlow()
waveglow.load_state_dict(torch.load("waveglow_epoch_100.pt"))
waveglow.eval()

# Synthesize text
text_input = "Hello, this is my synthesized voice."

# Convert text to tensor (this step requires proper text preprocessing)
text_sequence = torch.randint(0, 30, (100,))  # Replace with real text sequence encoding

# Generate mel-spectrogram
with torch.no_grad():
    mel_output, _, _, _ = tacotron2.inference(text_sequence)

# Generate audio from mel-spectrogram
with torch.no_grad():
    audio_output = waveglow.infer(mel_output)

# Save the generated audio
sf.write('generated_voice.wav', audio_output.cpu().numpy(), 22050)
