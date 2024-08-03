# File: preprocess_audio.py
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Directory containing your audio files
AUDIO_DIR = "my_voice.py"
OUTPUT_DIR = "Hello"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
sample_rate = 22050
n_mels = 80

def preprocess_audio(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=sample_rate)
    
    # Generate mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Save the spectrogram
    output_file = os.path.join(OUTPUT_DIR, os.path.basename(file_path).replace('.wav', '.npy'))
    np.save(output_file, log_mel_spectrogram)
    
    # Plot for inspection (optional)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()

def process_all_files():
    for filename in os.listdir(AUDIO_DIR):
        if filename.endswith(".wav"):
            file_path = os.path.join(AUDIO_DIR, filename)
            preprocess_audio(file_path)

if __name__ == "__main__":
    process_all_files()



