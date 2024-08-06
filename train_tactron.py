# File: train_tacotron2.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tacotron2.model import Tacotron2
from tacotron2.loss_function import Tacotron2Loss
from tacotron2.hparams import create_hparams

# Parameters
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
PREPROCESSED_DIR = "preprocessed_data"

# Create hyperparameters
hparams = create_hparams()

# Dataset Class
class VoiceDataset(Dataset):
    def __init__(self, preprocessed_dir):
        self.data = []
        for file in os.listdir(preprocessed_dir):
            if file.endswith(".npy"):
                self.data.append(os.path.join(preprocessed_dir, file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel_spec = np.load(self.data[idx])
        # Dummy text sequence for example purposes, replace with actual text encoding
        text_sequence = torch.randint(0, 30, (100,))
        text_length = torch.tensor(len(text_sequence), dtype=torch.int64)
        mel_length = torch.tensor(mel_spec.shape[1], dtype=torch.int64)
        return text_sequence, text_length, mel_spec, mel_length

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = VoiceDataset(PREPROCESSED_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model and optimizer
model = Tacotron2(hparams).to(device)  # Ensure model is on the correct device
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = Tacotron2Loss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()
        
        # Unpack the batch data and move to the correct device
        text_sequences, text_lengths, mels, mel_lengths = batch
        text_sequences, text_lengths = text_sequences.to(device), text_lengths.to(device)
        mels, mel_lengths = mels.to(device), mel_lengths.to(device)
        
        # Forward pass
        mel_output, mel_postnet_output, gate_output, alignments = model(
            (text_sequences, text_lengths, mels, None, mel_lengths)
        )
        
        loss = criterion([mel_output, mel_postnet_output, gate_output], [mels, mels])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader)}")

    # Save model checkpoint periodically
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"tacotron2_epoch_{epoch+1}.pt")
