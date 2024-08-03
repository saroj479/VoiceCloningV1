# File: train_waveglow.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from waveglow.model import WaveGlow
from waveglow.loss_function import WaveGlowLoss

# Parameters
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
PREPROCESSED_DIR = "preprocessed_data"

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
        return torch.tensor(mel_spec)

# Load dataset
dataset = VoiceDataset(PREPROCESSED_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model and optimizer
model = WaveGlow()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = WaveGlowLoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for mel_spec in dataloader:
        optimizer.zero_grad()
        audio_output = model.infer(mel_spec)
        
        loss = criterion(audio_output, mel_spec)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader)}")

    # Save model checkpoint periodically
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"waveglow_epoch_{epoch+1}.pt")
