import torch
import torch.nn as nn
import bitsandbytes as bnb  # Required for 8-bit Adam

# IMPORT YOUR DATA PIPELINE HERE
from data_prep import load_and_prep_nasa_data


# ==========================================
# 1. THE ARCHITECTURE (1D CNN)
# ==========================================
class SensorMiniBrain1D(nn.Module):
    def __init__(self, num_sensors=14, sequence_length=50):
        super(SensorMiniBrain1D, self).__init__()

        # Feature Extractors: Sliding across the 50 time steps
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=num_sensors, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Flattened size: 256 channels * 50 time steps = 12,800
        self.flattened_size = 256 * sequence_length

        # Decision Layers: Funneling down to 1 continuous output (e.g., RUL)
        self.decision_block = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.conv_block(x)
        flattened = features.view(features.size(0), -1)
        prediction = self.decision_block(flattened)
        return prediction


# ==========================================
# 2. HARDWARE & OPTIMIZATION SETUP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}\n")

# Initialize the model and send it to the GPU
model = SensorMiniBrain1D(num_sensors=14, sequence_length=50).to(device)

# Initialize 8-bit Adam Optimizer (Massive VRAM savings)
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.001)

# Loss function for regression (Mean Squared Error)
criterion = nn.MSELoss()

# Initialize the FP16 Gradient Scaler
scaler = torch.amp.GradScaler('cuda')

# ==========================================
# 3. THE REAL DATA PIPELINE
# ==========================================
print("Spinning up the Data Pipeline...")
# This calls your prep script, scales the data, and builds the batches
dataloader = load_and_prep_nasa_data('train_FD001.txt', sequence_length=50, batch_size=1024)
print("Data successfully loaded into memory!\n")

# ==========================================
# 4. THE FP16 TRAINING LOOP
# ==========================================
epochs = 25  # We need more epochs now that the data is real and complex

model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    batch_count = 0

    # The dataloader loops through all the 1024-sized batches in the dataset
    for batch_X, batch_y in dataloader:
        # Send this specific batch to the RTX 3050
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        # Forward pass with Automatic Mixed Precision (FP16)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

        # Backward pass using the scaler
        scaler.scale(loss).backward()

        # Optimizer step using the scaler
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        batch_count += 1

    # Calculate the average loss across the entire epoch
    avg_loss = epoch_loss / batch_count

    print(f"Epoch [{epoch + 1}/{epochs}] | Average Loss (MSE): {avg_loss:.4f}")

print("\nSuccess! The 1D Mini Brain has finished training on the NASA data.")

# Save the trained brain!
torch.save(model.state_dict(), "nasa_mini_brain_weights.pth")
print("Model weights saved to 'nasa_mini_brain_weights.pth'")