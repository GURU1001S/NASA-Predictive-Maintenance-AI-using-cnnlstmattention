import torch
import torch.nn as nn
import bitsandbytes as bnb  
from data_prep import load_and_prep_nasa_data
# ==========================================
# 1. THE ARCHITECTURE (1D CNN)
# ==========================================
class SensorMiniBrain1D(nn.Module):
    def __init__(self, num_sensors=14, sequence_length=50):
        super(SensorMiniBrain1D, self).__init__()
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
        self.flattened_size = 256 * sequence_length
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
model = SensorMiniBrain1D(num_sensors=14, sequence_length=50).to(device)
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
scaler = torch.amp.GradScaler('cuda')
# ==========================================
# 3. THE REAL DATA PIPELINE
# ==========================================
print("Spinning up the Data Pipeline...")
dataloader = load_and_prep_nasa_data('train_FD001.txt', sequence_length=50, batch_size=1024)
print("Data successfully loaded into memory!\n")
# ==========================================
# 4. THE FP16 TRAINING LOOP
# ==========================================
epochs = 25  
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    batch_count = 0
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        batch_count += 1
    avg_loss = epoch_loss / batch_count
    print(f"Epoch [{epoch + 1}/{epochs}] | Average Loss (MSE): {avg_loss:.4f}")
print("\nSuccess! The 1D Mini Brain has finished training on the NASA data.")
torch.save(model.state_dict(), "nasa_mini_brain_weights.pth")
print("Model weights saved to 'nasa_mini_brain_weights.pth'")
