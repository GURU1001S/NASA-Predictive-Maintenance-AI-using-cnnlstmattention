import torch
import torch.nn as nn
import bitsandbytes as bnb
from tqdm import tqdm  # The progress bar
from data_prep import load_and_prep_nasa_data
# ==========================================
# 1. THE HYBRID ARCHITECTURE (CNN-LSTM)
# ==========================================
class SensorAttentionBrain(nn.Module):
    def __init__(self, num_sensors=14, sequence_length=50):
        super(SensorAttentionBrain, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(num_sensors, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        features = self.conv_block(x).permute(0, 2, 1)  
        lstm_out, _ = self.lstm(features) 
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.fc(attn_out[:, -1, :])
# ==========================================
# 2. HARDWARE & SETUP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SensorAttentionBrain(num_sensors=14, sequence_length=50).to(device)
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
criterion = nn.MSELoss()
scaler = torch.amp.GradScaler('cuda')
# ==========================================
# 3. FD002 DATA LOADING
# ==========================================
dataloader = load_and_prep_nasa_data('train_FD004.txt', sequence_length=50, batch_size=1024)
# ==========================================
# 4. TRAINING LOOP WITH PROGRESS BAR
# ==========================================
epochs = 50 
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
    for batch_X, batch_y in progress_bar:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            preds = model(batch_X)
            loss = criterion(preds.view(-1), batch_y.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    avg_loss = epoch_loss / len(dataloader)
    scheduler.step(avg_loss)
    print(f"--- Epoch Summary: Avg Loss: {avg_loss:.4f} ---\n")
torch.save(model.state_dict(), "nasa_attention_fd004_weights.pth")
print("FD004 Weights Saved!")
