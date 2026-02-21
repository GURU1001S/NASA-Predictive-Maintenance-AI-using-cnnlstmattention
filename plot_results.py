import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. ARCHITECTURE (Same as before)
class SensorMiniBrain1D(nn.Module):
    def __init__(self, num_sensors=14, sequence_length=50):
        super(SensorMiniBrain1D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(num_sensors, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU()
        )
        self.flattened_size = 256 * sequence_length
        self.decision_block = nn.Sequential(
            nn.Linear(self.flattened_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.decision_block(self.conv_block(x).view(x.size(0), -1))

# 2. LOAD DATA & MODEL
print("Generating Visualization...")
cols = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
train_df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=cols)
test_df = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=cols)
true_rul = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])

active_sensors = [f's{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
scaler = MinMaxScaler().fit(train_df[active_sensors])
test_df[active_sensors] = scaler.transform(test_df[active_sensors])

# Extract and Edge-Pad
test_sequences = []
for unit_id in test_df['unit'].unique():
    data = test_df[test_df['unit'] == unit_id][active_sensors].values
    if len(data) >= 50:
        seq = data[-50:]
    else:
        seq = np.vstack((np.tile(data[0], (50-len(data), 1)), data))
    test_sequences.append(seq)

X_test = torch.tensor(np.transpose(np.array(test_sequences), (0, 2, 1)), dtype=torch.float32).cuda()
model = SensorMiniBrain1D().cuda()
model.load_state_dict(torch.load("nasa_mini_brain_weights.pth", weights_only=True))
model.eval()

with torch.no_grad():
    preds = model(X_test).cpu().numpy() * 362.0 # Un-normalize

# 3. PLOTTING
plt.figure(figsize=(12, 6))
plt.plot(true_rul.values, label='Actual RUL', color='blue', linewidth=2, marker='o', markersize=4)
plt.plot(preds, label='Predicted RUL', color='red', linestyle='--', linewidth=2, marker='x', markersize=4)

plt.title('NASA Turbofan Engine RUL Prediction: 1D CNN Performance', fontsize=14)
plt.xlabel('Engine Unit Number', fontsize=12)
plt.ylabel('Remaining Useful Life (Cycles)', fontsize=12)
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Add a text box with the RMSE score
rmse = np.sqrt(np.mean((preds - true_rul.values)**2))
plt.text(5, 250, f'Overall RMSE: {rmse:.2f} cycles', fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

plt.tight_layout()
plt.savefig('prediction_results.png') # Saves the graph as an image
plt.show()