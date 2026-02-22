import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# 1. MATCH THE NEW ATTENTION ARCHITECTURE
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
# 2. LOAD DATA
print("Loading FD004 Test Data...")
cols = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
train_df = pd.read_csv('train_FD004.txt', sep=r'\s+', header=None, names=cols)
test_df = pd.read_csv('test_FD004.txt', sep=r'\s+', header=None, names=cols)
true_rul = pd.read_csv('RUL_FD004.txt', sep=r'\s+', header=None, names=['RUL'])
# 3. STRICT REGIME-AWARE SCALING (FIXED)
os_cols = ['os1', 'os2', 'os3']
kmeans = KMeans(n_clusters=6, n_init=10, random_state=42).fit(train_df[os_cols])
train_df['regime'] = kmeans.labels_
test_df['regime'] = kmeans.predict(test_df[os_cols])
active_sensors = [f's{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
test_df[active_sensors] = test_df[active_sensors].astype(np.float32)
for r in range(6):
    mask_train = train_df['regime'] == r
    mask_test = test_df['regime'] == r
    if mask_test.any() and mask_train.any():
        scaler = StandardScaler().fit(train_df.loc[mask_train, active_sensors])
        test_df.loc[mask_test, active_sensors] = scaler.transform(test_df.loc[mask_test, active_sensors])
# 4. WINDOWING & INFERENCE
test_sequences = []
for unit_id in test_df['unit'].unique():
    data = test_df[test_df['unit'] == unit_id][active_sensors].values
    seq = data[-50:] if len(data) >= 50 else np.vstack((np.tile(data[0], (50-len(data), 1)), data))
    test_sequences.append(seq)
X_test = torch.tensor(np.transpose(np.array(test_sequences), (0, 2, 1)), dtype=torch.float32).cuda()
model = SensorAttentionBrain().cuda()
model.load_state_dict(torch.load("nasa_attention_fd004_weights.pth", weights_only=True))
model.eval()
with torch.no_grad():
    preds = model(X_test).cpu().numpy().flatten() * 125.0
# 5. SCIENTIFIC SCORING
actual = true_rul['RUL'].values
actual_clipped = np.clip(actual, a_min=None, a_max=125.0)
rmse_raw = np.sqrt(np.mean((preds - actual)**2))
rmse_clipped = np.sqrt(np.mean((preds - actual_clipped)**2))
print("\n" + "="*40)
print(f"Raw Absolute RMSE: {rmse_raw:.2f}")
print(f"Research (Clipped) RMSE: {rmse_clipped:.2f} <--- YOUR REAL SCORE")
print("="*40)
# ==========================================
# 6. PORTFOLIO VISUALIZATION
# ==========================================
print("Generating performance graph...")
sort_indices = np.argsort(actual_clipped)[::-1]
sorted_actual = actual_clipped[sort_indices]
sorted_preds = preds[sort_indices]
plt.figure(figsize=(14, 6))
plt.plot(sorted_actual, label='Actual RUL (Clipped at 125)', color='black', linewidth=2)
plt.scatter(range(len(sorted_preds)), sorted_preds,
            label='Attention Model Predictions', color='red', alpha=0.7, s=15, zorder=3)
plt.title(f'NASA CMAPSS FD004: Actual vs Predicted RUL (RMSE: {rmse_clipped:.2f})', fontsize=14, fontweight='bold')
plt.xlabel('Engine Units (Sorted by True Remaining Life)', fontsize=12)
plt.ylabel('Remaining Useful Life (Cycles)', fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fd004_attention_results_high_res.png', dpi=300)
plt.show()
