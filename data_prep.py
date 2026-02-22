import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
from torch.utils.data import TensorDataset, DataLoader
def load_and_prep_nasa_data(file_path, sequence_length=50, batch_size=1024):
    print(f"1. Loading raw CMAPSS data from {file_path}...")
    columns = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)
    print("2. Applying Piecewise RUL Labeling...")
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.rename(columns={'time_in_cycles': 'max_cycle'}, inplace=True)
    df = df.merge(max_cycles, on='unit_number', how='left')
    df['RUL'] = df['max_cycle'] - df['time_in_cycles']
    df['RUL'] = df['RUL'].clip(upper=125)
    df['RUL'] = df['RUL'] / 125.0
    df.drop('max_cycle', axis=1, inplace=True)
    print("3. Performing Regime-Aware Normalization (K-Means)...")
    op_settings = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    kmeans = KMeans(n_clusters=6, n_init=10, random_state=42).fit(df[op_settings])
    df['regime'] = kmeans.labels_
    active_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
                      'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
                      'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
    df[active_sensors] = df[active_sensors].astype(np.float64)
    for r in range(6):
        mask = df['regime'] == r
        if mask.any():
            scaler = StandardScaler()
            df.loc[mask, active_sensors] = scaler.fit_transform(df.loc[mask, active_sensors])
    print(f"4. Generating sliding windows [Sequence Length: {sequence_length}]...")
    sequence_data = []
    sequence_labels = []
    for unit_id in df['unit_number'].unique():
        unit_data = df[df['unit_number'] == unit_id]
        sensor_matrix = unit_data[active_sensors].values
        rul_array = df.loc[df['unit_number'] == unit_id, 'RUL'].values
        for i in range(len(unit_data) - sequence_length + 1):
            sequence_data.append(sensor_matrix[i: i + sequence_length])
            sequence_labels.append(rul_array[i + sequence_length - 1])
    X = np.array(sequence_data)
    y = np.array(sequence_labels)
    X = np.transpose(X, (0, 2, 1))
    print(f"Final Data Shape: {X.shape} | Labels Shape: {y.shape}")
    print("5. Converting to PyTorch Tensors and DataLoader...")
    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(tensor_X, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader
