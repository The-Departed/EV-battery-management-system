import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

class ResidualLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=1):
        super(ResidualLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take last sequence output
        return out

def create_sequences(data, seq_length=10):
    """Convert tabular data to sliding windows of seq_length."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i+seq_length)][['soh_physics_baseline', 'r_internal_ohms', 'cycle']].values
        y = data.iloc[i+seq_length]['residual_target']
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

def train_residual_lstm():
    """
    Step 3: Train the LSTM to learn the error of the physics model (Residual Learning)
    """
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "nasa" / "processed"
    model_dir = base_dir / "soh" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / "B0005_aging_features.csv"
    if not file_path.exists():
        print(f"⚠️ Processed file not found: {file_path}. Run Step 2 first.")
        return

    # Check for Server GPU!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧠 Training SOH Residual LSTM on {device}...")
    
    df = pd.read_csv(file_path)
    df['cycle'] = df['cycle'] / df['cycle'].max()
    
    X, y = create_sequences(df, seq_length=10)
    
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).view(-1, 1))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = ResidualLSTM(input_size=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    epochs = 150
    for epoch in range(epochs):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 25 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], MSE Loss: {loss.item():.6f}")

    # Save the trained brain
    save_path = model_dir / "lstm_residual_soh.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training Complete! Model saved to: {save_path.name}")
    print("✨ SOH_FINAL = SOH_PHYSICS + LSTM_RESIDUAL ✨")

if __name__ == "__main__":
    train_residual_lstm()
