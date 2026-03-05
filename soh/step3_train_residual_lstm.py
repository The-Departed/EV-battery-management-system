import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

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
    Trains on ALL 4 NASA batteries for a robust model.
    Includes 80/20 train/val split and saves loss curves.
    """
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "nasa" / "processed"
    model_dir = base_dir / "soh" / "models"
    plot_dir = base_dir / "results" / "paper_plots"
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    batteries = ["B0005", "B0006", "B0007", "B0018"]
    
    # Check for Server GPU!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧠 Training SOH Residual LSTM on {device}...")
    
    # Load and concatenate all batteries
    dfs = []
    for batt in batteries:
        fp = data_dir / f"{batt}_aging_features.csv"
        if not fp.exists():
            print(f"⚠️ {fp.name} not found. Run Step 2 first.")
            continue
        df = pd.read_csv(fp)
        df['cycle'] = df['cycle'] / df['cycle'].max()
        dfs.append(df)
    
    if not dfs:
        print("❌ No data files found. Aborting.")
        return
    
    # Create sequences from each battery separately (don't cross battery boundaries)
    X_all, y_all = [], []
    for df in dfs:
        X, y = create_sequences(df, seq_length=10)
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
    
    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    print(f"   Combined dataset: {len(X)} sequences from {len(dfs)} batteries")

    # ---- 80/20 Train / Validation Split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train)} | Val: {len(X_val)}")

    train_ds = TensorDataset(
        torch.from_numpy(X_train).to(device),
        torch.from_numpy(y_train).reshape(-1, 1).to(device),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).to(device),
        torch.from_numpy(y_val).reshape(-1, 1).to(device),
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    model = ResidualLSTM(input_size=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    epochs = 100
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # ---- Training phase ----
        model.train()
        epoch_train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * inputs.size(0)
        epoch_train_loss /= len(train_ds)

        # ---- Validation phase ----
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item() * inputs.size(0)
        epoch_val_loss /= len(val_ds)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train MSE: {epoch_train_loss:.6f}, Val MSE: {epoch_val_loss:.6f}")

    # Save the trained brain
    save_path = model_dir / "lstm_residual_soh.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training Complete! Model saved to: {save_path.name}")
    print("✨ SOH_FINAL = SOH_PHYSICS + LSTM_RESIDUAL ✨")

    # ---- Plot Training & Validation Loss Curves ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, epochs + 1), train_losses, color='blue', linewidth=1.5, label='Training Loss')
    ax.plot(range(1, epochs + 1), val_losses, color='orange', linewidth=1.5, label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('LSTM Residual SOH — Training & Validation Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(str(plot_dir / 'lstm_training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Loss curve saved to results/paper_plots/lstm_training_loss.png")

if __name__ == "__main__":
    train_residual_lstm()
