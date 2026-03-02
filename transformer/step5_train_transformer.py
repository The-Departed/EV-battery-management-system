import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

class BatteryThermalTransformer(nn.Module):
    def __init__(self, feature_dim=4, d_model=32, nhead=4, num_layers=2):
        super(BatteryThermalTransformer, self).__init__()
        self.embedding = nn.Linear(feature_dim, d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Predict Core Temperature target
        )

    def forward(self, src):
        # src: [batch_size, seq_len, features]
        x = self.embedding(src)
        x = self.transformer(x)
        
        final_timestep_out = x[:, -1, :] 
        core_temp_pred = self.regression_head(final_timestep_out)
        return core_temp_pred

def create_sliding_windows(df, window_size=60, stride=1):
    """Step 5 Data Prep: Sliding Windows to create massive training data."""
    X, y = [], []
    features = ['current_A', 'voltage_V', 'r0_ohms', 'temp_surface_C']
    target = 'temp_core_C_TARGET'
    
    data_arr = df[features].values.astype(np.float32)
    target_arr = df[target].values.astype(np.float32)
    
    for i in range(0, len(df) - window_size, stride):
        X.append(data_arr[i : i+window_size])
        y.append(target_arr[i+window_size])
        
    return np.array(X), np.array(y)

def train_thermal_transformer():
    """
    Step 5: Train the Transformer on the Augmented Aging Digital Twin dataset.
    """
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "digital_twin_sets"
    model_dir = base_dir / "transformer" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = data_dir / "augmented_aging_twin_dataset.csv"
    if not file_path.exists():
        print(f"⚠️ Dataset not found: {file_path}. Run Step 4 first.")
        return
        
    # Check for Server GPU!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧠 Preparing Sliding Window Data for Transformer on {device}...")
    df = pd.read_csv(file_path)
    
    for col in ['current_A', 'voltage_V', 'r0_ohms', 'temp_surface_C', 'temp_core_C_TARGET']:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        
    X, y = create_sliding_windows(df, window_size=60, stride=1)
    print(f"✅ Generated {len(X)} sequences of length 60 from the Digital Twin.")
    
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).view(-1, 1))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = BatteryThermalTransformer(feature_dim=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    print(f"🚀 Training Transformer on {device}...")
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.6f}")

    save_path = model_dir / "transformer_thermal_core.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training Complete! Model saved to: {save_path.name}")

if __name__ == "__main__":
    train_thermal_transformer()
