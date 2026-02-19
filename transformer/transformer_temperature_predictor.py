import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import json

# ============================================================================
# 1. Dataset Class
# ============================================================================
class BatteryDataset(Dataset):
    def __init__(self, csv_path, seq_len=60, features=None, target='temp_core_C'):
        self.data = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.features = features or ['current_A', 'voltage_V', 'soc', 'temp_surface_C', 'temp_ambient_C']
        self.target = target
        
        # Normalize features
        self.scaler = StandardScaler()
        self.data[self.features] = self.scaler.fit_transform(self.data[self.features])
        
        # Normalize target (optional, but good for convergence)
        self.target_scaler = StandardScaler()
        self.data[[self.target]] = self.target_scaler.fit_transform(self.data[[self.target]])
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        # Group by scenario to avoid jumping between discontinuous data
        for _, group in self.data.groupby('scenario_id'):
            group_vals = group[self.features].values
            target_vals = group[self.target].values
            
            for i in range(len(group) - seq_len):
                self.sequences.append(group_vals[i:i+seq_len])
                self.targets.append(target_vals[i+seq_len]) # Predict next step
                
        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.targets = torch.FloatTensor(np.array(self.targets)).unsqueeze(1)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# ============================================================================
# 2. Transformer Model
# ============================================================================
class BatteryTransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, output_dim=1):
        super(BatteryTransformerModel, self).__init__()
        
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model)) # Simple learnable positional encoding
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, src):
        # src shape: [batch_size, seq_len, input_dim]
        x = self.input_embedding(src) # [batch_size, seq_len, d_model]
        
        # Add positional encoding (broadcasting)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        x = self.transformer_encoder(x)
        
        # Use only the last time step for prediction
        x = x[:, -1, :] 
        
        output = self.decoder(x)
        return output

# ============================================================================
# 3. Training Loop
# ============================================================================
def train_model(train_path, val_path, epochs=10, batch_size=32, lr=0.001):
    print(f"Loading data from {train_path}...")
    dataset_train = BatteryDataset(train_path)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    dataset_val = BatteryDataset(val_path)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BatteryTransformerModel(input_dim=len(dataset_train.features)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': []}
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for seq, target in dataloader_train:
            seq, target = seq.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(dataloader_train)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, target in dataloader_val:
                seq, target = seq.to(device), target.to(device)
                output = model(seq)
                loss = criterion(output, target)
                val_loss += loss.item()
        val_loss /= len(dataloader_val)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
    return model, history, dataset_train

# ============================================================================
# 4. Evaluation and Export
# ============================================================================
def evaluate_and_export(model, test_path, dataset_train, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_test = BatteryDataset(test_path)
    
    # We need to use the scaler from training to correctly process test data
    dataset_test.scaler = dataset_train.scaler
    dataset_test.target_scaler = dataset_train.target_scaler
    # Re-process test data with training scalers
    dataset_test.data[dataset_test.features] = dataset_test.scaler.transform(pd.read_csv(test_path)[dataset_test.features])
    dataset_test.data[[dataset_test.target]] = dataset_test.target_scaler.transform(pd.read_csv(test_path)[[dataset_test.target]])
    
    # Re-create sequences (copy-paste logic from __init__ because I'm lazy to refactor right now)
    sequences = []
    targets = []
    data_backup = pd.read_csv(test_path) # Raw data for final export
    indices = []

    for _, group in dataset_test.data.groupby('scenario_id'):
        group_vals = group[dataset_test.features].values
        target_vals = group[dataset_test.target].values
        start_idx = group.index[0]
        
        for i in range(len(group) - dataset_test.seq_len):
            sequences.append(group_vals[i:i+dataset_test.seq_len])
            targets.append(target_vals[i+dataset_test.seq_len])
            indices.append(start_idx + i + dataset_test.seq_len)
            
    sequences = torch.FloatTensor(np.array(sequences)).to(device)
    targets = torch.FloatTensor(np.array(targets)).to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(sequences)
        
    # Inverse transform
    predictions_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    predictions_real = dataset_test.target_scaler.inverse_transform(predictions_np)
    targets_real = dataset_test.target_scaler.inverse_transform(targets_np)
    
    # Export results
    results_df = data_backup.iloc[indices].copy()
    results_df['predicted_temp_core_C'] = predictions_real
    results_df['actual_temp_core_C'] = targets_real
    results_df['error_abs'] = np.abs(predictions_real - targets_real)
    
    output_path = Path(output_dir) / "model_predictions.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # Metrics
    mae = np.mean(np.abs(predictions_real - targets_real))
    rmse = np.sqrt(np.mean((predictions_real - targets_real)**2))
    print(f"Test MAE: {mae:.4f} °C")
    print(f"Test RMSE: {rmse:.4f} °C")
    
    # NEW: Generate Paper-Style Plots
    plot_paper_replication_figures(results_df, output_dir)
    
    return results_df

def plot_paper_replication_figures(df, output_dir):
    """
    Generate plots replicating the style of the reference paper (Figs 6, 7, 8, 9).
    Focus: Core Temperature Estimation.
    """
    print("Generating paper-style plots...")
    output_dir = Path(output_dir)
    
    # Set style (optional if seaborn not installed, stick to matplotlib)
    plt.style.use('default') 
    
    # 1. Fig 9-like: Actual vs Estimated Tc over time
    # We pick one scenario to visualize clearly (e.g., the first one in the test set)
    scenario_ids = df['scenario_id'].unique()
    
    for i in range(min(3, len(scenario_ids))): # Plot first 3 scenarios
        sid = scenario_ids[i]
        subset = df[df['scenario_id'] == sid]
        time = subset['time_s'].values
        
        plt.figure(figsize=(10, 6))
        plt.plot(time, subset['actual_temp_core_C'], 'b-', label='Actual Tc', linewidth=1.5)
        plt.plot(time, subset['predicted_temp_core_C'], 'orange', label='Estimated Tc', linewidth=1.5, alpha=0.9)
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Core Temperature (K/°C)', fontsize=12) # User's graph has K, we have C
        plt.title(f'Actual vs Estimated Core Temp - Scenario {sid}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f"fig9_replication_scenario_{sid}.png", dpi=300)
        plt.close()

    # 2. Fig 6a-like: Error Plot
    # Plotting Absolute Error for the first scenario
    subset = df[df['scenario_id'] == scenario_ids[0]]
    time = subset['time_s'].values
    error = subset['error_abs'].values
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Core Temperature (°C)', color=color, fontsize=12)
    ax1.plot(time, subset['actual_temp_core_C'], color=color, label='Actual')
    ax1.plot(time, subset['predicted_temp_core_C'], color='cyan', linestyle='--', label='Predicted')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Absolute Error (°C)', color=color, fontsize=12)  # we already handled the x-label with ax1
    ax2.plot(time, error, color=color, label='Abs Error')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    
    plt.title('Prediction Accuracy & Error Analysis', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "fig6_replication_error_analysis.png", dpi=300)
    plt.close()
    
    print("✓ Paper-style plots saved.")

# ============================================================================
# 5. Main
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="results/datasets", help="Directory containing CSV datasets")
    parser.add_argument("--output_dir", type=str, default="results/model", help="Directory to save model and results")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find dataset files
    train_file = list(data_dir.glob("*_train.csv"))[0]
    val_file = list(data_dir.glob("*_val.csv"))[0]
    test_file = list(data_dir.glob("*_test.csv"))[0]
    
    print(f"Found training file: {train_file}")
    
    # Train
    model, history, train_dataset = train_model(train_file, val_file, epochs=args.epochs)
    
    # Save model
    torch.save(model.state_dict(), output_dir / "transformer_model.pth")
    print("Model saved.")
    
    # Plot history
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.savefig(output_dir / "training_loss.png")
    
    # Evaluate
    evaluate_and_export(model, test_file, train_dataset, output_dir)
