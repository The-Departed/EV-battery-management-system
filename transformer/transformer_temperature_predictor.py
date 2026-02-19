"""
Transformer-based Battery Core Temperature Predictor
=====================================================
GPU-Optimized: Mixed Precision (AMP), large batches, pinned memory DataLoaders.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast  # Mixed precision
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import json

# Detect device globally
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"  # Mixed precision only on GPU
print(f"[Device]: {DEVICE} | Mixed Precision: {USE_AMP}")

# ============================================================================
# 1. Dataset Class
# ============================================================================
class BatteryDataset(Dataset):
    def __init__(self, csv_path, seq_len=60, features=None, target='temp_core_C',
                 scaler=None, target_scaler=None):
        self.data = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.features = features or ['current_A', 'voltage_V', 'soc', 'temp_surface_C', 'temp_ambient_C']
        self.target = target
        
        # Fit or reuse scalers
        self.scaler = scaler or StandardScaler()
        self.target_scaler = target_scaler or StandardScaler()
        
        if scaler is None:
            self.data[self.features] = self.scaler.fit_transform(self.data[self.features])
        else:
            self.data[self.features] = self.scaler.transform(self.data[self.features])
        
        if target_scaler is None:
            self.data[[self.target]] = self.target_scaler.fit_transform(self.data[[self.target]])
        else:
            self.data[[self.target]] = self.target_scaler.transform(self.data[[self.target]])
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for _, group in self.data.groupby('scenario_id'):
            group_vals = group[self.features].values
            target_vals = group[self.target].values
            
            for i in range(len(group) - seq_len):
                self.sequences.append(group_vals[i:i+seq_len])
                self.targets.append(target_vals[i+seq_len])
                
        # Pin memory-friendly: keep as float32 numpy, convert in getitem
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (torch.from_numpy(self.sequences[idx]),
                torch.tensor(self.targets[idx]).unsqueeze(0))

# ============================================================================
# 2. Transformer Model
# ============================================================================
class BatteryTransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=256, dropout=0.1, output_dim=1):
        super().__init__()
        
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, src):
        x = self.input_embedding(src)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        return self.decoder(x[:, -1, :])

# ============================================================================
# 3. GPU-Optimized Training Loop
# ============================================================================
def train_model(train_path, val_path, epochs=10, batch_size=256, lr=0.001):
    print(f"\nLoading data...")
    dataset_train = BatteryDataset(train_path)
    dataset_val   = BatteryDataset(val_path,
                                   scaler=dataset_train.scaler,
                                   target_scaler=dataset_train.target_scaler)
    
    # GPU-optimized DataLoader: pin_memory + num_workers for fast data transfer
    num_workers = 0  # Keep 0 on Windows (avoids fork issues), increase on Linux
    pin_memory = DEVICE.type == "cuda"
    
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                  pin_memory=pin_memory, num_workers=num_workers)
    dataloader_val   = DataLoader(dataset_val, batch_size=batch_size*2, shuffle=False,
                                  pin_memory=pin_memory, num_workers=num_workers)
    
    print(f"Train: {len(dataset_train):,} sequences | Val: {len(dataset_val):,} sequences")
    print(f"Batch size: {batch_size} | Device: {DEVICE}")
    
    model = BatteryTransformerModel(input_dim=len(dataset_train.features)).to(DEVICE)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision scaler (GPU speedup ~2-3x)
    scaler_amp = GradScaler(enabled=USE_AMP)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    print(f"\nStarting training ({epochs} epochs)...\n")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for seq, target in dataloader_train:
            seq, target = seq.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            with autocast(enabled=USE_AMP):
                output = model(seq)
                loss = criterion(output, target)
            
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            train_loss += loss.item()
        
        train_loss /= len(dataloader_train)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq, target in dataloader_val:
                seq, target = seq.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
                with autocast(enabled=USE_AMP):
                    output = model(seq)
                    loss = criterion(output, target)
                val_loss += loss.item()
        val_loss /= len(dataloader_val)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_marker = " ← best"
        else:
            best_marker = ""
        
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.5f} | Val: {val_loss:.5f} | LR: {lr_now:.6f}{best_marker}")
    
    return model, history, dataset_train


# ============================================================================
# 4. Evaluation and Export
# ============================================================================
def evaluate_and_export(model, test_path, dataset_train, output_dir):
    dataset_test = BatteryDataset(test_path,
                                  scaler=dataset_train.scaler,
                                  target_scaler=dataset_train.target_scaler)
    
    dataloader_test = DataLoader(dataset_test, batch_size=512, shuffle=False,
                                 pin_memory=DEVICE.type == "cuda")
    
    data_backup = pd.read_csv(test_path)
    
    all_predictions = []
    all_targets = []
    all_indices = []
    
    model.eval()
    idx_offset = 0
    with torch.no_grad():
        for seq, target in dataloader_test:
            seq = seq.to(DEVICE, non_blocking=True)
            with autocast(enabled=USE_AMP):
                preds = model(seq)
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(target.numpy())
            all_indices.extend(range(idx_offset, idx_offset + len(seq)))
            idx_offset += len(seq)
    
    predictions_np = np.vstack(all_predictions)
    targets_np = np.vstack(all_targets)
    
    predictions_real = dataset_train.target_scaler.inverse_transform(predictions_np)
    targets_real = dataset_train.target_scaler.inverse_transform(targets_np)
    
    mae  = np.mean(np.abs(predictions_real - targets_real))
    rmse = np.sqrt(np.mean((predictions_real - targets_real) ** 2))
    print(f"\n📊 Test MAE:  {mae:.4f} °C")
    print(f"📊 Test RMSE: {rmse:.4f} °C")
    
    # Export
    output_dir = Path(output_dir)
    # Build results per row (simplified - map back by seq position)
    results_df = pd.DataFrame({
        'predicted_temp_core_C': predictions_real.flatten(),
        'actual_temp_core_C':    targets_real.flatten(),
        'error_abs': np.abs(predictions_real - targets_real).flatten()
    })
    
    # Attach scenario IDs for plotting (from test dataset)
    scenario_col = []
    time_col = []
    for _, group in dataset_test.data.groupby('scenario_id'):
        for i in range(len(group) - dataset_test.seq_len):
            scenario_col.append(group['scenario_id'].iloc[0])
            if 'time_s' in group.columns:
                time_col.append(group['time_s'].iloc[i + dataset_test.seq_len])
    
    results_df['scenario_id'] = scenario_col[:len(results_df)]
    if time_col:
        results_df['time_s'] = time_col[:len(results_df)]
    
    output_path = output_dir / "model_predictions.csv"
    results_df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to {output_path}")
    
    plot_paper_replication_figures(results_df, output_dir)
    return results_df


def plot_paper_replication_figures(df, output_dir):
    """Generate paper-style plots (Fig 6, 9 replication)."""
    print("Generating paper-style plots...")
    output_dir = Path(output_dir)
    plt.style.use('default')
    
    scenario_ids = df['scenario_id'].unique()
    
    # Fig 9-like: Actual vs Estimated Tc over time for 3 scenarios
    for i in range(min(3, len(scenario_ids))):
        sid = scenario_ids[i]
        subset = df[df['scenario_id'] == sid]
        time = subset['time_s'].values if 'time_s' in subset.columns else np.arange(len(subset))
        
        plt.figure(figsize=(10, 6))
        plt.plot(time, subset['actual_temp_core_C'], 'b-', label='Actual Tc', linewidth=1.5)
        plt.plot(time, subset['predicted_temp_core_C'], 'orange', label='Estimated Tc', linewidth=1.5, alpha=0.9)
        plt.xlabel('Drive Cycle Time (s)', fontsize=12)
        plt.ylabel('Core Temperature (°C)', fontsize=12)
        plt.title(f'Actual vs Estimated Core Temp — Scenario {sid}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f"fig9_replication_scenario_{sid}.png", dpi=300)
        plt.close()
    
    # Fig 6-like: Temperature + Absolute Error (dual axis)
    subset = df[df['scenario_id'] == scenario_ids[0]]
    time = subset['time_s'].values if 'time_s' in subset.columns else np.arange(len(subset))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(time, subset['actual_temp_core_C'], 'b-', label='Actual', linewidth=1.5)
    ax1.plot(time, subset['predicted_temp_core_C'], 'orange', linestyle='--', label='Estimated', linewidth=1.5)
    ax1.set_ylabel('Core Temperature (°C)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Core Temperature Prediction & Error', fontsize=14)
    
    ax2.plot(time, subset['error_abs'], 'r-', label='Absolute Error', linewidth=1)
    ax2.set_xlabel('Drive Cycle Time (s)', fontsize=12)
    ax2.set_ylabel('Absolute Error (°C)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig6_replication_error_analysis.png", dpi=300)
    plt.close()
    
    print("✓ Paper-style plots saved.")


# ============================================================================
# 5. Main
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU-Optimized Battery Temperature Predictor")
    parser.add_argument("--data_dir",   type=str, default="results/datasets")
    parser.add_argument("--output_dir", type=str, default="results/model")
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr",         type=float, default=0.001)
    args = parser.parse_args()
    
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find dataset files
    train_file = list(data_dir.glob("*_train.csv"))[0]
    val_file   = list(data_dir.glob("*_val.csv"))[0]
    test_file  = list(data_dir.glob("*_test.csv"))[0]
    print(f"📁 Training: {train_file}")
    
    # Train
    model, history, train_dataset = train_model(
        train_file, val_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # Save best model
    torch.save(model.state_dict(), output_dir / "transformer_model.pth")
    print("✓ Model saved.")
    
    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close()
    
    # Evaluate
    evaluate_and_export(model, test_file, train_dataset, output_dir)
