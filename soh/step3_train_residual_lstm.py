"""
Step 3: Residual SOH LSTM — Physics-Augmented Training
=======================================================
Trains a bidirectional LSTM to learn the nonlinear deviation from the
quadratic SOH physics baseline (fitted in Step 2).

Key fixes vs previous version:
  1. Leave-one-battery-out validation: B0018 is always held out.
     Previous code did a random 80/20 split across all batteries, which
     created data leakage (val sequences from the same battery, same aging
     stage as train) and inflated validation metrics artificially.
  2. Global cycle normalisation: divide by GLOBAL_MAX_CYCLES (200) instead
     of per-battery max. Ensures cycle 100 maps to the same "time" across
     all batteries regardless of their individual run length.
  3. Two-layer bidirectional LSTM with dropout (0.3 between layers) gives
     substantially more capacity for capturing the nonlinear SOH knee.
  4. Dropout on final linear head (0.2) adds regularisation.
  5. Early stopping (patience=20 epochs) on validation loss prevents overfit.
  6. Larger batch size (32) for smoother gradients.
  7. AdamW + CosineAnnealingLR for better convergence.
  8. ECM R0, C1, C2 loaded from Step 4 output (all 5 params saved now).
  9. ICA peak features (peak1_v, peak2_v, peak_ratio) included as extra
     aging indicators if available — richer than R0 alone.
 10. Sequence stride = 1 but split is at battery level, so no leakage.

Architecture:
  Input: [soh_physics_baseline, r0_ohms, cycle_norm,
          ica_peak1_v*, ica_peak2_v*, ica_peak_ratio*] × seq_len=10
         (* filled with 0 if ICA not available)
  → BiLSTM(hidden=128, layers=2, dropout=0.3)
  → Dropout(0.2)
  → Linear(256 → 64) → GELU → Linear(64 → 1)
  Output: residual_soh_correction

  Final SOH = SOH_physics_baseline + LSTM_residual
"""

import os
import warnings
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

warnings.filterwarnings("ignore", category=UserWarning)

GLOBAL_MAX_CYCLES = 200   # normalise cycle across all batteries to [0, 1]
SEQ_LEN           = 10
PATIENCE          = 20    # early stopping patience (epochs)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResidualLSTM(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0)
        out_dim = hidden_size * 2   # bidirectional
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(out_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # last timestep


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def create_sequences(df: pd.DataFrame, feature_cols: list[str],
                     target_col: str = 'residual_target',
                     seq_len: int = SEQ_LEN):
    xs, ys = [], []
    arr = df[feature_cols].values.astype(np.float32)
    tgt = df[target_col].values.astype(np.float32)
    for i in range(len(arr) - seq_len):
        xs.append(arr[i:i + seq_len])
        ys.append(tgt[i + seq_len])
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_residual_lstm():
    base_dir  = Path(__file__).parent.parent
    data_dir  = base_dir / "data" / "nasa" / "processed"
    model_dir = base_dir / "soh" / "models"
    plot_dir  = base_dir / "results" / "paper_plots"
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    batteries = ["B0005", "B0006", "B0007", "B0018"]
    val_batt  = "B0018"    # held-out battery (leave-one-out)

    # Load ECM parameters from Step 4 (all 5 params)
    ecm_path = base_dir / "data" / "digital_twin_sets" / "ecm_parameters.csv"
    ecm_df   = None
    if ecm_path.exists():
        ecm_df = pd.read_csv(ecm_path)
        print("   Loaded ECM parameters (R0, R1, C1, R2, C2) from Step 4")
    else:
        print("⚠️  ECM parameters not found — run Step 4 first; using placeholder R_internal")

    # Determine which feature columns are available
    has_ica = False

    all_dfs = {}
    for batt in batteries:
        fp = data_dir / f"{batt}_aging_features.csv"
        if not fp.exists():
            print(f"⚠️  {fp.name} not found. Run Step 2 first."); continue
        df = pd.read_csv(fp)

        # Merge ECM R0 if available (overrides crude r_internal_ohms)
        if ecm_df is not None:
            batt_ecm = ecm_df[ecm_df['battery'] == batt][['cycle', 'r0_ohms']].copy()
            df = df.merge(batt_ecm, on='cycle', how='left')
            if 'r0_ohms' in df.columns:
                df['r_internal_ohms'] = df['r0_ohms'].fillna(df['r_internal_ohms'])
                df.drop(columns=['r0_ohms'], inplace=True)

        # Global cycle normalisation
        df['cycle_norm'] = df['cycle'].astype(float) / GLOBAL_MAX_CYCLES

        # ICA features (may not exist in aging_features; they live in twin dataset)
        twin_path = base_dir / "data" / "digital_twin_sets" / "augmented_aging_twin_dataset.csv"
        if twin_path.exists():
            df_twin = pd.read_csv(twin_path, usecols=['battery', 'cycle',
                'ica_peak1_v', 'ica_peak2_v', 'ica_peak_ratio'])
            df_twin = df_twin[df_twin['battery'] == batt].drop_duplicates('cycle')
            df = df.merge(df_twin[['cycle', 'ica_peak1_v', 'ica_peak2_v', 'ica_peak_ratio']],
                          on='cycle', how='left')
            for col in ['ica_peak1_v', 'ica_peak2_v', 'ica_peak_ratio']:
                df[col] = df[col].fillna(0.0)
            has_ica = True

        df = df.sort_values('cycle').reset_index(drop=True)
        all_dfs[batt] = df

    if not all_dfs:
        print("❌ No data. Aborting."); return

    # Feature columns
    base_features = ['soh_physics_baseline', 'r_internal_ohms', 'cycle_norm']
    ica_features  = ['ica_peak1_v', 'ica_peak2_v', 'ica_peak_ratio'] if has_ica else []
    feature_cols  = base_features + ica_features
    input_size    = len(feature_cols)
    print(f"   Feature vector size: {input_size}  (ICA={'yes' if has_ica else 'no'})")

    # Build train / val splits at battery level
    train_X, train_y = [], []
    val_X,   val_y   = [], []

    for batt, df in all_dfs.items():
        if len(df) < SEQ_LEN + 1:
            continue
        xs, ys = create_sequences(df, feature_cols)
        if batt == val_batt:
            val_X.append(xs); val_y.append(ys)
        else:
            train_X.append(xs); train_y.append(ys)

    if not train_X or not val_X:
        print("❌ Insufficient data for train/val split."); return

    X_train = np.concatenate(train_X)
    y_train = np.concatenate(train_y)
    X_val   = np.concatenate(val_X)
    y_val   = np.concatenate(val_y)
    print(f"   Train: {len(X_train)} sequences | Val (held-out {val_batt}): {len(X_val)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧠 Training Residual SOH BiLSTM on {device}")

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).reshape(-1, 1).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).reshape(-1, 1).to(device)

    train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xv, yv), batch_size=128, shuffle=False)

    model     = ResidualLSTM(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    epochs       = 150
    best_val     = np.inf
    best_weights = None
    no_improve   = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * bx.size(0)
        tr_loss /= len(X_train)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                vl_loss += criterion(model(bx), by).item() * bx.size(0)
        vl_loss /= len(X_val)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if vl_loss < best_val:
            best_val     = vl_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"   Early stop at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

        if epoch % 10 == 0:
            print(f"   Epoch {epoch:3d} | Train MSE {tr_loss:.6f} | "
                  f"Val MSE {vl_loss:.6f} | Best {best_val:.6f}")

    # Restore best weights
    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # Save
    torch.save(model.state_dict(), model_dir / "lstm_residual_soh.pth")
    # Save feature column list so inference knows the expected input
    pd.DataFrame({'feature': feature_cols}).to_csv(
        model_dir / "lstm_feature_cols.csv", index=False)
    print("✅ Model + feature list saved.")

    # Loss curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, color='royalblue', label='Train MSE')
    ax.plot(val_losses,   color='firebrick', label=f'Val MSE (held-out {val_batt})')
    ax.set(xlabel='Epoch', ylabel='MSE Loss',
           title='BiLSTM Residual SOH — Training & Validation Loss')
    ax.legend(); ax.grid(ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(str(plot_dir / 'lstm_training_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("📈 Loss curve saved.")

    print(f"\n✅ Best validation MSE: {best_val:.6f}  "
          f"(RMSE ≈ {np.sqrt(best_val):.4f} SOH units)")


if __name__ == "__main__":
    train_residual_lstm()
