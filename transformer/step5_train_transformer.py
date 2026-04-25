import os, torch, torch.nn as nn, torch.optim as optim
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

class BatteryThermalTransformer(nn.Module):
    def __init__(self, feature_dim=4, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.dropout = dropout

    def forward(self, src, mc_dropout=False):
        # If mc_dropout, keep dropout active by using train() mode temporarily
        if mc_dropout:
            self.train()   # enables dropout (no batch norm issues)
        x = self.embedding(src)
        T = x.size(1)
        x = x + self.pos_encoding[:, :T, :]
        x = self.transformer(x)
        return self.regression_head(x[:, -1, :])

    def predict_with_uncertainty(self, x, n_samples=50):
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.forward(x, mc_dropout=True).cpu().numpy())
        preds = np.array(preds).squeeze(-1)   # (n_samples, batch)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std

def interpolate_to_1s(df):
    """Interpolate each (battery, cycle) time series to uniform 1s grid."""
    interpolated = []
    for (batt, cyc), grp in df.groupby(['battery', 'cycle']):
        grp = grp.sort_values('time_s')
        if len(grp) < 2: continue
        t_old = grp['time_s'].values
        t_new = np.arange(t_old[0], t_old[-1]+1, 1.0)
        new_data = {'battery': batt, 'cycle': cyc, 'time_s': t_new}
        for col in ['current_A', 'voltage_V', 'voltage_sim_V', 'r0_ohms', 'r1_ohms', 'r2_ohms',
                     'temp_surface_C', 'temp_surface_sim_C', 'temp_core_C_TARGET']:
            if col in grp.columns:
                f = interp1d(t_old, grp[col].values, kind='linear', fill_value='extrapolate')
                new_data[col] = f(t_new)
        interpolated.append(pd.DataFrame(new_data))
    return pd.concat(interpolated, ignore_index=True)

def load_and_align_data():
    base_dir = Path(__file__).parent.parent
    twin_path = base_dir / 'data/digital_twin_sets/augmented_aging_twin_dataset.csv'
    ev_path = base_dir / 'data/ev_validation_sets/ev_drive_cycle_dataset.csv'
    frames = []
    if twin_path.exists():
        twin = pd.read_csv(twin_path)
        print(f"Loaded NASA twin: {len(twin)} rows")
        # Interpolate to 1s
        twin = interpolate_to_1s(twin)
        print(f"After interpolation: {len(twin)} rows")
        frames.append(twin)
    else:
        return None
    if ev_path.exists():
        ev = pd.read_csv(ev_path)
        print(f"Loaded EV data: {len(ev)} rows")
        # Already 1s; just ensure time is monotonic
        ev = ev.sort_values(['battery', 'cycle', 'time_s'])
        frames.append(ev)
    else:
        ev = None
    df = pd.concat(frames, ignore_index=True)
    return df

def create_sliding_windows(df, window_size=60, stride=1):
    features = ['current_A', 'voltage_V', 'r0_ohms', 'temp_surface_C']
    target = 'temp_core_C_TARGET'
    X_all, y_all, groups = [], [], []
    for (batt, cyc), grp in df.groupby(['battery', 'cycle']):
        grp = grp.sort_values('time_s')
        if len(grp) <= window_size: continue
        data = grp[features].values.astype(np.float32)
        targ = grp[target].values.astype(np.float32)
        for i in range(0, len(grp)-window_size, stride):
            X_all.append(data[i:i+window_size])
            y_all.append(targ[i+window_size])
            groups.append(batt)   # store battery ID
    return np.array(X_all), np.array(y_all), groups

def train_thermal_transformer():
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / 'transformer/models'
    plot_dir = base_dir / 'results/paper_plots'
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = load_and_align_data()
    if df is None: return

    # Normalisation
    norm_cols = ['current_A', 'voltage_V', 'r0_ohms', 'temp_surface_C', 'temp_core_C_TARGET']
    stats = {}
    for col in norm_cols:
        mu, sigma = df[col].mean(), df[col].std() + 1e-8
        stats[col] = (mu, sigma)
        df[col] = (df[col] - mu) / sigma

    window_size = 60
    X, y, groups = create_sliding_windows(df, window_size=window_size, stride=1)

    # Leave-one-battery-out: hold out B0018
    train_mask = [g != 'B0018' for g in groups]
    val_mask   = [g == 'B0018' for g in groups]
    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    print(f"Train: {len(X_train)}, Val (held-out B0018): {len(X_val)}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val).unsqueeze(1))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = BatteryThermalTransformer(
        feature_dim=4, d_model=128, nhead=4, num_layers=4,
        dim_feedforward=256, dropout=0.1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_val_loss = float('inf')
    for epoch in range(1, 101):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*bx.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item()*bx.size(0)
        val_loss /= len(val_ds)
        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_dir / 'transformer_thermal_core.pth')
        if epoch % 10 == 0:
            _, sigma_tc = stats['temp_core_C_TARGET']
            rmse = np.sqrt(val_loss)*sigma_tc
            print(f"Epoch {epoch:3d} | Train MSE {train_loss:.6f} | Val MSE {val_loss:.6f} | Val RMSE {rmse:.4f}°C")

    # Final uncertainty on validation set (MC dropout)
    model.eval()
    all_preds_mean, all_preds_std = [], []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(device)
            mean, std = model.predict_with_uncertainty(bx, n_samples=50)
            all_preds_mean.append(mean)
            all_preds_std.append(std)
    all_preds_mean = np.concatenate(all_preds_mean)
    all_preds_std  = np.concatenate(all_preds_std)
    # Save for later use
    np.savez(model_dir / 'val_uncertainty.npz', mean=all_preds_mean, std=all_preds_std)

    # Save normalisation stats
    pd.DataFrame(stats, index=['mean','std']).to_csv(model_dir / 'normalisation_stats.csv')
    print("✅ Training complete.")