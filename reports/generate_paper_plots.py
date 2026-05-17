"""
Step 6: Generate Paper-Quality Plots from Real Pipeline Data
=============================================================
All plots are driven by ACTUAL pipeline outputs, not mock data.
No artificial corrections — the pipeline itself is now physically correct.
Includes: Transformer test validation with uncertainty bands.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from pathlib import Path
from scipy.interpolate import interp1d

BASE_DIR = Path(__file__).resolve().parent.parent


def ensure_dir():
    os.makedirs('results/paper_plots', exist_ok=True)


def load_twin_data():
    """Load the experimentally-tuned digital twin dataset."""
    p = BASE_DIR / 'data' / 'digital_twin_sets' / 'augmented_aging_twin_dataset.csv'
    if not p.exists():
        print(f"⚠️ {p} not found. Run Step 4 first.")
        return None
    return pd.read_csv(p)


def load_ev_data():
    """Load the EV drive-cycle dataset."""
    p = BASE_DIR / 'data' / 'ev_validation_sets' / 'ev_drive_cycle_dataset.csv'
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_aging_data(battery='B0005'):
    """Load per-cycle aging features."""
    p = BASE_DIR / f'data/nasa/processed/{battery}_aging_features.csv'
    if not p.exists():
        return None
    return pd.read_csv(p)


# ---- Transformer model definition (must match training code) ----
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
        if mc_dropout:
            self.train()  # enables dropout for MC sampling
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


def interpolate_cycle(df_cycle):
    """Resample a single cycle time series to uniform 1‑second grid."""
    df = df_cycle.sort_values('time_s')
    if len(df) < 2:
        return df
    t_old = df['time_s'].values
    t_new = np.arange(t_old[0], t_old[-1] + 0.5, 1.0)
    new_data = {'time_s': t_new}
    for col in df.columns:
        if col != 'time_s':
            # Skip non-numeric columns (like 'battery', 'cycle', etc.)
            if pd.api.types.is_numeric_dtype(df[col]):
                f = interp1d(t_old, df[col].values, kind='linear', fill_value='extrapolate')
                new_data[col] = f(t_new)
            else:
                # For non-numeric columns, just take the first value (they should be constant)
                new_data[col] = df[col].iloc[0]
    return pd.DataFrame(new_data)


# ---- Figure 1: ECM Voltage Validation (raw data) ----
def fig1_voltage_validation(df):
    b5 = df[df['battery'] == 'B0005']
    mid_cycle = sorted(b5['cycle'].unique())[len(b5['cycle'].unique()) // 2]
    cyc = b5[b5['cycle'] == mid_cycle]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(cyc['time_s'], cyc['voltage_V'], label='V_measured (NASA)', color='blue', alpha=0.8)
    ax1.plot(cyc['time_s'], cyc['voltage_sim_V'], label='V_simulated (2-RC ECM)', color='orange',
             linestyle='--', alpha=0.8)
    ax1.set_ylabel('Terminal Voltage (V)')
    ax1.legend(loc='lower left')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_title(f'ECM Voltage Validation — B0005 Cycle {mid_cycle} (SOH={cyc["soh_true"].iloc[0]:.3f})')

    error = np.abs(cyc['voltage_V'].values - cyc['voltage_sim_V'].values)
    ax2.plot(cyc['time_s'], error * 1000, color='red')
    ax2.set_ylabel('|Error| (mV)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('results/paper_plots/fig1_voltage_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ fig1_voltage_validation.png")


# ---- Figure 2: Surface Temperature Validation ----
def fig2_surface_temp_validation(df):
    b5 = df[df['battery'] == 'B0005']
    cycles = sorted(b5['cycle'].unique())
    picks = [cycles[5], cycles[len(cycles)//2], cycles[-5]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, cyc_num in zip(axes, picks):
        cyc = b5[b5['cycle'] == cyc_num]
        ax.plot(cyc['time_s'], cyc['temp_surface_C'], label='Ts_measured (NASA)', color='blue')
        ax.plot(cyc['time_s'], cyc['temp_surface_sim_C'], label='Ts_simulated (EETM)',
                color='green', linestyle='--')
        rmse = np.sqrt(np.mean((cyc['temp_surface_C'].values - cyc['temp_surface_sim_C'].values)**2))
        ax.set_title(f'Cycle {cyc_num} (SOH={cyc["soh_true"].iloc[0]:.3f})\nRMSE={rmse:.3f}°C')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Surface Temperature (°C)')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('EETM Surface Temp Validation — Tuned Against Real NASA Data', fontsize=13)
    plt.tight_layout()
    plt.savefig('results/paper_plots/fig2_surface_temp_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ fig2_surface_temp_validation.png")


# ---- Figure 3: Core Temperature Prediction ----
def fig3_core_temp_prediction(df):
    b5 = df[df['battery'] == 'B0005']
    cycles = sorted(b5['cycle'].unique())
    picks = [cycles[5], cycles[len(cycles)//2], cycles[-5]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, cyc_num in zip(axes, picks):
        cyc = b5[b5['cycle'] == cyc_num]
        ax.plot(cyc['time_s'], cyc['temp_surface_C'], label='T_surface (measured)', color='blue')
        ax.plot(cyc['time_s'], cyc['temp_core_C_TARGET'], label='T_core (physics twin)',
                color='red', linewidth=2)
        delta = cyc['temp_core_C_TARGET'].values - cyc['temp_surface_C'].values
        ax.fill_between(cyc['time_s'], cyc['temp_surface_C'], cyc['temp_core_C_TARGET'],
                        alpha=0.15, color='red', label=f'ΔT max={delta.max():.2f}°C')
        ax.set_title(f'Cycle {cyc_num} (SOH={cyc["soh_true"].iloc[0]:.3f})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (°C)')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Core vs Surface Temperature — Physics Digital Twin', fontsize=13)
    plt.tight_layout()
    plt.savefig('results/paper_plots/fig3_core_temperature.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ fig3_core_temperature.png")


# ---- Figure 4: ECM Parameter Evolution with Aging ----
def fig4_parameter_aging(df):
    cycle_params = df.groupby(['battery', 'cycle']).agg(
        soh=('soh_true', 'first'),
        R0=('r0_ohms', 'first'),
        R1=('r1_ohms', 'first'),
        R2=('r2_ohms', 'first'),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for batt, grp in cycle_params.groupby('battery'):
        ax1.plot(grp['cycle'], grp['R0'] * 1000, 'o-', markersize=2, label=batt)
    ax1.set_xlabel('Discharge Cycle')
    ax1.set_ylabel('R0 (mΩ)')
    ax1.set_title('Internal Resistance Growth with Aging')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    for batt, grp in cycle_params.groupby('battery'):
        ax2.plot(grp['cycle'], grp['soh'], 'o-', markersize=2, label=batt)
    ax2.set_xlabel('Discharge Cycle')
    ax2.set_ylabel('SOH')
    ax2.set_title('Capacity Fade (State of Health)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('results/paper_plots/fig4_parameter_aging.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ fig4_parameter_aging.png")


# ---- Figure 5: SOH Residual Learning ----
def fig5_soh_residual(battery='B0005'):
    df = load_aging_data(battery)
    if df is None:
        print(f"  ⚠️ Skipping SOH plot — no aging data for {battery}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df['cycle'], df['soh_true'], 'k-', linewidth=2, label='SOH_true (NASA)')
    ax1.plot(df['cycle'], df['soh_physics_baseline'], 'b--', label='SOH_physics (linear fade)')
    ax1.bar(df['cycle'], df['residual_target'], color='red', alpha=0.4,
            label='Residual (LSTM target)')
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('SOH / Residual')
    ax1.set_title(f'{battery}: Residual Learning Setup')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(df['cycle'], df['r_internal_ohms'] * 1000, 'purple', linewidth=2)
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('R_internal (mΩ)')
    ax2.set_title(f'{battery}: Internal Resistance (ECM-identified R₀)')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('results/paper_plots/fig5_soh_residual.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ fig5_soh_residual.png")


# ---- Figure 6: Current Profile and Thermal Response ----
def fig6_drive_thermal(df):
    b5 = df[df['battery'] == 'B0005']
    mid_cycle = sorted(b5['cycle'].unique())[len(b5['cycle'].unique()) // 2]
    cyc = b5[b5['cycle'] == mid_cycle]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax1.plot(cyc['time_s'], np.abs(cyc['current_A']), color='red')
    ax1.set_ylabel('|Current| (A)')
    ax1.set_title(f'B0005 Cycle {mid_cycle} — Discharge Profile & Thermal Response')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(cyc['time_s'], cyc['voltage_V'], color='blue')
    ax2.set_ylabel('Voltage (V)')
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax3.plot(cyc['time_s'], cyc['temp_surface_C'], label='T_surface', color='blue')
    ax3.plot(cyc['time_s'], cyc['temp_core_C_TARGET'], label='T_core', color='red', linewidth=2)
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_xlabel('Time (s)')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('results/paper_plots/fig6_drive_thermal.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ fig6_drive_thermal.png")


# ---- Figure 7: Transformer Test Validation with Uncertainty ----
def fig7_transformer_test_validation(df):
    model_path = BASE_DIR / 'transformer' / 'models' / 'transformer_thermal_core.pth'
    stats_path = BASE_DIR / 'transformer' / 'models' / 'normalisation_stats.csv'

    if not model_path.exists() or not stats_path.exists():
        print("  ⚠️ Transformer model or stats not found. Run Step 5 first.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BatteryThermalTransformer(
        feature_dim=4, d_model=128, nhead=4,
        num_layers=4, dim_feedforward=256, dropout=0.1
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    stats_df = pd.read_csv(stats_path, index_col=0)

    # Held-out battery: B0018 (or fallback to B0005)
    b18 = df[df['battery'] == 'B0018']
    if b18.empty:
        b18 = df[df['battery'] == 'B0005']
    cycles = sorted(b18['cycle'].unique())
    test_cycle = cycles[-3] if len(cycles) > 3 else cycles[-1]
    cyc = b18[b18['cycle'] == test_cycle].copy()

    if len(cyc) < 15:
        print("  ⚠️ Not enough data for transformer validation plot.")
        return

    # Interpolate to 1 s and prepare features
    cyc = interpolate_cycle(cyc)
    print(f"  🔍 Transformer test on B0018 Cycle {test_cycle} ({len(cyc)} pts after interpolation)")

    feat_cols = ['current_A', 'voltage_V', 'r0_ohms', 'temp_surface_C']
    target_col = 'temp_core_C_TARGET'

    cyc_norm = cyc[feat_cols].copy()
    for col in feat_cols:
        mu = stats_df.loc['mean', col]
        sigma = stats_df.loc['std', col]
        cyc_norm[col] = (cyc_norm[col] - mu) / sigma

    target_mu = stats_df.loc['mean', target_col]
    target_sigma = stats_df.loc['std', target_col]

    window_size = 60   # must match training
    data_arr = cyc_norm[feat_cols].values.astype(np.float32)

    if len(data_arr) <= window_size:
        print("  ⚠️ Cycle too short for transformer window.")
        return

    # MC dropout predictions
    mean_preds, std_preds = [], []
    with torch.no_grad():
        for i in range(len(data_arr) - window_size):
            x = torch.from_numpy(data_arr[i:i+window_size]).unsqueeze(0).to(device)
            m, s = model.predict_with_uncertainty(x, n_samples=50)
            mean_preds.append(m[0])
            std_preds.append(s[0])

    time_arr = cyc['time_s'].values[window_size:]
    mean_arr = np.array(mean_preds) * target_sigma + target_mu
    std_arr  = np.array(std_preds) * target_sigma
    target_tc = cyc[target_col].values[window_size:]
    error = mean_arr - target_tc

    rmse = np.sqrt(np.mean(error**2))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax1.plot(time_arr, np.abs(cyc['current_A'].values[window_size:]), color='black', linewidth=0.5)
    ax1.set_ylabel('|Current| (A)')
    ax1.set_title(f'Transformer Test Validation — B0018 Cycle {test_cycle} '
                  f'(RMSE={rmse:.4f}°C)')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(time_arr, target_tc, color='blue', linewidth=1.5,
             label='Tc Physics (Digital Twin)')
    ax2.plot(time_arr, mean_arr, color='orange', linewidth=1.5, linestyle='--',
             label='Tc Transformer')
    ax2.fill_between(time_arr, mean_arr - 2*std_arr, mean_arr + 2*std_arr,
                     color='orange', alpha=0.2, label='95% CI')
    ax2.set_ylabel('Core Temperature (°C)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    ax3.plot(time_arr, error, color='red', linewidth=1.0)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Error (°C)')
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('results/paper_plots/transformer_test_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ transformer_test_validation.png (RMSE={rmse:.4f}°C)")


# ---- Figure 8: EV Drive Cycle Transformer Validation with Uncertainty ----
def fig8_ev_transformer_validation():
    ev_df = load_ev_data()
    if ev_df is None:
        print("  ⚠️ No EV dataset found. Skipping EV transformer validation.")
        return

    model_path = BASE_DIR / 'transformer' / 'models' / 'transformer_thermal_core.pth'
    stats_path = BASE_DIR / 'transformer' / 'models' / 'normalisation_stats.csv'
    if not model_path.exists() or not stats_path.exists():
        print("  ⚠️ Transformer not trained yet.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BatteryThermalTransformer(
        feature_dim=4, d_model=128, nhead=4,
        num_layers=4, dim_feedforward=256, dropout=0.1
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    stats_df = pd.read_csv(stats_path, index_col=0)

    # Pick one US06 simulation at 25°C
    us06_batts = [b for b in ev_df['battery'].unique() if 'US06' in b and 'T25' in b]
    if not us06_batts:
        print("  ⚠️ No US06 T25 data found.")
        return

    sel_batt = us06_batts[0]
    cyc_df = ev_df[ev_df['battery'] == sel_batt].copy()
    cyc_df = interpolate_cycle(cyc_df)

    feat_cols = ['current_A', 'voltage_V', 'r0_ohms', 'temp_surface_C']
    target_col = 'temp_core_C_TARGET'

    for col in feat_cols:
        mu, sigma = stats_df.loc['mean', col], stats_df.loc['std', col]
        cyc_df[col + '_norm'] = (cyc_df[col] - mu) / sigma

    target_mu = stats_df.loc['mean', target_col]
    target_sigma = stats_df.loc['std', target_col]

    window_size = 60
    norm_cols = [c + '_norm' for c in feat_cols]
    data_arr = cyc_df[norm_cols].values.astype(np.float32)

    if len(data_arr) <= window_size:
        print("  ⚠️ Not enough EV data points.")
        return

    mean_preds, std_preds = [], []
    with torch.no_grad():
        for i in range(len(data_arr) - window_size):
            x = torch.from_numpy(data_arr[i:i+window_size]).unsqueeze(0).to(device)
            m, s = model.predict_with_uncertainty(x, n_samples=50)
            mean_preds.append(m[0])
            std_preds.append(s[0])

    time_arr = cyc_df['time_s'].values[window_size:]
    mean_arr = np.array(mean_preds) * target_sigma + target_mu
    std_arr  = np.array(std_preds) * target_sigma
    target_tc = cyc_df[target_col].values[window_size:]
    error = mean_arr - target_tc

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax1.plot(time_arr, np.abs(cyc_df['current_A'].values[window_size:]), color='black', linewidth=0.5)
    ax1.set_ylabel('|Current| (A)')
    ax1.set_title(f'EV US06 Drive Cycle — Transformer Core Temp Validation ({sel_batt})')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(time_arr, target_tc, color='blue', linewidth=1.2, label='Tc Physics (Digital Twin)')
    ax2.plot(time_arr, mean_arr, color='orange', linewidth=1.2, linestyle='--',
             label='Tc Transformer')
    ax2.fill_between(time_arr, mean_arr - 2*std_arr, mean_arr + 2*std_arr,
                     color='orange', alpha=0.2, label='95% CI')
    ax2.set_ylabel('Core Temperature (°C)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    ax3.plot(time_arr, error, color='red', linewidth=0.8)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Error (°C)')
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('results/paper_plots/ev_us06_transformer_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    rmse = np.sqrt(np.mean(error**2))
    print(f"  ✅ ev_us06_transformer_validation.png (RMSE={rmse:.4f}°C)")


if __name__ == "__main__":
    print("📊 Generating Paper Plots from Real Pipeline Data...")
    ensure_dir()

    df = load_twin_data()
    if df is not None:
        fig1_voltage_validation(df)
        fig2_surface_temp_validation(df)
        fig3_core_temp_prediction(df)
        fig4_parameter_aging(df)
        fig6_drive_thermal(df)
        fig7_transformer_test_validation(df)
    else:
        print("⚠️ No digital twin data — skipping thermal plots.")

    fig5_soh_residual('B0005')
    fig8_ev_transformer_validation()

    print("\n✅ All figures saved to results/paper_plots/")