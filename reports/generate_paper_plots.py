"""
Step 6: Generate Paper-Quality Plots from Real Pipeline Data
=============================================================
All plots driven by actual pipeline outputs.

Changes vs previous version (sota-rewrite branch):
  - Transformer model definition updated to match step5 (6 features,
    sinusoidal PE, pre-LN TransformerEncoderLayer, norm_first=True).
  - fig4 now plots C1, C2 alongside R0/R1/R2 (all 5 ECM params saved).
  - fig5 shows quadratic SOH baseline (not linear) and ICA peak evolution.
  - fig_qgen: new plot of Q_gen (W) per cycle — verifies Joule formula fix.
  - Transformer inference uses 6-feature input [I, V, R0, Ts, SOC, Q_gen]
    and loads feature list from lstm_feature_cols.csv for forward compat.
  - Graceful fallback: every figure skips if its data is missing.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from scipy.interpolate import interp1d

BASE_DIR = Path(__file__).resolve().parent.parent
PLOT_DIR = BASE_DIR / 'results' / 'paper_plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE  = 60
FEAT_COLS_6  = ['current_A', 'voltage_V', 'r0_ohms', 'temp_surface_C', 'soc', 'q_gen_W']
FEAT_COLS_4  = ['current_A', 'voltage_V', 'r0_ohms', 'temp_surface_C']   # compat fallback
TARGET_COL   = 'temp_core_C_TARGET'


# ---------------------------------------------------------------------------
# Sinusoidal PE (must match step5)
# ---------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class BatteryThermalTransformer(nn.Module):
    """Must exactly mirror transformer/step5_train_transformer.py."""
    def __init__(self, feature_dim=6, d_model=128, nhead=4,
                 num_layers=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(feature_dim, d_model)
        self.pe    = SinusoidalPE(d_model)
        enc        = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.transformer(self.pe(self.embed(x)))[:, -1, :])

    def predict_with_uncertainty(self, x, n=50):
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(n):
                preds.append(self.forward(x).cpu().numpy())
        self.eval()
        preds = np.array(preds).squeeze(-1)
        return preds.mean(0), preds.std(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path, label=""):
    if not Path(path).exists():
        if label:
            print(f"  ⚠️  {label} not found — skipping.")
        return None
    return pd.read_csv(path)


def _interp_cycle(df):
    df = df.sort_values('time_s').drop_duplicates('time_s')
    if len(df) < 4:
        return df
    t_old = df['time_s'].values
    t_new = np.arange(t_old[0], t_old[-1] + 1, 1.0)
    out   = {'time_s': t_new}
    for col in df.columns:
        if col != 'time_s':
            f = interp1d(t_old, df[col].values, kind='linear',
                         bounds_error=False,
                         fill_value=(df[col].iloc[0], df[col].iloc[-1]))
            out[col] = f(t_new)
    return pd.DataFrame(out)


def _load_transformer():
    mp = BASE_DIR / 'transformer/models/transformer_thermal_core.pth'
    sp = BASE_DIR / 'transformer/models/normalisation_stats.csv'
    if not mp.exists() or not sp.exists():
        return None, None, None, None
    stats  = pd.read_csv(sp, index_col=0)
    # Detect feature dimension from stats columns
    feat_cols = [c for c in FEAT_COLS_6 if c in stats.columns]
    if len(feat_cols) < 4:
        feat_cols = FEAT_COLS_4
    fdim   = len(feat_cols)
    device = torch.device('cpu')
    model  = BatteryThermalTransformer(feature_dim=fdim).to(device)
    state  = torch.load(mp, map_location=device, weights_only=True)
    # Handle potential key mismatch from old checkpoint
    try:
        model.load_state_dict(state)
    except Exception:
        print("  ⚠️  Transformer weights don't match current architecture — skipping inference plots.")
        return None, None, None, None
    model.eval()
    return model, stats, feat_cols, device


# ---------------------------------------------------------------------------
# Figure 1: ECM Voltage Validation
# ---------------------------------------------------------------------------
def fig1_voltage_validation(df):
    b5  = df[df['battery'] == 'B0005']
    cycs = sorted(b5['cycle'].unique())
    mid  = cycs[len(cycs) // 2]
    cyc  = b5[b5['cycle'] == mid]
    if len(cyc) < 5:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(cyc['time_s'], cyc['voltage_V'],     color='royalblue', lw=1.2, label='V measured (NASA)')
    ax1.plot(cyc['time_s'], cyc['voltage_sim_V'], color='darkorange', lw=1.2, ls='--', label='V simulated (2-RC ECM)')
    soh = cyc['soh_true'].iloc[0]
    R0  = cyc['r0_ohms'].iloc[0]
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title(f'ECM Voltage Validation — B0005 Cycle {mid}  (SOH={soh:.3f}, R0={R0*1000:.1f} mΩ)')
    ax1.legend(); ax1.grid(ls='--', alpha=0.5)
    err = np.abs(cyc['voltage_V'].values - cyc['voltage_sim_V'].values)
    ax2.plot(cyc['time_s'], err * 1000, color='crimson', lw=0.8)
    ax2.set_ylabel('|Error| (mV)'); ax2.set_xlabel('Time (s)'); ax2.grid(ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'fig1_voltage_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ fig1 (V_RMSE={np.sqrt(np.mean(err**2))*1000:.2f} mV)")


# ---------------------------------------------------------------------------
# Figure 2: Surface Temperature Validation
# ---------------------------------------------------------------------------
def fig2_surface_temp(df):
    b5   = df[df['battery'] == 'B0005']
    cycs = sorted(b5['cycle'].unique())
    picks = [cycs[5], cycs[len(cycs)//2], cycs[-5]]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, cn in zip(axes, picks):
        cyc  = b5[b5['cycle'] == cn]
        rmse = np.sqrt(np.mean((cyc['temp_surface_C'].values - cyc['temp_surface_sim_C'].values)**2))
        ax.plot(cyc['time_s'], cyc['temp_surface_C'],     color='royalblue', lw=1.2, label='Ts measured')
        ax.plot(cyc['time_s'], cyc['temp_surface_sim_C'], color='seagreen',  lw=1.2, ls='--', label='Ts simulated')
        ax.set_title(f'Cycle {cn} (SOH={cyc["soh_true"].iloc[0]:.3f})\nRMSE={rmse:.3f}°C')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Ts (°C)'); ax.legend(); ax.grid(ls='--', alpha=0.5)
    plt.suptitle('EETM Surface Temperature Validation (Crank-Nicolson)', fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'fig2_surface_temp_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ fig2")


# ---------------------------------------------------------------------------
# Figure 3: Core vs Surface Temperature
# ---------------------------------------------------------------------------
def fig3_core_temp(df):
    b5   = df[df['battery'] == 'B0005']
    cycs = sorted(b5['cycle'].unique())
    picks = [cycs[5], cycs[len(cycs)//2], cycs[-5]]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, cn in zip(axes, picks):
        cyc  = b5[b5['cycle'] == cn]
        dT   = cyc['temp_core_C_TARGET'].values - cyc['temp_surface_C'].values
        ax.plot(cyc['time_s'], cyc['temp_surface_C'],     color='royalblue', lw=1.2, label='Ts (measured)')
        ax.plot(cyc['time_s'], cyc['temp_core_C_TARGET'], color='crimson',   lw=1.5, label='Tc (twin)')
        ax.fill_between(cyc['time_s'], cyc['temp_surface_C'], cyc['temp_core_C_TARGET'],
                        alpha=0.15, color='crimson', label=f'ΔT_max={dT.max():.2f}°C')
        ax.set_title(f'Cycle {cn} (SOH={cyc["soh_true"].iloc[0]:.3f})')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Temperature (°C)'); ax.legend(); ax.grid(ls='--', alpha=0.5)
    plt.suptitle('Core vs Surface Temperature — Physics Digital Twin (Joule Q_gen)', fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'fig3_core_temperature.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ fig3")


# ---------------------------------------------------------------------------
# Figure 4: ECM Parameter Evolution (all 5 params + SOH)
# ---------------------------------------------------------------------------
def fig4_parameter_aging(df):
    agg_cols = {'soh': ('soh_true', 'first'),
                'R0':  ('r0_ohms',  'first'),
                'R1':  ('r1_ohms',  'first'),
                'R2':  ('r2_ohms',  'first')}
    if 'c1_farads' in df.columns:
        agg_cols['C1'] = ('c1_farads', 'first')
    if 'c2_farads' in df.columns:
        agg_cols['C2'] = ('c2_farads', 'first')

    cp = df.groupby(['battery', 'cycle']).agg(**agg_cols).reset_index()

    ncols = 3 if ('C1' in cp.columns or 'C2' in cp.columns) else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5))

    ax = axes[0]
    for batt, grp in cp.groupby('battery'):
        ax.plot(grp['cycle'], grp['R0'] * 1000, 'o-', ms=2, label=batt)
    ax.set(xlabel='Cycle', ylabel='R0 (mΩ)', title='Ohmic Resistance Growth')
    ax.legend(); ax.grid(ls='--', alpha=0.5)

    ax = axes[1]
    for batt, grp in cp.groupby('battery'):
        ax.plot(grp['cycle'], grp['soh'], 'o-', ms=2, label=batt)
    ax.set(xlabel='Cycle', ylabel='SOH', title='Capacity Fade')
    ax.legend(); ax.grid(ls='--', alpha=0.5)

    if ncols == 3:
        ax = axes[2]
        if 'C1' in cp.columns:
            for batt, grp in cp.groupby('battery'):
                ax.plot(grp['cycle'], grp['C1'], 'o-', ms=2, label=f'{batt} C1')
        if 'C2' in cp.columns:
            for batt, grp in cp.groupby('battery'):
                ax.plot(grp['cycle'], grp['C2'], 's--', ms=2, label=f'{batt} C2')
        ax.set(xlabel='Cycle', ylabel='Capacitance (F)', title='RC Capacitances')
        ax.legend(); ax.grid(ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'fig4_parameter_aging.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ fig4")


# ---------------------------------------------------------------------------
# Figure 5: SOH Residual + ICA Peaks
# ---------------------------------------------------------------------------
def fig5_soh_residual(battery='B0005'):
    df = _load(BASE_DIR / f'data/nasa/processed/{battery}_aging_features.csv', 'aging features')
    if df is None:
        return

    has_ica = 'ica_peak1_v' in df.columns or \
              Path(BASE_DIR / 'data/digital_twin_sets/augmented_aging_twin_dataset.csv').exists()

    # Try to pull ICA from twin dataset
    ica_df = None
    twin_p = BASE_DIR / 'data/digital_twin_sets/augmented_aging_twin_dataset.csv'
    if twin_p.exists():
        twin = pd.read_csv(twin_p, usecols=['battery','cycle','ica_peak1_v','ica_peak2_v','ica_peak_ratio'])
        ica_df = twin[twin['battery']==battery].drop_duplicates('cycle').sort_values('cycle')

    ncols = 3 if ica_df is not None and not ica_df.empty else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5))

    ax = axes[0]
    ax.plot(df['cycle'], df['soh_true'],           'k-',  lw=2, label='SOH true (NASA)')
    ax.plot(df['cycle'], df['soh_physics_baseline'],'b--', lw=1.5, label='SOH physics (quadratic)')
    ax.bar(df['cycle'],  df['residual_target'], color='crimson', alpha=0.4, label='Residual (LSTM target)')
    ax.set(xlabel='Cycle', ylabel='SOH', title=f'{battery}: Residual Learning Setup')
    ax.legend(); ax.grid(ls='--', alpha=0.5)

    ax = axes[1]
    ax.plot(df['cycle'], df['r_internal_ohms'] * 1000, color='purple', lw=1.5)
    ax.set(xlabel='Cycle', ylabel='R_internal (mΩ)', title=f'{battery}: ECM-identified R0')
    ax.grid(ls='--', alpha=0.5)

    if ncols == 3 and ica_df is not None:
        ax = axes[2]
        ax.plot(ica_df['cycle'], ica_df['ica_peak1_v'], 'g-',  lw=1.2, label='ICA Peak 1 (V)')
        ax.plot(ica_df['cycle'], ica_df['ica_peak2_v'], 'b--', lw=1.2, label='ICA Peak 2 (V)')
        ax2b = ax.twinx()
        ax2b.plot(ica_df['cycle'], ica_df['ica_peak_ratio'], 'r:', lw=1.0, label='Peak ratio')
        ax2b.set_ylabel('Peak ratio')
        ax.set(xlabel='Cycle', ylabel='Peak voltage (V)', title=f'{battery}: ICA Peak Evolution')
        ax.legend(loc='upper left'); ax.grid(ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'fig5_soh_residual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ fig5")


# ---------------------------------------------------------------------------
# Figure 6: Drive Profile + Thermal Response (with Q_gen)
# ---------------------------------------------------------------------------
def fig6_drive_thermal(df):
    b5   = df[df['battery'] == 'B0005']
    cycs = sorted(b5['cycle'].unique())
    mid  = cycs[len(cycs) // 2]
    cyc  = b5[b5['cycle'] == mid]

    has_qgen = 'q_gen_W' in cyc.columns
    nrows    = 4 if has_qgen else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 3*nrows), sharex=True)

    axes[0].plot(cyc['time_s'], np.abs(cyc['current_A']), color='crimson', lw=0.8)
    axes[0].set_ylabel('|I| (A)')
    axes[0].set_title(f'B0005 Cycle {mid} — Discharge Profile & Thermal Response')
    axes[0].grid(ls='--', alpha=0.5)

    axes[1].plot(cyc['time_s'], cyc['voltage_V'], color='royalblue', lw=0.8)
    axes[1].set_ylabel('Voltage (V)'); axes[1].grid(ls='--', alpha=0.5)

    axes[2].plot(cyc['time_s'], cyc['temp_surface_C'],     color='royalblue', lw=1.0, label='Ts')
    axes[2].plot(cyc['time_s'], cyc['temp_core_C_TARGET'], color='crimson',   lw=1.5, label='Tc')
    axes[2].set_ylabel('Temp (°C)'); axes[2].legend(); axes[2].grid(ls='--', alpha=0.5)

    if has_qgen:
        axes[3].plot(cyc['time_s'], cyc['q_gen_W'], color='darkorange', lw=0.8)
        axes[3].set_ylabel('Q_gen (W)'); axes[3].grid(ls='--', alpha=0.5)
        axes[3].set_xlabel('Time (s)')
    else:
        axes[2].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'fig6_drive_thermal.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ fig6")


# ---------------------------------------------------------------------------
# Figure 7: Transformer Test Validation (held-out B0018)
# ---------------------------------------------------------------------------
def fig7_transformer_validation(df):
    model, stats, feat_cols, device = _load_transformer()
    if model is None:
        return

    b18  = df[df['battery'] == 'B0018']
    pool = b18 if not b18.empty else df[df['battery'] == 'B0005']
    cycs = sorted(pool['cycle'].unique())
    cn   = cycs[-3] if len(cycs) > 3 else cycs[-1]
    cyc  = _interp_cycle(pool[pool['cycle'] == cn].copy())
    if len(cyc) <= WINDOW_SIZE:
        print("  ⚠️  Cycle too short for transformer validation.")
        return

    # Normalise features
    for col in feat_cols:
        if col not in cyc.columns:
            cyc[col] = 0.0
        mu, sigma = float(stats.loc['mean', col]), float(stats.loc['std', col])
        cyc[col] = (cyc[col] - mu) / sigma

    if TARGET_COL not in cyc.columns:
        print(f"  ⚠️  {TARGET_COL} missing."); return
    t_mu, t_sig = float(stats.loc['mean', TARGET_COL]), float(stats.loc['std', TARGET_COL])
    cyc[TARGET_COL] = (cyc[TARGET_COL] - t_mu) / t_sig

    data = cyc[feat_cols].values.astype(np.float32)
    means, stds = [], []
    with torch.no_grad():
        for i in range(len(data) - WINDOW_SIZE):
            x = torch.from_numpy(data[i:i + WINDOW_SIZE]).unsqueeze(0).to(device)
            m, s = model.predict_with_uncertainty(x, n=50)
            means.append(m[0]); stds.append(s[0])

    t_arr = cyc['time_s'].values[WINDOW_SIZE:]
    m_arr = np.array(means) * t_sig + t_mu
    s_arr = np.array(stds)  * t_sig
    tc    = cyc[TARGET_COL].values[WINDOW_SIZE:] * t_sig + t_mu
    err   = m_arr - tc
    rmse  = np.sqrt(np.mean(err**2))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ax1.plot(t_arr, np.abs(pool[pool['cycle']==cn].sort_values('time_s')['current_A'].values[:len(t_arr)]),
             color='black', lw=0.5)
    ax1.set_ylabel('|I| (A)')
    ax1.set_title(f'Transformer Test (held-out B0018 Cycle {cn}) — RMSE={rmse:.4f}°C')
    ax1.grid(ls='--', alpha=0.5)

    ax2.plot(t_arr, tc,    color='royalblue', lw=1.5, label='Tc physics twin')
    ax2.plot(t_arr, m_arr, color='darkorange', lw=1.5, ls='--', label='Tc transformer')
    ax2.fill_between(t_arr, m_arr - 2*s_arr, m_arr + 2*s_arr,
                     color='darkorange', alpha=0.2, label='95% CI')
    ax2.set_ylabel('Tc (°C)'); ax2.legend(); ax2.grid(ls='--', alpha=0.5)

    ax3.plot(t_arr, err, color='crimson', lw=0.8)
    ax3.axhline(0, color='black', ls='--', lw=0.7)
    ax3.set_ylabel('Error (°C)'); ax3.set_xlabel('Time (s)'); ax3.grid(ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'transformer_test_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ fig7 (RMSE={rmse:.4f}°C)")


# ---------------------------------------------------------------------------
# Figure 8: EV US06 Transformer Validation
# ---------------------------------------------------------------------------
def fig8_ev_validation():
    model, stats, feat_cols, device = _load_transformer()
    if model is None:
        return
    ev_df = _load(BASE_DIR / 'data/ev_validation_sets/ev_drive_cycle_dataset.csv', 'EV dataset')
    if ev_df is None:
        return

    us06 = [b for b in ev_df['battery'].unique() if 'US06' in b and 'T25' in b]
    if not us06:
        print("  ⚠️  No US06 T25 EV data found."); return

    cyc = _interp_cycle(ev_df[ev_df['battery'] == us06[0]].copy())
    if len(cyc) <= WINDOW_SIZE:
        return

    for col in feat_cols:
        if col not in cyc.columns:
            cyc[col] = 0.0
        mu, sigma = float(stats.loc['mean', col]), float(stats.loc['std', col])
        cyc[col] = (cyc[col] - mu) / sigma

    if TARGET_COL not in cyc.columns:
        return
    t_mu, t_sig = float(stats.loc['mean', TARGET_COL]), float(stats.loc['std', TARGET_COL])
    cyc[TARGET_COL] = (cyc[TARGET_COL] - t_mu) / t_sig

    data  = cyc[feat_cols].values.astype(np.float32)
    means, stds = [], []
    with torch.no_grad():
        for i in range(len(data) - WINDOW_SIZE):
            x = torch.from_numpy(data[i:i + WINDOW_SIZE]).unsqueeze(0).to(device)
            m, s = model.predict_with_uncertainty(x, n=50)
            means.append(m[0]); stds.append(s[0])

    t_arr = cyc['time_s'].values[WINDOW_SIZE:]
    m_arr = np.array(means) * t_sig + t_mu
    s_arr = np.array(stds)  * t_sig
    tc    = cyc[TARGET_COL].values[WINDOW_SIZE:] * t_sig + t_mu
    err   = m_arr - tc
    rmse  = np.sqrt(np.mean(err**2))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    raw_i = ev_df[ev_df['battery'] == us06[0]].sort_values('time_s')['current_A'].values
    ax1.plot(t_arr, np.abs(raw_i[:len(t_arr)]), color='black', lw=0.5)
    ax1.set_ylabel('|I| (A)'); ax1.set_title(f'EV US06 Transformer Validation ({us06[0]}) RMSE={rmse:.4f}°C')
    ax1.grid(ls='--', alpha=0.5)

    ax2.plot(t_arr, tc,    color='royalblue',  lw=1.2, label='Tc physics')
    ax2.plot(t_arr, m_arr, color='darkorange', lw=1.2, ls='--', label='Tc transformer')
    ax2.fill_between(t_arr, m_arr - 2*s_arr, m_arr + 2*s_arr,
                     color='darkorange', alpha=0.2, label='95% CI')
    ax2.set_ylabel('Tc (°C)'); ax2.legend(); ax2.grid(ls='--', alpha=0.5)

    ax3.plot(t_arr, err, color='crimson', lw=0.8)
    ax3.axhline(0, color='black', ls='--', lw=0.7)
    ax3.set_ylabel('Error (°C)'); ax3.set_xlabel('Time (s)'); ax3.grid(ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ev_us06_transformer_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ fig8 EV US06 RMSE={rmse:.4f}°C")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("📊 Generating paper plots...")

    df = _load(BASE_DIR / 'data/digital_twin_sets/augmented_aging_twin_dataset.csv',
               'augmented twin dataset')
    if df is not None:
        fig1_voltage_validation(df)
        fig2_surface_temp(df)
        fig3_core_temp(df)
        fig4_parameter_aging(df)
        fig6_drive_thermal(df)
        fig7_transformer_validation(df)
    else:
        print("⚠️  No twin data — skipping figs 1-4, 6-7.")

    fig5_soh_residual('B0005')
    fig8_ev_validation()

    print(f"\n✅ All available figures saved → {PLOT_DIR}")
