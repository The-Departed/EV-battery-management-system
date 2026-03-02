"""
Step 6: Generate Paper-Quality Plots from Real Pipeline Data
=============================================================
All plots are driven by ACTUAL pipeline outputs, not mock data.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path


def ensure_dir():
    os.makedirs('results/paper_plots', exist_ok=True)


def load_twin_data():
    """Load the experimentally-tuned digital twin dataset."""
    p = Path('data/digital_twin_sets/augmented_aging_twin_dataset.csv')
    if not p.exists():
        print(f"⚠️ {p} not found. Run Step 4 first.")
        return None
    return pd.read_csv(p)


def load_aging_data(battery='B0005'):
    """Load per-cycle aging features."""
    p = Path(f'data/nasa/processed/{battery}_aging_features.csv')
    if not p.exists():
        return None
    return pd.read_csv(p)


# ---- Figure 1: ECM Voltage Validation ----
def fig1_voltage_validation(df):
    """Real vs Simulated terminal voltage for a sample cycle."""
    # Pick a mid-aging cycle from B0005
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


# ---- Figure 2: Thermal Model Validation (Surface Temp) ----
def fig2_surface_temp_validation(df):
    """Simulated vs Measured surface temperature — proof of thermal calibration."""
    b5 = df[df['battery'] == 'B0005']
    # Pick early, mid, late cycles
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


# ---- Figure 3: Core Temperature Prediction (the key result) ----
def fig3_core_temp_prediction(df):
    """Surface vs predicted Core temperature showing thermal inertia."""
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
    """Show how identified R0 grows with aging (SOH decay)."""
    # One R0 per cycle
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
    """Show SOH ground truth vs physics baseline vs residual correction."""
    df = load_aging_data(battery)
    if df is None:
        print(f"  ⚠️ Skipping SOH plot — no aging data for {battery}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df['cycle'], df['soh_true'], 'k-', linewidth=2, label='SOH_true (NASA)')
    ax1.plot(df['cycle'], df['soh_physics_baseline'], 'b--', label='SOH_physics (biased)')
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
    ax2.set_title(f'{battery}: Internal Resistance from Pulse Response')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('results/paper_plots/fig5_soh_residual.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ fig5_soh_residual.png")


# ---- Figure 6: Current Profile and Thermal Response ----
def fig6_drive_thermal(df):
    """Show drive current and resulting thermal response for one cycle."""
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
    else:
        print("⚠️ No digital twin data — skipping thermal plots.")

    fig5_soh_residual('B0005')

    print("\n✅ All figures saved to results/paper_plots/")
