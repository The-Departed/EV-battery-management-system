"""
Visualization tools for ECM parameter identification results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_identification_results(cycle_num=1, save_path=None):
    """
    Plot parameter identification results for a cycle.
    
    Args:
        cycle_num: Cycle number
        save_path: Path to save figure
    """
    # Load data
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / "data" / "processed" / f"ecm_identification_cycle{cycle_num}.csv"
    
    if not data_file.exists():
        print(f"✗ Data file not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    
    # Load parameters
    params_file = project_root / "data" / "processed" / f"ecm_params_cycle{cycle_num}.csv"
    params = pd.read_csv(params_file).iloc[0]
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    time_min = df['time'].values / 60
    
    # 1. Voltage comparison (large plot)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_min, df['V_terminal_measured'], 'b-', linewidth=2, label='Measured', alpha=0.8)
    ax1.plot(time_min, df['V_terminal_model'], 'r--', linewidth=1.5, label='Model', alpha=0.8)
    ax1.set_ylabel('Terminal Voltage (V)', fontsize=11, fontweight='bold')
    ax1.set_title(f'ECM Parameter Identification - Cycle {cycle_num}', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(time_min, df['residual'] * 1000, 'g-', linewidth=1)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    rmse = np.sqrt(np.mean(df['residual']**2)) * 1000
    ax2.text(0.02, 0.95, f'RMSE = {rmse:.2f} mV', transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.set_ylabel('Residual (mV)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. RC voltages
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(time_min, df['V1'] * 1000, 'g-', linewidth=1.5, label='V1 (RC1)')
    ax3.plot(time_min, df['V2'] * 1000, 'm-', linewidth=1.5, label='V2 (RC2)')
    ax3.set_ylabel('RC Voltage (mV)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. SOC
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(time_min, df['SOC'] * 100, 'b-', linewidth=2)
    ax4.set_ylabel('SOC (%)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Parameters table
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    
    param_text = f"""
    Identified Parameters:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    R0 = {params['R0']:.6f} Ω  (Ohmic resistance)
    R1 = {params['R1']:.6f} Ω  |  C1 = {params['C1']:.2f} F  →  τ1 = {params['tau1']:.2f} s  (Fast/SEI)
    R2 = {params['R2']:.6f} Ω  |  C2 = {params['C2']:.2f} F  →  τ2 = {params['tau2']:.2f} s  (Slow/Diffusion)
    
    Performance: RMSE = {rmse:.2f} mV  |  MAE = {np.mean(np.abs(df['residual']))*1000:.2f} mV  |  Max Error = {np.max(np.abs(df['residual']))*1000:.2f} mV
    """
    ax5.text(0.05, 0.5, param_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.close()


def plot_residual_analysis(cycle_num=1, save_path=None):
    """
    Plot detailed residual analysis.
    
    Args:
        cycle_num: Cycle number
        save_path: Path to save figure
    """
    # Load data
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / "data" / "processed" / f"ecm_identification_cycle{cycle_num}.csv"
    
    if not data_file.exists():
        print(f"✗ Data file not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    residuals_mv = df['residual'].values * 1000
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Residual histogram
    axes[0, 0].hist(residuals_mv, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Residual (mV)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residual vs SOC
    axes[0, 1].scatter(df['SOC'] * 100, residuals_mv, alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('SOC (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Residual (mV)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Residual vs SOC', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residual vs voltage
    axes[1, 0].scatter(df['V_terminal_measured'], residuals_mv, alpha=0.5, s=10)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Terminal Voltage (V)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Residual (mV)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Residual vs Voltage', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    from scipy import stats
    stats.probplot(residuals_mv, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.close()


def plot_voltage_components(cycle_num=1, save_path=None):
    """
    Plot voltage component breakdown.
    
    Args:
        cycle_num: Cycle number
        save_path: Path to save figure
    """
    # Load data
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / "data" / "processed" / f"ecm_identification_cycle{cycle_num}.csv"
    
    if not data_file.exists():
        print(f"✗ Data file not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    params_file = project_root / "data" / "processed" / f"ecm_params_cycle{cycle_num}.csv"
    params = pd.read_csv(params_file).iloc[0]
    
    time_min = df['time'].values / 60
    
    # Calculate V_R0
    V_R0 = df['current'].values * params['R0']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot voltage components
    ax.plot(time_min, df['OCV'], label='OCV', linewidth=2, alpha=0.8)
    ax.plot(time_min, df['OCV'] - V_R0, label='OCV - V_R0', linewidth=1.5, alpha=0.8)
    ax.plot(time_min, df['OCV'] - V_R0 - df['V1'], label='OCV - V_R0 - V1', linewidth=1.5, alpha=0.8)
    ax.plot(time_min, df['V_terminal_model'], 'k--', label='V_terminal (model)', linewidth=2)
    ax.plot(time_min, df['V_terminal_measured'], 'r.', label='V_terminal (measured)', 
            markersize=3, alpha=0.5)
    
    ax.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
    ax.set_title(f'Voltage Component Breakdown - Cycle {cycle_num}', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.close()


def visualize_identification(cycle_num=1):
    """Generate all identification visualization plots."""
    print("\n" + "="*60)
    print(f"GENERATING IDENTIFICATION PLOTS - CYCLE {cycle_num}")
    print("="*60)
    
    # Create plots directory
    project_root = Path(__file__).parent.parent.parent
    plots_dir = project_root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n1. Identification Results Overview...")
    plot_identification_results(cycle_num, plots_dir / f"ecm_identification_cycle{cycle_num}.png")
    
    print("\n2. Residual Analysis...")
    plot_residual_analysis(cycle_num, plots_dir / f"ecm_residuals_cycle{cycle_num}.png")
    
    print("\n3. Voltage Components...")
    plot_voltage_components(cycle_num, plots_dir / f"ecm_components_cycle{cycle_num}.png")
    
    print("\n" + "="*60)
    print("✓ All identification plots generated")
    print("="*60)


if __name__ == "__main__":
    visualize_identification(cycle_num=1)
