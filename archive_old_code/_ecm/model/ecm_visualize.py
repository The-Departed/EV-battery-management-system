"""
Visualization tools for ECM simulation results.
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


def plot_ecm_simulation(results, save_path=None):
    """
    Plot ECM simulation results.
    
    Args:
        results: Dictionary from ECM.simulate()
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    time_min = results['time'] / 60  # Convert to minutes
    
    # 1. Terminal voltage
    axes[0].plot(time_min, results['V_terminal'], 'b-', linewidth=2, label='V_terminal')
    axes[0].plot(time_min, results['OCV'], 'r--', linewidth=1.5, alpha=0.7, label='OCV')
    axes[0].set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('ECM Simulation Results', fontsize=12, fontweight='bold')
    
    # 2. RC voltages
    axes[1].plot(time_min, results['V1'], 'g-', linewidth=1.5, label='V1 (RC1, fast)')
    axes[1].plot(time_min, results['V2'], 'm-', linewidth=1.5, label='V2 (RC2, slow)')
    axes[1].plot(time_min, results['V_R0'], 'c-', linewidth=1.5, alpha=0.7, label='V_R0 (ohmic)')
    axes[1].set_ylabel('Voltage Drop (V)', fontsize=11, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # 3. SOC
    axes[2].plot(time_min, results['SOC'] * 100, 'b-', linewidth=2)
    axes[2].set_ylabel('SOC (%)', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Current
    axes[3].plot(time_min, results['current'], 'r-', linewidth=2)
    axes[3].set_ylabel('Current (A)', fontsize=11, fontweight='bold')
    axes[3].set_xlabel('Time (min)', fontsize=11, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.close()


def plot_voltage_breakdown(results, save_path=None):
    """
    Plot voltage breakdown showing all components.
    
    Args:
        results: Dictionary from ECM.simulate()
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    time_min = results['time'] / 60
    
    # Stacked area plot
    ax.fill_between(time_min, 0, results['OCV'], 
                     alpha=0.3, label='OCV', color='blue')
    ax.fill_between(time_min, results['OCV'], 
                     results['OCV'] - results['V_R0'],
                     alpha=0.3, label='-V_R0', color='red')
    ax.fill_between(time_min, results['OCV'] - results['V_R0'],
                     results['OCV'] - results['V_R0'] - results['V1'],
                     alpha=0.3, label='-V1', color='green')
    ax.fill_between(time_min, results['OCV'] - results['V_R0'] - results['V1'],
                     results['V_terminal'],
                     alpha=0.3, label='-V2', color='purple')
    
    # Terminal voltage line
    ax.plot(time_min, results['V_terminal'], 'k-', linewidth=2, 
            label='V_terminal', zorder=10)
    
    ax.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
    ax.set_title('ECM Voltage Breakdown', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.close()


def plot_rc_dynamics(results, save_path=None):
    """
    Plot RC pair dynamics showing time constants.
    
    Args:
        results: Dictionary from ECM.simulate()
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    time = results['time']
    
    # V1 dynamics
    axes[0].plot(time, results['V1'], 'g-', linewidth=2)
    axes[0].set_ylabel('V1 (V)', fontsize=11, fontweight='bold')
    axes[0].set_title('RC1 Dynamics (Fast - SEI Layer)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, min(300, time[-1])])  # First 5 minutes
    
    # V2 dynamics
    axes[1].plot(time, results['V2'], 'm-', linewidth=2)
    axes[1].set_ylabel('V2 (V)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    axes[1].set_title('RC2 Dynamics (Slow - Diffusion)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.close()


def visualize_ecm_test():
    """Generate all ECM visualization plots."""
    print("\n" + "="*60)
    print("GENERATING ECM VISUALIZATION PLOTS")
    print("="*60)
    
    # Load simulation results
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "processed" / "ecm_simulation_test.csv"
    
    if not data_path.exists():
        print(f"✗ Simulation results not found at {data_path}")
        print("  Please run ecm_2rc.py first")
        return
    
    df = pd.read_csv(data_path)
    
    # Convert to dictionary format
    results = {
        'time': df['time'].values,
        'V_terminal': df['V_terminal'].values,
        'V1': df['V1'].values,
        'V2': df['V2'].values,
        'SOC': df['SOC'].values,
        'OCV': df['OCV'].values,
        'current': df['current'].values,
        'V_R0': df['current'].values * 0.03  # Reconstruct V_R0
    }
    
    # Create plots directory
    plots_dir = project_root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n1. ECM Simulation Overview...")
    plot_ecm_simulation(results, plots_dir / "ecm_simulation.png")
    
    print("\n2. Voltage Breakdown...")
    plot_voltage_breakdown(results, plots_dir / "ecm_voltage_breakdown.png")
    
    print("\n3. RC Dynamics...")
    plot_rc_dynamics(results, plots_dir / "ecm_rc_dynamics.png")
    
    print("\n" + "="*60)
    print("✓ All ECM plots generated")
    print("="*60)


if __name__ == "__main__":
    visualize_ecm_test()
