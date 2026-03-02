"""
SOC Visualization - Step 1.2
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

sns.set_style("whitegrid")


def plot_soc_vs_time(df: pd.DataFrame, cycle_nums: list = None, 
                     save_path: str = None):
    """Plot SOC vs time for selected cycles."""
    if cycle_nums is None:
        cycle_nums = [1, 2, 3, 4, 5]
    
    plt.figure(figsize=(14, 6))
    
    for cycle_num in cycle_nums:
        cycle_data = df[df['cycle'] == cycle_num]
        plt.plot(cycle_data['time'] / 60, cycle_data['soc'] * 100, 
                label=f'Cycle {cycle_num}', linewidth=2)
    
    plt.xlabel('Time (min)', fontsize=12)
    plt.ylabel('SOC (%)', fontsize=12)
    plt.title('State of Charge vs Time - Coulomb Counting', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_voltage_vs_soc(df: pd.DataFrame, cycle_nums: list = None,
                        save_path: str = None):
    """Plot terminal voltage vs SOC."""
    if cycle_nums is None:
        cycle_nums = [1, 20, 40, 80, 120, 160]
    
    plt.figure(figsize=(14, 6))
    
    for cycle_num in cycle_nums:
        cycle_data = df[df['cycle'] == cycle_num]
        plt.plot(cycle_data['soc'] * 100, cycle_data['voltage'], 
                label=f'Cycle {cycle_num}', linewidth=2)
    
    plt.xlabel('SOC (%)', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.title('Terminal Voltage vs SOC', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_soc_voltage_current(df: pd.DataFrame, cycle_num: int = 1,
                             save_path: str = None):
    """Plot SOC, voltage, and current for a single cycle."""
    cycle_data = df[df['cycle'] == cycle_num]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    time_min = cycle_data['time'] / 60
    
    # SOC
    ax1.plot(time_min, cycle_data['soc'] * 100, 'b-', linewidth=2)
    ax1.set_ylabel('SOC (%)', fontsize=12)
    ax1.set_title(f'Cycle {cycle_num} - SOC Estimation', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Voltage
    ax2.plot(time_min, cycle_data['voltage'], 'r-', linewidth=2)
    ax2.set_ylabel('Voltage (V)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Current
    ax3.plot(time_min, cycle_data['current'], 'g-', linewidth=2)
    ax3.set_ylabel('Current (A)', fontsize=12)
    ax3.set_xlabel('Time (min)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_capacity_fade_soc(df: pd.DataFrame, save_path: str = None):
    """Plot capacity fade effect on SOC."""
    # Get capacity and final SOC for each cycle
    cycle_stats = df.groupby('cycle').agg({
        'capacity': 'first',
        'soc': 'min'
    }).reset_index()
    
    cycle_stats['capacity'] = pd.to_numeric(cycle_stats['capacity'], errors='coerce')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Capacity
    ax1.plot(cycle_stats['cycle'], cycle_stats['capacity'], 
            'bo-', linewidth=2, markersize=4)
    ax1.set_ylabel('Capacity (Ah)', fontsize=12)
    ax1.set_title('Capacity Fade and SOC Depth', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Minimum SOC (depth of discharge)
    ax2.plot(cycle_stats['cycle'], cycle_stats['soc'] * 100, 
            'ro-', linewidth=2, markersize=4)
    ax2.set_ylabel('Min SOC (%)', fontsize=12)
    ax2.set_xlabel('Cycle Number', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-5, 10])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Generate all SOC visualizations."""
    print("="*60)
    print("Generating SOC Visualizations - Step 1.2")
    print("="*60)
    
    # Load data
    df = pd.read_csv("data/processed/B0005_discharge_soc.csv")
    print(f"\nLoaded {len(df)} samples with SOC")
    
    results_dir = Path("results/plots")
    
    # Plot 1: SOC vs time
    print("\n1. Plotting SOC vs time...")
    plot_soc_vs_time(df, cycle_nums=[1, 2, 3, 4, 5],
                     save_path=results_dir / "step2_soc_vs_time.png")
    
    # Plot 2: Voltage vs SOC
    print("\n2. Plotting voltage vs SOC...")
    plot_voltage_vs_soc(df, cycle_nums=[1, 20, 40, 80, 120, 160],
                        save_path=results_dir / "step2_voltage_vs_soc.png")
    
    # Plot 3: Complete profile
    print("\n3. Plotting complete profile for cycle 1...")
    plot_soc_voltage_current(df, cycle_num=1,
                             save_path=results_dir / "step2_cycle1_soc_profile.png")
    
    # Plot 4: Capacity fade effect
    print("\n4. Plotting capacity fade effect...")
    plot_capacity_fade_soc(df, save_path=results_dir / "step2_capacity_soc.png")
    
    print("\n" + "="*60)
    print("✓ SOC Visualizations Complete")
    print("="*60)


if __name__ == "__main__":
    main()
