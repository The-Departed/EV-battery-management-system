"""
Visualization utilities for battery data
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_cycle_voltage(df: pd.DataFrame, cycle_nums: list = None, 
                       save_path: str = None):
    """
    Plot voltage vs time for selected discharge cycles.
    
    Args:
        df: DataFrame with discharge data
        cycle_nums: List of cycle numbers to plot (default: first 5)
        save_path: Path to save figure
    """
    if cycle_nums is None:
        cycle_nums = sorted(df['cycle'].unique())[:5]
    
    plt.figure(figsize=(14, 6))
    
    for cycle_num in cycle_nums:
        cycle_data = df[df['cycle'] == cycle_num]
        plt.plot(cycle_data['time'], cycle_data['voltage'], 
                label=f'Cycle {cycle_num}', linewidth=1.5)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.title('Discharge Voltage vs Time', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_cycle_current_temperature(df: pd.DataFrame, cycle_num: int = 1,
                                   save_path: str = None):
    """
    Plot current and temperature for a single cycle.
    
    Args:
        df: DataFrame with discharge data
        cycle_num: Cycle number to plot
        save_path: Path to save figure
    """
    cycle_data = df[df['cycle'] == cycle_num]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Voltage
    ax1.plot(cycle_data['time'], cycle_data['voltage'], 'b-', linewidth=2)
    ax1.set_ylabel('Voltage (V)', fontsize=12)
    ax1.set_title(f'Cycle {cycle_num} - Complete Profile', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Current
    ax2.plot(cycle_data['time'], cycle_data['current'], 'r-', linewidth=2)
    ax2.set_ylabel('Current (A)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Temperature
    ax3.plot(cycle_data['time'], cycle_data['temperature'], 'g-', linewidth=2)
    ax3.set_ylabel('Temperature (°C)', fontsize=12)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_capacity_fade(df: pd.DataFrame, save_path: str = None):
    """
    Plot capacity fade over cycles.
    
    Args:
        df: DataFrame with discharge data (must have 'capacity' column)
        save_path: Path to save figure
    """
    if 'capacity' not in df.columns:
        print("Warning: No capacity data available")
        return
    
    # Get capacity per cycle
    capacity_per_cycle = df.groupby('cycle')['capacity'].first()
    
    # Convert to numeric
    capacity_per_cycle = pd.to_numeric(capacity_per_cycle, errors='coerce')
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(capacity_per_cycle.index, capacity_per_cycle.values, 
            'bo-', linewidth=2, markersize=6)
    
    plt.xlabel('Cycle Number', fontsize=12)
    plt.ylabel('Capacity (Ah)', fontsize=12)
    plt.title('Battery Capacity Fade', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_voltage_statistics(df: pd.DataFrame, save_path: str = None):
    """
    Plot voltage statistics across cycles.
    
    Args:
        df: DataFrame with discharge data
        save_path: Path to save figure
    """
    # Calculate statistics per cycle
    stats = df.groupby('cycle')['voltage'].agg(['min', 'max', 'mean']).reset_index()
    
    plt.figure(figsize=(14, 6))
    
    plt.fill_between(stats['cycle'], stats['min'], stats['max'], 
                     alpha=0.3, label='Min-Max Range')
    plt.plot(stats['cycle'], stats['mean'], 'r-', linewidth=2, 
            label='Mean Voltage')
    
    plt.xlabel('Cycle Number', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.title('Voltage Statistics Across Cycles', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def main():
    """
    Generate visualization plots for B0005 discharge data.
    """
    print("="*60)
    print("Generating Visualizations - Step 1.1")
    print("="*60)
    
    # Load processed data
    data_path = Path("data/processed/B0005_discharge.csv")
    
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run data_loader.py first.")
        return
    
    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} samples from {df['cycle'].nunique()} cycles")
    
    # Create results directory
    results_dir = Path("results/plots")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Voltage profiles for first 5 cycles
    print("\n1. Plotting voltage profiles...")
    plot_cycle_voltage(
        df, 
        cycle_nums=[1, 2, 3, 4, 5],
        save_path=results_dir / "step1_voltage_profiles.png"
    )
    
    # Plot 2: Complete profile for cycle 1
    print("\n2. Plotting complete profile for cycle 1...")
    plot_cycle_current_temperature(
        df,
        cycle_num=1,
        save_path=results_dir / "step1_cycle1_complete.png"
    )
    
    # Plot 3: Capacity fade
    print("\n3. Plotting capacity fade...")
    plot_capacity_fade(
        df,
        save_path=results_dir / "step1_capacity_fade.png"
    )
    
    # Plot 4: Voltage statistics
    print("\n4. Plotting voltage statistics...")
    plot_voltage_statistics(
        df,
        save_path=results_dir / "step1_voltage_statistics.png"
    )
    
    print("\n" + "="*60)
    print("✓ Visualizations Complete")
    print("="*60)
    print(f"\nPlots saved to {results_dir}/")


if __name__ == "__main__":
    main()
