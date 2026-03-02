"""
Visualization for EETM data
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


def plot_thermal_data(csv_file='calce_thermal_50SOC.csv', save_path=None):
    """
    Plot thermal data from CALCE experiment.
    
    Args:
        csv_file: Processed thermal CSV file
        save_path: Path to save figure
    """
    # Load data
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "processed" / csv_file
    
    if not data_path.exists():
        print(f"✗ Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    time_min = df['time'].values / 60  # Convert to minutes
    
    # 1. Surface Temperature
    axes[0].plot(time_min, df['Ts'], 'r-', linewidth=1.5, label='Ts (Surface)')
    axes[0].plot(time_min, df['Tamb'], 'b--', linewidth=1.5, alpha=0.7, label='Tamb (Ambient)')
    axes[0].set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    axes[0].set_title('CALCE Thermal Data - Step 2.1', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Current Profile
    axes[1].plot(time_min, df['current'], 'g-', linewidth=1.5)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Current (A)', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Voltage
    if 'voltage' in df.columns:
        axes[2].plot(time_min, df['voltage'], 'b-', linewidth=1.5)
        axes[2].set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No voltage data', ha='center', va='center',
                     transform=axes[2].transAxes, fontsize=14)
        axes[2].set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
    
    # 4. Temperature Rise (ΔT)
    dT = df['Ts'].values - df['Tamb'].values
    axes[3].plot(time_min, dT, 'm-', linewidth=1.5)
    axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[3].set_ylabel('ΔT = Ts - Tamb (°C)', fontsize=11, fontweight='bold')
    axes[3].set_xlabel('Time (min)', fontsize=11, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.close()


def plot_thermal_summary(csv_file='calce_thermal_50SOC.csv', save_path=None):
    """
    Plot thermal data summary statistics.
    
    Args:
        csv_file: Processed thermal CSV file
        save_path: Path to save figure
    """
    # Load data
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "processed" / csv_file
    
    if not data_path.exists():
        print(f"✗ Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Temperature histogram
    axes[0, 0].hist(df['Ts'], bins=50, edgecolor='black', alpha=0.7, color='red')
    axes[0, 0].axvline(x=df['Ts'].mean(), color='b', linestyle='--', linewidth=2, 
                       label=f'Mean: {df["Ts"].mean():.2f}°C')
    axes[0, 0].set_xlabel('Surface Temperature (°C)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Temperature Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Current histogram
    axes[0, 1].hist(df['current'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].axvline(x=0, color='k', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('Current (A)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Current Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Temperature vs Current scatter
    axes[1, 0].scatter(df['current'], df['Ts'], alpha=0.3, s=5)
    axes[1, 0].set_xlabel('Current (A)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Surface Temperature (°C)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Temperature vs Current', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Summary statistics table
    axes[1, 1].axis('off')
    
    summary_text = f"""
    THERMAL DATA SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Samples:        {len(df):,}
    Duration:       {df['time'].max()/60:.2f} minutes
    Sampling Rate:  ~{1/df['time'].diff().mean():.2f} Hz
    
    Temperature (Ts):
      Min:    {df['Ts'].min():.3f} °C
      Max:    {df['Ts'].max():.3f} °C
      Mean:   {df['Ts'].mean():.3f} °C
      Std:    {df['Ts'].std():.3f} °C
      ΔT:     {df['Ts'].max() - df['Ts'].min():.3f} °C
    
    Current:
      Min:    {df['current'].min():.3f} A
      Max:    {df['current'].max():.3f} A
      Mean:   {df['current'].mean():.3f} A
      RMS:    {np.sqrt(np.mean(df['current']**2)):.3f} A
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.close()


def visualize_step21():
    """Generate all Step 2.1 visualization plots."""
    print("\n" + "="*60)
    print("GENERATING STEP 2.1 PLOTS")
    print("="*60)
    
    # Create plots directory
    project_root = Path(__file__).parent.parent
    plots_dir = project_root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n1. Thermal Data Overview...")
    plot_thermal_data('calce_thermal_50SOC.csv', plots_dir / "step21_thermal_data.png")
    
    print("\n2. Thermal Data Summary...")
    plot_thermal_summary('calce_thermal_50SOC.csv', plots_dir / "step21_thermal_summary.png")
    
    print("\n" + "="*60)
    print("✓ All Step 2.1 plots generated")
    print("="*60)


def plot_heat_generation(csv_file='calce_with_heat.csv', save_path=None):
    """
    Plot heat generation Q(t) from ECM.
    
    Args:
        csv_file: CSV with heat data
        save_path: Path to save figure
    """
    # Load data
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "processed" / csv_file
    
    if not data_path.exists():
        print(f"✗ Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    time_min = df['time'].values / 60
    
    # 1. Heat Generation
    axes[0].plot(time_min, df['Q_total'], 'r-', linewidth=1.5, label='Q_total')
    if 'Q_joule' in df.columns:
        axes[0].plot(time_min, df['Q_joule'], 'r--', linewidth=1, alpha=0.7, label='Q_joule')
    axes[0].set_ylabel('Heat Generation (W)', fontsize=11, fontweight='bold')
    axes[0].set_title('Heat Generation from ECM - Step 2.2', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 2. Current (for reference)
    axes[1].plot(time_min, df['current'], 'g-', linewidth=1.5)
    axes[1].set_ylabel('Current (A)', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 3. Heat vs Current² (should be proportional)
    axes[2].scatter(df['current']**2, df['Q_total'], alpha=0.5, s=10)
    axes[2].set_xlabel('I² (A²)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Heat (W)', fontsize=11, fontweight='bold')
    axes[2].set_title('Joule Heating: Q ∝ I²', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Add linear fit line
    I2 = df['current'].values**2
    Q = df['Q_total'].values
    # Effective resistance from slope
    R_eff = np.polyfit(I2, Q, 1)[0]
    I2_fit = np.linspace(0, I2.max(), 100)
    Q_fit = R_eff * I2_fit
    axes[2].plot(I2_fit, Q_fit, 'r--', linewidth=2, alpha=0.7, 
                label=f'R_eff = {R_eff:.4f} Ω')
    axes[2].legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.close()


def visualize_step22():
    """Generate all Step 2.2 visualization plots."""
    print("\n" + "="*60)
    print("GENERATING STEP 2.2 PLOTS")
    print("="*60)
    
    # Create plots directory
    project_root = Path(__file__).parent.parent
    plots_dir = project_root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n1. Heat Generation...")
    plot_heat_generation('calce_with_heat.csv', plots_dir / "step22_heat_generation.png")
    
    print("\n" + "="*60)
    print("✓ All Step 2.2 plots generated")
    print("="*60)


def plot_eetm_dynamics(results, output_path, title="EETM Thermal Dynamics"):
    """
    Plot EETM thermal dynamics.
    
    Args:
        results: Dictionary from EETM simulation
        output_path: Path to save plot
        title: Plot title
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    time_min = results['time'] / 60  # Convert to minutes
    
    # 1. Temperature evolution
    ax = axes[0, 0]
    ax.plot(time_min, results['Tc'], 'r-', linewidth=2, label='Tc (core)')
    ax.plot(time_min, results['Ts'], 'b-', linewidth=2, label='Ts (surface)')
    ax.plot(time_min, results['Tamb'], 'k--', linewidth=1.5, label='Tamb')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Evolution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2. Temperature differences
    ax = axes[0, 1]
    ax.plot(time_min, results['dT_core_surface'], 'g-', linewidth=2, label='Tc - Ts')
    ax.plot(time_min, results['dT_surface_ambient'], 'm-', linewidth=2, label='Ts - Tamb')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Temperature Difference (°C)')
    ax.set_title('Temperature Gradients')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle=':', linewidth=1)
    
    # 3. Heat input
    ax = axes[1, 0]
    ax.plot(time_min, results['Q'], 'orange', linewidth=2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Heat Generation (W)')
    ax.set_title('Heat Input Q(t)')
    ax.grid(True, alpha=0.3)
    
    # 4. Heat flows
    ax = axes[1, 1]
    ax.plot(time_min, results['Q_core_surface'], 'r-', linewidth=2, label='Core → Surface')
    ax.plot(time_min, results['Q_surface_ambient'], 'b-', linewidth=2, label='Surface → Ambient')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Heat Flow (W)')
    ax.set_title('Heat Transfer Paths')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 5. Temperature rise from ambient
    ax = axes[2, 0]
    dTc = results['Tc'] - results['Tamb']
    dTs = results['Ts'] - results['Tamb']
    ax.plot(time_min, dTc, 'r-', linewidth=2, label='ΔTc')
    ax.plot(time_min, dTs, 'b-', linewidth=2, label='ΔTs')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Temperature Rise (°C)')
    ax.set_title('Temperature Rise from Ambient')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 6. Phase plane (Tc vs Ts)
    ax = axes[2, 1]
    scatter = ax.scatter(results['Ts'], results['Tc'], c=time_min, 
                        cmap='viridis', s=20, alpha=0.6)
    ax.plot(results['Ts'][0], results['Tc'][0], 'go', markersize=10, 
            label='Start', zorder=5)
    ax.plot(results['Ts'][-1], results['Tc'][-1], 'ro', markersize=10, 
            label='End', zorder=5)
    ax.plot([min(results['Ts']), max(results['Ts'])], 
            [min(results['Ts']), max(results['Ts'])], 
            'k--', linewidth=1, alpha=0.5, label='Tc = Ts')
    ax.set_xlabel('Ts (°C)')
    ax.set_ylabel('Tc (°C)')
    ax.set_title('Phase Plane (Tc vs Ts)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (min)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_step23():
    """Generate all Step 2.3 visualization plots."""
    print("\n" + "="*60)
    print("GENERATING STEP 2.3 PLOTS")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "processed" / "eetm_test_simulation.csv"
    plots_dir = project_root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {data_path.name}...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples")
    
    # Convert to results dictionary
    results = {
        'time': df['time'].values,
        'Tc': df['Tc'].values,
        'Ts': df['Ts'].values,
        'Tamb': df['Tamb'].values,
        'Q': df['Q'].values,
        'Q_core_surface': df['Q_core_surface'].values,
        'Q_surface_ambient': df['Q_surface_ambient'].values,
        'dT_core_surface': df['Tc'].values - df['Ts'].values,
        'dT_surface_ambient': df['Ts'].values - df['Tamb'].values
    }
    
    # Generate plot
    print("\n1. EETM Thermal Dynamics...")
    output_path = plots_dir / "step23_eetm_test.png"
    plot_eetm_dynamics(results, output_path, 
                       title="Step 2.3: EETM Model Test (Constant Heat Input)")
    
    print(f"✓ Plot saved to {output_path}")
    print("\n" + "="*60)
    print("✓ All Step 2.3 plots generated")
    print("="*60)


def plot_parameter_identification(data_path, output_path):
    """
    Plot parameter identification results.
    
    Args:
        data_path: Path to identification results CSV
        output_path: Path to save plot
    """
    df = pd.read_csv(data_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Step 2.4: EETM Parameter Identification', fontsize=16, fontweight='bold')
    
    time_min = df['time'].values / 60
    
    # 1. Measured vs Model Temperature
    ax = axes[0, 0]
    ax.plot(time_min, df['Ts_measured'], 'b-', linewidth=2, label='Ts measured', alpha=0.7)
    ax.plot(time_min, df['Ts_model'], 'r--', linewidth=2, label='Ts model')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Surface Temperature (°C)')
    ax.set_title('Temperature Fit')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2. Residuals
    ax = axes[0, 1]
    ax.plot(time_min, df['residual'], 'g-', linewidth=1, alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Residual (°C)')
    ax.set_title('Fitting Residuals (Measured - Model)')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    rmse = np.sqrt(np.mean(df['residual']**2))
    mae = np.mean(np.abs(df['residual']))
    ax.text(0.02, 0.98, f'RMSE = {rmse:.4f} °C\nMAE = {mae:.4f} °C',
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Core and Surface Temperatures
    ax = axes[1, 0]
    ax.plot(time_min, df['Tc_model'], 'r-', linewidth=2, label='Tc (core)', alpha=0.7)
    ax.plot(time_min, df['Ts_model'], 'b-', linewidth=2, label='Ts (surface)', alpha=0.7)
    ax.plot(time_min, df['Tamb'], 'k--', linewidth=1.5, label='Tamb', alpha=0.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Model Temperatures')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 4. Residual histogram
    ax = axes[1, 1]
    ax.hist(df['residual'], bins=50, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='k', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (°C)')
    ax.set_ylabel('Frequency')
    ax.set_title('Residual Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_res = np.mean(df['residual'])
    std_res = np.std(df['residual'])
    ax.text(0.98, 0.98, f'Mean = {mean_res:.4f} °C\nStd = {std_res:.4f} °C',
            transform=ax.transAxes, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_step24():
    """Generate all Step 2.4 visualization plots."""
    print("\n" + "="*60)
    print("GENERATING STEP 2.4 PLOTS")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "processed" / "eetm_identification_results.csv"
    plots_dir = project_root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {data_path.name}...")
    if not data_path.exists():
        print(f"✗ Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples")
    
    # Generate plot
    print("\n1. Parameter Identification Results...")
    output_path = plots_dir / "step24_parameter_identification.png"
    plot_parameter_identification(data_path, output_path)
    
    print(f"✓ Plot saved to {output_path}")
    print("\n" + "="*60)
    print("✓ All Step 2.4 plots generated")
    print("="*60)


def plot_validation_results(data_path, output_path):
    """
    Plot validation results.
    
    Args:
        data_path: Path to validation results CSV
        output_path: Path to save plot
    """
    df = pd.read_csv(data_path)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Step 2.5: EETM Model Validation', fontsize=16, fontweight='bold')
    
    time_min = df['time'].values / 60
    
    # 1. Temperature comparison
    ax = axes[0, 0]
    ax.plot(time_min, df['Ts_measured'], 'b-', linewidth=2, label='Ts measured', alpha=0.7)
    ax.plot(time_min, df['Ts_model'], 'r--', linewidth=2, label='Ts model')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Surface Temperature (°C)')
    ax.set_title('Surface Temperature: Measured vs Model')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2. Core and surface temperatures
    ax = axes[0, 1]
    ax.plot(time_min, df['Tc_model'], 'r-', linewidth=2, label='Tc (core)', alpha=0.7)
    ax.plot(time_min, df['Ts_model'], 'b-', linewidth=2, label='Ts (surface)', alpha=0.7)
    ax.plot(time_min, df['Tamb'], 'k--', linewidth=1.5, label='Tamb', alpha=0.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Model Predictions (Tc, Ts, Tamb)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3. Residuals over time
    ax = axes[1, 0]
    ax.plot(time_min, df['residual'], 'g-', linewidth=1, alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Residual (°C)')
    ax.set_title('Prediction Residuals (Measured - Model)')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    rmse = np.sqrt(np.mean(df['residual']**2))
    mae = np.mean(np.abs(df['residual']))
    max_err = np.max(np.abs(df['residual']))
    ax.text(0.02, 0.98, f'RMSE = {rmse:.4f} °C\nMAE = {mae:.4f} °C\nMax = {max_err:.4f} °C',
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 4. Residual histogram
    ax = axes[1, 1]
    ax.hist(df['residual'], bins=50, color='green', alpha=0.7, edgecolor='black', density=True)
    ax.axvline(0, color='k', linestyle='--', linewidth=2)
    
    # Add normal distribution fit
    mean_res = np.mean(df['residual'])
    std_res = np.std(df['residual'])
    x = np.linspace(df['residual'].min(), df['residual'].max(), 100)
    from scipy.stats import norm
    ax.plot(x, norm.pdf(x, mean_res, std_res), 'r-', linewidth=2, label='Normal fit')
    
    ax.set_xlabel('Residual (°C)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Residual Distribution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.text(0.98, 0.98, f'μ = {mean_res:.4f} °C\nσ = {std_res:.4f} °C',
            transform=ax.transAxes, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 5. Scatter plot: Measured vs Model
    ax = axes[2, 0]
    ax.scatter(df['Ts_measured'], df['Ts_model'], alpha=0.5, s=10, c=time_min, cmap='viridis')
    
    # Perfect prediction line
    min_temp = min(df['Ts_measured'].min(), df['Ts_model'].min())
    max_temp = max(df['Ts_measured'].max(), df['Ts_model'].max())
    ax.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', linewidth=2, label='Perfect fit')
    
    ax.set_xlabel('Ts Measured (°C)')
    ax.set_ylabel('Ts Model (°C)')
    ax.set_title('Parity Plot: Model vs Measured')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Add R² value
    ss_res = np.sum((df['Ts_measured'] - df['Ts_model'])**2)
    ss_tot = np.sum((df['Ts_measured'] - df['Ts_measured'].mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    ax.text(0.02, 0.98, f'R² = {r2:.6f}',
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 6. Temperature rise and gradients
    ax = axes[2, 1]
    dT_core_surface = df['Tc_model'] - df['Ts_model']
    dT_surface_amb = df['Ts_model'] - df['Tamb']
    
    ax.plot(time_min, dT_core_surface, 'r-', linewidth=2, label='Tc - Ts', alpha=0.7)
    ax.plot(time_min, dT_surface_amb, 'b-', linewidth=2, label='Ts - Tamb', alpha=0.7)
    ax.axhline(0, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Temperature Difference (°C)')
    ax.set_title('Thermal Gradients')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_step25():
    """Generate all Step 2.5 visualization plots."""
    print("\n" + "="*60)
    print("GENERATING STEP 2.5 PLOTS")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "processed" / "eetm_validation_results.csv"
    plots_dir = project_root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {data_path.name}...")
    if not data_path.exists():
        print(f"✗ Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples")
    
    # Generate plot
    print("\n1. Validation Results...")
    output_path = plots_dir / "step25_validation.png"
    plot_validation_results(data_path, output_path)
    
    print(f"✓ Plot saved to {output_path}")
    print("\n" + "="*60)
    print("✓ All Step 2.5 plots generated")
    print("="*60)


def plot_ekf_results(data_path, output_path):
    """
    Plot Extended Kalman Filter results.
    
    Args:
        data_path: Path to EKF results CSV
        output_path: Path to save plot
    """
    df = pd.read_csv(data_path)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Step 2.6: Extended Kalman Filter - Core Temperature Estimation', 
                 fontsize=16, fontweight='bold')
    
    time_min = df['time'].values / 60
    
    # 1. Core and surface temperature estimates
    ax = axes[0, 0]
    ax.plot(time_min, df['Tc_estimated'], 'r-', linewidth=2, label='Tc (core)', alpha=0.8)
    ax.plot(time_min, df['Ts_estimated'], 'b-', linewidth=2, label='Ts (surface)', alpha=0.8)
    ax.plot(time_min, df['Ts_measured'], 'b.', markersize=1, label='Ts measured', alpha=0.3)
    ax.plot(time_min, df['Tamb'], 'k--', linewidth=1.5, label='Tamb', alpha=0.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Estimates')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2. Tc with uncertainty bounds
    ax = axes[0, 1]
    Tc = df['Tc_estimated'].values
    sigma_Tc = df['sigma_Tc'].values
    ax.plot(time_min, Tc, 'r-', linewidth=2, label='Tc estimate')
    ax.fill_between(time_min, Tc - 2*sigma_Tc, Tc + 2*sigma_Tc, 
                     color='red', alpha=0.2, label='±2σ (95% confidence)')
    ax.fill_between(time_min, Tc - sigma_Tc, Tc + sigma_Tc, 
                     color='red', alpha=0.3, label='±1σ (68% confidence)')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Core Temperature (°C)')
    ax.set_title('Core Temperature with Uncertainty')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3. Uncertainty evolution
    ax = axes[1, 0]
    ax.plot(time_min, df['sigma_Tc'], 'r-', linewidth=2, label='σ_Tc')
    ax.plot(time_min, df['sigma_Ts'], 'b-', linewidth=2, label='σ_Ts')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Uncertainty (°C)')
    ax.set_title('State Uncertainty Evolution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 4. Innovation sequence
    ax = axes[1, 1]
    ax.plot(time_min, df['innovation'], 'g-', linewidth=0.5, alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Innovation (°C)')
    ax.set_title('Measurement Innovation (Ts_measured - Ts_predicted)')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    innov_mean = np.mean(df['innovation'])
    innov_std = np.std(df['innovation'])
    ax.text(0.02, 0.98, f'Mean = {innov_mean:.4f} °C\nStd = {innov_std:.4f} °C',
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 5. Core-surface temperature gradient
    ax = axes[2, 0]
    dT = df['Tc_estimated'] - df['Ts_estimated']
    ax.plot(time_min, dT, 'purple', linewidth=2)
    ax.axhline(0, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('ΔT (°C)')
    ax.set_title('Core-Surface Temperature Gradient (Tc - Ts)')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    dT_mean = np.mean(dT)
    dT_max = np.max(dT)
    ax.text(0.02, 0.98, f'Mean = {dT_mean:.3f} °C\nMax = {dT_max:.3f} °C',
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 6. Ts tracking performance
    ax = axes[2, 1]
    ax.scatter(df['Ts_measured'], df['Ts_estimated'], alpha=0.3, s=5, c=time_min, cmap='viridis')
    
    # Perfect tracking line
    min_temp = min(df['Ts_measured'].min(), df['Ts_estimated'].min())
    max_temp = max(df['Ts_measured'].max(), df['Ts_estimated'].max())
    ax.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', linewidth=2, label='Perfect tracking')
    
    ax.set_xlabel('Ts Measured (°C)')
    ax.set_ylabel('Ts Estimated (°C)')
    ax.set_title('Surface Temperature Tracking')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Add RMSE
    error = df['Ts_estimated'] - df['Ts_measured']
    rmse = np.sqrt(np.mean(error**2))
    ax.text(0.02, 0.98, f'RMSE = {rmse:.4f} °C',
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_step26():
    """Generate all Step 2.6 visualization plots."""
    print("\n" + "="*60)
    print("GENERATING STEP 2.6 PLOTS")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "processed" / "ekf_results.csv"
    plots_dir = project_root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {data_path.name}...")
    if not data_path.exists():
        print(f"✗ Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples")
    
    # Generate plot
    print("\n1. EKF Results...")
    output_path = plots_dir / "step26_ekf_estimation.png"
    plot_ekf_results(data_path, output_path)
    
    print(f"✓ Plot saved to {output_path}")
    print("\n" + "="*60)
    print("✓ All Step 2.6 plots generated")
    print("="*60)


if __name__ == "__main__":
    # Check which step to visualize
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '2.2':
            visualize_step22()
        elif sys.argv[1] == '2.3':
            visualize_step23()
        elif sys.argv[1] == '2.4':
            visualize_step24()
        elif sys.argv[1] == '2.5':
            visualize_step25()
        elif sys.argv[1] == '2.6':
            visualize_step26()
        else:
            visualize_step21()
    else:
        visualize_step21()

