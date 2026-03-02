"""
Visualization tools for ECM validation results.
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


def plot_validation_overview(save_path=None):
    """
    Plot validation overview for all tested cycles.
    """
    # Load data
    project_root = Path(__file__).parent.parent.parent
    metrics_file = project_root / "data" / "processed" / "ecm_validation_metrics.csv"
    
    if not metrics_file.exists():
        print(f"✗ Metrics file not found: {metrics_file}")
        return
    
    metrics = pd.read_csv(metrics_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. RMSE vs Cycle
    axes[0, 0].plot(metrics['cycle'], metrics['rmse_mv'], 'o-', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=150, color='g', linestyle='--', alpha=0.5, label='Excellent (<150mV)')
    axes[0, 0].axhline(y=250, color='orange', linestyle='--', alpha=0.5, label='Good (<250mV)')
    axes[0, 0].set_xlabel('Cycle Number', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('RMSE (mV)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Model Error vs Battery Aging', fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. R² vs Cycle
    axes[0, 1].plot(metrics['cycle'], metrics['r2'], 's-', linewidth=2, markersize=8, color='green')
    axes[0, 1].axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='Excellent (>0.95)')
    axes[0, 1].axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, label='Good (>0.90)')
    axes[0, 1].set_xlabel('Cycle Number', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('R² Score', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Model Fit Quality vs Battery Aging', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0.85, 1.0])
    
    # 3. Error metrics comparison
    x = np.arange(len(metrics))
    width = 0.35
    axes[1, 0].bar(x - width/2, metrics['rmse_mv'], width, label='RMSE', alpha=0.8)
    axes[1, 0].bar(x + width/2, metrics['mae_mv'], width, label='MAE', alpha=0.8)
    axes[1, 0].set_xlabel('Cycle Number', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Error (mV)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Error Metrics Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics['cycle'])
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Summary statistics
    axes[1, 1].axis('off')
    
    summary_text = f"""
    VALIDATION SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Cycles Tested: {len(metrics)}
    Cycle Range: {metrics['cycle'].min()} - {metrics['cycle'].max()}
    
    Performance Metrics:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    RMSE: {metrics['rmse_mv'].mean():.2f} ± {metrics['rmse_mv'].std():.2f} mV
    MAE:  {metrics['mae_mv'].mean():.2f} ± {metrics['mae_mv'].std():.2f} mV
    R²:   {metrics['r2'].mean():.4f} ± {metrics['r2'].std():.4f}
    MAPE: {metrics['mape_percent'].mean():.2f} ± {metrics['mape_percent'].std():.2f} %
    
    Range:
    RMSE: [{metrics['rmse_mv'].min():.2f}, {metrics['rmse_mv'].max():.2f}] mV
    R²:   [{metrics['r2'].min():.4f}, {metrics['r2'].max():.4f}]
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle('ECM Cross-Cycle Validation Results', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.close()


def plot_cycle_comparison(cycle_list=None, save_path=None):
    """
    Plot voltage comparison for multiple cycles.
    """
    project_root = Path(__file__).parent.parent.parent
    
    if cycle_list is None:
        # Auto-detect available cycles
        import glob
        files = glob.glob(str(project_root / "data" / "processed" / "ecm_validation_cycle*.csv"))
        cycle_list = [int(f.split('cycle')[1].split('.')[0]) for f in files]
        cycle_list.sort()
    
    n_cycles = len(cycle_list)
    fig, axes = plt.subplots(n_cycles, 1, figsize=(14, 4*n_cycles))
    
    if n_cycles == 1:
        axes = [axes]
    
    for idx, cycle in enumerate(cycle_list):
        data_file = project_root / "data" / "processed" / f"ecm_validation_cycle{cycle}.csv"
        
        if not data_file.exists():
            print(f"⚠ Cycle {cycle} data not found")
            continue
        
        df = pd.read_csv(data_file)
        time_min = df['time'].values / 60
        
        # Voltage comparison
        axes[idx].plot(time_min, df['voltage_measured'], 'b-', linewidth=2, 
                      label='Measured', alpha=0.7)
        axes[idx].plot(time_min, df['voltage_predicted'], 'r--', linewidth=1.5, 
                      label='Predicted', alpha=0.8)
        
        rmse = np.sqrt(np.mean(df['residual']**2)) * 1000
        r2 = 1 - np.sum(df['residual']**2) / np.sum((df['voltage_measured'] - df['voltage_measured'].mean())**2)
        
        axes[idx].set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'Cycle {cycle} - RMSE: {rmse:.2f} mV, R²: {r2:.4f}', 
                           fontsize=12, fontweight='bold')
        axes[idx].legend(loc='best')
        axes[idx].grid(True, alpha=0.3)
        
        if idx == n_cycles - 1:
            axes[idx].set_xlabel('Time (min)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.close()


def plot_residual_heatmap(cycle_list=None, save_path=None):
    """
    Plot residual heatmap across SOC and cycles.
    """
    project_root = Path(__file__).parent.parent.parent
    
    if cycle_list is None:
        import glob
        files = glob.glob(str(project_root / "data" / "processed" / "ecm_validation_cycle*.csv"))
        cycle_list = [int(f.split('cycle')[1].split('.')[0]) for f in files]
        cycle_list.sort()
    
    # Collect residuals vs SOC for all cycles
    soc_bins = np.linspace(0, 1, 21)  # 20 bins
    residual_matrix = []
    
    for cycle in cycle_list:
        data_file = project_root / "data" / "processed" / f"ecm_validation_cycle{cycle}.csv"
        
        if not data_file.exists():
            continue
        
        df = pd.read_csv(data_file)
        
        # Bin residuals by SOC
        binned_residuals = []
        for i in range(len(soc_bins) - 1):
            mask = (df['soc'] >= soc_bins[i]) & (df['soc'] < soc_bins[i+1])
            if mask.sum() > 0:
                binned_residuals.append(df.loc[mask, 'residual'].mean() * 1000)
            else:
                binned_residuals.append(np.nan)
        
        residual_matrix.append(binned_residuals)
    
    residual_matrix = np.array(residual_matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(residual_matrix, aspect='auto', cmap='RdBu_r', 
                   vmin=-200, vmax=200, interpolation='nearest')
    
    ax.set_yticks(range(len(cycle_list)))
    ax.set_yticklabels(cycle_list)
    ax.set_ylabel('Cycle Number', fontsize=11, fontweight='bold')
    
    soc_labels = [f'{int(s*100)}%' for s in soc_bins[:-1:4]]
    ax.set_xticks(range(0, len(soc_bins)-1, 4))
    ax.set_xticklabels(soc_labels)
    ax.set_xlabel('State of Charge (SOC)', fontsize=11, fontweight='bold')
    
    ax.set_title('Model Residuals vs SOC and Battery Aging', fontsize=12, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Residual (mV)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.close()


def visualize_validation():
    """Generate all validation visualization plots."""
    print("\n" + "="*60)
    print("GENERATING VALIDATION PLOTS")
    print("="*60)
    
    # Create plots directory
    project_root = Path(__file__).parent.parent.parent
    plots_dir = project_root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n1. Validation Overview...")
    plot_validation_overview(plots_dir / "ecm_validation_overview.png")
    
    print("\n2. Cycle Comparison...")
    plot_cycle_comparison(save_path=plots_dir / "ecm_validation_cycles.png")
    
    print("\n3. Residual Heatmap...")
    plot_residual_heatmap(save_path=plots_dir / "ecm_validation_heatmap.png")
    
    print("\n" + "="*60)
    print("✓ All validation plots generated")
    print("="*60)


if __name__ == "__main__":
    visualize_validation()
