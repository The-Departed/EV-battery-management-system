"""
OCV-SOC Visualization - Step 1.3
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ecm.ocv.ocv_model import OCVModel

sns.set_style("whitegrid")


def plot_ocv_soc_curve(ocv_data: pd.DataFrame, model: OCVModel,
                       save_path: str = None):
    """Plot OCV-SOC curve with fitted model."""
    plt.figure(figsize=(14, 6))
    
    # Plot data points
    plt.scatter(ocv_data['soc'] * 100, ocv_data['ocv'], 
               alpha=0.6, s=50, label='Extracted OCV points', color='blue')
    
    # Plot fitted curve
    soc_smooth = np.linspace(0, 1, 200)
    ocv_smooth = model.predict(soc_smooth)
    plt.plot(soc_smooth * 100, ocv_smooth, 
            'r-', linewidth=2.5, label=f'{model.method.capitalize()} fit (degree={model.degree})')
    
    plt.xlabel('SOC (%)', fontsize=12)
    plt.ylabel('OCV (V)', fontsize=12)
    plt.title('Open Circuit Voltage vs State of Charge', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_ocv_derivative(model: OCVModel, save_path: str = None):
    """Plot dOCV/dSOC (useful for understanding battery characteristics)."""
    soc = np.linspace(0.01, 0.99, 200)
    ocv = model.predict(soc)
    
    # Numerical derivative
    d_ocv_d_soc = np.gradient(ocv, soc)
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(soc * 100, d_ocv_d_soc, 'b-', linewidth=2)
    plt.xlabel('SOC (%)', fontsize=12)
    plt.ylabel('dOCV/dSOC (V)', fontsize=12)
    plt.title('OCV Sensitivity to SOC', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison(ocv_data: pd.DataFrame, save_path: str = None):
    """Compare different OCV models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    soc_smooth = np.linspace(0, 1, 200)
    
    models = [
        ('polynomial', 4),
        ('polynomial', 6),
        ('polynomial', 8),
        ('spline', 0.1)
    ]
    
    for idx, (method, param) in enumerate(models):
        ax = axes[idx // 2, idx % 2]
        
        # Fit model
        model = OCVModel(method=method, degree=param if method == 'polynomial' else int(param*100))
        model.fit(ocv_data['soc'].values, ocv_data['ocv'].values)
        
        # Evaluate
        metrics = model.evaluate(ocv_data['soc'].values, ocv_data['ocv'].values)
        
        # Plot
        ax.scatter(ocv_data['soc'] * 100, ocv_data['ocv'], 
                  alpha=0.4, s=30, label='Data', color='blue')
        ocv_smooth = model.predict(soc_smooth)
        ax.plot(soc_smooth * 100, ocv_smooth, 'r-', linewidth=2,
               label=f'{method.capitalize()} (param={param})')
        
        ax.set_xlabel('SOC (%)', fontsize=11)
        ax.set_ylabel('OCV (V)', fontsize=11)
        ax.set_title(f'{method.capitalize()} | RMSE={metrics["RMSE"]*1000:.2f} mV', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_residuals(ocv_data: pd.DataFrame, model: OCVModel, save_path: str = None):
    """Plot residuals (errors) of OCV model."""
    soc = ocv_data['soc'].values
    ocv_true = ocv_data['ocv'].values
    ocv_pred = model.predict(soc)
    residuals = (ocv_pred - ocv_true) * 1000  # Convert to mV
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Residuals vs SOC
    ax1.scatter(soc * 100, residuals, alpha=0.6, s=50, color='red')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('SOC (%)', fontsize=12)
    ax1.set_ylabel('Residual (mV)', fontsize=12)
    ax1.set_title('Model Residuals vs SOC', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax2.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='blue')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax2.set_xlabel('Residual (mV)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Generate all OCV visualizations."""
    print("="*60)
    print("Generating OCV Visualizations - Step 1.3")
    print("="*60)
    
    # Load data
    project_root = Path(__file__).parent.parent.parent
    ocv_data = pd.read_csv(project_root / "data" / "processed" / "B0005_ocv_soc.csv")
    
    # Fit model
    model = OCVModel(method='polynomial', degree=6)
    model.fit(ocv_data['soc'].values, ocv_data['ocv'].values)
    
    results_dir = project_root / "results" / "plots"
    
    # Plot 1: OCV-SOC curve
    print("\n1. Plotting OCV-SOC curve...")
    plot_ocv_soc_curve(ocv_data, model,
                      save_path=results_dir / "step3_ocv_soc_curve.png")
    
    # Plot 2: OCV derivative
    print("\n2. Plotting OCV sensitivity...")
    plot_ocv_derivative(model,
                       save_path=results_dir / "step3_ocv_derivative.png")
    
    # Plot 3: Model comparison
    print("\n3. Comparing different models...")
    plot_model_comparison(ocv_data,
                         save_path=results_dir / "step3_model_comparison.png")
    
    # Plot 4: Residuals
    print("\n4. Plotting residuals...")
    plot_residuals(ocv_data, model,
                  save_path=results_dir / "step3_residuals.png")
    
    print("\n" + "="*60)
    print("✓ OCV Visualizations Complete")
    print("="*60)


if __name__ == "__main__":
    main()
