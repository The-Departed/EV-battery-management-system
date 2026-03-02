"""
EETM Validation
Step 2.5: Validate identified thermal parameters

Validates the EETM model with identified parameters:
- Compare Ts_model vs Ts_measured
- Compute error metrics (RMSE, MAE, Max Error)
- Analyze residuals
- Check physical consistency
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from eetm.model import EETM2ndOrder


class EETMValidator:
    """
    Validate EETM model performance.
    """
    
    def __init__(self, params):
        """
        Initialize validator with identified parameters.
        
        Args:
            params: Dictionary with Rin, Rout, Cc, Cs
        """
        self.params = params
        self.eetm = EETM2ndOrder(
            params['Rin'],
            params['Rout'],
            params['Cc'],
            params['Cs']
        )
        
    def validate(self, time, Q, Ts_measured, Tamb, verbose=True):
        """
        Validate EETM model on given data.
        
        Args:
            time: Time vector (s)
            Q: Heat generation vector (W)
            Ts_measured: Measured surface temperature (°C)
            Tamb: Ambient temperature vector (°C)
            verbose: Print detailed results
            
        Returns:
            Dictionary with validation results
        """
        if verbose:
            print("\n" + "="*70)
            print("EETM MODEL VALIDATION")
            print("="*70)
            print(f"\nModel Parameters:")
            print(f"  Rin  = {self.params['Rin']:.4f} K/W")
            print(f"  Rout = {self.params['Rout']:.4f} K/W")
            print(f"  Cc   = {self.params['Cc']:.2f} J/K")
            print(f"  Cs   = {self.params['Cs']:.2f} J/K")
            print(f"\nValidation Data:")
            print(f"  Duration: {time[-1]:.1f} s ({time[-1]/60:.2f} min)")
            print(f"  Samples: {len(time)}")
            print(f"  Ts range: [{Ts_measured.min():.2f}, {Ts_measured.max():.2f}] °C")
            print(f"  Q range: [{Q.min():.4f}, {Q.max():.4f}] W")
        
        # Simulate with identified parameters
        if verbose:
            print(f"\nRunning EETM simulation...")
        
        results = self.eetm.simulate(
            time, Q, Tamb,
            Tc_init=Ts_measured[0],
            Ts_init=Ts_measured[0]
        )
        
        if verbose:
            print("✓ Simulation complete")
        
        # Calculate errors
        Ts_model = results['Ts']
        Tc_model = results['Tc']
        
        residuals = Ts_measured - Ts_model
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        max_error = np.max(np.abs(residuals))
        mean_error = np.mean(residuals)
        std_error = np.std(residuals)
        
        # R² coefficient
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((Ts_measured - np.mean(Ts_measured))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        # Normalized RMSE (percentage of temperature range)
        temp_range = Ts_measured.max() - Ts_measured.min()
        if temp_range > 0:
            nrmse = (rmse / temp_range) * 100
        else:
            nrmse = 0.0
        
        # Physical checks
        checks = {
            'Tc_gt_Ts': np.all(Tc_model >= Ts_model - 0.01),  # Allow small numerical error
            'Ts_gt_Tamb': np.all(Ts_model >= Tamb - 0.01),
            'temps_positive': np.all(Tc_model > 0) and np.all(Ts_model > 0),
            'no_nan': not (np.any(np.isnan(Tc_model)) or np.any(np.isnan(Ts_model))),
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print("VALIDATION METRICS")
            print(f"{'='*70}")
            print(f"\nAccuracy Metrics:")
            print(f"  RMSE          = {rmse:.4f} °C")
            print(f"  MAE           = {mae:.4f} °C")
            print(f"  Max Error     = {max_error:.4f} °C")
            print(f"  Mean Error    = {mean_error:.4f} °C")
            print(f"  Std Error     = {std_error:.4f} °C")
            print(f"  R²            = {r2:.6f}")
            print(f"  NRMSE         = {nrmse:.2f}%")
            
            print(f"\nTemperature Statistics:")
            print(f"  Tc range: [{Tc_model.min():.2f}, {Tc_model.max():.2f}] °C")
            print(f"  Ts range: [{Ts_model.min():.2f}, {Ts_model.max():.2f}] °C")
            print(f"  ΔT (Tc-Ts) range: [{(Tc_model-Ts_model).min():.4f}, "
                  f"{(Tc_model-Ts_model).max():.4f}] °C")
            
            print(f"\nPhysical Consistency Checks:")
            for check_name, passed in checks.items():
                status = "✓" if passed else "✗"
                print(f"  {status} {check_name}: {passed}")
            
            # Assess fit quality
            if rmse < 0.5:
                quality = "EXCELLENT"
                emoji = "🌟"
            elif rmse < 1.0:
                quality = "VERY GOOD"
                emoji = "✓"
            elif rmse < 2.0:
                quality = "GOOD"
                emoji = "✓"
            elif rmse < 5.0:
                quality = "ACCEPTABLE"
                emoji = "~"
            else:
                quality = "POOR"
                emoji = "✗"
            
            print(f"\n{'='*70}")
            print(f"{emoji} MODEL QUALITY: {quality}")
            print(f"{'='*70}")
        
        return {
            'time': time,
            'Ts_measured': Ts_measured,
            'Ts_model': Ts_model,
            'Tc_model': Tc_model,
            'Tamb': Tamb,
            'Q': Q,
            'residuals': residuals,
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'mean_error': mean_error,
            'std_error': std_error,
            'r2': r2,
            'nrmse': nrmse,
            'quality': quality,
            'physical_checks': checks,
            'all_checks_passed': all(checks.values())
        }
    
    def cross_validate(self, datasets, verbose=True):
        """
        Cross-validate on multiple datasets.
        
        Args:
            datasets: List of (time, Q, Ts, Tamb) tuples
            verbose: Print results
            
        Returns:
            List of validation results for each dataset
        """
        if verbose:
            print("\n" + "="*70)
            print("CROSS-VALIDATION")
            print("="*70)
            print(f"\nNumber of datasets: {len(datasets)}")
        
        results = []
        for i, (time, Q, Ts, Tamb) in enumerate(datasets):
            if verbose:
                print(f"\n--- Dataset {i+1}/{len(datasets)} ---")
            
            result = self.validate(time, Q, Ts, Tamb, verbose=False)
            results.append(result)
            
            if verbose:
                print(f"  RMSE = {result['rmse']:.4f} °C")
                print(f"  MAE  = {result['mae']:.4f} °C")
                print(f"  Quality: {result['quality']}")
        
        # Aggregate statistics
        if verbose:
            avg_rmse = np.mean([r['rmse'] for r in results])
            avg_mae = np.mean([r['mae'] for r in results])
            avg_r2 = np.mean([r['r2'] for r in results])
            
            print(f"\n{'='*70}")
            print("CROSS-VALIDATION SUMMARY")
            print(f"{'='*70}")
            print(f"Average RMSE: {avg_rmse:.4f} °C")
            print(f"Average MAE:  {avg_mae:.4f} °C")
            print(f"Average R²:   {avg_r2:.6f}")
        
        return results


def run_validation(verbose=True):
    """
    Run EETM validation using identified parameters.
    
    Returns:
        Dictionary with validation results
    """
    if verbose:
        print("\n" + "="*70)
        print("STEP 2.5: EETM MODEL VALIDATION")
        print("="*70)
    
    project_root = Path(__file__).parent.parent
    
    # Load identified parameters
    params_path = project_root / "data" / "processed" / "eetm_params.csv"
    
    if verbose:
        print(f"\nLoading parameters from {params_path.name}...")
    
    params_df = pd.read_csv(params_path)
    params = {
        row['parameter']: row['value'] 
        for _, row in params_df.iterrows()
    }
    
    if verbose:
        print(f"✓ Loaded parameters: Rin={params['Rin']:.3f}, Rout={params['Rout']:.3f}, "
              f"Cc={params['Cc']:.1f}, Cs={params['Cs']:.1f}")
    
    # Load validation data (use same data as training for now)
    data_path = project_root / "data" / "processed" / "calce_with_heat.csv"
    
    if verbose:
        print(f"\nLoading validation data from {data_path.name}...")
    
    df = pd.read_csv(data_path)
    
    if verbose:
        print(f"✓ Loaded {len(df)} samples")
    
    # Extract data
    time = df['time'].values
    Q = df['Q_total'].values
    Ts_measured = df['Ts'].values
    Tamb = df['Tamb'].values
    
    # Create validator
    validator = EETMValidator(params)
    
    # Validate
    results = validator.validate(time, Q, Ts_measured, Tamb, verbose=verbose)
    
    # Save validation results
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validation_df = pd.DataFrame({
        'time': results['time'],
        'Ts_measured': results['Ts_measured'],
        'Ts_model': results['Ts_model'],
        'Tc_model': results['Tc_model'],
        'Tamb': results['Tamb'],
        'Q': results['Q'],
        'residual': results['residuals']
    })
    
    output_path = output_dir / "eetm_validation_results.csv"
    validation_df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"\n✓ Validation results saved to {output_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'metric': ['RMSE', 'MAE', 'Max_Error', 'Mean_Error', 'Std_Error', 
                   'R2', 'NRMSE', 'Quality'],
        'value': [results['rmse'], results['mae'], results['max_error'],
                  results['mean_error'], results['std_error'], results['r2'],
                  results['nrmse'], results['quality']],
        'unit': ['°C', '°C', '°C', '°C', '°C', '-', '%', '-']
    })
    
    metrics_path = output_dir / "eetm_validation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    if verbose:
        print(f"✓ Validation metrics saved to {metrics_path}")
        print(f"\n{'='*70}")
        print("✓ Step 2.5 EETM Validation Complete")
        print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
    # Run validation
    results = run_validation(verbose=True)
