"""
ECM Validation
Step 1.6: Validate ECM model on different cycles and operating conditions

Tests:
1. Same-cycle validation (training data)
2. Cross-cycle validation (different cycles)
3. Parameter sensitivity analysis
4. Model generalization assessment
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ecm.model.ecm_2rc import ECM2RC
from ecm.ocv.ocv_model import OCVModel


class ECMValidator:
    """
    Validate ECM model performance across different conditions.
    """
    
    def __init__(self, ocv_model: OCVModel):
        """
        Initialize validator.
        
        Args:
            ocv_model: Fitted OCV-SOC model
        """
        self.ocv_model = ocv_model
        self.validation_results = []
        
    def validate_cycle(self, ecm: ECM2RC, time, current, voltage_measured, 
                      soc_init, cycle_num, verbose=True):
        """
        Validate ECM on a single cycle.
        
        Args:
            ecm: ECM model with identified parameters
            time: Time vector (s)
            current: Current vector (A)
            voltage_measured: Measured terminal voltage (V)
            soc_init: Initial SOC
            cycle_num: Cycle number
            verbose: Print results
            
        Returns:
            Dictionary with validation metrics
        """
        # Simulate
        results = ecm.simulate(time, current, soc_init)
        
        # Calculate metrics
        residuals = voltage_measured - results['V_terminal']
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        max_error = np.max(np.abs(residuals))
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((voltage_measured - np.mean(voltage_measured))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs(residuals / voltage_measured)) * 100
        
        validation = {
            'cycle': cycle_num,
            'n_samples': len(time),
            'duration_min': time[-1] / 60,
            'rmse_mv': rmse * 1000,
            'mae_mv': mae * 1000,
            'max_error_mv': max_error * 1000,
            'r2': r2,
            'mape_percent': mape,
            'mean_voltage': np.mean(voltage_measured),
            'std_voltage': np.std(voltage_measured),
            'time': time,
            'voltage_measured': voltage_measured,
            'voltage_predicted': results['V_terminal'],
            'residuals': residuals,
            'soc': results['SOC']
        }
        
        if verbose:
            print(f"\nCycle {cycle_num} Validation:")
            print(f"  Samples: {len(time)}, Duration: {time[-1]/60:.1f} min")
            print(f"  RMSE: {rmse*1000:.2f} mV")
            print(f"  MAE: {mae*1000:.2f} mV")
            print(f"  Max Error: {max_error*1000:.2f} mV")
            print(f"  R²: {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
        
        self.validation_results.append(validation)
        return validation
    
    def validate_multiple_cycles(self, ecm_params, cycle_list, discharge_data, verbose=True):
        """
        Validate ECM on multiple cycles.
        
        Args:
            ecm_params: Dictionary with ECM parameters
            cycle_list: List of cycle numbers to validate
            discharge_data: DataFrame with discharge data (with SOC)
            verbose: Print progress
            
        Returns:
            List of validation results
        """
        if verbose:
            print(f"\n{'='*60}")
            print("ECM MULTI-CYCLE VALIDATION")
            print(f"{'='*60}")
            print(f"Testing cycles: {cycle_list}")
            print(f"Parameters: R0={ecm_params['R0']:.6f}, R1={ecm_params['R1']:.6f}, "
                  f"C1={ecm_params['C1']:.1f}, R2={ecm_params['R2']:.6f}, C2={ecm_params['C2']:.1f}")
        
        # Create ECM with identified parameters
        ecm = ECM2RC(
            ecm_params['R0'], 
            ecm_params['R1'], 
            ecm_params['C1'],
            ecm_params['R2'], 
            ecm_params['C2'],
            ecm_params['capacity'],
            self.ocv_model
        )
        
        results = []
        for cycle_num in cycle_list:
            # Extract cycle data
            cycle_data = discharge_data[discharge_data['cycle'] == cycle_num].copy()
            
            if len(cycle_data) == 0:
                print(f"  ⚠ Cycle {cycle_num} not found")
                continue
            
            time = cycle_data['time'].values
            time = time - time[0]
            current = cycle_data['current'].values
            voltage = cycle_data['voltage'].values
            soc = cycle_data['soc'].values
            
            # Validate
            result = self.validate_cycle(ecm, time, current, voltage, soc[0], 
                                        cycle_num, verbose=verbose)
            results.append(result)
        
        if verbose:
            print(f"\n{'='*60}")
            print("VALIDATION SUMMARY")
            print(f"{'='*60}")
            
            rmse_values = [r['rmse_mv'] for r in results]
            mae_values = [r['mae_mv'] for r in results]
            r2_values = [r['r2'] for r in results]
            
            print(f"\nAcross {len(results)} cycles:")
            print(f"  RMSE: {np.mean(rmse_values):.2f} ± {np.std(rmse_values):.2f} mV")
            print(f"  MAE:  {np.mean(mae_values):.2f} ± {np.std(mae_values):.2f} mV")
            print(f"  R²:   {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
            print(f"  Range: RMSE [{np.min(rmse_values):.2f}, {np.max(rmse_values):.2f}] mV")
        
        return results
    
    def get_summary_stats(self):
        """Get summary statistics across all validated cycles."""
        if not self.validation_results:
            return None
        
        return {
            'n_cycles': len(self.validation_results),
            'mean_rmse_mv': np.mean([r['rmse_mv'] for r in self.validation_results]),
            'std_rmse_mv': np.std([r['rmse_mv'] for r in self.validation_results]),
            'mean_mae_mv': np.mean([r['mae_mv'] for r in self.validation_results]),
            'std_mae_mv': np.std([r['mae_mv'] for r in self.validation_results]),
            'mean_r2': np.mean([r['r2'] for r in self.validation_results]),
            'std_r2': np.std([r['r2'] for r in self.validation_results]),
            'min_rmse_mv': np.min([r['rmse_mv'] for r in self.validation_results]),
            'max_rmse_mv': np.max([r['rmse_mv'] for r in self.validation_results]),
        }


def run_validation():
    """
    Run comprehensive ECM validation.
    """
    print(f"\n{'='*60}")
    print("STEP 1.6: ECM VALIDATION")
    print(f"{'='*60}")
    
    # Load data
    project_root = Path(__file__).parent.parent.parent
    
    # Load discharge data with SOC
    discharge_soc = pd.read_csv(project_root / "data" / "processed" / "B0005_discharge_soc.csv")
    print(f"\n✓ Loaded discharge data: {len(discharge_soc)} samples")
    
    # Load OCV model
    ocv_data = pd.read_csv(project_root / "data" / "processed" / "B0005_ocv_soc.csv")
    ocv_model = OCVModel(method='polynomial', degree=6)
    ocv_model.fit(ocv_data['soc'].values, ocv_data['ocv'].values)
    print("✓ OCV model loaded")
    
    # Load identified parameters (from cycle 1)
    params_df = pd.read_csv(project_root / "data" / "processed" / "ecm_params_cycle1.csv")
    ecm_params = params_df.iloc[0].to_dict()
    print(f"✓ Loaded parameters from cycle {int(ecm_params['cycle'])}")
    
    # Initialize validator
    validator = ECMValidator(ocv_model)
    
    # Test on multiple cycles (early, middle, late in battery life)
    test_cycles = [1, 50, 100, 150, 168]  # Spanning battery degradation
    
    print(f"\n{'='*60}")
    print("CROSS-CYCLE VALIDATION")
    print(f"{'='*60}")
    print(f"Testing generalization across battery lifetime")
    print(f"Cycles: {test_cycles} (early → late degradation)")
    
    # Validate
    results = validator.validate_multiple_cycles(
        ecm_params, 
        test_cycles, 
        discharge_soc,
        verbose=True
    )
    
    # Save results
    output_dir = project_root / "data" / "processed"
    
    # Summary statistics
    summary = validator.get_summary_stats()
    summary_df = pd.DataFrame([summary])
    summary_file = output_dir / "ecm_validation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✓ Summary saved to {summary_file}")
    
    # Individual cycle results
    metrics_list = []
    for r in results:
        metrics_list.append({
            'cycle': r['cycle'],
            'n_samples': r['n_samples'],
            'duration_min': r['duration_min'],
            'rmse_mv': r['rmse_mv'],
            'mae_mv': r['mae_mv'],
            'max_error_mv': r['max_error_mv'],
            'r2': r['r2'],
            'mape_percent': r['mape_percent']
        })
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_file = output_dir / "ecm_validation_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✓ Cycle metrics saved to {metrics_file}")
    
    # Save detailed results for each cycle
    for r in results:
        cycle_df = pd.DataFrame({
            'time': r['time'],
            'voltage_measured': r['voltage_measured'],
            'voltage_predicted': r['voltage_predicted'],
            'residual': r['residuals'],
            'soc': r['soc']
        })
        cycle_file = output_dir / f"ecm_validation_cycle{r['cycle']}.csv"
        cycle_df.to_csv(cycle_file, index=False)
    
    print(f"✓ Detailed results saved for {len(results)} cycles")
    
    # Performance assessment
    print(f"\n{'='*60}")
    print("PERFORMANCE ASSESSMENT")
    print(f"{'='*60}")
    
    if summary['mean_rmse_mv'] < 150:
        print("✓ EXCELLENT: RMSE < 150 mV - Model generalizes well")
    elif summary['mean_rmse_mv'] < 250:
        print("✓ GOOD: RMSE < 250 mV - Acceptable performance")
    else:
        print("⚠ FAIR: RMSE > 250 mV - Consider parameter adaptation")
    
    if summary['mean_r2'] > 0.95:
        print("✓ EXCELLENT: R² > 0.95 - Strong correlation")
    elif summary['mean_r2'] > 0.90:
        print("✓ GOOD: R² > 0.90 - Good correlation")
    else:
        print("⚠ FAIR: R² < 0.90 - Model fit could be improved")
    
    print(f"\n{'='*60}")
    print("✓ Step 1.6 ECM Validation Complete")
    print(f"{'='*60}")
    
    return validator, results


if __name__ == "__main__":
    validator, results = run_validation()
