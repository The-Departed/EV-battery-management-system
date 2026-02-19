"""
ECM Parameter Identification
Step 1.5: Identify R0, R1, C1, R2, C2 from measured data using optimization

Method: Nonlinear Least Squares
Objective: Minimize ||V_measured - V_model||^2
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares, differential_evolution
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ecm.model.ecm_2rc import ECM2RC
from ecm.ocv.ocv_model import OCVModel


class ECMParameterIdentifier:
    """
    Identify 2-RC ECM parameters from experimental data.
    
    Uses nonlinear least squares to find optimal:
    - R0: Ohmic resistance
    - R1, C1: Fast RC pair (SEI)
    - R2, C2: Slow RC pair (diffusion)
    """
    
    def __init__(self, ocv_model: OCVModel, capacity: float):
        """
        Initialize parameter identifier.
        
        Args:
            ocv_model: Fitted OCV-SOC model
            capacity: Battery capacity (Ah)
        """
        self.ocv_model = ocv_model
        self.capacity = capacity
        self.best_params = None
        self.best_ecm = None
        
    def objective_function(self, params, time, current, voltage_measured, soc_init):
        """
        Objective function for optimization.
        
        Args:
            params: [R0, R1, C1, R2, C2]
            time: Time vector
            current: Current vector
            voltage_measured: Measured terminal voltage
            soc_init: Initial SOC
            
        Returns:
            residuals: V_measured - V_model
        """
        R0, R1, C1, R2, C2 = params
        
        # Parameter bounds check (soft constraint via penalty)
        if R0 <= 0 or R1 <= 0 or C1 <= 0 or R2 <= 0 or C2 <= 0:
            return np.ones_like(voltage_measured) * 1e6
        
        try:
            # Create ECM with current parameters
            ecm = ECM2RC(R0, R1, C1, R2, C2, self.capacity, self.ocv_model)
            
            # Simulate
            results = ecm.simulate(time, current, soc_init)
            
            # Calculate residuals
            residuals = voltage_measured - results['V_terminal']
            
            return residuals
            
        except Exception as e:
            # If simulation fails, return large residuals
            return np.ones_like(voltage_measured) * 1e6
    
    def identify_parameters(self, time, current, voltage, soc_init=1.0, 
                           method='least_squares', verbose=True):
        """
        Identify ECM parameters from experimental data.
        
        Args:
            time: Time vector (s)
            current: Current vector (A)
            voltage: Measured terminal voltage (V)
            soc_init: Initial SOC
            method: 'least_squares' or 'differential_evolution'
            verbose: Print progress
            
        Returns:
            Dictionary with identified parameters and metrics
        """
        if verbose:
            print(f"\n{'='*60}")
            print("PARAMETER IDENTIFICATION")
            print(f"{'='*60}")
            print(f"Data points: {len(time)}")
            print(f"Time range: {time[0]:.1f} - {time[-1]:.1f} s ({time[-1]/60:.1f} min)")
            print(f"Voltage range: {voltage.min():.3f} - {voltage.max():.3f} V")
            print(f"Current: {current.mean():.3f} ± {current.std():.3f} A")
        
        # Initial guess (from battery specs)
        p0 = [0.03, 0.01, 2000.0, 0.03, 20000.0]
        
        # Parameter bounds
        bounds_lower = [0.001, 0.001, 100.0, 0.001, 1000.0]
        bounds_upper = [0.2, 0.2, 10000.0, 0.2, 100000.0]
        
        if verbose:
            print(f"\nInitial guess:")
            print(f"  R0={p0[0]:.4f} Ω, R1={p0[1]:.4f} Ω, C1={p0[2]:.1f} F")
            print(f"  R2={p0[3]:.4f} Ω, C2={p0[4]:.1f} F")
            print(f"\nOptimization method: {method}")
        
        if method == 'least_squares':
            # Nonlinear least squares
            result = least_squares(
                fun=self.objective_function,
                x0=p0,
                args=(time, current, voltage, soc_init),
                bounds=(bounds_lower, bounds_upper),
                method='trf',
                verbose=2 if verbose else 0,
                max_nfev=100
            )
            params_opt = result.x
            success = result.success
            
        elif method == 'differential_evolution':
            # Global optimization
            bounds = list(zip(bounds_lower, bounds_upper))
            
            def cost_func(params):
                residuals = self.objective_function(params, time, current, voltage, soc_init)
                return np.sum(residuals**2)
            
            result = differential_evolution(
                func=cost_func,
                bounds=bounds,
                maxiter=50,
                popsize=10,
                disp=verbose
            )
            params_opt = result.x
            success = result.success
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Extract optimal parameters
        R0_opt, R1_opt, C1_opt, R2_opt, C2_opt = params_opt
        
        # Create optimal ECM
        ecm_opt = ECM2RC(R0_opt, R1_opt, C1_opt, R2_opt, C2_opt, 
                        self.capacity, self.ocv_model)
        
        # Simulate with optimal parameters
        results_opt = ecm_opt.simulate(time, current, soc_init)
        
        # Calculate metrics
        residuals = voltage - results_opt['V_terminal']
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        max_error = np.max(np.abs(residuals))
        
        if verbose:
            print(f"\n{'='*60}")
            print("IDENTIFICATION RESULTS")
            print(f"{'='*60}")
            print(f"\nOptimal parameters:")
            print(f"  R0 = {R0_opt:.6f} Ω")
            print(f"  R1 = {R1_opt:.6f} Ω")
            print(f"  C1 = {C1_opt:.2f} F")
            print(f"  R2 = {R2_opt:.6f} Ω")
            print(f"  C2 = {C2_opt:.2f} F")
            print(f"\nTime constants:")
            print(f"  τ1 = R1×C1 = {R1_opt*C1_opt:.2f} s")
            print(f"  τ2 = R2×C2 = {R2_opt*C2_opt:.2f} s")
            print(f"\nModel performance:")
            print(f"  RMSE = {rmse*1000:.2f} mV")
            print(f"  MAE = {mae*1000:.2f} mV")
            print(f"  Max Error = {max_error*1000:.2f} mV")
            print(f"  Success: {success}")
        
        # Store best results
        self.best_params = {
            'R0': R0_opt,
            'R1': R1_opt,
            'C1': C1_opt,
            'R2': R2_opt,
            'C2': C2_opt,
            'tau1': R1_opt * C1_opt,
            'tau2': R2_opt * C2_opt,
            'capacity': self.capacity
        }
        self.best_ecm = ecm_opt
        
        return {
            'parameters': self.best_params,
            'ecm': ecm_opt,
            'simulation': results_opt,
            'metrics': {
                'rmse': rmse,
                'mae': mae,
                'max_error': max_error,
                'residuals': residuals
            },
            'optimization': {
                'success': success,
                'method': method
            }
        }


def identify_for_cycle(cycle_num=1, method='least_squares'):
    """
    Identify ECM parameters for a specific discharge cycle.
    
    Args:
        cycle_num: Cycle number to use for identification
        method: Optimization method
        
    Returns:
        Identification results dictionary
    """
    print(f"\n{'='*60}")
    print(f"STEP 1.5: ECM PARAMETER IDENTIFICATION - CYCLE {cycle_num}")
    print(f"{'='*60}")
    
    # Load data
    project_root = Path(__file__).parent.parent.parent
    
    # Load discharge data with SOC
    discharge_soc = pd.read_csv(project_root / "data" / "processed" / "B0005_discharge_soc.csv")
    
    # Filter for specific cycle
    cycle_data = discharge_soc[discharge_soc['cycle'] == cycle_num].copy()
    
    if len(cycle_data) == 0:
        raise ValueError(f"Cycle {cycle_num} not found in data")
    
    print(f"\n✓ Loaded cycle {cycle_num}: {len(cycle_data)} samples")
    
    # Extract relevant columns
    time = cycle_data['time'].values
    time = time - time[0]  # Start from 0
    current = cycle_data['current'].values
    voltage = cycle_data['voltage'].values
    soc = cycle_data['soc'].values
    capacity = cycle_data['capacity'].iloc[0]
    
    print(f"  Capacity: {capacity:.3f} Ah")
    print(f"  Duration: {time[-1]/60:.1f} minutes")
    print(f"  SOC range: {soc.min()*100:.1f}% - {soc.max()*100:.1f}%")
    
    # Load OCV model
    ocv_data = pd.read_csv(project_root / "data" / "processed" / "B0005_ocv_soc.csv")
    ocv_model = OCVModel(method='polynomial', degree=6)
    ocv_model.fit(ocv_data['soc'].values, ocv_data['ocv'].values)
    
    print("\n✓ OCV model loaded")
    
    # Initialize identifier
    identifier = ECMParameterIdentifier(ocv_model, capacity)
    
    # Identify parameters
    soc_init = soc[0]
    results = identifier.identify_parameters(
        time, current, voltage, soc_init, 
        method=method, verbose=True
    )
    
    # Save results
    output_dir = project_root / "data" / "processed"
    
    # Save parameters
    params_df = pd.DataFrame([results['parameters']])
    params_df['cycle'] = cycle_num
    params_file = output_dir / f"ecm_params_cycle{cycle_num}.csv"
    params_df.to_csv(params_file, index=False)
    print(f"\n✓ Parameters saved to {params_file}")
    
    # Save simulation results
    sim_df = pd.DataFrame({
        'time': results['simulation']['time'],
        'V_terminal_measured': voltage,
        'V_terminal_model': results['simulation']['V_terminal'],
        'V1': results['simulation']['V1'],
        'V2': results['simulation']['V2'],
        'SOC': results['simulation']['SOC'],
        'OCV': results['simulation']['OCV'],
        'current': results['simulation']['current'],
        'residual': results['metrics']['residuals']
    })
    sim_file = output_dir / f"ecm_identification_cycle{cycle_num}.csv"
    sim_df.to_csv(sim_file, index=False)
    print(f"✓ Simulation results saved to {sim_file}")
    
    print(f"\n{'='*60}")
    print("✓ Step 1.5 Parameter Identification Complete")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    # Identify parameters for cycle 1
    results = identify_for_cycle(cycle_num=1, method='least_squares')
