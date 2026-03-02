"""
EETM Parameter Identification
Step 2.4: Identify thermal parameters using experimental data

Identifies 4 thermal parameters:
- Rin: Core-to-surface thermal resistance (K/W)
- Rout: Surface-to-ambient thermal resistance (K/W)
- Cc: Core thermal capacitance (J/K)
- Cs: Surface thermal capacitance (J/K)

Method: Nonlinear least squares (TRF algorithm)
Objective: Minimize ||Ts_measured - Ts_model||²
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, differential_evolution
from pathlib import Path
import time

from eetm.model import EETM2ndOrder


class ThermalParameterIdentifier:
    """
    Identify EETM thermal parameters from experimental data.
    """
    
    def __init__(self, time, Q, Ts_measured, Tamb):
        """
        Initialize parameter identifier.
        
        Args:
            time: Time vector (s)
            Q: Heat generation vector (W)
            Ts_measured: Measured surface temperature (°C)
            Tamb: Ambient temperature vector (°C)
        """
        self.time = time
        self.Q = Q
        self.Ts_measured = Ts_measured
        self.Tamb = Tamb
        
        # Initial conditions from data
        self.Tc_init = Ts_measured[0]
        self.Ts_init = Ts_measured[0]
        
        # Cost function evaluation counter
        self.n_eval = 0
        
    def residual_function(self, params):
        """
        Residual function for least squares optimization.
        
        Args:
            params: [Rin, Rout, Cc, Cs]
            
        Returns:
            residuals: Ts_measured - Ts_model
        """
        Rin, Rout, Cc, Cs = params
        
        # Create EETM with these parameters
        eetm = EETM2ndOrder(Rin, Rout, Cc, Cs)
        
        # Simulate
        try:
            results = eetm.simulate(
                self.time, self.Q, self.Tamb,
                Tc_init=self.Tc_init,
                Ts_init=self.Ts_init
            )
            
            # Calculate residuals
            residuals = self.Ts_measured - results['Ts']
            
            self.n_eval += 1
            
            # Print progress every 10 evaluations
            if self.n_eval % 10 == 0:
                rmse = np.sqrt(np.mean(residuals**2))
                print(f"  Eval {self.n_eval}: RMSE = {rmse:.4f} °C | "
                      f"Rin={Rin:.3f}, Rout={Rout:.3f}, Cc={Cc:.1f}, Cs={Cs:.1f}")
            
            return residuals
            
        except Exception as e:
            # Return large residuals if simulation fails
            print(f"  Simulation failed: {e}")
            return np.full_like(self.Ts_measured, 1e6)
    
    def objective_function(self, params):
        """
        Objective function (sum of squared residuals).
        
        Args:
            params: [Rin, Rout, Cc, Cs]
            
        Returns:
            cost: Sum of squared residuals
        """
        residuals = self.residual_function(params)
        return np.sum(residuals**2)
    
    def identify_parameters(self, method='trf', initial_guess=None, bounds=None):
        """
        Identify parameters using nonlinear least squares.
        
        Args:
            method: 'trf' (Trust Region Reflective) or 'lm' (Levenberg-Marquardt)
            initial_guess: Initial parameter guess [Rin, Rout, Cc, Cs]
            bounds: Parameter bounds (lower, upper)
            
        Returns:
            Dictionary with identified parameters and statistics
        """
        # Default initial guess (typical 18650 values)
        if initial_guess is None:
            initial_guess = [3.0, 15.0, 30.0, 15.0]  # [Rin, Rout, Cc, Cs]
        
        # Default bounds
        if bounds is None:
            # [Rin, Rout, Cc, Cs]
            lower = [0.1,  1.0,  5.0,  1.0]   # Lower bounds
            upper = [20.0, 50.0, 100.0, 50.0] # Upper bounds
            bounds = (lower, upper)
        
        print(f"\n{'='*70}")
        print("THERMAL PARAMETER IDENTIFICATION")
        print(f"{'='*70}")
        print(f"\nMethod: {method.upper()}")
        print(f"Initial guess: Rin={initial_guess[0]:.2f}, Rout={initial_guess[1]:.2f}, "
              f"Cc={initial_guess[2]:.1f}, Cs={initial_guess[3]:.1f}")
        print(f"Bounds: Rin [{bounds[0][0]:.1f}, {bounds[1][0]:.1f}], "
              f"Rout [{bounds[0][1]:.1f}, {bounds[1][1]:.1f}], "
              f"Cc [{bounds[0][2]:.1f}, {bounds[1][2]:.1f}], "
              f"Cs [{bounds[0][3]:.1f}, {bounds[1][3]:.1f}]")
        print(f"\nData points: {len(self.time)}")
        print(f"Duration: {self.time[-1]:.1f} s ({self.time[-1]/60:.2f} min)")
        
        # Reset counter
        self.n_eval = 0
        start_time = time.time()
        
        print(f"\nStarting optimization...")
        print(f"{'-'*70}")
        
        # Optimize
        result = least_squares(
            self.residual_function,
            x0=initial_guess,
            bounds=bounds,
            method=method,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=1000,
            verbose=0
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"{'-'*70}")
        print(f"\n✓ Optimization complete in {elapsed_time:.2f} s ({self.n_eval} evaluations)")
        
        # Extract results
        Rin_opt, Rout_opt, Cc_opt, Cs_opt = result.x
        
        # Simulate with optimal parameters
        eetm_opt = EETM2ndOrder(Rin_opt, Rout_opt, Cc_opt, Cs_opt)
        results_opt = eetm_opt.simulate(
            self.time, self.Q, self.Tamb,
            Tc_init=self.Tc_init,
            Ts_init=self.Ts_init
        )
        
        # Calculate statistics
        residuals = self.Ts_measured - results_opt['Ts']
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        max_error = np.max(np.abs(residuals))
        
        # R² coefficient
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.Ts_measured - np.mean(self.Ts_measured))**2)
        r2 = 1 - ss_res / ss_tot
        
        print(f"\n{'='*70}")
        print("IDENTIFIED PARAMETERS")
        print(f"{'='*70}")
        print(f"Rin  = {Rin_opt:.4f} K/W  (core-to-surface resistance)")
        print(f"Rout = {Rout_opt:.4f} K/W  (surface-to-ambient resistance)")
        print(f"Cc   = {Cc_opt:.2f} J/K   (core capacitance)")
        print(f"Cs   = {Cs_opt:.2f} J/K   (surface capacitance)")
        print(f"\nTime constants:")
        print(f"τ_core    = Cc·Rin  = {Cc_opt*Rin_opt:.1f} s ({Cc_opt*Rin_opt/60:.2f} min)")
        print(f"τ_surface = Cs·Rout = {Cs_opt*Rout_opt:.1f} s ({Cs_opt*Rout_opt/60:.2f} min)")
        
        print(f"\n{'='*70}")
        print("FIT STATISTICS")
        print(f"{'='*70}")
        print(f"RMSE      = {rmse:.4f} °C")
        print(f"MAE       = {mae:.4f} °C")
        print(f"Max Error = {max_error:.4f} °C")
        print(f"R²        = {r2:.6f}")
        
        # Assess fit quality
        if rmse < 0.5:
            quality = "EXCELLENT"
        elif rmse < 1.0:
            quality = "VERY GOOD"
        elif rmse < 2.0:
            quality = "GOOD"
        else:
            quality = "ACCEPTABLE"
        
        print(f"\nFit Quality: {quality}")
        
        # Return results
        return {
            'Rin': Rin_opt,
            'Rout': Rout_opt,
            'Cc': Cc_opt,
            'Cs': Cs_opt,
            'tau_core': Cc_opt * Rin_opt,
            'tau_surface': Cs_opt * Rout_opt,
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'r2': r2,
            'quality': quality,
            'n_eval': self.n_eval,
            'elapsed_time': elapsed_time,
            'success': result.success,
            'message': result.message,
            'Ts_model': results_opt['Ts'],
            'Tc_model': results_opt['Tc'],
            'residuals': residuals
        }
    
    def identify_global(self, n_iterations=50):
        """
        Global optimization using differential evolution.
        More robust but slower than local least squares.
        
        Args:
            n_iterations: Number of iterations
            
        Returns:
            Dictionary with identified parameters and statistics
        """
        print(f"\n{'='*70}")
        print("GLOBAL PARAMETER IDENTIFICATION (Differential Evolution)")
        print(f"{'='*70}")
        
        # Bounds
        bounds = [
            (0.1, 20.0),   # Rin
            (1.0, 50.0),   # Rout
            (5.0, 100.0),  # Cc
            (1.0, 50.0)    # Cs
        ]
        
        print(f"\nIterations: {n_iterations}")
        print(f"Data points: {len(self.time)}")
        
        # Reset counter
        self.n_eval = 0
        start_time = time.time()
        
        print(f"\nStarting global optimization...")
        print(f"{'-'*70}")
        
        # Optimize
        result = differential_evolution(
            self.objective_function,
            bounds=bounds,
            maxiter=n_iterations,
            popsize=15,
            seed=42,
            workers=1,
            updating='deferred',
            disp=True
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"{'-'*70}")
        print(f"\n✓ Global optimization complete in {elapsed_time:.2f} s")
        
        # Extract and return results (similar to identify_parameters)
        Rin_opt, Rout_opt, Cc_opt, Cs_opt = result.x
        
        # Simulate with optimal parameters
        eetm_opt = EETM2ndOrder(Rin_opt, Rout_opt, Cc_opt, Cs_opt)
        results_opt = eetm_opt.simulate(
            self.time, self.Q, self.Tamb,
            Tc_init=self.Tc_init,
            Ts_init=self.Ts_init
        )
        
        # Calculate statistics
        residuals = self.Ts_measured - results_opt['Ts']
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        max_error = np.max(np.abs(residuals))
        
        # R² coefficient
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.Ts_measured - np.mean(self.Ts_measured))**2)
        r2 = 1 - ss_res / ss_tot
        
        print(f"\n{'='*70}")
        print("IDENTIFIED PARAMETERS (GLOBAL)")
        print(f"{'='*70}")
        print(f"Rin  = {Rin_opt:.4f} K/W")
        print(f"Rout = {Rout_opt:.4f} K/W")
        print(f"Cc   = {Cc_opt:.2f} J/K")
        print(f"Cs   = {Cs_opt:.2f} J/K")
        print(f"\nRMSE = {rmse:.4f} °C")
        print(f"R²   = {r2:.6f}")
        
        return {
            'Rin': Rin_opt,
            'Rout': Rout_opt,
            'Cc': Cc_opt,
            'Cs': Cs_opt,
            'tau_core': Cc_opt * Rin_opt,
            'tau_surface': Cs_opt * Rout_opt,
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'r2': r2,
            'n_eval': self.n_eval,
            'elapsed_time': elapsed_time,
            'success': result.success,
            'Ts_model': results_opt['Ts'],
            'Tc_model': results_opt['Tc'],
            'residuals': residuals
        }


def run_parameter_identification(verbose=True):
    """
    Run parameter identification on CALCE data.
    
    Returns:
        Dictionary with identified parameters and results
    """
    if verbose:
        print("\n" + "="*70)
        print("STEP 2.4: EETM PARAMETER IDENTIFICATION")
        print("="*70)
    
    # Load data
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "processed" / "calce_with_heat.csv"
    
    if verbose:
        print(f"\nLoading data from {data_path.name}...")
    
    df = pd.read_csv(data_path)
    
    if verbose:
        print(f"✓ Loaded {len(df)} samples")
    
    # Extract data
    time = df['time'].values
    Q = df['Q_total'].values
    Ts_measured = df['Ts'].values
    Tamb = df['Tamb'].values
    
    if verbose:
        print(f"\nData summary:")
        print(f"  Duration: {time[-1]:.1f} s ({time[-1]/60:.2f} min)")
        print(f"  Ts range: [{Ts_measured.min():.2f}, {Ts_measured.max():.2f}] °C")
        print(f"  Q range: [{Q.min():.4f}, {Q.max():.4f}] W")
    
    # Create identifier
    identifier = ThermalParameterIdentifier(time, Q, Ts_measured, Tamb)
    
    # Identify parameters using least squares
    params = identifier.identify_parameters(method='trf')
    
    # Save results
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save identified parameters
    params_df = pd.DataFrame({
        'parameter': ['Rin', 'Rout', 'Cc', 'Cs', 'tau_core', 'tau_surface'],
        'value': [params['Rin'], params['Rout'], params['Cc'], params['Cs'],
                  params['tau_core'], params['tau_surface']],
        'unit': ['K/W', 'K/W', 'J/K', 'J/K', 's', 's']
    })
    params_path = output_dir / "eetm_params.csv"
    params_df.to_csv(params_path, index=False)
    
    if verbose:
        print(f"\n✓ Parameters saved to {params_path}")
    
    # Save simulation results
    results_df = pd.DataFrame({
        'time': time,
        'Ts_measured': Ts_measured,
        'Ts_model': params['Ts_model'],
        'Tc_model': params['Tc_model'],
        'residual': params['residuals'],
        'Q': Q,
        'Tamb': Tamb
    })
    results_path = output_dir / "eetm_identification_results.csv"
    results_df.to_csv(results_path, index=False)
    
    if verbose:
        print(f"✓ Results saved to {results_path}")
        print(f"\n{'='*70}")
        print("✓ Step 2.4 Parameter Identification Complete")
        print(f"{'='*70}")
    
    return params


if __name__ == "__main__":
    # Run parameter identification
    params = run_parameter_identification(verbose=True)
