"""
Extended Kalman Filter for EETM
Step 2.6: Real-time estimation of core temperature Tc

State Vector: x = [Tc, Ts]
Measurement: y = Ts (only surface temperature measured)
Input: u = [Q, Tamb]

The EKF estimates the latent core temperature Tc using only
surface temperature measurements.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time as pytime


class EETMKalmanFilter:
    """
    Extended Kalman Filter for 2nd-order EETM.
    
    Estimates core temperature Tc from surface temperature Ts measurements.
    """
    
    def __init__(self, Rin, Rout, Cc, Cs, dt=1.0):
        """
        Initialize EKF for EETM.
        
        Args:
            Rin: Core-to-surface thermal resistance (K/W)
            Rout: Surface-to-ambient thermal resistance (K/W)
            Cc: Core thermal capacitance (J/K)
            Cs: Surface thermal capacitance (J/K)
            dt: Time step (s)
        """
        self.Rin = Rin
        self.Rout = Rout
        self.Cc = Cc
        self.Cs = Cs
        self.dt = dt
        
        # State dimension
        self.n_states = 2  # [Tc, Ts]
        self.n_measurements = 1  # [Ts]
        
        # State vector: [Tc, Ts]
        self.x = np.zeros(self.n_states)
        
        # State covariance matrix
        self.P = np.eye(self.n_states) * 1.0  # Initial uncertainty
        
        # Process noise covariance (model uncertainty)
        self.Q = np.array([
            [0.01, 0.0],   # Tc process noise
            [0.0, 0.01]    # Ts process noise
        ])
        
        # Measurement noise covariance (sensor noise)
        self.R = np.array([[0.1**2]])  # Ts measurement noise (±0.1°C)
        
        # Measurement matrix (we measure Ts only)
        self.H = np.array([[0.0, 1.0]])  # y = [0, 1] * [Tc, Ts]^T = Ts
        
    def state_transition(self, x, Q, Tamb):
        """
        Compute state derivative for EETM dynamics.
        
        dx/dt = f(x, u)
        
        Args:
            x: State vector [Tc, Ts]
            Q: Heat generation (W)
            Tamb: Ambient temperature (°C)
            
        Returns:
            x_next: Next state (using Euler integration)
        """
        Tc, Ts = x
        
        # EETM state equations
        dTc_dt = (Q - (Tc - Ts) / self.Rin) / self.Cc
        dTs_dt = ((Tc - Ts) / self.Rin - (Ts - Tamb) / self.Rout) / self.Cs
        
        # Euler integration
        Tc_next = Tc + dTc_dt * self.dt
        Ts_next = Ts + dTs_dt * self.dt
        
        return np.array([Tc_next, Ts_next])
    
    def compute_jacobian_F(self, x, Q, Tamb):
        """
        Compute Jacobian of state transition function.
        
        F = ∂f/∂x
        
        Args:
            x: State vector [Tc, Ts]
            Q: Heat generation (W)
            Tamb: Ambient temperature (°C)
            
        Returns:
            F: Jacobian matrix (2x2)
        """
        Tc, Ts = x
        
        # Partial derivatives of dTc/dt
        dTc_dTc = -1.0 / (self.Rin * self.Cc)
        dTc_dTs = 1.0 / (self.Rin * self.Cc)
        
        # Partial derivatives of dTs/dt
        dTs_dTc = 1.0 / (self.Rin * self.Cs)
        dTs_dTs = -1.0 / (self.Rin * self.Cs) - 1.0 / (self.Rout * self.Cs)
        
        # Jacobian of continuous-time system
        F_continuous = np.array([
            [dTc_dTc, dTc_dTs],
            [dTs_dTc, dTs_dTs]
        ])
        
        # Discrete-time Jacobian (first-order approximation)
        F = np.eye(self.n_states) + F_continuous * self.dt
        
        return F
    
    def predict(self, Q, Tamb):
        """
        Prediction step of Kalman filter.
        
        Args:
            Q: Heat generation (W)
            Tamb: Ambient temperature (°C)
        """
        # Predict state
        self.x = self.state_transition(self.x, Q, Tamb)
        
        # Predict covariance
        F = self.compute_jacobian_F(self.x, Q, Tamb)
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, measurement):
        """
        Update step of Kalman filter.
        
        Args:
            measurement: Measured Ts (°C)
        """
        # Measurement residual (innovation)
        y = np.array([measurement])
        y_pred = self.H @ self.x
        innovation = y - y_pred
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ innovation
        
        # Update covariance
        I = np.eye(self.n_states)
        self.P = (I - K @ self.H) @ self.P
        
        return innovation[0], S[0, 0]
    
    def get_state(self):
        """Get current state estimate [Tc, Ts]."""
        return self.x.copy()
    
    def get_covariance(self):
        """Get current state covariance matrix."""
        return self.P.copy()
    
    def get_uncertainty(self):
        """
        Get 1-sigma uncertainty for each state.
        
        Returns:
            [sigma_Tc, sigma_Ts]
        """
        return np.sqrt(np.diag(self.P))
    
    def initialize(self, Tc_init, Ts_init, P_init=None):
        """
        Initialize filter state.
        
        Args:
            Tc_init: Initial core temperature (°C)
            Ts_init: Initial surface temperature (°C)
            P_init: Initial covariance (2x2), default uses identity
        """
        self.x = np.array([Tc_init, Ts_init])
        
        if P_init is not None:
            self.P = P_init
        else:
            self.P = np.eye(self.n_states) * 1.0


def run_ekf_estimation(verbose=True):
    """
    Run EKF for core temperature estimation on CALCE data.
    
    Returns:
        Dictionary with EKF results
    """
    if verbose:
        print("\n" + "="*70)
        print("STEP 2.6: EXTENDED KALMAN FILTER FOR CORE TEMPERATURE")
        print("="*70)
    
    project_root = Path(__file__).parent.parent
    
    # Load EETM parameters
    params_path = project_root / "data" / "processed" / "eetm_params.csv"
    
    if verbose:
        print(f"\nLoading EETM parameters from {params_path.name}...")
    
    params_df = pd.read_csv(params_path)
    params = {row['parameter']: row['value'] for _, row in params_df.iterrows()}
    
    Rin = params['Rin']
    Rout = params['Rout']
    Cc = params['Cc']
    Cs = params['Cs']
    
    if verbose:
        print(f"✓ Parameters: Rin={Rin:.3f}, Rout={Rout:.3f}, Cc={Cc:.1f}, Cs={Cs:.1f}")
    
    # Load data
    data_path = project_root / "data" / "processed" / "calce_with_heat.csv"
    
    if verbose:
        print(f"\nLoading data from {data_path.name}...")
    
    df = pd.read_csv(data_path)
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    
    if verbose:
        print(f"✓ Loaded {len(df)} samples")
    
    # Extract data
    time = df['time'].values
    Q = df['Q_total'].values
    Ts_measured = df['Ts'].values
    Tamb = df['Tamb'].values
    
    # Compute dt (time step)
    dt = np.median(np.diff(time))
    
    if verbose:
        print(f"\nData info:")
        print(f"  Duration: {time[-1] - time[0]:.1f} s ({(time[-1]-time[0])/60:.2f} min)")
        print(f"  Time step: {dt:.3f} s")
        print(f"  Samples: {len(time)}")
    
    # Initialize EKF
    ekf = EETMKalmanFilter(Rin, Rout, Cc, Cs, dt=dt)
    
    # Initialize with first measurement
    ekf.initialize(Ts_measured[0], Ts_measured[0])
    
    if verbose:
        print(f"\n✓ EKF initialized")
        print(f"  Initial state: Tc={Ts_measured[0]:.2f}°C, Ts={Ts_measured[0]:.2f}°C")
        print(f"  Process noise: σ_Tc={np.sqrt(ekf.Q[0,0]):.3f}, σ_Ts={np.sqrt(ekf.Q[1,1]):.3f}")
        print(f"  Measurement noise: σ_Ts={np.sqrt(ekf.R[0,0]):.3f}°C")
    
    # Storage for results
    Tc_est = np.zeros(len(time))
    Ts_est = np.zeros(len(time))
    sigma_Tc = np.zeros(len(time))
    sigma_Ts = np.zeros(len(time))
    innovations = np.zeros(len(time))
    innovation_vars = np.zeros(len(time))
    
    if verbose:
        print(f"\nRunning EKF...")
        start_time = pytime.time()
    
    # Run EKF
    for i in range(len(time)):
        # Predict
        ekf.predict(Q[i], Tamb[i])
        
        # Update with measurement
        innovation, innov_var = ekf.update(Ts_measured[i])
        
        # Store results
        state = ekf.get_state()
        uncertainty = ekf.get_uncertainty()
        
        Tc_est[i] = state[0]
        Ts_est[i] = state[1]
        sigma_Tc[i] = uncertainty[0]
        sigma_Ts[i] = uncertainty[1]
        innovations[i] = innovation
        innovation_vars[i] = innov_var
        
        # Progress
        if verbose and (i % 1000 == 0 or i == len(time) - 1):
            print(f"  Step {i+1}/{len(time)}: Tc={state[0]:.3f}±{uncertainty[0]:.3f}°C, "
                  f"Ts={state[1]:.3f}±{uncertainty[1]:.3f}°C")
    
    if verbose:
        elapsed = pytime.time() - start_time
        print(f"\n✓ EKF complete in {elapsed:.2f} s ({len(time)/elapsed:.0f} samples/s)")
    
    # Compute statistics
    Tc_std = np.std(Tc_est)
    Ts_error = Ts_est - Ts_measured
    rmse_Ts = np.sqrt(np.mean(Ts_error**2))
    mae_Ts = np.mean(np.abs(Ts_error))
    
    # Innovation statistics (should be zero-mean if filter is consistent)
    innov_mean = np.mean(innovations)
    innov_std = np.std(innovations)
    
    if verbose:
        print(f"\n{'='*70}")
        print("EKF RESULTS")
        print(f"{'='*70}")
        print(f"\nCore Temperature Estimates:")
        print(f"  Tc range: [{Tc_est.min():.3f}, {Tc_est.max():.3f}] °C")
        print(f"  Tc mean: {Tc_est.mean():.3f} °C")
        print(f"  Tc std: {Tc_std:.3f} °C")
        print(f"  σ_Tc (final): {sigma_Tc[-1]:.3f} °C")
        
        print(f"\nSurface Temperature Tracking:")
        print(f"  RMSE(Ts): {rmse_Ts:.4f} °C")
        print(f"  MAE(Ts): {mae_Ts:.4f} °C")
        
        print(f"\nInnovation Statistics (zero-mean expected):")
        print(f"  Mean: {innov_mean:.4f} °C")
        print(f"  Std: {innov_std:.4f} °C")
        
        print(f"\nCore-Surface Gradient:")
        dT_core_surface = Tc_est - Ts_est
        print(f"  ΔT (Tc-Ts) range: [{dT_core_surface.min():.3f}, {dT_core_surface.max():.3f}] °C")
        print(f"  ΔT mean: {dT_core_surface.mean():.3f} °C")
    
    # Save results
    output_dir = project_root / "data" / "processed"
    
    results_df = pd.DataFrame({
        'time': time,
        'Ts_measured': Ts_measured,
        'Tc_estimated': Tc_est,
        'Ts_estimated': Ts_est,
        'sigma_Tc': sigma_Tc,
        'sigma_Ts': sigma_Ts,
        'innovation': innovations,
        'Q': Q,
        'Tamb': Tamb
    })
    
    results_path = output_dir / "ekf_results.csv"
    results_df.to_csv(results_path, index=False)
    
    if verbose:
        print(f"\n✓ Results saved to {results_path}")
    
    # Save summary statistics
    stats_df = pd.DataFrame({
        'metric': ['Tc_min', 'Tc_max', 'Tc_mean', 'Tc_std', 'sigma_Tc_final',
                   'RMSE_Ts', 'MAE_Ts', 'innov_mean', 'innov_std'],
        'value': [Tc_est.min(), Tc_est.max(), Tc_est.mean(), Tc_std, sigma_Tc[-1],
                  rmse_Ts, mae_Ts, innov_mean, innov_std],
        'unit': ['°C', '°C', '°C', '°C', '°C', '°C', '°C', '°C', '°C']
    })
    
    stats_path = output_dir / "ekf_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    
    if verbose:
        print(f"✓ Statistics saved to {stats_path}")
        print(f"\n{'='*70}")
        print("✓ Step 2.6 EKF Implementation Complete")
        print(f"{'='*70}")
    
    return {
        'time': time,
        'Tc_est': Tc_est,
        'Ts_est': Ts_est,
        'Ts_measured': Ts_measured,
        'sigma_Tc': sigma_Tc,
        'sigma_Ts': sigma_Ts,
        'innovations': innovations,
        'Q': Q,
        'Tamb': Tamb,
        'rmse_Ts': rmse_Ts,
        'mae_Ts': mae_Ts,
        'Tc_std': Tc_std
    }


if __name__ == "__main__":
    # Run EKF
    results = run_ekf_estimation(verbose=True)
