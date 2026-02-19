"""
2nd-Order Equivalent Electro-Thermal Model (EETM)
Step 2.3: Core-Surface Thermal Dynamics

States:
- Tc: Core temperature (°C)
- Ts: Surface temperature (°C)

Parameters:
- Rin: Core-to-surface thermal resistance (K/W)
- Rout: Surface-to-ambient thermal resistance (K/W)
- Cc: Core thermal capacitance (J/K)
- Cs: Surface thermal capacitance (J/K)

Dynamics:
Cc·dTc/dt = Q(t) - (Tc - Ts)/Rin
Cs·dTs/dt = (Tc - Ts)/Rin - (Ts - Tamb)/Rout
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from pathlib import Path


class EETM2ndOrder:
    """
    2nd-order Equivalent Electro-Thermal Model.
    
    Two-state lumped thermal model:
    - Core temperature Tc (not measured, latent state)
    - Surface temperature Ts (measured)
    
    Heat flow: Q → Core → Surface → Ambient
    """
    
    def __init__(self, Rin, Rout, Cc, Cs):
        """
        Initialize EETM.
        
        Args:
            Rin: Core-to-surface thermal resistance (K/W)
            Rout: Surface-to-ambient thermal resistance (K/W)
            Cc: Core thermal capacitance (J/K)
            Cs: Surface thermal capacitance (J/K)
        """
        self.Rin = Rin
        self.Rout = Rout
        self.Cc = Cc
        self.Cs = Cs
        
        # Time constants
        self.tau_c = Cc * Rin  # Core time constant (s)
        self.tau_s = Cs * Rout  # Surface time constant (s)
        
    def state_equations(self, t, state, Q_func, Tamb_func):
        """
        EETM state-space equations.
        
        State vector: [Tc, Ts]
        
        Args:
            t: Time (s)
            state: State vector [Tc, Ts] (°C)
            Q_func: Function that returns heat generation at time t (W)
            Tamb_func: Function that returns ambient temperature at time t (°C)
            
        Returns:
            derivatives: [dTc/dt, dTs/dt] (°C/s)
        """
        Tc, Ts = state
        
        # Get inputs at time t
        Q = Q_func(t)
        Tamb = Tamb_func(t)
        
        # State equations
        dTc_dt = (Q - (Tc - Ts) / self.Rin) / self.Cc
        dTs_dt = ((Tc - Ts) / self.Rin - (Ts - Tamb) / self.Rout) / self.Cs
        
        return [dTc_dt, dTs_dt]
    
    def simulate(self, time, Q, Tamb, Tc_init=None, Ts_init=None):
        """
        Simulate EETM thermal response.
        
        Args:
            time: Time vector (s)
            Q: Heat generation vector (W)
            Tamb: Ambient temperature vector (°C)
            Tc_init: Initial core temperature (°C), default = Tamb[0]
            Ts_init: Initial surface temperature (°C), default = Tamb[0]
            
        Returns:
            Dictionary with simulation results
        """
        # Sort time if needed
        if not np.all(time[:-1] <= time[1:]):
            sort_idx = np.argsort(time)
            time = time[sort_idx]
            Q = Q[sort_idx]
            Tamb = Tamb[sort_idx]
        else:
            sort_idx = None
        
        # Initial conditions
        if Tc_init is None:
            Tc_init = Tamb[0]
        if Ts_init is None:
            Ts_init = Tamb[0]
        
        initial_state = [Tc_init, Ts_init]
        
        # Create interpolation functions
        from scipy.interpolate import interp1d
        Q_func = interp1d(time, Q, kind='linear', fill_value='extrapolate')
        Tamb_func = interp1d(time, Tamb, kind='linear', fill_value='extrapolate')
        
        # Solve ODE
        solution = solve_ivp(
            fun=lambda t, y: self.state_equations(t, y, Q_func, Tamb_func),
            t_span=(time[0], time[-1]),
            y0=initial_state,
            t_eval=time,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )
        
        # Extract states
        Tc = solution.y[0]
        Ts = solution.y[1]
        
        # Calculate heat flows
        Q_core_to_surface = (Tc - Ts) / self.Rin
        Q_surface_to_ambient = (Ts - Tamb) / self.Rout
        
        return {
            'time': time,
            'Tc': Tc,
            'Ts': Ts,
            'Tamb': Tamb,
            'Q': Q,
            'Q_core_surface': Q_core_to_surface,
            'Q_surface_ambient': Q_surface_to_ambient,
            'dT_core_surface': Tc - Ts,
            'dT_surface_ambient': Ts - Tamb
        }
    
    def get_params(self):
        """Return model parameters as dictionary."""
        return {
            'Rin': self.Rin,
            'Rout': self.Rout,
            'Cc': self.Cc,
            'Cs': self.Cs,
            'tau_c': self.tau_c,
            'tau_s': self.tau_s
        }
    
    def steady_state_temperature(self, Q, Tamb):
        """
        Calculate steady-state temperatures for constant Q and Tamb.
        
        At steady state: dTc/dt = 0, dTs/dt = 0
        
        Args:
            Q: Heat generation (W)
            Tamb: Ambient temperature (°C)
            
        Returns:
            (Tc_ss, Ts_ss): Steady-state core and surface temperatures (°C)
        """
        # At steady state:
        # Q = (Tc - Ts)/Rin = (Ts - Tamb)/Rout
        
        # Total thermal resistance
        R_total = self.Rin + self.Rout
        
        # Steady-state temperatures
        Tc_ss = Tamb + Q * R_total
        Ts_ss = Tamb + Q * self.Rout
        
        return Tc_ss, Ts_ss


def test_eetm_model(Q_constant=0.5, Tamb=25.0, duration=600, verbose=True):
    """
    Test EETM model with constant heat input.
    
    Args:
        Q_constant: Constant heat generation (W)
        Tamb: Ambient temperature (°C)
        duration: Simulation duration (s)
        verbose: Print results
        
    Returns:
        EETM model and simulation results
    """
    if verbose:
        print("\n" + "="*60)
        print("STEP 2.3: EETM MODEL - TEST SIMULATION")
        print("="*60)
    
    # Initial parameter estimates (typical for 18650 cell)
    Rin = 3.0   # K/W (core-to-surface)
    Rout = 15.0  # K/W (surface-to-ambient)
    Cc = 30.0   # J/K (core capacitance)
    Cs = 15.0   # J/K (surface capacitance)
    
    if verbose:
        print(f"\n✓ EETM Parameters (initial estimates):")
        print(f"  Rin = {Rin:.2f} K/W")
        print(f"  Rout = {Rout:.2f} K/W")
        print(f"  Cc = {Cc:.2f} J/K")
        print(f"  Cs = {Cs:.2f} J/K")
        print(f"  τ_core = Cc·Rin = {Cc*Rin:.1f} s")
        print(f"  τ_surface = Cs·Rout = {Cs*Rout:.1f} s")
    
    # Create EETM
    eetm = EETM2ndOrder(Rin, Rout, Cc, Cs)
    
    # Test scenario: constant heat input
    time = np.linspace(0, duration, 601)  # 1s resolution
    Q = np.full_like(time, Q_constant)
    Tamb_vec = np.full_like(time, Tamb)
    
    if verbose:
        print(f"\n✓ Test scenario:")
        print(f"  Heat: {Q_constant:.2f} W (constant)")
        print(f"  Ambient: {Tamb:.2f} °C")
        print(f"  Duration: {duration:.0f} s ({duration/60:.1f} min)")
    
    # Simulate
    if verbose:
        print(f"\nRunning simulation...")
    
    results = eetm.simulate(time, Q, Tamb_vec, Tc_init=Tamb, Ts_init=Tamb)
    
    if verbose:
        print("✓ Simulation complete")
    
    # Calculate steady-state
    Tc_ss, Ts_ss = eetm.steady_state_temperature(Q_constant, Tamb)
    
    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"\nInitial conditions:")
        print(f"  Tc(0) = {results['Tc'][0]:.2f} °C")
        print(f"  Ts(0) = {results['Ts'][0]:.2f} °C")
        
        print(f"\nFinal conditions (t = {duration}s):")
        print(f"  Tc(end) = {results['Tc'][-1]:.2f} °C")
        print(f"  Ts(end) = {results['Ts'][-1]:.2f} °C")
        
        print(f"\nSteady-state (analytical):")
        print(f"  Tc_ss = {Tc_ss:.2f} °C")
        print(f"  Ts_ss = {Ts_ss:.2f} °C")
        
        print(f"\nTemperature rise:")
        print(f"  ΔTc = {results['Tc'][-1] - Tamb:.2f} °C")
        print(f"  ΔTs = {results['Ts'][-1] - Tamb:.2f} °C")
        print(f"  Tc - Ts = {results['Tc'][-1] - results['Ts'][-1]:.2f} °C")
    
    # Save results
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame({
        'time': results['time'],
        'Tc': results['Tc'],
        'Ts': results['Ts'],
        'Tamb': results['Tamb'],
        'Q': results['Q'],
        'Q_core_surface': results['Q_core_surface'],
        'Q_surface_ambient': results['Q_surface_ambient']
    })
    
    output_path = output_dir / "eetm_test_simulation.csv"
    results_df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"\n✓ Results saved to {output_path}")
        print(f"\n{'='*60}")
        print("✓ Step 2.3 EETM Model Test Complete")
        print(f"{'='*60}")
    
    return eetm, results


if __name__ == "__main__":
    # Test EETM model
    eetm, results = test_eetm_model(Q_constant=0.5, Tamb=25.0, duration=600, verbose=True)
