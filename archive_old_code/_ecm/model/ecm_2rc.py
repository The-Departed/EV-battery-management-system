"""
2-RC Thevenin ECM (Electrical Circuit Model)
Step 1.4: Implement 2-RC equivalent circuit model

Circuit Structure:
                    R0
    V_terminal ─────┬──────┬──── OCV(SOC)
                    │      │
                   ┌┴┐    ┌┴┐
                R1 │ │  R2│ │
                   └┬┘    └┬┘
                    │      │
                   ─┴─    ─┴─  C1    C2
                    │      │
                   GND    GND

Equations:
    V_terminal = OCV(SOC) - V1 - V2 - I*R0
    dV1/dt = -V1/(R1*C1) + I/C1
    dV2/dt = -V2/(R2*C2) + I/C2
    dSOC/dt = -I/Capacity
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ecm.ocv.ocv_model import OCVModel


class ECM2RC:
    """
    2-RC Thevenin Equivalent Circuit Model for Li-ion battery.
    
    State variables:
    - V1: Voltage across RC pair 1 (fast dynamics)
    - V2: Voltage across RC pair 2 (slow dynamics)
    - SOC: State of Charge
    
    Parameters:
    - R0: Ohmic resistance (Ω)
    - R1, C1: RC pair 1 (SEI layer)
    - R2, C2: RC pair 2 (diffusion)
    - Capacity: Battery capacity (Ah)
    - OCV model: OCV-SOC relationship
    """
    
    def __init__(self, R0, R1, C1, R2, C2, capacity, ocv_model: OCVModel):
        """
        Initialize 2-RC ECM.
        
        Args:
            R0: Ohmic resistance (Ω)
            R1: Resistance of RC pair 1 (Ω)
            C1: Capacitance of RC pair 1 (F)
            R2: Resistance of RC pair 2 (Ω)
            C2: Capacitance of RC pair 2 (F)
            capacity: Battery capacity (Ah)
            ocv_model: Fitted OCV-SOC model
        """
        self.R0 = R0
        self.R1 = R1
        self.C1 = C1
        self.R2 = R2
        self.C2 = C2
        self.capacity = capacity
        self.ocv_model = ocv_model
        
        # Time constants
        self.tau1 = R1 * C1  # Fast time constant
        self.tau2 = R2 * C2  # Slow time constant
        
    def get_ocv(self, soc):
        """Get OCV for given SOC (with bounds)."""
        # Clip SOC to valid range [0, 1]
        soc_clipped = np.clip(soc, 0.0, 1.0)
        return self.ocv_model.predict(np.array([soc_clipped]))[0]
    
    def state_equations(self, t, state, current_func):
        """
        State space equations for ECM.
        
        State vector: [V1, V2, SOC]
        
        Args:
            t: Time (s)
            state: State vector [V1, V2, SOC]
            current_func: Function that returns current at time t
            
        Returns:
            derivatives: [dV1/dt, dV2/dt, dSOC/dt]
        """
        V1, V2, SOC = state
        
        # Get current at time t
        I = current_func(t)
        
        # State equations
        dV1_dt = -V1 / self.tau1 + I / self.C1
        dV2_dt = -V2 / self.tau2 + I / self.C2
        dSOC_dt = I / (self.capacity * 3600)  # I is negative for discharge, SOC decreases
        
        return [dV1_dt, dV2_dt, dSOC_dt]
    
    def terminal_voltage(self, V1, V2, SOC, I):
        """
        Calculate terminal voltage.
        
        V_terminal = OCV(SOC) - V1 - V2 - I*R0
        
        Args:
            V1: Voltage across RC1 (V)
            V2: Voltage across RC2 (V)
            SOC: State of charge (0-1)
            I: Current (A, positive for discharge)
            
        Returns:
            Terminal voltage (V)
        """
        OCV = self.get_ocv(SOC)
        V_terminal = OCV - V1 - V2 - I * self.R0
        return V_terminal
    
    def simulate(self, time, current, soc_init=1.0):
        """
        Simulate ECM response to current profile.
        
        Args:
            time: Time vector (s)
            current: Current vector (A, negative for discharge)
            soc_init: Initial SOC (0-1)
            
        Returns:
            Dictionary with simulation results
        """
        # Create current interpolation function
        from scipy.interpolate import interp1d
        current_func = interp1d(time, current, kind='linear', 
                               fill_value='extrapolate')
        
        # Initial state: [V1, V2, SOC]
        initial_state = [0.0, 0.0, soc_init]
        
        # Solve ODE
        solution = solve_ivp(
            fun=lambda t, y: self.state_equations(t, y, current_func),
            t_span=(time[0], time[-1]),
            y0=initial_state,
            t_eval=time,
            method='RK45'
        )
        
        # Extract states
        V1 = solution.y[0]
        V2 = solution.y[1]
        SOC = np.clip(solution.y[2], 0.0, 1.0)  # Ensure SOC stays in [0, 1]
        
        # Calculate terminal voltage
        V_terminal = np.zeros_like(time)
        OCV = np.zeros_like(time)
        
        for i in range(len(time)):
            I_curr = current[i]
            OCV[i] = self.get_ocv(SOC[i])
            V_terminal[i] = self.terminal_voltage(V1[i], V2[i], SOC[i], I_curr)
        
        return {
            'time': time,
            'V_terminal': V_terminal,
            'V1': V1,
            'V2': V2,
            'SOC': SOC,
            'OCV': OCV,
            'current': current,
            'V_R0': current * self.R0
        }
    
    def get_params(self):
        """Return model parameters as dictionary."""
        return {
            'R0': self.R0,
            'R1': self.R1,
            'C1': self.C1,
            'R2': self.R2,
            'C2': self.C2,
            'tau1': self.tau1,
            'tau2': self.tau2,
            'capacity': self.capacity
        }


def test_ecm_model():
    """
    Test ECM model with sample parameters.
    """
    print("="*60)
    print("STEP 1.4: 2-RC THEVENIN ECM - IMPLEMENTATION TEST")
    print("="*60)
    
    # Load OCV model
    project_root = Path(__file__).parent.parent.parent
    ocv_data = pd.read_csv(project_root / "data" / "processed" / "B0005_ocv_soc.csv")
    
    ocv_model = OCVModel(method='polynomial', degree=6)
    ocv_model.fit(ocv_data['soc'].values, ocv_data['ocv'].values)
    
    print("\n✓ OCV model loaded")
    
    # Initialize ECM with sample parameters
    ecm = ECM2RC(
        R0=0.03,      # Ohmic resistance (Ω)
        R1=0.01,      # RC1 resistance (Ω)
        C1=2000.0,    # RC1 capacitance (F)
        R2=0.03,      # RC2 resistance (Ω)
        C2=20000.0,   # RC2 capacitance (F)
        capacity=1.856,  # Initial capacity (Ah)
        ocv_model=ocv_model
    )
    
    print("\n✓ ECM initialized with sample parameters:")
    params = ecm.get_params()
    for key, value in params.items():
        if 'tau' in key:
            print(f"  {key}: {value:.1f} s")
        elif 'C' in key and key != 'capacity':
            print(f"  {key}: {value:.0f} F")
        else:
            print(f"  {key}: {value:.4f}")
    
    # Create test current profile (constant 2A discharge)
    time = np.linspace(0, 3600, 361)  # 1 hour, 10s resolution
    current = -2.0 * np.ones_like(time)  # 2A discharge (negative)
    
    print(f"\n✓ Test current profile: {abs(current[0]):.1f} A discharge for {time[-1]/60:.0f} min")
    
    # Simulate
    print("\nRunning simulation...")
    results = ecm.simulate(time, current, soc_init=1.0)
    
    print("\n✓ Simulation complete")
    
    # Display results
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"\nInitial conditions:")
    print(f"  SOC: {results['SOC'][0]*100:.1f}%")
    print(f"  OCV: {results['OCV'][0]:.3f} V")
    print(f"  V_terminal: {results['V_terminal'][0]:.3f} V")
    
    print(f"\nFinal conditions:")
    print(f"  SOC: {results['SOC'][-1]*100:.1f}%")
    print(f"  OCV: {results['OCV'][-1]:.3f} V")
    print(f"  V_terminal: {results['V_terminal'][-1]:.3f} V")
    
    print(f"\nVoltage drops:")
    print(f"  V_R0 (ohmic): {results['V_R0'][50]:.3f} V")
    print(f"  V1 (RC1): {results['V1'][50]:.3f} V")
    print(f"  V2 (RC2): {results['V2'][50]:.3f} V")
    print(f"  Total drop: {results['V_R0'][50] + results['V1'][50] + results['V2'][50]:.3f} V")
    
    # Save results for visualization
    results_df = pd.DataFrame({
        'time': results['time'],
        'V_terminal': results['V_terminal'],
        'V1': results['V1'],
        'V2': results['V2'],
        'SOC': results['SOC'],
        'OCV': results['OCV'],
        'current': results['current']
    })
    
    output_path = project_root / "data" / "processed" / "ecm_simulation_test.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    print("\n" + "="*60)
    print("✓ Step 1.4 Implementation Test Complete")
    print("="*60)
    
    return ecm, results


if __name__ == "__main__":
    ecm_model, simulation_results = test_ecm_model()
