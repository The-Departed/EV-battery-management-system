"""
Heat Generation Interface
Step 2.2: Compute heat generation Q(t) from ECM for EETM

Heat sources:
1. Joule heating: Q_joule = I²·R0 + I²·R1 + I²·R2
2. Reaction heat: Q_reaction = I·T·(dOCV/dT) [for now, simplified]

For Phase 2, we start with Joule heating only.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from ecm.model.ecm_2rc import ECM2RC
from ecm.ocv.ocv_model import OCVModel


class HeatGenerator:
    """
    Calculate battery heat generation from electrical model.
    
    Heat sources:
    - Joule heating (resistive losses)
    - Reaction/entropic heat (optional)
    """
    
    def __init__(self, ecm_params=None, include_reaction_heat=False):
        """
        Initialize heat generator.
        
        Args:
            ecm_params: Dictionary with R0, R1, R2, C1, C2, capacity
            include_reaction_heat: Include entropic heating (needs dOCV/dT)
        """
        self.ecm_params = ecm_params
        self.include_reaction_heat = include_reaction_heat
        
    def compute_joule_heat(self, current, R0=None, R1=None, R2=None, V1=None, V2=None):
        """
        Compute Joule heating: Q = I²R
        
        For 2-RC ECM:
        Q_joule = I²·R0 + I·V1 + I·V2
        
        Where:
        - I²·R0: Ohmic losses
        - I·V1: RC1 losses
        - I·V2: RC2 losses
        
        Args:
            current: Current vector (A)
            R0: Ohmic resistance (Ω)
            R1: RC1 resistance (Ω) [optional if V1 provided]
            R2: RC2 resistance (Ω) [optional if V2 provided]
            V1: Voltage across RC1 (V) [optional]
            V2: Voltage across RC2 (V) [optional]
            
        Returns:
            Heat generation (W)
        """
        if R0 is None and self.ecm_params:
            R0 = self.ecm_params.get('R0', 0.03)
        elif R0 is None:
            R0 = 0.03  # Default
        
        # Ohmic heating (always present)
        Q_ohmic = (current**2) * R0
        
        # RC losses (if V1, V2 available, use them; otherwise approximate)
        if V1 is not None and V2 is not None:
            # Exact: power dissipated in RC pairs
            Q_rc1 = current * V1
            Q_rc2 = current * V2
        elif R1 is not None and R2 is not None:
            # Approximate: assume steady-state V = I·R
            Q_rc1 = (current**2) * R1
            Q_rc2 = (current**2) * R2
        else:
            # No RC contribution
            Q_rc1 = 0
            Q_rc2 = 0
        
        Q_total = Q_ohmic + Q_rc1 + Q_rc2
        
        return Q_total
    
    def compute_total_heat(self, current, voltage=None, soc=None, temperature=None,
                          V1=None, V2=None):
        """
        Compute total heat generation.
        
        Q_total = Q_joule + Q_reaction
        
        Args:
            current: Current vector (A)
            voltage: Terminal voltage (V) [optional]
            soc: State of charge [0-1] [optional, for reaction heat]
            temperature: Cell temperature (K) [optional, for reaction heat]
            V1: Voltage across RC1 (V) [optional]
            V2: Voltage across RC2 (V) [optional]
            
        Returns:
            Heat generation (W)
        """
        # Joule heating
        if self.ecm_params:
            Q_joule = self.compute_joule_heat(
                current,
                R0=self.ecm_params['R0'],
                R1=self.ecm_params.get('R1'),
                R2=self.ecm_params.get('R2'),
                V1=V1,
                V2=V2
            )
        else:
            Q_joule = self.compute_joule_heat(current, V1=V1, V2=V2)
        
        # Reaction heat (entropic)
        if self.include_reaction_heat and soc is not None and temperature is not None:
            # Q_reaction = I * T * (dOCV/dSOC) * (dSOC/dT)
            # Simplified: Q_reaction ≈ I * T * dU_dT
            # For Li-ion, dU/dT ≈ -0.0002 V/K (typical)
            dU_dT = -0.0002  # V/K
            Q_reaction = current * temperature * dU_dT
        else:
            Q_reaction = 0
        
        Q_total = Q_joule + Q_reaction
        
        return Q_total
    
    def compute_heat_from_ecm(self, ecm_results, use_rc_voltages=True):
        """
        Compute heat from ECM simulation results.
        
        Args:
            ecm_results: Dictionary from ECM.simulate()
            use_rc_voltages: Use actual V1, V2 from simulation
            
        Returns:
            Heat generation vector (W)
        """
        current = ecm_results['current']
        
        if use_rc_voltages and 'V1' in ecm_results and 'V2' in ecm_results:
            V1 = ecm_results['V1']
            V2 = ecm_results['V2']
        else:
            V1 = None
            V2 = None
        
        soc = ecm_results.get('SOC', None)
        
        Q = self.compute_total_heat(
            current,
            soc=soc,
            V1=V1,
            V2=V2
        )
        
        return Q


def compute_heat_for_calce_data(calce_csv='calce_thermal_50SOC.csv',
                                ecm_params_file='ecm_params_cycle1.csv',
                                verbose=True):
    """
    Compute heat generation for CALCE thermal data using ECM parameters.
    
    Args:
        calce_csv: CALCE thermal data CSV
        ecm_params_file: ECM parameters CSV
        verbose: Print info
        
    Returns:
        DataFrame with heat generation added
    """
    if verbose:
        print("\n" + "="*60)
        print("STEP 2.2: HEAT INPUT INTERFACE")
        print("="*60)
    
    # Load CALCE data
    project_root = Path(__file__).parent.parent
    calce_path = project_root / "data" / "processed" / calce_csv
    
    if not calce_path.exists():
        raise FileNotFoundError(f"CALCE data not found: {calce_path}")
    
    df = pd.read_csv(calce_path)
    
    if verbose:
        print(f"\n✓ Loaded CALCE data: {len(df)} samples")
        print(f"  Current range: [{df['current'].min():.3f}, {df['current'].max():.3f}] A")
    
    # Load ECM parameters
    params_path = project_root / "data" / "processed" / ecm_params_file
    
    if params_path.exists():
        params_df = pd.read_csv(params_path)
        ecm_params = params_df.iloc[0].to_dict()
        
        if verbose:
            print(f"\n✓ Loaded ECM parameters:")
            print(f"  R0 = {ecm_params['R0']:.6f} Ω")
            print(f"  R1 = {ecm_params['R1']:.6f} Ω")
            print(f"  R2 = {ecm_params['R2']:.6f} Ω")
    else:
        # Use default parameters
        ecm_params = {
            'R0': 0.001127,
            'R1': 0.009899,
            'R2': 0.030116,
            'C1': 2000.05,
            'C2': 19999.50,
            'capacity': 1.856
        }
        if verbose:
            print(f"\n⚠ ECM parameters not found, using defaults")
            print(f"  R0 = {ecm_params['R0']:.6f} Ω")
    
    # Create heat generator
    heat_gen = HeatGenerator(ecm_params=ecm_params, include_reaction_heat=False)
    
    # Compute heat (Joule only for now)
    Q = heat_gen.compute_joule_heat(
        current=df['current'].values,
        R0=ecm_params['R0'],
        R1=ecm_params['R1'],
        R2=ecm_params['R2']
    )
    
    # Add to dataframe
    df['Q_joule'] = Q
    df['Q_total'] = Q  # For now, only Joule
    
    if verbose:
        print(f"\n✓ Computed heat generation:")
        print(f"  Q range: [{Q.min():.4f}, {Q.max():.4f}] W")
        print(f"  Q mean: {Q.mean():.4f} W")
        print(f"  Q_rms: {np.sqrt(np.mean(Q**2)):.4f} W")
        
        # Peak power during max discharge
        max_discharge_idx = np.argmin(df['current'])
        print(f"\n  Peak discharge current: {df['current'].iloc[max_discharge_idx]:.2f} A")
        print(f"  Heat at peak: {Q[max_discharge_idx]:.4f} W")
    
    # Save with heat
    output_path = project_root / "data" / "processed" / "calce_with_heat.csv"
    df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"\n✓ Saved data with heat to {output_path}")
        print(f"  Columns: {df.columns.tolist()}")
    
    print("\n" + "="*60)
    print("✓ Step 2.2 Complete: Heat Input Interface")
    print("="*60)
    
    return df


if __name__ == "__main__":
    # Test heat computation
    df = compute_heat_for_calce_data(verbose=True)
