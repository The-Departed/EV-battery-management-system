"""
SOC (State of Charge) Estimator using Coulomb Counting
Step 1.2: Implement SOC estimation from current integration
"""

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from pathlib import Path
from typing import Tuple, Optional


class SOCEstimator:
    """
    State of Charge estimation using Coulomb counting (current integration).
    
    SOC(t) = SOC(0) - (1/C_nominal) * ∫ I(t) dt
    
    For discharge: I < 0, SOC decreases
    For charge: I > 0, SOC increases
    """
    
    def __init__(self, nominal_capacity: float = None):
        """
        Initialize SOC estimator.
        
        Args:
            nominal_capacity: Battery nominal capacity in Ah
                            If None, will be estimated from data
        """
        self.nominal_capacity = nominal_capacity
    
    def estimate_capacity_from_cycle(self, time: np.ndarray, 
                                     current: np.ndarray) -> float:
        """
        Estimate capacity from a single discharge cycle.
        Uses trapezoidal integration of current over time.
        
        Args:
            time: Time vector in seconds
            current: Current vector in amperes (negative for discharge)
            
        Returns:
            Estimated capacity in Ah
        """
        # Integrate current over time (in seconds)
        charge_As = np.trapz(np.abs(current), time)  # Amp-seconds
        
        # Convert to Amp-hours
        capacity_Ah = charge_As / 3600.0
        
        return capacity_Ah
    
    def compute_soc(self, time: np.ndarray, current: np.ndarray,
                   soc_init: float = 1.0, 
                   capacity: float = None) -> np.ndarray:
        """
        Compute SOC using Coulomb counting.
        
        SOC(t) = SOC_0 - (1/C_n) * ∫[0,t] I(τ) dτ
        
        Args:
            time: Time vector in seconds
            current: Current vector in amperes (negative for discharge)
            soc_init: Initial SOC (0-1 scale, default 1.0 = 100%)
            capacity: Nominal capacity in Ah (uses self.nominal_capacity if None)
            
        Returns:
            SOC array (0-1 scale)
        """
        if capacity is None:
            if self.nominal_capacity is None:
                raise ValueError("Capacity must be provided or set in constructor")
            capacity = self.nominal_capacity
        
        # Integrate current over time using cumulative trapezoidal
        # For discharge: current < 0, integrated charge < 0
        # For charge: current > 0, integrated charge > 0
        charge_integrated = cumulative_trapezoid(current, time, initial=0)  # Amp-seconds
        
        # Convert to Amp-hours
        charge_Ah = charge_integrated / 3600.0
        
        # Compute SOC
        # charge_Ah is negative during discharge, positive during charge
        # SOC = SOC_init + (charge / capacity)
        # For discharge: charge is negative, so SOC decreases
        soc = soc_init + (charge_Ah / capacity)
        
        # Clip SOC to valid range [0, 1]
        soc = np.clip(soc, 0.0, 1.0)
        
        return soc
    
    def compute_soc_for_cycle(self, cycle_df: pd.DataFrame,
                             capacity: float = None) -> pd.DataFrame:
        """
        Compute SOC for a single cycle DataFrame.
        
        Args:
            cycle_df: DataFrame with 'time' and 'current' columns
            capacity: Nominal capacity in Ah
            
        Returns:
            DataFrame with added 'soc' column
        """
        df = cycle_df.copy()
        
        # Estimate capacity from this cycle if not provided
        if capacity is None:
            capacity = self.estimate_capacity_from_cycle(
                df['time'].values, 
                df['current'].values
            )
            print(f"Estimated capacity from cycle: {capacity:.4f} Ah")
        
        # Compute SOC
        soc = self.compute_soc(
            time=df['time'].values,
            current=df['current'].values,
            soc_init=1.0,  # Assume fully charged at start
            capacity=capacity
        )
        
        df['soc'] = soc
        
        return df
    
    def validate_soc(self, soc_estimated: np.ndarray, 
                    soc_true: np.ndarray) -> dict:
        """
        Validate SOC estimation against true SOC.
        
        Args:
            soc_estimated: Estimated SOC values
            soc_true: True SOC values
            
        Returns:
            Dictionary with error metrics (RMSE, MAE, Max Error)
        """
        error = soc_estimated - soc_true
        
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        max_error = np.max(np.abs(error))
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'Max_Error': max_error,
            'Mean_Error': np.mean(error),
            'Std_Error': np.std(error)
        }


def process_discharge_data(battery_id: str = "B0005") -> pd.DataFrame:
    """
    Load discharge data and compute SOC for all cycles.
    
    Args:
        battery_id: Battery identifier
        
    Returns:
        DataFrame with SOC computed for all cycles
    """
    # Load processed discharge data
    data_path = Path(f"data/processed/{battery_id}_discharge.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Initialize SOC estimator
    estimator = SOCEstimator()
    
    # Process each cycle
    all_cycles = []
    
    print("\n" + "="*60)
    print("Computing SOC for all discharge cycles")
    print("="*60)
    
    for cycle_num in sorted(df['cycle'].unique()):
        cycle_df = df[df['cycle'] == cycle_num].copy().reset_index(drop=True)
        
        # Reset time to start from 0
        cycle_df.loc[:, 'time'] = cycle_df['time'] - cycle_df['time'].iloc[0]
        
        # Get capacity if available
        if 'capacity' in cycle_df.columns:
            capacity = pd.to_numeric(cycle_df['capacity'].iloc[0], errors='coerce')
        else:
            capacity = None
        
        # Compute SOC
        if capacity is not None and not np.isnan(capacity):
            cycle_df_with_soc = estimator.compute_soc_for_cycle(cycle_df, capacity)
        else:
            # Estimate capacity from cycle
            capacity_est = estimator.estimate_capacity_from_cycle(
                cycle_df['time'].values,
                cycle_df['current'].values
            )
            cycle_df_with_soc = estimator.compute_soc_for_cycle(cycle_df, capacity_est)
        
        all_cycles.append(cycle_df_with_soc)
        
        if cycle_num <= 5 or cycle_num % 20 == 0:
            print(f"Cycle {cycle_num:3d}: Capacity = {capacity:.4f} Ah, "
                  f"SOC range = {cycle_df_with_soc['soc'].min():.3f} - {cycle_df_with_soc['soc'].max():.3f}")
    
    # Concatenate all cycles
    df_with_soc = pd.concat(all_cycles, ignore_index=True)
    
    # Save to processed data
    output_path = Path(f"data/processed/{battery_id}_discharge_soc.csv")
    df_with_soc.to_csv(output_path, index=False)
    print(f"\nSaved SOC data to {output_path}")
    
    return df_with_soc


def main():
    """
    Main execution for Step 1.2: SOC Estimation
    """
    print("="*60)
    print("STEP 1.2: SOC ESTIMATION - Coulomb Counting")
    print("="*60)
    
    # Process discharge data
    df = process_discharge_data("B0005")
    
    print("\n" + "="*60)
    print("SOC STATISTICS")
    print("="*60)
    print(f"\nTotal samples: {len(df)}")
    print(f"Number of cycles: {df['cycle'].nunique()}")
    print(f"\nSOC range: {df['soc'].min():.4f} - {df['soc'].max():.4f}")
    print(f"Mean SOC: {df['soc'].mean():.4f}")
    print(f"Std SOC: {df['soc'].std():.4f}")
    
    # Show sample data
    print("\n" + "="*60)
    print("SAMPLE DATA (Cycle 1, first 10 points)")
    print("="*60)
    cycle1 = df[df['cycle'] == 1].head(10)
    print(cycle1[['time', 'voltage', 'current', 'soc']])
    
    print("\n" + "="*60)
    print("✓ Step 1.2 Complete")
    print("="*60)


if __name__ == "__main__":
    main()
