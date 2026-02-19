"""
OCV-SOC Curve Fitting
Step 1.3: Extract and fit Open Circuit Voltage vs State of Charge relationship

OCV is the terminal voltage when no current flows (at rest/equilibrium).
This is a fundamental battery characteristic used in ECM modeling.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import curve_fit
from pathlib import Path
from typing import Tuple, Callable
import warnings
warnings.filterwarnings('ignore')


class OCVModel:
    """
    OCV-SOC relationship model.
    
    Fits the relationship: OCV = f(SOC)
    
    Methods:
    - Polynomial fit
    - Spline interpolation
    - Lookup table
    """
    
    def __init__(self, method: str = 'polynomial', degree: int = 6):
        """
        Initialize OCV model.
        
        Args:
            method: 'polynomial', 'spline', or 'lookup'
            degree: Polynomial degree or spline smoothing parameter
        """
        self.method = method
        self.degree = degree
        self.model = None
        self.coefficients = None
        self.soc_data = None
        self.ocv_data = None
        
    def extract_ocv_from_relaxation(self, df: pd.DataFrame, 
                                    current_threshold: float = 0.01,
                                    min_rest_time: float = 60.0) -> pd.DataFrame:
        """
        Extract OCV points from relaxation periods (when current ≈ 0).
        
        Args:
            df: DataFrame with time, voltage, current, soc
            current_threshold: Max current to consider as "rest" (A)
            min_rest_time: Minimum rest duration (s)
            
        Returns:
            DataFrame with SOC and OCV points
        """
        # Find rest periods (low current)
        df['is_rest'] = np.abs(df['current']) < current_threshold
        
        # Group consecutive rest periods
        df['rest_group'] = (df['is_rest'] != df['is_rest'].shift()).cumsum()
        
        ocv_points = []
        
        for group_id in df[df['is_rest']]['rest_group'].unique():
            rest_segment = df[df['rest_group'] == group_id]
            
            # Check if rest period is long enough
            if len(rest_segment) < 2:
                continue
                
            rest_duration = rest_segment['time'].iloc[-1] - rest_segment['time'].iloc[0]
            
            if rest_duration >= min_rest_time:
                # Take voltage at end of rest period as OCV
                soc = rest_segment['soc'].iloc[-1]
                ocv = rest_segment['voltage'].iloc[-1]
                
                ocv_points.append({
                    'soc': soc,
                    'ocv': ocv,
                    'rest_duration': rest_duration
                })
        
        return pd.DataFrame(ocv_points)
    
    def extract_ocv_from_discharge(self, df: pd.DataFrame,
                                   voltage_filter: bool = True) -> pd.DataFrame:
        """
        Extract approximate OCV from discharge data.
        Uses voltage during discharge with compensation for IR drop.
        
        For better accuracy, we sample points at low current or
        use the beginning of discharge cycles.
        
        Args:
            df: DataFrame with voltage, current, soc
            voltage_filter: Apply filtering to reduce noise
            
        Returns:
            DataFrame with SOC and approximate OCV
        """
        # Sample uniformly across SOC range
        soc_bins = np.linspace(0, 1, 50)
        ocv_points = []
        
        for i in range(len(soc_bins) - 1):
            soc_min, soc_max = soc_bins[i], soc_bins[i+1]
            
            # Get data in this SOC bin
            mask = (df['soc'] >= soc_min) & (df['soc'] < soc_max)
            bin_data = df[mask]
            
            if len(bin_data) == 0:
                continue
            
            # Use median voltage in this bin
            # (or use voltage at lowest current magnitude)
            if len(bin_data) > 0:
                # Find point with minimum current magnitude in this bin
                min_current_idx = bin_data['current'].abs().idxmin()
                soc_val = bin_data.loc[min_current_idx, 'soc']
                voltage_val = bin_data.loc[min_current_idx, 'voltage']
                
                ocv_points.append({
                    'soc': soc_val,
                    'ocv': voltage_val
                })
        
        return pd.DataFrame(ocv_points)
    
    def fit(self, soc: np.ndarray, ocv: np.ndarray):
        """
        Fit OCV-SOC model.
        
        Args:
            soc: SOC values (0-1)
            ocv: OCV values (V)
        """
        # Store data
        self.soc_data = soc
        self.ocv_data = ocv
        
        # Sort by SOC
        sort_idx = np.argsort(soc)
        soc_sorted = soc[sort_idx]
        ocv_sorted = ocv[sort_idx]
        
        if self.method == 'polynomial':
            # Polynomial fit
            self.coefficients = np.polyfit(soc_sorted, ocv_sorted, self.degree)
            self.model = np.poly1d(self.coefficients)
            
        elif self.method == 'spline':
            # Spline interpolation
            self.model = UnivariateSpline(soc_sorted, ocv_sorted, 
                                         s=self.degree, k=3)
            
        elif self.method == 'lookup':
            # Linear interpolation (lookup table)
            self.model = interp1d(soc_sorted, ocv_sorted, 
                                 kind='linear', fill_value='extrapolate')
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def predict(self, soc: np.ndarray) -> np.ndarray:
        """
        Predict OCV for given SOC values.
        
        Args:
            soc: SOC values (0-1)
            
        Returns:
            OCV values (V)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model(soc)
    
    def evaluate(self, soc_test: np.ndarray, ocv_test: np.ndarray) -> dict:
        """
        Evaluate model performance.
        
        Args:
            soc_test: Test SOC values
            ocv_test: True OCV values
            
        Returns:
            Dictionary with error metrics
        """
        ocv_pred = self.predict(soc_test)
        
        error = ocv_pred - ocv_test
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


def process_ocv_soc_curve(battery_id: str = "B0005",
                          method: str = 'polynomial',
                          degree: int = 6) -> Tuple[OCVModel, pd.DataFrame]:
    """
    Extract OCV-SOC data and fit curve.
    
    Args:
        battery_id: Battery identifier
        method: Fitting method
        degree: Polynomial degree or smoothing parameter
        
    Returns:
        Tuple of (fitted model, OCV data points)
    """
    # Load SOC data
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "processed" / f"{battery_id}_discharge_soc.csv"
    df = pd.read_csv(data_path)
    
    print("\n" + "="*60)
    print("Extracting OCV-SOC Data")
    print("="*60)
    
    # Use early cycles (less degradation)
    early_cycles = df[df['cycle'] <= 5].copy()
    
    # Extract OCV points
    ocv_model = OCVModel(method=method, degree=degree)
    ocv_data = ocv_model.extract_ocv_from_discharge(early_cycles)
    
    print(f"Extracted {len(ocv_data)} OCV-SOC points")
    print(f"SOC range: {ocv_data['soc'].min():.3f} - {ocv_data['soc'].max():.3f}")
    print(f"OCV range: {ocv_data['ocv'].min():.3f} - {ocv_data['ocv'].max():.3f} V")
    
    # Fit model
    print(f"\nFitting {method} model (degree={degree})...")
    ocv_model.fit(ocv_data['soc'].values, ocv_data['ocv'].values)
    
    # Evaluate
    metrics = ocv_model.evaluate(ocv_data['soc'].values, ocv_data['ocv'].values)
    print(f"\nModel Performance:")
    print(f"  RMSE: {metrics['RMSE']*1000:.2f} mV")
    print(f"  MAE:  {metrics['MAE']*1000:.2f} mV")
    print(f"  Max Error: {metrics['Max_Error']*1000:.2f} mV")
    
    # Save OCV data
    output_path = project_root / "data" / "processed" / f"{battery_id}_ocv_soc.csv"
    ocv_data.to_csv(output_path, index=False)
    print(f"\nSaved OCV data to {output_path}")
    
    return ocv_model, ocv_data


def main():
    """
    Main execution for Step 1.3: OCV-SOC Curve Fitting
    """
    print("="*60)
    print("STEP 1.3: OCV-SOC CURVE FITTING")
    print("="*60)
    
    # Fit polynomial model
    model_poly, ocv_data = process_ocv_soc_curve("B0005", method='polynomial', degree=6)
    
    # Display sample data
    print("\n" + "="*60)
    print("SAMPLE OCV-SOC DATA")
    print("="*60)
    print(ocv_data.head(10))
    
    # Test predictions
    print("\n" + "="*60)
    print("OCV PREDICTIONS AT KEY SOC POINTS")
    print("="*60)
    test_socs = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    test_ocvs = model_poly.predict(test_socs)
    
    for soc, ocv in zip(test_socs, test_ocvs):
        print(f"SOC = {soc*100:5.1f}%  →  OCV = {ocv:.4f} V")
    
    print("\n" + "="*60)
    print("✓ Step 1.3 Complete")
    print("="*60)


if __name__ == "__main__":
    main()
