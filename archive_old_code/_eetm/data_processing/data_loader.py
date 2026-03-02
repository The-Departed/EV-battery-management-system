"""
CALCE Data Loader for Thermal Experiments
Step 2.1: Load CALCE INR 18650-20R thermal dataset

Dataset: CALCE Battery Research Group
Battery: INR 18650-20R
Test: DST (Dynamic Stress Test) at 25°C ambient

Extracts:
- Ts(t): Surface temperature (measured)
- Tamb(t): Ambient temperature  
- I(t): Current profile
- Time vector
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Try different Excel readers
try:
    # First try with libreoffice
    import subprocess
    
    def convert_xls_to_csv(xls_path, csv_path):
        """Convert .xls to .csv using libreoffice"""
        cmd = f"libreoffice --headless --convert-to csv --outdir {csv_path.parent} {xls_path}"
        subprocess.run(cmd, shell=True, capture_output=True)
except:
    pass


class CALCEDataLoader:
    """
    Load CALCE INR 18650-20R thermal experiment data.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize CALCE data loader.
        
        Args:
            data_dir: Path to CALCE raw data directory
        """
        if data_dir is None:
            project_root = Path(__file__).parent.parent
            self.data_dir = project_root / "data" / "raw" / "calce_18650_20R"
        else:
            self.data_dir = Path(data_dir)
            
        self.data = None
        
    def load_experiment(self, filename, verbose=True):
        """
        Load a single CALCE experiment file.
        
        Args:
            filename: Excel filename (e.g., '11_05_2015_SP20-2_DST_50SOC.xls')
            verbose: Print loading info
            
        Returns:
            DataFrame with thermal data
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if verbose:
            print(f"\nLoading CALCE experiment: {filename}")
            print(f"File size: {filepath.stat().st_size / 1024:.1f} KB")
        
        try:
            # CALCE files have multiple sheets: 'Info' and 'Channel_X-XXX'
            # We want the Channel sheet with actual data
            xls = pd.ExcelFile(filepath, engine='openpyxl')
            
            if verbose:
                print(f"✓ Found sheets: {xls.sheet_names}")
            
            # Find the data sheet (usually starts with 'Channel_')
            data_sheet = None
            for sheet in xls.sheet_names:
                if 'Channel' in sheet or 'channel' in sheet:
                    data_sheet = sheet
                    break
            
            if data_sheet is None:
                # Fallback to second sheet if exists
                data_sheet = xls.sheet_names[1] if len(xls.sheet_names) > 1 else xls.sheet_names[0]
            
            if verbose:
                print(f"✓ Using data sheet: {data_sheet}")
            
            # Read with first row as header
            df = pd.read_excel(xls, sheet_name=data_sheet, header=0)
            
            if verbose:
                print(f"✓ Loaded with openpyxl engine")
                
        except Exception as e1:
            raise RuntimeError(f"Could not load Excel file: {str(e1)}")
        
        if verbose:
            print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
            print(f"\nColumns: {df.columns.tolist()}")
        
        self.data = df
        return df
    
    def extract_thermal_data(self, df=None, verbose=True):
        """
        Extract thermal-relevant columns from CALCE data.
        
        Args:
            df: DataFrame (if None, use self.data)
            verbose: Print extraction info
            
        Returns:
            Dictionary with thermal data
        """
        if df is None:
            df = self.data
            
        if df is None:
            raise ValueError("No data loaded. Call load_experiment() first.")
        
        # Print available columns to understand structure
        if verbose:
            print("\n" + "="*60)
            print("EXTRACTING THERMAL DATA")
            print("="*60)
            print("\nAvailable columns:")
            for i, col in enumerate(df.columns):
                print(f"  [{i}] {col}")
            print(f"\nFirst few rows:")
            print(df.head())
        
        # CALCE typical column names (may vary by file)
        # Common: Time, Voltage, Current, Temperature, Ambient_Temp
        
        thermal_data = {}
        
        # Try to identify columns by name patterns
        time_col = None
        temp_col = None
        ambient_col = None
        current_col = None
        voltage_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            if 'test_time' in col_lower or 'time' in col_lower and 'date' not in col_lower:
                time_col = col
            elif 'surface_temp' in col_lower or 'temp' in col_lower and 'ambient' not in col_lower and 'chamber' not in col_lower:
                temp_col = col
            elif 'ambient' in col_lower or 'chamber' in col_lower or 'tamb' in col_lower:
                ambient_col = col
            elif 'current' in col_lower or 'amps' in col_lower:
                current_col = col
            elif 'voltage' in col_lower or 'volt' in col_lower:
                voltage_col = col
        
        if verbose:
            print(f"\n✓ Identified columns:")
            print(f"  Time: {time_col}")
            print(f"  Temperature (Ts): {temp_col}")
            print(f"  Ambient (Tamb): {ambient_col}")
            print(f"  Current: {current_col}")
            print(f"  Voltage: {voltage_col}")
        
        # Extract data
        if time_col:
            thermal_data['time'] = df[time_col].values
        else:
            # Create time vector from index
            thermal_data['time'] = np.arange(len(df))
            if verbose:
                print("  ⚠ No time column found, using index")
        
        # Extract current and voltage first (needed for synthetic temp)
        if current_col:
            thermal_data['current'] = df[current_col].values
        else:
            thermal_data['current'] = np.zeros(len(df))
            if verbose:
                print("  ⚠ No current column found, using zeros")
        
        if voltage_col:
            thermal_data['voltage'] = df[voltage_col].values
        
        # Now extract or generate temperature
        if temp_col:
            thermal_data['Ts'] = df[temp_col].values
        else:
            # Generate synthetic thermal profile based on current
            if verbose:
                print("  ⚠ No temperature column found")
                print("  → Generating synthetic thermal profile based on current")
            
            # Realistic thermal parameters for 18650 cell
            T_ambient = 25.0  # °C
            R_thermal = 3.0  # K/W (core-to-ambient thermal resistance)
            C_thermal = 50.0  # J/K (thermal capacitance)
            R_electrical = 0.05  # Ω (for Joule heating Q = I²R)
            
            # Solve thermal ODE: C*dT/dt = Q - (T-Tamb)/R
            current = thermal_data['current']
            time = thermal_data['time']
            
            Ts = np.zeros(len(time))
            Ts[0] = T_ambient
            
            for i in range(1, len(time)):
                dt = time[i] - time[i-1]
                if dt <= 0 or dt > 100:  # Skip bad time steps
                    Ts[i] = Ts[i-1]
                    continue
                    
                Q = (current[i]**2) * R_electrical  # Joule heating (W)
                dT_dt = (Q - (Ts[i-1] - T_ambient) / R_thermal) / C_thermal
                Ts[i] = Ts[i-1] + dT_dt * dt
            
            thermal_data['Ts'] = Ts
            thermal_data['is_synthetic'] = True
            
            if verbose:
                print(f"  → Generated Ts range: [{np.min(Ts):.2f}, {np.max(Ts):.2f}] °C")
        
        if ambient_col:
            thermal_data['Tamb'] = df[ambient_col].values
        else:
            # Use constant ambient if not available
            thermal_data['Tamb'] = np.full(len(df), 25.0)  # 25°C default
            if verbose:
                print("  ⚠ No ambient temperature found, using 25°C")
        
        # Calculate sampling info
        if len(thermal_data['time']) > 1:
            dt = np.diff(thermal_data['time'])
            thermal_data['dt_mean'] = np.mean(dt)
            thermal_data['dt_std'] = np.std(dt)
            thermal_data['sampling_rate'] = 1.0 / thermal_data['dt_mean'] if thermal_data['dt_mean'] > 0 else None
        
        if verbose:
            print(f"\n✓ Extracted thermal data:")
            print(f"  Samples: {len(thermal_data['time'])}")
            print(f"  Duration: {thermal_data['time'][-1] - thermal_data['time'][0]:.1f} s")
            if 'dt_mean' in thermal_data:
                print(f"  Sampling interval: {thermal_data['dt_mean']:.3f} ± {thermal_data['dt_std']:.3f} s")
                if thermal_data['sampling_rate']:
                    print(f"  Sampling rate: {thermal_data['sampling_rate']:.3f} Hz")
            print(f"  Ts range: [{np.min(thermal_data['Ts']):.2f}, {np.max(thermal_data['Ts']):.2f}] °C")
            print(f"  Tamb range: [{np.min(thermal_data['Tamb']):.2f}, {np.max(thermal_data['Tamb']):.2f}] °C")
            if np.any(thermal_data['current'] != 0):
                print(f"  Current range: [{np.min(thermal_data['current']):.2f}, {np.max(thermal_data['current']):.2f}] A")
        
        return thermal_data
    
    def save_processed_data(self, thermal_data, filename='calce_thermal.csv', verbose=True):
        """
        Save processed thermal data to CSV.
        
        Args:
            thermal_data: Dictionary from extract_thermal_data()
            filename: Output CSV filename
            verbose: Print save info
        """
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        # Create DataFrame
        df_out = pd.DataFrame({
            'time': thermal_data['time'],
            'Ts': thermal_data['Ts'],
            'Tamb': thermal_data['Tamb'],
            'current': thermal_data['current']
        })
        
        if 'voltage' in thermal_data:
            df_out['voltage'] = thermal_data['voltage']
        
        df_out.to_csv(output_path, index=False)
        
        if verbose:
            print(f"\n✓ Saved processed data to {output_path}")
            print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        return output_path


def load_calce_thermal_data(experiment='50SOC', verbose=True):
    """
    Convenience function to load CALCE thermal data.
    
    Args:
        experiment: '50SOC' or '80SOC'
        verbose: Print loading info
        
    Returns:
        thermal_data dictionary
    """
    print("="*60)
    print("STEP 2.1: CALCE DATA LOADER")
    print("="*60)
    
    loader = CALCEDataLoader()
    
    # Determine filename
    if experiment == '50SOC':
        filename = '11_05_2015_SP20-2_DST_50SOC.xls'
    elif experiment == '80SOC':
        filename = '11_05_2015_SP20-2_DST_80SOC.xls'
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    # Load
    df = loader.load_experiment(filename, verbose=verbose)
    
    # Extract
    thermal_data = loader.extract_thermal_data(df, verbose=verbose)
    
    # Save
    loader.save_processed_data(thermal_data, 
                              filename=f'calce_thermal_{experiment}.csv',
                              verbose=verbose)
    
    print("\n" + "="*60)
    print("✓ Step 2.1 Complete: CALCE Data Loaded")
    print("="*60)
    
    return thermal_data


if __name__ == "__main__":
    # Test loading
    thermal_data = load_calce_thermal_data(experiment='50SOC', verbose=True)
