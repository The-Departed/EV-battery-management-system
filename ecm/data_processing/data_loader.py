"""
NASA Battery Dataset Loader
Loads and preprocesses NASA Li-ion Battery Dataset (CSV files)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class NASABatteryLoader:
    """
    Load NASA Battery Dataset and extract cycles.
    
    Dataset structure:
    - data/raw/cleaned_dataset/metadata.csv: cycle metadata
    - data/raw/cleaned_dataset/data/*.csv: individual cycle files
    
    Batteries available:
    - B0005, B0006, B0007, B0018, B0025, B0026, B0027, B0028, etc.
    
    Each cycle contains:
    - Voltage_measured (V)
    - Current_measured (A)
    - Temperature_measured (°C)
    - Time (s)
    """
    
    def __init__(self, data_dir: str = "data/raw/cleaned_dataset"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Load metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load and parse metadata.csv"""
        metadata_path = self.data_dir / "metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                f"Please download the dataset using: ./scripts/download_data.sh"
            )
        
        # Read metadata with proper handling
        df = pd.read_csv(metadata_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        return df
    
    def get_battery_ids(self) -> List[str]:
        """Get list of unique battery IDs in the dataset"""
        return sorted(self.metadata['battery_id'].unique().tolist())
    
    def get_battery_cycles(self, battery_id: str, 
                           cycle_type: str = None) -> pd.DataFrame:
        """
        Get all cycles for a specific battery.
        
        Args:
            battery_id: Battery identifier (e.g., 'B0005')
            cycle_type: Filter by 'charge', 'discharge', or None for all
            
        Returns:
            DataFrame with cycle metadata
        """
        # Filter by battery
        battery_df = self.metadata[self.metadata['battery_id'] == battery_id].copy()
        
        # Filter by cycle type if specified
        if cycle_type:
            battery_df = battery_df[battery_df['type'] == cycle_type]
        
        return battery_df.reset_index(drop=True)
    
    def load_cycle(self, filename: str) -> pd.DataFrame:
        """
        Load a single cycle data file.
        
        Args:
            filename: CSV filename (e.g., '00001.csv')
            
        Returns:
            DataFrame with cycle time-series data
        """
        file_path = self.data_dir / "data" / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Cycle file not found: {file_path}")
        
        # Load cycle data
        df = pd.read_csv(file_path)
        
        # Rename columns for consistency
        df = df.rename(columns={
            'Voltage_measured': 'voltage',
            'Current_measured': 'current',
            'Temperature_measured': 'temperature',
            'Time': 'time'
        })
        
        # Select relevant columns
        df = df[['time', 'voltage', 'current', 'temperature']]
        
        return df
    
    def process_battery(self, battery_id: str, 
                       save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and process all cycles for a battery.
        
        Args:
            battery_id: Battery identifier (e.g., 'B0005')
            save: Whether to save processed data to CSV
            
        Returns:
            Tuple of (discharge_df, charge_df)
        """
        print(f"\n{'='*60}")
        print(f"Processing Battery: {battery_id}")
        print(f"{'='*60}")
        
        # Get discharge cycles
        discharge_meta = self.get_battery_cycles(battery_id, 'discharge')
        print(f"Found {len(discharge_meta)} discharge cycles")
        
        # Get charge cycles
        charge_meta = self.get_battery_cycles(battery_id, 'charge')
        print(f"Found {len(charge_meta)} charge cycles")
        
        # Load all discharge cycles
        discharge_dfs = []
        for idx, row in discharge_meta.iterrows():
            try:
                cycle_df = self.load_cycle(row['filename'])
                cycle_df['cycle'] = idx + 1
                cycle_df['cycle_type'] = 'discharge'
                cycle_df['battery_id'] = battery_id
                
                # Add capacity if available
                if 'Capacity' in row and not pd.isna(row['Capacity']):
                    cycle_df['capacity'] = row['Capacity']
                
                discharge_dfs.append(cycle_df)
            except Exception as e:
                print(f"Warning: Could not load {row['filename']}: {e}")
        
        # Load all charge cycles
        charge_dfs = []
        for idx, row in charge_meta.iterrows():
            try:
                cycle_df = self.load_cycle(row['filename'])
                cycle_df['cycle'] = idx + 1
                cycle_df['cycle_type'] = 'charge'
                cycle_df['battery_id'] = battery_id
                charge_dfs.append(cycle_df)
            except Exception as e:
                print(f"Warning: Could not load {row['filename']}: {e}")
        
        # Concatenate
        discharge_df = pd.concat(discharge_dfs, ignore_index=True) if discharge_dfs else pd.DataFrame()
        charge_df = pd.concat(charge_dfs, ignore_index=True) if charge_dfs else pd.DataFrame()
        
        print(f"\nSuccessfully loaded:")
        print(f"  - {len(discharge_dfs)} discharge cycles ({len(discharge_df)} samples)")
        print(f"  - {len(charge_dfs)} charge cycles ({len(charge_df)} samples)")
        
        # Save to processed directory
        if save:
            if not discharge_df.empty:
                discharge_path = self.processed_dir / f"{battery_id}_discharge.csv"
                discharge_df.to_csv(discharge_path, index=False)
                print(f"\nSaved discharge data to {discharge_path}")
            
            if not charge_df.empty:
                charge_path = self.processed_dir / f"{battery_id}_charge.csv"
                charge_df.to_csv(charge_path, index=False)
                print(f"Saved charge data to {charge_path}")
        
        return discharge_df, charge_df
    
    def get_cycle_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for each cycle.
        
        Args:
            df: DataFrame containing cycle data
            
        Returns:
            DataFrame with cycle-level statistics
        """
        summary = df.groupby('cycle').agg({
            'time': ['min', 'max', 'count'],
            'voltage': ['min', 'max', 'mean'],
            'current': ['mean', 'std'],
            'temperature': ['min', 'max', 'mean']
        }).round(4)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        return summary


def main():
    """
    Example usage: Load B0005 battery data
    """
    print("="*60)
    print("NASA Battery Dataset Loader - Step 1.1")
    print("="*60)
    
    # Initialize loader
    loader = NASABatteryLoader()
    
    # Show available batteries
    batteries = loader.get_battery_ids()
    print(f"\nAvailable batteries: {batteries[:10]}...")  # Show first 10
    print(f"Total batteries: {len(batteries)}")
    
    # Process B0005 battery
    battery_id = "B0005"
    discharge_df, charge_df = loader.process_battery(battery_id)
    
    # Display results
    print("\n" + "="*60)
    print("DISCHARGE DATA")
    print("="*60)
    print(f"Total samples: {len(discharge_df)}")
    print(f"Number of cycles: {discharge_df['cycle'].nunique()}")
    print(f"\nFirst 10 rows:")
    print(discharge_df.head(10))
    print(f"\nData types:")
    print(discharge_df.dtypes)
    print(f"\nBasic statistics:")
    print(discharge_df[['time', 'voltage', 'current', 'temperature']].describe())
    
    # Cycle summary
    print("\n" + "="*60)
    print("CYCLE SUMMARY (Discharge)")
    print("="*60)
    summary = loader.get_cycle_summary(discharge_df)
    print(summary.head(10))
    
    # Voltage range
    print("\n" + "="*60)
    print("VOLTAGE CHARACTERISTICS")
    print("="*60)
    print(f"Min voltage: {discharge_df['voltage'].min():.4f} V")
    print(f"Max voltage: {discharge_df['voltage'].max():.4f} V")
    print(f"Mean voltage: {discharge_df['voltage'].mean():.4f} V")
    
    # Current range
    print("\n" + "="*60)
    print("CURRENT CHARACTERISTICS")
    print("="*60)
    print(f"Min current: {discharge_df['current'].min():.4f} A")
    print(f"Max current: {discharge_df['current'].max():.4f} A")
    print(f"Mean current: {discharge_df['current'].mean():.4f} A")
    
    # Temperature range
    print("\n" + "="*60)
    print("TEMPERATURE CHARACTERISTICS")
    print("="*60)
    print(f"Min temperature: {discharge_df['temperature'].min():.2f} °C")
    print(f"Max temperature: {discharge_df['temperature'].max():.2f} °C")
    print(f"Mean temperature: {discharge_df['temperature'].mean():.2f} °C")
    
    print("\n" + "="*60)
    print("CHARGE DATA")
    print("="*60)
    print(f"Total samples: {len(charge_df)}")
    print(f"Number of cycles: {charge_df['cycle'].nunique()}")
    
    print("\n" + "="*60)
    print("✓ Step 1.1 Complete")
    print("="*60)
    print("\nFiles saved:")
    print(f"  - data/processed/{battery_id}_discharge.csv")
    print(f"  - data/processed/{battery_id}_charge.csv")
    

if __name__ == "__main__":
    main()
