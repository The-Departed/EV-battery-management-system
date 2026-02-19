"""
Step 3.5: Dataset Builder
==========================

Build large-scale training/validation datasets for ML model training.

Features:
1. Batch generation of multiple scenarios
2. Combinatorial scenario coverage (drive cycles × temperatures × SOC)
3. Train/validation/test split
4. Data normalization and scaling
5. HDF5 export for efficient loading
6. Dataset statistics and metadata

Author: Battery Modeling Pipeline
Date: 2026-01-27
"""

import numpy as np
import pandas as pd
import h5py
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import itertools

# Import generators from previous steps
from generation.drive_cycles import DriveCycleLoader
from generation.temperature_profiles import TemperatureProfileGenerator
from generation.batch_simulator import BatchPhysicsSimulator, SimulationConfig
from generation.sensor_noise import SensorNoiseInjector, NoiseConfig


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    
    # Scenario coverage
    drive_cycles: List[str] = None  # ['UDDS', 'US06', 'HWFET']
    temp_profiles: List[Tuple[str, float, float]] = None  # [('constant', 0, 0), ...]
    soc_initial_range: List[float] = None  # [0.5, 0.7, 0.9, 1.0]
    
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Noise configuration
    add_noise: bool = True
    noise_config: NoiseConfig = None
    
    # Output settings
    output_dir: str = "results/datasets"
    dataset_name: str = "battery_thermal_dataset"
    save_format: str = "hdf5"  # 'hdf5' or 'csv'
    
    # Random seed
    random_seed: int = 42
    
    def __post_init__(self):
        """Set default values if not provided."""
        if self.drive_cycles is None:
            self.drive_cycles = ['UDDS', 'US06', 'HWFET']
        
        if self.temp_profiles is None:
            self.temp_profiles = [
                ('constant', 0, 0),      # 0°C
                ('constant', 25, 0),     # 25°C
                ('constant', 50, 0),     # 50°C
            ]
        
        if self.soc_initial_range is None:
            self.soc_initial_range = [0.5, 0.7, 0.9, 1.0]
        
        if self.noise_config is None:
            self.noise_config = NoiseConfig()


class DatasetBuilder:
    """
    Build large-scale training/validation datasets.
    
    Generates multiple simulation scenarios and organizes them
    into train/val/test splits with proper normalization.
    """
    
    def __init__(self, config: DatasetConfig = DatasetConfig()):
        """
        Initialize dataset builder.
        
        Parameters
        ----------
        config : DatasetConfig
            Dataset generation configuration
        """
        self.config = config
        
        # Set random seed
        np.random.seed(config.random_seed)
        
        # Initialize components
        self.drive_loader = DriveCycleLoader()
        self.temp_gen = TemperatureProfileGenerator()
        self.simulator = BatchPhysicsSimulator(SimulationConfig())
        self.noise_injector = SensorNoiseInjector(config.noise_config)
        
        # Create output directory
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Dataset builder initialized")
        print(f"  Output: {self.output_path}/{config.dataset_name}")
        print(f"  Format: {config.save_format}")
    
    def generate_scenario_list(self) -> List[Dict]:
        """
        Generate list of all scenario combinations.
        
        Returns
        -------
        scenarios : list of dict
            List of scenario configurations
        """
        scenarios = []
        
        # Combinatorial coverage: drive_cycles × temp_profiles × soc_initial
        for drive_cycle in self.config.drive_cycles:
            for temp_type, temp_param1, temp_param2 in self.config.temp_profiles:
                for soc_init in self.config.soc_initial_range:
                    scenario = {
                        'drive_cycle': drive_cycle,
                        'temp_type': temp_type,
                        'temp_param1': temp_param1,
                        'temp_param2': temp_param2,
                        'soc_initial': soc_init
                    }
                    scenarios.append(scenario)
        
        print(f"\n✓ Generated {len(scenarios)} scenario combinations:")
        print(f"  Drive cycles: {len(self.config.drive_cycles)}")
        print(f"  Temp profiles: {len(self.config.temp_profiles)}")
        print(f"  SOC initials: {len(self.config.soc_initial_range)}")
        
        return scenarios
    
    def simulate_scenario(self, scenario: Dict) -> pd.DataFrame:
        """
        Simulate a single scenario.
        
        Parameters
        ----------
        scenario : dict
            Scenario configuration
        
        Returns
        -------
        df : pd.DataFrame
            Simulation results as DataFrame
        """
        # Load drive cycle
        time, current = self.drive_loader.load_cycle(scenario['drive_cycle'])
        
        # Generate temperature profile
        if scenario['temp_type'] == 'constant':
            _, temp_k = self.temp_gen.generate_constant(
                duration=time[-1],
                temperature_c=scenario['temp_param1']
            )
        elif scenario['temp_type'] == 'sinusoidal':
            _, temp_k = self.temp_gen.generate_sinusoidal(
                duration=time[-1],
                temp_mean_c=scenario['temp_param1'],
                temp_amplitude_c=scenario['temp_param2'],
                period=time[-1]
            )
        else:
            raise ValueError(f"Unknown temp_type: {scenario['temp_type']}")
        
        # Run physics simulation
        results = self.simulator.simulate(
            time=time,
            current=current,
            temp_ambient_k=temp_k,
            soc_initial=scenario['soc_initial']
        )
        
        # Convert to DataFrame
        C_TO_K = 273.15
        df = pd.DataFrame({
            'time_s': results['time'],
            'current_A': results['current'],
            'voltage_V': results['voltage'],
            'soc': results['soc'],
            'temp_surface_C': results['temp_surface_k'] - C_TO_K,
            'temp_core_C': results['temp_core_k'] - C_TO_K,
            'temp_ambient_C': results['temp_ambient_k'] - C_TO_K,
            'heat_generation_W': results['heat_generation'],
            'power_W': results['power']
        })
        
        # Add scenario metadata
        for key, value in scenario.items():
            df[f'meta_{key}'] = value
        
        # Add noise if configured
        if self.config.add_noise:
            clean_data = {
                'current': df['current_A'].values,
                'voltage': df['voltage_V'].values,
                'temp_surface': df['temp_surface_C'].values,
                'temp_ambient': df['temp_ambient_C'].values
            }
            
            noisy_data = self.noise_injector.inject_dataset_noise(clean_data)
            
            # Add noisy measurements
            df['current_meas_A'] = noisy_data['current_noisy']
            df['voltage_meas_V'] = noisy_data['voltage_noisy']
            df['temp_surface_meas_C'] = noisy_data['temp_surface_noisy']
            df['temp_ambient_meas_C'] = noisy_data['temp_ambient_noisy']
            df['power_meas_W'] = noisy_data['power_noisy']
        
        return df
    
    def generate_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Generate full dataset with all scenarios.
        
        Returns
        -------
        dataset : dict
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        print("\n" + "="*80)
        print("GENERATING FULL DATASET")
        print("="*80 + "\n")
        
        # Generate scenario list
        scenarios = self.generate_scenario_list()
        
        # Shuffle scenarios for random train/val/test split
        np.random.shuffle(scenarios)
        
        # Simulate all scenarios
        print(f"\nSimulating {len(scenarios)} scenarios...")
        all_data = []
        
        for i, scenario in enumerate(scenarios):
            print(f"  [{i+1}/{len(scenarios)}] Simulating: {scenario['drive_cycle']} @ "
                  f"{scenario['temp_param1']}°C, SOC={scenario['soc_initial']:.1%}", end='\r')
            try:
                df = self.simulate_scenario(scenario)
                df['scenario_id'] = i
                all_data.append(df)
            except Exception as e:
                print(f"\n  Warning: Scenario {i} failed: {e}")
                continue
        
        print()  # New line after progress
        
        # Concatenate all scenarios
        full_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✓ Generated {len(full_df)} total samples from {len(all_data)} scenarios")
        
        # Split into train/val/test by scenario
        n_scenarios = len(all_data)
        n_train = int(n_scenarios * self.config.train_ratio)
        n_val = int(n_scenarios * self.config.val_ratio)
        
        train_scenarios = list(range(n_train))
        val_scenarios = list(range(n_train, n_train + n_val))
        test_scenarios = list(range(n_train + n_val, n_scenarios))
        
        dataset = {
            'train': full_df[full_df['scenario_id'].isin(train_scenarios)].reset_index(drop=True),
            'val': full_df[full_df['scenario_id'].isin(val_scenarios)].reset_index(drop=True),
            'test': full_df[full_df['scenario_id'].isin(test_scenarios)].reset_index(drop=True)
        }
        
        print(f"\n✓ Dataset split:")
        print(f"  Train: {len(dataset['train'])} samples ({len(train_scenarios)} scenarios)")
        print(f"  Val:   {len(dataset['val'])} samples ({len(val_scenarios)} scenarios)")
        print(f"  Test:  {len(dataset['test'])} samples ({len(test_scenarios)} scenarios)")
        
        return dataset
    
    def compute_normalization_params(
        self,
        train_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute normalization parameters from training set.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data
        
        Returns
        -------
        norm_params : dict
            Normalization parameters (mean, std) for each feature
        """
        # Features to normalize
        features = [
            'current_A', 'voltage_V', 'temp_surface_C', 
            'temp_ambient_C', 'soc', 'temp_core_C'
        ]
        
        if self.config.add_noise:
            features += [
                'current_meas_A', 'voltage_meas_V', 
                'temp_surface_meas_C', 'temp_ambient_meas_C'
            ]
        
        norm_params = {}
        for feature in features:
            if feature in train_df.columns:
                norm_params[feature] = {
                    'mean': float(train_df[feature].mean()),
                    'std': float(train_df[feature].std()),
                    'min': float(train_df[feature].min()),
                    'max': float(train_df[feature].max())
                }
        
        return norm_params
    
    def save_dataset(
        self,
        dataset: Dict[str, pd.DataFrame],
        norm_params: Dict[str, Dict[str, float]]
    ):
        """
        Save dataset to disk.
        
        Parameters
        ----------
        dataset : dict
            Dictionary with 'train', 'val', 'test' DataFrames
        norm_params : dict
            Normalization parameters
        """
        print("\n" + "="*80)
        print("SAVING DATASET")
        print("="*80 + "\n")
        
        if self.config.save_format == 'hdf5':
            self._save_hdf5(dataset, norm_params)
        elif self.config.save_format == 'csv':
            self._save_csv(dataset, norm_params)
        else:
            raise ValueError(f"Unknown format: {self.config.save_format}")
    
    def _save_hdf5(
        self,
        dataset: Dict[str, pd.DataFrame],
        norm_params: Dict[str, Dict[str, float]]
    ):
        """Save dataset in HDF5 format."""
        hdf5_path = self.output_path / f"{self.config.dataset_name}.h5"
        
        with h5py.File(hdf5_path, 'w') as f:
            # Save train/val/test splits
            for split_name, df in dataset.items():
                grp = f.create_group(split_name)
                
                # Save all columns as datasets
                for col in df.columns:
                    grp.create_dataset(col, data=df[col].values)
                
                # Add metadata
                grp.attrs['n_samples'] = len(df)
                grp.attrs['n_scenarios'] = df['scenario_id'].nunique()
            
            # Save normalization parameters
            norm_grp = f.create_group('normalization')
            norm_grp.attrs['params'] = json.dumps(norm_params)
            
            # Save dataset config
            f.attrs['config'] = json.dumps(asdict(self.config), default=str)
        
        print(f"✓ HDF5 dataset saved: {hdf5_path}")
        print(f"  Size: {hdf5_path.stat().st_size / 1024**2:.2f} MB")
    
    def _save_csv(
        self,
        dataset: Dict[str, pd.DataFrame],
        norm_params: Dict[str, Dict[str, float]]
    ):
        """Save dataset in CSV format."""
        for split_name, df in dataset.items():
            csv_path = self.output_path / f"{self.config.dataset_name}_{split_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"✓ CSV saved: {csv_path} ({len(df)} samples)")
        
        # Save normalization params
        norm_path = self.output_path / f"{self.config.dataset_name}_normalization.json"
        with open(norm_path, 'w') as f:
            json.dump(norm_params, f, indent=2)
        print(f"✓ Normalization params saved: {norm_path}")
        
        # Save config
        config_path = self.output_path / f"{self.config.dataset_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
        print(f"✓ Config saved: {config_path}")
    
    def plot_dataset_statistics(self, dataset: Dict[str, pd.DataFrame]):
        """
        Plot dataset statistics and distributions.
        
        Parameters
        ----------
        dataset : dict
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        print("\n" + "="*80)
        print("PLOTTING DATASET STATISTICS")
        print("="*80 + "\n")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle('Dataset Statistics', fontsize=16, fontweight='bold')
        
        features = [
            ('current_A', 'Current [A]'),
            ('voltage_V', 'Voltage [V]'),
            ('soc', 'SOC'),
            ('temp_surface_C', 'Surface Temp [°C]'),
            ('temp_core_C', 'Core Temp [°C]'),
            ('temp_ambient_C', 'Ambient Temp [°C]'),
            ('heat_generation_W', 'Heat Gen [W]'),
            ('power_W', 'Power [W]')
        ]
        
        for idx, (feature, label) in enumerate(features[:9]):
            ax = axes[idx // 3, idx % 3]
            
            for split_name, color in [('train', 'blue'), ('val', 'orange'), ('test', 'green')]:
                if feature in dataset[split_name].columns:
                    data = dataset[split_name][feature].values
                    ax.hist(data, bins=50, alpha=0.5, label=split_name, color=color, density=True)
            
            ax.set_xlabel(label, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{label} Distribution', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_path / f"{self.config.dataset_name}_statistics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Statistics plot saved: {plot_path}")
        plt.close()


def main():
    """Demonstration of dataset building."""
    
    print("\n" + "="*80)
    print("STEP 3.5: DATASET BUILDER")
    print("="*80 + "\n")
    
    # ========================================================================
    # Configure dataset generation
    # ========================================================================
    config = DatasetConfig(
        # Scenario coverage
        # Repeat cycles to get random variations (stochastic generation)
        drive_cycles=['UDDS']*5 + ['US06']*5 + ['HWFET']*5,  
        temp_profiles=[
            ('constant', -10, 0),    # Extreme Cold
            ('constant', 0, 0),      # Cold
            ('constant', 10, 0),     # Cool
            ('constant', 25, 0),     # Nominal
            ('constant', 40, 0),     # Warm
            ('constant', 50, 0),     # Hot
            ('constant', 60, 0),     # Extreme Hot
            ('sinusoidal', 25, 10),  # Daily cycle: 25±10°C
            ('sinusoidal', 10, 10),  # Cold Daily: 10±10°C
            ('sinusoidal', 40, 10),  # Hot Daily: 40±10°C
        ],
        soc_initial_range=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        
        # Split ratios
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        
        # Noise
        add_noise=True,
        noise_config=NoiseConfig(
            current_noise_std=0.01,
            voltage_noise_std=0.002,
            temp_noise_std=0.1,
            outlier_probability=0.001,
            random_seed=42
        ),
        
        # Output
        output_dir="results/datasets",
        dataset_name="battery_thermal_v1",
        save_format="csv",  # CSV for easier inspection
        
        random_seed=42
    )
    
    # ========================================================================
    # Build dataset
    # ========================================================================
    builder = DatasetBuilder(config)
    
    # Generate all scenarios
    dataset = builder.generate_dataset()
    
    # Compute normalization parameters
    print("\n" + "="*80)
    print("Computing normalization parameters...")
    norm_params = builder.compute_normalization_params(dataset['train'])
    print("✓ Normalization parameters computed\n")
    
    for feature, params in norm_params.items():
        print(f"  {feature:25s}: μ={params['mean']:8.4f}, σ={params['std']:8.4f}, "
              f"range=[{params['min']:8.4f}, {params['max']:8.4f}]")
    
    # Save dataset
    builder.save_dataset(dataset, norm_params)
    
    # Plot statistics
    builder.plot_dataset_statistics(dataset)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3.5 COMPLETE ✓")
    print("="*80)
    print(f"\nDataset generation successful!")
    print(f"  Scenarios: {len(config.drive_cycles)} × {len(config.temp_profiles)} × {len(config.soc_initial_range)} = {len(config.drive_cycles) * len(config.temp_profiles) * len(config.soc_initial_range)}")
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Val:   {len(dataset['val'])} samples")
    print(f"  Test:  {len(dataset['test'])} samples")
    print(f"  Total: {len(dataset['train']) + len(dataset['val']) + len(dataset['test'])} samples")
    print(f"\nDataset saved to: {builder.output_path}/")
    print(f"Next: Step 3.6 - Visualization & Sanity Checks")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
