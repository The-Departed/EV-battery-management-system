"""
Drive Cycle Loader
Step 3.1: Load and manage current profiles for battery simulation

Supports:
- UDDS (Urban Dynamometer Driving Schedule)
- US06 (Aggressive driving)
- HWFET (Highway Fuel Economy Test)
- Custom synthetic profiles

Drive cycles define the current demand I(t) over time.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class DriveCycleLoader:
    """
    Load and manage drive cycle current profiles.
    """
    
    # Standard drive cycle specifications
    CYCLE_SPECS = {
        'UDDS': {
            'duration': 12000,  # seconds (extended from 1369 for longer simulation)
            'description': 'Urban Dynamometer Driving Schedule (Extended)',
            'max_current': 4.0,  # A (example for 18650 cell)
            'rms_current': 1.5,
        },
        'US06': {
            'duration': 12000,  # seconds (extended from 600 for longer simulation)
            'description': 'US06 Supplemental FTP Driving Schedule (Extended)',
            'max_current': 6.0,  # A (aggressive)
            'rms_current': 2.5,
        },
        'HWFET': {
            'duration': 12000,  # seconds (extended from 765 for longer simulation)
            'description': 'Highway Fuel Economy Test Cycle (Extended)',
            'max_current': 3.0,  # A (highway cruising)
            'rms_current': 1.8,
        },
    }
    
    def __init__(self):
        """Initialize drive cycle loader."""
        self.project_root = Path(__file__).parent.parent
        self.cycles_dir = self.project_root / "data" / "drive_cycles"
        self.cycles_dir.mkdir(parents=True, exist_ok=True)
        
    def load_cycle(self, cycle_name: str, scale_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a drive cycle.
        
        Args:
            cycle_name: Name of cycle ('UDDS', 'US06', 'HWFET', or path to CSV)
            scale_factor: Scale current by this factor (default=1.0)
            
        Returns:
            (time, current): Time vector (s) and current vector (A)
        """
        if cycle_name.upper() in self.CYCLE_SPECS:
            # Load standard cycle
            return self._load_standard_cycle(cycle_name.upper(), scale_factor)
        elif Path(cycle_name).exists():
            # Load from file
            return self._load_from_file(cycle_name, scale_factor)
        else:
            raise ValueError(f"Unknown cycle: {cycle_name}")
    
    def _load_standard_cycle(self, cycle_name: str, scale_factor: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a standard drive cycle.
        
        For now, generates synthetic profiles matching standard specifications.
        In production, these would be loaded from reference datasets.
        """
        spec = self.CYCLE_SPECS[cycle_name]
        duration = spec['duration']
        
        if cycle_name == 'UDDS':
            return self._generate_udds(duration, spec['max_current'] * scale_factor)
        elif cycle_name == 'US06':
            return self._generate_us06(duration, spec['max_current'] * scale_factor)
        elif cycle_name == 'HWFET':
            return self._generate_hwfet(duration, spec['max_current'] * scale_factor)
        else:
            raise ValueError(f"Unknown cycle: {cycle_name}")
    
    def _generate_udds(self, duration: int, max_current: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate UDDS-like current profile.
        
        UDDS characteristics:
        - Urban driving with stops
        - Accelerations and decelerations
        - Mix of constant speed and idle
        """
        dt = 1.0  # 1 Hz sampling
        time = np.arange(0, duration, dt)
        current = np.zeros_like(time)
        
        # UDDS pattern: segments of acceleration, cruise, deceleration, idle
        t = 0
        while t < len(time):
            # Idle (0 current)
            idle_duration = int(np.random.uniform(5, 20))
            current[t:min(t+idle_duration, len(time))] = 0
            t += idle_duration
            
            if t >= len(time):
                break
            
            # Acceleration (discharge)
            accel_duration = int(np.random.uniform(10, 30))
            accel_current = np.random.uniform(0.3, 0.8) * max_current
            for i in range(min(accel_duration, len(time)-t)):
                current[t+i] = -accel_current * (1 - np.exp(-i/5))  # Ramp up
            t += accel_duration
            
            if t >= len(time):
                break
            
            # Cruise (moderate discharge)
            cruise_duration = int(np.random.uniform(20, 60))
            cruise_current = np.random.uniform(0.2, 0.5) * max_current
            current[t:min(t+cruise_duration, len(time))] = -cruise_current
            t += cruise_duration
            
            if t >= len(time):
                break
            
            # Deceleration (regenerative braking - charge)
            decel_duration = int(np.random.uniform(5, 15))
            regen_current = np.random.uniform(0.1, 0.3) * max_current
            current[t:min(t+decel_duration, len(time))] = regen_current
            t += decel_duration
        
        # Smooth transitions
        current = self._smooth_profile(current, window=5)
        
        return time, current
    
    def _generate_us06(self, duration: int, max_current: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate US06-like current profile.
        
        US06 characteristics:
        - Aggressive driving
        - High accelerations
        - Less idle time
        - Higher average current
        """
        dt = 1.0
        time = np.arange(0, duration, dt)
        current = np.zeros_like(time)
        
        t = 0
        while t < len(time):
            # Short idle
            idle_duration = int(np.random.uniform(2, 8))
            current[t:min(t+idle_duration, len(time))] = 0
            t += idle_duration
            
            if t >= len(time):
                break
            
            # Aggressive acceleration
            accel_duration = int(np.random.uniform(8, 20))
            accel_current = np.random.uniform(0.6, 1.0) * max_current
            for i in range(min(accel_duration, len(time)-t)):
                current[t+i] = -accel_current * (1 - np.exp(-i/3))
            t += accel_duration
            
            if t >= len(time):
                break
            
            # High-speed cruise
            cruise_duration = int(np.random.uniform(15, 40))
            cruise_current = np.random.uniform(0.5, 0.8) * max_current
            current[t:min(t+cruise_duration, len(time))] = -cruise_current
            t += cruise_duration
            
            if t >= len(time):
                break
            
            # Aggressive braking
            decel_duration = int(np.random.uniform(5, 12))
            regen_current = np.random.uniform(0.2, 0.5) * max_current
            current[t:min(t+decel_duration, len(time))] = regen_current
            t += decel_duration
        
        current = self._smooth_profile(current, window=3)
        
        return time, current
    
    def _generate_hwfet(self, duration: int, max_current: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate HWFET-like current profile.
        
        HWFET characteristics:
        - Highway driving
        - Relatively constant speed
        - Smooth accelerations
        - No idle periods
        """
        dt = 1.0
        time = np.arange(0, duration, dt)
        
        # Base highway cruise current
        base_current = 0.4 * max_current
        current = -base_current * np.ones_like(time)
        
        # Add gentle speed variations (overtaking, hills)
        num_variations = duration // 100
        for _ in range(num_variations):
            start = int(np.random.uniform(0, duration - 50))
            var_duration = int(np.random.uniform(20, 50))
            variation = np.random.uniform(-0.2, 0.3) * max_current
            
            # Create smooth variation
            for i in range(var_duration):
                if start + i < len(time):
                    envelope = np.sin(np.pi * i / var_duration)
                    current[start + i] += variation * envelope
        
        current = self._smooth_profile(current, window=10)
        
        return time, current
    
    def _smooth_profile(self, current: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Smooth current profile using moving average.
        
        Args:
            current: Current profile
            window: Window size for smoothing
            
        Returns:
            Smoothed current
        """
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(current, size=window, mode='nearest')
    
    def _load_from_file(self, filepath: str, scale_factor: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load drive cycle from CSV file.
        
        Expected format:
        - Column 'time' or 'Time' (seconds)
        - Column 'current' or 'Current' (A)
        """
        df = pd.read_csv(filepath)
        
        # Find time column
        time_col = None
        for col in ['time', 'Time', 'TIME', 't']:
            if col in df.columns:
                time_col = col
                break
        
        # Find current column
        current_col = None
        for col in ['current', 'Current', 'CURRENT', 'I', 'i']:
            if col in df.columns:
                current_col = col
                break
        
        if time_col is None or current_col is None:
            raise ValueError(f"Could not find time/current columns in {filepath}")
        
        time = df[time_col].values
        current = df[current_col].values * scale_factor
        
        return time, current
    
    def get_cycle_stats(self, time: np.ndarray, current: np.ndarray) -> dict:
        """
        Calculate statistics for a drive cycle.
        
        Args:
            time: Time vector (s)
            current: Current vector (A)
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'duration': time[-1] - time[0],
            'samples': len(time),
            'sampling_rate': len(time) / (time[-1] - time[0]),
            'current_min': current.min(),
            'current_max': current.max(),
            'current_mean': current.mean(),
            'current_rms': np.sqrt(np.mean(current**2)),
            'current_std': current.std(),
            'discharge_fraction': np.sum(current < 0) / len(current),
            'charge_fraction': np.sum(current > 0) / len(current),
            'idle_fraction': np.sum(current == 0) / len(current),
        }
        
        return stats
    
    def plot_cycle(self, time: np.ndarray, current: np.ndarray, 
                   title: str = "Drive Cycle", save_path: Optional[str] = None):
        """
        Plot drive cycle current profile.
        
        Args:
            time: Time vector (s)
            current: Current vector (A)
            title: Plot title
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        time_min = time / 60  # Convert to minutes
        
        # Current vs time
        ax = axes[0]
        ax.plot(time_min, current, 'b-', linewidth=1)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Time (min)', fontsize=11)
        ax.set_ylabel('Current (A)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats = self.get_cycle_stats(time, current)
        stats_text = (
            f"Duration: {stats['duration']:.0f} s ({stats['duration']/60:.1f} min)\n"
            f"RMS: {stats['current_rms']:.3f} A\n"
            f"Range: [{stats['current_min']:.3f}, {stats['current_max']:.3f}] A\n"
            f"Mean: {stats['current_mean']:.3f} A"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Current histogram
        ax = axes[1]
        ax.hist(current, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='k', linestyle='--', linewidth=1.5)
        ax.axvline(stats['current_mean'], color='r', linestyle='--', 
                   linewidth=1.5, label=f"Mean = {stats['current_mean']:.3f} A")
        ax.axvline(stats['current_rms'], color='g', linestyle='--',
                   linewidth=1.5, label=f"RMS = {stats['current_rms']:.3f} A")
        ax.set_xlabel('Current (A)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Current Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.close()


def demo_drive_cycles():
    """
    Demonstrate drive cycle loading and visualization.
    """
    print("\n" + "="*70)
    print("STEP 3.1: DRIVE CYCLE LOADER - DEMONSTRATION")
    print("="*70)
    
    loader = DriveCycleLoader()
    
    # Create plots directory
    plots_dir = loader.project_root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and plot each standard cycle
    for cycle_name in ['UDDS', 'US06', 'HWFET']:
        print(f"\n{'─'*70}")
        print(f"Loading {cycle_name}...")
        print(f"{'─'*70}")
        
        # Load cycle
        time, current = loader.load_cycle(cycle_name, scale_factor=1.0)
        
        # Get statistics
        stats = loader.get_cycle_stats(time, current)
        
        print(f"\n✓ {cycle_name} loaded successfully")
        print(f"\nCycle Statistics:")
        print(f"  Duration:     {stats['duration']:.0f} s ({stats['duration']/60:.1f} min)")
        print(f"  Samples:      {stats['samples']}")
        print(f"  Sample rate:  {stats['sampling_rate']:.1f} Hz")
        print(f"  Current range: [{stats['current_min']:.3f}, {stats['current_max']:.3f}] A")
        print(f"  Mean current: {stats['current_mean']:.3f} A")
        print(f"  RMS current:  {stats['current_rms']:.3f} A")
        print(f"  Std current:  {stats['current_std']:.3f} A")
        print(f"\nTime Distribution:")
        print(f"  Discharge:    {stats['discharge_fraction']*100:.1f}%")
        print(f"  Charge:       {stats['charge_fraction']*100:.1f}%")
        print(f"  Idle:         {stats['idle_fraction']*100:.1f}%")
        
        # Plot
        spec = loader.CYCLE_SPECS[cycle_name]
        title = f"{cycle_name} - {spec['description']}"
        save_path = plots_dir / f"step31_drivecycle_{cycle_name.lower()}.png"
        
        loader.plot_cycle(time, current, title=title, save_path=save_path)
    
    print(f"\n{'='*70}")
    print("✓ Step 3.1 Drive Cycle Loader Complete")
    print(f"{'='*70}")
    print(f"\nGenerated plots:")
    for cycle_name in ['UDDS', 'US06', 'HWFET']:
        print(f"  • step31_drivecycle_{cycle_name.lower()}.png")
    print()


if __name__ == "__main__":
    demo_drive_cycles()
