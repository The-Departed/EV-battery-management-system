"""
Step 3.2: Temperature Profile Generator
=======================================

Generates ambient temperature profiles Tamb(t) for battery thermal simulation:
- Constant temperatures (0°C, 25°C, 50°C)
- Step changes (thermal shock scenarios)
- Sinusoidal daily cycles (natural variation)

Author: Battery Modeling Pipeline
Date: 2026-01-27
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Optional


class TemperatureProfileGenerator:
    """
    Generate ambient temperature profiles for battery thermal testing.
    
    Supports:
    - Constant temperatures (steady-state operation)
    - Step changes (thermal shock, climate transitions)
    - Sinusoidal cycles (daily temperature variation)
    - Custom profiles from data
    
    Temperatures in Celsius, converted to Kelvin internally for physics.
    """
    
    def __init__(self, dt: float = 1.0):
        """
        Initialize temperature profile generator.
        
        Parameters
        ----------
        dt : float
            Time step in seconds (default: 1.0 s)
        """
        self.dt = dt
        self.C_TO_K = 273.15  # Celsius to Kelvin conversion
    
    def generate_constant(
        self,
        duration: float,
        temperature_c: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate constant temperature profile.
        
        Parameters
        ----------
        duration : float
            Duration in seconds
        temperature_c : float
            Constant temperature in Celsius
        
        Returns
        -------
        time : np.ndarray
            Time vector [s]
        temp_k : np.ndarray
            Temperature vector [K]
        """
        n_samples = int(duration / self.dt) + 1
        time = np.linspace(0, duration, n_samples)
        temp_k = np.full(n_samples, temperature_c + self.C_TO_K)
        
        return time, temp_k
    
    def generate_step(
        self,
        duration: float,
        temp_initial_c: float,
        temp_final_c: float,
        step_time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate step change temperature profile.
        
        Simulates:
        - Thermal shock testing
        - Climate zone transitions (e.g., garage → highway)
        - Sudden environmental changes
        
        Parameters
        ----------
        duration : float
            Total duration in seconds
        temp_initial_c : float
            Initial temperature in Celsius
        temp_final_c : float
            Final temperature after step in Celsius
        step_time : float
            Time of step change in seconds
        
        Returns
        -------
        time : np.ndarray
            Time vector [s]
        temp_k : np.ndarray
            Temperature vector [K] with step change
        """
        n_samples = int(duration / self.dt) + 1
        time = np.linspace(0, duration, n_samples)
        
        # Create step profile
        temp_c = np.full(n_samples, temp_initial_c)
        step_idx = int(step_time / self.dt)
        temp_c[step_idx:] = temp_final_c
        
        # Convert to Kelvin
        temp_k = temp_c + self.C_TO_K
        
        return time, temp_k
    
    def generate_ramp(
        self,
        duration: float,
        temp_initial_c: float,
        temp_final_c: float,
        ramp_start: float,
        ramp_duration: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate linear ramp temperature profile.
        
        Simulates:
        - Gradual climate changes
        - Thermal chamber testing
        - Seasonal transitions
        
        Parameters
        ----------
        duration : float
            Total duration in seconds
        temp_initial_c : float
            Initial temperature in Celsius
        temp_final_c : float
            Final temperature in Celsius
        ramp_start : float
            Time when ramp begins in seconds
        ramp_duration : float
            Duration of ramp in seconds
        
        Returns
        -------
        time : np.ndarray
            Time vector [s]
        temp_k : np.ndarray
            Temperature vector [K] with ramp
        """
        n_samples = int(duration / self.dt) + 1
        time = np.linspace(0, duration, n_samples)
        temp_c = np.full(n_samples, temp_initial_c)
        
        # Create ramp
        ramp_start_idx = int(ramp_start / self.dt)
        ramp_end_idx = int((ramp_start + ramp_duration) / self.dt)
        ramp_end_idx = min(ramp_end_idx, n_samples)
        
        n_ramp = ramp_end_idx - ramp_start_idx
        if n_ramp > 0:
            temp_c[ramp_start_idx:ramp_end_idx] = np.linspace(
                temp_initial_c, temp_final_c, n_ramp
            )
            temp_c[ramp_end_idx:] = temp_final_c
        
        # Convert to Kelvin
        temp_k = temp_c + self.C_TO_K
        
        return time, temp_k
    
    def generate_sinusoidal(
        self,
        duration: float,
        temp_mean_c: float,
        temp_amplitude_c: float,
        period: float = 86400.0,
        phase: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sinusoidal temperature profile.
        
        Simulates:
        - Daily temperature variation (default period = 24 hours)
        - Natural environmental cycles
        - Circadian thermal patterns
        
        Formula: T(t) = T_mean + A * sin(2π * t / period + φ)
        
        Parameters
        ----------
        duration : float
            Total duration in seconds
        temp_mean_c : float
            Mean temperature in Celsius
        temp_amplitude_c : float
            Temperature amplitude (peak-to-mean) in Celsius
        period : float
            Period of oscillation in seconds (default: 86400 s = 24 hours)
        phase : float
            Phase shift in radians (default: 0)
        
        Returns
        -------
        time : np.ndarray
            Time vector [s]
        temp_k : np.ndarray
            Temperature vector [K] with sinusoidal variation
        """
        n_samples = int(duration / self.dt) + 1
        time = np.linspace(0, duration, n_samples)
        
        # Generate sinusoidal profile
        omega = 2 * np.pi / period
        temp_c = temp_mean_c + temp_amplitude_c * np.sin(omega * time + phase)
        
        # Convert to Kelvin
        temp_k = temp_c + self.C_TO_K
        
        return time, temp_k
    
    def generate_multi_step(
        self,
        duration: float,
        temperature_sequence: list[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate multi-step temperature profile.
        
        Simulates:
        - Complex thermal test sequences
        - Multi-zone operation (city → highway → parking)
        - Standard test protocols
        
        Parameters
        ----------
        duration : float
            Total duration in seconds
        temperature_sequence : list of (time, temp_c) tuples
            List of (switch_time, temperature) pairs
            Example: [(0, 25), (300, 40), (600, 10), (900, 25)]
        
        Returns
        -------
        time : np.ndarray
            Time vector [s]
        temp_k : np.ndarray
            Temperature vector [K] with multiple steps
        """
        n_samples = int(duration / self.dt) + 1
        time = np.linspace(0, duration, n_samples)
        temp_c = np.zeros(n_samples)
        
        # Sort sequence by time
        sequence = sorted(temperature_sequence, key=lambda x: x[0])
        
        # Fill temperature profile
        for i in range(len(sequence)):
            time_i, temp_i = sequence[i]
            idx_i = int(time_i / self.dt)
            
            if i < len(sequence) - 1:
                time_next = sequence[i + 1][0]
                idx_next = int(time_next / self.dt)
                temp_c[idx_i:idx_next] = temp_i
            else:
                temp_c[idx_i:] = temp_i
        
        # Convert to Kelvin
        temp_k = temp_c + self.C_TO_K
        
        return time, temp_k
    
    def get_profile_stats(
        self,
        time: np.ndarray,
        temp_k: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute statistics for temperature profile.
        
        Parameters
        ----------
        time : np.ndarray
            Time vector [s]
        temp_k : np.ndarray
            Temperature vector [K]
        
        Returns
        -------
        stats : dict
            Dictionary containing:
            - duration: Total duration [s]
            - temp_mean_c: Mean temperature [°C]
            - temp_std_c: Standard deviation [°C]
            - temp_min_c: Minimum temperature [°C]
            - temp_max_c: Maximum temperature [°C]
            - temp_range_c: Temperature range [°C]
        """
        temp_c = temp_k - self.C_TO_K
        
        stats = {
            'duration': time[-1] - time[0],
            'samples': len(time),
            'temp_mean_c': float(np.mean(temp_c)),
            'temp_std_c': float(np.std(temp_c)),
            'temp_min_c': float(np.min(temp_c)),
            'temp_max_c': float(np.max(temp_c)),
            'temp_range_c': float(np.max(temp_c) - np.min(temp_c))
        }
        
        return stats
    
    def plot_profile(
        self,
        time: np.ndarray,
        temp_k: np.ndarray,
        title: str = "Temperature Profile",
        save_path: Optional[str] = None
    ):
        """
        Plot temperature profile with statistics.
        
        Parameters
        ----------
        time : np.ndarray
            Time vector [s]
        temp_k : np.ndarray
            Temperature vector [K]
        title : str
            Plot title
        save_path : str, optional
            Path to save figure (if None, figure is displayed)
        """
        # Convert to Celsius for plotting
        temp_c = temp_k - self.C_TO_K
        
        # Get statistics
        stats = self.get_profile_stats(time, temp_k)
        
        # Create figure with 2 panels
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Panel 1: Temperature vs Time
        ax1 = axes[0]
        ax1.plot(time / 60, temp_c, 'b-', linewidth=1.5, label='Tamb(t)')
        ax1.axhline(stats['temp_mean_c'], color='r', linestyle='--', 
                   linewidth=1, label=f"Mean = {stats['temp_mean_c']:.1f}°C")
        ax1.fill_between(
            time / 60,
            stats['temp_mean_c'] - stats['temp_std_c'],
            stats['temp_mean_c'] + stats['temp_std_c'],
            alpha=0.2, color='r', label=f"±1σ = {stats['temp_std_c']:.1f}°C"
        )
        ax1.set_xlabel('Time [min]', fontsize=12)
        ax1.set_ylabel('Temperature [°C]', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)
        
        # Add statistics box
        stats_text = (
            f"Duration: {stats['duration']:.0f} s ({stats['duration']/60:.1f} min)\n"
            f"Samples: {stats['samples']}\n"
            f"Range: [{stats['temp_min_c']:.1f}, {stats['temp_max_c']:.1f}] °C\n"
            f"Mean: {stats['temp_mean_c']:.1f} °C\n"
            f"Std: {stats['temp_std_c']:.1f} °C"
        )
        ax1.text(
            0.02, 0.98, stats_text,
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        # Panel 2: Temperature Histogram
        ax2 = axes[1]
        counts, bins, patches = ax2.hist(
            temp_c, bins=50, color='steelblue', 
            alpha=0.7, edgecolor='black', linewidth=0.5
        )
        ax2.axvline(stats['temp_mean_c'], color='r', linestyle='--', 
                   linewidth=2, label=f"Mean = {stats['temp_mean_c']:.1f}°C")
        ax2.set_xlabel('Temperature [°C]', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Temperature Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Demonstration of temperature profile generation."""
    
    print("\n" + "="*80)
    print("STEP 3.2: TEMPERATURE PROFILE GENERATOR")
    print("="*80 + "\n")
    
    # Create output directory
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    gen = TemperatureProfileGenerator(dt=1.0)
    
    # ========================================================================
    # Test 1: Constant Temperatures (Steady-State Testing)
    # ========================================================================
    print("Generating constant temperature profiles...")
    
    constant_temps = [
        (0, "cold"),      # Cold climate (winter, refrigeration)
        (25, "nominal"),  # Room temperature (standard testing)
        (50, "hot")       # Hot climate (summer, desert)
    ]
    
    for temp_c, label in constant_temps:
        time, temp_k = gen.generate_constant(duration=1800, temperature_c=temp_c)
        stats = gen.get_profile_stats(time, temp_k)
        
        print(f"\n  Constant {temp_c}°C ({label}):")
        print(f"    Duration: {stats['duration']:.0f} s ({stats['duration']/60:.1f} min)")
        print(f"    Samples: {stats['samples']}")
        print(f"    Mean: {stats['temp_mean_c']:.1f} °C")
        print(f"    Range: [{stats['temp_min_c']:.1f}, {stats['temp_max_c']:.1f}] °C")
        
        # Plot
        save_path = output_dir / f"step32_temp_constant_{label}.png"
        gen.plot_profile(
            time, temp_k,
            title=f"Constant Temperature: {temp_c}°C ({label.capitalize()})",
            save_path=str(save_path)
        )
    
    # ========================================================================
    # Test 2: Step Change (Thermal Shock)
    # ========================================================================
    print("\n" + "-"*80)
    print("Generating step change profiles...")
    
    step_profiles = [
        (25, 50, 600, "cold_to_hot"),   # Cold start → Hot operation
        (50, 0, 600, "hot_to_cold"),    # Hot → Refrigeration
        (25, -10, 600, "nominal_to_freeze")  # Room temp → Freezing
    ]
    
    for temp_init, temp_final, step_time, label in step_profiles:
        time, temp_k = gen.generate_step(
            duration=1800,
            temp_initial_c=temp_init,
            temp_final_c=temp_final,
            step_time=step_time
        )
        stats = gen.get_profile_stats(time, temp_k)
        
        print(f"\n  Step {temp_init}°C → {temp_final}°C at t={step_time}s:")
        print(f"    Duration: {stats['duration']:.0f} s ({stats['duration']/60:.1f} min)")
        print(f"    Mean: {stats['temp_mean_c']:.1f} °C")
        print(f"    Range: [{stats['temp_min_c']:.1f}, {stats['temp_max_c']:.1f}] °C")
        print(f"    Step magnitude: {abs(temp_final - temp_init):.1f} °C")
        
        # Plot
        save_path = output_dir / f"step32_temp_step_{label}.png"
        gen.plot_profile(
            time, temp_k,
            title=f"Step Change: {temp_init}°C → {temp_final}°C",
            save_path=str(save_path)
        )
    
    # ========================================================================
    # Test 3: Sinusoidal Daily Cycles
    # ========================================================================
    print("\n" + "-"*80)
    print("Generating sinusoidal daily temperature cycles...")
    
    # Simulate 48 hours (2 full days)
    duration_2days = 2 * 86400  # 2 * 24 hours
    
    sinusoidal_profiles = [
        (25, 5, "moderate"),   # Moderate climate: 20-30°C
        (35, 15, "desert"),    # Desert: 20-50°C
        (10, 10, "cold")       # Cold climate: 0-20°C
    ]
    
    for temp_mean, temp_amp, label in sinusoidal_profiles:
        time, temp_k = gen.generate_sinusoidal(
            duration=duration_2days,
            temp_mean_c=temp_mean,
            temp_amplitude_c=temp_amp,
            period=86400.0,  # 24 hours
            phase=-np.pi/2   # Peak at noon (t=12h)
        )
        stats = gen.get_profile_stats(time, temp_k)
        
        print(f"\n  Sinusoidal {label} (mean={temp_mean}°C, amp={temp_amp}°C):")
        print(f"    Duration: {stats['duration']:.0f} s ({stats['duration']/3600:.1f} hours)")
        print(f"    Mean: {stats['temp_mean_c']:.1f} °C")
        print(f"    Range: [{stats['temp_min_c']:.1f}, {stats['temp_max_c']:.1f}] °C")
        print(f"    Std: {stats['temp_std_c']:.1f} °C")
        
        # Plot
        save_path = output_dir / f"step32_temp_sinusoidal_{label}.png"
        gen.plot_profile(
            time, temp_k,
            title=f"Daily Cycle: {label.capitalize()} Climate",
            save_path=str(save_path)
        )
    
    # ========================================================================
    # Test 4: Multi-Step Sequence (Real-World Scenario)
    # ========================================================================
    print("\n" + "-"*80)
    print("Generating multi-step sequence (drive scenario)...")
    
    # Scenario: Morning commute → Work parking → Afternoon drive → Home
    sequence = [
        (0, 15),        # 0-10 min: Morning (cool)
        (600, 25),      # 10-20 min: Warming up (driving)
        (1200, 40),     # 20-40 min: Hot parking lot (sun exposure)
        (2400, 35),     # 40-50 min: Afternoon drive (AC on)
        (3000, 20)      # 50-60 min: Evening (cooling down)
    ]
    
    time, temp_k = gen.generate_multi_step(duration=3600, temperature_sequence=sequence)
    stats = gen.get_profile_stats(time, temp_k)
    
    print(f"\n  Multi-step drive scenario:")
    print(f"    Duration: {stats['duration']:.0f} s ({stats['duration']/60:.1f} min)")
    print(f"    Mean: {stats['temp_mean_c']:.1f} °C")
    print(f"    Range: [{stats['temp_min_c']:.1f}, {stats['temp_max_c']:.1f}] °C")
    print(f"    Steps: {len(sequence)}")
    
    # Plot
    save_path = output_dir / "step32_temp_multistep_drive.png"
    gen.plot_profile(
        time, temp_k,
        title="Multi-Step: Real-World Drive Scenario",
        save_path=str(save_path)
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3.2 COMPLETE ✓")
    print("="*80)
    print(f"\nGenerated {3 + 3 + 3 + 1} = 10 temperature profiles:")
    print("  • 3 constant temperatures (0°C, 25°C, 50°C)")
    print("  • 3 step changes (thermal shock scenarios)")
    print("  • 3 sinusoidal cycles (daily variation)")
    print("  • 1 multi-step sequence (real-world drive)")
    print(f"\nAll plots saved to: {output_dir}/")
    print("\nNext: Step 3.3 - Batch Physics Simulator (ECM + EETM)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
