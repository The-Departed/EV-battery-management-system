"""
Step 3.4: Sensor Noise Injection
=================================

Add realistic sensor noise to clean physics-based simulation data.

Noise types:
1. Gaussian noise (measurement uncertainty)
2. Quantization noise (ADC resolution)
3. Bias/drift (sensor calibration errors)
4. Outliers (intermittent faults)

Purpose: Create robust ML training data that reflects real sensor imperfections.

Author: Battery Modeling Pipeline
Date: 2026-01-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NoiseConfig:
    """Configuration for sensor noise injection."""
    
    # Gaussian noise (measurement uncertainty)
    current_noise_std: float = 0.01      # Current sensor noise [A]
    voltage_noise_std: float = 0.002     # Voltage sensor noise [V]
    temp_noise_std: float = 0.1          # Temperature sensor noise [°C]
    
    # Quantization (ADC resolution)
    current_resolution: float = 0.001    # Current ADC resolution [A]
    voltage_resolution: float = 0.001    # Voltage ADC resolution [V]
    temp_resolution: float = 0.01        # Temperature ADC resolution [°C]
    
    # Bias and drift
    current_bias: float = 0.0            # Current sensor bias [A]
    voltage_bias: float = 0.0            # Voltage sensor bias [V]
    temp_bias: float = 0.0               # Temperature sensor bias [°C]
    
    # Outliers
    outlier_probability: float = 0.001   # Probability of outlier
    outlier_magnitude: float = 3.0       # Outlier magnitude (multiples of std)
    
    # Random seed for reproducibility
    random_seed: Optional[int] = 42


class SensorNoiseInjector:
    """
    Inject realistic sensor noise into clean simulation data.
    
    Simulates real-world sensor imperfections:
    - Measurement noise (Gaussian)
    - Quantization effects (ADC)
    - Sensor bias and drift
    - Occasional outliers
    """
    
    def __init__(self, config: NoiseConfig = NoiseConfig()):
        """
        Initialize noise injector.
        
        Parameters
        ----------
        config : NoiseConfig
            Noise configuration parameters
        """
        self.config = config
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        print(f"✓ Sensor noise injector initialized")
        print(f"  Current noise: {config.current_noise_std:.4f} A (std)")
        print(f"  Voltage noise: {config.voltage_noise_std:.4f} V (std)")
        print(f"  Temperature noise: {config.temp_noise_std:.2f} °C (std)")
    
    def add_gaussian_noise(
        self,
        signal: np.ndarray,
        noise_std: float
    ) -> np.ndarray:
        """
        Add Gaussian (normal) noise to signal.
        
        Simulates random measurement uncertainty.
        
        Parameters
        ----------
        signal : np.ndarray
            Clean signal
        noise_std : float
            Standard deviation of noise
        
        Returns
        -------
        noisy_signal : np.ndarray
            Signal with added Gaussian noise
        """
        noise = np.random.normal(0, noise_std, size=signal.shape)
        return signal + noise
    
    def add_quantization_noise(
        self,
        signal: np.ndarray,
        resolution: float
    ) -> np.ndarray:
        """
        Add quantization noise (ADC discretization).
        
        Simulates analog-to-digital conversion with finite resolution.
        
        Parameters
        ----------
        signal : np.ndarray
            Clean signal
        resolution : float
            ADC resolution (LSB size)
        
        Returns
        -------
        quantized_signal : np.ndarray
            Quantized signal
        """
        return np.round(signal / resolution) * resolution
    
    def add_bias(
        self,
        signal: np.ndarray,
        bias: float
    ) -> np.ndarray:
        """
        Add constant bias to signal.
        
        Simulates sensor calibration offset.
        
        Parameters
        ----------
        signal : np.ndarray
            Clean signal
        bias : float
            Constant bias to add
        
        Returns
        -------
        biased_signal : np.ndarray
            Signal with added bias
        """
        return signal + bias
    
    def add_outliers(
        self,
        signal: np.ndarray,
        probability: float,
        magnitude: float,
        noise_std: float
    ) -> np.ndarray:
        """
        Add occasional outliers to signal.
        
        Simulates intermittent sensor faults or interference.
        
        Parameters
        ----------
        signal : np.ndarray
            Clean signal
        probability : float
            Probability of outlier at each sample (0-1)
        magnitude : float
            Outlier magnitude (multiples of noise std)
        noise_std : float
            Noise standard deviation for scaling
        
        Returns
        -------
        signal_with_outliers : np.ndarray
            Signal with added outliers
        """
        noisy_signal = signal.copy()
        n_samples = len(signal)
        
        # Random outlier locations
        outlier_mask = np.random.rand(n_samples) < probability
        n_outliers = np.sum(outlier_mask)
        
        if n_outliers > 0:
            # Random outlier directions and magnitudes
            outlier_values = np.random.randn(n_outliers) * magnitude * noise_std
            noisy_signal[outlier_mask] += outlier_values
        
        return noisy_signal
    
    def inject_noise_full(
        self,
        signal: np.ndarray,
        noise_std: float,
        resolution: float,
        bias: float,
        add_outliers: bool = True
    ) -> np.ndarray:
        """
        Apply full noise model (Gaussian + quantization + bias + outliers).
        
        Parameters
        ----------
        signal : np.ndarray
            Clean signal
        noise_std : float
            Gaussian noise standard deviation
        resolution : float
            ADC resolution
        bias : float
            Sensor bias
        add_outliers : bool
            Whether to add outliers
        
        Returns
        -------
        noisy_signal : np.ndarray
            Signal with full noise model
        """
        # Step 1: Add Gaussian noise
        noisy = self.add_gaussian_noise(signal, noise_std)
        
        # Step 2: Add bias
        noisy = self.add_bias(noisy, bias)
        
        # Step 3: Add outliers (optional)
        if add_outliers:
            noisy = self.add_outliers(
                noisy,
                self.config.outlier_probability,
                self.config.outlier_magnitude,
                noise_std
            )
        
        # Step 4: Quantize (ADC effect)
        noisy = self.add_quantization_noise(noisy, resolution)
        
        return noisy
    
    def inject_dataset_noise(
        self,
        clean_data: Dict[str, np.ndarray],
        add_outliers: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Inject noise into full simulation dataset.
        
        Parameters
        ----------
        clean_data : dict
            Dictionary with clean simulation signals
            Expected keys: time, current, voltage, temp_surface, temp_core, temp_ambient
        add_outliers : bool
            Whether to add outliers to measurements
        
        Returns
        -------
        noisy_data : dict
            Dictionary with noisy signals
        """
        noisy_data = clean_data.copy()
        
        # Add noise to current
        if 'current' in clean_data:
            noisy_data['current_noisy'] = self.inject_noise_full(
                clean_data['current'],
                self.config.current_noise_std,
                self.config.current_resolution,
                self.config.current_bias,
                add_outliers=add_outliers
            )
        
        # Add noise to voltage
        if 'voltage' in clean_data:
            noisy_data['voltage_noisy'] = self.inject_noise_full(
                clean_data['voltage'],
                self.config.voltage_noise_std,
                self.config.voltage_resolution,
                self.config.voltage_bias,
                add_outliers=add_outliers
            )
        
        # Add noise to surface temperature
        if 'temp_surface' in clean_data:
            noisy_data['temp_surface_noisy'] = self.inject_noise_full(
                clean_data['temp_surface'],
                self.config.temp_noise_std,
                self.config.temp_resolution,
                self.config.temp_bias,
                add_outliers=add_outliers
            )
        
        # Add noise to ambient temperature
        if 'temp_ambient' in clean_data:
            noisy_data['temp_ambient_noisy'] = self.inject_noise_full(
                clean_data['temp_ambient'],
                self.config.temp_noise_std,
                self.config.temp_resolution,
                self.config.temp_bias,
                add_outliers=False  # Ambient typically more stable
            )
        
        # Core temperature is NOT measured (latent state)
        # SOC is NOT measured directly (estimated)
        # Heat generation is NOT measured (computed)
        # Power is computed from noisy I and V
        if 'current_noisy' in noisy_data and 'voltage_noisy' in noisy_data:
            noisy_data['power_noisy'] = (
                noisy_data['current_noisy'] * noisy_data['voltage_noisy']
            )
        
        return noisy_data
    
    def compute_noise_statistics(
        self,
        clean_signal: np.ndarray,
        noisy_signal: np.ndarray,
        signal_name: str = "Signal"
    ) -> Dict[str, float]:
        """
        Compute noise statistics.
        
        Parameters
        ----------
        clean_signal : np.ndarray
            Original clean signal
        noisy_signal : np.ndarray
            Noisy signal
        signal_name : str
            Name of signal for reporting
        
        Returns
        -------
        stats : dict
            Noise statistics
        """
        noise = noisy_signal - clean_signal
        
        stats = {
            'signal_name': signal_name,
            'noise_mean': float(np.mean(noise)),
            'noise_std': float(np.std(noise)),
            'noise_rms': float(np.sqrt(np.mean(noise**2))),
            'noise_max_abs': float(np.max(np.abs(noise))),
            'snr_db': float(20 * np.log10(np.std(clean_signal) / (np.std(noise) + 1e-10)))
        }
        
        return stats
    
    def plot_noise_comparison(
        self,
        time: np.ndarray,
        clean_signal: np.ndarray,
        noisy_signal: np.ndarray,
        title: str = "Noise Injection",
        ylabel: str = "Signal",
        save_path: Optional[str] = None
    ):
        """
        Plot clean vs noisy signal comparison.
        
        Parameters
        ----------
        time : np.ndarray
            Time vector
        clean_signal : np.ndarray
            Clean signal
        noisy_signal : np.ndarray
            Noisy signal
        title : str
            Plot title
        ylabel : str
            Y-axis label
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        time_min = time / 60
        noise = noisy_signal - clean_signal
        
        # Panel 1: Clean and noisy signals
        ax1 = axes[0]
        ax1.plot(time_min, clean_signal, 'b-', linewidth=1.5, 
                alpha=0.7, label='Clean (ground truth)')
        ax1.plot(time_min, noisy_signal, 'r-', linewidth=1, 
                alpha=0.6, label='Noisy (measured)')
        ax1.set_xlabel('Time [min]', fontsize=11)
        ax1.set_ylabel(ylabel, fontsize=11)
        ax1.set_title(f'{title} - Signal Comparison', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)
        
        # Panel 2: Noise (residual)
        ax2 = axes[1]
        ax2.plot(time_min, noise, 'g-', linewidth=0.8, alpha=0.7)
        ax2.axhline(0, color='k', linestyle='--', linewidth=1)
        ax2.fill_between(time_min, -3*np.std(noise), 3*np.std(noise), 
                         alpha=0.2, color='gray', label='±3σ')
        ax2.set_xlabel('Time [min]', fontsize=11)
        ax2.set_ylabel('Noise (Measured - True)', fontsize=11)
        ax2.set_title('Noise Residual', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=10)
        
        # Statistics text
        stats_text = (
            f"Noise μ: {np.mean(noise):.4f}\n"
            f"Noise σ: {np.std(noise):.4f}\n"
            f"Noise RMS: {np.sqrt(np.mean(noise**2)):.4f}\n"
            f"Max |error|: {np.max(np.abs(noise)):.4f}\n"
            f"SNR: {20*np.log10(np.std(clean_signal)/(np.std(noise)+1e-10)):.1f} dB"
        )
        ax2.text(
            0.02, 0.98, stats_text,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
        )
        
        # Panel 3: Noise histogram
        ax3 = axes[2]
        counts, bins, patches = ax3.hist(
            noise, bins=50, color='steelblue', 
            alpha=0.7, edgecolor='black', linewidth=0.5, density=True
        )
        
        # Overlay Gaussian fit
        mu, sigma = np.mean(noise), np.std(noise)
        x = np.linspace(bins[0], bins[-1], 100)
        ax3.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2),
                'r-', linewidth=2, label=f'Gaussian (μ={mu:.4f}, σ={sigma:.4f})')
        
        ax3.set_xlabel('Noise', fontsize=11)
        ax3.set_ylabel('Probability Density', fontsize=11)
        ax3.set_title('Noise Distribution', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Demonstration of sensor noise injection."""
    
    print("\n" + "="*80)
    print("STEP 3.4: SENSOR NOISE INJECTION")
    print("="*80 + "\n")
    
    # Create output directory
    output_dir = Path("results/plots")
    data_dir = Path("results/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Load clean simulation data from Step 3.3
    # ========================================================================
    print("Loading clean simulation data from Step 3.3...")
    
    # Load UDDS @ 25°C simulation
    clean_df = pd.read_csv(data_dir / "step33_sim1_udds_25c.csv")
    
    print(f"✓ Loaded {len(clean_df)} samples from UDDS @ 25°C simulation")
    print(f"  Columns: {list(clean_df.columns)}\n")
    
    # Extract signals
    time = clean_df['time_s'].values
    current_clean = clean_df['current_A'].values
    voltage_clean = clean_df['voltage_V'].values
    temp_surface_clean = clean_df['temp_surface_C'].values
    temp_ambient_clean = clean_df['temp_ambient_C'].values
    soc_clean = clean_df['soc'].values
    
    # ========================================================================
    # Initialize noise injector
    # ========================================================================
    config = NoiseConfig(
        # Realistic sensor noise levels
        current_noise_std=0.01,      # 10 mA std
        voltage_noise_std=0.002,     # 2 mV std
        temp_noise_std=0.1,          # 0.1°C std
        
        # Typical ADC resolutions
        current_resolution=0.001,    # 1 mA
        voltage_resolution=0.001,    # 1 mV
        temp_resolution=0.01,        # 0.01°C
        
        # Small biases (calibration errors)
        current_bias=0.005,          # 5 mA offset
        voltage_bias=-0.001,         # -1 mV offset
        temp_bias=0.2,               # 0.2°C offset
        
        # Occasional outliers
        outlier_probability=0.001,   # 0.1% of samples
        outlier_magnitude=3.0,       # 3-sigma outliers
        
        random_seed=42
    )
    
    injector = SensorNoiseInjector(config)
    print()
    
    # ========================================================================
    # Inject noise into dataset
    # ========================================================================
    print("Injecting sensor noise into clean data...")
    
    clean_data = {
        'time': time,
        'current': current_clean,
        'voltage': voltage_clean,
        'temp_surface': temp_surface_clean,
        'temp_ambient': temp_ambient_clean,
        'soc': soc_clean
    }
    
    noisy_data = injector.inject_dataset_noise(clean_data, add_outliers=True)
    
    print("✓ Noise injection complete\n")
    
    # ========================================================================
    # Compute and display noise statistics
    # ========================================================================
    print("="*80)
    print("NOISE STATISTICS")
    print("="*80 + "\n")
    
    # Current noise stats
    current_stats = injector.compute_noise_statistics(
        current_clean,
        noisy_data['current_noisy'],
        "Current"
    )
    print(f"Current Sensor:")
    print(f"  Noise mean:   {current_stats['noise_mean']:+.6f} A")
    print(f"  Noise std:    {current_stats['noise_std']:.6f} A")
    print(f"  Noise RMS:    {current_stats['noise_rms']:.6f} A")
    print(f"  Max |error|:  {current_stats['noise_max_abs']:.6f} A")
    print(f"  SNR:          {current_stats['snr_db']:.1f} dB\n")
    
    # Voltage noise stats
    voltage_stats = injector.compute_noise_statistics(
        voltage_clean,
        noisy_data['voltage_noisy'],
        "Voltage"
    )
    print(f"Voltage Sensor:")
    print(f"  Noise mean:   {voltage_stats['noise_mean']:+.6f} V")
    print(f"  Noise std:    {voltage_stats['noise_std']:.6f} V")
    print(f"  Noise RMS:    {voltage_stats['noise_rms']:.6f} V")
    print(f"  Max |error|:  {voltage_stats['noise_max_abs']:.6f} V")
    print(f"  SNR:          {voltage_stats['snr_db']:.1f} dB\n")
    
    # Temperature noise stats
    temp_stats = injector.compute_noise_statistics(
        temp_surface_clean,
        noisy_data['temp_surface_noisy'],
        "Temperature"
    )
    print(f"Temperature Sensor:")
    print(f"  Noise mean:   {temp_stats['noise_mean']:+.6f} °C")
    print(f"  Noise std:    {temp_stats['noise_std']:.6f} °C")
    print(f"  Noise RMS:    {temp_stats['noise_rms']:.6f} °C")
    print(f"  Max |error|:  {temp_stats['noise_max_abs']:.6f} °C")
    print(f"  SNR:          {temp_stats['snr_db']:.1f} dB\n")
    
    # ========================================================================
    # Save noisy dataset
    # ========================================================================
    print("="*80)
    print("Saving noisy dataset...")
    
    noisy_df = pd.DataFrame({
        'time_s': time,
        # Clean signals (ground truth)
        'current_clean_A': current_clean,
        'voltage_clean_V': voltage_clean,
        'temp_surface_clean_C': temp_surface_clean,
        'temp_ambient_clean_C': temp_ambient_clean,
        'soc_clean': soc_clean,
        # Noisy signals (measured)
        'current_meas_A': noisy_data['current_noisy'],
        'voltage_meas_V': noisy_data['voltage_noisy'],
        'temp_surface_meas_C': noisy_data['temp_surface_noisy'],
        'temp_ambient_meas_C': noisy_data['temp_ambient_noisy'],
        'power_meas_W': noisy_data['power_noisy']
    })
    
    noisy_csv_path = data_dir / "step34_noisy_udds_25c.csv"
    noisy_df.to_csv(noisy_csv_path, index=False)
    print(f"✓ Noisy dataset saved: {noisy_csv_path} ({len(noisy_df)} samples)\n")
    
    # ========================================================================
    # Visualize noise injection
    # ========================================================================
    print("="*80)
    print("Generating noise comparison plots...")
    
    # Plot current noise
    injector.plot_noise_comparison(
        time,
        current_clean,
        noisy_data['current_noisy'],
        title="Current Sensor Noise",
        ylabel="Current [A]",
        save_path=str(output_dir / "step34_noise_current.png")
    )
    
    # Plot voltage noise
    injector.plot_noise_comparison(
        time,
        voltage_clean,
        noisy_data['voltage_noisy'],
        title="Voltage Sensor Noise",
        ylabel="Voltage [V]",
        save_path=str(output_dir / "step34_noise_voltage.png")
    )
    
    # Plot temperature noise
    injector.plot_noise_comparison(
        time,
        temp_surface_clean,
        noisy_data['temp_surface_noisy'],
        title="Temperature Sensor Noise",
        ylabel="Temperature [°C]",
        save_path=str(output_dir / "step34_noise_temperature.png")
    )
    
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("="*80)
    print("STEP 3.4 COMPLETE ✓")
    print("="*80)
    print(f"\nNoise injection successful!")
    print(f"  • Current:     σ={config.current_noise_std:.4f}A, bias={config.current_bias:.4f}A")
    print(f"  • Voltage:     σ={config.voltage_noise_std:.4f}V, bias={config.voltage_bias:.4f}V")
    print(f"  • Temperature: σ={config.temp_noise_std:.2f}°C, bias={config.temp_bias:.2f}°C")
    print(f"  • Outliers:    {config.outlier_probability*100:.2f}% probability")
    print(f"\nDataset saved: {noisy_csv_path}")
    print(f"Plots saved: {output_dir}/step34_noise_*.png")
    print(f"\nNext: Step 3.5 - Dataset Builder (create training/validation splits)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
