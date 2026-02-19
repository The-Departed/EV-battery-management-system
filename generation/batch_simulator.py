"""
Step 3.3: Batch Physics Simulator
==================================

Combines drive cycles I(t) and temperature profiles Tamb(t) to run
ECM + EETM physics simulations and generate synthetic datasets.

Simulation pipeline:
1. Load drive cycle I(t) and temperature profile Tamb(t)
2. Run ECM: I(t) → V(t), SOC(t)
3. Run EETM: I(t), Tamb(t) → Q_gen(t), Ts(t), Tc(t)
4. Collect all signals: [I, V, SOC, Ts, Tc, Tamb, Q_gen]
5. Save to structured dataset

Pure physics - no ML, no curve fitting.

Author: Battery Modeling Pipeline
Date: 2026-01-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# Import Phase 1 ECM
import sys
sys.path.append(str(Path(__file__).parent.parent))
from ecm.model.ecm_2rc import ECM2RC
from ecm.ocv.ocv_model import OCVModel

# Import Phase 2 EETM
from eetm.model.eetm_model import EETM2ndOrder


def compute_heat_generation(current: float, voltage: float, ocv: float) -> float:
    """
    Compute heat generation from electrical quantities.
    
    Q = I * (OCV - V)
    
    Where:
    - I is current (A, negative for discharge)
    - OCV is open circuit voltage (V)
    - V is terminal voltage (V)
    
    This accounts for all resistive losses in the ECM.
    
    Parameters
    ----------
    current : float
        Current [A] (negative = discharge)
    voltage : float
        Terminal voltage [V]
    ocv : float
        Open circuit voltage [V]
    
    Returns
    -------
    heat : float
        Heat generation rate [W]
    """
    return current * (ocv - voltage)


@dataclass
class SimulationConfig:
    """Configuration for batch simulation."""
    
    # ECM parameters (from Phase 1 identification)
    R0: float = 0.001127  # Ohmic resistance [Ω]
    R1: float = 0.009899  # RC pair 1 resistance [Ω]
    C1: float = 15290.0   # RC pair 1 capacitance [F]
    R2: float = 0.030116  # RC pair 2 resistance [Ω]
    C2: float = 3236.0    # RC pair 2 capacitance [F]
    
    # EETM parameters (from Phase 2 identification)
    Rin: float = 3.0      # Internal thermal resistance [K/W]
    Rout: float = 15.0    # External thermal resistance [K/W]
    Cc: float = 30.0      # Core heat capacity [J/K]
    Cs: float = 15.0      # Surface heat capacity [J/K]
    
    # Battery cell properties
    capacity_ah: float = 3.0      # Nominal capacity [Ah]
    voltage_nominal: float = 3.7  # Nominal voltage [V]
    soc_initial: float = 0.8      # Initial SOC [0-1]
    
    # Initial conditions
    temp_initial_k: float = 298.15  # Initial temperature [K] (25°C)
    
    # Simulation settings
    dt: float = 1.0  # Time step [s]


class BatchPhysicsSimulator:
    """
    Batch physics simulator combining ECM + EETM models.
    
    Generates synthetic datasets by running forward physics simulations
    with various drive cycles and temperature profiles.
    """
    
    def __init__(self, config: SimulationConfig = SimulationConfig()):
        """
        Initialize batch simulator.
        
        Parameters
        ----------
        config : SimulationConfig
            Simulation configuration with ECM and EETM parameters
        """
        self.config = config
        
        # Create simple OCV model (polynomial fit for typical Li-ion)
        # OCV ≈ 3.0 + 1.0*SOC (simplified linear model)
        # For better results, use actual OCV-SOC data
        soc_data = np.linspace(0, 1, 11)
        ocv_data = 3.0 + 1.0 * soc_data  # Simple linear: 3.0V @ SOC=0, 4.0V @ SOC=1
        self.ocv_model = OCVModel(method='polynomial', degree=3)
        self.ocv_model.fit(soc_data, ocv_data)
        
        # Initialize ECM model
        self.ecm = ECM2RC(
            R0=config.R0,
            R1=config.R1,
            C1=config.C1,
            R2=config.R2,
            C2=config.C2,
            capacity=config.capacity_ah,
            ocv_model=self.ocv_model
        )
        
        # Initialize EETM model
        self.eetm = EETM2ndOrder(
            Rin=config.Rin,
            Rout=config.Rout,
            Cc=config.Cc,
            Cs=config.Cs
        )
        
        print(f"✓ Batch simulator initialized")
        print(f"  ECM: 2-RC (R0={config.R0:.6f}Ω, R1={config.R1:.6f}Ω, R2={config.R2:.6f}Ω)")
        print(f"  EETM: 2-State (Rin={config.Rin:.1f}K/W, Rout={config.Rout:.1f}K/W)")
    
    def simulate(
        self,
        time: np.ndarray,
        current: np.ndarray,
        temp_ambient_k: np.ndarray,
        soc_initial: Optional[float] = None,
        temp_initial_k: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Run full ECM + EETM simulation.
        
        Parameters
        ----------
        time : np.ndarray
            Time vector [s]
        current : np.ndarray
            Current profile [A] (negative = discharge)
        temp_ambient_k : np.ndarray
            Ambient temperature profile [K]
        soc_initial : float, optional
            Initial SOC [0-1] (default: from config)
        temp_initial_k : float, optional
            Initial temperature [K] (default: from config)
        
        Returns
        -------
        results : dict
            Dictionary containing:
            - time: Time vector [s]
            - current: Current [A]
            - voltage: Terminal voltage [V]
            - soc: State of charge [0-1]
            - temp_surface_k: Surface temperature [K]
            - temp_core_k: Core temperature [K]
            - temp_ambient_k: Ambient temperature [K]
            - heat_generation: Heat generation rate [W]
            - power: Electrical power [W]
        """
        n_samples = len(time)
        
        # Initialize states
        if soc_initial is None:
            soc_initial = self.config.soc_initial
        if temp_initial_k is None:
            temp_initial_k = self.config.temp_initial_k
        
        # ============================================================
        # Step 1: Run ECM simulation
        # ============================================================
        ecm_results = self.ecm.simulate(
            time=time,
            current=current,
            soc_init=soc_initial
        )
        
        voltage = ecm_results['V_terminal']
        soc = ecm_results['SOC']
        ocv = ecm_results['OCV']
        
        # ============================================================
        # Step 2: Compute heat generation from ECM results
        # ============================================================
        heat_generation = np.zeros(n_samples)
        for k in range(n_samples):
            heat_generation[k] = compute_heat_generation(
                current=current[k],
                voltage=voltage[k],
                ocv=ocv[k]
            )
        
        # ============================================================
        # Step 3: Run EETM thermal simulation
        # ============================================================
        # Convert temperatures to Celsius for EETM (it works in Celsius internally)
        C_TO_K = 273.15
        temp_ambient_c = temp_ambient_k - C_TO_K
        temp_initial_c = temp_initial_k - C_TO_K
        
        eetm_results = self.eetm.simulate(
            time=time,
            Q=heat_generation,
            Tamb=temp_ambient_c,
            Tc_init=temp_initial_c,
            Ts_init=temp_initial_c
        )
        
        # Convert back to Kelvin
        temp_surface_k = eetm_results['Ts'] + C_TO_K
        temp_core_k = eetm_results['Tc'] + C_TO_K
        
        # ============================================================
        # Step 4: Compute electrical power
        # ============================================================
        power = voltage * current
        
        # Package results
        results = {
            'time': time,
            'current': current,
            'voltage': voltage,
            'soc': soc,
            'temp_surface_k': temp_surface_k,
            'temp_core_k': temp_core_k,
            'temp_ambient_k': temp_ambient_k,
            'heat_generation': heat_generation,
            'power': power
        }
        
        return results
    
    def compute_statistics(self, results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute statistics for simulation results.
        
        Parameters
        ----------
        results : dict
            Simulation results from simulate()
        
        Returns
        -------
        stats : dict
            Statistics dictionary
        """
        C_TO_K = 273.15
        
        stats = {
            # Time
            'duration_s': float(results['time'][-1] - results['time'][0]),
            'samples': len(results['time']),
            
            # Current
            'current_mean': float(np.mean(results['current'])),
            'current_rms': float(np.sqrt(np.mean(results['current']**2))),
            'current_min': float(np.min(results['current'])),
            'current_max': float(np.max(results['current'])),
            
            # Voltage
            'voltage_mean': float(np.mean(results['voltage'])),
            'voltage_min': float(np.min(results['voltage'])),
            'voltage_max': float(np.max(results['voltage'])),
            
            # SOC
            'soc_initial': float(results['soc'][0]),
            'soc_final': float(results['soc'][-1]),
            'soc_delta': float(results['soc'][-1] - results['soc'][0]),
            
            # Temperature (convert to Celsius)
            'temp_surface_mean_c': float(np.mean(results['temp_surface_k']) - C_TO_K),
            'temp_surface_min_c': float(np.min(results['temp_surface_k']) - C_TO_K),
            'temp_surface_max_c': float(np.max(results['temp_surface_k']) - C_TO_K),
            'temp_core_mean_c': float(np.mean(results['temp_core_k']) - C_TO_K),
            'temp_core_min_c': float(np.min(results['temp_core_k']) - C_TO_K),
            'temp_core_max_c': float(np.max(results['temp_core_k']) - C_TO_K),
            'temp_ambient_mean_c': float(np.mean(results['temp_ambient_k']) - C_TO_K),
            'temp_ambient_min_c': float(np.min(results['temp_ambient_k']) - C_TO_K),
            'temp_ambient_max_c': float(np.max(results['temp_ambient_k']) - C_TO_K),
            
            # Heat generation
            'heat_gen_mean': float(np.mean(results['heat_generation'])),
            'heat_gen_max': float(np.max(results['heat_generation'])),
            
            # Power and energy
            'power_mean': float(np.mean(results['power'])),
            'energy_total_wh': float(np.sum(results['power']) * self.config.dt / 3600),
        }
        
        return stats
    
    def save_results(
        self,
        results: Dict[str, np.ndarray],
        save_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save simulation results to CSV.
        
        Parameters
        ----------
        results : dict
            Simulation results from simulate()
        save_path : str
            Path to save CSV file
        metadata : dict, optional
            Additional metadata to save as comments
        """
        C_TO_K = 273.15
        
        # Convert to DataFrame (temperatures in Celsius for readability)
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
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        print(f"✓ Results saved: {save_path} ({len(df)} samples)")
    
    def plot_results(
        self,
        results: Dict[str, np.ndarray],
        title: str = "Simulation Results",
        save_path: Optional[str] = None
    ):
        """
        Plot simulation results (4-panel figure).
        
        Parameters
        ----------
        results : dict
            Simulation results from simulate()
        title : str
            Figure title
        save_path : str, optional
            Path to save figure
        """
        C_TO_K = 273.15
        time_min = results['time'] / 60
        
        # Create 4-panel figure
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # ================================================================
        # Panel 1: Current and Voltage
        # ================================================================
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(time_min, results['current'], 'b-', linewidth=1.5, label='Current')
        line2 = ax1_twin.plot(time_min, results['voltage'], 'r-', linewidth=1.5, label='Voltage')
        
        ax1.set_xlabel('Time [min]', fontsize=11)
        ax1.set_ylabel('Current [A]', fontsize=11, color='b')
        ax1_twin.set_ylabel('Voltage [V]', fontsize=11, color='r')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'{title} - Electrical Behavior', fontsize=12, fontweight='bold')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=10)
        
        # ================================================================
        # Panel 2: State of Charge
        # ================================================================
        ax2 = axes[1]
        ax2.plot(time_min, results['soc'] * 100, 'g-', linewidth=1.5)
        ax2.set_xlabel('Time [min]', fontsize=11)
        ax2.set_ylabel('SOC [%]', fontsize=11)
        ax2.set_title('State of Charge', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        # Annotate SOC change
        soc_delta = (results['soc'][-1] - results['soc'][0]) * 100
        ax2.text(
            0.02, 0.98,
            f"ΔSOC = {soc_delta:.1f}%\n"
            f"Initial: {results['soc'][0]*100:.1f}%\n"
            f"Final: {results['soc'][-1]*100:.1f}%",
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
        )
        
        # ================================================================
        # Panel 3: Temperatures
        # ================================================================
        ax3 = axes[2]
        ax3.plot(time_min, results['temp_core_k'] - C_TO_K, 'r-', 
                linewidth=1.5, label='Core (Tc)')
        ax3.plot(time_min, results['temp_surface_k'] - C_TO_K, 'orange', 
                linewidth=1.5, label='Surface (Ts)')
        ax3.plot(time_min, results['temp_ambient_k'] - C_TO_K, 'b--', 
                linewidth=1.5, label='Ambient (Tamb)')
        ax3.set_xlabel('Time [min]', fontsize=11)
        ax3.set_ylabel('Temperature [°C]', fontsize=11)
        ax3.set_title('Thermal States', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=10)
        
        # Annotate max temps
        temp_core_max = np.max(results['temp_core_k']) - C_TO_K
        temp_surface_max = np.max(results['temp_surface_k']) - C_TO_K
        ax3.text(
            0.02, 0.98,
            f"Max Tc: {temp_core_max:.1f}°C\n"
            f"Max Ts: {temp_surface_max:.1f}°C\n"
            f"ΔT (Tc-Ts): {temp_core_max - temp_surface_max:.1f}°C",
            transform=ax3.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        # ================================================================
        # Panel 4: Heat Generation and Power
        # ================================================================
        ax4 = axes[3]
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(time_min, results['heat_generation'], 'r-', 
                        linewidth=1.5, label='Heat Generation')
        line2 = ax4_twin.plot(time_min, results['power'], 'purple', 
                             linewidth=1.5, alpha=0.7, label='Power')
        
        ax4.set_xlabel('Time [min]', fontsize=11)
        ax4.set_ylabel('Heat Generation [W]', fontsize=11, color='r')
        ax4_twin.set_ylabel('Power [W]', fontsize=11, color='purple')
        ax4.tick_params(axis='y', labelcolor='r')
        ax4_twin.tick_params(axis='y', labelcolor='purple')
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Heat Generation and Electrical Power', fontsize=12, fontweight='bold')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Demonstration of batch physics simulation."""
    
    print("\n" + "="*80)
    print("STEP 3.3: BATCH PHYSICS SIMULATOR (ECM + EETM)")
    print("="*80 + "\n")
    
    # Import drive cycle and temperature profile generators
    from generation.drive_cycles import DriveCycleLoader
    from generation.temperature_profiles import TemperatureProfileGenerator
    
    # Create output directories
    output_dir = Path("results/plots")
    data_dir = Path("results/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generators
    drive_loader = DriveCycleLoader()
    temp_gen = TemperatureProfileGenerator(dt=1.0)
    
    # Initialize simulator
    config = SimulationConfig()
    simulator = BatchPhysicsSimulator(config)
    
    print()
    
    # ========================================================================
    # Simulation 1: UDDS @ 25°C (Standard Test Condition)
    # ========================================================================
    print("="*80)
    print("Simulation 1: UDDS Drive Cycle @ 25°C (Standard Condition)")
    print("="*80)
    
    # Load drive cycle
    time_udds, current_udds = drive_loader.load_cycle('UDDS', scale_factor=1.0)
    
    # Generate constant temperature
    _, temp_25c = temp_gen.generate_constant(
        duration=time_udds[-1],
        temperature_c=25
    )
    
    # Run simulation
    print("Running ECM + EETM simulation...")
    results_1 = simulator.simulate(
        time=time_udds,
        current=current_udds,
        temp_ambient_k=temp_25c,
        soc_initial=0.8
    )
    
    # Compute statistics
    stats_1 = simulator.compute_statistics(results_1)
    print(f"\nSimulation 1 Statistics:")
    print(f"  Duration: {stats_1['duration_s']:.0f} s ({stats_1['duration_s']/60:.1f} min)")
    print(f"  Current RMS: {stats_1['current_rms']:.3f} A")
    print(f"  SOC: {stats_1['soc_initial']:.1%} → {stats_1['soc_final']:.1%} (Δ={stats_1['soc_delta']:.1%})")
    print(f"  Voltage: {stats_1['voltage_min']:.3f} - {stats_1['voltage_max']:.3f} V")
    print(f"  Temp (Ts): {stats_1['temp_surface_min_c']:.2f} - {stats_1['temp_surface_max_c']:.2f}°C")
    print(f"  Temp (Tc): {stats_1['temp_core_min_c']:.2f} - {stats_1['temp_core_max_c']:.2f}°C")
    print(f"  Heat gen (mean): {stats_1['heat_gen_mean']:.3f} W")
    print(f"  Energy: {stats_1['energy_total_wh']:.3f} Wh")
    
    # Save results
    simulator.save_results(
        results_1,
        save_path=str(data_dir / "step33_sim1_udds_25c.csv")
    )
    
    # Plot
    simulator.plot_results(
        results_1,
        title="UDDS @ 25°C",
        save_path=str(output_dir / "step33_sim1_udds_25c.png")
    )
    
    # ========================================================================
    # Simulation 2: US06 @ 50°C (Aggressive + Hot)
    # ========================================================================
    print("\n" + "="*80)
    print("Simulation 2: US06 Drive Cycle @ 50°C (Aggressive + Hot)")
    print("="*80)
    
    # Load aggressive drive cycle
    time_us06, current_us06 = drive_loader.load_cycle('US06', scale_factor=1.0)
    
    # Generate hot temperature
    _, temp_50c = temp_gen.generate_constant(
        duration=time_us06[-1],
        temperature_c=50
    )
    
    # Run simulation
    print("Running ECM + EETM simulation...")
    results_2 = simulator.simulate(
        time=time_us06,
        current=current_us06,
        temp_ambient_k=temp_50c,
        soc_initial=0.9
    )
    
    # Compute statistics
    stats_2 = simulator.compute_statistics(results_2)
    print(f"\nSimulation 2 Statistics:")
    print(f"  Duration: {stats_2['duration_s']:.0f} s ({stats_2['duration_s']/60:.1f} min)")
    print(f"  Current RMS: {stats_2['current_rms']:.3f} A")
    print(f"  SOC: {stats_2['soc_initial']:.1%} → {stats_2['soc_final']:.1%} (Δ={stats_2['soc_delta']:.1%})")
    print(f"  Voltage: {stats_2['voltage_min']:.3f} - {stats_2['voltage_max']:.3f} V")
    print(f"  Temp (Ts): {stats_2['temp_surface_min_c']:.2f} - {stats_2['temp_surface_max_c']:.2f}°C")
    print(f"  Temp (Tc): {stats_2['temp_core_min_c']:.2f} - {stats_2['temp_core_max_c']:.2f}°C")
    print(f"  Heat gen (mean): {stats_2['heat_gen_mean']:.3f} W")
    print(f"  Energy: {stats_2['energy_total_wh']:.3f} Wh")
    
    # Save results
    simulator.save_results(
        results_2,
        save_path=str(data_dir / "step33_sim2_us06_50c.csv")
    )
    
    # Plot
    simulator.plot_results(
        results_2,
        title="US06 @ 50°C",
        save_path=str(output_dir / "step33_sim2_us06_50c.png")
    )
    
    # ========================================================================
    # Simulation 3: HWFET @ 0°C (Highway + Cold)
    # ========================================================================
    print("\n" + "="*80)
    print("Simulation 3: HWFET Drive Cycle @ 0°C (Highway + Cold)")
    print("="*80)
    
    # Load highway drive cycle
    time_hwfet, current_hwfet = drive_loader.load_cycle('HWFET', scale_factor=1.0)
    
    # Generate cold temperature
    _, temp_0c = temp_gen.generate_constant(
        duration=time_hwfet[-1],
        temperature_c=0
    )
    
    # Run simulation
    print("Running ECM + EETM simulation...")
    results_3 = simulator.simulate(
        time=time_hwfet,
        current=current_hwfet,
        temp_ambient_k=temp_0c,
        soc_initial=1.0
    )
    
    # Compute statistics
    stats_3 = simulator.compute_statistics(results_3)
    print(f"\nSimulation 3 Statistics:")
    print(f"  Duration: {stats_3['duration_s']:.0f} s ({stats_3['duration_s']/60:.1f} min)")
    print(f"  Current RMS: {stats_3['current_rms']:.3f} A")
    print(f"  SOC: {stats_3['soc_initial']:.1%} → {stats_3['soc_final']:.1%} (Δ={stats_3['soc_delta']:.1%})")
    print(f"  Voltage: {stats_3['voltage_min']:.3f} - {stats_3['voltage_max']:.3f} V")
    print(f"  Temp (Ts): {stats_3['temp_surface_min_c']:.2f} - {stats_3['temp_surface_max_c']:.2f}°C")
    print(f"  Temp (Tc): {stats_3['temp_core_min_c']:.2f} - {stats_3['temp_core_max_c']:.2f}°C")
    print(f"  Heat gen (mean): {stats_3['heat_gen_mean']:.3f} W")
    print(f"  Energy: {stats_3['energy_total_wh']:.3f} Wh")
    
    # Save results
    simulator.save_results(
        results_3,
        save_path=str(data_dir / "step33_sim3_hwfet_0c.csv")
    )
    
    # Plot
    simulator.plot_results(
        results_3,
        title="HWFET @ 0°C",
        save_path=str(output_dir / "step33_sim3_hwfet_0c.png")
    )
    
    # ========================================================================
    # Simulation 4: UDDS with Daily Temperature Cycle
    # ========================================================================
    print("\n" + "="*80)
    print("Simulation 4: UDDS with Daily Temperature Variation (25±10°C)")
    print("="*80)
    
    # Load drive cycle
    time_udds2, current_udds2 = drive_loader.load_cycle('UDDS', scale_factor=1.0)
    
    # Generate sinusoidal temperature (compressed to fit UDDS duration)
    _, temp_daily = temp_gen.generate_sinusoidal(
        duration=time_udds2[-1],
        temp_mean_c=25,
        temp_amplitude_c=10,
        period=time_udds2[-1],  # One full cycle over UDDS duration
        phase=-np.pi/2
    )
    
    # Run simulation
    print("Running ECM + EETM simulation...")
    results_4 = simulator.simulate(
        time=time_udds2,
        current=current_udds2,
        temp_ambient_k=temp_daily,
        soc_initial=0.7
    )
    
    # Compute statistics
    stats_4 = simulator.compute_statistics(results_4)
    print(f"\nSimulation 4 Statistics:")
    print(f"  Duration: {stats_4['duration_s']:.0f} s ({stats_4['duration_s']/60:.1f} min)")
    print(f"  Current RMS: {stats_4['current_rms']:.3f} A")
    print(f"  SOC: {stats_4['soc_initial']:.1%} → {stats_4['soc_final']:.1%} (Δ={stats_4['soc_delta']:.1%})")
    print(f"  Voltage: {stats_4['voltage_min']:.3f} - {stats_4['voltage_max']:.3f} V")
    print(f"  Temp (Ts): {stats_4['temp_surface_min_c']:.2f} - {stats_4['temp_surface_max_c']:.2f}°C")
    print(f"  Temp (Tc): {stats_4['temp_core_min_c']:.2f} - {stats_4['temp_core_max_c']:.2f}°C")
    print(f"  Temp (Tamb): {stats_4['temp_ambient_min_c']:.2f} - {stats_4['temp_ambient_max_c']:.2f}°C")
    print(f"  Heat gen (mean): {stats_4['heat_gen_mean']:.3f} W")
    print(f"  Energy: {stats_4['energy_total_wh']:.3f} Wh")
    
    # Save results
    simulator.save_results(
        results_4,
        save_path=str(data_dir / "step33_sim4_udds_daily.csv")
    )
    
    # Plot
    simulator.plot_results(
        results_4,
        title="UDDS with Daily Temperature Cycle",
        save_path=str(output_dir / "step33_sim4_udds_daily.png")
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3.3 COMPLETE ✓")
    print("="*80)
    print(f"\nGenerated 4 physics-based simulations:")
    print("  1. UDDS @ 25°C (standard test)")
    print("  2. US06 @ 50°C (aggressive + hot)")
    print("  3. HWFET @ 0°C (highway + cold)")
    print("  4. UDDS with daily temperature cycle (25±10°C)")
    print(f"\nDatasets saved to: {data_dir}/")
    print(f"Plots saved to: {output_dir}/")
    print("\nEach dataset contains:")
    print("  • Time, Current, Voltage, SOC")
    print("  • Surface temperature (Ts)")
    print("  • Core temperature (Tc)")
    print("  • Ambient temperature (Tamb)")
    print("  • Heat generation (Q_gen)")
    print("  • Electrical power")
    print("\nNext: Step 3.4 - Sensor Noise Injection")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
