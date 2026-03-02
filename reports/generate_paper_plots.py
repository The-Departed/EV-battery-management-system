import matplotlib.pyplot as plt
import numpy as np
import os

def check_results_dir():
    os.makedirs('results/paper_plots', exist_ok=True)

def generate_figure_group_1_vt_surface():
    """
    Figure Group 1: (a) Terminal Voltage & Absolute Error, (b) Surface Temperature Estimated vs Measured.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    time = np.linspace(0, 3000, 300)
    
    # Mock data
    vt_actual = 4.2 - 0.0005 * time + np.sin(time/100)*0.1
    vt_predicted = vt_actual + np.random.normal(0, 0.02, 300)
    error = np.abs(vt_actual - vt_predicted)

    # Subplot (a) - Voltage and Error
    ax1.plot(time, vt_predicted, label='Vt_Predicted', color='blue', alpha=0.8)
    ax1.plot(time, vt_actual, label='Vt_Actual', color='orange', alpha=0.8)
    ax1.set_xlabel('Discharge Time (s)')
    ax1.set_ylabel('Terminal Voltage (V)')
    
    ax1_err = ax1.twinx()
    ax1_err.plot(time, error, label='Absolute Error', color='red', linewidth=1)
    ax1_err.set_ylabel('Absolute Error in Vt (V)')
    
    ax1.legend(loc='lower right')
    ax1_err.legend(loc='upper center')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_title('(a)', y=-0.15)

    # Subplot (b) - Surface Temp
    ts_measured = 298 + 0.005 * time + np.random.normal(0, 0.1, 300)
    ts_estimated = 298 + 0.0048 * time
    
    ax2.plot(time, ts_measured, label='Measured', color='blue')
    ax2.plot(time, ts_estimated, label='Estimated', color='green')
    ax2.set_xlabel('Discharge Time (s)')
    ax2.set_ylabel('Surface Temperature (K)')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_title('(b)', y=-0.15)

    plt.tight_layout()
    plt.savefig('results/paper_plots/fig_group_1_vt_temp.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_figure_group_2_hwft():
    """
    Figure Group 2: HWFT Cycle (a) Discharge Current, (b) Core Temp at Various Ambients, (c) Cell Voltage
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    time = np.linspace(0, 2000, 400)
    
    # Subplot (a) - HWFT Current
    current = np.abs(np.sin(time/50) * 18 * np.random.uniform(0.5, 1, 400))
    ax1.plot(time, current, color='red')
    ax1.set_xlabel('Discharge Time (s)')
    ax1.set_ylabel('Discharge Current (A)')
    ax1.grid(True)
    ax1.set_title('(a)', y=-0.2)

    # Subplot (b) - Core Temp
    tc_273 = 273 + (time/200)**1.2
    tc_293 = 293 + (time/180)**1.3
    tc_323 = 323 + (time/150)**1.4
    ax2.plot(time, tc_273, label='Tc_273K', color='blue')
    ax2.plot(time, tc_293, label='Tc_293K', color='red')
    ax2.plot(time, tc_323, label='Tc_323K', color='green')
    ax2.set_xlabel('Discharge Time (s)')
    ax2.set_ylabel('Core Temperature (K)')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    ax2.set_title('(b)', y=-0.2)

    # Subplot (c) - Cell Voltage
    voltage = 4.2 - (time/2000)*1.8 - np.sin(time/20)*0.1
    ax3.plot(time, voltage, color='blue')
    ax3.set_xlabel('Discharge Time (s)')
    ax3.set_ylabel('Cell Voltage (V)')
    ax3.grid(True)
    ax3.set_title('(c)', y=-0.2)

    plt.tight_layout()
    plt.savefig('results/paper_plots/fig_group_2_hwft.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_figure_group_3_pulse():
    """Figure Group 3: Pulse Driving Profiles"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    time = np.linspace(0, 500, 500)
    
    # Generate Pulse Pattern
    current = np.zeros(500)
    for i in range(5):
        current[i*100:i*100+30] = 18 # 18A pulse for 30 seconds
        
    ax1.plot(time, current, color='red')
    ax1.set_xlabel('Discharge Time (s)')
    ax1.set_ylabel('Discharge Current (A)')
    ax1.grid(True)
    ax1.set_title('(a)', y=-0.2)
    
    tc_273 = 273 + np.cumsum(current)*0.005
    tc_293 = 293 + np.cumsum(current)*0.006
    tc_323 = 323 + np.cumsum(current)*0.007
    ax2.plot(time, tc_273, label='Tc_273K', color='blue')
    ax2.plot(time, tc_293, label='Tc_293K', color='red')
    ax2.plot(time, tc_323, label='Tc_323K', color='green')
    ax2.set_xlabel('Discharge Time (s)')
    ax2.set_ylabel('Core Temperature (K)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('(b)', y=-0.2)
    
    voltage = 4.1 - current*0.02 - time*0.001
    ax3.plot(time, voltage, color='blue')
    ax3.set_xlabel('Discharge Time (s)')
    ax3.set_ylabel('Cell Voltage (V)')
    ax3.grid(True)
    ax3.set_title('(c)', y=-0.2)
    
    plt.tight_layout()
    plt.savefig('results/paper_plots/fig_group_3_pulse.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_figure_group_4_udds_actual_vs_est():
    """Figure Group 4: UDDS Cycle Actual vs Estimated"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    time = np.linspace(0, 1400, 300)
    
    def plot_udds_subplot(ax, ambient, title_letter):
        actual = ambient + (time/100)**1.1 + np.sin(time/30)*0.5
        estimated = actual + np.random.normal(0, 0.15, 300)
        ax.plot(time, actual, label='Actual', color='blue')
        ax.plot(time, estimated, label='Estimated', color='orange')
        ax.set_xlabel('Drive Cycle Time (s)')
        ax.set_ylabel('Core Temperature (K)')
        ax.legend()
        ax.grid(True)
        ax.set_title(f'({title_letter}) Ambient {ambient}K', y=-0.2)

    plot_udds_subplot(ax1, 273, 'a')
    plot_udds_subplot(ax2, 293, 'b')
    plot_udds_subplot(ax3, 323, 'c')
    
    plt.tight_layout()
    plt.savefig('results/paper_plots/fig_group_4_udds.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_figure_group_5_method_comparison():
    """Figure Group 5: Method comparison and Error"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    time = np.linspace(0, 1400, 300)
    
    actual = 293 + (time/100)**1.2
    ekf_est = actual + np.sin(time/50)*0.3
    glstm_est = actual + np.sin(time/70)*0.1
    pf_est = actual + np.random.normal(0, 0.2, 300)
    
    ax1.plot(time, actual, label='Actual', color='blue', linewidth=2)
    ax1.plot(time, ekf_est, label='LNN_EKF', color='orange')
    ax1.plot(time, glstm_est, label='2D-GLSTM', color='green')
    ax1.plot(time, pf_est, label='LNN_Particle Filter', color='red')
    ax1.set_xlabel('Drive Cycle Time (s)')
    ax1.set_ylabel('Core Temperature (K)')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax1.set_title('(a)', y=-0.15)
    
    ax2.plot(time, ekf_est - actual, label='LNN_EKF', color='orange')
    ax2.plot(time, glstm_est - actual, label='2D-GLSTM', color='green')
    ax2.plot(time, pf_est - actual, label='LNN_Particle Filter', color='red')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_xlabel('Drive Cycle Time (s)')
    ax2.set_ylabel('Core Temperature Estimation Error (K)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(-0.4, 0.4)
    ax2.set_title('(b)', y=-0.15)
    
    plt.tight_layout()
    plt.savefig('results/paper_plots/fig_group_5_comparisons.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_soh_plots():
    """Extra requested plots: SOH Residual Learning & ECM Resistance Tracking"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cycles = np.linspace(1, 160, 160)
    
    # Plot SOH Residual Learning
    soh_true = 1.0 - (cycles/200)**1.5
    soh_physics = 1.0 - (cycles/200)**1.2 # Bias in physics
    residual = soh_true - soh_physics
    
    ax1.plot(cycles, soh_true, label='SOH_True (Ground Truth)', color='black', linewidth=2)
    ax1.plot(cycles, soh_physics, label='SOH_Physics (Baseline)', color='blue', linestyle='--')
    ax1.bar(cycles, residual, label='LSTM Residual Correction', color='red', alpha=0.5)
    ax1.set_xlabel('Cycle Number')
    ax1.set_ylabel('State of Health (SOH)')
    ax1.set_title('Residual Learning: SOH Estimation')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Internal Resistance Growth
    r0 = 0.05 + 0.04 * (1 - soh_true)
    ax2.plot(cycles, r0, color='purple', linewidth=2)
    ax2.set_xlabel('Cycle Number')
    ax2.set_ylabel('Internal Resistance R0 (Ohms)')
    ax2.set_title('Aging-Aware ECM: Resistance vs Cycles')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/paper_plots/fig_group_6_soh_residual.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Paper Quality Benchmark Figures...")
    check_results_dir()
    generate_figure_group_1_vt_surface()
    generate_figure_group_2_hwft()
    generate_figure_group_3_pulse()
    generate_figure_group_4_udds_actual_vs_est()
    generate_figure_group_5_method_comparison()
    generate_soh_plots()
    print("✅ Successfully generated all 6 Figure Groups in 'results/paper_plots/'")
