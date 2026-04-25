"""
Step 4: Experimentally‑Tuned Aging‑Aware Digital Twin for Core Temperature
==========================================================================
Fixes applied:
  - Real EPA drive cycles (UDDS, HWFET, US06) loaded from data/drive_cycles/
  - OCV‑SOC curve extracted from NASA data, split into fresh (SOH≥0.9) and aged (SOH≤0.75)
  - Entropic (reversible) heat added to thermal model
  - Multi‑start ECM identification with tight bounds and Savitzky‑Golay noise filtering
  - Initial SOC estimated from first measured voltage
  - Thermal parameters: median values used for EV generation (with aging scaling)
  - Validation log with voltage RMSE, surface temp RMSE, Q_gen, ΔT
  - Timing per cycle and capacity throughput check on EV drive cycles
"""

import os
import numpy as np
import pandas as pd
import time as timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import scipy.io

# =============================================================================
# 1. BUILD OCV-SOC CURVE – FRESH AND AGED
# =============================================================================
def build_ocv_curve():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "nasa"
    batteries = ["B0005", "B0006", "B0007", "B0018"]
    rated_ah = 2.0

    soc_pts, ocv_pts, soh_pts = [], [], []
    for batt in batteries:
        mat_file = data_dir / f"{batt}.mat"
        if not mat_file.exists():
            continue
        mat = scipy.io.loadmat(mat_file)
        cycles = mat[batt][0, 0]['cycle'][0]
        for entry in cycles:
            if entry['type'][0] == 'discharge':
                try:
                    d = entry['data']
                    v = d['Voltage_measured'][0, 0].flatten()
                    cap = d['Capacity'][0, 0].flatten()[0]
                    soh = cap / rated_ah
                    soc_pts.append(1.0)
                    ocv_pts.append(v[0])
                    soh_pts.append(soh)
                    soc_end = 1.0 - cap / rated_ah
                    if soc_end >= 0.0:
                        soc_pts.append(soc_end)
                        ocv_pts.append(v[-1])
                        soh_pts.append(soh)
                except Exception:
                    continue

    soc_pts = np.array(soc_pts)
    ocv_pts = np.array(ocv_pts)
    soh_pts = np.array(soh_pts)

    mask_fresh = soh_pts >= 0.9
    mask_aged  = soh_pts <= 0.75

    if mask_fresh.sum() > 6:
        coeffs_fresh = np.polyfit(soc_pts[mask_fresh], ocv_pts[mask_fresh], 5)
    else:
        coeffs_fresh = np.polyfit(soc_pts, ocv_pts, 5)

    if mask_aged.sum() > 6:
        coeffs_aged = np.polyfit(soc_pts[mask_aged], ocv_pts[mask_aged], 5)
    else:
        coeffs_aged = coeffs_fresh

    print(f"✅ Extracted OCV curves (fresh {mask_fresh.sum()} pts, aged {mask_aged.sum()} pts)")
    return list(coeffs_fresh), list(coeffs_aged)

OCV_COEFFS_FRESH, OCV_COEFFS_AGED = build_ocv_curve()

def ocv_from_soc(soc, soh=1.0):
    """Blend fresh and aged OCV polynomials based on current SOH."""
    soc = np.clip(soc, 0.0, 1.0)
    alpha = np.clip((soh - 0.75) / (0.9 - 0.75), 0.0, 1.0)
    coeffs = [alpha * c1 + (1 - alpha) * c2 for c1, c2 in zip(OCV_COEFFS_FRESH, OCV_COEFFS_AGED)]
    return np.polyval(coeffs, soc)

def soc_from_ocv(ocv, soh=1.0, tol=1e-4):
    """Inverse OCV lookup using bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if ocv_from_soc(mid, soh) < ocv:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2

# =============================================================================
# 2. ENTROPIC HEAT COEFFICIENT dU/dT (V/K) vs SOC
# =============================================================================
_ENTROPIC_SOC = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
_ENTROPIC_DUDT = np.array([0.0003, 0.00025, 0.00015, 0.00005, -0.00005,
                           -0.00015, -0.00025, -0.00035, -0.00045, -0.0005, -0.00055])

def dudt_from_soc(soc):
    return np.interp(np.clip(soc, 0, 1), _ENTROPIC_SOC, _ENTROPIC_DUDT)

# =============================================================================
# 3. REAL EPA DRIVE CYCLE CURRENT GENERATOR
# =============================================================================
MASS_KG = 1200.0
CRR = 0.01
RHO_AIR = 1.225
CD = 0.30
AREA = 2.2
GRAVITY = 9.81
CELLS_IN_PARALLEL = 74
CELL_AVERAGE_VOLTAGE = 3.7

DRIVE_CYCLE_DIR = Path(__file__).parent.parent / "data" / "drive_cycles"
CYCLE_FILE_MAP = {
    "UDDS":  "UDDS_epa_speed.csv",
    "HWFET": "HWFET_epa_speed.csv",
    "US06":  "US06_epa_speed.csv",
}

def _load_epa_trace(profile_name):
    fname = CYCLE_FILE_MAP[profile_name]
    file_path = DRIVE_CYCLE_DIR / fname
    if not file_path.exists():
        raise FileNotFoundError(f"Drive cycle file not found: {file_path}")
    df = pd.read_csv(file_path)
    return df['time_s'].values, df['speed_mps'].values

def generate_drive_cycle_current(profile_name, dt=1.0):
    if profile_name == "Aggressive":
        return generate_drive_cycle_current("US06", dt=dt)
    elif profile_name == "Mixed":
        t1, i1 = generate_drive_cycle_current("UDDS", dt=dt)
        t2, i2 = generate_drive_cycle_current("HWFET", dt=dt)
        t2_shifted = t2 + t1[-1] + dt
        return np.concatenate([t1, t2_shifted]), np.concatenate([i1, i2])

    time_full, speed_mps_full = _load_epa_trace(profile_name)
    if dt != 0.1:
        new_len = int(time_full[-1] / dt) + 1
        time_s = np.linspace(0, time_full[-1], new_len)
        speed_mps = np.interp(time_s, time_full, speed_mps_full)
    else:
        time_s = time_full
        speed_mps = speed_mps_full

    accel = np.gradient(speed_mps, time_s)
    F_roll = CRR * MASS_KG * GRAVITY
    F_drag = 0.5 * RHO_AIR * CD * AREA * speed_mps**2
    F_inertia = MASS_KG * accel
    F_total = F_roll + F_drag + F_inertia
    P_wheels = np.maximum(F_total * speed_mps, 0.0)
    P_per_cell = P_wheels / CELLS_IN_PARALLEL
    I_cell = P_per_cell / CELL_AVERAGE_VOLTAGE + 0.05
    return time_s, I_cell

# =============================================================================
# 4. ECM IDENTIFICATION (multi‑start, tight bounds, noise filtering)
# =============================================================================
def _filter_noise(arr, window=5):
    if len(arr) > window:
        return savgol_filter(arr, window, 2)
    return arr

def identify_ecm_params(time_s, current, voltage, soc_init, capacity_ah, soh=1.0):
    current_f = _filter_noise(current)
    voltage_f = _filter_noise(voltage)
    dt_arr = np.diff(time_s)
    n = len(current_f)
    Q = capacity_ah * 3600.0

    def simulate_ecm(params):
        R0, R1, C1, R2, C2 = params
        soc = soc_init
        V1, V2 = 0.0, 0.0
        v_sim = np.zeros(n)
        for k in range(n):
            I = current_f[k]
            v_sim[k] = ocv_from_soc(soc, soh) - I * R0 - V1 - V2
            if k < n - 1:
                dt = dt_arr[k]
                a1 = np.exp(-dt / (R1 * C1)) if R1 * C1 > 0 else 0.0
                a2 = np.exp(-dt / (R2 * C2)) if R2 * C2 > 0 else 0.0
                V1 = a1 * V1 + I * R1 * (1.0 - a1)
                V2 = a2 * V2 + I * R2 * (1.0 - a2)
                soc -= I * dt / Q
                soc = np.clip(soc, 0.0, 1.0)
        return v_sim

    def cost(params):
        return np.mean((simulate_ecm(params) - voltage_f) ** 2)

    bounds = [
        (0.030, 0.180),   # R0
        (0.005, 0.060),   # R1
        (500.0, 5000.0),  # C1
        (0.005, 0.060),   # R2
        (1000.0, 50000.0),# C2
    ]

    best_x, best_cost = None, np.inf
    rng = np.random.RandomState(42)
    for _ in range(20):
        x0 = [rng.uniform(*b) for b in bounds]
        res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-10})
        if res.fun < best_cost:
            best_cost = res.fun
            best_x = res.x
    return best_x, best_cost

def forward_ecm(time_s, current, ecm_params, soc_init, capacity_ah, soh=1.0):
    R0, R1, C1, R2, C2 = ecm_params
    dt_arr = np.diff(time_s)
    n = len(current)
    Q = capacity_ah * 3600.0
    soc = soc_init
    V1, V2 = 0.0, 0.0
    v_sim = np.zeros(n)
    v_ocv_arr = np.zeros(n)
    soc_arr = np.zeros(n)
    for k in range(n):
        I = current[k]
        v_ocv_arr[k] = ocv_from_soc(soc, soh)
        v_sim[k] = v_ocv_arr[k] - I * R0 - V1 - V2
        soc_arr[k] = soc
        if k < n - 1:
            dt = dt_arr[k]
            a1 = np.exp(-dt / (R1 * C1)) if R1 * C1 > 0 else 0.0
            a2 = np.exp(-dt / (R2 * C2)) if R2 * C2 > 0 else 0.0
            V1 = a1 * V1 + I * R1 * (1.0 - a1)
            V2 = a2 * V2 + I * R2 * (1.0 - a2)
            soc -= I * dt / Q
            soc = np.clip(soc, 0.0, 1.0)
    return v_sim, v_ocv_arr, soc_arr

# =============================================================================
# 5. THERMAL MODEL – ENTROPIC HEAT INCLUDED
# =============================================================================
def simulate_thermal_forward(time_s, current, v_sim, v_ocv_arr,
                             therm_params, T_ambient=25.0, soc_arr=None, capacity_ah=None):
    Rin, Rout, Cc, Cs = therm_params
    dt_arr = np.diff(time_s)
    n = len(current)

    Q_irrev = np.abs(current * (v_ocv_arr - v_sim))
    if soc_arr is None and capacity_ah is not None:
        soc_arr = 1.0 - np.cumsum(np.diff(time_s, prepend=time_s[0]) * current) / (capacity_ah * 3600.0)
    elif soc_arr is None:
        soc_arr = np.linspace(1.0, 0.0, n)
    Q_rev = current * (T_ambient + 273.15) * dudt_from_soc(np.clip(soc_arr, 0, 1))
    Q_gen = Q_irrev + Q_rev

    Tc = np.zeros(n)
    Ts = np.zeros(n)
    Tc[0] = T_ambient
    Ts[0] = T_ambient
    for k in range(n - 1):
        dt = dt_arr[k]
        dTc = (Q_gen[k] - (Tc[k] - Ts[k]) / Rin) / Cc
        dTs = ((Tc[k] - Ts[k]) / Rin - (Ts[k] - T_ambient) / Rout) / Cs
        Tc[k+1] = Tc[k] + dTc * dt
        Ts[k+1] = Ts[k] + dTs * dt
    return Tc, Ts

def tune_thermal_params(time_s, current, voltage_sim, voltage_ocv_arr,
                        temp_surface_measured, T_ambient, initial_guess=None, capacity_ah=2.0):
    dt_arr = np.diff(time_s)
    n = len(current)
    Q_irrev = np.abs(current * (voltage_ocv_arr - voltage_sim))
    soc_traj = 1.0 - np.cumsum(np.diff(time_s, prepend=time_s[0]) * current) / (capacity_ah * 3600.0)
    Q_rev = current * (T_ambient + 273.15) * dudt_from_soc(np.clip(soc_traj, 0, 1))
    Q_gen = Q_irrev + Q_rev

    def simulate_thermal(params):
        Rin, Rout, Cc, Cs = params
        Tc = np.zeros(n)
        Ts = np.zeros(n)
        Tc[0] = temp_surface_measured[0]
        Ts[0] = temp_surface_measured[0]
        for k in range(n - 1):
            dt = dt_arr[k]
            dTc = (Q_gen[k] - (Tc[k] - Ts[k]) / Rin) / Cc
            dTs = ((Tc[k] - Ts[k]) / Rin - (Ts[k] - T_ambient) / Rout) / Cs
            Tc[k+1] = Tc[k] + dTc * dt
            Ts[k+1] = Ts[k] + dTs * dt
        return Tc, Ts

    def cost(params):
        _, Ts_sim = simulate_thermal(params)
        return np.mean((Ts_sim - temp_surface_measured) ** 2)

    if initial_guess is None:
        initial_guess = [3.0, 15.0, 30.0, 15.0]
    bounds = [(0.5, 20.0), (2.0, 60.0), (5.0, 100.0), (2.0, 60.0)]
    res = minimize(cost, initial_guess, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 300, 'ftol': 1e-12})
    Tc_final, Ts_final = simulate_thermal(res.x)
    return res.x, res.fun, Tc_final, Ts_final

# =============================================================================
# 6. VISUALISATION & EV DATASET – USE MEDIAN THERMAL PARAMS + AGING SCALING
# =============================================================================
def run_physics_on_profile(time_s, current, ecm_params, therm_params,
                           T_ambient, capacity_ah=2.0, soh=1.0):
    v_sim, v_ocv, soc_arr = forward_ecm(time_s, current, ecm_params,
                                        soc_init=1.0, capacity_ah=capacity_ah, soh=soh)
    Tc, Ts = simulate_thermal_forward(time_s, current, v_sim, v_ocv,
                                      therm_params, T_ambient=T_ambient,
                                      soc_arr=soc_arr)
    return v_sim, v_ocv, Tc, Ts

def generate_multi_ambient_plot(profile_name, ecm_params, therm_params,
                                plot_dir, capacity_ah=2.0, soh=1.0):
    aging_factor = 1.0 + 0.2 * (1.0 - soh)
    therm_aged = [therm_params[0]*aging_factor, therm_params[1]*aging_factor,
                  therm_params[2], therm_params[3]]

    Tamb_list = [0.0, 20.0, 50.0]
    colors_tc = {0.0:'blue', 20.0:'red', 50.0:'green'}
    time_s, current = generate_drive_cycle_current(profile_name)
    Tc_results, Vs_results = {}, {}
    for Tamb in Tamb_list:
        v_sim, v_ocv, Tc, _ = run_physics_on_profile(
            time_s, current, ecm_params, therm_aged,
            T_ambient=Tamb, capacity_ah=capacity_ah, soh=soh)
        Tc_results[Tamb] = Tc
        Vs_results[Tamb] = v_sim
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,5))
    ax1.plot(time_s, current, color='black', linewidth=0.6)
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Current (A)')
    ax1.set_title(f'{profile_name} — Current Profile'); ax1.grid(True, linestyle='--', alpha=0.5)
    for Tamb in Tamb_list:
        ax2.plot(time_s, Tc_results[Tamb], color=colors_tc[Tamb], linewidth=1.2,
                 label=f'Tc @ {Tamb:.0f}°C')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Core Temperature (°C)')
    ax2.set_title(f'{profile_name} — Core Temperature'); ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.5)
    ax3.plot(time_s, Vs_results[20.0], color='purple', linewidth=0.8)
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Voltage (V)')
    ax3.set_title(f'{profile_name} — Simulated Voltage'); ax3.grid(True, linestyle='--', alpha=0.5)
    plt.suptitle(f'Physics Engine: {profile_name} Drive Cycle', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(str(plot_dir / f'{profile_name.lower()}_multi_temp_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved {profile_name.lower()}_multi_temp_visualization.png")

def generate_ev_drive_cycle_dataset(processed_dir, out_dir, plot_dir, median_therm):
    batteries = ["B0005","B0006","B0007","B0018"]
    drive_cycles = ["UDDS","HWFET","US06"]
    Tamb_list = [0.0, 25.0, 45.0]
    aging_step = 20
    ev_out_dir = out_dir.parent / "ev_validation_sets"
    ev_out_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    total_sims = 0
    start = timer.time()
    twin_csv = out_dir / "augmented_aging_twin_dataset.csv"
    if not twin_csv.exists():
        print("⚠️ Twin dataset not found. Skipping EV generation.")
        return
    df_twin = pd.read_csv(twin_csv)
    for battery in batteries:
        df_batt = df_twin[df_twin['battery'] == battery]
        if df_batt.empty: continue
        cycle_params = df_batt.groupby('cycle').agg(
            soh=('soh_true','first'), R0=('r0_ohms','first'),
            R1=('r1_ohms','first'), R2=('r2_ohms','first')).reset_index()
        cycles_avail = sorted(cycle_params['cycle'].values)
        aging_cycles = [c for c in cycles_avail if c % aging_step == 0]
        if not aging_cycles: aging_cycles = cycles_avail[::max(1,len(cycles_avail)//8)]
        print(f"\n🚗 {battery}: {len(aging_cycles)} aging states × {len(drive_cycles)} cycles × {len(Tamb_list)} temps")
        for cyc in aging_cycles:
            row = cycle_params[cycle_params['cycle'] == cyc].iloc[0]
            R0, R1, R2 = row['R0'], row['R1'], row['R2']
            soh = row['soh']
            C1, C2 = 15000.0, 3000.0
            ecm_p = [R0, R1, C1, R2, C2]
            aging_factor = 1.0 + 0.2 * (1.0 - soh)
            therm_p = [median_therm[0]*aging_factor, median_therm[1]*aging_factor,
                       median_therm[2], median_therm[3]]
            cap_ah = soh * 2.0
            for drive in drive_cycles:
                time_s, current = generate_drive_cycle_current(drive)
                throughput = np.sum(np.abs(current)) * (time_s[1]-time_s[0]) / 3600.0
                if throughput > 2.5:
                    print(f"⚠️ Throughput {throughput:.2f} Ah exceeds cell rating (2 Ah) in {drive}")
                for Tamb in Tamb_list:
                    v_sim, v_ocv, Tc, Ts = run_physics_on_profile(
                        time_s, current, ecm_p, therm_p,
                        T_ambient=Tamb, capacity_ah=cap_ah, soh=soh)
                    n = len(time_s)
                    for k in range(n):
                        all_records.append({
                            "battery": f"{battery}_{drive}_T{Tamb:.0f}_C{cyc}",
                            "cycle": cyc, "soh_true": soh,
                            "time_s": time_s[k], "current_A": current[k],
                            "voltage_V": v_sim[k], "voltage_sim_V": v_sim[k],
                            "r0_ohms": R0, "r1_ohms": R1, "r2_ohms": R2,
                            "temp_surface_C": Ts[k], "temp_surface_sim_C": Ts[k],
                            "temp_core_C_TARGET": Tc[k],
                        })
                    total_sims += 1
    elapsed = timer.time() - start
    print(f"\n⏱️  EV dataset generation: {elapsed:.1f}s for {total_sims} simulations")
    if not all_records: return
    df_ev = pd.DataFrame(all_records)
    ev_path = ev_out_dir / "ev_drive_cycle_dataset.csv"
    df_ev.to_csv(ev_path, index=False)
    total_hours = len(df_ev) / 3600.0
    print(f"✅ EV Dataset: {len(df_ev):,} rows ({total_hours:.1f} hours) → {ev_path.name}")

# =============================================================================
# 7. MAIN FUNCTION
# =============================================================================
def generate_aging_digital_twin():
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "data" / "nasa" / "processed"
    out_dir = base_dir / "data" / "digital_twin_sets"
    plot_dir = base_dir / "results" / "paper_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    batteries = ["B0005","B0006","B0007","B0018"]
    all_records = []
    thermal_list = []      # collect thermal params for median
    total_cycles = 0
    start = timer.time()
    last_soh = 1.0

    for battery in batteries:
        ts_file = processed_dir / f"{battery}_discharge_timeseries.csv"
        if not ts_file.exists():
            print(f"⚠️  {ts_file.name} not found. Run Step 2 first.")
            continue
        print(f"\n🔋 Processing {battery}...")
        df_ts = pd.read_csv(ts_file)
        cycles = sorted(df_ts['cycle'].unique())
        prev_thermal = None

        for cyc in cycles:
            df_cyc = df_ts[df_ts['cycle'] == cyc].reset_index(drop=True)
            if len(df_cyc) < 20: continue
            time_s  = df_cyc['time_s'].values
            current = df_cyc['current_A'].values
            voltage = df_cyc['voltage_V'].values
            temp_surf = df_cyc['temp_surface_C'].values
            soh     = df_cyc['soh_true'].values[0]
            cap_ah  = df_cyc['capacity_ah'].values[0]

            V_start = voltage[0]
            soc_init = soc_from_ocv(V_start, soh)

            t_ecm_start = timer.time()
            ecm_params, ecm_err = identify_ecm_params(time_s, current, voltage,
                                                      soc_init, cap_ah, soh)
            t_ecm = timer.time() - t_ecm_start
            R0, R1, C1, R2, C2 = ecm_params
            v_sim, v_ocv_arr, soc_arr = forward_ecm(time_s, current, ecm_params,
                                                    soc_init=soc_init, capacity_ah=cap_ah, soh=soh)

            Q_gen = np.abs(current * (v_ocv_arr - v_sim)) + current * (temp_surf[0]+273.15) * dudt_from_soc(soc_arr)
            q_mean = np.mean(np.abs(Q_gen))
            if q_mean < 1.0:
                print(f"   ⚠️  Cycle {cyc}: Q_gen = {q_mean:.3f} W (too low)")

            T_amb = temp_surf[0]
            therm_params, therm_err, Tc, Ts_sim = tune_thermal_params(
                time_s, current, v_sim, v_ocv_arr, temp_surf,
                T_ambient=T_amb, initial_guess=prev_thermal, capacity_ah=cap_ah)
            Rin, Rout, Cc, Cs = therm_params
            prev_thermal = list(therm_params)
            thermal_list.append(therm_params)
            ts_rmse = np.sqrt(np.mean((Ts_sim - temp_surf)**2))

            last_soh = soh

            n = len(current)
            for k in range(n):
                all_records.append({
                    "battery": battery, "cycle": cyc, "soh_true": soh,
                    "time_s": time_s[k], "current_A": current[k],
                    "voltage_V": voltage[k], "voltage_sim_V": v_sim[k],
                    "r0_ohms": R0, "r1_ohms": R1, "r2_ohms": R2,
                    "temp_surface_C": temp_surf[k], "temp_surface_sim_C": Ts_sim[k],
                    "temp_core_C_TARGET": Tc[k],
                })
            total_cycles += 1
            if total_cycles % 20 == 0:
                print(f"   Cycle {cyc:3d} | SOH={soh:.3f} | R0={R0:.4f}Ω | ECM time={t_ecm:.2f}s | "
                      f"ECM_MSE={ecm_err:.6f} | Ts_RMSE={ts_rmse:.3f}°C | Q_gen_mean={q_mean:.2f}W | "
                      f"Rin={Rin:.2f} Rout={Rout:.2f} Cc={Cc:.2f} Cs={Cs:.2f}")

    # Compute median thermal parameters across all cycles
    if thermal_list:
        thermal_df = pd.DataFrame(thermal_list, columns=['Rin','Rout','Cc','Cs'])
        median_therm = thermal_df.median().values
        print(f"\n📊 Median thermal params: Rin={median_therm[0]:.2f}, Rout={median_therm[1]:.2f}, Cc={median_therm[2]:.2f}, Cs={median_therm[3]:.2f}")
    else:
        median_therm = np.array([3.0, 15.0, 30.0, 15.0])

    # Save full twin dataset
    df_out = pd.DataFrame(all_records)
    out_path = out_dir / "augmented_aging_twin_dataset.csv"
    df_out.to_csv(out_path, index=False)
    print(f"✅ Saved {len(df_out)} rows to {out_path.name}")

    # Save ECM parameters for Step 3
    ecm_params_df = df_out.groupby(['battery','cycle']).agg(
        r0_ohms=('r0_ohms','first'), r1_ohms=('r1_ohms','first'), r2_ohms=('r2_ohms','first')).reset_index()
    ecm_params_df.to_csv(out_dir / "ecm_parameters.csv", index=False)
    print("✅ ECM parameters saved to ecm_parameters.csv")

    # Validation log
    validation_records = []
    for batt in batteries:
        bdf = df_out[df_out['battery']==batt]
        for cyc in sorted(bdf['cycle'].unique()):
            cdf = bdf[bdf['cycle']==cyc]
            if len(cdf)<2: continue
            v_rmse = np.sqrt(np.mean((cdf['voltage_V']-cdf['voltage_sim_V'])**2))
            ts_rmse = np.sqrt(np.mean((cdf['temp_surface_C']-cdf['temp_surface_sim_C'])**2))
            delta_tc = cdf['temp_core_C_TARGET'].values - cdf['temp_surface_C'].values
            q_approx = np.abs(cdf['current_A'].values * (cdf['voltage_V'].values - cdf['voltage_sim_V'].values))
            q_mean = np.mean(q_approx)
            validation_records.append({
                'battery':batt,'cycle':cyc,'soh':cdf['soh_true'].iloc[0],
                'V_RMSE_mV':v_rmse*1000, 'Ts_RMSE_C':ts_rmse,
                'Q_approx_mean_W':q_mean, 'max_core_surface_dT':delta_tc.max(),
                'mean_core_surface_dT':delta_tc.mean()
            })
    pd.DataFrame(validation_records).to_csv(out_dir / "validation_log.csv", index=False)
    print("✅ Validation log saved.")

    # Multi-ambient plots with median thermal params and aging
    plot_soh = last_soh
    aging_factor = 1.0 + 0.2 * (1.0 - plot_soh)
    therm_aged = [median_therm[0]*aging_factor, median_therm[1]*aging_factor,
                  median_therm[2], median_therm[3]]
    # Use last available ECM parameters (with typical C1,C2)
    if not df_out.empty:
        last_cycle_row = df_out[df_out['cycle'] == df_out['cycle'].max()].iloc[0]
        last_ecm = [last_cycle_row['r0_ohms'], last_cycle_row['r1_ohms'],
                    15000.0, last_cycle_row['r2_ohms'], 3000.0]
    else:
        last_ecm = [0.08, 0.01, 15000.0, 0.03, 3000.0]
    generate_multi_ambient_plot("Aggressive", last_ecm, therm_aged, plot_dir,
                                capacity_ah=plot_soh*2.0, soh=plot_soh)
    generate_multi_ambient_plot("Mixed", last_ecm, therm_aged, plot_dir,
                                capacity_ah=plot_soh*2.0, soh=plot_soh)

    print(f"\n🚗 Generating EV Real‑World Drive Cycle Dataset...")
    generate_ev_drive_cycle_dataset(processed_dir, out_dir, plot_dir, median_therm)

if __name__ == "__main__":
    generate_aging_digital_twin()