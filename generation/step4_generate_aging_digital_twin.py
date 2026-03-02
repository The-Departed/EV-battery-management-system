"""
Step 4: Experimentally-Tuned Aging-Aware Digital Twin for Core Temperature
==========================================================================
Based on: "Hybrid Electrical Circuit Model and Deep Learning-Based Core
Temperature Estimation" (Samanta, Surya, Williamson et al., IEEE TTE 2022)

Approach (matching the paper):
  1. Load REAL NASA discharge time-series (current, voltage, surface temp)
  2. Identify ECM parameters (R0, R1, C1, R2, C2) per cycle from real V-I data
  3. Run 2-RC ECM forward to compute terminal voltage and heat generation Q_gen
  4. Run 2-state EETM thermal model to compute CORE temperature (Tc) and
     surface temperature (Ts_sim)
  5. Tune thermal parameters (Rin, Rout, Cc, Cs) by minimising error between
     Ts_sim and the REAL measured surface temperature (this is the UKS-style
     calibration step that links experimental data to the digital twin)
  6. Once tuned, Tc is the physics "ground truth" for training the Transformer

The key difference from the old code: every single number that goes into the
simulator now comes from, or is validated against, the real NASA measurements.
"""

import os
import numpy as np
import pandas as pd
import time as timer
from pathlib import Path
from scipy.optimize import minimize

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


# ---------------------------------------------------------------------------
#  OCV-SOC relationship for 18650 NMC cell (3.0V – 4.2V)
# ---------------------------------------------------------------------------
def ocv_from_soc(soc):
    soc = np.clip(soc, 0.0, 1.0)
    return 3.0 + 1.2 * soc - 0.15 * soc**2 + 0.15 * soc**3


# ---------------------------------------------------------------------------
#  ECM Parameter Identification  (per-cycle, from real V-I data)
# ---------------------------------------------------------------------------
def identify_ecm_params(time_s, current, voltage, soc_init=1.0, capacity_ah=2.0):
    """
    Identify 2-RC ECM parameters [R0, R1, C1, R2, C2] by least-squares fit
    of simulated terminal voltage to measured terminal voltage.

    V_terminal = V_OCV(SOC) - I*R0 - V1 - V2
    dV1/dt = -V1/(R1*C1) + I/C1
    dV2/dt = -V2/(R2*C2) + I/C2
    SOC -= I * dt / Q
    """
    dt_arr = np.diff(time_s)
    n = len(current)
    Q = capacity_ah * 3600.0  # coulombs

    def simulate_ecm(params):
        R0, R1, C1, R2, C2 = params
        if R0 < 0.005 or R1 < 0.001 or R2 < 0.001 or C1 < 10 or C2 < 10:
            return np.ones(n) * 1e6  # penalty

        soc = soc_init
        V1, V2 = 0.0, 0.0
        v_sim = np.zeros(n)

        for k in range(n):
            I = current[k]
            v_sim[k] = ocv_from_soc(soc) - I * R0 - V1 - V2

            if k < n - 1:
                dt = dt_arr[k]
                alpha1 = np.exp(-dt / (R1 * C1))
                alpha2 = np.exp(-dt / (R2 * C2))
                V1 = alpha1 * V1 + I * R1 * (1.0 - alpha1)
                V2 = alpha2 * V2 + I * R2 * (1.0 - alpha2)
                soc -= I * dt / Q
                soc = np.clip(soc, 0.0, 1.0)

        return v_sim

    def cost(params):
        v_sim = simulate_ecm(params)
        return np.mean((v_sim - voltage) ** 2)

    # Initial guess (typical 18650 values from literature)
    x0 = [0.05, 0.01, 15000.0, 0.03, 3000.0]
    bounds = [
        (0.01, 0.20),   # R0
        (0.001, 0.10),  # R1
        (500, 50000),    # C1
        (0.001, 0.10),  # R2
        (100, 20000),    # C2
    ]

    result = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 200, 'ftol': 1e-10})

    return result.x, result.fun


# ---------------------------------------------------------------------------
#  Forward ECM simulation (reusable, after params are identified)
# ---------------------------------------------------------------------------
def forward_ecm(time_s, current, ecm_params, soc_init=1.0, capacity_ah=2.0):
    """Run the 2-RC ECM forward and return v_sim, v_ocv_arr, soc_arr."""
    R0, R1, C1, R2, C2 = ecm_params
    dt_arr = np.diff(time_s)
    n = len(current)
    Q = capacity_ah * 3600.0

    soc = soc_init
    V1, V2 = 0.0, 0.0
    v_sim = np.zeros(n)
    v_ocv_arr = np.zeros(n)

    for k in range(n):
        I = current[k]
        v_ocv_arr[k] = ocv_from_soc(soc)
        v_sim[k] = v_ocv_arr[k] - I * R0 - V1 - V2

        if k < n - 1:
            dt = dt_arr[k]
            a1 = np.exp(-dt / (R1 * C1))
            a2 = np.exp(-dt / (R2 * C2))
            V1 = a1 * V1 + I * R1 * (1.0 - a1)
            V2 = a2 * V2 + I * R2 * (1.0 - a2)
            soc -= I * dt / Q
            soc = np.clip(soc, 0.0, 1.0)

    return v_sim, v_ocv_arr


# ---------------------------------------------------------------------------
#  EETM Thermal Parameter Tuning  (calibrate against real surface temp)
# ---------------------------------------------------------------------------
def tune_thermal_params(time_s, current, voltage_sim, voltage_ocv_arr,
                        temp_surface_measured, T_ambient=25.0,
                        initial_guess=None):
    """
    Tune the 2-state EETM thermal parameters [Rin, Rout, Cc, Cs] by
    minimising the MSE between simulated surface temperature and the
    REAL measured surface temperature.

    This is the UKS-style calibration: the experimental Ts constrains the
    thermal model so that the *unobservable* Tc is physically reliable.

    Heat generation: Q_gen = |I * (V_OCV - V_terminal)|
    Core:    dTc/dt = (Q_gen - (Tc - Ts) / Rin) / Cc
    Surface: dTs/dt = ((Tc - Ts) / Rin - (Ts - Tamb) / Rout) / Cs
    """
    dt_arr = np.diff(time_s)
    n = len(current)
    Q_gen = np.abs(current * (voltage_ocv_arr - voltage_sim))

    def simulate_thermal(params):
        Rin, Rout, Cc, Cs = params
        if Rin < 0.1 or Rout < 0.1 or Cc < 1.0 or Cs < 1.0:
            return np.ones(n) * 1e6, np.ones(n) * 1e6

        Tc = np.zeros(n)
        Ts = np.zeros(n)
        Tc[0] = temp_surface_measured[0]
        Ts[0] = temp_surface_measured[0]

        for k in range(n - 1):
            dt = dt_arr[k]
            dTc = (Q_gen[k] - (Tc[k] - Ts[k]) / Rin) / Cc
            dTs = ((Tc[k] - Ts[k]) / Rin - (Ts[k] - T_ambient) / Rout) / Cs
            Tc[k + 1] = Tc[k] + dTc * dt
            Ts[k + 1] = Ts[k] + dTs * dt

        return Tc, Ts

    def cost(params):
        _, Ts_sim = simulate_thermal(params)
        return np.mean((Ts_sim - temp_surface_measured) ** 2)

    # Use previous cycle's result as warm-start if available
    if initial_guess is None:
        initial_guess = [3.0, 15.0, 30.0, 15.0]

    bounds = [
        (0.5, 20.0),   # Rin  (core-surface thermal resistance K/W)
        (2.0, 60.0),   # Rout (surface-ambient thermal resistance K/W)
        (5.0, 100.0),  # Cc   (core heat capacity J/K)
        (2.0, 60.0),   # Cs   (surface heat capacity J/K)
    ]

    result = minimize(cost, initial_guess, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 300, 'ftol': 1e-12})

    Tc_final, Ts_final = simulate_thermal(result.x)
    return result.x, result.fun, Tc_final, Ts_final


# ---------------------------------------------------------------------------
#  Main: process all batteries, all cycles
# ---------------------------------------------------------------------------
def generate_aging_digital_twin():
    """
    For every discharge cycle of every NASA battery:
      1) Identify ECM params from real V-I  (electrical calibration)
      2) Tune thermal params against real Ts (thermal calibration / UKS)
      3) Output dataset with physics-generated core T
    """
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "data" / "nasa" / "processed"
    out_dir = base_dir / "data" / "digital_twin_sets"
    out_dir.mkdir(parents=True, exist_ok=True)

    batteries = ["B0005", "B0006", "B0007", "B0018"]

    all_records = []
    total_cycles = 0
    start = timer.time()

    for battery in batteries:
        ts_file = processed_dir / f"{battery}_discharge_timeseries.csv"
        aging_file = processed_dir / f"{battery}_aging_features.csv"

        if not ts_file.exists():
            print(f"⚠️  {ts_file.name} not found. Run Step 2 first.")
            continue

        print(f"\n🔋 Processing {battery}...")
        df_ts = pd.read_csv(ts_file)

        cycles = sorted(df_ts['cycle'].unique())

        # Warm-start thermal params from previous cycle
        prev_thermal = None

        for cyc in cycles:
            df_cyc = df_ts[df_ts['cycle'] == cyc].reset_index(drop=True)
            if len(df_cyc) < 20:
                continue

            time_s  = df_cyc['time_s'].values
            current = df_cyc['current_A'].values
            voltage = df_cyc['voltage_V'].values
            temp_surf = df_cyc['temp_surface_C'].values
            soh     = df_cyc['soh_true'].values[0]
            cap_ah  = df_cyc['capacity_ah'].values[0]

            # Ambient = first temperature reading (battery at rest)
            T_amb = temp_surf[0]

            # ---- (A) ECM parameter identification from real V-I ----
            ecm_params, ecm_err = identify_ecm_params(
                time_s, current, voltage, soc_init=1.0, capacity_ah=cap_ah
            )
            R0, R1, C1, R2, C2 = ecm_params

            # ---- Forward ECM for V_sim and V_OCV ----
            v_sim, v_ocv_arr = forward_ecm(
                time_s, current, ecm_params,
                soc_init=1.0, capacity_ah=cap_ah
            )

            # ---- (B) Thermal parameter tuning against real Ts ----
            therm_params, therm_err, Tc, Ts_sim = tune_thermal_params(
                time_s, current, v_sim, v_ocv_arr, temp_surf,
                T_ambient=T_amb, initial_guess=prev_thermal
            )
            Rin, Rout, Cc, Cs = therm_params
            prev_thermal = list(therm_params)  # warm-start next cycle

            # Surface temp validation: RMSE
            ts_rmse = np.sqrt(np.mean((Ts_sim - temp_surf) ** 2))

            # ---- Build output records ----
            n = len(current)
            for k in range(n):
                all_records.append({
                    "battery":           battery,
                    "cycle":             cyc,
                    "soh_true":          soh,
                    "time_s":            time_s[k],
                    "current_A":         current[k],
                    "voltage_V":         voltage[k],
                    "voltage_sim_V":     v_sim[k],
                    "r0_ohms":           R0,
                    "r1_ohms":           R1,
                    "r2_ohms":           R2,
                    "temp_surface_C":    temp_surf[k],
                    "temp_surface_sim_C": Ts_sim[k],
                    "temp_core_C_TARGET": Tc[k],
                })

            total_cycles += 1
            if total_cycles % 20 == 0:
                print(f"   Cycle {cyc:3d} | SOH={soh:.3f} | R0={R0:.4f}Ω | "
                      f"ECM_MSE={ecm_err:.6f} | Ts_RMSE={ts_rmse:.3f}°C | "
                      f"Rin={Rin:.2f} Rout={Rout:.2f} Cc={Cc:.2f} Cs={Cs:.2f}")

        # Print summary for this battery
        print(f"  📊 {battery}: {total_cycles} total cycles processed so far")

    # ---- Save the complete dataset ----
    elapsed = timer.time() - start
    print(f"\n⏱️  Total processing time: {elapsed:.1f}s for {total_cycles} cycles")

    df_out = pd.DataFrame(all_records)
    out_path = out_dir / "augmented_aging_twin_dataset.csv"
    df_out.to_csv(out_path, index=False)
    print(f"✅ Saved {len(df_out)} rows to {out_path.name}")
    print(f"   Columns: {list(df_out.columns)}")

    # Print validation stats
    ts_err = df_out['temp_surface_C'] - df_out['temp_surface_sim_C']
    print(f"   Surface Temp validation: mean_err={ts_err.mean():.4f}°C, "
          f"RMSE={np.sqrt((ts_err**2).mean()):.4f}°C")
    print(f"   Core Temp range: [{df_out['temp_core_C_TARGET'].min():.2f}, "
          f"{df_out['temp_core_C_TARGET'].max():.2f}]°C")


if __name__ == "__main__":
    generate_aging_digital_twin()
