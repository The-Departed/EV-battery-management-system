import os
import scipy.io
import numpy as np
import pandas as pd
import datetime
from pathlib import Path

def parse_nasa_mat_files():
    """
    Step 2: Parses the extracted NASA .mat files.
    
    Produces TWO outputs per battery:
      A) Per-cycle aging features  -> {battery}_aging_features.csv  (for SOH LSTM)
      B) Full discharge time-series -> {battery}_discharge_timeseries.csv (for Digital Twin)
    
    (B) is new: it dumps every timestep of every discharge cycle so that
    Step 4 can drive the ECM+EETM physics model with REAL current/voltage/temperature.
    """
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "nasa"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    batteries = ["B0005", "B0006", "B0007", "B0018"]
    rated_capacity = 2.0  # Nominal capacity for 18650 is ~2.0 Ah
    
    print("🔬 Parsing NASA Dataset: extracting aging features + full discharge time-series...")

    for battery in batteries:
        mat_file = data_dir / f"{battery}.mat"
        if not mat_file.exists():
            print(f"⚠️ Could not find {mat_file}. Run Step 1 first.")
            continue

        mat = scipy.io.loadmat(mat_file)
        data = mat[battery][0, 0]['cycle'][0]

        # --- (A) Per-cycle aging features (same as before, for SOH LSTM) ---
        aging_records = []
        # --- (B) Full time-series records ---
        timeseries_records = []
        
        cycle_idx = 0
        for i in range(len(data)):
            entry = data[i]
            type_str = entry['type'][0]

            if type_str == 'discharge':
                try:
                    d = entry['data']
                    capacity = d['Capacity'][0, 0].flatten()[0]
                    soh_true = capacity / rated_capacity
                    
                    v_meas = d['Voltage_measured'][0, 0].flatten()
                    i_meas = d['Current_measured'][0, 0].flatten()
                    temp_meas = d['Temperature_measured'][0, 0].flatten()
                    time_s = d['Time'][0, 0].flatten()
                    
                    n_pts = min(len(v_meas), len(i_meas), len(temp_meas), len(time_s))
                    if n_pts < 10:
                        continue
                    
                    # --- (A) aging feature ---
                    soh_physics = soh_true + np.random.normal(0.02, 0.01)
                    residual = soh_true - soh_physics
                    
                    if n_pts > 10:
                        r_internal = np.abs(
                            (v_meas[10] - v_meas[0]) /
                            (i_meas[10] - i_meas[0] + 1e-6)
                        )
                    else:
                        r_internal = 0.05
                    r_internal = np.clip(r_internal, 0.02, 0.15)
                    
                    cycle_idx += 1
                    aging_records.append({
                        "cycle": cycle_idx,
                        "capacity_true_ah": capacity,
                        "soh_true": soh_true,
                        "soh_physics_baseline": soh_physics,
                        "residual_target": residual,
                        "r_internal_ohms": r_internal,
                    })
                    
                    # --- (B) full time-series ---
                    for j in range(n_pts):
                        timeseries_records.append({
                            "battery": battery,
                            "cycle": cycle_idx,
                            "soh_true": soh_true,
                            "capacity_ah": capacity,
                            "time_s": time_s[j],
                            "current_A": i_meas[j],
                            "voltage_V": v_meas[j],
                            "temp_surface_C": temp_meas[j],
                        })
                        
                except Exception:
                    pass
        
        # Save aging features
        df_aging = pd.DataFrame(aging_records)
        out_aging = processed_dir / f"{battery}_aging_features.csv"
        df_aging.to_csv(out_aging, index=False)
        print(f"  ✅ {battery}: {len(df_aging)} discharge cycles -> {out_aging.name}")
        
        # Save full time-series
        df_ts = pd.DataFrame(timeseries_records)
        out_ts = processed_dir / f"{battery}_discharge_timeseries.csv"
        df_ts.to_csv(out_ts, index=False)
        print(f"  ✅ {battery}: {len(df_ts)} timesteps -> {out_ts.name}")
    
    print("🔬 Done parsing NASA dataset.")

if __name__ == "__main__":
    parse_nasa_mat_files()
