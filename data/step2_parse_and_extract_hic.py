import os
import scipy.io
import numpy as np
import pandas as pd
import datetime
from pathlib import Path

def parse_nasa_mat_files():
    """
    Step 2: Parses the extracted NASA .mat files.
    Calculates Ground Truth SOH, Physics Baseline SOH, and Residuals.
    """
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "nasa"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    batteries = ["B0005", "B0006", "B0007", "B0018"]
    
    print("🔬 Parsing NASA Dataset and calculating Physics Baselines...")

    for battery in batteries:
        mat_file = data_dir / f"{battery}.mat"
        if not mat_file.exists():
            print(f"⚠️ Could not find {mat_file}. Run Step 1 first.")
            continue

        mat = scipy.io.loadmat(mat_file)
        data = mat[battery][0, 0]['cycle'][0]

        extracted_cycles = []
        rated_capacity = 2.0  # Nominal capacity for 18650 is ~2.0 Ah
        
        cycle_idx = 0
        for i in range(len(data)):
            entry = data[i]
            type_str = entry['type'][0]

            # We care about Discharge to find Ground Truth Capacity
            if type_str == 'discharge':
                try:
                    # Ground Truth Capacity calculation
                    capacity = entry['data']['Capacity'][0, 0][0, 0]
                    soh_true = capacity / rated_capacity
                    
                    # Simulated Physics Bias (Coulomb Counting assumption bias)
                    # Real physics models drift due to noise and non-linearity
                    soh_physics = soh_true + np.random.normal(0.02, 0.01) # Example physics bias
                    residual = soh_true - soh_physics

                    # Internal Resistance Approximation from pulse steps
                    # R = dV / dI
                    v_profile = entry['data']['Voltage_measured'][0,0][0]
                    i_profile = entry['data']['Current_measured'][0,0][0]
                    if len(v_profile) > 10 and len(i_profile) > 10:
                         r_internal = np.abs((v_profile[10] - v_profile[0]) / (i_profile[10] - i_profile[0] + 1e-6))
                    else:
                         r_internal = 0.05
                    
                    # Cap unreasonable resistance calculations
                    r_internal = np.clip(r_internal, 0.02, 0.15)
                         
                    cycle_idx += 1
                    extracted_cycles.append({
                        "cycle": cycle_idx,
                        "capacity_true_ah": capacity,
                        "soh_true": soh_true,
                        "soh_physics_baseline": soh_physics,
                        "residual_target": residual,
                        "r_internal_ohms": r_internal
                    })
                except Exception as e:
                    pass
        
        # Save to CSV
        df = pd.DataFrame(extracted_cycles)
        out_path = processed_dir / f"{battery}_aging_features.csv"
        df.to_csv(out_path, index=False)
        print(f"✅ Extracted {len(df)} cycles for {battery} -> Saved to {out_path.name}")

if __name__ == "__main__":
    parse_nasa_mat_files()
