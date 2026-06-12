"""
Step 2: Parse NASA .mat Files → Per-Battery CSVs
=================================================
Extracts per-cycle aging features and full discharge time-series from the
NASA Ames 18650 battery dataset (B0005, B0006, B0007, B0018).

Key fixes vs previous version:
  - SOH physics baseline: quadratic fit (captures the knee in capacity fade)
    instead of a single linear line first→last. For NASA cells the knee
    typically appears around cycle 100–120; a line misses this completely.
  - R_internal: removed the bogus ΔV/ΔI-at-10-pts estimate. A placeholder
    value (0.05 Ω) is used; Step 4 ECM identification overwrites it with the
    correct per-cycle R0 before the LSTM ever trains.
  - OCV rest points extracted from charge end-of-rest / between-cycle rest
    phases (type == 'impedance' entries in NASA mat), saved separately as
    {battery}_ocv_rest_points.csv for use in Step 4's OCV curve fitting.
  - Silent except: pass replaced with a logged warning so bad cycles are
    visible.
  - Cycle index is now the NASA sequential index (1-based), consistent with
    Step 4 so merge keys line up correctly.

Output files (in data/nasa/processed/):
  {battery}_aging_features.csv      — per-cycle SOH, baseline, residual, R
  {battery}_discharge_timeseries.csv — full V, I, T per timestep
  {battery}_ocv_rest_points.csv      — (soc, ocv, soh) triplets from rest phases
"""

import warnings
import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path


def _quadratic_soh_baseline(cycles: np.ndarray, soh: np.ndarray) -> np.ndarray:
    """
    Fit a quadratic (degree-2) polynomial to SOH vs cycle and return the
    fitted baseline values.  A quadratic captures the initial plateau +
    accelerating knee that all NASA cells show, unlike a straight line.
    Falls back to linear if fewer than 3 points.
    """
    if len(cycles) < 3:
        slope = (soh[-1] - soh[0]) / max(cycles[-1] - cycles[0], 1)
        return soh[0] + slope * (cycles - cycles[0])
    deg = 2
    coeffs = np.polyfit(cycles, soh, deg)
    baseline = np.polyval(coeffs, cycles)
    # Clamp: baseline should stay within [0, 1] and not rise above first point
    baseline = np.clip(baseline, 0.0, soh[0])
    return baseline


def _extract_ocv_rest_points(mat_data: np.ndarray, rated_ah: float) -> list[dict]:
    """
    NASA .mat files contain 'impedance' measurement entries that are taken
    during rest periods between discharge/charge cycles.  The voltage at the
    end of those entries is very close to OCV because the cell has rested.
    We also extract the end-of-charge voltage (last point of a charge cycle)
    as an additional near-OCV point.

    Returns list of {soc, ocv_v, soh} dicts.
    """
    rest_pts = []
    capacity_seen = []

    for entry in mat_data:
        etype = str(entry['type'][0]).strip()
        try:
            d = entry['data']

            if etype == 'discharge':
                cap = float(d['Capacity'][0, 0].flatten()[0])
                capacity_seen.append(cap)

            elif etype == 'charge':
                v = d['Voltage_measured'][0, 0].flatten()
                i_arr = d['Current_measured'][0, 0].flatten()
                if len(v) < 5:
                    continue
                # End-of-charge: last point where |I| < 0.05 A (CV phase end)
                low_i = np.where(np.abs(i_arr) < 0.05)[0]
                if len(low_i) > 0:
                    idx = low_i[-1]
                    v_ocv = float(v[idx])
                    soh_est = capacity_seen[-1] / rated_ah if capacity_seen else 1.0
                    # SOC at end of charge ≈ 1.0
                    rest_pts.append({'soc': 1.0, 'ocv_v': v_ocv, 'soh': soh_est})

            elif etype == 'impedance':
                v = d['Voltage_measured'][0, 0].flatten()
                if len(v) < 5:
                    continue
                v_rest = float(v[-1])   # last point ≈ settled OCV
                soh_est = capacity_seen[-1] / rated_ah if capacity_seen else 1.0
                # SOC unknown for impedance entries — use 0.5 as centre estimate;
                # these points are used to anchor the middle of the OCV curve.
                rest_pts.append({'soc': 0.5, 'ocv_v': v_rest, 'soh': soh_est})

        except Exception as exc:
            warnings.warn(f"OCV rest extraction skipped entry ({etype}): {exc}")
            continue

    return rest_pts


def parse_nasa_mat_files():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "nasa"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    batteries = ["B0005", "B0006", "B0007", "B0018"]
    rated_capacity = 2.0

    print("🔬 Parsing NASA Dataset: extracting aging features + discharge time-series + OCV rest points...")

    for battery in batteries:
        mat_file = data_dir / f"{battery}.mat"
        if not mat_file.exists():
            print(f"⚠️  {mat_file.name} not found. Run Step 1 first.")
            continue

        mat = scipy.io.loadmat(mat_file)
        mat_cycles = mat[battery][0, 0]['cycle'][0]

        aging_records = []
        timeseries_records = []
        cycle_idx = 0
        skipped = 0

        for i in range(len(mat_cycles)):
            entry = mat_cycles[i]
            type_str = str(entry['type'][0]).strip()

            if type_str != 'discharge':
                continue

            try:
                d = entry['data']
                capacity = float(d['Capacity'][0, 0].flatten()[0])
                soh_true = capacity / rated_capacity

                v_meas   = d['Voltage_measured'][0, 0].flatten()
                i_meas   = d['Current_measured'][0, 0].flatten()
                temp_meas= d['Temperature_measured'][0, 0].flatten()
                time_s   = d['Time'][0, 0].flatten()

                n_pts = min(len(v_meas), len(i_meas), len(temp_meas), len(time_s))
                if n_pts < 10:
                    skipped += 1
                    continue

                # Trim to common length
                v_meas    = v_meas[:n_pts]
                i_meas    = i_meas[:n_pts]
                temp_meas = temp_meas[:n_pts]
                time_s    = time_s[:n_pts]

                # Remove duplicate / non-monotone timestamps (dt <= 0)
                dt = np.diff(time_s)
                valid = np.concatenate([[True], dt > 0])
                v_meas    = v_meas[valid]
                i_meas    = i_meas[valid]
                temp_meas = temp_meas[valid]
                time_s    = time_s[valid]
                n_pts     = len(time_s)

                if n_pts < 10:
                    skipped += 1
                    continue

                cycle_idx += 1

                # Placeholder R_internal — overwritten by ECM R0 in Step 4
                r_internal = 0.05

                aging_records.append({
                    "cycle":             cycle_idx,
                    "capacity_true_ah":  capacity,
                    "soh_true":          soh_true,
                    "r_internal_ohms":   r_internal,
                })

                for j in range(n_pts):
                    timeseries_records.append({
                        "battery":       battery,
                        "cycle":         cycle_idx,
                        "soh_true":      soh_true,
                        "capacity_ah":   capacity,
                        "time_s":        float(time_s[j]),
                        "current_A":     float(i_meas[j]),
                        "voltage_V":     float(v_meas[j]),
                        "temp_surface_C":float(temp_meas[j]),
                    })

            except Exception as exc:
                skipped += 1
                warnings.warn(f"{battery} cycle {i}: skipped — {exc}")

        if skipped:
            print(f"  ⚠️  {battery}: {skipped} entries skipped (see warnings above)")

        # ---- SOH physics baseline: quadratic fit ----
        df_aging = pd.DataFrame(aging_records)
        if len(df_aging) > 2:
            cyc_arr = df_aging['cycle'].values.astype(float)
            soh_arr = df_aging['soh_true'].values
            df_aging['soh_physics_baseline'] = _quadratic_soh_baseline(cyc_arr, soh_arr)
        elif len(df_aging) == 2:
            first, last = df_aging['soh_true'].iloc[0], df_aging['soh_true'].iloc[-1]
            cyc_f, cyc_l = df_aging['cycle'].iloc[0], df_aging['cycle'].iloc[-1]
            slope = (last - first) / max(cyc_l - cyc_f, 1)
            df_aging['soh_physics_baseline'] = first + slope * (df_aging['cycle'] - cyc_f)
        else:
            df_aging['soh_physics_baseline'] = df_aging['soh_true']

        df_aging['residual_target'] = df_aging['soh_true'] - df_aging['soh_physics_baseline']

        out_aging = processed_dir / f"{battery}_aging_features.csv"
        df_aging.to_csv(out_aging, index=False)
        print(f"  ✅ {battery}: {len(df_aging)} cycles → {out_aging.name}")

        df_ts = pd.DataFrame(timeseries_records)
        out_ts = processed_dir / f"{battery}_discharge_timeseries.csv"
        df_ts.to_csv(out_ts, index=False)
        print(f"  ✅ {battery}: {len(df_ts)} timesteps → {out_ts.name}")

        # ---- OCV rest points ----
        ocv_pts = _extract_ocv_rest_points(mat_cycles, rated_capacity)
        if ocv_pts:
            df_ocv = pd.DataFrame(ocv_pts)
            out_ocv = processed_dir / f"{battery}_ocv_rest_points.csv"
            df_ocv.to_csv(out_ocv, index=False)
            print(f"  ✅ {battery}: {len(df_ocv)} OCV rest points → {out_ocv.name}")
        else:
            print(f"  ⚠️  {battery}: no OCV rest points found (impedance/charge entries missing)")

    print("🔬 Done parsing NASA dataset.")


if __name__ == "__main__":
    parse_nasa_mat_files()
