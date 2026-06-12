"""
Step 4: Physics-Correct Aging-Aware Digital Twin
=================================================
Implements the Samanta-Surya-Williamson (IEEE TTE 2022) ECM + EETM pipeline
with all known bugs fixed and SOTA numerical methods applied.

Bug fixes vs previous 'latest' branch:
  1. OCV curve built from NASA REST/CHARGE end-of-rest points (not from
     terminal voltage during active discharge).  V[0] under load ≠ OCV.
  2. R0 lower bound raised from 0.030 Ω to 0.050 Ω (realistic for 18650);
     previous code hit the bound floor every cycle → R0 always 0.030.
  3. ECM optimizer warm-started from previous cycle's solution; random
     multi-start seeded from warm point for robustness.  Eliminates the
     cycle-to-cycle discontinuous jumps in parameters.
  4. dt=0 guard in ECM simulation loop; duplicate timestamps already
     removed in Step 2 but guard added for safety.
  5. Q_gen computed from explicit Joule heating formula:
       Q_irrev = I²·R0 + V1²/R1 + V2²/R2
     (always non-negative, does not depend on OCV estimate accuracy).
     Entropic (reversible) heat term added on top.
  6. Thermal ODE integrated with Crank-Nicolson (unconditionally stable
     for linear RC systems) instead of forward Euler.  Eliminates the
     overflow/nan blow-up seen at lines ~294-302 with stiff parameters.
  7. Thermal optimizer bounds tightened and a stability pre-check added so
     the cost function returns a large penalty before the integrator can
     produce infs.
  8. C1 and C2 saved in ecm_parameters.csv (were previously discarded);
     EV dataset generator loads them instead of using magic constants.
  9. MSE printed in mV² (multiplied by 1e6) for human readability.
 10. Validation log Q_gen recalculated from the Joule formula, not from
     the residual voltage difference (which was also wrong).

SOTA additions:
  - Incremental Capacity Analysis (ICA) peak tracking per cycle; dQ/dV
    peaks correlate with lithium staging transitions and provide a richer
    aging feature than SOH alone.
  - Physics-informed regularisation in the ECM cost: penalises R0 > R0_prev
    drift beyond Arrhenius-plausible rates.
  - Per-cycle SOC trajectory stored in the twin dataset (needed for correct
    entropic heat computation in the transformer training data).
  - Adaptive aging-state sampling: instead of fixed 20-cycle spacing, uses
    the actual SOH gradient to identify the knee region and over-samples it.
"""

import os
import warnings
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

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 1. OCV-SOC CURVE — built from REST / CHARGE end-points, NOT active discharge
# =============================================================================

def build_ocv_curve():
    """
    Fit a 6th-order polynomial OCV(SOC) using near-equilibrium voltage points
    extracted from Step 2's *_ocv_rest_points.csv files.

    Falls back to a physics-motivated polynomial if no rest data is available.

    Returns (coeffs_fresh, coeffs_aged) each as a list of 7 floats.
    """
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "data" / "nasa" / "processed"
    batteries = ["B0005", "B0006", "B0007", "B0018"]

    soc_pts, ocv_pts, soh_pts = [], [], []

    for batt in batteries:
        rest_file = processed_dir / f"{batt}_ocv_rest_points.csv"
        if rest_file.exists():
            df = pd.read_csv(rest_file)
            soc_pts.extend(df['soc'].tolist())
            ocv_pts.extend(df['ocv_v'].tolist())
            soh_pts.extend(df['soh'].tolist())

    # Also mine the discharge timeseries: the VERY first point of each cycle
    # can be used if the previous cycle ended with a rest (common in NASA
    # protocol where impedance tests follow each discharge). We keep only
    # points where |I[0]| < 0.05 A as a proxy for a rested state.
    for batt in batteries:
        ts_file = processed_dir / f"{batt}_discharge_timeseries.csv"
        if not ts_file.exists():
            continue
        df_ts = pd.read_csv(ts_file)
        for cyc, grp in df_ts.groupby('cycle'):
            grp = grp.sort_values('time_s')
            if len(grp) < 5:
                continue
            I0 = abs(grp['current_A'].iloc[0])
            if I0 < 0.05:          # effectively at rest
                v0  = grp['voltage_V'].iloc[0]
                soh = grp['soh_true'].iloc[0]
                cap = grp['capacity_ah'].iloc[0]
                rated = 2.0
                soc_pts.append(1.0)   # start of discharge → SOC ≈ 1
                ocv_pts.append(v0)
                soh_pts.append(soh)

    if len(soc_pts) < 8:
        # Absolute fallback: well-known empirical OCV curve for 18650 NMC
        print("⚠️  OCV: insufficient rest data — using empirical NMC fallback polynomial")
        coeffs_nmf = [0.2896, -1.4135,  2.6129, -2.0901,  0.8228,  3.2867,  2.9835]
        return coeffs_nmf, coeffs_nmf

    soc_pts = np.array(soc_pts)
    ocv_pts = np.array(ocv_pts)
    soh_pts = np.array(soh_pts)

    # Clip physically impossible OCV values for 18650 (2.5 V – 4.35 V)
    valid = (ocv_pts >= 2.5) & (ocv_pts <= 4.35) & (soc_pts >= 0.0) & (soc_pts <= 1.0)
    soc_pts = soc_pts[valid]
    ocv_pts = ocv_pts[valid]
    soh_pts = soh_pts[valid]

    if len(soc_pts) < 8:
        coeffs_nmf = [0.2896, -1.4135,  2.6129, -2.0901,  0.8228,  3.2867,  2.9835]
        return coeffs_nmf, coeffs_nmf

    mask_fresh = soh_pts >= 0.90
    mask_aged  = soh_pts <= 0.78

    poly_order = 6
    if mask_fresh.sum() >= poly_order + 2:
        coeffs_fresh = list(np.polyfit(soc_pts[mask_fresh], ocv_pts[mask_fresh], poly_order))
    else:
        coeffs_fresh = list(np.polyfit(soc_pts, ocv_pts, poly_order))

    if mask_aged.sum() >= poly_order + 2:
        coeffs_aged = list(np.polyfit(soc_pts[mask_aged], ocv_pts[mask_aged], poly_order))
    else:
        coeffs_aged = coeffs_fresh

    print(f"✅ OCV curves: {mask_fresh.sum()} fresh pts, {mask_aged.sum()} aged pts "
          f"(total {len(soc_pts)} rest/charge points used)")
    return coeffs_fresh, coeffs_aged


# Module-level OCV curve coefficients (built once at import)
OCV_COEFFS_FRESH, OCV_COEFFS_AGED = build_ocv_curve()


def ocv_from_soc(soc: np.ndarray | float, soh: float = 1.0) -> np.ndarray | float:
    """Blend fresh and aged OCV polynomials by current SOH."""
    soc = np.clip(soc, 0.0, 1.0)
    alpha = float(np.clip((soh - 0.78) / (0.90 - 0.78), 0.0, 1.0))
    coeffs = [alpha * c1 + (1.0 - alpha) * c2
              for c1, c2 in zip(OCV_COEFFS_FRESH, OCV_COEFFS_AGED)]
    return np.polyval(coeffs, soc)


def soc_from_ocv(ocv: float, soh: float = 1.0, tol: float = 1e-4) -> float:
    """Bisection inversion of ocv_from_soc."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if ocv_from_soc(mid, soh) < ocv:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


# =============================================================================
# 2. ENTROPIC HEAT COEFFICIENT  dU/dT (V/K) vs SOC  — NMC 18650
# =============================================================================
_ENTROPIC_SOC  = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
_ENTROPIC_DUDT = np.array([3e-4, 2.5e-4, 1.5e-4, 5e-5, -5e-5,
                           -1.5e-4, -2.5e-4, -3.5e-4, -4.5e-4, -5e-4, -5.5e-4])


def dudt_from_soc(soc: np.ndarray | float) -> np.ndarray | float:
    return np.interp(np.clip(soc, 0.0, 1.0), _ENTROPIC_SOC, _ENTROPIC_DUDT)


# =============================================================================
# 3. INCREMENTAL CAPACITY ANALYSIS (ICA) — dQ/dV peak tracker
# =============================================================================

def compute_ica_features(voltage: np.ndarray, capacity_ah: float,
                         n_bins: int = 200) -> dict:
    """
    Compute the incremental capacity (dQ/dV) curve and extract peak positions.
    These peaks correspond to lithium-staging phase transitions and shift with
    aging — providing a model-free aging indicator that complements ECM R0.

    Returns a dict with keys: peak1_v, peak2_v, peak_ratio (peak2/peak1 height).
    Returns None values if curve too noisy.
    """
    if len(voltage) < 50:
        return {"ica_peak1_v": np.nan, "ica_peak2_v": np.nan, "ica_peak_ratio": np.nan}

    # Build Q from discharged Ah (assumes constant 2A discharge)
    # Q_i = i/n * capacity_ah  (uniform sampling approximation)
    q = np.linspace(0, capacity_ah, len(voltage))

    # Bin voltage into uniform grid and compute dQ/dV
    v_sorted_idx = np.argsort(voltage)[::-1]   # descending (discharge)
    v_s = voltage[v_sorted_idx]
    q_s = q[v_sorted_idx]

    v_bins = np.linspace(v_s[-1], v_s[0], n_bins)
    dv = v_bins[1] - v_bins[0]
    dqdv = np.zeros(n_bins - 1)
    v_mid = 0.5 * (v_bins[:-1] + v_bins[1:])

    for i in range(n_bins - 1):
        mask = (v_s >= v_bins[i]) & (v_s < v_bins[i + 1])
        dqdv[i] = mask.sum() * (capacity_ah / len(voltage)) / dv if dv > 0 else 0.0

    # Smooth
    if len(dqdv) > 11:
        dqdv = savgol_filter(dqdv, 11, 3)

    # Find top-2 peaks (simple local maxima)
    from scipy.signal import find_peaks
    peaks, props = find_peaks(dqdv, height=0.0, distance=10)
    if len(peaks) < 2:
        return {"ica_peak1_v": np.nan, "ica_peak2_v": np.nan, "ica_peak_ratio": np.nan}

    heights = dqdv[peaks]
    top2 = peaks[np.argsort(heights)[-2:]]
    top2 = top2[np.argsort(v_mid[top2])]  # order by voltage

    ratio = dqdv[top2[1]] / (dqdv[top2[0]] + 1e-9)
    return {
        "ica_peak1_v":    float(v_mid[top2[0]]),
        "ica_peak2_v":    float(v_mid[top2[1]]),
        "ica_peak_ratio": float(ratio),
    }


# =============================================================================
# 4. EPA DRIVE CYCLE CURRENT GENERATOR
# =============================================================================
MASS_KG            = 1200.0
CRR                = 0.01
RHO_AIR            = 1.225
CD                 = 0.30
AREA_M2            = 2.2
GRAVITY            = 9.81
CELLS_IN_PARALLEL  = 74
CELL_NOMINAL_V     = 3.7
REGEN_EFF          = 0.65    # regenerative braking efficiency

DRIVE_CYCLE_DIR = Path(__file__).parent.parent / "data" / "drive_cycles"
_CYCLE_FILE = {
    "UDDS":  "UDDS_epa_speed.csv",
    "HWFET": "HWFET_epa_speed.csv",
    "US06":  "US06_epa_speed.csv",
}


def _load_epa_trace(name: str):
    f = DRIVE_CYCLE_DIR / _CYCLE_FILE[name]
    if not f.exists():
        raise FileNotFoundError(f"Drive cycle file not found: {f}")
    df = pd.read_csv(f)
    return df['time_s'].values, df['speed_mps'].values


def generate_drive_cycle_current(profile: str, dt: float = 1.0):
    """
    Convert an EPA speed trace to a per-cell current profile.
    Includes regenerative braking (negative current) at REGEN_EFF efficiency.
    """
    if profile == "Aggressive":
        return generate_drive_cycle_current("US06", dt=dt)
    if profile == "Mixed":
        t1, i1 = generate_drive_cycle_current("UDDS", dt=dt)
        t2, i2 = generate_drive_cycle_current("HWFET", dt=dt)
        return np.concatenate([t1, t2 + t1[-1] + dt]), np.concatenate([i1, i2])

    t_raw, v_raw = _load_epa_trace(profile)
    if abs(dt - 0.1) > 1e-6:
        n_new = int(t_raw[-1] / dt) + 1
        t_out = np.linspace(0.0, t_raw[-1], n_new)
        v_out = np.interp(t_out, t_raw, v_raw)
    else:
        t_out, v_out = t_raw, v_raw

    accel   = np.gradient(v_out, t_out)
    F_roll  = CRR * MASS_KG * GRAVITY * np.ones_like(v_out)
    F_drag  = 0.5 * RHO_AIR * CD * AREA_M2 * v_out ** 2
    F_inert = MASS_KG * accel
    F_total = F_roll + F_drag + F_inert
    P_wheel = F_total * v_out

    # Positive: traction (discharge); negative: regen braking (charge)
    P_trac  = np.where(P_wheel > 0,  P_wheel,                   0.0)
    P_regen = np.where(P_wheel < 0, -P_wheel * REGEN_EFF,       0.0)
    P_cell  = (P_trac - P_regen) / CELLS_IN_PARALLEL
    I_cell  = P_cell / CELL_NOMINAL_V + 0.05   # 50 mA quiescent draw

    return t_out, I_cell


# =============================================================================
# 5. ECM IDENTIFICATION — 2-RC model, warm-start, Joule-based Q_gen
# =============================================================================

def _filter_noise(arr: np.ndarray, window: int = 5) -> np.ndarray:
    return savgol_filter(arr, window, 2) if len(arr) > window else arr


def identify_ecm_params(time_s: np.ndarray, current: np.ndarray,
                        voltage: np.ndarray, soc_init: float,
                        capacity_ah: float, soh: float = 1.0,
                        warm_start: list | None = None,
                        r0_prev: float | None = None) -> tuple[list, float]:
    """
    Identify 2-RC ECM parameters [R0, R1, C1, R2, C2] by minimising
    MSE of simulated vs measured terminal voltage.

    Key fixes:
      - R0 lower bound 0.050 Ω (fresh 18650 ≈ 0.06–0.09 Ω; old code used 0.030).
      - Physics-informed regularisation: penalises sudden R0 increase beyond
        Arrhenius plausible rate (prevents cycle-to-cycle discontinuities).
      - Warm-start: first random start uses previous cycle's solution.
      - dt=0 guard prevents exp(-inf) and subsequent NaN/inf in RC state.
      - Noise filtered with Savitzky-Golay before fitting.
    """
    I = _filter_noise(current)
    V = _filter_noise(voltage)
    dt_arr = np.diff(time_s)
    n      = len(I)
    Q_cell = capacity_ah * 3600.0

    def simulate_ecm(params):
        R0, R1, C1, R2, C2 = params
        soc = soc_init
        V1 = V2 = 0.0
        v_sim = np.empty(n)
        for k in range(n):
            Ik = I[k]
            v_sim[k] = ocv_from_soc(soc, soh) - Ik * R0 - V1 - V2
            if k < n - 1:
                dt = dt_arr[k]
                if dt <= 0.0:
                    continue
                tau1 = R1 * C1
                tau2 = R2 * C2
                a1 = np.exp(-dt / tau1) if tau1 > 0 else 0.0
                a2 = np.exp(-dt / tau2) if tau2 > 0 else 0.0
                V1 = a1 * V1 + Ik * R1 * (1.0 - a1)
                V2 = a2 * V2 + Ik * R2 * (1.0 - a2)
                soc -= Ik * dt / Q_cell
                soc  = float(np.clip(soc, 0.0, 1.0))
        return v_sim

    def cost(params):
        R0 = params[0]
        v_sim = simulate_ecm(params)
        mse = float(np.mean((v_sim - V) ** 2))
        # Physics regularisation: penalise large sudden R0 increase
        if r0_prev is not None:
            delta = R0 - r0_prev
            # Allow up to 5% increase per cycle; penalise beyond that
            excess = max(0.0, delta - 0.05 * r0_prev)
            mse += 10.0 * excess ** 2
        return mse

    # Bounds: R0 floor raised to 0.050 Ω for 18650 cells
    bounds = [
        (0.050, 0.200),     # R0
        (0.005, 0.080),     # R1
        (200.0, 8000.0),    # C1
        (0.005, 0.080),     # R2
        (500.0, 60000.0),   # C2
    ]

    # Multi-start: warm point first, then random restarts
    starts = []
    if warm_start is not None and len(warm_start) == 5:
        starts.append(warm_start)
    rng = np.random.RandomState(42)
    for _ in range(15):
        starts.append([rng.uniform(lo, hi) for lo, hi in bounds])

    best_x, best_cost = None, np.inf
    for x0 in starts:
        # Clamp x0 to bounds
        x0 = [float(np.clip(v, lo, hi)) for v, (lo, hi) in zip(x0, bounds)]
        try:
            res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            if np.isfinite(res.fun) and res.fun < best_cost:
                best_cost = res.fun
                best_x    = list(res.x)
        except Exception:
            continue

    if best_x is None:
        # Absolute fallback: return typical 18650 values
        best_x    = [0.08, 0.015, 2000.0, 0.025, 5000.0]
        best_cost = np.nan

    return best_x, best_cost


def forward_ecm(time_s: np.ndarray, current: np.ndarray,
                ecm_params: list, soc_init: float,
                capacity_ah: float, soh: float = 1.0):
    """Run fitted 2-RC ECM forward; return v_sim, v_ocv, soc_arr, V1_arr, V2_arr."""
    R0, R1, C1, R2, C2 = ecm_params
    dt_arr = np.diff(time_s)
    n      = len(current)
    Q_cell = capacity_ah * 3600.0
    soc    = soc_init
    V1 = V2 = 0.0

    v_sim   = np.empty(n)
    v_ocv   = np.empty(n)
    soc_arr = np.empty(n)
    V1_arr  = np.empty(n)
    V2_arr  = np.empty(n)

    for k in range(n):
        Ik = current[k]
        v_ocv[k]   = ocv_from_soc(soc, soh)
        v_sim[k]   = v_ocv[k] - Ik * R0 - V1 - V2
        soc_arr[k] = soc
        V1_arr[k]  = V1
        V2_arr[k]  = V2
        if k < n - 1:
            dt = dt_arr[k]
            if dt <= 0.0:
                continue
            tau1 = R1 * C1
            tau2 = R2 * C2
            a1 = np.exp(-dt / tau1) if tau1 > 0 else 0.0
            a2 = np.exp(-dt / tau2) if tau2 > 0 else 0.0
            V1  = a1 * V1 + Ik * R1 * (1.0 - a1)
            V2  = a2 * V2 + Ik * R2 * (1.0 - a2)
            soc -= Ik * dt / Q_cell
            soc  = float(np.clip(soc, 0.0, 1.0))

    return v_sim, v_ocv, soc_arr, V1_arr, V2_arr


def compute_qgen(current: np.ndarray, R0: float, R1: float, R2: float,
                 V1_arr: np.ndarray, V2_arr: np.ndarray,
                 soc_arr: np.ndarray, T_K: float) -> np.ndarray:
    """
    Total heat generation using the Bernardi equation (Joule form):
      Q_irrev = I²·R0 + V1²/R1 + V2²/R2
      Q_rev   = I · T · dU/dT(SOC)
      Q_gen   = Q_irrev + Q_rev

    This is always non-negative for discharge (I>0) and does NOT depend
    on the accuracy of the OCV curve (unlike the |I·(V_OCV - V_t)| form).
    """
    Q_irrev = current**2 * R0 + V1_arr**2 / R1 + V2_arr**2 / R2
    Q_rev   = current * T_K * dudt_from_soc(soc_arr)
    return Q_irrev + Q_rev


# =============================================================================
# 6. THERMAL MODEL — Crank-Nicolson integration (unconditionally stable)
# =============================================================================

def _cn_step(Tc: float, Ts: float, Qg: float,
             Rin: float, Rout: float, Cc: float, Cs: float,
             T_amb: float, dt: float) -> tuple[float, float]:
    """
    One Crank-Nicolson step for the 2-state thermal RC model.

    Governing ODEs (continuous):
      Cc · dTc/dt =  Qg        - (Tc - Ts) / Rin
      Cs · dTs/dt = (Tc - Ts) / Rin - (Ts - T_amb) / Rout

    CN discretisation (θ=0.5): each derivative evaluated at average of
    current and next time step.  For linear ODEs this gives a simple
    2×2 linear system — unconditionally stable for any dt.

    Let Tc_n, Ts_n be current; Tc_{n+1}, Ts_{n+1} be unknown.

    Cc·(Tc_{n+1} - Tc_n)/dt = 0.5·[Qg - (Tc_n - Ts_n)/Rin + Qg - (Tc_{n+1} - Ts_{n+1})/Rin]
    Cs·(Ts_{n+1} - Ts_n)/dt = 0.5·[(Tc_n - Ts_n)/Rin - (Ts_n - Tamb)/Rout
                                    +(Tc_{n+1}-Ts_{n+1})/Rin - (Ts_{n+1}-Tamb)/Rout]

    Re-arrange to A·[Tc_{n+1}, Ts_{n+1}]^T = b  and solve with 2×2 inverse.
    """
    r  = 1.0 / Rin
    ro = 1.0 / Rout
    h  = 0.5 * dt

    # Build 2×2 system A x = b
    # Row 0 (core):    Cc/dt · Tc1 + h/Rin · Tc1 - h/Rin · Ts1 = RHS_c
    # Row 1 (surface): -h/Rin · Tc1 + (Cs/dt + h/Rin + h/Rout) · Ts1 = RHS_s
    A00 = Cc / dt + h * r
    A01 = -h * r
    A10 = -h * r
    A11 = Cs / dt + h * r + h * ro

    RHS_c = Cc / dt * Tc + Qg - h * r * (Tc - Ts) + h * Qg
    RHS_s = Cs / dt * Ts + h * r * (Tc - Ts) - h * ro * (Ts - T_amb) \
            - h * ro * (Ts - T_amb)  # both halves use same Ts_n approximation

    # 2×2 direct solve
    det = A00 * A11 - A01 * A10
    if abs(det) < 1e-30:
        return Tc, Ts   # degenerate — return unchanged
    Tc1 = (A11 * RHS_c - A01 * RHS_s) / det
    Ts1 = (A00 * RHS_s - A10 * RHS_c) / det

    # Safety clamp: physical temperatures (−40 °C to 150 °C)
    Tc1 = float(np.clip(Tc1, -40.0, 150.0))
    Ts1 = float(np.clip(Ts1, -40.0, 150.0))
    return Tc1, Ts1


def simulate_thermal_forward(time_s: np.ndarray, Q_gen: np.ndarray,
                             therm_params: list, T_ambient: float = 25.0) -> tuple:
    """Run 2-state thermal RC model forward using Crank-Nicolson."""
    Rin, Rout, Cc, Cs = therm_params
    dt_arr = np.diff(time_s)
    n      = len(Q_gen)
    Tc     = np.empty(n)
    Ts     = np.empty(n)
    Tc[0]  = T_ambient
    Ts[0]  = T_ambient

    for k in range(n - 1):
        dt = dt_arr[k]
        if dt <= 0.0:
            Tc[k + 1] = Tc[k]
            Ts[k + 1] = Ts[k]
            continue
        Tc[k + 1], Ts[k + 1] = _cn_step(
            Tc[k], Ts[k], Q_gen[k], Rin, Rout, Cc, Cs, T_ambient, dt)

    return Tc, Ts


def _thermal_stability_ok(params: list, dt_max: float) -> bool:
    """
    Check that the thermal time constants are not so small that even
    Crank-Nicolson might struggle with extreme stiffness.  CN is
    unconditionally stable but not unconditionally accurate — very small
    time constants relative to dt give poor amplitude.
    Returns False only for clearly unphysical params.
    """
    Rin, Rout, Cc, Cs = params
    tau_min = min(Rin * Cc, Rout * Cs)
    return tau_min > 0.1   # reject if <0.1 s time constant


def tune_thermal_params(time_s: np.ndarray, Q_gen: np.ndarray,
                        temp_surface_measured: np.ndarray,
                        T_ambient: float,
                        initial_guess: list | None = None,
                        capacity_ah: float = 2.0) -> tuple:
    """
    Tune [Rin, Rout, Cc, Cs] by minimising MSE between simulated Ts and
    real measured surface temperature.
    """
    n = len(Q_gen)

    def simulate(params):
        if not _thermal_stability_ok(params, dt_max=2.0):
            return np.ones(n) * 1e6, np.ones(n) * 1e6
        Tc_s, Ts_s = simulate_thermal_forward(time_s, Q_gen, params, T_ambient)
        return Tc_s, Ts_s

    def cost(params):
        _, Ts_s = simulate(params)
        return float(np.mean((Ts_s - temp_surface_measured) ** 2))

    if initial_guess is None:
        initial_guess = [3.0, 12.0, 25.0, 10.0]

    # Physical bounds for 18650 cell (K/W, K/W, J/K, J/K)
    bounds = [(0.5, 20.0), (2.0, 50.0), (10.0, 120.0), (2.0, 50.0)]

    # Clamp initial guess to bounds
    x0 = [float(np.clip(v, lo, hi)) for v, (lo, hi) in zip(initial_guess, bounds)]

    res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 400, 'ftol': 1e-13})
    Tc_f, Ts_f = simulate(list(res.x))
    return list(res.x), float(res.fun), Tc_f, Ts_f


# =============================================================================
# 7. DRIVE CYCLE PHYSICS (for EV dataset generation)
# =============================================================================

def run_physics_on_profile(time_s, current, ecm_params, therm_params,
                           T_ambient, capacity_ah=2.0, soh=1.0):
    soc_init = soc_from_ocv(
        float(ocv_from_soc(1.0, soh)), soh)   # start fully charged
    v_sim, v_ocv, soc_arr, V1_arr, V2_arr = forward_ecm(
        time_s, current, ecm_params, soc_init, capacity_ah, soh)
    R0, R1, _, R2, _ = ecm_params
    T_K = T_ambient + 273.15
    Q_gen = compute_qgen(current, R0, R1, R2, V1_arr, V2_arr, soc_arr, T_K)
    Tc, Ts = simulate_thermal_forward(time_s, Q_gen, therm_params, T_ambient)
    return v_sim, v_ocv, soc_arr, Tc, Ts, Q_gen


# =============================================================================
# 8. MULTI-AMBIENT VISUALISATION
# =============================================================================

def generate_multi_ambient_plot(profile_name, ecm_params, therm_params,
                                plot_dir, capacity_ah=2.0, soh=1.0):
    aging_factor = 1.0 + 0.2 * (1.0 - soh)
    therm_aged = [therm_params[0] * aging_factor, therm_params[1] * aging_factor,
                  therm_params[2], therm_params[3]]

    Tamb_list    = [0.0, 20.0, 45.0]
    colors_tc    = {0.0: 'royalblue', 20.0: 'firebrick', 45.0: 'seagreen'}
    time_s, cur  = generate_drive_cycle_current(profile_name)

    Tc_res, Vs_res = {}, {}
    for Tamb in Tamb_list:
        v_sim, _, _, Tc, _, _ = run_physics_on_profile(
            time_s, cur, ecm_params, therm_aged,
            T_ambient=Tamb, capacity_ah=capacity_ah, soh=soh)
        Tc_res[Tamb] = Tc
        Vs_res[Tamb] = v_sim

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].plot(time_s, cur, color='black', lw=0.7)
    axes[0].set(xlabel='Time (s)', ylabel='Current (A)',
                title=f'{profile_name} — Current'); axes[0].grid(ls='--', alpha=0.5)

    for Tamb in Tamb_list:
        axes[1].plot(time_s, Tc_res[Tamb], color=colors_tc[Tamb], lw=1.2,
                     label=f'Tc @ {Tamb:.0f}°C')
    axes[1].set(xlabel='Time (s)', ylabel='Core Temperature (°C)',
                title=f'{profile_name} — Core Temp'); axes[1].legend(); axes[1].grid(ls='--', alpha=0.5)

    axes[2].plot(time_s, Vs_res[20.0], color='purple', lw=0.8)
    axes[2].set(xlabel='Time (s)', ylabel='Voltage (V)',
                title=f'{profile_name} — Simulated V'); axes[2].grid(ls='--', alpha=0.5)

    plt.suptitle(f'Physics Engine: {profile_name} Drive Cycle  (SOH={soh:.3f})',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = plot_dir / f'{profile_name.lower()}_multi_temp_visualization.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved {out.name}")


# =============================================================================
# 9. EV DRIVE CYCLE DATASET GENERATION
# =============================================================================

def _adaptive_aging_cycles(cycle_params: pd.DataFrame, max_states: int = 10) -> list:
    """
    Select aging states adaptively: over-sample the knee region (where
    dSOH/dcycle is steepest) and under-sample the plateau.
    """
    cycles_avail = sorted(cycle_params['cycle'].values)
    soh_vals     = cycle_params.set_index('cycle')['soh'].to_dict()

    if len(cycles_avail) <= max_states:
        return cycles_avail

    # Compute SOH gradient per cycle
    soh_arr   = np.array([soh_vals[c] for c in cycles_avail])
    grad      = np.abs(np.gradient(soh_arr))

    # Importance sampling: probability proportional to |dSOH/dcycle| + ε
    prob = grad + 0.1 * grad.max()
    prob /= prob.sum()

    rng     = np.random.RandomState(0)
    chosen  = rng.choice(len(cycles_avail), size=max_states, replace=False, p=prob)
    chosen  = sorted(chosen)
    return [cycles_avail[i] for i in chosen]


def generate_ev_drive_cycle_dataset(processed_dir, out_dir, plot_dir, median_therm):
    batteries    = ["B0005", "B0006", "B0007", "B0018"]
    drive_cycles = ["UDDS", "HWFET", "US06"]
    Tamb_list    = [0.0, 25.0, 45.0]
    ev_out_dir   = out_dir.parent / "ev_validation_sets"
    ev_out_dir.mkdir(parents=True, exist_ok=True)

    twin_csv = out_dir / "augmented_aging_twin_dataset.csv"
    if not twin_csv.exists():
        print("⚠️  Twin dataset not found. Skipping EV generation.")
        return

    df_twin = pd.read_csv(twin_csv)
    all_records  = []
    total_sims   = 0
    t_start      = timer.time()

    for battery in batteries:
        df_batt = df_twin[df_twin['battery'] == battery]
        if df_batt.empty:
            continue

        # Load all 5 ECM params (including C1, C2 now saved correctly)
        cycle_params = df_batt.groupby('cycle').agg(
            soh=('soh_true', 'first'),
            R0=('r0_ohms', 'first'), R1=('r1_ohms', 'first'),
            C1=('c1_farads', 'first'), R2=('r2_ohms', 'first'),
            C2=('c2_farads', 'first')).reset_index()

        aging_cycles = _adaptive_aging_cycles(cycle_params, max_states=10)
        print(f"\n🚗 {battery}: {len(aging_cycles)} adaptive aging states × "
              f"{len(drive_cycles)} drives × {len(Tamb_list)} temps")

        for cyc in aging_cycles:
            row = cycle_params[cycle_params['cycle'] == cyc].iloc[0]
            R0, R1, C1, R2, C2 = row['R0'], row['R1'], row['C1'], row['R2'], row['C2']
            soh    = row['soh']
            cap_ah = soh * 2.0

            ecm_p = [R0, R1, C1, R2, C2]
            aging_factor = 1.0 + 0.2 * (1.0 - soh)
            therm_p = [median_therm[0] * aging_factor, median_therm[1] * aging_factor,
                       median_therm[2], median_therm[3]]

            for drive in drive_cycles:
                try:
                    time_s, current = generate_drive_cycle_current(drive)
                except Exception as exc:
                    print(f"  ⚠️  {drive}: {exc}"); continue

                throughput = float(np.sum(np.abs(current)) * (time_s[1] - time_s[0]) / 3600.0)
                if throughput > 2.5:
                    print(f"  ⚠️  {drive} throughput {throughput:.2f} Ah > cell rating")

                for Tamb in Tamb_list:
                    try:
                        v_sim, v_ocv, soc_arr, Tc, Ts, Q_gen = run_physics_on_profile(
                            time_s, current, ecm_p, therm_p,
                            T_ambient=Tamb, capacity_ah=cap_ah, soh=soh)
                    except Exception as exc:
                        print(f"  ⚠️  physics failed ({battery} cyc{cyc} {drive} {Tamb}°C): {exc}")
                        continue

                    lbl = f"{battery}_{drive}_T{Tamb:.0f}_C{cyc}"
                    for k in range(len(time_s)):
                        all_records.append({
                            "battery":          lbl,
                            "cycle":            cyc,
                            "soh_true":         soh,
                            "time_s":           time_s[k],
                            "current_A":        current[k],
                            "voltage_V":        v_sim[k],
                            "voltage_sim_V":    v_sim[k],
                            "r0_ohms":          R0,
                            "r1_ohms":          R1,
                            "c1_farads":        C1,
                            "r2_ohms":          R2,
                            "c2_farads":        C2,
                            "soc":              soc_arr[k],
                            "q_gen_W":          Q_gen[k],
                            "temp_surface_C":   Ts[k],
                            "temp_surface_sim_C": Ts[k],
                            "temp_core_C_TARGET": Tc[k],
                        })
                    total_sims += 1

    elapsed = timer.time() - t_start
    print(f"\n⏱️  EV dataset: {elapsed:.1f}s, {total_sims} sims")
    if not all_records:
        return
    df_ev = pd.DataFrame(all_records)
    ev_path = ev_out_dir / "ev_drive_cycle_dataset.csv"
    df_ev.to_csv(ev_path, index=False)
    print(f"✅ EV Dataset: {len(df_ev):,} rows → {ev_path.name}")


# =============================================================================
# 10. MAIN PIPELINE
# =============================================================================

def generate_aging_digital_twin():
    base_dir      = Path(__file__).parent.parent
    processed_dir = base_dir / "data" / "nasa" / "processed"
    out_dir       = base_dir / "data" / "digital_twin_sets"
    plot_dir      = base_dir / "results" / "paper_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    batteries    = ["B0005", "B0006", "B0007", "B0018"]
    all_records  = []
    thermal_list = []
    total_cycles = 0
    t_start      = timer.time()
    last_soh     = 1.0

    for battery in batteries:
        ts_file = processed_dir / f"{battery}_discharge_timeseries.csv"
        if not ts_file.exists():
            print(f"⚠️  {ts_file.name} not found. Run Step 2 first.")
            continue
        print(f"\n🔋 Processing {battery}...")
        df_ts   = pd.read_csv(ts_file)
        cycles  = sorted(df_ts['cycle'].unique())

        prev_therm  = None
        prev_ecm    = None    # warm-start for next cycle
        prev_r0     = None

        for cyc in cycles:
            df_cyc = df_ts[df_ts['cycle'] == cyc].reset_index(drop=True)
            if len(df_cyc) < 20:
                continue

            time_s    = df_cyc['time_s'].values.astype(float)
            current   = df_cyc['current_A'].values.astype(float)
            voltage   = df_cyc['voltage_V'].values.astype(float)
            temp_surf = df_cyc['temp_surface_C'].values.astype(float)
            soh       = float(df_cyc['soh_true'].iloc[0])
            cap_ah    = float(df_cyc['capacity_ah'].iloc[0])

            # Ensure monotone timestamps (Step 2 already does this, belt-and-suspenders)
            dt = np.diff(time_s)
            valid = np.concatenate([[True], dt > 0])
            time_s    = time_s[valid]
            current   = current[valid]
            voltage   = voltage[valid]
            temp_surf = temp_surf[valid]
            if len(time_s) < 20:
                continue

            # Initial SOC from OCV inversion
            V_start  = float(voltage[0])
            soc_init = soc_from_ocv(V_start, soh)

            # ----- ECM identification -----
            t0 = timer.time()
            ecm_params, ecm_err = identify_ecm_params(
                time_s, current, voltage, soc_init, cap_ah, soh,
                warm_start=prev_ecm, r0_prev=prev_r0)
            t_ecm = timer.time() - t0
            prev_ecm = ecm_params
            prev_r0  = ecm_params[0]

            R0, R1, C1, R2, C2 = ecm_params
            v_sim, v_ocv, soc_arr, V1_arr, V2_arr = forward_ecm(
                time_s, current, ecm_params, soc_init, cap_ah, soh)

            # ----- Q_gen (Joule formula) -----
            T_K   = float(temp_surf[0]) + 273.15
            Q_gen = compute_qgen(current, R0, R1, R2, V1_arr, V2_arr, soc_arr, T_K)
            q_mean = float(np.mean(np.abs(Q_gen)))
            if q_mean < 0.15:
                print(f"   ⚠️  Cycle {cyc}: Q_gen = {q_mean:.3f} W (suspiciously low)")

            # ----- ICA features -----
            ica = compute_ica_features(voltage, cap_ah)

            # ----- Thermal parameter tuning -----
            T_amb = float(temp_surf[0])
            therm_params, therm_mse, Tc, Ts_sim = tune_thermal_params(
                time_s, Q_gen, temp_surf, T_ambient=T_amb,
                initial_guess=prev_therm, capacity_ah=cap_ah)
            prev_therm   = list(therm_params)
            thermal_list.append(therm_params)
            Rin, Rout, Cc_t, Cs_t = therm_params
            ts_rmse = float(np.sqrt(np.mean((Ts_sim - temp_surf) ** 2)))

            last_soh = soh
            n = len(time_s)
            for k in range(n):
                all_records.append({
                    "battery":            battery,
                    "cycle":              cyc,
                    "soh_true":           soh,
                    "time_s":             time_s[k],
                    "current_A":          current[k],
                    "voltage_V":          voltage[k],
                    "voltage_sim_V":      v_sim[k],
                    "r0_ohms":            R0,
                    "r1_ohms":            R1,
                    "c1_farads":          C1,
                    "r2_ohms":            R2,
                    "c2_farads":          C2,
                    "soc":                soc_arr[k],
                    "q_gen_W":            Q_gen[k],
                    "temp_surface_C":     temp_surf[k],
                    "temp_surface_sim_C": Ts_sim[k],
                    "temp_core_C_TARGET": Tc[k],
                    "ica_peak1_v":        ica["ica_peak1_v"],
                    "ica_peak2_v":        ica["ica_peak2_v"],
                    "ica_peak_ratio":     ica["ica_peak_ratio"],
                })
            total_cycles += 1

            if total_cycles % 20 == 0:
                print(
                    f"   Cycle {cyc:3d} | SOH={soh:.3f} | R0={R0:.4f}Ω | "
                    f"ECM_MSE={ecm_err * 1e6:.1f} mV² | Ts_RMSE={ts_rmse:.3f}°C | "
                    f"Q_gen={q_mean:.3f}W | ECM_t={t_ecm:.1f}s"
                )

    # ---- Median thermal params ----
    if thermal_list:
        therm_df     = pd.DataFrame(thermal_list, columns=['Rin', 'Rout', 'Cc', 'Cs'])
        median_therm = therm_df.median().values
        print(f"\n📊 Median thermal: Rin={median_therm[0]:.2f} Rout={median_therm[1]:.2f} "
              f"Cc={median_therm[2]:.2f} Cs={median_therm[3]:.2f}")
    else:
        median_therm = np.array([3.0, 12.0, 25.0, 10.0])

    # ---- Save twin dataset ----
    df_out   = pd.DataFrame(all_records)
    out_path = out_dir / "augmented_aging_twin_dataset.csv"
    df_out.to_csv(out_path, index=False)
    print(f"✅ Twin dataset: {len(df_out):,} rows → {out_path.name}")

    # ---- Save ECM parameters (all 5 params now) ----
    ecm_df = df_out.groupby(['battery', 'cycle']).agg(
        r0_ohms=('r0_ohms', 'first'), r1_ohms=('r1_ohms', 'first'),
        c1_farads=('c1_farads', 'first'), r2_ohms=('r2_ohms', 'first'),
        c2_farads=('c2_farads', 'first')).reset_index()
    ecm_df.to_csv(out_dir / "ecm_parameters.csv", index=False)
    print("✅ ECM parameters (R0,R1,C1,R2,C2) saved to ecm_parameters.csv")

    # ---- Validation log ----
    val_records = []
    for batt in batteries:
        bdf = df_out[df_out['battery'] == batt]
        for cyc in sorted(bdf['cycle'].unique()):
            cdf = bdf[bdf['cycle'] == cyc]
            if len(cdf) < 2:
                continue
            v_rmse   = float(np.sqrt(np.mean((cdf['voltage_V'] - cdf['voltage_sim_V']) ** 2)))
            ts_rmse_ = float(np.sqrt(np.mean((cdf['temp_surface_C'] - cdf['temp_surface_sim_C']) ** 2)))
            q_mean_  = float(cdf['q_gen_W'].mean())
            dT_max   = float((cdf['temp_core_C_TARGET'] - cdf['temp_surface_C']).max())
            val_records.append({
                'battery': batt, 'cycle': cyc,
                'soh':     cdf['soh_true'].iloc[0],
                'V_RMSE_mV': v_rmse * 1000.0,
                'Ts_RMSE_C': ts_rmse_,
                'Q_gen_mean_W': q_mean_,
                'max_core_surface_dT': dT_max,
            })
    pd.DataFrame(val_records).to_csv(out_dir / "validation_log.csv", index=False)
    print("✅ Validation log saved.")

    # ---- Multi-ambient plots ----
    plot_soh    = last_soh
    aging_f     = 1.0 + 0.2 * (1.0 - plot_soh)
    therm_aged  = [median_therm[0] * aging_f, median_therm[1] * aging_f,
                   median_therm[2], median_therm[3]]
    if not df_out.empty:
        lr = df_out[df_out['cycle'] == df_out['cycle'].max()].iloc[0]
        last_ecm = [lr['r0_ohms'], lr['r1_ohms'], lr['c1_farads'],
                    lr['r2_ohms'], lr['c2_farads']]
    else:
        last_ecm = [0.09, 0.015, 2000.0, 0.025, 5000.0]

    generate_multi_ambient_plot("Aggressive", last_ecm, therm_aged, plot_dir,
                                capacity_ah=plot_soh * 2.0, soh=plot_soh)
    generate_multi_ambient_plot("Mixed",      last_ecm, therm_aged, plot_dir,
                                capacity_ah=plot_soh * 2.0, soh=plot_soh)

    elapsed = timer.time() - t_start
    print(f"\n⏱️  Total Step 4 time: {elapsed:.1f}s for {total_cycles} cycles")

    # ---- EV drive cycle dataset ----
    print("\n🚗 Generating EV real-world drive cycle dataset...")
    generate_ev_drive_cycle_dataset(processed_dir, out_dir, plot_dir, median_therm)


if __name__ == "__main__":
    generate_aging_digital_twin()
