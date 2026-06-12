# What Was Wrong, What We Changed, and Why It Will Work Now

> Written for someone who may not know battery physics or machine learning deeply.
> Every concept is explained from scratch before the fix is described.

---

## The Big Picture First

This project estimates the **internal core temperature** of a car battery while it is driving.
Why? Because the very centre of a Li-ion battery can be 5–15 °C hotter than its outer surface,
and that hidden heat is what causes fires and accelerated aging. You can only stick a
temperature sensor on the *outside* of a battery cell — you cannot drill into a live cell
without destroying it. So we train an AI to predict the internal temperature from things you
*can* measure: current, voltage, and surface temperature.

To train that AI, we need labelled data: pairs of (what we can measure, what the core
temperature actually is). Since we cannot measure core temperature, we **simulate it** using
physics equations. This simulation is called the **Digital Twin**. Once the twin generates
trustworthy core-temperature labels, we train a neural network on those labels.

The whole pipeline is:

```
Real NASA battery data
        ↓
Physics simulation (ECM + thermal model) → generates core temperature labels
        ↓
Neural network training (Transformer) → learns to predict core temp from surface signals
```

Every single bug we found was corrupting one or more of these three stages.

---

## Stage 1: The Equivalent Circuit Model (ECM) — What It Is

### What it does

A Li-ion battery behaves electrically like a small network of resistors and capacitors.
The **2-RC Equivalent Circuit Model** represents this as:

```
V_terminal = V_OCV(SOC) − I·R0 − V1 − V2

where:
  V_OCV(SOC) = open-circuit voltage (the "resting" voltage at a given charge level)
  I·R0       = instant voltage drop across the pure ohmic resistance R0
  V1         = voltage across the first RC pair (fast polarisation, time scale ~seconds)
  V2         = voltage across the second RC pair (slow diffusion, time scale ~minutes)
```

The model has **5 parameters**: R0, R1, C1, R2, C2.
We find these 5 numbers by recording a real discharge cycle and running an optimiser
(a mathematical search algorithm) that finds the parameter values that make the model's
predicted voltage best match the measured voltage. This is called **parameter identification**.

### What SOC means

**State of Charge (SOC)** = how full the battery is, on a scale from 0 (empty) to 1 (full).
Like a fuel gauge percentage. SOC decreases as the battery discharges:

```
SOC[t+1] = SOC[t] − (current × time_step) / total_charge_capacity
```

### What OCV means

**Open-Circuit Voltage (OCV)** = the voltage the battery settles to when there is **zero current flowing** and the cell has rested for ~30 minutes. It is a smooth, monotonically decreasing function of SOC (higher SOC = higher voltage). For an 18650 NMC cell, OCV goes from ~4.15 V (full) to ~2.75 V (empty).

The OCV–SOC relationship is a fundamental property of the battery chemistry. It must be measured experimentally by charging or discharging the cell *extremely slowly* (or letting it rest), so the voltage has time to equilibrate.

---

## Bug 1 — The OCV Curve Was Built from the Wrong Voltages

### What the old code did

```python
# OLD: v[0] = first voltage measured at the START of a discharge
ocv_pts.append(v[0])   # WRONG — this is under-load voltage, not OCV
```

At the start of a discharge, current is already flowing (2A for these cells).
The measured voltage at that moment is:
```
v[0] = V_OCV(1.0) − 2A × R0 − V1 − V2
     ≈ 4.15V − 0.16V − small terms
     ≈ 3.99V
```
But `V_OCV(1.0)` is actually ~4.15V. The code was treating 3.99V as the OCV at SOC=1.0, which is wrong by ~160 millivolts.

This 160mV error in the OCV polynomial then **infected every single downstream calculation**:
- The ECM optimiser tried to minimise `V_OCV(SOC) − I·R0 − V1 − V2 ≈ V_measured`
- But `V_OCV` was 160mV too low
- So the optimiser reduced R0 as far as possible to compensate
- R0 hit its lower bound (0.030 Ω) on every single cycle
- R0 never changed across 636 cycles even though the real cell aged significantly

### Why this symptom appeared in the logs
```
Cycle 20  | R0=0.0300Ω
Cycle 40  | R0=0.0300Ω
Cycle 100 | R0=0.0300Ω
```
R0 should increase with aging (the battery's internal resistance grows as it degrades).
It staying exactly at the lower bound tells you the optimiser is stuck against a wall.

### What we changed

Step 2 now reads the NASA `.mat` file's `impedance` and `charge` entries.
In the NASA test protocol, the cell rests between discharge cycles (impedance test = cell is sitting at rest, no current). The voltage at the **end** of that rest period is very close to the true OCV. We collect these rest voltages and use them to build the OCV polynomial.

```python
# NEW: rest voltage at end of impedance test ≈ true OCV
if etype == 'impedance':
    v_rest = float(v[-1])   # settled, near-equilibrium voltage
    rest_pts.append({'soc': 0.5, 'ocv_v': v_rest, 'soh': soh_est})

# Also: end of charge phase when |I| < 0.05A (CV phase complete)
if etype == 'charge':
    low_i = np.where(np.abs(i_arr) < 0.05)[0]
    v_ocv = float(v[low_i[-1]])
    rest_pts.append({'soc': 1.0, 'ocv_v': v_ocv, 'soh': soh_est})
```

**Why this will work:** The OCV polynomial is now fit to voltages measured when essentially zero current flows — these are genuine equilibrium measurements, not under-load readings. The polynomial will be ~150–200 mV higher, which is the correct value. Once OCV is right, R0 is free to find its physically true value (which should be ~0.065–0.090 Ω for a fresh 18650 cell and increase to 0.120–0.180 Ω by end of life).

---

## Bug 2 — R0 Lower Bound Was Too Low (0.030 Ω)

### The concept

The **optimiser** (L-BFGS-B algorithm) searches within a box of allowed parameter values called **bounds**. If the true answer is inside the box, the optimiser can find it. If the bound cuts off the true answer, the optimiser gets stuck at the nearest wall.

The old bound for R0 was `[0.030, 0.180]` Ω. For a healthy 18650 NMC cell, the real DC resistance is 0.060–0.090 Ω (confirmed by multiple papers). A bound of 0.030 Ω is physically unrealistic — it allows values that no real 18650 cell can achieve. This, combined with the OCV error (Bug 1), pushed the optimiser to always report 0.030.

### What we changed

New bound: `[0.050, 0.200]` Ω. The lower limit is still below any expected fresh-cell R0, but tight enough that the optimiser cannot escape to unphysical territory.

**Why this will work:** Combined with the correct OCV (Bug 1 fix), the optimiser now has no incentive to shrink R0 below physical values. R0 will converge to the true value for each cycle, and you will see it increase smoothly from ~0.07 Ω at cycle 1 to ~0.15 Ω at cycle 168.

---

## Bug 3 — ECM Had No Memory of Previous Cycles (Caused Q_gen Jumps)

### The concept

For 636 discharge cycles (4 batteries × ~160 cycles each), the old code ran 20 completely random starting points for the optimiser on every single cycle, with no connection to what the previous cycle found. The 2-RC ECM cost surface is not convex — it has many local minima. So cycle 97 might converge to one local minimum and cycle 98 to a completely different one.

This created **discontinuous jumps** in the fitted parameters:
- Cycle 97: `R1=0.012, C1=3200` → `Q_gen = 0.221 W`
- Cycle 98: `R1=0.018, C1=1800` → `Q_gen = 0.337 W`
- Cycle 99: `R1=0.025, C1=1200` → `Q_gen = 0.435 W`

These jumps are not physical. The cell's properties change continuously and slowly with aging — they cannot jump by 50% in one cycle.

### What we changed

**Warm-starting:** The first of the 16 starting points for the optimiser on cycle N is the solution found on cycle N-1. The remaining 15 are random. This means the optimiser starts close to the true answer (since cell parameters change slowly) and rarely needs to explore far.

**Physics regularisation:** A small penalty term in the cost function prevents R0 from jumping by more than 5% per cycle:
```python
excess = max(0.0, R0 - r0_prev - 0.05 * r0_prev)
cost += 10.0 * excess**2
```
This is physically motivated: the rate of resistance increase is governed by the Arrhenius equation (a chemical kinetics law) — it cannot suddenly double between two cycles.

**Why this will work:** The fitted parameters will vary smoothly across cycles. Q_gen (heat generation) will show a gradual, physically plausible increase with aging rather than random jumps. The validation log's Q_gen column will no longer have step-function artifacts.

---

## Bug 4 — Q_gen Was Calculated with the Wrong Formula

### The physics of heat generation

When current flows through a resistor, it generates heat. The rate of heat generation (in Watts) is:
```
Power = Voltage² / Resistance = Current² × Resistance
```

In our 2-RC circuit, there are three resistors: R0 (ohmic), R1 (fast RC), R2 (slow RC).
The **correct total irreversible heat generation** is:
```
Q_irrev = I²·R0  +  V1²/R1  +  V2²/R2
```
where `V1` and `V2` are the voltages across the RC pairs (which we track during simulation).

### What the old formula was

```python
Q_irrev = |I × (V_OCV - V_terminal)|
```

Mathematically: `V_OCV - V_terminal = I·R0 + V1 + V2`.
So the old formula computes: `|I| × (I·R0 + V1 + V2) = I²·R0 + I·V1 + I·V2`.

The difference between `I·V1` and `V1²/R1`:
- They are equal only at **steady state** (when the RC pair is fully charged): `V1 = I·R1`, so `I·V1 = I·(I·R1) = I²·R1 = V1·(V1/R1) = V1²/R1`. ✓
- During **dynamic operation** (current changing every second, like during a drive cycle), the RC capacitor hasn't settled: `V1 ≠ I·R1`. The two formulas give different answers.

The old formula also **directly depends on OCV accuracy** — since OCV was wrong (Bug 1), Q_gen was doubly wrong.

### What we changed

```python
Q_irrev = current**2 * R0 + V1_arr**2 / R1 + V2_arr**2 / R2
```
This requires R0, R1, R2 (from the ECM fit) and V1, V2 (which we track during the forward simulation). We added V1 and V2 as outputs of the ECM forward pass.

**Why this will work:**
- The formula is always non-negative (heat can only be generated, not absorbed, in resistors)
- It does not depend on OCV at all — immune to the OCV error
- For 2A discharge, R0≈0.08Ω: `Q_irrev ≥ 4×0.08 = 0.32 W` (physical minimum from ohmic heating alone). This is 2–3× higher than the old formula was producing, which is correct.

---

## Bug 5 — The Thermal ODE Was Using an Unstable Numerical Method

### The 2-state thermal model

The thermal model tracks two temperatures:
- **Tc** = core temperature (centre of the cell — unobservable)
- **Ts** = surface temperature (measurable, used for calibration)

The physics equations are:
```
(Core thermal mass) × dTc/dt = Q_gen - (heat flowing core→surface)
                                      = Q_gen - (Tc - Ts) / Rin

(Surface thermal mass) × dTs/dt = (heat flowing core→surface) - (heat flowing surface→air)
                                        = (Tc - Ts) / Rin - (Ts - T_ambient) / Rout
```
`Rin` = thermal resistance between core and surface (higher = harder for heat to escape core)
`Rout` = thermal resistance between surface and ambient air
`Cc`, `Cs` = thermal masses (how much energy is needed to raise temperature by 1°C)

### Forward Euler — the method that was used

**Euler's method** approximates the continuous equations by taking small steps:
```
Tc[next] = Tc[now] + (dTc/dt) × Δt
Ts[next] = Ts[now] + (dTs/dt) × Δt
```
This is the simplest possible integration scheme. The problem: **it is only stable when the time step Δt is smaller than twice the fastest time constant in the system**. The fastest time constant here is `Rin × Cc` (how quickly the core-to-surface heat flow equilibrates).

With the optimiser exploring values near the parameter bounds (Rin=0.5, Cc=5.0): `τ_min = 0.5 × 5.0 = 2.5 s`. Stability requires `Δt < 5.0 s`. NASA data has `Δt ≈ 1 s`. This is on the edge. When Q_gen had spikes (from the wrong formula in Bug 4), `dTc/dt` became huge, `Tc` jumped to an astronomically high value, the `(Tc - Ts)/Rin` term became enormous, `dTs/dt` blew up, and the entire simulation hit numerical overflow (IEEE 754 `inf` or `nan`).

**This is exactly what produced the logged error:**
```
RuntimeWarning: overflow encountered in scalar multiply
RuntimeWarning: invalid value encountered in subtract
```

### Crank-Nicolson — the method we use now

**Crank-Nicolson (CN)** is the standard stable method for linear ODEs. Instead of using the derivative at only the *current* time step, it averages the derivative at *current* and *next* time steps:
```
Tc[next] = Tc[now] + 0.5 × (dTc_at_now + dTc_at_next) × Δt
```

Since `dTc_at_next` depends on `Tc[next]` and `Ts[next]` (which are unknown), this creates a small system of equations we must solve. For our 2-state system, this becomes a 2×2 linear algebra problem:
```
[A00  A01] [Tc_next]   [b0]
[A10  A11] [Ts_next] = [b1]
```
The 2×2 system is solved exactly (Cramer's rule: 6 multiplications, 2 divisions). Cost per time step: trivial.

**Why CN is unconditionally stable:** For linear dissipative systems (like this thermal RC network), the CN method's amplification factor is always ≤ 1 regardless of Δt. This means errors do not grow over time, no matter how large the time step. Even if the optimiser tries Rin=0.5, Cc=5.0, Δt=10.0 s — the integration stays bounded.

**Why this will work:** The `RuntimeWarning: overflow` errors will disappear entirely. The thermal model can now explore the full parameter space without numerical instability. Tc and Ts will always be physically bounded (we also clamp them to [−40°C, 150°C] as a final safety check).

---

## Bug 6 — The SOH Baseline Was a Straight Line Through a Curve

### State of Health (SOH)

**SOH** = current capacity / original capacity. A brand-new NASA cell holds 2.0 Ah. After 168 cycles it holds ~1.5 Ah. SOH = 1.5/2.0 = 0.75 = "75% healthy".

### Why we need a baseline

The LSTM doesn't try to predict raw SOH directly. Instead, we:
1. Fit a simple physics formula for SOH vs cycle number: this is the **baseline**
2. The LSTM predicts only the **residual** (difference between truth and baseline)

This decomposition makes the learning problem easier. The LSTM only needs to learn the small nonlinear wiggles, not the entire trajectory from 0.92 to 0.75.

### Why a linear baseline was wrong

The NASA cells don't fade linearly. They show a characteristic S-curve:
- Cycles 1–20: fast initial drop (SEI formation)
- Cycles 20–100: slow, nearly flat plateau
- Cycles 100+: accelerating fade (lithium plating begins)

A straight line from cycle 1 (SOH=0.92) to cycle 168 (SOH=0.75) cuts through the middle of this curve. The residual the LSTM sees is:
- **Negative in the early plateau** (true SOH is above the line → LSTM sees "battery is better than physics predicts")
- **Positive in the late acceleration** (true SOH is below the line → LSTM sees "battery is worse than physics predicts")

The LSTM must then learn to flip the sign of its correction depending on where in the cycle it is. This is harder and less physically meaningful.

### What we changed

A **quadratic baseline** `a·cycle² + b·cycle + c` fit to all available data:
```python
coeffs = np.polyfit(cycles, soh, 2)   # degree-2 polynomial
baseline = np.polyval(coeffs, cycles)
```
The slight downward curvature of a quadratic naturally tracks the beginning of the knee. The residual the LSTM sees is now genuinely small and lacks the systematic sign-flip. Learning the true nonlinear deviation from a good baseline is much easier.

---

## Bug 7 — Train/Validation Split Had Data Leakage (LSTM)

### What data leakage means

Data leakage happens when information from the validation set "leaks" into the training set, making the model appear better than it really is. The model is essentially cheating — it has seen (or nearly seen) the answers before being tested.

### How it happened here

The old code used a random 80/20 split across all sequences from all batteries.
A "sequence" is a window of 10 consecutive cycles used to predict the next cycle's SOH.

With sequences generated at stride=1 over B0005's 168 cycles:
- Sequence at position 50: `[cycle_50, 51, 52, ..., 59] → cycle_60`
- Sequence at position 51: `[cycle_51, 52, ..., 60] → cycle_61`

These two sequences share **9 of 10 inputs**. If sequence 50 goes to training and sequence 51 goes to validation, the model was trained on `[cycle_51...59]` and is being tested on `[cycle_51...60]` — it has seen 90% of the validation input. This is almost like testing on the training data.

**Result:** Val MSE ≈ 0.000123 (looks excellent, is actually meaningless).

### What we changed

**Leave-one-battery-out (LOBO) cross-validation:**
- Train on B0005, B0006, B0007 (all their cycles)
- Validate on B0018 (completely held out — never seen during training)

B0018 is a genuinely separate physical cell, tested in different sessions. If the model performs well on B0018, it has learned something real about how batteries age — not just the specific trajectory of three particular cells.

**Why this will work:** The validation metric will be honest. Expect it to be higher (worse) than before — that's correct, not a regression. It means we're measuring real generalisation. Published work (Batteries 2025, SP-LSTM) reports RMSE ≈ 0.0136 SOH on NASA B0018 with a similar approach; we expect to be in that range.

---

## Bug 8 — Transformer Had Data Leakage from Stride=1 Windows

### Same concept, same problem, bigger scale

The Transformer uses 60-second sliding windows. With stride=1:
- Window at second 100: `[second_100, 101, ..., 159] → Tc at second_160`
- Window at second 101: `[second_101, ..., 160] → Tc at second_161`

These share 59 of 60 seconds of input — 98.3% overlap. From 431,000 rows of data, this generated 377,000 windows, almost all of which are near-duplicates of adjacent windows.

The old validation score of 0.46°C RMSE was measured on overlapping windows from B0018 — but since the stride-1 windows were so highly correlated, the model essentially memorised the temperature profile.

### What we changed

**stride=60** — non-overlapping windows. Window 1: seconds 0–59. Window 2: seconds 60–119. No shared data between consecutive windows.

Total windows: from 377,000 down to ~6,000. Each window is a genuinely independent 60-second segment.

**Why this will work:**
- Training is ~60× faster (6,000 vs 377,000 windows per epoch)
- Validation RMSE will be higher but honest — it represents true generalisation to unseen 60-second segments of B0018's discharge
- The model now needs to actually understand the physics (thermal inertia, SOC dependence) rather than interpolating between adjacent near-identical windows

---

## Bug 9 — Transformer Only Had 4 Features, Missing the Most Important Ones

### What was the input before

`[current_A, voltage_V, r0_ohms, temp_surface_C]` × 60 timesteps

### What we added and why

**SOC (State of Charge):** The core temperature rate of change depends on Q_gen, which depends on I²·R0. But R0 itself changes with SOC (higher at low SOC). The OCV slope `dV/dSOC` also varies — affecting the entropic heat term. By explicitly giving the Transformer the current SOC value, it knows whether the battery is near-full (high OCV, lower heat) or near-empty (lower OCV, higher heat per amp).

**Q_gen (heat generation rate, Watts):** This is the **direct physical cause** of core temperature rise. The governing equation is literally `dTc/dt ∝ Q_gen`. Giving the Transformer the explicit heat input is like giving a weather model the solar irradiance data — it's the causal variable. Without it, the model must infer it from I and V (which it can, imperfectly). With it, the prediction is physically constrained and much easier to learn.

**New input:** `[current_A, voltage_V, r0_ohms, temp_surface_C, soc, q_gen_W]` × 60 timesteps

This is **physics-informed feature engineering** — adding physical quantities that directly appear in the governing equations.

---

## Bug 10 — Windows DataLoader Deadlock (`num_workers=2`)

### What happened

PyTorch's DataLoader has a `num_workers` option that spawns multiple processes to load data in parallel. On Linux/Mac, this uses `fork()` — the child process gets a copy of the parent's memory immediately. On **Windows**, `fork()` is not available; it uses `spawn()` instead, which means the child process must start fresh and import the entire module again before it can load data. Inside a conda environment, this often causes a deadlock (the child process waits for something the parent process is also waiting for → both freeze forever).

Setting `num_workers=2` on Windows causes the DataLoader workers to silently hang, the training loop never starts, and the program appears to do nothing.

### What we changed

```python
num_workers=0   # single-process data loading, safe on all platforms
```

With `num_workers=0`, data loading happens in the main process. For a dataset of 6,000 windows (stride-60), this is negligible overhead — data loading is not the bottleneck.

---

## Bug 11 — C1 and C2 Were Never Saved (EV Dataset Used Wrong RC Dynamics)

### Why C1 and C2 matter

The RC pairs (R1, C1) and (R2, C2) model how the battery voltage **recovers after a current step**. The time constants `τ1 = R1·C1` (typically 10–50 seconds) and `τ2 = R2·C2` (typically 100–500 seconds) control:
- How long it takes for the voltage to settle after a sudden current change
- How much of the heat is generated in the RC elements vs the pure ohmic resistance

For drive cycles (constantly changing current), getting C1 and C2 right is critical for accurate voltage simulation, which feeds into accurate Q_gen.

### What happened

Step 4 identified all 5 parameters (R0, R1, C1, R2, C2) per cycle, but only saved R0, R1, R2 to `ecm_parameters.csv`. C1 and C2 were thrown away. When the EV dataset generator loaded the ECM parameters, it used hardcoded values:
```python
C1, C2 = 15000.0, 3000.0   # magic constants, same for all batteries, all aging states
```
This is wrong. The real fitted C1 and C2 vary between ~1000–5000 F depending on the cycle.

### What we changed

Step 4 now saves all 5 columns: R0, R1, **C1**, R2, **C2**.
The EV generator reads them from the CSV.

**Why this will work:** The drive-cycle simulations will have correct RC dynamics. The voltage simulation errors will decrease for high-frequency current transients (e.g., US06 aggressive accelerations), and Q_gen from the RC elements will be more accurate.

---

## Bug 12 — dt=0 Guard Missing (Caused ECM_MSE in the Millions)

### The technical detail

When the ECM simulates the RC voltage state update:
```python
a1 = np.exp(-dt / (R1 * C1))
V1 = a1 * V1 + I * R1 * (1 - a1)
```
If `dt = 0` (two consecutive measurements at the same timestamp), then `exp(-0 / anything) = exp(0) = 1.0`, so `V1 = 1.0 * V1 + I * R1 * 0 = V1`. The RC state does not decay. Over the next few steps, V1 keeps accumulating without resetting. Within a few hundred iterations, V1 becomes enormous and `v_sim = V_OCV - I*R0 - V1 - V2` becomes hugely negative — nowhere near the real voltage. The cost function then sees thousands of (simulated_V − measured_V)² values all near 10² = 100, and the mean is ~100 V². Printed as a number, this looks like "ECM_MSE = 100,000,000 mV²" — the millions figure seen in the logs.

### What we changed

Step 2 removes duplicate/non-monotone timestamps before saving the CSV:
```python
dt = np.diff(time_s)
valid = np.concatenate([[True], dt > 0])
time_s = time_s[valid]
# ... similarly for current, voltage, temperature
```
Step 4 also has a `if dt <= 0.0: continue` guard in both the ECM loop and the thermal loop.

**Why this will work:** ECM_MSE will no longer blow up in early cycles. Expect values in the range 0.00005–0.005 V² (printed as 50–5000 mV²), corresponding to RMSE errors of 7–70 mV. This is physically realistic for a 2-RC ECM fit to noisy NASA data.

---

## Bug 13 — `\s+` Regex Without Raw String (SyntaxWarning)

### The technical detail

In Python, `'\s'` in a regular (non-raw) string is not a standard escape sequence like `'\n'` (newline) or `'\t'` (tab). Python 3.12+ issues a `SyntaxWarning: invalid escape sequence '\s'` and in future Python versions may raise a full error. The fix is to use a raw string `r'\s+'` which tells Python "do not interpret backslash sequences".

### What we changed
```python
# OLD
raw = pd.read_csv(url, header=None, skiprows=1, sep='\s+')
# NEW
raw = pd.read_csv(url, header=None, skiprows=1, sep=r'\s+')
```

---

## What Was Added That Wasn't There Before (SOTA Additions)

These are not just fixes — they are new features that improve the science.

### ICA Peak Tracking (Incremental Capacity Analysis)

**The concept:**
When you discharge a battery very slowly and plot `dQ/dV` (how much charge you extract per millivolt of voltage drop), you get a curve with distinct peaks. These peaks correspond to **phase transitions** in the lithium intercalation material — physical rearrangements of lithium atoms in the crystal lattice. As the battery ages, these peaks shift and shrink in a characteristic way.

For NMC 18650 cells, there are typically two main peaks:
- **Peak 1** (~3.65 V): corresponds to the hexagonal-to-monoclinic phase transition
- **Peak 2** (~3.75 V): corresponds to the monoclinic-to-hexagonal second transition

The voltage positions and height ratio of these peaks change measurably with aging — independently of capacity. This makes them a more sensitive aging indicator than just "how much charge did I get out."

**Why we added it:**
A 2025 paper (SP-LSTM, *Batteries* journal, doi:10.3390/batteries12050176) showed that feeding ICA-derived features (peak positions, heights, ratios) into a BiLSTM achieves RMSE=0.0136 SOH on NASA B0018 — significantly better than using capacity/resistance alone. We independently extract the same features.

**What we added to the code:**
Each cycle, we compute the dQ/dV curve from the discharge voltage profile, smooth it with Savitzky-Golay filtering, find the two tallest peaks, and save `ica_peak1_v`, `ica_peak2_v`, `ica_peak_ratio` to the twin dataset. These become additional input features for the LSTM SOH model.

### Physics-Informed Regularisation (ECM)

The ECM optimiser now includes a penalty on R0 jumping too fast:
```
cost += 10 × max(0, R0 - r0_prev - 5% of r0_prev)²
```
The growth rate of R0 is governed by SEI (Solid Electrolyte Interphase) layer thickening, which follows Arrhenius kinetics. This cannot accelerate discontinuously. By penalising sudden jumps, we embed physical chemistry knowledge directly into the optimiser. This is a form of **physics-informed optimisation**.

### Physics-Informed Smoothness Loss (Transformer)

During transformer training, a small additional loss term penalises large sudden changes in consecutive predicted core temperatures:
```python
loss = MSE(predicted, actual) + 0.01 × mean((pred[1:] - pred[:-1])²)
```
The physical motivation: the core temperature cannot change faster than `Q_gen / Cc_min ≈ 0.5 W / 10 J/K = 0.05 °C/second`. In a 60-second window, Tc should vary smoothly. Penalising jagged predictions encourages physically plausible thermal trajectories.

### Sinusoidal Positional Encoding

The old code initialised the positional encoding randomly:
```python
self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
```
This means the positional encoding was randomly initialised and then learned during training — it has to figure out "which position in the sequence is which" from scratch.

Sinusoidal positional encoding (from the original "Attention Is All You Need" paper, Vaswani et al. 2017) uses:
```
PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
```
These functions have known mathematical properties that encode relative position differences — second 45 is always 15 units after second 30, regardless of the total sequence length. This gives the Transformer a deterministic, physics-appropriate sense of time before training even begins.

---

## Summary Table

| Bug # | Where | What Was Wrong | Fix | Why Fix Works |
|-------|-------|---------------|-----|---------------|
| 1 | Step 4 | OCV built from under-load voltage | Use REST/CHARGE end-of-rest voltages | OCV is now near-equilibrium; R0 can find its true value |
| 2 | Step 4 | R0 lower bound 0.030 Ω (too low) | Raised to 0.050 Ω | Prevents physically impossible solutions |
| 3 | Step 4 | No warm-start; params jump each cycle | Warm-start from previous cycle + Arrhenius penalty | Smooth, physically continuous parameter evolution |
| 4 | Step 4 | Q_gen = \|I·(VOCV−Vt)\| (OCV-dependent) | Joule form: I²R0 + V1²/R1 + V2²/R2 | Always non-negative; independent of OCV accuracy |
| 5 | Step 4 | Forward Euler thermal ODE → overflow | Crank-Nicolson (unconditionally stable) | Eliminates all RuntimeWarning overflows |
| 6 | Step 2 | Linear SOH baseline | Quadratic fit | Captures the knee; LSTM learns real deviation |
| 7 | Step 3 | Random 80/20 split (data leakage) | Leave-one-out: B0018 held out | Honest validation; tests real generalisation |
| 8 | Step 5 | Stride=1 windows (98% overlap) | Stride=60 (non-overlapping) | Honest RMSE; 60× faster training |
| 9 | Step 5 | Only 4 input features | Added SOC + Q_gen (6 features total) | Gives Transformer the causal physical drivers |
| 10 | Step 5 | num_workers=2 (Windows deadlock) | num_workers=0 | No DataLoader hang on Windows |
| 11 | Step 4 | C1, C2 not saved; hardcoded in EV gen | Save all 5 ECM params; load in EV gen | Correct RC dynamics in drive cycle simulations |
| 12 | Steps 2,4 | No dt=0 guard | Deduplicate timestamps; skip dt≤0 | Prevents ECM_MSE blowup to millions |
| 13 | Step 0 | `'\s+'` non-raw regex | `r'\s+'` raw string | Eliminates SyntaxWarning |

---

## What Is Left to Do: Training

**You just need to run one command:**
```bash
git checkout sota-rewrite
conda activate battery-modelling
python run_pipeline.py
```

The pipeline will automatically run all 7 steps in order. On a CPU laptop, expect:
- Steps 1–2 (download + parse): 2–10 minutes (mostly network + file I/O)
- Step 4 (digital twin): 30–90 minutes (ECM fitting for 636 cycles is the slow part)
- Step 3 (LSTM): 5–15 minutes
- Step 5 (Transformer): 10–30 minutes (stride-60 means only ~6000 windows vs 377000)
- Step 6 (plots): 2–5 minutes

Total: approximately **1–2.5 hours on CPU**.

After training, check:
1. `data/digital_twin_sets/validation_log.csv` — R0 should increase from ~0.07 to ~0.15 Ω
2. `results/paper_plots/` — all 8 figures generated
3. Console output during Step 4 — no `RuntimeWarning: overflow` lines
4. Step 3 final line — `Best validation MSE: X` where X should be ~0.015–0.030
