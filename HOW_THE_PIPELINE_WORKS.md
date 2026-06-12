# How the Pipeline Works — A Complete Beginner's Guide

> Imagine you are explaining this to a smart friend who programs but has never studied batteries
> or machine learning. Start from zero, explain every step.

---

## The 30-Second Summary

We have real data from NASA battery lab experiments. We want to train an AI to predict the
*internal temperature* of a battery while it's discharging. We can't measure internal temperature
directly, so we:

1. **Download** the NASA data and real-world driving profiles
2. **Parse** the raw data into usable tables
3. **Simulate** what must have been happening inside the battery (physics)
4. **Train** two AI models: one that predicts how fast the battery is aging, and one that predicts internal temperature
5. **Plot** everything to check the results look right

Below is each step explained in full detail.

---

## Step 0: Download EPA Drive Cycles

**File:** [data/step0_download_epa_drive_cycles.py](data/step0_download_epa_drive_cycles.py)

**What it does:**
Downloads three standardised driving speed profiles from the US Environmental Protection Agency:
- **UDDS** (Urban Dynamometer Driving Schedule) — city driving. Lots of stop-and-go, low average speed (~32 km/h). The kind of driving where you brake at traffic lights every 30 seconds.
- **HWFET** (Highway Fuel Economy Test) — highway driving. Steady moderate speed (~78 km/h), no stops.
- **US06** (Supplemental Federal Test Procedure) — aggressive driving. High speeds, hard accelerations. The worst case for battery stress.

These profiles specify **vehicle speed in km/h vs time**. They don't tell us directly how much current the battery needs.

**Conversion to current:**
We use basic vehicle dynamics:
```
Force = mass × acceleration + aerodynamic drag + rolling resistance + hill climbing
Power = Force × speed
Current = Power / (battery_voltage × efficiency)
```
With regenerative braking modelled as a negative current (the motor acts as a generator,
putting energy back into the battery).

**Why we need this:**
The NASA tests use a simple constant 2A discharge — the same current all the time.
Real driving is not constant. We use the drive cycles to simulate the battery under realistic
variable-current conditions, which tests whether the digital twin generalises beyond the
constant-current training regime.

**Output files:**
```
data/drive_cycles/UDDS.csv       ← speed [km/h] vs time [s]
data/drive_cycles/HWFET.csv
data/drive_cycles/US06.csv
```

---

## Step 1: Download NASA Battery Data

**File:** [data/step1_download_nasa_data.py](data/step1_download_nasa_data.py)

**What it downloads:**
NASA `.mat` files (MATLAB format) from the NASA Data Repository for batteries B0005, B0006, B0007, B0018.

Each `.mat` file contains a MATLAB struct with entries for each test cycle:
- Every **discharge** cycle: time, current, voltage, temperature, capacity
- Every **charge** cycle: same
- Every **impedance** test: same (but with AC impedance measurement data)

**File sizes:** Each `.mat` file is ~50–150 MB.

**Output:**
```
data/raw/B0005.mat
data/raw/B0006.mat
data/raw/B0007.mat
data/raw/B0018.mat
```

---

## Step 2: Parse, Extract, and Engineer Features

**File:** [data/step2_parse_and_extract_hic.py](data/step2_parse_and_extract_hic.py)

This is the most complex pre-processing step. It does four things.

### 2a. Parse .mat → discharge CSV

For each battery, it reads every discharge cycle and creates a CSV with columns:
```
time_s | current_A | voltage_V | temperature_C | soc | capacity_ah
```

Each row is one measurement (~1 second apart). Total rows per battery: ~431,000.

SOC is computed by **Coulomb counting** — integrating the current over time:
```
SOC[t] = SOC[0] - integral(current × dt) / rated_capacity
```
This gives a SOC that counts down from 1.0 (full) to 0.0 (empty) within each cycle.

### 2b. Extract OCV rest points

Between discharge cycles, the NASA protocol includes a rest period (impedance test).
During rest, current is essentially zero and the cell voltage settles toward equilibrium.
We read the **final voltage** of each impedance test as an OCV measurement:
```python
v_rest = float(v[-1])   # last voltage in the impedance test sequence
soc_est = ?             # can be estimated from remaining capacity
```

We also read the end-of-charge voltage (CV phase complete, current ≈ 0 mA) as OCV at SOC=1.0.

These rest voltages are saved as `{battery}_ocv_rest_points.csv` and used in Step 4 to build
the OCV-SOC polynomial.

### 2c. Compute SOH per cycle

For each cycle, the discharge capacity (measured as the area under the current-time curve,
i.e., how many Amp-hours came out) is recorded.
```
SOH[cycle] = discharge_capacity[cycle] / rated_capacity
```
For B0005 at cycle 1: SOH ≈ 0.920. At cycle 168: SOH ≈ 0.750.

Then we fit a **quadratic baseline** to the SOH vs cycle number data:
```
SOH_baseline[cycle] = a × cycle² + b × cycle + c
```
The quadratic captures the gentle S-curve of capacity fade better than a straight line.

### 2d. Remove duplicate timestamps

If any two consecutive rows in a discharge CSV have the same timestamp (can happen from
integer-second rounding in NASA's logging software), the second row is removed.
This prevents division by zero in the thermal ODE integration and the ECM RC update formula.

**Output files:**
```
data/processed/{battery}_discharge.csv          ← full discharge traces
data/processed/{battery}_ocv_rest_points.csv    ← OCV-SOC calibration data
data/processed/{battery}_soh_per_cycle.csv      ← SOH per cycle + quadratic baseline
```

---

## Step 4: Generate the Digital Twin (The Physics Engine)

**File:** [generation/step4_generate_aging_digital_twin.py](generation/step4_generate_aging_digital_twin.py)

This is the heart of the project. It generates the **core temperature labels** that the
Transformer will be trained to predict. It does this through three nested processes.

### 4a. Build the OCV-SOC polynomial

Reads `{battery}_ocv_rest_points.csv` from Step 2. These points tell us:
"when the battery is at SOC=X (fresh cell), the open-circuit voltage is Y volts."

We fit a 6th-degree polynomial `V_OCV(SOC) = a0 + a1·SOC + a2·SOC² + ... + a6·SOC⁶` to
these points (separately for fresh cells SOH≥0.90 and aged cells SOH≤0.78).

If not enough rest points exist (can happen if the `.mat` file has incomplete impedance data),
we fall back to an empirical NMC polynomial:
```
V_OCV(SOC) = 3.481 + 0.718·SOC − 0.405·SOC² + ...  (6th order, literature values)
```

### 4b. ECM Parameter Identification (per cycle)

For each of the ~168 discharge cycles, we find the 5 parameters (R0, R1, C1, R2, C2) that
make the ECM's predicted voltage match the measured voltage as closely as possible.

**How the ECM simulation works (forward pass):**
```
for each time step:
    V1[next] = exp(-dt / (R1·C1)) × V1 + I × R1 × (1 - exp(-dt / (R1·C1)))
    V2[next] = exp(-dt / (R2·C2)) × V2 + I × R2 × (1 - exp(-dt / (R2·C2)))
    V_sim = V_OCV(SOC) − I·R0 − V1 − V2
```
Where V1 and V2 are the voltages across each RC pair (start at 0 at beginning of discharge).

**How the optimiser works:**
The L-BFGS-B algorithm starts with a guess for (R0, R1, C1, R2, C2), runs the forward pass to
get V_sim, computes `Cost = sum((V_sim - V_meas)²)`, then uses the gradient of the cost
w.r.t. the parameters to adjust them in the direction that reduces the cost. Repeats until
the cost no longer improves (convergence) or a maximum iteration limit is hit.

We run 17 starting points per cycle (1 warm-start + 16 random), pick the solution with lowest cost.

**Bounds:**
```
R0: [0.050, 0.200] Ω
R1: [0.001, 0.100] Ω
C1: [100, 10000] F
R2: [0.001, 0.100] Ω
C2: [100, 10000] F
```

**Warm-start:**
On cycle N, the first starting point is the fitted parameters from cycle N-1. Since parameters
evolve slowly, this almost always converges to the best solution quickly.

### 4c. Thermal Parameter Identification (per battery)

Using the ECM parameters from 4b, we compute Q_gen for every time step of every cycle.
Then we find the 4 thermal parameters (Rin, Rout, Cc, Cs) that make the simulated surface
temperature Ts match the measured surface temperature from the NASA data.

We use a single set of thermal parameters per battery (not per cycle) because the physical
cell construction doesn't change with aging.

**The Crank-Nicolson integrator:**
At each time step, instead of directly computing `Tc[next] = Tc + dt × dTc/dt`, we solve:
```
[A] × [Tc[next], Ts[next]] = [b]
```
where A is a 2×2 matrix built from Rin, Rout, Cc, Cs, dt. This 2×2 system is solved
exactly with Cramer's rule. The solution is numerically stable for any dt and any parameter values.

### 4d. Full Simulation Run

With both ECM and thermal parameters fitted, we run the complete simulation for every cycle
of every battery:
- ECM forward pass → V1, V2, Q_gen at every time step
- Thermal integration → Tc, Ts at every time step

The output is the augmented dataset: every row contains the original observables
(current, voltage, surface temp) **plus** the simulated quantities (Q_gen, Tc, SOC, V1, V2, R0, R1, C1, R2, C2).

### 4e. ICA Feature Computation

After computing the Q-V profile for each cycle, we compute the incremental capacity `dQ/dV`,
smooth it, find the two largest peaks, and record their voltage positions and height ratio.

**Output files:**
```
data/digital_twin_sets/B0005_twin.csv    ← ~80,000 rows per battery
data/digital_twin_sets/B0006_twin.csv
data/digital_twin_sets/B0007_twin.csv
data/digital_twin_sets/B0018_twin.csv
data/digital_twin_sets/ecm_parameters.csv    ← 5 ECM params per (battery, cycle)
data/digital_twin_sets/thermal_params.csv    ← 4 thermal params per battery
data/digital_twin_sets/validation_log.csv    ← MSE per cycle for quality check
```

---

## Step 3: Train the SOH (State of Health) Predictor

**File:** [soh/step3_train_residual_lstm.py](soh/step3_train_residual_lstm.py)

**Goal:** Given the last N cycles of a battery's ECM and ICA features, predict how healthy
the battery is on the next cycle.

### Data preparation

We take the ECM parameters + ICA features from Step 4, joined with the SOH from Step 2.
For each battery and each cycle, we have a row:
```
battery | cycle | soh_physics_baseline | r_internal_ohms | cycle_norm | ica_peak1_v | ica_peak2_v | ica_peak_ratio
```

`r_internal_ohms` = R0 + R1 (total DC internal resistance, a commonly used aging indicator).
`cycle_norm` = cycle / 200 (normalised time in cycle life).

### Sequence creation

For each battery, we create overlapping windows:
- Input: features at cycles [n, n+1, ..., n+9] (window size = 10)
- Label: SOH at cycle n+10

### Leave-one-out split

- **Training set:** All windows from B0005, B0006, B0007
- **Validation set:** All windows from B0018

B0018 never appears in training — it is a completely held-out test case.

### BiLSTM architecture

```
Input: [batch, 10, 6] — 10 cycles, 6 features each
  ↓
BiLSTM(hidden=128, num_layers=2, dropout=0.3)
  → reads sequence forward AND backward
  → output shape: [batch, 10, 256] (128 from each direction)
  ↓
Take last timestep output: [batch, 256]
  ↓
Dropout(0.2) → Linear(256→64) → GELU activation → Linear(64→1)
  ↓
Output: [batch, 1] — predicted SOH residual
```

The model predicts the **residual** (SOH_actual − SOH_baseline), not raw SOH.
Final SOH = SOH_baseline + residual.

### Training

- Loss: Mean Squared Error (MSE)
- Optimiser: AdamW (Adam with L2 weight decay = 0.01)
- Learning rate schedule: Cosine annealing from 5×10⁻⁴ to 5×10⁻⁷ over 150 epochs
- Early stopping: stop training if validation MSE doesn't improve for 20 consecutive epochs
- Gradient clipping: max norm = 1.0 (prevents exploding gradients)

**Expected results:**
- Validation MSE ~ 0.0002–0.0005 (corresponding to RMSE ~ 0.015–0.022 SOH)
- Best model checkpoint saved automatically

**Output files:**
```
soh/models/residual_lstm_best.pth       ← trained weights
soh/models/lstm_feature_cols.csv        ← list of feature column names (for reproducibility)
soh/results/soh_predictions.csv        ← predicted vs actual SOH for all cycles
```

---

## Step 5: Train the Core Temperature Transformer

**File:** [transformer/step5_train_transformer.py](transformer/step5_train_transformer.py)

**Goal:** Given a 60-second sequence of (current, voltage, R0, surface_temp, SOC, Q_gen),
predict the core temperature at the end of that sequence.

### Data preparation

We load all `*_twin.csv` files from Step 4. The 6 feature columns are:
```
current_A | voltage_V | r0_ohms | temp_surface_C | soc | q_gen_W
```
Label: `temp_core_C` (simulated by the thermal model in Step 4)

### Window creation — stride=60

We take **non-overlapping** 60-second windows:
- Window 1: rows 0–59 → label is `temp_core_C` at row 59
- Window 2: rows 60–119 → label is `temp_core_C` at row 119
- Window 3: rows 120–179 → label at row 179
- (no overlap between windows)

This is important: if windows overlapped by 59 rows, adjacent windows would be 98% identical
and the model would overfit massively.

### Transformer architecture

```
Input: [batch, 60, 6] — 60 seconds, 6 features

  ↓ Linear projection (6 → 64 dimensions)

  ↓ Sinusoidal Positional Encoding [1, 60, 64]
    (adds time-position information mathematically — second 45 is always 15 after second 30)

  ↓ 4× TransformerEncoderLayer(d_model=64, nhead=4, norm_first=True)
    Each layer:
      - Layer norm (pre-LN) → Self-attention → add residual
      - Layer norm (pre-LN) → Feed-forward MLP → add residual

  ↓ Take mean across time dimension (average all 60 positions)

  ↓ Linear(64→32) → ReLU → Linear(32→1)

Output: [batch, 1] — predicted core temperature in °C
```

### Training

- Loss: MSE(predicted_Tc, simulated_Tc) + 0.01 × smoothness_penalty
  - Smoothness penalty: `mean((pred[1:] - pred[:-1])²)` across a batch
  - Encourages physically plausible smooth temperature predictions
- Optimiser: AdamW, learning rate = 3×10⁻⁴
- Epochs: up to 100, with early stopping (patience=15 epochs)
- `num_workers=0` for Windows compatibility

**Train/val split:**
B0005, B0006, B0007 → training
B0018 → validation

**Expected results:**
- Val RMSE ~ 0.5–1.2 °C
- (Published SOTA on similar datasets: ~0.3–0.8 °C with more complex architectures)

**Output files:**
```
transformer/models/battery_thermal_best.pth     ← trained weights
transformer/results/transformer_predictions.csv
```

---

## Step 6: Generate Paper Plots

**File:** [reports/generate_paper_plots.py](reports/generate_paper_plots.py)

Generates 8 publication-quality figures saved to `results/paper_plots/`:

| Figure | What It Shows |
|--------|---------------|
| `fig1_dataset_overview.png` | SOH curves for all 4 batteries — shows the aging trajectory and why B0018 is a good test case |
| `fig2_ecm_fit.png` | Measured vs simulated voltage for one cycle — validates ECM fit quality |
| `fig3_thermal_fit.png` | Measured surface temp vs simulated Tc/Ts — validates thermal model and shows the Tc > Ts gap |
| `fig4_ecm_evolution.png` | R0, R1, C1, R2, C2 vs cycle number — shows aging causes monotonic R0 increase |
| `fig5_ica_peaks.png` | ICA peak positions and ratios vs cycle — shows the electrochemical aging signature |
| `fig6_qgen_evolution.png` | Q_gen vs time for early vs late cycles — shows more heat in aged cells |
| `fig7_soh_prediction.png` | LSTM SOH predictions vs truth on B0018 — key validation figure |
| `fig8_temperature_prediction.png` | Transformer Tc predictions vs truth on B0018 — key validation figure |

---

## Complete Data Flow Diagram

```
NASA B0005-B0018 .mat files         EPA Drive Cycles (UDDS/HWFET/US06)
         |                                          |
         ↓ step1                                    ↓ step0
   data/raw/*.mat                         data/drive_cycles/*.csv
         |
         ↓ step2
   ┌─────────────────────────────────────────────┐
   │ data/processed/                             │
   │   B000X_discharge.csv (current,V,T,SOC)    │
   │   B000X_ocv_rest_points.csv                │
   │   B000X_soh_per_cycle.csv                  │
   └─────────────────────────────────────────────┘
         |
         ↓ step4 (PHYSICS ENGINE — the slow step)
   ┌─────────────────────────────────────────────────────┐
   │ data/digital_twin_sets/                             │
   │   B000X_twin.csv (+ Q_gen, Tc, Ts, V1, V2, ICA)   │
   │   ecm_parameters.csv (R0,R1,C1,R2,C2 per cycle)   │
   │   thermal_params.csv (Rin,Rout,Cc,Cs per battery)  │
   └─────────────────────────────────────────────────────┘
              |                    |
              ↓ step3              ↓ step5
    ┌──────────────────┐   ┌──────────────────────────┐
    │ BiLSTM SOH Model │   │ Transformer Tc Model     │
    │ soh/models/      │   │ transformer/models/      │
    │ Val RMSE~0.018   │   │ Val RMSE~0.8°C           │
    └──────────────────┘   └──────────────────────────┘
              |                    |
              └────────┬───────────┘
                       ↓ step6
              results/paper_plots/
                (8 publication figures)
```

---

## Running the Full Pipeline

### One-shot (everything, in order):
```bash
conda activate battery-modelling
python run_pipeline.py
```

### Individual steps (if you want to run/re-run a specific stage):
```bash
python data/step0_download_epa_drive_cycles.py
python data/step1_download_nasa_data.py
python data/step2_parse_and_extract_hic.py
python generation/step4_generate_aging_digital_twin.py
python soh/step3_train_residual_lstm.py
python transformer/step5_train_transformer.py
python reports/generate_paper_plots.py
```

### Expected runtime on a CPU laptop (no GPU needed):
| Step | Time |
|------|------|
| step0 (drive cycles) | 1–3 min |
| step1 (download NASA) | 5–15 min (network speed) |
| step2 (parse) | 2–5 min |
| step4 (digital twin) | **30–90 min** (ECM fitting is slow) |
| step3 (LSTM) | 5–15 min |
| step5 (Transformer) | 10–30 min |
| step6 (plots) | 2–5 min |
| **Total** | **~55–165 min** |

### Signs of a successful run:
- No `RuntimeWarning: overflow` in step4's output
- `ecm_parameters.csv` has R0 values between 0.06 and 0.18 Ω
- R0 generally increases as cycle number increases (aging trend)
- Step3 prints something like `Best val MSE: 0.000312 (RMSE: 0.0177 SOH)`
- Step5 prints something like `Best val RMSE: 0.84 °C`
- All 8 figures are generated in `results/paper_plots/`

---

## Frequently Asked Questions

**Q: Why do we need both ECM and the thermal model? Can't we just use one?**
A: The ECM tells us how much electrical power is being dissipated (Q_gen). The thermal model
tells us where that heat goes (core vs surface). Without Q_gen, the thermal model has no
input. Without the thermal model, we cannot compute Tc from Q_gen. They are coupled.

**Q: Why does Tc matter if we can already measure Ts?**
A: Surface temperature lags behind core temperature. During a hard acceleration, the core
can spike to 60°C while the surface is still at 40°C. A BMS using only surface temperature
would miss this spike and might allow the battery to stay in a dangerous state for several
seconds. The digital twin provides the core temperature estimate in real-time.

**Q: Can this run in a real car?**
A: The Transformer model (once trained) runs in a few milliseconds per inference — fast enough
for real-time use. The ECM and thermal fitting (Step 4) is too slow for real-time, but it
runs offline during the design phase. In a production BMS, you would pre-fit the thermal
parameters during manufacturing/calibration, and run only the Transformer in real-time.

**Q: Why 18650 cells? Aren't EV batteries different?**
A: Most EV battery packs are built from 18650 cells (Tesla Model 3 uses ~4,416 of them).
The physics of heat generation and thermal diffusion are the same regardless of pack size.
A digital twin calibrated on a single cell can be extended to the pack level with a higher-order
thermal model. The 18650 NASA dataset is used because it is the most widely cited public dataset.

**Q: What happens in Step 4 if the ECM fit is bad for one cycle?**
A: That cycle's Q_gen will be inaccurate, and the thermal simulation for that cycle will have
slightly wrong Tc labels. The Transformer training will see a noisy label for that window.
In practice, with 168 cycles × 4 batteries, occasional bad fits are averaged out during training.
The physics regularisation (penalty on R0 jumps) prevents cascading errors across cycles.
