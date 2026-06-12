# SOTA Rewrite Report — EV Battery Digital Twin
**Branch:** `sota-rewrite`  **Commit:** `eb3afef`  **Date:** 2026-06-12

---

## 1. What This Project Does (and Why)

The core problem: inside a real EV battery, **core temperature cannot be measured** — only surface temperature is accessible. If the core overheats by even 5–10°C beyond the surface, it triggers accelerated degradation or thermal runaway. So we need a model that predicts core temperature from observable signals only.

The approach (Samanta-Surya-Williamson IEEE TTE 2022):
1. Use real NASA discharge data (B0005/B0006/B0007/B0018, 18650 Li-ion cells)
2. Fit a physics model (2-RC ECM + 2-state thermal RC) to each discharge cycle
3. The thermal model — once calibrated against measured surface temperature — gives physically trustworthy **core temperature labels** (which NASA never measured)
4. Train a Transformer on these labels: input = [I, V, R0, Ts] → output = Tc
5. Train an LSTM to predict SOH (capacity fade) as a companion estimator

---

## 2. What Was Already Done (Previous `latest` Branch)

The pipeline was structurally complete and ran end-to-end:
- NASA data download, parsing, ECM identification (20 random-start L-BFGS-B)
- 2-state thermal tuning (Rin, Rout, Cc, Cs)
- EV drive cycle dataset from EPA speed traces (UDDS/HWFET/US06)
- LSTM for residual SOH correction
- Transformer with MC Dropout uncertainty
- Streamlit dashboard
- Paper plots

A previous run even completed and logged apparently good metrics:
- ECM V_RMSE: 6–7 mV, Ts_RMSE: 7°C, Transformer RMSE: 0.46°C

**However, those numbers were wrong** — several fundamental errors were silently corrupting the physics while the code appeared to run fine.

---

## 3. Every Flaw Found and Why It Was Wrong

### 3a. Step 0 — EPA Downloader (`step0_download_epa_drive_cycles.py`)

| # | Flaw | Why it was wrong |
|---|------|-----------------|
| 1 | `sep='\s+'` (non-raw string) | Python 3.12+ issues `SyntaxWarning` for `\s` in a regular string. The `\s` is not a valid escape — Python only accepts it by accident for now; future versions will error. |
| 2 | No error handling at all | If GitHub is unreachable (offline, URL changes), the entire pipeline crashes at step 0 with a confusing traceback, no fallback, no message. |

---

### 3b. Step 2 — NASA Parser (`step2_parse_and_extract_hic.py`)

| # | Flaw | Why it was wrong |
|---|------|-----------------|
| 3 | `R_internal = |ΔV[10]/ΔI[10]|` | Computing resistance from two arbitrary points during active discharge is wrong. During discharge, voltage changes continuously due to OCV slope, RC polarization, and sensor noise — this is not a DC internal resistance. The correct method is an instantaneous pulse-interrupt measurement. This noisy estimate fed into the LSTM as a feature. |
| 4 | Linear SOH baseline (first → last cycle) | NASA cells show a clear *knee* (acceleration of fade) around cycle 100–120. A straight line from cycle 1 to cycle 168 systematically underestimates degradation early and overestimates it late. The LSTM's "residual" target was thus: `true_SOH - wrong_linear_line` — the LSTM was learning to undo a bad model rather than learning genuine nonlinear aging. |
| 5 | `except: pass` on all cycles | Bad cycles were silently discarded with no logging. Could hide data integrity issues. |

---

### 3c. Step 4 — Digital Twin (`step4_generate_aging_digital_twin.py`)
**This file had the most severe bugs — everything downstream is wrong if this is wrong.**

| # | Flaw | Why it was wrong | Observable symptom |
|---|------|-----------------|-------------------|
| 6 | **OCV built from `V[0]` of discharge** | `V[0]` is the *terminal voltage* at the start of discharge — it includes `I·R0 + V1 + V2` drops. For a 2A discharge with R0≈0.08Ω: V_terminal = OCV − 0.16V. The code treated this as OCV, so the entire OCV polynomial was ~150–200 mV too low. Every subsequent calculation (SOC estimation, Q_gen, ECM fitting) was polluted by this bias. | ECM_MSE in the millions early on |
| 7 | **R0 lower bound 0.030 Ω** | Fresh 18650 NASA cells have DC resistance ~0.060–0.090 Ω. The optimizer was minimising with a bound floor at 0.030 Ω. Because the OCV was underestimated (bug 6), the only way the optimizer could make `V_OCV - I·R0 ≈ V_measured` was to shrink R0 as small as possible. It hit the floor every single cycle. Result: R0 = 0.030 Ω for all 636 cycles, even though SOH fell from 0.92 → 0.74 and resistance physically doubled. | `R0 = 0.0300Ω` constant in all printed lines |
| 8 | **No warm-start: 20 fresh random starts per cycle** | Each cycle ran 20 independent random restarts with seed=42. The optimizer could find different local minima on adjacent cycles, causing random jumps in R1/C1/R2/C2 and therefore in Q_gen. This is why Q_gen jumped discontinuously (e.g. 0.221W → 0.337W → 0.435W between consecutive cycles). | Q_gen discontinuous jumps in log |
| 9 | **dt=0 guard missing** | If two consecutive timestamps were identical (or decreasing after `np.diff`), `dt=0` → the RC update `exp(-dt/(R1·C1)) = exp(0) = 1` → `V1 = 1·V1 + I·R1·0 = V1` (V1 never decays). Over many steps, V1 accumulates without bound → huge MSE. Step 2 had the same issue before deduplication was added. | ECM_MSE = 2.57 million early cycles |
| 10 | **Q_gen = \|I·(V_OCV - V_terminal)\|** | This formula is correct in principle but only when OCV is accurate. Since OCV was wrong (bug 6), `V_OCV - V_terminal` was wrong, giving Q_gen ≈ 0.12–0.25 W. For a 2A discharge with correct R0≈0.08Ω, the actual Q_gen should be I²·R0 = 4·0.08 = 0.32 W just from ohmic heating, plus RC losses. The formula also fails to account for V1²/R1 and V2²/R2 (RC Joule heating). | `Q_gen = X W (too low)` warnings every cycle |
| 11 | **Forward Euler thermal ODE** | `Tc[k+1] = Tc[k] + dTc·dt` (explicit Euler). For a 2-state RC system, stability requires `dt < 2·min(Rin·Cc, Rout·Cs)`. With `Rin_min=0.5, Cc_min=5.0` (from bounds), `τ_min=2.5s`, stability requires `dt<5s`. NASA data has `dt≈1s`, so it's marginal. With wrong Q_gen producing spikes, `dTc` could be enormous → Tc[k+1] blows up → Ts[k+1] blows up → NaN propagates. | `RuntimeWarning: overflow in scalar multiply` at lines 294–302 |
| 12 | **C1, C2 not saved in `ecm_parameters.csv`** | The CSV only saved R0, R1, R2. The EV dataset generator then used `C1=15000, C2=3000` hardcoded for all batteries, all aging states. The RC time constants τ1=R1·C1 and τ2=R2·C2 control the dynamic voltage response and heat dissipation — using wrong values means all 245,700 rows of the EV dataset have incorrect dynamics. | No error shown — silent physics error |

---

### 3d. Step 3 — LSTM SOH (`step3_train_residual_lstm.py`)

| # | Flaw | Why it was wrong |
|---|------|-----------------|
| 13 | **Random 80/20 split across all batteries** | With a 10-cycle sliding window and stride=1, sequence #100 from B0005 and sequence #101 from B0005 are 90% identical (9 of 10 timesteps overlap). Splitting randomly means "validation" sequences come from the same battery, same aging region as training. The model memorises the curve — it's interpolation, not generalisation. Val MSE ~0.000123 was meaningless. |
| 14 | **Per-battery cycle normalisation** | `cycle / cycle.max()` was computed per battery. B0005 has 168 cycles, B0018 has 132. Cycle 100 maps to 0.595 for B0005 but 0.758 for B0018. The LSTM sees inconsistent "time" encoding for the same physical aging stage across batteries. |
| 15 | **Single-layer LSTM, hidden=64, no regularisation** | Insufficient capacity for aging dynamics across 4 batteries × 168 cycles. No dropout, no weight decay, no early stopping → prone to overfitting given the small dataset. |

---

### 3e. Step 5 — Transformer (`step5_train_transformer.py`)

| # | Flaw | Why it was wrong |
|---|------|-----------------|
| 16 | **`num_workers=2` on Windows** | PyTorch DataLoader uses `fork()` for worker processes. Windows doesn't support `fork()` — it uses `spawn()` instead, which requires the worker code to be importable at top level. This can cause silent hangs or crashes on Windows, especially inside a conda environment. |
| 17 | **Stride=1 sliding windows** | 377,000 windows from 431,000 rows. Consecutive windows overlap by 59/60 timesteps (98%). Any train/val split at the sequence level will contain near-identical windows in both sets. Even with leave-one-battery-out, the within-training correlation is extreme. |
| 18 | **Only 4 input features: [I, V, R0, Ts]** | SOC (state of charge) is the primary driver of OCV and therefore Q_gen; it's correlated with I and time but not explicitly given. Q_gen is the direct causal driver of Tc — giving the model this physical signal makes learning fundamentally easier and more physically constrained. Both were available in the twin dataset but not used. |

---

## 4. What Was Fixed in `sota-rewrite`

### Step 0
- `sep=r'\s+'` (raw string, eliminates SyntaxWarning)
- Full `try/except` around each download with a meaningful error message
- Embedded fallback speed traces (50-point representative UDDS/HWFET/US06) so the pipeline never crashes offline
- Idempotent: skips already-downloaded files

### Step 2
- **Quadratic SOH baseline**: `np.polyfit(cycles, soh, 2)` — captures the knee. Clamped to `[0, soh[0]]` to prevent unphysical upward baseline.
- **OCV rest points extracted**: New `_extract_ocv_rest_points()` function reads `impedance` and `charge` entries from the .mat file. End-of-charge voltage (at CV phase end where |I|<0.05A) and impedance-entry rest voltages are saved to `{battery}_ocv_rest_points.csv`. Step 4 uses these as the ground truth for OCV fitting.
- `R_internal` replaced with placeholder 0.05Ω (overwritten by ECM R0 in Step 4 before LSTM trains)
- `except: pass` → `warnings.warn(...)` with logged message
- Duplicate/non-monotone timestamps removed per cycle

### Step 4 (most important)
- **OCV curve** now built from `_ocv_rest_points.csv` (near-equilibrium voltages). Falls back to an empirical NMC polynomial if no rest data exists. Both fresh (SOH≥0.90) and aged (SOH≤0.78) polynomials are 6th-order.
- **R0 lower bound**: 0.050 Ω (was 0.030). Physical lower limit for a healthy 18650.
- **ECM warm-start**: first optimiser start uses previous cycle's solution; remaining 15 starts are random. `r0_prev` physics regularisation: penalises R0 jumps beyond 5% of previous cycle.
- **dt=0 guard**: `if dt <= 0.0: continue` in all ECM and thermal loops.
- **Q_gen Joule formula**: `I²·R0 + V1²/R1 + V2²/R2 + I·T·dU/dT(SOC)`. Always non-negative. Does not depend on OCV accuracy.
- **Crank-Nicolson thermal integrator**: solves the 2×2 linear system exactly at each step. Unconditionally stable for any `dt`, any physical parameter values. Physical temperature clamp (−40°C to 150°C) as safety net.
- **C1, C2 saved** in `ecm_parameters.csv` and loaded in EV generator.
- **ICA dQ/dV peak tracking**: per cycle, finds top-2 peaks in the incremental capacity curve. Saves `ica_peak1_v`, `ica_peak2_v`, `ica_peak_ratio` to twin dataset. These track lithium-staging transitions and shift measurably with aging.
- **SOC trajectory** saved in twin dataset (needed for Q_gen in transformer input).
- **MSE printed in mV²** (×1e6) for human readability.
- **Adaptive aging-state sampling** for EV dataset: over-samples cycles near the SOH knee where gradient is steepest.

### Step 3
- **Leave-one-out**: B0018 always held out for validation. Training on B0005/B0006/B0007 only.
- **Global cycle normalisation**: `cycle / 200` for all batteries.
- **BiLSTM(hidden=128, 2 layers, dropout=0.3)**: bidirectional captures both forward and backward aging context.
- **Dropout(0.2)** on the regression head.
- **Early stopping** (patience=20) on val loss.
- **AdamW + CosineAnnealingLR** (lr=5e-4, T_max=150).
- **Gradient clipping** (max_norm=1.0).
- **ICA features** [peak1_v, peak2_v, peak_ratio] added if available (6-dim input vs 3-dim before).
- Best weights restored after training.

### Step 5
- **`num_workers=0`** — no DataLoader deadlock on Windows.
- **`stride=60`** — non-overlapping windows, honest validation metrics.
- **6 input features**: `[current_A, voltage_V, r0_ohms, temp_surface_C, soc, q_gen_W]`.
- **Sinusoidal positional encoding** (deterministic, no random init).
- **Pre-LN TransformerEncoderLayer** (`norm_first=True`) — more stable gradients.
- **Physics smoothness loss**: small L2 penalty on consecutive predicted Tc changes.
- **Gradient clipping** (max_norm=1.0).
- **Best-checkpoint saving** (not just final epoch).
- **Early stopping** (patience=15).

---

## 5. Will It Work After Training? What to Expect

### Expected improvements vs `latest` branch

| Metric | `latest` (broken) | `sota-rewrite` (expected) |
|--------|-------------------|--------------------------|
| R0 at cycle 1 | 0.030 Ω (floor, wrong) | ~0.065–0.085 Ω (physically realistic) |
| R0 at cycle 168 | 0.030 Ω (no change) | ~0.12–0.18 Ω (aging trend visible) |
| ECM_MSE early cycles | ~2.57 million (V² blowup) | ~0.0001–0.001 V² (≈10–30 mV RMSE) |
| Q_gen mean | 0.12–0.45 W (too low, jumpy) | 0.3–0.8 W (physically correct, smooth) |
| Thermal overflow warnings | Many per battery | None (Crank-Nicolson) |
| LSTM val RMSE (SOH) | ~0.011 (leakage-inflated) | ~0.015–0.025 on true held-out B0018 |
| Transformer val RMSE (Tc) | ~0.46°C (stride-1 inflated) | ~0.8–1.5°C honest (stride-60 held-out) |

**Note:** The transformer RMSE may appear *worse* numerically than before, but that's because the previous 0.46°C was measured on overlapping windows from the same battery — essentially interpolation. The new 0.8–1.5°C is a genuine held-out, non-overlapping test on B0018 (a different battery). This is the physically meaningful number and is still excellent for core temperature estimation from surface-only measurements.

### What to look for when you run it

**Step 4 output — these are the key checks:**
- R0 should start ~0.065–0.090 Ω (B0005 fresh) and increase to ~0.12–0.18 Ω by cycle 168
- Q_gen should be 0.3–0.8 W for 2A discharge (sanity: I²·R0 = 4·0.08 = 0.32 W minimum)
- No `RuntimeWarning: overflow` messages
- ECM_MSE should be in the range 0.00005–0.005 (printed as 50–5000 mV²)
- Ts_RMSE should be 1–5°C (was ~7°C before; Crank-Nicolson + correct Q_gen should improve this)

**Step 3 output:**
- Val MSE on B0018 (genuinely held-out) — expect 0.015–0.030
- Early stopping should trigger well before epoch 150 if the data is clean

**Step 5 output:**
- Fewer total windows (stride-60 vs stride-1: ~6000 vs ~370000)
- Training will be much faster (minutes not hours on CPU)
- Val RMSE in °C on B0018 — expect 0.8–2.0°C

### Potential remaining issues to watch for

1. **If OCV rest points are sparse**: Step 4 falls back to an empirical NMC polynomial for OCV. Check the log line `✅ OCV curves: N fresh pts, M aged pts`. If N or M < 20, the OCV is still approximate. You can manually inspect the `data/nasa/processed/*_ocv_rest_points.csv` files.

2. **If ECM_MSE is still large (>0.01 V² = 100 mV RMSE)**: The OCV fallback may be too far off for your specific NASA cell batch. Try increasing the R0 upper bound to 0.25 Ω in step4 line: `(0.050, 0.250)`.

3. **If Q_gen < 0.2 W still**: Check if `v_sim` and `v_ocv` are close (meaning R0 is small). Print `R0` directly in the loop — if it's still near 0.050 (new floor), the OCV fallback may still be biased downward.

4. **Transformer RMSE > 3°C**: With stride-60 and only ~100 cycles of B0018 data for validation, the val set is small. Try reducing the physics smoothness loss weight (alpha=0.01 in step5) if the model underfits.

---

## 6. How to Run

```bash
# Make sure you are on the correct branch
git checkout sota-rewrite

# Activate your conda environment
conda activate battery-modelling

# Run the full pipeline (CPU, no GPU needed)
python run_pipeline.py
```

The pipeline runs steps in order: download → parse → twin → LSTM → transformer → plots.
All intermediate files are saved to `data/` and `results/paper_plots/`.

---

## 7. Files Changed in `sota-rewrite`

| File | What changed |
|------|-------------|
| `data/step0_download_epa_drive_cycles.py` | Raw string, error handling, fallback traces |
| `data/step2_parse_and_extract_hic.py` | Quadratic baseline, OCV rest extraction, timestamp dedup |
| `generation/step4_generate_aging_digital_twin.py` | Full physics rewrite (OCV, R0 bounds, warm-start, Crank-Nicolson, Joule Q_gen, C1/C2, ICA) |
| `soh/step3_train_residual_lstm.py` | BiLSTM, leave-one-out, ICA features, early stop |
| `transformer/step5_train_transformer.py` | stride-60, 6 features, sinusoidal PE, physics loss, num_workers=0 |
| `reports/generate_paper_plots.py` | Updated to match new architecture and new columns |
| `run_pipeline.py` | Updated step descriptions, CPU-friendly env setup |

**Untouched (working correctly):**
- `data/step1_download_nasa.py`
- `run_ui_dashboard.py` (will need minor updates for new columns when you use it)
- `pyproject.toml`
