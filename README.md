# EV Battery Digital Twin — Core Temperature Estimation

> **Hybrid Electrical Circuit Model and Deep Learning-Based Core Temperature Estimation of Li-Ion Batteries**
>
> Based on: *Samanta, Surya, Williamson et al., IEEE Transactions on Transportation Electrification, 2022*

A fully automated, end-to-end pipeline that downloads real NASA Ames battery aging data, identifies physics model parameters from experiments, generates core temperature ground truth via a calibrated digital twin, and trains deep learning models for **State of Health (SOH)** estimation and **Core Temperature (Tc)** prediction.

---

## Architecture Overview
```
Real NASA 18650 Data (B0005, B0006, B0007, B0018)
│
▼
┌───────────────────────────────────────────────────────┐
│ Step 1 Download NASA Ames Battery Aging Dataset │
│ (.mat files from PHM S3 mirror) │
└───────────────────────┬───────────────────────────────┘
▼
┌───────────────────────────────────────────────────────┐
│ Step 0 Download EPA Drive Cycle Speed Traces │
│ (UDDS, HWFET, US06 from CIRCLES repo) │
└───────────────────────┬───────────────────────────────┘
▼
┌───────────────────────────────────────────────────────┐
│ Step 2 Parse .mat → Per-battery CSVs │
│ • aging_features.csv (linear-fade SOH) │
│ • discharge_timeseries.csv (full V, I, Ts) │
└───────────────────────┬───────────────────────────────┘
▼
┌───────────────────────────────────────────────────────┐
│ Step 4 Experimentally-Tuned Digital Twin │
│ ┌─────────────────────────────────────────────────┐ │
│ │ • Extracts OCV-SOC curve from NASA data │ │
│ │ • Multi-start ECM identification (20 starts) │ │
│ │ • Tight physical bounds + noise filtering │ │
│ │ • Estimates initial SOC from OCV curve │ │
│ │ • Entropic heat term included in thermal model │ │
│ │ • Median thermal parameters for EV generation │ │
│ │ • SOH-aging OCV blending (fresh vs aged) │ │
│ │ • ECM parameters saved for Step 3 │ │
│ │ • Validation log: V_RMSE, Ts_RMSE, Q_gen, ΔT │ │
│ └─────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────┐ │
│ │ • Real EPA drive cycles (UDDS/HWFET/US06) │ │
│ │ • Aging thermal scaling (β=0.2) │ │
│ │ • Multi-ambient visualisations (0/20/50°C) │ │
│ │ • 288 EV simulations: 4 batt × 8 aging × 3 │ │
│ │ drives × 3 temps │ │
│ └─────────────────────────────────────────────────┘ │
└───────────────────────┬───────────────────────────────┘
▼
┌───────────────────────────────────────────────────────┐
│ Step 3 LSTM Residual SOH │
│ • Uses ECM-identified R₀ from Step 4 │
│ • Learns deviation from linear fade baseline │
│ • 10-cycle sliding window │
│ • 80/20 train/val split, 100 epochs │
└───────────────────────┬───────────────────────────────┘
▼
┌───────────────────────────────────────────────────────┐
│ Step 5 Transformer Encoder │
│ • Interpolates NASA data to 1s, window_size=60 │
│ • Leave-one-battery-out validation (hold out B0018) │
│ • MC Dropout uncertainty quantification │
│ • d=128, 4 heads, 4 layers, 100 epochs │
│ • Input: [I, V, R0, Ts] → Target: Tc │
└───────────────────────┬───────────────────────────────┘
▼
┌───────────────────────────────────────────────────────┐
│ Step 6 Paper-Quality Plots │
│ • 8 figures with uncertainty bands │
│ • Transformer test on held-out B0018 │
│ • EV US06 validation with 95% CI │
└───────────────────────┬───────────────────────────────┘
▼
┌───────────────────────────────────────────────────────┐
│ Streamlit Interactive Dashboard │
│ • 9 pages: Overview, ECM, Thermal, Aging, │
│ Loss Curves, EV Drives, Transformer Validation, │
│ Live Inference (with uncertainty), Paper Plots │
└───────────────────────────────────────────────────────┘


---

## Key Methodological Approach

### Why This Works

The **core temperature of a Li-ion battery cannot be directly measured** in a real vehicle — only the surface temperature is available. This project creates a **physics-calibrated digital twin** that generates trustworthy core temperature labels for training a neural network.

1. **ECM Parameter Identification** — A 2-RC equivalent circuit model (R₀, R₁, C₁, R₂, C₂) is identified per discharge cycle using multi-start L-BFGS-B optimisation with tight physical bounds and Savitzky-Golay noise filtering.

2. **Thermal Model Calibration** — The 2-state EETM parameters (Rin, Rout, Cc, Cs) are tuned so the simulated surface temperature matches real measurements. The entropic heat term is included for physical accuracy.

3. **OCV-SOC Extraction** — The open-circuit voltage curve is extracted directly from the NASA discharge data, with separate fresh (SOH≥0.9) and aged (SOH≤0.75) curves blended by current SOH.

4. **Residual SOH Learning** — A linear fade baseline (first-to-last SOH) provides the physics approximation. An LSTM learns the real nonlinear deviation from this line, using ECM-identified R₀ as the aging feature.

5. **Real-World Drive Cycles** — The EPA drive cycles (UDDS, HWFET, US06) are converted to cell current via a vehicle dynamics model, replacing synthetic random patterns.

6. **Transformer with Uncertainty** — A Transformer Encoder predicts core temperature with Monte Carlo dropout for 95% confidence intervals. Held-out battery validation ensures genuine generalization.

### What Makes This Different

- **Every ECM and thermal parameter** is identified from real experimental data.
- The OCV curve is extracted from the same cells, not borrowed.
- Thermal aging is applied via a physics-based scaling factor (β=0.2).
- The Transformer is validated on a fully held-out battery (B0018).
- Uncertainty quantification makes the model safety-relevant.
- Real EPA drive cycles replace synthetic random patterns.

---

## Dataset

**NASA Ames Prognostics Center of Excellence — Battery Aging Dataset**

| Cell | Chemistry | Nominal Capacity | Cycles |
|------|-----------|------------------|--------|
| B0005 | 18650 Li-ion | 2.0 Ah | ~168 |
| B0006 | 18650 Li-ion | 2.0 Ah | ~168 |
| B0007 | 18650 Li-ion | 2.0 Ah | ~168 |
| B0018 | 18650 Li-ion | 2.0 Ah | ~132 |

---

## Project Structure
```
EV-battery-management-system/
├── run_pipeline.py # Master orchestrator
├── run_ui_dashboard.py # Streamlit dashboard
├── pyproject.toml # Dependencies
│
├── data/
│ ├── step0_download_epa_drive_cycles.py # EPA speed trace downloader
│ ├── step1_download_nasa.py # NASA .mat downloader
│ ├── step2_parse_and_extract_hic.py # Parse .mat → CSVs
│ ├── drive_cycles/ # EPA speed CSVs
│ │ ├── UDDS_epa_speed.csv
│ │ ├── HWFET_epa_speed.csv
│ │ └── US06_epa_speed.csv
│ ├── nasa/ # Raw .mat files
│ │ ├── B0005.mat ... B0018.mat
│ │ └── processed/
│ │ ├── B0005_aging_features.csv
│ │ ├── B0005_discharge_timeseries.csv
│ │ └── ...
│ ├── digital_twin_sets/
│ │ ├── augmented_aging_twin_dataset.csv
│ │ ├── ecm_parameters.csv
│ │ └── validation_log.csv
│ └── ev_validation_sets/
│ └── ev_drive_cycle_dataset.csv
│
├── generation/
│ └── step4_generate_aging_digital_twin.py # Full digital twin
│
├── soh/
│ ├── step3_train_residual_lstm.py # LSTM SOH trainer
│ └── models/
│ └── lstm_residual_soh.pth
│
├── transformer/
│ ├── step5_train_transformer.py # Transformer trainer
│ └── models/
│ ├── transformer_thermal_core.pth
│ ├── normalisation_stats.csv
│ └── val_uncertainty.npz
│
├── reports/
│ └── generate_paper_plots.py # Paper-quality figures
│
└── results/
└── paper_plots/
├── fig1_voltage_validation.png
├── fig2_surface_temp_validation.png
├── fig3_core_temperature.png
├── fig4_parameter_aging.png
├── fig5_soh_residual.png
├── fig6_drive_thermal.png
├── transformer_test_validation.png
├── ev_us06_transformer_validation.png
├── lstm_training_loss.png
├── transformer_training_loss.png
├── aggressive_multi_temp_visualization.png
├── mixed_multi_temp_visualization.png
└── us06_ev_triple_stack.png


---

## Pre-Requisites

- **Python 3.9+**
- **NVIDIA GPU** with CUDA support (optional; CPU fallback works)
- 8 GB disk space (for NASA data + generated datasets)

---

# 🔋 EV Battery Management System

## 🚀 Installation

### Option A: Using pip

```bash
git clone https://github.com/The-Departed/EV-battery-management-system.git
cd EV-battery-management-system

python -m venv .venv

# On Windows:
.\.venv\Scripts\activate

# On Linux/Mac:
source .venv/bin/activate

pip install -e .
```

### Option B: Using uv

```bash
git clone https://github.com/The-Departed/EV-battery-management-system.git
cd EV-battery-management-system

uv sync
```

---

## ▶️ Running the Pipeline

```bash
python run_pipeline.py
```

## 📊 Launching the Dashboard

```bash
streamlit run run_ui_dashboard.py --server.port 8501
```

---

## 🧠 Model Architectures

### 🔹 Residual LSTM — State of Health (SOH)

**Input**
```
[soh_physics_baseline, r_internal_ohms, cycle_norm] × 10 timesteps
```

**Architecture**
```
LSTM(hidden=64, layers=1) → Linear(64 → 1)
```

**Output**
```
SOH_final = SOH_physics + LSTM_residual
```

---

### 🔹 Transformer Encoder — Core Temperature

**Input**
```
[current_A, voltage_V, r0_ohms, temp_surface_C] × 60 timesteps (1 min)
```

**Architecture**
```
Linear(4 → 128)
+ Positional Encoding
→ 4 Encoder Layers (4 heads, FF=256, dropout=0.1)
→ Linear(128 → 32)
→ GELU
→ Linear(32 → 1)
```

**Uncertainty Estimation**
```
Monte Carlo Dropout (n=50 samples) → 95% Confidence Interval
```

---

## 🛠️ Key Fixes from Original Pipeline

| Issue | Fix |
|---|---|
| Synthetic drive cycles | Real EPA speed traces via vehicle model |
| Fake SOH baseline | Linear fade fitted to first & last SOH |
| ECM not converging | Multi-start (20) + tight bounds + noise filter |
| Wrong OCV curve | Extracted from NASA cells (fresh + aged) |
| Missing entropic heat | `dU/dT` lookup + `I·T·dU/dT` term added |
| Sampling mismatch | NASA interpolated to 1s, `window_size=60` |
| Data leakage | Transformer validated on held-out B0018 |
| No uncertainty | MC Dropout with 95% confidence intervals |
| Broken `r_internal` | Replaced with ECM-identified R₀ |
| No validation | `validation_log.csv` + Q_gen warnings |
| SOC always 1.0 | Estimated from OCV curve per cycle |
| No thermal aging | Aging factor applied to Rin/Rout |

---

## 📚 References

- A. Samanta, S. Surya, S. Williamson, et al. *Hybrid Electrical Circuit Model and Deep Learning-Based Core Temperature Estimation of Li-Ion Cells*, IEEE TTE, 2022.
- B. Saha and K. Goebel. *Battery Data Set*, NASA Ames Prognostics Data Repository, 2007.
- EPA Dynamometer Drive Schedules — https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules
```