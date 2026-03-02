# EV Battery Digital Twin — Core Temperature Estimation

> **Hybrid Electrical Circuit Model and Deep Learning-Based Core Temperature Estimation of Li-Ion Batteries**
>
> Based on: *Samanta, Surya, Williamson et al., IEEE Transactions on Transportation Electrification, 2022* (DOI: [10.1109/TTE.2022.3170359](https://doi.org/10.1109/TTE.2022.3170359))

A fully automated, end-to-end pipeline that downloads real NASA Ames battery aging data, identifies physics model parameters from experiments, generates core temperature ground truth via a calibrated digital twin, and trains deep learning models for **State of Health (SOH)** estimation and **Core Temperature (Tc)** prediction.

---

## Architecture Overview

```
Real NASA 18650 Data (B0005, B0006, B0007, B0018)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Step 1  Download NASA Ames Battery Aging Dataset     │
│          (.mat files from PHM S3 mirror)              │
└───────────────────────┬───────────────────────────────┘
                        ▼
┌───────────────────────────────────────────────────────┐
│  Step 2  Parse .mat → Per-battery CSVs                │
│          • aging_features.csv  (per-cycle SOH)        │
│          • discharge_timeseries.csv  (full V, I, Ts)  │
└──────────┬────────────────────────┬───────────────────┘
           ▼                        ▼
┌─────────────────────┐  ┌─────────────────────────────────────┐
│  Step 3  LSTM       │  │  Step 4  Experimentally-Tuned        │
│  Residual SOH       │  │  Aging-Aware Digital Twin             │
│  (physics + LSTM    │  │  ┌─────────────────────────────────┐ │
│   correction)       │  │  │ (a) 2-RC ECM identification     │ │
│  100 epochs, all 4  │  │  │     per-cycle from real V-I     │ │
│  batteries          │  │  │ (b) 2-state EETM thermal model  │ │
└─────────────────────┘  │  │     tuned against real Ts (UKS) │ │
                         │  │ (c) Core temp Tc = physics truth │ │
                         │  └─────────────────────────────────┘ │
                         └──────────────┬──────────────────────┘
                                        ▼
                         ┌──────────────────────────────────────┐
                         │  Step 5  Transformer Encoder          │
                         │  Core Temperature Prediction          │
                         │  d=128, 4 heads, 4 layers, 100 ep    │
                         │  Input: [I, V, R0, Ts] → Target: Tc  │
                         └──────────────┬───────────────────────┘
                                        ▼
                         ┌──────────────────────────────────────┐
                         │  Step 6  Paper-Quality Plots          │
                         │  6 figures → results/paper_plots/     │
                         └──────────────────────────────────────┘
                                        ▼
                         ┌──────────────────────────────────────┐
                         │  Streamlit Interactive Dashboard       │
                         │  Live LSTM + Transformer inference    │
                         └──────────────────────────────────────┘
```

---

## Key Methodological Approach

### Why This Works (and Why It Matters)

The **core temperature of a Li-ion battery cannot be directly measured** in a real vehicle — only the surface temperature is available via thermocouples. This project creates a **physics-calibrated digital twin** that generates trustworthy core temperature labels for training a neural network.

1. **ECM Parameter Identification** — A 2-RC equivalent circuit model (R₀, R₁, C₁, R₂, C₂) is identified per discharge cycle by minimising the error between simulated and **real measured terminal voltage** (NASA data) using L-BFGS-B optimisation.

2. **Thermal Model Calibration (UKS-style)** — The 2-state electrothermal model (EETM) parameters (Rᵢₙ, Rₒᵤₜ, Cₖ, Cₛ) are tuned so that the simulated **surface temperature matches the real measured surface temperature**. Once calibrated, the model's internal core temperature output is physically meaningful.

3. **Residual SOH Learning** — A simple physics baseline (Coulomb counting + capacity measurement) provides SOH estimates. An LSTM learns the **residual error** between this baseline and the true SOH, yielding corrected predictions.

4. **Transformer Core Temp Prediction** — A Transformer Encoder is trained on [current, voltage, R₀, T_surface] → T_core, using the physics-twin core temperature as the label. This replaces the need for invasive core temperature sensors.

### What Makes This Different from Mock/Synthetic-Only Approaches

- **Every parameter** in the ECM and thermal model is identified from real experimental data — not hardcoded.
- Parameters **warm-start** from the previous cycle, naturally capturing aging-induced resistance growth.
- The thermal model is **validated against real surface temperature** before its core temperature output is trusted.
- The Transformer never sees mock data — all inputs come from real NASA measurements.

---

## Dataset

**NASA Ames Prognostics Center of Excellence — Battery Aging Dataset**

| Cell | Chemistry | Nominal Capacity | Discharge Rate | Cycles |
|------|-----------|------------------|----------------|--------|
| B0005 | 18650 Li-ion | 2.0 Ah | ~2A CC | ~168 |
| B0006 | 18650 Li-ion | 2.0 Ah | ~2A CC | ~168 |
| B0007 | 18650 Li-ion | 2.0 Ah | ~2A CC | ~168 |
| B0018 | 18650 Li-ion | 2.0 Ah | ~2A CC | ~132 |

Sampling interval: ~18 seconds. Each discharge record includes time, current, voltage, and surface temperature.

---

## Project Structure

```
EV-battery-management-system/
├── run_pipeline.py                 # Master orchestrator (runs Steps 1→6)
├── run_ui_dashboard.py             # Streamlit interactive dashboard
├── pyproject.toml                  # Dependencies (managed by uv)
│
├── data/
│   ├── step1_download_nasa.py      # Step 1: Download .mat files
│   ├── step2_parse_and_extract_hic.py  # Step 2: Parse → CSVs
│   ├── nasa/                       # Raw .mat + processed CSVs
│   │   ├── B0005.mat ... B0018.mat
│   │   └── processed/
│   │       ├── B0005_aging_features.csv
│   │       ├── B0005_discharge_timeseries.csv
│   │       └── ...
│   └── digital_twin_sets/
│       └── augmented_aging_twin_dataset.csv
│
├── soh/
│   ├── step3_train_residual_lstm.py    # Step 3: LSTM residual SOH
│   └── models/
│       └── lstm_residual_soh.pth
│
├── generation/
│   └── step4_generate_aging_digital_twin.py  # Step 4: Physics digital twin
│
├── transformer/
│   ├── step5_train_transformer.py      # Step 5: Transformer core temp
│   └── models/
│       ├── transformer_thermal_core.pth
│       └── normalisation_stats.csv
│
├── reports/
│   ├── generate_paper_plots.py         # Step 6: Paper figures
│   └── (plots saved to results/paper_plots/)
│
├── results/
│   └── paper_plots/
│       ├── fig1_voltage_validation.png
│       ├── fig2_surface_temp_validation.png
│       ├── fig3_core_temperature.png
│       ├── fig4_parameter_aging.png
│       ├── fig5_soh_residual.png
│       └── fig6_drive_thermal.png
│
└── docs_gemini_architectural_plans/    # Architectural docs & planning
```

---

## Pre-Requisites

- **Python 3.9+** (tested on 3.13)
- **NVIDIA GPU** with CUDA support (pipeline defaults to `CUDA_VISIBLE_DEVICES=1`)
- **[uv](https://docs.astral.sh/uv/)** package manager (recommended) or pip

---

## Installation

### Option A: Using uv (Recommended)

```bash
git clone https://github.com/The-Departed/EV-battery-management-system.git
cd EV-battery-management-system

# Create venv and install all dependencies
uv sync
```

### Option B: Using pip

```bash
git clone https://github.com/The-Departed/EV-battery-management-system.git
cd EV-battery-management-system

python -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas scipy matplotlib seaborn scikit-learn h5py pyyaml requests streamlit plotly
```

---

## Running the Pipeline

### Full Pipeline (Steps 1–6)

```bash
# Activate the environment
source .venv/bin/activate

# Run the complete pipeline
python run_pipeline.py
```

This sequentially executes:

| Step | Script | What It Does | Output |
|------|--------|-------------|--------|
| 1 | `data/step1_download_nasa.py` | Downloads NASA battery .mat files from PHM S3 mirror (handles nested zips) | `data/nasa/B0005.mat` … `B0018.mat` |
| 2 | `data/step2_parse_and_extract_hic.py` | Parses .mat files → per-battery aging features + full discharge time-series CSVs | `data/nasa/processed/*.csv` |
| 3 | `soh/step3_train_residual_lstm.py` | Trains LSTM (hidden=64) on all 4 batteries for 100 epochs to learn SOH residual | `soh/models/lstm_residual_soh.pth` |
| 4 | `generation/step4_generate_aging_digital_twin.py` | Per-cycle ECM identification (L-BFGS-B on real V-I) + EETM thermal calibration (against real Ts) → core temp labels | `data/digital_twin_sets/augmented_aging_twin_dataset.csv` |
| 5 | `transformer/step5_train_transformer.py` | Trains Transformer Encoder (d=128, 4 heads, 4 layers) for 100 epochs with CosineAnnealing LR | `transformer/models/transformer_thermal_core.pth` |
| 6 | `reports/generate_paper_plots.py` | Generates 6 paper-quality PNG figures from real pipeline data | `results/paper_plots/fig*.png` |

> **Note:** Step 4 is CPU-bound (per-cycle scipy optimisation). Steps 3 & 5 use GPU. Total pipeline time: ~30–60 minutes depending on GPU.

### Running Individual Steps

```bash
# Skip download if .mat files already exist
python data/step2_parse_and_extract_hic.py

# Retrain just the transformer
CUDA_VISIBLE_DEVICES=1 python transformer/step5_train_transformer.py

# Regenerate plots only
python reports/generate_paper_plots.py
```

### GPU Selection

The pipeline defaults to GPU 1 (`CUDA_VISIBLE_DEVICES=1`). To change:

```bash
# Use GPU 0 instead
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py
```

### Running in tmux (Recommended for Long Training)

```bash
tmux new -s battery
source .venv/bin/activate
python run_pipeline.py
# Detach: Ctrl+B, then D
# Re-attach: tmux attach -t battery
```

---

## Launching the Dashboard

After the pipeline completes (or even partially — the dashboard gracefully handles missing outputs):

```bash
streamlit run run_ui_dashboard.py --server.port 8501
```

The dashboard provides 6 interactive pages:

| Page | Description |
|------|-------------|
| 📊 **Overview** | Fleet-level SOH KPIs, capacity fade curves, residual learning visualisation |
| ⚡ **ECM Voltage Validation** | Interactive cycle-by-cycle comparison of measured vs simulated voltage with error metrics |
| 🌡️ **Thermal Validation** | Surface temperature validation + core vs surface temperature plots |
| 🔧 **Parameter Aging** | R₀, R₁, R₂ evolution across cycles showing aging-induced resistance growth |
| 🧠 **Live Inference** | Run LSTM SOH correction and Transformer core temp prediction on any battery/cycle |
| 📄 **Paper Plots** | View the 6 pre-generated matplotlib figures from Step 6 |

The sidebar shows real-time pipeline status (which steps have been completed, dataset statistics, model availability).

---

## Model Architectures

### Residual LSTM (SOH)
```
Input:  [soh_physics_baseline, r_internal_ohms, cycle_norm] × 10 timesteps
LSTM:   hidden=64, layers=1
Output: residual correction → SOH_final = SOH_physics + LSTM_residual
```

### Transformer Encoder (Core Temperature)
```
Input:  [current_A, voltage_V, r0_ohms, temp_surface_C] × window_size timesteps
Embed:  Linear(4 → 128) + learned positional encoding
Encoder: 4 layers, 4 heads, feedforward=256, dropout=0.1
Head:   Linear(128 → 32) → GELU → Linear(32 → 1)
Output: predicted core temperature (°C)
```

---

## Generated Figures

| Figure | Description |
|--------|-------------|
| `fig1_voltage_validation.png` | 2-RC ECM simulated vs measured terminal voltage for a mid-aging cycle |
| `fig2_surface_temp_validation.png` | EETM surface temperature validation at early, mid, and late aging |
| `fig3_core_temperature.png` | Core vs surface temperature showing thermal gradient from physics twin |
| `fig4_parameter_aging.png` | R₀ growth and SOH fade across all 4 batteries |
| `fig5_soh_residual.png` | Residual learning setup: true SOH, physics baseline, and LSTM correction target |
| `fig6_drive_thermal.png` | Current profile, voltage response, and thermal response for a discharge cycle |

---

## References

1. A. Samanta, S. Surya, S. Williamson, et al., "Hybrid Electrical Circuit Model and Deep Learning-Based Core Temperature Estimation of Li-Ion Cells," *IEEE Transactions on Transportation Electrification*, 2022. DOI: [10.1109/TTE.2022.3170359](https://doi.org/10.1109/TTE.2022.3170359)
2. B. Saha and K. Goebel, "Battery Data Set," *NASA Ames Prognostics Data Repository*, 2007.
3. C. R. Birkl et al., "Degradation diagnostics for lithium ion cells," *Journal of Power Sources*, 2017.

---

## License

This project is for academic and research purposes.
