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
│  80/20 train/val    │  │  │     per-cycle from real V-I     │ │
│  100 epochs, all 4  │  │  │ (b) 2-state EETM thermal model  │ │
│  batteries          │  │  │     tuned against real Ts (UKS) │ │
│  + loss curve plot  │  │  │ (c) Core temp Tc = physics truth │ │
└─────────────────────┘  │  ├─────────────────────────────────┤ │
                         │  │ (d) Multi-ambient drive viz      │ │
                         │  │     Aggressive/Mixed at 0/20/50°C│ │
                         │  │ (e) EV drive-cycle dataset       │ │
                         │  │     288 sims: 4 batt × 8 aging   │ │
                         │  │     × 3 drives × 3 temps         │ │
                         │  │     (UDDS / HWFET / US06)        │ │
                         │  └─────────────────────────────────┘ │
                         └──────────────┬──────────────────────┘
                                        ▼
                         ┌──────────────────────────────────────┐
                         │  Step 5  Transformer Encoder          │
                         │  Core Temperature Prediction          │
                         │  d=128, 4 heads, 4 layers, 100 ep    │
                         │  Input: [I, V, R0, Ts] → Target: Tc  │
                         │  Trains on NASA twin + EV drive data  │
                         │  80/20 train/val + loss curve plot    │
                         └──────────────┬───────────────────────┘
                                        ▼
                         ┌──────────────────────────────────────┐
                         │  Step 6  Paper-Quality Plots          │
                         │  8 figures → results/paper_plots/     │
                         │  Including Transformer test           │
                         │  validation & EV US06 validation      │
                         └──────────────────────────────────────┘
                                        ▼
                         ┌──────────────────────────────────────┐
                         │  Streamlit Interactive Dashboard       │
                         │  9 pages: Overview, ECM, Thermal,     │
                         │  Aging, Loss Curves, EV Drives,       │
                         │  Transformer Validation, Inference,   │
                         │  Paper Plots                          │
                         └──────────────────────────────────────┘
```

---

## Key Methodological Approach

### Why This Works (and Why It Matters)

The **core temperature of a Li-ion battery cannot be directly measured** in a real vehicle — only the surface temperature is available via thermocouples. This project creates a **physics-calibrated digital twin** that generates trustworthy core temperature labels for training a neural network.

1. **ECM Parameter Identification** — A 2-RC equivalent circuit model (R₀, R₁, C₁, R₂, C₂) is identified per discharge cycle by minimising the error between simulated and **real measured terminal voltage** (NASA data) using L-BFGS-B optimisation.

2. **Thermal Model Calibration (UKS-style)** — The 2-state electrothermal model (EETM) parameters (Rᵢₙ, Rₒᵤₜ, Cₖ, Cₛ) are tuned so that the simulated **surface temperature matches the real measured surface temperature**. Once calibrated, the model's internal core temperature output is physically meaningful.

3. **Residual SOH Learning with Train/Val Split** — A simple physics baseline (Coulomb counting + capacity measurement) provides SOH estimates. An LSTM learns the **residual error** between this baseline and the true SOH. An 80/20 train/validation split (via `sklearn.model_selection.train_test_split`) is used to monitor generalisation, with train and validation loss curves plotted and saved.

4. **Multi-Ambient EV Drive-Cycle Data Generation** — The calibrated digital twin is applied to realistic EV drive cycles (UDDS, HWFET, US06) at multiple ambient temperatures (0°C, 25°C, 45°C) across 8 aging states per battery, producing **288 unique physics-based simulations** (~200 hours of synthetic driving data). Additionally, Aggressive and Mixed drive profiles are visualised at 0°C, 20°C, and 50°C to show temperature-dependent thermal behaviour.

5. **Transformer Core Temp Prediction with Expanded Training** — A Transformer Encoder is trained on both the NASA twin dataset and the EV drive-cycle dataset using [current, voltage, R₀, T_surface] → T_core mapping. The combined dataset provides far greater diversity in operating conditions. An 80/20 train/val split with loss curve tracking ensures convergence monitoring.

6. **Transformer Test Validation** — After training, the Transformer is evaluated on unseen data (late-aging cycles of B0018) and on EV drive cycles (US06), producing predicted vs actual core temperature comparison plots with estimation error analysis.

### What Makes This Different from Mock/Synthetic-Only Approaches

- **Every parameter** in the ECM and thermal model is identified from real experimental data — not hardcoded.
- Parameters **warm-start** from the previous cycle, naturally capturing aging-induced resistance growth.
- The thermal model is **validated against real surface temperature** before its core temperature output is trusted.
- The Transformer trains on **both real NASA cycles and physics-generated EV drive cycles**, covering diverse conditions (3 standard drive profiles × 3 temperatures × 4 batteries × 8 aging states).
- **Train/validation splits** with loss curves provide proper convergence evidence — no overfitting on training data.
- **Test validation on unseen data** (different battery, late aging) demonstrates generalisation capability.

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
├── run_ui_dashboard.py             # Streamlit interactive dashboard (9 pages)
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
│   ├── digital_twin_sets/
│   │   └── augmented_aging_twin_dataset.csv
│   └── ev_validation_sets/
│       └── ev_drive_cycle_dataset.csv  # 288 EV simulations
│
├── soh/
│   ├── step3_train_residual_lstm.py    # Step 3: LSTM residual SOH (80/20 split)
│   └── models/
│       └── lstm_residual_soh.pth
│
├── generation/
│   └── step4_generate_aging_digital_twin.py  # Step 4: Physics twin + EV data gen
│
├── transformer/
│   ├── step5_train_transformer.py      # Step 5: Transformer (NASA + EV data)
│   └── models/
│       ├── transformer_thermal_core.pth
│       └── normalisation_stats.csv
│
├── reports/
│   ├── generate_paper_plots.py         # Step 6: Paper figures (8 plots)
│   └── (plots saved to results/paper_plots/)
│
├── results/
│   └── paper_plots/
│       ├── fig1_voltage_validation.png
│       ├── fig2_surface_temp_validation.png
│       ├── fig3_core_temperature.png
│       ├── fig4_parameter_aging.png
│       ├── fig5_soh_residual.png
│       ├── fig6_drive_thermal.png
│       ├── transformer_test_validation.png
│       ├── ev_us06_transformer_validation.png
│       ├── lstm_training_loss.png
│       ├── transformer_training_loss.png
│       ├── aggressive_multi_temp_visualization.png
│       ├── mixed_multi_temp_visualization.png
│       └── us06_ev_triple_stack.png
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
| 3 | `soh/step3_train_residual_lstm.py` | Trains LSTM (hidden=64) on all 4 batteries for 100 epochs with 80/20 train/val split; saves loss curve | `soh/models/lstm_residual_soh.pth`, `lstm_training_loss.png` |
| 4 | `generation/step4_generate_aging_digital_twin.py` | Per-cycle ECM + EETM calibration → twin dataset; multi-ambient drive visualisations; 288 EV drive-cycle simulations (UDDS/HWFET/US06 × 3 temps × 4 batts × 8 aging) | `augmented_aging_twin_dataset.csv`, `ev_drive_cycle_dataset.csv`, multi-temp plots |
| 5 | `transformer/step5_train_transformer.py` | Trains Transformer Encoder (d=128, 4 heads, 4 layers) for 100 epochs on NASA + EV data with 80/20 split; saves loss curve | `transformer_thermal_core.pth`, `transformer_training_loss.png` |
| 6 | `reports/generate_paper_plots.py` | Generates 8 paper-quality figures including Transformer test validation and EV US06 validation | `results/paper_plots/*.png` |

> **Note:** Step 4 is CPU-bound (per-cycle scipy optimisation + 288 EV simulations). Steps 3 & 5 use GPU. Total pipeline time: ~45–90 minutes depending on GPU.

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

The dashboard provides 9 interactive pages:

| Page | Description |
|------|-------------|
| 📊 **Overview** | Fleet-level SOH KPIs, capacity fade curves, residual learning visualisation |
| ⚡ **ECM Voltage Validation** | Interactive cycle-by-cycle comparison of measured vs simulated voltage with error metrics |
| 🌡️ **Thermal Validation** | Surface temperature validation + core vs surface temperature plots |
| 🔧 **Parameter Aging** | R₀, R₁, R₂ evolution across cycles showing aging-induced resistance growth |
| 📈 **Training Loss Curves** | LSTM and Transformer train/val loss over 100 epochs with architecture details |
| 🚗 **EV Drive Cycles** | Multi-ambient Aggressive/Mixed visualisations, US06 triple-stack, and interactive 288-simulation dataset explorer |
| 🎯 **Transformer Validation** | Pre-generated test validation plots + interactive per-cycle predicted vs actual Tc with error |
| 🧠 **Live Inference** | Run LSTM SOH correction and Transformer core temp prediction on any battery/cycle |
| 📄 **Paper Plots** | View all 8 pre-generated matplotlib figures from Step 6 |

The sidebar shows real-time pipeline status (which steps have been completed, dataset statistics for both NASA twin and EV data, model availability).

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
Training data: NASA twin dataset + 288 EV drive-cycle simulations
```

### EV Drive-Cycle Data Generation (Step 4)
```
Drive Cycles:  UDDS (~1380s), HWFET (~765s), US06 (~600s)
Temperatures:  0°C, 25°C, 45°C
Batteries:     B0005, B0006, B0007, B0018
Aging States:  8 per battery (every 20th cycle)
Total:         4 × 8 × 3 × 3 = 288 unique simulations (~200 hrs synthetic)
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
| `transformer_test_validation.png` | Transformer predicted vs actual core temperature on unseen B0018 data with estimation error |
| `ev_us06_transformer_validation.png` | Transformer core temp prediction on EV US06 drive cycle: current, Tc comparison, error |
| `lstm_training_loss.png` | LSTM train/validation loss convergence over 100 epochs |
| `transformer_training_loss.png` | Transformer train/validation loss convergence over 100 epochs |
| `aggressive_multi_temp_visualization.png` | Aggressive drive profile — current, core temp at 0/20/50°C, voltage |
| `mixed_multi_temp_visualization.png` | Mixed drive profile — current, core temp at 0/20/50°C, voltage |
| `us06_ev_triple_stack.png` | US06 drive cycle — current, core temp at 0/25/45°C, voltage |

---

## References

1. A. Samanta, S. Surya, S. Williamson, et al., "Hybrid Electrical Circuit Model and Deep Learning-Based Core Temperature Estimation of Li-Ion Cells," *IEEE Transactions on Transportation Electrification*, 2022. DOI: [10.1109/TTE.2022.3170359](https://doi.org/10.1109/TTE.2022.3170359)
2. B. Saha and K. Goebel, "Battery Data Set," *NASA Ames Prognostics Data Repository*, 2007.
3. C. R. Birkl et al., "Degradation diagnostics for lithium ion cells," *Journal of Power Sources*, 2017.

---

## License

This project is for academic and research purposes.
