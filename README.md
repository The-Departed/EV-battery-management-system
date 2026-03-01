# EV Battery Core Temperature Estimation

A comprehensive physics-informed machine learning pipeline to estimate the internal core temperature of Li-ion batteries using surface and electrical measurements.

## The Approach

This project combines traditional electrical engineering (ECM/EETM) with modern AI (Transformers) to solve the difficult problem of internal battery temperature prediction:

1. **`ecm/` (Electrical Circuit Model):** Uses a 2-RC Thevenin circuit to model the battery's electrical behavior `V(t), SOC(t)`.
2. **`eetm/` (Electrical-Equivalent Thermal Model):** Uses a 2-state thermal model to calculate heat generation `Q(t)` and predict surface/core temperatures mathematically.
3. **`generation/` (Synthetic Batch Simulator):** Runs thousands of hours of driving scenarios (UDDS, US06, HWFET) through the ECM+EETM physics engine to generate massive datasets with added sensor noise.
4. **`transformer/` (AI Predictor):** Trains a mixed-precision multi-layer Transformer on the physics-generated data to predict `Core Temperature (Tc)` using only outside observable variables `[Current, Voltage, SOC, Surface Temp, Ambient Temp]`.

## Clean Project Structure

```text
EV-battery-management-system/
├── ecm/              # Phase 1: Electrical modeling (OCV, RC parameters)
├── eetm/             # Phase 2: Thermal modeling & heat generation
├── generation/       # Phase 3: GPU-Batched synthetic data generation
├── transformer/      # Phase 4: Transformer model & evaluation
├── data/             # Raw NASA reference data 
├── results/          # Output model weights, datasets, and plots
├── docs/             # Technical specifications and API guides
├── reports/          # Development progress logs from Phases 1-3
├── run_all.py        # Master pipeline execution script
└── README.md
```

## Quick Start
The environment is managed using `uv`. All code is highly optimized to use parallel multiprocessing on CPUs and Mixed Precision (AMP) on GPUs.

```bash
# Initialize uv environment
uv sync

# 1. Generate Synthetic Data (GPU accelerated if available, otherwise CPU map)
python run_all.py --step generate

# 2. Train Transformer Model (GPU Enabled)
python run_all.py --step train --epochs 10

# OR Run Everything at Once:
python run_all.py --step all
```

### Outputs
- **Datasets**: `results/datasets/` (Train/Val/Test CSVs + Distribution plots)
- **Model Results**: `results/model/` (Predictions CSV + Final accuracy figures reproducing paper standards)

