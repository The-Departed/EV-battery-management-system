# Battery Core Temperature Estimation

PhD-level project for core temperature estimation of Li-ion batteries using physics-based modeling and machine learning.

## Project Architecture

```
Real Experimental Data
        ↓
ECM (2-RC Thevenin Model)
        ↓
Heat Generation Model (Q)
        ↓
EETM (2nd-order thermal model)
        ↓
Synthetic Dataset Generation
        ↓
Transformer Model
        ↓
Core Temperature Prediction (Tc)
```

## Phase 1: ECM System Identification

Current phase focuses on:
- Loading NASA Li-ion Battery Dataset
- Implementing 2-RC Thevenin ECM
- Parameter identification (R0, R1, C1, R2, C2)
- OCV-SOC curve fitting
- Model validation

## Setup

```bash
# Initialize uv environment
uv sync

# Download NASA dataset
./scripts/download_data.sh
```

## Project Structure

- `data/` - Raw, processed, and synthetic datasets
- `ecm/` - Electrical Circuit Model implementation
- `eetm/` - Electrical-Equivalent Thermal Model
- `generation/` - Synthetic data generation
- `transformer/` - Transformer model for temperature prediction
- `results/` - Plots and metrics

## Current Status

✅ Project structure created
✅ Phase 1: ECM System Identification (Completed)
✅ Phase 2: Heat Generation Model (Completed)
✅ Phase 3: EETM Thermal Model (Completed)
🔄 Phase 4: Transformer Temperature Prediction (In Progress)

## Quick Start (Phase 4 Automation)

To run the full pipeline (Data Generation -> Model Training):

```bash
# 1. Generate Synthetic Data (100+ Hours, 400+ Scenarios)
python run_all.py --step generate

# 2. Train Transformer Model (GPU Enabled)
python run_all.py --step train --epochs 10

# OR Run Everything at Once:
python run_all.py --step all
```

### Outputs
- **Datasets**: `results/datasets/` (Train/Val/Test CSVs + Visualization Plots)
- **Model Results**: `results/model/` (Predictions CSV + Paper-Replication Plots)
