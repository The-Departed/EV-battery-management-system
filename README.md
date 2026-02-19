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
🔄 Phase 1, Step 1.1: Data Loader (in progress)
