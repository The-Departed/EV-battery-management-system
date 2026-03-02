# Grand Unified Battery Architecture: Server Execution Guide

This repository contains the full 5-part AI pipeline for an EV Battery State of Health (SOH) and Core Temperature (Tc) prediction system. It is designed to be executed sequentially, heavily utilizing **GPU (CUDA)** to accelerate dataset simulation and train the deep learning models.

## Pre-Requisites (For the Server)
Make sure the server environment is set up. You will need:
- An NVIDIA GPU available for PyTorch.
- Python 3.8+

Install required libraries:
```bash
pip install torch numpy pandas scipy matplotlib requests streamlit
```

## How to Run the Pipeline
All mathematical equations (2-RC ECM + 2-State EETM, SOH Residual Learning, Internal Resistance approximations) have already been coded and saved securely into the root folders.

You do not need to hunt down individual scripts. **All scripts, data loading, Physics generation, AI training, and model saves are handled by one execution script.**

### 1. Launch the Pipeline
Run the following command from the root folder:
```bash
python run_pipeline.py
```
**What this will do:**
- **Step 1:** Download the raw baseline 18650 Battery cells from the NASA Ames Prognostics center `(B0005.mat, B0006.mat...)`.
- **Step 2:** Parse the `.mat` data. It extracts the Ground Truth Capacity and approximates the internal resistance ($R = \Delta V / \Delta I$). 
- **Step 3:** Train a **PyTorch LSTM** on the GPU to learn the $Residual$ error of the Coulomb SOH counting method.
- **Step 4:** Leverage the GPU via **PyTorch tensors** to simulate over **333 hours** of dynamic driving data across 800 parallel environments where Resistance naturally grows with SOH.
- **Step 5:** Train the **PyTorch Transformer** on the server GPU, using an overlapping 60-second sliding window over those 333 hours of generated data, learning to perfectly predict `Core Temperature (Tc)`.
- **Plot Generation:** Outputs 6 highly specific, paper-ready PNG graphics into `results/paper_plots/`.

*(Note: Ensure your server does not sleep during Step 5, as the transformer will iterate millions of sequences.)*

### 2. View The Results
- Trained Models: `.pth` files will automatically be saved into `soh/models/` and `transformer/models/`.
- Generated Data: 333 hours of simulated Physics Ground Truth is exported as CSVs inside `data/digital_twin_sets/`.
- Presentation Figures: All plots generated are found in `results/paper_plots/`.

### 3. Launch The User Interface
If your server exposes ports (or if you sync the `.pth` files to your local machine), launch the interactive dashboard:
```bash
streamlit run run_ui_dashboard.py
```

*All architectural diagrams and project planning documents have been retained in the `docs_gemini_architectural_plans` directory.*
