# Task List - Phase 6: Residual Learning & Aging-Aware Digital Twin

- [x] **Phase A: NASA Data**
    - [x] Download and parse NASA `.mat` files extracting Ground Truth $SOH_{true}$.
- [x] **Phase B: Residual LSTM**
    - [x] Compute $SOH_{physics}$ baseline using Coulomb counting.
    - [x] Calculate $Residual = SOH_{true} - SOH_{physics}$.
    - [x] Train LSTM on sequence windows to predict the Residual error.
- [x] **Phase C: Aging-Aware ECM**
    - [x] Extract $\Delta V / \Delta I$ to map Resistance growth per cycle.
    - [x] Upgrade `gpu_batch_simulator.py` so $R_{internal}$ scales with $SOH$.
    - [x] Generate Physics Core Temp labels using UDDS/HWFET profiles.
- [x] **Phase D: Core Temp Transformer**
    - [x] Apply sliding window (stride=1) to generate 100k+ sequences.
    - [x] Train Transformer to predict Core Temp on the sliding windows.
- [x] **Phase E: Report & Plot Generation**
    - [x] Create `reports/generate_paper_plots.py` with all 5 required Figure Groups (Vt/Temp prediction, HWFT, Pulses, UDDS, and SOH tracking).
- [ ] **Phase F: Unified Dashboard**
    - [ ] Build the Streamlit dashboard executing the full Residual+Physics pipeline.
