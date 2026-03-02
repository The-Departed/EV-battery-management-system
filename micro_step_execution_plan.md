# The Micro-Step Execution Plan: Residual Learning & Aging-Aware Digital Twin

You have proposed an exceptionally strong, paper-worthy architecture. By using **Residual Learning**, we stop forcing the AI to learn basic physics from scratch, making the model infinitely more stable. 

Here is the exact step-by-step workflow for you and your friend to execute.

---

## Phase A: Getting the NASA Data (The Ground Truth)
*   **Step A1 (Downloader):** Run `data/download_nasa_aging.py` to get the B0005, B0006, B0007, and B0018 `.mat` files.
*   **Step A2 (Parser):** Parse the MATLAB files to extract the Ground Truth labels: the actual Capacity ($Q_{true}$) and the measured $SOH_{true}$ for every single cycle.

---

## Phase B: The SOH Residual LSTM (The Correction Engine)
*Crucial concept: The LSTM only learns the error of the physics model.*

*   **Step B1 (Physics Baseline):** For every charging cycle, we calculate a rough physics estimate of capacity: $Q_{est} = \int I dt / \Delta SOC$. This gives us $SOH_{physics} = Q_{est} / Q_{rated}$.
*   **Step B2 (Extract Health Indicators):** We extract 5-10 features per cycle $k$: $X_k = [Time_{4.2V}, V_{mean}, T_{avg}, \text{etc.}, SOH_{physics}]$. 
*   **Step B3 (Calculate the Residual Target):** We calculate the exact mistake our physics model made: $Residual_k = SOH_{true} - SOH_{physics}$.
*   **Step B4 (Train the LSTM):** We feed windows of 10 cycles $[X_{k-10}, ..., X_k]$ into the LSTM. The network is trained ONLY to predict the $Residual$. 
*   **Result:** Final SOH is calculated elegantly: $SOH_{final} = SOH_{physics} + Residual_{predicted}$.

---

## Phase C: The Aging-Aware Physics Engine & Digital Twin
*Crucial concept: A degraded battery has higher internal resistance, causing it to run hotter.*

*   **Step C1 (Extract Resistance Growth):** We analyze the NASA discharge pulses. We measure the instantaneous voltage drop: $R_{internal} = \Delta V / \Delta I$. We map this resistance growth to the cycle number/SOH.
*   **Step C2 (Upgrade the ECM):** We rewrite the `gpu_batch_simulator.py` physics equations. Instead of constant parameters, they are now functions of aging: $R_k = R_0 + \alpha(1 - SOH_k)$.
*   **Step C3 (Generate the Digital Twin Dataset):** We run the updated simulator on various realistic drive cycles (UDDS, HWFET, HWFT). Because $R$ increases as the battery ages, $Q = I^2 R$ dictates that the calculated Core Temperature will naturally and physically spike higher for older batteries.

---

## Phase D: The Thermal Transformer (Sliding Windows)
*Crucial concept: We use sliding windows to multiply our dataset size for data-hungry Transformers.*

*   **Step D1 (Create Sliding Windows):** From our thousands of seconds of driving data, we extract 60-second windows with a stride of 1 second. This turns 150 cycles into hundreds of thousands of training sequences, solving the Transformer data starvation problem.
*   **Step D2 (Train the Transformer):** Input = `[V, I, T_surface]`. Target = The Physics-Calculated Core Temp (from Phase C). The Transformer learns the thermal inertia.

---

## Phase E: Output & Paper Figure Generation
*   **Step E1 (Generate the Plots):** We run a comprehensive plotting script (`reports/generate_paper_plots.py`) to generate the exact 5 Figure Groups required for the research paper (Vt prediction, Surface Temp tracking, HWFT profiles, Pulse profiles, UDDS comparisons, and SOH tracking).

---

## Phase F: Live UI Dashboard (The Final Demo)
*   **Step F1 (Dashboard):** Streamlit UI that runs the physics model, queries the LSTM for the SOH Residual, updates the ECM resistance, and queries the Transformer for the real-time Core Temperature warning.
