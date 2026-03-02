# Phase 6 Execution Plan: Datasets, Models, and Steps

Here is the exact, concrete execution plan for how we move from the "Grand Unified Architecture" theory into actual Python code for your project. 

This document explicitly details **what datasets** we are using, **how many**, **what models** we are building, and the exact **step-by-step coding plan**.

---

## 1. The Datasets: What We Need and Why

We need exactly **two distinct datasets** to make this dual-engine system work.

### Dataset A: Real NASA Aging Data (The SOH Dataset)
*   **What it is:** Real-world battery charging data from the NASA Ames Prognostics Center.
*   **How Many:** 4 distinct battery cells (B0005, B0006, B0007, B0018).
*   **Why we need it:** We cannot fake how a battery ages. We need real-world data showing how voltage curves change over 150+ chemical charging cycles until the battery dies. 
*   **How we use it:** We will write a script to extract the "Time it takes to charge" and "Average Voltage" from each charging cycle. This extraction becomes the training data for the LSTM.
*   **Train/Test Split:** Train on Cells #5, #6, and #18. Test blindly on Cell #7.

### Dataset B: Synthetic Dynamic Driving Data (The Core Temp Dataset)
*   **What it is:** The data we generate using your existing `gpu_batch_simulator.py`.
*   **How Many:** We will generate 2,000+ realistic drive cycles (UDDS/US06/HWFET).
*   **The Upgrade:** Currently, your generator only simulates *brand new* batteries. We will update the math so that it randomly assigns an "Age" (SOH from 100% to 70%) to every drive cycle. For older batteries, it will mathematically increase the internal resistance, making the simulated core temperature spike higher and faster.
*   **How we use it:** This massive dataset becomes the training ground for the Transformer.
*   **Train/Test Split:** Train on 1,600 drive cycles. Test blindly on 400 drive cycles.

---

## 2. The Models: What We Build

We are building exactly **two AI Models** in Phase 6.

### Model A: The Long Short-Term Memory Network (LSTM)
*   **Architecture:** A lightweight deep learning model designed for sequential time-series forecasting.
*   **Input:** The daily chemical Health Indicators extracted from Dataset A.
*   **Output:** The current State of Health percentage (`SOH`) and the current Internal Ohmic Resistance (`R0`).
*   **Purpose:** To track the slow, year-over-year degradation of the battery. LSTMs are perfect for this because they possess a "cell state" that remembers trends over hundreds of days.

### Model B: The Multi-Task Physics-Informed Transformer
*   **Architecture:** Your existing PyTorch Transformer, but upgraded with a "Multi-Head" output.
*   **Input:** A 60-second rolling window of live driving sensors `[Voltage, Current, Surface_Temp, Ambient_Temp]`, PLUS the `R0` resistance value passed mathematically from the LSTM.
*   **Output:** The instantaneous prediction for `[Core_Temperature, SOH]`.
*   **Purpose:** To handle the fast, chaotic dynamics of highway driving and predict if the core is overheating right *now*. By injecting the LSTM's `R0` resistance into the Transformer, the Transformer instantly knows *why* the battery is getting so hot.

---

## 3. The Step-by-Step Execution Plan

To build this without getting overwhelmed, we execute in exactly 5 distinct coding steps.

### Step 1: Ingest NASA Data & Extract Features
-   **File:** `data/download_nasa_aging.py`
-   **Task:** Download the raw `.mat` files from the NASA repository. Parse them to separate the Charging cycles from the Discharging cycles.
-   **File:** `soh/health_indicators.py`
-   **Task:** Write the math to extract our Health Indicators (HIs) from the charging curves (e.g., Constant-Current charging duration).

### Step 2: Build & Train the LSTM Prognostic Engine
-   **File:** `soh/lstm_soh_estimator.py`
-   **Task:** Build the PyTorch LSTM network. Feed it the extracted HIs from Step 1. Train it to accurately predict the remaining Capacity and Resistance on the unseen Test Cell #7. Save the trained `.pth` model.

### Step 3: Upgrade the Synthetic Physics Generator
-   **File:** `generation/gpu_batch_simulator.py`
-   **Task:** Add the empirical Arrhenius aging equations. When the code generates a drive cycle, force it to randomly age the battery and output an `SOH` and `R0` column into the resulting CSVs. Run the generator to build our new massive `Dataset B`.

### Step 4: Upgrade & Train the Multi-Task Transformer
-   **File:** `transformer/transformer_temperature_predictor.py`
-   **Task:** Modify the model's final Linear layer to output 2 variables instead of 1. Train the Transformer on the newly generated aged data from Step 3. Verify that the Transformer accurately predicts higher core temperatures for older batteries.

### Step 5: The Grand Unified Dashboard Integration
-   **File:** `dashboard_api.py`
-   **Task:** Load BOTH the trained `.pth` LSTM model and the trained `.pth` Transformer model into the live server memory. Route the live data stream through the LSTM first, grab the output `R0`, and feed it straight into the Transformer.
-   **File:** `app.py`
-   **Task:** Build the visual Streamlit dashboard to show the stable SOH dial gauge (LSTM), and the fast-moving Core Temperature line graph (Transformer). Add the "Simulate 1-Year Time Jump" button to age the battery live on screen.
