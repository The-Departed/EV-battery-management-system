# Battery Digital Twin Project
## Physics-Based Modeling and Machine Learning
### Phases 1–3 Overview

**Author:** [Your Name]  
**Date:** 2026  
**Project:** Physics-Based Battery Thermal Digital Twin

---

# Project Overview Slide

## Goal of the Project
Build a **physics-consistent digital twin of a Li-ion battery** and use it to:
- Estimate internal (core) temperature
- Generate synthetic datasets
- Train ML models (Transformer)

## Global Pipeline

Drive Cycles → ECM → Heat → EETM → Synthetic Data → Transformer

---

# Phase 1: Electrical Modeling (ECM)
## Equivalent Circuit Model (2-RC)

**Objective:**  
Identify a physics-based electrical model of battery voltage dynamics.

**Dataset:**  
NASA Li-ion Battery Dataset (B0005)

---

# Phase 1 – Step 1.1
## Data Loader

### What is it?
Loads raw experimental battery cycling data:
- Current
- Voltage
- Temperature
- Capacity

### Why is it used?
To provide **real experimental signals** for system identification.

### Implementation
- Parse `.mat` files
- Extract discharge cycles
- Convert to CSV

### Results
- 168 discharge cycles
- ~50,000 samples
- Capacity fade: 1.856 Ah → 1.325 Ah

---

# Phase 1 – Step 1.2
## SOC Estimation (Coulomb Counting)

### What is it?
Computes State of Charge over time.

### Why?
SOC is a hidden state required for:
- OCV model
- ECM dynamics

### Formula
SOC(t) = SOC₀ + (1/C) ∫ I(t) dt

### Implementation
Numerical integration (trapezoidal rule)

### Results
- SOC range: 0–100%
- Mean SOC ≈ 45%

---

# Phase 1 – Step 1.3
## OCV–SOC Model

### What is it?
Polynomial mapping between SOC and open-circuit voltage.

### Why?
Needed to separate:
- equilibrium voltage
- dynamic voltage drops

### Formula
OCV(SOC) = a₀ + a₁·SOC + a₂·SOC² + ... + a₆·SOC⁶

### Results
- RMSE ≈ 25 mV
- OCV range: 3.15–4.19 V

---

# Phase 1 – Step 1.4
## ECM 2-RC Model

### What is it?
Thevenin equivalent circuit:
- R0 (ohmic)
- Two RC pairs (fast + slow dynamics)

### Why?
Captures transient and steady voltage behavior.

### Equations

V = OCV(SOC) - V₁ - V₂ - I·R₀  

dV₁/dt = -V₁/(R₁C₁) + I/C₁  
dV₂/dt = -V₂/(R₂C₂) + I/C₂  
dSOC/dt = I / (C·3600)

### Results
- Realistic voltage curves
- Time constants: 20 s and 600 s

---

# Phase 1 – Step 1.5
## Parameter Identification

### What is it?
Estimate R0, R1, R2, C1, C2 from data.

### Why?
To make ECM physically meaningful.

### Method
Nonlinear least squares:

min || V_measured - V_model ||

### Results (Cycle 1)
- R0 = 1.1 mΩ  
- R1 = 9.9 mΩ  
- R2 = 30 mΩ  
- RMSE ≈ 124 mV

---

# Phase 1 – Step 1.6
## Validation Across Lifetime

### What is it?
Test ECM on aged cycles.

### Why?
Check generalization.

### Results
| Cycle | RMSE |
|------|------|
| 1 | 123 mV |
| 50 | 119 mV |
| 150 | 277 mV |

### Insight
Fixed ECM degrades with aging → need adaptive parameters.

---

# Phase 2A: Thermal Modeling (EETM)
## 2-State Electro-Thermal Model

**Objective:**  
Model internal battery temperature dynamics.

---

# Phase 2 – Step 2.1
## Thermal Data Loader

### What is it?
Loads thermal experiment data.

### Why?
To tune thermal parameters.

### Reality
Public CALCE data had **no real temperature** → synthetic Ts used.

### Important
This phase = **simulation validation only**

---

# Phase 2 – Step 2.2
## Heat Generation

### What is it?
Computes heat from ECM.

### Why?
Thermal model input.

### Formula
Q(t) = I²·(R₀ + R₁ + R₂)

### Results
- Peak heat ≈ 0.66 W

---

# Phase 2 – Step 2.3
## EETM Model

### What is it?
2-state thermal system:
- Core temperature Tc
- Surface temperature Ts

### Equations

Cc·dTc/dt = Q - (Tc - Ts)/Rin  
Cs·dTs/dt = (Tc - Ts)/Rin - (Ts - Tamb)/Rout

---

# Phase 2 – Step 2.4
## Thermal Parameter Identification

### What is it?
Estimate:
- Rin, Rout
- Cc, Cs

### Method
Least squares on Ts.

### Results (synthetic)
- Rin = 3 K/W  
- Rout = 15 K/W  
- RMSE ≈ 0.14 °C

---

# Phase 2 – Step 2.5
## Validation

### What is it?
Compare predicted Ts with data.

### Results
- Sub-degree accuracy
- Physically consistent gradients

### Important
Not experimental yet → simulation-only.

---

# Phase 2 – Step 2.6
## Extended Kalman Filter

### What is it?
Estimate hidden Tc using Ts only.

### State
x = [Tc, Ts]

### EKF Equations
xₖ₊₁ = f(xₖ, uₖ)  
yₖ = Hxₖ = Ts

### Results
- Real-time estimation
- 38k samples/s
- Tc uncertainty tracked

---

# Phase 3: Synthetic Data Generator
## Digital Twin Data Factory

**Objective:**  
Generate ML-ready datasets using physics.

---

# Phase 3 – Step 3.1
## Drive Cycle Loader

### What is it?
Loads realistic current profiles:
- UDDS
- US06
- HWFET

### Why?
Simulate real driving conditions.

### Results
- RMS current: 1–3 A
- Peaks: ~6 A

---

# Phase 3 – Step 3.2
## Ambient Temperature Profiles

### What is it?
Generate Tamb(t):
- Constant
- Step
- Ramp
- Sinusoidal

### Why?
Expose ML to thermal diversity.

---

# Phase 3 – Step 3.3
## Batch Physics Simulator

### What is it?
Runs:
ECM → EETM

### Outputs
- I, V, SOC
- Q
- Ts, Tc, Tamb

### Results
- 48 scenarios
- Physically consistent dynamics

---

# Phase 3 – Step 3.4
## Sensor Noise Injection

### What is it?
Simulates real BMS sensors.

### Noise Types
- Gaussian
- Bias
- Outliers
- Quantization

### Why?
Prevent ML overfitting to perfect data.

---

# Phase 3 – Step 3.5
## Dataset Builder

### What is it?
Combines all scenarios.

### Features
Inputs:
[I, V, SOC, Ts, Tamb, Q]

Target:
[Tc]

### Results
- 43,744 samples
- Train/Val/Test split

---

# Phase 3 – Step 3.6
## Validation & Sanity Checks

### What is it?
Physics consistency verification.

### Checks
- SOC monotonic
- Tc–Ts coupling
- V–SOC correlation

### Result
Dataset is **ML-ready**.

---

# End of Phase 3

You now have:

- A digital twin
- A synthetic world
- Perfect labels
- Infinite training data

---

# Next: Phase 4 (Preview)

## Transformer Model

Input:
[I, V, SOC, Ts, Tamb, Q]

Output:
Tc

Baseline:
EKF vs Transformer

Goal:
Learn thermal dynamics from data.