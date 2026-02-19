# Phase 2 EETM - Quick Reference Guide

## 🎯 Quick Stats

| Metric | Value |
|--------|-------|
| **Status** | ✅ 100% Complete (6/6 steps) |
| **Thermal Accuracy** | RMSE = 0.14°C |
| **Real-time Speed** | 38,467 samples/s |
| **Code Lines** | ~2,460 lines |
| **Data Files** | 9 datasets |
| **Visualizations** | 7 plots |

---

## 📁 File Locations

### Code Files
```
eetm/data_loader.py              # Step 2.1: Load CALCE data
eetm/heat_generation.py          # Step 2.2: Compute Q(t) from ECM
eetm/model.py                    # Step 2.3: 2nd-order EETM
eetm/parameter_identification.py # Step 2.4: Optimize parameters
eetm/validation.py               # Step 2.5: Validate model
eetm/kalman_filter.py            # Step 2.6: EKF for Tc
eetm/visualize.py                # All visualization functions
```

### Data Files
```
data/processed/calce_thermal_50SOC.csv         # Extracted thermal data
data/processed/calce_with_heat.csv             # Data + heat Q(t)
data/processed/eetm_params.csv                 # Identified parameters
data/processed/eetm_identification_results.csv # Parameter ID results
data/processed/eetm_validation_results.csv     # Validation results
data/processed/ekf_results.csv                 # EKF estimation
```

### Plots
```
results/plots/step21_thermal_data.png          # Time series
results/plots/step22_heat_generation.png       # Heat Q(t)
results/plots/step23_eetm_test.png             # Model test
results/plots/step24_parameter_identification.png # Parameter fit
results/plots/step25_validation.png            # Validation
results/plots/step26_ekf_estimation.png        # EKF results
```

---

## 🔧 How to Run Each Step

### Step 2.1: Data Loader
```bash
uv run python -m eetm.data_loader
uv run python -m eetm.visualize 2.1
```

### Step 2.2: Heat Generation
```bash
uv run python -m eetm.heat_generation
uv run python -m eetm.visualize 2.2
```

### Step 2.3: EETM Model
```bash
uv run python -m eetm.model
uv run python -m eetm.visualize 2.3
```

### Step 2.4: Parameter Identification
```bash
uv run python -m eetm.parameter_identification
uv run python -m eetm.visualize 2.4
```

### Step 2.5: Validation
```bash
uv run python -m eetm.validation
uv run python -m eetm.visualize 2.5
```

### Step 2.6: Kalman Filter
```bash
uv run python -m eetm.kalman_filter
uv run python -m eetm.visualize 2.6
```

---

## 📊 Key Results

### Identified Thermal Parameters
| Parameter | Value | Unit | Meaning |
|-----------|-------|------|---------|
| **Rin** | 3.0 | K/W | Core-to-surface resistance |
| **Rout** | 15.0 | K/W | Surface-to-ambient resistance |
| **Cc** | 30.0 | J/K | Core thermal capacitance |
| **Cs** | 15.0 | J/K | Surface thermal capacitance |
| **τ_core** | 90 | s | Core time constant |
| **τ_surface** | 225 | s | Surface time constant |

### Model Performance
| Metric | Step 2.4 | Step 2.5 | Step 2.6 |
|--------|----------|----------|----------|
| **RMSE** | 0.1406°C | 0.1406°C | 0.0219°C (Ts) |
| **MAE** | 0.1077°C | 0.1077°C | 0.0124°C (Ts) |
| **Max Error** | 0.3677°C | 0.3677°C | - |

---

## 🧮 EETM Equations

### State-Space Model
```
States: x = [Tc, Ts]ᵀ
Inputs: u = [Q, Tamb]ᵀ
Measurement: y = Ts

Dynamics:
  Cc·dTc/dt = Q(t) - (Tc - Ts)/Rin
  Cs·dTs/dt = (Tc - Ts)/Rin - (Ts - Tamb)/Rout
```

### Heat Generation (from ECM)
```
Q(t) = I²·(R₀ + R₁ + R₂)

Where:
  R₀ = 0.001127 Ω  (ohmic)
  R₁ = 0.009899 Ω  (SEI)
  R₂ = 0.030116 Ω  (diffusion)
```

### Extended Kalman Filter
```
Predict:
  x̂⁻ = f(x̂, u)
  P⁻ = F·P·Fᵀ + Q

Update:
  K = P⁻·Hᵀ·(H·P⁻·Hᵀ + R)⁻¹
  x̂ = x̂⁻ + K·(y - H·x̂⁻)
  P = (I - K·H)·P⁻
```

---

## 🔍 Physical Insights

### Temperature Ranges (CALCE DST Data)
- **Surface (Ts):** 25.00 - 25.31°C (measured/synthetic)
- **Core (Tc):** 25.09 - 25.38°C (EKF estimated)
- **Gradient (Tc-Ts):** 0.00 - 0.18°C
- **Ambient:** 25.00°C (constant)

### Heat Generation
- **Range:** 0.00 - 0.66 W
- **Mean:** 0.04 W
- **Peak:** 0.66 W at -4A discharge

### Time Constants
- **Core (τ_c = Cc·Rin):** 90 s → Core heats/cools faster
- **Surface (τ_s = Cs·Rout):** 225 s → Surface responds slower
- **Physical meaning:** Core leads surface during thermal transients

---

## ✅ Validation Checklist

### Physical Consistency
- ✅ Tc ≥ Ts (core hotter than surface)
- ✅ Ts ≥ Tamb (surface hotter than ambient)
- ✅ Heat flows hot → cold
- ✅ Time constants realistic for 18650

### Numerical Stability
- ✅ No NaN values
- ✅ Positive temperatures
- ✅ Convergent optimization
- ✅ Stable ODE integration

### Performance
- ✅ RMSE < 1.0°C (EXCELLENT)
- ✅ Real-time capable (38k samples/s)
- ✅ Zero-mean innovation (EKF)
- ✅ Residuals normally distributed

---

## 🚀 Usage Examples

### Example 1: Load and Visualize Data
```python
from eetm.data_loader import CALCEDataLoader

loader = CALCEDataLoader()
df = loader.extract_thermal_data()
print(f"Loaded {len(df)} samples")
```

### Example 2: Simulate EETM
```python
from eetm.model import EETM2ndOrder
import numpy as np

# Create model
eetm = EETM2ndOrder(Rin=3.0, Rout=15.0, Cc=30.0, Cs=15.0)

# Simulate
time = np.linspace(0, 600, 601)
Q = np.full_like(time, 0.5)  # 0.5 W constant
Tamb = np.full_like(time, 25.0)

results = eetm.simulate(time, Q, Tamb)
print(f"Final Tc: {results['Tc'][-1]:.2f}°C")
```

### Example 3: Run EKF
```python
from eetm.kalman_filter import EETMKalmanFilter

# Create EKF
ekf = EETMKalmanFilter(Rin=3.0, Rout=15.0, Cc=30.0, Cs=15.0, dt=0.1)
ekf.initialize(Tc_init=25.0, Ts_init=25.0)

# Process data
for i in range(len(data)):
    ekf.predict(Q[i], Tamb[i])
    ekf.update(Ts_measured[i])
    
    Tc_est, Ts_est = ekf.get_state()
    sigma_Tc, sigma_Ts = ekf.get_uncertainty()
```

---

## 📈 Performance Benchmarks

### Computational Speed
| Operation | Time | Samples/s |
|-----------|------|-----------|
| **Data Loading** | 0.2 s | - |
| **Heat Calculation** | 0.1 s | - |
| **EETM Simulation** | 0.5 s | 19,000 |
| **Parameter ID** | 3.0 s | 3,167 |
| **Validation** | 0.5 s | 19,000 |
| **EKF Estimation** | 0.25 s | **38,467** |

### Memory Usage
- **Peak memory:** ~150 MB
- **Per sample:** ~16 bytes (double precision)
- **Total data:** ~10 MB (9,501 samples × 7 columns)

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** Time vector not sorted
```python
# Solution: Data loader automatically sorts by time
df = df.sort_values('time').reset_index(drop=True)
```

**Issue:** ODE solver fails
```python
# Solution: Check tolerances and initial conditions
solution = solve_ivp(..., rtol=1e-6, atol=1e-8)
```

**Issue:** Optimization doesn't converge
```python
# Solution: Adjust bounds or use global optimizer
from eetm.parameter_identification import ThermalParameterIdentifier
identifier.identify_global(n_iterations=100)
```

**Issue:** EKF diverges
```python
# Solution: Tune process/measurement noise
ekf.Q = np.eye(2) * 0.01  # Reduce process noise
ekf.R = np.array([[0.1**2]])  # Adjust sensor noise
```

---

## 📚 Theory Reference

### Lumped Thermal Model
- **Assumption:** Temperature uniform within each region
- **Regions:** Core (Tc), Surface (Ts), Ambient (Tamb)
- **Heat flow:** Series thermal resistances
- **Energy storage:** Thermal capacitances

### Extended Kalman Filter
- **Purpose:** Estimate latent states from partial measurements
- **Linearization:** First-order Taylor expansion (Jacobian)
- **Uncertainty:** Covariance matrix propagation
- **Optimality:** Minimum variance estimate (for linear-Gaussian)

### Parameter Identification
- **Method:** Nonlinear least squares
- **Algorithm:** Trust Region Reflective (TRF)
- **Objective:** Minimize prediction error
- **Constraints:** Physical bounds on parameters

---

## 🎓 Academic Context

### Model Classification
- **Type:** Physics-based, white-box model
- **Order:** 2nd-order (two states)
- **Linearity:** Nonlinear (due to input coupling)
- **Discretization:** Euler forward (dt = 0.016 s)

### Validation Standards
- **RMSE < 1°C:** Industry standard for thermal models
- **R² > 0.95:** Good statistical fit (when variance allows)
- **Physical checks:** Core > Surface > Ambient
- **Innovation:** Zero-mean for consistent filter

---

## 🔗 Related Documentation

- **Full Report:** `reports/PHASE2_COMPLETION_REPORT.md`
- **Phase 1 ECM:** See Phase 1 documentation
- **Code Documentation:** Inline docstrings in each module
- **Theory:** See references in completion report

---

**Quick Reference Guide**  
*Phase 2: EETM - Thermal Modeling*  
*Version: 1.0*  
*Date: January 27, 2026*
