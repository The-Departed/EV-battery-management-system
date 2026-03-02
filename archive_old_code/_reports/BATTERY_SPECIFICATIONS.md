# Battery Specifications: NASA B0005

**Battery Model:** 18650 Li-ion Cell  
**Dataset:** NASA Prognostics Center of Excellence (PCoE)  
**Battery ID:** B0005  
**Test Period:** April 2008 - July 2008

---

## 1. ELECTRICAL SPECIFICATIONS

### Nominal Parameters
| Parameter | Value | Unit | Notes |
|-----------|-------|------|-------|
| Nominal Capacity | **2.0** | Ah | Rated capacity |
| Initial Capacity | **1.856** | Ah | Measured at cycle 1 |
| Final Capacity | **1.325** | Ah | Measured at cycle 168 |
| Nominal Voltage | **3.7** | V | Typical Li-ion |
| Voltage Range | **2.5 - 4.2** | V | Operating window |
| Cut-off Voltage (Discharge) | **2.5** | V | End of discharge |
| Full Charge Voltage | **4.2** | V | Maximum charge |

### Chemistry
- **Type:** Li-ion (Lithium-ion)
- **Cathode:** LiCoO₂ (Lithium Cobalt Oxide) - typical for 18650 cells
- **Anode:** Graphite
- **Electrolyte:** Organic carbonate-based

### Test Conditions (NASA Dataset)
| Parameter | Value | Unit |
|-----------|-------|------|
| Discharge Current | **2.0** | A |
| Charge Current | **1.5** | A |
| Ambient Temperature | **24** | °C |
| Test Type | Constant Current (CC) discharge |
| Charge Protocol | CC-CV (Constant Current - Constant Voltage) |

---

## 2. PHYSICAL/GEOMETRIC SPECIFICATIONS

### Standard 18650 Cell Dimensions
| Parameter | Value | Unit | Notes |
|-----------|-------|------|-------|
| Diameter | **18.0** | mm | Standard 18650 |
| Height | **65.0** | mm | Standard 18650 |
| Weight | **45 - 48** | g | Typical for 2Ah cell |

### Calculated Geometric Parameters
```python
# Cylindrical geometry
diameter = 0.018  # m
height = 0.065    # m
radius = diameter / 2  # m

# Surface area (excluding ends for simplicity)
A_surface = π * diameter * height  # m²
A_surface ≈ 3.67e-3 m²

# Volume
V_total = π * radius² * height  # m³
V_total ≈ 1.65e-5 m³
```

---

## 3. THERMAL SPECIFICATIONS

### Temperature Characteristics
| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| Ambient Temperature | **24** | °C | Dataset metadata |
| Operating Temp (Min) | **23.2** | °C | Measured during test |
| Operating Temp (Max) | **41.5** | °C | Measured during test |
| Typical Temp Rise | **8 - 18** | °C | During 2A discharge |

### Estimated Thermal Parameters (Literature Values for 18650)
| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Heat Capacity | Cₚ | **900 - 1200** | J/(kg·K) |
| Thermal Conductivity (radial) | k_r | **1.0 - 5.0** | W/(m·K) |
| Thermal Conductivity (axial) | k_z | **20 - 40** | W/(m·K) |
| Convection Coefficient | h | **5 - 20** | W/(m²·K) |
| Density | ρ | **2400 - 2800** | kg/m³ |

---

## 4. ELECTRICAL EQUIVALENT CIRCUIT MODEL (ECM) PARAMETERS

### From Our Analysis

#### OCV-SOC Relationship
```
6th-degree polynomial fit:
OCV(SOC) = a₀ + a₁·SOC + a₂·SOC² + ... + a₆·SOC⁶

Key points:
- OCV @ 0% SOC:   3.25 V
- OCV @ 25% SOC:  3.47 V
- OCV @ 50% SOC:  3.54 V
- OCV @ 75% SOC:  3.71 V
- OCV @ 100% SOC: 4.14 V
```

#### Impedance Data (from metadata)
From impedance measurements in dataset:
- **Re (Electrolyte Resistance):** 0.053 - 0.062 Ω
- **Rct (Charge Transfer Resistance):** 0.16 - 0.21 Ω

### 2-RC Thevenin Model Structure
```
    R0
     ├─── R1 ─── C1 (RC pair 1: SEI layer, fast dynamics)
     │
     └─── R2 ─── C2 (RC pair 2: Diffusion, slow dynamics)
```

**Typical Parameter Ranges (to be identified):**
| Parameter | Typical Range | Unit | Description |
|-----------|--------------|------|-------------|
| R0 | 0.01 - 0.05 | Ω | Ohmic resistance |
| R1 | 0.005 - 0.02 | Ω | SEI resistance |
| C1 | 500 - 5000 | F | SEI capacitance |
| R2 | 0.01 - 0.05 | Ω | Diffusion resistance |
| C2 | 5000 - 50000 | F | Diffusion capacitance |

**Time Constants:**
- τ₁ = R1 × C1 ≈ 2.5 - 100 seconds (fast)
- τ₂ = R2 × C2 ≈ 50 - 2500 seconds (slow)

---

## 5. ENERGY & POWER SPECIFICATIONS

### Energy Capacity
```
Nominal Energy = Nominal Voltage × Capacity
E_nominal = 3.7 V × 2.0 Ah = 7.4 Wh

Actual Energy (initial):
E_actual = 3.7 V × 1.856 Ah ≈ 6.87 Wh
```

### Power Capability
```
Discharge Power (2A):
P_discharge = V_avg × I = 3.5 V × 2.0 A = 7.0 W

Power Density:
P_density = P / mass ≈ 7.0 W / 0.045 kg ≈ 156 W/kg
```

---

## 6. AGING/DEGRADATION DATA

### Capacity Fade
| Metric | Value |
|--------|-------|
| Initial Capacity | 1.856 Ah |
| Final Capacity (168 cycles) | 1.325 Ah |
| Total Fade | 0.531 Ah |
| Percentage Loss | **28.62%** |
| Fade Rate | ~0.0032 Ah/cycle |

### State of Health (SOH)
```
SOH = (Current Capacity / Nominal Capacity) × 100%
SOH_initial = (1.856 / 2.0) × 100% = 92.8%
SOH_final = (1.325 / 2.0) × 100% = 66.2%
```

---

## 7. FOR ECM + EETM MODELING

### Required Parameters for Implementation

**Electrical (ECM):**
- ✅ OCV-SOC curve (fitted, 6th-degree polynomial)
- ✅ Nominal capacity: 1.856 Ah (initial)
- 🔄 R0, R1, C1, R2, C2 (to be identified in Step 1.5)

**Thermal (EETM):**
- 🔄 Core-to-surface thermal resistance: Rc (to be identified)
- 🔄 Surface-to-ambient thermal resistance: Ru (to be identified)
- 🔄 Core heat capacity: Cc (to be estimated)
- 🔄 Surface heat capacity: Cs (to be estimated)
- ✅ Ambient temperature: 24°C
- ✅ Surface area: ~3.67e-3 m²

**Heat Generation:**
- 🔄 Joule heating: Q_joule = I² × R_total
- 🔄 Reaction heat: Q_reaction = I × T × (dOCV/dT)
- 🔄 Total heat: Q = Q_joule + Q_reaction

---

## 8. SUMMARY FOR MODELING

### Battery Configuration
```python
battery_config = {
    # Electrical
    'nominal_capacity': 2.0,      # Ah
    'initial_capacity': 1.856,    # Ah (measured)
    'voltage_min': 2.5,           # V
    'voltage_max': 4.2,           # V
    'nominal_voltage': 3.7,       # V
    
    # Geometric
    'diameter': 0.018,            # m
    'height': 0.065,              # m
    'surface_area': 3.67e-3,      # m²
    'volume': 1.65e-5,            # m³
    'mass': 0.045,                # kg (estimated)
    
    # Thermal
    'T_ambient': 24.0,            # °C
    'specific_heat': 1000,        # J/(kg·K) (estimated)
    'thermal_mass': 45,           # J/K (estimated)
    
    # Test Conditions
    'discharge_current': 2.0,     # A
    'charge_current': 1.5,        # A
}
```

---

## References

1. **NASA PCoE Battery Dataset**  
   https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset

2. **18650 Cell Specifications** (Industry Standard)  
   - Samsung ICR18650-26F
   - Panasonic NCR18650B
   - Similar LiCoO₂ chemistry cells

3. **Literature Values:**  
   - Forgez et al. "Thermal modeling of a cylindrical LiFePO4/graphite lithium-ion battery" (2010)
   - Bernardi et al. "A General Energy Balance for Battery Systems" (1985)

---

**Status:** ✅ Specifications documented  
**Ready for:** Step 1.4 - 2-RC ECM Implementation
