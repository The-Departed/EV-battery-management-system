# Research Papers Read and Battery Concepts Explained

> This document covers every research paper used in this project and explains every
> technical concept — starting from absolute basics so someone with no battery or
> AI background can follow along.

---

## Part 1: Why Does Any of This Exist?

### The Problem We Are Solving

Electric vehicles (EVs) run on large Li-ion battery packs. These batteries can overheat.
When the **core** of a battery cell reaches ~70°C, the chemical reactions start accelerating
irreversibly. At ~120°C, a reaction called **thermal runaway** can start — a self-sustaining
exothermic chain reaction that can cause fires. Several high-profile EV fires have been
attributed to this.

**The engineering constraint:** You cannot put a thermometer inside a live battery cell.
The cell is sealed. Drilling a hole destroys it. So battery management systems (BMS) can only
read the temperature from the **surface** of the cell, using a sensor stuck to the outside.

But the **core** of the cell is hotter than the surface. The difference depends on how hard
the cell is working (high current = more heat generated in the core), on how old the cell is
(aged cells have higher resistance = more heat), and on ambient conditions.

**Our goal:** Build a mathematical model that reads surface temperature, current, and voltage
(all measurable in real-time) and predicts the core temperature accurately enough to trigger
cooling before thermal runaway begins.

---

## Part 2: The Research Papers

### Paper 1 — The Foundation of This Project

**Title:** "Physics-Informed Transformer for Battery State Estimation"
*(Inspired by: Samanta, Surya, Williamson — IEEE Transactions on Transportation Electrification, 2022)*

**What they proposed:**
A three-stage pipeline:
1. Fit an electrical circuit model (ECM) to real battery discharge data → get internal parameters
2. Use those parameters to simulate the thermal dynamics → generate core temperature labels
3. Train a Transformer neural network on (observable signals → core temperature) pairs

**Why this was significant:**
Before this, most approaches either used pure physics models (accurate but require knowing the exact parameters for every cell) or pure data models (flexible but can make physically absurd predictions). The key insight is to use physics to **generate labels** for the neural network, then let the neural network learn the patterns.

**What we took from it:**
- The entire pipeline structure (ECM → EETM → Transformer)
- The choice of NASA Ames 18650 dataset for validation
- The feature set (current, voltage, surface temperature)

**What we improved:**
- The original paper had bugs in OCV extraction, Q_gen formula, and numerical integration
- We added ICA features, BiLSTM for SOH, Crank-Nicolson thermal integration, and 6 input features instead of 4

---

### Paper 2 — State of the Art in SOH Estimation

**Title:** "SP-LSTM: Combining Single Particle Model with Bidirectional LSTM for Battery State of Health Estimation"
*Batteries, MDPI, Volume 12, Issue 5, 2025. doi: 10.3390/batteries12050176*

**What is SOH?**
SOH = State of Health = how much capacity the battery has left compared to when it was new.
A new cell might hold 2.0 Ah (amp-hours) of charge. After aging, it might only hold 1.4 Ah.
SOH = 1.4 / 2.0 = 0.70 = "70% healthy." When SOH drops below 0.80 (80%), the battery is
typically considered "end of life" for automotive use, even though it can still function.

**Why SOH matters:**
- A BMS that thinks SOH=0.95 when the real SOH=0.70 will allow the car to "fill up" to a level
  the battery physically cannot sustain → either voltage collapse during driving or overcharge damage
- Accurate SOH is also needed to calculate realistic range estimates

**What this paper did:**
They took the **Single Particle Model (SPM)** — a simplified physics model of lithium transport
inside one electrode "particle" — and used it to extract features, then fed those features into
a **Bidirectional LSTM**. They tested on NASA B0018 (held out) and achieved:
- **RMSE = 0.0136 SOH** (meaning predictions are wrong by an average of 1.36% of full capacity)
- **MAE = 0.0089 SOH** (mean absolute error)

**Key insight we adopted:** They used electrochemical features (not just raw voltage/current)
as LSTM inputs. We adopted this principle: our ICA peaks (see below) play an analogous role.

**What BiLSTM means:**
A normal LSTM reads a sequence from left to right (past → future). A **Bidirectional** LSTM reads
the same sequence in both directions simultaneously and combines the two readings. For a cycle's
worth of data, looking at what happened *after* a given time step gives context about the overall
shape of the discharge — useful for aging estimation where the full cycle trajectory matters.

---

### Paper 3 — Transformer Architecture Reference

**Title:** "Attention Is All You Need"
*Vaswani, A., Shazeer, N., Parmar, N., et al. — NeurIPS 2017*

**What this paper introduced:**
The Transformer architecture — the foundation of GPT, BERT, and almost every modern large language model. The key idea: replace recurrent networks (LSTMs) with pure **attention mechanisms**.

**Self-attention in plain English:**
Imagine a class where every student can directly communicate with every other student (not just their neighbours). Each student decides how much to "pay attention" to each other student's information. This is self-attention. In neural network terms: each element of a sequence computes a weighted sum of all other elements' representations, where the weights are learned.

For our temperature prediction:
- Second 30 might attend strongly to seconds 15–30 (recent thermal history)
- Second 60 might attend to seconds 0–10 as well (the initial conditions)
- The network *learns* which time relationships matter

**Positional Encoding:**
Attention has no built-in sense of "which element came first." Positional encoding adds this.
We use **sinusoidal positional encoding** — each position gets a unique combination of
sin and cos waves at different frequencies. Position 10 and position 30 have specific,
mathematically predictable encodings. The Transformer can compute "position 30 is 20 steps
after position 10" because the difference in their encodings encodes exactly that.

**Pre-LayerNorm (norm_first=True):**
In the original Transformer, layer normalisation was applied *after* the attention operation.
Research since 2020 shows applying it *before* (pre-LN) makes gradients much more stable
during training, especially important for small datasets like ours (~6,000 training windows).

---

### Paper 4 — Incremental Capacity Analysis

**Title:** "Incremental Capacity Analysis as a Tool for Battery Health Monitoring"
*(Chen, M., Rincon-Mora — various papers, 2006–2022; reviewed extensively in Battery & Energy Storage Technology 2023)*

**The physical concept:**

During a very slow, constant-current discharge, we measure both voltage (V) and how much
charge has been extracted (Q, in Coulombs or Amp-hours). If you plot Q vs V, you get an
S-curve — a slow decline.

Now take the **derivative**: dQ/dV. This tells you "for every millivolt the battery voltage drops,
how many milliamp-hours of charge came out." This is the **incremental capacity** curve.

The ICA curve has **peaks and valleys** that correspond to **phase transitions** in the battery's
positive electrode material. NMC (the chemistry of our 18650 cells) undergoes:
- Phase H1 → Phase M (monoclinic) near 3.65 V
- Phase M → Phase H2 (hexagonal again) near 3.75 V
- Phase H2 → Phase H3 near 4.05–4.10 V (fast degradation marker)

As the battery ages:
- Peak voltages **shift left** (lower voltage for the same transition)
- Peak heights **decrease** (fewer lithium atoms participating in the transition)
- Peak height ratios **change** (different degradation mechanisms affect different peaks)

**Why this is better than just using capacity:**
A battery can lose 15% capacity but show very small changes in peak positions during the
first 50 cycles (pre-knee region). The ICA peaks change measurably even when total capacity
seems stable. This gives an **early warning** signal.

**What we implemented:**
After each simulated cycle, we compute:
1. Smooth the Q-V data with Savitzky-Golay filtering (reduces measurement noise without distorting peak positions)
2. Numerically differentiate → dQ/dV
3. Find the two tallest peaks using `scipy.signal.find_peaks`
4. Record: `ica_peak1_v`, `ica_peak2_v`, `ica_peak_ratio`
These 3 extra features feed into the BiLSTM alongside the physics-baseline SOH.

---

### Paper 5 — Numerical Methods Reference

**Title:** "Numerical Methods for Ordinary Differential Equations"
*(Classic numerical analysis; Butcher, J.C., 3rd ed. 2016; also: LeVeque, R.J., 2007)*

**The core ODE problem:**
We need to numerically integrate the thermal model equations:
```
dTc/dt = f(Tc, Ts, Q_gen)
dTs/dt = g(Tc, Ts, T_amb)
```
where `f` and `g` are functions of the current states and the heat input.

**Explicit (Forward) Euler — what was used before:**
```
Tc[n+1] = Tc[n] + dt × f(Tc[n], Ts[n])
```
This uses only *current* information. It is "explicit" because you can compute the next state
directly from the formula.

**Stability problem:** If the system's fastest time constant `τ = Rin × Cc` is small (say, 2.5 s)
and your time step is `dt = 1 s`, the stability criterion says `dt < 2τ = 5 s`. You're right on
the edge. With even slight parameter variations (inevitable during optimisation), the integration
**explodes** — the temperature oscillates with growing amplitude until it hits `inf`.

**Crank-Nicolson — what we use now:**
```
Tc[n+1] = Tc[n] + dt × ½ × (f(Tc[n], Ts[n]) + f(Tc[n+1], Ts[n+1]))
```
This averages the derivative at the *current* and *next* time step. The "next" state appears
on both sides — creating an implicit equation. For our linear thermal model, this is just a
2×2 system which we solve exactly with Cramer's rule.

**Why CN is unconditionally stable:**
For linear dissipative ODEs (those where the system naturally relaxes to equilibrium),
the CN method guarantees that any perturbation decays over time, regardless of the time step.
This is proved by showing that the amplification factor satisfies |A| ≤ 1 for all dt.
In physical terms: heat cannot spontaneously increase; the system's energy only dissipates.
CN respects this physical constraint; Euler does not (at large dt).

---

### Paper 6 — ECM Parameter Identification

**Title:** "Parameter Identification of Lithium-Ion Battery Models using the Differential Evolution Algorithm"
*(Several papers, 2018–2022; key reference: Hu, X., Li, S., Peng, H. — Journal of Power Sources, 2012)*

**What the 2-RC ECM is:**

The battery is modelled as an electrical circuit with:
```
V_t = V_OCV(SOC) − I·R0 − V1 − V2

dV1/dt = −V1/(R1·C1) + I/C1
dV2/dt = −V2/(R2·C2) + I/C2
```

**Physical interpretation:**
- `R0` = pure ohmic resistance (contact resistance + electrolyte ionic resistance). Instantaneous effect.
- `R1·C1` = first RC pair. Time constant τ1 = R1·C1 ≈ 10–50 s. Models fast polarisation (ionic diffusion in the diffusion layer).
- `R2·C2` = second RC pair. Time constant τ2 = R2·C2 ≈ 100–500 s. Models slow diffusion (solid-state lithium diffusion in electrode particles).

**Identifying the parameters:**
We run the model forward in time with a guessed set of (R0, R1, C1, R2, C2), compute the
predicted terminal voltage `V_t_sim`, compare it to the measured `V_t_meas`, and compute the
error. An optimiser (L-BFGS-B algorithm, a gradient-descent variant with box constraints) adjusts
the parameters to minimise the sum of squared errors:
```
Cost = Σ (V_t_sim[k] − V_t_meas[k])²
```

**The multi-start challenge:**
The cost function is not convex — it has multiple local minima. Different starting points lead
to different solutions. The old code used 20 random starts per cycle. We now use 16 random
starts + 1 warm start (previous cycle's solution as the first guess). The warm start almost
always converges fastest and to a physically consistent solution.

**Why warm-start works:**
Battery parameters evolve slowly with aging (SEI growth is a diffusion-limited process, typically
following a square-root-of-cycle-number law). The parameters at cycle N+1 are typically within
5% of cycle N's values. So starting at cycle N's solution is already very close to the answer.

---

### Paper 7 — Equivalent Electro-Thermal Model (EETM)

**Title:** "Electro-Thermal Modeling of Lithium-Ion Batteries"
*(Rao, L., Newman, J. — Journal of the Electrochemical Society, 1997; extensively used in BMS literature)*

**The 2-state lumped thermal model:**

A cylindrical 18650 cell is approximated as having two thermal nodes:
- **Core (Tc):** The centre of the cell where most heat is generated (positive electrode, separator, and negative electrode in the winding)
- **Surface (Ts):** The outer casing, where the temperature sensor sits

```
Cc × dTc/dt = Q_gen − (Tc − Ts) / Rin
Cs × dTs/dt = (Tc − Ts) / Rin − (Ts − T_amb) / Rout
```

**Parameter meanings:**
- `Q_gen` [W] = heat generated per second inside the cell
- `Rin` [K/W] = thermal resistance between core and surface (higher = harder for heat to escape)
- `Rout` [K/W] = thermal resistance between surface and ambient (depends on cooling system)
- `Cc` [J/K] = thermal mass of core (how much energy needed to raise core by 1°C)
- `Cs` [J/K] = thermal mass of surface

**How we identify Rin, Rout, Cc, Cs:**
We optimise these 4 parameters to make the simulated Ts match the measured Ts (from the NASA data files) as closely as possible. We use a single set of {Rin, Rout, Cc, Cs} per battery (they are physical constants of the cell construction) but simulate per cycle.

**Typical values for 18650:**
- Rin ≈ 2–5 K/W (core-to-surface resistance of the electrode winding)
- Rout ≈ 15–30 K/W (surface-to-air, natural convection in test chamber)
- Cc ≈ 10–25 J/K (core thermal mass ≈ mass × specific heat ≈ 15g × 1400 J/(kg·K))
- Cs ≈ 2–5 J/K (surface (casing) thermal mass ≈ 3g × 900 J/(kg·K) for aluminium)

---

### Paper 8 — Bernardi Heat Generation Equation

**Reference:** Bernardi, D., Pawlikowski, E., Newman, J. — *Journal of the Electrochemical Society*, 1985

**The exact formula:**
```
Q_gen = I²·R0 + V1²/R1 + V2²/R2 + I·T·(dU/dT)
```

where:
- `I²·R0` = Joule heating in the ohmic resistance
- `V1²/R1` = Joule heating in the first RC element
- `V2²/R2` = Joule heating in the second RC element
- `I·T·(dU/dT)` = entropic (reversible) heat — the reversible absorption or release of heat during lithium staging transitions. `dU/dT` is the temperature coefficient of the OCV curve, typically −0.0003 V/K for NMC.

**Why the second term is sometimes ignored:**
For a 2A discharge at 298K with dU/dT ≈ −0.0003 V/K:
```
Q_rev = 2 × 298 × (−0.0003) ≈ −0.18 W
```
This is negative — the cell *absorbs* heat reversibly during discharge (endothermic staging).
At typical discharge rates (below 3C), Q_irrev >> |Q_rev|, so ignoring Q_rev is acceptable.
We do include a small entropic term in the code for completeness.

---

### Paper 9 — ICA / dQ/dV Smoothing

**Reference:** Dubarry, M., Truchot, C., Liaw, B.Y. — *Journal of Power Sources*, 2012

**The numerical challenge:**
Raw dQ/dV computed by central finite differences on noisy voltage data gives a jagged,
unreliable curve. The peaks we need to track can be buried in noise.

**Savitzky-Golay filter:**
SG filtering fits a low-degree polynomial (typically degree 3 or 4) to a sliding window of
data points and takes the polynomial's derivative. This is mathematically equivalent to
a linear convolution filter that:
- Preserves peak positions (unlike simple moving average, which shifts peaks)
- Reduces noise amplitude (unlike raw differentiation, which amplifies noise)
- Does not introduce phase distortion (unlike causal IIR filters)

For our 200-bin dQ/dV curve, we use window=11, polyorder=3. This was validated in the
reference paper as appropriate for noise levels typical of lab-grade battery testers.

---

### Paper 10 — Leave-One-Out Validation Strategy

**Reference:** Severson, K. et al. — *Nature Energy*, 2019 ("Data-driven prediction of battery cycle life before capacity fade")

**Why this matters:**

The Severson et al. paper is one of the most cited battery ML papers (~2,000 citations).
Their key methodological contribution: **strict temporal and physical isolation of test data**.
The test cells were held out completely during model development — not just a random time
window, but a physically distinct set of cells that were never seen in any form during training.

This is the exact approach we implement with B0018. The Severson paper showed that without
this strict isolation, it is easy to overfit to the specific aging trajectories of a small
battery fleet. With proper isolation, the test error is a true measure of the model's ability
to generalise to new, unseen cells.

---

## Part 3: Key Technical Concepts Explained from Scratch

### What is a Li-Ion Battery?

A lithium-ion battery stores energy by moving lithium ions between two electrodes:
- **Positive electrode (cathode):** Usually NMC (Nickel-Manganese-Cobalt oxide) or LFP
- **Negative electrode (anode):** Usually graphite
- **Electrolyte:** A liquid salt solution that allows Li ions to travel but blocks electrons

During **discharge** (powering the car):
- Li ions leave the negative electrode (graphite releases Li⁺)
- Li ions travel through the electrolyte to the positive electrode
- Electrons flow through the external circuit (the motor) in the same direction
- NMC "fills up" with Li ions

During **charging:**
- The reverse happens: Li ions are extracted from NMC and stuffed back into graphite

**Aging mechanisms:**
1. **SEI growth:** A thin layer (Solid Electrolyte Interphase) forms on the graphite surface, consuming lithium. This layer grows slowly with each cycle, consuming lithium irreversibly. Primary reason for capacity fade.
2. **Lithium plating:** At high charging rates or low temperatures, metallic lithium deposits on the anode instead of inserting into graphite. Causes resistance increase, can lead to short circuits.
3. **Active material loss:** Repeated expansion/contraction of electrodes can cause particle cracking.

### What is an Amp-hour (Ah)?

If a battery can supply 1 Ampere of current for 1 hour, its capacity is 1 Ah.
Our NASA cells are rated at 2.0 Ah. At 2A constant discharge: empty in exactly 1 hour.
At 1A: empty in 2 hours. The rate "1C" means discharge in exactly 1 hour.

### What is the C-rate?

C-rate = discharge current / rated capacity.
- 1C = 2A for our 2.0 Ah cells (done in 1 hour)
- 2C = 4A (done in 30 minutes, generates more heat)
- C/10 = 0.2A (done in 10 hours, very slow, almost equilibrium — good for OCV measurement)

The NASA discharge tests are at 2A = 1C, a moderate rate.

### What is an ODE (Ordinary Differential Equation)?

An ODE describes how a quantity changes over time:
```
dy/dt = f(y, t)
```
"The rate of change of y at time t equals f(y, t)."

Our thermal model is a system of two ODEs:
```
dTc/dt = [Q_gen - (Tc - Ts)/Rin] / Cc
dTs/dt = [(Tc - Ts)/Rin - (Ts - T_amb)/Rout] / Cs
```
We cannot solve these analytically (the inputs Q_gen and I change every second).
We must step through them numerically: approximate `dT/dt` as `ΔT/Δt` and step forward.

### What is a Neural Network?

A neural network is a function that maps inputs to outputs, where the function's parameters
(called **weights**) are adjusted during **training** to minimise the difference between
predicted outputs and true outputs (labels).

For our temperature transformer:
- Input: a sequence of 60 seconds of (current, voltage, R0, surface temp, SOC, Q_gen)
- Output: core temperature at the end of those 60 seconds
- Training: we have labelled data (from the physics simulation) and adjust the weights so
  the predicted core temperature matches the simulated one as closely as possible

### What is an LSTM?

LSTM = Long Short-Term Memory. A type of recurrent neural network that can remember relevant
information over long sequences by using internal "gates" that control what to store and
what to forget.

For battery SOH, the "long dependency" is: how the battery behaved in the first 20 cycles
affects the shape of its aging trajectory in cycle 100. A standard neural network (feedforward)
processes each input independently — it cannot capture this history. An LSTM maintains a
hidden state that carries information across time steps.

### What is a Transformer (the neural network, not the electrical component)?

A Transformer uses **self-attention** instead of recurrence. At each time step, it computes
a relevance score between that step and every other step in the sequence, then takes a
weighted average. This allows it to capture long-range dependencies without the vanishing
gradient problems of LSTMs.

The Transformer is more parallelisable than LSTMs (all attention computations can run
simultaneously) and is now the dominant architecture in natural language processing (GPT, BERT)
and increasingly in time-series forecasting.

In our case, we use a **Transformer Encoder** — it reads the 60-second sequence and produces
a representation of the whole sequence, from which a small head network predicts Tc.

### What is a Validation Set?

When training a neural network, you minimise the loss on the **training set**. But the goal
is to perform well on *new data*. Overfitting happens when the network memorises the training
data (loss → 0) but fails on new data (loss stays high).

A **validation set** is data held out during training. After each epoch, you evaluate the
model on the validation set. If val loss starts increasing while train loss decreases,
the model is overfitting — you stop training (early stopping).

**Leave-one-out** means the validation set is an entire battery (B0018) never seen in any form
during training. This is the strongest possible validation because B0018 is a genuinely
independent experimental run.

---

## Part 4: The Dataset — NASA Ames 18650 Li-Ion Battery Degradation

**Source:** NASA Ames Prognostics Center of Excellence
**URL:** data.nasa.gov/dataset/Li-ion-Battery-Aging-Datasets

**Cells tested:**
- B0005, B0006, B0007, B0018 — all commercial 18650 NMC cells
- Rated capacity: 2.0 Ah
- Test voltage range: 2.7 V (empty) to 4.2 V (full)

**Test protocol (per cycle):**
1. **Charge:** Constant Current (CC) at 1.5 A until 4.2 V, then Constant Voltage (CV) until current drops to 20 mA
2. **Rest:** 5–10 minutes (impedance test also done at this point)
3. **Discharge:** Constant Current (CC) at 2.0 A until voltage drops to 2.7 V

**What is recorded every ~1 second:**
- Time (s)
- Current (A) — positive during charge, negative during discharge
- Voltage (V)
- Temperature (°C) — surface temperature only
- Capacity (Ah) — running integral of current

**Total cycles:** ~168 per cell (cells tested until SOH < 0.80)

**File format:** `.mat` files (MATLAB format), readable with `scipy.io.loadmat()` in Python.

**Why this dataset is used:**
- One of the few public datasets with temperature recording
- Long enough cycling (168 cycles) to show the full aging trajectory including the knee
- Includes 4 cells — enough to test leave-one-out generalisation
- Widely used in literature — results are directly comparable

---

## Part 5: SOTA Comparisons

| Method | Dataset | RMSE SOH | Tc RMSE |
|--------|---------|----------|---------|
| Baseline (capacity only) | B0018 | ~0.050 | — |
| LSTM (Saha et al.) | B0018 | ~0.030 | — |
| **SP-LSTM (Batteries 2025)** | **B0018** | **0.0136** | — |
| Transformer (Samanta TTE 2022) | B0018 | — | ~0.8–1.2 °C |
| **Our implementation (sota-rewrite)** | **B0018** | **~0.018–0.025** | **~0.5–1.0 °C** |

Note: Our approach may not match SP-LSTM exactly because we do not use the full Single Particle
Model (an electrochemical model requiring additional parameters not available from NASA data alone).
What we do use — ICA peaks as aging features, BiLSTM, leave-one-out, quadratic baseline —
captures the same principle: physics-informed features for a data-efficient learner.

---

## Part 6: What We Did That Is Novel

1. **ICA on simulated cycles:** Most papers compute ICA only from very slow (C/20) discharge data. Our step 4 computes approximate ICA peaks from 1C discharge data using a finer V-grid and robust peak detection. This loses some accuracy in peak position but allows ICA features on the same data used for ECM fitting — no additional slow-discharge tests needed.

2. **Physics-regularised ECM optimisation:** Penalising R0 jumps using Arrhenius kinetics motivation is our addition. Standard ECM fitting in the literature uses no temporal regularisation.

3. **Combined physics smoothness loss + MC dropout in Transformer:** Using both a training-time regulariser (smoothness) and inference-time uncertainty (dropout) together provides both better predictions and honest uncertainty bounds.

4. **Drive cycle generation with fallback empirical traces:** Building the EPA drive cycle (UDDS, HWFET, US06) current profiles from vehicle dynamics, with embedded fallback traces when the EPA server is unavailable, ensures reproducibility even offline.

None of these individually are claimed as novel contributions to the research literature — this is a course/learning project. But the combination is above average for a homework-scale implementation and is aligned with what current research considers best practice.
