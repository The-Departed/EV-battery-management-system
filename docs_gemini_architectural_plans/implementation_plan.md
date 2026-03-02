# Implementation Plan - Phase 6: Unified SOH & Core Temperature Transformer

## Goal Description
The user challenged the simulation accuracy: we cannot just "guess" how SOH degrades. We must use real-world, empirical battery physics equations (just like we did for ECM and EETM) to mathematically dictate exactly *how* and *why* Capacity fades and Resistance increases over time and cycles.

This plan details the integration of established Electrochemical Degradation Equations into our synthetic data generator, ensuring the Transformer learns true aging physics.

## The Physics of Battery Degradation (The Mathematical Engine)

Battery degradation is fundamentally driven by two main physical processes:
1.  **Solid Electrolyte Interphase (SEI) Layer Growth:** A parasitic chemical reaction on the negative electrode that consumes Lithium ions (reducing Capacity) and builds a thick, insulating crust (increasing Resistance).
2.  **Lithium Plating & Active Material Loss:** Mechanical stress (expansion/contraction) physically breaking the electrode.

### Equation 1: Capacity Fade ($C_{max}$)
Capacity loss is empirically modeled using a power-law relationship involving the number of charge/discharge equivalent full cycles ($EFC$), temperature ($T$), and a pre-exponential factor ($A$):

$Capacity\_Loss\_Percentage = A \cdot \exp\left(-\frac{E_a}{R \cdot T}\right) \cdot (EFC)^z$

Where:
-   $E_a$: Activation energy (the energy required for the parasitic reaction to occur).
-   $R$: Universal gas constant.
-   $T$: Absolute Temperature (Kelvin).
-   $z$: Power law factor (usually close to $0.5$, indicating degradation follows $t^{1/2}$ kinetics due to diffusion-limited SEI growth).
-   $EFC$: Equivalent Full Cycles (how much the battery has been driven).

*In Python, we will simulate $EFC$ increasing from 0 to 2000. As $EFC$ increases, $C_{max}$ mathematically drops from $2.0Ah$ down to $1.6Ah$ (80% SOH).*

### Equation 2: Internal Resistance Growth ($R_{0}$, $R_{1}$)
Resistance growth follows similar Arrhenius kinetics. As the insulating SEI crust thickens, it becomes harder for ions to pass through:

$Resistance\_Growth\_Percentage = B \cdot \exp\left(-\frac{E_{a,R}}{R \cdot T}\right) \cdot (EFC)^y$

Where $y$ is typically between $0.5$ and $1.0$. 

*In Python, as $EFC$ approaches 2000 (End of Life), $R_{0}$ (Ohmic Resistance) and $R_{1}$ (Polarization Resistance) will mathematically double ($200\%$ of original value).*

### Equation 3: SOH Definition
SOH is not a standalone physical property; it is a calculated ratio. We define it relative to the Capacity Fade:

$SOH_{capacity} = \frac{C_{current}}{C_{new}} \times 100\%$

However, because Resistance is actually the primary driver of *Thermal Danger*, our synthetic generator will output a combined `SOH` metric: as $SOH \rightarrow 80\%$, $C_{current} \rightarrow 0.8 \cdot C_{new}$ AND $R_{0,current} \rightarrow 2.0 \cdot R_{0,new}$.

## Proposed Changes

### Step 1: Upgrade the Physics Engine
#### [MODIFY] `generation/gpu_batch_simulator.py`
- Create a new method: `apply_aging_physics(cycle_number, E_a, z, T_avg)`.
- Update `GPUSimConfig` dynamically: for each generated batch, randomly sample an `EFC` (Equivalent Full Cycle) between `[0, 2000]`.
- Calculate the exact `C_max` and `R0`/`R1` values for that specific `EFC` using the empirical equations above.
- The `SOH` column in the CSV will simply be the ratio: `C_current / C_new`.

### Step 2: Implement the Multi-Task AI
#### [MODIFY] `transformer/transformer_temperature_predictor.py`
- Modify `BatteryTransformerModel` to output `[Core_Temp, SOH]`.
- Update the loss function to handle multi-target regression.

### Step 3: Upgrade the UI
#### [MODIFY] `app.py` & `dashboard_api.py`
- Feed live data into the new multi-head model and display live SOH percentages on a dial gauge, alongside the Core Temperature chart.

## Verification Plan
1.  **Physics Check:** Verify that a synthetic battery set to `EFC=1500` outputs a lower Capacity and higher Core Temperature than a battery at `EFC=50` for the same drive cycle.
2.  **Model Convergence:** Train the Transformer and verify it predicts SOH accurately on a holdout test dataset.
