# EV Battery Digital Twin — Core Temperature Estimation

> A physics + deep learning pipeline that predicts the **internal temperature** of a Li-ion
> battery cell — the temperature you cannot measure with a sensor — so that a car's battery
> management system can protect the battery before it overheats.

---

## Hey, What Does This Project Actually Do?

Imagine you are charging your electric car. Deep inside the battery pack, thousands of small
cylindrical cells (called 18650 cells — the kind in your laptop) are releasing heat as they
discharge. You can stick a temperature sensor on the *outside* of a cell, but the *inside*
can be 10–15°C hotter. By the time the surface gets hot enough to trigger an alarm, the core
may already be in a dangerous state.

This project:
1. Takes real battery aging data from NASA's lab (they tested cells through hundreds of discharge-charge cycles)
2. Uses physics equations to figure out what the **core temperature must have been** inside those cells during each test
3. Trains a neural network (Transformer) to predict core temperature from things you *can* measure in real-time (current, voltage, surface temperature)

Think of it like a car's oil pressure warning: you can't see inside the engine, but sensors tell you something's wrong before damage occurs. This project builds the equivalent for battery temperature.

---

## The Big Picture in 4 Steps

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP A: Get the data                                               │
│  Download NASA battery test records (4 cells, ~168 cycles each)     │
│  Download US EPA driving speed profiles (city, highway, aggressive) │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP B: Physics simulation (the "Digital Twin")                    │
│  Fit an electrical circuit model to each discharge cycle            │
│  → tells us internal resistance, heat generation                    │
│  Feed heat generation into a thermal model                          │
│  → computes core temperature (Tc) at every second                   │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP C: Train two AI models                                        │
│  1. BiLSTM: "How healthy is this battery right now?"  (SOH)        │
│  2. Transformer: "What is the core temperature right now?"  (Tc)    │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP D: Validate and plot                                          │
│  Test on B0018 (a completely separate battery, never seen before)   │
│  Generate publication-quality figures                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Why Should Anyone Care?

**Battery fires in EVs are rare but catastrophic.** Most are caused by thermal runaway — a
chain reaction that starts when the battery's internal temperature exceeds a critical threshold.
Thermal runaway propagates cell-to-cell, and a pack fire can be very difficult to extinguish.

Real-time core temperature prediction gives the BMS (the computer that manages the battery)
a few extra seconds of warning time — enough to:
- Reduce charging/discharging rate
- Activate cooling
- Alert the driver to stop

The Transformer model in this project runs in **milliseconds** on a cheap CPU, making it
deployable in any microcontroller-based BMS.

---

## What's in Each Folder

```
EV-battery-management-system/
│
├── data/
│   ├── step0_download_epa_drive_cycles.py   ← Downloads city/highway/aggressive speed profiles
│   ├── step1_download_nasa.py               ← Downloads the raw NASA battery test files
│   └── step2_parse_and_extract_hic.py       ← Converts raw data to usable tables
│
├── generation/
│   └── step4_generate_aging_digital_twin.py ← THE BIG ONE: physics simulation → Tc labels
│
├── soh/
│   └── step3_train_residual_lstm.py         ← Train the "how healthy is the battery?" model
│
├── transformer/
│   └── step5_train_transformer.py           ← Train the "what is the core temperature?" model
│
├── reports/
│   └── generate_paper_plots.py             ← Makes 8 nice figures for the paper
│
├── run_pipeline.py                          ← Run everything with one command
│
├── CHANGES.md                               ← Detailed list of every bug fixed and why
├── PAPERS_AND_CONCEPTS.md                   ← All research papers + concepts explained
└── HOW_THE_PIPELINE_WORKS.md               ← Step-by-step technical guide
```

---

## How to Run It

### 1. Set up the environment

```bash
# Clone the repo
git clone https://github.com/The-Departed/EV-battery-management-system.git
cd EV-battery-management-system

# If you use conda (recommended):
conda create -n battery-modelling python=3.11
conda activate battery-modelling
pip install -e .

# If you prefer pip + venv:
python -m venv .venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac/Linux
pip install -e .
```

### 2. Run the full pipeline

```bash
python run_pipeline.py
```

That's it. The script runs all 7 steps in the correct order, printing progress as it goes.

### 3. Check the results

After it finishes (typically **1–2.5 hours on a CPU laptop**):
- `results/paper_plots/` — 8 figures showing model performance
- `data/digital_twin_sets/ecm_parameters.csv` — how the battery's resistance changed over its life
- Console logs from step3 and step5 — the final RMSE numbers

---

## What Hardware Do You Need?

- **CPU laptop** — fully supported. No GPU required.
- Python 3.9 or newer
- ~8 GB free disk space (NASA data + generated datasets)
- ~4 GB RAM

The pipeline auto-detects if CUDA is available and uses it if so. On CPU, step4 (the physics
simulation) takes the longest — about 30–90 minutes. The neural network training is fast
because we use small models appropriate for the dataset size.

---

## The Dataset

NASA Ames Prognostics Center tested 4 identical 18650 Li-ion cells until they reached
end-of-life (< 80% original capacity). The cells were cycled at room temperature with a
standard charge-discharge protocol, and current, voltage, and surface temperature were recorded
every ~1 second for each of ~168 cycles.

| Cell | Used for | Cycles |
|------|----------|--------|
| B0005 | Training | ~168 |
| B0006 | Training | ~168 |
| B0007 | Training | ~168 |
| B0018 | Test only (never seen during training) | ~132 |

B0018 is held out completely — it never touches the training loop. This gives us an honest
answer to the question: "does the model generalise to a battery it has never seen?"

---

## The Two AI Models

### Model 1: BiLSTM — State of Health (SOH)

**Input:** Last 10 discharge cycles worth of features (internal resistance, ICA peaks, cycle number, physics-based SOH estimate)

**Output:** SOH — how much capacity remains (1.0 = brand new, 0.75 = typically end-of-life for EVs)

**How it works:** A Bidirectional LSTM reads the 10-cycle history forwards and backwards, then a small dense layer converts that to a SOH prediction. Instead of predicting raw SOH, it predicts the *deviation* from a physics-based estimate (quadratic fade model). This makes the learning problem easier.

**Expected accuracy on B0018:** RMSE ~0.018–0.025 (1.8–2.5% of full capacity)

---

### Model 2: Transformer Encoder — Core Temperature (Tc)

**Input:** Last 60 seconds of (current, voltage, internal resistance, surface temperature, state of charge, heat generation rate)

**Output:** Core temperature at the end of those 60 seconds, in °C

**How it works:** A Transformer Encoder reads the 60-second sequence using self-attention —
it can directly "look" at any pair of time steps to learn temporal relationships. The model
includes physics-informed features (heat generation rate, state of charge) that directly
appear in the governing thermal equations, making it much easier to learn accurate predictions.

**Expected accuracy on B0018:** RMSE ~0.5–1.2°C

**Uncertainty:** The model uses MC Dropout — running the same input through 50 times with
dropout randomly activated — to estimate a 95% confidence interval around each prediction.

---

## Key Improvements Over the Original Code

The original codebase (branch `latest`) had a cascade of physics bugs that corrupted every
single output:

| Problem | Impact | Fix |
|---------|--------|-----|
| OCV built from wrong voltages | R0 stuck at 0.030 Ω for all 168 cycles | Extract OCV from rest/equilibrium data |
| Wrong heat generation formula | Q_gen 3× too small + jumpy | Joule form: I²R0 + V1²/R1 + V2²/R2 |
| Forward Euler thermal ODE | `overflow` errors, wrong Tc | Crank-Nicolson (unconditionally stable) |
| Random validation split | Data leakage → fake 0.46°C RMSE | Leave-one-battery-out on B0018 |
| Stride=1 windows (98% overlap) | Model memorised training data | Stride=60 (non-overlapping windows) |
| Only 4 input features | Missing direct causal drivers | Added SOC + Q_gen (6 features total) |

See [CHANGES.md](CHANGES.md) for every bug explained in plain language.

---

## Documentation Files

This repo comes with three additional documents:

- **[CHANGES.md](CHANGES.md)** — Every bug that was found and fixed, with plain-English explanations of what was wrong, why it mattered, and why the fix works. Starts from basic physics and builds up.

- **[PAPERS_AND_CONCEPTS.md](PAPERS_AND_CONCEPTS.md)** — All research papers referenced, with summaries. Explains every technical concept (OCV, ECM, Crank-Nicolson, BiLSTM, attention, ICA) from scratch. No prior knowledge needed.

- **[HOW_THE_PIPELINE_WORKS.md](HOW_THE_PIPELINE_WORKS.md)** — Full step-by-step walkthrough of the entire pipeline. Includes diagrams, data flow, expected outputs, and FAQs.

- **[SOTA_REWRITE_REPORT.md](SOTA_REWRITE_REPORT.md)** — Technical report comparing our approach to the 2022 base paper and current (2025) state-of-the-art methods.

---

## References

1. A. Samanta, S. Surya, S. Williamson et al. — *Hybrid Electrical Circuit Model and Deep Learning-Based Core Temperature Estimation of Li-Ion Cells*, IEEE TTE, 2022.
2. B. Saha and K. Goebel — *Battery Data Set*, NASA Ames Prognostics Data Repository, 2007.
3. SP-LSTM Paper — *Batteries*, MDPI, Vol. 12, No. 5, 2025. doi:10.3390/batteries12050176
4. A. Vaswani et al. — *Attention Is All You Need*, NeurIPS, 2017.
5. D. Bernardi et al. — *A General Energy Balance for Battery Systems*, J. Electrochem. Soc., 1985.
6. M. Dubarry, B.Y. Liaw et al. — *Incremental Capacity Analysis and Close-to-Equilibrium OCV*, J. Power Sources, 2012.
7. K. Severson et al. — *Data-driven prediction of battery cycle life before capacity fade*, Nature Energy, 2019.
8. US EPA Dynamometer Drive Schedules — epa.gov/vehicle-and-fuel-emissions-testing
