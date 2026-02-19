# 🔋 Battery Modeling Project - File Index

## Quick Access

### 📊 View Results
- **Plots:** `results/plots/` (4 PNG files)
- **Report:** `results/STEP_1.1_REPORT.md`
- **Validation:** `STEP_1.1_VALIDATION.txt`
- **Status:** `PROJECT_STATUS.md`

### 🐍 Run Scripts
```bash
uv run python ecm/data_loader.py       # Load data
uv run python ecm/visualize.py         # Generate plots
uv run python ecm/step1_1_summary.py   # Show summary
./quick_commands.sh                     # Interactive menu
```

### 📁 Data Files
- **Discharge:** `data/processed/B0005_discharge.csv` (50,285 samples)
- **Charge:** `data/processed/B0005_charge.csv` (541,173 samples)

### 📖 Documentation
- `README.md` - Project overview
- `ecm/README_STEP1.1.md` - Step 1.1 details
- `PROJECT_STATUS.md` - Current progress
- `STEP_1.1_VALIDATION.txt` - Validation report

### 💻 Source Code
- `ecm/data_loader.py` - Main data loader
- `ecm/visualize.py` - Plotting functions
- `ecm/step1_1_summary.py` - Validation script

## Project Structure
```
Battery-modelling/
├── data/
│   ├── raw/cleaned_dataset/     # NASA dataset
│   └── processed/               # Processed CSVs
├── ecm/                         # ECM modules
├── results/
│   └── plots/                   # Visualizations
├── scripts/                     # Utility scripts
└── [docs and configs]
```

## Status: ✅ Step 1.1 Complete
**Next:** Step 1.2 - SOC Estimation (awaiting confirmation)
