"""
run_pipeline.py — Master execution script for the corrected battery digital twin.
Runs all steps in the correct order:
  1. Download NASA data
  2. Download EPA drive cycles
  3. Parse NASA data (with linear-fade SOH baseline)
  4. Generate digital twin (ECM + EETM, extracts OCV, saves ECM params)
  5. Train SOH residual LSTM (uses ECM-identified R₀)
  6. Train Transformer for core temperature (window_size=1, aligned sampling)
  7. Generate paper plots
"""

import subprocess
import sys
import os
from pathlib import Path

def run_script(script_path, step_name):
    print(f"\n{'='*60}")
    print(f"🚀 STEP: {step_name}")
    print(f"{'='*60}")
    
    # Ensure we use the Python executable from the current environment
    python_exe = sys.executable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"  # force GPU 1 if available
    
    result = subprocess.run(
        [python_exe, script_path],
        cwd=Path(__file__).parent,
        env=env,
        capture_output=False  # stream output directly to console
    )
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: Step '{step_name}' failed with return code {result.returncode}.")
        print("   Pipeline halted. Check the error messages above.")
        sys.exit(1)
    
    print(f"✅ STEP COMPLETE: {step_name}\n")

if __name__ == "__main__":
    print("=" * 60)
    print("🔋 CORRECTED BATTERY DIGITAL TWIN — FULL PIPELINE RUN")
    print("=" * 60)

    # Define steps in the CORRECT order
    steps = [
        ("data/step1_download_nasa.py", "Download NASA Battery Data"),
        ("data/step0_download_epa_drive_cycles.py", "Download EPA Drive Cycles"),
        ("data/step2_parse_and_extract_hic.py", "Parse NASA Data (Linear SOH Baseline)"),
        ("generation/step4_generate_aging_digital_twin.py", "Generate Digital Twin (ECM + EETM + OCV Extraction)"),
        ("soh/step3_train_residual_lstm.py", "Train SOH Residual LSTM (with ECM R₀)"),
        ("transformer/step5_train_transformer.py", "Train Core Temperature Transformer"),
        ("reports/generate_paper_plots.py", "Generate Paper Plots"),
    ]

    for script_path, step_name in steps:
        if not Path(script_path).exists():
            print(f"❌ ERROR: Script '{script_path}' not found.")
            print("   Make sure you are in the project root directory.")
            sys.exit(1)
        run_script(script_path, step_name)

    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETE — ALL AI MODELS TRAINED")
    print("=" * 60)
    print("📊 Check results/paper_plots/ for figures.")
    print("📊 Check soh/models/ and transformer/models/ for trained models.")
    print("📊 Check data/digital_twin_sets/validation_log.csv for quality metrics.")
    print("\n👉 To launch the dashboard:")
    print("   streamlit run run_ui_dashboard.py")
    print("=" * 60)