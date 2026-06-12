"""
run_pipeline.py — Master execution script for the SOTA battery digital twin.
Runs all steps in the correct order:
  1. Download NASA data
  2. Download EPA drive cycles
  3. Parse NASA data (quadratic SOH baseline + OCV rest-point extraction)
  4. Generate digital twin (ECM + EETM + ICA features — all bugs fixed)
  5. Train SOH residual BiLSTM (leave-one-out, ICA features)
  6. Train Core Temperature Transformer (stride-60, SOC+Q_gen features)
  7. Generate paper plots

SOTA branch changes vs 'latest':
  - Step 2: quadratic SOH baseline, OCV rest extraction, timestamp dedup
  - Step 4: OCV from rest/charge data, R0≥0.05, Crank-Nicolson thermal,
            Joule Q_gen, C1/C2 saved, warm-start ECM, ICA peak tracking
  - Step 3: BiLSTM(2-layer), leave-one-out on B0018, ICA features, early stop
  - Step 5: stride=60 (non-overlapping), 6 features inc. SOC+Q_gen,
            physics smoothness loss, num_workers=0 for Windows
"""

import subprocess
import sys
import os
import time
from pathlib import Path


def run_script(script_path: str, step_name: str) -> None:
    print(f"\n{'='*65}")
    print(f"🚀 {step_name}")
    print(f"{'='*65}")
    t0 = time.time()

    python_exe = sys.executable
    env = os.environ.copy()
    # CPU laptop: no GPU expected. Suppress CUDA warnings.
    env["CUDA_VISIBLE_DEVICES"] = ""

    result = subprocess.run(
        [python_exe, script_path],
        cwd=Path(__file__).parent,
        env=env,
        capture_output=False,
    )

    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌ Step '{step_name}' failed (rc={result.returncode}) after {elapsed:.1f}s.")
        print("   Pipeline halted. Fix the error above, then re-run.")
        sys.exit(1)

    print(f"✅ {step_name} — done in {elapsed:.1f}s")


if __name__ == "__main__":
    print("=" * 65)
    print("🔋 EV BATTERY DIGITAL TWIN — SOTA PIPELINE  (sota-rewrite branch)")
    print("=" * 65)

    steps = [
        ("data/step1_download_nasa.py",
         "Step 1 — Download NASA Battery Data"),
        ("data/step0_download_epa_drive_cycles.py",
         "Step 0 — Download EPA Drive Cycles"),
        ("data/step2_parse_and_extract_hic.py",
         "Step 2 — Parse NASA Data (quadratic SOH + OCV rest points)"),
        ("generation/step4_generate_aging_digital_twin.py",
         "Step 4 — Generate Digital Twin (ECM + EETM + ICA)"),
        ("soh/step3_train_residual_lstm.py",
         "Step 3 — Train SOH Residual BiLSTM"),
        ("transformer/step5_train_transformer.py",
         "Step 5 — Train Core Temperature Transformer"),
        ("reports/generate_paper_plots.py",
         "Step 6 — Generate Paper Plots"),
    ]

    for script_path, step_name in steps:
        if not Path(script_path).exists():
            print(f"❌ Script not found: {script_path}")
            sys.exit(1)
        run_script(script_path, step_name)

    print("\n" + "=" * 65)
    print("🎉 PIPELINE COMPLETE")
    print("=" * 65)
    print("  Figures  →  results/paper_plots/")
    print("  SOH model  →  soh/models/lstm_residual_soh.pth")
    print("  Thermal model  →  transformer/models/transformer_thermal_core.pth")
    print("  Validation  →  data/digital_twin_sets/validation_log.csv")
    print("\n  Dashboard:  streamlit run run_ui_dashboard.py")
    print("=" * 65)
