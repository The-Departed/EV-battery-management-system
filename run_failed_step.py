#!/usr/bin/env python
"""
run_failed_step.py — Run only the failed step (Generate Paper Plots)
without rerunning the entire pipeline.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("🎯 Running Failed Step: Generate Paper Plots")
    print("=" * 60)
    
    script_path = "reports/generate_paper_plots.py"
    step_name = "Generate Paper Plots"
    
    if not Path(script_path).exists():
        print(f"❌ ERROR: Script '{script_path}' not found.")
        sys.exit(1)
    
    python_exe = sys.executable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    
    print(f"\n🚀 Running: {script_path}\n")
    
    result = subprocess.run(
        [python_exe, script_path],
        cwd=Path(__file__).parent,
        env=env,
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: Step '{step_name}' failed with return code {result.returncode}.")
        sys.exit(1)
    
    print(f"\n✅ STEP COMPLETE: {step_name}\n")
    print("=" * 60)
    print("🎉 Paper plots generated successfully!")
    print("=" * 60)
    print("📊 Check results/paper_plots/ for figures.")

if __name__ == "__main__":
    main()
