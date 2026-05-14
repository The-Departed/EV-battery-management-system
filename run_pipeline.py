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
import io
from contextlib import redirect_stdout, redirect_stderr
import signal

def run_script(script_path, step_name, output_buffer):
    print(f"\n{'='*60}")
    output_buffer.write(f"\n{'='*60}\n")
    print(f"🚀 STEP: {step_name}")
    output_buffer.write(f"🚀 STEP: {step_name}\n")
    print(f"{'='*60}")
    output_buffer.write(f"{'='*60}\n")

    python_exe = sys.executable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"

    process = subprocess.Popen(
        [python_exe, script_path],
        cwd=Path(__file__).parent,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='')
        output_buffer.write(line)
    process.stdout.close()
    returncode = process.wait()

    if returncode != 0:
        msg = f"\n❌ ERROR: Step '{step_name}' failed with return code {returncode}.\n   Pipeline halted. Check the error messages above."
        print(msg)
        output_buffer.write(msg + "\n")
        sys.exit(1)

    print(f"✅ STEP COMPLETE: {step_name}\n")
    output_buffer.write(f"✅ STEP COMPLETE: {step_name}\n\n")

def main(output_buffer):
    print("=" * 60)
    output_buffer.write("=" * 60 + "\n")
    print("🔋 CORRECTED BATTERY DIGITAL TWIN — FULL PIPELINE RUN")
    output_buffer.write("🔋 CORRECTED BATTERY DIGITAL TWIN — FULL PIPELINE RUN\n")
    print("=" * 60)
    output_buffer.write("=" * 60 + "\n")

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
            msg = f"❌ ERROR: Script '{script_path}' not found.\n   Make sure you are in the project root directory."
            print(msg)
            output_buffer.write(msg + "\n")
            sys.exit(1)
        run_script(script_path, step_name, output_buffer)

    print("\n" + "=" * 60)
    output_buffer.write("\n" + "=" * 60 + "\n")
    print("🎉 PIPELINE COMPLETE — ALL AI MODELS TRAINED")
    output_buffer.write("🎉 PIPELINE COMPLETE — ALL AI MODELS TRAINED\n")
    print("=" * 60)
    output_buffer.write("=" * 60 + "\n")
    print("📊 Check results/paper_plots/ for figures.")
    output_buffer.write("📊 Check results/paper_plots/ for figures.\n")
    print("📊 Check soh/models/ and transformer/models/ for trained models.")
    output_buffer.write("📊 Check soh/models/ and transformer/models/ for trained models.\n")
    print("📊 Check data/digital_twin_sets/validation_log.csv for quality metrics.")
    output_buffer.write("📊 Check data/digital_twin_sets/validation_log.csv for quality metrics.\n")
    print("\n👉 To launch the dashboard:")
    output_buffer.write("\n👉 To launch the dashboard:\n")
    print("   streamlit run run_ui_dashboard.py")
    output_buffer.write("   streamlit run run_ui_dashboard.py\n")
    print("=" * 60)
    output_buffer.write("=" * 60 + "\n")


if __name__ == "__main__":
    md_filename = "pipeline_output.md"
    f = io.StringIO()
    interrupted = [False]

    def handle_interrupt(signum, frame):
        interrupted[0] = True
        output = f.getvalue()
        with open(md_filename, "w") as md_file:
            md_file.write("# Pipeline Output (Interrupted)\n\n")
            md_file.write("```")
            md_file.write("\n" + output + "\n")
            md_file.write("```")
        print("\n[!] Pipeline interrupted. Partial output written to pipeline_output.md.")
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_interrupt)

    try:
        main(f)
    finally:
        output = f.getvalue()
        with open(md_filename, "w") as md_file:
            md_file.write("# Pipeline Output\n\n")
            md_file.write("```")
            md_file.write("\n" + output + "\n")
            md_file.write("```")
        print(output)