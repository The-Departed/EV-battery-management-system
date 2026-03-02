import subprocess
import sys
import os
from pathlib import Path

# Force all child processes to use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def run_script(script_path):
    print(f"\n{'-'*60}")
    print(f"🚀 EXECUTING: {script_path}")
    print(f"{'-'*60}")
    
    # Run the script and stream the output directly to the console
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    result = subprocess.run([sys.executable, script_path], cwd=Path(__file__).parent, env=env)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: Script {script_path} crashed or failed.")
        print("🛑 PIPELINE HALTED.")
        sys.exit(1)
        
    print(f"✅ SUCCESS: {script_path} completed.\n")

if __name__ == "__main__":
    print("========================================================")
    print("🔋 GRAND UNIFIED BATTERY PIPELINE: MASTER EXECUTION RUN 🔋")
    print("========================================================")
    
    # The exact sequential order of the pipeline
    pipeline_scripts = [
        "data/step1_download_nasa.py",
        "data/step2_parse_and_extract_hic.py",
        "soh/step3_train_residual_lstm.py",
        "generation/step4_generate_aging_digital_twin.py",
        "transformer/step5_train_transformer.py",
        "reports/generate_paper_plots.py"
    ]
    
    for script in pipeline_scripts:
        if not os.path.exists(script):
            print(f"❌ ERROR: Cannot find {script}. Make sure you are in the root directory.")
            sys.exit(1)
        run_script(script)
        
    print("\n========================================================")
    print("🎉 PIPELINE COMPLETE! ALL AI MODELS SUCCESSFULLY TRAINED! 🎉")
    print("========================================================")
    print("👉 NEXT STEP: Launch the UI by running this command in your terminal:")
    print("             streamlit run run_ui_dashboard.py")
    print("========================================================")
