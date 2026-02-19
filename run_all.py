import os
import subprocess
import sys
import argparse
from pathlib import Path

def run_command(command, description):
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*80}\n")
    
    # Use unbuffered output to see progress in real-time
    result = subprocess.run(command, shell=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    
    if result.returncode != 0:
        print(f"❌ Error in {description}")
        sys.exit(1)
    else:
        print(f"✓ {description} completed successfully")

def main():
    parser = argparse.ArgumentParser(description="EV Battery BMS - Master Pipeline")
    parser.add_argument("--step", type=str, choices=['all', 'generate', 'train'], default='all', help="Which step to run")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.absolute()
    
    print(f"🚀 Starting EV Battery BMS Pipeline (Step: {args.step.upper()})")

    # 1. Data Generation
    if args.step in ['all', 'generate']:
        print("\nStep 1: Data Generation")
        print("-----------------------")
        # Ensure generation module exists/works
        run_command(f"python -m generation.dataset_builder", "Data Generation (Dataset Builder)")
    
    # 2. Train Transformer Model
    if args.step in ['all', 'train']:
        print("\nStep 2: Model Training")
        print("---------------------")
        # Check if data exists
        if not (base_dir / "results/datasets/battery_thermal_v1_train.csv").exists() and args.step == 'train':
            print("❌ Training data not found! Please run with --step generate first.")
            sys.exit(1)
            
        run_command(f"python transformer/transformer_temperature_predictor.py --epochs {args.epochs}", "Transformer Training & Evaluation") 
    
    print(f"\n\n✓ PIPELINE COMPLETE ({args.step.upper()})")
    if args.step in ['all', 'train']:
        print("Check results in results/model/ for plots and predictions.")

if __name__ == "__main__":
    main()
