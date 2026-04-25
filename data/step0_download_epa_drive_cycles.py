"""
download_epa_drive_cycles.py
----------------------------
Downloads UDDS, HWFET, and US06 speed traces (10 Hz, m/s)
from the CIRCLES-ENERGY-MODELS GitHub repository and saves them
as CSV files in data/drive_cycles/.
"""

import os
import pandas as pd
from pathlib import Path

# ---------- Config ----------
DRIVE_CYCLES = ["UDDS", "HWFET", "US06"]
BASE_URL = "https://raw.githubusercontent.com/CIRCLES-consortium/CIRCLES-ENERGY-MODELS/main"
OUTPUT_DIR = Path("data/drive_cycles")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for cycle in DRIVE_CYCLES:
    url = f"{BASE_URL}/{cycle}_drivecycle_speeds_10Hz.csv"
    print(f"Downloading {cycle} from {url} ...")

    # The raw files have one header line then space-separated floats
    # We read everything as a single column, then split.
    raw = pd.read_csv(url, header=None, skiprows=1, sep='\s+')
    speed = raw.values.flatten().astype(float)

    # Create time column: 10 Hz -> dt = 0.1 s
    time = [i * 0.1 for i in range(len(speed))]

    # Build a clean DataFrame
    df = pd.DataFrame({
        "time_s": time,
        "speed_mps": speed
    })

    # Save as a normal CSV (comma-separated, with header)
    output_path = OUTPUT_DIR / f"{cycle}_epa_speed.csv"
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} rows to {output_path}")

print("\nDone. All three drive cycles are ready in data/drive_cycles/")