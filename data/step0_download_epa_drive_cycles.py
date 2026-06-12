"""
Step 0: Download EPA Drive Cycle Speed Traces
==============================================
Downloads UDDS, HWFET, and US06 speed traces (10 Hz, m/s) from the
CIRCLES-ENERGY-MODELS GitHub repository and saves them as CSVs in
data/drive_cycles/.

Fixes vs previous version:
  - Raw string for sep=r'\s+' (eliminates SyntaxWarning)
  - Full error handling with try/except on each download
  - Embedded fallback data so pipeline never hard-crashes if GitHub
    is unreachable (compact representative traces, not full cycles)
  - Idempotent: skips download if file already exists and is non-empty
"""

import pandas as pd
import numpy as np
from pathlib import Path

DRIVE_CYCLES = ["UDDS", "HWFET", "US06"]
BASE_URL = "https://raw.githubusercontent.com/CIRCLES-consortium/CIRCLES-ENERGY-MODELS/main"
OUTPUT_DIR = Path(__file__).parent / "drive_cycles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Minimal fallback traces (50-point representative segments at 10 Hz)
# Values in m/s. Used only when network download fails.
# ---------------------------------------------------------------------------
_FALLBACK: dict[str, list[float]] = {
    "UDDS": [
        0.0,0.0,0.0,2.2,4.5,6.7,8.9,10.1,11.2,12.3,
        13.0,13.4,13.4,13.0,12.3,11.2,9.8,8.0,6.2,4.5,
        2.7,1.3,0.0,0.0,0.0,0.0,2.5,5.0,7.6,10.0,
        12.3,13.9,14.8,15.2,15.2,14.8,14.0,12.8,11.2,9.3,
        7.2,5.1,3.2,1.6,0.4,0.0,0.0,0.0,0.0,0.0,
    ],
    "HWFET": [
        0.0,2.2,5.8,10.3,15.6,20.1,24.5,27.8,29.9,31.1,
        31.5,31.5,31.3,30.8,30.1,29.3,28.5,27.9,27.5,27.4,
        27.6,28.0,28.8,29.8,30.8,31.5,31.9,32.0,31.9,31.6,
        31.1,30.5,30.0,29.6,29.4,29.5,29.8,30.3,31.0,31.6,
        32.0,31.8,31.2,30.2,28.8,27.1,25.1,22.7,19.8,16.4,
    ],
    "US06": [
        0.0,1.3,4.0,8.0,13.0,18.0,22.5,26.4,29.8,32.7,
        35.0,36.9,38.3,39.3,40.0,40.3,40.3,40.0,39.4,38.5,
        37.4,36.0,34.3,32.5,30.5,28.4,26.2,24.0,21.8,19.6,
        17.5,15.6,13.8,12.3,11.0,10.0,9.2,8.7,8.3,8.2,
        8.3,8.6,9.1,9.8,10.7,11.7,12.8,14.1,15.5,17.0,
    ],
}


def _make_fallback_df(cycle: str) -> pd.DataFrame:
    speeds = _FALLBACK[cycle]
    time = [i * 0.1 for i in range(len(speeds))]
    return pd.DataFrame({"time_s": time, "speed_mps": speeds})


def download_cycle(cycle: str) -> bool:
    output_path = OUTPUT_DIR / f"{cycle}_epa_speed.csv"
    if output_path.exists() and output_path.stat().st_size > 100:
        print(f"  ✅ {cycle}: already exists, skipping download.")
        return True

    url = f"{BASE_URL}/{cycle}_drivecycle_speeds_10Hz.csv"
    print(f"  Downloading {cycle} from {url} ...")
    try:
        raw = pd.read_csv(url, header=None, skiprows=1, sep=r'\s+')
        speed = raw.values.flatten().astype(float)
        time = [i * 0.1 for i in range(len(speed))]
        df = pd.DataFrame({"time_s": time, "speed_mps": speed})
        df.to_csv(output_path, index=False)
        print(f"  ✅ {cycle}: saved {len(df)} rows → {output_path.name}")
        return True
    except Exception as exc:
        print(f"  ⚠️  {cycle}: download failed ({exc}). Writing fallback trace.")
        df = _make_fallback_df(cycle)
        df.to_csv(output_path, index=False)
        print(f"  ⚠️  {cycle}: fallback saved ({len(df)} pts). Re-run when network available.")
        return False


if __name__ == "__main__":
    print("📥 Downloading EPA drive cycles...")
    results = {c: download_cycle(c) for c in DRIVE_CYCLES}
    ok = sum(results.values())
    print(f"\n{'✅' if ok == 3 else '⚠️ '} {ok}/3 cycles downloaded from network "
          f"({'all good' if ok == 3 else 'fallback used for failures'}).")
    print("Drive cycles ready in data/drive_cycles/")
