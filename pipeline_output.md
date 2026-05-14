# Pipeline Output

```
============================================================
🔋 CORRECTED BATTERY DIGITAL TWIN — FULL PIPELINE RUN
============================================================

============================================================
🚀 STEP: Download NASA Battery Data
============================================================
✅ NASA dataset already downloaded and extracted in data/nasa/.
✅ STEP COMPLETE: Download NASA Battery Data


============================================================
🚀 STEP: Download EPA Drive Cycles
============================================================
/dist_home/aksay/pranesh/EV-battery-management-system/data/step0_download_epa_drive_cycles.py:26: SyntaxWarning: invalid escape sequence '\s'
  raw = pd.read_csv(url, header=None, skiprows=1, sep='\s+')
Downloading UDDS from https://raw.githubusercontent.com/CIRCLES-consortium/CIRCLES-ENERGY-MODELS/main/UDDS_drivecycle_speeds_10Hz.csv ...
  Saved 13691 rows to data/drive_cycles/UDDS_epa_speed.csv
Downloading HWFET from https://raw.githubusercontent.com/CIRCLES-consortium/CIRCLES-ENERGY-MODELS/main/HWFET_drivecycle_speeds_10Hz.csv ...
  Saved 7641 rows to data/drive_cycles/HWFET_epa_speed.csv
Downloading US06 from https://raw.githubusercontent.com/CIRCLES-consortium/CIRCLES-ENERGY-MODELS/main/US06_drivecycle_speeds_10Hz.csv ...
  Saved 6001 rows to data/drive_cycles/US06_epa_speed.csv

Done. All three drive cycles are ready in data/drive_cycles/
✅ STEP COMPLETE: Download EPA Drive Cycles


============================================================
🚀 STEP: Parse NASA Data (Linear SOH Baseline)
============================================================
🔬 Parsing NASA Dataset: extracting aging features + full discharge time-series...
  ✅ B0005: 168 cycles -> B0005_aging_features.csv
  ✅ B0005: 50285 timesteps -> B0005_discharge_timeseries.csv
  ✅ B0006: 168 cycles -> B0006_aging_features.csv
  ✅ B0006: 50285 timesteps -> B0006_discharge_timeseries.csv
  ✅ B0007: 168 cycles -> B0007_aging_features.csv
  ✅ B0007: 50285 timesteps -> B0007_discharge_timeseries.csv
  ✅ B0018: 132 cycles -> B0018_aging_features.csv
  ✅ B0018: 34866 timesteps -> B0018_discharge_timeseries.csv
🔬 Done parsing NASA dataset.
✅ STEP COMPLETE: Parse NASA Data (Linear SOH Baseline)


============================================================
🚀 STEP: Generate Digital Twin (ECM + EETM + OCV Extraction)
============================================================
/dist_home/aksay/pranesh/EV-battery-management-system/generation/step4_generate_aging_digital_twin.py:297: RuntimeWarning: overflow encountered in scalar multiply
  Ts[k+1] = Ts[k] + dTs * dt
/dist_home/aksay/pranesh/EV-battery-management-system/generation/step4_generate_aging_digital_twin.py:297: RuntimeWarning: invalid value encountered in scalar add
  Ts[k+1] = Ts[k] + dTs * dt
/dist_home/aksay/pranesh/EV-battery-management-system/generation/step4_generate_aging_digital_twin.py:302: RuntimeWarning: overflow encountered in square
  return np.mean((Ts_sim - temp_surface_measured) ** 2)
/dist_home/aksay/pranesh/EV-battery-management-system/generation/step4_generate_aging_digital_twin.py:294: RuntimeWarning: overflow encountered in scalar divide
  dTc = (Q_gen[k] - (Tc[k] - Ts[k]) / Rin) / Cc
/dist_home/aksay/pranesh/EV-battery-management-system/generation/step4_generate_aging_digital_twin.py:295: RuntimeWarning: overflow encountered in scalar divide
  dTs = ((Tc[k] - Ts[k]) / Rin - (Ts[k] - T_ambient) / Rout) / Cs
/dist_home/aksay/pranesh/EV-battery-management-system/generation/step4_generate_aging_digital_twin.py:296: RuntimeWarning: invalid value encountered in scalar add
  Tc[k+1] = Tc[k] + dTc * dt
/dist_home/aksay/pranesh/EV-battery-management-system/.venv/lib/python3.13/site-packages/scipy/optimize/_numdiff.py:686: RuntimeWarning: invalid value encountered in subtract
  df = [f_eval - f0 for f_eval in f_evals]
✅ Extracted OCV curves (fresh 257 pts, aged 522 pts)

🔋 Processing B0005...
   ⚠️  Cycle 1: Q_gen = 0.277 W (too low)
   ⚠️  Cycle 2: Q_gen = 0.277 W (too low)
   ⚠️  Cycle 3: Q_gen = 0.277 W (too low)
   ⚠️  Cycle 4: Q_gen = 0.278 W (too low)
   ⚠️  Cycle 5: Q_gen = 0.278 W (too low)
   ⚠️  Cycle 6: Q_gen = 0.277 W (too low)
   ⚠️  Cycle 7: Q_gen = 0.277 W (too low)
   ⚠️  Cycle 8: Q_gen = 0.281 W (too low)
   ⚠️  Cycle 9: Q_gen = 0.282 W (too low)
   ⚠️  Cycle 10: Q_gen = 0.282 W (too low)
   ⚠️  Cycle 11: Q_gen = 0.284 W (too low)
   ⚠️  Cycle 12: Q_gen = 0.285 W (too low)
   ⚠️  Cycle 13: Q_gen = 0.287 W (too low)
   ⚠️  Cycle 14: Q_gen = 0.287 W (too low)
   ⚠️  Cycle 15: Q_gen = 0.287 W (too low)
   ⚠️  Cycle 16: Q_gen = 0.288 W (too low)
   ⚠️  Cycle 17: Q_gen = 0.290 W (too low)
   ⚠️  Cycle 18: Q_gen = 0.291 W (too low)
   ⚠️  Cycle 19: Q_gen = 0.291 W (too low)
   ⚠️  Cycle 20: Q_gen = 0.283 W (too low)
   Cycle  20 | SOH=0.924 | R0=0.0300Ω | ECM time=0.75s | ECM_MSE=2574981.266425 | Ts_RMSE=0.722°C | Q_gen_mean=0.28W | Rin=0.66 Rout=30.70 Cc=42.98 Cs=22.50
   ⚠️  Cycle 21: Q_gen = 0.286 W (too low)
   ⚠️  Cycle 22: Q_gen = 0.285 W (too low)
   ⚠️  Cycle 23: Q_gen = 0.287 W (too low)
   ⚠️  Cycle 24: Q_gen = 0.285 W (too low)
   ⚠️  Cycle 25: Q_gen = 0.291 W (too low)
   ⚠️  Cycle 26: Q_gen = 0.289 W (too low)
   ⚠️  Cycle 27: Q_gen = 0.291 W (too low)
   ⚠️  Cycle 28: Q_gen = 0.291 W (too low)
   ⚠️  Cycle 29: Q_gen = 0.289 W (too low)
   ⚠️  Cycle 30: Q_gen = 0.291 W (too low)
   ⚠️  Cycle 31: Q_gen = 0.296 W (too low)
   ⚠️  Cycle 32: Q_gen = 0.297 W (too low)
   ⚠️  Cycle 33: Q_gen = 0.297 W (too low)
   ⚠️  Cycle 34: Q_gen = 0.296 W (too low)
   ⚠️  Cycle 35: Q_gen = 0.297 W (too low)
   ⚠️  Cycle 36: Q_gen = 0.296 W (too low)
   ⚠️  Cycle 37: Q_gen = 0.295 W (too low)
   ⚠️  Cycle 38: Q_gen = 0.296 W (too low)
   ⚠️  Cycle 39: Q_gen = 0.295 W (too low)
   ⚠️  Cycle 40: Q_gen = 0.296 W (too low)
   Cycle  40 | SOH=0.887 | R0=0.0300Ω | ECM time=1.38s | ECM_MSE=2230700.869035 | Ts_RMSE=0.799°C | Q_gen_mean=0.30W | Rin=0.54 Rout=36.23 Cc=55.28 Cs=10.44
   ⚠️  Cycle 41: Q_gen = 0.295 W (too low)
   ⚠️  Cycle 42: Q_gen = 0.295 W (too low)
   ⚠️  Cycle 43: Q_gen = 0.291 W (too low)
   ⚠️  Cycle 44: Q_gen = 0.295 W (too low)
   ⚠️  Cycle 45: Q_gen = 0.294 W (too low)
   ⚠️  Cycle 46: Q_gen = 0.294 W (too low)
   ⚠️  Cycle 47: Q_gen = 0.294 W (too low)
   ⚠️  Cycle 48: Q_gen = 0.300 W (too low)
   ⚠️  Cycle 49: Q_gen = 0.298 W (too low)
   ⚠️  Cycle 50: Q_gen = 0.297 W (too low)
   ⚠️  Cycle 51: Q_gen = 0.296 W (too low)
   ⚠️  Cycle 52: Q_gen = 0.295 W (too low)
   ⚠️  Cycle 53: Q_gen = 0.295 W (too low)
   ⚠️  Cycle 54: Q_gen = 0.294 W (too low)
   ⚠️  Cycle 55: Q_gen = 0.293 W (too low)
   ⚠️  Cycle 56: Q_gen = 0.292 W (too low)
   ⚠️  Cycle 57: Q_gen = 0.291 W (too low)
   ⚠️  Cycle 58: Q_gen = 0.290 W (too low)
   ⚠️  Cycle 59: Q_gen = 0.290 W (too low)
   ⚠️  Cycle 60: Q_gen = 0.289 W (too low)
   Cycle  60 | SOH=0.847 | R0=0.0300Ω | ECM time=1.34s | ECM_MSE=1114039.691707 | Ts_RMSE=0.825°C | Q_gen_mean=0.29W | Rin=0.58 Rout=39.52 Cc=57.13 Cs=9.73
   ⚠️  Cycle 61: Q_gen = 0.288 W (too low)
   ⚠️  Cycle 62: Q_gen = 0.286 W (too low)
   ⚠️  Cycle 63: Q_gen = 0.287 W (too low)
   ⚠️  Cycle 64: Q_gen = 0.231 W (too low)
   ⚠️  Cycle 65: Q_gen = 0.231 W (too low)
   ⚠️  Cycle 66: Q_gen = 0.231 W (too low)
   ⚠️  Cycle 67: Q_gen = 0.230 W (too low)
   ⚠️  Cycle 68: Q_gen = 0.230 W (too low)
   ⚠️  Cycle 69: Q_gen = 0.229 W (too low)
   ⚠️  Cycle 70: Q_gen = 0.229 W (too low)
   ⚠️  Cycle 71: Q_gen = 0.229 W (too low)
   ⚠️  Cycle 72: Q_gen = 0.227 W (too low)
   ⚠️  Cycle 73: Q_gen = 0.227 W (too low)
   ⚠️  Cycle 74: Q_gen = 0.227 W (too low)
   ⚠️  Cycle 75: Q_gen = 0.226 W (too low)
   ⚠️  Cycle 76: Q_gen = 0.226 W (too low)
   ⚠️  Cycle 77: Q_gen = 0.226 W (too low)
   ⚠️  Cycle 78: Q_gen = 0.227 W (too low)
   ⚠️  Cycle 79: Q_gen = 0.226 W (too low)
   ⚠️  Cycle 80: Q_gen = 0.225 W (too low)
   Cycle  80 | SOH=0.782 | R0=0.0300Ω | ECM time=1.28s | ECM_MSE=119043.299638 | Ts_RMSE=0.799°C | Q_gen_mean=0.22W | Rin=0.73 Rout=45.47 Cc=53.56 Cs=7.56
   ⚠️  Cycle 81: Q_gen = 0.224 W (too low)
   ⚠️  Cycle 82: Q_gen = 0.224 W (too low)
   ⚠️  Cycle 83: Q_gen = 0.224 W (too low)
   ⚠️  Cycle 84: Q_gen = 0.223 W (too low)
   ⚠️  Cycle 85: Q_gen = 0.223 W (too low)
   ⚠️  Cycle 86: Q_gen = 0.222 W (too low)
   ⚠️  Cycle 87: Q_gen = 0.222 W (too low)
   ⚠️  Cycle 88: Q_gen = 0.222 W (too low)
   ⚠️  Cycle 89: Q_gen = 0.221 W (too low)
   ⚠️  Cycle 90: Q_gen = 0.225 W (too low)
   ⚠️  Cycle 91: Q_gen = 0.226 W (too low)
   ⚠️  Cycle 92: Q_gen = 0.224 W (too low)
   ⚠️  Cycle 93: Q_gen = 0.223 W (too low)
   ⚠️  Cycle 94: Q_gen = 0.222 W (too low)
   ⚠️  Cycle 95: Q_gen = 0.221 W (too low)
   ⚠️  Cycle 96: Q_gen = 0.221 W (too low)
   ⚠️  Cycle 97: Q_gen = 0.221 W (too low)
   ⚠️  Cycle 98: Q_gen = 0.337 W (too low)
   ⚠️  Cycle 99: Q_gen = 0.435 W (too low)
   ⚠️  Cycle 100: Q_gen = 0.433 W (too low)
   Cycle 100 | SOH=0.743 | R0=0.0300Ω | ECM time=1.26s | ECM_MSE=0.619650 | Ts_RMSE=0.778°C | Q_gen_mean=0.43W | Rin=0.90 Rout=46.71 Cc=51.79 Cs=6.25
   ⚠️  Cycle 101: Q_gen = 0.433 W (too low)
   ⚠️  Cycle 102: Q_gen = 0.433 W (too low)
   ⚠️  Cycle 103: Q_gen = 0.435 W (too low)
   ⚠️  Cycle 104: Q_gen = 0.437 W (too low)
   ⚠️  Cycle 105: Q_gen = 0.435 W (too low)
   ⚠️  Cycle 106: Q_gen = 0.433 W (too low)
   ⚠️  Cycle 107: Q_gen = 0.432 W (too low)
   ⚠️  Cycle 108: Q_gen = 0.429 W (too low)
   ⚠️  Cycle 109: Q_gen = 0.431 W (too low)
   ⚠️  Cycle 110: Q_gen = 0.430 W (too low)
   ⚠️  Cycle 111: Q_gen = 0.428 W (too low)
   ⚠️  Cycle 112: Q_gen = 0.427 W (too low)
   ⚠️  Cycle 113: Q_gen = 0.428 W (too low)
   ⚠️  Cycle 114: Q_gen = 0.426 W (too low)
   ⚠️  Cycle 115: Q_gen = 0.426 W (too low)
   ⚠️  Cycle 116: Q_gen = 0.425 W (too low)
   ⚠️  Cycle 117: Q_gen = 0.425 W (too low)
   ⚠️  Cycle 118: Q_gen = 0.425 W (too low)
   ⚠️  Cycle 119: Q_gen = 0.424 W (too low)
   ⚠️  Cycle 120: Q_gen = 0.429 W (too low)
   Cycle 120 | SOH=0.717 | R0=0.0300Ω | ECM time=1.22s | ECM_MSE=0.641611 | Ts_RMSE=0.768°C | Q_gen_mean=0.43W | Rin=0.98 Rout=48.15 Cc=52.43 Cs=5.65
   ⚠️  Cycle 121: Q_gen = 0.430 W (too low)
   ⚠️  Cycle 122: Q_gen = 0.426 W (too low)
   ⚠️  Cycle 123: Q_gen = 0.423 W (too low)
   ⚠️  Cycle 124: Q_gen = 0.423 W (too low)
   ⚠️  Cycle 125: Q_gen = 0.422 W (too low)
   ⚠️  Cycle 126: Q_gen = 0.423 W (too low)
   ⚠️  Cycle 127: Q_gen = 0.421 W (too low)
   ⚠️  Cycle 128: Q_gen = 0.419 W (too low)
   ⚠️  Cycle 129: Q_gen = 0.419 W (too low)
   ⚠️  Cycle 130: Q_gen = 0.419 W (too low)
   ⚠️  Cycle 131: Q_gen = 0.419 W (too low)
   ⚠️  Cycle 132: Q_gen = 0.417 W (too low)
   ⚠️  Cycle 133: Q_gen = 0.420 W (too low)
   ⚠️  Cycle 134: Q_gen = 0.422 W (too low)
   ⚠️  Cycle 135: Q_gen = 0.418 W (too low)
   ⚠️  Cycle 136: Q_gen = 0.417 W (too low)
   ⚠️  Cycle 137: Q_gen = 0.415 W (too low)
   ⚠️  Cycle 138: Q_gen = 0.416 W (too low)
   ⚠️  Cycle 139: Q_gen = 0.416 W (too low)
   ⚠️  Cycle 140: Q_gen = 0.415 W (too low)
   Cycle 140 | SOH=0.675 | R0=0.0300Ω | ECM time=1.19s | ECM_MSE=0.663324 | Ts_RMSE=0.812°C | Q_gen_mean=0.42W | Rin=1.56 Rout=48.01 Cc=46.49 Cs=3.40
   ⚠️  Cycle 141: Q_gen = 0.415 W (too low)
   ⚠️  Cycle 142: Q_gen = 0.413 W (too low)
   ⚠️  Cycle 143: Q_gen = 0.413 W (too low)
   ⚠️  Cycle 144: Q_gen = 0.413 W (too low)
   ⚠️  Cycle 145: Q_gen = 0.411 W (too low)
   ⚠️  Cycle 146: Q_gen = 0.411 W (too low)
   ⚠️  Cycle 147: Q_gen = 0.409 W (too low)
   ⚠️  Cycle 148: Q_gen = 0.409 W (too low)
   ⚠️  Cycle 149: Q_gen = 0.411 W (too low)/dist_home/aksay/pranesh/EV-battery-management-system/generation/step4_generate_aging_digital_twin.py:302: RuntimeWarning: overflow encountered in square
  return np.mean((Ts_sim - temp_surface_measured) ** 2)
/dist_home/aksay/pranesh/EV-battery-management-system/.venv/lib/python3.13/site-packages/scipy/optimize/_numdiff.py:686: RuntimeWarning: invalid value encountered in subtract
  df = [f_eval - f0 for f_eval in f_evals]
/dist_home/aksay/pranesh/EV-battery-management-system/generation/step4_generate_aging_digital_twin.py:297: RuntimeWarning: overflow encountered in scalar multiply
  Ts[k+1] = Ts[k] + dTs * dt
/dist_home/aksay/pranesh/EV-battery-management-system/generation/step4_generate_aging_digital_twin.py:297: RuntimeWarning: invalid value encountered in scalar add
  Ts[k+1] = Ts[k] + dTs * dt

```