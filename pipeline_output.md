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
/dist_home/aksay/miniconda3/lib/python3.12/site-packages/scipy/optimize/_numdiff.py:596: RuntimeWarning: invalid value encountered in subtract
  df = fun(x1) - f0
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
   Cycle  20 | SOH=0.924 | R0=0.0300Ω | ECM time=0.73s | ECM_MSE=2574981.266425 | Ts_RMSE=0.722°C | Q_gen_mean=0.28W | Rin=0.66 Rout=30.70 Cc=42.98 Cs=22.50
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
   Cycle  40 | SOH=0.887 | R0=0.0300Ω | ECM time=1.34s | ECM_MSE=2230700.869035 | Ts_RMSE=0.799°C | Q_gen_mean=0.30W | Rin=0.54 Rout=36.23 Cc=55.28 Cs=10.44
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
   Cycle  60 | SOH=0.847 | R0=0.0300Ω | ECM time=1.33s | ECM_MSE=1114039.691707 | Ts_RMSE=0.825°C | Q_gen_mean=0.29W | Rin=0.58 Rout=39.52 Cc=57.13 Cs=9.73
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
   Cycle  80 | SOH=0.782 | R0=0.0300Ω | ECM time=1.26s | ECM_MSE=119043.299638 | Ts_RMSE=0.799°C | Q_gen_mean=0.22W | Rin=0.73 Rout=45.47 Cc=53.56 Cs=7.56
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
   Cycle 100 | SOH=0.743 | R0=0.0300Ω | ECM time=1.23s | ECM_MSE=0.619650 | Ts_RMSE=0.778°C | Q_gen_mean=0.43W | Rin=0.90 Rout=46.71 Cc=51.79 Cs=6.25
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
   Cycle 120 | SOH=0.717 | R0=0.0300Ω | ECM time=1.20s | ECM_MSE=0.641611 | Ts_RMSE=0.768°C | Q_gen_mean=0.43W | Rin=0.98 Rout=48.15 Cc=52.43 Cs=5.65
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
   Cycle 140 | SOH=0.675 | R0=0.0300Ω | ECM time=1.17s | ECM_MSE=0.663324 | Ts_RMSE=0.812°C | Q_gen_mean=0.42W | Rin=1.56 Rout=48.01 Cc=46.49 Cs=3.40
   ⚠️  Cycle 141: Q_gen = 0.415 W (too low)
   ⚠️  Cycle 142: Q_gen = 0.413 W (too low)
   ⚠️  Cycle 143: Q_gen = 0.413 W (too low)
   ⚠️  Cycle 144: Q_gen = 0.413 W (too low)
   ⚠️  Cycle 145: Q_gen = 0.411 W (too low)
   ⚠️  Cycle 146: Q_gen = 0.411 W (too low)
   ⚠️  Cycle 147: Q_gen = 0.409 W (too low)
   ⚠️  Cycle 148: Q_gen = 0.409 W (too low)
/dist_home/aksay/pranesh/EV-battery-management-system/generation/step4_generate_aging_digital_twin.py:302: RuntimeWarning: overflow encountered in square
  return np.mean((Ts_sim - temp_surface_measured) ** 2)
/dist_home/aksay/miniconda3/lib/python3.12/site-packages/scipy/optimize/_numdiff.py:596: RuntimeWarning: invalid value encountered in subtract
  df = fun(x1) - f0
/dist_home/aksay/pranesh/EV-battery-management-system/generation/step4_generate_aging_digital_twin.py:297: RuntimeWarning: overflow encountered in scalar multiply
  Ts[k+1] = Ts[k] + dTs * dt
/dist_home/aksay/pranesh/EV-battery-management-system/generation/step4_generate_aging_digital_twin.py:297: RuntimeWarning: invalid value encountered in scalar add
  Ts[k+1] = Ts[k] + dTs * dt
   ⚠️  Cycle 149: Q_gen = 0.411 W (too low)
   ⚠️  Cycle 150: Q_gen = 0.411 W (too low)
   ⚠️  Cycle 151: Q_gen = 0.418 W (too low)
   ⚠️  Cycle 152: Q_gen = 0.415 W (too low)
   ⚠️  Cycle 153: Q_gen = 0.413 W (too low)
   ⚠️  Cycle 154: Q_gen = 0.411 W (too low)
   ⚠️  Cycle 155: Q_gen = 0.411 W (too low)
   ⚠️  Cycle 156: Q_gen = 0.409 W (too low)
   ⚠️  Cycle 157: Q_gen = 0.410 W (too low)
   ⚠️  Cycle 158: Q_gen = 0.409 W (too low)
   ⚠️  Cycle 159: Q_gen = 0.408 W (too low)
   ⚠️  Cycle 160: Q_gen = 0.408 W (too low)
   Cycle 160 | SOH=0.652 | R0=0.0300Ω | ECM time=1.14s | ECM_MSE=0.675122 | Ts_RMSE=0.826°C | Q_gen_mean=0.41W | Rin=2.01 Rout=48.06 Cc=43.33 Cs=2.73
   ⚠️  Cycle 161: Q_gen = 0.410 W (too low)
   ⚠️  Cycle 162: Q_gen = 0.408 W (too low)
   ⚠️  Cycle 163: Q_gen = 0.408 W (too low)
   ⚠️  Cycle 164: Q_gen = 0.408 W (too low)
   ⚠️  Cycle 165: Q_gen = 0.406 W (too low)
   ⚠️  Cycle 166: Q_gen = 0.406 W (too low)
   ⚠️  Cycle 167: Q_gen = 0.413 W (too low)
   ⚠️  Cycle 168: Q_gen = 0.415 W (too low)

🔋 Processing B0006...
   ⚠️  Cycle 1: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 2: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 3: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 4: Q_gen = 0.302 W (too low)
   ⚠️  Cycle 5: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 6: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 7: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 8: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 9: Q_gen = 0.302 W (too low)
   ⚠️  Cycle 10: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 11: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 12: Q_gen = 0.302 W (too low)
   Cycle  12 | SOH=0.967 | R0=0.0300Ω | ECM time=0.72s | ECM_MSE=2761828.789743 | Ts_RMSE=0.654°C | Q_gen_mean=0.30W | Rin=0.58 Rout=32.54 Cc=47.53 Cs=24.92
   ⚠️  Cycle 13: Q_gen = 0.302 W (too low)
   ⚠️  Cycle 14: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 15: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 16: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 17: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 18: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 19: Q_gen = 0.301 W (too low)
   ⚠️  Cycle 20: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 21: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 22: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 23: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 24: Q_gen = 0.302 W (too low)
   ⚠️  Cycle 25: Q_gen = 0.302 W (too low)
   ⚠️  Cycle 26: Q_gen = 0.301 W (too low)
   ⚠️  Cycle 27: Q_gen = 0.301 W (too low)
   ⚠️  Cycle 28: Q_gen = 0.300 W (too low)
   ⚠️  Cycle 29: Q_gen = 0.295 W (too low)
   ⚠️  Cycle 30: Q_gen = 0.301 W (too low)
   ⚠️  Cycle 31: Q_gen = 0.308 W (too low)
   ⚠️  Cycle 32: Q_gen = 0.307 W (too low)
   Cycle  32 | SOH=0.941 | R0=0.0300Ω | ECM time=1.40s | ECM_MSE=2783903.718207 | Ts_RMSE=0.627°C | Q_gen_mean=0.31W | Rin=0.51 Rout=32.07 Cc=66.81 Cs=11.02
   ⚠️  Cycle 33: Q_gen = 0.305 W (too low)
   ⚠️  Cycle 34: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 35: Q_gen = 0.301 W (too low)
   ⚠️  Cycle 36: Q_gen = 0.301 W (too low)
   ⚠️  Cycle 37: Q_gen = 0.299 W (too low)
   ⚠️  Cycle 38: Q_gen = 0.298 W (too low)
   ⚠️  Cycle 39: Q_gen = 0.297 W (too low)
   ⚠️  Cycle 40: Q_gen = 0.296 W (too low)
   ⚠️  Cycle 41: Q_gen = 0.294 W (too low)
   ⚠️  Cycle 42: Q_gen = 0.293 W (too low)
   ⚠️  Cycle 43: Q_gen = 0.291 W (too low)
   ⚠️  Cycle 44: Q_gen = 0.298 W (too low)
   ⚠️  Cycle 45: Q_gen = 0.293 W (too low)
   ⚠️  Cycle 46: Q_gen = 0.291 W (too low)
   ⚠️  Cycle 47: Q_gen = 0.290 W (too low)
   ⚠️  Cycle 48: Q_gen = 0.307 W (too low)
   ⚠️  Cycle 49: Q_gen = 0.304 W (too low)
   ⚠️  Cycle 50: Q_gen = 0.301 W (too low)
   ⚠️  Cycle 51: Q_gen = 0.296 W (too low)
   ⚠️  Cycle 52: Q_gen = 0.294 W (too low)
   Cycle  52 | SOH=0.865 | R0=0.0300Ω | ECM time=1.34s | ECM_MSE=1556245.403308 | Ts_RMSE=0.684°C | Q_gen_mean=0.29W | Rin=0.74 Rout=42.49 Cc=67.40 Cs=7.21
   ⚠️  Cycle 53: Q_gen = 0.291 W (too low)
   ⚠️  Cycle 54: Q_gen = 0.289 W (too low)
   ⚠️  Cycle 55: Q_gen = 0.286 W (too low)
   ⚠️  Cycle 56: Q_gen = 0.231 W (too low)
   ⚠️  Cycle 57: Q_gen = 0.231 W (too low)
   ⚠️  Cycle 58: Q_gen = 0.229 W (too low)
   ⚠️  Cycle 59: Q_gen = 0.229 W (too low)
   ⚠️  Cycle 60: Q_gen = 0.227 W (too low)
   ⚠️  Cycle 61: Q_gen = 0.226 W (too low)
   ⚠️  Cycle 62: Q_gen = 0.225 W (too low)
   ⚠️  Cycle 63: Q_gen = 0.225 W (too low)
   ⚠️  Cycle 64: Q_gen = 0.224 W (too low)
   ⚠️  Cycle 65: Q_gen = 0.223 W (too low)
   ⚠️  Cycle 66: Q_gen = 0.222 W (too low)
   ⚠️  Cycle 67: Q_gen = 0.221 W (too low)
   ⚠️  Cycle 68: Q_gen = 0.221 W (too low)
   ⚠️  Cycle 69: Q_gen = 0.221 W (too low)
   ⚠️  Cycle 70: Q_gen = 0.219 W (too low)
   ⚠️  Cycle 71: Q_gen = 0.219 W (too low)
   ⚠️  Cycle 72: Q_gen = 0.218 W (too low)
   Cycle  72 | SOH=0.762 | R0=0.0300Ω | ECM time=1.30s | ECM_MSE=16327.570982 | Ts_RMSE=0.698°C | Q_gen_mean=0.22W | Rin=1.92 Rout=46.00 Cc=57.94 Cs=2.78
   ⚠️  Cycle 73: Q_gen = 0.218 W (too low)
   ⚠️  Cycle 74: Q_gen = 0.217 W (too low)
   ⚠️  Cycle 75: Q_gen = 0.314 W (too low)
   ⚠️  Cycle 76: Q_gen = 0.424 W (too low)
   ⚠️  Cycle 77: Q_gen = 0.424 W (too low)
   ⚠️  Cycle 78: Q_gen = 0.223 W (too low)
   ⚠️  Cycle 79: Q_gen = 0.316 W (too low)
   ⚠️  Cycle 80: Q_gen = 0.427 W (too low)
   ⚠️  Cycle 81: Q_gen = 0.423 W (too low)
   ⚠️  Cycle 82: Q_gen = 0.423 W (too low)
   ⚠️  Cycle 83: Q_gen = 0.423 W (too low)
   ⚠️  Cycle 84: Q_gen = 0.422 W (too low)
   ⚠️  Cycle 85: Q_gen = 0.421 W (too low)
   ⚠️  Cycle 86: Q_gen = 0.420 W (too low)
   ⚠️  Cycle 87: Q_gen = 0.419 W (too low)
   ⚠️  Cycle 88: Q_gen = 0.420 W (too low)
   ⚠️  Cycle 89: Q_gen = 0.419 W (too low)
   ⚠️  Cycle 90: Q_gen = 0.226 W (too low)
   ⚠️  Cycle 91: Q_gen = 0.226 W (too low)
   ⚠️  Cycle 92: Q_gen = 0.222 W (too low)
   Cycle  92 | SOH=0.757 | R0=0.0300Ω | ECM time=1.25s | ECM_MSE=5847.842495 | Ts_RMSE=0.688°C | Q_gen_mean=0.22W | Rin=1.22 Rout=48.77 Cc=56.26 Cs=4.43
   ⚠️  Cycle 93: Q_gen = 0.433 W (too low)
   ⚠️  Cycle 94: Q_gen = 0.430 W (too low)
   ⚠️  Cycle 95: Q_gen = 0.428 W (too low)
   ⚠️  Cycle 96: Q_gen = 0.427 W (too low)
   ⚠️  Cycle 97: Q_gen = 0.425 W (too low)
   ⚠️  Cycle 98: Q_gen = 0.423 W (too low)
   ⚠️  Cycle 99: Q_gen = 0.423 W (too low)
   ⚠️  Cycle 100: Q_gen = 0.422 W (too low)
   ⚠️  Cycle 101: Q_gen = 0.421 W (too low)
   ⚠️  Cycle 102: Q_gen = 0.420 W (too low)
   ⚠️  Cycle 103: Q_gen = 0.420 W (too low)
   ⚠️  Cycle 104: Q_gen = 0.427 W (too low)
   ⚠️  Cycle 105: Q_gen = 0.423 W (too low)
   ⚠️  Cycle 106: Q_gen = 0.421 W (too low)
   ⚠️  Cycle 107: Q_gen = 0.422 W (too low)
   ⚠️  Cycle 108: Q_gen = 0.418 W (too low)
   ⚠️  Cycle 109: Q_gen = 0.417 W (too low)
   ⚠️  Cycle 110: Q_gen = 0.417 W (too low)
   ⚠️  Cycle 111: Q_gen = 0.416 W (too low)
   ⚠️  Cycle 112: Q_gen = 0.416 W (too low)
   Cycle 112 | SOH=0.692 | R0=0.0300Ω | ECM time=1.20s | ECM_MSE=0.772965 | Ts_RMSE=0.744°C | Q_gen_mean=0.42W | Rin=1.33 Rout=47.85 Cc=45.65 Cs=4.15
   ⚠️  Cycle 113: Q_gen = 0.415 W (too low)
   ⚠️  Cycle 114: Q_gen = 0.415 W (too low)
   ⚠️  Cycle 115: Q_gen = 0.414 W (too low)
   ⚠️  Cycle 116: Q_gen = 0.414 W (too low)
   ⚠️  Cycle 117: Q_gen = 0.413 W (too low)
   ⚠️  Cycle 118: Q_gen = 0.412 W (too low)
   ⚠️  Cycle 119: Q_gen = 0.413 W (too low)
   ⚠️  Cycle 120: Q_gen = 0.418 W (too low)
   ⚠️  Cycle 121: Q_gen = 0.423 W (too low)
   ⚠️  Cycle 122: Q_gen = 0.419 W (too low)
   ⚠️  Cycle 123: Q_gen = 0.415 W (too low)
   ⚠️  Cycle 124: Q_gen = 0.415 W (too low)
   ⚠️  Cycle 125: Q_gen = 0.411 W (too low)
   ⚠️  Cycle 126: Q_gen = 0.413 W (too low)
   ⚠️  Cycle 127: Q_gen = 0.411 W (too low)
   ⚠️  Cycle 128: Q_gen = 0.409 W (too low)
   ⚠️  Cycle 129: Q_gen = 0.407 W (too low)
/dist_home/aksay/pranesh/EV-battery-management-system/generation/step4_generate_aging_digital_twin.py:302: RuntimeWarning: overflow encountered in square
  return np.mean((Ts_sim - temp_surface_measured) ** 2)
/dist_home/aksay/miniconda3/lib/python3.12/site-packages/scipy/optimize/_numdiff.py:596: RuntimeWarning: invalid value encountered in subtract
  df = fun(x1) - f0
/dist_home/aksay/pranesh/EV-battery-management-system/generation/step4_generate_aging_digital_twin.py:297: RuntimeWarning: overflow encountered in scalar multiply
  Ts[k+1] = Ts[k] + dTs * dt
/dist_home/aksay/pranesh/EV-battery-management-system/generation/step4_generate_aging_digital_twin.py:297: RuntimeWarning: invalid value encountered in scalar add
  Ts[k+1] = Ts[k] + dTs * dt
   ⚠️  Cycle 130: Q_gen = 0.409 W (too low)
   ⚠️  Cycle 131: Q_gen = 0.407 W (too low)
   ⚠️  Cycle 132: Q_gen = 0.405 W (too low)
   Cycle 132 | SOH=0.658 | R0=0.0300Ω | ECM time=1.17s | ECM_MSE=0.778642 | Ts_RMSE=0.769°C | Q_gen_mean=0.41W | Rin=2.22 Rout=46.18 Cc=42.03 Cs=2.38
   ⚠️  Cycle 133: Q_gen = 0.406 W (too low)
   ⚠️  Cycle 134: Q_gen = 0.413 W (too low)
   ⚠️  Cycle 135: Q_gen = 0.408 W (too low)
   ⚠️  Cycle 136: Q_gen = 0.406 W (too low)
   ⚠️  Cycle 137: Q_gen = 0.402 W (too low)
   ⚠️  Cycle 138: Q_gen = 0.404 W (too low)
   ⚠️  Cycle 139: Q_gen = 0.402 W (too low)
   ⚠️  Cycle 140: Q_gen = 0.402 W (too low)
   ⚠️  Cycle 141: Q_gen = 0.398 W (too low)
   ⚠️  Cycle 142: Q_gen = 0.396 W (too low)
   ⚠️  Cycle 143: Q_gen = 0.397 W (too low)
   ⚠️  Cycle 144: Q_gen = 0.396 W (too low)
   ⚠️  Cycle 145: Q_gen = 0.393 W (too low)
   ⚠️  Cycle 146: Q_gen = 0.392 W (too low)
   ⚠️  Cycle 147: Q_gen = 0.392 W (too low)
   ⚠️  Cycle 148: Q_gen = 0.393 W (too low)
   ⚠️  Cycle 149: Q_gen = 0.391 W (too low)
   ⚠️  Cycle 150: Q_gen = 0.391 W (too low)
   ⚠️  Cycle 151: Q_gen = 0.399 W (too low)
   ⚠️  Cycle 152: Q_gen = 0.393 W (too low)
   Cycle 152 | SOH=0.632 | R0=0.0300Ω | ECM time=1.16s | ECM_MSE=0.769468 | Ts_RMSE=0.774°C | Q_gen_mean=0.39W | Rin=1.81 Rout=44.23 Cc=38.63 Cs=2.98
   ⚠️  Cycle 153: Q_gen = 0.389 W (too low)
   ⚠️  Cycle 154: Q_gen = 0.386 W (too low)
   ⚠️  Cycle 155: Q_gen = 0.386 W (too low)
   ⚠️  Cycle 156: Q_gen = 0.382 W (too low)
   ⚠️  Cycle 157: Q_gen = 0.382 W (too low)
   ⚠️  Cycle 158: Q_gen = 0.380 W (too low)
   ⚠️  Cycle 159: Q_gen = 0.380 W (too low)
   ⚠️  Cycle 160: Q_gen = 0.377 W (too low)
   ⚠️  Cycle 161: Q_gen = 0.376 W (too low)
   ⚠️  Cycle 162: Q_gen = 0.375 W (too low)
   ⚠️  Cycle 163: Q_gen = 0.373 W (too low)
   ⚠️  Cycle 164: Q_gen = 0.368 W (too low)
   ⚠️  Cycle 165: Q_gen = 0.370 W (too low)
   ⚠️  Cycle 166: Q_gen = 0.370 W (too low)
   ⚠️  Cycle 167: Q_gen = 0.374 W (too low)
   ⚠️  Cycle 168: Q_gen = 0.375 W (too low)

🔋 Processing B0007...
   ⚠️  Cycle 1: Q_gen = 0.284 W (too low)
   ⚠️  Cycle 2: Q_gen = 0.284 W (too low)
   ⚠️  Cycle 3: Q_gen = 0.286 W (too low)
   ⚠️  Cycle 4: Q_gen = 0.287 W (too low)
   Cycle   4 | SOH=0.940 | R0=0.0300Ω | ECM time=0.75s | ECM_MSE=2619817.561431 | Ts_RMSE=0.876°C | Q_gen_mean=0.29W | Rin=0.61 Rout=37.77 Cc=43.61 Cs=23.98
   ⚠️  Cycle 5: Q_gen = 0.287 W (too low)
   ⚠️  Cycle 6: Q_gen = 0.283 W (too low)
   ⚠️  Cycle 7: Q_gen = 0.283 W (too low)
   ⚠️  Cycle 8: Q_gen = 0.289 W (too low)
   ⚠️  Cycle 9: Q_gen = 0.292 W (too low)
   ⚠️  Cycle 10: Q_gen = 0.292 W (too low)
   ⚠️  Cycle 11: Q_gen = 0.293 W (too low)
   ⚠️  Cycle 12: Q_gen = 0.295 W (too low)
   ⚠️  Cycle 13: Q_gen = 0.297 W (too low)
   ⚠️  Cycle 14: Q_gen = 0.296 W (too low)
   ⚠️  Cycle 15: Q_gen = 0.295 W (too low)
   ⚠️  Cycle 16: Q_gen = 0.297 W (too low)
   ⚠️  Cycle 17: Q_gen = 0.300 W (too low)
   ⚠️  Cycle 18: Q_gen = 0.299 W (too low)
   ⚠️  Cycle 19: Q_gen = 0.299 W (too low)
   ⚠️  Cycle 20: Q_gen = 0.290 W (too low)
   ⚠️  Cycle 21: Q_gen = 0.293 W (too low)
   ⚠️  Cycle 22: Q_gen = 0.292 W (too low)
   ⚠️  Cycle 23: Q_gen = 0.296 W (too low)
   ⚠️  Cycle 24: Q_gen = 0.295 W (too low)
   Cycle  24 | SOH=0.935 | R0=0.0300Ω | ECM time=0.72s | ECM_MSE=2688470.141496 | Ts_RMSE=0.835°C | Q_gen_mean=0.29W | Rin=0.61 Rout=35.87 Cc=44.06 Cs=24.24
   ⚠️  Cycle 25: Q_gen = 0.298 W (too low)
   ⚠️  Cycle 26: Q_gen = 0.299 W (too low)
   ⚠️  Cycle 27: Q_gen = 0.298 W (too low)
   ⚠️  Cycle 28: Q_gen = 0.298 W (too low)
   ⚠️  Cycle 29: Q_gen = 0.299 W (too low)
   ⚠️  Cycle 30: Q_gen = 0.298 W (too low)
   ⚠️  Cycle 31: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 32: Q_gen = 0.304 W (too low)
   ⚠️  Cycle 33: Q_gen = 0.304 W (too low)
   ⚠️  Cycle 34: Q_gen = 0.304 W (too low)
   ⚠️  Cycle 35: Q_gen = 0.304 W (too low)
   ⚠️  Cycle 36: Q_gen = 0.304 W (too low)
   ⚠️  Cycle 37: Q_gen = 0.304 W (too low)
   ⚠️  Cycle 38: Q_gen = 0.304 W (too low)
   ⚠️  Cycle 39: Q_gen = 0.304 W (too low)
   ⚠️  Cycle 40: Q_gen = 0.304 W (too low)
   ⚠️  Cycle 41: Q_gen = 0.304 W (too low)
   ⚠️  Cycle 42: Q_gen = 0.304 W (too low)
   ⚠️  Cycle 43: Q_gen = 0.298 W (too low)
   ⚠️  Cycle 44: Q_gen = 0.304 W (too low)
   Cycle  44 | SOH=0.903 | R0=0.0300Ω | ECM time=1.35s | ECM_MSE=2783689.785091 | Ts_RMSE=0.792°C | Q_gen_mean=0.30W | Rin=0.50 Rout=33.43 Cc=55.24 Cs=11.50
   ⚠️  Cycle 45: Q_gen = 0.304 W (too low)
   ⚠️  Cycle 46: Q_gen = 0.304 W (too low)
   ⚠️  Cycle 47: Q_gen = 0.304 W (too low)
   ⚠️  Cycle 48: Q_gen = 0.306 W (too low)
   ⚠️  Cycle 49: Q_gen = 0.305 W (too low)
   ⚠️  Cycle 50: Q_gen = 0.305 W (too low)
   ⚠️  Cycle 51: Q_gen = 0.304 W (too low)
   ⚠️  Cycle 52: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 53: Q_gen = 0.303 W (too low)
   ⚠️  Cycle 54: Q_gen = 0.302 W (too low)
   ⚠️  Cycle 55: Q_gen = 0.301 W (too low)
   ⚠️  Cycle 56: Q_gen = 0.301 W (too low)
   ⚠️  Cycle 57: Q_gen = 0.301 W (too low)
   ⚠️  Cycle 58: Q_gen = 0.300 W (too low)
   ⚠️  Cycle 59: Q_gen = 0.300 W (too low)
   ⚠️  Cycle 60: Q_gen = 0.299 W (too low)
   ⚠️  Cycle 61: Q_gen = 0.298 W (too low)
   ⚠️  Cycle 62: Q_gen = 0.298 W (too low)
   ⚠️  Cycle 63: Q_gen = 0.297 W (too low)
   ⚠️  Cycle 64: Q_gen = 0.298 W (too low)
   Cycle  64 | SOH=0.852 | R0=0.0300Ω | ECM time=1.31s | ECM_MSE=1254948.558743 | Ts_RMSE=0.891°C | Q_gen_mean=0.30W | Rin=0.56 Rout=41.81 Cc=60.29 Cs=9.88
   ⚠️  Cycle 65: Q_gen = 0.296 W (too low)
   ⚠️  Cycle 66: Q_gen = 0.295 W (too low)
   ⚠️  Cycle 67: Q_gen = 0.294 W (too low)
   ⚠️  Cycle 68: Q_gen = 0.294 W (too low)
   ⚠️  Cycle 69: Q_gen = 0.293 W (too low)
   ⚠️  Cycle 70: Q_gen = 0.293 W (too low)
   ⚠️  Cycle 71: Q_gen = 0.238 W (too low)
   ⚠️  Cycle 72: Q_gen = 0.237 W (too low)
   ⚠️  Cycle 73: Q_gen = 0.237 W (too low)
   ⚠️  Cycle 74: Q_gen = 0.237 W (too low)
   ⚠️  Cycle 75: Q_gen = 0.237 W (too low)
   ⚠️  Cycle 76: Q_gen = 0.237 W (too low)
   ⚠️  Cycle 77: Q_gen = 0.237 W (too low)
   ⚠️  Cycle 78: Q_gen = 0.238 W (too low)
   ⚠️  Cycle 79: Q_gen = 0.236 W (too low)
   ⚠️  Cycle 80: Q_gen = 0.236 W (too low)
   ⚠️  Cycle 81: Q_gen = 0.236 W (too low)
   ⚠️  Cycle 82: Q_gen = 0.236 W (too low)
   ⚠️  Cycle 83: Q_gen = 0.235 W (too low)
   ⚠️  Cycle 84: Q_gen = 0.235 W (too low)
   Cycle  84 | SOH=0.805 | R0=0.0300Ω | ECM time=1.27s | ECM_MSE=363483.048347 | Ts_RMSE=0.883°C | Q_gen_mean=0.23W | Rin=0.98 Rout=52.62 Cc=59.22 Cs=5.32
   ⚠️  Cycle 85: Q_gen = 0.236 W (too low)
   ⚠️  Cycle 86: Q_gen = 0.236 W (too low)
   ⚠️  Cycle 87: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 88: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 89: Q_gen = 0.236 W (too low)
   ⚠️  Cycle 90: Q_gen = 0.297 W (too low)
   ⚠️  Cycle 91: Q_gen = 0.239 W (too low)
   ⚠️  Cycle 92: Q_gen = 0.238 W (too low)
   ⚠️  Cycle 93: Q_gen = 0.237 W (too low)
   ⚠️  Cycle 94: Q_gen = 0.236 W (too low)
   ⚠️  Cycle 95: Q_gen = 0.236 W (too low)
   ⚠️  Cycle 96: Q_gen = 0.236 W (too low)
   ⚠️  Cycle 97: Q_gen = 0.235 W (too low)
   ⚠️  Cycle 98: Q_gen = 0.235 W (too low)
   ⚠️  Cycle 99: Q_gen = 0.235 W (too low)
   ⚠️  Cycle 100: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 101: Q_gen = 0.235 W (too low)
   ⚠️  Cycle 102: Q_gen = 0.235 W (too low)
   ⚠️  Cycle 103: Q_gen = 0.235 W (too low)
   ⚠️  Cycle 104: Q_gen = 0.236 W (too low)
   Cycle 104 | SOH=0.787 | R0=0.0300Ω | ECM time=1.24s | ECM_MSE=165829.629714 | Ts_RMSE=0.904°C | Q_gen_mean=0.24W | Rin=1.08 Rout=56.76 Cc=60.05 Cs=4.80
   ⚠️  Cycle 105: Q_gen = 0.236 W (too low)
   ⚠️  Cycle 106: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 107: Q_gen = 0.237 W (too low)
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
   ⚠️  Cycle 108: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 109: Q_gen = 0.235 W (too low)
   ⚠️  Cycle 110: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 111: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 112: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 113: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 114: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 115: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 116: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 117: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 118: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 119: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 120: Q_gen = 0.236 W (too low)
   ⚠️  Cycle 121: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 122: Q_gen = 0.235 W (too low)
   ⚠️  Cycle 123: Q_gen = 0.234 W (too low)
   ⚠️  Cycle 124: Q_gen = 0.234 W (too low)
   Cycle 124 | SOH=0.754 | R0=0.0300Ω | ECM time=1.20s | ECM_MSE=1692.562238 | Ts_RMSE=0.850°C | Q_gen_mean=0.23W | Rin=0.64 Rout=60.00 Cc=54.67 Cs=8.56
   ⚠️  Cycle 125: Q_gen = 0.341 W (too low)
   ⚠️  Cycle 126: Q_gen = 0.457 W (too low)
   ⚠️  Cycle 127: Q_gen = 0.455 W (too low)
   ⚠️  Cycle 128: Q_gen = 0.453 W (too low)
   ⚠️  Cycle 129: Q_gen = 0.455 W (too low)
   ⚠️  Cycle 130: Q_gen = 0.455 W (too low)
   ⚠️  Cycle 131: Q_gen = 0.453 W (too low)
   ⚠️  Cycle 132: Q_gen = 0.453 W (too low)
   ⚠️  Cycle 133: Q_gen = 0.454 W (too low)
   ⚠️  Cycle 134: Q_gen = 0.456 W (too low)
   ⚠️  Cycle 135: Q_gen = 0.454 W (too low)
   ⚠️  Cycle 136: Q_gen = 0.453 W (too low)
   ⚠️  Cycle 137: Q_gen = 0.453 W (too low)
   ⚠️  Cycle 138: Q_gen = 0.452 W (too low)
   ⚠️  Cycle 139: Q_gen = 0.452 W (too low)
   ⚠️  Cycle 140: Q_gen = 0.452 W (too low)
   ⚠️  Cycle 141: Q_gen = 0.451 W (too low)
   ⚠️  Cycle 142: Q_gen = 0.449 W (too low)
   ⚠️  Cycle 143: Q_gen = 0.450 W (too low)
   ⚠️  Cycle 144: Q_gen = 0.450 W (too low)
   Cycle 144 | SOH=0.723 | R0=0.0300Ω | ECM time=1.16s | ECM_MSE=0.690804 | Ts_RMSE=0.785°C | Q_gen_mean=0.45W | Rin=0.79 Rout=60.00 Cc=53.44 Cs=6.99
   ⚠️  Cycle 145: Q_gen = 0.449 W (too low)
   ⚠️  Cycle 146: Q_gen = 0.449 W (too low)
   ⚠️  Cycle 147: Q_gen = 0.447 W (too low)
   ⚠️  Cycle 148: Q_gen = 0.448 W (too low)
   ⚠️  Cycle 149: Q_gen = 0.449 W (too low)
   ⚠️  Cycle 150: Q_gen = 0.449 W (too low)
   ⚠️  Cycle 151: Q_gen = 0.452 W (too low)
   ⚠️  Cycle 152: Q_gen = 0.451 W (too low)
   ⚠️  Cycle 153: Q_gen = 0.449 W (too low)
   ⚠️  Cycle 154: Q_gen = 0.448 W (too low)
   ⚠️  Cycle 155: Q_gen = 0.448 W (too low)
   ⚠️  Cycle 156: Q_gen = 0.447 W (too low)
   ⚠️  Cycle 157: Q_gen = 0.447 W (too low)
   ⚠️  Cycle 158: Q_gen = 0.446 W (too low)
   ⚠️  Cycle 159: Q_gen = 0.447 W (too low)
   ⚠️  Cycle 160: Q_gen = 0.446 W (too low)
   ⚠️  Cycle 161: Q_gen = 0.447 W (too low)
   ⚠️  Cycle 162: Q_gen = 0.446 W (too low)
   ⚠️  Cycle 163: Q_gen = 0.445 W (too low)
   ⚠️  Cycle 164: Q_gen = 0.445 W (too low)
   Cycle 164 | SOH=0.703 | R0=0.0300Ω | ECM time=1.14s | ECM_MSE=0.713488 | Ts_RMSE=0.768°C | Q_gen_mean=0.45W | Rin=1.02 Rout=60.00 Cc=51.43 Cs=5.36
   ⚠️  Cycle 165: Q_gen = 0.445 W (too low)
   ⚠️  Cycle 166: Q_gen = 0.444 W (too low)
   ⚠️  Cycle 167: Q_gen = 0.450 W (too low)
   ⚠️  Cycle 168: Q_gen = 0.450 W (too low)

🔋 Processing B0018...
   ⚠️  Cycle 1: Q_gen = 0.302 W (too low)
   ⚠️  Cycle 2: Q_gen = 0.301 W (too low)
   ⚠️  Cycle 3: Q_gen = 0.301 W (too low)
   ⚠️  Cycle 4: Q_gen = 0.300 W (too low)
   ⚠️  Cycle 5: Q_gen = 0.300 W (too low)
   ⚠️  Cycle 6: Q_gen = 0.300 W (too low)
   ⚠️  Cycle 7: Q_gen = 0.298 W (too low)
   ⚠️  Cycle 8: Q_gen = 0.298 W (too low)
   ⚠️  Cycle 9: Q_gen = 0.298 W (too low)
   ⚠️  Cycle 10: Q_gen = 0.299 W (too low)
   ⚠️  Cycle 11: Q_gen = 0.298 W (too low)
   ⚠️  Cycle 12: Q_gen = 0.296 W (too low)
   ⚠️  Cycle 13: Q_gen = 0.295 W (too low)
   ⚠️  Cycle 14: Q_gen = 0.294 W (too low)
   ⚠️  Cycle 15: Q_gen = 0.295 W (too low)
   ⚠️  Cycle 16: Q_gen = 0.294 W (too low)
   Cycle  16 | SOH=0.886 | R0=0.0300Ω | ECM time=1.25s | ECM_MSE=2180769.013372 | Ts_RMSE=0.673°C | Q_gen_mean=0.29W | Rin=0.54 Rout=34.53 Cc=61.90 Cs=11.59
   ⚠️  Cycle 17: Q_gen = 0.294 W (too low)
   ⚠️  Cycle 18: Q_gen = 0.291 W (too low)
   ⚠️  Cycle 19: Q_gen = 0.290 W (too low)
   ⚠️  Cycle 20: Q_gen = 0.291 W (too low)
   ⚠️  Cycle 21: Q_gen = 0.290 W (too low)
   ⚠️  Cycle 22: Q_gen = 0.287 W (too low)
   ⚠️  Cycle 23: Q_gen = 0.287 W (too low)
   ⚠️  Cycle 24: Q_gen = 0.287 W (too low)
   ⚠️  Cycle 25: Q_gen = 0.291 W (too low)
   ⚠️  Cycle 26: Q_gen = 0.291 W (too low)
   ⚠️  Cycle 27: Q_gen = 0.288 W (too low)
   ⚠️  Cycle 28: Q_gen = 0.286 W (too low)
   ⚠️  Cycle 29: Q_gen = 0.286 W (too low)
   ⚠️  Cycle 30: Q_gen = 0.284 W (too low)
   ⚠️  Cycle 31: Q_gen = 0.282 W (too low)
   ⚠️  Cycle 32: Q_gen = 0.282 W (too low)
   ⚠️  Cycle 33: Q_gen = 0.228 W (too low)
   ⚠️  Cycle 34: Q_gen = 0.227 W (too low)
   ⚠️  Cycle 35: Q_gen = 0.228 W (too low)
   ⚠️  Cycle 36: Q_gen = 0.227 W (too low)
   Cycle  36 | SOH=0.819 | R0=0.0300Ω | ECM time=1.08s | ECM_MSE=544079.611905 | Ts_RMSE=0.671°C | Q_gen_mean=0.23W | Rin=0.62 Rout=41.70 Cc=61.44 Cs=11.64
   ⚠️  Cycle 37: Q_gen = 0.224 W (too low)
   ⚠️  Cycle 38: Q_gen = 0.226 W (too low)
   ⚠️  Cycle 39: Q_gen = 0.225 W (too low)
   ⚠️  Cycle 40: Q_gen = 0.285 W (too low)
   ⚠️  Cycle 41: Q_gen = 0.227 W (too low)
   ⚠️  Cycle 42: Q_gen = 0.227 W (too low)
   ⚠️  Cycle 43: Q_gen = 0.226 W (too low)
   ⚠️  Cycle 44: Q_gen = 0.224 W (too low)
   ⚠️  Cycle 45: Q_gen = 0.225 W (too low)
   ⚠️  Cycle 46: Q_gen = 0.291 W (too low)
   ⚠️  Cycle 47: Q_gen = 0.289 W (too low)
   ⚠️  Cycle 48: Q_gen = 0.286 W (too low)
   ⚠️  Cycle 49: Q_gen = 0.284 W (too low)
   ⚠️  Cycle 50: Q_gen = 0.229 W (too low)
   ⚠️  Cycle 51: Q_gen = 0.229 W (too low)
   ⚠️  Cycle 52: Q_gen = 0.230 W (too low)
   ⚠️  Cycle 53: Q_gen = 0.226 W (too low)
   ⚠️  Cycle 54: Q_gen = 0.228 W (too low)
   ⚠️  Cycle 55: Q_gen = 0.227 W (too low)
   ⚠️  Cycle 56: Q_gen = 0.285 W (too low)
   Cycle  56 | SOH=0.837 | R0=0.0300Ω | ECM time=1.07s | ECM_MSE=879764.188835 | Ts_RMSE=0.704°C | Q_gen_mean=0.28W | Rin=0.67 Rout=40.88 Cc=62.34 Cs=10.52
   ⚠️  Cycle 57: Q_gen = 0.229 W (too low)
   ⚠️  Cycle 58: Q_gen = 0.229 W (too low)
   ⚠️  Cycle 59: Q_gen = 0.227 W (too low)
   ⚠️  Cycle 60: Q_gen = 0.225 W (too low)
   ⚠️  Cycle 61: Q_gen = 0.225 W (too low)
   ⚠️  Cycle 62: Q_gen = 0.226 W (too low)
   ⚠️  Cycle 63: Q_gen = 0.224 W (too low)
   ⚠️  Cycle 64: Q_gen = 0.225 W (too low)
   ⚠️  Cycle 65: Q_gen = 0.224 W (too low)
   ⚠️  Cycle 66: Q_gen = 0.225 W (too low)
   ⚠️  Cycle 67: Q_gen = 0.223 W (too low)
   ⚠️  Cycle 68: Q_gen = 0.223 W (too low)
   ⚠️  Cycle 69: Q_gen = 0.342 W (too low)
   ⚠️  Cycle 70: Q_gen = 0.439 W (too low)
   ⚠️  Cycle 71: Q_gen = 0.226 W (too low)
   ⚠️  Cycle 72: Q_gen = 0.224 W (too low)
   ⚠️  Cycle 73: Q_gen = 0.345 W (too low)
   ⚠️  Cycle 74: Q_gen = 0.439 W (too low)
   ⚠️  Cycle 75: Q_gen = 0.439 W (too low)
   ⚠️  Cycle 76: Q_gen = 0.440 W (too low)
   Cycle  76 | SOH=0.740 | R0=0.0300Ω | ECM time=0.94s | ECM_MSE=0.667156 | Ts_RMSE=0.680°C | Q_gen_mean=0.44W | Rin=0.86 Rout=50.05 Cc=59.67 Cs=8.42
   ⚠️  Cycle 77: Q_gen = 0.438 W (too low)
   ⚠️  Cycle 78: Q_gen = 0.440 W (too low)
   ⚠️  Cycle 79: Q_gen = 0.438 W (too low)
   ⚠️  Cycle 80: Q_gen = 0.437 W (too low)
   ⚠️  Cycle 81: Q_gen = 0.439 W (too low)
   ⚠️  Cycle 82: Q_gen = 0.436 W (too low)
   ⚠️  Cycle 83: Q_gen = 0.436 W (too low)
   ⚠️  Cycle 84: Q_gen = 0.436 W (too low)
   ⚠️  Cycle 85: Q_gen = 0.434 W (too low)
   ⚠️  Cycle 86: Q_gen = 0.441 W (too low)
   ⚠️  Cycle 87: Q_gen = 0.440 W (too low)
   ⚠️  Cycle 88: Q_gen = 0.438 W (too low)
/dist_home/aksay/miniconda3/lib/python3.12/site-packages/scipy/optimize/_numdiff.py:596: RuntimeWarning: invalid value encountered in subtract
  df = fun(x1) - f0
   ⚠️  Cycle 89: Q_gen = 0.437 W (too low)
   ⚠️  Cycle 90: Q_gen = 0.432 W (too low)
   ⚠️  Cycle 91: Q_gen = 0.442 W (too low)
   ⚠️  Cycle 92: Q_gen = 0.437 W (too low)
   ⚠️  Cycle 93: Q_gen = 0.437 W (too low)
   ⚠️  Cycle 94: Q_gen = 0.437 W (too low)
   ⚠️  Cycle 95: Q_gen = 0.436 W (too low)
   ⚠️  Cycle 96: Q_gen = 0.438 W (too low)
   Cycle  96 | SOH=0.704 | R0=0.0300Ω | ECM time=0.86s | ECM_MSE=0.708789 | Ts_RMSE=0.672°C | Q_gen_mean=0.44W | Rin=1.64 Rout=57.60 Cc=60.94 Cs=4.30
   ⚠️  Cycle 97: Q_gen = 0.436 W (too low)
   ⚠️  Cycle 98: Q_gen = 0.433 W (too low)
   ⚠️  Cycle 99: Q_gen = 0.433 W (too low)
   ⚠️  Cycle 100: Q_gen = 0.434 W (too low)
   ⚠️  Cycle 101: Q_gen = 0.433 W (too low)
   ⚠️  Cycle 102: Q_gen = 0.432 W (too low)
   ⚠️  Cycle 103: Q_gen = 0.432 W (too low)
   ⚠️  Cycle 104: Q_gen = 0.432 W (too low)
   ⚠️  Cycle 105: Q_gen = 0.432 W (too low)
   ⚠️  Cycle 106: Q_gen = 0.442 W (too low)
   ⚠️  Cycle 107: Q_gen = 0.440 W (too low)
   ⚠️  Cycle 108: Q_gen = 0.438 W (too low)
   ⚠️  Cycle 109: Q_gen = 0.437 W (too low)
   ⚠️  Cycle 110: Q_gen = 0.437 W (too low)
   ⚠️  Cycle 111: Q_gen = 0.436 W (too low)
   ⚠️  Cycle 112: Q_gen = 0.434 W (too low)
   ⚠️  Cycle 113: Q_gen = 0.436 W (too low)
   ⚠️  Cycle 114: Q_gen = 0.436 W (too low)
   ⚠️  Cycle 115: Q_gen = 0.433 W (too low)
   ⚠️  Cycle 116: Q_gen = 0.437 W (too low)
   Cycle 116 | SOH=0.694 | R0=0.0300Ω | ECM time=0.81s | ECM_MSE=0.714593 | Ts_RMSE=0.668°C | Q_gen_mean=0.44W | Rin=1.21 Rout=56.29 Cc=57.06 Cs=6.50
   ⚠️  Cycle 117: Q_gen = 0.434 W (too low)
   ⚠️  Cycle 118: Q_gen = 0.434 W (too low)
   ⚠️  Cycle 119: Q_gen = 0.433 W (too low)
   ⚠️  Cycle 120: Q_gen = 0.432 W (too low)
   ⚠️  Cycle 121: Q_gen = 0.439 W (too low)
   ⚠️  Cycle 122: Q_gen = 0.435 W (too low)
   ⚠️  Cycle 123: Q_gen = 0.437 W (too low)
   ⚠️  Cycle 124: Q_gen = 0.434 W (too low)
   ⚠️  Cycle 125: Q_gen = 0.433 W (too low)
   ⚠️  Cycle 126: Q_gen = 0.434 W (too low)
   ⚠️  Cycle 127: Q_gen = 0.436 W (too low)
   ⚠️  Cycle 128: Q_gen = 0.434 W (too low)
   ⚠️  Cycle 129: Q_gen = 0.432 W (too low)
   ⚠️  Cycle 130: Q_gen = 0.431 W (too low)
   ⚠️  Cycle 131: Q_gen = 0.435 W (too low)
   ⚠️  Cycle 132: Q_gen = 0.432 W (too low)

📊 Median thermal params: Rin=0.79, Rout=46.00, Cc=54.65, Cs=7.31
✅ Saved 185721 rows to augmented_aging_twin_dataset.csv
✅ ECM parameters saved to ecm_parameters.csv
✅ Validation log saved.
  📊 Saved aggressive_multi_temp_visualization.png
  📊 Saved mixed_multi_temp_visualization.png

🚗 Generating EV Real‑World Drive Cycle Dataset...

🚗 B0005: 8 aging states × 3 cycles × 3 temps
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06

🚗 B0006: 8 aging states × 3 cycles × 3 temps
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06

🚗 B0007: 8 aging states × 3 cycles × 3 temps
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06

🚗 B0018: 6 aging states × 3 cycles × 3 temps
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06
⚠️ Throughput 4.26 Ah exceeds cell rating (2 Ah) in UDDS
⚠️ Throughput 5.96 Ah exceeds cell rating (2 Ah) in HWFET
⚠️ Throughput 7.35 Ah exceeds cell rating (2 Ah) in US06

⏱️  EV dataset generation: 5.1s for 270 simulations
✅ EV Dataset: 246,240 rows (68.4 hours) → ev_drive_cycle_dataset.csv
✅ STEP COMPLETE: Generate Digital Twin (ECM + EETM + OCV Extraction)


============================================================
🚀 STEP: Train SOH Residual LSTM (with ECM R₀)
============================================================
   Loaded ECM parameters for R0
🧠 Training SOH Residual LSTM on cuda
   Combined dataset: 596 sequences from 4 batteries
   Train: 476 | Val: 120
Epoch [10/100], Train MSE: 0.000463, Val MSE: 0.000481
Epoch [20/100], Train MSE: 0.000427, Val MSE: 0.000428
Epoch [30/100], Train MSE: 0.000400, Val MSE: 0.000423
Epoch [40/100], Train MSE: 0.000396, Val MSE: 0.000423
Epoch [50/100], Train MSE: 0.000393, Val MSE: 0.000420
Epoch [60/100], Train MSE: 0.000409, Val MSE: 0.000468
Epoch [70/100], Train MSE: 0.000391, Val MSE: 0.000442
Epoch [80/100], Train MSE: 0.000385, Val MSE: 0.000415
Epoch [90/100], Train MSE: 0.000390, Val MSE: 0.000450
Epoch [100/100], Train MSE: 0.000386, Val MSE: 0.000419
✅ Model saved.
📈 Loss curve saved.
✅ STEP COMPLETE: Train SOH Residual LSTM (with ECM R₀)


============================================================
🚀 STEP: Train Core Temperature Transformer
============================================================
✅ STEP COMPLETE: Train Core Temperature Transformer


============================================================
🚀 STEP: Generate Paper Plots
============================================================
📊 Generating Paper Plots from Real Pipeline Data...
  ✅ fig1_voltage_validation.png
  ✅ fig2_surface_temp_validation.png
  ✅ fig3_core_temperature.png
  ✅ fig4_parameter_aging.png
  ✅ fig6_drive_thermal.png
Traceback (most recent call last):
  File "/dist_home/aksay/pranesh/EV-battery-management-system/reports/generate_paper_plots.py", line 483, in <module>
    fig7_transformer_test_validation(df)
  File "/dist_home/aksay/pranesh/EV-battery-management-system/reports/generate_paper_plots.py", line 309, in fig7_transformer_test_validation
    cyc = interpolate_cycle(cyc)
          ^^^^^^^^^^^^^^^^^^^^^^
  File "/dist_home/aksay/pranesh/EV-battery-management-system/reports/generate_paper_plots.py", line 97, in interpolate_cycle
    f = interp1d(t_old, df[col].values, kind='linear', fill_value='extrapolate')
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/dist_home/aksay/miniconda3/lib/python3.12/site-packages/scipy/interpolate/_interpolate.py", line 303, in __init__
    y = y.astype(np.float64)
        ^^^^^^^^^^^^^^^^^^^^
ValueError: could not convert string to float: 'B0018'

❌ ERROR: Step 'Generate Paper Plots' failed with return code 1.
   Pipeline halted. Check the error messages above.

```